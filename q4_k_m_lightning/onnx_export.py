"""Export a standalone mixed Q4_K_M ONNX GenAI package, not a GGUF runtime."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper

from export import REVISION, digest

FORMATS = {6, 8, 12}


def original_qk_rows(data, heads):
    rows, width = data.shape
    if rows % (heads * 2):
        raise ValueError("Invalid Q/K head dimensions")
    return data.reshape(heads, rows // heads // 2, 2, width).swapaxes(1, 2).reshape(rows, width)


def target_name(name, mapping):
    match = re.fullmatch(r"backbone\.layers\.(\d+)\.mixer\.(up|down)_experts", name)
    if match:
        return f"blk.{match[1]}.ffn_{match[2]}_exps.weight"
    return mapping.get_name(name, try_suffixes=(".weight", ".bias"))


def external_tensor(name, dtype, shape, offset, length):
    tensor = onnx.TensorProto(name=name, data_type=dtype, dims=shape)
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", "model.onnx.data"), ("offset", str(offset)), ("length", str(length))):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    return tensor


def replace_nodes(model, matrices):
    counts = Counter()
    nodes = []
    for node in model.graph.node:
        if node.op_type == "Gather" and node.input[0] in matrices:
            matrix = matrices[node.input[0]]
            replacement = helper.make_node("Embedding", [node.input[1], matrix["packed_name"]],
                list(node.output), domain="lightning_q4", n=matrix["n"], k=matrix["k"],
                format=matrix["format"])
            counts["Embedding"] += 1
        elif node.op_type in ("RowLinear", "MatMul") and node.input[1] in matrices:
            matrix = matrices[node.input[1]]
            replacement = helper.make_node("Linear", [node.input[0], matrix["packed_name"]],
                list(node.output), domain="lightning_q4", n=matrix["n"], k=matrix["k"],
                format=matrix["format"])
            counts["Linear"] += 1
        elif node.op_type in ("MoeDeviceBF16", "MoeCohortBF16") and node.input[3] in matrices:
            attributes = {a.name: a.i for a in node.attribute}
            if attributes.get("topk") != 6 or attributes.get("source_order") != 1:
                raise ValueError("Expected six source-ordered expert routes")
            up, down = matrices[node.input[3]], matrices[node.input[4]]
            if (up["shape"][:2] != [128, 1856] or down["shape"][:2] != [128, 2688]
                    or up["k"] != 2688 or down["k"] != 1856):
                raise ValueError("Expected Lightning expert dimensions")
            replacement = helper.make_node("MoE",
                list(node.input[:3]) + [up["packed_name"], down["packed_name"]], list(node.output),
                domain="lightning_q4", experts=128, hidden=2688, intermediate=1856,
                up_format=up["format"], down_format=down["format"])
            counts["MoE"] += 1
        else:
            replacement = copy.deepcopy(node)
        nodes.append(replacement)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid("lightning_q4", 1))
    return dict(counts)


def build_package(baseline, quantized, output, library, config):
    import gguf

    source = onnx.load(baseline / "model.onnx", load_external_data=False)
    model = copy.deepcopy(source)
    reader = gguf.GGUFReader(quantized)
    if int(reader.get_field("general.file_type").contents()) != int(gguf.LlamaFileType.MOSTLY_Q4_K_M):
        raise ValueError("Expected upstream Q4_K_M mixed preset")
    tensors = {t.name: t for t in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.NEMOTRON_H_MOE, config["num_hidden_layers"])
    matrices, initializers, retained = {}, [], []
    with (output / "model.onnx.data").open("xb") as weights:
        for parent in source.graph.initializer:
            target = target_name(parent.name, mapping)
            tensor = tensors.get(target)
            offset = weights.tell()
            if tensor is not None and int(tensor.tensor_type) in FORMATS:
                data = tensor.data.view(np.uint8)
                k = int(tensor.shape[0])
                n = int(tensor.shape[1])
                if list(parent.dims) != (
                        [int(tensor.shape[2]), n, k] if len(tensor.shape) == 3 else
                        [n, k] if parent.name == "backbone.embeddings.weight" else [k, n]):
                    raise ValueError(f"Source/packed shape mismatch: {parent.name}")
                block, block_bytes = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
                packed_shape = ([int(tensor.shape[2])] if len(tensor.shape) == 3 else []) + [
                    n, k // block * block_bytes]
                data = data.reshape(packed_shape)
                if parent.name.endswith((".q_proj.weight", ".k_proj.weight")):
                    heads = config["num_attention_heads"] if parent.name.endswith(".q_proj.weight") else (
                        config["num_key_value_heads"])
                    data = original_qk_rows(data, heads)
                flat = data.reshape(-1, data.shape[-1])
                for start in range(0, len(flat), 1024):
                    weights.write(np.ascontiguousarray(flat[start:start + 1024]).tobytes())
                packed_name = parent.name + ".q4km"
                initializers.append(external_tensor(packed_name, onnx.TensorProto.UINT8,
                                                    packed_shape, offset, weights.tell() - offset))
                matrices[parent.name] = {
                    "packed_name": packed_name, "reference_name": target, "format": int(tensor.tensor_type),
                    "format_name": tensor.tensor_type.name, "n": n, "k": k, "shape": packed_shape,
                    "bytes": weights.tell() - offset,
                    "qk_rows_restored": parent.name.endswith((".q_proj.weight", ".k_proj.weight")),
                }
            else:
                fields = {e.key: e.value for e in parent.external_data}
                if fields:
                    location = Path(fields["location"])
                    if location.is_absolute() or ".." in location.parts:
                        raise ValueError("External data must stay within the parent package")
                    with (baseline / location).open("rb") as stream:
                        stream.seek(int(fields.get("offset", 0)))
                        raw = stream.read(int(fields["length"]))
                        if len(raw) != int(fields["length"]):
                            raise ValueError("Truncated source initializer")
                elif parent.HasField("raw_data"):
                    raw = parent.raw_data
                else:
                    raw = onnx.numpy_helper.to_array(parent).tobytes()
                weights.write(raw)
                initializers.append(external_tensor(parent.name, parent.data_type, list(parent.dims),
                                                    offset, len(raw)))
                retained.append(parent.name)
    if len(matrices) != 164:
        raise ValueError(f"Expected all 164 quantized matrices, found {len(matrices)}")
    counts = replace_nodes(model, matrices)
    if counts != {"Embedding": 1, "Linear": 117, "MoE": 23}:
        raise ValueError(f"Incomplete quantized graph replacement: {counts}")
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(initializers)
    onnx.save_model(model, output / "model.onnx")
    onnx.checker.check_model(str(output / "model.onnx"))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "chat_template.jinja"):
        if (baseline / name).exists():
            shutil.copy2(baseline / name, output / name)
    runtime = json.loads((baseline / "genai_config.json").read_text())
    runtime["model"]["context_length"] = config["max_position_embeddings"]
    runtime["search"]["max_length"] = config["max_position_embeddings"]
    runtime["model"]["decoder"]["session_options"]["custom_ops_library"] = "liblightning_ops.so"
    (output / "genai_config.json").write_text(json.dumps(runtime, indent=2) + "\n")
    shutil.copy2(library, output / "liblightning_ops.so")
    report = {
        "format": "ONNX", "preset": "Q4_K_M", "mtp": False,
        "llama_cpp_quantizer_revision": REVISION,
        "runtime_requires_gguf": False, "runtime_requires_llama_cpp": False,
        "source_graph_sha256": digest(baseline / "model.onnx"),
        "quantized_weight_reference_sha256": digest(quantized),
        "graph_sha256": digest(output / "model.onnx"),
        "external_weights_sha256": digest(output / "model.onnx.data"),
        "external_weight_bytes": (output / "model.onnx.data").stat().st_size,
        "library_sha256": digest(output / "liblightning_ops.so"),
        "quantized_matrices": matrices, "retained_initializers": retained, "operators": counts,
        "execution": "Bounded temporary BF16 weight dequantization, not a fully BF16 resident model. "
                     "Expert cohorts reuse source-row reduction plans; routes synchronize to CPU.",
        "qualified": False, "benchmarked": False,
    }
    (output / "q4km_onnx.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--bf16-package", required=True, type=Path)
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--custom-library", required=True, type=Path)
    parser.add_argument("--quantizer", type=Path)
    parser.add_argument("--quantized-reference", type=Path,
                        help="Optional previously audited offline weight cache; never needed at runtime")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an ONNX package")
    config = json.loads((args.model_dir / "config.json").read_text())
    if (config.get("hidden_size") != 2688 or config.get("num_hidden_layers") != 52
            or config.get("architectures") != ["NemotronHForCausalLM"]):
        parser.error("Expected the original Lightning model")
    revision = subprocess.check_output(
        ["git", "-C", str(args.llama_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION or subprocess.check_output(
            ["git", "-C", str(args.llama_source), "status", "--porcelain"], text=True).strip():
        parser.error("Expected clean pinned offline quantizer source")
    quantized = args.quantized_reference
    if quantized is None:
        if args.work_dir is None or args.quantizer is None:
            parser.error("Provide --work-dir and --quantizer for the offline quantization step")
        subprocess.run([sys.executable, str(Path(__file__).with_name("export.py")),
            "--model-dir", str(args.model_dir), "--llama-source", str(args.llama_source),
            "--quantizer", str(args.quantizer), "--out", str(args.work_dir)], check=True)
        quantized = args.work_dir / "target-Q4_K_M.gguf"
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    args.out.mkdir(parents=True)
    report = build_package(args.bf16_package, quantized, args.out, args.custom_library, config)
    print(json.dumps({key: report[key] for key in ("format", "operators", "external_weight_bytes",
                                                  "runtime_requires_gguf", "qualified")}, indent=2))


if __name__ == "__main__":
    main()
