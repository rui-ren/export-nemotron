"""Reference Q4_K_M export from original Lightning BF16, with auditable type mix."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REVISION = "5047748f114564fcf271e6a648f60e2d2eae6264"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(16 * 1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def audit(source, quantized, output):
    import gguf

    before, after = gguf.GGUFReader(source), gguf.GGUFReader(quantized)
    original = {tensor.name: tensor for tensor in before.tensors}
    if len(original) != len(after.tensors) or set(original) != {t.name for t in after.tensors}:
        raise ValueError("Quantization changed the tensor inventory")
    field = after.get_field("general.file_type")
    if field is None or int(field.contents()) != int(gguf.LlamaFileType.MOSTLY_Q4_K_M):
        raise ValueError("Expected the upstream mixed Q4_K_M preset")
    types, type_bytes, tensors = Counter(), Counter(), []
    for tensor in after.tensors:
        parent = original[tensor.name]
        if not np.array_equal(parent.shape, tensor.shape):
            raise ValueError(f"Shape changed: {tensor.name}")
        if parent.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32):
            raise ValueError("Source must be unquantized BF16/F32")
        same_type = tensor.tensor_type == parent.tensor_type
        data = tensor.data.reshape(-1, tensor.data.shape[-1])
        parent_data = parent.data.reshape(-1, parent.data.shape[-1])
        squared_error, squared_reference, max_error = 0.0, 0.0, 0.0
        for start in range(0, len(data), 256):
            block = data[start:start + 256]
            parent_block = parent_data[start:start + 256]
            values = gguf.dequantize(block, tensor.tensor_type)
            reference = gguf.dequantize(parent_block, parent.tensor_type)
            if not np.isfinite(values).all() or not np.isfinite(reference).all():
                raise ValueError(f"Nonfinite tensor: {tensor.name}")
            if same_type and not np.array_equal(block.view(np.uint8), parent_block.view(np.uint8)):
                raise ValueError(f"Unquantized tensor changed: {tensor.name}")
            difference = values.astype(np.float64) - reference
            squared_error += float(np.square(difference).sum())
            squared_reference += float(np.square(reference.astype(np.float64)).sum())
            max_error = max(max_error, float(np.abs(difference).max(initial=0)))
        kind = tensor.tensor_type.name
        types[kind] += 1
        type_bytes[kind] += tensor.n_bytes
        tensors.append({
            "name": tensor.name, "ggml_shape": tensor.shape.tolist(), "type": kind,
            "bytes": tensor.n_bytes, "elements": tensor.n_elements,
            "unchanged_bytes": same_type,
            "relative_weight_l2": (squared_error / max(squared_reference, 1e-300)) ** 0.5,
            "max_absolute_weight_error": max_error,
        })
    report = {
        "format": "GGUF", "preset": "Q4_K_M", "llama_cpp_revision": REVISION,
        "mtp": False, "importance_matrix": None, "onnx_runtime_ready": False,
        "all_values_finite": True, "tensor_inventory_and_shapes_preserved": True,
        "bf16_source": {"path": str(source.resolve()), "bytes": source.stat().st_size,
                        "sha256": digest(source)},
        "quantized": {"path": str(quantized.resolve()), "bytes": quantized.stat().st_size,
                      "gb": quantized.stat().st_size / 1e9,
                      "gib": quantized.stat().st_size / 2**30, "sha256": digest(quantized)},
        "tensor_types": dict(types), "tensor_type_bytes": dict(type_bytes),
        "effective_weight_bits": sum(type_bytes.values()) * 8 / sum(t["elements"] for t in tensors),
        "tensors": tensors,
        "limitation": "Structural/weight checks, not a model-quality or BF16-parity certificate. "
                      "Stock Q4_K_M may fall back on non-256-divisible rows.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--quantizer", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.llama_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error(f"Expected pinned llama.cpp revision {REVISION}")
    if subprocess.check_output(["git", "-C", str(args.llama_source), "status", "--porcelain"],
                               text=True).strip():
        parser.error("Reference quantizer source must be clean")
    if args.threads < 1:
        parser.error("--threads must be positive")
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    source = args.out / "target-bf16.gguf"
    quantized = args.out / "target-Q4_K_M.gguf"
    report = args.out / "quantization.json"
    if report.exists():
        parser.error("Refusing to overwrite completed evidence")
    if not args.audit_only:
        if args.model_dir is None or args.quantizer is None:
            parser.error("Export requires --model-dir and --quantizer")
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("Export destination must be new or empty")
        config = json.loads((args.model_dir / "config.json").read_text())
        if (config.get("architectures") != ["NemotronHForCausalLM"]
                or config.get("hidden_size") != 2688 or config.get("dtype") != "bfloat16"):
            parser.error("Expected the original Lightning BF16 safetensors checkpoint")
        args.out.mkdir(parents=True, exist_ok=True)
        commands = [
            [sys.executable, str(args.llama_source / "convert_hf_to_gguf.py"),
             str(args.model_dir), "--outfile", str(source), "--outtype", "bf16", "--no-nextn"],
            [str(args.quantizer), str(source), str(quantized), "Q4_K_M", str(args.threads)],
        ]
        (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        for index, command in enumerate(commands):
            with (args.out / f"export-{index}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = audit(source, quantized, report)
    print(json.dumps({key: result[key] for key in (
        "quantized", "tensor_types", "effective_weight_bits", "onnx_runtime_ready")}, indent=2))


if __name__ == "__main__":
    main()
