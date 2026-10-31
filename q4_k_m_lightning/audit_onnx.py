"""Verify every ONNX packed/retained initializer and the native decoder ABI."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx

from export import digest


def data(tensor, directory):
    fields = {e.key: e.value for e in tensor.external_data}
    location = Path(fields["location"])
    if location.is_absolute() or ".." in location.parts:
        raise ValueError("Initializer escapes the package")
    return np.memmap(directory / location, mode="r", dtype=np.uint8,
                     offset=int(fields["offset"]), shape=(int(fields["length"]),))


def audit(package, baseline, reference, config):
    import gguf

    model = onnx.load(package / "model.onnx", load_external_data=False)
    original = onnx.load(baseline / "model.onnx", load_external_data=False)
    onnx.checker.check_model(str(package / "model.onnx"))
    for field in ("input", "output"):
        if [v.SerializeToString() for v in getattr(model.graph, field)] != [
                v.SerializeToString() for v in getattr(original.graph, field)]:
            raise ValueError(f"Native decoder {field} ABI changed")
    report = json.loads((package / "q4km_onnx.json").read_text())
    old = {t.name: t for t in original.graph.initializer}
    tensors = {t.name: t for t in gguf.GGUFReader(reference).tensors}
    quantized = {v["packed_name"]: (name, v) for name, v in report["quantized_matrices"].items()}
    checked, retained = [], []
    for tensor in model.graph.initializer:
        actual = data(tensor, package)
        if tensor.name not in quantized:
            expected = data(old[tensor.name], baseline)
            if not np.array_equal(actual, expected):
                raise ValueError(f"Retained initializer changed: {tensor.name}")
            retained.append(tensor.name)
            continue
        name, info = quantized[tensor.name]
        parent = tensors[info["reference_name"]]
        if tensor.data_type != onnx.TensorProto.UINT8 or list(tensor.dims) != info["shape"]:
            raise ValueError("Invalid packed initializer dtype/shape")
        expected = parent.data.reshape(info["shape"])
        actual = actual.reshape(info["shape"])
        if info["qk_rows_restored"]:
            heads = config["num_attention_heads"] if name.endswith(".q_proj.weight") else (
                config["num_key_value_heads"])
            actual = actual.reshape(heads, 2, actual.shape[0] // heads // 2, -1).swapaxes(
                1, 2).reshape(expected.shape)
        actual, expected = actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1])
        for start in range(0, len(actual), 1024):
            if not np.array_equal(actual[start:start + 1024], expected[start:start + 1024]):
                raise ValueError(f"Quantized bits changed: {name}")
        checked.append(name)
    if len(checked) != 164 or len(retained) != 237:
        raise ValueError("Incomplete ONNX weight audit")
    return {
        "complete": True, "quantized_matrices_bitwise_checked": len(checked),
        "retained_initializers_bitwise_checked": len(retained), "native_decoder_abi_unchanged": True,
        "graph_sha256": digest(package / "model.onnx"),
        "external_weights_sha256": digest(package / "model.onnx.data"),
        "quantized_matrices": checked, "retained_initializers": retained,
        "limitation": "Structural/weight checks are not CUDA execution or model quality qualification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--quantized-reference", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite audit evidence")
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    result = audit(args.model, args.baseline, args.quantized_reference,
                   json.loads((args.model_dir / "config.json").read_text()))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
