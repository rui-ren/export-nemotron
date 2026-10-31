"""Verify every target checkpoint tensor against the unquantized reference GGUF."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np
from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite source audit")
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(args.gguf)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    config = json.loads((args.model_dir / "config.json").read_text())
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.NEMOTRON_H_MOE, config["num_hidden_layers"])
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    checks, covered = [], Counter()
    for shard in sorted(set(index.values())):
        with safe_open(args.model_dir / shard, framework="pt", device="cpu") as source:
            for name in sorted(name for name, file in index.items()
                               if file == shard and not name.startswith("mtp.")):
                expected = source.get_tensor(name).float()
                expert = re.fullmatch(
                    r"backbone.layers.(\d+).mixer.experts.(\d+).(up|down)_proj.weight", name)
                normalized = name.replace(".dt_bias", ".dt.bias")
                if name.endswith(".gate.e_score_correction_bias"):
                    layer = name.split(".")[2]
                    target = f"blk.{layer}.exp_probs_b.bias"
                elif expert:
                    layer, expert_id, direction = expert.groups()
                    target = f"blk.{layer}.ffn_{direction}_exps.weight"
                else:
                    target = mapping.get_name(normalized, try_suffixes=(".weight", ".bias"))
                if target not in tensors:
                    raise ValueError(f"No GGUF target for {name}: {target}")
                tensor = tensors[target]
                transformation = "reshape/expert-stack or exact dtype widening"
                if name.endswith(".A_log"):
                    expected = -torch.exp(expected)
                    transformation = "negative FP32 exp"
                elif name.endswith((".q_proj.weight", ".k_proj.weight")):
                    heads = config["num_attention_heads"] if name.endswith(".q_proj.weight") else (
                        config["num_key_value_heads"])
                    expected = expected.reshape(heads, 2, expected.shape[0] // heads // 2, -1).swapaxes(
                        1, 2).reshape(expected.shape)
                    transformation = "upstream per-head Q/K row permutation"
                if expert:
                    data = tensor.data[int(expert_id)]
                else:
                    data = tensor.data
                actual = gguf.dequantize(data, tensor.tensor_type).reshape(-1)
                if not np.array_equal(actual, expected.numpy().reshape(-1)):
                    raise ValueError(f"Source values changed: {name} -> {target}")
                covered[target] += 1
                checks.append({"source": name, "target": target, "exact": True,
                               "transformation": transformation})
    if set(covered) != set(tensors):
        raise ValueError(f"GGUF tensors without source checks: {set(tensors) - set(covered)}")
    report = {
        "complete": True, "source_tensors_checked": len(checks), "gguf_tensors_covered": len(covered),
        "mtp_excluded": True, "checks": checks,
        "source_context_length": config["max_position_embeddings"],
        "gguf_declared_context_length": reader.get_field("nemotron_h_moe.context_length").contents(),
        "context_limit_note": "Do not infer longer-context support from upstream GGUF metadata.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exact source audit: {len(checks)} checkpoint tensors -> {len(covered)} GGUF tensors")


if __name__ == "__main__":
    main()
