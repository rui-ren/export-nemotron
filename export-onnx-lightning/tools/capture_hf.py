"""Capture a teacher-forced reference using Transformers, in a separate process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dt-min", type=float, default=0.0,
                        help="Match the ONNX export; 0 selects unclamped recurrent semantics")
    args = parser.parse_args()
    from transformers import AutoConfig, AutoModelForCausalLM

    tokens = [int(value) for value in args.token_ids.split(",")]
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    config.time_step_min = args.dt_min
    model, loading = AutoModelForCausalLM.from_pretrained(
        args.model_dir, config=config, dtype=torch.bfloat16,
        device_map={"": f"cuda:{args.device}"}, attn_implementation="eager",
        output_loading_info=True, local_files_only=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Reference model did not load exactly: {loading}")
    unexpected = [name for name in loading.get("unexpected_keys", []) if not name.startswith("mtp.")]
    if unexpected:
        raise RuntimeError(f"Unexpected backbone weights; use a compatible Transformers build: {unexpected[:10]}")
    model.eval()
    candidates = [module for name, module in model.named_modules()
                  if name in ("model.norm_f", "backbone.norm_f")]
    if len(candidates) != 1:
        raise RuntimeError("Cannot identify the target's final normalization for the MTP hidden-state tap")
    hidden = []
    hook = candidates[0].register_forward_hook(
        lambda module, inputs, output: hidden.append(output.detach().float().cpu().numpy()))
    logits, cache = [], None
    with torch.inference_mode():
        for token in tokens:
            output = model(input_ids=torch.tensor([[token]], device=f"cuda:{args.device}"),
                           past_key_values=cache, use_cache=True, return_dict=True)
            cache = output.past_key_values
            logits.append(output.logits.float().cpu().numpy())
    hook.remove()
    if len(hidden) != len(tokens):
        raise RuntimeError("Reference hidden-state hook did not fire exactly once per token")
    data = {"token_ids": np.asarray(tokens, np.int64), "logits": np.stack(logits),
            "hidden_states": np.stack(hidden)}
    for index, kind in enumerate(config.layers_block_type):
        if kind in ("mamba", "linear_attention"):
            if not hasattr(cache, "get_recurrent_state") or not hasattr(cache, "get_conv_state"):
                raise RuntimeError("Reference cache API differs; use current compatible Transformers main")
            conv = cache.get_conv_state(index, state_idx=0)
            state = cache.get_recurrent_state(index, state_idx=0)
            data[f"present_conv.{index}"] = conv[:, :, -(config.conv_kernel - 1):].float().cpu().numpy()
            data[f"present_ssm.{index}"] = state.transpose(-1, -2).float().cpu().numpy()
        elif kind in ("attention", "full_attention"):
            layer = cache.layers[index]
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise RuntimeError("Cannot identify reference attention cache tensors")
            data[f"present_key.{index}"] = layer.keys[:, :, :len(tokens)].float().cpu().numpy()
            data[f"present_value.{index}"] = layer.values[:, :, :len(tokens)].float().cpu().numpy()
    np.savez(args.record, **data)
    print(json.dumps({"reference": "transformers", "tokens": len(tokens),
                      "dt_min": args.dt_min, "record": str(args.record)}, indent=2))


if __name__ == "__main__":
    main()
