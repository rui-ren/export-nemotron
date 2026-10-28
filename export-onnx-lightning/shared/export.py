"""Export Lightning as BF16, custom BF16, W4A16, or a separate MTP decode graph."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile


def parse_args(argv=None, variant=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Complete original safetensors checkpoint")
    parser.add_argument("--out", required=True, type=Path, help="New or empty destination directory")
    if variant is None:
        parser.add_argument("--variant", choices=("bf16", "custom", "int4", "mtp"), required=True)
    else:
        parser.set_defaults(variant=variant)
    parser.add_argument("--max-length", type=int, default=2048, help="Fixed attention-cache capacity")
    parser.add_argument("--attention", choices=("gqa", "standard"), default="gqa",
                        help="ORT CUDA GQA, or explicit standard-ONNX attention")
    parser.add_argument("--block-size", type=int, default=32, help="INT4 group size, power of two >= 16")
    parser.add_argument("--dt-min", type=float, default=0.0,
                        help="Mamba lower time-step clamp; 0 matches the un-clamped recurrent update")
    parser.add_argument("--mtp-backend", choices=("standard", "custom"), default="standard")
    parser.add_argument("--num-layers", type=int, default=0, help="Debug only: 0 = full backbone")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="float32 is a tiny-model CPU diagnostic, not the deployment format")
    args = parser.parse_args(argv)
    if args.max_length < 1 or args.num_layers < 0:
        parser.error("--max-length must be positive; --num-layers must be >= 0")
    if args.block_size < 16 or args.block_size & (args.block_size - 1):
        parser.error("--block-size must be a power of two >= 16")
    if not math.isfinite(args.dt_min) or args.dt_min < 0:
        parser.error("--dt-min must be finite and >= 0")
    if args.variant == "mtp" and args.num_layers:
        parser.error("--num-layers does not apply to the MTP graph")
    return args


def build_model(checkpoint, store, options):
    import torch
    from onnx import helper

    from .graph import CUSTOM, FLOAT, Graph, INT32, INT64, MS
    from .layers import Layers

    config = checkpoint.config
    dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32
    g = Graph(store, dtype)
    layers = Layers(checkpoint, g, options)
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    token_ids = g.input("input_ids", INT64, [1, 1])
    g.input("seqlens_k", INT32, [1])
    g.input("total_sequence_length", INT32, [])
    embedding = layers.weight("backbone.embeddings.weight", (vocab, hidden))
    x = g.op("Gather", [embedding, token_ids], axis=0)

    if options.variant == "mtp":
        if config.get("num_nextn_predict_layers") != 1 or config.get("mtp_layers_block_type") != ["attention", "moe"]:
            raise ValueError("MTP export supports exactly one attention+MoE prediction stage")
        previous = g.input("target_hidden_states", g.type, [1, 1, hidden])
        token_norm = layers.rms(x, "mtp.layers.0.enorm.weight")
        hidden_norm = layers.rms(previous, "mtp.layers.0.hnorm.weight")
        fused = g.op("Concat", [token_norm, hidden_norm], axis=-1)
        x = layers.linear(fused, "mtp.layers.0.eh_proj.weight", hidden, 2 * hidden)
        pattern = config["mtp_layers_block_type"]
        prefix = "mtp.layers."
    else:
        pattern = config["layers_block_type"]
        count = options.num_layers or len(pattern)
        if count > len(pattern):
            raise ValueError("--num-layers exceeds the checkpoint's layer count")
        pattern = pattern[:count]
        prefix = "backbone.layers."

    for index, kind in enumerate(pattern):
        layer_prefix = f"{prefix}{index}."
        normalized = layers.rms(x, layer_prefix + "norm.weight")
        cache_id = f"mtp.{index}" if options.variant == "mtp" else str(index)
        mixer_prefix = layer_prefix + "mixer."
        if kind == "mamba":
            result = layers.mamba(normalized, mixer_prefix, cache_id)
        elif kind == "attention":
            result = layers.attention(normalized, mixer_prefix, cache_id)
        elif kind == "moe":
            result = layers.moe(normalized, mixer_prefix)
        else:
            raise ValueError(f"Unsupported layer type: {kind}")
        x = g.op("Add", [x, result])
        print(f"Built {layer_prefix}{kind}; external data {store.file.tell() / 2**30:.2f} GiB", flush=True)

    final_weight = "mtp.layers.1.final_layernorm.weight" if options.variant == "mtp" else "backbone.norm_f.weight"
    x = layers.rms(x, final_weight)
    g.output(x, "hidden_states", g.type, [1, 1, hidden])
    logits = layers.linear(x, "lm_head.weight", vocab, hidden, quantize=False)
    g.output(g.cast(logits, FLOAT), "logits", FLOAT, [1, 1, vocab])
    graph = helper.make_graph(g.nodes, "lightning_" + options.variant, g.inputs, g.outputs,
                              initializer=store.initializers)
    domains = {node.domain for node in g.nodes}
    imports = [helper.make_opsetid("", 21)]
    imports += [helper.make_opsetid(domain, 1) for domain in (MS, CUSTOM) if domain in domains]
    model = helper.make_model(graph, opset_imports=imports, producer_name="lightning_safetensors_export")
    model.ir_version = 10
    manifest = {
        "format_version": 2, "variant": options.variant, "dtype": options.dtype,
        "kernel_contract": "bolt_style_lightning_v1" if layers.custom else None,
        "ssm_state_layout": "batch,heads,state_size,head_dim",
        "attention": options.attention, "max_length": options.max_length,
        "dt_min": options.dt_min, "block_size": options.block_size if layers.quantized else None,
        "decode_only": True, "batch_size": 1, "tokens_per_call": 1,
        "custom_library_required": layers.custom, "custom_domain": CUSTOM if layers.custom else None,
        "debug_partial_backbone": options.variant != "mtp" and len(pattern) != len(config["layers_block_type"]),
        "hidden_states_contract": "final normalized hidden states, also used as MTP target input",
        "mtp_input_contract": (
            "target_hidden_states=h_t; input_ids=x_(t+1); output predicts x_(t+2); "
            "MTP attention cache must represent the same aligned draft history"
        ) if options.variant == "mtp" else None,
        "states": g.state, "config": config, **checkpoint.fingerprints,
    }
    return model, manifest


def main(argv=None, variant=None):
    options = parse_args(argv, variant)
    import onnx
    from .checkpoint import Checkpoint, validate_config
    from .graph import TensorStore

    checkpoint = Checkpoint(options.model_dir)
    validate_config(checkpoint.config)
    custom = options.variant in ("custom", "int4") or (
        options.variant == "mtp" and options.mtp_backend == "custom")
    if custom and (checkpoint.config["n_routed_experts"] > 512 or checkpoint.config["num_experts_per_tok"] > 32):
        raise ValueError("Custom MoeRoute supports at most 512 experts and top-k <= 32")
    if options.out.exists() and any(options.out.iterdir()):
        raise FileExistsError(f"Destination is not empty: {options.out}")
    options.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=options.out) as temporary:
        staging = Path(temporary)
        with TensorStore(staging / "model.onnx.data") as store:
            model, manifest = build_model(checkpoint, store, options)
        onnx.save_model(model, staging / "model.onnx")
        onnx.checker.check_model(str(staging / "model.onnx"))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name in ("model.onnx.data", "model.onnx", "manifest.json"):
            os.replace(staging / name, options.out / name)
    print(f"Export complete: {options.out}")
    print("Single-token decode graph; initialize caches and replay prompt tokens to prefill.")


if __name__ == "__main__":
    main()
