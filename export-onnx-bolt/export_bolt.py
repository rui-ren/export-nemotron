"""Export the Nemotron Bolt text backbone to ONNX with external weights."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_argument_group("1. Where to read the checkpoint")
    source.add_argument("--model-dir", default=os.environ.get("BOLT_DIR"),
                        help="Folder containing config.json and the safetensors index (or BOLT_DIR)")
    source.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Device used for weight conversion, not the inference provider")

    output = parser.add_argument_group("2. What to export")
    output.add_argument("--out", required=True, type=Path, help="New or empty output folder")
    output.add_argument("--num-layers", type=int, default=0, help="0 = all; a smaller stack is only for debugging")
    output.add_argument("--q8", choices=["bf16", "int8"], default="bf16")
    output.add_argument("--iq2", choices=["bf16", "int4", "lut4", "native"], default="bf16")
    output.add_argument("--experts", choices=["int4", "int8", "lut4", "native"], default="int4")

    runtime = parser.add_argument_group("3. Advanced runtime options (leave off while learning)")
    runtime.add_argument("--decode", action="store_true", help="Batch 1, static token count and fixed KV buffers")
    runtime.add_argument("--fused", action="store_true", help="Use the repository's bolt CUDA custom ops; requires --decode")
    runtime.add_argument("--verify-rows", type=int, default=0,
                         help="1..8 speculative verification tokens; implies --decode --fused")
    runtime.add_argument("--max-len", type=int, default=4096, help="KV buffer capacity in decode mode")
    runtime.add_argument("--dt-min", type=float, default=0.0, help="Mamba time-step lower clamp; 0 = no clamp")
    runtime.add_argument("--aux", action="store_true", help="Output residuals entering layers 1,18,35,51,68,85")
    runtime.add_argument("--debug-layers", type=int, nargs="*", default=[], help="Also output selected mixer results")
    args = parser.parse_args(argv)

    if not args.model_dir:
        parser.error("Provide --model-dir or set BOLT_DIR")
    if args.num_layers < 0 or args.max_len < 1:
        parser.error("--num-layers must be >= 0 and --max-len must be > 0")
    if not 0 <= args.verify_rows <= 8:
        parser.error("--verify-rows must be 0 (off) or 1..8")
    if args.verify_rows:
        args.decode = args.fused = True
    if args.fused and not args.decode:
        parser.error("--fused requires --decode")
    if args.verify_rows > args.max_len:
        parser.error("--max-len must be at least --verify-rows")
    if args.dt_min < 0 or not math.isfinite(args.dt_min):
        parser.error("--dt-min must be finite and >= 0")
    if args.fused and args.experts == "int8":
        parser.error("Fused expert ops support int4, lut4 or native, not int8")
    if not args.fused and (args.iq2 in ("lut4", "native") or args.experts in ("lut4", "native")):
        parser.error("lut4/native layouts require --fused; ORT int4 ops cannot decode these layouts")
    if args.experts in ("lut4", "native") and not args.verify_rows:
        parser.error("lut4/native experts require --verify-rows (MoeVerify)")
    if args.iq2 == "lut4" and args.verify_rows < 2:
        parser.error("lut4 Q4Gemv requires --verify-rows between 2 and 8")
    return args


def main(argv=None):
    options = parse_args(argv)

    # Keep heavy imports here so --help works without PyTorch or a checkpoint.
    import onnx
    import torch

    from checkpoint import SafetensorsCheckpoint
    from layers import LayerBuilder
    from model import AUX_LAYERS, build_model
    from tensor_store import ExternalTensorStore

    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for CPU weight conversion.")
    checkpoint = SafetensorsCheckpoint(options.model_dir)

    layer_types = checkpoint.config["layers_block_type"]
    layer_count = options.num_layers or len(layer_types)
    if not 1 <= layer_count <= len(layer_types):
        raise ValueError(f"--num-layers must be between 1 and {len(layer_types)}, or 0 for all")
    if options.aux and layer_count <= max(AUX_LAYERS):
        raise ValueError("--aux requires all auxiliary target layers (through layer 85)")
    if any(index < 0 or index >= layer_count for index in options.debug_layers):
        raise ValueError("--debug-layers indices must be inside the exported layer stack")
    if options.out.exists() and any(options.out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {options.out}. Choose a new folder.")
    options.out.mkdir(parents=True, exist_ok=True)

    # The store owns the file handle, so it closes even if graph construction fails.
    data_path = options.out / "model.onnx.data"
    with ExternalTensorStore(data_path) as store:
        layers = LayerBuilder(options, checkpoint, store)
        model = build_model(layers)
    graph_path = options.out / "model.onnx"
    onnx.save_model(model, graph_path)
    print(f"Graph:   {graph_path}")
    print(f"Weights: {data_path} ({store.offset / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
