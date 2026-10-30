"""Replay the frozen 71-input/512-output historical GenAI timing protocol."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

from build import digest

ROOT = Path(__file__).resolve().parent


def verify_mapped_runtime(
    expected: dict[str, str], maps: Path = Path("/proc/self/maps")
) -> dict:
    """Verify every mapped copy, not merely files in the requested runtime directory."""
    mapped = {}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        name = path.name.removesuffix(" (deleted)")
        if name not in expected:
            continue
        if fields[5].endswith(" (deleted)"):
            raise ValueError(f"Mapped runtime was deleted: {fields[5]}")
        resolved = str(path.resolve(strict=True))
        if resolved not in mapped:
            checksum = digest(path)
            if checksum != expected[name]:
                raise ValueError(f"Mapped runtime mismatch: {path}")
            mapped[resolved] = {"name": name, "sha256": checksum}
    missing = expected.keys() - {entry["name"] for entry in mapped.values()}
    if missing:
        raise ValueError(f"Runtime libraries not mapped: {sorted(missing)}")
    return mapped


def verify_package(model: Path, *, allow_rebuilt: bool) -> dict[str, str]:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    expected = {
        "model.onnx": provenance["graph_sha256"],
        "genai_config.json": provenance["config_sha256"],
        **{name: entry["sha256"] for name, entry in provenance["required_assets"].items()},
    }
    actual = {}
    for name, checksum in {**expected, **provenance["measured_library_sha256"]}.items():
        actual[name] = digest(model / name)
        rebuilt = (
            name.endswith(".so") and name not in provenance["required_unchanged_libraries"]
        )
        if actual[name] != checksum and not (allow_rebuilt and rebuilt):
            raise ValueError(f"Historical package mismatch: {name}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Complete diagnostic asset package"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New evidence file (never overwritten)"
    )
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--allow-rebuilt-libraries", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    if os.environ.get("BOLT_MAMBA_CUMSUM_HEAD_TILE", "1") != "1":
        parser.error("This frozen prompt requires BOLT_MAMBA_CUMSUM_HEAD_TILE=1")
    os.environ["BOLT_MAMBA_CUMSUM_HEAD_TILE"] = "1"
    provenance = json.loads((ROOT / "provenance.json").read_text())
    if digest(ROOT / "historical_generate.py") != provenance["historical_generate_sha256"]:
        raise ValueError("Historical timing helper changed")
    bindings = verify_package(args.model, allow_rebuilt=args.allow_rebuilt_libraries)
    runtime = {}
    for name, checksum in provenance["runtime_sha256"].items():
        runtime[name] = digest(args.runtime_dir / name)
        if runtime[name] != checksum:
            raise ValueError(f"Qualified runtime mismatch: {name}")

    import onnxruntime_genai as og
    from historical_generate import generate

    if digest(Path(og.__file__)) != runtime[Path(og.__file__).name]:
        raise ValueError("Imported GenAI does not match --runtime-dir")
    og.register_execution_provider_library(
        "CUDAExecutionProvider",
        str(args.runtime_dir.resolve() / "libonnxruntime_providers_cuda.so"),
    )
    model = og.Model(str(args.model.resolve()))
    mapped_runtime = verify_mapped_runtime(runtime)
    historical = json.loads((ROOT / "benchmark.json").read_text())
    capacity = json.loads((args.model / "genai_config.json").read_text())["model"][
        "context_length"
    ]
    result = {
        "bindings": bindings,
        "runtime_sha256": runtime,
        "mapped_runtime": mapped_runtime,
        "rebuilt_libraries_allowed": args.allow_rebuilt_libraries,
        "prompt_ids": historical["prompt_ids"],
        "runs": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Claim the evidence path before inference, retaining partial results on failure.
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    for index in range(5):
        run = generate(og, model, historical["prompt_ids"], 512, capacity)
        run["warmup"] = index < 2
        run["historical_ids_exact"] = run["tokens"] == historical["generated_ids"]
        result["runs"].append(run)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        if len(run["tokens"]) != 512 or not run["historical_ids_exact"]:
            raise RuntimeError(
                f"Historical output mismatch in run {index}; evidence preserved"
            )
    result["median_tok_s"] = statistics.median(run["tok_s"] for run in result["runs"][2:])
    result["status"] = "PASS_FIXED_512_HISTORICAL_TOKEN_REPLAY"
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"median_tok_s": result["median_tok_s"], "evidence": str(args.out)}))


if __name__ == "__main__":
    main()
