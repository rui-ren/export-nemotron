"""Rebuild the archived diagnostic operators without editing their arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = re.compile(r'^(\s*#\s*include\s+)"([^"]+)"', re.MULTILINE)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_sources(output: Path) -> Path:
    """Verify immutable originals; relocate only absolute local include paths."""
    manifest = json.loads((ROOT / "provenance.json").read_text())["source_files"]
    stage = output / "sources"
    original_paths = {entry["original"]: name for name, entry in manifest.items()}
    for name, entry in manifest.items():
        path = ROOT / "sources" / name
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Archived source changed: {name}")
        text = path.read_text()

        def relocate(match: re.Match[str]) -> str:
            include = match[2]
            if not Path(include).is_absolute():
                return match[0]
            if include not in original_paths:
                raise ValueError(f"Unarchived include: {include}")
            return f'{match[1]}"{stage / original_paths[include]}"'

        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(INCLUDE.sub(relocate, text))
    return stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--ort-include", type=Path)
    parser.add_argument("--torch", type=Path, help="Installed torch package directory")
    parser.add_argument("--cublas", type=Path, help="Qualified libcublas.so.13 path")
    parser.add_argument("--cudart", type=Path, help="Qualified libcudart.so.13 path")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--objcopy", default="objcopy")
    args = parser.parse_args()
    output = args.build_dir.resolve()
    stage = prepare_sources(output)
    if args.prepare_only:
        print(f"Verified and staged sources in {stage}")
        return
    for name in ("cuda_home", "ort_include", "torch", "cublas", "cudart"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required to compile")
    for path in (
        args.cuda_home / "bin/nvcc",
        args.ort_include / "onnxruntime_cxx_api.h",
        args.torch / "include/ATen/ATen.h",
        args.cublas,
        args.cudart,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    objects = output / "objects"
    libraries = output / "lib"
    objects.mkdir(parents=True, exist_ok=True)
    libraries.mkdir(parents=True, exist_ok=True)
    # Keep the caller's cuBLAS path spelling: historical kernels check it via dladdr.
    replacements = {
        "SRC": stage,
        "OUT": objects,
        "LIB": libraries,
        "NVCC": args.cuda_home.absolute() / "bin/nvcc",
        "CUDA": args.cuda_home.absolute(),
        "ORT": args.ort_include.absolute(),
        "TORCH": args.torch.absolute(),
        "CUBLAS": args.cublas.absolute(),
        "CUBLAS_DIR": args.cublas.absolute().parent,
        "CUDART": args.cudart.absolute(),
        "CXX": args.cxx,
        "OBJCOPY": args.objcopy,
    }
    plan = json.loads((ROOT / "build-plan.json").read_text())
    executed = []
    for command in plan["commands"]:
        expanded = []
        for token in command:
            for name, value in replacements.items():
                token = token.replace(f"@{name}@", str(value))
            expanded.append(token)
        print(shlex.join(expanded), flush=True)
        subprocess.run(expanded, check=True)
        executed.append(expanded)
    receipt = {
        "commands": executed,
        "library_sha256": {path.name: digest(path) for path in sorted(libraries.glob("*.so"))},
        "historical_binary_identity": "Not assumed; compare hashes and qualify rebuilt binaries",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
