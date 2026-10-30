"""Load the scale-fixed native graph using an existing HF-derived asset package."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

_GRAPH_BYTES = Path(__file__).with_name("model.onnx").read_bytes()
_LIBRARIES = (
    "libbolt_upstream.so",
    "libbolt_q8_fp16.so",
    "libbolt_iq2_source.so",
)
_PROVIDER_LOCK = Lock()
_PROVIDER_REGISTRATIONS = []


def _mapped_cuda_providers() -> set[Path]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return set()
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name == "libonnxruntime_providers_cuda.so":
            paths.add(Path(fields[5]).resolve())
    return paths


def _register_cuda_provider(og, provider: Path) -> None:
    with _PROVIDER_LOCK:
        for runtime, registered in _PROVIDER_REGISTRATIONS:
            if runtime is og:
                if registered != provider:
                    raise ValueError(
                        f"CUDA provider already registered from {registered}, not {provider}"
                    )
                return
        mapped = _mapped_cuda_providers()
        if mapped and mapped != {provider}:
            raise ValueError(
                f"Conflicting mapped CUDA providers: {mapped}; requested {provider}"
            )
        try:
            og.register_execution_provider_library("CUDAExecutionProvider", str(provider))
        except RuntimeError as error:
            if "already registered under CUDAExecutionProvider" not in str(error):
                raise
            # GenAI exposes no registration-path getter. Accept an external
            # registration only when exactly the requested provider is mapped.
            if _mapped_cuda_providers() != {provider}:
                raise RuntimeError(
                    "CUDA was registered externally; cannot verify its library path. "
                    "Omit cuda_provider_library to use the application's registration."
                ) from error
        _PROVIDER_REGISTRATIONS.append((og, provider))


def create_model(asset_root: str | Path, *, cuda_provider_library: str | Path | None = None):
    """Keep weights/configuration at asset_root; overlay only the native graph.

    Requires the qualified Nemotron Bolt GenAI runtime, not an arbitrary stock
    onnxruntime-genai wheel. Copy the three built DSOs into asset_root first.
    """
    import onnxruntime_genai as og

    root = Path(asset_root).resolve(strict=True)
    for name in ("genai_config.json", *_LIBRARIES):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    if cuda_provider_library is not None:
        provider = Path(cuda_provider_library).resolve(strict=True)
        _register_cuda_provider(og, provider)
    config = og.Config(str(root))
    config.overlay(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "custom_ops_libraries": list(_LIBRARIES),
                        },
                    }
                }
            }
        )
    )
    config.add_model_data("model.onnx", _GRAPH_BYTES)
    return og.Model(config)
