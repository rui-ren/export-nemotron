"""Load the scale-fixed native graph using an existing HF-derived asset package."""

from __future__ import annotations

import json
from pathlib import Path

_GRAPH_BYTES = Path(__file__).with_name("model.onnx").read_bytes()
_LIBRARIES = (
    "libbolt_upstream.so",
    "libbolt_q8_fp16.so",
    "libbolt_iq2_source.so",
)


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
        og.register_execution_provider_library("CUDAExecutionProvider", str(provider))
    config = og.Config(str(root))
    config.overlay(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "session_options": {
                            "custom_ops_libraries": list(_LIBRARIES),
                        }
                    }
                }
            }
        )
    )
    config.add_model_data("model.onnx", _GRAPH_BYTES)
    return og.Model(config)
