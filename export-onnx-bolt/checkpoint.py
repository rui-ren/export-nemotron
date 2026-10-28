"""Read model configuration and named weights from safetensors checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import torch

BACKBONE_PREFIX = "language_model.backbone."


class SafetensorsCheckpoint:
    """Read config and named tensors from the original safetensors folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with (self.path / "model.safetensors.index.json").open(encoding="utf-8") as file:
            self.weight_map: dict[str, str] = json.load(file)["weight_map"]
        with (self.path / "config.json").open(encoding="utf-8") as file:
            self.config = json.load(file)["llm_config"]
        self._open_shards = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str, device="cpu") -> torch.Tensor:
        from safetensors import safe_open

        filename = self.weight_map[name]
        if filename not in self._open_shards:
            # "pt" asks for PyTorch tensors; the file format is still safetensors.
            self._open_shards[filename] = safe_open(str(self.path / filename), "pt", device="cpu")
        tensor = self._open_shards[filename].get_tensor(name)
        return tensor.to(device) if device != "cpu" else tensor

    def get_many(self, names, device="cuda") -> torch.Tensor:
        return torch.stack([self.get(name) for name in names]).to(device)
