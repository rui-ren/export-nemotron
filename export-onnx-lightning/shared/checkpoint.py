"""Read Lightning's original floating-point safetensors, one tensor at a time."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        config_bytes = (self.path / "config.json").read_bytes()
        index_bytes = (self.path / "model.safetensors.index.json").read_bytes()
        self.config = json.loads(config_bytes)
        self.weight_map: dict[str, str] = json.loads(index_bytes)["weight_map"]
        self.fingerprints = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        }

    def get(self, name: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(f"Required Lightning weight is absent from the index: {name}")
        with safe_open(str(self.path / self.weight_map[name]), framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(name)
        if not tensor.is_floating_point():
            raise TypeError(f"{name}: expected floating-point weights, got {tensor.dtype}")
        if shape is not None and tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
        return tensor


def validate_config(config: dict) -> None:
    """Reject architectural differences instead of silently exporting wrong math."""
    required = (
        "hidden_size", "vocab_size", "layers_block_type", "layer_norm_epsilon",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups", "conv_kernel",
        "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
        "moe_shared_expert_intermediate_size", "routed_scaling_factor",
    )
    for key in required:
        if key not in config:
            raise ValueError(f"Missing Lightning configuration field: {key}")
    dimensions = (
        "hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim",
        "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups", "conv_kernel",
        "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
        "moe_shared_expert_intermediate_size",
    )
    for key in dimensions:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["conv_kernel"] < 2:
        raise ValueError("conv_kernel must be at least 2")
    for key in ("layer_norm_epsilon", "routed_scaling_factor"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config.get("model_type") != "nemotron_h":
        raise ValueError("This exporter supports model_type=nemotron_h only")
    if config.get("moe_latent_size") is not None:
        raise ValueError("Latent experts are not supported by the Lightning exporter")
    if config.get("n_group", 1) != 1 or config.get("topk_group", 1) != 1:
        raise ValueError("Grouped expert routing is not implemented")
    if not config.get("norm_topk_prob", True):
        raise ValueError("This exporter requires normalized top-k routing probabilities")
    if config.get("mlp_hidden_act", "relu2") != "relu2":
        raise ValueError("Only relu2 expert activation is supported")
    if config.get("mamba_hidden_act", "silu") != "silu":
        raise ValueError("Only SiLU Mamba convolution is supported")
    for flag in ("attention_bias", "mlp_bias", "mamba_proj_bias", "use_bias", "residual_in_fp32"):
        if config.get(flag, False):
            raise ValueError(f"Unsupported Lightning option: {flag}=True")
    if not config.get("use_conv_bias", True):
        raise ValueError("The Mamba implementation requires convolution bias")
    if config["mamba_num_heads"] % config["n_groups"]:
        raise ValueError("Mamba head count must be divisible by n_groups")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("Attention head count must be divisible by KV head count")
    if not 1 <= config["num_experts_per_tok"] <= config["n_routed_experts"]:
        raise ValueError("Invalid routed expert count/top-k")
    if not config["layers_block_type"] or set(config["layers_block_type"]) - {"mamba", "attention", "moe"}:
        raise ValueError("Unsupported or empty layer pattern")
