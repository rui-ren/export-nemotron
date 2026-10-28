"""Generate a small checkpoint in the exact Lightning naming convention."""
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def create_checkpoint(path: Path, gpu=False):
    path.mkdir(parents=True, exist_ok=True)
    hidden, states, intermediate, shared = (128, 8, 64, 128) if gpu else (8, 3, 12, 16)
    config = {
        "model_type": "nemotron_h", "architectures": ["NemotronHForCausalLM"],
        "dtype": "bfloat16", "hidden_size": hidden, "vocab_size": 11,
        "layers_block_type": ["mamba", "attention", "moe"], "num_hidden_layers": 3,
        "layer_norm_epsilon": 1e-5, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": hidden // 2,
        "mamba_num_heads": 2, "mamba_head_dim": hidden // 2, "ssm_state_size": states, "n_groups": 1,
        "conv_kernel": 4, "use_conv_bias": True, "mamba_hidden_act": "silu",
        "n_routed_experts": 3, "num_experts_per_tok": 2, "moe_intermediate_size": intermediate,
        "moe_shared_expert_intermediate_size": shared, "routed_scaling_factor": 2.5,
        "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "moe_latent_size": None,
        "mlp_hidden_act": "relu2", "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
    }
    generator = torch.Generator().manual_seed(123)
    tensors = {}

    def weight(name, shape, ones=False):
        value = torch.ones(shape) if ones else torch.randn(shape, generator=generator) * 0.12
        tensors[name] = value.to(torch.bfloat16)

    weight("backbone.embeddings.weight", (11, hidden))
    weight("backbone.norm_f.weight", (hidden,), True)
    weight("lm_head.weight", (11, hidden))

    def attention(prefix):
        for projection, rows in (("q", hidden), ("k", hidden // 2), ("v", hidden // 2), ("o", hidden)):
            weight(prefix + projection + "_proj.weight", (rows, hidden))

    def moe(prefix):
        weight(prefix + "gate.weight", (3, hidden))
        weight(prefix + "gate.e_score_correction_bias", (3,))
        for expert in range(3):
            weight(prefix + f"experts.{expert}.up_proj.weight", (intermediate, hidden))
            weight(prefix + f"experts.{expert}.down_proj.weight", (hidden, intermediate))
        weight(prefix + "shared_experts.up_proj.weight", (shared, hidden))
        weight(prefix + "shared_experts.down_proj.weight", (hidden, shared))

    for index in range(3):
        weight(f"backbone.layers.{index}.norm.weight", (hidden,), True)
    prefix = "backbone.layers.0.mixer."
    for name, shape in (
        ("in_proj.weight", (2 * hidden + 2 * states + 2, hidden)), ("out_proj.weight", (hidden, hidden)),
        ("conv1d.weight", (hidden + 2 * states, 1, 4)), ("conv1d.bias", (hidden + 2 * states,)),
        ("A_log", (2,)), ("dt_bias", (2,)),
    ):
        weight(prefix + name, shape)
    weight(prefix + "D", (2,), True)
    weight(prefix + "norm.weight", (hidden,), True)
    attention("backbone.layers.1.mixer.")
    moe("backbone.layers.2.mixer.")
    for name in ("enorm", "hnorm", "norm"):
        weight(f"mtp.layers.0.{name}.weight", (hidden,), True)
    weight("mtp.layers.0.eh_proj.weight", (hidden, 2 * hidden))
    attention("mtp.layers.0.mixer.")
    weight("mtp.layers.1.norm.weight", (hidden,), True)
    weight("mtp.layers.1.final_layernorm.weight", (hidden,), True)
    moe("mtp.layers.1.mixer.")
    save_file(tensors, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}))
    return config
