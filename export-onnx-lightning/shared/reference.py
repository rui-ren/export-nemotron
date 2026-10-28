"""Independent eager PyTorch decode math for tiny tests and MTP validation.

This is deliberately not a fast serving engine. Real-model target validation
should additionally compare against Transformers using capture_hf.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .quantization import dequantize_int4, quantize_int4


def mamba_core(projection, past_conv, past_ssm, conv_w, conv_b, dt_bias, a, skip, norm,
               heads, dim, states, groups, epsilon, dt_min):
    dtype = projection.dtype
    inner = heads * dim
    channels = inner + 2 * groups * states
    gate, xbc, dt = projection.split([inner, channels, heads], dim=-1)
    history = torch.cat([past_conv, xbc.unsqueeze(-1)], dim=-1)
    conv = (history.float() * conv_w.reshape(1, channels, -1).float()).sum(-1) + conv_b.float()
    conv = F.silu(conv).to(dtype)
    x, b, c = conv.split([inner, groups * states, groups * states], dim=-1)
    b = b.reshape(1, groups, states).repeat_interleave(heads // groups, dim=1).float()
    c = c.reshape(1, groups, states).repeat_interleave(heads // groups, dim=1).float()
    dt = F.softplus(dt.float() + dt_bias.float()).clamp_min(dt_min).reshape(1, heads, 1, 1)
    decay = torch.exp(dt * a.reshape(1, heads, 1, 1))
    x = x.float().reshape(1, heads, dim, 1)
    state = past_ssm.transpose(-1, -2) * decay + (dt * b.unsqueeze(2)) * x
    scan = ((state * c.unsqueeze(2)).sum(-1) + x.squeeze(-1) * skip.reshape(1, heads, 1)).to(dtype)
    gated = scan.reshape(1, inner).float() * F.silu(gate.float())
    grouped = gated.reshape(1, groups, inner // groups)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + epsilon)
    y = (normalized.reshape(1, inner).to(dtype).float() * norm.float()).to(dtype)
    return y, history[:, :, 1:], state.transpose(-1, -2).contiguous(), conv, scan.reshape(1, inner)


def experts_core(x, indices, probabilities, up, down):
    values = []
    for expert in indices.reshape(-1):
        expert_id = int(expert.item())
        activated = F.relu(F.linear(x, up[expert_id])).square()
        values.append(F.linear(activated, down[expert_id]))
    stacked = torch.stack(values, dim=0)
    return (stacked.float() * probabilities.reshape(-1, 1, 1)).sum(0).to(x.dtype)


class TorchDecoder:
    def __init__(self, checkpoint, options, device="cpu"):
        self.source, self.c, self.options = checkpoint, checkpoint.config, options
        self.device = device
        self.dtype = torch.bfloat16 if options.dtype == "bfloat16" else torch.float32
        self.states = {}
        self.length = 0

    def weight(self, name, quantize=False):
        value = self.source.get(name)
        if quantize and self.options.variant == "int4":
            packed, scales = quantize_int4(value, self.options.block_size, self.dtype)
            value = dequantize_int4(packed, scales, value.shape[-1])
        return value.to(self.device, self.dtype)

    def linear(self, x, name, quantize=True):
        return F.linear(x, self.weight(name, quantize))

    def rms(self, x, name):
        x32 = x.float()
        normalized = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.c["layer_norm_epsilon"])
        return (normalized * self.weight(name).float()).to(self.dtype)

    def moe(self, x, prefix):
        flat = x.reshape(1, -1)
        # Router multiplication and normalization stay in FP32.
        gate = self.source.get(prefix + "gate.weight").to(self.device).float()
        bias = self.source.get(prefix + "gate.e_score_correction_bias").to(self.device).float()
        scores = F.linear(flat.float(), gate).sigmoid()
        indices = torch.topk(scores + bias, self.c["num_experts_per_tok"], sorted=True).indices
        probabilities = scores.gather(1, indices)
        probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20)
        probabilities = probabilities * self.c["routed_scaling_factor"]
        outputs = []
        for index in indices.reshape(-1):
            name = prefix + f"experts.{int(index.item())}."
            activated = F.relu(self.linear(flat, name + "up_proj.weight")).square()
            outputs.append(self.linear(activated, name + "down_proj.weight"))
        routed = (torch.stack(outputs).float() * probabilities.reshape(-1, 1, 1)).sum(0).to(self.dtype)
        shared = F.relu(self.linear(flat, prefix + "shared_experts.up_proj.weight")).square()
        shared = self.linear(shared, prefix + "shared_experts.down_proj.weight")
        return (routed + shared).reshape_as(x)

    def attention(self, x, prefix, layer):
        heads, kv, dim = self.c["num_attention_heads"], self.c["num_key_value_heads"], self.c["head_dim"]
        query = self.linear(x, prefix + "q_proj.weight").reshape(1, 1, heads, dim).transpose(1, 2)
        key = self.linear(x, prefix + "k_proj.weight").reshape(1, kv, dim)
        value = self.linear(x, prefix + "v_proj.weight").reshape(1, kv, dim)
        shape = (1, kv, self.options.max_length, dim)
        for kind, update in (("key", key), ("value", value)):
            name = f"present_{kind}.{layer}"
            past = self.states.get(name, torch.zeros(shape, device=self.device, dtype=self.dtype))
            present = past.clone()
            present[:, :, self.length] = update
            self.states[name] = present
        keys = self.states[f"present_key.{layer}"][:, :, :self.length + 1].repeat_interleave(heads // kv, dim=1)
        values = self.states[f"present_value.{layer}"][:, :, :self.length + 1].repeat_interleave(heads // kv, dim=1)
        scores = query.float() @ keys.float().transpose(-1, -2) * (dim ** -0.5)
        probs = scores.softmax(-1).to(self.dtype)
        output = (probs @ values).transpose(1, 2).reshape(1, 1, heads * dim)
        return self.linear(output, prefix + "o_proj.weight")

    def mamba(self, x, prefix, layer):
        c = self.c
        heads, dim, states, groups = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"], c["n_groups"]
        inner = heads * dim
        channels = inner + 2 * groups * states
        projection = self.linear(x, prefix + "in_proj.weight").reshape(1, -1)
        pc_name, ps_name = f"present_conv.{layer}", f"present_ssm.{layer}"
        conv = self.states.get(pc_name, torch.zeros(
            (1, channels, c["conv_kernel"] - 1), dtype=self.dtype, device=self.device))
        ssm = self.states.get(ps_name, torch.zeros(
            (1, heads, states, dim), dtype=torch.float32, device=self.device))
        result = mamba_core(
            projection, conv, ssm, self.weight(prefix + "conv1d.weight"), self.weight(prefix + "conv1d.bias"),
            self.source.get(prefix + "dt_bias").to(self.device).float(),
            -self.source.get(prefix + "A_log").to(self.device).float().exp(),
            self.source.get(prefix + "D").to(self.device).float(), self.weight(prefix + "norm.weight"),
            heads, dim, states, groups, c["layer_norm_epsilon"], self.options.dt_min,
        )
        y, self.states[pc_name], self.states[ps_name], _, _ = result
        return self.linear(y.reshape(1, 1, inner), prefix + "out_proj.weight")

    @torch.inference_mode()
    def step(self, token: int, target_hidden_states=None):
        if self.length >= self.options.max_length:
            raise ValueError("Attention cache capacity exceeded")
        x = self.weight("backbone.embeddings.weight")[token].reshape(1, 1, -1)
        if self.options.variant == "mtp":
            if target_hidden_states is None:
                raise ValueError("MTP needs aligned target_hidden_states")
            fused = torch.cat([self.rms(x, "mtp.layers.0.enorm.weight"),
                               self.rms(target_hidden_states, "mtp.layers.0.hnorm.weight")], dim=-1)
            x = self.linear(fused, "mtp.layers.0.eh_proj.weight")
            prefix, pattern = "mtp.layers.", ["attention", "moe"]
        else:
            prefix = "backbone.layers."
            pattern = self.c["layers_block_type"][:self.options.num_layers or None]
        for index, kind in enumerate(pattern):
            base = f"{prefix}{index}."
            normalized = self.rms(x, base + "norm.weight")
            cache_id = f"mtp.{index}" if self.options.variant == "mtp" else str(index)
            if kind == "mamba":
                mixed = self.mamba(normalized, base + "mixer.", cache_id)
            elif kind == "attention":
                mixed = self.attention(normalized, base + "mixer.", cache_id)
            else:
                mixed = self.moe(normalized, base + "mixer.")
            x = x + mixed
        norm = "mtp.layers.1.final_layernorm.weight" if self.options.variant == "mtp" else "backbone.norm_f.weight"
        hidden = self.rms(x, norm)
        logits = self.linear(hidden, "lm_head.weight", quantize=False).float()
        self.length += 1
        return {"logits": logits, "hidden_states": hidden, **self.states}
