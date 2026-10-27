"""Weight access + dequantization helpers for NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB (ggml IQ2_XXS / Q8_0 + modelopt NVFP4)."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from gguf.quants import IQ2_XXS
from safetensors import safe_open

MODEL_DIR = os.environ.get("BOLT_DIR", "/data/tlwu/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB")
PFX = "language_model.backbone."

IQ2_XXS.init_grid()
_KSIGNS = torch.tensor(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy(), dtype=torch.int64)
_GRID = torch.tensor(np.asarray(IQ2_XXS.grid, dtype=np.float32).reshape(256, 8))  # values 8/25/43
_FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class Checkpoint:
    def __init__(self, path: str = MODEL_DIR):
        self.path = path
        self.wmap = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.cfg = cfg["llm_config"]
        self._open: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.wmap

    def get(self, name: str, device="cpu") -> torch.Tensor:
        fn = self.wmap[name]
        f = self._open.get(fn)
        if f is None:
            f = safe_open(os.path.join(self.path, fn), "pt", device="cpu")
            self._open[fn] = f
        t = f.get_tensor(name)
        return t.to(device) if device != "cpu" else t

    def get_many(self, names, device="cuda") -> torch.Tensor:
        return torch.stack([self.get(n) for n in names]).to(device)


def dq_iq2xxs(blk: torch.Tensor) -> torch.Tensor:
    """blk: uint8 [..., nb, 66] -> float32 [..., nb*256]"""
    dev = blk.device
    lead = blk.shape[:-2]
    nb = blk.shape[-2]
    b = blk.reshape(-1, 66)
    n = b.shape[0]
    d = b[:, :2].contiguous().view(torch.float16).to(torch.float32)  # [n,1]
    qs = b[:, 2:].contiguous().view(torch.int32).reshape(n, 8, 2).to(torch.int64) & 0xFFFFFFFF
    aux0 = qs[..., 0]  # 4 grid idx bytes
    aux1 = qs[..., 1]
    db = d.reshape(n, 1) * (0.5 + (aux1 >> 28).to(torch.float32)) * 0.25  # [n,8]
    gi = torch.stack([(aux0 >> (8 * i)) & 0xFF for i in range(4)], dim=-1)  # [n,8,4]
    si = torch.stack([(aux1 >> (7 * i)) & 0x7F for i in range(4)], dim=-1)
    ks = _KSIGNS.to(dev)[si]  # [n,8,4]
    bits = (ks.unsqueeze(-1) >> torch.arange(8, device=dev)) & 1  # [n,8,4,8]
    sign = 1.0 - 2.0 * bits.to(torch.float32)
    grid = _GRID.to(dev)[gi]  # [n,8,4,8]
    out = db.reshape(n, 8, 1, 1) * grid * sign
    return out.reshape(*lead, nb * 256)


def iq2xxs_to_int4(blk: torch.Tensor, block: int = 32):
    """IQ2_XXS -> symmetric int4 (zero point 8), block-32 scales.

    The IQ2_XXS magnitudes {8,25,43}*db are mapped to the odd levels {1,3,5}*scale (scale = 8.514*db, the LS fit),
    which keeps the signs exactly and the magnitudes within ~1.3% rms.
    Returns (packed uint8 [..., K/2] low nibble first, scales float32 [..., K/32]).
    """
    v = dq_iq2xxs(blk)  # [..., K]
    K = v.shape[-1]
    v = v.reshape(*v.shape[:-1], K // block, block)
    # per-32 sub-block db is the common scale: recover from max magnitude (43*db) when the block has a 43, else fit
    amax = v.abs().amax(dim=-1, keepdim=True)
    # level index from magnitude relative to the 3 grid values; db = amax/43 holds only if a 43 is present, so use LS fit instead
    # LS: unit u chosen so that |v| ~ u*{1,3,5}; we know |v|/db in {8,25,43} -> level = round((|v|/db - 8)/17.5)*2+1
    # db is recovered exactly from the raw block scale; recompute it here.
    n = blk.reshape(-1, 66)
    d = n[:, :2].contiguous().view(torch.float16).to(torch.float32)
    aux1 = n[:, 2:].contiguous().view(torch.int32).reshape(-1, 8, 2)[..., 1].to(torch.int64) & 0xFFFFFFFF
    db = d.reshape(-1, 1) * (0.5 + (aux1 >> 28).to(torch.float32)) * 0.25  # [n,8]
    db = db.reshape(*v.shape[:-2], K // block, 1)
    mag = v.abs() / db.clamp_min(1e-30)
    level = torch.where(mag < 16.5, 1, torch.where(mag < 34, 3, 5)).to(torch.int64)
    sign = (v < 0)
    q = torch.where(sign, 8 - level, 8 + level)  # 3,5,7 | 9,11,13
    scale = db * (8.514)
    q = q.reshape(*q.shape[:-2], K).to(torch.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()
    return packed, scale.squeeze(-1)


def dq_q8_0(blk: torch.Tensor) -> torch.Tensor:
    """uint8 [..., nb, 34] -> float32 [..., nb*32]"""
    lead = blk.shape[:-2]
    nb = blk.shape[-2]
    b = blk.reshape(-1, 34)
    d = b[:, :2].contiguous().view(torch.float16).to(torch.float32)
    x = b[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    return (d * x).reshape(*lead, nb * 32)


def q8_0_split(blk: torch.Tensor):
    """Q8_0 -> (uint8 weights [..., K] with zero point 128, fp16 scales [..., K/32])."""
    lead = blk.shape[:-2]
    nb = blk.shape[-2]
    b = blk.reshape(-1, nb, 34)
    d = b[..., :2].contiguous().view(torch.float16).reshape(*lead, nb)
    q = (b[..., 2:].contiguous().view(torch.int8).to(torch.int16) + 128).to(torch.uint8)
    return q.reshape(*lead, nb * 32), d


def dq_nvfp4(w: torch.Tensor, ws: torch.Tensor, ws2: torch.Tensor) -> torch.Tensor:
    """modelopt NVFP4: w uint8 [N,K/2], ws fp8 [N,K/16], ws2 fp32 scalar -> float32 [N,K]"""
    dev = w.device
    lut = _FP4.to(dev)
    lo = lut[(w & 0xF).long()]
    hi = lut[(w >> 4).long()]
    v = torch.stack([lo, hi], dim=-1).reshape(w.shape[0], -1)
    s = ws.to(torch.float32).repeat_interleave(16, dim=-1) * ws2.to(torch.float32)
    return v * s


def iq2xxs_to_int8(blk: torch.Tensor, block: int = 32):
    """Lossless IQ2_XXS -> int8 (zero point 128), block-32 scales (= the IQ2_XXS sub-block scale db).

    Returns (uint8 [..., K], float32 scales [..., K/32])."""
    v = dq_iq2xxs(blk)
    K = v.shape[-1]
    n = blk.reshape(-1, 66)
    d = n[:, :2].contiguous().view(torch.float16).to(torch.float32)
    aux1 = n[:, 2:].contiguous().view(torch.int32).reshape(-1, 8, 2)[..., 1].to(torch.int64) & 0xFFFFFFFF
    db = (d.reshape(-1, 1) * (0.5 + (aux1 >> 28).to(torch.float32)) * 0.25).reshape(*v.shape[:-1], K // block, 1)
    vb = v.reshape(*v.shape[:-1], K // block, block)
    q = torch.round(vb / db.clamp_min(1e-30)) + 128
    return q.reshape(*v.shape).to(torch.uint8), db.squeeze(-1)


def iq2xxs_to_lut4(blk: torch.Tensor, block: int = 32):
    """Exact IQ2_XXS in 4 bits: code = magnitude index (0,1,2 <-> 8,25,43) + 4 * (negative), low nibble first, block-32 scales db.

    Same packing / scale layout as iq2xxs_to_int4; the kernels (bolt iq2=1) decode the codes with a byte-permute LUT.
    Returns (packed uint8 [..., K/2], float32 scales [..., K/32])."""
    v = dq_iq2xxs(blk)
    K = v.shape[-1]
    n = blk.reshape(-1, 66)
    d = n[:, :2].contiguous().view(torch.float16).to(torch.float32)
    aux1 = n[:, 2:].contiguous().view(torch.int32).reshape(-1, 8, 2)[..., 1].to(torch.int64) & 0xFFFFFFFF
    db = (d.reshape(-1, 1) * (0.5 + (aux1 >> 28).to(torch.float32)) * 0.25).reshape(*v.shape[:-1], K // block, 1)
    mag = (v.reshape(*v.shape[:-1], K // block, block).abs() / db.clamp_min(1e-30))
    idx = torch.where(mag < 16.5, 0, torch.where(mag < 34, 1, 2)).to(torch.int64)
    code = (idx + 4 * (v.reshape(*v.shape[:-1], K // block, block) < 0).to(torch.int64)).reshape(*v.shape).to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), db.squeeze(-1)


GGUF_PATH = os.environ.get("BOLT_GGUF", "/data/tlwu/bolt_gguf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")


def nvfp4_from_ggml(arr: np.ndarray):
    """ggml NVFP4 rows ([N, K/64*36] bytes: per 64 values 4 UE4M3 scales + 32 nibble bytes, element j / j+8 share a byte)
    -> modelopt layout: (w uint8 [N, K/2] with elements 2i / 2i+1 in the low / high nibble, scales uint8 [N, K/16] e4m3 bytes)."""
    n = arr.shape[0]
    b = np.ascontiguousarray(arr).reshape(n, -1, 36)
    ws = b[:, :, :4].reshape(n, -1)
    qs = b[:, :, 4:].reshape(n, -1, 4, 8)
    elems = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)  # [n, nb, 4, 16]
    w = (elems[..., 0::2] | (elems[..., 1::2] << 4)).reshape(n, -1)
    return torch.from_numpy(w.copy()), torch.from_numpy(ws.copy())


class GgufCheckpoint(Checkpoint):
    """The HF checkpoint, but with the quantised weights (IQ2_XXS experts / shared experts, NVFP4 mixers, Q8_0 projections + lm_head), the
    router (fp32 in the GGUF) and the router bias taken from NVIDIA's GGUF: same blocks / formats, a different (better) quantisation realisation."""

    def __init__(self, path: str = MODEL_DIR, gguf_path: str = GGUF_PATH):
        super().__init__(path)
        from gguf import GGUFReader

        self._T = {t.name: t for t in GGUFReader(gguf_path).tensors}

    def _a(self, n):
        return self._T[n].data

    def get(self, name: str, device="cpu") -> torch.Tensor:
        import re

        m = re.match(re.escape(PFX) + r"layers\.(\d+)\.mixer\.(.*)", name)
        out = None
        if m:
            il, rest = int(m.group(1)), m.group(2)
            mm = re.match(r"experts\.(\d+)\.(up|down)_proj\.weight$", rest)
            if mm:
                a = self._a(f"blk.{il}.ffn_{mm.group(2)}_exps.weight")[int(mm.group(1))]
                out = torch.from_numpy(np.ascontiguousarray(a).reshape(a.shape[0], -1, 66))
            mm = re.match(r"shared_experts\.(up|down)_proj\.weight$", rest)
            if mm:
                a = self._a(f"blk.{il}.ffn_{mm.group(1)}_shexp.weight")
                out = torch.from_numpy(np.ascontiguousarray(a).reshape(a.shape[0], -1, 66))
            for hf, gn in (("fc1_latent_proj", "ffn_latent_down"), ("fc2_latent_proj", "ffn_latent_up"), ("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                if rest == hf + ".weight":
                    a = self._a(f"blk.{il}.{gn}.weight")
                    out = torch.from_numpy(np.ascontiguousarray(a).reshape(a.shape[0], -1, 34))
            for hf, gn in (("in_proj", "ssm_in"), ("out_proj", "ssm_out")):
                if rest == hf + ".weight":
                    out = nvfp4_from_ggml(np.array(self._a(f"blk.{il}.{gn}.weight")))[0]
                elif rest == hf + ".weight_scale":
                    out = nvfp4_from_ggml(np.array(self._a(f"blk.{il}.{gn}.weight")))[1].view(torch.float8_e4m3fn)
                elif rest == hf + ".weight_scale_2":
                    out = torch.tensor(np.array(self._a(f"blk.{il}.{gn}.scale")).reshape(-1)[0], dtype=torch.float32)
            if rest == "gate.weight":
                out = torch.from_numpy(np.array(self._a(f"blk.{il}.ffn_gate_inp.weight")))
            elif rest == "gate.e_score_correction_bias":
                out = torch.from_numpy(np.array(self._a(f"blk.{il}.exp_probs_b.bias")))
        elif name == "language_model.lm_head.weight":
            a = self._a("output.weight")
            out = torch.from_numpy(np.ascontiguousarray(a).reshape(a.shape[0], -1, 34))
        if out is None:
            return super().get(name, device)
        return out.to(device) if device != "cpu" else out


def iq2xxs_to_native(blk: torch.Tensor):
    """ggml IQ2_XXS blocks [..., K/256, 66] -> the bolt `iq2=2` layout: records uint8 [..., K/32, 8] (4 grid-index bytes + one u32 of
    four 7-bit sign indices and a 4-bit scale, as in ggml) and the fp16 super-block scales [..., K/256] as raw bits in a bf16-typed
    tensor. Same 2.0625 bits per weight as the checkpoint, decoded exactly in the kernels."""
    nb = blk.shape[-2]
    rec = blk[..., 2:].reshape(*blk.shape[:-2], nb * 8, 8).contiguous()
    d = blk[..., :2].contiguous().view(torch.int16).squeeze(-1).view(torch.bfloat16)
    return rec, d

