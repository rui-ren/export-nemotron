"""Decode checkpoint bytes or repack them for the selected runtime.

Start with dequantize_q8_0: each block stores a scale and 32 signed bytes.
dequantize_nvfp4 uses a small lookup table. IQ2_XXS is more advanced: it
stores codebook indices and signs, rather than ordinary two-bit integers.

The packing formulas below are preserved from the original bolt-weights.py.
"""
from __future__ import annotations

import numpy as np
import torch
from gguf.quants import IQ2_XXS

IQ2_XXS.init_grid()
IQ2_SIGN_TABLE = torch.tensor(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy(), dtype=torch.int64)
IQ2_MAGNITUDE_TABLE = torch.tensor(np.asarray(IQ2_XXS.grid, dtype=np.float32).reshape(256, 8))
FP4_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)

def dequantize_iq2xxs(blk: torch.Tensor) -> torch.Tensor:
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
    ks = IQ2_SIGN_TABLE.to(dev)[si]  # [n,8,4]
    bits = (ks.unsqueeze(-1) >> torch.arange(8, device=dev)) & 1  # [n,8,4,8]
    sign = 1.0 - 2.0 * bits.to(torch.float32)
    grid = IQ2_MAGNITUDE_TABLE.to(dev)[gi]  # [n,8,4,8]
    out = db.reshape(n, 8, 1, 1) * grid * sign
    return out.reshape(*lead, nb * 256)


def iq2xxs_to_int4(blk: torch.Tensor, block: int = 32):
    """IQ2_XXS -> symmetric int4 (zero point 8), block-32 scales.

    The IQ2_XXS magnitudes {8,25,43}*db are mapped to the odd levels {1,3,5}*scale (scale = 8.514*db, the LS fit),
    which keeps the signs exactly and the magnitudes within ~1.3% rms.
    Returns (packed uint8 [..., K/2] low nibble first, scales float32 [..., K/32]).
    """
    v = dequantize_iq2xxs(blk)  # [..., K]
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


def dequantize_q8_0(blk: torch.Tensor) -> torch.Tensor:
    """Turn each [2 scale bytes + 32 signed value bytes] block into 32 floats."""
    leading_shape = blk.shape[:-2]
    block_count = blk.shape[-2]
    blocks = blk.reshape(-1, 34)
    scales = blocks[:, :2].contiguous().view(torch.float16).to(torch.float32)
    signed_values = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)
    weights = scales * signed_values
    return weights.reshape(*leading_shape, block_count * 32)


def q8_0_split(blk: torch.Tensor):
    """Q8_0 -> (uint8 weights [..., K] with zero point 128, fp16 scales [..., K/32])."""
    lead = blk.shape[:-2]
    nb = blk.shape[-2]
    b = blk.reshape(-1, nb, 34)
    d = b[..., :2].contiguous().view(torch.float16).reshape(*lead, nb)
    q = (b[..., 2:].contiguous().view(torch.int8).to(torch.int16) + 128).to(torch.uint8)
    return q.reshape(*lead, nb * 32), d


def dequantize_nvfp4(w: torch.Tensor, ws: torch.Tensor, ws2: torch.Tensor) -> torch.Tensor:
    """Unpack two FP4 values per byte, then apply block and global scales.

    w: packed bytes [N, K/2]; ws: FP8 scales [N, K/16]; ws2: scalar.
    The result is a normal float32 weight matrix [N, K].
    """
    lookup_table = FP4_VALUES.to(w.device)
    low_values = lookup_table[(w & 0xF).long()]
    high_values = lookup_table[(w >> 4).long()]
    values = torch.stack([low_values, high_values], dim=-1).reshape(w.shape[0], -1)
    block_scales = ws.to(torch.float32).repeat_interleave(16, dim=-1)
    combined_scales = block_scales * ws2.to(torch.float32)
    return values * combined_scales


def iq2xxs_to_int8(blk: torch.Tensor, block: int = 32):
    """Lossless IQ2_XXS -> int8 (zero point 128), block-32 scales (= the IQ2_XXS sub-block scale db).

    Returns (uint8 [..., K], float32 scales [..., K/32])."""
    v = dequantize_iq2xxs(blk)
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
    v = dequantize_iq2xxs(blk)
    K = v.shape[-1]
    n = blk.reshape(-1, 66)
    d = n[:, :2].contiguous().view(torch.float16).to(torch.float32)
    aux1 = n[:, 2:].contiguous().view(torch.int32).reshape(-1, 8, 2)[..., 1].to(torch.int64) & 0xFFFFFFFF
    db = (d.reshape(-1, 1) * (0.5 + (aux1 >> 28).to(torch.float32)) * 0.25).reshape(*v.shape[:-1], K // block, 1)
    mag = (v.reshape(*v.shape[:-1], K // block, block).abs() / db.clamp_min(1e-30))
    idx = torch.where(mag < 16.5, 0, torch.where(mag < 34, 1, 2)).to(torch.int64)
    code = (idx + 4 * (v.reshape(*v.shape[:-1], K // block, block) < 0).to(torch.int64)).reshape(*v.shape).to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), db.squeeze(-1)


def iq2xxs_to_native(blk: torch.Tensor):
    """ggml IQ2_XXS blocks [..., K/256, 66] -> the bolt `iq2=2` layout: records uint8 [..., K/32, 8] (4 grid-index bytes + one u32 of
    four 7-bit sign indices and a 4-bit scale, as in ggml) and the fp16 super-block scales [..., K/256] as raw bits in a bf16-typed
    tensor. Same 2.0625 bits per weight as the checkpoint, decoded exactly in the kernels."""
    nb = blk.shape[-2]
    rec = blk[..., 2:].reshape(*blk.shape[:-2], nb * 8, 8).contiguous()
    d = blk[..., :2].contiguous().view(torch.int16).squeeze(-1).view(torch.bfloat16)
    return rec, d
