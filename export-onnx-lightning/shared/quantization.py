"""Calibration-free symmetric W4A16 RTN; unsigned nibbles have zero point 8."""
from __future__ import annotations

import torch


def quantize_int4(weight: torch.Tensor, block_size: int, scale_dtype: torch.dtype):
    """[N,K] -> uint8 [N,ceil(K/block),block/2], scales [N,ceil(K/block)].

    Values are rounded using the stored scale. Adjacent K values occupy the
    low/high nibbles. Padding represents zero, not an extra input feature.
    """
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError("INT4 block size must be a power of two >= 16")
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("INT4 input must be a finite matrix")
    rows, columns = weight.shape
    blocks = (columns + block_size - 1) // block_size
    padded = torch.zeros(rows, blocks * block_size, dtype=torch.float32)
    padded[:, :columns] = weight.float()
    values = padded.reshape(rows, blocks, block_size)
    maximum = values.abs().amax(dim=-1)
    scales = torch.where(maximum > 0, maximum / 7, torch.ones_like(maximum)).to(scale_dtype)
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("INT4 scales are not representable in the requested dtype")
    integers = torch.round(values / scales.float().unsqueeze(-1)).clamp(-7, 7).to(torch.int16) + 8
    packed = (integers[..., 0::2] | (integers[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()


def dequantize_int4(packed, scales, columns: int) -> torch.Tensor:
    low, high = packed & 15, packed >> 4
    integers = torch.stack([low, high], dim=-1).flatten(-2).float() - 8
    values = integers * scales.float().unsqueeze(-1)
    return values.flatten(-2)[..., :columns]
