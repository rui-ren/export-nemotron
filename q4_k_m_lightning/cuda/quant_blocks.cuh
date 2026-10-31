#pragma once
#include <cuda_fp16.h>
#include <cstdint>

__host__ __device__ inline float HalfAt(const uint8_t* p) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | p[1] << 8)));
}

__host__ __device__ inline float DecodeWeight(const uint8_t* row, int column, int format) {
  if (format == 8) {
    const auto* p = row + static_cast<int64_t>(column / 32) * 34;
    return HalfAt(p) * static_cast<float>(static_cast<int8_t>(p[2 + column % 32]));
  }
  if (format == 6) {
    const auto* p = row + static_cast<int64_t>(column / 32) * 22;
    const int j = column % 32;
    const int low = (p[6 + j % 16] >> (j / 16 * 4)) & 15;
    const int high = (p[2 + j / 8] >> (j % 8)) & 1;
    return HalfAt(p) * static_cast<float>((low | high << 4) - 16);
  }
  const auto* p = row + static_cast<int64_t>(column / 256) * 144;
  const int j = column % 256, group = j / 32;
  const auto* s = p + 4;
  const int scale = group < 4 ? s[group] & 63 :
      (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
  const int minimum = group < 4 ? s[group + 4] & 63 :
      (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
  const int q = (p[16 + (j / 64) * 32 + j % 32] >> ((j / 32 % 2) * 4)) & 15;
  return (HalfAt(p) * scale) * q - HalfAt(p + 2) * minimum;
}
