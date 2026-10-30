#ifdef TEST_IQ2
#include "../iq2-compat/bolt_ops.cu"
#else
#include "../compat/bolt_ops.cu"
#endif

#include <cstdio>

namespace {

#ifndef TEST_IQ2
__device__ __align__(16) uint8_t q8_weights[4096][64];
__device__ __half q8_scales[4096][2];
#endif

__global__ void CheckScales(int* failures, int* witnesses) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 4096) return;
  // Non-BF16 FP16 scales exercise the premature-rounding regression.
  const float base = __half2float(__ushort_as_half(0x2001u + index));
  uint32_t lo[4], hi[4];
#ifdef TEST_IQ2
  const float scale = base * (0.5f + static_cast<float>(index % 16)) * 0.25f;
  const uint32_t sign_a = index & 255, sign_b = (index * 17) & 255;
  const uint4 raw = make_uint4(index % 256, (index + 113) % 256, sign_a, sign_b);
  DequantRow<kFmtIq2N>(raw, scale, lo, hi);
  const uint64_t grids[2] = {kIq2xxsGrid[raw.x], kIq2xxsGrid[raw.y]};
  for (int i = 0; i < 16; ++i) {
    const int group = i / 8, j = i % 8;
    const float magnitude = static_cast<float>((grids[group] >> (8 * j)) & 255u);
    const uint32_t sign = group == 0 ? sign_a : sign_b;
    const float grid = (sign & (1u << j)) ? -magnitude : magnitude;
    const int slot = group * 2 + j / 4;
    const uint32_t pair = (j % 4 < 2) ? lo[slot] : hi[slot];
    const uint16_t actual = static_cast<uint16_t>(pair >> (16 * (j % 2)));
    const uint16_t expected = __bfloat16_as_ushort(__float2bfloat16(grid * scale));
    const float early_scale = B2F(__float2bfloat16(scale));
    const uint16_t old = __bfloat16_as_ushort(__float2bfloat16(grid * early_scale));
    if (actual != expected) atomicAdd(failures, 1);
    if (old != expected) atomicAdd(witnesses, 1);
  }
#else
  // Exercise the actual FP16 scale loader before the original Q8 reconstruction.
  uint8_t* weights = q8_weights[index];
  __half* scales = q8_scales[index];
  scales[0] = scales[1] = __float2half(base);
  for (int i = 0; i < 64; ++i) weights[i] = static_cast<uint8_t>(index + i);
  for (int t = 0; t < 4; ++t) {
    uint4 raw;
    float scale;
    LoadWeightRow<kFmtQ8F16, false>(weights, scales, 0, 64, 0, t, raw, &scale);
    if (scale != base) atomicAdd(failures, 1);
    DequantRow<kFmtQ8F16>(raw, scale, lo, hi);
    for (int s = 0; s < 4; ++s) {
      for (int word = 0; word < 4; ++word) {
        const uint32_t pair = word < 2 ? lo[s] : hi[s];
        const uint16_t actual = static_cast<uint16_t>(pair >> (16 * (word % 2)));
        const float q = static_cast<float>(weights[16 * t + 4 * word + s]) - 128.f;
        const uint16_t expected = __bfloat16_as_ushort(__float2bfloat16(q * base));
        const float early_scale = B2F(__float2bfloat16(base));
        const uint16_t old = __bfloat16_as_ushort(__float2bfloat16(q * early_scale));
        if (actual != expected) atomicAdd(failures, 1);
        if (old != expected) atomicAdd(witnesses, 1);
      }
    }
  }
#endif
}

bool CheckCuda(cudaError_t error) {
  if (error == cudaSuccess) return true;
  std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
  return false;
}

}  // namespace

int main() {
  int* counts = nullptr;
  if (!CheckCuda(cudaMallocManaged(&counts, 2 * sizeof(int)))) return 1;
  counts[0] = counts[1] = 0;
  CheckScales<<<32, 128>>>(counts, counts + 1);
  if (!CheckCuda(cudaGetLastError()) || !CheckCuda(cudaDeviceSynchronize())) {
    cudaFree(counts);
    return 1;
  }
  std::printf("mismatches=%d premature_rounding_witnesses=%d\n", counts[0], counts[1]);
  const bool pass = counts[0] == 0 && counts[1] > 0;
  return CheckCuda(cudaFree(counts)) && pass ? 0 : 1;
}
