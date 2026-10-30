#include "baseline_snapshot.cu"

namespace {
__device__ __forceinline__ __nv_bfloat16 complete_bf16(
    uint8_t packed, uint8_t scale, int parity) {
  // original_complete_weight already rounds BF16; this store is lossless,
  // including the sign of zero, and does not apply the global scale.
  return __float2bfloat16_rn(original_complete_weight(packed, scale, parity));
}

template <int TILE>
__global__ void r8_shared_geometry(
    const __nv_bfloat16* x, const uint8_t* packed, const uint8_t* scales,
    const float* global, __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 weights[TILE];
  const int row = blockIdx.x;
  const int lane = threadIdx.x % 16;
  const int r = threadIdx.x / 16;
  float total = 0.f;
#pragma unroll 1
  for (int begin = 0; begin < 8192; begin += TILE) {
#pragma unroll
    for (int i = threadIdx.x; i < TILE; i += 128) {
      const int column = begin + i;
      weights[i] = complete_bf16(
          packed[size_t(row) * 4096 + column / 2],
          scales[size_t(row) * 512 + column / 16], column & 1);
    }
    __syncthreads();
    // Each R subgroup consumes ascending source K lanes. The accumulator is
    // never reset at a tile boundary: 512 ordered FP32 FMAs per thread.
#pragma unroll
    for (int i = lane; i < TILE; i += 16)
      total = __fmaf_rn(
          __bfloat162float(weights[i]),
          __bfloat162float(x[size_t(r) * 8192 + begin + i]), total);
    __syncthreads();
  }
#pragma unroll
  for (int offset = 8; offset; offset /= 2)
    total = __fadd_rn(total, __shfl_down_sync(0xffffffffu, total, offset, 16));
  if (lane == 0)
    output[size_t(r) * 4096 + row] =
        __float2bfloat16_rn(__fmul_rn(total, global[0]));
}

template <int TILE>
int dispatch_geometry(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output ||
      rows != 8 || n != 4096 || k != 8192) return -1;
  r8_shared_geometry<TILE><<<4096, 128, 0,
                           static_cast<cudaStream_t>(opaque_stream)>>>(
      static_cast<const __nv_bfloat16*>(x), packed, scales, global,
      static_cast<__nv_bfloat16*>(output));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

__global__ void geometry_codec(float* original, float* actual, int parity) {
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    const uint8_t packed = uint8_t((i & 15) * 17);
    original[i] = original_complete_weight(packed, i >> 4, parity);
    actual[i] = __bfloat162float(complete_bf16(packed, i >> 4, parity));
  }
}

__global__ void geometry_values(
    const uint8_t* packed, const uint8_t* scales, float* output) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(4096) * 8192) return;
  const int row = index / 8192, column = index % 8192;
  output[index] = __bfloat162float(complete_bf16(
      packed[size_t(row) * 4096 + column / 2],
      scales[size_t(row) * 512 + column / 16], column & 1));
}
}

extern "C" int launch_r8_tile128(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* stream) {
  return dispatch_geometry<128>(x, packed, scales, global, output, rows, n, k, stream);
}

extern "C" int launch_r8_tile256(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* stream) {
  return dispatch_geometry<256>(x, packed, scales, global, output, rows, n, k, stream);
}

extern "C" int probe_geometry_codec(
    float* original, float* actual, int parity, void* stream) {
  if (!original || !actual || parity < 0 || parity > 1) return -1;
  geometry_codec<<<1, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      original, actual, parity);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

extern "C" int probe_geometry_values(
    const uint8_t* packed, const uint8_t* scales, float* output, void* stream) {
  if (!packed || !scales || !output) return -1;
  geometry_values<<<(4096 * 8192) / 256, 256, 0,
                    static_cast<cudaStream_t>(stream)>>>(packed, scales, output);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
