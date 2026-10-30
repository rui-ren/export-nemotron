// Immutable accepted implementation supplies the codec, probes and out fallback.
#define launch_native_nvfp4_m1 accepted_snapshot_launch
#include "baseline_snapshot.cu"
#undef launch_native_nvfp4_m1

namespace {
template <int R, int RowsPerBlock>
__global__ void tile_parallel_input(
    const __nv_bfloat16* x, const uint8_t* packed, const uint8_t* scales,
    const float* global, __nv_bfloat16* output) {
  const int local_row = threadIdx.x / 128;
  const int row = blockIdx.x * RowsPerBlock + local_row;
  const int lane = threadIdx.x % 16;
  const int tile_index = (threadIdx.x % 128) / 16;
  // Eight independent K512 chains, one 16-lane group per source tile.
  __shared__ float partials[RowsPerBlock][8][R][16];
  float tile[R] = {};
  for (int group = 0; group < 32; group += 4) {
    float weights[4];
    if constexpr (R == 8)
      record_weights(packed, scales, row, 4096,
                     tile_index * 512 + group * 16, lane, weights);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int column = tile_index * 512 + lane + (group + step) * 16;
      float weight;
      if constexpr (R == 8) weight = weights[step];
      else weight = original_complete_weight(
          packed[size_t(row) * 2048 + column / 2],
          scales[size_t(row) * 256 + column / 16], column & 1);
#pragma unroll
      for (int r = 0; r < R; ++r)
        tile[r] = __fmaf_rn(
            weight, __bfloat162float(x[size_t(r) * 4096 + column]), tile[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) partials[local_row][tile_index][r][lane] = tile[r];
  __syncthreads();
  // Do not collapse the initial +0 or reassociate tile/lane additions.
  if (tile_index == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float total = 0.f;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        total = __fadd_rn(total, partials[local_row][t][r][lane]);
      float sum = __shfl_sync(0x0000ffffu, total, 0, 16);
#pragma unroll
      for (int i = 1; i < 16; ++i)
        sum = __fadd_rn(sum, __shfl_sync(0x0000ffffu, total, i, 16));
      if (lane == 0)
        output[size_t(r) * 18560 + row] =
            __float2bfloat16_rn(__fmul_rn(sum, global[0]));
    }
  }
}
}

extern "C" int launch_native_nvfp4_m1(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output || (rows != 1 && rows != 8))
    return -1;
  if (n != 18560 || k != 4096)
    return accepted_snapshot_launch(x, packed, scales, global, output,
                                    rows, n, k, opaque_stream);
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  if (rows == 1)
    tile_parallel_input<1, 2><<<9280, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), packed, scales, global,
        static_cast<__nv_bfloat16*>(output));
  else
    tile_parallel_input<8, 1><<<18560, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), packed, scales, global,
        static_cast<__nv_bfloat16*>(output));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
