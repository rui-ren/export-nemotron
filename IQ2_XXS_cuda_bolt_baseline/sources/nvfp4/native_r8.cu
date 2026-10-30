#define launch_native_nvfp4_m1 r8_probe_snapshot_launch
#define probe_native_nvfp4_codec r8_probe_original_codec
#define probe_native_nvfp4_record_values r8_probe_original_record_values
#include "baseline_snapshot.cu"
#undef launch_native_nvfp4_m1
#undef probe_native_nvfp4_codec
#undef probe_native_nvfp4_record_values
#include "native_r8_plan.h"

namespace {
template <int OUTPUT_ROWS>
__global__ void input_reuse(
    const __nv_bfloat16* x, const uint8_t* packed, const uint8_t* scales,
    const float* global, __nv_bfloat16* output) {
  const int lane = threadIdx.x % 16;
  const int tile_index = threadIdx.x / 16;
  const int base_row = blockIdx.x * OUTPUT_ROWS;
  __shared__ float partials[OUTPUT_ROWS][8][8][16];
  float tile[OUTPUT_ROWS][8] = {};
  for (int group = 0; group < 32; group += 4) {
    float weights[OUTPUT_ROWS][4];
#pragma unroll
    for (int p = 0; p < OUTPUT_ROWS; ++p)
      record_weights(packed, scales, base_row + p, 4096,
                     tile_index * 512 + group * 16, lane, weights[p]);
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int column = tile_index * 512 + lane + (group + step) * 16;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        // Reuse one activation across output rows, not partial accumulators.
        const float activation = __bfloat162float(x[size_t(r) * 4096 + column]);
#pragma unroll
        for (int p = 0; p < OUTPUT_ROWS; ++p)
          tile[p][r] = __fmaf_rn(weights[p][step], activation, tile[p][r]);
      }
    }
  }
#pragma unroll
  for (int p = 0; p < OUTPUT_ROWS; ++p)
#pragma unroll
    for (int r = 0; r < 8; ++r) partials[p][tile_index][r][lane] = tile[p][r];
  __syncthreads();
  // Each 16-lane subgroup reduces a distinct output row. Preserve +0, then
  // ordered tile additions, then the source serial lane 0..15 additions.
  if (tile_index < OUTPUT_ROWS) {
    const int p = tile_index;
    const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      float total = 0.f;
#pragma unroll
      for (int t = 0; t < 8; ++t)
        total = __fadd_rn(total, partials[p][t][r][lane]);
      float sum = __shfl_sync(mask, total, 0, 16);
#pragma unroll
      for (int i = 1; i < 16; ++i)
        sum = __fadd_rn(sum, __shfl_sync(mask, total, i, 16));
      if (lane == 0)
        output[size_t(r) * 18560 + base_row + p] =
            __float2bfloat16_rn(__fmul_rn(sum, global[0]));
    }
  }
}

template <int TILE, bool RECORD>
__global__ void output_shared(
    const __nv_bfloat16* x, const uint8_t* packed, const uint8_t* scales,
    const float* global, __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 weights[TILE];
  const int row = blockIdx.x, lane = threadIdx.x % 16, r = threadIdx.x / 16;
  float total = 0.f;
#pragma unroll 1
  for (int begin = 0; begin < 8192; begin += TILE) {
    if constexpr (RECORD) {
#pragma unroll
      for (int record = threadIdx.x / 16 * 64; record < TILE; record += 512) {
        float values[4];
        record_weights(packed, scales, row, 8192, begin + record, lane, values);
#pragma unroll
        for (int step = 0; step < 4; ++step)
          weights[record + lane + step * 16] = __float2bfloat16_rn(values[step]);
      }
    } else {
#pragma unroll
      for (int i = threadIdx.x; i < TILE; i += 128) {
        const int column = begin + i;
        weights[i] = __float2bfloat16_rn(original_complete_weight(
            packed[size_t(row) * 4096 + column / 2],
            scales[size_t(row) * 512 + column / 16], column & 1));
      }
    }
    __syncthreads();
    // The accumulator is continuous across tiles: 512 ascending ordered FMAs.
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

template <int OUTPUT_ROWS>
int input_launch(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output ||
      !native_r8_plan::input(rows, n, k)) return -1;
  input_reuse<OUTPUT_ROWS><<<18560 / OUTPUT_ROWS, 128, 0,
                            static_cast<cudaStream_t>(opaque_stream)>>>(
      static_cast<const __nv_bfloat16*>(x), packed, scales, global,
      static_cast<__nv_bfloat16*>(output));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

template <int TILE, bool RECORD>
int output_launch(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output ||
      !native_r8_plan::output(rows, n, k)) return -1;
  output_shared<TILE, RECORD><<<4096, 128, 0,
                               static_cast<cudaStream_t>(opaque_stream)>>>(
      static_cast<const __nv_bfloat16*>(x), packed, scales, global,
      static_cast<__nv_bfloat16*>(output));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
}  // namespace

#define INPUT_ENTRY(NAME, ROWS) \
extern "C" int NAME(const void* x, const uint8_t* p, const uint8_t* s, const float* g, \
                   void* y, int r, int n, int k, void* stream) { \
  return input_launch<ROWS>(x, p, s, g, y, r, n, k, stream); \
}
INPUT_ENTRY(launch_input_reuse1, 1)
INPUT_ENTRY(launch_input_reuse2, 2)
INPUT_ENTRY(launch_input_reuse4, 4)
INPUT_ENTRY(launch_input_reuse8, 8)
#undef INPUT_ENTRY

#define OUTPUT_ENTRY(NAME, TILE, RECORD) \
extern "C" int NAME(const void* x, const uint8_t* p, const uint8_t* s, const float* g, \
                   void* y, int r, int n, int k, void* stream) { \
  return output_launch<TILE, RECORD>(x, p, s, g, y, r, n, k, stream); \
}
OUTPUT_ENTRY(launch_output_scalar256, 256, false)
OUTPUT_ENTRY(launch_output_scalar512, 512, false)
OUTPUT_ENTRY(launch_output_record512, 512, true)
OUTPUT_ENTRY(launch_output_record1024, 1024, true)
#undef OUTPUT_ENTRY
