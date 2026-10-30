// The included source is read-only and its digest is pinned by build.sh/probe.py.
#include "baseline_snapshot.cu"

namespace {
template <int P>
__device__ __forceinline__ void load_ahead(
    const uint8_t* packed, const uint8_t* scales, int row, int lane,
    int begin, float (&weights)[P]) {
#pragma unroll
  for (int step = 0; step < P; ++step) {
    const int column = lane + (begin + step) * 16;
    // Keep the accepted scalar codec/load path, including complete BF16 rounding.
    weights[step] = original_complete_weight(
        packed[size_t(row) * 4096 + column / 2],
        scales[size_t(row) * 512 + column / 16], column & 1);
  }
}

template <int R, int P>
__global__ void out_lookahead(
    const __nv_bfloat16* x, const uint8_t* packed, const uint8_t* scales,
    const float* global, __nv_bfloat16* output) {
  const int lane = threadIdx.x % 16;
  const int row = blockIdx.x * 8 + threadIdx.x / 16;
  float total[R] = {};
  float current[P];
  load_ahead<P>(packed, scales, row, lane, 0, current);
  // Only codec/load scheduling changes: each lane still executes 512 ordered
  // FMAs per physical row. There are no independent partial accumulators.
#pragma unroll 1
  for (int begin = 0; begin < 512; begin += P) {
    float upcoming[P];
    if (begin + P < 512)
      load_ahead<P>(packed, scales, row, lane, begin + P, upcoming);
#pragma unroll
    for (int step = 0; step < P; ++step) {
      const int column = lane + (begin + step) * 16;
#pragma unroll
      for (int r = 0; r < R; ++r)
        total[r] = __fmaf_rn(
            current[step], __bfloat162float(x[size_t(r) * 8192 + column]), total[r]);
    }
#pragma unroll
    for (int step = 0; step < P; ++step)
      if (begin + P < 512) current[step] = upcoming[step];
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float sum = total[r];
#pragma unroll
    for (int offset = 8; offset; offset /= 2)
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset, 16));
    if (lane == 0)
      output[size_t(r) * 4096 + row] =
          __float2bfloat16_rn(__fmul_rn(sum, global[0]));
  }
}

// Instrument the exact helper used above, never a predecoded runtime path.
__global__ void out_complete_values(
    const uint8_t* packed, const uint8_t* scales, float* output) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(4096) * 8192) return;
  const int row = index / 8192, column = index % 8192;
  output[index] = original_complete_weight(
      packed[size_t(row) * 4096 + column / 2],
      scales[size_t(row) * 512 + column / 16], column & 1);
}

template <int P>
int dispatch_out(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output ||
      (rows != 1 && rows != 8) || n != 4096 || k != 8192) return -1;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  if (rows == 1)
    out_lookahead<1, P><<<512, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), packed, scales, global,
        static_cast<__nv_bfloat16*>(output));
  else
    out_lookahead<8, P><<<512, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), packed, scales, global,
        static_cast<__nv_bfloat16*>(output));
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
}

extern "C" int launch_out_lookahead1(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* stream) {
  return dispatch_out<1>(x, packed, scales, global, output, rows, n, k, stream);
}

extern "C" int launch_out_lookahead4(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* stream) {
  return dispatch_out<4>(x, packed, scales, global, output, rows, n, k, stream);
}

extern "C" int probe_out_complete_values(
    const uint8_t* packed, const uint8_t* scales, float* output, void* stream) {
  if (!packed || !scales || !output) return -1;
  out_complete_values<<<(4096 * 8192) / 256, 256, 0,
                        static_cast<cudaStream_t>(stream)>>>(packed, scales, output);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
