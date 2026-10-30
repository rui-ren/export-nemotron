#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
__global__ void router_logits(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    float* output, int rows, int experts) {
  const int lane = threadIdx.x % 32;
  const int expert = blockIdx.x * 4 + threadIdx.x / 32;
  float sums[8] = {};
  // Observed source GEMVX: 32 lanes, a continuous stride-32 FP32 chain.
  for (int begin = 0; begin < 128; begin += 4) {
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int column = lane + (begin + step) * 32;
      const float w = __bfloat162float(weight[size_t(expert) * 4096 + column]);
#pragma unroll
      for (int row = 0; row < 8; ++row)
        if (row < rows)
          sums[row] = __fmaf_rn(
              w, __bfloat162float(input[size_t(row) * 4096 + column]), sums[row]);
    }
  }
#pragma unroll
  for (int row = 0; row < 8; ++row) {
    if (row < rows) {
      float sum = sums[row];
#pragma unroll
      for (int offset = 16; offset; offset /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
      if (lane == 0) output[row * experts + expert] = sum;
    }
  }
}

__global__ void normalize_router(
    const float* scores, const int64_t* ids, int32_t* output_ids,
    float* weights, int experts) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int id = lane < 9 ? int(ids[row * 9 + lane]) : 0;
  const float selected = lane < 9 ? scores[row * experts + id] : 0.f;
  float sum = selected;
  // Source Reduce.cuh uses eight threads: lane 0 first accumulates score 8,
  // then offsets 4,2,1. Zero padding a warp gives the same positive-FP32 order.
#pragma unroll
  for (int offset = 16; offset; offset /= 2)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
  const float denominator = __fadd_rn(__shfl_sync(0xffffffffu, sum, 0), 1e-20f);
  if (lane < 9) {
    output_ids[row * 9 + lane] = id;
    weights[row * 9 + lane] = __fmul_rn(__fdiv_rn(selected, denominator), 5.f);
  }
}

bool valid_experts(int e) {
  return e == 240 || e == 304 || e == 368 || e == 432 || e == 496;
}
}

extern "C" int launch_native_router_logits(
    const void* input, const void* weight, float* output,
    int rows, int experts, void* opaque_stream) {
  if (!input || !weight || !output || rows < 1 || rows > 8 || !valid_experts(experts))
    return -1;
  router_logits<<<experts / 4, 128, 0, static_cast<cudaStream_t>(opaque_stream)>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<const __nv_bfloat16*>(weight),
      output, rows, experts);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

extern "C" int launch_native_router_normalize(
    const float* scores, const int64_t* ids, int32_t* output_ids, float* weights,
    int rows, int experts, void* opaque_stream) {
  if (!scores || !ids || !output_ids || !weights ||
      rows < 1 || rows > 8 || !valid_experts(experts)) return -1;
  normalize_router<<<rows, 32, 0, static_cast<cudaStream_t>(opaque_stream)>>>(
      scores, ids, output_ids, weights, experts);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
