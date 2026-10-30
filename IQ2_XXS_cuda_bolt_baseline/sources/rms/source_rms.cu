#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Match the pinned PyTorch float mean's vector-four, four-accumulator tree.
__global__ void source_rms(const __nv_bfloat16* x, const __nv_bfloat16* gamma,
                           __nv_bfloat16* y, float* variance, int columns,
                           float epsilon) {
  __shared__ float scratch[512];
  const int lane = threadIdx.x;
  const int width = blockDim.x;
  const int row = blockIdx.x;
  float sums[4] = {0.f, 0.f, 0.f, 0.f};
  for (int vector = lane; vector * 4 + 3 < columns; vector += width) {
    #pragma unroll
    for (int component = 0; component < 4; ++component) {
      const float value = __bfloat162float(x[row * columns + vector * 4 + component]);
      sums[component] = __fadd_rn(sums[component], __fmul_rn(value, value));
    }
  }
  float sum = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
  scratch[lane] = sum;
  for (int offset = width / 2; offset >= 32; offset /= 2) {
    __syncthreads();
    if (lane < offset) {
      sum = __fadd_rn(sum, scratch[lane + offset]);
      scratch[lane] = sum;
    }
  }
  __syncthreads();
  if (lane < 32) {
    for (int offset = 16; offset > 0; offset /= 2) {
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (lane == 0) {
      const float mean = __fmul_rn(sum, 1.f / columns);
      if (variance) variance[row] = mean;
      scratch[0] = rsqrtf(__fadd_rn(mean, epsilon));
    }
  }
  __syncthreads();
  const float inverse = scratch[0];
  for (int column = lane; column < columns; column += width) {
    const float normalized = __fmul_rn(__bfloat162float(x[row * columns + column]), inverse);
    y[row * columns + column] = __float2bfloat16_rn(
        __fmul_rn(normalized, __bfloat162float(gamma[column])));
  }
}

extern "C" int launch_source_rms(const void* x, const void* gamma, void* y,
                                float* variance, int rows, int columns,
                                int source_rows, float epsilon, void* stream) {
  if (rows <= 0 || columns != 4096 || source_rows <= 0) return cudaErrorInvalidValue;
  int height = 1;
  while (height * 2 <= source_rows && height < 16) height *= 2;
  const int width = 512 / height;
  source_rms<<<rows, width, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(gamma),
      static_cast<__nv_bfloat16*>(y), variance, columns, epsilon);
  return cudaGetLastError();
}
