#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <algorithm>
#include "source_q8_dims.h"
#include "../projection_scratch.h"

#ifndef SOURCE_CUBLAS_PATH
#error "Pin source cuBLAS at build time"
#endif

namespace {
thread_local size_t peak_bytes = 0;
__global__ void decode(const uint8_t* packed, const __half* scales,
                       __nv_bfloat16* weight, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    weight[i] = __float2bfloat16_rn(
        float(int(packed[i]) - 128) * __half2float(scales[i / 32]));
}
__global__ void pad(const __nv_bfloat16* x, __nv_bfloat16* padded,
                    int rows, int m, int position, int k) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(m) * k) {
    const int row = i / k;
    padded[i] = row >= position && row < position + rows
        ? x[i - size_t(position) * k] : __float2bfloat16_rn(0.f);
  }
}
}

extern "C" const char* full_q8_wide_cublas_path() {
  Dl_info info{};
  using Gemm = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
      int, int, int, const void*, const void*, cudaDataType_t, int, const void*,
      cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
      cublasComputeType_t, cublasGemmAlgo_t);
  return dladdr(reinterpret_cast<void*>(static_cast<Gemm>(&cublasGemmEx)), &info)
      ? info.dli_fname : "";
}
extern "C" size_t full_q8_wide_last_peak_extra_bytes() { return peak_bytes; }
extern "C" size_t full_q8_wide_retained_scratch_bytes() {
  return projection_retained_bytes<__nv_bfloat16>();
}
extern "C" int full_q8_wide_release_scratch() {
  return projection_release_scratch<__nv_bfloat16>();
}

// Diagnostic ABI: device operands, host dimensions, caller's CUDA stream.
// Retain source M, absolute row position, and independent M1 cached projections.
extern "C" int launch_full_q8_wide(
    const void* x, const uint8_t* packed, const void* scales, void* y,
    int rows, int n, int k, int source_rows, int position, void* opaque_stream) {
  peak_bytes = 0;
  if (!x || !packed || !scales || !y || rows < 1 || rows > 8 ||
      !source_q8_shape(n, k) ||
      source_rows < 1 || source_rows > 135168 || position < 0 ||
      (source_rows != 1 && position >= source_rows)) return -1;
  if (std::strcmp(full_q8_wide_cublas_path(), SOURCE_CUBLAS_PATH)) return -3;
  constexpr size_t workspace_bytes = 32ull << 20;
  const size_t explicit_bytes = 2 * (size_t(n) * k + size_t(source_rows) * (k + n))
                                + workspace_bytes;
  if (explicit_bytes > (4ull << 30)) return -2;
  int status = 0;
  auto* cached = projection_scratch<__nv_bfloat16>(status);
  if (!cached) return status;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  ProjectionLease<__nv_bfloat16> lease(*cached, stream);
  auto& state = lease.state;
  const size_t weight_bytes = size_t(n) * k * 2;
  const size_t input_bytes = size_t(source_rows) * k * 2;
  const size_t output_bytes = size_t(source_rows) * n * 2;
  size_t projected = state.projected_bytes(
      weight_bytes, input_bytes, output_bytes, workspace_bytes);
  if (projected > (4ull << 30)) {
    if ((status = state.release())) return status;
    projected = explicit_bytes;
  }
  size_t before = 0, total = 0;
  if (cudaMemGetInfo(&before, &total) != cudaSuccess) return -4;
  // Reserve handle overhead, and fail before allocation if the caller is near
  // the full-model 72GiB device-residency limit.
  if (total - before + projected - state.bytes() + (128ull << 20) > (72ull << 30))
    return -5;
  auto sample = [&]() {
    size_t available = 0;
    if (cudaMemGetInfo(&available, &total) == cudaSuccess && available < before &&
        before - available > peak_bytes) peak_bytes = before - available;
  };
#define CUDA_CHECK(call) do { auto e = (call); if (e != cudaSuccess) return 1000 + e; } while (0)
#define BLAS_CHECK(call) do { auto e = (call); if (e != CUBLAS_STATUS_SUCCESS) return 2000 + e; } while (0)
  CUDA_CHECK(state.reserve(weight_bytes, input_bytes, output_bytes, workspace_bytes));
  if (!state.handle) BLAS_CHECK(cublasCreate(&state.handle));
  BLAS_CHECK(cublasSetStream(state.handle, stream));
  BLAS_CHECK(cublasSetWorkspace(state.handle, state.workspace, workspace_bytes));
  BLAS_CHECK(cublasSetMathMode(state.handle,
      static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                               CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
  sample();
  decode<<<(size_t(n) * k + 255) / 256, 256, 0, stream>>>(
      packed, static_cast<const __half*>(scales), state.weight, size_t(n) * k);
  CUDA_CHECK(cudaGetLastError());
  const float alpha = 1.f, beta = 0.f;
  CUDA_CHECK(cudaMemsetAsync(y, 0, size_t(rows) * n * 2, stream));
  const int iterations = source_rows == 1 ? rows : 1;
  for (int row = 0; row < iterations; ++row) {
    const auto* input = static_cast<const __nv_bfloat16*>(x) + size_t(row) * k;
    const int valid = source_rows == 1 ? 1 : std::min(rows, source_rows - position);
    if (source_rows != 1) {
      pad<<<(size_t(source_rows) * k + 255) / 256, 256, 0, stream>>>(
          input, state.input, valid, source_rows, position, k);
      CUDA_CHECK(cudaGetLastError());
      input = state.input;
    }
    // Pinned Torch dispatch: BF16 A/B/C, FP32 accumulation, tensor-op default.
    BLAS_CHECK(cublasGemmEx(state.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, source_rows, k,
        &alpha, state.weight, CUDA_R_16BF, k, input, CUDA_R_16BF, k,
        &beta, state.output, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    const size_t count = size_t(valid) * n;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<__nv_bfloat16*>(y) + size_t(row) * n,
        state.output + size_t(source_rows == 1 ? 0 : position) * n,
        count * 2, cudaMemcpyDeviceToDevice, stream));
    sample();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  lease.fenced = true;
  sample();
  return 0;
}
