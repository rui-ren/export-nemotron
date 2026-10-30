#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <algorithm>
#include "projection_scratch.h"

#ifndef SOURCE_CUBLAS_PATH
#error "Build with the pinned source cuBLAS path"
#endif

extern "C" int launch_native_nvfp4_m1(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);

namespace {
thread_local size_t last_peak_extra_bytes = 0;
__device__ float e4m3(uint8_t v) {
  const int e = (v >> 3) & 15, m = v & 7;
  if (e == 15 && m == 7) return nanf("");
  const float a = e ? ldexpf(1.f + m * .125f, e - 7) : m * 0.001953125f;
  return (v & 128) ? -a : a;
}
__global__ void decode(const uint8_t* w, const uint8_t* s, float* out, size_t count, int k) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint8_t code = (w[i / 2] >> ((i & 1) * 4)) & 15;
  const float lut[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
  const float local = ((code & 8) ? -lut[code & 7] : lut[code & 7]) *
                      e4m3(s[(i / k) * (k / 16) + (i % k) / 16]);
  out[i] = __bfloat162float(__float2bfloat16_rn(local));
}
__global__ void cast_pad(const __nv_bfloat16* x, float* out,
                         int valid, int m, int position, int k) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(m) * k) {
    const int row = i / k;
    out[i] = row >= position && row < position + valid
        ? __bfloat162float(x[i - size_t(position) * k]) : 0.f;
  }
}
__global__ void epilogue(const float* dot, const float* global, __nv_bfloat16* y,
                         size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) y[i] = __float2bfloat16_rn(dot[i] * global[0]);
}
}

extern "C" const char* full_nvfp4_cublas_path() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&cublasSgemm), &info)) return "";
  return info.dli_fname;
}

extern "C" size_t full_nvfp4_last_peak_extra_bytes() {
  return last_peak_extra_bytes;
}
extern "C" size_t full_nvfp4_retained_scratch_bytes() {
  return projection_retained_bytes<float>();
}
extern "C" int full_nvfp4_release_scratch() {
  return projection_release_scratch<float>();
}

// All pointers except source_rows (passed by value) are device pointers.
// Scratch persists, but weight contents are rebuilt. Not graph-capture safe.
extern "C" int launch_full_nvfp4(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, int source_rows, int position, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output || rows <= 0 || n <= 0 ||
      rows > 8 || k <= 0 || k % 16 || source_rows < 1 || position < 0 ||
      (source_rows != 1 && position >= source_rows)) return -1;
  // A parent process may already have loaded a different library with SONAME 13.
  // Fail closed rather than silently losing source GEMM qualification.
  if (std::strcmp(full_nvfp4_cublas_path(), SOURCE_CUBLAS_PATH)) return -3;
  const int m = source_rows;
  if (n > 32768 || k > 32768 || m > 32768) return -2;
  const size_t bytes = (size_t(n) * k + size_t(m) * (k + n)) * sizeof(float);
  if (bytes > (2ull << 30)) return -2;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  if (m == 1 && (rows == 1 || rows == 8) &&
      ((n == 18560 && k == 4096) || (n == 4096 && k == 8192))) {
    size_t available = 0, total = 0;
    last_peak_extra_bytes = 0;
    if (cudaMemGetInfo(&available, &total) != cudaSuccess) return -4;
    if (total - available + (128ull << 20) > (72ull << 30)) return -5;
    const int native_status = launch_native_nvfp4_m1(
        x, packed, scales, global, output, rows, n, k, opaque_stream);
    if (native_status) return native_status;
    // No scratch or host readback: ORT orders consumers on this same stream.
    return 0;
  }
  int status = 0;
  auto* cached = projection_scratch<float>(status);
  if (!cached) return status;
  ProjectionLease<float> lease(*cached, stream);
  auto& state = lease.state;
  const size_t weight_bytes = size_t(n) * k * 4;
  const size_t input_bytes = size_t(m) * k * 4;
  const size_t output_bytes = size_t(m) * n * 4;
  size_t projected = state.projected_bytes(weight_bytes, input_bytes, output_bytes, 0);
  if (projected > (2ull << 30)) {
    if ((status = state.release())) return status;
    projected = bytes;
  }
  size_t free_before = 0, total = 0;
  last_peak_extra_bytes = 0;
  if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) return -4;
  if (total - free_before + projected - state.bytes() + (128ull << 20) > (72ull << 30))
    return -5;
  auto sample_memory = [&]() {
    size_t available = 0;
    if (cudaMemGetInfo(&available, &total) == cudaSuccess && available < free_before) {
      const size_t used = free_before - available;
      if (used > last_peak_extra_bytes) last_peak_extra_bytes = used;
    }
  };
  cudaError_t ce;
  if ((ce = cudaMemsetAsync(output, 0, size_t(rows) * n * 2, stream)) != cudaSuccess)
    return 1000 + ce;
  if ((ce = state.reserve(weight_bytes, input_bytes, output_bytes, 0)) != cudaSuccess)
    return 1000 + ce;
  cublasStatus_t be;
  if (!state.handle && (be = cublasCreate(&state.handle)) != CUBLAS_STATUS_SUCCESS)
    return 2000 + be;
  if ((be = cublasSetStream(state.handle, stream)) != CUBLAS_STATUS_SUCCESS) return 2000 + be;
  if ((be = cublasSetMathMode(state.handle, CUBLAS_DEFAULT_MATH)) != CUBLAS_STATUS_SUCCESS)
    return 2000 + be;
  sample_memory();
  decode<<<(size_t(n) * k + 255) / 256, 256, 0, stream>>>(
      packed, scales, state.weight, size_t(n) * k, k);
  const float alpha = 1.f, beta = 0.f;
  const int iterations = m == 1 ? rows : 1;
  for (int row = 0; row < iterations; ++row) {
    const int valid = m == 1 ? 1 : std::min(rows, m - position);
    const int start = m == 1 ? 0 : position;
    cast_pad<<<(size_t(m) * k + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x) + size_t(row) * k,
        state.input, valid, m, start, k);
    // F.linear(X[M,K], W[N,K]) dispatch: column-major W^T * X.
    be = cublasSgemm(state.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                    &alpha, state.weight, k, state.input, k, &beta, state.output, n);
    if (be != CUBLAS_STATUS_SUCCESS) return 2000 + be;
    sample_memory();
    epilogue<<<(size_t(valid) * n + 255) / 256, 256, 0, stream>>>(
        state.output + size_t(start) * n, global,
        static_cast<__nv_bfloat16*>(output) + size_t(row) * n,
        size_t(valid) * n);
  }
  if ((ce = cudaGetLastError()) != cudaSuccess) return 1000 + ce;
  // Keep the baseline completion boundary before sharing the arena with another call.
  ce = cudaStreamSynchronize(stream);
  lease.fenced = ce == cudaSuccess;
  sample_memory();
  return ce == cudaSuccess ? 0 : 1000 + ce;
}
