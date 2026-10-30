#define launch_full_q8_wide launch_fallback_full_q8_wide
#include "operators/q8wide/source_q8.cu"
#undef launch_full_q8_wide
#include <atomic>

extern "C" int launch_native_source_projection(
    const void*, const uint8_t*, const void*, void*, int, int, int, int, int, void*);

namespace {
std::atomic<uint64_t> wide_native_m1_calls{0};
}

extern "C" uint64_t full_q8_wide_native_m1_calls() {
  return wide_native_m1_calls.load(std::memory_order_relaxed);
}

extern "C" int launch_full_q8_wide(
    const void* x, const uint8_t* packed, const void* scales, void* y,
    int rows, int n, int k, int source_rows, int position, void* opaque_stream) {
  if (source_rows != 1 || rows != 8)
    return launch_fallback_full_q8_wide(
        x, packed, scales, y, rows, n, k, source_rows, position, opaque_stream);
  peak_bytes = 0;
  if (!x || !packed || !scales || !y || position < 0 ||
      !source_q8_shape(n, k)) return -1;
  if (std::strcmp(full_q8_wide_cublas_path(), SOURCE_CUBLAS_PATH)) return -3;
  size_t available = 0, total = 0;
  if (cudaMemGetInfo(&available, &total) != cudaSuccess) return -4;
  if (total - available + (128ull << 20) > (72ull << 30)) return -5;
  const int status = launch_native_source_projection(
      x, packed, scales, y, rows, n, k, 0, 1, opaque_stream);
  if (status) return status;
  // The native kernel owns no temporary buffers; consumers use the caller stream.
  wide_native_m1_calls.fetch_add(1, std::memory_order_relaxed);
  return 0;
}
