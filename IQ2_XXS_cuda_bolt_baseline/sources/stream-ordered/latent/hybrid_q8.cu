#define launch_full_q8 launch_fallback_full_q8
#include "operators/q8/source_q8.cu"
#undef launch_full_q8
#include <atomic>

extern "C" int launch_native_source_projection(
    const void*, const uint8_t*, const void*, void*, int, int, int, int, int, void*);

namespace {
std::atomic<uint64_t> q8_native_m1_calls{0};
}

extern "C" uint64_t full_q8_native_m1_calls() {
  return q8_native_m1_calls.load(std::memory_order_relaxed);
}

extern "C" int launch_full_q8(
    const void* x, const uint8_t* packed, const void* scales, void* y,
    int rows, int n, int k, int source_rows, int position, void* opaque_stream) {
  if (source_rows != 1 || rows != 8)
    return launch_fallback_full_q8(
        x, packed, scales, y, rows, n, k, source_rows, position, opaque_stream);
  peak_bytes = 0;
  if (!x || !packed || !scales || !y || rows < 1 || rows > 8 || position < 0 ||
      !((n == 1024 && k == 4096) || (n == 4096 && k == 1024))) return -1;
  if (std::strcmp(full_q8_cublas_path(), SOURCE_CUBLAS_PATH)) return -3;
  size_t available = 0, total = 0;
  if (cudaMemGetInfo(&available, &total) != cudaSuccess) return -4;
  if (total - available + (128ull << 20) > (72ull << 30)) return -5;
  const int status = launch_native_source_projection(
      x, packed, scales, y, rows, n, k, 0, 1, opaque_stream);
  if (status) return status;
  // The native kernel owns no temporary buffers; consumers use the caller stream.
  q8_native_m1_calls.fetch_add(1, std::memory_order_relaxed);
  return 0;
}
