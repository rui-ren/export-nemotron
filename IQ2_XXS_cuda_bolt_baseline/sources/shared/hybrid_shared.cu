#define launch_full_iq2 launch_fallback_full_iq2
#define full_iq2_retained_scratch_bytes fallback_iq2_retained_scratch_bytes
#define full_iq2_release_scratch fallback_iq2_release_scratch
#include "/home/mobius/bolt-native-split-projection-20261028/operators/iq2/source_iq2.cu"
#undef launch_full_iq2
#undef full_iq2_retained_scratch_bytes
#undef full_iq2_release_scratch
#include <atomic>
#include <memory>
#include <vector>

extern "C" int launch_native_source_projection(
    const void*, const uint8_t*, const void*, void*, int, int, int, int, int, void*);
extern "C" int launch_native_split_projection(
    const void*, const uint8_t*, const void*, float*, void*, int, int, int, int, void*);

namespace {
constexpr size_t partial_bytes = 2 * 8 * 4096 * sizeof(float);
std::atomic<uint64_t> shared_native_r8_calls{0}, shared_parallel_calls{0};

struct PartialScratch {
  int device;
  float* data = nullptr;
  cudaEvent_t done = nullptr;
  bool recorded = false, poisoned = false;
  explicit PartialScratch(int value) : device(value) {}
  int release() {
    int current = -1;
    auto error = cudaGetDevice(&current);
    if (error != cudaSuccess) return 1000 + error;
    error = cudaSetDevice(device);
    if (error != cudaSuccess) return 1000 + error;
    auto first = recorded ? cudaEventSynchronize(done) : cudaSuccess;
    if (data) {
      error = cudaFree(data);
      if (first == cudaSuccess) first = error;
      data = nullptr;
    }
    if (done) {
      error = cudaEventDestroy(done);
      if (first == cudaSuccess) first = error;
      done = nullptr;
    }
    recorded = false;
    error = cudaSetDevice(current);
    if (first == cudaSuccess) first = error;
    return first == cudaSuccess ? 0 : 1000 + first;
  }
  ~PartialScratch() { release(); }
};

// One bounded buffer per host thread/device. An event also orders reuse after
// a caller switches CUDA streams; the retained bytes never contain weights.
thread_local std::vector<std::unique_ptr<PartialScratch>> partial_scratch;

PartialScratch* acquire_partials(cudaStream_t stream, int& status) {
  int device = -1;
  auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) { status = 1000 + error; return nullptr; }
  if (partial_scratch.size() <= size_t(device)) partial_scratch.resize(device + 1);
  auto& slot = partial_scratch[device];
  if (!slot) slot = std::make_unique<PartialScratch>(device);
  if (slot->poisoned) { status = -8; return nullptr; }
  if (!slot->done) {
    error = cudaEventCreateWithFlags(&slot->done, cudaEventDisableTiming);
    if (error != cudaSuccess) { status = 1000 + error; return nullptr; }
  }
  if (!slot->data) {
    error = cudaMalloc(&slot->data, partial_bytes);
    if (error != cudaSuccess) { status = 1000 + error; return nullptr; }
  }
  if (slot->recorded) {
    error = cudaStreamWaitEvent(stream, slot->done, 0);
    if (error != cudaSuccess) { status = 1000 + error; return nullptr; }
  }
  return slot.get();
}

size_t retained_partials() {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess ||
      size_t(device) >= partial_scratch.size() || !partial_scratch[device] ||
      !partial_scratch[device]->data) return 0;
  return partial_bytes;
}
}

extern "C" uint64_t full_iq2_native_m1_calls() {
  return shared_native_r8_calls.load(std::memory_order_relaxed);
}
extern "C" uint64_t full_iq2_parallel_m1_calls() {
  return shared_parallel_calls.load(std::memory_order_relaxed);
}
extern "C" size_t full_iq2_parallel_retained_bytes() { return retained_partials(); }
extern "C" size_t full_iq2_retained_scratch_bytes() {
  return fallback_iq2_retained_scratch_bytes() + retained_partials();
}
extern "C" int full_iq2_release_scratch() {
  int status = fallback_iq2_release_scratch();
  int device = -1;
  const auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) return status ? status : 1000 + error;
  if (size_t(device) < partial_scratch.size() && partial_scratch[device]) {
    const int partial_status = partial_scratch[device]->release();
    partial_scratch[device].reset();
    if (!status) status = partial_status;
  }
  return status;
}

extern "C" int launch_full_iq2(
    const void* x, const uint8_t* packed, const void* scales, void* y,
    int rows, int n, int k, int source_rows, int position, void* opaque_stream) {
  if (source_rows != 1 ||
      !((n == 5376 && k == 4096) || (n == 4096 && k == 5376)))
    return launch_fallback_full_iq2(
        x, packed, scales, y, rows, n, k, source_rows, position, opaque_stream);
  peak_bytes = 0;
  if (!x || !packed || !scales || !y || rows < 1 || rows > 8 || position < 0) return -1;
  if (std::strcmp(full_iq2_cublas_path(), SOURCE_CUBLAS_PATH)) return -3;
  size_t available = 0, total = 0;
  if (cudaMemGetInfo(&available, &total) != cudaSuccess) return -4;
  const size_t extra = n == 4096 && !retained_partials() ? partial_bytes : 0;
  if (total - available + extra + (128ull << 20) > (72ull << 30)) return -5;
  int status = 0;
  if (n == 4096) {
    auto stream = static_cast<cudaStream_t>(opaque_stream);
    auto* slot = acquire_partials(stream, status);
    if (!slot) return status;
    if (extra) {
      size_t after = 0;
      if (cudaMemGetInfo(&after, &total) == cudaSuccess && after < available)
        peak_bytes = available - after;
    }
    status = launch_native_split_projection(
        x, packed, scales, slot->data, y, rows, n, k, 1, opaque_stream);
    const auto error = cudaEventRecord(slot->done, stream);
    if (error != cudaSuccess) {
      slot->poisoned = true;
      cudaStreamSynchronize(stream);
      return status ? status : 1000 + error;
    }
    slot->recorded = true;
    if (!status) shared_parallel_calls.fetch_add(1, std::memory_order_relaxed);
  } else {
    status = launch_native_source_projection(
        x, packed, scales, y, rows, n, k, 1, 1, opaque_stream);
  }
  if (!status && rows == 8)
    shared_native_r8_calls.fetch_add(1, std::memory_order_relaxed);
  return status;
}
