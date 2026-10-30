#include <cuda.h>

#include <atomic>
#include <cstdio>
#include <thread>

namespace {
std::atomic<int> loads{0}, unloads{0};
std::atomic<int> attempts{0}, fail_at{0};
std::atomic<int> failures{0};
CUresult CountedLoad(CUmodule* module, const void* image) {
  if (++attempts == fail_at) return CUDA_ERROR_INVALID_IMAGE;
  const auto status = cuModuleLoadData(module, image);
  if (status == CUDA_SUCCESS) ++loads;
  return status;
}
CUresult CountedUnload(CUmodule module) {
  const auto status = cuModuleUnload(module);
  if (status == CUDA_SUCCESS) ++unloads;
  return status;
}
}  // namespace

#define cuModuleLoadData CountedLoad
#define cuModuleUnload CountedUnload
#include "bolt_mamba_e6.cu"
#undef cuModuleLoadData
#undef cuModuleUnload

int main() {
  if (cuInit(0) != CUDA_SUCCESS) return 1;
  CUdevice device;
  CUcontext context;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS) return 1;
  setenv("BOLT_MAMBA_CUMSUM_HEAD_TILE", "1", 1);
  for (int i = 0; i < 6; ++i) {
    std::thread worker([&]() {
      try {
        CheckSourceDriver(cuCtxSetCurrent(context));
        auto& source = SourceSsdForContext();
        auto& parallel = E6Modules();
        if (&source != &SourceSsdForContext() || &parallel != &E6Modules()) ++failures;
        // Leave no context current; cleanup must restore the owning context itself.
        CheckSourceDriver(cuCtxSetCurrent(nullptr));
      } catch (...) {
        ++failures;
      }
    });
    worker.join();
    if (loads != 4 * (i + 1) || unloads != loads) ++failures;
  }
  std::thread failed_constructor([&]() {
    CheckSourceDriver(cuCtxSetCurrent(context));
    // Exercise partial construction: the first member loads, the second fails.
    fail_at = attempts + 2;
    try {
      SourceSsdForContext();
      ++failures;
    } catch (const std::exception&) {
    }
  });
  failed_constructor.join();
  if (loads != unloads) ++failures;
  cuDevicePrimaryCtxRelease(device);
  std::printf("loads=%d unloads=%d failures=%d\n", int(loads), int(unloads), int(failures));
  return failures ? 1 : 0;
}
