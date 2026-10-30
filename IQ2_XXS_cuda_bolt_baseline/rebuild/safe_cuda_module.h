#pragma once

#include <cuda.h>
#include <memory>
#include <stdexcept>

// Module ownership is tied to its creating context, not the context (if any)
// left current when a short-lived inference thread exits.
class SafeCudaModule {
 public:
  SafeCudaModule() {
    if (cuCtxGetCurrent(&context_) != CUDA_SUCCESS || !context_)
      throw std::runtime_error("CUDA module requires a current context");
  }
  SafeCudaModule(const SafeCudaModule&) = delete;
  SafeCudaModule& operator=(const SafeCudaModule&) = delete;
  ~SafeCudaModule() noexcept {
    if (!module_ || cuCtxPushCurrent(context_) != CUDA_SUCCESS) return;
    // Queued launches can outlive the host call. Complete them before unload.
    cuCtxSynchronize();
    cuModuleUnload(module_);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  operator CUmodule() const { return module_; }
  CUmodule* operator&() { return &module_; }

 private:
  CUcontext context_ = nullptr;
  CUmodule module_ = nullptr;
};
