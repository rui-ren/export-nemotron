#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <mutex>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace {

// One bounded arena per device and library, not one decoded matrix per graph node.
// We reuse storage only: every invocation still decodes its current weight operands.
template <typename T>
struct ProjectionScratch {
  cublasHandle_t handle = nullptr;
  T *weight = nullptr, *input = nullptr, *output = nullptr;
  void* workspace = nullptr;
  size_t weight_capacity = 0, input_capacity = 0, output_capacity = 0;
  size_t workspace_capacity = 0;
  int device = -1;
  std::mutex mutex;

  size_t bytes() const {
    return weight_capacity + input_capacity + output_capacity + workspace_capacity;
  }

  size_t projected_bytes(size_t w, size_t x, size_t y, size_t work) const {
    return std::max(w, weight_capacity) + std::max(x, input_capacity) +
           std::max(y, output_capacity) + std::max(work, workspace_capacity);
  }

  template <typename U>
  static cudaError_t grow(U*& pointer, size_t& capacity, size_t required) {
    if (required <= capacity) return cudaSuccess;
    if (pointer) {
      const auto error = cudaFree(pointer);
      if (error != cudaSuccess) return error;
      pointer = nullptr;
      capacity = 0;
    }
    const auto error = cudaMalloc(reinterpret_cast<void**>(&pointer), required);
    if (error == cudaSuccess) capacity = required;
    return error;
  }

  cudaError_t reserve(size_t w, size_t x, size_t y, size_t work) {
    auto error = grow(weight, weight_capacity, w);
    if (error == cudaSuccess) error = grow(input, input_capacity, x);
    if (error == cudaSuccess) error = grow(output, output_capacity, y);
    if (error == cudaSuccess) error = grow(workspace, workspace_capacity, work);
    return error;
  }

  int release() {
    if (device < 0) return 0;
    int previous = -1;
    auto error = cudaGetDevice(&previous);
    if (error != cudaSuccess) return 1000 + error;
    if (previous != device) {
      error = cudaSetDevice(device);
      if (error != cudaSuccess) return 1000 + error;
    }
    int status = 0;
    if (handle) {
      const auto result = cublasDestroy(handle);
      if (result == CUBLAS_STATUS_SUCCESS) handle = nullptr;
      else status = 2000 + result;
    }
    auto free_buffer = [&](auto*& pointer, size_t& capacity) {
      if (!pointer) return;
      const auto result = cudaFree(pointer);
      if (result == cudaSuccess) {
        pointer = nullptr;
        capacity = 0;
      } else if (!status) {
        status = 1000 + result;
      }
    };
    free_buffer(workspace, workspace_capacity);
    free_buffer(output, output_capacity);
    free_buffer(input, input_capacity);
    free_buffer(weight, weight_capacity);
    if (previous != device) {
      error = cudaSetDevice(previous);
      if (!status && error != cudaSuccess) status = 1000 + error;
    }
    return status;
  }

  ~ProjectionScratch() { release(); }
};

template <typename T>
ProjectionScratch<T>* projection_scratch(int& status) {
  int device = -1;
  const auto error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    status = 1000 + error;
    return nullptr;
  }
  // Fixed metadata storage avoids allocation/exception paths in the C launch ABI.
  static std::array<ProjectionScratch<T>, 64> states;
  if (device < 0 || device >= static_cast<int>(states.size())) {
    status = -6;
    return nullptr;
  }
  status = 0;
  return &states[device];
}

template <typename T>
struct ProjectionLease {
  ProjectionScratch<T>& state;
  std::unique_lock<std::mutex> lock;
  cudaStream_t stream;
  bool fenced = false;

  ProjectionLease(ProjectionScratch<T>& scratch, cudaStream_t current_stream)
      : state(scratch), lock(scratch.mutex), stream(current_stream) {
    cudaGetDevice(&state.device);
  }

  ~ProjectionLease() {
    // Error returns must also fence before another caller can overwrite scratch.
    if (!fenced) cudaStreamSynchronize(stream);
  }
};

template <typename T>
size_t projection_retained_bytes() {
  int status = 0;
  auto* state = projection_scratch<T>(status);
  if (!state) return 0;
  std::lock_guard<std::mutex> lock(state->mutex);
  return state->bytes();
}

template <typename T>
int projection_release_scratch() {
  int status = 0;
  auto* state = projection_scratch<T>(status);
  if (!state) return status;
  std::lock_guard<std::mutex> lock(state->mutex);
  return state->release();
}

}  // namespace
