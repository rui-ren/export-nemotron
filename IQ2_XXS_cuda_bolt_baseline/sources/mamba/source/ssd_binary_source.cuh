// Executable-only kernels from the independently observed long-prompt source profile.
// See source-profile.json and *_abi.json for configuration and compilation provenance.
#include "source_profile_launch.h"
#include "observed_cumsum_profiles.h"
#include "state_cubin.h"

const ObservedCumsumProfile& ObservedCumsumSpec() {
  const char* requested = std::getenv("BOLT_MAMBA_CUMSUM_HEAD_TILE");
  if (!requested)
    throw std::runtime_error("Mamba source cumsum profile must be supplied by upstream calibration");
  for (const auto& profile : kObservedCumsumProfiles) {
    if (std::to_string(profile.head_tile) == requested) {
      static const int initial = profile.head_tile;
      if (initial != profile.head_tile)
        throw std::runtime_error("Mamba source profile cannot change after its first use");
      return profile;
    }
  }
  throw std::runtime_error("Invalid upstream Mamba cumsum head tile");
}

void CheckSourceDriver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string("MambaScanVerify E5 source kernel: ") +
                             (message ? message : "CUDA driver failure"));
  }
}

struct SourceSsdBinary {
  CUmodule cumsum_module, state_module;
  CUfunction cumsum, state;
  unsigned int cumsum_grid, cumsum_threads, cumsum_shared;
  SourceSsdBinary() {
    const auto& profile = ObservedCumsumSpec();
    cumsum_grid = profile.grid;
    cumsum_threads = profile.threads;
    cumsum_shared = profile.shared;
    CheckSourceDriver(cuModuleLoadData(&cumsum_module, profile.image));
    CheckSourceDriver(cuModuleLoadData(&state_module, v5_state_cubin));
    CheckSourceDriver(cuModuleGetFunction(&cumsum, cumsum_module, "_chunk_cumsum_fwd_kernel"));
    CheckSourceDriver(cuModuleGetFunction(&state, state_module, "_chunk_state_fwd_kernel"));
    CheckSourceDriver(cuFuncSetAttribute(cumsum, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       cumsum_shared));
    CheckSourceDriver(cuFuncSetAttribute(state, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       kProfileSTATE_SHARED));
  }
};

SourceSsdBinary& SourceSsdForContext() {
  ObservedCumsumSpec();
  // Context-keyed immutable executable handles; all sequence data remains in ABI tensors.
  thread_local std::vector<std::pair<CUcontext, SourceSsdBinary*>> modules;
  CUcontext current = nullptr;
  CheckSourceDriver(cuCtxGetCurrent(&current));
  for (auto& entry : modules)
    if (entry.first == current) return *entry.second;
  auto* module = new SourceSsdBinary;
  modules.emplace_back(current, module);
  return *module;
}

__global__ void SourceBiasBf16(const float* bias, bf16* rounded) {
  rounded[threadIdx.x] = __float2bfloat16(bias[threadIdx.x]);
}

__global__ void SourceInitialState(float* state, const float* initial, const float* cs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 1048576) {
    const int h = index / 8192;
    state[index] = state[index] + initial[index] * __expf(cs[h * 128 + 127]);
  }
}

void LaunchOriginalSourceStates(cudaStream_t stream, const bf16* history,
                                const float* initial, const float* bias, const float* A,
                                float* state, int start, int rows, float tmin) {
  auto& kernels = SourceSsdForContext();
  float* scratch = nullptr;
  if (cudaMallocAsync(&scratch, 32768 * sizeof(float) + 128 * sizeof(bf16), stream)
      != cudaSuccess)
    throw std::runtime_error("MambaScanVerify E5: source-state scratch allocation failed");
  float* dt = scratch;
  float* cs = scratch + 16384;
  bf16* rounded_bias = reinterpret_cast<bf16*>(scratch + 32768);
  SourceBiasBf16<<<1, 128, 0, stream>>>(bias, rounded_bias);
  const bf16* raw = history + 10240;
  const bf16* B = history + 8192;
  int heads = 128, chunk = 128, history_batch = 2654208, history_row = 10368;
  int output_batch = 16384, output_chunk = 128, output_head = 128;
  int hd = 64, ds = 128, ratio = 16, x_head = 64, b_head = 128;
  int state_batch = 1048576, state_chunk = 1048576, state_head = 8192, state_n = 64;
  int zero = 0;
  float tmax = INFINITY;
  void* unused_scratch = nullptr;
  for (int r = 0; r < rows; ++r) {
    int length = start + r + 1;
    void* cumsum_args[] = {
      &raw, &A, &rounded_bias, &dt, &cs, &length, &heads, &chunk, &tmin, &tmax,
      &history_batch, &history_row, &output_batch, &output_chunk, &output_head,
      &output_batch, &output_chunk, &output_head, &unused_scratch, &unused_scratch};
    CheckSourceDriver(cuLaunchKernel(kernels.cumsum, 1, 1, kernels.cumsum_grid,
                                    kernels.cumsum_threads, 1, 1, kernels.cumsum_shared,
                                    stream, cumsum_args, nullptr));
    float* row_state = state + r * 1048576;
    void* state_args[] = {
      &history, &B, &row_state, &dt, &cs, &hd, &ds, &chunk, &length, &ratio,
      &history_batch, &history_row, &x_head, &history_batch, &history_row, &b_head,
      &state_batch, &state_chunk, &state_head, &state_n,
      &output_batch, &output_chunk, &output_head,
      &output_batch, &output_chunk, &output_head, &zero, &zero,
      &unused_scratch, &unused_scratch};
    CheckSourceDriver(cuLaunchKernel(kernels.state, kProfileSTATE_GRID, 1, 128,
                                    kProfileSTATE_THREADS, 1, 1, kProfileSTATE_SHARED,
                                    stream, state_args, nullptr));
    SourceInitialState<<<4096, 256, 0, stream>>>(row_state, initial, cs);
  }
  cudaFreeAsync(scratch, stream);
}
