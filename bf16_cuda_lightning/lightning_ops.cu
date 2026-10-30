// Source-derived single-token adaptation of src/bolt/bolt_ops.cu; see README.md for numerical differences.
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
using Shape = std::vector<int64_t>;
constexpr auto BF = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto F32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto I32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
constexpr auto U8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

__device__ __forceinline__ float B2F(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float RoundBF(float v) { return B2F(__float2bfloat16(v)); }
__device__ __forceinline__ float Silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float WarpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
int PositiveInt(int64_t v, const char* message) {
  Require(v > 0 && v <= std::numeric_limits<int>::max(), message);
  return static_cast<int>(v);
}
int CeilDiv(int n, int d) { return 1 + (n - 1) / d; }
Shape Dimensions(const Ort::ConstValue& v) { return v.GetTensorTypeAndShapeInfo().GetShape(); }
void CheckTensor(const Ort::KernelContext& c, size_t i, const Shape& expected,
                 ONNXTensorElementDataType type, const char* message) {
  auto info = c.GetInput(i).GetTensorTypeAndShapeInfo();
  Require(info.GetElementType() == type && info.GetShape() == expected, message);
  int64_t count = 1;
  for (auto d : expected) {
    Require(d > 0 && count <= std::numeric_limits<int64_t>::max() / d,
            "Tensor element count exceeds int64 range");
    count *= d;
  }
  Require(static_cast<uint64_t>(count) <= std::numeric_limits<size_t>::max() / 4,
          "Tensor byte count exceeds addressable range");
  Require(info.GetElementCount() == static_cast<size_t>(count), message);
}
void CheckArity(const Ort::KernelContext& c, size_t inputs, size_t outputs) {
  Require(c.GetInputCount() == inputs && c.GetOutputCount() == outputs,
          "lightning: incorrect input/output count");
}
template <typename T> const T* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const T*>(c.GetInput(i).GetTensorRawData());
}
template <typename T> T* Out(Ort::UnownedValue& v) {
  return reinterpret_cast<T*>(v.GetTensorMutableRawData());
}
cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}
void CheckLaunch() {
  const auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void CheckBlas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("lightning: cuBLAS failed with status " + std::to_string(status));
}
struct AsyncBuffer {
  void* data = nullptr;
  cudaStream_t stream;
  AsyncBuffer(size_t bytes, cudaStream_t s) : stream(s) {
    const auto error = cudaMallocAsync(&data, bytes, stream);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  }
  ~AsyncBuffer() { if (data) cudaFreeAsync(data, stream); }
};
OrtStatus* ComputeError(const OrtApi& api, const std::exception& e) {
  return api.CreateStatus(dynamic_cast<const std::invalid_argument*>(&e) ? ORT_INVALID_ARGUMENT : ORT_FAIL, e.what());
}

// Bolt's convolution channel ownership and kernel-4 vector load, with a generic kernel-width path.
__global__ void MambaConvKernel(const bf16* proj, int di, int channels, int kernel,
                               const bf16* past, const bf16* weight, const bf16* bias,
                               bf16* present, bf16* xbc) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const int64_t base = static_cast<int64_t>(c) * (kernel - 1);
  const float current = B2F(proj[di + c]);
  float acc;
  if (kernel == 4) {
    const uint2 wv = __ldg(reinterpret_cast<const uint2*>(weight + static_cast<int64_t>(c) * 4));
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
    const float x0 = B2F(past[base]), x1 = B2F(past[base + 1]), x2 = B2F(past[base + 2]);
    acc = B2F(bias[c]) + B2F(wp[0]) * x0 + B2F(wp[1]) * x1 + B2F(wp[2]) * x2 + B2F(wp[3]) * current;
    present[base] = __float2bfloat16(x1);
    present[base + 1] = __float2bfloat16(x2);
    present[base + 2] = __float2bfloat16(current);
  } else {
    acc = B2F(bias[c]);
    for (int j = 0; j < kernel; ++j) {
      const float x = j + 1 == kernel ? current : B2F(past[base + j]);
      acc += B2F(weight[static_cast<int64_t>(c) * kernel + j]) * x;
      if (j > 0) present[base + j - 1] = __float2bfloat16(x);
    }
  }
  // causal-conv1d uses fast math; Torch's later norm gate does not.
  xbc[c] = __float2bfloat16(__fdividef(acc, 1.0f + __expf(-acc)));
}

// Bolt's single-token QP=2 state-partitioned kernel, retaining [heads, DS, HD] storage.
template <int DS, int HD, int QP>
__global__ void MambaSsmKernel(const bf16* proj, int di, int channels, int heads_per_group, int groups,
                              const bf16* xbc, const float* past, const float* dt_bias,
                              const float* a, const float* skip, float tmin, const int32_t* position,
                              float* present, float* ypre) {
  if (position && position[0] > 0) return;
  const int h = blockIdx.x, p = threadIdx.x % HD, part = threadIdx.x / HD;
  const int g = h / heads_per_group;
  constexpr int NH = DS / QP;
  __shared__ float sB[DS], sC[DS], sPart[QP][HD], sDt;
  for (int n = threadIdx.x; n < DS; n += blockDim.x) {
    sB[n] = B2F(xbc[di + g * DS + n]);
    sC[n] = B2F(xbc[di + groups * DS + g * DS + n]);
  }
  if (threadIdx.x == 0) {
    const float raw = B2F(proj[di + channels + h]) + dt_bias[h];
    sDt = fmaxf(raw > 20.f ? raw :
               (position ? logf(__expf(raw) + 1.f) : log1pf(expf(raw))), tmin);
  }
  __syncthreads();
  const float decay = expf(a[h] * sDt), x = B2F(xbc[h * HD + p]), dtx = sDt * x;
  const bool initial = position && position[0] == 0;
  const int64_t base = (static_cast<int64_t>(h) * DS + part * NH) * HD + p;
  float acc = 0.f;
#pragma unroll
  for (int n = 0; n < NH; ++n) {
    const int64_t offset = base + static_cast<int64_t>(n) * HD;
    const float st = initial ? RoundBF(sDt * sB[part * NH + n]) * x
                            : past[offset] * decay + dtx * sB[part * NH + n];
    present[offset] = st;
    acc += sC[part * NH + n] * (initial ? sB[part * NH + n] : st);
  }
  sPart[part][p] = acc;
  __syncthreads();
  if (part == 0) {
    float sum = initial ? 0.f : skip[h] * x;
#pragma unroll
    for (int q = 0; q < QP; ++q) sum += sPart[q][p];
    ypre[h * HD + p] = initial ? RoundBF(sum * sDt) * x + skip[h] * x : sum;
  }
}

// Match the pinned native Triton update's four states per lane and fused readout.
__global__ void MambaSsmSourceUpdateKernel(
    const bf16* proj, int di, int channels, int heads_per_group, int groups,
    const bf16* xbc, const float* past, const float* dt_bias, const float* a,
    const float* skip, float tmin, const int32_t* position, float* present, float* ypre) {
  if (position[0] == 0) return;
  const int h = blockIdx.x, p = blockIdx.y * 4 + threadIdx.x / 32;
  const int n = (threadIdx.x & 31) * 4, g = h / heads_per_group;
  const float raw = B2F(proj[di + channels + h]) + dt_bias[h];
  const float dt = fmaxf(raw > 20.f ? raw : logf(__expf(raw) + 1.f), tmin);
  const float decay = __expf(a[h] * dt), x = B2F(xbc[h * 64 + p]);
  float state[4], c[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int64_t offset = (static_cast<int64_t>(h) * 128 + n + j) * 64 + p;
    const float b = B2F(xbc[di + g * 128 + n + j]);
    c[j] = B2F(xbc[di + groups * 128 + g * 128 + n + j]);
    state[j] = __fmaf_rn(decay, past[offset], (dt * b) * x);
    present[offset] = state[j];
  }
  float acc = state[1] * c[1];
  acc = __fmaf_rn(state[0], c[0], acc);
  acc = __fmaf_rn(state[2], c[2], acc);
  acc = __fmaf_rn(state[3], c[3], acc);
  acc = WarpSum(acc);
  if ((threadIdx.x & 31) == 0) ypre[h * 64 + p] = __fmaf_rn(x, skip[h], acc);
}

// Fixture dimensions use the same recurrence and layout without template restrictions.
__global__ void MambaSsmGenericKernel(const bf16* proj, int di, int channels, int hd, int ds,
                                     int heads_per_group, int groups, const bf16* xbc,
                                     const float* past, const float* dt_bias, const float* a,
                                     const float* skip, float tmin, const int32_t* position,
                                     float* present, float* ypre) {
  const int feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature >= di) return;
  const int h = feature / hd, p = feature % hd, g = h / heads_per_group;
  const float raw = B2F(proj[di + channels + h]) + dt_bias[h];
  const float dt = fmaxf(raw > 20.f ? raw :
                        (position ? logf(__expf(raw) + 1.f) : log1pf(expf(raw))), tmin);
  const float decay = expf(a[h] * dt), x = B2F(xbc[feature]), dtx = dt * x;
  const bool initial = position && position[0] == 0;
  float acc = 0.f;
  for (int n = 0; n < ds; ++n) {
    const int64_t offset = (static_cast<int64_t>(h) * ds + n) * hd + p;
    const float b = B2F(xbc[di + g * ds + n]);
    const float st = initial ? RoundBF(dt * b) * x : past[offset] * decay + dtx * b;
    present[offset] = st;
    acc += B2F(xbc[di + groups * ds + g * ds + n]) * (initial ? b : st);
  }
  ypre[feature] = skip[h] * x + (initial ? RoundBF(acc * dt) * x : acc);
}

int RmsThreads(int width, int rows = 1) {
  const int count = width > 128 ? width / 4 : width;
  int threads = 1;
  while (threads < 512 && threads <= count / 2) threads *= 2;
  int height = 1;
  while (height < 512 && height <= rows / 2) height *= 2;
  const int initial_width = threads < 32 ? threads : 32;
  height = height < 512 / initial_width ? height : 512 / initial_width;
  return threads < 512 / height ? threads : 512 / height;
}

template <bool Gated>
__device__ __forceinline__ float RmsInput(const bf16* x, const float* readout, int64_t i) {
  if constexpr (Gated) return RoundBF(readout[i]) * Silu(B2F(x[i]));
  return B2F(x[i]);
}

// Native contiguous FP32 mean: four accumulators, shared halves, then ascending warp steps.
template <bool Gated>
__global__ void SourceRmsKernel(const bf16* x, const float* readout, const float* weight,
                               int width, float eps, bf16* y, int weight_rows = 0) {
  const int t = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * width;
  __shared__ float reduction[512];
  float values[4] = {};
  if (width > 128) {
    for (int i = t * 4; i + 3 < width; i += blockDim.x * 4) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float v = RmsInput<Gated>(x, readout, base + i + j);
        values[j] += v * v;
      }
    }
    const int tail = width - width % 4 + t;
    if (tail < width) {
      const float v = RmsInput<Gated>(x, readout, base + tail);
      values[0] += v * v;
    }
  } else {
    for (int i = t; i < width; i += blockDim.x) {
      const float v = RmsInput<Gated>(x, readout, base + i);
      values[0] += v * v;
    }
  }
  float sum = ((values[0] + values[1]) + values[2]) + values[3];
  reduction[t] = sum;
  for (int offset = blockDim.x / 2; offset >= 32; offset /= 2) {
    __syncthreads();
    if (t < offset) {
      sum += reduction[t + offset];
      reduction[t] = sum;
    }
  }
  __syncthreads();
  if (t < 32) {
    const int lanes = blockDim.x < 32 ? blockDim.x : 32;
    const unsigned mask = __activemask();
    for (int offset = 1; offset < lanes; offset *= 2)
      sum += __shfl_down_sync(mask, sum, offset, lanes);
    if (t == 0) reduction[0] = rsqrtf(sum * (1.f / width) + eps);
  }
  __syncthreads();
  for (int i = t; i < width; i += blockDim.x) {
    float v = RmsInput<Gated>(x, readout, base + i) * reduction[0];
    if constexpr (Gated) v = RoundBF(v);
    const int64_t weight_base = Gated ? (weight_rows ? blockIdx.x % weight_rows : blockIdx.x) * width : 0;
    y[base + i] = __float2bfloat16(v * weight[weight_base + i]);
  }
}

// Lightning rounds the SSM readout before gating, and normalized values before norm_w multiplication.
__global__ void MambaNormKernel(const bf16* proj, int group_size, float eps, const float* ypre,
                               const float* norm_w, bf16* y) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * group_size;
  __shared__ float red[8];
  float ss = 0.f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float v = RoundBF(ypre[base + i]) * Silu(B2F(proj[base + i]));
    ss += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / group_size + eps);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float v = RoundBF(ypre[base + i]) * Silu(B2F(proj[base + i]));
    y[base + i] = __float2bfloat16(RoundBF(v * rs) * norm_w[base + i]);
  }
}

struct MambaStepState {
  const OrtApi& api;
  int heads, hd, ds, groups, di, channels, stride;
  float eps, tmin;
  MambaStepState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    heads = PositiveInt(k.GetAttribute<int64_t>("heads"), "MambaStep: invalid heads");
    hd = PositiveInt(k.GetAttribute<int64_t>("head_dim"), "MambaStep: invalid head_dim");
    ds = PositiveInt(k.GetAttribute<int64_t>("d_state"), "MambaStep: invalid d_state");
    groups = PositiveInt(k.GetAttribute<int64_t>("n_groups"), "MambaStep: invalid n_groups");
    eps = k.GetAttribute<float>("epsilon");
    tmin = k.GetAttribute<float>("time_step_min");
    Require(heads % groups == 0 && std::isfinite(eps) && eps > 0.f &&
            std::isfinite(tmin) && tmin >= 0.f, "MambaStep: invalid groups/epsilon/time_step_min");
    di = PositiveInt(static_cast<int64_t>(heads) * hd, "MambaStep: inner size overflow");
    channels = PositiveInt(static_cast<int64_t>(di) + 2LL * groups * ds, "MambaStep: channels overflow");
    stride = PositiveInt(static_cast<int64_t>(di) + channels + heads, "MambaStep: projection size overflow");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      Require((c.GetInputCount() == 9 || c.GetInputCount() == 10) && c.GetOutputCount() == 5,
              "MambaStep: expected nine legacy inputs or ten position-aware inputs");
      const int32_t* position = nullptr;
      if (c.GetInputCount() == 10) {
        CheckTensor(c, 9, {1}, I32, "MambaStep: position must be INT32 [1]");
        position = In<int32_t>(c, 9);
      }
      CheckTensor(c, 0, {1, stride}, BF, "MambaStep: proj must be BF16 [1,inner+channels+heads]");
      const auto cw = Dimensions(c.GetInput(3));
      Require(cw.size() == 3 && cw[0] == channels && cw[1] == 1 && cw[2] >= 2 &&
              cw[2] <= std::numeric_limits<int>::max(), "MambaStep: conv_w must be [channels,1,kernel>=2]");
      const int kernel = static_cast<int>(cw[2]);
      CheckTensor(c, 1, {1, channels, kernel - 1}, BF, "MambaStep: invalid convolution state");
      CheckTensor(c, 2, {1, heads, ds, hd}, F32, "MambaStep: state must be FLOAT [1,heads,d_state,head_dim]");
      CheckTensor(c, 3, cw, BF, "MambaStep: invalid convolution weights");
      CheckTensor(c, 4, {channels}, BF, "MambaStep: invalid convolution bias");
      for (size_t i = 5; i < 8; ++i)
        CheckTensor(c, i, {heads}, F32, "MambaStep: per-head parameters must be FLOAT [heads]");
      CheckTensor(c, 8, {di}, F32, "MambaStep: norm_w must be FLOAT [inner]");
      auto y = c.GetOutput(0, Shape{1, di});
      auto conv = c.GetOutput(1, Shape{1, channels, kernel - 1});
      auto state = c.GetOutput(2, Shape{1, heads, ds, hd});
      auto xbc = c.GetOutput(3, Shape{1, channels});
      auto ypre = c.GetOutput(4, Shape{1, di});
      const auto s = Stream(api, raw);
      MambaConvKernel<<<CeilDiv(channels, 128), 128, 0, s>>>(
          In<bf16>(c, 0), di, channels, kernel, In<bf16>(c, 1), In<bf16>(c, 3),
          In<bf16>(c, 4), Out<bf16>(conv), Out<bf16>(xbc));
      CheckLaunch();
      if (hd == 64 && ds == 128) {
        MambaSsmKernel<128, 64, 2><<<heads, 128, 0, s>>>(
            In<bf16>(c, 0), di, channels, heads / groups, groups, Out<bf16>(xbc),
            In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
            tmin, position, Out<float>(state), Out<float>(ypre));
        CheckLaunch();
        if (position) {
          MambaSsmSourceUpdateKernel<<<dim3(heads, 16), 128, 0, s>>>(
              In<bf16>(c, 0), di, channels, heads / groups, groups, Out<bf16>(xbc),
              In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
              tmin, position, Out<float>(state), Out<float>(ypre));
        }
      } else {
        MambaSsmGenericKernel<<<CeilDiv(di, 128), 128, 0, s>>>(
            In<bf16>(c, 0), di, channels, hd, ds, heads / groups, groups, Out<bf16>(xbc),
            In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
            tmin, position, Out<float>(state), Out<float>(ypre));
      }
      CheckLaunch();
      if (position) {
        SourceRmsKernel<true><<<groups, RmsThreads(di / groups, groups), 0, s>>>(
            In<bf16>(c, 0), Out<float>(ypre), In<float>(c, 8), di / groups, eps, Out<bf16>(y));
      } else {
        MambaNormKernel<<<groups, 256, 0, s>>>(
            In<bf16>(c, 0), di / groups, eps, Out<float>(ypre), In<float>(c, 8), Out<bf16>(y));
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

struct RmsNormState {
  const OrtApi& api;
  float eps;
  RmsNormState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    eps = Ort::ConstKernelInfo(info).GetAttribute<float>("epsilon");
    Require(std::isfinite(eps) && eps > 0.f, "RMSNorm: invalid epsilon");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 2, 1);
      const auto shape = Dimensions(c.GetInput(0));
      Require((shape.size() == 2 || shape.size() == 3) &&
              (shape.size() == 2 || shape[0] == 1), "RMSNorm: expected [T,H] or [1,T,H]");
      const int width = PositiveInt(shape.back(), "RMSNorm: invalid width");
      const int rows = PositiveInt(shape[shape.size() - 2], "RMSNorm: invalid rows");
      CheckTensor(c, 0, shape, BF, "RMSNorm: invalid BF16 activation");
      CheckTensor(c, 1, {width}, F32, "RMSNorm: weight must be FLOAT [width]");
      auto output = c.GetOutput(0, shape);
      SourceRmsKernel<false><<<rows, RmsThreads(width), 0, Stream(api, raw)>>>(
          In<bf16>(c, 0), nullptr, In<float>(c, 1), width, eps, Out<bf16>(output));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

__device__ __forceinline__ bool Aligned(const void* p, uintptr_t alignment) {
  return (reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0;
}

// Bolt's eight-BF16 vector loads, guarded for incomplete and unaligned rows.
__device__ __forceinline__ float DotBF16x8(const bf16* x, const bf16* w, int remaining) {
  float acc = 0.f;
  if (remaining >= 8 && Aligned(x, 16) && Aligned(w, 16)) {
    const uint4 xv = *reinterpret_cast<const uint4*>(x);
    const uint4 wv = *reinterpret_cast<const uint4*>(w);
    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += B2F(xp[i]) * B2F(wp[i]);
  } else {
    for (int i = 0; i < 8 && i < remaining; ++i) acc += B2F(x[i]) * B2F(w[i]);
  }
  return acc;
}

// Bolt's (row, ceil(E/8)) grid and one warp per logit, without the H%256 restriction.
__global__ void RouterLogitsKernel(const bf16* x, const bf16* gate, int hidden, int experts, float* scores) {
  x += static_cast<int64_t>(blockIdx.x) * hidden;
  scores += static_cast<int64_t>(blockIdx.x) * experts;
  const int e = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (e >= experts) return;
  const bf16* w = gate + static_cast<int64_t>(e) * hidden;
  float acc = 0.f;
  for (int64_t k = lane * 8; k < hidden; k += 256)
    acc += DotBF16x8(x + k, w + k, hidden - static_cast<int>(k));
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) scores[e] = 1.f / (1.f + expf(-acc));
}

__global__ void WidenRouterKernel(const bf16* x, const bf16* gate, int hidden,
                                  int64_t count, float* values) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) values[i] = B2F(i < hidden ? x[i] : gate[i - hidden]);
}

__global__ void RouterSigmoidKernel(float* scores, int experts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < experts) scores[i] = 1.f / (1.f + expf(-scores[i]));
}

// Bolt's iterative warp argmax, with explicit availability and smallest-ID ties even at -infinity.
__global__ void RouterTopkKernel(const float* scores, const float* bias, int experts, int topk,
                                float scale, bool source_order, int32_t* indices, float* probs) {
  scores += static_cast<int64_t>(blockIdx.x) * experts;
  indices += static_cast<int64_t>(blockIdx.x) * topk;
  probs += static_cast<int64_t>(blockIdx.x) * topk;
  const int lane = threadIdx.x;
  __shared__ int selected_ids[32], order[32];
  __shared__ float selected_scores[32];
  float val[16], sc[16];
  bool available[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + 32 * i;
    available[i] = e < experts;
    sc[i] = e < experts ? scores[e] : 0.f;
    val[i] = e < experts ? sc[i] + bias[e] : -INFINITY;
    if (isnan(val[i])) val[i] = -INFINITY;
  }
  int my_idx = -1;
  float my_sc = 0.f, sum = 0.f;
  for (int j = 0; j < topk; ++j) {
    float best = -INFINITY;
    int id = -1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int e = lane + 32 * i;
      if (available[i] && (id < 0 || val[i] > best || (val[i] == best && e < id))) {
        best = val[i];
        id = e;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float other = __shfl_xor_sync(0xffffffff, best, o);
      const int oi = __shfl_xor_sync(0xffffffff, id, o);
      if (oi >= 0 && (id < 0 || other > best || (other == best && oi < id))) {
        best = other;
        id = oi;
      }
    }
    const int owner = id & 31, slot = id >> 5;
    const float selected = __shfl_sync(0xffffffff, sc[slot], owner);
    if (lane == owner) available[slot] = false;
    if (lane == j) { my_idx = id; my_sc = selected; }
    sum += selected;
  }
  if (source_order) {
    if (lane < topk) {
      selected_ids[lane] = my_idx;
      selected_scores[lane] = my_sc;
    }
    __syncthreads();
    const float threshold = selected_scores[topk - 1] + bias[selected_ids[topk - 1]];
    if (lane < topk) {
      const bool equal = selected_scores[lane] + bias[selected_ids[lane]] == threshold;
      int rank = 0;
      for (int k = 0; k < topk; ++k) {
        const bool other_equal = selected_scores[k] + bias[selected_ids[k]] == threshold;
        rank += (!other_equal && equal) ||
                (other_equal == equal && selected_ids[k] < selected_ids[lane]);
      }
      order[rank] = lane;
    }
    __syncthreads();
    int threads = 1;
    while (threads <= topk / 2) threads *= 2;
    float partial = 0.f;
    if (lane < threads)
      for (int k = lane; k < topk; k += threads) partial += selected_scores[order[k]];
    for (int offset = 1; offset < threads; offset *= 2)
      partial += __shfl_down_sync(0xffffffff, partial, offset, threads);
    sum = __shfl_sync(0xffffffff, partial, 0);
    if (lane < topk) {
      my_idx = selected_ids[order[lane]];
      my_sc = selected_scores[order[lane]];
    }
  }
  if (lane < topk) {
    indices[lane] = my_idx;
    probs[lane] = my_sc / (sum + 1e-20f) * scale;
  }
}

struct MoeRouteState {
  const OrtApi& api;
  int topk;
  float scale;
  bool source_order = false;
  cublasHandle_t blas = nullptr;
  std::mutex blas_mutex;
  MoeRouteState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = PositiveInt(k.GetAttribute<int64_t>("topk"), "MoeRoute: invalid topk");
    scale = k.GetAttribute<float>("scale");
    Require(topk <= 32 && std::isfinite(scale) && scale >= 0.f, "MoeRoute: topk<=32 and finite nonnegative scale required");
    int64_t flag = 0;
    const auto status = api.KernelInfoGetAttribute_int64(info, "source_order", &flag);
    if (status) api.ReleaseStatus(status);
    Require(flag == 0 || flag == 1, "MoeRoute: source_order must be zero or one");
    source_order = flag == 1;
    if (source_order) CheckBlas(cublasCreate(&blas));
  }
  ~MoeRouteState() { if (blas) cublasDestroy(blas); }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 3, 3);
      const auto xs = Dimensions(c.GetInput(0)), ws = Dimensions(c.GetInput(1));
      Require(xs.size() == 2, "MoeRoute: x must be [T,H]");
      const int rows = PositiveInt(xs[0], "MoeRoute: invalid T");
      const int hidden = PositiveInt(xs[1], "MoeRoute: invalid H");
      Require(ws.size() == 2, "MoeRoute: gate must be [E,H]");
      const int experts = PositiveInt(ws[0], "MoeRoute: invalid E");
      Require(experts <= 512 && topk <= experts, "MoeRoute: E<=512 and topk<=E required");
      CheckTensor(c, 0, {rows, hidden}, BF, "MoeRoute: x must be BF16 [T,H]");
      CheckTensor(c, 1, {experts, hidden}, BF, "MoeRoute: gate must be BF16 [E,H]");
      CheckTensor(c, 2, {experts}, F32, "MoeRoute: bias must be FLOAT [E]");
      auto indices = c.GetOutput(0, Shape{rows, topk});
      auto probs = c.GetOutput(1, Shape{rows, topk});
      auto scores = c.GetOutput(2, Shape{rows, experts});
      const auto s = Stream(api, raw);
      if (source_order) {
        std::lock_guard<std::mutex> lock(blas_mutex);
        const int input_count = PositiveInt(static_cast<int64_t>(rows) * hidden, "MoeRoute: input too large");
        const int score_count = PositiveInt(static_cast<int64_t>(rows) * experts, "MoeRoute: scores too large");
        const int64_t count = static_cast<int64_t>(hidden) * (experts + rows);
        AsyncBuffer buffer(static_cast<size_t>(count) * sizeof(float), s);
        auto values = static_cast<float*>(buffer.data);
        WidenRouterKernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, s>>>(
            In<bf16>(c, 0), In<bf16>(c, 1), input_count, count, values);
        CheckLaunch();
        CheckBlas(cublasSetStream(blas, s));
        const float alpha = 1.f, beta = 0.f;
        // FP32 routing is sensitive to logical GEMM rows, even before BF16 expert math.
        for (int row = 0; row < rows; ++row)
          CheckBlas(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, experts, 1, hidden, &alpha,
                               values + input_count, hidden, values + static_cast<int64_t>(row) * hidden,
                               hidden, &beta, Out<float>(scores) + static_cast<int64_t>(row) * experts, experts));
        RouterSigmoidKernel<<<CeilDiv(score_count, 128), 128, 0, s>>>(Out<float>(scores), score_count);
      } else {
        RouterLogitsKernel<<<dim3(rows, CeilDiv(experts, 8)), 256, 0, s>>>(
            In<bf16>(c, 0), In<bf16>(c, 1), hidden, experts, Out<float>(scores));
      }
      CheckLaunch();
      RouterTopkKernel<<<rows, 32, 0, s>>>(
          Out<float>(scores), In<float>(c, 2), experts, topk, scale, source_order,
          Out<int32_t>(indices), Out<float>(probs));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

// Bolt's packed 32-nibble chunk; unlike Bolt, scale and BF16-round each dequantized value before the dot.
__device__ __forceinline__ float DotInt4x32(const uint8_t* w, const bf16* scale,
                                          const bf16* x, int column, int remaining, int block) {
  float acc = 0.f;
  if (remaining >= 32 && Aligned(w, 16) && Aligned(x, 16)) {
    const uint4 packed = *reinterpret_cast<const uint4*>(w);
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 xv = *reinterpret_cast<const uint4*>(x + q * 8);
      const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (words[q] >> (8 * b)) & 255u;
        const int offset = q * 8 + b * 2;
        const float lo = RoundBF((static_cast<int>(byte & 15u) - 8) * B2F(scale[(column + offset) / block]));
        const float hi = RoundBF((static_cast<int>(byte >> 4) - 8) * B2F(scale[(column + offset + 1) / block]));
        acc += lo * B2F(xp[b * 2]) + hi * B2F(xp[b * 2 + 1]);
      }
    }
  } else {
    for (int i = 0; i < 32 && i < remaining; ++i) {
      const uint8_t byte = w[i / 2];
      const int nibble = (i & 1) ? byte >> 4 : byte & 15;
      acc += RoundBF((nibble - 8) * B2F(scale[(column + i) / block])) * B2F(x[i]);
    }
  }
  return acc;
}

__global__ void W4LinearKernel(const bf16* x, const uint8_t* weight, const bf16* scale,
                               int rows, int columns, int block, bf16* y) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const int blocks = 1 + (columns - 1) / block;
  const uint8_t* w = weight + static_cast<int64_t>(row) * blocks * (block / 2);
  const bf16* s = scale + static_cast<int64_t>(row) * blocks;
  float acc = 0.f;
  for (int64_t col = lane * 32; col < columns; col += 1024)
    acc += DotInt4x32(w + col / 2, s, x + col, static_cast<int>(col),
                     columns - static_cast<int>(col), block);
  acc = WarpSum(acc);
  if (lane == 0) y[row] = __float2bfloat16(acc);
}

struct W4LinearState {
  const OrtApi& api;
  int rows, columns, block;
  W4LinearState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    rows = PositiveInt(k.GetAttribute<int64_t>("N"), "W4Linear: invalid N");
    columns = PositiveInt(k.GetAttribute<int64_t>("K"), "W4Linear: invalid K");
    block = PositiveInt(k.GetAttribute<int64_t>("block_size"), "W4Linear: invalid block size");
    Require(block >= 16 && (block & (block - 1)) == 0, "W4Linear: block size must be a power of two >=16");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 3, 1);
      auto shape = Dimensions(c.GetInput(0));
      Require((shape.size() == 2 || shape.size() == 3) && shape.back() == columns &&
              shape[0] == 1 && (shape.size() == 2 || shape[1] == 1),
              "W4Linear: only one BF16 activation row is supported");
      CheckTensor(c, 0, shape, BF, "W4Linear: invalid BF16 activation");
      const int blocks = CeilDiv(columns, block);
      CheckTensor(c, 1, {rows, blocks, block / 2}, U8, "W4Linear: invalid packed weights");
      CheckTensor(c, 2, {rows, blocks}, BF, "W4Linear: invalid BF16 scales");
      shape.back() = rows;
      auto y = c.GetOutput(0, shape);
      W4LinearKernel<<<CeilDiv(rows, 8), 256, 0, Stream(api, raw)>>>(
          In<bf16>(c, 0), In<uint8_t>(c, 1), In<bf16>(c, 2), rows, columns, block, Out<bf16>(y));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

// Bolt's (selected expert, ceil(output_rows/8)) grid, generalized from latent=1024.
template <bool Quantized, bool Up>
__global__ void MoeFcKernel(const bf16* x, const int32_t* indices, int experts, int hidden, int intermediate,
                           const void* weights, const bf16* scales, int block, bf16* h, float* part) {
  const int pair = blockIdx.x, e = indices[pair];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rows = Up ? intermediate : hidden, columns = Up ? hidden : intermediate;
  const int n = blockIdx.y * 8 + warp;
  if (n >= rows) return;
  assert(e >= 0 && e < experts);
  if (e < 0 || e >= experts) return;
  const int64_t row = static_cast<int64_t>(e) * rows + n;
  const bf16* xr = Up ? x : x + static_cast<int64_t>(pair) * intermediate;
  float acc = 0.f;
  if constexpr (Quantized) {
    const int blocks = 1 + (columns - 1) / block;
    const uint8_t* wr = static_cast<const uint8_t*>(weights) + row * blocks * (block / 2);
    const bf16* sr = scales + row * blocks;
    for (int64_t col = static_cast<int64_t>(lane) * 32; col < columns; col += 1024)
      acc += DotInt4x32(wr + col / 2, sr, xr + col, static_cast<int>(col), columns - static_cast<int>(col), block);
  } else {
    const bf16* wr = static_cast<const bf16*>(weights) + row * columns;
    for (int64_t col = static_cast<int64_t>(lane) * 8; col < columns; col += 256)
      acc += DotBF16x8(xr + col, wr + col, columns - static_cast<int>(col));
  }
  acc = WarpSum(acc);
  if (lane == 0) {
    if constexpr (Up) {
      const float r = fmaxf(RoundBF(acc), 0.f);
      h[static_cast<int64_t>(pair) * intermediate + n] = __float2bfloat16(r * r);
    } else {
      part[static_cast<int64_t>(pair) * hidden + n] = RoundBF(acc);
    }
  }
}

// Bolt's float4 reduction over routed pairs, with alignment and final-column tails.
__global__ void MoeFinalizeKernel(const float* part, const float* probs, const int32_t* indices,
                                  int topk, int hidden, bool source_order, bf16* out) {
  __shared__ int order[32];
  if (threadIdx.x < topk) {
    int rank = 0;
    for (int k = 0; k < topk; ++k)
      rank += indices[k] < indices[threadIdx.x] ||
              (indices[k] == indices[threadIdx.x] && k < threadIdx.x);
    order[rank] = threadIdx.x;
  }
  __syncthreads();
  const int64_t n = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (n >= hidden) return;
  float acc[4] = {};
  if (source_order) {
    // Native grouped_mm restores route order, then Torch's strided six-row sum
    // accumulates four independent lanes before a left-associated final sum.
    float lanes[4][4] = {};
    for (int k = 0; k < topk; ++k)
      for (int j = 0; j < 4 && n + j < hidden; ++j)
        lanes[k % 4][j] += probs[k] * part[static_cast<int64_t>(k) * hidden + n + j];
    for (int j = 0; j < 4; ++j)
      acc[j] = ((lanes[0][j] + lanes[1][j]) + lanes[2][j]) + lanes[3][j];
  } else for (int k = 0; k < topk; ++k) {
    const int pair = order[k];
    const float* p = part + static_cast<int64_t>(pair) * hidden + n;
    if (n + 4 <= hidden && Aligned(p, 16)) {
      const float4 v = *reinterpret_cast<const float4*>(p);
      acc[0] += probs[pair] * v.x;
      acc[1] += probs[pair] * v.y;
      acc[2] += probs[pair] * v.z;
      acc[3] += probs[pair] * v.w;
    } else {
      for (int j = 0; j < 4 && n + j < hidden; ++j) acc[j] += probs[pair] * p[j];
    }
  }
  if (n + 4 <= hidden && Aligned(out + n, 8)) {
    uint2 packed;
    bf16* p = reinterpret_cast<bf16*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = __float2bfloat16(acc[j]);
    *reinterpret_cast<uint2*>(out + n) = packed;
  } else {
    for (int j = 0; j < 4 && n + j < hidden; ++j) out[n + j] = __float2bfloat16(acc[j]);
  }
}

__global__ void ReluSquaredKernel(bf16* value, int64_t count) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n < count) {
    const float r = fmaxf(B2F(value[n]), 0.f);
    value[n] = __float2bfloat16(r * r);
  }
}
__global__ void ExpertPartKernel(const bf16* value, float* part, int64_t count) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n < count) part[n] = B2F(value[n]);
}

#include "device_decode.cuh"

template <bool Quantized>
struct MoeDecodeState {
  const OrtApi& api;
  int topk, block = 32;
  bool source_order = false;
  bool device_decode = false;
  cublasHandle_t blas = nullptr;
  std::mutex blas_mutex;
  MoeDecodeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = PositiveInt(k.GetAttribute<int64_t>("topk"), "MoeDecode: invalid topk");
    Require(topk <= 32, "MoeDecode: topk must be in [1,32]");
    int64_t flag = 0;
    const auto status = api.KernelInfoGetAttribute_int64(info, "source_order", &flag);
    if (status) api.ReleaseStatus(status);
    Require(flag == 0 || flag == 1, "MoeDecode: source_order must be zero or one");
    source_order = flag == 1;
    Require(!source_order || topk == 6, "MoeDecode: source_order requires six routed experts");
    if constexpr (Quantized) {
      block = PositiveInt(k.GetAttribute<int64_t>("block_size"), "MoeDecode: invalid block_size");
      Require(block % 2 == 0, "MoeDecode: block_size must be positive and even");
    } else {
      CheckBlas(cublasCreate(&blas));
    }
  }
  ~MoeDecodeState() { if (blas) cublasDestroy(blas); }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, Quantized ? 7 : 5, 3);
      const auto xs = Dimensions(c.GetInput(0)), ws = Dimensions(c.GetInput(3));
      Require(xs.size() == 2 && xs[0] == 1, "MoeDecode: x must be [1,H]");
      const int hidden = PositiveInt(xs[1], "MoeDecode: invalid H");
      Require(ws.size() == (Quantized ? 4u : 3u), "MoeDecode: invalid up weight rank");
      const int experts = PositiveInt(ws[0], "MoeDecode: invalid E");
      const int intermediate = PositiveInt(ws[1], "MoeDecode: invalid I");
      Require(topk <= experts, "MoeDecode: topk must not exceed E");
      Require(CeilDiv(hidden, 8) <= 65535 && CeilDiv(intermediate, 8) <= 65535,
              "MoeDecode: output rows exceed CUDA grid.y limit");
      CheckTensor(c, 0, {1, hidden}, BF, "MoeDecode: x must be BF16 [1,H]");
      CheckTensor(c, 1, {1, topk}, I32, "MoeDecode: idx must be INT32 [1,topk]");
      CheckTensor(c, 2, {1, topk}, F32, "MoeDecode: probs must be FLOAT [1,topk]");
      const bf16* s1 = nullptr;
      const bf16* s2 = nullptr;
      if constexpr (Quantized) {
        const int bh = CeilDiv(hidden, block), bi = CeilDiv(intermediate, block);
        CheckTensor(c, 3, {experts, intermediate, bh, block / 2}, U8, "MoeDecode: invalid packed up weights");
        CheckTensor(c, 4, {experts, intermediate, bh}, BF, "MoeDecode: invalid BF16 up scales");
        CheckTensor(c, 5, {experts, hidden, bi, block / 2}, U8, "MoeDecode: invalid packed down weights");
        CheckTensor(c, 6, {experts, hidden, bi}, BF, "MoeDecode: invalid BF16 down scales");
        s1 = In<bf16>(c, 4);
        s2 = In<bf16>(c, 6);
      } else {
        CheckTensor(c, 3, {experts, intermediate, hidden}, BF, "MoeDecodeBF16: invalid up weights");
        CheckTensor(c, 4, {experts, hidden, intermediate}, BF, "MoeDecodeBF16: invalid down weights");
      }
      auto y = c.GetOutput(0, Shape{1, hidden});
      auto h = c.GetOutput(1, Shape{topk, intermediate});
      auto part = c.GetOutput(2, Shape{topk, hidden});
      const auto s = Stream(api, raw);
      if constexpr (!Quantized) {
        if (device_decode && source_order && topk == 6 && hidden == 2688 && intermediate == 1856) {
          AsyncBuffer scratch(2 * 6 * 1856 * sizeof(float), s);
          DeviceExpertTiledMmaKernel<true><<<dim3(1856 / 64, 6, 2), 128, 0, s>>>(
              In<bf16>(c, 0), In<int32_t>(c, 1), In<bf16>(c, 3), experts,
              static_cast<float*>(scratch.data));
          DeviceExpertActivationKernel<<<CeilDiv(6 * 1856, 256), 256, 0, s>>>(
              static_cast<const float*>(scratch.data), Out<bf16>(h));
          DeviceExpertTiledMmaKernel<false><<<dim3(2688 / 64, 6), 128, 0, s>>>(
              Out<bf16>(h), In<int32_t>(c, 1), In<bf16>(c, 4), experts, Out<float>(part));
          MoeFinalizeKernel<<<CeilDiv(CeilDiv(hidden, 4), 64), 64, 0, s>>>(
              Out<float>(part), In<float>(c, 2), In<int32_t>(c, 1), topk, hidden, true, Out<bf16>(y));
          CheckLaunch();
          return nullptr;
        }
        std::lock_guard<std::mutex> lock(blas_mutex);
        AsyncBuffer down_buffer(static_cast<size_t>(topk) * hidden * sizeof(bf16), s);
        auto down_values = static_cast<bf16*>(down_buffer.data);
        std::vector<int32_t> cohort(topk);
        auto error = cudaMemcpyAsync(cohort.data(), In<int32_t>(c, 1), topk * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, s);
        if (error == cudaSuccess) error = cudaStreamSynchronize(s);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        CheckBlas(cublasSetStream(blas, s));
        const float alpha = 1.f, beta = 0.f;
        // Match HF's one logical row per expert rather than a different batched GEMM reduction.
        for (int k = 0; k < topk; ++k) {
          Require(cohort[k] >= 0 && cohort[k] < experts, "MoeDecodeBF16: expert index out of range");
          const auto offset = static_cast<int64_t>(cohort[k]) * intermediate * hidden;
          CheckBlas(cublasGemmEx(
              blas, CUBLAS_OP_T, CUBLAS_OP_N, intermediate, 1, hidden, &alpha,
              In<bf16>(c, 3) + offset, CUDA_R_16BF, hidden,
              In<bf16>(c, 0), CUDA_R_16BF, hidden, &beta,
              Out<bf16>(h) + static_cast<int64_t>(k) * intermediate, CUDA_R_16BF, intermediate,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        ReluSquaredKernel<<<CeilDiv(topk * intermediate, 256), 256, 0, s>>>(
            Out<bf16>(h), static_cast<int64_t>(topk) * intermediate);
        CheckLaunch();
        for (int k = 0; k < topk; ++k) {
          const auto offset = static_cast<int64_t>(cohort[k]) * intermediate * hidden;
          CheckBlas(cublasGemmEx(
              blas, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, intermediate, &alpha,
              In<bf16>(c, 4) + offset, CUDA_R_16BF, intermediate,
              Out<bf16>(h) + static_cast<int64_t>(k) * intermediate, CUDA_R_16BF, intermediate,
              &beta, down_values + static_cast<int64_t>(k) * hidden, CUDA_R_16BF, hidden,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        ExpertPartKernel<<<CeilDiv(topk * hidden, 256), 256, 0, s>>>(
            down_values, Out<float>(part), static_cast<int64_t>(topk) * hidden);
        CheckLaunch();
      } else {
        MoeFcKernel<Quantized, true><<<dim3(topk, CeilDiv(intermediate, 8)), 256, 0, s>>>(
            In<bf16>(c, 0), In<int32_t>(c, 1), experts, hidden, intermediate,
            c.GetInput(3).GetTensorRawData(), s1, block, Out<bf16>(h), Out<float>(part));
        CheckLaunch();
        MoeFcKernel<Quantized, false><<<dim3(topk, CeilDiv(hidden, 8)), 256, 0, s>>>(
            Out<bf16>(h), In<int32_t>(c, 1), experts, hidden, intermediate,
            c.GetInput(5).GetTensorRawData(), s2, block, Out<bf16>(h), Out<float>(part));
        CheckLaunch();
      }
      MoeFinalizeKernel<<<CeilDiv(CeilDiv(hidden, 4), 64), 64, 0, s>>>(
          Out<float>(part), In<float>(c, 2), In<int32_t>(c, 1), topk, hidden, source_order, Out<bf16>(y));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

#include "chunked_prefill.cuh"

template <typename Derived, typename State>
struct LightningOp : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try { *kernel = new State(api, info); return nullptr; }
    catch (const std::exception& e) { return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what()); }
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
};
struct MambaStepOp : LightningOp<MambaStepOp, MambaStepState> {
  const char* GetName() const { return "MambaStep"; }
  size_t GetInputTypeCount() const { return 10; }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t i) const {
    return i == 9 ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_REQUIRED;
  }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType types[] = {BF, BF, F32, BF, BF, F32, F32, F32, F32, I32};
    return types[i];
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType types[] = {BF, BF, F32, BF, F32};
    return types[i];
  }
};
struct MambaChunkOp : LightningOp<MambaChunkOp, MambaChunkState> {
  const char* GetName() const { return "MambaChunk"; }
  size_t GetInputTypeCount() const { return 10; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return MambaStepOp().GetInputType(i);
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    return MambaStepOp().GetOutputType(i);
  }
};
struct MoeRouteOp : LightningOp<MoeRouteOp, MoeRouteState> {
  const char* GetName() const { return "MoeRoute"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i < 2 ? BF : F32; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return i == 0 ? I32 : F32; }
};
struct RmsNormOp : LightningOp<RmsNormOp, RmsNormState> {
  const char* GetName() const { return "RMSNorm"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i == 0 ? BF : F32; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct RowLinearOp : LightningOp<RowLinearOp, RowLinearState> {
  const char* GetName() const { return "RowLinear"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct CachedAttentionChunkOp : LightningOp<CachedAttentionChunkOp, CachedAttentionChunkState> {
  const char* GetName() const { return "CachedAttentionChunk"; }
  size_t GetInputTypeCount() const { return 7; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i >= 5 ? I32 : BF; }
  OrtMemType GetInputMemoryType(size_t i) const { return i == 6 ? OrtMemTypeCPUInput : OrtMemTypeDefault; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct MambaWindowOp : LightningOp<MambaWindowOp, MambaWindowState> {
  const char* GetName() const { return "MambaWindow"; }
  size_t GetInputTypeCount() const { return 10; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i == 9 ? I32 : (i == 2 || (i >= 5 && i <= 8) ? F32 : BF);
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return i == 2 || i == 4 ? F32 : BF; }
};
struct W4LinearOp : LightningOp<W4LinearOp, W4LinearState> {
  const char* GetName() const { return "W4Linear"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i == 1 ? U8 : BF; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
template <bool Quantized>
struct MoeDecodeOp : LightningOp<MoeDecodeOp<Quantized>, MoeDecodeState<Quantized>> {
  const char* GetName() const { return Quantized ? "MoeDecode" : "MoeDecodeBF16"; }
  size_t GetInputTypeCount() const { return Quantized ? 7 : 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    if (i == 1) return I32;
    if (i == 2) return F32;
    if (Quantized && (i == 3 || i == 5)) return U8;
    return BF;
  }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return i == 2 ? F32 : BF; }
};
struct MoeChunkOp : LightningOp<MoeChunkOp, MoeChunkState> {
  const char* GetName() const { return "MoeChunkBF16"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return MoeDecodeOp<false>().GetInputType(i); }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return MoeDecodeOp<false>().GetOutputType(i); }
};
struct MoeCohortOp : LightningOp<MoeCohortOp, MoeCohortState> {
  const char* GetName() const { return "MoeCohortBF16"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return MoeDecodeOp<false>().GetInputType(i); }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return MoeDecodeOp<false>().GetOutputType(i); }
};
struct MoeDeviceOp : LightningOp<MoeDeviceOp, MoeDeviceState> {
  const char* GetName() const { return "MoeDeviceBF16"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return MoeDecodeOp<false>().GetInputType(i); }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return MoeDecodeOp<false>().GetOutputType(i); }
};
struct MoeVerifyOp : LightningOp<MoeVerifyOp, MoeVerifyState> {
  const char* GetName() const { return "MoeVerifyBF16"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return MoeDecodeOp<false>().GetInputType(i); }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return MoeDecodeOp<false>().GetOutputType(i); }
};
}  // namespace

#if defined(_WIN32)
#define LIGHTNING_EXPORT __declspec(dllexport)
#else
#define LIGHTNING_EXPORT __attribute__((visibility("default")))
#endif
extern "C" LIGHTNING_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto api = api_base->GetApi(ORT_API_VERSION);
  if (!api) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "lightning: unsupported ORT API");
  Ort::InitApi(api);
  try {
    static MambaStepOp mamba;
    static MambaChunkOp mamba_chunk;
    static MambaWindowOp mamba_window;
    static MoeChunkOp moe_chunk;
    static MoeCohortOp moe_cohort;
    static MoeDeviceOp moe_device;
    static MoeVerifyOp moe_verify;
    static RmsNormOp rms;
    static RowLinearOp row_linear;
    static CachedAttentionChunkOp attention_chunk;
    static MoeRouteOp route;
    static W4LinearOp linear;
    static MoeDecodeOp<false> bf16_decode;
    static MoeDecodeOp<true> w4_decode;
    static Ort::CustomOpDomain domain("lightning");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&mamba);
      domain.Add(&mamba_chunk);
      domain.Add(&mamba_window);
      domain.Add(&moe_chunk);
      domain.Add(&moe_cohort);
      domain.Add(&moe_device);
      domain.Add(&moe_verify);
      domain.Add(&rms);
      domain.Add(&row_linear);
      domain.Add(&attention_chunk);
      domain.Add(&route);
      domain.Add(&linear);
      domain.Add(&bf16_decode);
      domain.Add(&w4_decode);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) { return api->CreateStatus(ORT_FAIL, e.what()); }
}
