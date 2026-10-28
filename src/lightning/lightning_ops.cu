// Source-derived single-token adaptation of src/bolt/bolt_ops.cu; see README.md for numerical differences.
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
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
  xbc[c] = __float2bfloat16(Silu(acc));
}

// Bolt's single-token QP=2 state-partitioned kernel, retaining [heads, DS, HD] storage.
template <int DS, int HD, int QP>
__global__ void MambaSsmKernel(const bf16* proj, int di, int channels, int heads_per_group, int groups,
                              const bf16* xbc, const float* past, const float* dt_bias,
                              const float* a, const float* skip, float tmin, float* present, float* ypre) {
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
    sDt = fmaxf(raw > 20.f ? raw : log1pf(expf(raw)), tmin);
  }
  __syncthreads();
  const float decay = expf(a[h] * sDt), x = B2F(xbc[h * HD + p]), dtx = sDt * x;
  const int64_t base = (static_cast<int64_t>(h) * DS + part * NH) * HD + p;
  float acc = 0.f;
#pragma unroll
  for (int n = 0; n < NH; ++n) {
    const int64_t offset = base + static_cast<int64_t>(n) * HD;
    const float st = past[offset] * decay + dtx * sB[part * NH + n];
    present[offset] = st;
    acc += sC[part * NH + n] * st;
  }
  sPart[part][p] = acc;
  __syncthreads();
  if (part == 0) {
    float sum = skip[h] * x;
#pragma unroll
    for (int q = 0; q < QP; ++q) sum += sPart[q][p];
    ypre[h * HD + p] = sum;
  }
}

// Fixture dimensions use the same recurrence and layout without template restrictions.
__global__ void MambaSsmGenericKernel(const bf16* proj, int di, int channels, int hd, int ds,
                                     int heads_per_group, int groups, const bf16* xbc,
                                     const float* past, const float* dt_bias, const float* a,
                                     const float* skip, float tmin, float* present, float* ypre) {
  const int feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature >= di) return;
  const int h = feature / hd, p = feature % hd, g = h / heads_per_group;
  const float raw = B2F(proj[di + channels + h]) + dt_bias[h];
  const float dt = fmaxf(raw > 20.f ? raw : log1pf(expf(raw)), tmin);
  const float decay = expf(a[h] * dt), x = B2F(xbc[feature]), dtx = dt * x;
  float acc = 0.f;
  for (int n = 0; n < ds; ++n) {
    const int64_t offset = (static_cast<int64_t>(h) * ds + n) * hd + p;
    const float st = past[offset] * decay + dtx * B2F(xbc[di + g * ds + n]);
    present[offset] = st;
    acc += B2F(xbc[di + groups * ds + g * ds + n]) * st;
  }
  ypre[feature] = skip[h] * x + acc;
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
      CheckArity(c, 9, 5);
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
            tmin, Out<float>(state), Out<float>(ypre));
      } else {
        MambaSsmGenericKernel<<<CeilDiv(di, 128), 128, 0, s>>>(
            In<bf16>(c, 0), di, channels, hd, ds, heads / groups, groups, Out<bf16>(xbc),
            In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
            tmin, Out<float>(state), Out<float>(ypre));
      }
      CheckLaunch();
      MambaNormKernel<<<groups, 256, 0, s>>>(
          In<bf16>(c, 0), di / groups, eps, Out<float>(ypre), In<float>(c, 8), Out<bf16>(y));
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
  const int e = blockIdx.y * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (e >= experts) return;
  const bf16* w = gate + static_cast<int64_t>(e) * hidden;
  float acc = 0.f;
  for (int64_t k = lane * 8; k < hidden; k += 256)
    acc += DotBF16x8(x + k, w + k, hidden - static_cast<int>(k));
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) scores[e] = 1.f / (1.f + expf(-acc));
}

// Bolt's iterative warp argmax, with explicit availability and smallest-ID ties even at -infinity.
__global__ void RouterTopkKernel(const float* scores, const float* bias, int experts, int topk,
                                float scale, int32_t* indices, float* probs) {
  const int lane = threadIdx.x;
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
  if (lane < topk) {
    indices[lane] = my_idx;
    probs[lane] = my_sc / (sum + 1e-20f) * scale;
  }
}

struct MoeRouteState {
  const OrtApi& api;
  int topk;
  float scale;
  MoeRouteState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = PositiveInt(k.GetAttribute<int64_t>("topk"), "MoeRoute: invalid topk");
    scale = k.GetAttribute<float>("scale");
    Require(topk <= 32 && std::isfinite(scale) && scale >= 0.f, "MoeRoute: topk<=32 and finite nonnegative scale required");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 3, 3);
      const auto xs = Dimensions(c.GetInput(0)), ws = Dimensions(c.GetInput(1));
      Require(xs.size() == 2 && xs[0] == 1, "MoeRoute: x must be [1,H]");
      const int hidden = PositiveInt(xs[1], "MoeRoute: invalid H");
      Require(ws.size() == 2, "MoeRoute: gate must be [E,H]");
      const int experts = PositiveInt(ws[0], "MoeRoute: invalid E");
      Require(experts <= 512 && topk <= experts, "MoeRoute: E<=512 and topk<=E required");
      CheckTensor(c, 0, {1, hidden}, BF, "MoeRoute: x must be BF16 [1,H]");
      CheckTensor(c, 1, {experts, hidden}, BF, "MoeRoute: gate must be BF16 [E,H]");
      CheckTensor(c, 2, {experts}, F32, "MoeRoute: bias must be FLOAT [E]");
      auto indices = c.GetOutput(0, Shape{1, topk});
      auto probs = c.GetOutput(1, Shape{1, topk});
      auto scores = c.GetOutput(2, Shape{1, experts});
      const auto s = Stream(api, raw);
      RouterLogitsKernel<<<dim3(1, CeilDiv(experts, 8)), 256, 0, s>>>(
          In<bf16>(c, 0), In<bf16>(c, 1), hidden, experts, Out<float>(scores));
      CheckLaunch();
      RouterTopkKernel<<<1, 32, 0, s>>>(
          Out<float>(scores), In<float>(c, 2), experts, topk, scale, Out<int32_t>(indices), Out<float>(probs));
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

// Bolt's (selected expert, ceil(output_rows/8)) grid and warp ownership, generalized from latent=1024.
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
__global__ void MoeFinalizeKernel(const float* part, const float* probs, int topk, int hidden, bf16* out) {
  const int64_t n = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (n >= hidden) return;
  float acc[4] = {};
  for (int k = 0; k < topk; ++k) {
    const float* p = part + static_cast<int64_t>(k) * hidden + n;
    if (n + 4 <= hidden && Aligned(p, 16)) {
      const float4 v = *reinterpret_cast<const float4*>(p);
      acc[0] += probs[k] * v.x;
      acc[1] += probs[k] * v.y;
      acc[2] += probs[k] * v.z;
      acc[3] += probs[k] * v.w;
    } else {
      for (int j = 0; j < 4 && n + j < hidden; ++j) acc[j] += probs[k] * p[j];
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

template <bool Quantized>
struct MoeDecodeState {
  const OrtApi& api;
  int topk, block = 32;
  MoeDecodeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    topk = PositiveInt(k.GetAttribute<int64_t>("topk"), "MoeDecode: invalid topk");
    Require(topk <= 32, "MoeDecode: topk must be in [1,32]");
    if constexpr (Quantized) {
      block = PositiveInt(k.GetAttribute<int64_t>("block_size"), "MoeDecode: invalid block_size");
      Require(block % 2 == 0, "MoeDecode: block_size must be positive and even");
    }
  }
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
      MoeFcKernel<Quantized, true><<<dim3(topk, CeilDiv(intermediate, 8)), 256, 0, s>>>(
          In<bf16>(c, 0), In<int32_t>(c, 1), experts, hidden, intermediate,
          c.GetInput(3).GetTensorRawData(), s1, block, Out<bf16>(h), Out<float>(part));
      CheckLaunch();
      MoeFcKernel<Quantized, false><<<dim3(topk, CeilDiv(hidden, 8)), 256, 0, s>>>(
          Out<bf16>(h), In<int32_t>(c, 1), experts, hidden, intermediate,
          c.GetInput(Quantized ? 5 : 4).GetTensorRawData(), s2, block, Out<bf16>(h), Out<float>(part));
      CheckLaunch();
      MoeFinalizeKernel<<<CeilDiv(CeilDiv(hidden, 4), 64), 64, 0, s>>>(
          Out<float>(part), In<float>(c, 2), topk, hidden, Out<bf16>(y));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

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
  size_t GetInputTypeCount() const { return 9; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType types[] = {BF, BF, F32, BF, BF, F32, F32, F32, F32};
    return types[i];
  }
  size_t GetOutputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType types[] = {BF, BF, F32, BF, F32};
    return types[i];
  }
};
struct MoeRouteOp : LightningOp<MoeRouteOp, MoeRouteState> {
  const char* GetName() const { return "MoeRoute"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i < 2 ? BF : F32; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return i == 0 ? I32 : F32; }
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
    static MoeRouteOp route;
    static MoeDecodeOp<false> bf16_decode;
    static MoeDecodeOp<true> w4_decode;
    static Ort::CustomOpDomain domain("lightning");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&mamba);
      domain.Add(&route);
      domain.Add(&bf16_decode);
      domain.Add(&w4_decode);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) { return api->CreateStatus(ORT_FAIL, e.what()); }
}
