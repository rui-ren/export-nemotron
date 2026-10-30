// Nemotron-3.5 Bolt decode ops for ONNX Runtime (CUDA EP, custom-op ABI), domain "bolt".
//
//   MambaStep   : Mamba2 mixer core of R <= 16 rows (conv1d+SiLU, dt/decay, SSM state update, D skip, gated grouped RMSNorm).
//                 Replaces the ~35-node ONNX chain between in_proj and out_proj; states are updated in place friendly.
//   MoeRoute    : fp32 router (sigmoid, bias-corrected top-k, normalisation, scaling) emitting the dense top-k selection /
//                 weight matrices consumed by QMoE (router_probs / router_weights).
//   Relu2       : y = bf16(relu(x)^2)
//   Nvfp4Gemv   : y[R,N] = x[R,K] @ W^T with a modelopt NVFP4 weight (u8 [N,K/2], fp8 e4m3 block-16 scales, fp32 global scale).
#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <mutex>
#include <stdexcept>
#include <vector>



#ifndef BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
#define BOLT_MAMBA_ROUND_CONV_BEFORE_SILU 0
#endif
#ifndef BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM
#define BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM 0
#endif
#ifndef BOLT_IQ2_NATIVE_FULL_SCALE
#define BOLT_IQ2_NATIVE_FULL_SCALE 0
#endif

namespace {

using bf16 = __nv_bfloat16;
constexpr int kMaxRows = 16;

__device__ __forceinline__ float B2F(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float Silu(float v) { return v / (1.0f + expf(-v)); }

__device__ __forceinline__ float MambaNormReadout(float v) {
#if BOLT_MAMBA_ROUND_READOUT_BEFORE_NORM
  // Preserve the BF16 scan-output boundary before FP32 gate/variance math.
  return B2F(__float2bfloat16(v));
#else
  return v;
#endif
}

void CheckLaunch() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

cudaStream_t Stream(const OrtApi& api, OrtKernelContext* context) {
  void* raw = nullptr;
  Ort::ThrowOnError(api.KernelContext_GetGPUComputeStream(context, &raw));
  return static_cast<cudaStream_t>(raw);
}

template <typename T>
const T* In(const Ort::KernelContext& c, size_t i) {
  return reinterpret_cast<const T*>(c.GetInput(i).GetTensorRawData());
}
template <typename T>
T* Out(Ort::UnownedValue& v) {
  return reinterpret_cast<T*>(v.GetTensorMutableRawData());
}
int64_t Dim(const Ort::ConstValue& v, size_t i) { return v.GetTensorTypeAndShapeInfo().GetShape().at(i); }
int64_t Count(const Ort::ConstValue& v) {
  int64_t n = 1;
  for (auto d : v.GetTensorTypeAndShapeInfo().GetShape()) n *= d;
  return n;
}

// ------------------------------------------------------------------------------------------------------------------
// MambaStep
// ------------------------------------------------------------------------------------------------------------------
// proj row layout: [gate (di) | xBC (conv_dim = di + 2*ng*ds) | dt (heads)]
template <int R_FIXED = 0, bool ALL_ROWS = false>
__global__ void MambaConvKernel(const bf16* __restrict__ proj, int R_rt, int proj_stride, int di, int conv_dim,
                                const bf16* __restrict__ conv_state_in, const bf16* __restrict__ conv_w,
                                const bf16* __restrict__ conv_b, bf16* __restrict__ conv_state_out,
                                bf16* __restrict__ xbc_act, const int* __restrict__ sel, bool all_rows_rt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  const int R = R_FIXED > 0 ? R_FIXED : R_rt;
  const bool all_rows = R_FIXED > 0 ? ALL_ROWS : all_rows_rt;
  // Verify mode: the input state is row `sel` of a [rows, conv_dim, 3] buffer and the state after token r goes to row r.
  if (sel) conv_state_in += static_cast<int64_t>(*sel) * conv_dim * 3;
  float inp[3 + (R_FIXED > 0 ? R_FIXED : kMaxRows)];
  inp[0] = B2F(conv_state_in[c * 3 + 0]);
  inp[1] = B2F(conv_state_in[c * 3 + 1]);
  inp[2] = B2F(conv_state_in[c * 3 + 2]);
  const uint2 wv = __ldg(reinterpret_cast<const uint2*>(conv_w + c * 4));
  const bf16* wp = reinterpret_cast<const bf16*>(&wv);
  const float w0 = B2F(wp[0]), w1 = B2F(wp[1]), w2 = B2F(wp[2]), w3 = B2F(wp[3]);
  const float bias = B2F(conv_b[c]);
  if constexpr (R_FIXED > 0) {
#pragma unroll
    for (int r = 0; r < R_FIXED; ++r) inp[3 + r] = B2F(proj[static_cast<int64_t>(r) * proj_stride + di + c]);
#pragma unroll
    for (int r = 0; r < R_FIXED; ++r) {
      float acc = bias + w0 * inp[r] + w1 * inp[r + 1] + w2 * inp[r + 2] + w3 * inp[r + 3];
#if BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
      acc = B2F(__float2bfloat16(acc));
#endif
      xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
    }
    if constexpr (ALL_ROWS) {
#pragma unroll
      for (int r = 0; r < R_FIXED; ++r) {
        bf16* o = conv_state_out + (static_cast<int64_t>(r) * conv_dim + c) * 3;
        o[0] = __float2bfloat16(inp[r + 1]);
        o[1] = __float2bfloat16(inp[r + 2]);
        o[2] = __float2bfloat16(inp[r + 3]);
      }
    } else {
      conv_state_out[c * 3 + 0] = __float2bfloat16(inp[R_FIXED]);
      conv_state_out[c * 3 + 1] = __float2bfloat16(inp[R_FIXED + 1]);
      conv_state_out[c * 3 + 2] = __float2bfloat16(inp[R_FIXED + 2]);
    }
  } else {
    for (int r = 0; r < R; ++r) inp[3 + r] = B2F(proj[static_cast<int64_t>(r) * proj_stride + di + c]);
    for (int r = 0; r < R; ++r) {
      float acc = bias + w0 * inp[r] + w1 * inp[r + 1] + w2 * inp[r + 2] + w3 * inp[r + 3];
#if BOLT_MAMBA_ROUND_CONV_BEFORE_SILU
      acc = B2F(__float2bfloat16(acc));
#endif
      xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
    }
    if (all_rows) {
      for (int r = 0; r < R; ++r) {
        bf16* o = conv_state_out + (static_cast<int64_t>(r) * conv_dim + c) * 3;
        o[0] = __float2bfloat16(inp[r + 1]);
        o[1] = __float2bfloat16(inp[r + 2]);
        o[2] = __float2bfloat16(inp[r + 3]);
      }
    } else {
      conv_state_out[c * 3 + 0] = __float2bfloat16(inp[R]);
      conv_state_out[c * 3 + 1] = __float2bfloat16(inp[R + 1]);
      conv_state_out[c * 3 + 2] = __float2bfloat16(inp[R + 2]);
    }
  }
}

// Block (h, pb) owns head h and p-tile pb of width PT (HD = PT * gridDim.y). State layout [heads, ds, hd] (hd contiguous), fp32.
// Thread (p_local, part) owns state[n in part][p_base + p_local]; the QP parts split d_state.
template <int DS, int HD, int QP, int PT = HD, int R_FIXED = 0, bool ALL_ROWS = false>
__global__ void MambaSsmKernel(const bf16* __restrict__ proj, int R_rt, int proj_stride, int di, int conv_dim, int heads,
                               int heads_per_group, const bf16* __restrict__ xbc_act,
                               const float* __restrict__ ssm_in, const float* __restrict__ dt_bias,
                               const float* __restrict__ a_neg, const float* __restrict__ d_skip, float tmin,
                               float* __restrict__ ssm_out, float* __restrict__ ypre, const int* __restrict__ sel,
                               bool all_rows_rt) {
  const int h = blockIdx.x;
  const int p_base = blockIdx.y * PT;
  const int g = h / heads_per_group;
  const int tid = threadIdx.x;
  const int pl = tid % PT;
  const int p = p_base + pl;
  const int half = tid / PT;  // 0 .. QP-1
  const int R = R_FIXED > 0 ? R_FIXED : R_rt;
  const bool all_rows = R_FIXED > 0 ? ALL_ROWS : all_rows_rt;
  constexpr int NH = DS / QP;
  constexpr int MAX_R = R_FIXED > 0 ? R_FIXED : kMaxRows;
  __shared__ float sB[MAX_R][DS];
  __shared__ float sC[MAX_R][DS];
  __shared__ float sPart[MAX_R][QP][PT];
  __shared__ float sDt[MAX_R];
  const int b_off = di + g * DS;
  const int c_off = di + (heads / heads_per_group) * DS + g * DS;
  for (int i = tid; i < R * DS; i += blockDim.x) {
    const int r = i / DS, n = i % DS;
    sB[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + b_off + n]);
    sC[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + c_off + n]);
  }
  if (tid < R) {
    const float raw = B2F(proj[static_cast<int64_t>(tid) * proj_stride + di + conv_dim + h]) + dt_bias[h];
    float dt = raw > 20.0f ? raw : log1pf(expf(raw));
    sDt[tid] = fmaxf(dt, tmin);
  }
  __syncthreads();
  float st[NH];
  const int64_t row_stride = static_cast<int64_t>(heads) * DS * HD;
  const float* sp = ssm_in + (sel ? static_cast<int64_t>(*sel) * row_stride : 0) + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
  for (int n = 0; n < NH; ++n) st[n] = sp[static_cast<int64_t>(n) * HD];
  const float A = a_neg[h];
  const float D = d_skip[h];
  if constexpr (R_FIXED > 0) {
#pragma unroll
    for (int r = 0; r < R_FIXED; ++r) {
      const float dt = sDt[r];
      const float dA = expf(A * dt);
      const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p]);
      const float dtx = dt * x;
      float acc = 0.f;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        const int nn = half * NH + n;
        st[n] = st[n] * dA + dtx * sB[r][nn];
        acc += sC[r][nn] * st[n];
      }
      sPart[r][half][pl] = acc;
      if constexpr (ALL_ROWS) {
        float* orow = ssm_out + r * row_stride + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
        for (int n = 0; n < NH; ++n) orow[static_cast<int64_t>(n) * HD] = st[n];
      }
    }
    if constexpr (!ALL_ROWS) {
      float* op = ssm_out + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
      for (int n = 0; n < NH; ++n) op[static_cast<int64_t>(n) * HD] = st[n];
    }
  } else {
    for (int r = 0; r < R; ++r) {
      const float dt = sDt[r];
      const float dA = expf(A * dt);
      const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p]);
      const float dtx = dt * x;
      float acc = 0.f;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        const int nn = half * NH + n;
        st[n] = st[n] * dA + dtx * sB[r][nn];
        acc += sC[r][nn] * st[n];
      }
      sPart[r][half][pl] = acc;
      if (all_rows) {
        float* orow = ssm_out + r * row_stride + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
        for (int n = 0; n < NH; ++n) orow[static_cast<int64_t>(n) * HD] = st[n];
      }
    }
    if (!all_rows) {
      float* op = ssm_out + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
      for (int n = 0; n < NH; ++n) op[static_cast<int64_t>(n) * HD] = st[n];
    }
  }
  __syncthreads();
  for (int i = tid; i < R * PT; i += blockDim.x) {
    const int r = i / PT, pp = i % PT;
    const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p_base + pp]);
    float sum = D * x;
#pragma unroll
    for (int q = 0; q < QP; ++q) sum += sPart[r][q][pp];
    ypre[static_cast<int64_t>(r) * di + h * HD + p_base + pp] = sum;
  }
}

#include "cached_source_kernel.cuh"

// grid (R, groups), block 256; gate is proj[r][0..di)
__global__ void MambaNormKernel(const bf16* __restrict__ proj, int proj_stride, int di, int group_size, float eps,
                                const float* __restrict__ ypre, const float* __restrict__ norm_w, bf16* __restrict__ y) {
  const int r = blockIdx.x, g = blockIdx.y;
  const int64_t base = static_cast<int64_t>(r) * di + static_cast<int64_t>(g) * group_size;
  const bf16* gate = proj + static_cast<int64_t>(r) * proj_stride + static_cast<int64_t>(g) * group_size;
  __shared__ float red[32];
  if (group_size == 1024 && blockDim.x == 256) {
    const int tid = threadIdx.x;
    float v[4];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + k * 256;
      const float gt = B2F(__ldg(gate + i));
      v[k] = MambaNormReadout(__ldg(ypre + base + i)) * (gt / (1.0f + expf(-gt)));
      ss += v[k] * v[k];
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
      float w = tid < 8 ? red[tid] : 0.f;
      for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffff, w, o);
      if (tid == 0) red[0] = w;
    }
    __syncthreads();
    const float rs = rsqrtf(red[0] / group_size + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + k * 256;
      y[base + i] = __float2bfloat16(v[k] * rs * __ldg(norm_w + g * 1024 + i));
    }
    return;
  }
  float ss = 0.f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = MambaNormReadout(ypre[base + i]) * (gt / (1.0f + expf(-gt)));
    ss += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / group_size + eps);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = MambaNormReadout(ypre[base + i]) * (gt / (1.0f + expf(-gt)));
    y[base + i] = __float2bfloat16(v * rs * norm_w[g * group_size + i]);
  }
}

#include "norm_source_kernel.cuh"

struct MambaStepState {
  const OrtApi& api;
  int heads, hd, ds, ng;
  float eps, tmin;
  bool verify;
  MambaStepState(const OrtApi& a, const OrtKernelInfo* info, bool verify_mode = false) : api(a), verify(verify_mode) {
    Ort::ConstKernelInfo k(info);
    heads = static_cast<int>(k.GetAttribute<int64_t>("heads"));
    hd = static_cast<int>(k.GetAttribute<int64_t>("head_dim"));
    ds = static_cast<int>(k.GetAttribute<int64_t>("d_state"));
    ng = static_cast<int>(k.GetAttribute<int64_t>("n_groups"));
    eps = k.GetAttribute<float>("epsilon");
    tmin = k.GetAttribute<float>("time_step_min");
    if (hd != 64 || ds != 128) throw std::invalid_argument("MambaStep: only head_dim=64, d_state=128 are implemented");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto proj = c.GetInput(0);
      const int R = static_cast<int>(Dim(proj, 0));
      if (R < 1 || R > kMaxRows) throw std::invalid_argument("MambaStep: rows must be in [1,16]");
      const int di = heads * hd;
      const int conv_dim = di + 2 * ng * ds;
      const int stride = di + conv_dim + heads;
      if (Dim(proj, 1) != stride) throw std::invalid_argument("MambaStep: bad proj width");
      auto y = c.GetOutput(0, std::vector<int64_t>{R, di});
      const int* sel = verify ? In<int>(c, 9) : nullptr;
      if (verify && Dim(c.GetInput(1), 0) < R) throw std::invalid_argument("MambaVerify: state buffer needs >= R rows");
      auto conv_out = c.GetOutput(1, c.GetInput(1).GetTensorTypeAndShapeInfo().GetShape());
      auto ssm_out = c.GetOutput(2, c.GetInput(2).GetTensorTypeAndShapeInfo().GetShape());
      auto xbc_act = c.GetOutput(3, std::vector<int64_t>{R, conv_dim});
      auto ypre = c.GetOutput(4, std::vector<int64_t>{R, di});
      cudaStream_t s = Stream(api, raw);
      if (R == 8 && verify) {
        MambaConvKernel<8, true><<<(conv_dim + 63) / 64, 64, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, In<bf16>(c, 1),
                                                                     In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv_out),
                                                                     Out<bf16>(xbc_act), sel, verify);
        MambaCachedSourceKernel<<<dim3(heads, 8), 256, 0, s>>>(
            In<bf16>(c, 0), R, stride, di, conv_dim, heads, ng, Out<bf16>(xbc_act),
            In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
            Out<float>(ssm_out), Out<float>(ypre), sel);
      } else {
        MambaConvKernel<<<(conv_dim + 127) / 128, 128, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, In<bf16>(c, 1),
                                                               In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv_out),
                                                               Out<bf16>(xbc_act), sel, verify);
        // QP = 4 state parts (256 threads/head) for the multi-row verify path, 2 parts (128 threads) for single-token decode
        if (verify) MambaCachedSourceKernel<<<dim3(heads,8),256,0,s>>>(
            In<bf16>(c,0),R,stride,di,conv_dim,heads,ng,Out<bf16>(xbc_act),
            In<float>(c,2),In<float>(c,5),In<float>(c,6),In<float>(c,7),
            Out<float>(ssm_out),Out<float>(ypre),sel);
        else if (R > 1) MambaSsmKernel<128, 64, 4><<<heads, 256, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, heads, heads / ng,
                                                       Out<bf16>(xbc_act), In<float>(c, 2), In<float>(c, 5),
                                                       In<float>(c, 6), In<float>(c, 7), tmin, Out<float>(ssm_out),
                                                       Out<float>(ypre), sel, verify);
        else MambaSsmKernel<128, 64, 2><<<heads, 128, 0, s>>>(In<bf16>(c, 0), R, stride, di, conv_dim, heads, heads / ng,
                                                       Out<bf16>(xbc_act), In<float>(c, 2), In<float>(c, 5),
                                                       In<float>(c, 6), In<float>(c, 7), tmin, Out<float>(ssm_out),
                                                       Out<float>(ypre), sel, verify);
      }
      LaunchMambaNormSource(s,R,ng,In<bf16>(c,0),stride,di,di/ng,eps,
                            Out<float>(ypre),In<float>(c,8),Out<bf16>(y));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

// ------------------------------------------------------------------------------------------------------------------
#include "ssd_binary_source.cuh"
#include "parallel_source_readout.cuh"
#include "mamba_scan.cuh"


template <typename Derived, typename State>
struct BoltOp : Ort::CustomOpBase<Derived, State, true> {
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try {
      *kernel = new State(api, info);
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, e.what());
    }
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const { return INPUT_OUTPUT_REQUIRED; }
};

constexpr auto BF = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
constexpr auto F32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto U8 = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
constexpr auto I32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

struct MambaScanVerifyOp : BoltOp<MambaScanVerifyOp, MambaScanVerifyState> {
  const char* GetName() const { return "MambaScanVerifyE6"; }
  size_t GetInputTypeCount() const { return 14; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType t[14] = {
        BF, BF, F32, BF, BF, F32, F32, F32, F32, I32, I32, BF, I32, F32};
    return t[i];
  }
  OrtMemType GetInputMemoryType(size_t i) const {
    return i == 10 ? OrtMemTypeCPUInput : OrtMemTypeDefault;
  }
  size_t GetOutputTypeCount() const { return 8; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[8] = {BF, BF, F32, BF, F32, BF, I32, F32};
    return t[i];
  }
};

struct E6DebugState : MambaScanVerifyState {
  E6DebugState(const OrtApi& a,const OrtKernelInfo* info):MambaScanVerifyState(a,info,true){}
};
struct E6DebugOp : BoltOp<E6DebugOp,E6DebugState> {
  const char* GetName() const { return "MambaScanVerifyE6Debug"; }
  size_t GetInputTypeCount() const { return 14; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType t[14]={
        BF,BF,F32,BF,BF,F32,F32,F32,F32,I32,I32,BF,I32,F32};
    return t[i];
  }
  OrtMemType GetInputMemoryType(size_t i) const {
    return i==10 ? OrtMemTypeCPUInput:OrtMemTypeDefault;
  }
  size_t GetOutputTypeCount() const { return 10; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    static const ONNXTensorElementDataType t[10]={BF,BF,F32,BF,F32,BF,I32,F32,F32,BF};
    return t[i];
  }
};
} // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,const OrtApiBase* base) {
  const OrtApi* api=base->GetApi(24);
  if (!api) return nullptr;
  try {
    Ort::InitApi(api);
    static MambaScanVerifyOp operation;
    static E6DebugOp debug_operation;
    static Ort::CustomOpDomain domain("bolt_source_mamba");
    static std::once_flag once;
    std::call_once(once,[&](){ domain.Add(&operation);domain.Add(&debug_operation); });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch(const std::exception& error) {
    return api->CreateStatus(ORT_FAIL,error.what());
  }
}

extern "C" int BoltMambaScanProbe(
    void* stream_raw,const void* proj_raw,const void* conv_raw,const float* ssm,
    const void* w_raw,const void* b_raw,const float* bias,const float* A,const float* D,
    const float* norm,const int* sel,const void* past_raw,const float* initial,
    void* y_raw,void* conv_out_raw,float* state,void* xbc_raw,float* ypre,
    void* hist_raw,int* lengths,float* scratch,int start,int R,int scan) {
  try {
    auto stream=static_cast<cudaStream_t>(stream_raw);
    auto proj=static_cast<const bf16*>(proj_raw);
    auto xbc=static_cast<bf16*>(xbc_raw);
    MambaConvKernel<<<80,128,0,stream>>>(
        proj,R,18560,8192,10240,static_cast<const bf16*>(conv_raw),
        static_cast<const bf16*>(w_raw),static_cast<const bf16*>(b_raw),
        static_cast<bf16*>(conv_out_raw),xbc,sel,true);
    if (scan) {
      LaunchSourceScan(stream,proj,xbc,static_cast<const bf16*>(past_raw),
          static_cast<bf16*>(hist_raw),lengths,initial,bias,A,D,state,ypre,
          scratch,scratch+R*128*256,start,R,128,8,128,64,0.f);
    } else {
      MambaCachedSourceKernel<<<dim3(128,8),256,0,stream>>>(
          proj,R,18560,8192,10240,128,8,xbc,ssm,bias,A,D,state,ypre,sel);
      E6CachedDiagnostics<<<256,256,0,stream>>>(
          proj,bias,A,scratch+R*128*256,R);
    }
    LaunchMambaNormSource(stream,R,8,proj,18560,8192,1024,1e-5f,
                          ypre,norm,static_cast<bf16*>(y_raw));
    return static_cast<int>(cudaGetLastError());
  } catch(const std::exception& error) {
    fprintf(stderr,"E6 probe: %s\n",error.what());
    return -1;
  }
}
