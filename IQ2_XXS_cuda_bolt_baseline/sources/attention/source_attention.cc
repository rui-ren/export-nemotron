// Source eager attention with dynamic logical prefill M; not Torch-free.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifndef SOURCE_CUBLAS_PATH
#error "Source cuBLAS must be pinned"
#endif

namespace {
thread_local size_t peak_allocated_bytes = 0;
void cu(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
constexpr int64_t capacity = 135168;
const std::vector<int64_t> cache_shape{1, 2, capacity, 128};
}

extern "C" const char* source_attention_cublas_path() {
  Dl_info info{};
  return dladdr(reinterpret_cast<void*>(&cublasSgemm_v2), &info) ? info.dli_fname : "";
}
extern "C" size_t source_attention_allocator_peak_bytes() { return peak_allocated_bytes; }

struct SourceAttentionKernel {
  const OrtApi& api;
  SourceAttentionKernel(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    if (k.GetAttribute<int64_t>("num_heads") != 32 ||
        k.GetAttribute<int64_t>("kv_num_heads") != 2 ||
        k.GetAttribute<int64_t>("do_rotary") != 0)
      throw std::invalid_argument("SourceAttention requires heads32/KV2/no rotary");
    // Match the frozen source process, rather than inheriting ATen C++ defaults.
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowBF16ReductionCuBLAS(false);
    at::globalContext().setAllowFP16ReductionCuBLAS(false);
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext ctx(raw);
      const auto q = ctx.GetInput(0), k = ctx.GetInput(1), v = ctx.GetInput(2);
      const auto pk = ctx.GetInput(3), pv = ctx.GetInput(4);
      auto shape = [](const auto& value) { return value.GetTensorTypeAndShapeInfo().GetShape(); };
      const auto qs = shape(q);
      if (qs.size() != 3 || qs[0] != 1 || qs[1] < 1 || qs[1] > 8 || qs[2] != 4096 ||
          shape(k) != std::vector<int64_t>{1, qs[1], 256} ||
          shape(v) != std::vector<int64_t>{1, qs[1], 256} ||
          shape(pk) != cache_shape || shape(pv) != cache_shape)
        throw std::invalid_argument("SourceAttention unqualified Q/K/V/cache shape");
      for (size_t i = 5; i < 9; ++i) {
        const auto dims = shape(ctx.GetInput(i));
        if (dims != std::vector<int64_t>{1} && !(i == 6 && dims.empty()))
          throw std::invalid_argument("SourceAttention metadata must be INT32[1] (capacity also scalar)");
      }
      const int64_t rows = qs[1];
      const int64_t total = *ctx.GetInput(6).GetTensorData<int32_t>();
      const int64_t mode = *ctx.GetInput(7).GetTensorData<int32_t>();
      const int64_t past = *ctx.GetInput(8).GetTensorData<int32_t>();
      if (total != capacity || mode < 1 || mode > capacity || past < 0 ||
          past > capacity - rows || (mode > 1 && past >= mode))
        throw std::invalid_argument("SourceAttention invalid logical prefill M, past position or cache capacity");
      if (std::strcmp(source_attention_cublas_path(), SOURCE_CUBLAS_PATH))
        throw std::runtime_error("SourceAttention wrong loaded cuBLAS");
      if (at::globalContext().allowTF32CuBLAS() || at::globalContext().allowTF32CuDNN() ||
          at::globalContext().allowBF16ReductionCuBLAS() !=
              at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK ||
          at::globalContext().allowFP16ReductionCuBLAS() !=
              at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK)
        throw std::runtime_error("SourceAttention global source math flags changed");
      int device = -1;
      cu(cudaGetDevice(&device));
      if (device != 0) throw std::runtime_error("SourceAttention requires internal CUDA0");
      auto stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());
      const size_t length_bound = mode > 1 ? mode : past + rows;
      const size_t query_bound = mode > 1 ? mode : 1;
      const size_t scratch_bound = 256 * query_bound * length_bound +
          65536 * length_bound + (512ull << 20);
      size_t free = 0, device_total = 0;
      cu(cudaMemGetInfo(&free, &device_total));
      if (scratch_bound > free || device_total - free + scratch_bound > (72ull << 30))
        throw std::runtime_error("SourceAttention source eager matrices exceed the 72GiB residency budget");
      int32_t seqlen = -1;
      cu(cudaMemcpyAsync(&seqlen, ctx.GetInput(5).GetTensorData<int32_t>(), sizeof(seqlen),
                         cudaMemcpyDeviceToHost, stream));
      cu(cudaStreamSynchronize(stream));
      if (seqlen != past + rows - 1)
        throw std::invalid_argument("SourceAttention seqlens != explicit past_position+R-1");
      c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device));
      c10::InferenceMode inference;
      const auto options = at::TensorOptions().device(at::kCUDA, device).dtype(at::kBFloat16);
      auto view = [&](const auto& value, const std::vector<int64_t>& dims) {
        return at::from_blob(const_cast<Ort::BFloat16_t*>(value.template GetTensorData<Ort::BFloat16_t>()),
                             dims, options);
      };
      auto query = view(q, qs);
      auto key = view(k, {1, rows, 2, 128}).transpose(1, 2);
      auto val = view(v, {1, rows, 2, 128}).transpose(1, 2);
      auto old_k = view(pk, cache_shape), old_v = view(pv, cache_shape);
      auto out = ctx.GetOutput(0, qs);
      auto outk = ctx.GetOutput(1, cache_shape), outv = ctx.GetOutput(2, cache_shape);
      auto result = at::from_blob(out.GetTensorMutableData<Ort::BFloat16_t>(), qs, options);
      auto present_k = at::from_blob(outk.GetTensorMutableData<Ort::BFloat16_t>(), cache_shape, options);
      auto present_v = at::from_blob(outv.GetTensorMutableData<Ort::BFloat16_t>(), cache_shape, options);
      if (present_k.data_ptr() != old_k.data_ptr()) present_k.copy_(old_k);
      if (present_v.data_ptr() != old_v.data_ptr()) present_v.copy_(old_v);
      // Preserve all native writes, including dummy rows; attention reads causal prefixes only.
      present_k.narrow(2, past, rows).copy_(key);
      present_v.narrow(2, past, rows).copy_(val);
      const auto allocated_before = c10::cuda::CUDACachingAllocator::getDeviceStats(device)
                                        .allocated_bytes[0].current;
      const int64_t valid = mode > 1 ? std::min<int64_t>(rows, mode - past) : 0;
      const int iterations = mode > 1 ? 1 + rows - valid : rows;
      const bool hoist = mode == 1 && rows == 8;
      at::Tensor expanded_k, expanded_v;
      if (hoist) {
        // All eight cache writes precede attention. Materialize repeat_kv once;
        // each row still reads only its original causal prefix.
        const int64_t length = past + rows;
        auto logical_k = at::zeros({1, 2, length, 128}, options);
        auto logical_v = at::zeros_like(logical_k);
        logical_k.copy_(present_k.narrow(2, 0, length));
        logical_v.copy_(present_v.narrow(2, 0, length));
        expanded_k = logical_k.unsqueeze(2).expand({1, 2, 16, length, 128})
                         .reshape({1, 32, length, 128});
        expanded_v = logical_v.unsqueeze(2).expand({1, 2, 16, length, 128})
                         .reshape({1, 32, length, 128});
      }
      for (int iteration = 0; iteration < iterations; ++iteration) {
        const bool prefill = mode > 1 && iteration == 0;
        const int64_t row = prefill ? 0 : mode > 1 ? valid + iteration - 1 : iteration;
        const int64_t m = prefill ? mode : 1, length = prefill ? mode : past + row + 1;
        const int64_t start = prefill ? past : 0;
        const int64_t count = prefill ? valid : 1;
        auto logical_q = at::zeros({1, m, 4096}, options);
        logical_q.narrow(1, start, count).copy_(query.narrow(1, row, count));
        auto qh = logical_q.view({1, m, 32, 128}).transpose(1, 2);
        at::Tensor kh, vh;
        if (hoist) {
          // Matrix-local strides stay (128,1); only head-batch stride uses Lmax.
          // Do not make these prefix views contiguous.
          kh = expanded_k.narrow(2, 0, length);
          vh = expanded_v.narrow(2, 0, length);
        } else {
          auto logical_k = at::zeros({1, 2, length, 128}, options);
          auto logical_v = at::zeros_like(logical_k);
          const int64_t available = prefill ? std::min<int64_t>(past + rows, mode) : length;
          logical_k.narrow(2, 0, available).copy_(present_k.narrow(2, 0, available));
          logical_v.narrow(2, 0, available).copy_(present_v.narrow(2, 0, available));
          // Source repeat_kv: two KV heads become 32 query heads in groups of16.
          kh = logical_k.unsqueeze(2).expand({1, 2, 16, length, 128})
                   .reshape({1, 32, length, 128});
          vh = logical_v.unsqueeze(2).expand({1, 2, 16, length, 128})
                   .reshape({1, 32, length, 128});
        }
        auto positions = prefill ?
            at::arange(mode, options.dtype(at::kLong)) :
            at::full({1}, past + row, options.dtype(at::kLong));
        auto mask = at::full({m, length}, -3.3895313892515355e38, options);
        if (m != 1) mask = at::triu(mask, 1);
        mask = mask * (at::arange(length, options.dtype(at::kLong)).unsqueeze(0) >
                       positions.reshape({m, 1}));
        // Each eager boundary is BF16 except FP32 softmax; cached mask retains negative zero.
        auto scores = at::matmul(qh, kh.transpose(2, 3)) * std::pow(128.0, -0.5);
        scores = scores + mask.unsqueeze(0).unsqueeze(0);
        auto probabilities = at::softmax(scores, -1, at::kFloat).to(at::kBFloat16);
        auto attended = at::matmul(probabilities, vh).transpose(1, 2).contiguous()
                            .view({1, m, 4096});
        result.narrow(1, row, count).copy_(attended.narrow(1, start, count));
      }
      cu(cudaStreamSynchronize(stream));
      peak_allocated_bytes = static_cast<size_t>(std::max<int64_t>(0,
          c10::cuda::CUDACachingAllocator::getDeviceStats(device).allocated_bytes[0].peak -
          allocated_before));
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
    }
  }
};

struct SourceAttentionOp : Ort::CustomOpBase<SourceAttentionOp, SourceAttentionKernel, true> {
  const char* GetName() const { return "SourceAttention"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 9; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i < 5 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtMemType GetInputMemoryType(size_t i) const {
    return i >= 6 ? OrtMemTypeCPUInput : OrtMemTypeDefault;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi& a, const OrtKernelInfo* info, void** out) const noexcept {
    try { *out = new SourceAttentionKernel(a, info); return nullptr; }
    catch (const std::exception& e) { return a.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "SourceAttention requires CAPI24");
  Ort::InitApi(api);
  try {
    static SourceAttentionOp op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_full_attention");
    static const bool added = [&] { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  }
}
