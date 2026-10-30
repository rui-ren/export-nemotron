// Diagnostic source router: pinned SGEMM and unchanged ATen CUDA pointwise/topk/reduction.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifndef SOURCE_CUBLAS_PATH
#error "Pin the source cuBLAS path"
#endif

extern "C" int launch_native_router_logits(
    const void*, const void*, float*, int, int, void*);
extern "C" int launch_native_router_normalize(
    const float*, const int64_t*, int32_t*, float*, int, int, void*);

namespace {
std::atomic<uint64_t> native_m1_calls{0};
thread_local size_t peak_extra_bytes = 0;
thread_local size_t allocator_peak_bytes = 0;
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("SourceRouter cuBLAS status " + std::to_string(status));
}
struct Handle {
  cublasHandle_t value = nullptr;
  Handle() { blas_check(cublasCreate(&value)); }
  ~Handle() { if (value) cublasDestroy(value); }
};
bool supported_experts(int64_t e) {
  return e == 240 || e == 304 || e == 368 || e == 432 || e == 496;
}
}

extern "C" const char* full_router_cublas_path() {
  Dl_info info{};
  return dladdr(reinterpret_cast<void*>(&cublasSgemm_v2), &info) ? info.dli_fname : "";
}
extern "C" size_t full_router_peak_extra_bytes() { return peak_extra_bytes; }
extern "C" size_t full_router_allocator_peak_bytes() { return allocator_peak_bytes; }
extern "C" uint64_t full_router_native_m1_calls() {
  return native_m1_calls.load(std::memory_order_relaxed);
}

struct SourceRouterKernel {
  const OrtApi& api;
  SourceRouterKernel(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    if (k.GetAttribute<int64_t>("topk") != 9 || k.GetAttribute<float>("scale") != 5.f ||
        k.GetAttribute<int64_t>("n_group") != 1 ||
        k.GetAttribute<int64_t>("topk_group") != 1 ||
        k.GetAttribute<int64_t>("norm_topk_prob") != 1)
      throw std::invalid_argument("SourceRouter supports only top9, scale5, group1, normalized");
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext ctx(raw);
      const auto x = ctx.GetInput(0), w = ctx.GetInput(1);
      const auto b = ctx.GetInput(2), m = ctx.GetInput(3);
      const auto p = ctx.GetInput(4);
      const auto xs = x.GetTensorTypeAndShapeInfo().GetShape();
      const auto ws = w.GetTensorTypeAndShapeInfo().GetShape();
      const auto bs = b.GetTensorTypeAndShapeInfo().GetShape();
      if (xs.size() != 2 || xs[0] < 1 || xs[0] > 8 || xs[1] != 4096 ||
          ws.size() != 2 || !supported_experts(ws[0]) || ws[1] != 4096 ||
          bs != std::vector<int64_t>{ws[0]} ||
          m.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1} ||
          p.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1})
        throw std::invalid_argument("SourceRouter unqualified shape: R1..8,H4096,E240/304/368/432/496");
      const int rows = xs[0], experts = ws[0], source_rows = *m.GetTensorData<int32_t>();
      const int position = *p.GetTensorData<int32_t>();
      if (source_rows < 1 || source_rows > 135168 || position < 0 ||
          (source_rows > 1 && position >= source_rows))
        throw std::invalid_argument("SourceRouter invalid logical M or past position");
      if (std::strcmp(full_router_cublas_path(), SOURCE_CUBLAS_PATH))
        throw std::runtime_error("SourceRouter wrong preloaded cuBLAS");
      int device = -1;
      cuda_check(cudaGetDevice(&device));
      if (device != 0) throw std::runtime_error("SourceRouter requires visible internal CUDA device0");
      const auto stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());
      c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device));
      c10::InferenceMode inference;
      const auto options = at::TensorOptions().device(at::kCUDA, device);
      auto input = at::from_blob(const_cast<Ort::BFloat16_t*>(x.GetTensorData<Ort::BFloat16_t>()),
                                {rows, 4096}, options.dtype(at::kBFloat16));
      auto weight = at::from_blob(const_cast<Ort::BFloat16_t*>(w.GetTensorData<Ort::BFloat16_t>()),
                                 {experts, 4096}, options.dtype(at::kBFloat16));
      auto bias = at::from_blob(const_cast<float*>(b.GetTensorData<float>()),
                               {experts}, options.dtype(at::kFloat));
      auto ids_out = ctx.GetOutput(0, std::vector<int64_t>{rows, 9});
      auto weights_out = ctx.GetOutput(1, std::vector<int64_t>{rows, 9});
      auto scores_out = ctx.GetOutput(2, std::vector<int64_t>{rows, experts});
      auto output_ids = at::from_blob(ids_out.GetTensorMutableData<int32_t>(),
                                     {rows, 9}, options.dtype(at::kInt));
      auto output_weights = at::from_blob(weights_out.GetTensorMutableData<float>(),
                                         {rows, 9}, options.dtype(at::kFloat));
      auto output_scores = at::from_blob(scores_out.GetTensorMutableData<float>(),
                                        {rows, experts}, options.dtype(at::kFloat));
      size_t before = 0, total = 0;
      cuda_check(cudaMemGetInfo(&before, &total));
      peak_extra_bytes = 0;
      const auto allocated_before = c10::cuda::CUDACachingAllocator::getDeviceStats(device)
                                        .allocated_bytes[0].current;
      allocator_peak_bytes = 0;
      auto sample = [&] {
        size_t available = 0;
        cuda_check(cudaMemGetInfo(&available, &total));
        if (available < before) peak_extra_bytes = std::max(peak_extra_bytes, before - available);
        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        // Process-global historical peak is a conservative bound, not a reset of caller state.
        allocator_peak_bytes = std::max(allocator_peak_bytes, static_cast<size_t>(
            std::max<int64_t>(0, stats.allocated_bytes[0].peak - allocated_before)));
      };
      if (source_rows == 1) {
        auto logits = at::empty({rows, experts}, options.dtype(at::kFloat));
        if (launch_native_router_logits(
                x.GetTensorData<Ort::BFloat16_t>(), w.GetTensorData<Ort::BFloat16_t>(),
                logits.data_ptr<float>(), rows, experts, stream))
          throw std::runtime_error("Native source router logits failed");
        auto scores = at::sigmoid(logits);
        auto ids = std::get<1>(at::topk(scores + bias.unsqueeze(0), 9, -1, true, false));
        if (launch_native_router_normalize(
                scores.data_ptr<float>(), ids.data_ptr<int64_t>(),
                ids_out.GetTensorMutableData<int32_t>(),
                weights_out.GetTensorMutableData<float>(), rows, experts, stream))
          throw std::runtime_error("Native source router normalization failed");
        output_scores.copy_(scores);
        cuda_check(cudaStreamSynchronize(stream));
        sample();
        native_m1_calls.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }
      static thread_local Handle handle;
      blas_check(cublasSetStream(handle.value, stream));
      blas_check(cublasSetMathMode(handle.value, CUBLAS_DEFAULT_MATH));
      auto fpweight = weight.to(at::kFloat);
      const float one = 1.f, zero = 0.f;
      const int iterations = source_rows == 1 ? rows : 1;
      for (int row = 0; row < iterations; ++row) {
        // Preserve the original logical M for both SGEMM and the nine-value row reduction.
        auto fpinput = at::zeros({source_rows, 4096}, options.dtype(at::kFloat));
        const int count = source_rows == 1 ? 1 : std::min(rows, source_rows - position);
        const int start = source_rows == 1 ? 0 : position;
        fpinput.narrow(0, start, count).copy_(input.narrow(0, row, count));
        auto logits = at::empty({source_rows, experts}, options.dtype(at::kFloat));
        blas_check(cublasSgemm(handle.value, CUBLAS_OP_T, CUBLAS_OP_N,
            experts, source_rows, 4096, &one, fpweight.data_ptr<float>(), 4096,
            fpinput.data_ptr<float>(), 4096, &zero, logits.data_ptr<float>(), experts));
        auto scores = at::sigmoid(logits);
        // With n_group=topk_group=1 the original group mask is identically true.
        // ATen topk(sorted=false) retains source threshold/tie order, not score order.
        auto ids = std::get<1>(at::topk(scores + bias.unsqueeze(0), 9, -1, true, false));
        auto selected = at::gather(scores, 1, ids);
        auto denominator = selected.sum(-1, true) + 1e-20;
        selected.div_(denominator);
        auto normalized = selected * 5.0;
        output_ids.narrow(0, row, count).copy_(ids.narrow(0, start, count));
        output_weights.narrow(0, row, count).copy_(normalized.narrow(0, start, count));
        output_scores.narrow(0, row, count).copy_(scores.narrow(0, start, count));
        if (source_rows > 1 && count < rows) {
          // Even discarded rows must satisfy the routed kernel's unique-ID contract.
          output_ids.narrow(0, count, rows - count).copy_(
              at::arange(9, options.dtype(at::kInt)).expand({rows - count, 9}));
          output_weights.narrow(0, count, rows - count).zero_();
          output_scores.narrow(0, count, rows - count).zero_();
        }
        sample();
      }
      cuda_check(cudaStreamSynchronize(stream));
      sample();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
    }
  }
};

struct SourceRouterOp : Ort::CustomOpBase<SourceRouterOp, SourceRouterKernel, true> {
  const char* GetName() const { return "SourceRouter"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 5; }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    const ONNXTensorElementDataType types[] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32};
    return types[i];
  }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    return i == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtMemType GetInputMemoryType(size_t i) const {
    return i >= 3 ? OrtMemTypeCPUInput : OrtMemTypeDefault;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi& a, const OrtKernelInfo* info, void** out) const noexcept {
    try { *out = new SourceRouterKernel(a, info); return nullptr; }
    catch (const std::exception& e) { return a.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "SourceRouter requires C API24");
  Ort::InitApi(api);
  try {
    static SourceRouterOp op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_full_router");
    static const bool added = [&] { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  }
}
