#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <climits>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" int launch_source_rms(const void*, const void*, void*, float*,
                                int, int, int, float, void*);

struct SourceRmsKernel {
  const OrtApi& api;
  float epsilon;
  SourceRmsKernel(const OrtApi& ort_api, const OrtKernelInfo* info)
      : api(ort_api), epsilon(Ort::ConstKernelInfo(info).GetAttribute<float>("epsilon")) {}

  OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
    try {
      Ort::KernelContext ctx(context);
      const auto x = ctx.GetInput(0);
      const auto gamma = ctx.GetInput(1);
      const auto source_rows = ctx.GetInput(2);
      const auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const auto gamma_shape = gamma.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.empty() || shape.back() != 4096 ||
          gamma_shape != std::vector<int64_t>{4096} ||
          source_rows.GetTensorTypeAndShapeInfo().GetElementCount() != 1) {
        throw std::runtime_error("SourceRMS requires BF16[...,4096], gamma[4096], source_rows[1]");
      }
      const auto count = x.GetTensorTypeAndShapeInfo().GetElementCount();
      if (count == 0 || count / 4096 > INT_MAX) {
        throw std::runtime_error("SourceRMS invalid row count");
      }
      const int logical_rows = *source_rows.GetTensorData<int32_t>();
      auto y = ctx.GetOutput(0, shape);
      const int error = launch_source_rms(
          x.GetTensorData<Ort::BFloat16_t>(), gamma.GetTensorData<Ort::BFloat16_t>(),
          y.GetTensorMutableData<Ort::BFloat16_t>(), nullptr, count / 4096, 4096,
          logical_rows, epsilon, ctx.GetGPUComputeStream());
      if (error != 0) {
        throw std::runtime_error("SourceRMS CUDA launch error " + std::to_string(error));
      }
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
  }
};

struct SourceRmsOp : Ort::CustomOpBase<SourceRmsOp, SourceRmsKernel, true> {
  const char* GetName() const { return "SourceRMS"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 3; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 : ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  OrtMemType GetInputMemoryType(size_t index) const {
    return index == 2 ? OrtMemTypeCPUInput : OrtMemTypeDefault;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** out) const noexcept {
    try {
      *out = new SourceRmsKernel(api, info);
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
  }
};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "SourceRMS requires ORT C API24");
  Ort::InitApi(api);
  try {
    static SourceRmsOp op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_source");
    static const bool added = [&]() { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
  }
}
