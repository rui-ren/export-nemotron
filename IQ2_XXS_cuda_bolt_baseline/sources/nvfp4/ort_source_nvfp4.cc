#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <climits>
#include <stdexcept>
#include <vector>

extern "C" int launch_full_nvfp4(const void*, const uint8_t*, const uint8_t*,
                               const float*, void*, int, int, int, int, int, void*);
struct SourceNvfp4Kernel {
  const OrtApi& api;
  SourceNvfp4Kernel(const OrtApi& a, const OrtKernelInfo*) : api(a) {}
  OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
    try {
      Ort::KernelContext ctx(context);
      const auto x = ctx.GetInput(0), w = ctx.GetInput(1), s = ctx.GetInput(2);
      const auto g = ctx.GetInput(3), m = ctx.GetInput(4);
      const auto p = ctx.GetInput(5);
      const auto xs = x.GetTensorTypeAndShapeInfo().GetShape();
      const auto ws = w.GetTensorTypeAndShapeInfo().GetShape();
      const auto ss = s.GetTensorTypeAndShapeInfo().GetShape();
      if (xs.size() != 2 || ws.size() != 2 || xs[0] <= 0 || xs[0] > INT_MAX ||
          xs[1] <= 0 || xs[1] > INT_MAX || ws[0] <= 0 || ws[0] > INT_MAX ||
          xs[1] % 16 || ws[1] != xs[1] / 2 ||
          ss != std::vector<int64_t>{ws[0], xs[1] / 16} ||
          g.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1} ||
          m.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1} ||
          p.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1})
        throw std::runtime_error("SourceNvfp4 requires x[R,K], W[N,K/2], scales[N,K/16], global[1], source_rows[1]");
      auto y = ctx.GetOutput(0, std::vector<int64_t>{xs[0], ws[0]});
      const int status = launch_full_nvfp4(
          x.GetTensorData<Ort::BFloat16_t>(), w.GetTensorData<uint8_t>(),
          s.GetTensorData<uint8_t>(), g.GetTensorData<float>(),
          y.GetTensorMutableData<Ort::BFloat16_t>(), xs[0], ws[0], xs[1],
          *m.GetTensorData<int32_t>(), *p.GetTensorData<int32_t>(), ctx.GetGPUComputeStream());
      if (status) throw std::runtime_error("SourceNvfp4 launch error " + std::to_string(status));
      return nullptr;
    } catch (const std::exception& e) { return api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};
struct SourceNvfp4Op : Ort::CustomOpBase<SourceNvfp4Op, SourceNvfp4Kernel, true> {
  const char* GetName() const { return "SourceNvfp4"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 6; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    if (i == 0) return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    if (i <= 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    return i == 3 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; }
  OrtMemType GetInputMemoryType(size_t i) const { return i >= 4 ? OrtMemTypeCPUInput : OrtMemTypeDefault; }
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** out) const noexcept {
    try { *out = new SourceNvfp4Kernel(api, info); return nullptr; }
    catch (const std::exception& e) { return api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "SourceNvfp4 requires C API24");
  Ort::InitApi(api);
  try {
    static SourceNvfp4Op op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_full_nvfp4");
    static const bool added = [&]() { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) { return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
}
