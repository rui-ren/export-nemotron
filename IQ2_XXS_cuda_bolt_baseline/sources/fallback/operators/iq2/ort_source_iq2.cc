#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <vector>

extern "C" int launch_full_iq2(const void*, const uint8_t*, const void*, void*,
                             int, int, int, int, int, void*);
extern "C" int launch_full_iq2_relu2(void*, int, int, void*);
extern "C" int full_iq2_supported_dimensions(int, int);
struct SourceIQ2Kernel {
  const OrtApi& api;
  int64_t activation = 0;
  SourceIQ2Kernel(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    auto status = api.KernelInfoGetAttribute_int64(info, "relu2", &activation);
    if (status) { api.ReleaseStatus(status); activation = 0; }
    if (activation != 0 && activation != 1)
      throw std::runtime_error("SourceIQ2 relu2 must be 0 or 1");
  }
  OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
    try {
      Ort::KernelContext ctx(context);
      const auto x = ctx.GetInput(0), w = ctx.GetInput(1);
      const auto s = ctx.GetInput(2), m = ctx.GetInput(3);
      const auto p = ctx.GetInput(4);
      const auto xs = x.GetTensorTypeAndShapeInfo().GetShape();
      const auto ws = w.GetTensorTypeAndShapeInfo().GetShape();
      const auto ss = s.GetTensorTypeAndShapeInfo().GetShape();
      if (xs.size() != 2 || ws.size() != 3 || xs[0] < 1 || xs[0] > 8 ||
          !full_iq2_supported_dimensions(ws[0], xs[1]) ||
          ws[1] != xs[1] / 32 || ws[2] != 8 ||
          ss != std::vector<int64_t>{ws[0], xs[1] / 256} ||
          m.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1} ||
          p.GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1})
        throw std::runtime_error("SourceIQ2 unsupported shapes or native packed layout");
      const int source_rows = *m.GetTensorData<int32_t>();
      if (activation && ws[0] != 2048 && ws[0] != 5376)
        throw std::runtime_error("SourceIQ2 relu2 is qualified only for up projections");
      auto y = ctx.GetOutput(0, std::vector<int64_t>{xs[0], ws[0]});
      auto* data = y.GetTensorMutableData<Ort::BFloat16_t>();
      int status = launch_full_iq2(
          x.GetTensorData<Ort::BFloat16_t>(), w.GetTensorData<uint8_t>(),
          s.GetTensorData<Ort::Float16_t>(), data,
          xs[0], ws[0], xs[1], source_rows, *p.GetTensorData<int32_t>(),
          ctx.GetGPUComputeStream());
      if (!status && activation)
        status = launch_full_iq2_relu2(data, xs[0], ws[0], ctx.GetGPUComputeStream());
      if (status) throw std::runtime_error("SourceIQ2 launch error " + std::to_string(status));
      return nullptr;
    } catch (const std::exception& e) { return api.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};
struct SourceIQ2Op : Ort::CustomOpBase<SourceIQ2Op, SourceIQ2Kernel, true> {
  const char* GetName() const { return "SourceIQ2"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 5; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    const ONNXTensorElementDataType types[] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32};
    return types[i];
  }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  OrtMemType GetInputMemoryType(size_t i) const {
    return i >= 3 ? OrtMemTypeCPUInput : OrtMemTypeDefault;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi& a, const OrtKernelInfo* info, void** out) const noexcept {
    try { *out = new SourceIQ2Kernel(a, info); return nullptr; }
    catch (const std::exception& e) { return a.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "SourceIQ2 requires C API24");
  Ort::InitApi(api);
  try {
    static SourceIQ2Op op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_full_iq2");
    static const bool added = [&]() { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) { return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
}
