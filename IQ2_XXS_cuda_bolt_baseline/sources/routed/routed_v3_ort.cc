#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <vector>

extern "C" int routed_v3_supported(int, int);
extern "C" int routed_v3_launch(const void*, const int32_t*, const float*,
    const uint8_t*, const void*, const uint8_t*, const void*,
    void*, void*, float*, int32_t*, int, int, int, int, void*);

struct RoutedV3Kernel {
  const OrtApi& api;
  RoutedV3Kernel(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    int64_t topk = 0, iq2 = 0;
    Ort::ThrowOnError(api.KernelInfoGetAttribute_int64(info, "topk", &topk));
    Ort::ThrowOnError(api.KernelInfoGetAttribute_int64(info, "iq2", &iq2));
    if (topk != 9 || iq2 != 2)
      throw std::runtime_error("RoutedV3 requires topk=9 and native iq2=2");
  }
  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext ctx(raw);
      auto x = ctx.GetInput(0), ids = ctx.GetInput(1), weights = ctx.GetInput(2);
      auto up = ctx.GetInput(3), us = ctx.GetInput(4);
      auto down = ctx.GetInput(5), ds = ctx.GetInput(6);
      auto shape = [](auto& value) { return value.GetTensorTypeAndShapeInfo().GetShape(); };
      const auto xs = shape(x), is = shape(ids), ws = shape(weights);
      const auto u = shape(up), d = shape(down);
      if (xs.size() != 2 || xs[0] < 1 || xs[0] > 8 || xs[1] != 1024 ||
          is != std::vector<int64_t>{xs[0], 9} || ws != is ||
          u.size() != 4 || d.size() != 4 ||
          u[0] > 496 || u[1] > 2560 || !routed_v3_supported(u[0], u[1]) ||
          u[2] != 32 || u[3] != 8 ||
          shape(us) != std::vector<int64_t>{u[0], u[1], 4} ||
          d != std::vector<int64_t>{u[0], 1024, u[1] / 32, 8} ||
          shape(ds) != std::vector<int64_t>{u[0], 1024, u[1] / 256})
        throw std::runtime_error("RoutedV3 invalid seven-input native IQ2 shapes/E/I/R/top9");
      const int r = xs[0], e = u[0], i = u[1], p = r * 9;
      auto y = ctx.GetOutput(0, std::vector<int64_t>{r, 1024});
      auto h = ctx.GetOutput(1, std::vector<int64_t>{p, i});
      auto part = ctx.GetOutput(2, std::vector<int64_t>{p, 1024});
      auto meta = ctx.GetOutput(3, std::vector<int64_t>{1 + p + p * 8});
      const int error = routed_v3_launch(
          x.GetTensorData<Ort::BFloat16_t>(), ids.GetTensorData<int32_t>(),
          weights.GetTensorData<float>(), up.GetTensorData<uint8_t>(),
          us.GetTensorData<Ort::Float16_t>(), down.GetTensorData<uint8_t>(),
          ds.GetTensorData<Ort::Float16_t>(), y.GetTensorMutableData<Ort::BFloat16_t>(),
          h.GetTensorMutableData<Ort::BFloat16_t>(), part.GetTensorMutableData<float>(),
          meta.GetTensorMutableData<int32_t>(), r, e, i, 9, ctx.GetGPUComputeStream());
      if (error) {
        const char* reason = error == -2 ? "ID outside [0,E)" :
            error == -4 ? "duplicate expert ID within a row" :
            error == -5 ? "nonfinite route weight" :
            error == -3 ? "unpinned cuBLAS binding" : "CUDA/cuBLAS/ABI failure";
        throw std::runtime_error(std::string("RoutedV3 ") + reason + " status=" + std::to_string(error));
      }
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
  }
};
struct RoutedV3Op : Ort::CustomOpBase<RoutedV3Op, RoutedV3Kernel, true> {
  const char* GetName() const { return "SourceRoutedIQ2M1"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 7; }
  size_t GetOutputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    const ONNXTensorElementDataType types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16};
    return types[i];
  }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    const ONNXTensorElementDataType types[] = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32};
    return types[i];
  }
  OrtStatusPtr CreateKernelV2(const OrtApi& a, const OrtKernelInfo* info, void** out) const noexcept {
    try { *out = new RoutedV3Kernel(a, info); return nullptr; }
    catch (const std::exception& e) { return a.CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
  }
};
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "RoutedV3 requires C API24");
  Ort::InitApi(api);
  try {
    static RoutedV3Op op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_routed_v3");
    static const bool added = [&]() { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) { return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what()); }
}
