#define RegisterCustomOps RegisterUnusedStandaloneBoltCustomOps
#include "bolt_ops.cu"
#undef RegisterCustomOps

namespace {

struct Q8RankState {
  const OrtApi& api;
  Q8RankState(const OrtApi& a, const OrtKernelInfo*) : api(a) {}

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      auto x = c.GetInput(0);
      const auto w = c.GetInput(1), sc = c.GetInput(2);
      auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const auto ws = w.GetTensorTypeAndShapeInfo().GetShape();
      const auto ss = sc.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.empty() || ws.size() != 3 || ss.size() != 2)
        throw std::invalid_argument("Q8 FP16 scales: invalid tensor ranks");
      const int64_t k = shape.back();
      int64_t rows = 1;
      for (size_t i = 0; i + 1 < shape.size(); ++i) {
        if (shape[i] < 1 || shape[i] > 8 || rows > 8 / shape[i])
          throw std::invalid_argument("Q8 FP16 scales: flattened rows must be in [1,8]");
        rows *= shape[i];
      }
      if (k <= 0 || k % 64 || k > std::numeric_limits<int>::max() ||
          ws[0] <= 0 || ws[0] > std::numeric_limits<int>::max() ||
          ws[1] != k / 32 || ws[2] != 32 || ss[0] != ws[0] || ss[1] != ws[1])
        throw std::invalid_argument("Q8 FP16 scales: inconsistent packed weight geometry");
      // Match upstream QGemvState: preserve all leading dimensions.
      shape.back() = ws[0];
      auto y = c.GetOutput(0, shape);
      LaunchMmaGemv<kFmtQ8F16>(
          In<bf16>(c, 0), In<uint8_t>(c, 1), In<__half>(c, 2), nullptr,
          static_cast<int>(rows), static_cast<int>(ws[0]), static_cast<int>(k),
          Out<bf16>(y), Stream(api, raw));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct Q8RankOp : BoltOp<Q8RankOp, Q8RankState> {
  const char* GetName() const { return "Q8GemvFp16Scales"; }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    static const ONNXTensorElementDataType types[] = {
        BF, U8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16};
    return types[i];
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto* api = api_base->GetApi(ORT_API_VERSION);
  if (!api)
    return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static Q8RankOp op;
    static Ort::CustomOpDomain domain("bolt_fast_q8");
    static std::once_flag registered;
    std::call_once(registered, [&]() { domain.Add(&op); });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
