#define RegisterCustomOps RegisterUnusedStandaloneBoltCustomOps
#include "bolt_ops.cu"
#undef RegisterCustomOps

namespace {

struct ValidatedMoeVerifyState : MoeVerifyState {
  using MoeVerifyState::MoeVerifyState;

  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context(raw);
      if (iq2 == 2 && Dim(context.GetInput(3), 1) % 256 != 0)
        throw std::invalid_argument("MoeVerify iq2=2: inter size must be a multiple of 256");
      return MoeVerifyState::ComputeV2(raw);
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};

struct ValidatedMoeVerifyOp : BoltOp<ValidatedMoeVerifyOp, ValidatedMoeVerifyState> {
  const char* GetName() const { return "MoeVerify"; }
  size_t GetInputTypeCount() const { return 7; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return MoeVerifyOp{}.GetInputType(i);
  }
  size_t GetOutputTypeCount() const { return 4; }
  ONNXTensorElementDataType GetOutputType(size_t i) const {
    return MoeVerifyOp{}.GetOutputType(i);
  }
};

}  // namespace

extern "C" int BoltIQ2ProjectionProbe(
    cudaStream_t stream, const void* x, const void* packed, const void* scales,
    void* output, int rows, int n, int k) {
  try {
    LaunchMmaGemv<kFmtIq2N>(
        static_cast<const bf16*>(x), static_cast<const uint8_t*>(packed),
        static_cast<const bf16*>(scales), nullptr, rows, n, k,
        static_cast<bf16*>(output), stream);
    return static_cast<int>(cudaGetLastError());
  } catch (...) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
}

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const auto* api = api_base->GetApi(ORT_API_VERSION);
  if (!api)
    return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Unsupported ORT API");
  Ort::InitApi(api);
  try {
    static Q4GemvOp shared;
    static ValidatedMoeVerifyOp routed;
    static Ort::CustomOpDomain domain("bolt_iq2_source");
    static std::once_flag registered;
    std::call_once(registered, [&]() {
      domain.Add(&shared);
      domain.Add(&routed);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) {
    return api->CreateStatus(ORT_FAIL, e.what());
  }
}
