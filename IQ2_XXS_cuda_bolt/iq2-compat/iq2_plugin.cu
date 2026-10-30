#define RegisterCustomOps RegisterUnusedStandaloneBoltCustomOps
#include "bolt_ops.cu"
#undef RegisterCustomOps

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
    static MoeVerifyOp routed;
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
