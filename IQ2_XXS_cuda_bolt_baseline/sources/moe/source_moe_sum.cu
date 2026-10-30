#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

__global__ void ordered_expert_sum(const float* part, const float* weights, const int* ids,
                                   __nv_bfloat16* output, int topk) {
  const int row = blockIdx.x;
  const int column = blockIdx.y * blockDim.x + threadIdx.x;
  if (column >= 1024) return;
  int order[32];
  for (int j = 0; j < topk; ++j) {
    int slot = j;
    while (slot > 0 && ids[row * topk + order[slot - 1]] > ids[row * topk + j]) {
      order[slot] = order[slot - 1];
      --slot;
    }
    order[slot] = j;
  }
  float sum = 0.f;
  // The source visits experts by ascending ID and materializes each FP32 product.
  for (int j = 0; j < topk; ++j) {
    const int pair = row * topk + order[j];
    const float weighted = __fmul_rn(part[pair * 1024 + column], weights[pair]);
    sum = __fadd_rn(sum, weighted);
  }
  output[row * 1024 + column] = __float2bfloat16_rn(sum);
}

extern "C" int launch_source_moe_sum(const float* part, const float* weights, const int* ids,
                                    void* output, int rows, int topk, void* stream) {
  if (rows <= 0 || topk < 1 || topk > 32) return cudaErrorInvalidValue;
  ordered_expert_sum<<<dim3(rows, 4), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      part, weights, ids, static_cast<__nv_bfloat16*>(output), topk);
  return cudaGetLastError();
}

struct SourceMoeKernel {
  const OrtApi& api;
  SourceMoeKernel(const OrtApi& value, const OrtKernelInfo*) : api(value) {}
  OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
    try {
      Ort::KernelContext ctx(context);
      auto part = ctx.GetInput(0);
      auto weights = ctx.GetInput(1);
      auto ids = ctx.GetInput(2);
      const auto ws = weights.GetTensorTypeAndShapeInfo().GetShape();
      const auto ps = part.GetTensorTypeAndShapeInfo().GetShape();
      if (ws.size() != 2 || ws[0] < 1 || ws[0] > 8 || ws[1] < 1 || ws[1] > 32 ||
          ids.GetTensorTypeAndShapeInfo().GetShape() != ws ||
          ps != std::vector<int64_t>{ws[0] * ws[1], 1024}) {
        throw std::runtime_error("SourceMoeSum requires part[R*topk,1024], weights/ids[R,topk]");
      }
      auto output = ctx.GetOutput(0, std::vector<int64_t>{ws[0], 1024});
      const int error = launch_source_moe_sum(
          part.GetTensorData<float>(), weights.GetTensorData<float>(), ids.GetTensorData<int>(),
          output.GetTensorMutableData<Ort::BFloat16_t>(), ws[0], ws[1], ctx.GetGPUComputeStream());
      if (error) throw std::runtime_error("SourceMoeSum CUDA launch error " + std::to_string(error));
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
  }
};

struct SourceMoeOp : Ort::CustomOpBase<SourceMoeOp, SourceMoeKernel, true> {
  const char* GetName() const { return "SourceMoeSum"; }
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; }
  size_t GetInputTypeCount() const { return 3; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** out) const noexcept {
    try {
      *out = new SourceMoeKernel(api, info);
      return nullptr;
    } catch (const std::exception& error) {
      return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
    }
  }
};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(24);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "SourceMoeSum requires ORT C API24");
  Ort::InitApi(api);
  try {
    static SourceMoeOp op;
    op.version = 24;
    static Ort::CustomOpDomain domain("bolt_source_moe");
    static const bool added = [&]() { domain.Add(&op); return true; }();
    (void)added;
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& error) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
  }
}
