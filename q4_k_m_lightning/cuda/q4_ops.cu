#define RegisterCustomOps LightningBF16RegisterCustomOps
#include "lightning_ops.cu"
#undef RegisterCustomOps

namespace {
#include "quant_blocks.cuh"

int RowBytes(int format, int k) {
  const int block = format == 12 ? 256 : 32;
  Require(format == 12 || format == 6 || format == 8, "Q4: unsupported block format");
  Require(k > 0 && k % block == 0, "Q4: incompatible row width");
  return PositiveInt(static_cast<int64_t>(k / block) * (format == 12 ? 144 : format == 6 ? 22 : 34),
                     "Q4: packed row is too large");
}

__global__ void DecodeMatrix(const uint8_t* packed, int n, int k, int row_bytes,
                             int format, bool transpose, bf16* output) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(n) * k) return;
  const int row = i / k, column = i % k;
  output[transpose ? static_cast<int64_t>(column) * n + row : i] =
      __float2bfloat16(DecodeWeight(packed + static_cast<int64_t>(row) * row_bytes, column, format));
}

void Decode(const uint8_t* packed, int n, int k, int format, bool transpose,
            bf16* output, cudaStream_t stream) {
  DecodeMatrix<<<static_cast<unsigned>((static_cast<int64_t>(n) * k + 255) / 256), 256, 0, stream>>>(
      packed, n, k, RowBytes(format, k), format, transpose, output);
  CheckLaunch();
}

struct QuantMatrix {
  const OrtApi& api;
  int n, k, format, row_bytes;
  QuantMatrix(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo attributes(info);
    n = PositiveInt(attributes.GetAttribute<int64_t>("n"), "Q4: invalid N");
    k = PositiveInt(attributes.GetAttribute<int64_t>("k"), "Q4: invalid K");
    format = PositiveInt(attributes.GetAttribute<int64_t>("format"), "Q4: invalid format");
    row_bytes = RowBytes(format, k);
  }
};

struct QuantDequantState : QuantMatrix {
  QuantDequantState(const OrtApi& a, const OrtKernelInfo* info) : QuantMatrix(a, info) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 1, 1);
      CheckTensor(c, 0, {n, row_bytes}, U8, "Q4: invalid packed matrix");
      auto output = c.GetOutput(0, Shape{n, k});
      Decode(In<uint8_t>(c, 0), n, k, format, false, Out<bf16>(output), Stream(api, raw));
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

__global__ void QuantEmbeddingKernel(const int64_t* ids, int count, const uint8_t* packed,
                                     int n, int k, int row_bytes, int format, bf16* output) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(count) * k) return;
  const int64_t token = ids[i / k];
  if (token >= 0 && token < n)
    output[i] = __float2bfloat16(DecodeWeight(packed + token * row_bytes, i % k, format));
}

struct QuantEmbeddingState : QuantMatrix {
  QuantEmbeddingState(const OrtApi& a, const OrtKernelInfo* info) : QuantMatrix(a, info) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 2, 1);
      CheckTensor(c, 1, {n, row_bytes}, U8, "Q4: invalid packed embedding");
      auto shape = Dimensions(c.GetInput(0));
      Require(shape.size() == 2 && shape[0] == 1, "Q4: embedding requires batch one IDs");
      const int count = PositiveInt(shape[1], "Q4: empty token input");
      Require(count <= 4096, "Q4: maximum embedding chunk is 4096");
      CheckTensor(c, 0, shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "Q4: invalid token IDs");
      const auto stream = Stream(api, raw);
      std::vector<int64_t> ids(count);
      auto error = cudaMemcpyAsync(ids.data(), In<int64_t>(c, 0), count * sizeof(int64_t),
                                   cudaMemcpyDeviceToHost, stream);
      if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
      for (auto id : ids) Require(id >= 0 && id < n, "Q4: token ID out of range");
      shape.push_back(k);
      auto output = c.GetOutput(0, shape);
      QuantEmbeddingKernel<<<static_cast<unsigned>((static_cast<int64_t>(count) * k + 255) / 256),
                             256, 0, stream>>>(
          In<int64_t>(c, 0), count, In<uint8_t>(c, 1), n, k, row_bytes, format, Out<bf16>(output));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

struct QuantLinearState : QuantMatrix {
  cublasHandle_t blas = nullptr;
  std::mutex mutex;
  QuantLinearState(const OrtApi& a, const OrtKernelInfo* info) : QuantMatrix(a, info) {
    CheckBlas(cublasCreate(&blas));
  }
  ~QuantLinearState() { if (blas) cublasDestroy(blas); }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 2, 1);
      auto shape = Dimensions(c.GetInput(0));
      Require((shape.size() == 2 || (shape.size() == 3 && shape[0] == 1)) && shape.back() == k,
              "Q4: expected [T,K] or [1,T,K] activation");
      const int rows = PositiveInt(shape[shape.size() - 2], "Q4: invalid rows");
      Require(rows <= 4096, "Q4: maximum projection chunk is 4096");
      CheckTensor(c, 0, shape, BF, "Q4: invalid activation");
      CheckTensor(c, 1, {n, row_bytes}, U8, "Q4: invalid packed projection");
      shape.back() = n;
      auto output = c.GetOutput(0, shape);
      const auto stream = Stream(api, raw);
      std::lock_guard<std::mutex> lock(mutex);
      AsyncBuffer weights(static_cast<size_t>(n) * k * sizeof(bf16), stream);
      Decode(In<uint8_t>(c, 1), n, k, format, true, static_cast<bf16*>(weights.data), stream);
      CheckBlas(cublasSetStream(blas, stream));
      const float alpha = 1.f, beta = 0.f;
      for (int row = 0; row < rows; ++row)
        CheckBlas(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, k, &alpha,
            weights.data, CUDA_R_16BF, n, In<bf16>(c, 0) + static_cast<int64_t>(row) * k,
            CUDA_R_16BF, k, &beta, Out<bf16>(output) + static_cast<int64_t>(row) * n,
            CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

struct QuantMoeState {
  const OrtApi& api;
  int experts, hidden, intermediate, up_format, down_format;
  cublasHandle_t blas = nullptr;
  std::unique_ptr<CohortGemm> up_plan, down_plan;
  std::mutex mutex;
  QuantMoeState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo attrs(info);
    experts = PositiveInt(attrs.GetAttribute<int64_t>("experts"), "Q4: invalid experts");
    hidden = PositiveInt(attrs.GetAttribute<int64_t>("hidden"), "Q4: invalid hidden");
    intermediate = PositiveInt(attrs.GetAttribute<int64_t>("intermediate"), "Q4: invalid intermediate");
    up_format = PositiveInt(attrs.GetAttribute<int64_t>("up_format"), "Q4: invalid up format");
    down_format = PositiveInt(attrs.GetAttribute<int64_t>("down_format"), "Q4: invalid down format");
    Require(experts >= 6 && hidden == 2688 && intermediate == 1856,
            "Q4: MoE requires Lightning dimensions and six source-ordered routes");
    RowBytes(up_format, hidden); RowBytes(down_format, intermediate);
    CheckBlas(cublasCreate(&blas));
  }
  ~QuantMoeState() { if (blas) cublasDestroy(blas); }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 5, 3);
      const auto shape = Dimensions(c.GetInput(0));
      Require(shape.size() == 2 && shape[1] == hidden, "Q4: expected [T,H]");
      const int rows = PositiveInt(shape[0], "Q4: invalid rows");
      Require(rows <= 4096, "Q4: maximum chunk is 4096");
      constexpr int topk = 6;
      CheckTensor(c, 0, shape, BF, "Q4: invalid activation");
      CheckTensor(c, 1, {rows, topk}, I32, "Q4: invalid expert IDs");
      CheckTensor(c, 2, {rows, topk}, F32, "Q4: invalid route probabilities");
      const int up_bytes = RowBytes(up_format, hidden), down_bytes = RowBytes(down_format, intermediate);
      CheckTensor(c, 3, {experts, intermediate, up_bytes}, U8, "Q4: invalid packed up experts");
      CheckTensor(c, 4, {experts, hidden, down_bytes}, U8, "Q4: invalid packed down experts");
      auto y = c.GetOutput(0, Shape{rows, hidden});
      auto h = c.GetOutput(1, Shape{rows * topk, intermediate});
      auto part = c.GetOutput(2, Shape{rows, topk, hidden});
      const auto stream = Stream(api, raw);
      std::lock_guard<std::mutex> lock(mutex);
      AsyncBuffer metadata(static_cast<size_t>(rows * topk + 2 * experts) * sizeof(int), stream);
      auto* ranks = static_cast<int*>(metadata.data);
      auto* counts = ranks + rows * topk;
      auto* offsets = counts + experts;
      ExpertCohortKernel<<<experts, 1, 0, stream>>>(In<int32_t>(c, 1), rows, topk, ranks, counts);
      std::vector<int> host_counts(experts), host_offsets(experts);
      auto error = cudaMemcpyAsync(host_counts.data(), counts, experts * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream);
      if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
      int total = 0;
      for (int e = 0; e < experts; ++e) {
        host_offsets[e] = total;
        Require(host_counts[e] >= 0 && host_counts[e] <= rows, "Q4: duplicate expert IDs");
        total += host_counts[e];
      }
      Require(total == rows * topk, "Q4: expert ID out of range");
      ExpertOffsetsKernel<<<CeilDiv(experts, 128), 128, 0, stream>>>(counts, experts, offsets);
      AsyncBuffer packed(static_cast<size_t>(total) * hidden * 2 * sizeof(bf16), stream);
      auto* input = static_cast<bf16*>(packed.data);
      auto* down = input + static_cast<int64_t>(total) * hidden;
      AsyncBuffer activation(static_cast<size_t>(total) * intermediate * sizeof(bf16), stream);
      auto* activated = static_cast<bf16*>(activation.data);
      ExpertGatherChunkKernel<<<CeilDiv(total * hidden, 256), 256, 0, stream>>>(
          In<bf16>(c, 0), In<int32_t>(c, 1), ranks, offsets, rows, topk, hidden, input);
      if (!up_plan) {
        auto up = std::make_unique<CohortGemm>(), down_gemm = std::make_unique<CohortGemm>();
        up->Initialize(hidden, intermediate); down_gemm->Initialize(intermediate, hidden);
        up_plan = std::move(up); down_plan = std::move(down_gemm);
      }
      size_t workspace_bytes = 0;
      for (int count : host_counts) if (count > 1) {
        workspace_bytes = std::max(workspace_bytes, up_plan->Workspace(count));
        workspace_bytes = std::max(workspace_bytes, down_plan->Workspace(count));
      }
      AsyncBuffer workspace(std::max<size_t>(1, workspace_bytes), stream);
      AsyncBuffer weights(static_cast<size_t>(hidden) * intermediate * sizeof(bf16), stream);
      CheckBlas(cublasSetStream(blas, stream));
      const float alpha = 1.f, beta = 0.f;
      for (int e = 0; e < experts; ++e) {
        const int count = host_counts[e], offset = host_offsets[e];
        if (!count) continue;
        Decode(In<uint8_t>(c, 3) + static_cast<int64_t>(e) * intermediate * up_bytes,
               intermediate, hidden, up_format, false, static_cast<bf16*>(weights.data), stream);
        auto* src = input + static_cast<int64_t>(offset) * hidden;
        auto* mid = activated + static_cast<int64_t>(offset) * intermediate;
        auto* dst = down + static_cast<int64_t>(offset) * hidden;
        if (count > 1)
          up_plan->Run(static_cast<bf16*>(weights.data), src, mid, count,
                       workspace.data, workspace_bytes, stream);
        else
          CheckBlas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, intermediate, 1, hidden,
              &alpha, weights.data, CUDA_R_16BF, hidden, src, CUDA_R_16BF, hidden,
              &beta, mid, CUDA_R_16BF, intermediate, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        ReluSquaredKernel<<<CeilDiv(count * intermediate, 256), 256, 0, stream>>>(
            mid, static_cast<int64_t>(count) * intermediate);
        Decode(In<uint8_t>(c, 4) + static_cast<int64_t>(e) * hidden * down_bytes,
               hidden, intermediate, down_format, false, static_cast<bf16*>(weights.data), stream);
        if (count > 1)
          down_plan->Run(static_cast<bf16*>(weights.data), mid, dst, count,
                         workspace.data, workspace_bytes, stream);
        else
          CheckBlas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, intermediate,
              &alpha, weights.data, CUDA_R_16BF, intermediate, mid, CUDA_R_16BF, intermediate,
              &beta, dst, CUDA_R_16BF, hidden, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      ExpertScatterActivationKernel<<<CeilDiv(total * intermediate, 256), 256, 0, stream>>>(
          activated, In<int32_t>(c, 1), ranks, offsets, total, intermediate, Out<bf16>(h));
      ExpertFinalizeChunkKernel<<<CeilDiv(rows * hidden, 256), 256, 0, stream>>>(
          down, In<int32_t>(c, 1), In<float>(c, 2), ranks, offsets, rows, topk, hidden,
          Out<bf16>(y), Out<float>(part));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

template <typename Derived, typename State>
struct QuantOp : LightningOp<Derived, State> {
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return BF; }
};
struct QuantDequantOp : QuantOp<QuantDequantOp, QuantDequantState> {
  const char* GetName() const { return "Dequantize"; }
  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return U8; }
};
struct QuantEmbeddingOp : QuantOp<QuantEmbeddingOp, QuantEmbeddingState> {
  const char* GetName() const { return "Embedding"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i ? U8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
};
struct QuantLinearOp : QuantOp<QuantLinearOp, QuantLinearState> {
  const char* GetName() const { return "Linear"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t i) const { return i ? U8 : BF; }
};
struct QuantMoeOp : LightningOp<QuantMoeOp, QuantMoeState> {
  const char* GetName() const { return "MoE"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i == 0 ? BF : i == 1 ? I32 : i == 2 ? F32 : U8;
  }
  size_t GetOutputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetOutputType(size_t i) const { return i == 2 ? F32 : BF; }
};
}

extern "C" LIGHTNING_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  if (auto* status = LightningBF16RegisterCustomOps(options, api_base)) return status;
  const auto* api = api_base->GetApi(ORT_API_VERSION);
  try {
    static QuantDequantOp decode;
    static QuantEmbeddingOp embedding;
    static QuantLinearOp linear;
    static QuantMoeOp moe;
    static Ort::CustomOpDomain domain("lightning_q4");
    static std::once_flag registered;
    std::call_once(registered, [&] {
      domain.Add(&decode); domain.Add(&embedding); domain.Add(&linear); domain.Add(&moe);
    });
    Ort::UnownedSessionOptions(options).Add(domain);
    return nullptr;
  } catch (const std::exception& e) { return api->CreateStatus(ORT_FAIL, e.what()); }
}
