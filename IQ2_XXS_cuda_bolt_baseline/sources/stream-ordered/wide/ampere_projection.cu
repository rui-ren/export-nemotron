// The included coworker implementation retains its native register-dequantized
// tensor-core path; the two scale corrections live in the isolated source copy.
#include "bolt_ops.cu"

extern "C" int launch_native_source_projection(
    const void* x, const uint8_t* weight, const void* scales, void* output,
    int rows, int n, int k, int format, int k_warps, void* opaque_stream) {
  if (!x || !weight || !scales || !output || rows < 1 || rows > 8 ||
      n < 1 || k < 256 || k % 256 || (format != 0 && format != 1)) return -1;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  const auto* input = static_cast<const bf16*>(x);
  auto* result = static_cast<bf16*>(output);
  auto launch = [&](auto kind) {
    constexpr int fmt = decltype(kind)::value;
    switch (k_warps) {
      case 1: LaunchMmaGemvImpl<fmt, false, 1, 1>(input, weight, scales, nullptr, rows, n, k, result, stream); break;
      case 2: LaunchMmaGemvImpl<fmt, false, 1, 2>(input, weight, scales, nullptr, rows, n, k, result, stream); break;
      case 4: LaunchMmaGemvImpl<fmt, false, 1, 4>(input, weight, scales, nullptr, rows, n, k, result, stream); break;
      case 8: LaunchMmaGemvImpl<fmt, false, 1, 8>(input, weight, scales, nullptr, rows, n, k, result, stream); break;
      case 16: LaunchMmaGemvImpl<fmt, false, 1, 16>(input, weight, scales, nullptr, rows, n, k, result, stream); break;
      default: return -2;
    }
    return 0;
  };
  const int status = format == 0 ? launch(std::integral_constant<int, kFmtQ8>{})
                                : launch(std::integral_constant<int, kFmtIq2N>{});
  if (status) return status;
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

extern "C" int native_source_decode_weights(
    const uint8_t* packed, const void* scales, void* output,
    int n, int k, int format, void* opaque_stream);

__global__ void decode_source_values(const uint8_t* packed, const __half* scales,
                                      bf16* output, int n, int k, int format) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(n) * k) return;
  if (format == 0) {
    output[i] = __float2bfloat16_rn((float(packed[i]) - 128.f) * __half2float(scales[i / 32]));
    return;
  }
  const uint8_t* record = packed + (i / 32) * 8;
  const uint32_t control = uint32_t(record[4]) | (uint32_t(record[5]) << 8) |
      (uint32_t(record[6]) << 16) | (uint32_t(record[7]) << 24);
  const int group = (i % 32) / 8, lane = i % 8;
  const uint32_t sign7 = (control >> (7 * group)) & 127u;
  const uint32_t signs = sign7 | ((__popc(sign7) & 1u) << 7);
  const float magnitude = float((kIq2xxsGrid[record[group]] >> (8 * lane)) & 255u);
  const float scale = (__half2float(scales[(i / k) * (k / 256) + (i % k) / 256]) *
                        (float(control >> 28) + 0.5f)) * 0.25f;
  output[i] = __float2bfloat16_rn(magnitude * scale * (((signs >> lane) & 1) ? -1.f : 1.f));
}

extern "C" int native_source_decode_weights(
    const uint8_t* packed, const void* scales, void* output,
    int n, int k, int format, void* opaque_stream) {
  if (!packed || !scales || !output || n < 1 || k < 256 || k % 256 ||
      (format != 0 && format != 1)) return -1;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  decode_source_values<<<(size_t(n) * k + 255) / 256, 256, 0, stream>>>(
      packed, static_cast<const __half*>(scales), static_cast<bf16*>(output), n, k, format);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
