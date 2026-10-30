#define RegisterCustomOps RegisterCoworkerCustomOps
#include "native_projection.cu"
#undef RegisterCustomOps

extern "C" int routed_v3_supported(int experts, int width) {
  return (experts == 240 && (width == 1536 || width == 2048 || width == 2304)) ||
         (experts == 304 && (width == 2048 || width == 2304 || width == 2560)) ||
         (experts == 368 && (width == 2048 || width == 2304 || width == 2560)) ||
         (experts == 432 && (width == 2304 || width == 2560)) ||
         (experts == 496 && width == 2560);
}

extern "C" size_t routed_v3_last_device_extra_bytes() { return 0; }
extern "C" size_t routed_v3_last_explicit_bytes() { return 0; }
extern "C" const char* routed_v3_cublas_path() {
  return "NOT_USED_NATIVE_REGISTER_DEQUANTIZATION";
}

extern "C" int routed_v3_launch(
    const void* x, const int32_t* ids, const float* weights,
    const uint8_t* up, const void* up_scale, const uint8_t* down, const void* down_scale,
    void* output, void* h, float* part, int32_t* meta,
    int rows, int experts, int width, int topk, void* opaque_stream) {
  if (rows < 1 || rows > 8 || topk != 9 || !routed_v3_supported(experts, width)) return -1;
  if (!x || !ids || !weights || !up || !up_scale || !down || !down_scale ||
      !output || !h || !part || !meta) return -6;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
#define NATIVE_CUDA_CHECK(call) do { auto e = (call); if (e != cudaSuccess) return 1000 + e; } while (0)
  const int pairs = rows * topk;
  std::vector<int32_t> host_ids(pairs), host_meta(1 + pairs + pairs * 8, -1), unique;
  std::vector<float> host_weights(pairs);
  NATIVE_CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), ids, pairs * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream));
  NATIVE_CUDA_CHECK(cudaMemcpyAsync(host_weights.data(), weights, pairs * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
  NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int p = 0; p < pairs; ++p) {
    if (host_ids[p] < 0 || host_ids[p] >= experts) return -2;
    if (!std::isfinite(host_weights[p])) return -5;
    for (int q = p / topk * topk; q < p; ++q)
      if (host_ids[q] == host_ids[p]) return -4;
    int group = 0;
    while (group < int(unique.size()) && unique[group] != host_ids[p]) ++group;
    if (group == int(unique.size())) unique.push_back(host_ids[p]);
    host_meta[1 + pairs + group * 8 + p / topk] = p;
  }
  host_meta[0] = unique.size();
  for (size_t u = 0; u < unique.size(); ++u) host_meta[1 + u] = unique[u];
  NATIVE_CUDA_CHECK(cudaMemcpyAsync(meta, host_meta.data(), host_meta.size() * sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream));
  projection_wgmma<true, 1><<<dim3(width / 64, pairs), 128, 0, stream>>>(
      static_cast<const bf16*>(x), up, static_cast<const __half*>(up_scale),
      h, rows, width, 1024, meta, pairs);
  projection_wgmma<true, 2><<<dim3(16, pairs), 128, 0, stream>>>(
      static_cast<const bf16*>(h), down, static_cast<const __half*>(down_scale),
      part, rows, 1024, width, meta, pairs);
  MoeFinalizeKernel<<<dim3(rows, 4), 64, 0, stream>>>(
      part, weights, topk, static_cast<bf16*>(output));
  NATIVE_CUDA_CHECK(cudaGetLastError());
  NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
#undef NATIVE_CUDA_CHECK
}

#include "routed_v3_ort.cc"
