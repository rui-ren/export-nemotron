#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace {
__device__ __forceinline__ float original_complete_weight(uint8_t packed, uint8_t scale, int parity) {
  const uint32_t code = (packed >> (parity * 4)) & 15u;
  const uint32_t magnitude = code & 7u;
  const float value = magnitude < 2 ? float(magnitude) * 0.5f
      : __uint_as_float((((magnitude >> 1) + 126u) << 23) | ((magnitude & 1u) << 22));
  const float signed_value = __uint_as_float(__float_as_uint(value) ^ ((code & 8u) << 28));
  const float local_scale = __half2float(
      __half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(scale), __NV_E4M3)));
  return __bfloat162float(__float2bfloat16_rn(signed_value * local_scale));
}

__device__ __forceinline__ float complete_weight(uint8_t packed, uint8_t scale, int parity) {
  const uint32_t code = (packed >> (parity * 4)) & 15u;
  const uint32_t magnitude = code & 7u;
  const float value = magnitude < 2 ? float(magnitude) * 0.5f
      : __uint_as_float((((magnitude >> 1) + 126u) << 23) | ((magnitude & 1u) << 22));
  const float signed_value = __uint_as_float(__float_as_uint(value) ^ ((code & 8u) << 28));
  const float local_scale = __half2float(
      __half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(scale), __NV_E4M3)));
  // Finite E2M1*E4M3 products fit BF16 exactly. The exhaustive codec probe
  // also checks the mask against the original promoted-BF16 NaN/zero bits.
  return __uint_as_float(__float_as_uint(__fmul_rn(signed_value, local_scale)) & 0xffff0000u);
}

__device__ __forceinline__ void record_weights(
    const uint8_t* packed, const uint8_t* scales, int row, int k, int column,
    int lane, float (&weights)[4]) {
  const size_t index = size_t(row) * (k / 2) + column / 2;
  const uint4 lo = __ldg(reinterpret_cast<const uint4*>(packed + index));
  const uint4 hi = __ldg(reinterpret_cast<const uint4*>(packed + index + 16));
  const uint32_t codes[4] = {
      lane < 8 ? lo.x : lo.y, lane < 8 ? lo.z : lo.w,
      lane < 8 ? hi.x : hi.y, lane < 8 ? hi.z : hi.w};
  const uint32_t local = __ldg(reinterpret_cast<const uint32_t*>(
      scales + size_t(row) * (k / 16) + column / 16));
#pragma unroll
  for (int step = 0; step < 4; ++step)
    weights[step] = complete_weight(
        uint8_t(codes[step] >> (((lane >> 1) & 3) * 8)),
        uint8_t(local >> (step * 8)), lane & 1);
}

template <int R, bool GEMVX>
__global__ void source_ordered_nvfp4(
    const __nv_bfloat16* x, const uint8_t* packed, const uint8_t* scales,
    const float* global, __nv_bfloat16* output, int n, int k) {
  const int lane = threadIdx.x % 16, local_row = threadIdx.x / 16;
  const int row = blockIdx.x * 8 + local_row;
  float total[R] = {};
  if constexpr (GEMVX) {
    // Actual GEMVX parameters select stride 16 and K64 per-lane loop tiles.
    // Its accumulator continues across tiles, followed by a 16-lane tree.
    for (int begin = 0; begin < 512; begin += 4) {
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        const int column = lane + (begin + step) * 16;
        const float weight = row < n ? original_complete_weight(
            packed[size_t(row) * (k / 2) + column / 2],
            scales[size_t(row) * (k / 16) + column / 16], column & 1) : 0.f;
#pragma unroll
        for (int r = 0; r < R; ++r)
          total[r] = __fmaf_rn(weight, __bfloat162float(x[size_t(r) * k + column]), total[r]);
      }
    }
  } else {
    // In-projection GEMV2T instead resets each stride-16 K512 tile's chain
    // and adds tile partials before the final serial lane reduction.
    for (int begin = 0; begin < k; begin += 512) {
      float tile[R] = {};
      for (int group = 0; group < 32; group += 4) {
        float weights[4];
        // The host gate admits only N divisible by 8 and aligned K64 records.
        if constexpr (R == 8)
          record_weights(packed, scales, row, k, begin + group * 16, lane, weights);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
          const int column = begin + lane + (group + step) * 16;
          float weight;
          if constexpr (R == 8) weight = weights[step];
          else weight = row < n ? original_complete_weight(
                packed[size_t(row) * (k / 2) + column / 2],
                scales[size_t(row) * (k / 16) + column / 16], column & 1) : 0.f;
#pragma unroll
          for (int r = 0; r < R; ++r)
            tile[r] = __fmaf_rn(weight, __bfloat162float(x[size_t(r) * k + column]), tile[r]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) total[r] = __fadd_rn(total[r], tile[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    float sum = total[r];
    if constexpr (GEMVX) {
#pragma unroll
      for (int offset = 8; offset; offset /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset, 16));
    } else {
      sum = __shfl_sync(0xffffffffu, total[r], 0, 16);
#pragma unroll
      for (int i = 1; i < 16; ++i)
        sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, total[r], i, 16));
    }
    if (lane == 0 && row < n)
      output[size_t(r) * n + row] = __float2bfloat16_rn(__fmul_rn(sum, global[0]));
  }
}

__global__ void probe_codec(float* original, float* candidate, int parity) {
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    const uint8_t packed = uint8_t((i & 15) * 17);
    original[i] = original_complete_weight(packed, i >> 4, parity);
    candidate[i] = complete_weight(packed, i >> 4, parity);
  }
}

__global__ void probe_record_values(
    const uint8_t* packed, const uint8_t* scales, float* output, int n, int k) {
  const size_t record = size_t(blockIdx.x) * 8 + threadIdx.x / 16;
  if (record >= size_t(n) * (k / 64)) return;
  const int row = record / (k / 64), column = (record % (k / 64)) * 64;
  const int lane = threadIdx.x % 16;
  float weights[4];
  record_weights(packed, scales, row, k, column, lane, weights);
#pragma unroll
  for (int step = 0; step < 4; ++step)
    output[size_t(row) * k + column + lane + step * 16] = weights[step];
}
}

extern "C" int probe_native_nvfp4_record_values(
    const uint8_t* packed, const uint8_t* scales, float* output,
    int n, int k, void* stream) {
  if (!packed || !scales || !output ||
      !((n == 18560 && k == 4096) || (n == 4096 && k == 8192))) return -1;
  const size_t records = size_t(n) * (k / 64);
  probe_record_values<<<(records + 7) / 8, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      packed, scales, output, n, k);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

extern "C" int probe_native_nvfp4_codec(
    float* original, float* candidate, int parity, void* stream) {
  if (!original || !candidate || parity < 0 || parity > 1) return -1;
  probe_codec<<<1, 128, 0, static_cast<cudaStream_t>(stream)>>>(original, candidate, parity);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}

extern "C" int launch_native_nvfp4_m1(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* opaque_stream) {
  if (!x || !packed || !scales || !global || !output || (rows != 1 && rows != 8)) return -1;
  if (!((n == 18560 && k == 4096) || (n == 4096 && k == 8192))) return -2;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  auto dispatch = [&](auto row_count, auto gemvx) {
    source_ordered_nvfp4<decltype(row_count)::value, decltype(gemvx)::value>
        <<<(n + 7) / 8, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x), packed, scales, global,
            static_cast<__nv_bfloat16*>(output), n, k);
  };
  if (rows == 1) {
    if (k == 8192) dispatch(std::integral_constant<int, 1>{}, std::true_type{});
    else dispatch(std::integral_constant<int, 1>{}, std::false_type{});
  } else {
    if (k == 8192) dispatch(std::integral_constant<int, 8>{}, std::true_type{});
    else dispatch(std::integral_constant<int, 8>{}, std::false_type{});
  }
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
