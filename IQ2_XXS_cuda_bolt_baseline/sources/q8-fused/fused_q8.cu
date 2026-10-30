#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
using bf16 = __nv_bfloat16;

__device__ __forceinline__ uint64_t descriptor(const bf16* pointer) {
  const uint64_t address = __cvta_generic_to_shared(pointer);
  return ((address & 0x3ffffu) >> 4) | (uint64_t(8) << 16) | (uint64_t(64) << 32);
}

__device__ __forceinline__ int core_offset(int row, int column) {
  return (row / 8) * 512 + (column / 8) * 64 + (row % 8) * 8 + column % 8;
}

__device__ __forceinline__ void source_wgmma(float* accum, uint64_t a, uint64_t b) {
  asm volatile(
      "{\n"
      ".reg .pred use_accum;\n"
      "setp.ne.b32 use_accum, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3}, %4, %5, use_accum, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3])
      : "l"(a), "l"(b)
      : "memory");
}

__device__ __forceinline__ void group_barrier(int group) {
  // IDs 1..3 isolate unequal partition loops; barrier 0 is the final CTA join.
  asm volatile("bar.sync %0, 128;\n" : : "r"(group + 1) : "memory");
}

template <int PARTITION_K, int PARTITIONS>
__global__ void __launch_bounds__(PARTITIONS * 128) fused_q8_projection(
    const bf16* x, const uint8_t* packed, const __half* scales, bf16* output,
    int n, int k) {
  __shared__ __align__(1024) bf16 shared_a[PARTITIONS][64 * 64];
  __shared__ __align__(1024) bf16 shared_b[PARTITIONS][8 * 64];
  __shared__ float partials[PARTITIONS][8 * 64];
  const int group = threadIdx.x / 128, tid = threadIdx.x % 128;
  const int base = blockIdx.x * 64;
  const int start = group * PARTITION_K;
  const int end = min(start + PARTITION_K, k);
  bf16* a = shared_a[group];
  bf16* b = shared_b[group];
  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  for (int begin = start; begin < end; begin += 64) {
    const int row = tid / 2, column = (tid % 2) * 32;
    const bool valid = base + row < n;
    const size_t index = size_t(min(base + row, n - 1)) * k + begin + column;
    const uint4 lo = __ldg(reinterpret_cast<const uint4*>(packed + index));
    const uint4 hi = __ldg(reinterpret_cast<const uint4*>(packed + index + 16));
    const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const float scale = __half2float(scales[index / 32]);
#pragma unroll
    for (int pair = 0; pair < 16; ++pair) {
      const uint32_t word = words[pair / 2];
      const int shift = (pair % 2) * 16;
      const float first = __fmul_rn(float(int((word >> shift) & 255u) - 128), scale);
      const float second = __fmul_rn(float(int((word >> (shift + 8)) & 255u) - 128), scale);
      *reinterpret_cast<__nv_bfloat162*>(a + core_offset(row, column + pair * 2)) =
          __floats2bfloat162_rn(valid ? first : 0.f, valid ? second : 0.f);
    }
#pragma unroll
    for (int i = tid; i < 8 * 64; i += 128) {
      const int row_x = i / 64, column_x = i % 64;
      b[core_offset(row_x, column_x)] = x[size_t(row_x) * k + begin + column_x];
    }
    group_barrier(group);
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int step = 0; step < 4; ++step)
      source_wgmma(accum, descriptor(a + step * 128), descriptor(b + step * 128));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    group_barrier(group);
  }
  const int warp = tid / 32, lane = tid % 32;
  const int row = warp * 16 + lane / 4, column = (lane % 4) * 2;
  partials[group][column * 64 + row] = accum[0];
  partials[group][(column + 1) * 64 + row] = accum[1];
  partials[group][column * 64 + row + 8] = accum[2];
  partials[group][(column + 1) * 64 + row + 8] = accum[3];
  // Every group has waited for WGMMA before publishing its complete FP32 partial.
  __syncthreads();
  if (group == 0) {
    auto reduce = [&](int out_row, int out_column) {
      if (base + out_row >= n) return;
      float value = 0.f;
#pragma unroll
      for (int part = 0; part < PARTITIONS; ++part)
        value = __fadd_rn(value, partials[part][out_column * 64 + out_row]);
      output[size_t(out_column) * n + base + out_row] = __float2bfloat16_rn(value);
    };
    reduce(row, column);
    reduce(row, column + 1);
    reduce(row + 8, column);
    reduce(row + 8, column + 1);
  }
}
}

extern "C" int launch_native_source_projection(
    const void*, const uint8_t*, const void*, void*, int, int, int, int, int, void*);

extern "C" int launch_fused_or_accepted_projection(
    const void* x, const uint8_t* packed, const void* scales, void* output,
    int rows, int n, int k, int format, int k_partitions, void* opaque_stream) {
  if (rows != 8 || format != 0 || k_partitions != 1 || k != 4096 ||
      (n != 256 && n != 1024 && n != 4096))
    return launch_native_source_projection(
        x, packed, scales, output, rows, n, k, format, k_partitions, opaque_stream);
  if (!x || !packed || !scales || !output) return -1;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  if (n == 4096)
    fused_q8_projection<2048, 2><<<(n + 63) / 64, 256, 0, stream>>>(
        static_cast<const bf16*>(x), packed, static_cast<const __half*>(scales),
        static_cast<bf16*>(output), n, k);
  else
    fused_q8_projection<1408, 3><<<(n + 63) / 64, 384, 0, stream>>>(
        static_cast<const bf16*>(x), packed, static_cast<const __half*>(scales),
        static_cast<bf16*>(output), n, k);
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
