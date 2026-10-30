#define launch_native_source_projection launch_ampere_source_projection
#include "ampere_projection.cu"
#undef launch_native_source_projection

namespace {
__device__ __forceinline__ uint64_t descriptor(const bf16* pointer) {
  const uint64_t address = __cvta_generic_to_shared(pointer);
  // Unswizzled 8x8 BF16 core matrices: 128-byte K stride and 1024-byte M stride.
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

template <bool IQ2>
__device__ __forceinline__ bf16 complete_weight(
    const uint8_t* packed, const __half* scales, const uint64_t* grid,
    int row, int column, int k) {
  const size_t index = size_t(row) * k + column;
  if constexpr (!IQ2) {
    return __float2bfloat16_rn((float(packed[index]) - 128.f) * __half2float(scales[index / 32]));
  } else {
    const uint8_t* record = packed + (index / 32) * 8;
    const uint32_t control = uint32_t(record[4]) | (uint32_t(record[5]) << 8) |
        (uint32_t(record[6]) << 16) | (uint32_t(record[7]) << 24);
    const int group = (column % 32) / 8, lane = column % 8;
    const uint32_t sign7 = (control >> (7 * group)) & 127u;
    const uint32_t signs = sign7 | ((__popc(sign7) & 1u) << 7);
    const float magnitude = float((grid[record[group]] >> (8 * lane)) & 255u);
    const float scale = (__half2float(scales[size_t(row) * (k / 256) + column / 256]) *
                          (float(control >> 28) + 0.5f)) * 0.25f;
    return __float2bfloat16_rn(
        magnitude * scale * (((signs >> lane) & 1) ? -1.f : 1.f));
  }
}

template <bool IQ2, int PH = 0>
__global__ void __launch_bounds__(128) projection_wgmma(
    const bf16* x, const uint8_t* packed, const __half* scales, void* output,
    int rows, int n, int k, const int32_t* meta = nullptr, int pairs = 0) {
  __shared__ __align__(1024) bf16 shared_a[64 * 64];
  __shared__ __align__(1024) bf16 shared_b[8 * 64];
  __shared__ uint64_t grid[256];
  if constexpr (IQ2) {
    for (int i = threadIdx.x; i < 256; i += 128) grid[i] = kIq2xxsGrid[i];
    __syncthreads();
  }
  int expert = 0;
  const int32_t* pair_of = nullptr;
  if constexpr (PH != 0) {
    if (blockIdx.y >= meta[0]) return;
    expert = meta[1 + blockIdx.y];
    pair_of = meta + 1 + pairs + blockIdx.y * 8;
  }
  const int base = blockIdx.x * 64;
  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  float reduced[4] = {0.f, 0.f, 0.f, 0.f};
  // Actual M1 source launch c[0x350], not partition sizes fitted to outputs.
  const int partition_k = PH == 0
      ? (!IQ2 && n == 1024 && k == 4096 ? 1408
         : IQ2 && n == 4096 && k == 5376 ? 2688 : 0)
      : 0;
  for (int begin = 0; begin < k; begin += 64) {
    if constexpr (IQ2) {
      // One thread decodes one aligned 32-weight record instead of serially
      // refetching its control word and scale for every scalar weight.
      const int row = threadIdx.x / 2, column = (threadIdx.x % 2) * 32;
      const bool valid = base + row < n;
      const size_t weight_row = size_t(expert) * n + min(base + row, n - 1);
      const size_t index = weight_row * k + begin + column;
      const uint2 record = __ldg(reinterpret_cast<const uint2*>(packed + (index / 32) * 8));
      const float scale = (__half2float(scales[weight_row * (k / 256) + (begin + column) / 256]) *
                             (float(record.y >> 28) + 0.5f)) * 0.25f;
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const uint64_t magnitudes = grid[(record.x >> (8 * group)) & 255u];
        const uint32_t sign7 = (record.y >> (7 * group)) & 127u;
        const uint32_t signs = sign7 | ((__popc(sign7) & 1u) << 7);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const float first = float((magnitudes >> (16 * pair)) & 255u) * scale *
              (((signs >> (2 * pair)) & 1u) ? -1.f : 1.f);
          const float second = float((magnitudes >> (16 * pair + 8)) & 255u) * scale *
              (((signs >> (2 * pair + 1)) & 1u) ? -1.f : 1.f);
          *reinterpret_cast<__nv_bfloat162*>(shared_a + core_offset(row, column + group * 8 + pair * 2)) =
              __floats2bfloat162_rn(valid ? first : 0.f, valid ? second : 0.f);
        }
      }
    } else {
      // One thread consumes a complete aligned Q8 record: two vector loads,
      // one original FP16 scale, then paired BF16 stores in the existing layout.
      const int row = threadIdx.x / 2, column = (threadIdx.x % 2) * 32;
      const bool valid = base + row < n;
      const size_t index = size_t(expert * n + min(base + row, n - 1)) * k + begin + column;
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
        *reinterpret_cast<__nv_bfloat162*>(shared_a + core_offset(row, column + pair * 2)) =
            __floats2bfloat162_rn(valid ? first : 0.f, valid ? second : 0.f);
      }
    }
#pragma unroll
    for (int i = threadIdx.x; i < 8 * 64; i += 128) {
      const int row = i / 64, column = i % 64;
      int input_row = row;
      bool valid = row < rows;
      if constexpr (PH == 2) {
        input_row = pair_of[row];
        valid = input_row >= 0;
      }
      shared_b[core_offset(row, column)] = valid
          ? x[size_t(input_row) * k + begin + column] : __float2bfloat16(0.f);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    // Each observed partition uses four ascending HGMMA K16 steps per K64 tile.
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      source_wgmma(accum, descriptor(shared_a + step * 128),
                   descriptor(shared_b + step * 128));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    __syncthreads();
    if (partition_k && ((begin + 64) % partition_k == 0 || begin + 64 == k)) {
      // Source splitKreduce adds FP32 partials serially before its only BF16 cast.
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        reduced[i] = __fadd_rn(reduced[i], accum[i]);
        accum[i] = 0.f;
      }
    }
  }
  if (partition_k) {
#pragma unroll
    for (int i = 0; i < 4; ++i) accum[i] = reduced[i];
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = base + warp * 16 + lane / 4, column = (lane % 4) * 2;
  auto store = [&](int out_row, int out_column, float value) {
    if (out_row >= n || out_column >= rows) return;
    int target_row = out_column;
    if constexpr (PH != 0) {
      target_row = pair_of[out_column];
      if (target_row < 0) return;
    }
    const size_t index = size_t(target_row) * n + out_row;
    const bf16 rounded = __float2bfloat16_rn(value);
    if constexpr (PH == 2) {
      static_cast<float*>(output)[index] = __bfloat162float(rounded);
    } else if constexpr (PH == 1) {
      const float relu = fmaxf(__bfloat162float(rounded), 0.f);
      static_cast<bf16*>(output)[index] = __float2bfloat16_rn(relu * relu);
    } else {
      static_cast<bf16*>(output)[index] = rounded;
    }
  };
  store(row, column, accum[0]);
  store(row, column + 1, accum[1]);
  store(row + 8, column, accum[2]);
  store(row + 8, column + 1, accum[3]);
}
}

extern "C" int launch_native_source_projection(
    const void* x, const uint8_t* packed, const void* scales, void* output,
    int rows, int n, int k, int format, int k_partitions, void* opaque_stream) {
  if (!x || !packed || !scales || !output || rows < 1 || rows > 8 ||
      n < 1 || k < 256 || k % 256 || (format != 0 && format != 1) || k_partitions != 1) return -1;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  auto launch = [&](auto iq2) {
    projection_wgmma<decltype(iq2)::value><<<(n + 63) / 64, 128, 0, stream>>>(
        static_cast<const bf16*>(x), packed, static_cast<const __half*>(scales),
        static_cast<bf16*>(output), rows, n, k);
  };
  if (format == 1) launch(std::true_type{});
  else launch(std::false_type{});
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
