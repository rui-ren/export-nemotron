#include "/home/mobius/bolt-native-stream-ordered-20261028/wide/native_projection.cu"

namespace {
template <bool IQ2>
__global__ void __launch_bounds__(128) parallel_source_partition(
    const bf16* x, const uint8_t* packed, const __half* scales, float* partials,
    int rows, int n, int k, int partition_k) {
  __shared__ __align__(1024) bf16 shared_a[64 * 64];
  __shared__ __align__(1024) bf16 shared_b[8 * 64];
  __shared__ uint64_t grid[256];
  if constexpr (IQ2) {
    for (int i = threadIdx.x; i < 256; i += 128) grid[i] = kIq2xxsGrid[i];
    __syncthreads();
  }
  const int base = blockIdx.x * 64;
  const int start = blockIdx.y * partition_k;
  const int end = min(start + partition_k, k);
  float accum[4] = {0.f, 0.f, 0.f, 0.f};
  for (int begin = start; begin < end; begin += 64) {
    const int row = threadIdx.x / 2, column = (threadIdx.x % 2) * 32;
    const bool valid = base + row < n;
    const size_t weight_row = min(base + row, n - 1);
    const size_t index = weight_row * k + begin + column;
    if constexpr (IQ2) {
      const uint2 record = __ldg(
          reinterpret_cast<const uint2*>(packed + (index / 32) * 8));
      const float scale = (__half2float(
          scales[weight_row * (k / 256) + (begin + column) / 256]) *
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
          *reinterpret_cast<__nv_bfloat162*>(
              shared_a + core_offset(row, column + group * 8 + pair * 2)) =
              __floats2bfloat162_rn(valid ? first : 0.f, valid ? second : 0.f);
        }
      }
    } else {
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
        *reinterpret_cast<__nv_bfloat162*>(
            shared_a + core_offset(row, column + pair * 2)) =
            __floats2bfloat162_rn(valid ? first : 0.f, valid ? second : 0.f);
      }
    }
#pragma unroll
    for (int i = threadIdx.x; i < 8 * 64; i += 128) {
      const int input_row = i / 64, input_column = i % 64;
      shared_b[core_offset(input_row, input_column)] = input_row < rows
          ? x[size_t(input_row) * k + begin + input_column] : __float2bfloat16(0.f);
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      source_wgmma(accum, descriptor(shared_a + step * 128),
                   descriptor(shared_b + step * 128));
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    __syncthreads();
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = base + warp * 16 + lane / 4, column = (lane % 4) * 2;
  auto store = [&](int out_row, int out_column, float value) {
    if (out_row < n && out_column < rows)
      partials[(size_t(blockIdx.y) * rows + out_column) * n + out_row] = value;
  };
  store(row, column, accum[0]);
  store(row, column + 1, accum[1]);
  store(row + 8, column, accum[2]);
  store(row + 8, column + 1, accum[3]);
}

__global__ void source_partition_reduce(
    const float* partials, bf16* output, int count, int partitions) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  // Match the source reducer, including its initial +0 and sole BF16 boundary.
  float value = 0.f;
  for (int part = 0; part < partitions; ++part)
    value = __fadd_rn(value, partials[size_t(part) * count + index]);
  output[index] = __float2bfloat16_rn(value);
}
}

extern "C" int launch_native_split_projection(
    const void* x, const uint8_t* packed, const void* scales, float* partials,
    void* output, int rows, int n, int k, int format, void* opaque_stream) {
  const int partition_k =
      format == 1 && n == 4096 && k == 5376 ? 2688 :
      format == 0 && k == 4096 && (n == 256 || n == 1024) ? 1408 :
      format == 0 && k == 4096 && n == 4096 ? 2048 : 0;
  if (!x || !packed || !scales || !partials || !output || rows < 1 ||
      rows > 8 || !partition_k) return -1;
  const int partitions = (k + partition_k - 1) / partition_k;
  auto stream = static_cast<cudaStream_t>(opaque_stream);
  auto launch = [&](auto iq2) {
    parallel_source_partition<decltype(iq2)::value>
        <<<dim3((n + 63) / 64, partitions), 128, 0, stream>>>(
            static_cast<const bf16*>(x), packed, static_cast<const __half*>(scales),
            partials, rows, n, k, partition_k);
  };
  if (format == 1) launch(std::true_type{});
  else launch(std::false_type{});
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return 1000 + error;
  source_partition_reduce<<<(rows * n + 255) / 256, 256, 0, stream>>>(
      partials, static_cast<bf16*>(output), rows * n, partitions);
  error = cudaGetLastError();
  return error == cudaSuccess ? 0 : 1000 + error;
}
