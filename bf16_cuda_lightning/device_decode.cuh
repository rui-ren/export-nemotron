// Included inside lightning_ops.cu's anonymous namespace.

__global__ void DeviceExpertActivationKernel(const float* partial, bf16* activated) {
  constexpr int count = 6 * 1856;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float r = fmaxf(RoundBF(partial[i] + partial[count + i]), 0.f);
    activated[i] = __float2bfloat16(r * r);
  }
}

template <bool Up, int Rows = 1>
__global__ void __launch_bounds__(128) DeviceExpertTiledMmaKernel(
    const bf16* input, const int32_t* indices, const bf16* weights,
    int experts, float* output) {
  constexpr int k = Up ? 2688 : 1856;
  constexpr int n = Up ? 1856 : 2688;
  constexpr int slices = Up ? 2 : 1;
  constexpr int chunks = k / slices / 64;
  __shared__ __align__(16) bf16 tiles[2][64 * 64];
  __shared__ __align__(16) bf16 activation[k];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int g = lane / 4, t = lane % 4;
  const int token = Rows == 1 ? 0 : blockIdx.y / 6;
  const int route = Rows == 1 ? blockIdx.y : blockIdx.y % 6, slice = blockIdx.z;
  const int first_row = blockIdx.x * 64;
  const int row = first_row + warp * 16 + g;
  const int expert = indices[token * 6 + route];
  const int64_t offset = ((static_cast<int64_t>(token) * slices + slice) * 6 + route) * n + row;
  if (expert < 0 || expert >= experts) {
    if (t == 0) { output[offset] = nanf(""); output[offset + 8] = nanf(""); }
    return;
  }
  const bf16* weight = weights + static_cast<int64_t>(expert) * n * k;
  const bf16* x = input + token * (Up ? k : 6 * k) + (Up ? 0 : route * k);
  for (int i = threadIdx.x; i < k; i += blockDim.x) activation[i] = x[i];
  auto prefetch = [&](int chunk) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int vector = threadIdx.x + i * 128;
      const int r = vector / 8, column = (vector % 8) * 8;
      const unsigned destination = __cvta_generic_to_shared(tiles[chunk % 2] + r * 64 + column);
      const bf16* source = weight + static_cast<int64_t>(first_row + r) * k
                          + slice * (k / slices) + chunk * 64 + column;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                   "r"(destination), "l"(source));
    }
    asm volatile("cp.async.commit_group;");
  };
  prefetch(0);
  float acc[4] = {};
  // Keep the source's contiguous split-K partitions and ascending MMA order.
#pragma unroll 1
  for (int chunk = 0; chunk < chunks; ++chunk) {
    if (chunk + 1 < chunks) {
      prefetch(chunk + 1);
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const unsigned address = __cvta_generic_to_shared(
          tiles[chunk % 2] + (warp * 16 + lane % 16) * 64
          + step * 16 + (lane / 16) * 8);
      uint32_t a0, a1, a2, a3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(address));
      const int base = slice * (k / slices) + chunk * 64 + step * 16 + t * 2;
      const uint32_t b0 = g == 0 ? *reinterpret_cast<const uint32_t*>(activation + base) : 0;
      const uint32_t b1 = g == 0 ? *reinterpret_cast<const uint32_t*>(activation + base + 8) : 0;
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    __syncthreads();
  }
  if (t == 0) {
    output[offset] = Up ? acc[0] : RoundBF(acc[0]);
    output[offset + 8] = Up ? acc[2] : RoundBF(acc[2]);
  }
}
