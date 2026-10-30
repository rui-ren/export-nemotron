// Same source recurrence/readout, with eight adjacent head channels per warp.
__global__ void MambaCachedSourceKernel(
    const bf16* proj, int R, int stride, int di, int conv_dim, int heads, int groups,
    const bf16* xbc, const float* state_in, const float* bias, const float* A,
    const float* D, float* state_out, float* ypre, const int* selected) {
  const int local_p = threadIdx.x % 8;
  const int lane = threadIdx.x / 8;
  const int p = blockIdx.y * 8 + local_p;
  const int h = blockIdx.x, g = h / (heads / groups);
  const int n = lane * 4;
  const int row_size = heads * 128 * 64;
  const int base = (h * 128 + n) * 64 + p;
  const int state_row = selected ? *selected : 0;
  __shared__ float timesteps[16], decays[16], partial[256];
  if (threadIdx.x < R) {
    const int r = threadIdx.x;
    const float raw = B2F(proj[r * stride + di + conv_dim + h]) + bias[h];
    const float dt = raw <= 20.f ? logf(__expf(raw) + 1.f) : raw;
    timesteps[r] = dt;
    decays[r] = __expf(__fmul_rn(A[h], dt));
  }
  __syncthreads();
  float s[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    s[k] = state_in[state_row * row_size + base + k * 64];
  for (int r = 0; r < R; ++r) {
    const float dt = timesteps[r], decay = decays[r];
    const float x = B2F(xbc[r * conv_dim + h * 64 + p]);
    float c[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float b = B2F(xbc[r * conv_dim + di + g * 128 + n + k]);
      c[k] = B2F(xbc[r * conv_dim + di + groups * 128 + g * 128 + n + k]);
      const float inject = __fmul_rn(__fmul_rn(b, dt), x);
      s[k] = fmaf(decay, s[k], inject);
      state_out[r * row_size + base + k * 64] = s[k];
    }
    float sum = fmaf(s[0], c[0], __fmul_rn(s[1], c[1])) +
                fmaf(s[2], c[2], __fmul_rn(s[3], c[3]));
    partial[threadIdx.x] = sum;
    __syncthreads();
    // Reproduce source XOR16, XOR8 and XOR4 across warps, in the same
    // operand order. The remaining XOR2/XOR1 are local to the packed warp.
    const float a = __fadd_rn(sum, partial[(lane ^ 16) * 8 + local_p]);
    const float b = __fadd_rn(partial[(lane ^ 8) * 8 + local_p],
                             partial[(lane ^ 24) * 8 + local_p]);
    const float c0 = __fadd_rn(partial[(lane ^ 4) * 8 + local_p],
                              partial[(lane ^ 20) * 8 + local_p]);
    const float d = __fadd_rn(partial[(lane ^ 12) * 8 + local_p],
                             partial[(lane ^ 28) * 8 + local_p]);
    sum = __fadd_rn(__fadd_rn(a, b), __fadd_rn(c0, d));
    __syncthreads();
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 16));
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 8));
    if (lane == 0) ypre[r * di + h * 64 + p] = fmaf(x, D[h], sum);
  }
}
