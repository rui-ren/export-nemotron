// One source SSD chunk, replayed through graph-visible prefix history.
constexpr int kScanCapacity = 256;

__global__ void ScanHistoryKernel(const bf16* proj, const bf16* xbc, const bf16* past,
                                 bf16* present, int* lengths, int R, int start,
                                 int width, int stride, int di, int conv_dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kScanCapacity * width) {
    const int row = i / width, col = i % width;
    if (row < start) present[i] = past[i];
    else if (row < start + R)
      present[i] = col < conv_dim ? xbc[(row - start) * conv_dim + col]
                                  : proj[(row - start) * stride + di + col];
    else present[i] = __float2bfloat16(0.f);
  }
  if (i < R) lengths[i] = start + i + 1;
}

// CB and all temporal scaling stay FP32 until the source dot-operand boundary.
__global__ void ScanCoefficientsKernel(const bf16* hist, const float* bias, const float* A,
                                      float* coeff, float* dt_cs, int start, int R,
                                      int heads, int groups, int ds, int hd, float tmin) {
  const int h = blockIdx.x, tid = threadIdx.x;
  const int di = heads * hd, conv_dim = di + 2 * groups * ds, width = conv_dim + heads;
  const int total = start + R, g = h / (heads / groups);
  __shared__ float dt[kScanCapacity], cs[kScanCapacity];
  if (tid < kScanCapacity) {
    float raw = tid < total ? B2F(hist[tid * width + conv_dim + h]) : 0.f;
    raw += B2F(__float2bfloat16(bias[h]));
    // Pinned Triton 3 softplus is log(exp(raw) + 1), not log1p(exp(raw)).
    dt[tid] = tid < total ? fmaxf(raw <= 20.f ? logf(__expf(raw) + 1.f) : raw, tmin) : 0.f;
    cs[tid] = dt[tid] * A[h];
  }
  __syncthreads();
  // Inclusive parallel prefix, never restarted at an R8 boundary.
  for (int offset = 1; offset < kScanCapacity; offset *= 2) {
    float left = tid >= offset ? cs[tid - offset] : 0.f;
    __syncthreads();
    if (tid >= offset) cs[tid] = cs[tid] + left;
    __syncthreads();
  }
  dt_cs[(h * 2) * kScanCapacity + tid] = dt[tid];
  dt_cs[(h * 2 + 1) * kScanCapacity + tid] = cs[tid];
  for (int i = tid; i < R * kScanCapacity; i += blockDim.x) {
    const int r = i / kScanCapacity, j = i % kScanCapacity, t = start + r;
    float cb = 0.f;
    if (j <= t) {
      for (int n = 0; n < ds; ++n)
        cb = fmaf(B2F(hist[t * width + di + groups * ds + g * ds + n]),
                  B2F(hist[j * width + di + g * ds + n]), cb);
      cb = (cb * __expf(fminf(cs[t] - cs[j], 0.f))) * dt[j];
    }
    coeff[(r * heads + h) * kScanCapacity + j] = B2F(__float2bfloat16(cb));
  }
}

__global__ void ScanReadoutKernel(const bf16* hist, const float* initial, const float* coeff,
                                 const float* dt_cs, const float* D, float* ypre,
                                 int start, int R, int heads, int groups, int ds, int hd) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R * heads * hd) return;
  const int p = i % hd, h = (i / hd) % heads, r = i / (hd * heads), t = start + r;
  const int di = heads * hd, width = di + 2 * groups * ds + heads, g = h / (heads / groups);
  const float* cs = dt_cs + (h * 2 + 1) * kScanCapacity;
  float prior = 0.f;
  for (int n = 0; n < ds; ++n)
    prior = fmaf(B2F(hist[t * width + di + groups * ds + g * ds + n]),
                 B2F(__float2bfloat16(initial[(h * ds + n) * hd + p])), prior);
  float acc = prior * __expf(cs[t]);
  for (int j = 0; j <= t; ++j)
    acc = fmaf(coeff[(r * heads + h) * kScanCapacity + j],
               B2F(hist[j * width + h * hd + p]), acc);
  ypre[i] = acc + B2F(hist[t * width + h * hd + p]) * B2F(__float2bfloat16(D[h]));
}

__global__ void ScanStateKernel(const bf16* hist, const float* initial, const float* dt_cs,
                               float* state, int start, int R, int heads, int groups,
                               int ds, int hd) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_size = heads * ds * hd;
  if (i >= R * row_size) return;
  const int index = i % row_size, r = i / row_size;
  const int p = index % hd, n = (index / hd) % ds, h = index / (hd * ds);
  const int t = start + r, g = h / (heads / groups), di = heads * hd;
  const int width = di + 2 * groups * ds + heads;
  const float* dt = dt_cs + (h * 2) * kScanCapacity;
  const float* cs = dt + kScanCapacity;
  float acc = 0.f;
  for (int j = 0; j <= t; ++j) {
    const float scale = __expf(fminf(cs[t] - cs[j], 0.f)) * dt[j];
    const float scaled_b = B2F(__float2bfloat16(
        B2F(hist[j * width + di + g * ds + n]) * scale));
    acc = fmaf(B2F(hist[j * width + h * hd + p]), scaled_b, acc);
  }
  // Chunk-state passing retains the original state in FP32.
  state[i] = acc + initial[index] * __expf(cs[t]);
}

void LaunchSourceScan(cudaStream_t stream, const bf16* proj, const bf16* xbc,
                      const bf16* past, bf16* present, int* lengths,
                      const float* initial, const float* bias, const float* A,
                      const float* D, float* state, float* ypre,
                      float* coeff, float* dt_cs, int start, int R,
                      int heads, int groups, int ds, int hd, float tmin) {
  const int di = heads * hd, conv_dim = di + 2 * groups * ds;
  const int width = conv_dim + heads, stride = di + width;
  ScanHistoryKernel<<<(kScanCapacity * width + 255) / 256, 256, 0, stream>>>(
      proj, xbc, past, present, lengths, R, start, width, stride, di, conv_dim);
  LaunchOriginalSourceStates(stream, present, initial, bias, A, state, start, R, tmin);
  E6ParallelReadout(stream,present,initial,bias,A,D,ypre,coeff,dt_cs,start,R,tmin);
}

struct MambaScanVerifyState : MambaStepState {
  int chunk_size;
  bool diagnostics;
  MambaScanVerifyState(const OrtApi& a, const OrtKernelInfo* info,bool debug=false)
      : MambaStepState(a, info, true),diagnostics(debug) {
    chunk_size = static_cast<int>(Ort::ConstKernelInfo(info).GetAttribute<int64_t>("chunk_size"));
    if (chunk_size != 128 || heads != 128 || ng != 8 || ds != 128 || hd != 64)
      throw std::invalid_argument(
          "MambaScanVerify E5 requires chunk_size=128, heads=128, groups=8, dstate=128, head_dim=64");
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const int R = static_cast<int>(Dim(c.GetInput(0), 0));
      const int di = heads * hd, conv_dim = di + 2 * ng * ds, width = conv_dim + heads;
      const int mode = *In<int>(c, 10);
      const int snapshots = static_cast<int>(Dim(c.GetInput(1), 0));
      if (mode < 0 || mode > 2 || R < 1 || R > 16 || snapshots < R ||
          Dim(c.GetInput(0), 1) != di + width ||
          c.GetInput(1).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{snapshots, conv_dim, 3} ||
          c.GetInput(2).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{snapshots, heads, ds, hd} ||
          Count(c.GetInput(3)) != conv_dim * 4 || Count(c.GetInput(4)) != conv_dim ||
          Count(c.GetInput(5)) != heads || Count(c.GetInput(6)) != heads ||
          Count(c.GetInput(7)) != heads || Count(c.GetInput(8)) != di ||
          Count(c.GetInput(9)) != 1 || Count(c.GetInput(10)) != 1 ||
          c.GetInput(11).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{256, width} ||
          c.GetInput(12).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{snapshots} ||
          c.GetInput(13).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{heads, ds, hd})
        throw std::invalid_argument("MambaScanVerify: invalid mode or tensor shape");
      cudaStream_t s = Stream(api, raw);
      int selected = 0, start = 0;
      if (cudaMemcpyAsync(&selected, In<int>(c, 9), sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
          cudaStreamSynchronize(s) != cudaSuccess)
        throw std::runtime_error("MambaScanVerify: cannot read state_row; CUDA graph capture unsupported");
      if (selected < 0 || selected >= snapshots)
        throw std::invalid_argument("MambaScanVerify: state_row out of range");
      if (mode == 2) {
        if (cudaMemcpyAsync(&start, In<int>(c, 12) + selected, sizeof(int), cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaStreamSynchronize(s) != cudaSuccess)
          throw std::runtime_error("MambaScanVerify: cannot read history length");
        if (start < 1 || start + R > chunk_size)
          throw std::invalid_argument("MambaScanVerify: invalid history or SSD chunk boundary exceeded");
      }
      auto hist = c.GetOutput(5, std::vector<int64_t>{256, width});
      auto lengths = c.GetOutput(6, std::vector<int64_t>{snapshots});
      auto initial = c.GetOutput(7, std::vector<int64_t>{heads, ds, hd});
      cudaMemsetAsync(Out<int>(lengths), 0xff, snapshots * sizeof(int), s);
      if (mode == 0) {
        if (diagnostics) {
          auto dt=c.GetOutput(8,std::vector<int64_t>{128,2,256});
          auto co=c.GetOutput(9,std::vector<int64_t>{R,128,128});
          E6CachedDiagnostics<<<256,256,0,s>>>(
              In<bf16>(c,0),In<float>(c,5),In<float>(c,6),Out<float>(dt),R);
          cudaMemsetAsync(Out<bf16>(co),0,R*128*128*sizeof(bf16),s);
        }
        cudaMemcpyAsync(Out<bf16>(hist), In<bf16>(c, 11), 256 * width * sizeof(bf16), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(Out<float>(initial), In<float>(c, 13), heads * ds * hd * sizeof(float), cudaMemcpyDeviceToDevice, s);
        return MambaStepState::ComputeV2(raw);
      }
      const int64_t row_size = static_cast<int64_t>(heads) * ds * hd;
      cudaMemcpyAsync(Out<float>(initial),
                      mode == 1 ? In<float>(c, 2) + selected * row_size : In<float>(c, 13),
                      row_size * sizeof(float), cudaMemcpyDeviceToDevice, s);
      auto y = c.GetOutput(0, std::vector<int64_t>{R, di});
      auto conv = c.GetOutput(1, c.GetInput(1).GetTensorTypeAndShapeInfo().GetShape());
      auto state = c.GetOutput(2, c.GetInput(2).GetTensorTypeAndShapeInfo().GetShape());
      auto xbc = c.GetOutput(3, std::vector<int64_t>{R, conv_dim});
      auto ypre = c.GetOutput(4, std::vector<int64_t>{R, di});
      MambaConvKernel<<<(conv_dim + 127) / 128, 128, 0, s>>>(
          In<bf16>(c, 0), R, di + width, di, conv_dim, In<bf16>(c, 1),
          In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv), Out<bf16>(xbc), In<int>(c, 9), true);
      float* coeff = nullptr;
      if (cudaMallocAsync(&coeff, (R * heads * 256 + heads * 512) * sizeof(float), s) != cudaSuccess)
        throw std::runtime_error("MambaScanVerify: scratch allocation failed");
      LaunchSourceScan(s, In<bf16>(c, 0), Out<bf16>(xbc), In<bf16>(c, 11),
                       Out<bf16>(hist), Out<int>(lengths), Out<float>(initial),
                       In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
                       Out<float>(state), Out<float>(ypre), coeff, coeff + R * heads * 256,
                       start, R, heads, ng, ds, hd, tmin);
      if (diagnostics) {
        auto dt=c.GetOutput(8,std::vector<int64_t>{128,2,256});
        auto co=c.GetOutput(9,std::vector<int64_t>{R,128,128});
        cudaMemcpyAsync(Out<float>(dt),coeff+R*heads*256,128*512*sizeof(float),
                        cudaMemcpyDeviceToDevice,s);
        E6GatherCoefficients<<<(R*128*128+255)/256,256,0,s>>>(
            coeff,Out<bf16>(co),R*128*128);
      }
      LaunchMambaNormSource(s,R,ng,In<bf16>(c,0),di+width,di,di/ng,eps,
                            Out<float>(ypre),In<float>(c,8),Out<bf16>(y));
      cudaFreeAsync(coeff, s);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) {
      return api.CreateStatus(ORT_FAIL, e.what());
    }
  }
};
