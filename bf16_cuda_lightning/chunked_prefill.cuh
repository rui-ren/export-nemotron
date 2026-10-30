// Included inside lightning_ops.cu's anonymous namespace.

struct RowLinearState {
  const OrtApi& api;
  cublasHandle_t blas = nullptr;
  std::mutex mutex;
  RowLinearState(const OrtApi& a, const OrtKernelInfo*) : api(a) {
    CheckBlas(cublasCreate(&blas));
  }
  ~RowLinearState() {
    if (blas) cublasDestroy(blas);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 2, 1);
      const auto xs = Dimensions(c.GetInput(0)), ws = Dimensions(c.GetInput(1));
      Require((xs.size() == 2 || (xs.size() == 3 && xs[0] == 1)) &&
              ws.size() == 2 && xs.back() == ws[0], "RowLinear: expected [T,K] or [1,T,K] @ [K,N]");
      const int rows = PositiveInt(xs[xs.size() - 2], "RowLinear: invalid rows");
      const int k = PositiveInt(ws[0], "RowLinear: invalid K");
      const int n = PositiveInt(ws[1], "RowLinear: invalid N");
      CheckTensor(c, 0, xs, BF, "RowLinear: invalid BF16 activation");
      CheckTensor(c, 1, ws, BF, "RowLinear: invalid BF16 weights");
      auto output_shape = xs;
      output_shape.back() = n;
      auto output = c.GetOutput(0, output_shape);
      const auto stream = Stream(api, raw);
      std::lock_guard<std::mutex> lock(mutex);
      const auto x = In<bf16>(c, 0), weight = In<bf16>(c, 1);
      auto y = Out<bf16>(output);
      CheckBlas(cublasSetStream(blas, stream));
      const float alpha = 1.f, beta = 0.f;
      for (int row = 0; row < rows; ++row)
        CheckBlas(cublasGemmEx(
            blas, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, k, &alpha,
            weight, CUDA_R_16BF, n, x + static_cast<int64_t>(row) * k, CUDA_R_16BF, k, &beta,
            y + static_cast<int64_t>(row) * n, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

__global__ void ChunkPositionsKernel(int32_t* positions, int start, int rows) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < rows) positions[t] = start + t;
}

__global__ void ChunkAppendKvKernel(
    const bf16* key, const bf16* value, int width, int dim, int capacity, int position,
    bf16* cache_key, bf16* cache_value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= width) return;
  const int offset = (i / dim * capacity + position) * dim + i % dim;
  cache_key[offset] = key[i];
  cache_value[offset] = value[i];
}

struct CachedAttentionChunkState {
  const OrtApi& api;
  int64_t heads, kv_heads;
  Ort::Op attention{nullptr};
  CachedAttentionChunkState(const OrtApi& a, const OrtKernelInfo* info) : api(a) {
    Ort::ConstKernelInfo k(info);
    heads = k.GetAttribute<int64_t>("num_heads");
    kv_heads = k.GetAttribute<int64_t>("kv_num_heads");
    Require(heads == 32 && kv_heads == 2, "CachedAttentionChunk: Lightning32/2 heads required");
    const char* names[] = {"T", "T_CACHE", "T_KV_SCALE", "M"};
    ONNXTensorElementDataType types[] = {BF, BF, F32, I32};
    int64_t rotary = 0;
    Ort::OpAttr attributes[] = {
        {"num_heads", &heads, 1, ORT_OP_ATTR_INT},
        {"kv_num_heads", &kv_heads, 1, ORT_OP_ATTR_INT},
        {"do_rotary", &rotary, 1, ORT_OP_ATTR_INT}};
    attention = Ort::Op::Create(info, "GroupQueryAttention", "com.microsoft", 1,
                               names, types, 4, attributes, 3, 7, 3);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 7, 3);
      const auto qs = Dimensions(c.GetInput(0)), cache_shape = Dimensions(c.GetInput(3));
      Require(qs.size() == 3 && qs[0] == 1 && qs[2] == heads * 128 &&
              cache_shape.size() == 4 && cache_shape[0] == 1 &&
              cache_shape[1] == kv_heads && cache_shape[3] == 128,
              "CachedAttentionChunk: invalid query/cache shape");
      const int rows = PositiveInt(qs[1], "CachedAttentionChunk: invalid rows");
      const int capacity = PositiveInt(cache_shape[2], "CachedAttentionChunk: invalid capacity");
      CheckTensor(c, 0, qs, BF, "CachedAttentionChunk: invalid queries");
      for (int i = 1; i < 3; ++i)
        CheckTensor(c, i, {1, rows, kv_heads * 128}, BF, "CachedAttentionChunk: invalid K/V");
      for (int i = 3; i < 5; ++i)
        CheckTensor(c, i, cache_shape, BF, "CachedAttentionChunk: invalid cache");
      CheckTensor(c, 5, {1}, I32, "CachedAttentionChunk: invalid GPU lengths");
      CheckTensor(c, 6, {1}, I32, "CachedAttentionChunk: total length must be CPU INT32[1]");
      const int total = In<int32_t>(c, 6)[0];
      Require(total >= rows && total <= capacity, "CachedAttentionChunk: cache capacity exceeded");
      const int start = total - rows;
      auto y = c.GetOutput(0, qs);
      auto key = c.GetOutput(1, cache_shape), value = c.GetOutput(2, cache_shape);
      const auto stream = Stream(api, raw);
      const size_t cache_bytes = static_cast<size_t>(kv_heads) * capacity * 128 * sizeof(bf16);
      for (int i = 0; i < 2; ++i) {
        auto output = i == 0 ? Out<bf16>(key) : Out<bf16>(value);
        if (output != In<bf16>(c, 3 + i)) {
          const auto error = cudaMemcpyAsync(output, In<bf16>(c, 3 + i), cache_bytes, cudaMemcpyDeviceToDevice, stream);
          if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        }
      }
      AsyncBuffer positions(rows * sizeof(int32_t), stream);
      ChunkPositionsKernel<<<CeilDiv(rows, 128), 128, 0, stream>>>(
          static_cast<int32_t*>(positions.data), start, rows);
      Ort::MemoryInfo gpu("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
      auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      const Shape query_shape{1, 1, heads * 128}, length_shape{1};
      for (int t = 0; t < rows; ++t) {
        int32_t length = start + t + 1;
        // Native Flash uses its ordinary64-column tile until split-KV is beneficial.
        // Read-only GQA avoids ORT's forced split-KV append kernel in that regime.
        const bool read_only = length <= 128;
        if (read_only)
          ChunkAppendKvKernel<<<CeilDiv(kv_heads * 128, 128), 128, 0, stream>>>(
              In<bf16>(c, 1) + static_cast<int64_t>(t) * kv_heads * 128,
              In<bf16>(c, 2) + static_cast<int64_t>(t) * kv_heads * 128,
              kv_heads * 128, 128, capacity, length - 1, Out<bf16>(key), Out<bf16>(value));
        const Shape kv_shape{1, read_only ? 0 : 1, kv_heads * 128};
        Ort::Value inputs[] = {
            Ort::Value::CreateTensor(gpu, const_cast<bf16*>(In<bf16>(c, 0)) + static_cast<int64_t>(t) * heads * 128,
                heads * 128 * sizeof(bf16), query_shape.data(), 3, BF),
            Ort::Value::CreateTensor(gpu, const_cast<bf16*>(In<bf16>(c, 1)) + static_cast<int64_t>(t) * kv_heads * 128,
                (read_only ? 0 : kv_heads * 128 * sizeof(bf16)), kv_shape.data(), 3, BF),
            Ort::Value::CreateTensor(gpu, const_cast<bf16*>(In<bf16>(c, 2)) + static_cast<int64_t>(t) * kv_heads * 128,
                (read_only ? 0 : kv_heads * 128 * sizeof(bf16)), kv_shape.data(), 3, BF),
            Ort::Value::CreateTensor(gpu, Out<bf16>(key), cache_bytes, cache_shape.data(), 4, BF),
            Ort::Value::CreateTensor(gpu, Out<bf16>(value), cache_bytes, cache_shape.data(), 4, BF),
            Ort::Value::CreateTensor(gpu, static_cast<int32_t*>(positions.data) + t,
                sizeof(int32_t), length_shape.data(), 1, I32),
            Ort::Value::CreateTensor(cpu, &length, sizeof(int32_t), length_shape.data(), 1, I32)};
        Ort::Value outputs[] = {
            Ort::Value::CreateTensor(gpu, Out<bf16>(y) + static_cast<int64_t>(t) * heads * 128,
                heads * 128 * sizeof(bf16), query_shape.data(), 3, BF),
            Ort::Value::CreateTensor(gpu, Out<bf16>(key), cache_bytes, cache_shape.data(), 4, BF),
            Ort::Value::CreateTensor(gpu, Out<bf16>(value), cache_bytes, cache_shape.data(), 4, BF)};
        attention.Invoke(raw, inputs, 7, outputs, 3);
      }
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

__global__ void MambaChunkConvKernel(
    const bf16* proj, int rows, int stride, int di, int channels, int kernel,
    const bf16* past, const bf16* weight, const bf16* bias, bf16* present, bf16* xbc) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
  if (c >= channels) return;
  float acc = B2F(bias[c]);
  for (int j = 0; j < kernel; ++j) {
    const int source = t - kernel + 1 + j;
    const float x = B2F(source < 0 ? past[static_cast<int64_t>(c) * (kernel - 1) + source + kernel - 1]
                                   : proj[static_cast<int64_t>(source) * stride + di + c]);
    acc += B2F(weight[static_cast<int64_t>(c) * kernel + j]) * x;
  }
  xbc[static_cast<int64_t>(t) * channels + c] =
      __float2bfloat16(__fdividef(acc, 1.f + __expf(-acc)));
  if (t == 0) {
    for (int j = 0; j < kernel - 1; ++j) {
      const int source = rows - kernel + 1 + j;
      present[static_cast<int64_t>(c) * (kernel - 1) + j] =
          source < 0 ? past[static_cast<int64_t>(c) * (kernel - 1) + source + kernel - 1]
                     : proj[static_cast<int64_t>(source) * stride + di + c];
    }
  }
}

__global__ void MambaChunkSourceKernel(
    const bf16* proj, int rows, int stride, int di, int channels, int heads_per_group, int groups,
    const bf16* xbc, const float* past, const float* dt_bias, const float* a, const float* skip,
    float tmin, const int32_t* position, float* present, float* ypre,
    float* penultimate = nullptr) {
  const int h = blockIdx.x, p = blockIdx.y * 4 + threadIdx.x / 32;
  const int n = (threadIdx.x & 31) * 4, g = h / heads_per_group;
  float state[4];
  const bool cold = position[0] == 0;
  for (int j = 0; j < 4; ++j) {
    const int64_t offset = (static_cast<int64_t>(h) * 128 + n + j) * 64 + p;
    state[j] = cold ? present[offset] : past[offset];
    if (penultimate && cold && rows == 2) penultimate[offset] = state[j];
  }
  for (int t = cold ? 1 : 0; t < rows; ++t) {
    const auto projection = proj + static_cast<int64_t>(t) * stride;
    const auto convolved = xbc + static_cast<int64_t>(t) * channels;
    const float raw = B2F(projection[di + channels + h]) + dt_bias[h];
    const float dt = fmaxf(raw > 20.f ? raw : logf(__expf(raw) + 1.f), tmin);
    const float decay = __expf(a[h] * dt), x = B2F(convolved[h * 64 + p]);
    float c[4];
    for (int j = 0; j < 4; ++j) {
      const float b = B2F(convolved[di + g * 128 + n + j]);
      c[j] = B2F(convolved[di + groups * 128 + g * 128 + n + j]);
      state[j] = __fmaf_rn(decay, state[j], (dt * b) * x);
      if (penultimate && t == rows - 2)
        penultimate[(static_cast<int64_t>(h) * 128 + n + j) * 64 + p] = state[j];
    }
    float acc = state[1] * c[1];
    acc = __fmaf_rn(state[0], c[0], acc);
    acc = __fmaf_rn(state[2], c[2], acc);
    acc = __fmaf_rn(state[3], c[3], acc);
    acc = WarpSum(acc);
    if ((threadIdx.x & 31) == 0)
      ypre[static_cast<int64_t>(t) * di + h * 64 + p] = __fmaf_rn(x, skip[h], acc);
  }
  for (int j = 0; j < 4; ++j)
    present[(static_cast<int64_t>(h) * 128 + n + j) * 64 + p] = state[j];
}

__global__ void MambaChunkGateKernel(const bf16* proj, int rows, int stride, int di, bf16* gate) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<int64_t>(rows) * di) gate[i] = proj[(i / di) * stride + i % di];
}

struct MambaChunkState : MambaStepState {
  using MambaStepState::MambaStepState;
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const auto shape = Dimensions(c.GetInput(0));
      Require(shape.size() == 2, "MambaChunk: proj must be [T,stride]");
      if (shape[0] == 1) return MambaStepState::ComputeV2(raw);
      CheckArity(c, 10, 5);
      const int rows = PositiveInt(shape[0], "MambaChunk: invalid T");
      Require(rows <= 4096 && hd == 64 && ds == 128, "MambaChunk: T<=4096, HD=64, DS=128 required");
      CheckTensor(c, 0, {rows, stride}, BF, "MambaChunk: invalid projection");
      const auto cw = Dimensions(c.GetInput(3));
      Require(cw.size() == 3 && cw[0] == channels && cw[1] == 1 && cw[2] == 4,
              "MambaChunk: convolution width4 required");
      CheckTensor(c, 1, {1, channels, 3}, BF, "MambaChunk: invalid convolution state");
      CheckTensor(c, 2, {1, heads, ds, hd}, F32, "MambaChunk: invalid SSM state");
      CheckTensor(c, 3, cw, BF, "MambaChunk: invalid convolution weights");
      CheckTensor(c, 4, {channels}, BF, "MambaChunk: invalid convolution bias");
      for (size_t i = 5; i < 8; ++i)
        CheckTensor(c, i, {heads}, F32, "MambaChunk: invalid per-head parameter");
      CheckTensor(c, 8, {di}, F32, "MambaChunk: invalid normalization weight");
      CheckTensor(c, 9, {1}, I32, "MambaChunk: position must be INT32[1]");
      auto y = c.GetOutput(0, Shape{rows, di});
      auto conv = c.GetOutput(1, Shape{1, channels, 3});
      auto state = c.GetOutput(2, Shape{1, heads, ds, hd});
      auto xbc = c.GetOutput(3, Shape{rows, channels});
      auto ypre = c.GetOutput(4, Shape{rows, di});
      const auto s = Stream(api, raw);
      // Separate storage prevents the all-token convolution from racing an aliased past.
      AsyncBuffer history(static_cast<size_t>(channels) * 3 * sizeof(bf16), s);
      auto error = cudaMemcpyAsync(history.data, In<bf16>(c, 1), static_cast<size_t>(channels) * 3 * sizeof(bf16),
                                   cudaMemcpyDeviceToDevice, s);
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
      MambaChunkConvKernel<<<dim3(CeilDiv(channels, 128), rows), 128, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, channels, 4, static_cast<const bf16*>(history.data),
          In<bf16>(c, 3), In<bf16>(c, 4), Out<bf16>(conv), Out<bf16>(xbc));
      MambaSsmKernel<128, 64, 2><<<heads, 128, 0, s>>>(
          In<bf16>(c, 0), di, channels, heads / groups, groups, Out<bf16>(xbc),
          In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
          tmin, In<int32_t>(c, 9), Out<float>(state), Out<float>(ypre));
      MambaChunkSourceKernel<<<dim3(heads, 16), 128, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, channels, heads / groups, groups, Out<bf16>(xbc),
          In<float>(c, 2), In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
          tmin, In<int32_t>(c, 9), Out<float>(state), Out<float>(ypre));
      AsyncBuffer gate(static_cast<size_t>(rows) * di * sizeof(bf16), s);
      MambaChunkGateKernel<<<CeilDiv(rows * di, 256), 256, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, static_cast<bf16*>(gate.data));
      // Keep exactly the qualified one-token reduction and BF16 store boundaries.
      SourceRmsKernel<true><<<rows * groups, RmsThreads(di / groups, groups), 0, s>>>(
          static_cast<bf16*>(gate.data), Out<float>(ypre), In<float>(c, 8),
          di / groups, eps, Out<bf16>(y), groups);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

__global__ void MambaWindowConvKernel(
    const bf16* proj, int rows, int stride, int di, int channels,
    const bf16* past, bf16* penultimate) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  for (int j = 0; j < 3; ++j) {
    const int source = rows - 4 + j;
    penultimate[c * 3 + j] = source < 0 ? past[c * 3 + source + 3]
        : proj[static_cast<int64_t>(source) * stride + di + c];
  }
}

struct MambaWindowState : MambaStepState {
  using MambaStepState::MambaStepState;
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      CheckArity(c, 10, 5);
      const auto shape = Dimensions(c.GetInput(0));
      Require(shape.size() == 2, "MambaWindow: projection must be [T,stride]");
      const int rows = PositiveInt(shape[0], "MambaWindow: invalid T");
      Require(rows <= 4096 && hd == 64 && ds == 128, "MambaWindow: unsupported dimensions");
      CheckTensor(c, 0, {rows, stride}, BF, "MambaWindow: invalid projection");
      CheckTensor(c, 1, {2, 1, channels, 3}, BF, "MambaWindow: invalid convolution window");
      CheckTensor(c, 2, {2, 1, heads, ds, hd}, F32, "MambaWindow: invalid recurrent window");
      CheckTensor(c, 3, {channels, 1, 4}, BF, "MambaWindow: convolution width must be four");
      CheckTensor(c, 4, {channels}, BF, "MambaWindow: invalid convolution bias");
      for (size_t i = 5; i < 8; ++i)
        CheckTensor(c, i, {heads}, F32, "MambaWindow: invalid per-head parameter");
      CheckTensor(c, 8, {di}, F32, "MambaWindow: invalid normalization weight");
      CheckTensor(c, 9, {1}, I32, "MambaWindow: position must be INT32[1]");
      auto y = c.GetOutput(0, {rows, di});
      auto conv = c.GetOutput(1, {2, 1, channels, 3});
      auto state = c.GetOutput(2, {2, 1, heads, ds, hd});
      auto xbc = c.GetOutput(3, {rows, channels});
      auto ypre = c.GetOutput(4, {rows, di});
      const auto s = Stream(api, raw);
      const int64_t state_size = static_cast<int64_t>(heads) * ds * hd;
      AsyncBuffer history(static_cast<size_t>(channels) * 3 * sizeof(bf16), s);
      auto copy = [&](void* dst, const void* src, size_t bytes) {
        const auto error = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
      };
      copy(history.data, In<bf16>(c, 1) + channels * 3, channels * 3 * sizeof(bf16));
      auto past = static_cast<const bf16*>(history.data);
      auto final_state = Out<float>(state) + state_size;
      auto past_state = In<float>(c, 2) + state_size;
      copy(Out<float>(state), past_state, state_size * sizeof(float));
      MambaWindowConvKernel<<<CeilDiv(channels, 128), 128, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, channels, past, Out<bf16>(conv));
      MambaChunkConvKernel<<<dim3(CeilDiv(channels, 128), rows), 128, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, channels, 4, past, In<bf16>(c, 3),
          In<bf16>(c, 4), Out<bf16>(conv) + channels * 3, Out<bf16>(xbc));
      MambaSsmKernel<128, 64, 2><<<heads, 128, 0, s>>>(
          In<bf16>(c, 0), di, channels, heads / groups, groups, Out<bf16>(xbc),
          past_state, In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
          tmin, In<int32_t>(c, 9), final_state, Out<float>(ypre));
      MambaChunkSourceKernel<<<dim3(heads, 16), 128, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, channels, heads / groups, groups, Out<bf16>(xbc),
          past_state, In<float>(c, 5), In<float>(c, 6), In<float>(c, 7),
          tmin, In<int32_t>(c, 9), final_state, Out<float>(ypre), Out<float>(state));
      AsyncBuffer gate(static_cast<size_t>(rows) * di * sizeof(bf16), s);
      MambaChunkGateKernel<<<CeilDiv(rows * di, 256), 256, 0, s>>>(
          In<bf16>(c, 0), rows, stride, di, static_cast<bf16*>(gate.data));
      SourceRmsKernel<true><<<rows * groups, RmsThreads(di / groups, groups), 0, s>>>(
          static_cast<bf16*>(gate.data), Out<float>(ypre), In<float>(c, 8),
          di / groups, eps, Out<bf16>(y), groups);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

__global__ void ExpertCohortKernel(const int32_t* indices, int rows, int topk,
                                   int* ranks, int* counts) {
  const int expert = blockIdx.x;
  if (threadIdx.x != 0) return;
  int count = 0;
  for (int i = 0; i < rows * topk; ++i)
    if (indices[i] == expert) ranks[i] = count++;
  counts[expert] = count;
}

__global__ void ExpertGatherChunkKernel(const bf16* x, const int32_t* indices,
                                        const int* ranks, const int* offsets,
                                        int rows, int topk, int hidden, bf16* packed) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(rows) * topk * hidden) return;
  const int route = i / hidden, column = i % hidden;
  const int slot = offsets[indices[route]] + ranks[route];
  packed[static_cast<int64_t>(slot) * hidden + column] =
      x[static_cast<int64_t>(route / topk) * hidden + column];
}

__global__ void ExpertOffsetsKernel(const int* counts, int experts, int* offsets) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= experts) return;
  int sum = 0;
  for (int j = 0; j < e; ++j) sum += counts[j];
  offsets[e] = sum;
}

__global__ void ExpertScatterActivationKernel(
    const bf16* packed, const int32_t* indices, const int* ranks, const int* offsets,
    int routes, int intermediate, bf16* activated) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(routes) * intermediate) return;
  const int route = i / intermediate, column = i % intermediate;
  const int slot = offsets[indices[route]] + ranks[route];
  activated[i] = packed[static_cast<int64_t>(slot) * intermediate + column];
}

__global__ void ExpertFinalizeChunkKernel(const bf16* down, const int32_t* indices,
                                         const float* probs, const int* ranks, const int* offsets,
                                         int rows, int topk, int hidden, bf16* y, float* part) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(rows) * hidden) return;
  const int row = i / hidden, column = i % hidden;
  float lanes[4] = {};
  for (int k = 0; k < topk; ++k) {
    const int route = row * topk + k, slot = offsets[indices[route]] + ranks[route];
    const float value = B2F(down[static_cast<int64_t>(slot) * hidden + column]);
    part[static_cast<int64_t>(route) * hidden + column] = value;
    lanes[k % 4] += probs[route] * value;
  }
  y[i] = __float2bfloat16(((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
}

struct CohortGemm {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  std::vector<cublasLtMatrixLayout_t> inputs, outputs;
  std::vector<size_t> workspace_sizes;
  int hidden = 0, width = 0;
  static constexpr size_t workspace_limit = 64 * 1024 * 1024;
  ~CohortGemm() {
    for (auto layout : inputs) if (layout) cublasLtMatrixLayoutDestroy(layout);
    for (auto layout : outputs) if (layout) cublasLtMatrixLayoutDestroy(layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (handle) cublasLtDestroy(handle);
  }
  void Layout(int rows) {
    if (!inputs[rows])
      CheckBlas(cublasLtMatrixLayoutCreate(&inputs[rows], CUDA_R_16BF, hidden, rows, hidden));
    if (!outputs[rows])
      CheckBlas(cublasLtMatrixLayoutCreate(&outputs[rows], CUDA_R_16BF, width, rows, width));
  }
  void Initialize(int k, int n) {
    hidden = k; width = n;
    inputs.resize(4097, nullptr);
    outputs.resize(4097, nullptr);
    workspace_sizes.resize(4097, std::numeric_limits<size_t>::max());
    CheckBlas(cublasLtCreate(&handle));
    CheckBlas(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transpose = CUBLAS_OP_T;
    CheckBlas(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &transpose, sizeof(transpose)));
    CheckBlas(cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_16BF, hidden, width, hidden));
    CheckBlas(cublasLtMatmulPreferenceCreate(&preference));
    CheckBlas(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_limit, sizeof(workspace_limit)));
    const uint32_t alignment = 16;
    for (auto attribute : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      CheckBlas(cublasLtMatmulPreferenceSetAttribute(preference, attribute, &alignment, sizeof(alignment)));
    Layout(1);
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    // Keep the one-row reduction configuration, not the multi-row heuristic's different split-K.
    CheckBlas(cublasLtMatmulAlgoGetHeuristic(handle, operation, weight_layout, inputs[1], outputs[1],
                                           outputs[1], preference, 1, &result, &count));
    Require(count == 1 && result.state == CUBLAS_STATUS_SUCCESS,
            "MoeCohortBF16: no source-row Lt algorithm");
    algorithm = result.algo;
  }
  size_t Workspace(int rows) {
    Require(rows > 0 && rows <= 4096, "MoeCohortBF16: invalid cohort size");
    if (workspace_sizes[rows] == std::numeric_limits<size_t>::max()) {
      Layout(rows);
      cublasLtMatmulHeuristicResult_t result{};
      CheckBlas(cublasLtMatmulAlgoCheck(handle, operation, weight_layout, inputs[rows],
                                       outputs[rows], outputs[rows], &algorithm, &result));
      Require(result.state == CUBLAS_STATUS_SUCCESS && result.workspaceSize <= workspace_limit,
              "MoeCohortBF16: unsupported cohort algorithm/workspace");
      workspace_sizes[rows] = result.workspaceSize;
    }
    return workspace_sizes[rows];
  }
  void Run(const bf16* weight, const bf16* input, bf16* output, int rows,
           void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    const float alpha = 1.f, beta = 0.f;
    CheckBlas(cublasLtMatmul(handle, operation, &alpha, weight, weight_layout,
                             input, inputs[rows], &beta, output, outputs[rows],
                             output, outputs[rows], &algorithm, workspace, workspace_bytes, stream));
  }
};

struct MoeChunkState : MoeDecodeState<false> {
  bool cohort_gemm = false;
  std::unique_ptr<CohortGemm> up_plan, down_plan;
  MoeChunkState(const OrtApi& a, const OrtKernelInfo* info) : MoeDecodeState<false>(a, info) {}
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const auto xs = Dimensions(c.GetInput(0)), ws = Dimensions(c.GetInput(3));
      Require(xs.size() == 2, "MoeChunkBF16: x must be [T,H]");
      if (xs[0] == 1) return MoeDecodeState<false>::ComputeV2(raw);
      CheckArity(c, 5, 3);
      const int rows = PositiveInt(xs[0], "MoeChunkBF16: invalid rows");
      const int hidden = PositiveInt(xs[1], "MoeChunkBF16: invalid hidden");
      Require(rows <= 4096 && ws.size() == 3 && source_order && topk == 6,
              "MoeChunkBF16: T<=4096 and six source-ordered routes required");
      const int experts = PositiveInt(ws[0], "MoeChunkBF16: invalid experts");
      const int intermediate = PositiveInt(ws[1], "MoeChunkBF16: invalid intermediate");
      CheckTensor(c, 0, xs, BF, "MoeChunkBF16: invalid BF16 activation");
      CheckTensor(c, 1, {rows, topk}, I32, "MoeChunkBF16: invalid indices");
      CheckTensor(c, 2, {rows, topk}, F32, "MoeChunkBF16: invalid probabilities");
      CheckTensor(c, 3, {experts, intermediate, hidden}, BF, "MoeChunkBF16: invalid up weights");
      CheckTensor(c, 4, {experts, hidden, intermediate}, BF, "MoeChunkBF16: invalid down weights");
      auto y = c.GetOutput(0, Shape{rows, hidden});
      auto h = c.GetOutput(1, Shape{rows * topk, intermediate});
      auto part = c.GetOutput(2, Shape{rows, topk, hidden});
      const auto s = Stream(api, raw);
      std::lock_guard<std::mutex> lock(blas_mutex);
      AsyncBuffer metadata(static_cast<size_t>(rows * topk + 2 * experts) * sizeof(int), s);
      auto ranks = static_cast<int*>(metadata.data), counts = ranks + rows * topk, offsets = counts + experts;
      ExpertCohortKernel<<<experts, 1, 0, s>>>(In<int32_t>(c, 1), rows, topk, ranks, counts);
      std::vector<int> host_counts(experts), host_offsets(experts);
      auto error = cudaMemcpyAsync(host_counts.data(), counts, experts * sizeof(int), cudaMemcpyDeviceToHost, s);
      if (error == cudaSuccess) error = cudaStreamSynchronize(s);
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
      int total = 0;
      for (int e = 0; e < experts; ++e) {
        host_offsets[e] = total;
        Require(host_counts[e] >= 0 && host_counts[e] <= rows, "MoeChunkBF16: duplicate or invalid routes");
        total += host_counts[e];
      }
      Require(total == rows * topk, "MoeChunkBF16: expert indices out of range");
      ExpertOffsetsKernel<<<CeilDiv(experts, 128), 128, 0, s>>>(counts, experts, offsets);
      AsyncBuffer packed(static_cast<size_t>(total) * hidden * 2 * sizeof(bf16), s);
      auto input = static_cast<bf16*>(packed.data), down = input + static_cast<int64_t>(total) * hidden;
      AsyncBuffer activation(static_cast<size_t>(total) * intermediate * sizeof(bf16), s);
      auto activated = static_cast<bf16*>(activation.data);
      ExpertGatherChunkKernel<<<CeilDiv(total * hidden, 256), 256, 0, s>>>(
          In<bf16>(c, 0), In<int32_t>(c, 1), ranks, offsets, rows, topk, hidden, input);
      const bool use_cohorts = cohort_gemm && hidden == 2688 && intermediate == 1856;
      size_t workspace_bytes = 0;
      if (use_cohorts) {
        if (!up_plan) {
          auto up = std::make_unique<CohortGemm>(), down = std::make_unique<CohortGemm>();
          up->Initialize(hidden, intermediate);
          down->Initialize(intermediate, hidden);
          up_plan = std::move(up); down_plan = std::move(down);
        }
        for (auto count : host_counts) if (count > 1) {
          workspace_bytes = std::max(workspace_bytes, up_plan->Workspace(count));
          workspace_bytes = std::max(workspace_bytes, down_plan->Workspace(count));
        }
      }
      std::unique_ptr<AsyncBuffer> workspace;
      if (workspace_bytes) workspace = std::make_unique<AsyncBuffer>(workspace_bytes, s);
      CheckBlas(cublasSetStream(blas, s));
      const float alpha = 1.f, beta = 0.f;
      for (int e = 0; e < experts; ++e) {
        if (use_cohorts && host_counts[e] > 1) {
          up_plan->Run(In<bf16>(c, 3) + static_cast<int64_t>(e) * intermediate * hidden,
                       input + static_cast<int64_t>(host_offsets[e]) * hidden,
                       activated + static_cast<int64_t>(host_offsets[e]) * intermediate,
                       host_counts[e], workspace ? workspace->data : nullptr, workspace_bytes, s);
          continue;
        }
        for (int row = 0; row < host_counts[e]; ++row)
        CheckBlas(cublasGemmEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, intermediate, 1, hidden, &alpha,
            In<bf16>(c, 3) + static_cast<int64_t>(e) * intermediate * hidden, CUDA_R_16BF, hidden,
            input + static_cast<int64_t>(host_offsets[e] + row) * hidden, CUDA_R_16BF, hidden, &beta,
            activated + static_cast<int64_t>(host_offsets[e] + row) * intermediate, CUDA_R_16BF, intermediate,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      ReluSquaredKernel<<<CeilDiv(total * intermediate, 256), 256, 0, s>>>(activated, total * intermediate);
      for (int e = 0; e < experts; ++e) {
        if (use_cohorts && host_counts[e] > 1) {
          down_plan->Run(In<bf16>(c, 4) + static_cast<int64_t>(e) * intermediate * hidden,
                         activated + static_cast<int64_t>(host_offsets[e]) * intermediate,
                         down + static_cast<int64_t>(host_offsets[e]) * hidden,
                         host_counts[e], workspace ? workspace->data : nullptr, workspace_bytes, s);
          continue;
        }
        for (int row = 0; row < host_counts[e]; ++row)
        CheckBlas(cublasGemmEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, hidden, 1, intermediate, &alpha,
            In<bf16>(c, 4) + static_cast<int64_t>(e) * intermediate * hidden, CUDA_R_16BF, intermediate,
            activated + static_cast<int64_t>(host_offsets[e] + row) * intermediate, CUDA_R_16BF, intermediate, &beta,
            down + static_cast<int64_t>(host_offsets[e] + row) * hidden, CUDA_R_16BF, hidden,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      ExpertScatterActivationKernel<<<CeilDiv(total * intermediate, 256), 256, 0, s>>>(
          activated, In<int32_t>(c, 1), ranks, offsets, total, intermediate, Out<bf16>(h));
      ExpertFinalizeChunkKernel<<<CeilDiv(rows * hidden, 256), 256, 0, s>>>(
          down, In<int32_t>(c, 1), In<float>(c, 2), ranks, offsets, rows, topk, hidden,
          Out<bf16>(y), Out<float>(part));
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};

struct MoeCohortState : MoeChunkState {
  MoeCohortState(const OrtApi& a, const OrtKernelInfo* info) : MoeChunkState(a, info) {
    cohort_gemm = true;
  }
};

struct MoeDeviceState : MoeCohortState {
  MoeDeviceState(const OrtApi& a, const OrtKernelInfo* info) : MoeCohortState(a, info) {
    device_decode = true;
  }
};

__global__ void VerifyExpertActivationKernel(const float* partial, bf16* activated) {
  constexpr int count = 6 * 1856;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 2 * count) {
    const int base = (i / count) * 2 * count + i % count;
    const float r = fmaxf(RoundBF(partial[base] + partial[base + count]), 0.f);
    activated[i] = __float2bfloat16(r * r);
  }
}

struct MoeVerifyState : MoeDeviceState {
  using MoeDeviceState::MoeDeviceState;
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext c(raw);
      const auto xs = Dimensions(c.GetInput(0));
      CheckArity(c, 5, 3);
      const auto ws = Dimensions(c.GetInput(3));
      if (xs != Shape{2, 2688} || ws.size() != 3 || ws[1] != 1856 || ws[2] != 2688)
        return MoeDeviceState::ComputeV2(raw);
      Require(source_order && topk == 6, "MoeVerify: requires source-ordered six-expert routing");
      const int experts = PositiveInt(ws[0], "MoeVerify: invalid experts");
      Require(experts >= 6, "MoeVerify: requires at least six experts");
      CheckTensor(c, 0, xs, BF, "MoeVerify: invalid activation");
      CheckTensor(c, 1, {2, 6}, I32, "MoeVerify: invalid indices");
      CheckTensor(c, 2, {2, 6}, F32, "MoeVerify: invalid probabilities");
      CheckTensor(c, 3, ws, BF, "MoeVerify: invalid up weights");
      CheckTensor(c, 4, {experts, 2688, 1856}, BF, "MoeVerify: invalid down weights");
      auto y = c.GetOutput(0, {2, 2688});
      auto h = c.GetOutput(1, {12, 1856});
      auto part = c.GetOutput(2, {2, 6, 2688});
      const auto s = Stream(api, raw);
      AsyncBuffer scratch(2 * 2 * 6 * 1856 * sizeof(float), s);
      DeviceExpertTiledMmaKernel<true, 2><<<dim3(1856 / 64, 12, 2), 128, 0, s>>>(
          In<bf16>(c, 0), In<int32_t>(c, 1), In<bf16>(c, 3), experts,
          static_cast<float*>(scratch.data));
      VerifyExpertActivationKernel<<<CeilDiv(12 * 1856, 256), 256, 0, s>>>(
          static_cast<const float*>(scratch.data), Out<bf16>(h));
      DeviceExpertTiledMmaKernel<false, 2><<<dim3(2688 / 64, 12), 128, 0, s>>>(
          Out<bf16>(h), In<int32_t>(c, 1), In<bf16>(c, 4), experts, Out<float>(part));
      for (int row = 0; row < 2; ++row)
        MoeFinalizeKernel<<<CeilDiv(CeilDiv(2688, 4), 64), 64, 0, s>>>(
            Out<float>(part) + row * 6 * 2688, In<float>(c, 2) + row * 6,
            In<int32_t>(c, 1) + row * 6, 6, 2688, true, Out<bf16>(y) + row * 2688);
      CheckLaunch();
      return nullptr;
    } catch (const std::exception& e) { return ComputeError(api, e); }
  }
};
