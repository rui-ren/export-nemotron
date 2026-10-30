void E6LaunchCB(CUfunction function, cudaStream_t stream, const bf16* history, float* cb, int length) {
  auto a_ptr = history+9216;
  auto b_ptr = history+8192;
  auto out_ptr = cb;
  int seqlen = length;
  int chunk_size = 128;
  int K = 128;
  int ngroups = 8;
  int stride_a_batch = 2654208;
  int stride_a_seqlen = 10368;
  int stride_a_head = 128;
  int stride_b_batch = 2654208;
  int stride_b_seqlen = 10368;
  int stride_b_head = 128;
  int stride_out_batch = 131072;
  int stride_out_chunk = 131072;
  int stride_out_head = 16384;
  int stride_outm = 128;
  int stride_seq_idx_batch = 0;
  int stride_seq_idx_seqlen = 0;
  void* unused_scratch = nullptr;
  void* arguments[] = {&a_ptr,&b_ptr,&out_ptr,&seqlen,&chunk_size,&K,&ngroups,&stride_a_batch,&stride_a_seqlen,&stride_a_head,&stride_b_batch,&stride_b_seqlen,&stride_b_head,&stride_out_batch,&stride_out_chunk,&stride_out_head,&stride_outm,&stride_seq_idx_batch,&stride_seq_idx_seqlen,&unused_scratch,&unused_scratch};
  CheckSourceDriver(cuLaunchKernel(function,kProfileCB_GRID,1,8,kProfileCB_THREADS,1,1,kProfileCB_SHARED,stream,arguments,nullptr));
}
