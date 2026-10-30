__device__ __forceinline__ float NormSourceDiv(float a,float b) {
  float out;
  asm("div.full.f32 %0,%1,%2;":"=f"(out):"f"(a),"f"(b));
  return out;
}
__device__ __forceinline__ float NormSourceExp(float x) {
  float scaled=__fmul_rn(x,1.4426950408889634f),out;
  asm("ex2.approx.f32 %0,%1;":"=f"(out):"f"(scaled));
  return out;
}
__global__ void MambaNormSourceKernel(const bf16* proj,int stride,int di,float eps,
                                     const float* readout,const float* w,bf16* y) {
  const int row=blockIdx.x,group=blockIdx.y,tid=threadIdx.x;
  const int base=row*di+group*1024;
  float v[8];
#pragma unroll
  for(int k=0;k<8;++k) {
    int i=tid*8+k;
    float z=B2F(proj[row*stride+group*1024+i]);
    float act=__fmul_rn(z,NormSourceDiv(1.f,__fadd_rn(NormSourceExp(-z),1.f)));
    v[k]=__fmul_rn(MambaNormReadout(readout[base+i]),act);
  }
  float sum=(fmaf(v[0],v[0],__fmul_rn(v[1],v[1]))+
             fmaf(v[2],v[2],__fmul_rn(v[3],v[3])))+
            (fmaf(v[4],v[4],__fmul_rn(v[5],v[5]))+
             fmaf(v[6],v[6],__fmul_rn(v[7],v[7])));
#pragma unroll
  for(int delta=16;delta;delta/=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta);
  __shared__ float partial[4];
  if((tid&31)==0) partial[tid>>5]=sum;
  __syncthreads();
  sum=partial[tid&3];
  sum+=__shfl_xor_sync(0xffffffff,sum,2);
  sum+=__shfl_xor_sync(0xffffffff,sum,1);
  float denominator=__fadd_rn(__fmul_rn(sum,1.f/1024.f),eps),root;
  asm("sqrt.approx.ftz.f32 %0,%1;":"=f"(root):"f"(denominator));
  float rstd=NormSourceDiv(1.f,root);
#pragma unroll
  for(int k=0;k<8;++k) {
    int i=tid*8+k;
    y[base+i]=__float2bfloat16(__fmul_rn(__fmul_rn(v[k],rstd),w[group*1024+i]));
  }
}
void LaunchMambaNormSource(cudaStream_t s,int rows,int groups,const bf16* proj,
                           int stride,int di,int group_size,float eps,
                           const float* readout,const float* w,bf16* y) {
  if(group_size==1024)
    MambaNormSourceKernel<<<dim3(rows,groups),128,0,s>>>(proj,stride,di,eps,readout,w,y);
  else
    MambaNormKernel<<<dim3(rows,groups),256,0,s>>>(proj,stride,di,group_size,eps,readout,w,y);
}
