// Original pinned mamba_ssm arithmetic kernels. The arrays contain executable
// code, not model weights, fixtures, output values, or sequence-specific state.
#include "cb_cubin.h"
#include "readout_cubin.h"
#include "cb_launch.inc"
#include "readout_launch.inc"

struct E6ParallelModules {
  CUmodule cb_module, readout_module;
  CUfunction cb, readout;
  E6ParallelModules() {
    CheckSourceDriver(cuModuleLoadData(&cb_module,e6_cb_cubin));
    CheckSourceDriver(cuModuleLoadData(&readout_module,e6_readout_cubin));
    CheckSourceDriver(cuModuleGetFunction(&cb,cb_module,"_bmm_chunk_fwd_kernel"));
    CheckSourceDriver(cuModuleGetFunction(&readout,readout_module,"_chunk_scan_fwd_kernel"));
    int bytes = 0;
    CheckSourceDriver(cuFuncGetAttribute(&bytes,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,cb));
    CheckSourceDriver(cuFuncSetAttribute(cb,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,kProfileCB_SHARED));
    CheckSourceDriver(cuFuncSetAttribute(readout,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,kProfileREADOUT_SHARED));
  }
};

E6ParallelModules& E6Modules() {
  thread_local std::vector<std::pair<CUcontext,E6ParallelModules*>> entries;
  CUcontext context = nullptr;
  CheckSourceDriver(cuCtxGetCurrent(&context));
  for (auto& e:entries) if (e.first==context) return *e.second;
  auto* value = new E6ParallelModules;
  entries.emplace_back(context,value);
  return *value;
}

__global__ void E6CaptureCoefficients(const float* cb,const float* dt,const float* cs,
                                    float* coefficients,float* packed,int start,int rows) {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  if (index < 128*512) {
    int h=index/512, item=index%512, part=item/256, j=item%256;
    packed[index] = j<128 ? (part ? cs[h*128+j] : dt[h*128+j]) : 0.f;
  }
  if (index < rows*128*256) {
    int j=index%256, h=(index/256)%128, t=start+index/(128*256);
    float value=0.f;
    if (j<=t && j<128) {
      float decay=__expf(fminf(cs[h*128+t]-cs[h*128+j],0.f));
      value=__fmul_rn(__fmul_rn(cb[(h/16)*16384+t*128+j],decay),dt[h*128+j]);
    }
    coefficients[index]=B2F(__float2bfloat16(value));
  }
}

__global__ void E6GatherCoefficients(const float* source,bf16* output,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<count) {
    int j=i%128, row=i/128;
    output[i]=__float2bfloat16(source[row*256+j]);
  }
}

__global__ void E6CachedDiagnostics(const bf16* proj,const float* bias,const float* A,
                                    float* packed,int rows) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=128*512) return;
  int h=i/512, part=(i%512)/256, j=i%256;
  float value=0.f;
  if (j<rows) {
    float raw=B2F(proj[j*18560+18432+h])+bias[h];
    float dt=raw<=20.f ? logf(__expf(raw)+1.f):raw;
    value=part ? __fmul_rn(A[h],dt):dt;
  }
  packed[i]=value;
}

void E6ParallelReadout(cudaStream_t stream,const bf16* history,const float* initial,
                       const float* bias,const float* A,const float* D,float* output,
                       float* coefficients,float* packed,int start,int rows,float tmin,
                       float* cb_debug=nullptr) {
  auto& source = SourceSsdForContext();
  auto& parallel = E6Modules();
  float* scratch=nullptr;
  constexpr int floats=32768+131072+1048576;
  if (cudaMallocAsync(&scratch,floats*sizeof(float)+256*sizeof(bf16),stream)!=cudaSuccess)
    throw std::runtime_error("E6 parallel SSD scratch allocation failed");
  float* dt=scratch;
  float* cs=dt+16384;
  float* cb=cs+16384;
  float* full_output=cb+131072;
  bf16* rounded_bias=reinterpret_cast<bf16*>(scratch+floats);
  bf16* rounded_D=rounded_bias+128;
  SourceBiasBf16<<<1,128,0,stream>>>(bias,rounded_bias);
  SourceBiasBf16<<<1,128,0,stream>>>(D,rounded_D);
  const bf16* raw=history+10240;
  int length=start+rows,heads=128,chunk=128,history_batch=2654208,history_row=10368;
  int batch_stride=16384,chunk_stride=128,head_stride=128;
  float tmax=INFINITY;
  void* unused=nullptr;
  void* args[]={&raw,&A,&rounded_bias,&dt,&cs,&length,&heads,&chunk,&tmin,&tmax,
                &history_batch,&history_row,&batch_stride,&chunk_stride,&head_stride,
                &batch_stride,&chunk_stride,&head_stride,&unused,&unused};
  CheckSourceDriver(cuLaunchKernel(source.cumsum,1,1,source.cumsum_grid,source.cumsum_threads,
                                  1,1,source.cumsum_shared,stream,args,nullptr));
  E6LaunchCB(parallel.cb,stream,history,cb,length);
  E6LaunchREADOUT(parallel.readout,stream,history,cb,dt,cs,initial,rounded_D,full_output,length);
  cudaMemcpyAsync(output,full_output+start*8192,rows*8192*sizeof(float),cudaMemcpyDeviceToDevice,stream);
  int count=std::max(rows*128*256,128*512);
  E6CaptureCoefficients<<<(count+255)/256,256,0,stream>>>(
      cb,dt,cs,coefficients,packed,start,rows);
  if (cb_debug)
    cudaMemcpyAsync(cb_debug,cb,131072*sizeof(float),cudaMemcpyDeviceToDevice,stream);
  cudaFreeAsync(scratch,stream);
}
