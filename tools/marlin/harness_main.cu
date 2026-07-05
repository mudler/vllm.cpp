// Correctness harness: feed vLLM's dumped repacked MoE-NVFP4 inputs into OUR
// vendored marlin_mm and dump C_ours for comparison against C_golden.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "core/scalar_type.hpp"
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_mm.h"

#define CK(x) do{ cudaError_t e=(x); if(e){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e));exit(1);} }while(0)

static std::vector<char> readf(const char* p){
  FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(1);}
  fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<char> b(n); if(fread(b.data(),1,n,f)!=(size_t)n){exit(1);} fclose(f); return b;
}
static void* dev(const std::vector<char>& h){ void* d; CK(cudaMalloc(&d,h.size())); CK(cudaMemcpy(d,h.data(),h.size(),cudaMemcpyHostToDevice)); return d; }

int main(){
  const char* D="/tmp/moe_dump";
  const int E=8,K=256,N=128,M=16,TOPK=2,BLK=16,sorted_len=152,ws_len=192;
  auto A=readf((std::string(D)+"/a.bin").c_str());
  auto B=readf((std::string(D)+"/b_q_weight.bin").c_str());
  auto BS=readf((std::string(D)+"/b_scales.bin").c_str());
  auto GS=readf((std::string(D)+"/global_scale.bin").c_str());
  auto WS=readf((std::string(D)+"/workspace.bin").c_str());
  auto SID=readf((std::string(D)+"/sorted_ids.bin").c_str());
  auto EID=readf((std::string(D)+"/expert_ids.bin").c_str());
  auto NP=readf((std::string(D)+"/num_pad.bin").c_str());
  auto TW=readf((std::string(D)+"/topk_w.bin").c_str());
  void *a=dev(A),*b=dev(B),*bs=dev(BS),*gs=dev(GS),*ws=dev(WS),*sid=dev(SID),*eid=dev(EID),*np=dev(NP),*tw=dev(TW);
  CK(cudaMemset(ws,0,WS.size()));  // reset reduction locks
  void* c; CK(cudaMalloc(&c,(size_t)M*TOPK*N*2)); CK(cudaMemset(c,0,(size_t)M*TOPK*N*2));
  float* ctmp; CK(cudaMalloc((void**)&ctmp,(size_t)N*sorted_len*sizeof(float)));

  int sms=-1; CK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0));
  const vllm::ScalarType at=vllm::kBFloat16, bt=vllm::kFE2M1f, ct=vllm::kBFloat16, st=vllm::kFE4M3fn;
  marlin_moe_wna16::marlin_mm(
     a,b,c,ctmp,nullptr,nullptr,bs,gs,nullptr,nullptr,nullptr,nullptr,
     sid,eid,np,tw, BLK, E, TOPK, /*mul_topk*/false, M,N,K, ws,
     at,bt,ct,st, /*has_bias*/false,/*act_order*/false,/*k_full*/true,/*has_zp*/false,
     /*num_groups*/K/16,/*group_size*/16, /*dev*/0, /*stream*/0, -1,-1, sms, 0,
     /*atomic_add*/false,/*fp32_reduce*/true,/*zp_float*/false);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<char> out((size_t)M*TOPK*N*2);
  CK(cudaMemcpy(out.data(),c,out.size(),cudaMemcpyDeviceToHost));
  FILE* f=fopen((std::string(D)+"/C_ours.bin").c_str(),"wb"); fwrite(out.data(),1,out.size(),f); fclose(f);
  printf("OK: ran marlin_mm, wrote C_ours.bin (%zu bytes)\n", out.size());
  return 0;
}
