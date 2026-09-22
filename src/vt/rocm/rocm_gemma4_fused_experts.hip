// Fused top-k ExpertGeGLU for Gemma-4 MoE decode (T=1).
// Opt-in: VT_GEMMA4_FUSED_EXPERTS=1
// Two-phase multi-block: (1) act[i]=gelu(gate)*up  (2) ysum += w * act@Wdn
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdlib>

#include "vt/device.h"

namespace vllm {
namespace {

constexpr int kMaxK = 8;
constexpr int kBlock = 256;

__device__ inline float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// grid: (I, G)  — one block per (expert, intermediate index)
__global__ void ExpertActKernel(__hip_bfloat16* __restrict__ act,  // [G, I]
                                const __hip_bfloat16* __restrict__ x,  // [H]
                                const __hip_bfloat16* const* __restrict__ gu_ptrs, int I,
                                int H) {
  const int i = static_cast<int>(blockIdx.x);
  const int e = static_cast<int>(blockIdx.y);
  if (i >= I) return;
  const __hip_bfloat16* gu = gu_ptrs[e];
  const __hip_bfloat16* grow = gu + static_cast<int64_t>(i) * H;
  const __hip_bfloat16* urow = gu + static_cast<int64_t>(I + i) * H;

  float gate = 0.f, up = 0.f;
  for (int h = static_cast<int>(threadIdx.x); h < H; h += kBlock) {
    const float xv = __bfloat162float(x[h]);
    gate += xv * __bfloat162float(grow[h]);
    up += xv * __bfloat162float(urow[h]);
  }
  __shared__ float sg[256], su[256];
  sg[threadIdx.x] = gate;
  su[threadIdx.x] = up;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) {
      sg[threadIdx.x] += sg[threadIdx.x + s];
      su[threadIdx.x] += su[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    act[static_cast<int64_t>(e) * I + i] = __float2bfloat16(gelu_tanh(sg[0]) * su[0]);
  }
}

// grid: (H,) — accumulate all experts into ysum
__global__ void ExpertDownMixKernel(__hip_bfloat16* __restrict__ ysum,
                                    const __hip_bfloat16* __restrict__ act,  // [G,I]
                                    const __hip_bfloat16* const* __restrict__ dn_ptrs,
                                    const float* __restrict__ wts, int G, int I, int H) {
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  float acc = 0.f;
  for (int e = 0; e < G; ++e) {
    const __hip_bfloat16* drow = dn_ptrs[e] + static_cast<int64_t>(h) * I;
    const __hip_bfloat16* arow = act + static_cast<int64_t>(e) * I;
    float dot = 0.f;
    for (int i = 0; i < I; ++i) {
      dot += __bfloat162float(arow[i]) * __bfloat162float(drow[i]);
    }
    acc += wts[e] * dot;
  }
  ysum[h] = __float2bfloat16(acc);
}

int FusedEnabled() {
  static const int on = [] {
    if (const char* e = std::getenv("VT_GEMMA4_FUSED_EXPERTS")) return e[0] == '1' ? 1 : 0;
    return 0;
  }();
  return on;
}

struct BufTls {
  void** d_gu = nullptr;
  void** d_dn = nullptr;
  float* d_wts = nullptr;
  __hip_bfloat16* d_act = nullptr;
  int act_cap = 0;
};

BufTls& TLS() {
  static thread_local BufTls t;
  return t;
}

}  // namespace

bool RunGemma4FusedTopkExpertGeGLU(vt::Queue& q, void* ysum, const void* x,
                                   const uint16_t* const* gu_ptrs,
                                   const uint16_t* const* dn_ptrs, const float* wts, int G,
                                   int64_t I, int64_t H) {
  if (!FusedEnabled() || G <= 0 || G > kMaxK || I <= 0 || H <= 0) return false;
  if (!ysum || !x || !gu_ptrs || !dn_ptrs || !wts) return false;

  auto& tls = TLS();
  const size_t act_need = static_cast<size_t>(G) * static_cast<size_t>(I);
  if (tls.act_cap < static_cast<int>(act_need)) {
    if (tls.d_act) (void)hipFree(tls.d_act);
    if (hipMalloc(&tls.d_act, act_need * sizeof(__hip_bfloat16)) != hipSuccess) return false;
    tls.act_cap = static_cast<int>(act_need);
  }
  if (!tls.d_gu) {
    if (hipMalloc(&tls.d_gu, sizeof(void*) * kMaxK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_dn, sizeof(void*) * kMaxK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_wts, sizeof(float) * kMaxK) != hipSuccess) return false;
  }

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  const void* h_gu[kMaxK];
  const void* h_dn[kMaxK];
  for (int i = 0; i < G; ++i) {
    h_gu[i] = gu_ptrs[i];
    h_dn[i] = dn_ptrs[i];
  }
  if (hipMemcpyAsync(tls.d_gu, h_gu, sizeof(void*) * static_cast<size_t>(G),
                     hipMemcpyHostToDevice, st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_dn, h_dn, sizeof(void*) * static_cast<size_t>(G),
                     hipMemcpyHostToDevice, st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_wts, wts, sizeof(float) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;

  dim3 grid1(static_cast<unsigned>(I), static_cast<unsigned>(G));
  ExpertActKernel<<<grid1, kBlock, 0, st>>>(
      tls.d_act, static_cast<const __hip_bfloat16*>(x),
      reinterpret_cast<const __hip_bfloat16* const*>(tls.d_gu), static_cast<int>(I),
      static_cast<int>(H));

  const int grid2 = static_cast<int>((H + kBlock - 1) / kBlock);
  ExpertDownMixKernel<<<grid2, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(ysum), tls.d_act,
      reinterpret_cast<const __hip_bfloat16* const*>(tls.d_dn), tls.d_wts, G, static_cast<int>(I),
      static_cast<int>(H));
  return true;
}

}  // namespace vllm
