// Gemma-4 MoE Expert GeGLU decode (T=1), custom RDNA4 HIP kernels.
// VT_GEMMA4_CUSTOM_EXPERT=1 to use from gemma4_moe.cpp.
// y = beta*y + alpha * Down(Gelu(Gate(x)) * Up(x))
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

__device__ inline float gelu_tanh(float g) {
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  return 0.5f * g * (1.f + tanhf(inner));
}

// Block-level float sum (kBlock == 256).
__device__ inline float BlockSum256(float v) {
  __shared__ float sm[256];
  const int tid = static_cast<int>(threadIdx.x);
  sm[tid] = v;
  __syncthreads();
#pragma unroll
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) sm[tid] += sm[tid + s];
    __syncthreads();
  }
  return sm[0];
}

// grid: I  — act[i] = gelu(dot(x,Wgu[i])) * dot(x,Wgu[I+i])
__global__ void ExpertActKernel(float* __restrict__ act, const __hip_bfloat16* __restrict__ x,
                                const __hip_bfloat16* __restrict__ w_gu, int I, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I) return;

  const __hip_bfloat16* grow = w_gu + static_cast<int64_t>(i) * H;
  const __hip_bfloat16* urow = w_gu + static_cast<int64_t>(I + i) * H;

  float gate = 0.f, up = 0.f;
  // Vectorize by 2 when H even
  const int tid = static_cast<int>(threadIdx.x);
  for (int h = tid; h < H; h += kBlock) {
    const float xv = __bfloat162float(x[h]);
    gate += xv * __bfloat162float(grow[h]);
    up += xv * __bfloat162float(urow[h]);
  }
  gate = BlockSum256(gate);
  up = BlockSum256(up);
  if (tid == 0) act[i] = gelu_tanh(gate) * up;
}

// grid: ceil(H / kBlock) — y[h] = beta*y[h] + alpha * dot(act, Wdn[h])
__global__ void ExpertDownKernel(__hip_bfloat16* __restrict__ y, const float* __restrict__ act,
                                 const __hip_bfloat16* __restrict__ w_dn, int I, int H, float alpha,
                                 float beta) {
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  const __hip_bfloat16* drow = w_dn + static_cast<int64_t>(h) * I;
  float acc = 0.f;
  int i = 0;
  for (; i + 3 < I; i += 4) {
    acc += act[i] * __bfloat162float(drow[i]);
    acc += act[i + 1] * __bfloat162float(drow[i + 1]);
    acc += act[i + 2] * __bfloat162float(drow[i + 2]);
    acc += act[i + 3] * __bfloat162float(drow[i + 3]);
  }
  for (; i < I; ++i) acc += act[i] * __bfloat162float(drow[i]);
  float v = alpha * acc;
  if (beta != 0.f) v += beta * __bfloat162float(y[h]);
  y[h] = __float2bfloat16(v);
}

struct ActBuf {
  float* p = nullptr;
  int cap = 0;
};

ActBuf& TlsAct() {
  static thread_local ActBuf b;
  return b;
}

bool EnsureAct(int I) {
  auto& b = TlsAct();
  if (b.cap >= I && b.p) return true;
  if (b.p) (void)hipFree(b.p);
  b.p = nullptr;
  b.cap = 0;
  if (hipMalloc(&b.p, static_cast<size_t>(I) * sizeof(float)) != hipSuccess) return false;
  b.cap = I;
  return true;
}

}  // namespace

// Returns false if launch/setup failed (caller falls back to hipBLAS path).
bool ExpertGeGLUBf16M1Rocm(Queue& q, void* y, const void* x, const void* w_gu, const void* w_dn,
                           int I, int H, float alpha, float beta) {
  if (!y || !x || !w_gu || !w_dn || I <= 0 || H <= 0) return false;
  if (!EnsureAct(I)) return false;

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  auto* act = TlsAct().p;

  ExpertActKernel<<<I, kBlock, 0, st>>>(
      act, static_cast<const __hip_bfloat16*>(x), static_cast<const __hip_bfloat16*>(w_gu), I, H);

  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertDownKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(y), act, static_cast<const __hip_bfloat16*>(w_dn), I, H, alpha,
      beta);

  return hipGetLastError() == hipSuccess;
}

// Top-k: sequential custom experts with alpha/beta mix into ysum.
bool ExpertGeGLUBf16TopKM1Rocm(Queue& q, void* ysum, const void* x, const void* const* w_gu,
                               const void* const* w_dn, const float* wts, int G, int I, int H) {
  if (G <= 0 || !ysum || !x || !w_gu || !w_dn || !wts) return false;
  for (int g = 0; g < G; ++g) {
    const float beta = (g == 0) ? 0.f : 1.f;
    if (!ExpertGeGLUBf16M1Rocm(q, ysum, x, w_gu[g], w_dn[g], I, H, wts[g], beta)) return false;
  }
  return true;
}

}  // namespace vt::rocm
