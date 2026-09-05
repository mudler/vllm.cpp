// Fast FP8-E4M3FN weight × BF16 activation kernels for Gemma-4 MoE.
// Channel scales BF16 per output row: y[m,n] = alpha * scale[n] * dot(x[m], dequant(W[n])) + beta*y
//
// M=1: fused Expert GeGLU (decode).
// M>1: weight stays FP8 — one dequant of W row into LDS, then all M acts (prefill batch).
// No host/device expand of full expert packs to BF16.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <vector>

#include <cstdint>
#include <cstdio>

#include "vt/device.h"
#include "vt/dtype.h"

namespace vt::rocm {
namespace {

constexpr int kBlock = 256;

__device__ inline float F8E4M3ToF32(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1U;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFU;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7U;
  const float sm = sign ? -1.0f : 1.0f;
  if (exp == 0xFU && mant == 0x7U) return 0.f;
  if (exp == 0U) return sm * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sm * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

// Shared 256-entry FP8 E4M3 LUT — much cheaper than ldexpf per weight load.
__device__ inline void FillF8Lut(float lut[256]) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < 256) lut[tid] = F8E4M3ToF32(static_cast<uint8_t>(tid));
  __syncthreads();
}

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

// Fewer barriers than BlockSum256 — warp reduce then 8-way (blockDim=256).
__device__ inline float WarpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
  return v;
}
__device__ inline float BlockSumFast(float v) {
  __shared__ float wsum[8];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = WarpSum(v);
  if (lane == 0) wsum[warp] = v;
  __syncthreads();
  // Only warp 0 reduces the 8 warp sums; publish to wsum[0] for all warps.
  if (warp == 0) {
    v = (lane < 8) ? wsum[lane] : 0.f;
    v = WarpSum(v);
    if (lane == 0) wsum[0] = v;
  }
  __syncthreads();
  return wsum[0];
}

__device__ inline float gelu_tanh(float g) {
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  return 0.5f * g * (1.f + tanhf(inner));
}

// Dot x_bf16[K] with pre-dequantized float weight row w_f[K].
__device__ inline float DotBf16xFloat(const __hip_bfloat16* __restrict__ xrow,
                                      const float* __restrict__ w_f, int K) {
  float acc = 0.f;
  const int tid = static_cast<int>(threadIdx.x);
  const int K4 = K & ~3;
  for (int k = tid * 4; k < K4; k += kBlock * 4) {
    acc += __bfloat162float(xrow[k + 0]) * w_f[k + 0];
    acc += __bfloat162float(xrow[k + 1]) * w_f[k + 1];
    acc += __bfloat162float(xrow[k + 2]) * w_f[k + 2];
    acc += __bfloat162float(xrow[k + 3]) * w_f[k + 3];
  }
  for (int k = K4 + tid; k < K; k += kBlock) {
    acc += __bfloat162float(xrow[k]) * w_f[k];
  }
  return BlockSum256(acc);
}

// M=1 GEMV: grid N, shmem K floats for x. FP8 via shared LUT.
__global__ void Fp8ChannelGemvFastKernel(__hip_bfloat16* __restrict__ y,
                                         const __hip_bfloat16* __restrict__ x,
                                         const uint8_t* __restrict__ W,  // [N,K]
                                         const __hip_bfloat16* __restrict__ scale, int N, int K,
                                         float alpha, float beta) {
  __shared__ float lut[256];
  FillF8Lut(lut);
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  for (int k = tid; k < K; k += kBlock) x_cache[k] = __bfloat162float(x[k]);
  __syncthreads();

  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;
  const uint8_t* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  float acc = 0.f;
  const int K16 = K & ~15;
  for (int k = tid * 16; k < K16; k += kBlock * 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) acc += x_cache[k + t] * lut[wrow[k + t]];
  }
  for (int k = K16 + tid; k < K; k += kBlock) acc += x_cache[k] * lut[wrow[k]];
  acc = BlockSum256(acc);
  if (tid == 0) {
    const float s = __bfloat162float(scale[n]);
    float v = alpha * s * acc;
    if (beta != 0.f) v += beta * __bfloat162float(y[n]);
    y[n] = __float2bfloat16(v);
  }
}

// M>=1 GEMM: y[M,N] = alpha * (x[M,K] @ W[N,K]^T) * scale[N] + beta * y
// grid = N. Dequant weight row once into LDS; stream all M activations.
// shmem = K floats (weight row). Weights stay FP8 in global memory.
__global__ void Fp8ChannelGemmMKernel(__hip_bfloat16* __restrict__ y,  // [M,N]
                                      const __hip_bfloat16* __restrict__ x,  // [M,K]
                                      const uint8_t* __restrict__ W,         // [N,K]
                                      const __hip_bfloat16* __restrict__ scale, int M, int N,
                                      int K, float alpha, float beta) {
  extern __shared__ float smem[];
  float* w_f = smem;
  const int tid = static_cast<int>(threadIdx.x);
  const int n = static_cast<int>(blockIdx.x);
  if (n >= N) return;

  __shared__ float lut[256];
  // Fill lut once (all threads); then dequant W row via lut into w_f.
  {
    if (tid < 256) lut[tid] = F8E4M3ToF32(static_cast<uint8_t>(tid));
    __syncthreads();
  }
  const uint8_t* wrow = W + static_cast<int64_t>(n) * static_cast<int64_t>(K);
  for (int k = tid; k < K; k += kBlock) w_f[k] = lut[wrow[k]];
  __syncthreads();

  const float s = __bfloat162float(scale[n]);
  for (int m = 0; m < M; ++m) {
    const __hip_bfloat16* xrow = x + static_cast<int64_t>(m) * static_cast<int64_t>(K);
    float acc = DotBf16xFloat(xrow, w_f, K);
    if (tid == 0) {
      __hip_bfloat16* ymn = y + static_cast<int64_t>(m) * static_cast<int64_t>(N) + n;
      float v = alpha * s * acc;
      if (beta != 0.f) v += beta * __bfloat162float(*ymn);
      *ymn = __float2bfloat16(v);
    }
    // ensure all threads done before next m reuses reduction sm in DotBf16xFloat
    __syncthreads();
  }
}

// Fused gate|up act T=1: grid = I. LUT + optional global x_f32.
__global__ void ExpertFp8ActKernel(float* __restrict__ act, const __hip_bfloat16* __restrict__ x,
                                   const uint8_t* __restrict__ w_gu,
                                   const __hip_bfloat16* __restrict__ s_gu, int I, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I) return;
  __shared__ float lut[256];
  FillF8Lut(lut);
  extern __shared__ float smem[];
  float* x_cache = smem;
  const int tid = static_cast<int>(threadIdx.x);
  for (int h = tid; h < H; h += kBlock) x_cache[h] = __bfloat162float(x[h]);
  __syncthreads();

  const uint8_t* grow = w_gu + static_cast<int64_t>(i) * H;
  const uint8_t* urow = w_gu + static_cast<int64_t>(I + i) * H;
  float gate = 0.f, up = 0.f;
  const int H16 = H & ~15;
  for (int h = tid * 16; h < H16; h += kBlock * 16) {
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const int hh = h + t;
      const float xv = x_cache[hh];
      gate += xv * lut[grow[hh]];
      up += xv * lut[urow[hh]];
    }
  }
  for (int h = H16 + tid; h < H; h += kBlock) {
    const float xv = x_cache[h];
    gate += xv * lut[grow[h]];
    up += xv * lut[urow[h]];
  }
  gate = BlockSum256(gate);
  up = BlockSum256(up);
  if (tid == 0) {
    const float sg = __bfloat162float(s_gu[i]);
    const float su = __bfloat162float(s_gu[I + i]);
    act[i] = gelu_tanh(sg * gate) * (su * up);
  }
}

__global__ void ExpertFp8DownKernel(__hip_bfloat16* __restrict__ y, const float* __restrict__ act,
                                    const uint8_t* __restrict__ w_dn,
                                    const __hip_bfloat16* __restrict__ s_dn, int I, int H,
                                    float alpha, float beta) {
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  const uint8_t* drow = w_dn + static_cast<int64_t>(h) * I;
  float acc = 0.f;
  int i = 0;
  for (; i + 3 < I; i += 4) {
    acc += act[i] * lut[drow[i]];
    acc += act[i + 1] * lut[drow[i + 1]];
    acc += act[i + 2] * lut[drow[i + 2]];
    acc += act[i + 3] * lut[drow[i + 3]];
  }
  for (; i < I; ++i) acc += act[i] * lut[drow[i]];
  const float sd = __bfloat162float(s_dn[h]);
  float v = alpha * sd * acc;
  if (beta != 0.f) v += beta * __bfloat162float(y[h]);
  y[h] = __float2bfloat16(v);
}

__global__ void Bf16ToF32VecKernel(float* __restrict__ out, const __hip_bfloat16* __restrict__ in,
                                   int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = __bfloat162float(in[i]);
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

void MatmulBTFp8ChannelRocm(Queue& q, void* out, const void* a, const void* b_fp8,
                            const void* scale_bf16, int M, int N, int K, float alpha,
                            float beta) {
  if (M <= 0 || N <= 0 || K <= 0 || !out || !a || !b_fp8 || !scale_bf16) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const size_t shmem = static_cast<size_t>(K) * sizeof(float);
  if (shmem > 48 * 1024) return;

  if (M == 1) {
    Fp8ChannelGemvFastKernel<<<N, kBlock, shmem, s>>>(
        static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
        static_cast<const uint8_t*>(b_fp8), static_cast<const __hip_bfloat16*>(scale_bf16), N, K,
        alpha, beta);
    return;
  }

  // Prefill / multi-token: keep W in FP8; dequant one row at a time into LDS.
  // NOTE: serial-in-M — prefer DequantFp8ChannelBf16Rocm + hipBLAS for large M.
  Fp8ChannelGemmMKernel<<<N, kBlock, shmem, s>>>(
      static_cast<__hip_bfloat16*>(out), static_cast<const __hip_bfloat16*>(a),
      static_cast<const uint8_t*>(b_fp8), static_cast<const __hip_bfloat16*>(scale_bf16), M, N, K,
      alpha, beta);
}

// out[n,k] = scale[n] * f8_e4m3_to_f32(w[n,k]) as bf16
__global__ void Fp8ChannelDequantBf16Kernel(__hip_bfloat16* __restrict__ out,
                                            const uint8_t* __restrict__ w,
                                            const __hip_bfloat16* __restrict__ scale, int N,
                                            int K) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
  const int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(K);
  if (idx >= total) return;
  const int n = static_cast<int>(idx / K);
  const float s = __bfloat162float(scale[n]);
  const float v = s * F8E4M3ToF32(w[idx]);
  out[idx] = __float2bfloat16(v);
}

void DequantFp8ChannelBf16Rocm(Queue& q, void* out_bf16, const void* fp8, const void* scale_bf16,
                               int N, int K) {
  if (!out_bf16 || !fp8 || !scale_bf16 || N <= 0 || K <= 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int64_t total = static_cast<int64_t>(N) * static_cast<int64_t>(K);
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  Fp8ChannelDequantBf16Kernel<<<blocks, threads, 0, s>>>(
      static_cast<__hip_bfloat16*>(out_bf16), static_cast<const uint8_t*>(fp8),
      static_cast<const __hip_bfloat16*>(scale_bf16), N, K);
}

bool ExpertGeGLUFp8M1Rocm(Queue& q, void* y, const void* x, const void* fp8_gu, const void* s_gu,
                          const void* fp8_dn, const void* s_dn, int I, int H, float alpha,
                          float beta) {
  if (!y || !x || !fp8_gu || !s_gu || !fp8_dn || !s_dn || I <= 0 || H <= 0) return false;
  const size_t shmem = static_cast<size_t>(H) * sizeof(float);
  if (shmem > 48 * 1024) return false;
  if (!EnsureAct(I)) return false;
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  float* act = TlsAct().p;
  ExpertFp8ActKernel<<<I, kBlock, shmem, st>>>(
      act, static_cast<const __hip_bfloat16*>(x), static_cast<const uint8_t*>(fp8_gu),
      static_cast<const __hip_bfloat16*>(s_gu), I, H);
  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertFp8DownKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(y), act, static_cast<const uint8_t*>(fp8_dn),
      static_cast<const __hip_bfloat16*>(s_dn), I, H, alpha, beta);
  return hipGetLastError() == hipSuccess;
}

// --- True top-k fused FP8 ExpertGeGLU (decode T=1) ---------------------------
// 1 act grid (I) looping G + 1 down. (Multi-row/LDS-act tiled variants regressed
// on gfx1201 lab A/B 2026-08-09 — keep this path.)
constexpr int kMaxTopK = 8;

// grid = I — x already expanded to float in global (L2-resident); no per-block x smem.
__global__ void ExpertFp8ActTopKKernel(float* __restrict__ act,  // [G,I]
                                       const float* __restrict__ x_f,  // [H]
                                       const uint8_t* const* __restrict__ w_gu,
                                       const __hip_bfloat16* const* __restrict__ s_gu, int G,
                                       int I, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= I) return;
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);

  for (int g = 0; g < G; ++g) {
    const uint8_t* gu = w_gu[g];
    const __hip_bfloat16* sg = s_gu[g];
    const uint8_t* grow = gu + static_cast<int64_t>(i) * H;
    const uint8_t* urow = gu + static_cast<int64_t>(I + i) * H;
    float gate = 0.f, up = 0.f;
    const int H16 = H & ~15;
    for (int h = tid * 16; h < H16; h += kBlock * 16) {
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        const int hh = h + t;
        const float xv = x_f[hh];
        gate += xv * lut[grow[hh]];
        up += xv * lut[urow[hh]];
      }
    }
    for (int h = H16 + tid; h < H; h += kBlock) {
      const float xv = x_f[h];
      gate += xv * lut[grow[h]];
      up += xv * lut[urow[h]];
    }
    gate = BlockSumFast(gate);
    up = BlockSumFast(up);
    if (tid == 0) {
      const float sgate = __bfloat162float(sg[i]);
      const float sup = __bfloat162float(sg[I + i]);
      act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
    }
    __syncthreads();
  }
}

__global__ void ExpertFp8DownTopKKernel(__hip_bfloat16* __restrict__ ysum,
                                        const float* __restrict__ act,  // [G,I]
                                        const uint8_t* const* __restrict__ w_dn,
                                        const __hip_bfloat16* const* __restrict__ s_dn,
                                        const float* __restrict__ wts, int G, int I, int H) {
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  float acc = 0.f;
  for (int g = 0; g < G; ++g) {
    const uint8_t* drow = w_dn[g] + static_cast<int64_t>(h) * I;
    const float* arow = act + static_cast<int64_t>(g) * I;
    float dot = 0.f;
    int i = 0;
    for (; i + 3 < I; i += 4) {
      dot += arow[i] * lut[drow[i]];
      dot += arow[i + 1] * lut[drow[i + 1]];
      dot += arow[i + 2] * lut[drow[i + 2]];
      dot += arow[i + 3] * lut[drow[i + 3]];
    }
    for (; i < I; ++i) dot += arow[i] * lut[drow[i]];
    acc += wts[g] * __bfloat162float(s_dn[g][h]) * dot;
  }
  ysum[h] = __float2bfloat16(acc);
}

struct TopKTls {
  const void** d_fgu = nullptr;
  const void** d_sgu = nullptr;
  const void** d_fdn = nullptr;
  const void** d_sdn = nullptr;
  float* d_wts = nullptr;
  float* d_act = nullptr;
  int act_cap = 0;
};
TopKTls& TlsTopK() {
  static thread_local TopKTls t;
  return t;
}

bool ExpertGeGLUFp8TopKM1Rocm(Queue& q, void* ysum, const void* x, const void* const* fp8_gu,
                              const void* const* s_gu, const void* const* fp8_dn,
                              const void* const* s_dn, const float* wts, int G, int I, int H) {
  if (G <= 0 || G > kMaxTopK || !ysum || !x || !fp8_gu || !s_gu || !fp8_dn || !s_dn || !wts)
    return false;
  if (I <= 0 || H <= 0) return false;
  // Act no longer needs H*f32 dynamic smem (x expanded to global).
  if (H <= 0) return false;

  const int dev = q.device.index;
  if (dev < 0 || dev >= 8) return false;
  {
    int cur = -1;
    if (hipGetDevice(&cur) != hipSuccess || cur != dev) {
      if (hipSetDevice(dev) != hipSuccess) return false;
    }
  }

  // Per-device TLS: dual-GPU expert-run must not reuse GPU0 scratch on GPU1.
  struct TopKDevTls {
    const void** d_fgu = nullptr;
    const void** d_sgu = nullptr;
    const void** d_fdn = nullptr;
    const void** d_sdn = nullptr;
    float* d_wts = nullptr;
    float* d_act = nullptr;
    float* d_xf = nullptr;
    int act_cap = 0;
    int xf_cap = 0;
    int bound_dev = -1;
  };
  static thread_local TopKDevTls slots[8];
  auto& tls = slots[dev];
  if (tls.bound_dev != dev) {
    tls = TopKDevTls{};
    tls.bound_dev = dev;
  }

  const int act_need = G * I;
  if (tls.act_cap < act_need || !tls.d_act) {
    if (tls.d_act) (void)hipFree(tls.d_act);
    tls.d_act = nullptr;
    tls.act_cap = 0;
    if (hipMalloc(&tls.d_act, static_cast<size_t>(act_need) * sizeof(float)) != hipSuccess)
      return false;
    tls.act_cap = act_need;
  }
  if (tls.xf_cap < H || !tls.d_xf) {
    if (tls.d_xf) (void)hipFree(tls.d_xf);
    tls.d_xf = nullptr;
    tls.xf_cap = 0;
    if (hipMalloc(&tls.d_xf, static_cast<size_t>(H) * sizeof(float)) != hipSuccess) return false;
    tls.xf_cap = H;
  }
  if (!tls.d_fgu) {
    if (hipMalloc(&tls.d_fgu, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_sgu, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_fdn, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_sdn, sizeof(void*) * kMaxTopK) != hipSuccess) return false;
    if (hipMalloc(&tls.d_wts, sizeof(float) * kMaxTopK) != hipSuccess) return false;
  }

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  if (hipMemcpyAsync(tls.d_fgu, fp8_gu, sizeof(void*) * static_cast<size_t>(G),
                     hipMemcpyHostToDevice, st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_sgu, s_gu, sizeof(void*) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_fdn, fp8_dn, sizeof(void*) * static_cast<size_t>(G),
                     hipMemcpyHostToDevice, st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_sdn, s_dn, sizeof(void*) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;
  if (hipMemcpyAsync(tls.d_wts, wts, sizeof(float) * static_cast<size_t>(G), hipMemcpyHostToDevice,
                     st) != hipSuccess)
    return false;

  // Expand x once → L2; act blocks read x_f (no 704× smem fills of H).
  {
    const int xt = 256;
    const int xg = (H + xt - 1) / xt;
    Bf16ToF32VecKernel<<<xg, xt, 0, st>>>(tls.d_xf, static_cast<const __hip_bfloat16*>(x), H);
  }
  dim3 grid_act(static_cast<unsigned>(I));
  ExpertFp8ActTopKKernel<<<grid_act, kBlock, 0, st>>>(
      tls.d_act, tls.d_xf, reinterpret_cast<const uint8_t* const*>(tls.d_fgu),
      reinterpret_cast<const __hip_bfloat16* const*>(tls.d_sgu), G, I, H);

  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertFp8DownTopKKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(ysum), tls.d_act,
      reinterpret_cast<const uint8_t* const*>(tls.d_fdn),
      reinterpret_cast<const __hip_bfloat16* const*>(tls.d_sdn), tls.d_wts, G, I, H);

  return hipGetLastError() == hipSuccess;
}

// Contiguous FP8 packs + device top-k indices (no host pointer gather / D2H).
// grid = (I, G) — one block per (intermediate row, expert); experts run in parallel.
// gu_base [E,2I,H] u8, sgu [E,2I] bf16.
// x is BF16 [H] — convert on load (skips separate Bf16ToF32 launch + d_xf write).
__global__ void ExpertFp8ActTopKIndexedKernel(float* __restrict__ act,
                                              const __hip_bfloat16* __restrict__ x,
                                              const uint8_t* __restrict__ gu_base,
                                              const __hip_bfloat16* __restrict__ sgu_base,
                                              const int32_t* __restrict__ idx, int G, int I, int H) {
  const int i = static_cast<int>(blockIdx.x);
  const int g = static_cast<int>(blockIdx.y);
  if (i >= I || g >= G) return;
  __shared__ float lut[256];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);
  const int e = idx[g];

  const size_t gu_one = static_cast<size_t>(2) * static_cast<size_t>(I) * static_cast<size_t>(H);
  const size_t sgu_one = static_cast<size_t>(2) * static_cast<size_t>(I);
  const uint8_t* gu = gu_base + static_cast<size_t>(e) * gu_one;
  const __hip_bfloat16* sg = sgu_base + static_cast<size_t>(e) * sgu_one;
  const uint8_t* grow = gu + static_cast<int64_t>(i) * H;
  const uint8_t* urow = gu + static_cast<int64_t>(I + i) * H;

  float gate = 0.f, up = 0.f;
  const int H16 = H & ~15;
  for (int h = tid * 16; h < H16; h += kBlock * 16) {
#pragma unroll
    for (int tt = 0; tt < 16; ++tt) {
      const int hh = h + tt;
      const float xv = __bfloat162float(x[hh]);
      gate += xv * lut[grow[hh]];
      up += xv * lut[urow[hh]];
    }
  }
  for (int h = H16 + tid; h < H; h += kBlock) {
    const float xv = __bfloat162float(x[h]);
    gate += xv * lut[grow[h]];
    up += xv * lut[urow[h]];
  }
  gate = BlockSumFast(gate);
  up = BlockSumFast(up);
  if (tid == 0) {
    const float sgate = __bfloat162float(sg[i]);
    const float sup = __bfloat162float(sg[I + i]);
    act[static_cast<int64_t>(g) * I + i] = gelu_tanh(sgate * gate) * (sup * up);
  }
}

__global__ void ExpertFp8DownTopKIndexedKernel(__hip_bfloat16* __restrict__ ysum,
                                               const float* __restrict__ act,
                                               const uint8_t* __restrict__ dn_base,
                                               const __hip_bfloat16* __restrict__ sdn_base,
                                               const int32_t* __restrict__ idx,
                                               const float* __restrict__ wts, int G, int I, int H) {
  __shared__ float lut[256];
  __shared__ int sidx[kMaxTopK];
  __shared__ float swts[kMaxTopK];
  FillF8Lut(lut);
  const int tid = static_cast<int>(threadIdx.x);
  if (tid < G) {
    sidx[tid] = idx[tid];
    swts[tid] = wts[tid];
  }
  __syncthreads();
  const int h = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (h >= H) return;
  const size_t dn_one = static_cast<size_t>(H) * static_cast<size_t>(I);
  const size_t sdn_one = static_cast<size_t>(H);
  float acc = 0.f;
  for (int g = 0; g < G; ++g) {
    const int e = sidx[g];
    const uint8_t* drow = dn_base + static_cast<size_t>(e) * dn_one + static_cast<size_t>(h) * I;
    const float* arow = act + static_cast<int64_t>(g) * I;
    float dot = 0.f;
    int i = 0;
    for (; i + 3 < I; i += 4) {
      dot += arow[i] * lut[drow[i]];
      dot += arow[i + 1] * lut[drow[i + 1]];
      dot += arow[i + 2] * lut[drow[i + 2]];
      dot += arow[i + 3] * lut[drow[i + 3]];
    }
    for (; i < I; ++i) dot += arow[i] * lut[drow[i]];
    const __hip_bfloat16* sd = sdn_base + static_cast<size_t>(e) * sdn_one;
    acc += swts[g] * __bfloat162float(sd[h]) * dot;
  }
  ysum[h] = __float2bfloat16(acc);
}

__global__ void ApplyExpertScaleRwKernel(float* __restrict__ rw, const int32_t* __restrict__ ri,
                                         const float* __restrict__ escale, int G, int E) {
  const int g = static_cast<int>(threadIdx.x);
  if (g >= G) return;
  const int e = ri[g];
  if (e >= 0 && e < E) rw[g] *= escale[e];
}

struct IdxTls {
  float* d_act = nullptr;
  float* d_xf = nullptr;
  int act_cap = 0;
  int xf_cap = 0;
  int bound_dev = -1;
};
IdxTls& IdxTlsSlot(int dev) {
  static thread_local IdxTls slots[8];
  return slots[dev];
}

// Alloc-only (safe to call at Prepare). Graph-safe: no hipSetDevice/hipMalloc once warm.
bool EnsureIdxTls(int dev, int G, int I, int H) {
  if (dev < 0 || dev >= 8 || G <= 0 || G > kMaxTopK || I <= 0 || H <= 0) return false;
  auto& tls = IdxTlsSlot(dev);
  const int act_need = G * I;
  if (tls.bound_dev == dev && tls.d_act && tls.d_xf && tls.act_cap >= act_need &&
      tls.xf_cap >= H)
    return true;  // hot path: no device API
  if (hipSetDevice(dev) != hipSuccess) return false;
  if (tls.bound_dev != dev) {
    tls = IdxTls{};
    tls.bound_dev = dev;
  }
  if (tls.act_cap < act_need || !tls.d_act) {
    if (tls.d_act) (void)hipFree(tls.d_act);
    tls.d_act = nullptr;
    tls.act_cap = 0;
    if (hipMalloc(&tls.d_act, static_cast<size_t>(act_need) * sizeof(float)) != hipSuccess)
      return false;
    tls.act_cap = act_need;
  }
  if (tls.xf_cap < H || !tls.d_xf) {
    if (tls.d_xf) (void)hipFree(tls.d_xf);
    tls.d_xf = nullptr;
    tls.xf_cap = 0;
    if (hipMalloc(&tls.d_xf, static_cast<size_t>(H) * sizeof(float)) != hipSuccess) return false;
    tls.xf_cap = H;
  }
  return true;
}

bool ExpertGeGLUFp8TopKIndexedRocm(Queue& q, void* ysum, const void* x, const void* gu_base,
                                   const void* dn_base, const void* sgu_base, const void* sdn_base,
                                   const int32_t* idx_dev, const float* wts_dev, int G, int I,
                                   int H) {
  if (G <= 0 || G > kMaxTopK || !ysum || !x || !gu_base || !dn_base || !sgu_base || !sdn_base ||
      !idx_dev || !wts_dev)
    return false;
  if (I <= 0 || H <= 0) return false;
  const int dev = q.device.index;
  if (!EnsureIdxTls(dev, G, I, H)) return false;
  auto& tls = IdxTlsSlot(dev);

  hipStream_t st = static_cast<hipStream_t>(q.handle);
  dim3 grid_act(static_cast<unsigned>(I), static_cast<unsigned>(G));
  ExpertFp8ActTopKIndexedKernel<<<grid_act, kBlock, 0, st>>>(
      tls.d_act, static_cast<const __hip_bfloat16*>(x), static_cast<const uint8_t*>(gu_base),
      static_cast<const __hip_bfloat16*>(sgu_base), idx_dev, G, I, H);
  const int grid_dn = (H + kBlock - 1) / kBlock;
  ExpertFp8DownTopKIndexedKernel<<<grid_dn, kBlock, 0, st>>>(
      static_cast<__hip_bfloat16*>(ysum), tls.d_act, static_cast<const uint8_t*>(dn_base),
      static_cast<const __hip_bfloat16*>(sdn_base), idx_dev, wts_dev, G, I, H);
  return hipGetLastError() == hipSuccess;
}

void ApplyExpertScaleRwRocm(Queue& q, float* rw_dev, const int32_t* ri_dev, const float* escale_dev,
                            int G, int E) {
  if (!rw_dev || !ri_dev || !escale_dev || G <= 0) return;
  hipStream_t st = static_cast<hipStream_t>(q.handle);
  ApplyExpertScaleRwKernel<<<1, G, 0, st>>>(rw_dev, ri_dev, escale_dev, G, E);
}

// Pre-allocate per-device ExpertGeGLU scratch (act + x_f) so decode never hipMalloc mid-step.
bool PrewarmExpertGeGLUFp8TopKIndexedRocm(int dev, int G, int I, int H) {
  return EnsureIdxTls(dev, G, I, H);
}

// --- Prefill MoE gather / weighted scatter (device-only) ---

__global__ void MoeGatherRowsKernel(__hip_bfloat16* __restrict__ out,
                                    const __hip_bfloat16* __restrict__ in,
                                    const int32_t* __restrict__ tids, int n, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= n) return;
  const int t = tids[i];
  const __hip_bfloat16* src = in + static_cast<int64_t>(t) * H;
  __hip_bfloat16* dst = out + static_cast<int64_t>(i) * H;
  for (int h = static_cast<int>(threadIdx.x); h < H; h += static_cast<int>(blockDim.x)) {
    dst[h] = src[h];
  }
}

// One block per (row i); threads walk H. Atomic not needed if tokens unique per expert chunk
// (same token can appear once per expert with top-k — different experts write different
// increments to same acc row → need atomics on bf16 via float CAS or float acc).
// Use float staging: each thread does atomicAdd on float buffer — too heavy.
// Simpler: serialize rows with one warp per row doing non-atomic read-modify-write
// when we guarantee single-threaded updates per (token,h) within a launch.
// For top-k, same token appears in multiple experts → concurrent expert launches are
// sequential in host loop, so within one scatter launch each token appears once.
// Safe non-atomic RMW per (i) row.
__global__ void MoeWeightedScatterAddKernel(__hip_bfloat16* __restrict__ acc,
                                            const __hip_bfloat16* __restrict__ y,
                                            const int32_t* __restrict__ tids,
                                            const float* __restrict__ wts, int n, int H) {
  const int i = static_cast<int>(blockIdx.x);
  if (i >= n) return;
  const int t = tids[i];
  const float w = wts[i];
  const __hip_bfloat16* src = y + static_cast<int64_t>(i) * H;
  __hip_bfloat16* dst = acc + static_cast<int64_t>(t) * H;
  for (int h = static_cast<int>(threadIdx.x); h < H; h += static_cast<int>(blockDim.x)) {
    const float v = w * __bfloat162float(src[h]) + __bfloat162float(dst[h]);
    dst[h] = __float2bfloat16(v);
  }
}

__global__ void MoeZeroBf16Kernel(__hip_bfloat16* __restrict__ buf, int64_t nelem) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + static_cast<int64_t>(threadIdx.x);
  if (i < nelem) buf[i] = __float2bfloat16(0.f);
}

void MoeGatherRowsRocm(Queue& q, void* out_bf16, const void* in_bf16, const int32_t* token_ids,
                       int n, int H) {
  if (n <= 0 || H <= 0 || !out_bf16 || !in_bf16 || !token_ids) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  MoeGatherRowsKernel<<<n, kBlock, 0, s>>>(
      static_cast<__hip_bfloat16*>(out_bf16), static_cast<const __hip_bfloat16*>(in_bf16),
      token_ids, n, H);
}

void MoeWeightedScatterAddRocm(Queue& q, void* acc_bf16, const void* y_bf16,
                               const int32_t* token_ids, const float* weights, int n, int H) {
  if (n <= 0 || H <= 0 || !acc_bf16 || !y_bf16 || !token_ids || !weights) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  // Threads in a block share a row — need sync between h-loop iterations? No, each h
  // unique per thread. Multiple threads write different h of same row — OK.
  MoeWeightedScatterAddKernel<<<n, kBlock, 0, s>>>(
      static_cast<__hip_bfloat16*>(acc_bf16), static_cast<const __hip_bfloat16*>(y_bf16),
      token_ids, weights, n, H);
}

void MoeZeroBf16Rocm(Queue& q, void* buf_bf16, int64_t nelem) {
  if (!buf_bf16 || nelem <= 0) return;
  hipStream_t s = static_cast<hipStream_t>(q.handle);
  const int block = 256;
  const int grid = static_cast<int>((nelem + block - 1) / block);
  MoeZeroBf16Kernel<<<grid, block, 0, s>>>(static_cast<__hip_bfloat16*>(buf_bf16), nelem);
}

}  // namespace vt::rocm
