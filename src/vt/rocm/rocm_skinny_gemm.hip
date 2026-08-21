// ROCm skinny GEMM for decode (BACKEND-ROCM; issue #487).
// Port of vLLM's wvSplitK_hf_sml_ (csrc/rocm/skinny_gemms.cu:351-573) — the
// split-K, LDS-staged, CU-count-aware kernel that wins M<=4 shapes, replacing
// the 128x128-macro-tile rocBLAS GEMM that dominates decode GPU time. gfx1100
// (GFX1X/wave32) bf16. De-torched; semantics mirror the donor exactly.
//
// Layout mapping to our MatmulBT (out[M,N] = a[M,K] @ b[N,K]^T, decode M<=4):
//   donor in_a (weight [N,K]) = our b;  donor in_b (x [M,K]) = our a;
//   donor M_in = our N (output dim);  donor N_in = our M;  C[N_in,M_in] = out[M,N].

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"

namespace vt::rocm {
namespace {

constexpr int kLdsSize = 64 * 1024;  // gfx1100 (non-gfx95x)
constexpr int kThrds = 32;           // wave32
constexpr int kWvPrGrp = 16;
constexpr int kYtile = 2;            // proven decode config (donor gfx1x, N small)
constexpr int kUnrl = 2;
constexpr int kAChunk = 8;

using scalar8 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
union bigType {
  __hip_bfloat16 h[kAChunk];
  float f[kAChunk / 2];
  scalar8 h8;
};

__device__ __forceinline__ unsigned int min__(uint32_t a, uint32_t b) {
  return min(a, b);
}

// Donor's persistent-workgroup splitter (host-side in skinny_gemms.cu:1169).
inline int mindiv(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int rnds[13];
  for (int i = 0; i < 13; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = 12; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

// A (activation) fits LDS. N = decode batch (our M). bf16, f32 accum.
template <int N>
__global__ void __launch_bounds__(kWvPrGrp * kThrds)
    wvSplitKSml(const int K, const int Kbp, const int Kap, const int M,
                const __hip_bfloat16* __restrict__ B, const __hip_bfloat16* __restrict__ A,
                __hip_bfloat16* C, const int _WvPrGrp, const int CuCount) {
  constexpr int max_lds_len = kLdsSize / 2;  // bf16 elements
  __shared__ __hip_bfloat16 s[max_lds_len];

  // Stage the activation row(s) A [N,K] into LDS.
  for (uint32_t k = (threadIdx.y * kThrds + threadIdx.x) * kAChunk;
       k < min__(Kap * N, max_lds_len); k += kThrds * kWvPrGrp * kAChunk) {
    *((bigType*)(&s[k])) = *((const bigType*)(&A[k]));
  }
  __syncthreads();

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * kYtile;

  while (m < static_cast<uint32_t>(M)) {
    float sum[N][kYtile] = {};
    for (uint32_t k1 = 0; k1 < static_cast<uint32_t>(K); k1 += kThrds * kAChunk * kUnrl) {
      bigType bigA[N][kUnrl] = {};
      bigType bigB[kYtile][kUnrl];
#pragma unroll
      for (uint32_t k2 = 0; k2 < kUnrl; k2++) {
        uint32_t k = k1 + k2 * kThrds * kAChunk;
        uint32_t k_ = k + threadIdx.x * kAChunk;
        const __hip_bfloat16* B_ = &B[min__(k_, K - kAChunk)];
        for (int y = 0; y < kYtile; y++)
          bigB[y][k2].h8 = __builtin_nontemporal_load(
              (const scalar8*)(&B_[min__(y + m, M - 1) * Kbp]));
      }
#pragma unroll
      for (uint32_t k2 = 0; k2 < kUnrl; k2++) {
        uint32_t k = k1 + k2 * kThrds * kAChunk;
        uint32_t k_ = k + threadIdx.x * kAChunk;
        if (k_ >= static_cast<uint32_t>(K)) break;
        for (int n = 0; n < N; n++) bigA[n][k2] = *((const bigType*)(&(s[k_ + Kap * n])));
      }
      // Interleaved MAC; bf16 pairs unpacked to f32 (donor DOT2C bf16 branch).
      for (uint32_t k2 = 0; k2 < kUnrl; k2++) {
        for (int n = 0; n < N; n++) {
          for (int y = 0; y < kYtile; y++) {
#pragma unroll
            for (uint32_t b = 0; b < kAChunk / 2; b++) {
              float2 a2 = __bfloat1622float2(*((__hip_bfloat162*)(&(bigA[n][k2].h[b * 2]))));
              float2 b2 = __bfloat1622float2(*((__hip_bfloat162*)(&(bigB[y][k2].h[b * 2]))));
              sum[n][y] += (a2.x * b2.x) + (a2.y * b2.y);
            }
          }
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
    // Wave32 reduction: DPP row_shr 8/4/2/1 then shfl_xor(16).
    for (int n = 0; n < N; n++) {
      for (int y = 0; y < kYtile; y++) {
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf, 1);
        sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf, 1);
        sum[n][y] += __shfl_xor(sum[n][y], 16);
      }
    }
    if (threadIdx.x == (kThrds - 1)) {
      for (int n = 0; n < N; n++)
        for (int y = 0; y < kYtile; y++) C[m + y + n * M] = __float2bfloat16(sum[n][y]);
    }
    m += CuCount * _WvPrGrp * kYtile;
  }
}

int DeviceCuCount(int device) {
  static const int cache = [] {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return 96;  // gfx1100 fallback
    int n = 0;
    if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
        n <= 0)
      return 96;
    return n;
  }();
  (void)device;
  return cache;
}

}  // namespace

// out[M,N] = a[M,K] @ b[N,K]^T, bf16 in/out. Only called for the decode-skinny
// gate below (M in 1..4, K%8==0); the caller keeps every other shape on the
// BLAS path.
void WvSplitKBT(hipStream_t s, void* out, const void* a, const void* b, int M, int N,
                int K, int device) {
  const int cu = DeviceCuCount(device);
  dim3 grid(cu), block(kThrds, kWvPrGrp);
  const int wvPrGrp = mindiv(N, cu * kYtile, kWvPrGrp);
  auto* C = static_cast<__hip_bfloat16*>(out);
  auto* A = static_cast<const __hip_bfloat16*>(a);
  auto* B = static_cast<const __hip_bfloat16*>(b);
  switch (M) {
    case 1:
      wvSplitKSml<1><<<grid, block, 0, s>>>(K, K, K, N, B, A, C, wvPrGrp, cu);
      break;
    case 2:
      wvSplitKSml<2><<<grid, block, 0, s>>>(K, K, K, N, B, A, C, wvPrGrp, cu);
      break;
    case 3:
      wvSplitKSml<3><<<grid, block, 0, s>>>(K, K, K, N, B, A, C, wvPrGrp, cu);
      break;
    case 4:
      wvSplitKSml<4><<<grid, block, 0, s>>>(K, K, K, N, B, A, C, wvPrGrp, cu);
      break;
    default:
      throw std::runtime_error("vt rocm: wvSplitK unsupported M=" + std::to_string(M));
  }
  if (hipGetLastError() != hipSuccess)
    throw std::runtime_error("vt rocm: wvSplitK launch failed");
}

}  // namespace vt::rocm
