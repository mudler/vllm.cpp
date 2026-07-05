// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// Per-tensor FP8 (W8A16) dequant-GEMM for the CUDA backend. Mirrors the NVFP4
// W4A16 kernel (cuda_matmul_nvfp4.cu) — same naive/tiled/WMMA structure — but the
// weight is raw IEEE fp8-e4m3fn (one byte per element, per-tensor scale, no
// packing and no group scale), which is simpler than fp4.
//
// out[M,N] = act[M,K] (f32/bf16) @ dequant(w).T, where w is a torch Linear weight
// [N=out_features, K=in_features] read DIRECTLY from device memory and
// dequantized on the fly (no host bf16 weight tensor). The decode is bit-for-bit
// vllm::DequantFp8ToBf16 (the authoritative reference):
//   value = bf16_round( f8_e4m3(byte) * weight_scale )
// so this GEMM and Matmul(act, DequantFp8ToBf16(w).T) differ only in K-reduction
// order (matmul tolerance), not in the per-element product.
//
// This is the fp8-resident decode-path optimization: the attention q/k/v/o and
// GDN in_proj_qkv/z/out_proj projections of the 35B checkpoint (per-tensor fp8 on
// disk) were dequantized to bf16 at LOAD and run through cublas bf16 gemv at
// decode; keeping them fp8-resident + this in-kernel-dequant GEMV halves their
// decode weight bandwidth AND their device memory.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: matmul_fp8: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Tensor-core (bf16 WMMA) prefill path toggle (A/B; default ON). Set
// VT_FP8_WMMA=0 to fall back to the CUDA-core tiled kernel.
bool WmmaEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP8_WMMA");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// f32 load/store overloads: bf16 converts on the way in/out, math is f32.
__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, same as host F32ToBF16
}

// IEEE fp8-e4m3fn -> f32, mirroring vllm::F8E4M3ToF32 (1 sign, 4 exp, 3 mant,
// bias 7, no inf, NaN only at 0x7F/0xFF).
__device__ inline float F8E4M3ToF32Dev(uint8_t byte) {
  const uint32_t sign = static_cast<uint32_t>(byte >> 7) & 0x1u;
  const uint32_t exp = static_cast<uint32_t>(byte >> 3) & 0xFu;
  const uint32_t mant = static_cast<uint32_t>(byte) & 0x7u;
  const float sign_mul = sign ? -1.0f : 1.0f;
  if (exp == 0xFu && mant == 0x7u) return CUDART_NAN_F;
  if (exp == 0u) return sign_mul * (static_cast<float>(mant) * (1.0f / 512.0f));
  const float mantissa = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
  return sign_mul * ldexpf(mantissa, static_cast<int>(exp) - 7);
}

// Dequant one fp8 byte to its bf16-rounded weight value (bit-for-bit
// vllm::DequantFp8ToBf16: round the f8*scale product through bf16 before use).
__device__ __forceinline__ float DecodeFp8(uint8_t b, float scale) {
  return __bfloat162float(__float2bfloat16(F8E4M3ToF32Dev(b) * scale));
}

// One output column's fp8 W8A16 dot product against a precomputed dequant LUT
// (lut[byte] = bf16(f8_e4m3(byte)*scale), f32) — the inner loop is pure load +
// fma, no per-element decode ALU. The weight row is K raw fp8 bytes; read
// 128-bit-wide (uint4 = 16 bytes) where the row is 16-byte aligned, scalar for
// the tail. Ascending-k order matches the scalar loop => bit-identical.
template <typename Tin>
__device__ __forceinline__ float Fp8ColDotLut(const uint8_t* wrow, const Tin* arow, int64_t k_dim,
                                              const float* lut) {
  float acc = 0.0f;
  int64_t k = 0;
  if ((reinterpret_cast<uintptr_t>(wrow) & 0xf) == 0) {
    const int64_t vec = (k_dim / 16) * 16;  // bytes covered by full uint4 loads
    const uint4* wrow4 = reinterpret_cast<const uint4*>(wrow);
    for (; k < vec; k += 16) {
      const uint4 pk = wrow4[k / 16];
      const uint32_t words[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
        for (int bi = 0; bi < 4; ++bi) {
          const uint8_t byte = static_cast<uint8_t>((words[wi] >> (bi * 8)) & 0xffu);
          acc += Load(arow, k + wi * 4 + bi) * lut[byte];
        }
      }
    }
  }
  for (; k < k_dim; ++k) acc += Load(arow, k) * lut[wrow[k]];
  return acc;
}

// Naive one-thread-per-output kernel (grid.x over N, grid.y over M). Used for
// small M (decode): weight is L2-resident so coalescing buys little and this
// avoids the tiled kernel's shared-mem/sync overhead. The block first builds the
// 256-entry dequant LUT in shared (fp8 byte -> bf16(byte*scale) as f32),
// eliminating the per-element ldexpf/decode from the K-loop so the GEMV is
// bandwidth-bound (the whole point of fp8-resident). Graph-capture-safe.
template <typename Tin, typename Tout>
__global__ void MatmulFp8Naive(Tout* out, const Tin* act, const uint8_t* weight, float scale,
                               int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  __shared__ float lut[256];
  for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x))
    lut[i] = DecodeFp8(static_cast<uint8_t>(i), scale);
  __syncthreads();

  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;
  const uint8_t* wrow = weight + n * k_dim;
  const Tin* arow = act + m * k_dim;
  Store(out, m * n_cols + n, Fp8ColDotLut(wrow, arow, k_dim, lut));
}

// Shared-memory tiled dequant-GEMM. A block owns output tile [BM x BN]; each
// thread owns a [TM x TN] register sub-tile. Weight tile [BN x BK] decoded into
// shared with coalesced byte reads (consecutive threads read consecutive weight
// bytes of a row). blockDim.x == (BM/TM)*(BN/TN).
template <typename Tin, typename Tout, int BM, int BN, int BK, int TM, int TN>
__global__ void MatmulFp8Tiled(Tout* out, const Tin* act, const uint8_t* weight, float scale,
                               int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  __shared__ float As[BM * BK];
  __shared__ float Ws[BN * BK];

  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;

  const int t_row = (tid / (BN / TN)) * TM;
  const int t_col = (tid % (BN / TN)) * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      const int64_t gr = row0 + r;
      As[idx] = (gr < m_rows && kt + c < k_dim) ? Load(act, gr * k_dim + kt + c) : 0.0f;
    }
    for (int idx = tid; idx < BN * BK; idx += kThreads) {
      const int nl = idx / BK, kl = idx % BK;
      const int64_t gn = col0 + nl;
      Ws[idx] = (gn < n_cols && kt + kl < k_dim) ? DecodeFp8(weight[gn * k_dim + kt + kl], scale)
                                                 : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float ar[TM], br[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ar[i] = As[(t_row + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) br[j] = Ws[(t_col + j) * BK + k];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ar[i] * br[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int64_t gr = row0 + t_row + i;
    if (gr >= m_rows) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int64_t gc = col0 + t_col + j;
      if (gc < n_cols) Store(out, gr * n_cols + gc, acc[i][j]);
    }
  }
}

// Tensor-core (bf16 WMMA) prefill path. Same math as MatmulFp8Tiled but the
// K-reduction runs on Blackwell tensor cores: the fp8 weight tile is dequantized
// INTO SHARED as bf16 (CUDA-core decode, exactly DecodeFp8's bf16-rounded value)
// and the bf16 activation tile is staged into shared; each warp runs wmma
// m16n16k16 bf16xbf16 -> f32 over the tile. bf16 activations only.
namespace wmma = nvcuda::wmma;

template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MatmulFp8Wmma(Tout* out, const __nv_bfloat16* act, const uint8_t* weight,
                              float scale, int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M;
  constexpr int WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16;
  constexpr int NF = WNPER / 16;
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      const int64_t gr = row0 + r;
      As[idx] = (gr < m_rows && kt + c < k_dim) ? act[gr * k_dim + kt + c]
                                                : __float2bfloat16(0.0f);
    }
    for (int idx = tid; idx < BN * BK; idx += kThreads) {
      const int nl = idx / BK, kl = idx % BK;
      const int64_t gn = col0 + nl;
      Ws[idx] = (gn < n_cols && kt + kl < k_dim)
                    ? __float2bfloat16(DecodeFp8(weight[gn * k_dim + kt + kl], scale))
                    : __float2bfloat16(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[MF];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[NF];
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
        wmma::load_matrix_sync(af[mi], &As[(wm * WMPER + mi * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int ni = 0; ni < NF; ++ni)
        wmma::load_matrix_sync(bf[ni], &Ws[(wn * WNPER + ni * 16) * BK + kk * 16], BK);
#pragma unroll
      for (int mi = 0; mi < MF; ++mi)
#pragma unroll
        for (int ni = 0; ni < NF; ++ni) wmma::mma_sync(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni)
      wmma::store_matrix_sync(&Cs[(wm * WMPER + mi * 16) * BN + (wn * WNPER + ni * 16)],
                              acc[mi][ni], BN, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < BM * BN; idx += kThreads) {
    const int r = idx / BN, c = idx % BN;
    const int64_t gr = row0 + r, gc = col0 + c;
    if (gr < m_rows && gc < n_cols) Store(out, gr * n_cols + gc, Cs[idx]);
  }
}

// Below this many activation rows the naive kernel wins (weight L2-resident,
// tiling overhead not amortized); at/above it the tiled/WMMA kernels win. Mirrors
// the NVFP4 kernel's tuned threshold.
constexpr int64_t kTileMinRows = 32;

template <typename Tin, typename Tout>
void Launch(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& weight, float scale,
            int64_t m, int64_t n, int64_t k) {
  if (m < kTileMinRows) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(m));
    MatmulFp8Naive<Tin, Tout><<<grid, kBlock, 0, s>>>(out.Ptr<Tout>(), act.Ptr<Tin>(),
                                                      weight.Ptr<uint8_t>(), scale, m, n, k);
    Check(cudaGetLastError(), "kernel launch (naive)");
    return;
  }
  // BF16 activations at large M run on tensor cores. BK=16 (fp8 has no group
  // alignment; 16 matches the wmma K-fragment). Tile 64x64, warps 2x2.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (WmmaEnabled()) {
      constexpr int BM = 64, BN = 64, BK = 16, WARPS_M = 2, WARPS_N = 2;
      constexpr int kThreads = WARPS_M * WARPS_N * 32;
      const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                      static_cast<unsigned>((m + BM - 1) / BM));
      MatmulFp8Wmma<Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
          out.Ptr<Tout>(), act.Ptr<__nv_bfloat16>(), weight.Ptr<uint8_t>(), scale, m, n, k);
      Check(cudaGetLastError(), "kernel launch (wmma)");
      return;
    }
  }
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int kThreads = (BM / TM) * (BN / TN);
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
  MatmulFp8Tiled<Tin, Tout, BM, BN, BK, TM, TN><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), weight.Ptr<uint8_t>(), scale, m, n, k);
  Check(cudaGetLastError(), "kernel launch (tiled)");
}

template <typename Tin>
void LaunchByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& weight, float scale,
                 int64_t m, int64_t n, int64_t k) {
  switch (out.dtype) {
    case DType::kF32: Launch<Tin, float>(s, out, act, weight, scale, m, n, k); break;
    case DType::kBF16: Launch<Tin, __nv_bfloat16>(s, out, act, weight, scale, m, n, k); break;
    default: VT_CHECK(false, "cuda matmul_fp8: unsupported out dtype (f32/bf16 only)");
  }
}

void MatmulFp8KernelCuda(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight,
                         float weight_scale) {
  const int64_t m = act.shape[0], k = act.shape[1], n = weight.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchByOut<float>(s, out, act, weight, weight_scale, m, n, k);
      break;
    case DType::kBF16:
      LaunchByOut<__nv_bfloat16>(s, out, act, weight, weight_scale, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_fp8: unsupported act dtype (f32/bf16 only)");
  }
}

// Registers the CUDA FP8 dequant-GEMM during static init (table fill only, no
// CUDA calls — see cuda_ops.cu for the pre-main discipline rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulFp8, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulFp8Fn>(&MatmulFp8KernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
