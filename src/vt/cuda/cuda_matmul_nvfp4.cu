// vllm.cpp original (vt runtime, inventory deviation §9.1); no upstream mirror.
// NVFP4 W4A16 dequant-GEMM for the CUDA backend (M2.2a, correctness-grade).
//
// out[M,N] = act[M,K] (f32/bf16) @ dequant(w).T, where w is a modelopt
// W4A16_NVFP4 weight [N=out_features, K=in_features] read DIRECTLY from device
// memory and dequantized on the fly (no host bf16 weight tensor). The decode is
// bit-for-bit vllm::DequantNvfp4ToBf16 (the authoritative reference):
//   * two 4-bit E2M1 codes per packed byte, low nibble = input elem 2i, high
//     nibble = 2i+1; nibble bit 3 = sign, bits 0..2 index the magnitude LUT
//     {0, .5, 1, 1.5, 2, 3, 4, 6};
//   * one IEEE fp8-e4m3fn scale per 16-element input group;
//   * group scale = f32(fp8_scale) * weight_scale_2 (per-tensor, multiplied);
//   * the dequanted value is ROUNDED TO BF16 before use — exactly the byte
//     DequantNvfp4ToBf16 stores — so this GEMM and Matmul(act, dequant(w).T)
//     differ only in K-reduction order, not in the per-element product.
// These are standard IEEE fp8-e4m3fn scales: the GGUF killgate fork's UE4M3
// x0.5 LUT trap (.agents/gguf-nvfp4-notes.md) does NOT apply to modelopt NVFP4.
//
// Layout (M2.4-tile): SHARED-MEMORY TILED GEMM with register accumulation. A
// thread-block computes a [BM x BN] output tile; per K-tile it cooperatively
// (a) loads the activation tile [BM x BK] into shared memory (coalesced, act is
// K-contiguous) and (b) decodes the fp4 weight tile [BN x BK] into shared memory
// with COALESCED packed-byte reads — consecutive threads read consecutive packed
// bytes of a weight row (the naive one-thread-per-output kernel had each thread
// stride a full weight row = fully uncoalesced, the GB10 bottleneck). Each thread
// then accumulates a [TM x TN] register sub-tile over the shared tiles. The
// per-element decode is byte-for-byte the naive kernel's (same nibble order,
// group-16 fp8-e4m3 scale, scale2, bf16-round) so the GEMM stays token-for-token
// correct; only the K-reduction order changes (tiled), within matmul tolerance.
//
// Layout (M2.6-wmma): the PREFILL (large-M) bf16 path runs on Blackwell TENSOR
// CORES. The fp4 weight tile is decoded into shared memory AS BF16 (CUDA-core
// decode, same bf16-rounded value) and the bf16 activation tile is staged into
// shared; each warp then runs nvcuda::wmma m16n16k16 bf16xbf16 -> f32 over the
// tile (mirrors a cutlass W4A16 dequant-GEMM). For the MoE grouped GEMM the P
// token-major pair-rows are first GROUPED BY EXPERT on device (counting sort +
// ragged per-BM-tile map) so each expert's weight is decoded once per row tile
// and fed to the tensor cores — buying both the ~BM-x weight-bandwidth reduction
// the per-pair kernel lacked and the tensor-core MMA. VT_NVFP4_WMMA=0 falls back
// to the CUDA-core tiled kernels (A/B toggle). Small-M decode keeps the naive
// kernel (graph-safe). Measured GB10: 1024-token prefill TTFT 14.4s -> 6.1s.
#include <cuda_bf16.h>
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
    throw std::runtime_error(std::string("vt cuda: matmul_nvfp4: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

// Tensor-core (bf16 WMMA) NVFP4 GEMM path toggle (A/B; default ON). Set
// VT_NVFP4_WMMA=0 to fall back to the CUDA-core tiled kernels (baseline).
bool WmmaEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NVFP4_WMMA");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

// M=1/decode-path 128-bit vectorized fp4 weight loads (A/B; default ON). Set
// VT_FP4_VEC=0 to force the byte-wise scalar reduction (bit-identical, slower).
bool Fp4VecEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FP4_VEC");
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

// E2M1 (fp4) magnitude LUT, indexed by the low 3 bits of the nibble; bit 3 is
// the sign. 1x-scaled — modelopt NVFP4, NOT the 2x GGUF kvalues_mxfp4 LUT.
__constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Decode one packed fp4 byte (two E2M1 codes) into its two dequanted bf16-rounded
// weight values, given the already-computed group scale gs = f8(scale)*scale2.
// Bit-for-bit vllm::DequantNvfp4ToBf16: low nibble = elem 2j, high = 2j+1; bit 3
// = sign; magnitude via the LUT; product rounded through bf16.
__device__ __forceinline__ void DecodeFp4Byte(uint8_t b, float gs, float& w_lo, float& w_hi) {
  const uint8_t low = b & 0x0Fu;
  const uint8_t high = b >> 4;
  const float lo_mag = kE2M1[low & 0x7u] * ((low & 0x8u) ? -1.0f : 1.0f);
  const float hi_mag = kE2M1[high & 0x7u] * ((high & 0x8u) ? -1.0f : 1.0f);
  w_lo = __bfloat162float(__float2bfloat16(lo_mag * gs));
  w_hi = __bfloat162float(__float2bfloat16(hi_mag * gs));
}

// Decode + accumulate the 4 packed bytes (8 fp4 elements) held in one 32-bit
// word, at activation offset `base`. Byte b (little-endian: (word>>8b)&0xff)
// carries elements base+2b (low nibble) and base+2b+1 (high nibble) -- the exact
// element order of the scalar loop, so this stays bit-identical.
template <typename Tin>
__device__ __forceinline__ void AccumFp4Word(float& acc, uint32_t word, float gs, const Tin* arow,
                                             int64_t base) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const uint8_t byte = static_cast<uint8_t>((word >> (b * 8)) & 0xffu);
    float w_lo, w_hi;
    DecodeFp4Byte(byte, gs, w_lo, w_hi);
    acc += Load(arow, base + 2 * b) * w_lo;
    acc += Load(arow, base + 2 * b + 1) * w_hi;
  }
}

// One output column's fp4 W4A16 dot product: acc = sum_k act[k] * dequant(w[k]).
// Bandwidth-optimized for the M=1/decode path -- the packed weight row is read
// with 128-bit (uint4 = 16 byte = 2 group-16) loads instead of byte-at-a-time,
// cutting the weight LDG count 16x and issuing full-width memory transactions.
// The accumulation order is unchanged (group ascending, byte ascending, low then
// high nibble) so the result is BIT-IDENTICAL to the scalar loop below; only the
// load width differs. Falls back to the scalar path when the row is not 16-byte
// aligned (k_dim not a multiple of 32) or has an odd group tail.
template <typename Tin>
__device__ __forceinline__ float Fp4ColDot(const uint8_t* prow, const uint8_t* srow,
                                            const Tin* arow, int64_t k_dim, float scale2,
                                            bool use_vec) {
  const int64_t groups = k_dim / 16;
  float acc = 0.0f;
  const bool aligned16 = use_vec && ((k_dim / 2) % 16 == 0) &&
                         ((reinterpret_cast<uintptr_t>(prow) & 0xf) == 0);
  if (aligned16) {
    const uint4* prow4 = reinterpret_cast<const uint4*>(prow);
    const int64_t cpairs = groups / 2;  // each uint4 spans two group-16 blocks
    for (int64_t c = 0; c < cpairs; ++c) {
      const uint4 pk = prow4[c];
      const float gs0 = F8E4M3ToF32Dev(srow[2 * c]) * scale2;      // group 2c   (bytes 0..7)
      const float gs1 = F8E4M3ToF32Dev(srow[2 * c + 1]) * scale2;  // group 2c+1 (bytes 8..15)
      const int64_t base0 = (2 * c) * 16;
      const int64_t base1 = (2 * c + 1) * 16;
      AccumFp4Word(acc, pk.x, gs0, arow, base0);      // group 2c   elems 0..7
      AccumFp4Word(acc, pk.y, gs0, arow, base0 + 8);  // group 2c   elems 8..15
      AccumFp4Word(acc, pk.z, gs1, arow, base1);      // group 2c+1 elems 0..7
      AccumFp4Word(acc, pk.w, gs1, arow, base1 + 8);  // group 2c+1 elems 8..15
    }
    if (groups & 1) {  // odd trailing group: scalar tail (same order)
      const int64_t g = groups - 1;
      const float gs = F8E4M3ToF32Dev(srow[g]) * scale2;
      const int64_t base = g * 16;
      for (int j = 0; j < 8; ++j) {
        float w_lo, w_hi;
        DecodeFp4Byte(prow[g * 8 + j], gs, w_lo, w_hi);
        acc += Load(arow, base + 2 * j) * w_lo;
        acc += Load(arow, base + 2 * j + 1) * w_hi;
      }
    }
    return acc;
  }
  for (int64_t g = 0; g < groups; ++g) {
    const float gs = F8E4M3ToF32Dev(srow[g]) * scale2;
    const int64_t base = g * 16;
    for (int j = 0; j < 8; ++j) {
      float w_lo, w_hi;
      DecodeFp4Byte(prow[g * 8 + j], gs, w_lo, w_hi);
      acc += Load(arow, base + 2 * j) * w_lo;
      acc += Load(arow, base + 2 * j + 1) * w_hi;
    }
  }
  return acc;
}

// Naive one-thread-per-output NVFP4 dequant-GEMM (grid.x over N, grid.y over M).
// Retained for SMALL m (decode): there the weight fits L2 so coalescing buys
// little, and this kernel avoids the tiled kernel's shared-mem/sync overhead —
// measured faster than the tiled path below for small m/N (GB10 nsys, 2026-07-04).
// The weight row is read 128-bit-wide via Fp4ColDot (bit-identical to the scalar
// reduction); this is the M2.8 decode-path bandwidth optimization.
template <typename Tin, typename Tout>
__global__ void MatmulNvfp4KernelNaive(Tout* out, const Tin* act, const uint8_t* packed,
                                       const uint8_t* scale, float scale2, int64_t m_rows,
                                       int64_t n_cols, int64_t k_dim, bool use_vec) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;

  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* prow = packed + n * packed_cols;
  const uint8_t* srow = scale + n * groups;
  const Tin* arow = act + m * k_dim;
  Store(out, m * n_cols + n, Fp4ColDot(prow, srow, arow, k_dim, scale2, use_vec));
}

// Shared-memory tiled NVFP4 dequant-GEMM. out[m,n] = sum_k act[m,k] *
// bf16(e2m1(w[n,k]) * group_scale[n,k/16]). A block owns output tile
// [row0..row0+BM, col0..col0+BN]; each thread owns a [TM x TN] register sub-tile.
// BK must be a multiple of 16 (group-16 scale alignment). blockDim.x must equal
// (BM/TM)*(BN/TN).
template <typename Tin, typename Tout, int BM, int BN, int BK, int TM, int TN>
__global__ void MatmulNvfp4Tiled(Tout* out, const Tin* act, const uint8_t* packed,
                                 const uint8_t* scale, float scale2, int64_t m_rows,
                                 int64_t n_cols, int64_t k_dim) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  __shared__ float As[BM * BK];  // activation tile [BM][BK]
  __shared__ float Ws[BN * BK];  // decoded weight tile [BN][BK]

  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  // This thread's output sub-tile origin within the block tile.
  const int t_row = (tid / (BN / TN)) * TM;
  const int t_col = (tid % (BN / TN)) * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    // Cooperative coalesced load of the activation tile [BM][BK]. Guard both the
    // M edge (row0+r >= M) and the K edge (kt+c >= K, partial last K-tile) with 0.
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK;
      const int c = idx % BK;
      const int64_t gr = row0 + r;
      As[idx] = (gr < m_rows && kt + c < k_dim) ? Load(act, gr * k_dim + kt + c) : 0.0f;
    }
    // Cooperative coalesced load + on-the-fly decode of the weight tile [BN][BK].
    // Iterate over PACKED BYTES (one byte -> two k elements); consecutive threads
    // read consecutive packed bytes of a weight row => coalesced fp4 weight reads.
    for (int idx = tid; idx < BN * (BK / 2); idx += kThreads) {
      const int nl = idx / (BK / 2);
      const int bc = idx % (BK / 2);  // packed-byte column within the K-tile
      const int kl = 2 * bc;          // element column within the K-tile
      const int64_t gn = col0 + nl;
      // K is a multiple of 16 (even), so a byte's two elements (kt+kl, kt+kl+1)
      // are both in-range iff kt+kl < K — one guard covers the whole byte.
      if (gn < n_cols && kt + kl < k_dim) {
        const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
        const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
        float w_lo, w_hi;
        DecodeFp4Byte(b, gs, w_lo, w_hi);
        Ws[nl * BK + kl] = w_lo;
        Ws[nl * BK + kl + 1] = w_hi;
      } else {
        Ws[nl * BK + kl] = 0.0f;
        Ws[nl * BK + kl + 1] = 0.0f;
      }
    }
    __syncthreads();

    // Register-blocked accumulate over the K-tile.
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

// --- Tensor-core (bf16 WMMA) NVFP4 W4A16 GEMM (M2.6 prefill unlock) ----------
// Same math as MatmulNvfp4Tiled (out[m,n] = sum_k act[m,k] * bf16(e2m1(w[n,k]) *
// group_scale[n,k/16])) but the K-reduction runs on Blackwell tensor cores. The
// fp4 weight tile is dequantized INTO SHARED as bf16 (CUDA-core dequant, exactly
// DecodeFp4Byte's bf16-rounded value) and the bf16 activation tile is staged into
// shared; each warp then runs wmma m16n16k16 bf16xbf16 -> f32 over the tile. This
// mirrors a cutlass W4A16 dequant-GEMM (block tile [BM x BN], warp tiling
// [WARPS_M x WARPS_N], K-loop over BK; dequant on CUDA cores, MMA on tensor
// cores). BF16 activations only — the f32-act large-M case keeps the CUDA-core
// tiled path (no test exercises f32-act large-M, and the model feeds bf16).
namespace wmma = nvcuda::wmma;

template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MatmulNvfp4Wmma(Tout* out, const __nv_bfloat16* act, const uint8_t* packed,
                                const uint8_t* scale, float scale2, int64_t m_rows,
                                int64_t n_cols, int64_t k_dim) {
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M;  // rows a warp owns
  constexpr int WNPER = BN / WARPS_N;  // cols a warp owns
  constexpr int MF = WMPER / 16;       // wmma M-fragments per warp
  constexpr int NF = WNPER / 16;       // wmma N-fragments per warp
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    // Stage the bf16 activation tile [BM][BK] (coalesced, act is K-contiguous).
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      const int64_t gr = row0 + r;
      As[idx] = (gr < m_rows && kt + c < k_dim) ? act[gr * k_dim + kt + c]
                                                : __float2bfloat16(0.0f);
    }
    // Decode the fp4 weight tile [BN][BK] into shared bf16 (coalesced packed-byte
    // reads; one byte -> two consecutive K elements), stored [BN][BK] so a
    // col_major wmma B-load with ldm=BK yields B[k,n] = weight[n,k].
    for (int idx = tid; idx < BN * (BK / 2); idx += kThreads) {
      const int nl = idx / (BK / 2), bc = idx % (BK / 2), kl = 2 * bc;
      const int64_t gn = col0 + nl;
      if (gn < n_cols && kt + kl < k_dim) {
        const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
        const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
        float w_lo, w_hi;
        DecodeFp4Byte(b, gs, w_lo, w_hi);
        Ws[nl * BK + kl] = __float2bfloat16(w_lo);
        Ws[nl * BK + kl + 1] = __float2bfloat16(w_hi);
      } else {
        Ws[nl * BK + kl] = __float2bfloat16(0.0f);
        Ws[nl * BK + kl + 1] = __float2bfloat16(0.0f);
      }
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

// Below this many activation rows the naive one-thread-per-output kernel wins
// (weight L2-resident, tiling overhead not amortized); at/above it the tiled
// register-blocked kernel wins (weight streamed, coalescing + reuse pay off).
// Measured on GB10 (nsys, 2026-07-04): decode m=1..2 favors naive by ~2-3x,
// prefill m>=128 favors tiled by ~2x (and kills the multi-100ms lm_head spikes).
constexpr int64_t kTileMinRows = 32;

template <typename Tin, typename Tout>
void Launch(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& packed,
            const Tensor& scale, float scale2, int64_t m, int64_t n, int64_t k) {
  if (m < kTileMinRows) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(m));
    MatmulNvfp4KernelNaive<Tin, Tout><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<Tin>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(), scale2, m, n,
        k, Fp4VecEnabled());
    Check(cudaGetLastError(), "kernel launch (naive)");
    return;
  }
  // BF16 activations at large M run on tensor cores (wmma bf16 -> f32). Tile
  // [BM x BN] = 64x64, BK=32, warps 2x2 (128 threads); each warp owns 32x32 =
  // 2x2 wmma m16n16k16 fragments. Dequant into shared bf16 stays on CUDA cores.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (WmmaEnabled()) {
    constexpr int BM = 64, BN = 64, BK = 32, WARPS_M = 2, WARPS_N = 2;
    constexpr int kThreads = WARPS_M * WARPS_N * 32;
    const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN),
                    static_cast<unsigned>((m + BM - 1) / BM));
    MatmulNvfp4Wmma<Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<__nv_bfloat16>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(),
        scale2, m, n, k);
    Check(cudaGetLastError(), "kernel launch (wmma)");
    return;
    }
  }
  // GB10-tuned tile: [BM x BN] = 64x64, BK=32, [TM x TN] = 4x4 => 256
  // threads/block. Weight tile is loaded once per block and reused across all BM
  // rows; coalesced packed-byte + scale reads feed the register-blocked accumulate.
  constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
  constexpr int kThreads = (BM / TM) * (BN / TN);
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
  MatmulNvfp4Tiled<Tin, Tout, BM, BN, BK, TM, TN><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), packed.Ptr<uint8_t>(), scale.Ptr<uint8_t>(), scale2, m, n,
      k);
  Check(cudaGetLastError(), "kernel launch (tiled)");
}

template <typename Tin>
void LaunchByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& packed,
                 const Tensor& scale, float scale2, int64_t m, int64_t n, int64_t k) {
  switch (out.dtype) {
    case DType::kF32:
      Launch<Tin, float>(s, out, act, packed, scale, scale2, m, n, k);
      break;
    case DType::kBF16:
      Launch<Tin, __nv_bfloat16>(s, out, act, packed, scale, scale2, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4: unsupported out dtype (f32/bf16 only)");
  }
}

// --- Fused MoE grouped NVFP4 GEMM (M2.4) --------------------------------------
// out[p, n] = sum_k act[row(p), k] * bf16(e2m1(W_e[n,k]) * gscale_e[n,k/16]),
// where e = expert_ids[p], W_e = packed_ptrs[e]/scale_ptrs[e]/scale2s[e], and
// row(p) = row_map ? row_map[p] : p. One block over (p rows, N tiles) replaces
// the per-expert loop of MatmulNvfp4 launches — the launch-count reduction is
// the win (GB10 is launch-overhead bound). The per-element decode is byte-for-
// byte MatmulNvfp4Kernel's, so each output row is bit-identical to running the
// single-expert kernel for that pair's expert.
// Naive grouped GEMM (one thread per output, grid-strided over pair rows).
// Retained for SMALL P (decode): weight L2-resident, coalescing buys little.
template <typename Tin, typename Tout>
__global__ void MoeGroupedGemmNvfp4KernelNaive(Tout* out, const Tin* act, const int32_t* expert_ids,
                                               const int32_t* row_map, const int64_t* packed_ptrs,
                                               const int64_t* scale_ptrs, const float* scale2s,
                                               int64_t p_rows, int64_t n_cols, int64_t k_dim,
                                               bool use_vec) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= n_cols) return;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
    const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
    const float scale2 = scale2s[e];
    const uint8_t* prow = packed + n * packed_cols;
    const uint8_t* srow = scale + n * groups;
    const Tin* arow = act + r * k_dim;
    Store(out, p * n_cols + n, Fp4ColDot(prow, srow, arow, k_dim, scale2, use_vec));
  }
}

// Tiled grouped GEMM: one block computes a [1 x BN] slice of a single pair row p
// (MoE pairs are token-major, NOT expert-sorted, so consecutive rows use
// different expert weights — no cross-row weight reuse to exploit). Within the
// block the activation row is loaded once per K-tile into shared and REUSED
// across the BN columns, and the fp4 weight tile [BN x BK] is decoded into shared
// with COALESCED packed-byte reads (the naive kernel had thread n stride a full
// weight row = uncoalesced). blockDim.x == BN (one thread per output column).
template <typename Tin, typename Tout, int BN, int BK>
__global__ void MoeGroupedGemmNvfp4Tiled(Tout* out, const Tin* act, const int32_t* expert_ids,
                                         const int32_t* row_map, const int64_t* packed_ptrs,
                                         const int64_t* scale_ptrs, const float* scale2s,
                                         int64_t p_rows, int64_t n_cols, int64_t k_dim) {
  __shared__ float As[BK];       // activation row slice [BK]
  __shared__ float Ws[BN * BK];  // decoded weight tile [BN][BK]

  const int tid = static_cast<int>(threadIdx.x);  // 0..BN-1 => output column
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t gc = col0 + tid;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;

  // Grid-stride over the pair rows: P = T*top_k can exceed the gridDim.y max.
  for (int64_t p = blockIdx.y; p < p_rows; p += gridDim.y) {
    const int64_t e = expert_ids[p];
    const int64_t r = row_map != nullptr ? row_map[p] : p;
    const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
    const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
    const float scale2 = scale2s[e];
    const Tin* arow = act + r * k_dim;

    float acc = 0.0f;
    for (int64_t kt = 0; kt < k_dim; kt += BK) {
      // Guard the K edge (kt+idx >= K, partial last K-tile) with 0.
      for (int idx = tid; idx < BK; idx += BN)
        As[idx] = (kt + idx < k_dim) ? Load(arow, kt + idx) : 0.0f;
      for (int idx = tid; idx < BN * (BK / 2); idx += BN) {
        const int nl = idx / (BK / 2);
        const int bc = idx % (BK / 2);
        const int kl = 2 * bc;
        const int64_t gn = col0 + nl;
        if (gn < n_cols && kt + kl < k_dim) {
          const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
          const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
          float w_lo, w_hi;
          DecodeFp4Byte(b, gs, w_lo, w_hi);
          Ws[nl * BK + kl] = w_lo;
          Ws[nl * BK + kl + 1] = w_hi;
        } else {
          Ws[nl * BK + kl] = 0.0f;
          Ws[nl * BK + kl + 1] = 0.0f;
        }
      }
      __syncthreads();
      if (gc < n_cols) {
#pragma unroll
        for (int k = 0; k < BK; ++k) acc += As[k] * Ws[tid * BK + k];
      }
      __syncthreads();
    }
    if (gc < n_cols) Store(out, p * n_cols + gc, acc);
  }
}

// --- Expert-grouped tensor-core (bf16 WMMA) MoE grouped GEMM (M2.6) ----------
// The MoE pairs are token-major (row p uses expert eids[p], NOT expert-sorted),
// so the per-pair tiled kernel above re-reads each expert's whole weight once per
// row (no cross-row reuse) AND cannot feed tensor cores (M=1 per weight). This
// path first GROUPS the P pair-rows by expert on device (counting sort), then
// runs a dense bf16 WMMA GEMM per expert over its contiguous row block — the
// weight is decoded once per BM-row tile (mirrors cutlass/Marlin grouped MoE:
// token-sorted activations + per-tile expert map, ragged tail masked). Grouping
// buys BOTH the ~BM-x weight-bandwidth reduction and the tensor-core MMA. Runs
// only for large P (prefill), which is never CUDA-graph-captured, so the
// stream-ordered scratch alloc below is safe; small-P decode keeps the naive
// kernel. Row bookkeeping over sorted positions:
//   sp[pos] = original pair index p (output row = out[sp[pos], :])
//   sr[pos] = source activation row = row_map ? row_map[p] : p
constexpr int kGroupBM = 64;

__global__ void MoeHistKernel(const int32_t* eids, int32_t* count, int64_t p_rows) {
  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p < p_rows) atomicAdd(&count[eids[p]], 1);
}

// Exclusive prefix sum of count[E] -> offset[E] (E small; single thread serial).
__global__ void MoeOffsetsKernel(const int32_t* count, int32_t* offset, int e_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int32_t acc = 0;
  for (int e = 0; e < e_count; ++e) {
    offset[e] = acc;
    acc += count[e];
  }
}

__global__ void MoeScatterKernel(const int32_t* eids, const int32_t* row_map,
                                 const int32_t* offset, int32_t* cursor, int32_t* sp, int32_t* sr,
                                 int64_t p_rows) {
  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= p_rows) return;
  const int e = eids[p];
  const int pos = offset[e] + atomicAdd(&cursor[e], 1);
  sp[pos] = static_cast<int32_t>(p);
  sr[pos] = row_map != nullptr ? row_map[p] : static_cast<int32_t>(p);
}

// Build the ragged per-BM-tile expert map (no tile crosses an expert boundary);
// pad the unused tail tiles with tile_rows=0 (grid.y is an upper bound). Single
// thread serial over E (small).
__global__ void MoeTileMapKernel(const int32_t* count, const int32_t* offset, int32_t* tile_expert,
                                 int32_t* tile_row0, int32_t* tile_rows, int e_count, int bm,
                                 int max_tiles) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int t = 0;
  for (int e = 0; e < e_count; ++e) {
    const int c = count[e], o = offset[e];
    for (int off = 0; off < c; off += bm) {
      tile_expert[t] = e;
      tile_row0[t] = o + off;
      tile_rows[t] = (c - off < bm) ? (c - off) : bm;
      ++t;
    }
  }
  for (; t < max_tiles; ++t) tile_rows[t] = 0;
}

template <typename Tin, typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MoeGroupedGemmNvfp4Wmma(Tout* out, const Tin* act, const int32_t* sp,
                                        const int32_t* sr, const int64_t* packed_ptrs,
                                        const int64_t* scale_ptrs, const float* scale2s,
                                        const int32_t* tile_expert, const int32_t* tile_row0,
                                        const int32_t* tile_rows, int64_t n_cols, int64_t k_dim) {
  const int rcount = tile_rows[blockIdx.y];
  if (rcount == 0) return;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M, WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16, NF = WNPER / 16;
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int e = tile_expert[blockIdx.y];
  const int row0 = tile_row0[blockIdx.y];
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;
  const int64_t packed_cols = k_dim / 2, groups = k_dim / 16;
  const auto* packed = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(packed_ptrs[e]));
  const auto* scale = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(scale_ptrs[e]));
  const float scale2 = scale2s[e];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    for (int idx = tid; idx < BM * BK; idx += kThreads) {
      const int r = idx / BK, c = idx % BK;
      __nv_bfloat16 v = __float2bfloat16(0.0f);
      if (r < rcount && kt + c < k_dim) {
        const int64_t srow = sr[row0 + r];
        v = static_cast<__nv_bfloat16>(act[srow * k_dim + kt + c]);
      }
      As[idx] = v;
    }
    for (int idx = tid; idx < BN * (BK / 2); idx += kThreads) {
      const int nl = idx / (BK / 2), bc = idx % (BK / 2), kl = 2 * bc;
      const int64_t gn = col0 + nl;
      if (gn < n_cols && kt + kl < k_dim) {
        const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
        const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]) * scale2;
        float w_lo, w_hi;
        DecodeFp4Byte(b, gs, w_lo, w_hi);
        Ws[nl * BK + kl] = __float2bfloat16(w_lo);
        Ws[nl * BK + kl + 1] = __float2bfloat16(w_hi);
      } else {
        Ws[nl * BK + kl] = __float2bfloat16(0.0f);
        Ws[nl * BK + kl + 1] = __float2bfloat16(0.0f);
      }
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
    const int64_t gc = col0 + c;
    if (r < rcount && gc < n_cols) Store(out, static_cast<int64_t>(sp[row0 + r]) * n_cols + gc,
                                         Cs[idx]);
  }
}

// Groups P pairs by expert on device then runs the WMMA grouped GEMM. Scratch is
// stream-ordered (prefill-only; not graph-captured).
template <typename Tin, typename Tout>
void LaunchGroupedWmma(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                       const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                       const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  constexpr int BM = kGroupBM, BN = 64, BK = 32, WARPS_M = 2, WARPS_N = 2;
  const int max_tiles = static_cast<int>((p + BM - 1) / BM + e_count);
  // Single scratch block, int32 sub-slices: count,offset,cursor [E]; sp,sr [P];
  // tile_expert,tile_row0,tile_rows [max_tiles].
  const int64_t n_i32 = 3 * e_count + 2 * p + 3 * max_tiles;
  int32_t* scratch = nullptr;
  Check(cudaMallocAsync(&scratch, static_cast<size_t>(n_i32) * sizeof(int32_t), s),
        "moe grouped scratch alloc");
  int32_t* count = scratch;
  int32_t* offset = count + e_count;
  int32_t* cursor = offset + e_count;
  int32_t* sp = cursor + e_count;
  int32_t* sr = sp + p;
  int32_t* tile_expert = sr + p;
  int32_t* tile_row0 = tile_expert + max_tiles;
  int32_t* tile_rows = tile_row0 + max_tiles;

  Check(cudaMemsetAsync(count, 0, static_cast<size_t>(e_count) * sizeof(int32_t), s), "memset count");
  Check(cudaMemsetAsync(cursor, 0, static_cast<size_t>(e_count) * sizeof(int32_t), s),
        "memset cursor");
  const int32_t* eids = expert_ids.Ptr<int32_t>();
  const int32_t* rmap = row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr;
  const int hb = 256;
  MoeHistKernel<<<static_cast<unsigned>((p + hb - 1) / hb), hb, 0, s>>>(eids, count, p);
  MoeOffsetsKernel<<<1, 1, 0, s>>>(count, offset, static_cast<int>(e_count));
  MoeScatterKernel<<<static_cast<unsigned>((p + hb - 1) / hb), hb, 0, s>>>(eids, rmap, offset,
                                                                          cursor, sp, sr, p);
  MoeTileMapKernel<<<1, 1, 0, s>>>(count, offset, tile_expert, tile_row0, tile_rows,
                                   static_cast<int>(e_count), BM, max_tiles);

  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>(max_tiles));
  MoeGroupedGemmNvfp4Wmma<Tin, Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), sp, sr, packed_ptrs.Ptr<int64_t>(),
      scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), tile_expert, tile_row0, tile_rows, n, k);
  Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (wmma)");
  Check(cudaFreeAsync(scratch, s), "moe grouped scratch free");
}

template <typename Tin, typename Tout>
void LaunchGrouped(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                   const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                   const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  const int64_t y = p < 65535 ? p : 65535;  // grid.y max; kernel strides over p
  if (p < kTileMinRows) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(y));
    MoeGroupedGemmNvfp4KernelNaive<Tin, Tout><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<Tin>(), expert_ids.Ptr<int32_t>(),
        row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, packed_ptrs.Ptr<int64_t>(),
        scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), p, n, k, Fp4VecEnabled());
    Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (naive)");
    return;
  }
  // Large-P (prefill) bf16: expert-grouped tensor-core (WMMA) path.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (WmmaEnabled()) {
      LaunchGroupedWmma<Tin, Tout>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                   scale2s, p, n, k, e_count);
      return;
    }
  }
  constexpr int BN = 128, BK = 32;  // 128 threads/block; weight tile 128x32
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>(y));
  MoeGroupedGemmNvfp4Tiled<Tin, Tout, BN, BK><<<grid, BN, 0, s>>>(
      out.Ptr<Tout>(), act.Ptr<Tin>(), expert_ids.Ptr<int32_t>(),
      row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, packed_ptrs.Ptr<int64_t>(),
      scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), p, n, k);
  Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (tiled)");
}

template <typename Tin>
void LaunchGroupedByOut(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                        const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                        const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  switch (out.dtype) {
    case DType::kF32:
      LaunchGrouped<Tin, float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k, e_count);
      break;
    case DType::kBF16:
      LaunchGrouped<Tin, __nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k, e_count);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_nvfp4: unsupported out dtype (f32/bf16 only)");
  }
}

void MoeGroupedGemmNvfp4KernelCuda(Queue& q, Tensor& out, const Tensor& act,
                                   const Tensor& expert_ids, const Tensor* row_map,
                                   const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                                   const Tensor& scale2s) {
  const int64_t p = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (p == 0 || n == 0) return;
  const int64_t e_count = packed_ptrs.shape[0];  // number of experts
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchGroupedByOut<float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k, e_count);
      break;
    case DType::kBF16:
      LaunchGroupedByOut<__nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k, e_count);
      break;
    default: VT_CHECK(false, "cuda moe_grouped_gemm_nvfp4: unsupported act dtype (f32/bf16 only)");
  }
}

void MatmulNvfp4KernelCuda(Queue& q, Tensor& out, const Tensor& act, const Tensor& weight_packed,
                           const Tensor& weight_scale, float weight_scale_2) {
  const int64_t m = act.shape[0], k = act.shape[1], n = weight_packed.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchByOut<float>(s, out, act, weight_packed, weight_scale, weight_scale_2, m, n, k);
      break;
    case DType::kBF16:
      LaunchByOut<__nv_bfloat16>(s, out, act, weight_packed, weight_scale, weight_scale_2, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4: unsupported act dtype (f32/bf16 only)");
  }
}

// Registers the CUDA NVFP4 dequant-GEMM during static init (table fill only, no
// CUDA calls — see cuda_ops.cu for the pre-main discipline rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fn>(&MatmulNvfp4KernelCuda)));
    RegisterOp(OpId::kMoeGroupedGemmNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmNvfp4Fn>(&MoeGroupedGemmNvfp4KernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
