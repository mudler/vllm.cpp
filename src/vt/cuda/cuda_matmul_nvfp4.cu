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

// Decode-specialized MoE grouped GEMM path toggle (M2.9; A/B, default ON). Set
// VT_MOE_DECODE=0 to force the prefill-tuned BM=64 WMMA tile for small-M decode
// (the pre-M2.9 behavior) for same-binary A/B. See LaunchGrouped for the gate.
bool MoeDecodeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_DECODE");
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

// Persistent scratch for the MoE expert-grouping counting-sort + ragged tile map
// (grown on demand, kept alive) so the decode path never pays the per-call
// cudaMallocAsync/cudaFreeAsync — at conc-8 the grouped GEMM runs ~80x/step, and
// the stream-ordered alloc/free serialize behind the (occupancy-bound) kernel.
// Mirrors the argmax persistent-scratch discipline (cuda_sample.cu). The model
// forward drives ONE CUDA stream, so consecutive grouped-GEMM calls are already
// serialized on the stream => reusing one buffer is race-free (the hist/scatter/
// tilemap kernels rewrite it every call). cudaMalloc (not async) fires once at
// warmup / on growth, never on the steady-state decode path.
int32_t* g_moe_scratch = nullptr;
size_t g_moe_scratch_cap = 0;  // capacity in int32 elements
int32_t* EnsureMoeScratch(int64_t elems) {
  if (static_cast<size_t>(elems) > g_moe_scratch_cap) {
    if (g_moe_scratch) cudaFree(g_moe_scratch);
    Check(cudaMalloc(&g_moe_scratch, static_cast<size_t>(elems) * sizeof(int32_t)),
          "moe grouped persistent scratch");
    g_moe_scratch_cap = static_cast<size_t>(elems);
  }
  return g_moe_scratch;
}

// Small-M decode gate for the grouped WMMA path: at/below this many (token,expert)
// pairs we run the decode-specialized BM=16 tile (mirrors vLLM fused_moe's
// BLOCK_SIZE_M=16 for small M — moe_align_block_size pads each expert's ragged
// token run to block_size, one block per (expert-segment x N-tile), NOT a full
// prefill tile per expert). conc-8 decode = 8 tok x top_k 8 = 64 pairs; conc-64
// still fits. Above it (prefill), the BM=64 tile stays byte-identical.
constexpr int64_t kMoeDecodeMaxP = 512;
constexpr int kMoeDecodeBM = 16;

// Groups P pairs by expert on device (the moe_align_block_size analog) then runs
// the WMMA grouped GEMM over the ragged per-tile expert map. Templated on the
// M-tile so the decode path can size the tile to the ~1-row-per-expert reality
// (BM=16, one warp of M) while prefill keeps the large-M BM=64 tile. Persistent
// scratch (no per-call malloc/free). Not graph-captured on either path.
template <typename Tin, typename Tout, int BM, int WARPS_M, int WARPS_N>
void LaunchGroupedWmma(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                       const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                       const Tensor& scale2s, int64_t p, int64_t n, int64_t k, int64_t e_count) {
  constexpr int BN = 64, BK = 32;
  const int max_tiles = static_cast<int>((p + BM - 1) / BM + e_count);
  // Single scratch block, int32 sub-slices: count,offset,cursor [E]; sp,sr [P];
  // tile_expert,tile_row0,tile_rows [max_tiles].
  const int64_t n_i32 = 3 * e_count + 2 * p + 3 * max_tiles;
  int32_t* scratch = EnsureMoeScratch(n_i32);
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
  // bf16: expert-grouped tensor-core (WMMA) path. Decode/small-M (p small) runs
  // the BM=16 decode tile (vLLM BLOCK_SIZE_M=16 for small M); prefill/large-M
  // keeps the byte-identical BM=64 tile.
  if constexpr (std::is_same_v<Tin, __nv_bfloat16>) {
    if (WmmaEnabled()) {
      if (MoeDecodeEnabled() && p <= kMoeDecodeMaxP) {
        LaunchGroupedWmma<Tin, Tout, kMoeDecodeBM, 1, 2>(s, out, act, expert_ids, row_map,
                                                         packed_ptrs, scale_ptrs, scale2s, p, n, k,
                                                         e_count);
      } else {
        LaunchGroupedWmma<Tin, Tout, kGroupBM, 2, 2>(s, out, act, expert_ids, row_map, packed_ptrs,
                                                     scale_ptrs, scale2s, p, n, k, e_count);
      }
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

// --- TRUE W4A4 (fp4 activations x fp4 weights) — the 27B path (notes §7) --------
// vt::ScaledFp4Quant (mirror vllm scaled_fp4_quant / cvt_warp_fp16_to_fp4) +
// vt::MatmulNvfp4Fp4 (mirror cutlass_scaled_fp4_mm_sm120a). The GEMM here is the
// CORRECTNESS-FIRST compute: both fp4 operands are dequantized to bf16 in shared
// and the K-reduction runs on the bf16 tensor cores (numerically the exact
// vllm::RunNvfp4Emulation result — what the true-W4A4 oracle golden should match),
// then the accumulator is scaled by the folded `alpha`. The native block-scaled
// fp4xfp4 MMA (Blackwell mxf4 tensor cores) is the throughput follow-up (notes
// §7.6 step 6); the bf16-dequant path is the token-exact fallback.

// IEEE fp8-e4m3fn RNE-saturating cast (hardware conversion == vllm F32ToF8E4M3).
__device__ __forceinline__ uint8_t F32ToFp8Dev(float f) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3));
}

// f32 -> E2M1 nibble (bit-matches vllm CastToFp4 + Fp4ToNibble). Input pre-scaled.
__device__ __forceinline__ uint8_t CastToFp4NibbleDev(float x) {
  const float a = fabsf(x);
  uint8_t idx;
  if (a <= 0.25f) idx = 0;
  else if (a < 0.75f) idx = 1;
  else if (a <= 1.25f) idx = 2;
  else if (a < 1.75f) idx = 3;
  else if (a <= 2.5f) idx = 4;
  else if (a < 3.5f) idx = 5;
  else if (a <= 5.0f) idx = 6;
  else idx = 7;
  if (idx == 0) return 0;
  return static_cast<uint8_t>((x < 0.0f ? 0x8u : 0x0u) | idx);
}

// Opt-in switch for the NATIVE block-scaled fp4xfp4 MMA + its matching
// fast-reciprocal activation quant (mirror of vllm's flashinfer-cutlass sm120a
// path). DEFAULT OFF: the default forward keeps the bf16-dequant WMMA GEMM +
// exact-division quant, which is byte-identical to the CPU RefScaledFp4Quant /
// RunNvfp4Emulation reference (and reproduces vllm's EMULATION token stream).
// Set VT_NVFP4_FP4_NATIVE=1 to select the native sm120a-mirror path (validated
// bit-exact vs the CPU reference; on the 27B near-tie it still lands on vllm's
// emulation token, not cutlass's — see notes §7.2 / the qwen27 gate).
inline bool NativeFp4MmaEnabled() {
  const char* e = std::getenv("VT_NVFP4_FP4_NATIVE");
  return e && e[0] == '1';
}

// ScaledFp4Quant: one thread per (row, 16-group). Emits packed fp4 [M,K/2] +
// fp8 block scale [M,K/16]. Reciprocal selected by approx_recip (notes §7.2).
// Tin f32/bf16.
template <typename Tin>
__global__ void ScaledFp4QuantKernel(uint8_t* packed, uint8_t* scale, const Tin* x,
                                     float input_global_scale, int64_t m_rows, int64_t k_dim,
                                     bool approx_recip) {
  const int64_t groups = k_dim / 16;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= m_rows * groups) return;
  const int64_t row = gid / groups;
  const int64_t g = gid % groups;
  const int64_t base = row * k_dim + g * 16;
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(Load(x, base + j)));
  float sc = input_global_scale * (vmax * (1.0f / 6.0f));
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  scale[row * groups + g] = sc8;  // LINEAR store; a separate SwizzleBlockscaleKernel reorders.
  // PART B (VT_SWIZZLE_IN_QUANT direct-emit, FOLLOW-UP — not yet shipped): store sc8
  // DIRECTLY at vLLM's swizzled offset here, eliminating the separate
  // SwizzleBlockscaleKernel pass. Verified bit-identical to
  // cuda_matmul_nvfp4_cutlass.cu SwizzleBlockscaleKernel:
  //   numKTiles = round_up(groups,4)/4;  Kp = numKTiles*4  (swizzled cols)
  //   mTileIdx=row>>7; outerMIdx=row&31; innerMIdx=(row>>5)&3; kTileIdx=g>>2; innerKIdx=g&3;
  //   sfOff = ((mTileIdx*numKTiles + kTileIdx)*32 + outerMIdx)*16 + innerMIdx*4 + innerKIdx;
  //   scale[sfOff] = sc8;   // scale buffer must be the swizzled [round_up(M,128), Kp] shape
  // (row&31 == (row%128)%32 and (row>>5)&3 == (row%128)/32 since 128 is a multiple of 32, so
  // this equals SwizzleBlockscaleKernel's ((((a0*(Kp/4)+a3)*32+a2)*4+a1)*4+a4) exactly.)
  // STRUCTURAL WORK REQUIRED (why it is a follow-up, not shipped here): the padding slots
  // rows[M,round_up(M,128)) / cols[groups,Kp) must be pre-zeroed (SwizzleBlockscaleKernel
  // zero-fills them) and this must be a SEPARATE quant mode — the linear scale is still
  // consumed by the non-cutlass WMMA MatmulNvfp4Fp4 path — so the ScaledFp4Quant op gains a
  // swizzled-output variant used ONLY on the cutlass fuse sites. GPU-validate against the gate.
  // outputScale = SFScaleVal / SFValue. In native mode (approx_recip) this uses
  // the fast approximate reciprocal, mirroring the NATIVE vllm cvt_warp_fp16_to_fp4
  // (nvfp4_utils.cuh:283-286 reciprocal_approximate_ftz) — the sub-bucket delta
  // the sm120a oracle uses. Default (emulation-grade) uses exact division, which
  // is byte-identical to the CPU RefScaledFp4Quant reference (notes §7.2).
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      float rcp;
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(sfv));
      out_scale = input_global_scale * rcp;
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo = fminf(fmaxf(Load(x, base + j) * out_scale, -6.0f), 6.0f);
    const float hi = fminf(fmaxf(Load(x, base + j + 1) * out_scale, -6.0f), 6.0f);
    packed[(base + j) / 2] =
        static_cast<uint8_t>(CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
}

void ScaledFp4QuantKernelCuda(Queue& q, Tensor& out_packed, Tensor& out_scale, const Tensor& x,
                              float input_global_scale_inv) {
  const int64_t m = x.shape[0], k = x.shape[1];
  if (m == 0 || k == 0) return;
  cudaStream_t s = AsStream(q);
  const int64_t total = m * (k / 16);
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  auto* pk = out_packed.Ptr<uint8_t>();
  auto* sc = out_scale.Ptr<uint8_t>();
  const bool approx = NativeFp4MmaEnabled();  // native path -> fast-reciprocal quant
  switch (x.dtype) {
    case DType::kF32:
      ScaledFp4QuantKernel<float><<<grid, kBlock, 0, s>>>(pk, sc, x.Ptr<float>(),
                                                          input_global_scale_inv, m, k, approx);
      break;
    case DType::kBF16:
      ScaledFp4QuantKernel<__nv_bfloat16><<<grid, kBlock, 0, s>>>(
          pk, sc, x.Ptr<__nv_bfloat16>(), input_global_scale_inv, m, k, approx);
      break;
    default: VT_CHECK(false, "cuda scaled_fp4_quant: unsupported x dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "scaled_fp4_quant kernel launch");
}

// FUSED silu-mul -> NVFP4 quant (mirror vllm ActivationQuantFusionPass /
// silu_and_mul_nvfp4_quant, act_quant_fusion.py:40 + activation_nvfp4_quant_fusion
// _kernels.cu). Removes the bf16 intermediate the unfused MoeSiluMul(->bf16) +
// ScaledFp4Quant writes+reads on the memory-bound prefill. One thread per
// (row, 16-group), same grid as ScaledFp4QuantKernel. gate/up are [M,I] (our
// two-input MoeSiluMul form, qwen3_5.cpp DenseMlpBlock). BIT-IDENTICAL to
// MoeSiluMul(->bf16) then ScaledFp4Quant: silu(gate)*up is ROUNDED THROUGH bf16
// (MoeSiluMul stores bf16) before quant, reusing the exact ScaledFp4Quant epilogue.
template <typename Tin>
__global__ void SiluMulFp4QuantKernel(uint8_t* packed, uint8_t* scale, const Tin* gate,
                                      const Tin* up, float input_global_scale, int64_t m_rows,
                                      int64_t i_dim, bool approx_recip) {
  const int64_t groups = i_dim / 16;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= m_rows * groups) return;
  const int64_t row = gid / groups;
  const int64_t g = gid % groups;
  const int64_t base = row * i_dim + g * 16;
  // silu(gate)*up, rounded through bf16 (== MoeSiluMul store to bf16, cuda_ops.cu:149).
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float gg = Load(gate, base + j);
    const float uu = Load(up, base + j);
    const float sm = (gg / (1.0f + expf(-gg))) * uu;
    v[j] = __bfloat162float(__float2bfloat16(sm));
  }
  // --- exact ScaledFp4QuantKernel epilogue (cuda_matmul_nvfp4.cu:960-989) over v ---
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) vmax = fmaxf(vmax, fabsf(v[j]));
  float sc = input_global_scale * (vmax * (1.0f / 6.0f));
  sc = fminf(fmaxf(sc, -448.0f), 448.0f);
  const uint8_t sc8 = F32ToFp8Dev(sc);
  scale[row * groups + g] = sc8;
  const float sfv = F8E4M3ToF32Dev(sc8);
  float out_scale = 0.0f;
  if (sfv != 0.0f) {
    if (approx_recip) {
      float rcp;
      asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(sfv));
      out_scale = input_global_scale * rcp;
    } else {
      out_scale = input_global_scale / sfv;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j += 2) {
    const float lo = fminf(fmaxf(v[j] * out_scale, -6.0f), 6.0f);
    const float hi = fminf(fmaxf(v[j + 1] * out_scale, -6.0f), 6.0f);
    packed[(base + j) / 2] =
        static_cast<uint8_t>(CastToFp4NibbleDev(lo) | (CastToFp4NibbleDev(hi) << 4));
  }
}

void SiluMulFp4QuantKernelCuda(Queue& q, Tensor& out_packed, Tensor& out_scale,
                               const Tensor& gate, const Tensor& up,
                               float input_global_scale_inv) {
  const int64_t m = gate.shape[0], i = gate.shape[1];
  if (m == 0 || i == 0) return;
  cudaStream_t s = AsStream(q);
  const int64_t total = m * (i / 16);
  constexpr int kBlock = 256;
  const dim3 grid(static_cast<unsigned>((total + kBlock - 1) / kBlock));
  auto* pk = out_packed.Ptr<uint8_t>();
  auto* sc = out_scale.Ptr<uint8_t>();
  const bool approx = NativeFp4MmaEnabled();  // pair fused/unfused reciprocal choice
  switch (gate.dtype) {
    case DType::kF32:
      SiluMulFp4QuantKernel<float><<<grid, kBlock, 0, s>>>(
          pk, sc, gate.Ptr<float>(), up.Ptr<float>(), input_global_scale_inv, m, i, approx);
      break;
    case DType::kBF16:
      SiluMulFp4QuantKernel<__nv_bfloat16><<<grid, kBlock, 0, s>>>(
          pk, sc, gate.Ptr<__nv_bfloat16>(), up.Ptr<__nv_bfloat16>(), input_global_scale_inv, m,
          i, approx);
      break;
    default: VT_CHECK(false, "cuda silu_mul_fp4_quant: unsupported dtype (f32/bf16 only)");
  }
  Check(cudaGetLastError(), "silu_mul_fp4_quant kernel launch");
}

// Naive fp4xfp4 GEMM (small-M / decode). out[m,n] = alpha * sum_k
// (a_fp4[m,k]·f8(a_scale[m,k/16]))·(b_fp4[n,k]·f8(b_scale[n,k/16])). One thread
// per output; decode both operands on the fly. Bit-consistent with the WMMA path
// (same per-element decode), only K-order differs.
template <typename Tout>
__global__ void MatmulNvfp4Fp4Naive(Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
                                    const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
                                    int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* aprow = a_packed + m * packed_cols;
  const uint8_t* asrow = a_scale + m * groups;
  const uint8_t* bprow = b_packed + n * packed_cols;
  const uint8_t* bsrow = b_scale + n * groups;
  float acc = 0.0f;
  for (int64_t g = 0; g < groups; ++g) {
    const float asf = F8E4M3ToF32Dev(asrow[g]);
    const float bsf = F8E4M3ToF32Dev(bsrow[g]);
    for (int j = 0; j < 8; ++j) {
      const uint8_t ab = aprow[g * 8 + j];
      const uint8_t bb = bprow[g * 8 + j];
      const float a_lo = kE2M1[ab & 0x7u] * ((ab & 0x8u) ? -1.0f : 1.0f) * asf;
      const float a_hi = kE2M1[(ab >> 4) & 0x7u] * ((ab & 0x80u) ? -1.0f : 1.0f) * asf;
      const float b_lo = kE2M1[bb & 0x7u] * ((bb & 0x8u) ? -1.0f : 1.0f) * bsf;
      const float b_hi = kE2M1[(bb >> 4) & 0x7u] * ((bb & 0x80u) ? -1.0f : 1.0f) * bsf;
      acc += a_lo * b_lo + a_hi * b_hi;
    }
  }
  Store(out, m * n_cols + n, alpha * acc);
}

// Decode a fp4 tile [ROWS][BK] (packed [ROWS][K/2], fp8 group scales [ROWS][K/16])
// into shared bf16 `dst` [ROWS][BK]. `outer0` = the tile's first row in the full
// [ROWS_total][K] operand; used by both the A (M) and B (N) operands (symmetric).
template <int BK>
__device__ __forceinline__ void DecodeFp4TileToShared(__nv_bfloat16* dst, const uint8_t* packed,
                                                      const uint8_t* scale, int64_t outer0,
                                                      int64_t outer_max, int64_t kt, int64_t k_dim,
                                                      int tid, int kThreads, int tile_rows) {
  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  for (int idx = tid; idx < tile_rows * (BK / 2); idx += kThreads) {
    const int nl = idx / (BK / 2), bc = idx % (BK / 2), kl = 2 * bc;
    const int64_t gn = outer0 + nl;
    if (gn < outer_max && kt + kl < k_dim) {
      const uint8_t b = packed[gn * packed_cols + kt / 2 + bc];
      const float gs = F8E4M3ToF32Dev(scale[gn * groups + (kt + kl) / 16]);
      float w_lo, w_hi;
      DecodeFp4Byte(b, gs, w_lo, w_hi);
      dst[nl * BK + kl] = __float2bfloat16(w_lo);
      dst[nl * BK + kl + 1] = __float2bfloat16(w_hi);
    } else {
      dst[nl * BK + kl] = __float2bfloat16(0.0f);
      dst[nl * BK + kl + 1] = __float2bfloat16(0.0f);
    }
  }
}

// Tensor-core (bf16 WMMA) fp4xfp4 GEMM: decode BOTH operands to bf16 in shared,
// run wmma m16n16k16, scale the accumulator by alpha at store. Same tiling as
// MatmulNvfp4Wmma; the only additions are the A-operand fp4 decode and the alpha.
template <typename Tout, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void MatmulNvfp4Fp4Wmma(Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
                                   const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
                                   int64_t m_rows, int64_t n_cols, int64_t k_dim) {
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  constexpr int WMPER = BM / WARPS_M, WNPER = BN / WARPS_N;
  constexpr int MF = WMPER / 16, NF = WNPER / 16;
  __shared__ __nv_bfloat16 As[BM * BK];
  __shared__ __nv_bfloat16 Ws[BN * BK];
  __shared__ float Cs[BM * BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32, wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int64_t row0 = static_cast<int64_t>(blockIdx.y) * BM;
  const int64_t col0 = static_cast<int64_t>(blockIdx.x) * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MF][NF];
#pragma unroll
  for (int mi = 0; mi < MF; ++mi)
#pragma unroll
    for (int ni = 0; ni < NF; ++ni) wmma::fill_fragment(acc[mi][ni], 0.0f);

  for (int64_t kt = 0; kt < k_dim; kt += BK) {
    DecodeFp4TileToShared<BK>(As, a_packed, a_scale, row0, m_rows, kt, k_dim, tid, kThreads, BM);
    DecodeFp4TileToShared<BK>(Ws, b_packed, b_scale, col0, n_cols, kt, k_dim, tid, kThreads, BN);
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
    if (gr < m_rows && gc < n_cols) Store(out, gr * n_cols + gc, alpha * Cs[idx]);
  }
}

// --- NATIVE block-scaled fp4xfp4 MMA (Blackwell sm120a mxf4nvf4 tensor cores) --
// Mirrors vllm's cutlass_scaled_fp4_mm_sm120a: the fp4 operands and their fp8
// (e4m3) group-16 block scales are fed DIRECTLY to the block-scaled tensor-core
// MMA (no bf16 dequant), fp32 accumulate, then x alpha. This is the true W4A4
// GEMM the 27B oracle golden was captured with (native != bf16-dequant on
// near-ties). Only compiled when the TU is built for the architecture-specific
// sm_120a/sm_121a target (__CUDA_ARCH_SPECIFIC__); the host gates the launch on
// the VT_FP4_MMA_SM120A build define so non-a builds keep the bf16 path.
//
// The per-thread operand/scale layout was validated device-vs-CPU bit-exact on
// GB10 sm_121a (m16n8k64 e2m1, scale_vec::4X ue4m3):
//   lane: g=lane/4 (0..7), t=lane%4 (0..3)
//   A frags a0(row g, k=t*8+j) a1(row g+8,..) a2(row g,32+t*8+j) a3(row g+8,..)
//   B frags b0(n g, k=t*8+j)   b1(n g,32+t*8+j);  nibble j -> k-elem j
//   D       d0(g,t*2) d1(g,t*2+1) d2(g+8,t*2) d3(g+8,t*2+1)
//   A scale: row r held in lane (r%8)*4 + (r>=8?1:0), byte b = k-block b
//   B scale: col n held in lane n*4,                  byte b = k-block b
__device__ __forceinline__ uint8_t GetNib(const uint8_t* p, int64_t row, int64_t col, int64_t k) {
  const uint8_t byte = p[row * (k / 2) + col / 2];
  return (col & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xFu);
}

template <typename Tout>
__global__ void MatmulNvfp4Fp4Native(Tout* out, const uint8_t* a_packed, const uint8_t* a_scale,
                                     const uint8_t* b_packed, const uint8_t* b_scale, float alpha,
                                     int64_t M, int64_t N, int64_t K) {
  const int lane = static_cast<int>(threadIdx.x);
  const int g = lane / 4, t = lane % 4;
  const int64_t m0 = static_cast<int64_t>(blockIdx.y) * 16;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * 8;
  const int64_t groups = K / 16;
  float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#if defined(__CUDA_ARCH_SPECIFIC__)
  for (int64_t k0 = 0; k0 < K; k0 += 64) {
    const int64_t rA = m0 + g, rA8 = m0 + g + 8, rB = n0 + g;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int64_t ka = k0 + t * 8 + j, kb = k0 + 32 + t * 8 + j;
      if (rA < M) {
        if (ka < K) a0 |= static_cast<uint32_t>(GetNib(a_packed, rA, ka, K)) << (4 * j);
        if (kb < K) a2 |= static_cast<uint32_t>(GetNib(a_packed, rA, kb, K)) << (4 * j);
      }
      if (rA8 < M) {
        if (ka < K) a1 |= static_cast<uint32_t>(GetNib(a_packed, rA8, ka, K)) << (4 * j);
        if (kb < K) a3 |= static_cast<uint32_t>(GetNib(a_packed, rA8, kb, K)) << (4 * j);
      }
      if (rB < N) {
        if (ka < K) b0 |= static_cast<uint32_t>(GetNib(b_packed, rB, ka, K)) << (4 * j);
        if (kb < K) b1 |= static_cast<uint32_t>(GetNib(b_packed, rB, kb, K)) << (4 * j);
      }
    }
    // 1.0 in fp8-e4m3 = 0x38: harmless default for unused/tail scale bytes.
    uint32_t sfa = 0x38383838u, sfb = 0x38383838u;
    const int64_t scaleRowA = (t == 0) ? (m0 + g) : (t == 1 ? (m0 + g + 8) : -1);
    if (scaleRowA >= 0 && scaleRowA < M) {
      uint32_t v = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const int64_t gc = k0 / 16 + b;
        const uint8_t sv = (gc < groups) ? a_scale[scaleRowA * groups + gc] : 0x38u;
        v |= static_cast<uint32_t>(sv) << (8 * b);
      }
      sfa = v;
    }
    if (t == 0) {
      const int64_t cB = n0 + g;
      if (cB < N) {
        uint32_t v = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int64_t gc = k0 / 16 + b;
          const uint8_t sv = (gc < groups) ? b_scale[cB * groups + gc] : 0x38u;
          v |= static_cast<uint32_t>(sv) << (8 * b);
        }
        sfb = v;
      }
    }
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {%15, %16}, %17, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa), "n"(0), "n"(0), "r"(sfb), "n"(0), "n"(0));
  }
#else
  (void)a_packed; (void)a_scale; (void)b_packed; (void)b_scale; (void)groups; (void)t; (void)g;
#endif
  const int64_t r = m0 + g, r8 = m0 + g + 8, c0 = n0 + t * 2, c1 = n0 + t * 2 + 1;
  if (r < M && c0 < N) Store(out, r * N + c0, alpha * d0);
  if (r < M && c1 < N) Store(out, r * N + c1, alpha * d1);
  if (r8 < M && c0 < N) Store(out, r8 * N + c0, alpha * d2);
  if (r8 < M && c1 < N) Store(out, r8 * N + c1, alpha * d3);
}

template <typename Tout>
void LaunchFp4Fp4(cudaStream_t s, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                  const Tensor& b_packed, const Tensor& b_scale, float alpha, int64_t m, int64_t n,
                  int64_t k) {
  auto* ap = a_packed.Ptr<uint8_t>();
  auto* as = a_scale.Ptr<uint8_t>();
  auto* bp = b_packed.Ptr<uint8_t>();
  auto* bs = b_scale.Ptr<uint8_t>();
#if defined(VT_FP4_MMA_SM120A)
  if (NativeFp4MmaEnabled()) {
    const dim3 grid(static_cast<unsigned>((n + 7) / 8), static_cast<unsigned>((m + 15) / 16));
    MatmulNvfp4Fp4Native<Tout><<<grid, 32, 0, s>>>(out.Ptr<Tout>(), ap, as, bp, bs, alpha, m, n, k);
    Check(cudaGetLastError(), "matmul_nvfp4_fp4 kernel launch (native sm120a)");
    return;
  }
#endif
  if (m < kTileMinRows || !WmmaEnabled()) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(m));
    MatmulNvfp4Fp4Naive<Tout><<<grid, kBlock, 0, s>>>(out.Ptr<Tout>(), ap, as, bp, bs, alpha, m, n,
                                                      k);
    Check(cudaGetLastError(), "matmul_nvfp4_fp4 kernel launch (naive)");
    return;
  }
  constexpr int BM = 64, BN = 64, BK = 32, WARPS_M = 2, WARPS_N = 2;
  constexpr int kThreads = WARPS_M * WARPS_N * 32;
  const dim3 grid(static_cast<unsigned>((n + BN - 1) / BN), static_cast<unsigned>((m + BM - 1) / BM));
  MatmulNvfp4Fp4Wmma<Tout, BM, BN, BK, WARPS_M, WARPS_N><<<grid, kThreads, 0, s>>>(
      out.Ptr<Tout>(), ap, as, bp, bs, alpha, m, n, k);
  Check(cudaGetLastError(), "matmul_nvfp4_fp4 kernel launch (wmma)");
}

void MatmulNvfp4Fp4KernelCuda(Queue& q, Tensor& out, const Tensor& a_packed, const Tensor& a_scale,
                              const Tensor& b_packed, const Tensor& b_scale, float alpha) {
  const int64_t m = a_packed.shape[0], k = a_packed.shape[1] * 2, n = b_packed.shape[0];
  if (m == 0 || n == 0) return;
  cudaStream_t s = AsStream(q);
  switch (out.dtype) {
    case DType::kF32:
      LaunchFp4Fp4<float>(s, out, a_packed, a_scale, b_packed, b_scale, alpha, m, n, k);
      break;
    case DType::kBF16:
      LaunchFp4Fp4<__nv_bfloat16>(s, out, a_packed, a_scale, b_packed, b_scale, alpha, m, n, k);
      break;
    default: VT_CHECK(false, "cuda matmul_nvfp4_fp4: unsupported out dtype (f32/bf16 only)");
  }
}

// Registers the CUDA NVFP4 dequant-GEMM during static init (table fill only, no
// CUDA calls — see cuda_ops.cu for the pre-main discipline rationale).
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fn>(&MatmulNvfp4KernelCuda)));
    RegisterOp(OpId::kScaledFp4Quant, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ScaledFp4QuantFn>(&ScaledFp4QuantKernelCuda)));
    RegisterOp(OpId::kSiluMulFp4Quant, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<SiluMulFp4QuantFn>(&SiluMulFp4QuantKernelCuda)));
    RegisterOp(OpId::kMatmulNvfp4Fp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MatmulNvfp4Fp4Fn>(&MatmulNvfp4Fp4KernelCuda)));
    RegisterOp(OpId::kMoeGroupedGemmNvfp4, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<MoeGroupedGemmNvfp4Fn>(&MoeGroupedGemmNvfp4KernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
