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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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

// Naive one-thread-per-output NVFP4 dequant-GEMM (grid.x over N, grid.y over M).
// Retained for SMALL m (decode): there the weight fits L2 so coalescing buys
// little, and this kernel avoids the tiled kernel's shared-mem/sync overhead —
// measured faster than the tiled path below for small m/N (GB10 nsys, 2026-07-04).
template <typename Tin, typename Tout>
__global__ void MatmulNvfp4KernelNaive(Tout* out, const Tin* act, const uint8_t* packed,
                                       const uint8_t* scale, float scale2, int64_t m_rows,
                                       int64_t n_cols, int64_t k_dim) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = blockIdx.y;
  if (n >= n_cols || m >= m_rows) return;

  const int64_t packed_cols = k_dim / 2;
  const int64_t groups = k_dim / 16;
  const uint8_t* prow = packed + n * packed_cols;
  const uint8_t* srow = scale + n * groups;
  const Tin* arow = act + m * k_dim;

  float acc = 0.0f;
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
  Store(out, m * n_cols + n, acc);
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
        k);
    Check(cudaGetLastError(), "kernel launch (naive)");
    return;
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
                                               int64_t p_rows, int64_t n_cols, int64_t k_dim) {
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
    float acc = 0.0f;
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
    Store(out, p * n_cols + n, acc);
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

template <typename Tin, typename Tout>
void LaunchGrouped(cudaStream_t s, Tensor& out, const Tensor& act, const Tensor& expert_ids,
                   const Tensor* row_map, const Tensor& packed_ptrs, const Tensor& scale_ptrs,
                   const Tensor& scale2s, int64_t p, int64_t n, int64_t k) {
  const int64_t y = p < 65535 ? p : 65535;  // grid.y max; kernel strides over p
  if (p < kTileMinRows) {
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock), static_cast<unsigned>(y));
    MoeGroupedGemmNvfp4KernelNaive<Tin, Tout><<<grid, kBlock, 0, s>>>(
        out.Ptr<Tout>(), act.Ptr<Tin>(), expert_ids.Ptr<int32_t>(),
        row_map != nullptr ? row_map->Ptr<int32_t>() : nullptr, packed_ptrs.Ptr<int64_t>(),
        scale_ptrs.Ptr<int64_t>(), scale2s.Ptr<float>(), p, n, k);
    Check(cudaGetLastError(), "moe_grouped_gemm_nvfp4 kernel launch (naive)");
    return;
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
                        const Tensor& scale2s, int64_t p, int64_t n, int64_t k) {
  switch (out.dtype) {
    case DType::kF32:
      LaunchGrouped<Tin, float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k);
      break;
    case DType::kBF16:
      LaunchGrouped<Tin, __nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k);
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
  cudaStream_t s = AsStream(q);
  switch (act.dtype) {
    case DType::kF32:
      LaunchGroupedByOut<float>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs, scale2s,
                                p, n, k);
      break;
    case DType::kBF16:
      LaunchGroupedByOut<__nv_bfloat16>(s, out, act, expert_ids, row_map, packed_ptrs, scale_ptrs,
                                        scale2s, p, n, k);
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
