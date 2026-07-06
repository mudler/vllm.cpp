// Ported from: vllm/v1/attention/backends/flash_attn.py @ e24d1b24
//   (FlashAttentionImpl.forward SEMANTICS: causal GQA softmax over the paged K/V,
//    softmax_scale = self.scale, cu_seqlens_q = query_start_loc, seqused_k =
//    seq_lens, block_table = block_table_tensor, window_size = None for the
//    qwen35 full-attn layers → plain causal). Cache READ is the NHD layout
//    FlashAttentionBackend::get_kv_cache_shape allocates, indexed by TENSOR
//    STRIDES (the two dim-1 unbind slices; block stride 2*bs*H*D) — NOT cpu_attn's
//    HND arithmetic (M1.6 Task-3 layout trap).
//
// TWO code paths, one op (M2.4 prefill flash rewrite):
//   * DECODE (num_tokens == num_reqs, i.e. every request has query_len 1): the
//     M1.6 correctness-grade kernel — one block per (query token, q-head),
//     block-cooperative online softmax. Fully device-side, NO host reads →
//     stays CUDA-graph capturable (decode is graph-captured; prefill is not).
//     Kept byte-for-byte to avoid regressing the decode latency path.
//   * PREFILL (num_tokens > num_reqs): a FlashAttention-style tiled kernel —
//     one block per (query-token tile, q-head); warp-per-query-row online
//     softmax with shared-memory K/V tiles reused across the BM rows of the
//     tile (bandwidth amortization), warp-shuffle Q·Kᵀ (no per-key block sync).
//     GQA (hq/num_kv_heads query heads per KV head) reuse is L2-served across
//     the adjacent per-head blocks. Mirrors flash_attn_varlen_func's online
//     softmax; algebraically identical to the decode kernel / CPU reference.
//
// Prefill is never CUDA-graph-captured (see cuda_matmul_nvfp4.cu), so the
// launcher may read query_start_loc D2H to build per-request query tiles.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <mma.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "vt/ops.h"

namespace vt::cuda {
namespace {

constexpr int kPagedBlock = 256;

// Prefill flash tiling. BM query rows (warps) per block; BN keys per K/V tile.
// head_dim <= kMaxEpl*32 = 256 (the gate models' head_dim) uses the flash path;
// larger head_dim falls back to the block-cooperative kernel (still correct).
constexpr int kBM = 16;     // query rows per block (warps)
constexpr int kBN = 32;     // keys per shared-memory K/V tile
constexpr int kMaxEpl = 8;  // elems-per-lane cap → head_dim <= 256

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

__device__ inline float Load(const float* p, int64_t i) { return p[i]; }
__device__ inline float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ inline void Store(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void Store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

// ===========================================================================
// DECODE path (M1.6, unchanged): one block per (query token, q-head). The block
// scans query_start_loc to find token t's request r, derives the absolute
// position p, and streams keys 0..p (causal) from the paged cache with a
// block-cooperative online (flash) softmax. Fully device-side / graph-safe.
// ===========================================================================
template <typename TQ, typename TKV, typename Tout>
__global__ void PagedAttentionKernel(Tout* out, const TQ* query, const TKV* k_cache,
                                     const TKV* v_cache, const int32_t* block_table,
                                     const int32_t* seq_lens, const int32_t* query_start_loc,
                                     int64_t num_reqs, int64_t hq, int64_t num_kv_heads, int64_t d,
                                     int64_t block_size, int64_t bt_row, int64_t bt_col,
                                     int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                     int64_t vc_pg, int64_t vc_hd, float scale, bool causal) {
  const int64_t t = blockIdx.x;  // global query-token index
  const int64_t h = blockIdx.y;  // q-head
  // Find request r with query_start_loc[r] <= t < query_start_loc[r+1].
  int64_t r = -1, q0 = 0, q1 = 0;
  for (int64_t rr = 0; rr < num_reqs; ++rr) {
    const int64_t a = query_start_loc[rr], b = query_start_loc[rr + 1];
    if (t >= a && t < b) {
      r = rr;
      q0 = a;
      q1 = b;
      break;
    }
  }
  if (r < 0) return;  // padding token beyond the last request

  const int64_t query_len = q1 - q0;
  const int64_t seqlen = seq_lens[r];
  const int64_t context = seqlen - query_len;
  const int64_t p = context + (t - q0);
  const int64_t jmax = causal ? p : seqlen - 1;
  const int64_t g = h / (hq / num_kv_heads);
  const int64_t qoff = (t * hq + h) * d;

  extern __shared__ float smem[];
  float* acc = smem;      // [d] running output accumulator
  float* red = smem + d;  // [blockDim.x] reduction scratch
  __shared__ float s_score, s_m, s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) acc[e] = 0.0f;
  if (threadIdx.x == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int64_t j = 0; j <= jmax; ++j) {
    const int64_t blk = block_table[r * bt_row + (j / block_size) * bt_col];
    const int64_t off = j % block_size;
    const int64_t kbase = blk * kc_blk + off * kc_pg + g * kc_hd;
    float part = 0.0f;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      part += Load(query, qoff + e) * Load(k_cache, kbase + e);
    red[threadIdx.x] = part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) s_score = red[0] * scale;
    __syncthreads();

    const float s = s_score;
    const float m_new = fmaxf(s_m, s);
    const float corr = expf(s_m - m_new);  // 0 on the first key (s_m == -inf)
    const float pw = expf(s - m_new);
    const int64_t vbase = blk * vc_blk + off * vc_pg + g * vc_hd;
    for (int64_t e = threadIdx.x; e < d; e += blockDim.x)
      acc[e] = acc[e] * corr + pw * Load(v_cache, vbase + e);
    __syncthreads();
    if (threadIdx.x == 0) {
      s_l = s_l * corr + pw;
      s_m = m_new;
    }
    __syncthreads();
  }

  const float inv = 1.0f / s_l;
  for (int64_t e = threadIdx.x; e < d; e += blockDim.x) Store(out, qoff + e, acc[e] * inv);
}

// ===========================================================================
// PREFILL path: FlashAttention-style tiled kernel.
//   grid  = (num_tiles, hq); block = dim3(32, BM) → BM warps, one query row each.
//   tiles[i] = (request r, first local query row) — never crosses a request.
// Each warp owns one query row (D distributed across 32 lanes, kMaxEpl per lane);
// all BM warps share the K/V tile staged in shared memory (loaded once per tile,
// reused across the rows → bandwidth amortization). Per key: warp-shuffle dot,
// online-softmax rescale of the register accumulator. No per-key block sync.
// ===========================================================================
template <typename TQ, typename TKV, typename Tout>
__global__ void PagedFlashKernel(Tout* out, const TQ* query, const TKV* k_cache,
                                 const TKV* v_cache, const int32_t* block_table,
                                 const int32_t* seq_lens, const int32_t* query_start_loc,
                                 const int2* tiles, int num_tiles, int hq, int num_kv_heads, int d,
                                 int block_size, int64_t bt_row, int64_t bt_col, int64_t kc_blk,
                                 int64_t kc_pg, int64_t kc_hd, int64_t vc_blk, int64_t vc_pg,
                                 int64_t vc_hd, float scale, bool causal, int bn) {
  const int tile_idx = blockIdx.x;
  const int h = blockIdx.y;  // q-head
  if (tile_idx >= num_tiles) return;

  const int2 td = tiles[tile_idx];
  const int r = td.x;       // request
  const int local0 = td.y;  // first local query row in this tile
  const int q0 = query_start_loc[r];
  const int q1 = query_start_loc[r + 1];
  const int qlen = q1 - q0;
  const int seqlen = seq_lens[r];
  const int context = seqlen - qlen;
  const int g = h / (hq / num_kv_heads);

  const int warp = threadIdx.y;  // 0..BM-1 → query row
  const int lane = threadIdx.x;  // 0..31
  const int bm = blockDim.y;
  const int local_row = local0 + warp;
  const int t = q0 + local_row;          // global query token
  const bool active = local_row < qlen;  // valid query row?
  const int p = context + local_row;     // absolute position (if active)
  const int my_jmax = causal ? p : (seqlen - 1);

  const int epl = (d + 31) / 32;  // elems per lane (<= kMaxEpl)
  float q_reg[kMaxEpl];
  float o_reg[kMaxEpl];
#pragma unroll
  for (int i = 0; i < kMaxEpl; ++i) {
    q_reg[i] = 0.0f;
    o_reg[i] = 0.0f;
  }
  if (active) {
    const int64_t qoff = (static_cast<int64_t>(t) * hq + h) * d;
    for (int i = 0; i < epl; ++i) {
      const int e = lane + 32 * i;
      if (e < d) q_reg[i] = Load(query, qoff + e);
    }
  }
  float m = -CUDART_INF_F, l = 0.0f;

  // Iterate K/V tiles up to the LAST active row's jmax so every warp participates
  // in each shared-memory load barrier (rows with smaller jmax process a prefix).
  // local0 < qlen is guaranteed by the host tile builder.
  const int last_row = min(local0 + bm, qlen) - 1;
  const int block_jmax = causal ? (context + last_row) : (seqlen - 1);

  extern __shared__ float smem[];
  float* ksm = smem;           // [bn * d]
  float* vsm = smem + bn * d;  // [bn * d]
  const int nthreads = bm * 32;
  const int tid = warp * 32 + lane;

  for (int j0 = 0; j0 <= block_jmax; j0 += bn) {
    const int tile_keys = min(bn, block_jmax - j0 + 1);
    // Cooperative K/V tile load: [tile_keys, d] → shared memory (as f32).
    for (int idx = tid; idx < tile_keys * d; idx += nthreads) {
      const int kk = idx / d, ee = idx % d;
      const int j = j0 + kk;
      const int blk = block_table[static_cast<int64_t>(r) * bt_row + (j / block_size) * bt_col];
      const int off = j % block_size;
      ksm[idx] = Load(k_cache, static_cast<int64_t>(blk) * kc_blk +
                                   static_cast<int64_t>(off) * kc_pg +
                                   static_cast<int64_t>(g) * kc_hd + ee);
      vsm[idx] = Load(v_cache, static_cast<int64_t>(blk) * vc_blk +
                                   static_cast<int64_t>(off) * vc_pg +
                                   static_cast<int64_t>(g) * vc_hd + ee);
    }
    __syncthreads();

    if (active) {
      const int kmax = min(j0 + tile_keys - 1, my_jmax);
      for (int j = j0; j <= kmax; ++j) {
        const int koff = (j - j0) * d;
        float dot = 0.0f;
        for (int i = 0; i < epl; ++i) {
          const int e = lane + 32 * i;
          if (e < d) dot += q_reg[i] * ksm[koff + e];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        dot = __shfl_sync(0xffffffffu, dot, 0);  // broadcast the row score

        const float s = dot * scale;
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new);  // 0 on the first key (m == -inf)
        const float pw = __expf(s - m_new);
        for (int i = 0; i < epl; ++i) {
          const int e = lane + 32 * i;
          if (e < d) o_reg[i] = o_reg[i] * corr + pw * vsm[koff + e];
        }
        l = l * corr + pw;
        m = m_new;
      }
    }
    __syncthreads();  // before the next tile overwrites shared memory
  }

  if (active) {
    const float inv = 1.0f / l;
    const int64_t qoff = (static_cast<int64_t>(t) * hq + h) * d;
    for (int i = 0; i < epl; ++i) {
      const int e = lane + 32 * i;
      if (e < d) Store(out, qoff + e, o_reg[i] * inv);
    }
  }
}

// ===========================================================================
// PREFILL path, TENSOR-CORE (WMMA) FlashAttention. Mirrors flash_attn_varlen_func
// (vllm/v1/attention/backends/flash_attn.py @ e24d1b24): bf16 QKᵀ / P·V on the
// tensor cores, f32 online-softmax accumulation, no per-key block barrier. Query
// is cast f32→bf16 (vLLM stores q/k/v in bf16 for FlashAttention), the paged KV
// cache is read bf16 (or f32→bf16). Reuses the DeltaH/ChunkO WMMA tiling
// (cuda_gdn.cu:1021 Q@Kᵀ Arow×Bcol; :1070 A@V Arow×Brow, acc pre-loaded).
//
// One block = (16 query rows = one WMMA m-tile, one q-head). head_dim d must be
// a multiple of 16 (gate model d=256 → 16 k-tiles). BN keys per K/V tile.
//   grid = (num_tiles, hq); block = kWmmaWarps warps.
//   Qb[16,d] bf16 staged once; per key-tile: Kb/Vb[BN,d] bf16 staged from the
//   paged cache; S = Q·Kᵀ (WMMA, f32 acc) → per-row online softmax (f32,
//   causal-masked) → P bf16 → O += P·V (WMMA, acc = rescaled running O).
// ===========================================================================
namespace attn_wmma = nvcuda::wmma;
constexpr int kWmmaM = 16;        // WMMA tile M=N=K
constexpr int kWmmaBN = 64;       // keys per K/V tile (multiple of 16)
constexpr int kWmmaWarps = 8;     // warps per block

using AccFrag = attn_wmma::fragment<attn_wmma::accumulator, kWmmaM, kWmmaM, kWmmaM, float>;
using ArowFrag =
    attn_wmma::fragment<attn_wmma::matrix_a, kWmmaM, kWmmaM, kWmmaM, __nv_bfloat16, attn_wmma::row_major>;
using BcolFrag =
    attn_wmma::fragment<attn_wmma::matrix_b, kWmmaM, kWmmaM, kWmmaM, __nv_bfloat16, attn_wmma::col_major>;
using BrowFrag =
    attn_wmma::fragment<attn_wmma::matrix_b, kWmmaM, kWmmaM, kWmmaM, __nv_bfloat16, attn_wmma::row_major>;

template <typename TQ, typename TKV, typename Tout>
__global__ void PagedFlashWmmaKernel(Tout* out, const TQ* query, const TKV* k_cache,
                                     const TKV* v_cache, const int32_t* block_table,
                                     const int32_t* seq_lens, const int32_t* query_start_loc,
                                     const int2* tiles, int num_tiles, int hq, int num_kv_heads,
                                     int d, int block_size, int64_t bt_row, int64_t bt_col,
                                     int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                     int64_t vc_pg, int64_t vc_hd, float scale, bool causal) {
  const int tile_idx = blockIdx.x;
  const int h = blockIdx.y;  // q-head
  if (tile_idx >= num_tiles) return;

  const int2 td = tiles[tile_idx];
  const int r = td.x;       // request
  const int local0 = td.y;  // first local query row in this tile
  const int q0 = query_start_loc[r];
  const int q1 = query_start_loc[r + 1];
  const int qlen = q1 - q0;
  const int seqlen = seq_lens[r];
  const int context = seqlen - qlen;
  const int g = h / (hq / num_kv_heads);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nwarps = static_cast<int>(blockDim.x) / 32;
  const int nthreads = static_cast<int>(blockDim.x);

  // Shared layout: Qb[16,d] | Kb[BN,d] | Vb[BN,d] (bf16) ; Osm[16,d] | Ssm[16,BN]
  // (f32) ; Pb[16,BN] (bf16) ; plus per-row m/l/corr scalars.
  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qb = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Kb = Qb + kWmmaM * d;
  __nv_bfloat16* Vb = Kb + kWmmaBN * d;
  float* Osm = reinterpret_cast<float*>(Vb + kWmmaBN * d);
  float* Ssm = Osm + kWmmaM * d;
  __nv_bfloat16* Pb = reinterpret_cast<__nv_bfloat16*>(Ssm + kWmmaM * kWmmaBN);
  __shared__ float s_m[kWmmaM], s_l[kWmmaM], s_corr[kWmmaM];

  // Stage Q (cast to bf16) and zero the running O.
  for (int e = tid; e < kWmmaM * d; e += nthreads) {
    const int row = e / d, col = e % d;
    const int local_row = local0 + row;
    __nv_bfloat16 qv = __float2bfloat16(0.0f);
    if (local_row < qlen) {
      const int64_t qoff = (static_cast<int64_t>(q0 + local_row) * hq + h) * d;
      qv = __float2bfloat16(Load(query, qoff + col));
    }
    Qb[e] = qv;
    Osm[e] = 0.0f;
  }
  for (int i = tid; i < kWmmaM; i += nthreads) {
    s_m[i] = -CUDART_INF_F;
    s_l[i] = 0.0f;
  }
  __syncthreads();

  const int last_row = min(local0 + kWmmaM, qlen) - 1;
  const int block_jmax = causal ? (context + last_row) : (seqlen - 1);
  const int nqk = d / kWmmaM;         // QKᵀ k-steps (also = # of Q k-tiles)
  const int nqk_n = kWmmaBN / kWmmaM; // QKᵀ n-tiles (key sub-tiles)
  const int npv_n = d / kWmmaM;       // P·V n-tiles (over head_dim)

  for (int j0 = 0; j0 <= block_jmax; j0 += kWmmaBN) {
    const int tile_keys = min(kWmmaBN, block_jmax - j0 + 1);
    // Stage the full [BN, d] K/V tile → bf16 shared. Rows [tile_keys, BN) are
    // ZEROED (not left stale): P·V multiplies masked probs (Pb=0) by these rows,
    // and 0 * NaN = NaN would poison the f32 accumulator if the padding held
    // stale NaN from a prior kernel's shared memory. Zeroing keeps 0*0 = 0.
    for (int idx = tid; idx < kWmmaBN * d; idx += nthreads) {
      const int kk = idx / d, ee = idx % d;
      if (kk < tile_keys) {
        const int j = j0 + kk;
        const int blk = block_table[static_cast<int64_t>(r) * bt_row + (j / block_size) * bt_col];
        const int off = j % block_size;
        Kb[idx] = __float2bfloat16(Load(k_cache, static_cast<int64_t>(blk) * kc_blk +
                                                     static_cast<int64_t>(off) * kc_pg +
                                                     static_cast<int64_t>(g) * kc_hd + ee));
        Vb[idx] = __float2bfloat16(Load(v_cache, static_cast<int64_t>(blk) * vc_blk +
                                                     static_cast<int64_t>(off) * vc_pg +
                                                     static_cast<int64_t>(g) * vc_hd + ee));
      } else {
        Kb[idx] = __float2bfloat16(0.0f);
        Vb[idx] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    // S = Q · Kᵀ  [16, BN]  (WMMA, f32 accumulate). n-tiles over the key sub-tiles.
    for (int nt = warp; nt < nqk_n; nt += nwarps) {
      const int jj0 = nt * kWmmaM;
      AccFrag acc;
      attn_wmma::fill_fragment(acc, 0.0f);
      for (int kk = 0; kk < nqk; ++kk) {
        ArowFrag a;
        BcolFrag b;  // K stored [BN,d] row-major → col-major load of ld=d gives Kᵀ
        attn_wmma::load_matrix_sync(a, Qb + kk * kWmmaM, d);
        attn_wmma::load_matrix_sync(b, Kb + jj0 * d + kk * kWmmaM, d);
        attn_wmma::mma_sync(acc, a, b, acc);
      }
      attn_wmma::store_matrix_sync(Ssm + jj0, acc, kWmmaBN, attn_wmma::mem_row_major);
    }
    __syncthreads();

    // Per-row online softmax (thread i owns query row i). Scale + causal mask,
    // running max/sum, write P (bf16) and the O-rescale factor corr[i].
    if (tid < kWmmaM) {
      const int i = tid;
      const int local_row = local0 + i;
      const bool active = local_row < qlen;
      const int jmax_i = !active ? -1 : (causal ? (context + local_row) : (seqlen - 1));
      float row_max = -CUDART_INF_F;
      for (int jj = 0; jj < tile_keys; ++jj) {
        const int j = j0 + jj;
        if (j <= jmax_i) {
          const float s = Ssm[i * kWmmaBN + jj] * scale;
          Ssm[i * kWmmaBN + jj] = s;  // store scaled for the exp pass
          row_max = fmaxf(row_max, s);
        }
      }
      const float m_old = s_m[i];
      const float m_new = fmaxf(m_old, row_max);
      const float corr = (m_new == -CUDART_INF_F) ? 1.0f : __expf(m_old - m_new);
      float row_sum = 0.0f;
      for (int jj = 0; jj < kWmmaBN; ++jj) {
        const int j = j0 + jj;
        float p = 0.0f;
        if (jj < tile_keys && j <= jmax_i) {
          p = __expf(Ssm[i * kWmmaBN + jj] - m_new);
          row_sum += p;
        }
        Pb[i * kWmmaBN + jj] = __float2bfloat16(p);
      }
      s_l[i] = s_l[i] * corr + row_sum;
      s_m[i] = m_new;
      s_corr[i] = corr;
    }
    __syncthreads();

    // Rescale running O by corr[i] (per row) before accumulating P·V.
    for (int e = tid; e < kWmmaM * d; e += nthreads) {
      Osm[e] *= s_corr[e / d];
    }
    __syncthreads();

    // O += P · V  [16, d]  (WMMA, acc pre-loaded from the rescaled O).
    for (int nt = warp; nt < npv_n; nt += nwarps) {
      const int vi0 = nt * kWmmaM;
      AccFrag acc;
      attn_wmma::load_matrix_sync(acc, Osm + vi0, d, attn_wmma::mem_row_major);
      for (int kk = 0; kk < nqk_n; ++kk) {
        ArowFrag a;  // P[16, BN] row-major
        BrowFrag b;  // V[BN, d] row-major
        attn_wmma::load_matrix_sync(a, Pb + kk * kWmmaM, kWmmaBN);
        attn_wmma::load_matrix_sync(b, Vb + (kk * kWmmaM) * d + vi0, d);
        attn_wmma::mma_sync(acc, a, b, acc);
      }
      attn_wmma::store_matrix_sync(Osm + vi0, acc, d, attn_wmma::mem_row_major);
    }
    __syncthreads();  // before the next tile overwrites Kb/Vb/Ssm/Pb
  }

  // Normalize by the softmax denominator and store active rows.
  for (int e = tid; e < kWmmaM * d; e += nthreads) {
    const int row = e / d, col = e % d;
    const int local_row = local0 + row;
    if (local_row < qlen) {
      const float inv = 1.0f / s_l[row];
      const int64_t qoff = (static_cast<int64_t>(q0 + local_row) * hq + h) * d;
      Store(out, qoff + col, Osm[e] * inv);
    }
  }
}

// ===========================================================================
// PREFILL path, TENSOR-CORE (WMMA) FlashAttention with GQA K/V REUSE.
// Mirrors flash_attn_varlen_func's GQA loop (vllm/v1/attention/backends/
// flash_attn.py @ e24d1b24): num_queries_per_kv = num_heads // num_kv_heads
// (:711); FlashAttention loads a KV head's K/V ONCE and attends all its query
// heads, instead of re-reading K/V per q-head.
//
// The per-head PagedFlashWmmaKernel above launches grid.y = hq blocks, and the
// 8 q-heads that share one KV head (gate: hq=16, num_kv_heads=2 → qpk=8) each
// re-stage that KV head's K/V from global → ~8x redundant K/V traffic. This
// kernel launches grid.y = hq/QG blocks; each block stages a KV head's [BN,d]
// K/V tile ONCE into shared and loops the QG query heads sharing it, reusing the
// staged K/V. K/V global traffic drops by QG (gate: QG=2 → 2x fewer K/V reads).
//
// Why QG=2 and not the full qpk=8: online softmax must keep each q-head's f32 O
// accumulator [16,d] resident across the whole key-tile loop (it is updated per
// tile). At d=256 that is 16 KiB per head. GB10/sm_121 caps opt-in shared at
// ~99 KiB/block (cudaDevAttrMaxSharedMemoryPerBlockOptin=101376), so 8 O buffers
// (128 KiB) alone exceed the ceiling — full 8x reuse is physically impossible on
// this GPU. QG=2 (BN=32) fits at ~83 KiB and halves the redundant K/V traffic;
// per-head Q is kept resident (no per-tile Q re-read). Ssm/Pb are single buffers
// reused per head in the inner loop. All QK^T/softmax/P·V math is byte-identical
// to PagedFlashWmmaKernel — only the K/V staging is shared across QG heads.
//
//   grid = (num_tiles, hq/QG); block = kWmmaWarps warps.
//   Shared: Qb[QG,16,d] | Kb[BN,d] | Vb[BN,d] (bf16) ; Osm[QG,16,d] | Ssm[16,BN]
//   (f32) ; Pb[16,BN] (bf16) ; per-(head,row) m/l/corr scalars.
// ===========================================================================
constexpr int kGqaBN = 32;  // keys per K/V tile (multiple of 16; smaller than
                            // the per-head BN=64 so the QG O buffers fit shared)

template <typename TQ, typename TKV, typename Tout, int QG>
__global__ void PagedFlashWmmaGqaKernel(Tout* out, const TQ* query, const TKV* k_cache,
                                        const TKV* v_cache, const int32_t* block_table,
                                        const int32_t* seq_lens, const int32_t* query_start_loc,
                                        const int2* tiles, int num_tiles, int hq, int num_kv_heads,
                                        int d, int block_size, int64_t bt_row, int64_t bt_col,
                                        int64_t kc_blk, int64_t kc_pg, int64_t kc_hd, int64_t vc_blk,
                                        int64_t vc_pg, int64_t vc_hd, float scale, bool causal) {
  const int tile_idx = blockIdx.x;
  const int grp = blockIdx.y;  // group over hq/QG groups of QG consecutive q-heads
  if (tile_idx >= num_tiles) return;

  const int h0 = grp * QG;                    // first q-head in this group
  const int g = h0 / (hq / num_kv_heads);     // shared KV head (QG | qpk ⇒ all QG same g)

  const int2 td = tiles[tile_idx];
  const int r = td.x;       // request
  const int local0 = td.y;  // first local query row in this tile
  const int q0 = query_start_loc[r];
  const int q1 = query_start_loc[r + 1];
  const int qlen = q1 - q0;
  const int seqlen = seq_lens[r];
  const int context = seqlen - qlen;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nwarps = static_cast<int>(blockDim.x) / 32;
  const int nthreads = static_cast<int>(blockDim.x);

  // Shared layout: Qb[QG,16,d] | Kb[BN,d] | Vb[BN,d] (bf16) ; Osm[QG,16,d] |
  // Ssm[16,BN] (f32) ; Pb[16,BN] (bf16). Ssm/Pb are single (reused per head).
  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qb = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Kb = Qb + QG * kWmmaM * d;
  __nv_bfloat16* Vb = Kb + kGqaBN * d;
  float* Osm = reinterpret_cast<float*>(Vb + kGqaBN * d);
  float* Ssm = Osm + QG * kWmmaM * d;
  __nv_bfloat16* Pb = reinterpret_cast<__nv_bfloat16*>(Ssm + kWmmaM * kGqaBN);
  __shared__ float s_m[QG][kWmmaM], s_l[QG][kWmmaM], s_corr[QG][kWmmaM];

  // Stage Q (cast bf16) for all QG heads and zero the running O.
  for (int e = tid; e < QG * kWmmaM * d; e += nthreads) {
    const int hh = e / (kWmmaM * d);
    const int rem = e - hh * kWmmaM * d;
    const int row = rem / d, col = rem % d;
    const int local_row = local0 + row;
    __nv_bfloat16 qv = __float2bfloat16(0.0f);
    if (local_row < qlen) {
      const int h = h0 + hh;
      const int64_t qoff = (static_cast<int64_t>(q0 + local_row) * hq + h) * d;
      qv = __float2bfloat16(Load(query, qoff + col));
    }
    Qb[e] = qv;
    Osm[e] = 0.0f;
  }
  for (int i = tid; i < QG * kWmmaM; i += nthreads) {
    s_m[i / kWmmaM][i % kWmmaM] = -CUDART_INF_F;
    s_l[i / kWmmaM][i % kWmmaM] = 0.0f;
  }
  __syncthreads();

  const int last_row = min(local0 + kWmmaM, qlen) - 1;
  const int block_jmax = causal ? (context + last_row) : (seqlen - 1);
  const int nqk = d / kWmmaM;          // QKᵀ k-steps
  const int nqk_n = kGqaBN / kWmmaM;   // QKᵀ n-tiles (key sub-tiles)
  const int npv_n = d / kWmmaM;        // P·V n-tiles (over head_dim)

  for (int j0 = 0; j0 <= block_jmax; j0 += kGqaBN) {
    const int tile_keys = min(kGqaBN, block_jmax - j0 + 1);
    // Stage the [BN,d] K/V tile ONCE → bf16 shared, shared by all QG heads. Rows
    // [tile_keys, BN) are ZEROED (not left stale): P·V multiplies masked probs
    // (Pb=0) by these rows, and 0 * NaN = NaN would poison the f32 accumulator.
    for (int idx = tid; idx < kGqaBN * d; idx += nthreads) {
      const int kk = idx / d, ee = idx % d;
      if (kk < tile_keys) {
        const int j = j0 + kk;
        const int blk = block_table[static_cast<int64_t>(r) * bt_row + (j / block_size) * bt_col];
        const int off = j % block_size;
        Kb[idx] = __float2bfloat16(Load(k_cache, static_cast<int64_t>(blk) * kc_blk +
                                                     static_cast<int64_t>(off) * kc_pg +
                                                     static_cast<int64_t>(g) * kc_hd + ee));
        Vb[idx] = __float2bfloat16(Load(v_cache, static_cast<int64_t>(blk) * vc_blk +
                                                     static_cast<int64_t>(off) * vc_pg +
                                                     static_cast<int64_t>(g) * vc_hd + ee));
      } else {
        Kb[idx] = __float2bfloat16(0.0f);
        Vb[idx] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    // Inner loop over the QG query heads sharing this staged K/V tile.
    for (int hh = 0; hh < QG; ++hh) {
      __nv_bfloat16* Qh = Qb + hh * kWmmaM * d;
      float* Oh = Osm + hh * kWmmaM * d;

      // S = Qh · Kᵀ  [16, BN]  (WMMA, f32 accumulate).
      for (int nt = warp; nt < nqk_n; nt += nwarps) {
        const int jj0 = nt * kWmmaM;
        AccFrag acc;
        attn_wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk < nqk; ++kk) {
          ArowFrag a;
          BcolFrag b;  // K stored [BN,d] row-major → col-major load of ld=d gives Kᵀ
          attn_wmma::load_matrix_sync(a, Qh + kk * kWmmaM, d);
          attn_wmma::load_matrix_sync(b, Kb + jj0 * d + kk * kWmmaM, d);
          attn_wmma::mma_sync(acc, a, b, acc);
        }
        attn_wmma::store_matrix_sync(Ssm + jj0, acc, kGqaBN, attn_wmma::mem_row_major);
      }
      __syncthreads();

      // Per-row online softmax (thread i owns query row i of head hh).
      if (tid < kWmmaM) {
        const int i = tid;
        const int local_row = local0 + i;
        const bool active = local_row < qlen;
        const int jmax_i = !active ? -1 : (causal ? (context + local_row) : (seqlen - 1));
        float row_max = -CUDART_INF_F;
        for (int jj = 0; jj < tile_keys; ++jj) {
          const int j = j0 + jj;
          if (j <= jmax_i) {
            const float s = Ssm[i * kGqaBN + jj] * scale;
            Ssm[i * kGqaBN + jj] = s;  // store scaled for the exp pass
            row_max = fmaxf(row_max, s);
          }
        }
        const float m_old = s_m[hh][i];
        const float m_new = fmaxf(m_old, row_max);
        const float corr = (m_new == -CUDART_INF_F) ? 1.0f : __expf(m_old - m_new);
        float row_sum = 0.0f;
        for (int jj = 0; jj < kGqaBN; ++jj) {
          const int j = j0 + jj;
          float p = 0.0f;
          if (jj < tile_keys && j <= jmax_i) {
            p = __expf(Ssm[i * kGqaBN + jj] - m_new);
            row_sum += p;
          }
          Pb[i * kGqaBN + jj] = __float2bfloat16(p);
        }
        s_l[hh][i] = s_l[hh][i] * corr + row_sum;
        s_m[hh][i] = m_new;
        s_corr[hh][i] = corr;
      }
      __syncthreads();

      // Rescale running O by corr[i] (per row) before accumulating P·V.
      for (int e = tid; e < kWmmaM * d; e += nthreads) {
        Oh[e] *= s_corr[hh][e / d];
      }
      __syncthreads();

      // O += P · V  [16, d]  (WMMA, acc pre-loaded from the rescaled O).
      for (int nt = warp; nt < npv_n; nt += nwarps) {
        const int vi0 = nt * kWmmaM;
        AccFrag acc;
        attn_wmma::load_matrix_sync(acc, Oh + vi0, d, attn_wmma::mem_row_major);
        for (int kk = 0; kk < nqk_n; ++kk) {
          ArowFrag a;  // P[16, BN] row-major
          BrowFrag b;  // V[BN, d] row-major
          attn_wmma::load_matrix_sync(a, Pb + kk * kWmmaM, kGqaBN);
          attn_wmma::load_matrix_sync(b, Vb + (kk * kWmmaM) * d + vi0, d);
          attn_wmma::mma_sync(acc, a, b, acc);
        }
        attn_wmma::store_matrix_sync(Oh + vi0, acc, d, attn_wmma::mem_row_major);
      }
      __syncthreads();  // before the next head reuses Ssm/Pb / next tile overwrites K/V
    }
  }

  // Normalize by the softmax denominator and store active rows (all QG heads).
  for (int e = tid; e < QG * kWmmaM * d; e += nthreads) {
    const int hh = e / (kWmmaM * d);
    const int rem = e - hh * kWmmaM * d;
    const int row = rem / d, col = rem % d;
    const int local_row = local0 + row;
    if (local_row < qlen) {
      const int h = h0 + hh;
      const float inv = 1.0f / s_l[hh][row];
      const int64_t qoff = (static_cast<int64_t>(q0 + local_row) * hq + h) * d;
      Store(out, qoff + col, Osm[e] * inv);
    }
  }
}

// ===========================================================================
// PREFILL path, TENSOR-CORE (WMMA) FlashAttention-2 with REGISTER-RESIDENT O.
// (VT_ATTN_FLASH2, default ON; VT_ATTN_FLASH2=0 -> proven PagedFlashWmmaGqaKernel.)
//
// STEP-1 measured limiter of PagedFlashWmmaGqaKernel on GB10/sm_121: the f32
// running-O accumulator Osm[QG,16,d] = 32 KiB (QG=2, d=256) dominates a 86.4 KiB
// shared footprint, capping the kernel at floor(102400/86400) = 1 block/SM =
// 8 warps/SM = 16.7% theoretical occupancy (registers allow 5 blocks; not the
// limit). A latency-bound kernel at 16.7% occupancy sits at ~1% of the bf16
// tensor roofline. THE fix mirrors FlashAttention-2 (Dao 2023; vllm/v1/attention/
// backends/flash_attn.py @ e24d1b24 delegates to flash_attn_varlen_func, whose
// CUTLASS kernel keeps the O accumulator in registers across the K/V loop):
// hold O in the WMMA accumulator FRAGMENTS (registers) instead of shared. This
// removes the 32 KiB Osm; shared drops to ~36 KiB -> 2 blocks/SM = 16 warps/SM =
// 33% occupancy (2x), doubling the latency-hiding warps at fixed registers.
//
// Register-O layout: each of the 8 warps owns a contiguous 2-n-tile (32-col)
// slice of O for BOTH QG heads -> 4 accumulator fragments per warp (2 heads x
// 2 n-tiles), 32 f32/thread. The online-softmax per-row corr rescale is applied
// to a fragment WITHOUT relying on the (opaque) accumulator element layout: a
// corr tile corrbuf[i][j]=corr[row i] is loaded into an accumulator fragment with
// the SAME type/layout as O, so element-wise Ofrag.x[k] *= cfrag.x[k] scales
// O(i,j) by corr(i) exactly (both fragments share the deterministic x[k]<->(i,j)
// map). K and V SHARE one [BN,d] bf16 buffer via a two-phase tile (K-phase does
// QKᵀ+softmax for all QG heads into Pb[QG]; V-phase overwrites with V and does
// P·V), which frees the shared budget the register-O reclaim needs. All softmax
// numerics (sequential-over-keys running max/sum, f32) are identical to the GQA
// kernel; only O storage + K/V staging change.
//
//   grid = (num_tiles, hq/QG); block = kWmmaWarps(=8) warps.
//   Shared: Qb[QG,16,d] | KVb[BN,d] | Pb[QG,16,BN] (bf16) ; Ssm[16,BN] |
//   corrbuf[QG,16,16] (f32). O lives in registers (accumulator fragments).
// ===========================================================================
constexpr int kF2BN = 32;  // keys per K/V tile (K and V share KVb, two-phase).

template <typename TQ, typename TKV, typename Tout, int QG>
__global__ void PagedFlashWmmaGqaFlash2Kernel(Tout* out, const TQ* query, const TKV* k_cache,
                                              const TKV* v_cache, const int32_t* block_table,
                                              const int32_t* seq_lens,
                                              const int32_t* query_start_loc, const int2* tiles,
                                              int num_tiles, int hq, int num_kv_heads, int d,
                                              int block_size, int64_t bt_row, int64_t bt_col,
                                              int64_t kc_blk, int64_t kc_pg, int64_t kc_hd,
                                              int64_t vc_blk, int64_t vc_pg, int64_t vc_hd,
                                              float scale, bool causal) {
  const int tile_idx = blockIdx.x;
  const int grp = blockIdx.y;
  if (tile_idx >= num_tiles) return;

  const int h0 = grp * QG;
  const int g = h0 / (hq / num_kv_heads);

  const int2 td = tiles[tile_idx];
  const int r = td.x;
  const int local0 = td.y;
  const int q0 = query_start_loc[r];
  const int q1 = query_start_loc[r + 1];
  const int qlen = q1 - q0;
  const int seqlen = seq_lens[r];
  const int context = seqlen - qlen;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int nwarps = static_cast<int>(blockDim.x) / 32;  // 8
  const int nthreads = static_cast<int>(blockDim.x);

  extern __shared__ char smem_raw[];
  __nv_bfloat16* Qb = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* KVb = Qb + QG * kWmmaM * d;
  __nv_bfloat16* Pb = KVb + kF2BN * d;
  float* Ssm = reinterpret_cast<float*>(Pb + QG * kWmmaM * kF2BN);
  float* corrbuf = Ssm + kWmmaM * kF2BN;  // [QG][16][16]
  __shared__ float s_m[QG][kWmmaM], s_l[QG][kWmmaM];

  const int nqk = d / kWmmaM;          // QKᵀ k-steps
  const int nqk_n = kF2BN / kWmmaM;    // QKᵀ n-tiles = 2
  // Each warp owns n-tiles [2*warp, 2*warp+2) of O for both heads -> 4 fragments.
  const int kNtPerWarp = 2;  // npv_n(16) / nwarps(8)

  // O accumulator lives in registers: Of[hh][j], j in [0,kNtPerWarp).
  AccFrag Of[QG][2];
#pragma unroll
  for (int hh = 0; hh < QG; ++hh)
#pragma unroll
    for (int j = 0; j < kNtPerWarp; ++j) attn_wmma::fill_fragment(Of[hh][j], 0.0f);

  // Stage Q (cast bf16) for all QG heads.
  for (int e = tid; e < QG * kWmmaM * d; e += nthreads) {
    const int hh = e / (kWmmaM * d);
    const int rem = e - hh * kWmmaM * d;
    const int row = rem / d, col = rem % d;
    const int local_row = local0 + row;
    __nv_bfloat16 qv = __float2bfloat16(0.0f);
    if (local_row < qlen) {
      const int h = h0 + hh;
      const int64_t qoff = (static_cast<int64_t>(q0 + local_row) * hq + h) * d;
      qv = __float2bfloat16(Load(query, qoff + col));
    }
    Qb[e] = qv;
  }
  for (int i = tid; i < QG * kWmmaM; i += nthreads) {
    s_m[i / kWmmaM][i % kWmmaM] = -CUDART_INF_F;
    s_l[i / kWmmaM][i % kWmmaM] = 0.0f;
  }
  __syncthreads();

  const int last_row = min(local0 + kWmmaM, qlen) - 1;
  const int block_jmax = causal ? (context + last_row) : (seqlen - 1);

  for (int j0 = 0; j0 <= block_jmax; j0 += kF2BN) {
    const int tile_keys = min(kF2BN, block_jmax - j0 + 1);

    // ---- K-phase: stage K into KVb.
    for (int idx = tid; idx < kF2BN * d; idx += nthreads) {
      const int kk = idx / d, ee = idx % d;
      if (kk < tile_keys) {
        const int j = j0 + kk;
        const int blk = block_table[static_cast<int64_t>(r) * bt_row + (j / block_size) * bt_col];
        const int off = j % block_size;
        KVb[idx] = __float2bfloat16(Load(k_cache, static_cast<int64_t>(blk) * kc_blk +
                                                      static_cast<int64_t>(off) * kc_pg +
                                                      static_cast<int64_t>(g) * kc_hd + ee));
      } else {
        KVb[idx] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    // Per head: S = Qh·Kᵀ (WMMA) -> online softmax -> Pb[hh], corrbuf[hh].
    for (int hh = 0; hh < QG; ++hh) {
      __nv_bfloat16* Qh = Qb + hh * kWmmaM * d;
      __nv_bfloat16* Ph = Pb + hh * kWmmaM * kF2BN;
      for (int nt = warp; nt < nqk_n; nt += nwarps) {
        const int jj0 = nt * kWmmaM;
        AccFrag acc;
        attn_wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk < nqk; ++kk) {
          ArowFrag a;
          BcolFrag b;
          attn_wmma::load_matrix_sync(a, Qh + kk * kWmmaM, d);
          attn_wmma::load_matrix_sync(b, KVb + jj0 * d + kk * kWmmaM, d);
          attn_wmma::mma_sync(acc, a, b, acc);
        }
        attn_wmma::store_matrix_sync(Ssm + jj0, acc, kF2BN, attn_wmma::mem_row_major);
      }
      __syncthreads();

      float* cbuf = corrbuf + hh * kWmmaM * kWmmaM;
      if (tid < kWmmaM) {
        const int i = tid;
        const int local_row = local0 + i;
        const bool active = local_row < qlen;
        const int jmax_i = !active ? -1 : (causal ? (context + local_row) : (seqlen - 1));
        float row_max = -CUDART_INF_F;
        for (int jj = 0; jj < tile_keys; ++jj) {
          const int j = j0 + jj;
          if (j <= jmax_i) {
            const float s = Ssm[i * kF2BN + jj] * scale;
            Ssm[i * kF2BN + jj] = s;
            row_max = fmaxf(row_max, s);
          }
        }
        const float m_old = s_m[hh][i];
        const float m_new = fmaxf(m_old, row_max);
        const float corr = (m_new == -CUDART_INF_F) ? 1.0f : __expf(m_old - m_new);
        float row_sum = 0.0f;
        for (int jj = 0; jj < kF2BN; ++jj) {
          const int j = j0 + jj;
          float p = 0.0f;
          if (jj < tile_keys && j <= jmax_i) {
            p = __expf(Ssm[i * kF2BN + jj] - m_new);
            row_sum += p;
          }
          Ph[i * kF2BN + jj] = __float2bfloat16(p);
        }
        s_l[hh][i] = s_l[hh][i] * corr + row_sum;
        s_m[hh][i] = m_new;
        for (int c = 0; c < kWmmaM; ++c) cbuf[i * kWmmaM + c] = corr;  // broadcast row corr
      }
      __syncthreads();
    }

    // ---- V-phase: overwrite KVb with V.
    for (int idx = tid; idx < kF2BN * d; idx += nthreads) {
      const int kk = idx / d, ee = idx % d;
      if (kk < tile_keys) {
        const int j = j0 + kk;
        const int blk = block_table[static_cast<int64_t>(r) * bt_row + (j / block_size) * bt_col];
        const int off = j % block_size;
        KVb[idx] = __float2bfloat16(Load(v_cache, static_cast<int64_t>(blk) * vc_blk +
                                                      static_cast<int64_t>(off) * vc_pg +
                                                      static_cast<int64_t>(g) * vc_hd + ee));
      } else {
        KVb[idx] = __float2bfloat16(0.0f);
      }
    }
    __syncthreads();

    // O_reg = O_reg*corr + Ph·V for each owned (head, n-tile). corr via a fragment
    // of identical accumulator layout -> element-wise multiply is exact per row.
    for (int hh = 0; hh < QG; ++hh) {
      __nv_bfloat16* Ph = Pb + hh * kWmmaM * kF2BN;
      float* cbuf = corrbuf + hh * kWmmaM * kWmmaM;
      AccFrag cfrag;
      attn_wmma::load_matrix_sync(cfrag, cbuf, kWmmaM, attn_wmma::mem_row_major);
#pragma unroll
      for (int j = 0; j < kNtPerWarp; ++j) {
        const int nt = warp * kNtPerWarp + j;
        AccFrag& acc = Of[hh][j];
#pragma unroll
        for (int k = 0; k < acc.num_elements; ++k) acc.x[k] *= cfrag.x[k];
        for (int kk = 0; kk < nqk_n; ++kk) {
          ArowFrag a;  // P[16,BN] row-major
          BrowFrag b;  // V[BN,d] row-major
          attn_wmma::load_matrix_sync(a, Ph + kk * kWmmaM, kF2BN);
          attn_wmma::load_matrix_sync(b, KVb + (kk * kWmmaM) * d + nt * kWmmaM, d);
          attn_wmma::mma_sync(acc, a, b, acc);
        }
      }
    }
    __syncthreads();  // before next tile overwrites KVb/Pb/Ssm/corrbuf
  }

  // Normalize by the softmax denominator and store active rows. Each warp stages
  // its owned O fragments to a private [16,16] f32 slot in KVb (bf16, free now)
  // reinterpreted as f32, then its 32 lanes scatter them to global with the 1/l
  // normalization. Warp slots are disjoint (warp*256 f32 < KVb's 4096 f32).
  __syncthreads();
  float* store_scr = reinterpret_cast<float*>(KVb) + warp * kWmmaM * kWmmaM;
  const int lane = tid & 31;
  for (int hh = 0; hh < QG; ++hh) {
    const int h = h0 + hh;
#pragma unroll
    for (int j = 0; j < kNtPerWarp; ++j) {
      const int nt = warp * kNtPerWarp + j;
      attn_wmma::store_matrix_sync(store_scr, Of[hh][j], kWmmaM, attn_wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < kWmmaM * kWmmaM; idx += 32) {
        const int row = idx / kWmmaM, col = idx % kWmmaM;
        const int local_row = local0 + row;
        if (local_row < qlen) {
          const float inv = 1.0f / s_l[hh][row];
          const int gcol = nt * kWmmaM + col;
          const int64_t qoff = (static_cast<int64_t>(q0 + local_row) * hq + h) * d;
          Store(out, qoff + gcol, store_scr[idx] * inv);
        }
      }
      __syncwarp();
    }
  }
}


// --- Launchers -------------------------------------------------------------

template <typename TQ, typename TKV, typename Tout>
void LaunchDecode(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                  const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                  const Tensor& query_start_loc, const PagedAttentionArgs& args, int64_t num_tokens,
                  int64_t hq, int64_t d, int64_t num_reqs, int64_t num_kv_heads,
                  int64_t block_size) {
  const dim3 grid(static_cast<unsigned>(num_tokens), static_cast<unsigned>(hq));
  const size_t shmem = (static_cast<size_t>(d) + kPagedBlock) * sizeof(float);
  PagedAttentionKernel<TQ, TKV, Tout><<<grid, kPagedBlock, shmem, s>>>(
      out.Ptr<Tout>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
      block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(), num_reqs,
      hq, num_kv_heads, d, block_size, block_table.stride[0], block_table.stride[1],
      k_cache.stride[0], k_cache.stride[1], k_cache.stride[2], v_cache.stride[0], v_cache.stride[1],
      v_cache.stride[2], args.scale, args.causal);
  Check(cudaGetLastError(), "paged_attention decode launch");
}

template <typename TQ, typename TKV, typename Tout>
void LaunchPrefillFlash(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                        const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                        const Tensor& query_start_loc, const PagedAttentionArgs& args, int64_t hq,
                        int64_t d, int64_t num_reqs, int64_t num_kv_heads, int64_t block_size) {
  // Read query_start_loc D2H (prefill is not graph-captured) to build per-request
  // query tiles: each request's [0, qlen) rows split into ceil(qlen/BM) tiles.
  std::vector<int32_t> qsl(static_cast<size_t>(num_reqs + 1));
  Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(), qsl.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, s),
        "paged flash qsl D2H");
  Check(cudaStreamSynchronize(s), "paged flash qsl sync");

  std::vector<int2> tiles;
  tiles.reserve(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int32_t qlen = qsl[static_cast<size_t>(r + 1)] - qsl[static_cast<size_t>(r)];
    for (int32_t ts = 0; ts < qlen; ts += kBM) tiles.push_back(int2{static_cast<int>(r), ts});
  }
  const int num_tiles = static_cast<int>(tiles.size());
  if (num_tiles == 0) return;

  int2* d_tiles = nullptr;
  Check(cudaMallocAsync(&d_tiles, tiles.size() * sizeof(int2), s), "paged flash tiles alloc");
  Check(cudaMemcpyAsync(d_tiles, tiles.data(), tiles.size() * sizeof(int2), cudaMemcpyHostToDevice,
                        s),
        "paged flash tiles H2D");

  const int bn = kBN;
  const size_t shmem = static_cast<size_t>(2) * bn * d * sizeof(float);
  auto* kernel = PagedFlashKernel<TQ, TKV, Tout>;
  if (shmem > 48u * 1024u) {
    Check(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shmem)),
          "paged flash smem opt-in");
  }
  const dim3 grid(static_cast<unsigned>(num_tiles), static_cast<unsigned>(hq));
  const dim3 block(32, kBM);
  kernel<<<grid, block, shmem, s>>>(
      out.Ptr<Tout>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
      block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(), d_tiles,
      num_tiles, static_cast<int>(hq), static_cast<int>(num_kv_heads), static_cast<int>(d),
      static_cast<int>(block_size), block_table.stride[0], block_table.stride[1], k_cache.stride[0],
      k_cache.stride[1], k_cache.stride[2], v_cache.stride[0], v_cache.stride[1], v_cache.stride[2],
      args.scale, args.causal, bn);
  Check(cudaGetLastError(), "paged_attention prefill flash launch");
  Check(cudaFreeAsync(d_tiles, s), "paged flash tiles free");
}

template <typename TQ, typename TKV, typename Tout>
void LaunchPrefillWmma(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                       const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                       const Tensor& query_start_loc, const PagedAttentionArgs& args, int64_t hq,
                       int64_t d, int64_t num_reqs, int64_t num_kv_heads, int64_t block_size) {
  // Same host-side tile build as LaunchPrefillFlash (prefill is not graph-captured).
  std::vector<int32_t> qsl(static_cast<size_t>(num_reqs + 1));
  Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(), qsl.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, s),
        "paged wmma qsl D2H");
  Check(cudaStreamSynchronize(s), "paged wmma qsl sync");

  std::vector<int2> tiles;
  tiles.reserve(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int32_t qlen = qsl[static_cast<size_t>(r + 1)] - qsl[static_cast<size_t>(r)];
    for (int32_t ts = 0; ts < qlen; ts += kWmmaM) tiles.push_back(int2{static_cast<int>(r), ts});
  }
  const int num_tiles = static_cast<int>(tiles.size());
  if (num_tiles == 0) return;

  int2* d_tiles = nullptr;
  Check(cudaMallocAsync(&d_tiles, tiles.size() * sizeof(int2), s), "paged wmma tiles alloc");
  Check(cudaMemcpyAsync(d_tiles, tiles.data(), tiles.size() * sizeof(int2), cudaMemcpyHostToDevice,
                        s),
        "paged wmma tiles H2D");

  const size_t bf16_bytes =
      (static_cast<size_t>(kWmmaM) * d + 2ull * kWmmaBN * d + static_cast<size_t>(kWmmaM) * kWmmaBN) *
      sizeof(__nv_bfloat16);
  const size_t f32_bytes =
      (static_cast<size_t>(kWmmaM) * d + static_cast<size_t>(kWmmaM) * kWmmaBN) * sizeof(float);
  const size_t shmem = bf16_bytes + f32_bytes;
  auto* kernel = PagedFlashWmmaKernel<TQ, TKV, Tout>;
  if (shmem > 48u * 1024u) {
    Check(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shmem)),
          "paged wmma smem opt-in");
  }
  const dim3 grid(static_cast<unsigned>(num_tiles), static_cast<unsigned>(hq));
  const dim3 block(32 * kWmmaWarps);
  kernel<<<grid, block, shmem, s>>>(
      out.Ptr<Tout>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
      block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(), d_tiles,
      num_tiles, static_cast<int>(hq), static_cast<int>(num_kv_heads), static_cast<int>(d),
      static_cast<int>(block_size), block_table.stride[0], block_table.stride[1], k_cache.stride[0],
      k_cache.stride[1], k_cache.stride[2], v_cache.stride[0], v_cache.stride[1], v_cache.stride[2],
      args.scale, args.causal);
  Check(cudaGetLastError(), "paged_attention prefill wmma launch");
  Check(cudaFreeAsync(d_tiles, s), "paged wmma tiles free");
}

// GQA K/V-reuse WMMA prefill: grid.y = hq/QG, each block stages a KV head's K/V
// once and loops QG query heads. QG must divide qpk (else a group would span two
// KV heads). Same host tile build as LaunchPrefillWmma.
template <typename TQ, typename TKV, typename Tout, int QG>
void LaunchPrefillWmmaGqa(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                          const Tensor& query_start_loc, const PagedAttentionArgs& args, int64_t hq,
                          int64_t d, int64_t num_reqs, int64_t num_kv_heads, int64_t block_size) {
  std::vector<int32_t> qsl(static_cast<size_t>(num_reqs + 1));
  Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(), qsl.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, s),
        "paged wmma-gqa qsl D2H");
  Check(cudaStreamSynchronize(s), "paged wmma-gqa qsl sync");

  std::vector<int2> tiles;
  tiles.reserve(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int32_t qlen = qsl[static_cast<size_t>(r + 1)] - qsl[static_cast<size_t>(r)];
    for (int32_t ts = 0; ts < qlen; ts += kWmmaM) tiles.push_back(int2{static_cast<int>(r), ts});
  }
  const int num_tiles = static_cast<int>(tiles.size());
  if (num_tiles == 0) return;

  int2* d_tiles = nullptr;
  Check(cudaMallocAsync(&d_tiles, tiles.size() * sizeof(int2), s), "paged wmma-gqa tiles alloc");
  Check(cudaMemcpyAsync(d_tiles, tiles.data(), tiles.size() * sizeof(int2), cudaMemcpyHostToDevice,
                        s),
        "paged wmma-gqa tiles H2D");

  const size_t bf16_bytes =
      (static_cast<size_t>(QG) * kWmmaM * d + 2ull * kGqaBN * d + static_cast<size_t>(kWmmaM) * kGqaBN) *
      sizeof(__nv_bfloat16);
  const size_t f32_bytes =
      (static_cast<size_t>(QG) * kWmmaM * d + static_cast<size_t>(kWmmaM) * kGqaBN) * sizeof(float);
  const size_t shmem = bf16_bytes + f32_bytes;
  auto* kernel = PagedFlashWmmaGqaKernel<TQ, TKV, Tout, QG>;
  if (shmem > 48u * 1024u) {
    Check(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shmem)),
          "paged wmma-gqa smem opt-in");
  }
  const int num_groups = static_cast<int>(hq) / QG;
  const dim3 grid(static_cast<unsigned>(num_tiles), static_cast<unsigned>(num_groups));
  const dim3 block(32 * kWmmaWarps);
  kernel<<<grid, block, shmem, s>>>(
      out.Ptr<Tout>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
      block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(), d_tiles,
      num_tiles, static_cast<int>(hq), static_cast<int>(num_kv_heads), static_cast<int>(d),
      static_cast<int>(block_size), block_table.stride[0], block_table.stride[1], k_cache.stride[0],
      k_cache.stride[1], k_cache.stride[2], v_cache.stride[0], v_cache.stride[1], v_cache.stride[2],
      args.scale, args.causal);
  Check(cudaGetLastError(), "paged_attention prefill wmma-gqa launch");
  Check(cudaFreeAsync(d_tiles, s), "paged wmma-gqa tiles free");
}

// FlashAttention-2 restructured GQA prefill launcher (VT_ATTN_FLASH2). Same host
// tile build as LaunchPrefillWmmaGqa; BN=kF2BN, K/V share one shared buffer.
template <typename TQ, typename TKV, typename Tout, int QG>
void LaunchPrefillWmmaGqaFlash2(cudaStream_t s, Tensor& out, const Tensor& query,
                                const Tensor& k_cache, const Tensor& v_cache,
                                const Tensor& block_table, const Tensor& seq_lens,
                                const Tensor& query_start_loc, const PagedAttentionArgs& args,
                                int64_t hq, int64_t d, int64_t num_reqs, int64_t num_kv_heads,
                                int64_t block_size) {
  std::vector<int32_t> qsl(static_cast<size_t>(num_reqs + 1));
  Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(), qsl.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, s),
        "paged wmma-flash2 qsl D2H");
  Check(cudaStreamSynchronize(s), "paged wmma-flash2 qsl sync");

  std::vector<int2> tiles;
  tiles.reserve(static_cast<size_t>(num_reqs));
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int32_t qlen = qsl[static_cast<size_t>(r + 1)] - qsl[static_cast<size_t>(r)];
    for (int32_t ts = 0; ts < qlen; ts += kWmmaM) tiles.push_back(int2{static_cast<int>(r), ts});
  }
  const int num_tiles = static_cast<int>(tiles.size());
  if (num_tiles == 0) return;

  int2* d_tiles = nullptr;
  Check(cudaMallocAsync(&d_tiles, tiles.size() * sizeof(int2), s), "paged wmma-flash2 tiles alloc");
  Check(cudaMemcpyAsync(d_tiles, tiles.data(), tiles.size() * sizeof(int2), cudaMemcpyHostToDevice,
                        s),
        "paged wmma-flash2 tiles H2D");

  const size_t bf16_bytes =
      (static_cast<size_t>(QG) * kWmmaM * d + static_cast<size_t>(kF2BN) * d +
       static_cast<size_t>(QG) * kWmmaM * kF2BN) *
      sizeof(__nv_bfloat16);
  const size_t f32_bytes =
      (static_cast<size_t>(kWmmaM) * kF2BN + static_cast<size_t>(QG) * kWmmaM * kWmmaM) *
      sizeof(float);
  const size_t shmem = bf16_bytes + f32_bytes;
  auto* kernel = PagedFlashWmmaGqaFlash2Kernel<TQ, TKV, Tout, QG>;
  if (shmem > 48u * 1024u) {
    Check(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shmem)),
          "paged wmma-flash2 smem opt-in");
  }
  const int num_groups = static_cast<int>(hq) / QG;
  const dim3 grid(static_cast<unsigned>(num_tiles), static_cast<unsigned>(num_groups));
  const dim3 block(32 * kWmmaWarps);
  kernel<<<grid, block, shmem, s>>>(
      out.Ptr<Tout>(), query.Ptr<TQ>(), k_cache.Ptr<TKV>(), v_cache.Ptr<TKV>(),
      block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(), query_start_loc.Ptr<int32_t>(), d_tiles,
      num_tiles, static_cast<int>(hq), static_cast<int>(num_kv_heads), static_cast<int>(d),
      static_cast<int>(block_size), block_table.stride[0], block_table.stride[1], k_cache.stride[0],
      k_cache.stride[1], k_cache.stride[2], v_cache.stride[0], v_cache.stride[1], v_cache.stride[2],
      args.scale, args.causal);
  Check(cudaGetLastError(), "paged_attention prefill wmma-flash2 launch");
  Check(cudaFreeAsync(d_tiles, s), "paged wmma-flash2 tiles free");
}


// Prefill flash is opt-out via VT_PAGED_FLASH=0 (A/B toggle vs the decode-grade
// block kernel). Evaluated once.
bool PrefillFlashEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_PAGED_FLASH");
    return !(e != nullptr && e[0] == '0');
  }();
  return enabled;
}

// bf16 tensor-core (WMMA) prefill flash attention. Default ON; opt-out via
// VT_ATTN_WMMA=0 (falls back to the CUDA-core register-tiled flash path).
bool PrefillWmmaEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_ATTN_WMMA");
    return !(e != nullptr && e[0] == '0');
  }();
  return enabled;
}

// GQA K/V-reuse WMMA prefill (one KV-head K/V staging shared by QG q-heads).
// Default ON; opt-out via VT_ATTN_GQA=0 (falls back to the per-head WMMA kernel).
bool PrefillWmmaGqaEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_ATTN_GQA");
    return !(e != nullptr && e[0] == '0');
  }();
  return enabled;
}

// FlashAttention-2 restructured GQA prefill (BN=64, K/V share one shared buffer,
// QKᵀ over 4 warps) — the measured-limiter fix over PagedFlashWmmaGqaKernel.
// Default ON; opt-out via VT_ATTN_FLASH2=0 (falls back to the proven GQA kernel).
bool PrefillWmmaFlash2Enabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_ATTN_FLASH2");
    return !(e != nullptr && e[0] == '0');
  }();
  return enabled;
}

// GQA reuse group size: how many consecutive q-heads share one staged K/V tile.
// Must divide qpk = hq/num_kv_heads. Bounded by shared memory: at d=256 each
// head's f32 O accumulator is 16 KiB, and GB10/sm_121 caps opt-in shared at
// ~99 KiB/block, so QG=2 (~83 KiB) is the largest that fits with a full BN=32
// K/V tile. See PagedFlashWmmaGqaKernel header.
constexpr int kGqaQG = 2;

// TQ = query dtype, TKV = KV-cache dtype (decoupled: Phase-1 bf16 KV cache keeps
// an f32 query with a bf16 cache — attention still accumulates in f32, the cache
// reads convert bf16→f32 via Load()). Tout is dispatched here from out.dtype.
template <typename TQ, typename TKV>
void LaunchPaged(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                 const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                 const Tensor& query_start_loc, const PagedAttentionArgs& args) {
  const int64_t num_tokens = query.shape[0], hq = query.shape[1], d = query.shape[2];
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t num_kv_heads = k_cache.shape[2], block_size = k_cache.shape[1];
  if (num_tokens == 0 || hq == 0 || d == 0) return;

  // DECODE (every request query_len 1 ⟺ num_tokens == num_reqs) or head_dim too
  // large for the register-tiled flash path: keep the graph-safe block kernel.
  // Otherwise PREFILL → flash. num_tokens/num_reqs are host-known (no device read).
  const bool is_prefill = num_tokens > num_reqs;
  // bf16 tensor-core prefill: head_dim a multiple of the WMMA tile (gate d=256),
  // and only for a bf16 KV cache — the deployment path (vLLM's bf16 flash_attn
  // KV store). An f32 cache keeps the f32 CUDA-core flash (unit-test anchors).
  const bool wmma = is_prefill && (d % kWmmaM == 0) && PrefillWmmaEnabled() &&
                    std::is_same<TKV, __nv_bfloat16>::value;
  // GQA K/V reuse: eligible when qpk = hq/num_kv_heads is a multiple of the reuse
  // group size (else a group would span two KV heads). Mirrors flash_attn's
  // load-K/V-once-per-KV-head GQA loop; halves redundant K/V traffic vs per-head.
  const int64_t qpk = (num_kv_heads > 0) ? hq / num_kv_heads : 0;
  const bool gqa = wmma && PrefillWmmaGqaEnabled() && (qpk % kGqaQG == 0) &&
                   (hq % kGqaQG == 0);
  const bool flash2 = gqa && PrefillWmmaFlash2Enabled();
  const bool prefill = is_prefill && d <= kMaxEpl * 32 && PrefillFlashEnabled();
  switch (out.dtype) {
    case DType::kF32:
      if (flash2) {
        LaunchPrefillWmmaGqaFlash2<TQ, TKV, float, kGqaQG>(s, out, query, k_cache, v_cache,
                                                          block_table, seq_lens,
                                                          query_start_loc, args, hq, d,
                                                          num_reqs, num_kv_heads, block_size);
      } else if (gqa) {
        LaunchPrefillWmmaGqa<TQ, TKV, float, kGqaQG>(s, out, query, k_cache, v_cache, block_table,
                                                     seq_lens, query_start_loc, args, hq, d,
                                                     num_reqs, num_kv_heads, block_size);
      } else if (wmma) {
        LaunchPrefillWmma<TQ, TKV, float>(s, out, query, k_cache, v_cache, block_table, seq_lens,
                                          query_start_loc, args, hq, d, num_reqs, num_kv_heads,
                                          block_size);
      } else if (prefill) {
        LaunchPrefillFlash<TQ, TKV, float>(s, out, query, k_cache, v_cache, block_table, seq_lens,
                                           query_start_loc, args, hq, d, num_reqs, num_kv_heads,
                                           block_size);
      } else {
        LaunchDecode<TQ, TKV, float>(s, out, query, k_cache, v_cache, block_table, seq_lens,
                                     query_start_loc, args, num_tokens, hq, d, num_reqs,
                                     num_kv_heads, block_size);
      }
      break;
    case DType::kBF16:
      if (flash2) {
        LaunchPrefillWmmaGqaFlash2<TQ, TKV, __nv_bfloat16, kGqaQG>(s, out, query, k_cache, v_cache,
                                                          block_table, seq_lens,
                                                          query_start_loc, args, hq, d,
                                                          num_reqs, num_kv_heads, block_size);
      } else if (gqa) {
        LaunchPrefillWmmaGqa<TQ, TKV, __nv_bfloat16, kGqaQG>(
            s, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc, args, hq, d,
            num_reqs, num_kv_heads, block_size);
      } else if (wmma) {
        LaunchPrefillWmma<TQ, TKV, __nv_bfloat16>(s, out, query, k_cache, v_cache, block_table,
                                                  seq_lens, query_start_loc, args, hq, d, num_reqs,
                                                  num_kv_heads, block_size);
      } else if (prefill) {
        LaunchPrefillFlash<TQ, TKV, __nv_bfloat16>(s, out, query, k_cache, v_cache, block_table,
                                                   seq_lens, query_start_loc, args, hq, d, num_reqs,
                                                   num_kv_heads, block_size);
      } else {
        LaunchDecode<TQ, TKV, __nv_bfloat16>(s, out, query, k_cache, v_cache, block_table, seq_lens,
                                             query_start_loc, args, num_tokens, hq, d, num_reqs,
                                             num_kv_heads, block_size);
      }
      break;
    default: VT_CHECK(false, "cuda paged_attention: unsupported out dtype");
  }
}

// Dispatch on (query dtype, KV-cache dtype). Both f32 and bf16 caches are valid
// (Phase-1 bf16 KV cache mirrors vLLM's bf16 flash_attn KV store); the query may
// independently be f32 (Phase 1) or bf16.
template <typename TQ>
void LaunchPagedByKv(cudaStream_t s, Tensor& out, const Tensor& query, const Tensor& k_cache,
                     const Tensor& v_cache, const Tensor& block_table, const Tensor& seq_lens,
                     const Tensor& query_start_loc, const PagedAttentionArgs& args) {
  switch (k_cache.dtype) {
    case DType::kF32:
      LaunchPaged<TQ, float>(s, out, query, k_cache, v_cache, block_table, seq_lens,
                             query_start_loc, args);
      break;
    case DType::kBF16:
      LaunchPaged<TQ, __nv_bfloat16>(s, out, query, k_cache, v_cache, block_table, seq_lens,
                                     query_start_loc, args);
      break;
    default: VT_CHECK(false, "cuda paged_attention: unsupported KV-cache dtype (f32/bf16 only)");
  }
}

void PagedAttentionKernelCuda(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                              const Tensor& v_cache, const Tensor& block_table,
                              const Tensor& seq_lens, const Tensor& query_start_loc,
                              const PagedAttentionArgs& args) {
  switch (query.dtype) {
    case DType::kF32:
      LaunchPagedByKv<float>(AsStream(q), out, query, k_cache, v_cache, block_table, seq_lens,
                             query_start_loc, args);
      break;
    case DType::kBF16:
      LaunchPagedByKv<__nv_bfloat16>(AsStream(q), out, query, k_cache, v_cache, block_table,
                                     seq_lens, query_start_loc, args);
      break;
    default: VT_CHECK(false, "cuda paged_attention: unsupported input dtype (f32/bf16 only)");
  }
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kPagedAttention, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernelCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
