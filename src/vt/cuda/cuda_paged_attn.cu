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

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

// Prefill flash is opt-out via VT_PAGED_FLASH=0 (A/B toggle vs the decode-grade
// block kernel). Evaluated once.
bool PrefillFlashEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VT_PAGED_FLASH");
    return !(e != nullptr && e[0] == '0');
  }();
  return enabled;
}

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
  const bool prefill = num_tokens > num_reqs && d <= kMaxEpl * 32 && PrefillFlashEnabled();
  switch (out.dtype) {
    case DType::kF32:
      if (prefill) {
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
      if (prefill) {
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
