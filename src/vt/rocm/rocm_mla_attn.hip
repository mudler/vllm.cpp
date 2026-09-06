// ROCm MLA attention — the two ops GLM-5.3 non-flash refuses on `gfx1151`
// (`BACKEND-ROCM`, issue #2926, spec `.agents/specs/rocm-mla-attention-nonflash.md`).
//
//   kMlaPrefillAttention — varlen multi-head attention over this step's own
//                          new-token K/V, causal (bottom-right aligned) or not.
//   kMlaDecodeAttention  — the ABSORBED MQA decode over the compressed latent
//                          cache, walked through the block table.
//
// WHY THESE TWO, AND WHY NOW. `6b97a6800` (#2511) narrowed managed allocation
// to `PageableMemoryAccess == 1` and made the unified claim follow the
// allocator (`include/vt/rocm/rocm_arch.h:180-195`). `gfx1151` reports 0, so
// `docs/ROCM.md:83-85` gives it plain `hipMalloc` and NO reference tier — and
// an op with no native ROCm kernel is then a REFUSAL, not a slow path. These
// are the two the GLM-5.3 forward reaches, so before this file the model could
// not emit a token on this board at all.
//
// The set is a MEASUREMENT, not a reading. `9f3e6e223`'s GLM-5.3 run on this
// board named the five ops it observed on the then-eligible tier:
// `ConcatAndCacheMla`, `ConcatMlaNopeRope`, `MlaPrefillAttention`,
// `BatchedMatmul`, `MlaDecodeAttention`. Three landed native at `928d89a9b`
// (#2715 W1); these are the other two. The DSA indexer pair is NOT in that set
// and does not block generation on a dense step — `run_indexer` needs
// `sparse_step`, which mirrors upstream's `use_dense_mha = prefill_max_seq_len
// <= self.topk_tokens`. It stays owed.
//
// UPSTREAM (pinned oracle `~/_git/vllm` @ `5559679229`).
//   MlaPrefillAttention <- vllm/v1/attention/backends/mla/prefill/flash_attn.py
//                          :153-248 (`_flash_attn_varlen_diff_headdims`,
//                          `run_prefill_new_tokens` causal / `run_prefill_
//                          context_chunk` non-causal)
//   MlaDecodeAttention  <- vllm/v1/attention/backends/mla/triton_mla.py:189-260
//                          (`TritonMLAImpl.forward_mqa`, ONE buffer as both K
//                          and V, `:236-244`), NUMERICS from upstream's own CPU
//                          kernel vllm/csrc/cpu/mla_decode.cpp
//                          (`mla_decode_kvcache_cpu_impl`)
//
// PORTED FROM, and this is the one place the mirror is not the CUDA sibling.
// `cuda_mla_prefill.cu`'s prefill arm is the vendored FlashAttention-2 launcher
// behind `#ifdef VLLM_CPP_FLASH_ATTN`, and `cuda_mla_attn.cu:140-750` is the
// two-stage split-KV flash decode. Neither is a transcription target here. The
// numerics come from the CPU references, which are this tree's independent
// ports of upstream's own semantics:
//   src/vt/cpu/cpu_mla_prefill.cpp   (MlaPrefillAttentionKernel)
//   src/vt/cpu/cpu_mla_attn.cpp      (MlaDecodeAttentionKernel)
// and the three optional arms' exact rules come from the CUDA decode kernel,
// cited at each site:
//   src/vt/cuda/cuda_mla_attn.cu:196       (the sliding-window start)
//   src/vt/cuda/cuda_mla_attn.cu:255-266   (the selected-slot visit map)
//   src/vt/cuda/cuda_mla_attn.cu:281-289   (a dead selected slot scores nothing)
//   src/vt/cuda/cuda_mla_attn.cu:398-407   (the per-head attention sink)
//
// WHY THERE IS NO SPLIT-KV SCHEDULE HERE, stated rather than left to be found.
// `.agents/specs/rocm-mla-dsa-ops.md` priced the decode op at ~600 HIP lines
// and HIGH risk, and its four risks are all properties of the SPLIT: a >48 KiB
// dynamic shared-memory request with no HIP opt-in, a `multiprocessor_count`
// probe `DeviceCaps` does not carry, a grow-only capture-safe workspace, and a
// sink stage 2 must count once per row rather than once per split. A
// single-pass kernel — one block per output row, one streaming online-softmax
// walk over the keys — computes the same function and has none of them. The
// split's purpose is filling a GPU when `batch * heads` is small; that is a
// SPEED property, no speed number is admissible from this board for this model
// (`docs/ROCM.md:60-61`), and the split stays owed as its own wave.
//
// WAVEFRONT WIDTH is not assumed. The dot product uses the `__syncthreads()`
// shared-memory tree reduce `rocm_rmsnorm.hip:67-98` and
// `rocm_mla_fused_norm_rope.hip:98-116` already use, with no warp-level
// primitive and no `warpSize`, so a 64-lane wavefront needs no special case.
//
// REDUCTION ORDER, and why an NMSE bound is the honest gate. The key loop is
// SEQUENTIAL inside the block, in the CPU reference's own order, so the online
// softmax reduces over keys identically. Only the dot product's order over the
// head dimension differs (a tree here, a sequential sum there). That is the one
// documented source of the difference, which is why the gate is NMSE <= 5e-4
// and not bit-exactness.
// `hip_bf16.h` BEFORE `hip_runtime.h`, which is the order every bf16-using
// sibling in this directory uses (`rocm_mla_fused_norm_rope.hip:1-2`,
// `rocm_paged_attn.hip:4-5`). Without it `__hip_bfloat16` and
// `__float2bfloat16` are undeclared and the TU does not compile -- measured,
// as five errors from one cause, on rc job 1e41b641.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vt/ops.h"
#include "vt/rocm/rocm_device_bind.h"

namespace vt::rocm {
namespace {

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm mla_attn: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) { return static_cast<hipStream_t>(q.handle); }

// 256 threads: four whole wavefronts, and the width `rocm_rmsnorm.hip:41` fixes
// for the reason stated there — it keeps the reduction order identical across
// the ROCm kernels in this tree. It is a power of two, which the tree reduce
// below requires.
constexpr int kBlock = 256;

// The shared-memory ceiling this file will ask for. Every architecture
// guarantees 48 KiB of dynamic shared memory per block and HIP has no opt-in
// above it, so a geometry that would need more is REFUSED BY NAME rather than
// launched into an opaque `hipErrorInvalidValue`. At GLM-5.3 geometry the ask
// is (576 + 512 + 256) * 4 = 5,376 B.
constexpr size_t kMaxSharedBytes = 48u * 1024u;

// Integer clamps, spelled locally rather than through the bare `min`/`max` HIP
// injects into the global namespace. This translation unit also includes
// `<algorithm>`, and a two-name overload set that a build could resolve either
// way is the kind of thing that compiles on one toolchain and not the next.
__device__ inline int IMin(int a, int b) { return a < b ? a : b; }
__device__ inline int IMax(int a, int b) { return a > b ? a : b; }

__device__ inline float Ld(const float* p, int64_t i) { return p[i]; }
__device__ inline float Ld(const __hip_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ inline void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ inline void St(__hip_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);  // round-to-nearest-even, as the CUDA path and the host do
}

// The shared-memory tree reduce, `rocm_rmsnorm.hip:80-86` unchanged. On return
// `red[0]` holds the sum on EVERY thread's view and a `__syncthreads()` has
// executed, so the caller may read it without another barrier.
__device__ inline float BlockSum(float* red, float part) {
  red[threadIdx.x] = part;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  return red[0];
}

// ─── kMlaPrefillAttention ───────────────────────────────────────────────────
// One block per (query row, head). `cpu_mla_prefill.cpp`'s loop, with its
// two-pass softmax replaced by the streaming online form — the same answer, and
// the only form whose working set is O(head_dim) rather than O(len_k), which is
// what lets a block hold it in shared memory at any sequence length.
template <typename T>
__global__ __launch_bounds__(kBlock) void MlaPrefillAttnKernelHip(
    T* __restrict__ out, float* __restrict__ lse, const T* __restrict__ query,
    const T* __restrict__ key, const T* __restrict__ value,
    const int32_t* __restrict__ qsl, const int32_t* __restrict__ ksl, int num_reqs,
    int64_t q_s0, int64_t q_s1, int64_t k_s0, int64_t k_s1, int64_t v_s0, int64_t v_s1,
    int64_t o_s0, int64_t o_s1, int64_t lse_s0, int qk_head_dim, int v_head_dim,
    float scale, bool causal, int win_left) {
  const int t = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);

  // Which request owns this query row. `cu_seqlens_q` is a DEVICE tensor
  // (ops.cpp requires `cu_seqlens_q.device == q.device`), so the lookup happens
  // here rather than on the host. The scan takes the FIRST span whose end is
  // past `t`, which is also correct for a zero-length span — upstream's
  // `cu_seqlens` may carry one and `cpu_mla_prefill.cpp` skips it with
  // `if (len_q == 0) continue;`.
  int b = 0;
  while (b < num_reqs && t >= qsl[b + 1]) ++b;
  if (b >= num_reqs) return;

  const int q_begin = qsl[b];
  const int len_q = qsl[b + 1] - q_begin;
  const int k_begin = ksl[b];
  const int len_k = ksl[b + 1] - k_begin;
  const int iq = t - q_begin;

  // FlashAttention's BOTTOM-RIGHT causal alignment: query `i` sees keys
  // `j <= i + (len_k - len_q)`. `cpu_mla_prefill.cpp`'s `causal_shift`,
  // element for element. A TOP-LEFT implementation differs from this whenever
  // `len_k != len_q`, which is exactly the chunked-context call.
  const int causal_shift = len_k - len_q;
  const int visible = causal ? IMin(len_k, IMax(0, iq + causal_shift + 1)) : len_k;
  // `p - left` clamped at 0, with `p = iq + causal_shift` — the SAME
  // bottom-right position the causal bound uses, so the two agree by
  // construction. `win_left < 0` is the absent state (every DeepSeek / MiniCPM3
  // / Kimi-Linear caller) and puts the start back at 0.
  const int first =
      win_left < 0 ? 0 : IMin(visible, IMax(0, iq + causal_shift - win_left));

  extern __shared__ float smem[];
  float* q_s = smem;                    // [qk_head_dim]
  float* acc_s = q_s + qk_head_dim;     // [v_head_dim]
  float* red = acc_s + v_head_dim;      // [kBlock]

  const T* qp = query + static_cast<int64_t>(t) * q_s0 + static_cast<int64_t>(h) * q_s1;
  for (int d = tid; d < qk_head_dim; d += kBlock) q_s[d] = Ld(qp, d);
  for (int d = tid; d < v_head_dim; d += kBlock) acc_s[d] = 0.0f;
  __syncthreads();

  // Every thread carries the same `m` and `l`: both are derived only from
  // `red[0]`, which the tree reduce leaves identical on all of them.
  float m = -INFINITY;
  float l = 0.0f;
  for (int j = first; j < visible; ++j) {
    const T* kp = key + static_cast<int64_t>(k_begin + j) * k_s0 +
                  static_cast<int64_t>(h) * k_s1;
    float part = 0.0f;
    for (int d = tid; d < qk_head_dim; d += kBlock) part += q_s[d] * Ld(kp, d);
    const float qk = BlockSum(red, part) * scale;

    // The online softmax, `cpu_mla_attn.cpp`'s update verbatim. The `isinf(m)`
    // guard is what keeps the FIRST key from computing `expf(-inf - -inf)`,
    // which is `expf(NaN)`.
    const float m_new = fmaxf(m, qk);
    const float rescale = isinf(m) ? 0.0f : expf(m - m_new);
    const float p = expf(qk - m_new);
    const T* vp = value + static_cast<int64_t>(k_begin + j) * v_s0 +
                  static_cast<int64_t>(h) * v_s1;
    for (int d = tid; d < v_head_dim; d += kBlock) {
      acc_s[d] = acc_s[d] * rescale + p * Ld(vp, d);
    }
    l = l * rescale + p;
    m = m_new;
    // `red` is rewritten at the top of the next iteration, and the accumulator
    // update above still has readers of `red[0]` in flight without this.
    __syncthreads();
  }

  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  T* op = out + static_cast<int64_t>(t) * o_s0 + static_cast<int64_t>(h) * o_s1;
  for (int d = tid; d < v_head_dim; d += kBlock) St(op, d, acc_s[d] * inv);
  if (lse != nullptr && tid == 0) {
    // FlashAttention's varlen LSE layout is [num_heads, total_q]; a row with no
    // visible keys is -inf, which is what MergeAttnStates' both-(-inf) branch
    // is written for.
    lse[static_cast<int64_t>(h) * lse_s0 + t] = l > 0.0f ? m + logf(l) : -INFINITY;
  }
}

// ─── kMlaDecodeAttention ────────────────────────────────────────────────────
// One block per (request, head). `cpu_mla_attn.cpp`'s loop, which is upstream's
// own `mla_decode_kvcache_cpu_impl` shape: walk the block table, run the online
// softmax, and read V as the LEADING `v_head_dim` slice of the SAME cache row
// the QK dot spanned in full (`triton_mla.py:236`).
template <typename T>
__global__ __launch_bounds__(kBlock) void MlaDecodeAttnKernelHip(
    T* __restrict__ out, float* __restrict__ lse, const T* __restrict__ query,
    const T* __restrict__ kv_cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens, int64_t q_s0, int64_t q_s1, int64_t c_s0,
    int64_t c_s1, int64_t bt_s0, int64_t o_s0, int64_t o_s1, int64_t lse_s0,
    int head_size, int v_head_dim, int block_size, float scale, int win_left,
    const int32_t* __restrict__ sel, const int32_t* __restrict__ sel_cnt, int64_t sel_s0,
    int topk, const float* __restrict__ attn_sink) {
  const int b = static_cast<int>(blockIdx.x);
  const int h = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int seq_len = seq_lens[b];

  // The sliding-window start (`cuda_mla_attn.cu:196`, `cpu_mla_attn.cpp`): a
  // decode query IS the last position of its own sequence, so
  // `AttentionWindow{left, 0}` keeps keys `[seq_len - 1 - left, seq_len)` and
  // only the loop START moves.
  const int kv_start = win_left < 0 ? 0 : IMax(0, seq_len - 1 - win_left);
  // The DSA selected-slot arm (`cuda_mla_attn.cu:255-266`). CLAMPED against
  // `topk` because `ops.cpp` can only refuse an over-large count on
  // host-readable memory — a device read there would sync the decode step — so
  // the kernel contains what the validator could not see.
  const int n_keys =
      sel != nullptr ? IMin(topk, IMax(0, sel_cnt[b])) : IMax(0, seq_len - kv_start);

  extern __shared__ float smem[];
  float* q_s = smem;                   // [head_size]
  float* acc_s = q_s + head_size;      // [v_head_dim]
  float* red = acc_s + v_head_dim;     // [kBlock]

  const T* qp = query + static_cast<int64_t>(b) * q_s0 + static_cast<int64_t>(h) * q_s1;
  for (int d = tid; d < head_size; d += kBlock) q_s[d] = Ld(qp, d);
  for (int d = tid; d < v_head_dim; d += kBlock) acc_s[d] = 0.0f;
  __syncthreads();

  // THE PER-HEAD ATTENTION SINK, seeded ONCE PER ROW (`cuda_mla_attn.cu:398-407`
  // says why the split arm has to put it in stage 2 instead; a single-pass
  // kernel has no per-split double count to avoid). It is one extra logit that
  // joins the DENOMINATOR and contributes no value: start the running max at the
  // sink and the denominator at `exp(sink - sink) == 1`, accumulator at zero.
  // Null — every caller before W5 — keeps the `-inf`/0 seeds and the loop is
  // bit-identical to the pre-sink one.
  float m = attn_sink != nullptr ? attn_sink[h] : -INFINITY;
  float l = attn_sink != nullptr ? 1.0f : 0.0f;

  for (int i = 0; i < n_keys; ++i) {
    // Visit index -> token position. On the ABSENT arm this is `kv_start + i`,
    // so the sequence of positions — and therefore the online softmax's
    // reduction order — is exactly the contiguous loop's.
    const int j = sel != nullptr ? sel[static_cast<int64_t>(b) * sel_s0 + i] : kv_start + i;
    // A `-1` (upstream's "no token" sentinel, sparse_attn_indexer.py:431-432) or
    // an out-of-range position contributes NOTHING. `cuda_mla_attn.cu:281-289`
    // spells that as a -inf score; skipping is the same answer and is what
    // `cpu_mla_attn.cpp` does for `-1`. The predicate depends only on `b` and
    // `i`, so the `continue` is BLOCK-UNIFORM and skips no `__syncthreads()`.
    //
    // A RECORDED DIVERGENCE, not a new one. On an out-of-range position
    // `cpu_mla_attn.cpp` REFUSES by name where the CUDA arm returns a number. A
    // device kernel cannot throw, so this arm can only take the CUDA rule. The
    // divergence predates this file and stays in `.agents/specs/dots3-note.md`
    // `## Owed`.
    if (sel != nullptr && (j < 0 || j >= seq_len)) continue;
    const int blk = block_table[static_cast<int64_t>(b) * bt_s0 + j / block_size];
    const T* src = kv_cache + static_cast<int64_t>(blk) * c_s0 +
                   static_cast<int64_t>(j % block_size) * c_s1;

    float part = 0.0f;
    for (int d = tid; d < head_size; d += kBlock) part += q_s[d] * Ld(src, d);
    const float qk = BlockSum(red, part) * scale;

    const float m_new = fmaxf(m, qk);
    const float rescale = isinf(m) ? 0.0f : expf(m - m_new);
    const float p = expf(qk - m_new);
    // V IS the leading `v_head_dim` slice of the row the dot just spanned.
    for (int d = tid; d < v_head_dim; d += kBlock) {
      acc_s[d] = acc_s[d] * rescale + p * Ld(src, d);
    }
    l = l * rescale + p;
    m = m_new;
    __syncthreads();
  }

  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
  T* op = out + static_cast<int64_t>(b) * o_s0 + static_cast<int64_t>(h) * o_s1;
  for (int d = tid; d < v_head_dim; d += kBlock) St(op, d, acc_s[d] * inv);
  if (lse != nullptr && tid == 0) {
    // triton_decode_attention.py:636 `lse_val = e_max + tl.log(e_sum)`.
    lse[static_cast<int64_t>(b) * lse_s0 + h] = l > 0.0f ? m + logf(l) : -INFINITY;
  }
}

// The shared-memory ask, in one place so the guard and the launch cannot drift.
inline size_t SharedBytes(int64_t dot_dim, int64_t v_head_dim) {
  return static_cast<size_t>(dot_dim + v_head_dim + kBlock) * sizeof(float);
}

inline void RequireSharedFits(size_t bytes, const char* op, int64_t dot_dim,
                              int64_t v_head_dim) {
  VT_CHECK(bytes <= kMaxSharedBytes,
           std::string("rocm ") + op + ": this geometry needs " + std::to_string(bytes) +
               " B of dynamic shared memory (dot_dim " + std::to_string(dot_dim) +
               " + v_head_dim " + std::to_string(v_head_dim) + " + a " +
               std::to_string(kBlock) +
               "-thread reduction scratch), and HIP offers no opt-in above the "
               "48 KiB every architecture guarantees. Refusing rather than "
               "launching into hipErrorInvalidValue.");
}

template <typename T>
void LaunchMlaPrefill(hipStream_t s, Tensor& out, Tensor* lse, const Tensor& query,
                      const Tensor& key, const Tensor& value, const Tensor& cu_seqlens_q,
                      const Tensor& cu_seqlens_k, const MlaPrefillAttentionArgs& args) {
  const int64_t total_q = query.shape[0];
  const int64_t num_heads = query.shape[1];
  const int64_t qk_head_dim = query.shape[2];
  const int64_t v_head_dim = value.shape[2];
  if (total_q == 0) return;
  const size_t shbytes = SharedBytes(qk_head_dim, v_head_dim);
  RequireSharedFits(shbytes, "mla_prefill_attention", qk_head_dim, v_head_dim);
  const int win_left = args.window_size.has_value()
                           ? static_cast<int>(args.window_size->left)
                           : -1;
  const dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(num_heads));
  MlaPrefillAttnKernelHip<T><<<grid, kBlock, shbytes, s>>>(
      out.Ptr<T>(), lse != nullptr ? lse->Ptr<float>() : nullptr, query.Ptr<T>(),
      key.Ptr<T>(), value.Ptr<T>(), cu_seqlens_q.Ptr<int32_t>(),
      cu_seqlens_k.Ptr<int32_t>(), static_cast<int>(cu_seqlens_q.shape[0] - 1),
      query.stride[0], query.stride[1], key.stride[0], key.stride[1], value.stride[0],
      value.stride[1], out.stride[0], out.stride[1],
      lse != nullptr ? lse->stride[0] : 0, static_cast<int>(qk_head_dim),
      static_cast<int>(v_head_dim), args.scale, args.causal, win_left);
  Check(hipGetLastError(), "mla_prefill_attention launch");
}

template <typename T>
void LaunchMlaDecode(hipStream_t s, Tensor& out, Tensor* lse, const Tensor& query,
                     const Tensor& kv_cache, const Tensor& block_table,
                     const Tensor& seq_lens, const MlaDecodeAttentionArgs& args) {
  const int64_t batch = query.shape[0];
  const int64_t heads = query.shape[1];
  const int64_t head_size = query.shape[2];
  const int64_t v_head_dim = out.shape[2];
  if (batch == 0 || heads == 0) return;
  const size_t shbytes = SharedBytes(head_size, v_head_dim);
  RequireSharedFits(shbytes, "mla_decode_attention", head_size, v_head_dim);
  const int win_left = args.window_size.has_value()
                           ? static_cast<int>(args.window_size->left)
                           : -1;
  const int32_t* sel =
      args.topk_indices != nullptr ? args.topk_indices->Ptr<int32_t>() : nullptr;
  const int32_t* sel_cnt =
      args.valid_counts != nullptr ? args.valid_counts->Ptr<int32_t>() : nullptr;
  const int64_t sel_s0 = sel != nullptr ? args.topk_indices->stride[0] : 0;
  const int topk = sel != nullptr ? static_cast<int>(args.topk_indices->shape[1]) : 0;
  const float* sink = args.attn_sink != nullptr ? args.attn_sink->Ptr<float>() : nullptr;
  const dim3 grid(static_cast<unsigned>(batch), static_cast<unsigned>(heads));
  MlaDecodeAttnKernelHip<T><<<grid, kBlock, shbytes, s>>>(
      out.Ptr<T>(), lse != nullptr ? lse->Ptr<float>() : nullptr, query.Ptr<T>(),
      kv_cache.Ptr<T>(), block_table.Ptr<int32_t>(), seq_lens.Ptr<int32_t>(),
      query.stride[0], query.stride[1], kv_cache.stride[0], kv_cache.stride[1],
      block_table.stride[0], out.stride[0], out.stride[1],
      lse != nullptr ? lse->stride[0] : 0, static_cast<int>(head_size),
      static_cast<int>(v_head_dim), static_cast<int>(kv_cache.shape[1]), args.scale,
      win_left, sel, sel_cnt, sel_s0, topk, sink);
  Check(hipGetLastError(), "mla_decode_attention launch");
}

}  // namespace

// The registered MlaPrefillAttentionFn. Every argument check already ran in
// `ops.cpp`; a backend writes zero validation beyond what only it can know.
void MlaPrefillAttentionKernelRocm(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                                   const Tensor& key, const Tensor& value,
                                   const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k,
                                   const MlaPrefillAttentionArgs& args) {
  EnsureQueueDevice(q);
  switch (query.dtype) {
    case DType::kF32:
      LaunchMlaPrefill<float>(AsStream(q), out, lse, query, key, value, cu_seqlens_q,
                              cu_seqlens_k, args);
      break;
    case DType::kBF16:
      LaunchMlaPrefill<__hip_bfloat16>(AsStream(q), out, lse, query, key, value,
                                       cu_seqlens_q, cu_seqlens_k, args);
      break;
    default:
      // Named rather than silent: an f16 model would otherwise meet a wrong
      // answer or an opaque launch failure instead of the missing arm.
      VT_CHECK(false,
               "rocm mla_prefill_attention: unsupported dtype — this arm carries f32 "
               "and bf16, the two a GLM-5.3 GGUF arm produces. f16 is owed "
               "(BACKEND-ROCM, #2926).");
  }
}

// The registered MlaDecodeAttentionFn.
//
// `MlaDecodeAttentionArgs::num_kv_splits` is IGNORED here, stated rather than
// implied. It is a scheduling hint for a split-KV kernel; this arm is
// single-pass, so there is nothing for it to select. `ops.cpp` validates only
// `>= 0`, and the CUDA arm overrides it with its own heuristic when it is 0.
void MlaDecodeAttentionKernelRocm(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                                  const Tensor& kv_cache, const Tensor& block_table,
                                  const Tensor& seq_lens,
                                  const MlaDecodeAttentionArgs& args) {
  EnsureQueueDevice(q);
  switch (query.dtype) {
    case DType::kF32:
      LaunchMlaDecode<float>(AsStream(q), out, lse, query, kv_cache, block_table, seq_lens,
                             args);
      break;
    case DType::kBF16:
      LaunchMlaDecode<__hip_bfloat16>(AsStream(q), out, lse, query, kv_cache, block_table,
                                      seq_lens, args);
      break;
    default:
      VT_CHECK(false,
               "rocm mla_decode_attention: unsupported dtype — this arm carries f32 "
               "and bf16, the two a GLM-5.3 GGUF arm produces. f16 is owed "
               "(BACKEND-ROCM, #2926).");
  }
}

}  // namespace vt::rocm
