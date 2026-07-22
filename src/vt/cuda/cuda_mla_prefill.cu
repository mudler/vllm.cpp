// MLA prefill + the chunked-context primitives (CUDA) — MLA campaign W5.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin e24d1b24) ───────
//
//   OURS                        <-  UPSTREAM
//   MlaPrefillAttentionCuda     <-  vllm/v1/attention/backends/mla/prefill/
//                                   flash_attn.py:153-203
//                                   `_flash_attn_varlen_diff_headdims`
//                                   (+ `:205 run_prefill_new_tokens` causal /
//                                      `:229 run_prefill_context_chunk` non-causal)
//   PadVKernel / SliceOutKernel <-  `flash_attn.py:164-168` (the ZERO pad of V up
//                                   to the QK head dim) and `:196-197` (the slice
//                                   of the output back to v_head_dim)
//   LaunchMlaPrefillFA2Bf16     <-  the `flash_attn_varlen_func` call itself
//     (in cuda_flash_attn_fa2.cu)   (`:182-188`), i.e. the vendored FA-2
//   GatherMlaCacheCuda          <-  vllm/csrc/libtorch_stable/cache_kernels.cu
//                                   :992-1064 `gather_and_maybe_dequant_cache`
//                                   (host wrapper `:1099-1157`)
//   MergeAttnStatesCuda         <-  vllm/csrc/libtorch_stable/attention/
//                                   merge_attn_states.cu:18-192
//                                   `merge_attn_states_kernel`
//
// WHAT ACTUALLY RUNS UPSTREAM, verified rather than assumed (AGENTS.md
// whole-chain rule). Three different answers, and they matter:
//   * The MLA PREFILL ATTENTION is NOT vLLM's own csrc — it is FlashAttention,
//     reached through `vllm_flash_attn`, selected by
//     `mla/prefill/selector.py:66-76` which gives every non-sm100 device the
//     single entry `[FLASH_ATTN]` and HARD-RAISES if it is unavailable
//     (`:191-194`). W0 OBSERVED the oracle logging
//     `Using FLASH_ATTN MLA prefill backend` on sm_121. That is why this file
//     calls the VENDORED FA-2 rather than writing a kernel.
//   * The CACHE GATHER and the LSE MERGE *are* vLLM's own csrc (the two
//     `libtorch_stable` TUs above); there is no flashinfer / cutlass / TRT-LLM
//     variant in the dense-bf16 path. Their fp8 siblings (`cp_gather_cache`,
//     `USE_FP8_OUTPUT`) are out of campaign scope and refused in ops.cpp.
//
// ─── HOW THE FA-2 LAUNCHER WAS GENERALIZED (W4 left this as W5's job) ───────
// W4 recorded that the vendored FA-2 LAUNCHER did not fit MLA DECODE (separate
// 4-D k/v caches, symmetric head_dim {128,256}, a combine with no notion of
// V width != QK width) but that generalizing it for PREFILL was tractable. It
// was, and the reason is that upstream ALSO does not ask FA-2 for asymmetric
// head dims: `requires_v_padding` is TRUE on GB10 (`flash_attn.py:88-99` exempts
// only FA3-on-SM90 and FA4), so upstream ZERO-PADS V from 128 to 192 and slices
// the output back. The kernel therefore stays a plain SYMMETRIC head_dim-192
// instantiation, and the whole generalization is:
//   (1) two new explicit instantiations of the EXISTING generic template
//       `run_mha_fwd_splitkv_dispatch<bf16, 192, {true,false}>`
//       (flash_fwd_split_hdim192_bf16{,_causal}_sm80.cu — no template edited);
//   (2) a NEW launcher entry point `LaunchMlaPrefillFA2Bf16` for the CONTIGUOUS
//       varlen mode (cu_seqlens_k instead of block_table + seqused_k), which the
//       vendored kernel already supports (`flash_fwd_kernel.h:584-590`);
//   (3) the pad/slice pair below.
// `LaunchPrefillFA2Bf16` — the paged launcher the 27B / 35B / Qwen3-dense
// prefill paths call — is NOT TOUCHED, so those paths are byte-identical by
// construction, not by measurement.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vt/cuda/graph_safe_scratch.h"
#include "vt/ops.h"

namespace vt::cuda {

#ifdef VLLM_CPP_FLASH_ATTN
// Defined in cuda_flash_attn_fa2.cu (the vendored FA-2 adapter).
void LaunchMlaPrefillFA2Bf16(cudaStream_t s, Tensor& out, float* lse_out,
                             int64_t lse_row_stride, const Tensor& query, const Tensor& key,
                             const Tensor& value, const Tensor& cu_seqlens_q,
                             const Tensor& cu_seqlens_k,
                             const MlaPrefillAttentionArgs& args, int64_t num_reqs,
                             int64_t hq, int64_t d);
#endif  // VLLM_CPP_FLASH_ATTN

namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: mla_prefill: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

__device__ __forceinline__ float Ld(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float Ld(const __nv_bfloat16* p, int64_t i) {
  return __bfloat162float(p[i]);
}
__device__ __forceinline__ float Ld(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ __forceinline__ void St(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ void St(__nv_bfloat16* p, int64_t i, float v) {
  p[i] = __float2bfloat16(v);
}
__device__ __forceinline__ void St(__half* p, int64_t i, float v) { p[i] = __float2half(v); }

// ─── grow-only per-stream scratch (the V pad + the padded output) ───────────
// Same house discipline as the W4 decode workspace: on growth the old block is
// RETIRED, never freed, because a captured graph may have baked the pointer.
struct StreamScratch {
  void* buf = nullptr;
  size_t bytes = 0;
};

[[maybe_unused]] std::mutex& ScratchMutex() {
  static std::mutex mu;
  return mu;
}

[[maybe_unused]] StreamScratch& ScratchFor(cudaStream_t s) {
  static std::unordered_map<cudaStream_t, StreamScratch> m;
  return m[s];
}

[[maybe_unused]] void* EnsureScratch(size_t need, cudaStream_t s) {
  StreamScratch& sc = ScratchFor(s);
  if (need > sc.bytes) {
    RetireGraphScratch(sc.buf);
    Check(cudaMallocAsync(&sc.buf, need, s), "cudaMallocAsync mla prefill workspace");
    sc.bytes = need;
  }
  return sc.buf;
}

// ─── V zero-pad, flash_attn.py:164-168 ──────────────────────────────────────
// `torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)`. Zero is
// exact: the padded output columns are sum_j p[j]*0 == 0, so the slice-back
// loses nothing. The destination is CONTIGUOUS [total_k, h, dqk].
template <typename T>
__global__ void PadVKernel(T* __restrict__ dst, const T* __restrict__ src, int64_t total_k,
                           int heads, int dv, int dqk, int64_t src_row_stride,
                           int64_t src_head_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = total_k * heads * dqk;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % dqk);
  const int h = static_cast<int>((idx / dqk) % heads);
  const int64_t t = idx / (static_cast<int64_t>(dqk) * heads);
  const T zero = T(0.0f);
  dst[idx] = d < dv ? src[t * src_row_stride + h * src_head_stride + d] : zero;
}

// ─── output slice-back, flash_attn.py:196-197 ───────────────────────────────
// `attn_out = attn_out[..., : v.shape[-1]]`.
template <typename T>
__global__ void SliceOutKernel(T* __restrict__ dst, const T* __restrict__ src, int64_t total_q,
                               int heads, int dv, int dqk, int64_t dst_row_stride,
                               int64_t dst_head_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = total_q * heads * dv;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % dv);
  const int h = static_cast<int>((idx / dv) % heads);
  const int64_t t = idx / (static_cast<int64_t>(dv) * heads);
  dst[t * dst_row_stride + h * dst_head_stride + d] =
      src[(t * heads + h) * static_cast<int64_t>(dqk) + d];
}

void MlaPrefillAttentionCuda(Queue& q, Tensor& out, Tensor* lse, const Tensor& query,
                             const Tensor& key, const Tensor& value,
                             const Tensor& cu_seqlens_q, const Tensor& cu_seqlens_k,
                             const MlaPrefillAttentionArgs& args) {
  const int64_t total_q = query.shape[0];
  const int64_t total_k = key.shape[0];
  const int64_t heads = query.shape[1];
  const int64_t dqk = query.shape[2];
  const int64_t dv = value.shape[2];
  const int64_t num_reqs = cu_seqlens_q.shape[0] - 1;
  if (total_q == 0 || num_reqs == 0) return;
#ifndef VLLM_CPP_FLASH_ATTN
  (void)out; (void)lse; (void)key; (void)value; (void)cu_seqlens_k; (void)args;
  (void)total_k; (void)heads; (void)dqk; (void)dv;
  throw std::runtime_error(
      "cuda mla_prefill_attention: built without the vendored FlashAttention-2 "
      "(VLLM_CPP_FLASH_ATTN). MLA prefill on sm_121 IS FlashAttention — the upstream "
      "selector has no fallback below it (mla/prefill/selector.py:191-194) — so there "
      "is nothing to degrade to.");
#else
  if (query.dtype != DType::kBF16) {
    throw std::runtime_error(
        "cuda mla_prefill_attention: bf16 only (the vendored FA-2 MLA prefill "
        "instantiation is bf16; upstream's supported_dtypes for this backend are "
        "fp16/bf16 and the model is bf16)");
  }
  cudaStream_t s = AsStream(q);
  std::lock_guard<std::mutex> lk(ScratchMutex());

  // Workspace: the zero-padded V [total_k, heads, dqk] followed by the padded
  // output [total_q, heads, dqk], both contiguous and bf16.
  const size_t v_elems = static_cast<size_t>(total_k) * static_cast<size_t>(heads) *
                         static_cast<size_t>(dqk);
  const size_t o_elems = static_cast<size_t>(total_q) * static_cast<size_t>(heads) *
                         static_cast<size_t>(dqk);
  auto* base = static_cast<__nv_bfloat16*>(
      EnsureScratch((v_elems + o_elems) * sizeof(__nv_bfloat16), s));
  __nv_bfloat16* vpad = base;
  __nv_bfloat16* opad = base + v_elems;

  if (total_k > 0) {
    const int threads = 256;
    const int64_t blocks = (static_cast<int64_t>(v_elems) + threads - 1) / threads;
    PadVKernel<__nv_bfloat16><<<static_cast<unsigned>(blocks), threads, 0, s>>>(
        vpad, value.Ptr<__nv_bfloat16>(), total_k, static_cast<int>(heads),
        static_cast<int>(dv), static_cast<int>(dqk), value.stride[0], value.stride[1]);
    Check(cudaGetLastError(), "pad-V launch");
  }

  Tensor vpad_t{};
  vpad_t.data = vpad;
  vpad_t.dtype = DType::kBF16;
  vpad_t.device = query.device;
  vpad_t.rank = 3;
  vpad_t.shape[0] = total_k;
  vpad_t.shape[1] = heads;
  vpad_t.shape[2] = dqk;
  vpad_t.stride[0] = heads * dqk;
  vpad_t.stride[1] = dqk;
  vpad_t.stride[2] = 1;

  Tensor opad_t = vpad_t;
  opad_t.data = opad;
  opad_t.shape[0] = total_q;

  LaunchMlaPrefillFA2Bf16(s, opad_t, lse != nullptr ? lse->Ptr<float>() : nullptr,
                          lse != nullptr ? lse->stride[0] : 0, query, key, vpad_t,
                          cu_seqlens_q, cu_seqlens_k, args, num_reqs, heads, dqk);

  {
    const int threads = 256;
    const int64_t total = total_q * heads * dv;
    const int64_t blocks = (total + threads - 1) / threads;
    SliceOutKernel<__nv_bfloat16><<<static_cast<unsigned>(blocks), threads, 0, s>>>(
        out.Ptr<__nv_bfloat16>(), opad, total_q, static_cast<int>(heads),
        static_cast<int>(dv), static_cast<int>(dqk), out.stride[0], out.stride[1]);
    Check(cudaGetLastError(), "slice-out launch");
  }
#endif  // VLLM_CPP_FLASH_ATTN
}

// ─── vt::GatherMlaCache — cache_kernels.cu:992-1064 ─────────────────────────
// One block per token (upstream `dim3 grid(num_tokens)`, `:1145`), CTA_SIZE
// threads striding the entry. The index arithmetic is `:1013-1031` verbatim.
template <typename T>
__global__ void GatherMlaCacheKernel(T* __restrict__ dst, const T* __restrict__ src_cache,
                                     const int32_t* __restrict__ block_table,
                                     const int32_t* __restrict__ cu_seq_lens,
                                     const int32_t* __restrict__ token_to_seq,
                                     const int32_t* __restrict__ seq_starts, int num_tokens,
                                     int block_size, int head_dim, int64_t block_table_stride,
                                     int64_t cache_block_stride, int64_t cache_entry_stride,
                                     int64_t dst_entry_stride) {
  for (int token_id = blockIdx.x; token_id < num_tokens; token_id += gridDim.x) {
    const int batch_id = token_to_seq[token_id];
    const int batch_start = cu_seq_lens[batch_id];
    const int batch_end = cu_seq_lens[batch_id + 1];
    if (token_id >= batch_end) return;  // upstream `:1019`
    int batch_offset = token_id - batch_start;
    if (seq_starts != nullptr) batch_offset += seq_starts[batch_id];
    const int block_table_id = batch_offset / block_size;
    const int slot_id = batch_offset % block_size;
    const int block_id =
        block_table[static_cast<int64_t>(batch_id) * block_table_stride + block_table_id];
    const int64_t cache_offset =
        static_cast<int64_t>(block_id) * cache_block_stride + slot_id * cache_entry_stride;
    T* d = dst + static_cast<int64_t>(token_id) * dst_entry_stride;
    const T* sp = src_cache + cache_offset;
    for (int idx = threadIdx.x; idx < head_dim; idx += blockDim.x) d[idx] = sp[idx];
  }
}

void GatherMlaCacheCuda(Queue& q, Tensor& dst, const Tensor& src_cache,
                        const Tensor& block_table, const Tensor& cu_seq_lens,
                        const Tensor& token_to_seq, const Tensor* seq_starts,
                        int64_t num_tokens) {
  if (num_tokens == 0) return;
  cudaStream_t s = AsStream(q);
  const int block_size = static_cast<int>(src_cache.shape[1]);
  const int head_dim = static_cast<int>(src_cache.shape[2]);
  // `constexpr int32_t thread_block_size = 64;` (`:1142`).
  const int threads = 64;
  const unsigned grid = static_cast<unsigned>(num_tokens);
  const int32_t* starts = seq_starts != nullptr ? seq_starts->Ptr<int32_t>() : nullptr;

#define VT_GATHER_MLA(T)                                                                   \
  GatherMlaCacheKernel<T><<<grid, threads, 0, s>>>(                                        \
      dst.Ptr<T>(), src_cache.Ptr<T>(), block_table.Ptr<int32_t>(),                        \
      cu_seq_lens.Ptr<int32_t>(), token_to_seq.Ptr<int32_t>(), starts,                     \
      static_cast<int>(num_tokens), block_size, head_dim, block_table.stride[0],           \
      src_cache.stride[0], src_cache.stride[1], dst.stride[0])
  switch (dst.dtype) {
    case DType::kF32: VT_GATHER_MLA(float); break;
    case DType::kBF16: VT_GATHER_MLA(__nv_bfloat16); break;
    case DType::kF16: VT_GATHER_MLA(__half); break;
    default: throw std::runtime_error("cuda gather_mla_cache: unsupported dtype");
  }
#undef VT_GATHER_MLA
  Check(cudaGetLastError(), "gather_mla_cache launch");
}

// ─── vt::MergeAttnStates — merge_attn_states.cu:18-192 ──────────────────────
// One thread per (token, head, element). Upstream vectorizes 128 bits per
// thread; we keep it scalar so the op accepts any head_size and any (strided)
// layout — the arithmetic, including BOTH -inf edge cases, is identical and is
// what the gate checks. Vectorization is a W9 concern.
template <typename T>
__global__ void MergeAttnStatesKernel(
    T* __restrict__ output, float* __restrict__ output_lse, const T* __restrict__ prefix_output,
    const float* __restrict__ prefix_lse, const T* __restrict__ suffix_output,
    const float* __restrict__ suffix_lse, int num_tokens, int num_heads, int head_size,
    int prefix_num_tokens, int64_t out_row_stride, int64_t out_head_stride,
    int64_t p_row_stride, int64_t p_head_stride, int64_t s_row_stride, int64_t s_head_stride,
    int64_t p_lse_stride, int64_t s_lse_stride, int64_t o_lse_stride) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(num_tokens) * num_heads * head_size;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % head_size);
  const int h = static_cast<int>((idx / head_size) % num_heads);
  const int t = static_cast<int>(idx / (static_cast<int64_t>(head_size) * num_heads));

  const int64_t p_off = t * p_row_stride + h * p_head_stride + d;
  const int64_t s_off = t * s_row_stride + h * s_head_stride + d;
  const int64_t o_off = t * out_row_stride + h * out_head_stride + d;

  // `:66-89` — no context for this token: take the suffix verbatim.
  if (t >= prefix_num_tokens) {
    St(output, o_off, Ld(suffix_output, s_off));
    if (output_lse != nullptr && d == 0) {
      output_lse[h * o_lse_stride + t] = suffix_lse[h * s_lse_stride + t];
    }
    return;
  }

  float p_lse = prefix_lse[h * p_lse_stride + t];
  float s_lse = suffix_lse[h * s_lse_stride + t];
  if (isinf(p_lse)) p_lse = -INFINITY;  // `:97-98`
  if (isinf(s_lse)) s_lse = -INFINITY;
  const float max_lse = fmaxf(p_lse, s_lse);

  // `:100-134` — both -inf would be 0/0; emit the prefix (expected all zeros).
  if (isinf(max_lse)) {
    St(output, o_off, Ld(prefix_output, p_off));
    if (output_lse != nullptr && d == 0) output_lse[h * o_lse_stride + t] = max_lse;
    return;
  }

  const float p_se = __expf(p_lse - max_lse);
  const float s_se = __expf(s_lse - max_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;
  St(output, o_off, Ld(prefix_output, p_off) * p_scale + Ld(suffix_output, s_off) * s_scale);
  if (output_lse != nullptr && d == 0) {
    output_lse[h * o_lse_stride + t] = logf(out_se) + max_lse;
  }
}

void MergeAttnStatesCuda(Queue& q, Tensor& output, Tensor* output_lse,
                         const Tensor& prefix_output, const Tensor& prefix_lse,
                         const Tensor& suffix_output, const Tensor& suffix_lse,
                         int64_t prefill_tokens_with_context) {
  const int64_t num_tokens = output.shape[0];
  const int64_t num_heads = output.shape[1];
  const int64_t head_size = output.shape[2];
  if (num_tokens == 0) return;
  cudaStream_t s = AsStream(q);
  const int64_t total = num_tokens * num_heads * head_size;
  const int threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;
  const int prefix_num_tokens = static_cast<int>(
      prefill_tokens_with_context < 0 ? num_tokens : prefill_tokens_with_context);

#define VT_MERGE_ATTN(T)                                                                    \
  MergeAttnStatesKernel<T><<<static_cast<unsigned>(blocks), threads, 0, s>>>(               \
      output.Ptr<T>(), output_lse != nullptr ? output_lse->Ptr<float>() : nullptr,          \
      prefix_output.Ptr<T>(), prefix_lse.Ptr<float>(), suffix_output.Ptr<T>(),              \
      suffix_lse.Ptr<float>(), static_cast<int>(num_tokens), static_cast<int>(num_heads),   \
      static_cast<int>(head_size), prefix_num_tokens, output.stride[0], output.stride[1],   \
      prefix_output.stride[0], prefix_output.stride[1], suffix_output.stride[0],            \
      suffix_output.stride[1], prefix_lse.stride[0], suffix_lse.stride[0],                  \
      output_lse != nullptr ? output_lse->stride[0] : 0)
  switch (output.dtype) {
    case DType::kF32: VT_MERGE_ATTN(float); break;
    case DType::kBF16: VT_MERGE_ATTN(__nv_bfloat16); break;
    case DType::kF16: VT_MERGE_ATTN(__half); break;
    default: throw std::runtime_error("cuda merge_attn_states: unsupported dtype");
  }
#undef VT_MERGE_ATTN
  Check(cudaGetLastError(), "merge_attn_states launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(
        OpId::kMlaPrefillAttention, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<MlaPrefillAttentionFn>(&MlaPrefillAttentionCuda)));
    RegisterOp(OpId::kGatherMlaCache, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GatherMlaCacheFn>(&GatherMlaCacheCuda)));
    RegisterOp(OpId::kMergeAttnStates, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<MergeAttnStatesFn>(&MergeAttnStatesCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
