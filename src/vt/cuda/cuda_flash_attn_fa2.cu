// FlashAttention-2 prefill launcher (torch-free, sm_121a) — vendored from
// vllm-project/flash-attention @ 2c839c33 (the exact FA-2 source vLLM 0.24.0
// builds as _vllm_fa2_C and runs on GB10/sm_121; its flash_fwd_splitkv_kernel is
// what vLLM's own profile shows for the 27B/35B prefill:
// flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<256,64,64,4,...>>). This TU
// is a thin, torch-free replacement for FA-2's flash_api.cpp mha_varlen_fwd: it
// fills Flash_fwd_params directly from our vt::Tensor views and calls the
// CUTLASS-templated splitkv dispatch. The kernel itself is compiled from the
// vendored flash_attn/src/*.cu instantiations.
//
// Interface matched to vLLM's flash_attn backend prefill call
// (vllm/v1/attention/backends/flash_attn.py -> flash_attn_varlen_func ->
// vllm_flash_attn/flash_attn_interface.py varlen path): head_dim 256, GQA
// (h_h_k_ratio), bf16 in / bf16 out, per-sequence causal, PAGED KV (block_table)
// + varlen (query_start_loc as cu_seqlens_q, seq_lens as seqused_k).
//
// num_splits is ALWAYS 1 here (Split=false: the main kernel writes O directly
// through the varlen cu_seqlens_q offsets; no combine pass). This is exactly
// vLLM's FA-2 varlen behaviour — flash_attn_interface.py:309-310 raises
// NotImplementedError("FA2 does not support num_splits > 1") on the varlen
// path — and it is structurally required: the split-KV combine kernel writes O
// with BATCHED addressing (flash_fwd_kernel.h:1291, batch_idx*o_batch_stride),
// which is wrong for a ragged varlen-packed O. A previous revision carried the
// decode-style num_splits heuristic + accumulator pool; it measured
// num_splits==1 at every gate shape and crashed when forced >1 on ragged
// prefill (.agents/parity-ledger.md 2026-07-06 FA-2 split-KV row) — removed.
//
// The launcher is SYNC-FREE on the production path: max_seqlen_q comes from
// PagedAttentionArgs::query_start_loc_host and max_seqlen_k from
// PagedAttentionArgs::max_seq_len (both host-known per step, set by
// FullAttnBlockPaged from CommonAttentionMetadata — the same device-resident-
// metadata pattern the WMMA prefill launchers use). The previous revision did a
// per-layer D2H copy + cudaStreamSynchronize here, re-introducing the exact
// pipeline drain the query_start_loc_host mechanism was built to remove (ops.h
// PagedAttentionArgs note: ~10-12 syncs/step, prefill only 43.7% GPU-busy);
// op-unit-test callers without host metadata still get the D2H+sync fallback.
//
// It is also CAST-FREE: bf16 query/out only. The previous revision accepted an
// f32 query/out and bridged with f32<->bf16 cast kernels + per-call
// cudaMallocAsync scratch — measured to ERASE the attention win end-to-end
// (.agents/parity-ledger.md 2026-07-06: "the f32<->bf16 CAST kernels (~35ms/run)
// ... ERASE the attention win"). The production 27B path now hands bf16
// natively (fused preamble emits bf16 q/k, attention output is bf16 into the
// sigmoid gate); anything else falls through to the WMMA ladder.
//
// Gated at compile time by VLLM_CPP_FLASH_ATTN and at runtime by
// VT_FA2_PREFILL (see cuda_paged_attn.cu Fa2PrefillEnabled); eligibility
// (prefill segment, head_dim 256, bf16 q/KV/out) is enforced by the LaunchPaged
// dispatch gate. Decode stays on the existing kernels: the decode path is
// CUDA-graph captured (this launcher's pooled scratch + host-side grid sizing
// are not capture-safe) and decode attention is already at parity
// (PagedAttentionDecodeGqaKernel, ledger 2026-07-06 decode-GQA row).
#ifdef VLLM_CPP_FLASH_ATTN

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cutlass/numeric_types.h>  // cutlass::bfloat16_t

// Torch-free FA-2 params struct + the splitkv dispatch *declaration* (the
// definition lives in the vendored flash_attn/src/*.cu instantiations, linked in).
#include "namespace_config.h"
#include "flash.h"

#include "vt/ops.h"

namespace vt::cuda {
namespace {

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cuda flash-attn-2 prefill: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

inline int RoundMultiple(int x, int m) { return (x + m - 1) / m * m; }

// Persistent (grow-only) device scratch for softmax_lse [nheads, total_q] f32.
// The kernel unconditionally writes the LSE; we never consume it. Sized by the
// largest (hq * total_q) seen and reused across layers/steps — the prefill
// token count is bounded by max_num_batched_tokens, so this converges after the
// first full-budget step (no per-call cudaMalloc/Free on the hot path).
struct LsePool {
  std::mutex mu;
  float* ptr = nullptr;
  size_t floats = 0;

  float* Ensure(size_t need) {
    std::lock_guard<std::mutex> lock(mu);
    if (need > floats) {
      if (ptr != nullptr) cudaFree(ptr);
      if (cudaMalloc(&ptr, need * sizeof(float)) != cudaSuccess) {
        ptr = nullptr;
        floats = 0;
        throw std::runtime_error("cuda flash-attn-2 prefill: softmax_lse pool alloc failed");
      }
      floats = need;
    }
    return ptr;
  }
};

LsePool& SoftmaxLsePool() {
  static LsePool pool;
  return pool;
}

}  // namespace

// Launch FA-2 paged varlen prefill for a bf16 query + bf16 KV cache + bf16 out,
// head_dim 256. Layouts (elements, matching our LaunchPaged dispatch):
//   query [total_q, hq, d]                          (varlen-packed, bf16)
//   k/v   [num_blocks, block_size, num_kv_heads, d] (paged, NHD block layout, bf16)
//   out   [total_q, hq, d]                          (varlen-packed, bf16)
//   block_table [num_reqs, max_blocks] i32, seq_lens [num_reqs] i32 (K lengths
//   incl. context), query_start_loc [num_reqs+1] i32 (cumulative q offsets).
void LaunchPrefillFA2Bf16(cudaStream_t s, Tensor& out, const Tensor& query,
                          const Tensor& k_cache, const Tensor& v_cache,
                          const Tensor& block_table, const Tensor& seq_lens,
                          const Tensor& query_start_loc, const PagedAttentionArgs& args,
                          int64_t hq, int64_t d, int64_t num_reqs, int64_t num_kv_heads,
                          int64_t block_size) {
  const int64_t total_q = query.shape[0];
  if (total_q == 0 || num_reqs == 0 || hq == 0 || d == 0) return;
  if (query.dtype != DType::kBF16 || out.dtype != DType::kBF16) {
    throw std::runtime_error(
        "cuda flash-attn-2 prefill: bf16 query/out required (dispatch gate must enforce)");
  }

  // FA-2 needs max_seqlen_q/max_seqlen_k on the host — for GRID SIZING and the
  // rounded dims only; the per-request causal geometry reads the DEVICE
  // cu_seqlens_q/seqused_k via BlockInfo, so host UPPER BOUNDS are safe.
  // Production callers hand both through PagedAttentionArgs
  // (query_start_loc_host + max_seq_len, built once per step); callers without
  // them (op unit tests) fall back to a small D2H + sync, same as the WMMA
  // launchers' fallback.
  int max_seqlen_q = 0;
  int max_seqlen_k = args.max_seq_len;
  if (args.query_start_loc_host != nullptr) {
    for (int64_t i = 0; i < num_reqs; ++i) {
      max_seqlen_q = std::max(
          max_seqlen_q, args.query_start_loc_host[i + 1] - args.query_start_loc_host[i]);
    }
  }
  if (max_seqlen_q <= 0 || max_seqlen_k <= 0) {
    std::vector<int32_t> qsl(static_cast<size_t>(num_reqs) + 1);
    std::vector<int32_t> sk(static_cast<size_t>(num_reqs));
    Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(),
                          sizeof(int32_t) * (num_reqs + 1), cudaMemcpyDeviceToHost, s),
          "query_start_loc D2H");
    Check(cudaMemcpyAsync(sk.data(), seq_lens.Ptr<int32_t>(), sizeof(int32_t) * num_reqs,
                          cudaMemcpyDeviceToHost, s),
          "seq_lens D2H");
    Check(cudaStreamSynchronize(s), "seqlen sync");
    max_seqlen_q = 0;
    max_seqlen_k = 0;
    for (int64_t i = 0; i < num_reqs; ++i) {
      max_seqlen_q = std::max(max_seqlen_q, qsl[i + 1] - qsl[i]);
      max_seqlen_k = std::max(max_seqlen_k, sk[i]);
    }
  }
  if (max_seqlen_q == 0 || max_seqlen_k == 0) return;

  // softmax_lse [nheads, total_q] f32 (unpadded/varlen LSE). Written by the
  // kernel; not consumed. Grow-only pool (no per-call malloc on the hot path).
  float* softmax_lse =
      SoftmaxLsePool().Ensure(static_cast<size_t>(hq) * static_cast<size_t>(total_q));

  FLASH_NAMESPACE::Flash_fwd_params p{};  // zero-init: nulls knew/rotary/alibi/accum/leftpad
  p.is_bf16 = true;

  p.q_ptr = query.data;
  p.k_ptr = k_cache.data;
  p.v_ptr = v_cache.data;
  p.o_ptr = out.data;

  // Strides in ELEMENTS. Varlen q/o: no batch stride (cu_seqlens_q drives offset).
  p.q_row_stride = query.stride[0];
  p.q_head_stride = query.stride[1];
  p.o_row_stride = out.stride[0];
  p.o_head_stride = out.stride[1];
  // Paged k/v [num_blocks, block_size, num_kv_heads, d]:
  //   batch_stride = per-block, row_stride = per page-position, head_stride = per head.
  p.k_batch_stride = k_cache.stride[0];
  p.k_row_stride = k_cache.stride[1];
  p.k_head_stride = k_cache.stride[2];
  p.v_batch_stride = v_cache.stride[0];
  p.v_row_stride = v_cache.stride[1];
  p.v_head_stride = v_cache.stride[2];

  p.cu_seqlens_q = query_start_loc.Ptr<int32_t>();
  p.cu_seqlens_k = nullptr;               // paged: block_table + seqused_k drive K
  p.seqused_k = seq_lens.Ptr<int32_t>();  // per-request K length (BlockInfo::actual_seqlen_k)
  p.softmax_lse_ptr = softmax_lse;

  p.b = static_cast<int>(num_reqs);
  p.h = static_cast<int>(hq);
  p.h_k = static_cast<int>(num_kv_heads);
  p.h_h_k_ratio = static_cast<int>(hq / num_kv_heads);
  p.seqlen_q = max_seqlen_q;
  p.seqlen_k = max_seqlen_k;
  p.seqlen_q_rounded = RoundMultiple(max_seqlen_q, 128);
  p.seqlen_k_rounded = RoundMultiple(max_seqlen_k, 128);
  p.d = static_cast<int>(d);
  p.d_rounded = RoundMultiple(static_cast<int>(d), 32);
  p.total_q = static_cast<int>(total_q);

  p.scale_softmax = args.scale;
  p.scale_softmax_log2 = args.scale * static_cast<float>(M_LOG2E);
  p.softcap = 0.0f;

  // Dropout disabled (inference): keep-prob 1.0.
  p.p_dropout = 1.0f;
  p.p_dropout_in_uint8_t = uint8_t(255);
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = args.scale;
  p.philox_args = at::PhiloxCudaState(0, 0);

  p.is_causal = args.causal;
  p.window_size_left = -1;
  p.window_size_right = args.causal ? 0 : -1;
  p.is_seqlens_k_cumulative = true;  // ignored while cu_seqlens_k == nullptr
  p.is_rotary_interleaved = false;
  p.rotary_dim = 0;

  p.block_table = block_table.Ptr<int32_t>();
  p.block_table_batch_stride = block_table.stride[0];
  p.page_block_size = static_cast<int>(block_size);

  p.unpadded_lse = true;  // LSE is [nheads, total_q]
  p.seqlenq_ngroups_swapped = false;

  // ALWAYS 1 on the varlen prefill path (see header): Split=false, the main
  // kernel writes O directly through cu_seqlens_q. o_batch_stride is unused on
  // this path (varlen q_offset ignores it) but set consistently anyway.
  p.num_splits = 1;
  p.o_batch_stride = static_cast<int64_t>(max_seqlen_q) * p.o_row_stride;

  // head_dim 256 is gated by the caller. Causal is per-layer (qwen3.5 mixes full-
  // attn causal / GDN); dispatch both instantiations (compiled for sm_121a).
  // With num_splits==1 run_mha_fwd_splitkv_dispatch runs the Split=false kernel
  // only — no combine pass, exactly vLLM's varlen prefill.
  if (args.causal) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(p, s);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p, s);
  }
  Check(cudaGetLastError(), "splitkv dispatch launch");
}

}  // namespace vt::cuda

#endif  // VLLM_CPP_FLASH_ATTN
