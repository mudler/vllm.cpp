// FlashAttention-2 prefill launcher (torch-free, sm_121a) — vendored from
// vllm-project/flash-attention @ 2c839c33 (the exact FA-2 source vLLM 0.24.0
// builds as _vllm_fa2_C and runs on GB10/sm_121; its flash_fwd_splitkv_kernel is
// what vLLM's own nsys trace shows for prefill). This TU is a thin, torch-free
// replacement for FA-2's flash_api.cpp: it fills Flash_fwd_params directly from
// our vt::Tensor views and calls the CUTLASS-templated splitkv dispatch. The
// kernel itself is compiled from the vendored flash_attn/src/*.cu instantiations.
//
// Requirements matched to our WMMA prefill path: head_dim 256, GQA (h_h_k_ratio),
// bf16, per-sequence causal, PAGED KV (block_table) + varlen (query_start_loc as
// cu_seqlens_q, seq_lens as seqused_k). Paged KV forces FA-2's split kernel with
// num_splits=1 (Split=false → the main kernel writes O directly; no combine pass),
// exactly as FA-2's mha_varlen_fwd does (run_mha_fwd(params, stream, paged_KV)).
//
// Gated at compile time by VLLM_CPP_FLASH_ATTN and at runtime by VT_ATTN_FA2
// (default OFF); the proven WMMA kernel remains the default prefill path.
#ifdef VLLM_CPP_FLASH_ATTN

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
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

}  // namespace

// Launch FA-2 paged prefill for a bf16 query + bf16 KV cache, head_dim 256.
// Layouts (elements, matching our LaunchPaged dispatch):
//   query [total_q, hq, d]                          (varlen-packed)
//   k/v   [num_blocks, block_size, num_kv_heads, d] (paged, NHD block layout)
//   out   [total_q, hq, d]
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

  // FA-2 needs max_seqlen_q/max_seqlen_k on the host (grid + rounded dims). Small
  // D2H (num_reqs ints) + one stream sync — the same host-build shape our WMMA
  // prefill launcher already uses. query_start_loc is cumulative; seq_lens holds
  // the per-request K length (context + query) — our seqused_k.
  std::vector<int32_t> qsl(static_cast<size_t>(num_reqs) + 1);
  std::vector<int32_t> sk(static_cast<size_t>(num_reqs));
  Check(cudaMemcpyAsync(qsl.data(), query_start_loc.Ptr<int32_t>(),
                        sizeof(int32_t) * (num_reqs + 1), cudaMemcpyDeviceToHost, s),
        "query_start_loc D2H");
  Check(cudaMemcpyAsync(sk.data(), seq_lens.Ptr<int32_t>(), sizeof(int32_t) * num_reqs,
                        cudaMemcpyDeviceToHost, s),
        "seq_lens D2H");
  Check(cudaStreamSynchronize(s), "seqlen sync");
  int max_seqlen_q = 0;
  int max_seqlen_k = 0;
  for (int64_t i = 0; i < num_reqs; ++i) {
    max_seqlen_q = std::max(max_seqlen_q, qsl[i + 1] - qsl[i]);
    max_seqlen_k = std::max(max_seqlen_k, sk[i]);
  }
  if (max_seqlen_q == 0 || max_seqlen_k == 0) return;

  // softmax_lse [nheads, total_q] f32 (unpadded/varlen LSE). Written by the main
  // splitkv kernel; we don't consume it but the kernel requires a valid buffer.
  float* softmax_lse = nullptr;
  Check(cudaMallocAsync(&softmax_lse, sizeof(float) * hq * total_q, s), "softmax_lse alloc");

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

  p.num_splits = 1;         // paged varlen: no KV split (Split=false, direct O write)
  p.unpadded_lse = true;    // LSE is [nheads, total_q]
  p.seqlenq_ngroups_swapped = false;

  // head_dim 256 is gated by the caller. Causal is per-layer (qwen3.5 mixes full-
  // attn causal / GDN); dispatch both instantiations (compiled for sm_121a).
  if (args.causal) {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(p, s);
  } else {
    FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(p, s);
  }
  Check(cudaGetLastError(), "splitkv dispatch launch");
  Check(cudaFreeAsync(softmax_lse, s), "softmax_lse free");
}

}  // namespace vt::cuda

#endif  // VLLM_CPP_FLASH_ATTN
