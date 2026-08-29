// DeepSeek-V4-Flash W3 — the genuinely-NEW attention primitives, as portable
// host (CPU) reference implementations. This is the FORWARD CODE for the two
// items W3 owns:
//
//   (A) the DSA "Lightning Indexer" sparse top-k SELECTION math
//       (DeepseekV4Indexer, attention.py:689-857; the MQA-logit kernel
//        v1/attention/ops/triton_fp8_mqa_logits.py:48-156; the short-context
//        all-select attention.py:70-86 / :813-831), and
//   (B) the 512-wide MLA output seams that V2/V3 do NOT have —
//       per-head ATTENTION SINKS in the softmax (flashinfer_sparse.py:777,:896)
//       and the grouped OUTPUT-LoRA (wo_a bmm einsum + wo_b,
//       nvidia/ops/o_proj.py:28-73).
//
// WHY host/CPU reference (honest scope): the full DeepSeek-V4-Flash forward is a
// multi-Spark campaign — the checkpoint is 156.7 GiB (does not fit one GB10, see
// deepseek_v4.h) and the forward also needs MHC (W5) + the sqrtsoftplus/hash MoE
// (W6), neither ported yet. So W3 lands + UNIT-GATES these primitives against a
// hand-derived small case and a from-first-principles double-precision reference
// (tests/vllm/models/test_deepseek_v4_dsa.cpp), rather than a full-model gate. The
// eventual GPU forward (W7) will call the SAME math from a CUDA kernel; this file
// pins the numerics portably so the kernel port has an oracle.
//
// SACRED-inert: additive TU only. It does NOT touch the DeepSeek-V2 MLA path
// (src/vllm/model_executor/layers/attention/mla_attention.{h,cpp},
//  src/vt/cuda/cuda_mla_attn.cu) — per the W3 brief, extending the SHARED MLA
// block risks V2, so V4's new geometry lands as a V4-specific path here and the
// shared-mla extraction is a NAMED follow-on (W7 integration). Reuse points that
// integration WILL share are cited inline.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   OURS                          <-  UPSTREAM (vllm/, @ 0.26.0.dev0)
//   DsaIndexerLogits              <-  v1/attention/ops/triton_fp8_mqa_logits.py
//                                     :120-156 (scores=q·k, *kv_scale, ReLU,
//                                      *weights, sum over heads) — the ReLU is
//                                      the load-bearing nuance
//   DsaIndexerWeightFold          <-  model_executor/layers/sparse_attn_indexer.py
//                                     :203-207 (weight * q_scale * softmax_scale
//                                      * head_scale), softmax_scale=head_dim**-0.5
//                                      head_scale=n_head**-0.5 (attention.py:735,:843)
//   DsaTopkSelect                 <-  sparse_attn_indexer.py:488-497
//                                     (top_k_per_row) + the short-context
//                                     all-candidate select (attention.py:70-86,
//                                     :813-831): if #candidates <= topk, take
//                                     [0..n-1] then -1 padding
//   SoftmaxWithSink               <-  attention sinks: flashinfer_sparse.py
//                                     sinks=self.attn_sink (:777,:896); the sink
//                                     is an extra per-head logit in the softmax
//                                     denominator that carries NO value (init
//                                     -inf = no effect, attention.py:219-222)
//   GroupedOutputLora             <-  nvidia/ops/o_proj.py:58-73
//                                     (z[b,h,d]=einsum "bhr,hdr->bhd"; then
//                                      wo_b(z.flatten)); shapes attention.py
//                                     :243-260 (wo_a in=n_heads*head_dim/n_groups,
//                                      out=n_groups*o_lora_rank, is_bmm; wo_b
//                                      n_groups*o_lora_rank -> hidden_size)
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::deepseek_v4 {

// ── (A) DSA Lightning Indexer ────────────────────────────────────────────────

// Fold the per-(token,head) indexer gate weight the way the fused q-rope-quant
// kernel does (sparse_attn_indexer.py:203-207): the value the MQA-logit sum
// multiplies each head's ReLU'd score by is
//     weight[t,h] * softmax_scale * head_scale
// with softmax_scale = index_head_dim**-0.5 and head_scale = index_n_heads**-0.5.
// (The fp8 path also folds a per-token q_scale here; the fp32 reference has
// q_scale == 1, so it is omitted — documented, not silently dropped.)
// `weights_proj` is [T, H] row-major; returns [T, H] row-major folded weights.
std::vector<float> DsaIndexerWeightFold(const std::vector<float>& weights_proj,
                                        int64_t num_tokens, int64_t index_n_heads,
                                        int64_t index_head_dim);

// The MQA "Lightning Indexer" logit for one query row t over key row s:
//     logit[t,s] = sum_h folded_weights[t,h] * ReLU( dot(q[t,h,:], k[s,:]) )
// (triton_fp8_mqa_logits.py:125-132 — dot, *kv_scale(==1 fp32), ReLU, *weights,
//  sum over heads). Keys outside the causal window [win_start[t], win_end[t])
// are set to -inf so a plain top-k needs no masking (matches the kernel's
// -inf prefill, :207).
//
//   q               : [num_tokens, index_n_heads, index_head_dim] row-major
//   k               : [num_keys,   index_head_dim]                row-major (MQA: 1 KV head)
//   folded_weights  : [num_tokens, index_n_heads]                row-major (DsaIndexerWeightFold)
//   win_start/win_end: per-query causal candidate range [start,end) into keys
// Returns logits [num_tokens, num_keys] row-major.
std::vector<float> DsaIndexerLogits(const std::vector<float>& q,
                                    const std::vector<float>& k,
                                    const std::vector<float>& folded_weights,
                                    const std::vector<int64_t>& win_start,
                                    const std::vector<int64_t>& win_end,
                                    int64_t num_tokens, int64_t num_keys,
                                    int64_t index_n_heads, int64_t index_head_dim);

// Per-row sparse top-k selection (sparse_attn_indexer.py:488-497 top_k_per_row +
// the short-context all-select attention.py:70-86,:813-831). For each query row:
//   - let n = number of valid (finite) candidates in [win_start,win_end);
//   - if n <= topk: emit [win_start, win_start+1, ..., win_start+n-1] then -1 pad
//     (short-context: EVERY candidate selected, in ascending key order);
//   - else: emit the topk key indices with the largest logits, ascending-index
//     order among the chosen set; ties broken by SMALLER key index (stable).
// Returns [num_tokens, topk] row-major int64 index buffer (-1 = no token).
std::vector<int64_t> DsaTopkSelect(const std::vector<float>& logits,
                                   const std::vector<int64_t>& win_start,
                                   const std::vector<int64_t>& win_end,
                                   int64_t num_tokens, int64_t num_keys,
                                   int64_t topk);

// ── (B) 512-wide MLA output seams (NEW vs V2/V3) ─────────────────────────────

// Attention-sink softmax over one query's `num_keys` scores with an extra
// per-head sink logit (flashinfer_sparse.py sinks=self.attn_sink). The sink
// participates in the softmax DENOMINATOR but carries no value, so the returned
// probabilities sum to < 1 by exactly the sink's share:
//     m       = max(max_j scores[j], sink)
//     denom   = sum_j exp(scores[j]-m) + exp(sink-m)
//     prob[j] = exp(scores[j]-m) / denom
// A sink of -inf reduces to a plain softmax (the -inf param init, no effect).
// Numerically stable (max-subtraction). Returns [num_keys] probabilities.
std::vector<float> SoftmaxWithSink(const std::vector<float>& scores, float sink);

// Grouped OUTPUT-LoRA (o_proj.py:58-73). The per-head attention output is
// reshaped into `n_groups` groups of `heads_per_group*head_dim` and each group
// is projected by its own [o_lora_rank x in_per_group] slab (the wo_a bmm /
// einsum "bhr,hdr->bhd"), the per-group results are concatenated to
// [n_groups*o_lora_rank] and projected to hidden by wo_b:
//     z[t,g,:]  = wo_a[g] @ o_group[t,g,:]                      (per-group matmul)
//     out[t,:]  = wo_b @ concat_g z[t,g,:]
// (fp32 reference; the inverse-RoPE + fp8 quant that precede the einsum on GPU
//  reuse our decoupled-RoPE machinery with a negated angle and are a W7 seam —
//  omitted here so the gate isolates the grouped-LoRA linear algebra.)
//
//   o        : [num_tokens, n_heads, head_dim]                 row-major
//   wo_a     : [n_groups, o_lora_rank, in_per_group]           row-major,
//              in_per_group = n_heads*head_dim/n_groups = heads_per_group*head_dim
//   wo_b     : [hidden_size, n_groups*o_lora_rank]             row-major
// Returns out [num_tokens, hidden_size] row-major.
// `W` is `float` (the ported upstream-parity arm) or `uint16_t` (bf16 bit
// patterns -- the carried tower's FP8-sourced half at the model dtype, W1d #2186).
// Defined in `deepseek_v4_dsa.cpp` with both instantiations explicit, so the two
// arms share ONE body and cannot drift apart.
template <typename W>
std::vector<float> GroupedOutputLora(const std::vector<float>& o,
                                     const std::vector<W>& wo_a,
                                     const std::vector<W>& wo_b,
                                     int64_t num_tokens, int64_t n_heads,
                                     int64_t head_dim, int64_t n_groups,
                                     int64_t o_lora_rank, int64_t hidden_size);

}  // namespace vllm::deepseek_v4
