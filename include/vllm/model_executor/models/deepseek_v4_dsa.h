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

#include "vt/ops.h"
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


// `KV-DSV4-MULTICACHE` W5 (#2323) — DENSE-CAUSAL MLA attention over a PAGED cache.
//
// The paged counterpart of the forward's step-5 loop, for the arms with no
// indexer selection. It exists so DeepSeek-V4 can stop attending over a
// contiguous `DeepseekV4KvCache::deck` that grows without bound and read the
// runner's paged topology instead.
//
// THE MAPPING THAT MAKES ONE DECODE OP SERVE T QUERIES. `vt::MlaDecodeAttention`
// attends ONE query per batch row over `[0, seq_lens[b])`. A V4 step carries T
// queries whose global positions are `kv_base + t` and whose causal key set is
// `[0, kv_base + t]` (`deepseek_v4.cpp`, the dense-causal `sel` arm). Presenting
// the step as `batch = T` with `seq_lens[t] = kv_base + t + 1` therefore
// reproduces the causal mask EXACTLY, with no mask tensor and no per-token
// launch. Every row shares the same blocks, so `block_table` is the same row
// repeated.
//
//   q       [T * num_heads * head_dim] f32, row-major (t, h, d)
//   sink    [num_heads] f32 — per-head, denominator-only
//   returns [T * num_heads * head_dim] f32
//
// `no_sink` is the `kNoAttnSink` miswire: it feeds `-inf`, which contributes
// nothing to the denominator and so is exactly "no sink".
std::vector<float> PagedCausalMlaAttention(vt::Queue& queue, const std::vector<float>& q,
                                           vt::Tensor& kv_cache, int64_t num_blocks,
                                           int64_t block_size, int64_t num_tokens,
                                           int64_t num_heads, int64_t head_dim,
                                           int64_t kv_base, const std::vector<float>& sink,
                                           float scale, bool no_sink,
                                           // KV-DSV4-MULTICACHE W5 (#2323): the
                                           // SLIDING WINDOW, in tokens. 0 = none
                                           // (attend the whole causal prefix).
                                           // Upstream attends a `swa_only` layer
                                           // over its window and NOTHING else
                                           // (`flashmla.py`, `k_cache=swa_cache`
                                           // with `extra_k_cache=None`), so a
                                           // full-context read of such a layer
                                           // diverges above the window.
                                           int64_t sliding_window = 0,
                                           // MODEL-DSV4-DSA-COMPOSE W1 (#2286):
                                           // the per-(token,head) log-sum-exp,
                                           // `[num_tokens * num_heads]`. Needed
                                           // only to MERGE this pass with another
                                           // over a disjoint key set; null
                                           // otherwise and then never computed.
                                           std::vector<float>* out_lse = nullptr);


// `MODEL-DSV4-DSA-COMPOSE` W1 (#2286) — merge a window pass with a COMPRESSED-HISTORY
// pass, the way upstream's single fused two-cache kernel does it in one call.
//
// `window_out`/`window_lse` are an already-computed sliding-window pass INCLUDING
// the per-head sink. This attends `comp_rows` (`[n_rows, head_dim]` compressed
// latents), then merges the two states by their log-sum-exps.
//
// THE SINK IS DELIBERATELY ABSENT FROM THIS PASS. `vt::MergeAttnStates` combines
// by LSE, each `log sum exp(scores)`, so a sink seeded into BOTH passes lands in
// the merged denominator TWICE and yields a plausible, slightly-too-small result
// that no token gate would catch. It belongs to exactly one contributor, and the
// window pass already carries it.
//
// Every compressed row is visible to every query -- a closed window is history,
// not a windowed neighbour -- so this pass takes no window and no causal bound
// beyond the rows that exist.
std::vector<float> MergeWindowAndCompressed(vt::Queue& queue,
                                            const std::vector<float>& window_out,
                                            const std::vector<float>& window_lse,
                                            const std::vector<float>& q,
                                            const std::vector<float>& comp_rows,
                                            int64_t n_rows, int64_t num_tokens,
                                            int64_t num_heads, int64_t head_dim,
                                            float scale);

// `MODEL-DSV4-DSA-COMPOSE` W1 (#2286) — ONE compressor layer's decode step,
// composed. This is what the paged forward must call before its refusal can
// narrow; the refusal stays until it does, because removing the guard without the
// capability turns a loud refusal into a silent wrong answer.
//
// The `compress_ratio == 128` shape only: `coff == 1`, so no overlapped gathering
// window and no Lightning Indexer. `compress_ratio == 4` is `coff == 2` and is W3.
//
// Per step, in order:
//   1. score = comp_wgate @ x, the pool score this layer selects with
//      (`compressor.py:279-287`).
//   2. `CompressorStepCycle` appends `(kv, score + ape)` to the carried state and
//      pools at each boundary `(pos + 1) % compress_ratio == 0`, returning the rows
//      it emitted this step.
//   3. Those rows append to the layer's compressed history.
//   4. The window pass attends the sliding window WITH the sink, keeping its LSE.
//   5. `MergeWindowAndCompressed` folds the compressed history in, with NO sink --
//      the window pass owns it, and a merged denominator may count it once.
//
// **`kv` IS THE MLA LATENT, AND THAT IS THE COLLAPSED GEOMETRY'S CONVENTION, NOT
// UPSTREAM'S.** Upstream's compressor owns a `fused_wkv_wgate` producing BOTH its
// KV and its gate from the hidden state (`compressor.py:279-287`), and the real
// artifact stores `attn.compressor.wkv.weight` for exactly that. This tree does
// not materialize it -- `deepseek_v4_weights.cpp` accounts it and says so -- since
// the collapsed compressor reuses the MLA's own `kraw`. So on the REAL artifact
// this function would pool the wrong operand: finite, plausible, and not what
// upstream pools. Listed under the row's `## Owed`.
//
// `state_kv`, `state_score` and `comp_rows` are CARRIED ACROSS STEPS by the
// caller. The compressor is a state machine, and its failure mode is a plausible
// value several tokens after the mistake rather than an immediate one.
std::vector<float> CompressorLayerStep(
    vt::Queue& queue, const std::vector<float>& x, const std::vector<float>& kv,
    const std::vector<float>& q, const std::vector<float>& comp_wgate,
    const std::vector<float>& comp_ape, const std::vector<float>& comp_norm_weight,
    const std::vector<float>& attn_sink, vt::Tensor& window_cache,
    int64_t num_blocks, int64_t block_size, std::vector<float>* state_kv,
    std::vector<float>* state_score, std::vector<float>* comp_rows,
    const std::vector<int64_t>& positions, int64_t kv_base, int64_t num_tokens,
    int64_t num_heads, int64_t hidden, int64_t head_dim, int64_t compress_ratio,
    int64_t sliding_window, float eps, float scale,
    // The pooled row's rope tail is rotated at the WINDOW'S BASE position
    // (`fused_compress_quant_cache.py:272-297`). 0 leaves it unrotated.
    int64_t rope_dim = 0, double rope_theta = 10000.0);

// `MODEL-DSV4-DSA-COMPOSE` W3 (#2286) — the INDEXER's compressed keys.
//
// The indexer owns a second `DeepseekCompressor` at `head_dim = index_head_dim`
// (`attention.py:768-776`), and its pooled rows are the KEYS its top-k scores
// against -- not an attention contributor. So this produces keys and nothing
// else; the selection consumes them.
//
// It is the SAME cycle: both compressors go through `compress_norm_rope_store_*`
// and the dispatch names the split ("triton (indexer/AMD)",
// `compressor.py:414-415`). So the rope rule is identical -- the last
// `rope_head_dim` elements, at `(position / compress_ratio) * compress_ratio`.
//
// `rope_dim` is the MODEL's `qk_rope_head_dim`, not a function of
// `index_head_dim`: `compressor.py:240` sets `self.rope_head_dim =
// config.qk_rope_head_dim` whatever the compressor's own head is. At the real
// geometry that is 64 inside a 128-wide indexer head.
//
// `coff` is always 2 here, because the indexer exists only at
// `compress_ratio == 4` (`attention.py:274`).
//
//   returns the compressed key rows emitted this step, [k * index_head_dim]
std::vector<float> IndexerCompressedKeys(
    const std::vector<float>& x, const std::vector<float>& idx_wk,
    const std::vector<float>& idx_comp_wgate, const std::vector<float>& idx_comp_ape,
    const std::vector<float>& idx_comp_norm_weight, std::vector<float>* state_kv,
    std::vector<float>* state_score, const std::vector<int64_t>& positions,
    int64_t num_tokens, int64_t hidden, int64_t index_head_dim, int64_t compress_ratio,
    int64_t rope_dim, double rope_theta, float eps);

// `MODEL-DSV4-DSA-COMPOSE` W3 (#2286) — the indexer's SELECTION, over compressed
// rows.
//
// `_fill_short_context_topk_indices` (`attention.py:71-87`) fixes what an index
// means: `num_compressed = (position + 1) / compress_ratio` rows are available at
// a position, a selection index addresses one of THOSE, and `-1` pads the row out
// to `top_k`. So this returns compressed-row indices, never token indices, and a
// caller must honour `-1` as padding rather than read it as row zero.
//
// The scores are `q . k` over the indexer's heads, folded by the per-head
// weights, which is the same shape the per-token path used -- only the candidate
// set changes, from tokens to closed compressed rows.
//
//   iq        [num_tokens, index_n_heads * index_head_dim]
//   keys      [n_rows, index_head_dim]  the compressed keys, oldest first
//   folded    [num_tokens, index_n_heads]  the weights_proj fold
//   returns   [num_tokens * top_k] compressed-row indices, `-1` padded
std::vector<int64_t> IndexerSelectCompressed(const std::vector<float>& iq,
                                             const std::vector<float>& keys,
                                             const std::vector<float>& folded,
                                             const std::vector<int64_t>& positions,
                                             int64_t num_tokens, int64_t n_rows,
                                             int64_t index_n_heads,
                                             int64_t index_head_dim, int64_t top_k,
                                             int64_t compress_ratio);

// `MODEL-DSV4-DSA-COMPOSE` W3 (#2286) — gather the SELECTED compressed rows.
//
// The `cr == 4` family attends its sliding window plus the compressed rows the
// indexer CHOSE, where the `cr == 128` family attends the window plus ALL closed
// rows. So the only difference at the attention is which rows reach
// `MergeWindowAndCompressed`, and this is what narrows them.
//
// `-1` IS PADDING and is dropped, never read as row zero. A selection row is
// padded out to `top_k` whenever fewer rows have closed than `top_k`, so the
// padded slots are the common case rather than an edge one, and treating one as a
// row attends a real key the indexer did not choose.
//
// Rows come back in SELECTION order, best first, because that is the order the
// indices arrive in; the merge is order-independent, so this is a property of the
// buffer rather than a requirement on it.
//
//   comp_rows  [n_rows, head_dim]   every closed row, oldest first
//   sel        [top_k]              one token's indices, `-1` padded
//   returns    [k, head_dim] with k <= top_k, the selected rows compacted
std::vector<float> GatherSelectedCompressed(const std::vector<float>& comp_rows,
                                            const std::vector<int64_t>& sel,
                                            int64_t n_rows, int64_t head_dim);

}  // namespace vllm::deepseek_v4
