// Qwen4-Exp (Qwen3.8-Flash-Next) W4 — Qwen Sparse Attention: the indexer, its
// side cache, and the CONSUMER. Portable host (CPU) reference implementations.
//
// Row MODEL-MM-QWEN4-EXP, issue #1991, spec .agents/specs/qwen4-exp-flash-next.md.
//
// ─── THE POINT OF THIS FILE: A GATHER, NOT A MASK ───────────────────────────
// llama.cpp #27739 records, with the mechanism named, that a sparse MASK over a
// dense cache costs the same as dense attention under CUDA flash attention,
// because `flash_attn_mask_to_KV_max` only scans back to the first tile that is
// not all -inf. llama.cpp #27742 is mask-only and therefore buys correctness
// without decode speed. So the consumer here is `QsaGatherAttention`, which
// expands selected block `b` into tokens [CR*b, CR*b + CR), appends the ragged
// tail, and reduces over ONLY those rows. `QsaMaskedAttention` exists solely as
// the red-first reference the gather is gated against; it is O(kv_len) by
// construction and must never be the production path.
//
// Both consumers report `keys_visited`, and both COUNT it at the key-row read
// rather than assign it from the index buffer. That distinction is the whole
// instrument: a counter set to `sel.size()` is an assertion about `indices`, and
// a body doing the dense work under the gather's name still satisfies it (W4
// fresh review, mutation M22c, green on the first revision of this file). Counted
// at the `Dot`, the gather reports `selected * num_q_heads * 2` and the mask
// reports `kv_len * num_q_heads * 2`, so the ratio between them is measured.
// Correctness alone cannot distinguish the two implementations, which is exactly
// why a token gate would have let a mask through.
//
// ─── ORACLES (AGENTS.md 'vLLM is the reference' / 'When vLLM has no impl') ───
// vLLM registers NO `qwen4_exp` at origin/main = 6a5e8f5979 (read 2026-08-26),
// so per the developer direction recorded in the spec's `## Oracles`,
// transformers supplies the ALGORITHM and vLLM supplies the OPS.
//
//   ALGORITHM  huggingface/transformers v5.16.0 (the lane pin)
//              src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
//              `Qwen4ExpTextQSAIndexer` :611-717, `Qwen4ExpTextAttention` :720+
//              (authored delta: modular_qwen4_exp.py :367-507)
//   OPS        vllm-project/vllm origin/main 6a5e8f5979
//
// ─── PORT MAP, file:line on BOTH sides ──────────────────────────────────────
//  OURS                      <- ALGORITHM (transformers)   + OPS (vLLM)
//  QsaValidateConfig         <- configuration_qwen4_exp.py :221-231
//                               (kv_heads == 1; budget % ratio == 0;
//                                rotary_dim = int(head_dim * partial_rotary
//                                _factor) must fit indexer_head_dim)
//  QsaSideCacheSpec          <- Cache.update_indexer
//                             + v1/kv_cache_interface.py MLAAttentionSpec with
//                               compress_ratio (:386), whose page runs off
//                               storage_block_size = block_size //
//                               compress_ratio (:393-395). Key-only: one
//                               vector per state, not 2x for K+V.
//  QsaCompressedSlot         <- v1/attention/backends/mla/compressor_utils.py
//                               :49-61 `_compressed_slot_mapping_kernel`
//                               (`(pos + 1) % COMPRESS_RATIO == 0` boundary,
//                                `pos // COMPRESS_RATIO` slot, PAD_ID -1)
//  QsaRmsNorm                <- modeling_qwen4_exp.py :167-179
//                               `out * rsqrt(mean(x^2) + eps) * (1.0 + weight)`
//                               — NOT vLLM's `out * weight` polarity
//  QsaApplyRotaryLeadingHalf <- modeling_qwen4_exp.py :566-570,:573-608
//                               `rotate_half` over the LEADING rotary_dim dims
//  QsaCompressNormRope       <- modeling_qwen4_exp.py :677-688 (mean pool over a
//                               NON-overlapping window of compress_ratio, the
//                               `.to(raw_keys.dtype)` round-trip, k_layernorm,
//                               RoPE at the block's FIRST token)
//                             + scaffolding from the TRITON head_dim=128 kernel
//                               models/deepseek_v4/common/ops/fused_compress_
//                               quant_cache.py :677-830: boundary predicate
//                               :729-731, gather window :735-736, paged store
//                               :783-795, block-start RoPE :816.
//  QsaBlockScores            <- modeling_qwen4_exp.py :690-693
//                               relu(q.k) summed over index heads, / sqrt(D)
//                             + v1/attention/ops/triton_fp8_mqa_logits.py
//                               :120-156 (the ReLU-then-sum MQA logit shape)
//  QsaTopkBlocks             <- modeling_qwen4_exp.py :695
//                             + csrc/libtorch_stable/sampler.cu :391-411
//                               (the rowLen <= topK shortcut: every candidate,
//                                ASCENDING, -1 padded) and :515 (ties resolve to
//                                the LOWER candidate index). Both are inherited
//                                for consistency with the op and neither is
//                                exercised by the fixtures — see QsaTopkBlocks.
//  QsaSelectedTokenIndices   <- modeling_qwen4_exp.py :697-703 (block -> token
//                               remap, the ALWAYS-attended ragged tail, and the
//                               budget + compress_ratio - 1 buffer width)
//  QsaGatherAttention        <- NOTHING UPSTREAM. Every DSv4 sparse consumer
//                               attends the COMPRESSED MLA KV (one state per
//                               four tokens); MiniMax-M3's consumers attend raw
//                               tokens but only at KV-page granularity. QSA
//                               attends RAW tokens selected at ratio-4
//                               granularity, which no vLLM consumer does.
//  QsaMaskedAttention        <- modeling_qwen4_exp.py :705-717 + :491-496
//                               (the scatter mask and its `&` with the causal
//                                mask). REFERENCE ONLY.
//
// ─── TWO THINGS DELIBERATELY NOT INHERITED FROM DeepSeek-V4 ─────────────────
// 1. `weights_proj`. DSv4 folds a per-(token,head) learned logit weight
//    (our DsaIndexerWeightFold, sparse_attn_indexer.py :203-207, plus a
//    head_scale of n_heads**-0.5 from models/deepseek_v4/attention.py :930,
//    `self.n_head**-0.5` in the indexer call).
//    QSA has no such tensor and no head_scale: its weight is the constant
//    1/sqrt(indexer_head_dim), applied AFTER the sum over heads.
// 2. The RoPE geometry. The DSv4 indexer kernel is GPT-J style over a TRAILING
//    contiguous span (`rope_pair_local = pair_idx - NOPE_PAIRS`, adjacent
//    even/odd pairs, fused_compress_quant_cache.py :806-818). QSA is NeoX
//    `rotate_half` over the LEADING rotary_dim dims with the NoPE dims trailing.
//    The halves are swapped end for end AND the pairing convention differs.
// A third: DeepSeek-V4's compressor does NOT mean-pool, in either of its two
// implementations. `score = tl.softmax(score, dim=0); sum(kv * score)` is a
// LEARNED softmax pool over an OVERLAPPING window of
// `(1 + OVERLAP) * COMPRESS_RATIO`, driven by a score channel this checkpoint
// does not have. Those line numbers are the TRITON
// `_fused_kv_compress_norm_rope_insert_indexer_attn`
// (fused_compress_quant_cache.py :677, softmax at :769); the CuteDSL
// `SparseAttnCompressNormRopeStoreC4Kernel` is a different file
// (models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py :75) and does
// the same online softmax at :1121-1130. Only the scaffolding is ported.
//
// ─── WHY HOST REFERENCE, AND WHAT IS UNREACHED ──────────────────────────────
// AGENTS.md 'Nothing lands dead': this TU lands UNREACHED. `Qwen4ExpTextModel`
// does not exist yet — its PLE layer (#1987), hyper-connection stream (#1988)
// and GGUF reader (#1989) are three sibling waves, and the registry entry and
// runner wiring belong to W5. Row MODEL-MM-QWEN4-EXP owns that wiring and issue
// #1978 tracks it; the spec lists this slice under `## Owed`. The device arm is
// owed too: the eventual CUDA kernel ports this same math and these functions
// are its portable oracle, exactly as deepseek_v4_dsa.h is for cuda_deepseek_v4.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm::qwen4_exp {

// ── Config ───────────────────────────────────────────────────────────────────

// The published Qwen3.8-Flash-Next values are the defaults. `rotary_dim` is NOT
// an independent key: upstream derives it as
// `int(head_dim * partial_rotary_factor)` = int(256 * 0.25) = 64 and then
// requires it to fit `indexer_head_dim` (configuration_qwen4_exp.py :225-231).
struct QsaConfig {
  int64_t index_n_heads = 4;
  int64_t index_kv_heads = 1;  // upstream requires EXACTLY 1
  int64_t index_head_dim = 128;
  int64_t token_budget = 2048;  // TOKENS, not blocks
  int64_t compress_ratio = 4;
  int64_t rotary_dim = 64;  // int(head_dim * partial_rotary_factor)
  float rms_norm_eps = 1e-6f;

  // block_topk = token_budget // compress_ratio (modeling_qwen4_exp.py :622).
  int64_t block_topk() const { return token_budget / compress_ratio; }
  // The index buffer is `token_budget + compress_ratio - 1` wide because the
  // incomplete trailing block is ALWAYS attended on top of the budget
  // (modeling_qwen4_exp.py :661-666,:700-701).
  int64_t index_width() const { return token_budget + compress_ratio - 1; }
};

// Mirrors `Qwen4ExpTextConfig.validate_architecture`'s QSA clause
// (configuration_qwen4_exp.py :210-231). Throws on violation.
void QsaValidateConfig(const QsaConfig& cfg);

// ── The side cache ───────────────────────────────────────────────────────────

// `MLAAttentionSpec(num_kv_heads=1, head_size=index_head_dim,
//                   compress_ratio=compress_ratio)`.
//
// `MLAAttentionSpec` is NOT an MLA claim. MiniMax-M3 is plain GQA and uses it
// for its own indexer cache, with the upstream comment "Key-only:
// MLAAttentionSpec budgets one vector/token (not 2x for K+V)". It is a BUDGET
// shape.
//
// CORRECTED IN FLOW AT W5c, issue
// [#2198](https://github.com/mudler/vllm.cpp/issues/2198). The two comments
// this replaces named a field called `tokens_per_state` and anchored it at
// `v1/attention/backends/mla/indexer.py:624-628`. `grep -rn tokens_per_state`
// over the pinned vLLM tree (`5559679229`) returns ZERO hits, tree-wide, and
// so does a search for the docstring they quoted; the cited anchor is
// `_prepare_decode_tensors` and is unrelated. The real field is
// **`compress_ratio`** (`vllm/v1/kv_cache_interface.py:386`, defaulted to 1),
// and the page runs off `storage_block_size = block_size // compress_ratio`
// (`:393-395`). `MLAAttentionSpec.merge` (`:424-435`) asserts ONE
// `compress_ratio` per KV group, which is why W5c publishes a single spec for
// all twelve QSA layers. This tree was already correct where it counts —
// `include/vllm/v1/kv_cache_interface.h` spells it `compress_ratio` — so the
// defect was a citation that would have sent the wave writing the KV spec
// looking for a field that does not exist.
struct QsaSideCacheSpec {
  int64_t num_kv_heads = 1;
  int64_t head_size = 128;
  // A LOCAL name with no upstream referent, deliberately left alone by #2198:
  // its arithmetic is right (64 B/token/layer, pinned by
  // `tests/vllm/models/test_qwen4_exp_qsa.cpp`) and it matches
  // `MLAAttentionSpec::real_page_size_bytes()` exactly, so renaming it would
  // churn this TU and its suite to fix a citation the comments above now carry.
  // The spec that reaches the runner spells it `compress_ratio`.
  int64_t tokens_per_state = 4;
  int64_t elem_bytes = 2;  // bf16

  // Only a COMPLETE block produces a state: the compressor early-exits unless
  // `(position + 1) % compress_ratio == 0` (compressor_utils.py :52,
  // fused_compress_quant_cache.py :729-731). So this floors, and the ragged tail
  // costs no state at all — it is attended from the raw KV cache instead.
  int64_t StatesForTokens(int64_t num_tokens) const;
  // elem_bytes * num_kv_heads * head_size / tokens_per_state. 64 B at the
  // published shape, a quarter of what a per-token index cache would cost.
  int64_t BytesPerTokenPerLayer() const;
  int64_t BytesForTokens(int64_t num_tokens) const;
};

QsaSideCacheSpec QsaMakeSideCacheSpec(const QsaConfig& cfg, int64_t elem_bytes);

// `get_compressed_slot_mapping`'s per-token body (compressor_utils.py :49-61):
// the slot a token's compressed state is written to, or -1 (PAD_ID) when the
// token is not a block boundary and therefore writes nothing.
int64_t QsaCompressedSlot(int64_t position, int64_t compress_ratio);

// ── Primitives ───────────────────────────────────────────────────────────────

// `Qwen4ExpTextRMSNorm.forward` (modeling_qwen4_exp.py :167-179). Note the
// weight polarity: upstream is `out * (1.0 + weight)` with a ZERO-initialised
// weight, where vLLM's RMSNorm is `out * weight` with a ones-initialised one.
// `round_to_bf16` mirrors the closing `.type_as(x)` on a bf16 model path.
std::vector<float> QsaRmsNorm(const std::vector<float>& x, int64_t rows,
                              int64_t dim, const std::vector<float>& weight,
                              float eps, bool round_to_bf16);

// `apply_rotary_pos_emb` (modeling_qwen4_exp.py :573-604) over the LEADING
// `rotary_dim` dims, NeoX `rotate_half` pairing (:566-571), NoPE dims trailing
// and untouched. `cos`/`sin` are [rows, rotary_dim] and already carry the
// interleaved-mRoPE section layout; this function does not build them.
// `round_to_bf16` mirrors bf16 elementwise arithmetic, which rounds per op.
std::vector<float> QsaApplyRotaryLeadingHalf(const std::vector<float>& x,
                                             int64_t rows, int64_t head_dim,
                                             const std::vector<float>& cos,
                                             const std::vector<float>& sin,
                                             int64_t rotary_dim,
                                             bool round_to_bf16);

// The pooled-key build, i.e. the side cache's contents.
//
// `raw_keys` is [num_keys, index_head_dim] and must be RAW: un-normed and
// un-roped, exactly as `Cache.update_indexer` stores it (modeling_qwen4_exp.py
// :657-658). Per complete block, mean-pool `compress_ratio` consecutive keys in
// float, round back to the cache dtype, apply `k_layernorm`, then apply RoPE at
// the position of the block's FIRST token (:677-688). Returns
// [num_keys / compress_ratio, index_head_dim].
//
// `cos`/`sin` are the FULL-position tables, [num_positions, rotary_dim]; the
// function selects row `compress_ratio * b` for block b, which is the same
// value the Triton kernel computes as
// `compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO`
// (fused_compress_quant_cache.py :816).
//
// RECONCILIATION. Upstream forms blocks over the VISIBLE token indices of a
// padded batch (`local_visible_indices`, :668-679) and takes
// `group_starts = block_token_indices[:, 0]`. A serving engine's ragged batch
// has no interior masking — visible is the contiguous range [0, kv_len) — so
// the two coincide and block b is exactly tokens [CR*b, CR*b + CR). This
// function assumes that contiguity and asserts `num_keys % compress_ratio == 0`
// on the caller's slice; it does not accept an arbitrary visibility set.
std::vector<float> QsaCompressNormRope(const std::vector<float>& raw_keys,
                                       int64_t num_keys,
                                       const std::vector<float>& k_norm_weight,
                                       const std::vector<float>& cos,
                                       const std::vector<float>& sin,
                                       const QsaConfig& cfg,
                                       bool round_to_bf16);

// One score per COMPLETE block for one query token (modeling_qwen4_exp.py
// :690-693):
//     score[b] = sum_h ReLU(dot(q[h,:], block_key[b,:])) / sqrt(index_head_dim)
// `q` is [index_n_heads, index_head_dim], already q_layernorm'd and roped.
// There is no per-head weight and no head_scale — see the header note on what
// is not inherited from DeepSeek-V4. The division is applied AFTER the sum, as
// upstream writes it.
std::vector<float> QsaBlockScores(const std::vector<float>& q,
                                  const std::vector<float>& block_keys,
                                  int64_t num_blocks, const QsaConfig& cfg);

// `scores.topk(min(block_topk, num_complete_blocks)).indices`
// (modeling_qwen4_exp.py :695), with two op-side semantics taken from vLLM's
// `top_k_per_row` rather than invented here:
//   * when num_blocks <= k every candidate is selected and the result is
//     ASCENDING block order (sampler.cu :391-402, "Indices are not sorted by
//     their corresponding logit");
//   * ties resolve to the LOWER block index (sampler.cu :515,
//     `logit == otherLogit && i < j`).
// Both are inherited so a QSA top-k resolves the way the op it mirrors resolves
// one. NEITHER is load-bearing here and neither is exercised: the fixtures
// produce no exact tie, and `QsaSelectedTokenIndices` re-sorts, so reversing
// either leaves the suite green. Do not read them as gated behaviour.
// Returns the selected block ids in DESCENDING score order, which is the order
// `torch.topk` produces; `QsaSelectedTokenIndices` re-sorts them.
std::vector<int64_t> QsaTopkBlocks(const std::vector<float>& scores,
                                   int64_t num_blocks, int64_t k);

// The GATHER index buffer for one query token: selected block ids expanded to
// tokens [CR*b, CR*b + CR), plus the ALWAYS-attended ragged tail
// [CR*num_complete_blocks, kv_len), sorted ASCENDING and padded to
// `cfg.index_width()` with -1.
//
// WHY ASCENDING, when upstream's buffer is in score-rank order. Upstream
// scatters the buffer into a boolean mask (:705-712), so its order is
// unobservable. A GATHER's order IS observable: it fixes the reduction order of
// the softmax. Ascending is the order that makes a sub-budget gather reduce over
// exactly the dense sequence, which is what makes the bit-identity oracle below
// hold, and it is the order vLLM's own all-select shortcut emits
// (sampler.cu :391-402).
//
// THE FREE ORACLE. When kv_len <= token_budget + compress_ratio - 1 every
// candidate is selected, so this returns exactly [0, kv_len) followed by -1
// padding, and `QsaGatherAttention` over it is bit-identical to dense attention.
// llama.cpp #27742 measures a max logit delta of 0.0 over all 2051 such rows.
// That gates the whole selection and masking path with no checkpoint. The dense
// side of that `==` is `QsaMaskedAttention` over the full causal prefix — a
// separate walk over every cached row. Comparing the gather to a second call to
// ITSELF, as the first revision of the test did, only restates that the function
// is deterministic: scaling its output by 2 left that case green.
std::vector<int32_t> QsaSelectedTokenIndices(
    const std::vector<int64_t>& selected_blocks, int64_t num_complete_blocks,
    int64_t kv_len, const QsaConfig& cfg);

// ── Consumers ────────────────────────────────────────────────────────────────

// THE GATHER CONSUMER — the point of the wave.
//
// Dense GQA over ONLY the gathered rows. `q` is [num_q_heads, head_dim] for one
// query token; `k` and `v` are [kv_len, num_kv_heads, head_dim]; `indices` is
// one row of `QsaSelectedTokenIndices`, terminated by -1. Returns
// [num_q_heads, head_dim].
//
// `keys_visited`, when non-null, receives the number of key-row reads this call
// PERFORMED — incremented at the `Dot`, never assigned from `indices`. An honest
// gather reads each selected row once per query head in each of the two softmax
// passes, so the value is `selected * num_q_heads * 2`; the mask reference
// reports `kv_len * num_q_heads * 2` in the same unit. It is the observable that
// separates the two consumers: they agree on every output value and disagree
// here by a factor of kv_len / selected_count. A gate that checks only the
// values measures correctness and NOT the speed lever, which is precisely how a
// mask-only implementation passes a token gate while forfeiting the lever.
//
// WHAT IT DOES NOT SEE, stated so nobody has to rediscover it. It counts READS,
// so a body that iterates 0..kv_len and `continue`s past the unselected rows
// without touching them still reports the sparse figure — correctly, because the
// loop counter is not the cost llama.cpp #27739 measures; the key-row traffic is.
// And it can only count reads that go through this function's own read site.
std::vector<float> QsaGatherAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      const std::vector<int32_t>& indices,
                                      int64_t kv_len, int64_t num_q_heads,
                                      int64_t num_kv_heads, int64_t head_dim,
                                      int64_t* keys_visited);

// THE MASK REFERENCE — red-first only, never the production path.
//
// Dense GQA over the WHOLE cache with -inf at every unselected position, i.e.
// what upstream's scatter mask produces (modeling_qwen4_exp.py :705-717). Kept
// so the gather has something to be gated against and so the cost difference is
// measurable, and named `Reference` in its own doc rather than in a comment
// somewhere else. `keys_visited` is COUNTED at the read like the gather's, and
// comes out at `kv_len * num_q_heads * 2` because this walk reads every cached
// row whatever the mask says.
//
// Over the full causal prefix — an index buffer of exactly [0, kv_len) — this is
// plain dense attention with no position removed, which is what makes it the
// independent side of the sub-budget bit-identity oracle.
std::vector<float> QsaMaskedAttention(const std::vector<float>& q,
                                      const std::vector<float>& k,
                                      const std::vector<float>& v,
                                      const std::vector<int32_t>& indices,
                                      int64_t kv_len, int64_t num_q_heads,
                                      int64_t num_kv_heads, int64_t head_dim,
                                      int64_t* keys_visited);

}  // namespace vllm::qwen4_exp
