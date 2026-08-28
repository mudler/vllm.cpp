// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) W5b-5 — `Qwen4ExpTextAttention` as ONE
// production block, and the first place under `src/` that COMPOSES the Qwen
// Sparse Attention indexer.
//
// Issue [#2211](https://github.com/mudler/vllm.cpp/issues/2211), wave issue
// [#2031](https://github.com/mudler/vllm.cpp/issues/2031), campaign issue
// [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHY THIS FILE EXISTS, IN THE SPEC'S OWN WORDS ───────────────────────────
// `## Owed`: "**W5b OWES THE INDEXER COMPOSITION IN PRODUCTION CODE, AND FOUR
// SETTINGS WITH IT.** … That composition exists in exactly one place: the
// `RunIndexer` helper in `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`.
// Nothing under `src/` composes it, so nothing outside that helper enforces any
// of the four settings the collapse depends on".
//
// The four are stated once, here, and asserted once, in `Qwen4ExpQsaIndex`:
//
//   1. `weights` is all ones `[T, index_n_heads]`. This is what collapses
//      `vt::DsaIndexerLogits`'s per-head fold to a single constant, which is
//      what makes that op QSA's block score at all. QSA has no `weights_proj`.
//   2. `n_head_scale == 1.0f`, NOT DeepSeek-V4's `n_head ** -0.5`. QSA has no
//      tensor for it and upstream's scoring line has no such factor.
//   3. `softmax_scale == index_head_dim ** -0.5`, QSA's own scale
//      (`modeling_qwen4_exp.py:693`, `/ math.sqrt(self.index_head_dim)`).
//   4. `win_end == kv_len / compress_ratio` per query token — the COMPLETE
//      VISIBLE blocks, not the whole cache.
//
// **TWO OF THE FOUR ARE INVISIBLE TO ANY SELECTION-BASED GATE, BY CONSTRUCTION,
// AND THAT IS WHY THIS HEADER NAMES THEM.** Top-k is invariant under a positive
// rescale of every score, so a wrong `n_head_scale` or `softmax_scale` cannot
// move a selected set — spec mutation M26 measures exactly that survival, and
// `DsaIndexerLogitsArgs` says the same thing in its own comment. `Qwen4ExpQsaIndex`
// therefore takes an OPTIONAL `logits` out-parameter and the gate compares it BY
// VALUE against the oracle's own pre-top-k score tensor. A selection gate here
// would be an instrument nobody wired up.
//
// ─── THE OTHER TRAP THIS BLOCK CARRIES ───────────────────────────────────────
// A mask-shaped consumer passes a token gate and forfeits the whole point of the
// row: `exp(-inf - m)` is exactly +0, so a sparse mask over a dense cache agrees
// with a gather VALUE FOR VALUE, and under CUDA flash attention it costs the
// full dense prefix (llama.cpp #27739). This block routes the consumer through
// `vt::Qwen4ExpQsaGatherAttention` and forwards that op's `keys_visited`
// instrument, so the block's own gate can run the NaN-poison and unmapped-tail
// probes W5b-4 established rather than trusting the shape of the call.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// vLLM registers `qwen4_exp` at NO revision, so under AGENTS.md "When vLLM has
// no implementation" the ALGORITHM oracle is transformers **5.16.0**, this row's
// accepted lane pin, at `models/qwen4_exp/modeling_qwen4_exp.py`:
// `Qwen4ExpTextAttention.forward` (:785-841) and `Qwen4ExpTextQSAIndexer.forward`
// (:631-716). Every op this block calls is a vLLM-mirrored primitive; the block
// is the composition, and the composition is what upstream has and vLLM does not.
//
// ─── SCOPE, AND WHAT IS NOT HERE ─────────────────────────────────────────────
// This block is REACHED ONLY BY ITS TEST at its merge commit, and the spec's
// `## Owed` records that with the row and the issue that own the wiring:
// `Qwen4ExpTextModel::Forward` does not exist yet, so `ModelRegistry::Forward`
// still refuses `Qwen4ExpForConditionalGeneration` by name. Also not here: the
// PAGED cache (this block takes contiguous per-sequence K/V, which is the shape
// the two `vt::` ops already accept and all the KV-cache group can express
// today), the cos/sin table build (taken as an operand, so the interleaved-mRoPE
// section layout stays owed by the wave that builds it), and the CUDA arm.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_QSA_BLOCK_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_QSA_BLOCK_H_

#include <cstdint>
#include <memory>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev
#include "vllm/model_executor/models/qwen4_exp.h"          // Qwen4ExpParams
#include "vllm/model_executor/models/qwen4_exp_weights.h"  // Qwen4ExpQsaWeights
#include "vt/tensor.h"

namespace vllm {

// The per-sequence caches one QSA layer reads and writes. CONTIGUOUS, not paged,
// and that is a scope statement rather than a design preference: both `vt::` ops
// this block drives address their caches as flat `[rows, …]` arrays and never
// read `stride[0]` (see `Qwen4ExpQsaGatherAttention`'s own contract), and the KV
// group that would make them paged is blocked behind the runner work in
// [#2131](https://github.com/mudler/vllm.cpp/issues/2131). The paged store is
// listed under `## Owed`.
//
// `key`/`value` hold the RAW model K/V — what upstream's `past_key_values.update`
// returns — and `index_key` holds the RAW, UN-normed and UN-roped indexer keys,
// which is what `Cache.update_indexer` stores and what
// `vt::Qwen4ExpQsaCompress` expects. Storing them normed or roped would be a
// double application, because the compressor does both.
struct Qwen4ExpQsaCaches {
  vt::Tensor key;        // [max_kv, num_kv_heads, head_dim]     READ-WRITE
  vt::Tensor value;      // [max_kv, num_kv_heads, head_dim]     READ-WRITE
  vt::Tensor index_key;  // [max_kv, indexer_head_dim]           READ-WRITE
};

// Owning device-resident output of one QSA block: a [T, hidden_size] view plus
// the shared_ptr that returns its pool block to the DevicePool when the last
// reference drops. Mirrors `GdnBlockOutput` / `MoeBlockOutput` exactly, so the
// composing forward keeps one release idiom across all three block kinds.
struct Qwen4ExpQsaBlockOutput {
  vt::Tensor tensor;              // [T, H] at the block's dtype
  std::shared_ptr<void> storage;  // owns the pool block (Pool().Put on release)
};

// The composed indexer's result for one step: the selected BLOCK ids and how
// many of them each query token got.
//
// The ids are what `vt::DsaTopkSelect` emits — ASCENDING, `-1`-terminated,
// all-select below `block_topk` — which is exactly what
// `vt::Qwen4ExpQsaGatherAttention` consumes. Nothing between the two reorders
// them, and nothing may: the ascending order is what makes a sub-budget
// selection reduce over the same positions in the same order dense attention
// would, and therefore bit-identical to it.
struct Qwen4ExpQsaSelection {
  vt::Tensor block_ids;  // [T, block_topk] i32, ascending, -1 = no block
  vt::Tensor counts;     // [T] i32
  std::shared_ptr<void> storage;  // owns both pool blocks
};

// `Qwen4ExpTextQSAIndexer.forward` (:631-716) as a COMPOSITION of three ops —
// `vt::Qwen4ExpQsaCompress`, `vt::DsaIndexerLogits`, `vt::DsaTopkSelect` — with
// the four settings this header names applied where no test helper is watching.
//
//   q_index    [T, index_n_heads, index_head_dim]  the indexer query, ALREADY
//              q-layernormed and roped by the caller (the block below does it)
//   caches.index_key rows [0, kv_len)  the raw indexer keys, UN-normed/UN-roped
//   k_norm_w   [index_head_dim]  the RAW HuggingFace gamma; the compressor
//              applies `(1.0 + w)` itself, mirroring `Qwen4ExpTextRMSNorm`
//   cos/sin    [>= kv_len, rotary_dim] f32 FULL-position tables; the compressor
//              reads row `compress_ratio * b`, the block's FIRST token
//   kv_lens    [T] i32, the causal visible length of each query token
//   logits     OPTIONAL [T, kv_len / compress_ratio] f32 OUT — the VALUE-gate
//              surface. Two of the four settings cannot be seen any other way;
//              see the header comment. `nullptr` on the production path.
//
// The block-key scratch is allocated per call and dropped; a wave that gives QSA
// a real KV-cache group turns it into the side cache's paged store.
Qwen4ExpQsaSelection Qwen4ExpQsaIndex(dense_attn::Dev d, const Qwen4ExpQsaParams& qsa,
                                      float rms_norm_eps, const vt::Tensor& q_index,
                                      const Qwen4ExpQsaCaches& caches, const vt::Tensor& k_norm_w,
                                      const vt::Tensor& cos, const vt::Tensor& sin,
                                      const vt::Tensor& kv_lens, int64_t kv_len,
                                      bool round_intermediates_to_bf16,
                                      vt::Tensor* logits = nullptr);

// One `Qwen4ExpTextAttention` block, end to end.
//
//   hidden     [T, hidden_size]  the block input — on this architecture that is
//              the gated residual's COLLAPSED 2560-wide output, never the
//              10240-wide stream, which only the hyper-connection ops see
//   positions  [T] i32, the position of each new token; used for the q/k RoPE
//   cos_sin    [P, rotary_dim] the packed cos|sin cache `vt::RopeFromCache`
//              reads: columns [0, rot/2) are cos and [rot/2, rot) are sin
//   cos/sin    [P, rotary_dim] the SEPARATE full tables the compressor reads.
//              Two layouts for one set of angles, because the two ops were
//              ported from two upstreams that spell it differently. The caller
//              must build both from ONE table, and the block CROSS-CHECKS that
//              rather than trusting it: equal heights, then a BOUNDED SAMPLE of
//              rows compared value for value. The sample is what a table-wide
//              construction difference shows in and is not a per-row guarantee;
//              `CheckRopeLayoutsAgree` in the .cpp states exactly what it can
//              and cannot see, and why a full comparison is not paid per call.
//   past_len   how many tokens the caches already hold; the new tokens land at
//              rows [past_len, past_len + T)
//   keys_visited  OPTIONAL, forwarded to `vt::Qwen4ExpQsaGatherAttention`. It is
//              counted AT THE KEY-ROW READ and is the discriminator between a
//              gather and a mask; see that op's contract for what it cannot see
//              and which two probes close the gap.
Qwen4ExpQsaBlockOutput RunQwen4ExpQsaBlock(dense_attn::Dev d, const Qwen4ExpQsaWeights& w,
                                           const Qwen4ExpParams& params, const vt::Tensor& hidden,
                                           const vt::Tensor& positions, const vt::Tensor& cos_sin,
                                           const vt::Tensor& cos, const vt::Tensor& sin,
                                           const Qwen4ExpQsaCaches& caches, int64_t past_len,
                                           int64_t* keys_visited = nullptr);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_QSA_BLOCK_H_
