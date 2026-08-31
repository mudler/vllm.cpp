#pragma once

// DSV4-DSPARK-DRAFTER — the DSpark block drafter's ENTRY, as pure host code.
//
// Spec: `.agents/specs/dsv4-dspark-drafter.md`. Oracle: `exllamav3` @ the
// registered pin `2398c05635fbbad01a0a51dce63c85c6c8a8450e`
// (`.agents/oracles/exllamav3.md`); vLLM implements no DSpark, and its
// `nvidia/mtp.py` head is a DIFFERENT architecture.
//
// This is the lever the 44-47 tok/s target actually uses: that figure is measured
// WITH DSpark K5 speculative decoding at acceptance 0.65/0.44/0.31/0.17/0.07,
// about 2.64 accepted tokens per step.

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4.h"

namespace vllm::dspark {

// THE TAP. `exllamav3/modules/transformer.py:198-203`:
//
//     x_ = x.mean(dim = 2) if self.attn_hc else x
//
// With hyperconnections the trunk residual is a STREAM STACK
// `[num_tokens, hc_mult, hidden]`, and the tap the drafter consumes is its mean
// over the STREAM dimension -- "streams start as broadcast copies of the
// embedding", so the mean is the collapsed hidden state. Taken AFTER the layer's
// residual add, and only on `layer_instance == 0` (`transformer.py:144`).
//
// Pinned in code rather than left to the caller because every neighbouring choice
// -- the raw stack, the pre-residual state, the final-norm output -- yields a
// drafter that runs and emits CORRECT tokens (verification is lossless) while
// drafting badly. There is no failing test for a wrong tap, only an acceptance
// rate nobody can account for.
//
//   x    [num_tokens, hc_mult, hidden], stream-major within a token
//   out  [num_tokens, hidden]
std::vector<float> StreamMeanTap(const std::vector<float>& x, int64_t num_tokens,
                                 int64_t hc_mult, int64_t hidden);

// THE ENTRY. `DSparkInputLayer.project_taps`
// (`exllamav3/modules/arch_specific/dspark.py:194-197`): concatenate the taps on
// the last axis, project once through `main_proj`, then `main_norm`.
//
// `main_proj` is `Linear(n_taps * hidden, hidden)` and
// `dspark_target_layer_ids = [40, 41, 42]` gives `n_taps = 3`, which is the
// artifact's measured `[4096, 12288]`. The projection is applied ONCE per set of
// taps, not once per block: all three blocks then read the same `dspark_main_x`.
//
//   taps  n_taps buffers, each [num_tokens, hidden], already stream-meaned
//   out   [num_tokens, hidden]
std::vector<float> ProjectTaps(const std::vector<std::vector<float>>& taps,
                               const DeepseekV4MtpTensorView& main_proj,
                               const std::vector<float>& main_norm_weight, float eps,
                               int64_t num_tokens, int64_t hidden);

// W-1, THE TAP SEAM. What the trunk fills when a caller wants taps.
//
// `layer_ids` are the trunk layers to tap, upstream's `dspark_target_layer_ids`
// ([40, 41, 42] in the artifact). `taps` comes back with ONE `[num_tokens,
// hidden]` buffer per requested layer, IN THE ORDER REQUESTED -- that order is
// load-bearing, because `main_proj`'s input columns are the taps concatenated in
// exactly this order and a permutation is silent.
//
// A tap is taken from the layer's POST-block manifold state, the fold of that
// layer's output back through `MhcPost`, which is the same state upstream exports
// after its residual add. It is then collapsed by `StreamMeanTap`.
struct TapRequest {
  std::vector<int64_t> layer_ids;
  std::vector<std::vector<float>> taps;  // filled by the forward
};

// W-4, THE DRAFT LOOP. `DeepseekV4MTPModel.sample_from_state`
// (`exllamav3/architecture/deepseek_v4_mtp.py:311-340`).
//
// The trunk head runs over ALL block positions at once; only this loop is
// sequential, and all it does per step is one embedding gather, one rank-256
// GEMV and an argmax. That is what makes a 5-token block affordable: the
// expensive part is parallel and the serial part is a bigram correction.
//
//     out[0] = seed
//     for i in [0, block):
//       emb    = markov_w1[out[i]]        // Embedding, [rank]
//       bias   = markov_w2 @ emb          // Linear rank -> vocab, [vocab]
//       out[i+1] = argmax(logits[i] + bias)
//
// `markov_w1` is an Embedding of width `dspark_markov_rank` and `markov_w2` a
// Linear whose `in_features` is that same rank
// (`deepseek_v4_mtp.py:176-190`); both are `[vocab, rank]` in the artifact, at
// rank 256.
//
//   logits  [block, vocab] from the SHARED trunk head
//   returns block + 1 ids, `[seed, drafts...]` — the caller crops the seed
std::vector<int32_t> MarkovDraftLoop(const std::vector<float>& logits, int32_t seed_id,
                                     const std::vector<float>& markov_w1,
                                     const std::vector<float>& markov_w2, int64_t block,
                                     int64_t vocab, int64_t rank);

// W-3, ONE BLOCK'S KV ROWS. `DSparkAttention.update_kv_rows`
// (`exllamav3/modules/arch_specific/dspark.py:134-155`):
//
//     kv = wkv(main_x).view(bsz, s, 1, D)
//     ext.rope(kv, kv, ..., positions, kv_norm_w, eps, ..., rotate_dims=1,
//              rotate_offset=D - rd)
//     kl.write_rows(kv, positions, block_table)
//
// The drafter's KV does NOT come from its own forward: every block derives its
// rows from the SAME projected tap state, at the target's own positions, so the
// draft cache stays aligned with the target's block tables.
//
// ORDER AND WIDTH, both read out of the kernel rather than inferred. In
// `exllamav3_ext/rope.cu` the body is `load_head(); apply_norm(); apply_rope();`
// (:248-251), the norm weight is required to be `head_dim` wide (:379) and the RMS
// divides by `head_dim` (:207). So the norm covers the WHOLE head_dim and the
// rotation then applies to `[head_dim - rope_dim, head_dim)`.
//
// That is the TRUNK's convention and NOT `vt::kFusedNormRope`'s, which implements
// DeepSeek-V2/V3 MLA where the norm covers only the nope half and the decoupled
// rope part stays unnormed. The two are easy to confuse and the artifact settles
// it: `mtp.0.attn.kv_norm.weight` is `[512]`, the full head_dim, where the V2
// convention would make it `[448]`.
//
//   main_x    [num_tokens, hidden]  the shared, already-projected tap state
//   wkv       [head_dim, hidden]    this block's own KV projection
//   returns   [num_tokens, head_dim]
std::vector<float> BlockKvRows(const std::vector<float>& main_x,
                               const std::vector<float>& wkv,
                               const std::vector<float>& kv_norm_w, float eps,
                               const std::vector<int32_t>& positions, double rope_theta,
                               int64_t num_tokens, int64_t hidden, int64_t head_dim,
                               int64_t rope_dim);

}  // namespace vllm::dspark

namespace vllm {

// W-1's public entry: run the trunk and collect its taps. One `[num_tokens,
// hidden]` stream mean per requested layer, in REQUEST order.
std::vector<std::vector<float>> DeepseekV4TrunkTapsHost(
    const DeepseekV4HostWeights& hw, const DeepseekV4Params& p,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<int64_t>& layer_ids);

}  // namespace vllm
