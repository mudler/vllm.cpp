// DFlash2 draft model — the CANDIDATE SELECTOR half (SPEC-DFLASH2 W3, #1314).
//
// BEYOND-PIN. Ported from vllm/model_executor/models/qwen3_dflash2.py @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`:
// `_topk` (:60-64), `_score_edges` (:208-228), `CandidateSelector` (:231-276),
// and `DFlash2Qwen3ForCausalLM.compute_candidates` (:326-356). The parity pin
// `555967922` does not carry the architecture at all, and nothing here advances
// it.
//
// THE HEAD MOVED under this row (#1404). W2 cited
// `19c9351904df4c63042671bc67a866ca48dc7d6f`; this file cites the new head and
// every anchor above was re-read there. Diffing the two heads for this file
// returns exactly two changes, both in the enclosing model rather than in the
// selector's math:
//
//   * `set_model_tag("dflash2_candidate_selector")` around the selector's
//     construction. A DELIBERATE NON-PORT — see
//     `Dflash2SelectorWeights::kNonPortSetModelTag` (qwen3_dflash.h) for why an
//     engine with no torch.compile cache has nothing for it to disambiguate.
//   * the LM-head guard widened to accept `UnquantizedLinearMethod` beside
//     `UnquantizedEmbeddingMethod`, which is the folded-in vllm#52883 fix.
//     PORTED — see `RefuseQuantizedDflash2LmHead`.
//
// The selector's own math (`_score_edges`, the two codebooks, the projection,
// the top-k, the org-vocab rebase, `output_multiplier`, `final_logit_softcapping`)
// is BYTE-IDENTICAL at the two heads.
//
// WHAT THE UPSTREAM SPLIT LOOKS LIKE HERE. Upstream's `compute_candidates`
// applies the TARGET's `lm_head` to the draft's hidden states itself, because
// its `DFlashQwen3ForCausalLM.forward` returns hidden states. This engine's
// draft forward already applies `weights.lm_head` and returns
// `[T, draft_vocab]` f32 logits (`Qwen3DFlashModel::ForwardBlockLogits*`), and
// that lm_head IS the target's — the loader shares it, exactly as upstream's
// loader does. So `ComputeCandidates` below starts one step later, at the
// top-k, and the head application is not duplicated.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm {

// One row's surviving candidates, as `compute_candidates` returns them: `ids` in
// the TARGET vocabulary (after the org-vocab rebase) and `values` after
// `output_multiplier` and `final_logit_softcapping`. Both are `[rows, top_k]`
// row-major, ordered by DESCENDING value with ties broken by ASCENDING id — the
// order `vt::TopKValuesIndices` pins.
struct Dflash2CandidateSet {
  std::vector<int64_t> ids;
  std::vector<float> values;
  int64_t rows = 0;
  int64_t top_k = 0;
};

// The two VOCAB-PARALLEL properties upstream reads off
// `lm_head.shard_indices`. Both are ZERO on every path this engine ships: the
// DFlash lane's `lm_head` is the raw unpadded checkpoint tensor and there is no
// vocab-parallel sharding, so upstream's `shard_indices` degenerate. They are
// implemented and gated SYNTHETICALLY rather than claimed as checkpoint
// coverage — the posture `## Upstream chain` of the spec already records for the
// output scalars — because the arithmetic has to be right the day a sharded head
// arrives, and a candidate id that is not rebased indexes the wrong codebook row
// and moves acceptance without raising anything.
struct Dflash2CandidateArgs {
  // `lm_head.shard_indices.num_org_vocab_padding`: that many columns at the END
  // of each logits row are forced to -inf BEFORE the top-k, so a padded head can
  // never contribute a candidate.
  int64_t num_org_vocab_padding = 0;
  // `lm_head.shard_indices.org_vocab_start_index`: added to every surviving id
  // AFTER the top-k, rebasing this shard's column space into the full vocabulary.
  int64_t org_vocab_start_index = 0;
};

// The DFlash2 candidate selector, as a static surface on the draft model (the
// weights it reads live on `Qwen3DFlashWeights::candidate_selector`).
class Qwen3DFlash2Model {
 public:
  // `compute_candidates` from the top-k onward
  // (`DFlash2Qwen3ForCausalLM.compute_candidates` @ the PR head).
  //
  // `logits` is `[rows, vocab]` f32 — the SAMPLE rows of the draft's block
  // forward, i.e. the mask positions and not the anchor. Returns the top
  // `selector_top_k` (id, value) pairs of each row with the org-vocab rebase,
  // `output_multiplier` and `final_logit_softcapping` applied to the VALUES in
  // that order, which is upstream's order and is why a wrong scalar reorders the
  // top-K one step later rather than raising.
  static Dflash2CandidateSet ComputeCandidates(const std::vector<float>& logits, int64_t rows,
                                               int64_t vocab,
                                               const Qwen3DFlashWeights& weights,
                                               vt::Queue& queue,
                                               const Dflash2CandidateArgs& args = {});

  // `CandidateSelector.forward` (@ the PR head): the hidden projection followed
  // by `_score_edges`. `hidden` is `[rows, H]` f32 — the post-final-norm hidden
  // of the same sample rows `ComputeCandidates` read, in the same order.
  // `anchors` is one verified anchor token per request. Returns the edge lattice
  // `[num_reqs, num_steps, top_k, top_k]` f32, row-major, which the W4 path walk
  // consumes.
  static std::vector<float> SelectorEdgeScores(const Dflash2CandidateSet& candidates,
                                               const std::vector<float>& hidden,
                                               const std::vector<int32_t>& anchors,
                                               int64_t num_reqs, int64_t num_steps,
                                               const Qwen3DFlashWeights& weights,
                                               const HfConfig& config, vt::Queue& queue);
};

// The LM-head guard, ported from `compute_candidates`'s own first statement @
// the PR head, where it reads
//
//     if not isinstance(self.lm_head.quant_method,
//                       (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)):
//         raise ValueError("DFlash2 requires an unquantized target LM head ...")
//
// The second class in that tuple is the FOLDED-IN vllm#52883 fix and the reason
// it exists is not cosmetic: a `ParallelLMHead` returns `UnquantizedLinearMethod`
// rather than `UnquantizedEmbeddingMethod` whenever a quant config leaves the
// head itself unquantized (INC, ModelOpt, fp8 with excluded layers), so the
// NARROW guard refused perfectly valid unquantized heads. Ours mirrors the WIDE
// form: what is refused is a head whose weights are not readable as dense
// floats, and an unquantized head of either provenance is admitted.
//
// Why it is a refusal and not a dequantize-on-read: the selector's candidate set
// IS the target's top-K, so a head that cannot produce exact logits produces a
// different candidate set, and the whole defect is invisible — the verify is
// lossless, the emitted tokens stay the target's, and only acceptance falls.
void RefuseQuantizedDflash2LmHead(const Qwen3DFlashWeights& weights);

}  // namespace vllm
