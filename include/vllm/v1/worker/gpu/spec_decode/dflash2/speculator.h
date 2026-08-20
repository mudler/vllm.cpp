// DFlash2 speculator (SPEC-DFLASH2 W3, #1314) — the candidate-selection half.
//
// BEYOND-PIN. Ported from `DFlash2Speculator._generate_draft`
// (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:191-227 @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`).
// Upstream's `_generate_draft` is five steps:
//
//   1. run the draft model                         (W2, landed)
//   2. gather the SAMPLE rows of its hidden states (here)
//   3. compute_candidates -> (ids, values)         (here, qwen3_dflash2.h)
//   4. candidate_selector  -> the edge lattice     (here, qwen3_dflash2.h)
//   5. _sample_path + _cache_draft_logits          (W4)
//
// This file owns 2 and the sequencing of 3 and 4, and REFUSES 5 by name. W2's
// `RefuseDflash2CandidateSelector` is retired by this wave: the selector is no
// longer the missing mechanism, the WALK is.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vllm/model_executor/models/qwen3_dflash2.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

namespace vllm::v1 {

// What one block propose has after the selector has run and before the walk.
struct Dflash2ProposeState {
  Dflash2CandidateSet candidates;   // [num_reqs*num_steps, top_k]
  std::vector<float> edge_scores;   // [num_reqs, num_steps, top_k, top_k] f32
  int64_t num_reqs = 0;
  int64_t num_steps = 0;
  int64_t top_k = 0;
};

// Steps 2-4 above, over ONE block propose. This is the function BOTH propose
// paths call — `GPUModelRunner::propose_drafts_block` (the production decode
// path) and `DflashProposeBlock` — so the sequence a user arrives through and
// the sequence a gate can drive are the SAME code rather than two copies. W2
// carried two copies of its refusal and only one of them was gated; that is what
// spec `## Owed` O7 records, and collapsing the duplicate is how this wave
// avoids repeating it.
//
// `block_logits` is `[num_reqs*(1+k), draft_vocab]` f32 and `block_hidden` is
// `[num_reqs*(1+k), H]` f32 (the block forward's `final_out`), both in
// per-request block-row order: request r owns rows [r*(1+k) .. r*(1+k)+k], row
// +0 being the ANCHOR. The SAMPLE rows are +1..+k — the mask positions — which
// is upstream's `sample_indices` gather, and the anchor row is deliberately not
// among them: it carries a verified token and predicts nothing.
//
// `anchors` is each proposing row's verified anchor token in the target
// vocabulary; the selector seeds step 0's predecessor with it.
Dflash2ProposeState Dflash2SelectCandidates(const std::vector<float>& block_logits,
                                            const std::vector<float>& block_hidden,
                                            const std::vector<int32_t>& anchors,
                                            int num_reqs, int k,
                                            const Qwen3DFlashWeights& weights,
                                            const HfConfig& config, vt::Queue& queue);

// The W4 boundary, refused BY NAME. A DFlash2 draft reaching here has run its
// grouped convolution (W2), its target-head top-k, its codebook lattice and its
// edge scores (W3); what is missing is the PATH WALK that turns those scores
// into k draft tokens, its inverse-CDF arm at T>0 and the realized-q draft-logit
// cache the rejection sampler reads.
//
// Falling through to `SampleDflashBlockDrafts` instead would SUCCEED and be
// silent: the DFlash1 per-slot argmax proposes well-formed tokens, the verify is
// lossless, so the engine still emits the target's tokens and only ACCEPTANCE
// falls. That is why this is a refusal rather than a fallback, and why it sits
// AFTER the scoring: everything before it is implemented and gated, the choice
// among the scored paths is not.
//
// IT TAKES THE SCORED LATTICE, and that is not decoration. The message names the
// lattice it is declining to walk -- how many requests, steps and candidates,
// and how many transitions were scored -- which tells a user exactly how far the
// port got rather than only where it stopped. It is also what makes the
// SELECTOR's execution observable at this call site: delete the
// `Dflash2SelectCandidates` call above it and pass a default-constructed state,
// and the counts read zero and the model gate goes red. Without that argument
// the refusal fires identically whether the selector ran or not, which is the
// shape spec `## Owed` O7 records for W2's refusal.
void RefuseDflash2PathWalk(const Qwen3DFlashWeights& weights,
                           const Dflash2ProposeState& scored);

}  // namespace vllm::v1
