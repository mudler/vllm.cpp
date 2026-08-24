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
//   5. _sample_path                                (here, W4)
//   6. _cache_draft_logits                         (probabilistic only; owed)
//
// This file owns 2, the sequencing of 3 and 4, and the PATH WALK of 5. W3's
// `RefuseDflash2PathWalk` is retired by this wave, exactly as W2's
// `RefuseDflash2CandidateSelector` was retired by W3: the walk is no longer the
// missing mechanism, so there is no mechanism left to refuse on the greedy arm a
// user can configure. What replaces it is a GUARD in the other direction --
// `RefuseDflash1ArgmaxOnDflash2Block` -- so that losing the walk's call site
// fails loudly instead of silently drafting with the DFlash1 per-slot argmax.
//
// STEP 6 IS NOT HERE, and its absence is upstream's own at the moved head. The
// realized-score cache is what feeds the rejection sampler a proposal
// distribution, and `_generate_draft` calls it only `if self.draft_logits is not
// None` -- which `DraftModelSpeculator.__init__` sets only for
// `draft_sample_method == "probabilistic"`. That value is refused BY NAME here
// (`vllm::ParseSpeculativeConfigJson`, src/vllm/config/speculative.cpp) and this
// engine verifies accept-iff-equal, so no configuration can reach step 6 and
// nothing could read what it wrote. Its layout is recorded rather than built:
// upstream's `draft_logits_spec` returns `(torch.float32, -inf)` for DFlash2 --
// fp32 rather than the head dtype because rounding real selector scores to bf16
// moves a candidate row's argmax 0.81% of the time, and an `-inf` fill because
// the cache kernel writes only the K candidate columns and every column it never
// touches has to read as impossible. See `## Owed` of
// .agents/specs/dflash2-spec-decode.md.
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

// SPEC-DFLASH2 W8 (#1837): the DEVICE-RESIDENT propose state — the same
// candidates and lattice, never leaving the device. This is what upstream's
// `_generate_draft` holds between `compute_candidates` and
// `_selector_walk_kernel`: device tensors end to end.
struct Dflash2ProposeStateDevice {
  Dflash2CandidateSetDevice candidates;             // ids/values [num_reqs*num_steps, top_k]
  Qwen3DFlash2Model::Dflash2EdgeScoresDevice edges; // [num_reqs, num_steps, top_k, top_k] f32
  int64_t num_reqs = 0;
  int64_t num_steps = 0;
  int64_t top_k = 0;
};

// SPEC-DFLASH2 W8 (#1837): steps 2-4 over the block forward's DEVICE outputs
// (`DflashBlockDeviceOut`): the sample-row gather is a device vt::IndexSelect
// (upstream's `last_hidden_states[self.sample_indices[:num_sample]]`), the
// top-k, the value scalars, the projection and the edge lattice all run on
// device, and NOTHING is downloaded. The production runner calls this; the
// host-vector `Dflash2SelectCandidates` above is a marshaling shell over the
// same cores, kept for `DflashProposeBlock` and the unit surface. `block_logits`
// is [num_reqs*(1+k), draft_vocab] f32 and `block_hidden` [num_reqs*(1+k), H]
// bf16 (the post-final-norm bits the pre-W8 f32 detour round-tripped exactly).
Dflash2ProposeStateDevice Dflash2SelectCandidatesDevice(
    const vt::Tensor& block_logits, const vt::Tensor& block_hidden,
    const std::vector<int32_t>& anchors, int num_reqs, int k,
    const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue);

// The PATH WALK (SPEC-DFLASH2 W4, #1314) — `DFlash2Speculator._sample_path`
// (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:148-172 @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`),
// which is one `_selector_walk_kernel` launch of `num_reqs` programs.
//
// Turns the selector's lattice into the k drafted tokens of every proposing row:
// start at the verified anchor -- already present as EVERY predecessor slot of
// step 0, so the walk enters at slot 0 -- take the best child, and read the next
// step's block at the predecessor row just chosen. `vt::Dflash2PathWalk`
// (include/vt/ops.h) carries the tie-break, the all -inf rule and why the greedy
// arm is the only one.
//
// Returns `[num_reqs][num_steps]` token ids in the TARGET vocabulary, which is
// what `compute_candidates` emits after the org-vocab rebase and what the
// verify compares against. Row r of the result belongs to proposing row r of
// `scored`, in the same order `Dflash2SelectCandidates` was given.
//
// THE LATTICE MAKES ONE ROUND TRIP, and that is named rather than hidden. The
// selector downloads `edge_scores` (its own observable, and what the D9 flip
// gate reads) and this function uploads them again for the walk. At the
// published shapes that is `num_reqs * k * K * K` f32 -- 8 KB per request per
// step -- so it is small, and it is still a transfer upstream does not make,
// because upstream never leaves the device between the two. Fusing them (and
// computing the candidates inside the block forward, which is what would put a
// DFlash2 draft back on the paged CUDA-graph fast path W3 took it off) is a
// SPEED change and belongs to the wave that takes a number; this row claims
// none. Recorded at `## Owed` of .agents/specs/dflash2-spec-decode.md.
struct Dflash2WalkResult {
  std::vector<std::vector<int32_t>> draft_token_ids;  // [num_reqs][num_steps]
};

Dflash2WalkResult Dflash2WalkPath(const Dflash2ProposeState& scored, vt::Queue& queue);

// SPEC-DFLASH2 W8 (#1837): the SAME walk over the device-resident state, with
// exactly ONE download in the whole selector+walk — the [num_reqs, num_steps]
// i64 drafted token ids, which is all upstream ever brings back either. The
// "lattice makes one round trip" note above described the pre-W8 host shells;
// this entry is the fusion that note deferred, and the host `Dflash2WalkPath`
// is now a marshaling shell over it.
Dflash2WalkResult Dflash2WalkPathDevice(const Dflash2ProposeStateDevice& scored,
                                        vt::Queue& queue);

// THE GUARD THAT REPLACES W3's REFUSAL, pointing the other way.
//
// W1, W2 and W3 each refused a DFlash2 draft by name because a mechanism was
// missing. Nothing on the greedy arm is missing now, so there is nothing left to
// refuse -- but the thing that made those refusals necessary has not changed:
// sampling a DFlash2 block with the DFlash1 per-slot argmax SUCCEEDS. It returns
// well-formed tokens, the verify is lossless, the engine still emits the
// target's tokens, and only ACCEPTANCE falls. No token gate in this repository
// can see it.
//
// So both propose paths call this immediately before they would fall back to
// `SampleDflashBlockDrafts`, and it throws when the draft is a DFlash2 one. On a
// DFlash1 draft it is a no-op and runs on every propose. Its value is that
// deleting the walk's call site -- the mutation `.agents/reachability.md`
// requires a reviewer to make -- becomes a LOUD failure instead of a green run
// that drafts worse tokens.
void RefuseDflash1ArgmaxOnDflash2Block(const Qwen3DFlashWeights& weights);

}  // namespace vllm::v1
