// Ported from: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
// (`_prepare_decode_inputs_kernel` :597-645 + the `prepare_decode_inputs`
// wrapper :648-671, and `_update_draft_inputs_kernel` :674-738 + the
// `update_draft_inputs` wrapper :741-771) @ 5559679229bc961848b121ccdeaa8fa5d79bec98,
// driven from `AutoRegressiveSpeculator.propose` (:240-274) and
// `_generate_draft` (:426-471).
//
// Scope (SPEC-MTP-K-GT-1, issue #81): the DRAFT-DECODE half of the
// autoregressive propose — the half `prepare_prefill_inputs.h` explicitly
// deferred ("k>1 adds prepare_decode_inputs (:591-665), DEFERRED"). Its sibling
// prepares the ONE draft prefill forward that k=1 early-exits after
// (speculator.py:236-238); this prepares the k-1 single-token draft decode steps
// that follow it, and advances that state between them.
//
// ─── THE DECODE ENTRY STATE, EXACTLY (per request r) ────────────────────────
// After the draft prefill, `_prefill` (:363-371) leaves three per-request values
// behind: the step-0 draft token, the draft model's hidden state at the
// request's `last_token_indices` row, and that row's POSITION. From those,
// `prepare_decode_inputs` (:628-645) builds the first decode step's inputs:
//
//   input_ids[r]  = draft_tokens[r][0]                          # :630-631
//   positions[r]  = min(prefill_position[r] + 1,                # :637-639
//                       max_model_len - 1)
//   seq_lens[r]   = min(target_seq_lens[r] - num_rejected[r]    # :641-645
//                       + 1, max_model_len)
//   query_start_loc = identity 0..num_reqs                      # :617-621
//
// The identity query_start_loc is the whole shape of a draft decode step: every
// request contributes EXACTLY one query token, which is why upstream can replay
// one captured graph across the k-1 steps (:255-263).
//
// ─── THE STEP ADVANCE, EXACTLY (per request r, at draft step `step`) ────────
// `update_draft_inputs` (:695-738) records the step's token and, unless this was
// the last step, feeds it forward:
//
//   draft_tokens[r][step] = token                               # :696-702
//   if step >= num_speculative_steps - 1: return                # :704-706
//   input_ids[r] = token                                        # :710
//   next_hidden[r] = hidden[r]                                  # :714-726
//   positions[r] = min(positions[r] + 1, max_model_len - 1)     # :732-734
//   seq_lens[r]  = min(seq_lens[r] + 1, max_model_len)          # :736-737
//
// The hidden carry (:714-726) is a straight per-request row copy because a
// decode step's forward already produces exactly one row per request, so it is
// the identity here and needs no gather. Only the PREFILL to first-decode
// handoff gathers, through Qwen3_5MTPModel::GatherHiddenRows. That copy is
// therefore not represented in this header's state, which carries only the
// integer inputs the two kernels own.
//
// ─── ADVANCE_DRAFT_POSITIONS ────────────────────────────────────────────────
// Both kernels take an `ADVANCE_DRAFT_POSITIONS` constexpr (:610, :690). It is
// TRUE for Eagle and standard MTP, which is what Qwen3.5/3.6 are
// (`advance_draft_positions`, :52-59); the false case is Gemma4 MTP, whose
// Q-only head shares the target KV and keeps positions constant, and which is
// not a Qwen3.5 architecture. This port takes the true branch only. When the
// Gemma4 MTP head lands it needs the false branch here, not a second copy.
//
// ─── CUDA-GRAPH REQUEST-COUNT PADDING (deliberate deviation) ────────────────
// Upstream pads query_start_loc and seq_lens out to `max_num_reqs` (:617-626)
// because its draft decode IS graph-captured (:255-263) and the captured graph
// reads the persistent buffers' padded tail. Our draft decode is not captured,
// and its only consumer is BuildFullAttnStepDevInputs (qwen3_5.cpp), which
// requires `query_start_loc.size() == num_reqs + 1` and `seq_lens.size() ==
// num_reqs` exactly. Padding here would be dead work that the caller would have
// to slice back off, so these arrays are built at the request count. This is the
// same deviation, for the same reason, that BlockTable::compute_slot_mapping
// records for upstream's slot_mapping tail-pad. A future captured draft decode
// restores the pad here.
#ifndef VLLM_V1_WORKER_GPU_SPEC_DECODE_AUTOREGRESSIVE_PREPARE_DECODE_INPUTS_H_
#define VLLM_V1_WORKER_GPU_SPEC_DECODE_AUTOREGRESSIVE_PREPARE_DECODE_INPUTS_H_

#include <cstdint>
#include <vector>

namespace vllm::v1 {

// The draft model's per-step decode inputs, mirroring the `InputBuffers` rows
// the two kernels above read and write. One query token per request, so every
// array is request-indexed and `input_ids` doubles as the token stream.
struct SpecDecodeInputs {
  // [num_reqs] the token each request drafts FROM this step (the previous
  // step's drafted token).
  std::vector<int32_t> input_ids;
  // [num_reqs] its position. i32 because Qwen3_5MTPModel::ForwardPaged takes i32
  // positions, as the target and standalone forwards do.
  std::vector<int32_t> positions;
  // [num_reqs] the request's draft-side sequence length AFTER this step's token.
  std::vector<int32_t> seq_lens;
  // [num_reqs + 1] the identity 0..num_reqs: one query token per request.
  std::vector<int32_t> query_start_loc;

  int num_reqs() const { return static_cast<int>(input_ids.size()); }
};

// Build the FIRST draft decode step's inputs (speculator.py:628-645 + :617-621).
// Pure function of the prefill's outcome; see the header for the exact rule.
//
//   draft_tokens_step0  [num_reqs] i32 — the drafted token from the prefill
//   target_seq_lens     [num_reqs] i32 — the verify step's per-request seq_lens
//   num_rejected        [num_reqs] i32 — the RejectionSampler's per-request count
//   prefill_positions   [num_reqs] i32 — the target position at each request's
//                                        last_token_index (speculator.py:353)
//   max_model_len                      — the clamp bound (:638, :644)
SpecDecodeInputs prepare_decode_inputs(
    const std::vector<int32_t>& draft_tokens_step0,
    const std::vector<int32_t>& target_seq_lens,
    const std::vector<int32_t>& num_rejected,
    const std::vector<int32_t>& prefill_positions, int max_model_len);

// Record `step`'s drafted tokens and advance the decode state for the next step
// (speculator.py:695-738).
//
//   drafted        [num_reqs] i32 — this step's per-request drafted token
//   step                          — the draft step index, in [0, k)
//   num_speculative_steps         — k
//   out_draft_tokens [num_reqs * k] i32 — row-major per-request draft buffer,
//                                         written at column `step`
//   inputs                        — advanced IN PLACE, and left untouched when
//                                   this was the final step (:704-706)
//   max_model_len                 — the clamp bound (:733, :737)
//
// Returns true when a further decode step follows, i.e. exactly upstream's
// "did not take the :704-706 early return".
bool update_draft_inputs(const std::vector<int32_t>& drafted, int step,
                         int num_speculative_steps,
                         std::vector<int32_t>& out_draft_tokens,
                         SpecDecodeInputs& inputs, int max_model_len);

// The draft KV slot each request's decode token writes, for a batch of
// one-token-per-request queries.
//
// The host equivalent of `self.block_tables.compute_slot_mappings(idx_mapping,
// query_start_loc, positions, num_tokens)` (speculator.py:392-397) at
// total_cp_world_size == 1, which is what BlockTable::compute_slot_mapping
// reduces to: `block_table[r][pos / block_size] * block_size + pos %
// block_size`. The DRAFT KV group (`fa_draft`) has the same block geometry as
// the target's full-attention group, so the target's block table addresses it
// unchanged — which is the same reason the draft PREFILL reuses the target's
// slot mapping verbatim (speculator.py:222-234).
//
//   block_table  [num_reqs * cols] i32, row-major
//   positions    [num_reqs]        i32
std::vector<int64_t> draft_decode_slot_mapping(
    const std::vector<int32_t>& block_table, int block_table_num_cols,
    const std::vector<int32_t>& positions, int block_size);

}  // namespace vllm::v1

#endif  // VLLM_V1_WORKER_GPU_SPEC_DECODE_AUTOREGRESSIVE_PREPARE_DECODE_INPUTS_H_
