// Ported from vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
// (`_prepare_decode_inputs_kernel` / `_update_draft_inputs_kernel`) @ 555967922.
// See the header for scope, the exact per-request rules, and the two recorded
// deviations. SPEC-MTP-K-GT-1, issue #81.
#include "vllm/v1/worker/gpu/spec_decode/autoregressive/prepare_decode_inputs.h"

#include <algorithm>
#include <cstddef>

#include "vt/backend.h"

namespace vllm::v1 {
namespace {

// tl.minimum over int64, kept as one named helper so the two clamps below read
// as the upstream lines they mirror rather than as arithmetic.
int32_t Min32(int64_t a, int64_t b) {
  return static_cast<int32_t>(a < b ? a : b);
}

}  // namespace

SpecDecodeInputs prepare_decode_inputs(
    const std::vector<int32_t>& draft_tokens_step0,
    const std::vector<int32_t>& target_seq_lens,
    const std::vector<int32_t>& num_rejected,
    const std::vector<int32_t>& prefill_positions, int max_model_len) {
  const size_t num_reqs = draft_tokens_step0.size();
  VT_CHECK(num_reqs > 0, "prepare_decode_inputs: empty batch");
  VT_CHECK(target_seq_lens.size() == num_reqs &&
               num_rejected.size() == num_reqs &&
               prefill_positions.size() == num_reqs,
           "prepare_decode_inputs: every per-request array must have num_reqs "
           "entries");
  VT_CHECK(max_model_len > 0, "prepare_decode_inputs: max_model_len must be "
                              "positive");

  SpecDecodeInputs out;
  out.input_ids.resize(num_reqs);
  out.positions.resize(num_reqs);
  out.seq_lens.resize(num_reqs);
  out.query_start_loc.resize(num_reqs + 1);

  for (size_t r = 0; r < num_reqs; ++r) {
    // :630-631 — the step-0 draft token becomes this step's input id.
    out.input_ids[r] = draft_tokens_step0[r];
    // :637-639 — advance one position past the prefill's last sampled row, and
    // clamp to max_model_len - 1 so the block-table lookup stays in range.
    out.positions[r] =
        Min32(static_cast<int64_t>(prefill_positions[r]) + 1,
              static_cast<int64_t>(max_model_len) - 1);
    // :641-645 — the draft's own sequence length: the target's, less the
    // rejected tail this step rolled back, plus the one token drafted now.
    const int64_t accepted_len =
        static_cast<int64_t>(target_seq_lens[r]) - num_rejected[r];
    out.seq_lens[r] = Min32(accepted_len + 1, max_model_len);
  }
  // :617-621 — one query token per request, so query_start_loc is the identity.
  for (size_t r = 0; r <= num_reqs; ++r) {
    out.query_start_loc[r] = static_cast<int32_t>(r);
  }
  return out;
}

bool update_draft_inputs(const std::vector<int32_t>& drafted, int step,
                         int num_speculative_steps,
                         std::vector<int32_t>& out_draft_tokens,
                         SpecDecodeInputs& inputs, int max_model_len) {
  const size_t num_reqs = drafted.size();
  VT_CHECK(num_reqs > 0, "update_draft_inputs: empty batch");
  VT_CHECK(inputs.input_ids.size() == num_reqs &&
               inputs.positions.size() == num_reqs &&
               inputs.seq_lens.size() == num_reqs,
           "update_draft_inputs: decode state must match the batch");
  VT_CHECK(step >= 0 && step < num_speculative_steps,
           "update_draft_inputs: draft step out of range");
  VT_CHECK(out_draft_tokens.size() ==
               num_reqs * static_cast<size_t>(num_speculative_steps),
           "update_draft_inputs: draft buffer must be [num_reqs, k]");

  // :696-702 — record the step's token into column `step` of the per-request
  // draft row. This happens on EVERY step, including the last one.
  const size_t k = static_cast<size_t>(num_speculative_steps);
  for (size_t r = 0; r < num_reqs; ++r) {
    out_draft_tokens[r * k + static_cast<size_t>(step)] = drafted[r];
  }

  // :704-706 — the final step drafts its token and stops. Nothing is fed
  // forward, which is what makes the caller's loop bound and this the one place
  // that decides it.
  if (step >= num_speculative_steps - 1) return false;

  for (size_t r = 0; r < num_reqs; ++r) {
    // :710 — the token just drafted is the next step's input.
    inputs.input_ids[r] = drafted[r];
    // :732-734 / :736-737 — ADVANCE_DRAFT_POSITIONS is true for standard MTP
    // (:52-59), so both position and sequence length advance by one, each
    // clamped to the model length exactly as upstream clamps them.
    inputs.positions[r] = Min32(static_cast<int64_t>(inputs.positions[r]) + 1,
                                static_cast<int64_t>(max_model_len) - 1);
    inputs.seq_lens[r] =
        Min32(static_cast<int64_t>(inputs.seq_lens[r]) + 1, max_model_len);
  }
  return true;
}

std::vector<int64_t> draft_decode_slot_mapping(
    const std::vector<int32_t>& block_table, int block_table_num_cols,
    const std::vector<int32_t>& positions, int block_size) {
  const size_t num_reqs = positions.size();
  VT_CHECK(num_reqs > 0, "draft_decode_slot_mapping: empty batch");
  VT_CHECK(block_size > 0 && block_table_num_cols > 0,
           "draft_decode_slot_mapping: block geometry must be positive");
  VT_CHECK(block_table.size() >=
               num_reqs * static_cast<size_t>(block_table_num_cols),
           "draft_decode_slot_mapping: block table smaller than the batch");

  std::vector<int64_t> slots(num_reqs);
  for (size_t r = 0; r < num_reqs; ++r) {
    const int64_t pos = positions[r];
    const int64_t block_index = pos / block_size;
    VT_CHECK(block_index < block_table_num_cols,
             "draft_decode_slot_mapping: draft position past the request's "
             "block table (the draft ran beyond the allocated KV)");
    const int64_t block_number =
        block_table[r * static_cast<size_t>(block_table_num_cols) +
                    static_cast<size_t>(block_index)];
    slots[r] = block_number * block_size + (pos - block_index * block_size);
  }
  return slots;
}

}  // namespace vllm::v1
