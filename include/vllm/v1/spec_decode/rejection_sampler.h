// Ported from: vllm/v1/worker/gpu/spec_decode/rejection_sampler.py (the
// `RejectionSampler` module) + vllm/v1/worker/gpu/spec_decode/
// rejection_sampler_utils.py::rejection_sample @ e24d1b24.
//
// Scope (SPEC-MTP increment I3 / row SPEC-REJECTION): the VERIFY half of
// speculative decoding — given the target model's EXPANDED logits (1 + k_i rows
// per request; see StepInputs::cu_num_logits) and the draft token ids that were
// scheduled for the step, decide per request how many drafts are accepted, emit
// the accepted token sequence plus the bonus/replacement token, and report the
// per-request num_sampled / num_rejected the scheduler rollback and the GDN spec
// state selection consume.
//
// ─── THE GREEDY ACCEPT RULE (temperature 0), EXACTLY ───────────────────────
// Mirrors the `is_greedy` branch of `_rejection_kernel`
// (rejection_sampler_utils.py:564-585) + the accepted-length store at :628, the
// greedy short-circuit of `_resample_kernel` (:846-849, "Greedy + non-bonus
// token. No resampling needed because the target argmax is already in the
// sampled tensor"), and `_insert_resampled_kernel` (:828-841, num_sampled += 1
// and, for the bonus position, the argmax insert).
//
// For request r with expanded rows [cu[r], cu[r+1]) and k_r = cu[r+1] - cu[r] - 1:
//
//   accepted = true; len = 0
//   for i in [0, k_r):
//     if !accepted: break                       # upstream `elif accepted:` guard
//     target_argmax = argmax(logits[cu[r] + i])
//     draft         = draft_sampled[cu[r] + i + 1]     # NOTE the +1, :534
//     accepted     &= (draft == target_argmax)
//     sampled[r][i] = accepted ? draft : target_argmax
//     len          += accepted
//   if len == k_r:                              # every draft accepted
//     sampled[r][k_r] = argmax(logits[cu[r] + k_r])    # the BONUS token
//   num_sampled[r]  = len + 1
//   num_rejected[r] = (1 + k_r) - num_sampled[r] = k_r - len
//
// Three properties this pins down, each of which the I5 e2e greedy gate depends
// on:
//   1. ACCEPT IFF EQUAL. A draft is accepted only when it equals the target's
//      argmax at its own position, so every emitted token is a token the
//      non-speculative greedy run would have emitted.
//   2. STOP AT THE FIRST MISMATCH. `accepted` is sticky-false and the loop body
//      is guarded by it, so no draft after a rejection is ever accepted, and the
//      mismatch position emits the TARGET argmax (the replacement).
//   3. EXACTLY ONE EXTRA TOKEN. Whether the run ends in a rejection (the
//      replacement) or a full accept (the bonus), num_sampled = len + 1. A
//      request with k_r == 0 therefore emits exactly one token — the plain
//      greedy argmax — which is why the k = 0 path is byte-identical to the
//      non-speculative sampler.
// A placeholder draft id of -1 is rejected by construction (an argmax is >= 0),
// mirroring the upstream `-1` padding contract (`test_placeholder_draft_token_
// rejected`, tests/v1/spec_decode/test_rejection_sampler_utils.py:285).
//
// ─── DEFERRED (the obvious seams; M-mtp-3, spec §5) ───────────────────────
//   * The STOCHASTIC path (`accepted &= target_logprob > log(u) + draft_logprob`,
//     rejection_sampler_utils.py:589-627) and the residual-distribution
//     `_resample_kernel` Gumbel draw (:775-799). Everything below keys off
//     temperature == 0; a temperature > 0 request must NOT be routed here yet.
//   * `RejectionSampler.__call__`'s `apply_sampling_params` over the EXPANDED
//     batch (rejection_sampler.py:113-120), which needs
//     `expanded_idx_mapping` / `expanded_local_pos`. Greedy argmax is invariant
//     under temperature, top-k and top-p, so this is a no-op for the greedy gate
//     workload; it is REQUIRED before penalties / logit-bias / bad-words are
//     supported under spec decode.
//   * Block verification (:535+), synthetic acceptance rates, and logprobs over
//     the expanded batch (`_get_logprobs_tensors`, rejection_sampler.py:67-99).
#ifndef VLLM_V1_SPEC_DECODE_REJECTION_SAMPLER_H_
#define VLLM_V1_SPEC_DECODE_REJECTION_SAMPLER_H_

#include <cstdint>
#include <vector>

#include "vt/tensor.h"

namespace vllm::v1 {

// The per-request result of one verify step. Mirrors the fields of upstream's
// SamplerOutput that the spec path populates (`sampled_token_ids`,
// `num_sampled`, `num_rejected`; vllm/v1/worker/gpu/spec_decode/
// rejection_sampler.py:154-160).
struct RejectionSamplerOutput {
  // Per request, the accepted draft tokens followed by the bonus/replacement
  // token. Length == num_sampled[r] (always >= 1 for a sampling row).
  std::vector<std::vector<int32_t>> sampled_token_ids;
  // accepted_length + 1 per request. Feeds InputBatch::num_accepted_tokens
  // (the GDN recurrent-state slot select, spec §3 step 3).
  std::vector<int32_t> num_sampled;
  // (1 + k_r) - num_sampled[r] per request. Feeds the scheduler's
  // `num_computed_tokens -= num_rejected` rollback (scheduler.py:1580-1612,
  // landed by I2 at src/vllm/v1/core/sched/scheduler.cpp:589-618).
  std::vector<int32_t> num_rejected;
};

// The greedy rejection sampler. Stateless; `num_speculative_steps` only sizes
// the `sampled` scratch row (upstream `sampled = new_empty(num_reqs,
// num_speculative_steps + 1)`, rejection_sampler_utils.py:1026-1028).
class RejectionSampler {
 public:
  explicit RejectionSampler(int num_speculative_steps)
      : num_speculative_steps_(num_speculative_steps) {}

  int num_speculative_steps() const { return num_speculative_steps_; }

  // Run one verify step.
  //   logits         [num_logits, vocab] f32 on `q`'s device — the EXPANDED
  //                  verify logits (row cu[r]+j is request r's j-th spec
  //                  position). num_logits == cu_num_logits.back().
  //   draft_sampled  [num_logits] host i32 — `input_ids[logits_indices]`
  //                  (rejection_sampler.py:111). Row cu[r]+i+1 holds request r's
  //                  i-th draft token; row cu[r] holds the previous step's
  //                  token and is never compared.
  //   cu_num_logits  [num_reqs + 1] host i32 — StepInputs::cu_num_logits.
  //   is_chunked_prefilling  optional [num_reqs] host flags; a true entry zeroes
  //                  both num_sampled and num_rejected for that row, mirroring
  //                  `_get_num_sampled_and_rejected_kernel`
  //                  (gpu/input_batch.py:408-433). Empty == no row is prefilling.
  RejectionSamplerOutput forward(
      vt::Queue& q, const vt::Tensor& logits,
      const std::vector<int32_t>& draft_sampled,
      const std::vector<int32_t>& cu_num_logits,
      const std::vector<char>& is_chunked_prefilling = {}) const;

 private:
  int num_speculative_steps_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SPEC_DECODE_REJECTION_SAMPLER_H_
