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
#include <memory>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm::v1 {

class DeviceScratch;

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

// ─── SPEC-DFLASH2 A2-2: THE DEVICE-RESIDENT ACCEPT WALK ─────────────────────
//
// Upstream's `RejectionSampler.__call__` returns a `SamplerOutput` whose
// `sampled_token_ids`, `num_sampled` and `num_rejected` are DEVICE tensors
// (vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:262-272 @ pin
// 5559679229). Nothing crosses to the host inside the sampler; the one D2H of
// the step is issued afterwards by `AsyncOutput` on the COPY stream
// (vllm/v1/worker/gpu/model_runner.py:1492-1499, async_utils.py:29-44).
//
// Our `forward` below did both halves in one call and paid a full MAIN-QUEUE
// `Synchronize` for the copy. On a speculative step that is one of the two
// compute-stream drains SPEC-DFLASH2 A2 exists to remove (the other is the
// speculator's own draft download), so the walk is split in two:
//
//   * `verify` issues the kernel and hands back THIS object — the buffers the
//     kernel wrote, still on the device, with nothing waited on;
//   * the caller chooses where the bytes cross. `forward` copies on the SAME
//     queue and drains it, which is byte-for-byte the pre-split behaviour and
//     the route every synchronous caller keeps. The async verify arm forks a
//     COPY queue off the main queue with an event, copies there, and blocks the
//     host on that copy event alone, so the wait is a copy-queue event rather
//     than a main-queue drain. The HOST still waits in step; see `## Owed` in
//     `.agents/specs/dflash2-async-spec-sampler.md` for what A2-4 and A2-5 move.
//
// LIFETIME IS THE POINT OF THE TYPE, not a detail of it. On a discrete backend
// the accept walk is still queued when `verify` returns, and its two kernels
// read or write SIX buffers. Freeing any of them before the caller has waited is
// a device use-after-free — and a unified-memory backend (CPU, GB10) cannot show
// it, because there the staging is an in-place wrap of the caller's host vector
// and the kernels have already finished. So the invariant is stated here per
// buffer, because "the object owns everything" was written once and was FALSE:
//
//   OWNED, and freed in the destructor — "still on device" and "still alive" are
//   one statement for these five:
//     * `sampled_`      [num_reqs, width] i32, the accept kernel's output;
//     * `num_sampled_`  [num_reqs] i32, its other output;
//     * `draft_`        the uploaded `draft_sampled` staging it reads;
//     * `cu_`           the uploaded `cu_num_logits` staging it reads;
//     * `target_argmax_` [num_logits] i32, the buffer the argmax kernel writes
//       and the accept kernel reads. This one is owned HERE and is a parameter
//       of `vt::GreedyRejectionSample` precisely because it cannot be a private
//       static in the backend: a process-global grow-only scratch that a later,
//       larger call `cudaFree`s is a free under this object's still-queued
//       accept kernel, and the damage it does is a wrong accept prefix that no
//       token gate in this tree can see (SPEC-DFLASH2 A2-2, #2802).
//
//   OWNED, BUT NOT FREE-AT-WILL — "which buffers" is only half the invariant and
//   the other half is WHEN. A destructor that frees these five without waiting
//   is the same defect as the backend global, relocated from the backend into
//   this object's own scope, and it has a concrete caller: on the copy-queue
//   route in `GPUModelRunner::sample_tokens_async` the device result is a
//   BLOCK-SCOPED LOCAL destroyed at the close of the `else`, while A2-4's stated
//   job is to move `SynchronizeEvent(verify_ready_event_)` past `propose_drafts`.
//   Move that wait past the closing brace and the destructor frees `sampled_`,
//   `num_sampled_` and `target_argmax_` while the D2H copy is still writing them:
//   garbage accept prefix, wrong emitted ids, nothing raised, and — because the
//   verify is lossless — invisible to every token gate in this tree (#1366).
//
//   So `Release` DRAINS BEFORE IT FREES. It synchronizes the queue `verify`
//   issued the walk on, and the queue of the last `CopyToHost` when that is a
//   different one, and only then frees. This is a SAFETY NET, not a licence: a
//   caller that destroys the object before its own wait now pays a full drain in
//   the destructor instead of corrupting silently. That trade is deliberate. A
//   stall is visible in the G4 `nsys` read A2-5 must take anyway; a wrong accept
//   prefix is visible to nothing. A2-4 still has to destroy this object AFTER
//   the wait it moves — the net makes the mistake slow rather than wrong, and it
//   does not make the mistake correct.
//
//   NOT OWNED, and therefore a CALLER OBLIGATION with no destructor to enforce
//   it:
//     * `logits` — the [num_logits, vocab] tensor passed to `verify`. It is the
//       forward's own output buffer on the device path
//       (`GPUModelRunner::assemble_sample_logits` hands the sampler
//       `exec_state_.logits.device_tensor` directly), and the next step's
//       forward WRITES THAT BUFFER. The argmax kernel reads it after `verify`
//       returns, so the caller must not free it, reuse it, or run the next
//       forward until it has waited for this object's outputs. Every caller in
//       the tree satisfies this today because the wait is still in step. A2-4
//       and A2-5 move that wait, and moving it past the next forward without
//       double-buffering the logits is the failure this paragraph names in
//       advance.
//     * on a unified-memory backend the `draft_` / `cu_` staging IS the caller's
//       `draft_sampled` / `cu_num_logits` vectors, wrapped in place. They must
//       outlive this object, exactly as they must outlive a `forward` call.
//     * the QUEUES. The drain above holds a `vt::Queue*` to the queue `verify`
//       was given and to the queue the last `CopyToHost` was given, so both must
//       outlive this object. That is the ordinary shape already — the runner's
//       `queue_` and its async copy queue are both members — and it is a
//       scope inversion a reviewer can see, unlike the corruption it replaces.
//       Destroying a queue that still has work on it is undefined on CUDA
//       regardless of this type.
class RejectionSamplerDeviceOutput {
 public:
  RejectionSamplerDeviceOutput() = default;
  ~RejectionSamplerDeviceOutput();
  RejectionSamplerDeviceOutput(RejectionSamplerDeviceOutput&& other) noexcept;
  RejectionSamplerDeviceOutput& operator=(
      RejectionSamplerDeviceOutput&& other) noexcept;
  RejectionSamplerDeviceOutput(const RejectionSamplerDeviceOutput&) = delete;
  RejectionSamplerDeviceOutput& operator=(const RejectionSamplerDeviceOutput&) =
      delete;

  // num_reqs == cu_num_logits.size() - 1; `width` is the `sampled` row stride,
  // upstream's `num_speculative_steps + 1`
  // (rejection_sampler_utils.py:1026-1028).
  int64_t num_reqs() const { return num_reqs_; }
  int64_t width() const { return width_; }
  vt::Device device() const { return device_; }

  // Issue the two D2H copies on `q`. NO `Synchronize` and NO event recorded —
  // the caller owns the wait, which is exactly what lets that wait be a
  // copy-queue event instead of a main-queue drain. `sampled_out` receives
  // num_reqs * width i32 values, `num_sampled_out` receives num_reqs. `q` is
  // RECORDED, so the destructor can drain it if the caller frees this object
  // before waiting; it must therefore outlive this object.
  void CopyToHost(vt::Queue& q, int32_t* sampled_out,
                  int32_t* num_sampled_out) const;

  // The accept walk's [num_logits] i32 argmax scratch, so a test can assert the
  // ownership rule above rather than read it: two device results that are alive
  // at the same time must not share this pointer. That is the ONE property of
  // the fix a unified-memory tier can observe — on CPU the kernel has already
  // finished, so a shared buffer corrupts nothing there and only the aliasing
  // itself is visible. Null when num_reqs == 0.
  const void* target_argmax_scratch() const { return target_argmax_; }

 private:
  friend class RejectionSampler;
  void Release();

  vt::Backend* backend_ = nullptr;
  vt::Device device_{};
  void* sampled_ = nullptr;      // [num_reqs, width] i32, device-resident
  void* num_sampled_ = nullptr;  // [num_reqs] i32, device-resident
  // [num_logits] i32, device-resident: written by the argmax kernel, read by the
  // accept kernel, both after `verify` returns. Owned here for that reason.
  void* target_argmax_ = nullptr;
  int64_t num_reqs_ = 0;
  int64_t width_ = 0;
  // The queues the object's own work is on, so `Release` can drain before it
  // frees (see "OWNED, BUT NOT FREE-AT-WILL" above). `verify_q_` is the queue
  // the accept walk was issued on; `copy_q_` is the queue of the last
  // `CopyToHost`, null when the caller never copied. Both are borrowed and must
  // outlive this object.
  vt::Queue* verify_q_ = nullptr;
  mutable vt::Queue* copy_q_ = nullptr;
  // The kernel's INPUTS, held for the same reason the outputs are.
  std::unique_ptr<DeviceScratch> draft_;
  std::unique_ptr<DeviceScratch> cu_;
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

  // A2-2 half one: run the accept walk and leave EVERY output on the device.
  // Same arguments and same refusals as `forward`, minus the host-only
  // `is_chunked_prefilling` (which `finalize` applies, because it never reaches
  // the kernel). No copy is issued and no queue is drained.
  RejectionSamplerDeviceOutput verify(
      vt::Queue& q, const vt::Tensor& logits,
      const std::vector<int32_t>& draft_sampled,
      const std::vector<int32_t>& cu_num_logits) const;

  // A2-2 half two: the pure-HOST reduction over the walk's downloaded outputs.
  // `host_sampled` is [num_reqs * width] and `host_num_sampled` is [num_reqs],
  // exactly as `RejectionSamplerDeviceOutput::CopyToHost` wrote them. Mirrors
  // `get_num_sampled_and_rejected` (gpu/input_batch.py:408-453): num_rejected =
  // num_logits - num_sampled, and a still-chunked-prefilling row samples nothing
  // and rejects nothing. Static because it touches no device state — which is
  // also why the split is safe: whichever queue carried the bytes, this half is
  // the same function of the same numbers.
  static RejectionSamplerOutput finalize(
      const std::vector<int32_t>& host_sampled, int64_t width,
      const std::vector<int32_t>& host_num_sampled,
      const std::vector<int32_t>& cu_num_logits,
      const std::vector<char>& is_chunked_prefilling);

 private:
  int num_speculative_steps_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SPEC_DECODE_REJECTION_SAMPLER_H_
