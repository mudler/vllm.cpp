// Ported from: vllm/v1/sample/sampler.py @ e24d1b24 (the `Sampler` nn.Module).
//
// The V1 Sampler assembles the ordered sampling pipeline over the model's final
// logits [num_reqs, vocab] (row-major f32) and produces a SamplerOutput. The
// ORDER is the correctness core and mirrors Sampler.forward + Sampler.sample
// EXACTLY (sampler.py docstring steps 1-9):
//   1. If logprobs requested, snapshot raw_logprobs = compute_logprobs(logits)
//      BEFORE any mutation (logprobs_mode raw_logprobs; raw_logits is a stub).
//   2. logits -> float32 (already f32 here).
//   3. apply allowed_token_ids whitelist mask.
//   4. apply bad_words exclusion.
//   5. non-argmax-invariant procs, in upstream order: min_tokens then logit_bias.
//   6. penalties (repetition, frequency, presence).
//   7. sample(): greedy argmax snapshot (if !all_random); if all_greedy return
//      it; apply_temperature (temp<eps->1.0 guard when !all_random);
//      argmax-invariant procs (min_p); top_k_top_p; probs=softmax; random_sample;
//      where(temp<eps, greedy, random) per row for a mixed batch.
//   8. gather_logprobs(raw_logprobs, num_logprobs, sampled): top-k values+indices,
//      the sampled token's logprob, and ranks via batched_count_greater_than
//      (rank = #{vocab logprobs >= the sampled token's logprob}; 1-based — the
//      max-logprob token has rank 1). Concat [sampled_col | topk] -> [n, k+1].
//   9. Return SamplerOutput{ sampled_token_ids [n,1], logprobs_tensors }.
//
// The Sampler is device-agnostic: it composes the vt sampling ops (M1.7 Task 2/3)
// on whatever Queue's device the logits live on. Per-row host decisions (the
// where-merge, materializing the top_k/top_p/temperature/seed vectors) mirror the
// Task 2/3 host-array-then-op pattern. CPU is the correctness gate at T0; a
// CUDA-Queue run is dgx-pending.
//
// DEFERRED (marked 1:1 stubs, see sampler.cpp): logprob_token_ids gather
// (generative-scoring), spec-decode bonus-token (predict_bonus_token),
// thinking-budget, and the logprobs_mode variants beyond raw_logprobs.
#ifndef VLLM_V1_SAMPLE_SAMPLER_H_
#define VLLM_V1_SAMPLE_SAMPLER_H_

#include <cstdint>
#include <vector>

#include "vllm/v1/engine/types.h"     // SamplerOutput
#include "vllm/v1/sample/metadata.h"  // SamplingMetadata
#include "vt/tensor.h"

namespace vllm::v1 {

// LogprobsMode (vllm/config/model.py::LogprobsMode). Only raw_logprobs is
// implemented at T0; the raw_logits / processed_* modes are marked stubs (the
// all-greedy processed-logprobs branch and the raw_logits clone are deferred).
enum class LogprobsMode {
  kRawLogprobs,       // default: compute_logprobs(logits) before any mutation
  kRawLogits,         // STUB (deferred): clone the raw logits
  kProcessedLogprobs, // STUB (deferred): logprobs after temperature/top-k/top-p
  kProcessedLogits,   // STUB (deferred): logits after temperature/top-k/top-p
};

// Sampler (sampler.py::Sampler). Stateless apart from the logprobs_mode; the
// forward call carries all per-step state via the SamplingMetadata.
class Sampler {
 public:
  explicit Sampler(LogprobsMode logprobs_mode = LogprobsMode::kRawLogprobs)
      : logprobs_mode_(logprobs_mode) {}

  // Sampler.forward. `logits` [num_reqs, vocab] f32 is mutated in place through
  // the pipeline (upstream's logits.to(float32) is a no-op copy when already f32,
  // so upstream mutates in place too). Returns the sampled tokens (+ logprobs).
  SamplerOutput forward(vt::Queue& q, vt::Tensor& logits,
                        const SamplingMetadata& sampling_metadata) const;

  // Async-decode device greedy fast path (VT_ASYNC_DECODE). When this batch is a
  // pure all-greedy step with NO active logits processor (no penalties / bad_words
  // / allowed_token_ids / min_tokens / logit_bias) and NO logprobs request — the
  // exact conditions under which forward()'s all-greedy result is a bare
  // argmax(logits) — run vt::GreedyArgmax straight into the caller-provided DEVICE
  // int64 buffer `out_device_ids` [num_reqs] and return true, WITHOUT the blocking
  // full-token download (DeviceBuffer::download's Synchronize) forward() performs.
  // The token ids stay ON DEVICE so the runner can build the next step's input_ids
  // on-GPU and read them back on a side stream (mirroring vLLM's
  // prev_sampled_token_ids). The GreedyArgmax kernel + its lowest-index tie-break
  // are UNCHANGED — only the RESIDENCE of the result differs, so the tokens are
  // BIT-IDENTICAL to forward()'s greedy path. Returns false (does nothing) when
  // any of the above conditions fails; the caller must then fall back to forward().
  bool sample_greedy_device(vt::Queue& q, vt::Tensor& logits,
                            vt::Tensor& out_device_ids,
                            const SamplingMetadata& sampling_metadata) const;

 private:
  // Sampler.sample. Runs steps 7a-7f; returns the [num_reqs] host token ids.
  std::vector<int64_t> sample(vt::Queue& q, vt::Tensor& logits,
                              const SamplingMetadata& sampling_metadata) const;

  LogprobsMode logprobs_mode_;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_SAMPLE_SAMPLER_H_
