// Ported from: vllm/sampling_params.py @ e24d1b24
//
// Scope: the T0 subset the V1 engine actually passes around (see the M1.1
// engine-core-types plan). Field names, defaults, enum int values, the
// sampling_type derivation and the _verify_args / __post_init__ semantics are
// mirrored 1:1 with upstream; only Python-specific concerns are dropped.
//
// DEFERRED (T1/T2) upstream fields, intentionally omitted here — a future
// porter slots them in without reshaping the struct:
//   - logit_bias,
//     allowed_token_ids, bad_words / _bad_words_token_ids, extra_args
//   (structured_outputs (StructuredOutputsParams) is now ported below — M3.4)
//   - logprob_token_ids, flat_logprobs, num_logprobs()
//   - thinking_token_budget, repetition_detection (RepetitionDetectionParams),
//     routed_experts_prompt_start, skip_reading_prefix_cache
//   - skip_clone / clone(), for_sampler_warmup()
//   - internal post-init state: output_text_buffer_length,
//     _all_stop_token_ids (the detokenizer computes its own stop buffer len)
//   - engine-time helpers: from_optional(), update_from_generation_config(),
//     update_from_tokenizer(), verify(model_config, ...) and its
//     _validate_* family, the eos_token_id / all_stop_token_ids /
//     bad_words_token_ids properties
//   - BeamSearchParams (separate struct)
//
// DEVIATIONS, recorded:
//   - Upstream raises ValueError / VLLMValidationError; here Verify() throws
//     std::runtime_error with the upstream-equivalent message text.
//   - `stop` / `stop_token_ids` are always list-form (upstream accepts a bare
//     str or None and normalizes in __post_init__); the str/None union has no
//     C++ analogue, so callers pass the already-normalized vectors.
//   - Python bool coercions (`logprobs is True -> 1`) have no analogue.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vllm {

// sampling_params._SAMPLING_EPS: below this temperature, sampling is greedy.
inline constexpr double kSamplingEps = 1e-5;
// sampling_params._MAX_TEMP: temperatures in (0, _MAX_TEMP) are clamped up to
// avoid nan/inf in downstream tensors.
inline constexpr double kMaxTemp = 1e-2;
// envs.VLLM_MAX_N_SEQUENCES default (upper bound on `n`).
inline constexpr int kMaxNSequences = 16384;

// SamplingType (IntEnum): int values are load-bearing (serialized / used as
// tensor row selectors upstream).
enum class SamplingType : int {
  kGreedy = 0,
  kRandom = 1,
  kRandomSeed = 2,
};

// RequestOutputKind (Enum): controls how much output each RequestOutput
// carries.
enum class RequestOutputKind : int {
  // Return entire output so far in every RequestOutput.
  kCumulative = 0,
  // Return only deltas in each RequestOutput.
  kDelta = 1,
  // Do not return intermediate RequestOutput.
  kFinalOnly = 2,
};

// StructuredOutputsParams (sampling_params.py:72-142 @ e24d1b24). Exactly one
// of json / json_object / regex / choice / grammar / structural_tag selects the
// structured-output constraint; the rest are lowering options. Field names 1:1
// with upstream.
//
// DEVIATIONS, recorded:
//   - upstream `json: str | dict | None` — here std::optional<std::string>: the
//     caller (OpenAI response_format layer, M3.4 Task 5) pre-serializes a dict
//     schema to its JSON string, mirroring upstream's `json.dumps` str branch in
//     get_structured_output_key.
//   - upstream `json_object: bool | None = None` — kept as std::optional<bool>
//     so has_value() faithfully mirrors upstream's `is not None` in the
//     mutual-exclusion count (a bare bool cannot distinguish unset from False).
//   - `_backend` / `_backend_was_auto` are init=False upstream (set only by the
//     Processor); ported as `backend` / `backend_was_auto`.
struct StructuredOutputsParams {
  // One of these selects the constraint (sampling_params.py:73-78).
  std::optional<std::string> json;
  std::optional<std::string> regex;
  std::optional<std::vector<std::string>> choice;
  std::optional<std::string> grammar;
  std::optional<bool> json_object;
  // Other lowering options (sampling_params.py:80-83).
  bool disable_any_whitespace = false;
  bool disable_additional_properties = false;
  std::optional<std::string> whitespace_pattern;
  std::optional<std::string> structural_tag;
  // sampling_params.py:85-88 (_backend / _backend_was_auto): CAUTION — set only
  // by the Processor's structured-output validation.
  std::optional<std::string> backend;
  bool backend_was_auto = false;

  // __post_init__ (sampling_params.py:90-111): exactly one constraint field must
  // be set. Throws std::runtime_error (upstream ValueError) on count != 1. Not
  // run automatically for a plain struct — SamplingParams::PostInit() calls it.
  void Verify() const;

  // all_constraints_none (sampling_params.py:113-127): true iff none of the six
  // constraint fields is set.
  bool all_constraints_none() const;

  // all_non_structural_tag_constraints_none (sampling_params.py:129-142): true
  // iff none of the five non-structural-tag constraint fields is set.
  bool all_non_structural_tag_constraints_none() const;
};

// Sampling parameters for text generation (T0 field subset). Defaults match
// upstream SamplingParams exactly.
struct SamplingParams {
  // Number of outputs to return for the given prompt request.
  int n = 1;
  // Penalizes new tokens based on presence in the generated text so far.
  double presence_penalty = 0.0;
  // Penalizes new tokens based on their frequency in the generated text.
  double frequency_penalty = 0.0;
  // Penalizes new tokens based on presence in prompt + generated text.
  double repetition_penalty = 1.0;
  // Randomness of sampling; 0 means greedy.
  double temperature = 1.0;
  // Cumulative probability of top tokens to consider; in (0, 1].
  double top_p = 1.0;
  // Number of top tokens to consider; 0 (or -1) considers all tokens.
  int top_k = 0;
  // Minimum token probability relative to the most likely token; in [0, 1].
  double min_p = 0.0;
  // Random seed for generation (unset => nondeterministic).
  std::optional<int64_t> seed;
  // Strings that stop generation when produced (already list-form).
  std::vector<std::string> stop;
  // Token IDs that stop generation when produced (already list-form).
  std::vector<int32_t> stop_token_ids;
  // Whether to ignore EOS and keep generating past it.
  bool ignore_eos = false;
  // Maximum number of tokens to generate per output (unset => unbounded).
  std::optional<int> max_tokens = 16;
  // Minimum tokens to generate before EOS / stop_token_ids can end generation.
  int min_tokens = 0;
  // Number of sample logprobs per token; unset => none, -1 => all vocab.
  std::optional<int> logprobs;
  // Number of prompt logprobs per token; unset => none, -1 => all vocab.
  std::optional<int> prompt_logprobs;
  // Whether to detokenize the output.
  bool detokenize = true;
  // Whether to skip special tokens in the output.
  bool skip_special_tokens = true;
  // Whether to add spaces between special tokens in the output.
  bool spaces_between_special_tokens = true;
  // Whether to include the stop strings in the output text.
  bool include_stop_str_in_output = false;
  // How much output each RequestOutput carries.
  RequestOutputKind output_kind = RequestOutputKind::kCumulative;

  // structured_outputs (SamplingParams.structured_outputs @ e24d1b24): the
  // structured-output / guided-decoding constraint, or unset. PostInit()
  // validates it (mutual exclusion) when present.
  std::optional<StructuredOutputsParams> structured_outputs;

  // The model's EOS token id. Upstream: `_eos_token_id` — a non-init field set
  // engine-side by update_from_generation_config, exposed via the read-only
  // `eos_token_id` property (which is what check_stop reads). It is NOT a
  // user-facing sampling knob: the engine populates it before the params reach
  // the scheduler, and Verify() must not validate it (upstream doesn't).
  std::optional<int32_t> eos_token_id;

  // sampling_type (cached_property): greedy when temperature < _SAMPLING_EPS,
  // random_seed when a seed is set, else random.
  SamplingType Type() const;

  // _verify_args: pure validation only. Throws std::runtime_error with the
  // upstream-equivalent message on any invalid field. const (no mutation).
  // NOTE: this is NOT sufficient on its own — it does not normalize fields nor
  // run the greedy n-check, so it accepts states upstream rejects at
  // construction (e.g. temperature=0 with n=2). Callers that build a
  // SamplingParams for the engine must call PostInit(), not Verify() alone.
  void Verify() const;

  // __post_init__ equivalent: normalize in place (clamp near-zero temperature,
  // drop seed == -1, force greedy sub-params when greedy), then run Verify()
  // and the greedy n-check. Upstream ALWAYS runs __post_init__ at construction,
  // so this is MANDATORY: every SamplingParams that enters the engine must have
  // PostInit() called on it (the InputProcessor / EngineCoreRequest
  // construction path in M1.8 does this). Verify() alone is NOT a substitute —
  // it neither normalizes fields nor enforces the greedy n==1 rule, so a caller
  // using Verify() by itself would accept invalid states upstream rejects.
  void PostInit();

 private:
  // _verify_greedy_sampling: n must be 1 under greedy sampling.
  void VerifyGreedySampling() const;
};

}  // namespace vllm
