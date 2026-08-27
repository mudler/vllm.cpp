// Ported from: vllm/sampling_params.py @ e24d1b24
//
// Scope: the T0 subset the V1 engine actually passes around (see the M1.1
// engine-core-types plan). Field names, defaults, enum int values, the
// sampling_type derivation and the _verify_args / __post_init__ semantics are
// mirrored 1:1 with upstream; only Python-specific concerns are dropped.
//
// WIRED (ROAD-V1-C7, `SAMPLE-LOGIT-FILTERS`): logit_bias, allowed_token_ids,
// bad_words + bad_words_token_ids (upstream `_bad_words_token_ids`) and the
// all_stop_token_ids set (upstream `_all_stop_token_ids`) are now ported below —
// they feed the already-implemented sampler logits processors via the
// InputBatch::build_sampling_metadata wiring. logit_bias clamp to [-100, 100] is
// applied at the request-construction boundary (from_optional's role, ported in
// entrypoints/openai/protocol.cpp), mirroring sampling_params.py:388-413.
//
// DEFERRED (T1/T2) upstream fields, intentionally omitted here — a future
// porter slots them in without reshaping the struct:
//   (extra_args is now ported below, as a STRING-VALUED slice — KV-EVENTS W3)
//   (structured_outputs (StructuredOutputsParams) is now ported below — M3.4)
//   - flat_logprobs
//   (logprob_token_ids + num_logprobs() are now ported below —
//    `SAMPLE-LOGPROB-TOKEN-IDS`, specs/logprob-token-ids.md. Their vocab-range
//    validation stays engine-time, like allowed_token_ids'.)
//   - thinking_token_budget, repetition_detection (RepetitionDetectionParams),
//     routed_experts_prompt_start, skip_reading_prefix_cache
//   - skip_clone / clone(), for_sampler_warmup()
//   - internal post-init state: output_text_buffer_length
//   - engine-time helpers: from_optional(), verify(model_config, ...) and its
//     _validate_* family, the eos_token_id / all_stop_token_ids /
//     bad_words_token_ids properties (update_from_generation_config /
//     update_from_tokenizer are ported on InputProcessor, M1.8)
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
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "vllm/logits_processor_callback.h"

namespace vllm {

// sampling_params._SAMPLING_EPS: below this temperature, sampling is greedy.
inline constexpr double kSamplingEps = 1e-5;
// sampling_params._MAX_TEMP: temperatures in (0, _MAX_TEMP) are clamped up to
// avoid nan/inf in downstream tensors.
inline constexpr double kMaxTemp = 1e-2;
// envs.VLLM_MAX_N_SEQUENCES default (upper bound on `n`).
inline constexpr int kMaxNSequences = 16384;
// sampling_params.MAX_LOGPROB_TOKEN_IDS (sampling_params.py:31-33): upper bound
// on the `logprob_token_ids` list length. Upstream ties it to the per-request
// row width its V2 sampler's LogprobTokenIdsState allocates; our V1-shaped
// sampler builds the row per step, so the bound is carried purely to keep the
// accepted request surface identical.
inline constexpr int kMaxLogprobTokenIds = 128;

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
// Ported from: the dict vllm/config/model.py::ModelConfig.get_diff_sampling_param
// returns, and that vllm/entrypoints/openai/*/serving.py stores as
// `self.default_sampling_params`.
//
// The SERVER-WIDE sampling defaults a checkpoint's own generation_config.json
// asks for. Every field is optional because "the checkpoint declared nothing"
// and "the checkpoint declared the neutral value" resolve differently: an unset
// field falls through to the OpenAI neutral default, a set one does not. That
// distinction is the whole rule, so it is carried in the type rather than in a
// sentinel.
//
// `max_tokens` is already renamed from the file's `max_new_tokens`, exactly
// where upstream renames it (get_diff_sampling_param, "Huggingface definition
// of max_new_tokens is equivalent to vLLM's max_tokens").
struct DefaultSamplingParams {
  std::optional<double> repetition_penalty;
  std::optional<double> temperature;
  std::optional<int> top_k;
  std::optional<double> top_p;
  std::optional<double> min_p;
  std::optional<int> max_tokens;

  bool empty() const {
    return !repetition_penalty.has_value() && !temperature.has_value() &&
           !top_k.has_value() && !top_p.has_value() && !min_p.has_value() &&
           !max_tokens.has_value();
  }

  // The set fields as `{key: value, ...}`, for the startup line upstream logs
  // ("Default vLLM sampling parameters have been overridden by ..."). Empty
  // string when nothing is set.
  std::string ToString() const;
};

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
  // logprob_token_ids (sampling_params.py:278-283): score an EXPLICIT set of
  // vocab ids instead of the top-k — "more efficient than logprobs=-1 when you
  // only need logprobs for a small set of tokens". vLLM's generative-scoring
  // endpoint drives it (generative_scoring/serving.py:247-255: max_tokens=1 +
  // logprob_token_ids=label_token_ids). The sampler returns the sampled token
  // plus exactly these ids; when `logprobs` is ALSO set the explicit ids WIN
  // (sampler.py:133-136). Unset => the ordinary top-k path, byte-identical.
  std::optional<std::vector<int32_t>> logprob_token_ids;
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

  // stream_interval (sampling_params.py:302 @ vllm#49754): per-request override
  // of the engine-level `--stream-interval` — the number of newly generated
  // tokens to batch into each streamed RequestOutput. Only RAISES the interval
  // above the engine setting (values below it are clamped up, see
  // output_processor RequestState::FromNewRequest). The first and final outputs
  // are always emitted immediately. Unset => use the engine interval unchanged.
  std::optional<int> stream_interval;

  // structured_outputs (SamplingParams.structured_outputs @ e24d1b24): the
  // structured-output / guided-decoding constraint, or unset. PostInit()
  // validates it (mutual exclusion) when present.
  std::optional<StructuredOutputsParams> structured_outputs;

  // logit_bias (sampling_params.py:318): token_id -> additive bias. When set,
  // the sampler adds the bias to each listed token's logit. Callers building
  // from an OpenAI request MUST have already converted string keys to int and
  // clamped each bias to [-100, 100] (sampling_params.py:388-413 / from_optional).
  std::optional<std::map<int32_t, float>> logit_bias;

  // allowed_token_ids (sampling_params.py:321): if set, only these token ids
  // keep their logits (all others are masked to -inf). Upstream requires a
  // non-empty list of in-range ids (validated in verify(), model-config-aware —
  // the vocab-range check is engine-time and stays deferred; the empty-list
  // check is enforced in Verify()).
  std::optional<std::vector<int32_t>> allowed_token_ids;

  // bad_words (sampling_params.py:337): words never to generate (only the final
  // token of a matching token sequence is blocked). Always list-form (upstream
  // __post_init__ normalizes None -> []). Tokenized engine-side into
  // bad_words_token_ids by InputProcessor::UpdateFromTokenizer
  // (update_from_tokenizer, sampling_params.py:659).
  std::vector<std::string> bad_words;
  // _bad_words_token_ids (sampling_params.py:341): the per-word token-id n-grams
  // update_from_tokenizer produces; unset until the tokenizer pass runs. The
  // sampler's apply_bad_words consumes these.
  std::optional<std::vector<std::vector<int32_t>>> bad_words_token_ids;

  // _all_stop_token_ids (sampling_params.py:313): the union of stop_token_ids
  // and (engine-time) the eos id(s), used by the MinTokens logits processor to
  // know which tokens to mask while output_len < min_tokens. PostInit() seeds it
  // from stop_token_ids (sampling_params.py:500); the engine adds eos ids via
  // InputProcessor::UpdateFromGenerationConfig.
  std::set<int32_t> all_stop_token_ids;

  // Custom logits processor (ROAD-V1-C7 `custom_logit_processor`). A single host
  // callback invoked per decode step, before sampling, to modify THIS request's
  // logits (see logits_processor_callback.h). Recorded deviation: vLLM carries a
  // `logits_processors` plugin list (sampling_params.py) and SGLang a per-request
  // Python callable (custom_logit_processor.py:24); we carry one C-ABI function
  // pointer + user_data (the C ABI's `vllm_logits_processor`). fn == nullptr (the
  // default) => no processor, the byte-identical sampler path. Not validated by
  // Verify() (an opaque host callback has no upstream validation analogue).
  LogitsProcessorCallback logits_processor;

  // The model's EOS token id. Upstream: `_eos_token_id` — a non-init field set
  // engine-side by update_from_generation_config, exposed via the read-only
  // `eos_token_id` property (which is what check_stop reads). It is NOT a
  // user-facing sampling knob: the engine populates it before the params reach
  // the scheduler, and Verify() must not validate it (upstream doesn't).
  std::optional<int32_t> eos_token_id;

  // extra_args (sampling_params.py `extra_args: dict[str, Any] | None`): the
  // out-of-band per-request knobs the engine reads but the sampler never does.
  // Upstream reads three keys out of it in Request.__init__
  // (vllm/v1/request.py:116-127): kv_transfer_params, ec_transfer_params, and
  // kv_cache_report_mode.
  //
  // RECORDED RESTRICTION: this is a STRING-VALUED slice of upstream's
  // `dict[str, Any]`. Only kv_cache_report_mode (a plain string, KV-EVENTS W3)
  // is consumed today; the other two keys are nested dicts belonging to
  // KV-CONNECTORS / EC transfer, which are deferred with their own rows and
  // would dictate a different value shape. Modelling the ported slice as
  // map<string,string> matches the precedent already in the tree
  // (include/vllm/v1/kv_offload/kv_connector.h:109 models kv_transfer_params
  // the same way). nullopt (the default) is upstream's `None` and leaves every
  // consumer on its default -> byte-identical.
  //
  // The HTTP door to this (`vllm_xargs` on the OpenAI request) is still
  // DEFERRED — see include/vllm/entrypoints/openai/protocol.h — so today this
  // is settable from the C++ engine API only. Not validated by Verify()
  // (upstream does not validate extra_args either).
  std::optional<std::map<std::string, std::string>> extra_args;

  // sampling_type (cached_property): greedy when temperature < _SAMPLING_EPS,
  // random_seed when a seed is set, else random.
  SamplingType Type() const;

  // num_logprobs (property, sampling_params.py:724-729): the number of sample
  // logprobs to return per output token, or unset when none were requested.
  // `logprobs` if set, ELSE len(logprob_token_ids) if set, else unset. This —
  // not the raw `logprobs` field — is what the scheduler's slice gate
  // (scheduler.py:1818) and the LogprobsProcessor (logprobs.py) read.
  std::optional<int> num_logprobs() const;

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
