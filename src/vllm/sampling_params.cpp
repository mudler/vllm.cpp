// Ported from: vllm/sampling_params.py @ e24d1b24
// See include/vllm/sampling_params.h for scope, deferrals and deviations.
#include "vllm/sampling_params.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vllm {
namespace {

// Formats a double the way the upstream f-strings do for the common cases
// (e.g. 0.5 -> "0.5", 2.0 -> "2"): enough for message parity without dragging
// in Python's repr algorithm.
std::string Fmt(double v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

}  // namespace

SamplingType SamplingParams::Type() const {
  if (temperature < kSamplingEps) {
    return SamplingType::kGreedy;
  }
  if (seed.has_value()) {
    return SamplingType::kRandomSeed;
  }
  return SamplingType::kRandom;
}

void SamplingParams::Verify() const {
  if (n < 1) {
    throw std::runtime_error("n must be at least 1, got " + std::to_string(n) +
                             ".");
  }
  if (n > kMaxNSequences) {
    throw std::runtime_error(
        "n must be at most " + std::to_string(kMaxNSequences) + ", got " +
        std::to_string(n) +
        ". To increase this limit, set the VLLM_MAX_N_SEQUENCES "
        "environment variable.");
  }
  if (!(-2.0 <= presence_penalty && presence_penalty <= 2.0)) {
    throw std::runtime_error("presence_penalty must be in [-2, 2], got " +
                             Fmt(presence_penalty) + ".");
  }
  if (!(-2.0 <= frequency_penalty && frequency_penalty <= 2.0)) {
    throw std::runtime_error("frequency_penalty must be in [-2, 2], got " +
                             Fmt(frequency_penalty) + ".");
  }
  if (!std::isfinite(repetition_penalty)) {
    throw std::runtime_error(
        "repetition_penalty must be a finite number, got " +
        Fmt(repetition_penalty) + ".");
  }
  if (repetition_penalty <= 0.0) {
    throw std::runtime_error(
        "repetition_penalty must be greater than zero, got " +
        Fmt(repetition_penalty) + ".");
  }
  if (!std::isfinite(temperature)) {
    throw std::runtime_error("temperature must be a finite number, got " +
                             Fmt(temperature) + ".");
  }
  if (temperature < 0.0) {
    throw std::runtime_error("temperature must be non-negative, got " +
                             Fmt(temperature) + ".");
  }
  if (temperature > 2.0) {
    throw std::runtime_error("temperature must be in [0, 2], got " +
                             Fmt(temperature) + ".");
  }
  if (!(0.0 < top_p && top_p <= 1.0)) {
    throw std::runtime_error("top_p must be in (0, 1], got " + Fmt(top_p) +
                             ".");
  }
  // quietly accept -1 as disabled, but prefer 0
  if (top_k < -1) {
    throw std::runtime_error("top_k must be 0 (disable), or at least 1, got " +
                             std::to_string(top_k) + ".");
  }
  if (!(0.0 <= min_p && min_p <= 1.0)) {
    throw std::runtime_error("min_p must be in [0, 1], got " + Fmt(min_p) +
                             ".");
  }
  if (max_tokens.has_value() && *max_tokens < 1) {
    throw std::runtime_error("max_tokens must be at least 1, got " +
                             std::to_string(*max_tokens) + ".");
  }
  if (min_tokens < 0) {
    throw std::runtime_error(
        "min_tokens must be greater than or equal to 0, got " +
        std::to_string(min_tokens) + ".");
  }
  if (max_tokens.has_value() && min_tokens > *max_tokens) {
    throw std::runtime_error(
        "min_tokens must be less than or equal to max_tokens=" +
        std::to_string(*max_tokens) + ", got " + std::to_string(min_tokens) +
        ".");
  }
  if (logprobs.has_value() && *logprobs != -1 && *logprobs < 0) {
    throw std::runtime_error("logprobs must be non-negative or -1, got " +
                             std::to_string(*logprobs) + ".");
  }
  if (prompt_logprobs.has_value() && *prompt_logprobs != -1 &&
      *prompt_logprobs < 0) {
    throw std::runtime_error(
        "prompt_logprobs must be non-negative or -1, got " +
        std::to_string(*prompt_logprobs) + ".");
  }
  // stop_token_ids element type is enforced by std::vector<int32_t>.
  for (const std::string& stop_str : stop) {
    if (stop_str.empty()) {
      throw std::runtime_error("stop cannot contain an empty string.");
    }
  }
  if (!stop.empty() && !detokenize) {
    throw std::runtime_error(
        "stop strings are only supported when detokenize is True. "
        "Set detokenize=True to use stop.");
  }
}

void SamplingParams::VerifyGreedySampling() const {
  if (n > 1) {
    throw std::runtime_error("n must be 1 when using greedy sampling, got " +
                             std::to_string(n) + ".");
  }
}

void StructuredOutputsParams::Verify() const {
  // __post_init__ (sampling_params.py:90-111): exactly one constraint field.
  const int count = (json.has_value() ? 1 : 0) + (regex.has_value() ? 1 : 0) +
                    (choice.has_value() ? 1 : 0) +
                    (grammar.has_value() ? 1 : 0) +
                    (json_object.has_value() ? 1 : 0) +
                    (structural_tag.has_value() ? 1 : 0);
  if (count > 1) {
    throw std::runtime_error(
        "You can only use one kind of structured outputs constraint but "
        "multiple are specified.");
  }
  if (count < 1) {
    throw std::runtime_error(
        "You must use one kind of structured outputs constraint but none are "
        "specified.");
  }
}

bool StructuredOutputsParams::all_constraints_none() const {
  // sampling_params.py:113-127.
  return !json.has_value() && !regex.has_value() && !choice.has_value() &&
         !grammar.has_value() && !json_object.has_value() &&
         !structural_tag.has_value();
}

bool StructuredOutputsParams::all_non_structural_tag_constraints_none() const {
  // sampling_params.py:129-142.
  return !json.has_value() && !regex.has_value() && !choice.has_value() &&
         !grammar.has_value() && !json_object.has_value();
}

void SamplingParams::PostInit() {
  if (structured_outputs.has_value()) {
    // Upstream runs StructuredOutputsParams.__post_init__ at its construction;
    // as a plain struct here it is validated at the enclosing PostInit().
    structured_outputs->Verify();
  }
  if (temperature > 0.0 && temperature < kMaxTemp) {
    // Maxed out to _MAX_TEMP to avoid nan/inf in downstream tensors.
    temperature = std::max(temperature, kMaxTemp);
  }

  if (seed.has_value() && *seed == -1) {
    seed.reset();
  }

  Verify();

  if (temperature < kSamplingEps) {
    // Zero temperature means greedy sampling.
    top_p = 1.0;
    top_k = 0;
    min_p = 0.0;
    VerifyGreedySampling();
  }
}

}  // namespace vllm
