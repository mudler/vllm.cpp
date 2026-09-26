// XOR inference entry point (MODEL-XOR).
//
// Decision pipeline: tokenize state + options into the xor prompt format,
// forward the Qwen3.5 MoE backbone (ForwardMoeHidden), apply lm_head to get
// vocab logits, read the logit at each candidate option's token position,
// run forward+reverse option-order evaluation, calibrate, and return
// per-option scores.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

class LoadedModel;
namespace tok { class Tokenizer; }

struct XorDecisionResult {
  std::vector<float> scores;     // calibrated per-option logits
  int64_t prompt_tokens = 0;
};

XorDecisionResult XorInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options);

}  // namespace vllm
