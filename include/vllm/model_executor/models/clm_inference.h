// CLM decision inference entry point (MODEL-CLM).
//
// Full decision pipeline: tokenize the state and each option text separately,
// forward the Qwen3 dense backbone (ForwardHidden) for each to get last-token
// hidden states, apply the state-head and action-head MLPs (L2-normalized),
// and return per-option pre-softmax logits via scaled cosine similarity.
// Exposed so the /v1/systemone decision callback can reach the CLM capability
// without going through the text-generation/embedding forward.
//
// This is the CLM analogue of kev_inference.h: where kev returns PointerHead
// logits from a Qwen3.5 dense backbone via a single causal-row forward, CLM is
// a bi-encoder that forwards state and each option SEPARATELY (N+1 forward
// passes) and scores via dual MLP heads + scaled cosine.
//
// The device queue is stored on the ClmLoadedModel during PrepareClm (the
// ModelFactory prepare callback), so ClmInference retrieves it from the model.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

class LoadedModel;

namespace tok {
class Tokenizer;
}  // namespace tok

// Result of a CLM decision inference call.
struct ClmDecisionResult {
  std::vector<float> scores;     // pre-softmax per-option logits
  int64_t prompt_tokens = 0;     // total tokens forwarded (state + all options)
};

struct ClmDecisionResult ClmInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options);

}  // namespace vllm
