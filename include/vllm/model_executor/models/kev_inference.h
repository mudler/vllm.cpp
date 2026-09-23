// kev decision inference entry point (MODEL-KEV Phase 5c).
//
// Full decision pipeline: resolve Qwen special tokens, tokenize the
// state/instructions/options, encode the question as a causal row, forward
// the Qwen3.5 dense backbone to post-final-norm hidden states
// (ForwardDenseHidden), extract the <decide> and per-option <box_end> hidden
// states, apply the PointerHead readout, and return per-option logits
// (pre-softmax, post-temperature). Exposed so the /v1/systemone decision
// callback can reach the kev capability without going through the
// text-generation/embedding forward.
//
// This is the kev analogue of laya_inference.h: where Laya returns decision
// scores from a ModernBERT encoder + Laya head, kev returns PointerHead
// logits from a Qwen3.5 dense backbone. Both back the same /v1/systemone API
// through the DecisionFn callback.
//
// The device queue is NOT a parameter: it is stored on the KevLoadedModel
// during PrepareKev (the ModelFactory prepare callback), so KevInference
// retrieves it from the model. This avoids exposing the queue through any
// shared header (runner.h or model_loader.h) and keeps the change inside
// kev-owned files.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

class LoadedModel;
namespace tok {
class Tokenizer;
}

// Result of one kev decision forward pass.
struct KevDecisionResult {
  std::vector<float> scores;       // per-option logits (pre-softmax, post-temp)
  std::vector<float> act_logits;   // empty — kev has no escalation head
  int64_t prompt_tokens = 0;
};

// Run kev decision inference for one question.
//
//   model:         a Kev-loaded model (checked via ModelAs)
//   tokenizer:     a loaded Qwen3.5 tokenizer
//   state:         text to analyze (the "state" in the sequence)
//   qtype:         "choice", "score", or "noul"
//   instructions:  question instructions (placed after <|fim_middle|>)
//   options:       option texts (each wrapped in <|box_start|>/<|box_end|>)
KevDecisionResult KevInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options);

}  // namespace vllm
