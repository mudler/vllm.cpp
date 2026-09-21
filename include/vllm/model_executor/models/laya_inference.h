// Laya decision inference entry point (MODEL-LAYA Phase 4).
//
// Full decision pipeline: build the input sequence, run the ModernBERT encoder,
// run the Laya decision head (type_emb + head layers + scorer + act_head),
// and return per-option scores + escalate logits. Exposed so the server
// endpoint (/v1/systemone) can reach the decision capability without going
// through the text-generation/embedding forward.
//
// This is the decision-model analogue of gliner2_ner.h: where GLiNER2.5
// returns extracted entities, Laya returns decision scores and act_logits.
// Both back the same /v1/systemone API — the server routes to the correct
// callback based on the resolved architecture.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

class LoadedModel;
namespace tok {
class Tokenizer;
}

// Result of one Laya decision forward pass.
struct LayaDecisionResult {
  std::vector<float> scores;       // per-option raw logits (from scorer)
  std::vector<float> act_logits;   // [n_act] escalate decision
  int64_t prompt_tokens = 0;
};

// Run Laya decision inference for one question.
//
//   model:         a Laya-loaded model (checked via ModelAs)
//   tokenizer:     a loaded BPE tokenizer (ModernBERT/GPT-2 style)
//   state:         text to analyze (the "state" in the sequence)
//   qtype:         "choice", "score", or "noul"
//   instructions:  question instructions (placed in the head text)
//   options:       option texts (each prefixed with a MASK marker)
//   max_len:       total sequence cap (512 for the full model)
//   head_max_len:  head budget for type+instructions+options (192)
LayaDecisionResult LayaInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options,
    int max_len = 512,
    int head_max_len = 192);

}  // namespace vllm
