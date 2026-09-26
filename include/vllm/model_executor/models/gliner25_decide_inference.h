// GLiNER2.5-Decide inference entry point (MODEL-GLINER25-DECIDE).
//
// Full decision pipeline: tokenize the state + options, build the
// [CLS] text [SEP] [P] task [L] label1 [L] label2 ... [SEP] sequence,
// forward the DeBERTa v2 encoder, extract label embeddings at [L] marker
// positions, run the classification head, and return per-option logits.
//
// Exposed so the /v1/systemone decision callback can reach the
// GLiNER2.5-Decide capability without going through the text-generation
// or embedding forward.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {

class LoadedModel;
namespace tok { class Tokenizer; }

// Result of a GLiNER2.5-Decide inference call.
struct Gliner25DecideResult {
  std::vector<float> scores;     // pre-softmax per-option logits
  int64_t prompt_tokens = 0;     // total tokens in the sequence
};

Gliner25DecideResult Gliner25DecideInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options);

}  // namespace vllm
