// cua-s1-forms inference entry point — full pipeline: strings → probabilities.
//
// Wraps ByteCollate + ForwardHost + softmax into a single call that mirrors
// the Python inference path: context + options in, probabilities + winner out.
//
// Not yet wired to the server; that is Phase 4 of the spec.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/cua_s1.h"
#include "vllm/model_executor/models/cua_s1_collator.h"

namespace vllm {
namespace cua_s1 {

struct CuaS1ScoreResult {
  std::vector<float> probabilities;
  int64_t winner = 0;
  float confidence = 0.0F;
  int64_t prompt_tokens = 0;
};

inline CuaS1ScoreResult CuaS1Inference(
    const Params& params, const Weights& weights,
    const std::string& context,
    const std::vector<std::string>& options) {
  CuaS1ScoreResult result;

  const CollatedBatch batch = ByteCollate(params, context, options);
  result.prompt_tokens =
      batch.ctx_len + batch.n_opt * batch.opt_len;

  const std::vector<float> logits = ForwardHost(
      params, weights,
      batch.context_ids, batch.context_mask,
      batch.option_ids, batch.option_tok_mask, batch.option_mask);

  const int64_t n = static_cast<int64_t>(logits.size());
  result.probabilities.resize(static_cast<size_t>(n));

  // Softmax.
  float max_logit = logits.empty() ? 0.0F : logits[0];
  for (float v : logits) {
    if (v > max_logit) max_logit = v;
  }
  float sum = 0.0F;
  for (int64_t i = 0; i < n; ++i) {
    result.probabilities[static_cast<size_t>(i)] =
        std::exp(logits[static_cast<size_t>(i)] - max_logit);
    sum += result.probabilities[static_cast<size_t>(i)];
  }
  if (sum > 0.0F) {
    for (float& p : result.probabilities) p /= sum;
  }

  // Winner = argmax, confidence = max probability.
  result.winner = 0;
  result.confidence = result.probabilities.empty() ? 0.0F : result.probabilities[0];
  for (int64_t i = 1; i < n; ++i) {
    if (result.probabilities[static_cast<size_t>(i)] > result.confidence) {
      result.confidence = result.probabilities[static_cast<size_t>(i)];
      result.winner = i;
    }
  }

  return result;
}

}  // namespace cua_s1

// Production score inference: casts the LoadedModel to its CuaS1LoadedModel
// inner type and runs the full scoring pipeline (ByteCollate + ForwardHost +
// softmax). Declared here so server_main.cpp can call it; defined in
// cua_s1_registry.cpp. LoadedModel is forward-declared to avoid pulling
// model_registry.h into this header.
class LoadedModel;

cua_s1::CuaS1ScoreResult CuaS1ScoreInference(
    const LoadedModel& model,
    const std::string& context,
    const std::vector<std::string>& options);

}  // namespace vllm
