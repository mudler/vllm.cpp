// Ported from: vllm/reasoning/step3_reasoning_parser.py @ e24d1b24
// (name "step3"). Step3 marks only the END of reasoning with </think>: all text
// before the first </think> is reasoning, all text after is content. A leading
// <think> is optional and, if present, stays part of the reasoning span (only
// </think> is treated as a marker). Not a BaseThinking subclass - it has no
// start marker to strip.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Step3ReasoningParser final : public ReasoningParser {
 public:
  // step3_reasoning_parser.py:93.
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // step3_reasoning_parser.py:56.
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // step3_reasoning_parser.py:110 - reasoning ends once </think> appears.
  bool is_reasoning_end(const std::string& text) const override;

 private:
  const std::string end_ = "</think>";
};

}  // namespace vllm::entrypoints::openai
