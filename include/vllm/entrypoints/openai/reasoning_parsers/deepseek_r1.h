// Ported from: vllm/reasoning/deepseek_r1_reasoning_parser.py @ e24d1b24
// (name "deepseek_r1"). DeepSeek R1 wraps reasoning in <think>...</think>. The
// start marker is frequently hard-coded by the chat template and NOT emitted by
// the model, so the streaming override treats a leading (start-less) span as
// reasoning up to the first </think>.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/basic.h"

namespace vllm::entrypoints::openai {

class DeepSeekR1ReasoningParser final : public BaseThinkingReasoningParser {
 public:
  const std::string& start_token() const override { return start_; }
  const std::string& end_token() const override { return end_; }

  // deepseek_r1_reasoning_parser.py:28 - handles the start-marker-less stream.
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  const std::string start_ = "<think>";
  const std::string end_ = "</think>";
};

}  // namespace vllm::entrypoints::openai
