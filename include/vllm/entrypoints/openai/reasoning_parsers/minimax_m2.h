// Ported from: vllm/reasoning/minimax_m2_reasoning_parser.py @ e24d1b24.
//
// Two registered parsers:
//   - "minimax_m2" (MiniMaxM2ReasoningParser): the model emits only the </think>
//     END marker (no <think>); everything before </think> is reasoning, after is
//     content. Behaviourally identical to step3's split, but is_reasoning_end
//     considers the LAST of {<think>, </think>} (upstream:39-45).
//   - "minimax_m2_append_think" (MiniMaxM2AppendThinkReasoningParser): a
//     pass-through that prepends a literal "<think>" and returns EVERYTHING as
//     content (reasoning is always absent). Used when the caller wants the raw
//     transcript with the opening marker restored.
//
// DEVIATION: upstream "minimax_m2" subclasses an engine-based parser adapter
// (MinimaxM2ParserReasoningAdapter, token-ID streaming). The text-only seam
// re-expresses the documented behaviour directly, matching the upstream test
// contract (tests/reasoning/test_minimax_m2_reasoning_parser.py).
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class MiniMaxM2ReasoningParser final : public ReasoningParser {
 public:
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;
  bool is_reasoning_end(const std::string& text) const override;

 private:
  const std::string start_ = "<think>";
  const std::string end_ = "</think>";
};

class MiniMaxM2AppendThinkReasoningParser final : public ReasoningParser {
 public:
  // minimax_m2_reasoning_parser.py:63 - always (None, "<think>" + output).
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;
  // minimax_m2_reasoning_parser.py:50 - prepend "<think>" on the first delta;
  // everything is content.
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;
  bool is_reasoning_end(const std::string& text) const override;

 private:
  const std::string start_ = "<think>";
  const std::string end_ = "</think>";
};

}  // namespace vllm::entrypoints::openai
