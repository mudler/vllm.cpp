// Ported from: vllm/reasoning/basic_parsers.py @ e24d1b24
// (BaseThinkingReasoningParser). The start_token/end_token think-block engine
// that most reasoning parsers subclass (<think>...</think>,
// [THINK]...[/THINK], <seed:think>...</seed:think>, ...).
//
// TEXT-ONLY (see reasoning_parsers/abstract.h): the upstream token-ID membership
// tests collapse to substring checks over the detokenized text. The "skip a lone
// special-token delta" guard (basic_parsers.py:113) becomes "delta_text equals
// exactly the start/end marker" - the case where a detokenizer surfaces the
// atomic special token on its own.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class BaseThinkingReasoningParser : public ReasoningParser {
 public:
  BaseThinkingReasoningParser() = default;

  // Subclasses supply the block delimiters (basic_parsers.py:30-40).
  virtual const std::string& start_token() const = 0;
  virtual const std::string& end_token() const = 0;

  // basic_parsers.py:157 (extract_reasoning). Strip a leading start marker if
  // present; text up to the first end marker is reasoning, the rest is content.
  // For models that may not emit the start marker, the whole prefix is reasoning.
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // basic_parsers.py:98 (extract_reasoning_streaming).
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // basic_parsers.py:72 (is_reasoning_end) - TEXT form: scanning from the end,
  // the reasoning has ended iff an end marker sits after the last start marker
  // (or an end marker exists with no start marker at all).
  bool is_reasoning_end(const std::string& text) const override;
};

}  // namespace vllm::entrypoints::openai
