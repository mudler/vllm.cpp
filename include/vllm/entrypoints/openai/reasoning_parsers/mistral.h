// Ported from: vllm/reasoning/mistral_reasoning_parser.py @ e24d1b24
// (name "mistral"). Mistral (Magistral) wraps reasoning in [THINK]...[/THINK].
// A valid trace starts with [THINK]; a bare [/THINK] with no preceding [THINK]
// is treated as ordinary content (the markers are stripped but nothing is
// classified as reasoning).
//
// DEVIATION: upstream binds the [THINK]/[/THINK] ids from the MistralTokenizer's
// special-token table and requires a MistralTokenizer instance. The text-only
// seam hard-codes the marker strings and drops the tokenizer type check.
#pragma once

#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/basic.h"

namespace vllm::entrypoints::openai {

class MistralReasoningParser final : public BaseThinkingReasoningParser {
 public:
  const std::string& start_token() const override { return start_; }
  const std::string& end_token() const override { return end_; }

  // mistral_reasoning_parser.py:120 (custom BOT/EOT partition). Streaming reuses
  // the base engine (mistral does not override it).
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // mistral_reasoning_parser.py:65 - reasoning ends only when an [/THINK]
  // follows the last [THINK] (a start marker is required).
  bool is_reasoning_end(const std::string& text) const override;

 private:
  const std::string start_ = "[THINK]";
  const std::string end_ = "[/THINK]";
};

}  // namespace vllm::entrypoints::openai
