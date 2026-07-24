// Ported from: vllm/reasoning/olmo3_reasoning_parser.py @ e24d1b24
// (name "olmo3"). Olmo 3 does NOT use special tokens for reasoning: <think> and
// </think> are ordinary vocabulary text that the pre-tokenizer may split across
// several tokens. The parser therefore works entirely in STRING space, buffering
// delta text until a (possibly split) marker completes. A leading <think> is
// optional - templates that hard-code it mean the parser sees only </think>.
//
// This is the one STATEFUL parser in the batch: the streaming buffer carries the
// partial-marker state across deltas, so ONE instance must live for the whole
// stream. The non-streaming path (extract_reasoning) is a stateless split.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"

namespace vllm::entrypoints::openai {

// olmo3_reasoning_parser.py:76 (Olmo3ReasoningBuffer). The streaming state
// machine: accumulate text, emit reasoning/content deltas as markers complete.
class Olmo3ReasoningBuffer {
 public:
  enum class State { kReasoning, kContent };

  // olmo3:88 (process_buffer).
  std::optional<DeltaMessage> process_buffer();
  // olmo3:140 (add_text).
  std::optional<DeltaMessage> add_text(const std::string& delta_text);

  const std::string& buffer() const { return buffer_; }

  const std::string think_start = "<think>";
  const std::string think_end = "</think>";

 private:
  std::string buffer_;
  // Start in REASONING so hard-coded-<think> templates (only </think> emitted)
  // classify the opening span as reasoning (olmo3:86).
  State state_ = State::kReasoning;
};

class Olmo3ReasoningParser final : public ReasoningParser {
 public:
  // olmo3:271 (regex split: optional <think>, reasoning up to first </think>).
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // olmo3:299 (buffer-driven streaming).
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  bool is_reasoning_end(const std::string& text) const override;

 private:
  Olmo3ReasoningBuffer buffer_;
  const std::string think_start_ = "<think>";
  const std::string think_end_ = "</think>";
};

}  // namespace vllm::entrypoints::openai
