// Ported from: vllm/tool_parsers/hermes_tool_parser.py @ e24d1b24
//
// Hermes2ProToolParser — the `<tool_call>{json}</tool_call>` format. This is
// the format the gate model Qwen3.6-35B (qwen35moe) emits (docs/features/
// tool_calling.md:317 maps Qwen non-Coder models to `hermes`). Ports both the
// NON-STREAMING extract_tool_calls (Task 2) and the STREAMING incremental parse
// extract_tool_calls_streaming (Task 3).
#pragma once

#include <optional>
#include <regex>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class HermesToolParser : public ToolParser {
 public:
  HermesToolParser() = default;

  // hermes_tool_parser.py:70 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // hermes_tool_parser.py:204 (extract_tool_calls_streaming). Re-parses the full
  // current_text each call, diffing against the sent state to emit only new
  // content / tool names / argument fragments.
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // hermes_tool_parser.py:36-37 — the tool-call marker tokens.
  static constexpr const char* kToolCallStartToken = "<tool_call>";
  static constexpr const char* kToolCallEndToken = "</tool_call>";

 protected:
  // The wrapper tokens + the two-capture extraction regex, exposed as virtuals
  // so a subclass that only swaps the wrapper tag (LongcatFlashToolParser,
  // longcat_tool_parser.py:11 — `<longcat_tool_call>`) can reuse ALL of the
  // Hermes extraction/streaming logic verbatim (the C++ analogue of the Python
  // subclass overriding self.tool_call_{start,end}_token + self.tool_call_regex
  // in __init__). The base implementations return the Hermes `<tool_call>` /
  // `</tool_call>` tokens and their regex.
  virtual const std::string& tool_call_start() const;
  virtual const std::string& tool_call_end() const;
  virtual const std::regex& tool_call_pattern() const;

 private:
  // hermes_tool_parser.py:57 — how much non-tool-call content has been sent.
  std::size_t sent_content_idx_ = 0;
};

}  // namespace vllm::entrypoints::openai
