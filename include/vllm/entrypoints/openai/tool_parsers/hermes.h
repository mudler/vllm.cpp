// Ported from: vllm/tool_parsers/hermes_tool_parser.py @ e24d1b24
//
// Hermes2ProToolParser — the `<tool_call>{json}</tool_call>` format. This is
// the format the gate model Qwen3.6-35B (qwen35moe) emits (docs/features/
// tool_calling.md:317 maps Qwen non-Coder models to `hermes`). Only the
// NON-STREAMING extract_tool_calls is ported here (Task 2); the streaming
// method + adjust_request are Task 3.
#pragma once

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

  // hermes_tool_parser.py:36-37 — the tool-call marker tokens.
  static constexpr const char* kToolCallStartToken = "<tool_call>";
  static constexpr const char* kToolCallEndToken = "</tool_call>";
};

}  // namespace vllm::entrypoints::openai
