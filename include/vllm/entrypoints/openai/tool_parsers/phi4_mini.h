// Ported from: vllm/tool_parsers/phi4mini_tool_parser.py @ e24d1b24
//
// Phi4MiniJsonToolParser (name "phi4_mini_json") - for the phi-4-mini models
// with the tool_chat_template_llama.jinja template. Format: a `functools`
// prefix followed by a JSON array of {"name","arguments"} objects, e.g.
//   functools[{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
// Only the NON-STREAMING extract_tool_calls is meaningful; upstream's streaming
// path is a stub that returns None (streaming is NOT implemented for this model).
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Phi4MiniJsonToolParser : public ToolParser {
 public:
  Phi4MiniJsonToolParser() = default;

  // phi4mini_tool_parser.py:55 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // phi4mini_tool_parser.py:116 (extract_tool_calls_streaming). Upstream returns
  // None unconditionally (streaming unimplemented); we mirror that with nullopt.
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // phi4mini_tool_parser.py:53 - the bot token that prefixes the JSON array.
  static constexpr const char* kBotToken = "functools";
};

}  // namespace vllm::entrypoints::openai
