// Ported from: vllm/tool_parsers/granite_20b_fc_tool_parser.py @ e24d1b24
//
// Granite20bFCToolParser (name "granite-20b-fc") - for the
// granite-20b-functioncalling model (examples/tool_chat_template_granite20b_fc.
// jinja). Format: one or more `<function_call> {json}` segments, each a bare
// {"name","arguments"} object (NOT wrapped in an array). Ports the NON-STREAMING
// extract_tool_calls (JSONDecoder.raw_decode over each segment) and the STREAMING
// extract_tool_calls_streaming (partial-JSON loop; argument-diff engine shared
// with granite via granite_stream_emit).
//
// DEVIATIONS from upstream (see also utils.h):
//   - streaming TOKEN-ID span params dropped (text-only), per the abstract/hermes
//     convention.
//   - regex `<function_call>\s*` -> literal find + consume_space (the token has
//     no regex metacharacters).
//   - JSONDecoder.raw_decode / partial_json_parser -> nlohmann parse of the
//     bounded segment (non-streaming) and partial_json_loads (streaming).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Granite20bFCToolParser : public ToolParser {
 public:
  Granite20bFCToolParser() = default;

  // granite_20b_fc_tool_parser.py:57 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // granite_20b_fc_tool_parser.py:115 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // granite_20b_fc_tool_parser.py:53 - the single start marker.
  static constexpr const char* kBotToken = "<function_call>";

 private:
  // See GraniteToolParser: INSERTION-ordered streaming history (utils.h).
  std::vector<nlohmann::ordered_json> granite_prev_tool_call_arr_;
};

}  // namespace vllm::entrypoints::openai
