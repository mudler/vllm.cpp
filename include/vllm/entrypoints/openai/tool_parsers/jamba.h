// Ported from: vllm/tool_parsers/jamba_tool_parser.py @ e24d1b24
//
// JambaToolParser (name "jamba") - for the AI21 Jamba models. Format:
//   <tool_calls>[{"name":..,"arguments":{..}}, ...]</tool_calls>
// A JSON ARRAY of tool calls wrapped in the special <tool_calls> markers (Jamba
// DOES support parallel calls). Ports the NON-STREAMING extract_tool_calls and
// the STREAMING incremental array parse.
//
// MAJOR DEVIATION (vocab -> text rework):
//   Upstream reads the tokenizer VOCAB (self.tool_calls_start/end_token_id) and
//   inspects delta_token_ids to (a) reject construction when the markers are not
//   in the vocab / a Mistral tokenizer is used, and (b) suppress the lone
//   start-token chunk via `start_token_id in delta_token_ids and len==1`. Our
//   text-only seam has neither a tokenizer nor token-id spans, so:
//     - the constructor drops the vocab/Mistral-tokenizer validation entirely;
//     - the lone-start-token suppression is achieved by the SAME text path that
//       already handles it - when the parsable region after <tool_calls> is empty
//       or only "[" (an empty/partial array), partial parsing yields a
//       zero-length array and we return nullopt. This reproduces upstream's
//       "don't emit for the control token" behaviour WITHOUT reading token ids.
//   The argument-diff still keys off `delta_text` (which the text seam DOES
//   provide), so no token-id span is needed.
//
// Other (shared) deviations: streaming token-id params dropped; upstream
// partial_json_parser.loads -> our partial_json_loads (utils).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class JambaToolParser : public ToolParser {
 public:
  JambaToolParser() = default;

  // jamba_tool_parser.py:83 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // jamba_tool_parser.py:127 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // jamba_tool_parser.py:49-50 - the tool-call wrapper markers.
  static constexpr const char* kToolCallsStart = "<tool_calls>";
  static constexpr const char* kToolCallsEnd = "</tool_calls>";

 private:
  // Insertion-ordered history of the parsed tool-call array (upstream
  // self.prev_tool_call_arr; ordered so serialized argument key order is stable).
  std::vector<nlohmann::ordered_json> jamba_prev_tool_call_arr_;
};

}  // namespace vllm::entrypoints::openai
