// Ported from: vllm/tool_parsers/ernie45_tool_parser.py @ e24d1b24
//
// Ernie45ToolParser (name "ernie45") - for Baidu ERNIE-4.5 "thinking" MoE models.
// Format (one or more blocks):
//   <tool_call>\n{"name":..,"arguments":{..}}\n</tool_call>
// optionally preceded by a </think> reasoning close and/or a <response>...
// </response> span. Ports the NON-STREAMING extract_tool_calls (regex findall
// over <tool_call>...</tool_call>) and the STREAMING incremental parse (a buffer
// that flushes one complete <tool_call> block at a time).
//
// MAJOR DEVIATION (vocab -> text rework, REQUIRED by the text-only seam):
//   Upstream is VOCAB-AWARE: __init__ resolves think/response/newline SPECIAL
//   TOKEN IDS from the tokenizer, and the streaming parse gates on those ids
//   (previous_token_ids / delta_token_ids) to (a) strip a leading "\n" that a
//   tokenizer emits right after </think> | <response> | </response>, and (b) peel
//   a <response>...</response> wrapper off plain content. Our seam has neither a
//   tokenizer nor token-id spans. We therefore REWORK every token-id test to the
//   equivalent TEXT test:
//     - "newline_token_id in delta_token_ids" -> delta_text starts with "\n";
//     - "prev token in {</think>,<response>,</response>}" -> previous_text ends
//       with one of those literals;
//     - "response_start/end_token_id in delta_token_ids" -> the literal
//       <response> | </response> occurs in delta_text.
//   The TOOL-CALL extraction path (the load-bearing behaviour the ported tests
//   check) is already pure text and ports verbatim. The content-cleanup rework is
//   documented and covered by added edge tests; it is a best-effort text analogue
//   of the tokenizer-boundary cleanup, not a token-exact reproduction.
//
// Other deviations: ctor drops the tokenizer/vocab wiring; streaming signature
// drops the token-id spans; tool-call ids use make_tool_call_id().
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Ernie45ToolParser : public ToolParser {
 public:
  Ernie45ToolParser() = default;

  // ernie45_tool_parser.py:72 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // ernie45_tool_parser.py:118 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // ernie45_tool_parser.py:40-46 - the marker literals.
  static constexpr const char* kThinkEnd = "</think>";
  static constexpr const char* kResponseStart = "<response>";
  static constexpr const char* kResponseEnd = "</response>";
  static constexpr const char* kToolCallStart = "<tool_call>";
  static constexpr const char* kToolCallEnd = "</tool_call>";

 private:
  // Streaming state (ernie45_tool_parser.py:37-70).
  std::vector<nlohmann::ordered_json> ernie_prev_tool_call_arr_;
  int ernie_current_tool_id_ = -1;
  std::vector<std::string> ernie_streamed_args_for_tool_;
  std::string buffer_;
};

}  // namespace vllm::entrypoints::openai
