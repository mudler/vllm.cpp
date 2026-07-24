// Ported from: vllm/tool_parsers/granite_tool_parser.py @ e24d1b24
//
// GraniteToolParser (name "granite") - for the IBM Granite 3.0 / 3.1 models
// (examples/tool_chat_template_granite.jinja). Format: an OPTIONAL `<|tool_call|>`
// token (Granite 3.0) or `<tool_call>` string (Granite 3.1) prefix, then a JSON
// LIST of {"name","arguments"} objects. Ports the NON-STREAMING extract_tool_calls
// and the STREAMING extract_tool_calls_streaming (the partial-JSON incremental
// parse; the argument-diff engine lives in granite_stream_emit in utils).
//
// DEVIATIONS from upstream (see also utils.h):
//   - upstream's streaming signature takes previous/current/delta TOKEN-ID spans;
//     the granite streaming parse is TEXT-only (never reads token ids), so those
//     three params are dropped to keep the serving seam small (same deviation the
//     abstract/hermes ports already document).
//   - partial_json_parser -> our hand-written partial_json_loads (utils.cpp).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class GraniteToolParser : public ToolParser {
 public:
  GraniteToolParser() = default;

  // granite_tool_parser.py:55 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // granite_tool_parser.py:102 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // granite_tool_parser.py:51-53 - the two accepted tool-call markers.
  static constexpr const char* kBotToken = "<|tool_call|>";   // Granite 3.0
  static constexpr const char* kBotString = "<tool_call>";    // Granite 3.1

 private:
  // granite_tool_parser.py:248 (self.prev_tool_call_arr). The base carries a
  // std::vector<nlohmann::json>, but the granite streaming diff needs INSERTION-
  // ordered objects (see utils.h), so we keep our own ordered_json history here.
  std::vector<nlohmann::ordered_json> granite_prev_tool_call_arr_;
};

}  // namespace vllm::entrypoints::openai
