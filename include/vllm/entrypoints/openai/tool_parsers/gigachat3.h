// Ported from: vllm/tool_parsers/gigachat3_tool_parser.py @ e24d1b24
//
// GigaChat3ToolParser (name "gigachat3") - for Sber GigaChat 3 / 3.1. Two header
// forms introduce the single function call:
//   "function call<|role_sep|>\n{json}"   (GigaChat 3, after a <|message_sep|>)
//   "<|function_call|>{json}"             (GigaChat 3.1)
// where {json} == {"name":..,"arguments":{..}} and generation may end with "</s>".
// A single tool call per turn (index 0 only). Ports the NON-STREAMING
// extract_tool_calls and the STREAMING incremental parse (name-first, then a
// startswith-prefix argument diff).
//
// DEVIATIONS from upstream shape (all documented):
//   - Upstream __init__ takes a tokenizer (only used by adjust_request to force
//     skip_special_tokens=False). The text seam surfaces the header/sep literals
//     unconditionally, so the ctor drops the tokenizer + adjust_request hook.
//   - The streaming signature drops the previous/current/delta TOKEN-ID spans -
//     the upstream gigachat3 streaming parse ALREADY ignores every token-id
//     argument (it reads only current_text + delta_text), so this is a pure
//     signature trim with NO behavioural rework.
//   - Tool-call id uses make_tool_call_id() (upstream already calls the same
//     helper).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class GigaChat3ToolParser : public ToolParser {
 public:
  GigaChat3ToolParser() = default;

  // gigachat3_tool_parser.py:67 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // gigachat3_tool_parser.py:121 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // Streaming state (gigachat3_tool_parser.py:52-57).
  bool tool_started_ = false;
  bool tool_name_sent_ = false;
  std::optional<std::string> tool_id_;
  bool end_content_ = false;
  // prev_tool_call_arr[0]["arguments_str"] - the last-seen raw argument text.
  std::string prev_arguments_str_;
  bool prev_arguments_str_set_ = false;
  std::string prev_name_;
  std::vector<std::string> streamed_args_for_tool_;
};

}  // namespace vllm::entrypoints::openai
