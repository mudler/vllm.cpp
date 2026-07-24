// Ported from: vllm/tool_parsers/granite_20b_fc_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/granite_20b_fc.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

namespace vllm::entrypoints::openai {

ExtractedToolCallInformation Granite20bFCToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string token = kBotToken;

  // granite_20b_fc_tool_parser.py:60-63 - no marker -> plain content.
  if (model_output.find(token) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    // granite_20b_fc_tool_parser.py:67 - all `<function_call>` match positions.
    std::vector<std::size_t> match_starts;
    for (std::size_t p = model_output.find(token); p != std::string::npos;
         p = model_output.find(token, p + token.size())) {
      match_starts.push_back(p);
    }

    // granite_20b_fc_tool_parser.py:70-85 - raw_decode the JSON after each marker
    // up to the next marker (or end).
    std::vector<nlohmann::ordered_json> raw_function_calls;
    for (std::size_t i = 0; i < match_starts.size(); ++i) {
      // regex `<function_call>\s*`: end = marker end + trailing whitespace.
      const std::size_t start_of_json =
          consume_space(match_starts[i] + token.size(), model_output);
      const std::size_t next_start = (i + 1 < match_starts.size())
                                         ? match_starts[i + 1]
                                         : model_output.size();
      const std::string segment =
          model_output.substr(start_of_json, next_start - start_of_json);

      // raw_decode: parse ONE complete value, ignoring trailing data. A truncated
      // value raises (upstream) -> we throw to hit the fallback below.
      auto [value, end_idx] = partial_json_loads(segment, /*allow_partial_str=*/true);
      if (!is_complete_json(segment.substr(0, end_idx))) {
        throw std::runtime_error("incomplete JSON in function call segment");
      }
      raw_function_calls.push_back(std::move(value));
    }

    // granite_20b_fc_tool_parser.py:88-100.
    std::vector<ToolCall> tool_calls;
    for (const nlohmann::ordered_json& function_call : raw_function_calls) {
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = function_call.at("name").get<std::string>();
      tc.function.arguments = function_call.at("arguments").dump();
      tool_calls.push_back(std::move(tc));
    }

    // granite_20b_fc_tool_parser.py:102-107 - content before the first marker.
    const std::string content =
        model_output.substr(0, model_output.find(token));
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content.empty()) info.content = content;
    return info;

  } catch (const std::exception&) {
    // granite_20b_fc_tool_parser.py:109-113 - any error -> whole output content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> Granite20bFCToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  const std::string token = kBotToken;

  // granite_20b_fc_tool_parser.py:125-131 - hold back a partial marker; stream
  // plain text until the marker opens the sequence.
  if (current_text.size() < token.size() &&
      token.compare(0, current_text.size(), current_text) == 0) {
    return std::nullopt;
  }
  if (current_text.compare(0, token.size(), token) != 0) {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }

  // granite_20b_fc_tool_parser.py:137 - Allow.ALL vs Allow.ALL & ~Allow.STR.
  const bool allow_partial_str = current_tool_name_sent;
  try {
    std::vector<nlohmann::ordered_json> tool_call_arr;
    std::vector<bool> is_complete;
    try {
      // granite_20b_fc_tool_parser.py:142-154 - walk each `<function_call> obj`.
      std::size_t start_idx = consume_space(token.size(), current_text);
      while (start_idx < current_text.size()) {
        const std::string tail = current_text.substr(start_idx);
        auto [obj, end_idx] = partial_json_loads(tail, allow_partial_str);
        is_complete.push_back(
            is_complete_json(current_text.substr(start_idx, end_idx)));
        start_idx += end_idx;
        start_idx = consume_space(start_idx, current_text);
        start_idx += token.size();
        start_idx = consume_space(start_idx, current_text);
        tool_call_arr.push_back(std::move(obj));
      }
    } catch (const MalformedPartialJson&) {
      // granite_20b_fc_tool_parser.py:155-157 - not enough tokens yet.
      return std::nullopt;
    }

    // granite_20b_fc_tool_parser.py:159-269 (shared with granite).
    return granite_stream_emit(granite_prev_tool_call_arr_, current_tool_id,
                               current_tool_name_sent, streamed_args_for_tool,
                               tool_call_arr, is_complete);
  } catch (const std::exception&) {
    // granite_20b_fc_tool_parser.py:271-276 - drop the chunk on any error.
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
