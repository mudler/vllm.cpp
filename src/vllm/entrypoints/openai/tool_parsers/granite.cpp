// Ported from: vllm/tool_parsers/granite_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/granite.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

namespace vllm::entrypoints::openai {

namespace {

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && IsSpace(s[b])) ++b;
  while (e > b && IsSpace(s[e - 1])) --e;
  return s.substr(b, e - b);
}
std::string Lstrip(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && IsSpace(s[b])) ++b;
  return s.substr(b);
}
// Python str.removeprefix.
std::string RemovePrefix(const std::string& s, const std::string& prefix) {
  if (s.compare(0, prefix.size(), prefix) == 0) return s.substr(prefix.size());
  return s;
}
// Python str[i:].startswith(prefix).
bool StartsWithAt(const std::string& s, std::size_t i, const std::string& prefix) {
  return s.compare(i, prefix.size(), prefix) == 0;
}

}  // namespace

ExtractedToolCallInformation GraniteToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // granite_tool_parser.py:58-63 - strip, drop either marker prefix, lstrip.
  std::string stripped = Strip(model_output);
  stripped = RemovePrefix(stripped, kBotToken);
  stripped = RemovePrefix(stripped, kBotString);
  stripped = Lstrip(stripped);

  // granite_tool_parser.py:64-67 - must be a JSON list.
  if (stripped.empty() || stripped[0] != '[') {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    // granite_tool_parser.py:69-73 - json.loads; must be a list. ordered_json
    // preserves argument key order for the serialized arguments string.
    const nlohmann::ordered_json raw_function_calls =
        nlohmann::ordered_json::parse(stripped);
    if (!raw_function_calls.is_array()) {
      throw std::runtime_error("Expected list");
    }

    // granite_tool_parser.py:76-88.
    std::vector<ToolCall> tool_calls;
    for (const nlohmann::ordered_json& function_call : raw_function_calls) {
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = function_call.at("name").get<std::string>();
      // arguments are JSON but stored AS A STRING (json.dumps).
      tc.function.arguments = function_call.at("arguments").dump();
      tool_calls.push_back(std::move(tc));
    }

    // granite_tool_parser.py:90-94 - content is None when tools are called.
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    return info;

  } catch (const std::exception&) {
    // granite_tool_parser.py:96-100 - any error -> whole output as content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> GraniteToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // granite_tool_parser.py:112-122 - skip whitespace + either marker, then the
  // remainder must open a JSON array; otherwise stream the delta as content.
  std::size_t start_idx = consume_space(0, current_text);
  if (StartsWithAt(current_text, start_idx, kBotToken)) {
    start_idx = consume_space(start_idx + std::string(kBotToken).size(), current_text);
  }
  if (StartsWithAt(current_text, start_idx, kBotString)) {
    start_idx = consume_space(start_idx + std::string(kBotString).size(), current_text);
  }
  if (current_text.empty() || start_idx >= current_text.size() ||
      current_text[start_idx] != '[') {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }

  // granite_tool_parser.py:128 - Allow.ALL vs Allow.ALL & ~Allow.STR.
  const bool allow_partial_str = current_tool_name_sent;
  try {
    std::vector<nlohmann::ordered_json> tool_call_arr;
    std::vector<bool> is_complete;
    try {
      // granite_tool_parser.py:133-143.
      const std::string tail = current_text.substr(start_idx);
      auto [tool_calls, end_idx] = partial_json_loads(tail, allow_partial_str);
      if (!tool_calls.is_array()) {
        DeltaMessage msg;
        msg.content = delta_text;
        return msg;
      }
      for (const nlohmann::ordered_json& e : tool_calls) tool_call_arr.push_back(e);
      is_complete.assign(tool_call_arr.size(), true);
      if (!tool_call_arr.empty() && !is_complete_json(tail.substr(0, end_idx))) {
        is_complete.back() = false;
      }
    } catch (const MalformedPartialJson&) {
      // granite_tool_parser.py:144-146 - not enough tokens yet.
      return std::nullopt;
    }

    // granite_tool_parser.py:150-151.
    if (tool_call_arr.empty()) return std::nullopt;

    // granite_tool_parser.py:154-249 (shared with granite-20b-fc).
    return granite_stream_emit(granite_prev_tool_call_arr_, current_tool_id,
                               current_tool_name_sent, streamed_args_for_tool,
                               tool_call_arr, is_complete);
  } catch (const std::exception&) {
    // granite_tool_parser.py:251-256 - drop the chunk on any error.
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
