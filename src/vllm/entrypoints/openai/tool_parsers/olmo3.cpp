// Ported from: vllm/tool_parsers/olmo3_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/olmo3.h"

#include <cstddef>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/tool_parsers/pythonic_core.h"

namespace vllm::entrypoints::openai {

namespace {

using pythonic_core::PyCall;

bool IsWs(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsWs(s[b])) ++b;
  while (e > b && IsWs(s[e - 1])) --e;
  return s.substr(b, e - b);
}

bool StartsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
bool EndsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::string ReplaceAll(std::string s, char from, char to) {
  for (char& c : s) {
    if (c == from) c = to;
  }
  return s;
}

// olmo3_tool_parser.py:84-88 - join the non-blank, stripped lines with ", " and
// wrap the result in a [...] list literal so the pythonic grammar sees a single
// list-of-calls.
std::string JoinLinesToList(const std::string& text) {
  std::string joined;
  std::istringstream iss(text);
  std::string line;
  bool first = true;
  while (std::getline(iss, line)) {
    const std::string stripped = Strip(line);
    if (stripped.empty()) continue;
    if (!first) joined += ", ";
    joined += stripped;
    first = false;
  }
  return "[" + joined + "]";
}

const char kOpenTag[] = "<function_calls>";
const char kCloseTag[] = "</function_calls>";

}  // namespace

ExtractedToolCallInformation Olmo3ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // olmo3_tool_parser.py:77-88. Remove the <function_calls> wrapper (leftmost,
  // non-greedy match), then fold the newline-separated calls into a list.
  std::string inner = model_output;
  const std::size_t open = model_output.find(kOpenTag);
  if (open != std::string::npos) {
    const std::size_t after = open + sizeof(kOpenTag) - 1;
    const std::size_t close = model_output.find(kCloseTag, after);
    if (close != std::string::npos) {
      inner = Strip(model_output.substr(after, close - after));
    }
  }

  const std::string list_text = JoinLinesToList(inner);

  // olmo3_tool_parser.py:90-130 - regex gate + ast parse collapse into one
  // strict parse. On any failure: plain content (the ORIGINAL output), no calls.
  const std::optional<std::vector<PyCall>> calls =
      pythonic_core::parse_call_list(list_text);
  if (!calls.has_value()) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  std::vector<ToolCall> tool_calls;
  tool_calls.reserve(calls->size());
  for (const PyCall& call : *calls) {
    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = call.name;
    tc.function.arguments = call.arguments.dump();
    tool_calls.push_back(std::move(tc));
  }
  ExtractedToolCallInformation info;
  info.tools_called = true;
  info.tool_calls = std::move(tool_calls);
  // content is None when tools are called.
  return info;
}

std::optional<DeltaMessage> Olmo3ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // olmo3_tool_parser.py:142-145 - all tool calls start with '<'; before that
  // the delta is plain content.
  if (current_text.empty() || current_text.front() != '<') {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }

  try {
    // olmo3_tool_parser.py:148-152 - strip the (possibly partial) wrapper.
    std::string text = current_text;
    if (StartsWith(text, kOpenTag)) {
      text = text.substr(sizeof(kOpenTag) - 1);
    }
    if (EndsWith(text, kCloseTag)) {
      text = text.substr(0, text.size() - (sizeof(kCloseTag) - 1));
    }

    const std::optional<std::pair<std::string, std::string>> mv =
        pythonic_core::make_valid_python(text);
    if (!mv.has_value()) return std::nullopt;
    const std::string& added_text = mv->second;

    // olmo3_tool_parser.py:159-163 - fold newline-separated calls into a list.
    const std::string list_text = JoinLinesToList(mv->first);
    const std::optional<std::vector<PyCall>> calls =
        pythonic_core::parse_call_list(list_text);
    if (!calls.has_value()) return std::nullopt;

    std::vector<DeltaToolCall> tool_deltas;
    for (std::size_t index = 0; index < calls->size(); ++index) {
      if (static_cast<int>(index) < current_tool_id) continue;  // current_tool_index
      current_tool_id = static_cast<int>(index);
      if (streamed_args_for_tool.size() == index) {
        streamed_args_for_tool.emplace_back();
      }

      // olmo3_tool_parser.py:186-193 - NOTE the ")" (single char) and
      // added_text[:-1] here, vs pythonic's ")]" / [:-2]: olmo3's make_valid_python
      // runs on the UN-wrapped text, so its added_text lacks pythonic's outer "]".
      const bool new_call_complete = (index < calls->size() - 1) ||
                                     (added_text.find(')') == std::string::npos);
      if (new_call_complete) ++current_tool_id;

      std::string withheld_suffix;
      if (!new_call_complete) {
        withheld_suffix = added_text.substr(0, added_text.size() - 1);  // [:-1]
        if (added_text.back() == ')') withheld_suffix += '}';
      }
      withheld_suffix = ReplaceAll(withheld_suffix, '\'', '"');

      const PyCall& call = (*calls)[index];
      const std::string call_args = call.arguments.dump();
      const std::string call_id = make_tool_call_id();

      const std::optional<DeltaToolCall> delta = pythonic_core::compute_tool_delta(
          streamed_args_for_tool[index], call_id, call.name, call_args,
          static_cast<int>(index), withheld_suffix);
      if (delta.has_value()) {
        tool_deltas.push_back(*delta);
        if (delta->function.arguments.has_value()) {
          streamed_args_for_tool[index] += *delta->function.arguments;
        }
      }
    }

    // olmo3_tool_parser.py:214-215 - HACK: mark prev_tool_call_arr non-empty so
    // the serving layer sets finish_reason=tool_calls.
    if (!tool_deltas.empty() && prev_tool_call_arr.empty()) {
      prev_tool_call_arr.push_back(
          nlohmann::json{{"arguments", nlohmann::json::object()}});
    }

    if (!tool_deltas.empty()) {
      DeltaMessage msg;
      msg.tool_calls = std::move(tool_deltas);
      return msg;
    }
    if (added_text.empty() && current_tool_id > 0) {
      // Emit an empty content delta once all calls are done (finish_reason).
      DeltaMessage msg;
      msg.content = "";
      return msg;
    }
    return std::nullopt;
  } catch (const pythonic_core::ToolDeltaError&) {
    return std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
