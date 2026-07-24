// Ported from: vllm/tool_parsers/ernie45_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/ernie45.h"

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// ernie45_tool_parser.py:48-50 - <tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>,
// DOTALL. ECMAScript `.` is not DOTALL, so [\s\S]. The lazy \{[\s\S]*?\} still
// balances the outer braces because it must be followed by \s*</tool_call>.
const std::regex& ToolCallRegex() {
  static const std::regex re(
      R"(<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>)");
  return re;
}

// str.rstrip("\n").
std::string RStripNewlines(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0 && s[e - 1] == '\n') --e;
  return s.substr(0, e);
}

// str.lstrip("\n").
std::string LStripNewlines(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && s[b] == '\n') ++b;
  return s.substr(b);
}

bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool StartsWith(const std::string& s, const std::string& pre) {
  return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

}  // namespace

ExtractedToolCallInformation Ernie45ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // ernie45_tool_parser.py:77-81 - sanity check.
  if (model_output.find(kToolCallStart) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    std::vector<ToolCall> tool_calls;
    for (auto it = std::sregex_iterator(model_output.begin(), model_output.end(),
                                        ToolCallRegex());
         it != std::sregex_iterator(); ++it) {
      const std::string tool_call_json = (*it)[1].str();
      const nlohmann::ordered_json dict = nlohmann::ordered_json::parse(tool_call_json);
      // args_str = json.dumps(dict.get("arguments", {}), ensure_ascii=False).
      nlohmann::ordered_json args =
          dict.contains("arguments") ? dict.at("arguments")
                                     : nlohmann::ordered_json::object();
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name =
          dict.contains("name") ? dict.at("name").get<std::string>() : "";
      tc.function.arguments = args.dump();
      tool_calls.push_back(std::move(tc));
    }

    // ernie45_tool_parser.py:103-105 - content = output[:first <tool_call>]
    // .rstrip("\n").
    const std::string content = RStripNewlines(
        model_output.substr(0, model_output.find(kToolCallStart)));

    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content.empty()) info.content = content;
    return info;

  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> Ernie45ToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  buffer_ += delta_text;
  std::string cur_text = buffer_;
  const std::size_t start_idx = cur_text.find(kToolCallStart);

  if (start_idx == std::string::npos) {
    buffer_.clear();
    // ernie45_tool_parser.py:133-135 - suppress content once a tool call is done.
    if (ernie_current_tool_id_ > 0) cur_text.clear();
    // ernie45_tool_parser.py:136-139 - DEVIATION: "all previous tokens are the
    // newline token" is not reproducible without token ids; the closest text
    // analogue (previous_text is only newlines) strips leading newlines.
    if (ernie_current_tool_id_ == -1 && !previous_text.empty() &&
        previous_text.find_first_not_of('\n') == std::string::npos) {
      // strip("\n") both ends.
      cur_text = LStripNewlines(RStripNewlines(cur_text));
    }

    std::string content = cur_text;

    // ernie45_tool_parser.py:144-154 - <response>/</response> peel (DEVIATION:
    // token-id membership -> literal-in-delta_text test).
    if (delta_text.find(kResponseStart) != std::string::npos) {
      content = LStripNewlines(content);
      const std::size_t rs = content.find(kResponseStart);
      if (rs != std::string::npos) {
        content = content.substr(rs + std::string(kResponseStart).size());
      }
      const std::size_t re_ = content.rfind(kResponseEnd);
      if (re_ != std::string::npos) content = content.substr(0, re_);
    } else if (delta_text.find(kResponseEnd) != std::string::npos) {
      const std::size_t re_ = content.rfind(kResponseEnd);
      if (re_ != std::string::npos) content = content.substr(0, re_);
    }

    // ernie45_tool_parser.py:155-162 - strip a single leading "\n" that follows a
    // </think>|<response>|</response> boundary (DEVIATION: token ids -> literals).
    const bool prev_is_parser_marker =
        EndsWith(previous_text, kThinkEnd) ||
        EndsWith(previous_text, kResponseStart) ||
        EndsWith(previous_text, kResponseEnd);
    if (prev_is_parser_marker && StartsWith(delta_text, "\n")) {
      content = LStripNewlines(content);
    }

    if (content.empty()) return DeltaMessage{};
    DeltaMessage m;
    m.content = content;
    return m;
  }

  const std::size_t end_idx = cur_text.find(kToolCallEnd);
  if (end_idx != std::string::npos) {
    // ernie45_tool_parser.py:168-175 - lazily init the per-call state.
    if (ernie_current_tool_id_ == -1) {
      ernie_current_tool_id_ = 0;
      ernie_prev_tool_call_arr_.clear();
      ernie_streamed_args_for_tool_.clear();
    }
    while (static_cast<int>(ernie_prev_tool_call_arr_.size()) <= ernie_current_tool_id_)
      ernie_prev_tool_call_arr_.push_back(nlohmann::ordered_json::object());
    while (static_cast<int>(ernie_streamed_args_for_tool_.size()) <= ernie_current_tool_id_)
      ernie_streamed_args_for_tool_.emplace_back();

    const std::string block =
        cur_text.substr(0, end_idx + std::string(kToolCallEnd).size());
    const ExtractedToolCallInformation extracted = extract_tool_calls(block, request);

    if (extracted.tool_calls.empty()) return std::nullopt;
    const ToolCall& tool_call = extracted.tool_calls[0];

    nlohmann::ordered_json entry;
    entry["name"] = tool_call.function.name;
    entry["arguments"] = nlohmann::ordered_json::parse(tool_call.function.arguments);
    ernie_prev_tool_call_arr_[static_cast<std::size_t>(ernie_current_tool_id_)] = entry;
    ernie_streamed_args_for_tool_[static_cast<std::size_t>(ernie_current_tool_id_)] =
        tool_call.function.arguments;

    DeltaToolCall d;
    d.index = ernie_current_tool_id_;
    d.id = tool_call.id;
    d.type = tool_call.type;
    d.function.name = tool_call.function.name;
    d.function.arguments = tool_call.function.arguments;

    DeltaMessage delta;
    delta.content = extracted.content;
    delta.tool_calls = std::vector<DeltaToolCall>{std::move(d)};

    ernie_current_tool_id_ += 1;
    buffer_ = cur_text.substr(end_idx + std::string(kToolCallEnd).size());
    return delta;
  }

  // ernie45_tool_parser.py:210-212 - keep buffering from the <tool_call> start;
  // emit any content that preceded it.
  buffer_ = cur_text.substr(start_idx);
  const std::string content = RStripNewlines(cur_text.substr(0, start_idx));
  if (content.empty()) return DeltaMessage{};
  DeltaMessage m;
  m.content = content;
  return m;
}

}  // namespace vllm::entrypoints::openai
