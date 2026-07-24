// Ported from: vllm/tool_parsers/apertus_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/apertus.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

namespace vllm::entrypoints::openai {

namespace {

// apertus_tool_parser.py:94-99 - <|tools_prefix|>(.*?)(?:<|tools_suffix|>|$), DOTALL.
const std::regex& ToolCallRegex() {
  static const std::regex re(
      R"(<\|tools_prefix\|>([\s\S]*?)(?:<\|tools_suffix\|>|$))");
  return re;
}

bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && static_cast<unsigned char>(s[b]) <= ' ') ++b;
  while (e > b && static_cast<unsigned char>(s[e - 1]) <= ' ') --e;
  return s.substr(b, e - b);
}

std::string LStrip(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && static_cast<unsigned char>(s[b]) <= ' ') ++b;
  return s.substr(b);
}

std::string ReplaceAll(std::string s, const std::string& from,
                       const std::string& to) {
  if (from.empty()) return s;
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

// str.split(sep)[0] - the substring before the FIRST sep (whole string if none).
std::string BeforeFirst(const std::string& s, const std::string& sep) {
  const std::size_t p = s.find(sep);
  return p == std::string::npos ? s : s.substr(0, p);
}

// str.split(sep)[-1] - the substring after the LAST sep (whole string if none).
std::string AfterLast(const std::string& s, const std::string& sep) {
  const std::size_t p = s.rfind(sep);
  return p == std::string::npos ? s : s.substr(p + sep.size());
}

// Python truthiness of a parsed JSON value.
bool JsonTruthy(const nlohmann::ordered_json& v) {
  if (v.is_null()) return false;
  if (v.is_string()) return !v.get<std::string>().empty();
  if (v.is_object() || v.is_array()) return !v.empty();
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number_float()) return v.get<double>() != 0.0;
  if (v.is_number()) return v.get<int64_t>() != 0;
  return true;
}

// json.JSONDecoder().raw_decode equivalent (see apertus.h DEVIATION): parse the
// COMPLETE leading JSON value out of `s`, returning {value, end_idx}. nullopt when
// `s` does not begin with a complete value (truncated / empty) - the caller then
// takes the tolerant partial-parse fallback.
std::optional<std::pair<nlohmann::ordered_json, std::size_t>> RawDecode(
    const std::string& s) {
  try {
    auto [val, end] = partial_json_loads(s, /*allow_partial_str=*/true);
    if (nlohmann::json::accept(s.substr(0, end))) {
      return std::make_pair(std::move(val), end);
    }
    return std::nullopt;
  } catch (const MalformedPartialJson&) {
    return std::nullopt;
  }
}

}  // namespace

ExtractedToolCallInformation ApertusToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  std::smatch m;
  if (!std::regex_search(model_output, m, ToolCallRegex())) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    const std::string matched_text = m[1].str();
    const std::string stripped_text = LStrip(matched_text);

    nlohmann::ordered_json parsed_json;
    std::string trailing_in_group;
    if (auto rd = RawDecode(stripped_text)) {
      parsed_json = rd->first;
      trailing_in_group = stripped_text.substr(rd->second);
    } else {
      // Fallback for token-truncated requests. A MalformedPartialJson here
      // propagates to the outer catch -> whole output as content (matches
      // upstream: the fallback lives in the inner try, its raise reaches the
      // outer `except Exception`).
      parsed_json = partial_json_loads(matched_text, /*allow_partial_str=*/true)
                        .first;
      trailing_in_group = "";
    }

    // if not isinstance(list): [parsed] if parsed else [].
    nlohmann::ordered_json arr;
    if (parsed_json.is_array()) {
      arr = std::move(parsed_json);
    } else {
      arr = nlohmann::ordered_json::array();
      if (JsonTruthy(parsed_json)) arr.push_back(parsed_json);
    }

    std::vector<ToolCall> tool_calls;
    for (const nlohmann::ordered_json& obj : arr) {
      if (obj.is_object() && !obj.empty()) {
        const auto first = obj.begin();
        ToolCall tc;
        tc.id = make_tool_call_id();
        tc.type = "function";
        tc.function.name = first.key();
        tc.function.arguments = first.value().dump();  // json.dumps(ensure_ascii=False)
        tool_calls.push_back(std::move(tc));
      }
    }

    // Content before the block, plus any surfaced trailing text.
    std::string content_str =
        Strip(model_output.substr(0, static_cast<std::size_t>(m.position(0))));

    if (!Strip(trailing_in_group).empty()) {
      const std::string trailing = Strip(ReplaceAll(trailing_in_group, kToolsSuffix, ""));
      if (!trailing.empty()) content_str = Strip(content_str + "\n" + trailing);
    }

    const std::string after_suffix = Strip(ReplaceAll(
        model_output.substr(
            static_cast<std::size_t>(m.position(0) + m.length(0))),
        kToolsSuffix, ""));
    if (!after_suffix.empty()) content_str = Strip(content_str + "\n" + after_suffix);

    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content_str.empty()) info.content = content_str;
    return info;

  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::string ApertusToolParser::buffer_delta_text(const std::string& delta_text) {
  buffered_delta_text_ += delta_text;
  const std::string text = buffered_delta_text_;
  const std::string tags[] = {kToolsPrefix, kToolsSuffix};
  for (const std::string& tag : tags) {
    if (EndsWith(text, tag)) {
      buffered_delta_text_.clear();
      return text;
    }
    for (std::size_t i = tag.size() - 1; i > 0; --i) {
      if (EndsWith(text, tag.substr(0, i))) {
        buffered_delta_text_ = text.substr(text.size() - i);
        return text.substr(0, text.size() - i);
      }
    }
  }
  buffered_delta_text_.clear();
  return text;
}

std::optional<DeltaMessage> ApertusToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text_in, const ChatCompletionRequest& /*request*/) {
  const std::string delta_text = buffer_delta_text(delta_text_in);
  if (delta_text.empty()) return std::nullopt;

  if (current_text.find(kToolsPrefix) == std::string::npos) {
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }

  try {
    return extract_streaming(current_text, delta_text);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<DeltaMessage> ApertusToolParser::extract_streaming(
    const std::string& current_text, const std::string& delta_text) {
  const std::string prefix = kToolsPrefix;
  const std::string suffix = kToolsSuffix;

  auto rfind_i = [](const std::string& s, const std::string& sub) -> long {
    const std::size_t p = s.rfind(sub);
    return p == std::string::npos ? -1L : static_cast<long>(p);
  };
  const long prefix_idx = rfind_i(current_text, prefix);
  const long suffix_idx = rfind_i(current_text, suffix);

  bool is_inside_tools = prefix_idx > suffix_idx;
  bool json_completed = false;
  std::optional<long> json_end_idx;

  if (is_inside_tools) {
    const long json_start = prefix_idx + static_cast<long>(prefix.size());
    const std::string s = LStrip(current_text.substr(static_cast<std::size_t>(json_start)));
    if (auto rd = RawDecode(s)) {
      const long idx = static_cast<long>(rd->second);
      json_end_idx = static_cast<long>(current_text.size()) -
                     static_cast<long>(s.size()) + idx;
      json_completed = true;
      is_inside_tools = false;
    }
  }

  const bool just_finished =
      (delta_text.find(suffix) != std::string::npos) || json_completed;

  // 1. Fast path: emit normal text when fully outside a tool block.
  if (!is_inside_tools && !just_finished) {
    const std::string text = ReplaceAll(ReplaceAll(delta_text, prefix, ""), suffix, "");
    if (!text.empty()) {
      DeltaMessage m;
      m.content = text;
      return m;
    }
    return std::nullopt;
  }

  // 2. Leading / trailing normal text directly adjacent to the tool block.
  std::string content_str;
  if (delta_text.find(prefix) != std::string::npos) {
    content_str += ReplaceAll(BeforeFirst(delta_text, prefix), suffix, "");
  }
  if (just_finished) {
    if (json_completed && json_end_idx.has_value()) {
      const long delta_start_idx =
          static_cast<long>(current_text.size()) - static_cast<long>(delta_text.size());
      const long content_start = std::max(*json_end_idx, delta_start_idx);
      if (content_start < static_cast<long>(current_text.size())) {
        content_str += ReplaceAll(
            current_text.substr(static_cast<std::size_t>(content_start)), suffix, "");
      }
    } else {
      content_str += AfterLast(delta_text, suffix);
    }
  }

  // 3. Isolate the JSON array string for the active block.
  const long json_start = prefix_idx + static_cast<long>(prefix.size());
  const long json_end = (suffix_idx > prefix_idx)
                            ? suffix_idx
                            : (json_end_idx.has_value()
                                   ? *json_end_idx
                                   : static_cast<long>(current_text.size()));
  std::string json_str;
  if (json_end > json_start) {
    json_str = current_text.substr(static_cast<std::size_t>(json_start),
                                   static_cast<std::size_t>(json_end - json_start));
  }

  std::vector<DeltaToolCall> tool_calls =
      parse_and_diff_json(json_str, /*is_final=*/!is_inside_tools);

  if (!tool_calls.empty() || !content_str.empty()) {
    DeltaMessage m;
    if (!content_str.empty()) m.content = content_str;
    if (!tool_calls.empty()) m.tool_calls = std::move(tool_calls);
    return m;
  }
  return std::nullopt;
}

std::vector<DeltaToolCall> ApertusToolParser::parse_and_diff_json(
    const std::string& json_str, bool is_final) {
  nlohmann::ordered_json parsed;
  try {
    parsed = partial_json_loads(json_str, /*allow_partial_str=*/true).first;
  } catch (const MalformedPartialJson&) {
    return {};
  }

  nlohmann::ordered_json arr;
  if (parsed.is_array()) {
    arr = std::move(parsed);
  } else {
    arr = nlohmann::ordered_json::array();
    if (JsonTruthy(parsed)) arr.push_back(parsed);
  }
  if (arr.empty()) return {};

  std::vector<DeltaToolCall> tool_calls;
  const int latest_index = static_cast<int>(arr.size()) - 1;

  // Catch up over any tools skipped in one large delta.
  while (current_tool_id_ < latest_index) {
    if (current_tool_id_ >= 0) {
      if (!current_tool_name_sent_) emit_tool_name(arr, current_tool_id_, tool_calls);
      if (auto d = get_tool_diff(arr, current_tool_id_, /*is_final=*/true))
        tool_calls.push_back(*d);
    }
    current_tool_id_ += 1;
    current_tool_name_sent_ = false;
    while (static_cast<int>(streamed_args_for_tool_.size()) <= current_tool_id_)
      streamed_args_for_tool_.emplace_back();
  }

  // Stream the currently active tool.
  if (current_tool_id_ >= 0) {
    if (!current_tool_name_sent_) emit_tool_name(arr, current_tool_id_, tool_calls);
    if (auto d = get_tool_diff(arr, current_tool_id_, is_final))
      tool_calls.push_back(*d);
  }

  return tool_calls;
}

void ApertusToolParser::emit_tool_name(const nlohmann::ordered_json& parsed,
                                       int index,
                                       std::vector<DeltaToolCall>& tool_calls) {
  const nlohmann::ordered_json& obj = parsed[static_cast<std::size_t>(index)];
  if (obj.is_object() && !obj.empty()) {
    const auto first = obj.begin();
    const std::string name = first.key();
    current_tool_name_sent_ = true;
    DeltaToolCall d;
    d.index = index;
    d.type = "function";
    d.id = make_tool_call_id();
    d.function.name = name;
    d.function.arguments = std::string("");  // model_dump keeps arguments="".
    tool_calls.push_back(std::move(d));
  }
}

std::optional<DeltaToolCall> ApertusToolParser::get_tool_diff(
    const nlohmann::ordered_json& parsed, int index, bool is_final) {
  const nlohmann::ordered_json& obj = parsed[static_cast<std::size_t>(index)];
  if (!obj.is_object() || obj.empty()) return std::nullopt;

  const auto first = obj.begin();
  const nlohmann::ordered_json& args = first.value();
  if (args.is_null()) return std::nullopt;

  std::string args_json = args.dump();

  if (!is_final) {
    while (!args_json.empty()) {
      const char c = args_json.back();
      if (c == '}' || c == '"' || c == ']' || c == ' ' || c == ',') {
        args_json.pop_back();
      } else {
        break;
      }
    }
  }

  std::string& prev_sent = streamed_args_for_tool_[static_cast<std::size_t>(index)];
  if (args_json == prev_sent) return std::nullopt;

  const std::string prefix = find_common_prefix(prev_sent, args_json);
  if (prefix.size() < prev_sent.size()) {
    // Partial parser structurally revised a past assumption; backtrack.
    prev_sent = prefix;
    return std::nullopt;
  }

  const std::string diff = args_json.substr(prev_sent.size());
  if (!diff.empty()) {
    prev_sent = args_json;
    DeltaToolCall d;
    d.index = index;
    d.function.arguments = diff;
    return d;
  }
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
