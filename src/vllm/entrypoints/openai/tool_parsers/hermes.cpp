// Ported from: vllm/tool_parsers/hermes_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// hermes_tool_parser.py:38-40:
//   tool_call_regex = re.compile(
//       r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
// There are two possible captures: the JSON between the open/close tags
// (group 1), or between an open tag and end-of-string (group 2, the
// unterminated tail). `.` under re.DOTALL matches newlines — ECMAScript `.`
// does not, so we use `[\s\S]` to emulate DOTALL.
const std::regex& tool_call_regex() {
  static const std::regex re(
      R"(<tool_call>([\s\S]*?)</tool_call>|<tool_call>([\s\S]*))");
  return re;
}

// ─── Streaming helpers (tool_parsers/utils.py + hermes_tool_parser.py) ────────

// Python str.strip() / str.rstrip() over ASCII whitespace.
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
std::string Rstrip(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0 && IsSpace(s[e - 1])) --e;
  return s.substr(0, e);
}

// tool_parsers/utils.py:42 (partial_tag_overlap): length of the longest prefix
// of `tag` that matches a suffix of `text` (0 when none). E.g. text ending in
// "<tool_" → 6 for tag "<tool_call>".
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k >= 1; --k) {
    if (text.compare(text.size() - k, k, tag, 0, k) == 0) return k;
  }
  return 0;
}

// tool_parsers/utils.py:141 (is_complete_json). json::accept parses WITHOUT
// throwing and returns whether the input is a complete, valid JSON document.
bool IsCompleteJson(const std::string& s) { return nlohmann::json::accept(s); }

// hermes_tool_parser.py:168 (_extract_tool_name): the tool name, or nullopt when
// it hasn't fully arrived yet.
std::optional<std::string> ExtractToolName(const std::string& tc_json) {
  static const std::regex re(R"rx("name"\s*:\s*"([^"]+)")rx");
  std::smatch m;
  if (std::regex_search(tc_json, m, re)) return m[1].str();
  return std::nullopt;
}

// hermes_tool_parser.py:174 (_extract_tool_args): given {"name":..,"arguments":
// X}, return X. When complete, strips the trailing '}' closing the OUTER object.
std::optional<std::string> ExtractToolArgs(const std::string& tc_json,
                                           bool is_complete) {
  static const std::regex re(R"rx("arguments"\s*:\s*)rx");
  std::smatch m;
  if (!std::regex_search(tc_json, m, re)) return std::nullopt;
  std::string raw = m.suffix().str();  // tc_json[match.end():]
  if (is_complete) {
    raw = Rstrip(raw);
    if (!raw.empty() && raw.back() == '}') {
      raw.pop_back();
      raw = Rstrip(raw);
    }
  }
  return raw;
}

// hermes_tool_parser.py:141 (_extract_tool_call_jsons): (json_text, is_complete)
// for each <tool_call> region in `text`.
std::vector<std::pair<std::string, bool>> ExtractToolCallJsons(
    const std::string& text) {
  const std::string start_tok = HermesToolParser::kToolCallStartToken;
  const std::string end_tok = HermesToolParser::kToolCallEndToken;
  std::vector<std::pair<std::string, bool>> results;
  std::size_t pos = 0;
  for (;;) {
    const std::size_t start = text.find(start_tok, pos);
    if (start == std::string::npos) break;
    const std::size_t json_start = start + start_tok.size();
    const std::size_t json_end = text.find(end_tok, json_start);
    if (json_end != std::string::npos) {
      results.emplace_back(Strip(text.substr(json_start, json_end - json_start)),
                           true);
      pos = json_end + end_tok.size();
    } else {
      std::string raw = text.substr(json_start);
      const std::size_t overlap = PartialTagOverlap(raw, end_tok);
      if (overlap) raw = raw.substr(0, raw.size() - overlap);
      const std::string tc_json = Strip(raw);
      const bool is_complete = tc_json.empty() ? false : IsCompleteJson(tc_json);
      results.emplace_back(tc_json, is_complete);
      break;
    }
  }
  return results;
}

}  // namespace

ExtractedToolCallInformation HermesToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // hermes_tool_parser.py:75-79 — sanity check; avoid unnecessary processing.
  if (model_output.find(kToolCallStartToken) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    // hermes_tool_parser.py:87-107 — findall over the two-capture regex, then
    // json.loads(match[0] if match[0] else match[1]) for each block and build a
    // ToolCall with the arguments re-serialized back to a JSON *string*.
    std::vector<ToolCall> tool_calls;
    for (auto it = std::sregex_iterator(model_output.begin(),
                                        model_output.end(), tool_call_regex());
         it != std::sregex_iterator(); ++it) {
      const std::smatch& match = *it;
      // Prefer group 1 (between tags); fall back to group 2 (unterminated tail).
      const std::string raw =
          match[1].matched ? match[1].str() : match[2].str();

      // json.loads — throws on malformed JSON (caught below -> fallback).
      const nlohmann::json function_call = nlohmann::json::parse(raw);

      ToolCall tc;
      tc.id = make_tool_call_id();  // default_factory=make_tool_call_id.
      tc.type = "function";
      // .at() throws (KeyError-equivalent) when "name"/"arguments" is absent,
      // matching upstream's dict lookups inside the try/except.
      tc.function.name = function_call.at("name").get<std::string>();
      // function call args are JSON but stored AS A STRING
      // (json.dumps(..., ensure_ascii=False)). nlohmann dump() is compact —
      // whitespace differs from CPython's ", "/": " separators, but the JSON is
      // semantically identical (clients json.loads it). Deviation: whitespace.
      tc.function.arguments = function_call.at("arguments").dump();
      tool_calls.push_back(std::move(tc));
    }

    // hermes_tool_parser.py:109-114 — content is the text before the first
    // <tool_call>; None when empty.
    const std::string content =
        model_output.substr(0, model_output.find(kToolCallStartToken));
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content.empty()) {
      info.content = content;
    }
    return info;

  } catch (const std::exception&) {
    // hermes_tool_parser.py:116-120 — on ANY error, log + fall back to the whole
    // output as content with tools_called=false.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> HermesToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/,
    const ChatCompletionRequest& /*request*/) {
  // hermes_tool_parser.py:204-274. Re-parse the full current_text each call and
  // diff against the sent state to emit only new content / names / arg deltas.
  try {
    // hermes_tool_parser.py:225 — content before any tool call (holds back a
    // suffix that could be a partial <tool_call> tag). Inlined _extract_content
    // (hermes_tool_parser.py:122) because it touches sent_content_idx_.
    std::optional<std::string> content;
    {
      std::size_t sendable_idx;
      const std::size_t start = current_text.find(kToolCallStartToken);
      if (start == std::string::npos) {
        const std::size_t overlap =
            PartialTagOverlap(current_text, kToolCallStartToken);
        sendable_idx = current_text.size() - overlap;
      } else {
        sendable_idx = start;
      }
      if (sendable_idx > sent_content_idx_) {
        content = current_text.substr(sent_content_idx_,
                                      sendable_idx - sent_content_idx_);
        sent_content_idx_ = sendable_idx;
      }
    }

    const std::vector<std::pair<std::string, bool>> tool_call_jsons =
        ExtractToolCallJsons(current_text);
    std::vector<DeltaToolCall> tool_call_deltas;

    for (std::size_t i = 0; i < tool_call_jsons.size(); ++i) {
      const std::string& tc_json = tool_call_jsons[i].first;
      const bool is_complete = tool_call_jsons[i].second;

      if (i >= prev_tool_call_arr.size()) {
        prev_tool_call_arr.push_back(nlohmann::json::object());
        streamed_args_for_tool.emplace_back();
      }

      // hermes_tool_parser.py:234-250 — stream the tool name ONCE (name-first).
      if (!prev_tool_call_arr[i].contains("name")) {
        const std::optional<std::string> name = ExtractToolName(tc_json);
        if (!name.has_value()) {
          // Can't advance to tool i+1 until tool i's name is ready.
          break;
        }
        prev_tool_call_arr[i]["name"] = *name;
        DeltaToolCall d;
        d.index = static_cast<int>(i);
        d.type = "function";
        d.id = make_tool_call_id();
        d.function.name = *name;
        tool_call_deltas.push_back(std::move(d));
      }

      // hermes_tool_parser.py:252-262 — stream new argument text (the diff).
      // Inlined _compute_args_diff (hermes_tool_parser.py:192) — touches state.
      const std::optional<std::string> args =
          ExtractToolArgs(tc_json, is_complete);
      if (args.has_value() && args->size() > streamed_args_for_tool[i].size()) {
        std::string args_diff = args->substr(streamed_args_for_tool[i].size());
        streamed_args_for_tool[i] = *args;
        prev_tool_call_arr[i]["arguments"] = *args;
        if (!args_diff.empty()) {
          DeltaToolCall d;
          d.index = static_cast<int>(i);
          d.function.arguments = std::move(args_diff);
          tool_call_deltas.push_back(std::move(d));
        }
      }
    }

    // hermes_tool_parser.py:264-270.
    if ((content.has_value() && !content->empty()) ||
        !tool_call_deltas.empty()) {
      DeltaMessage msg;
      if (content.has_value() && !content->empty()) msg.content = content;
      if (!tool_call_deltas.empty()) msg.tool_calls = std::move(tool_call_deltas);
      return msg;
    }
    return std::nullopt;

  } catch (const std::exception&) {
    // hermes_tool_parser.py:272-274 — on ANY error, drop the delta.
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
