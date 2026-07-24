// REIMPLEMENTED FROM THE WIRE FORMAT of vllm/parser/kimi_k2.py @ e24d1b24.
// See kimi_k2.h for the grammar citation and the recorded deviations.
#include "vllm/entrypoints/openai/tool_parsers/kimi_k2.h"

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

bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

// Python str.strip() over ASCII whitespace (both ends). Mirrors the header
// `.strip()` (kimi_k2.py:182) and the args `.strip()` (kimi_k2.py:237).
std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsSpace(s[b])) ++b;
  while (e > b && IsSpace(s[e - 1])) --e;
  return s.substr(b, e - b);
}

bool IsWhitespaceOnly(const std::string& s) {
  for (const char c : s) {
    if (!IsSpace(c)) return false;
  }
  return true;
}

// tool_parsers/utils.py:42 (partial_tag_overlap): length of the longest prefix
// of `tag` that matches a suffix of `text` (0 when none). Used so a marker split
// across a delta boundary is never emitted as content/arguments.
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k >= 1; --k) {
    if (text.compare(text.size() - k, k, tag, 0, k) == 0) return k;
  }
  return 0;
}

// _TOOL_ID_RE (kimi_k2.py:49) `(?P<id>.+:\d+)` anchored at the start of the
// stripped header (re.match). `.` excludes newline (ECMAScript, like Python re
// without DOTALL); the header is a single stripped line so this is exact.
const std::regex& ToolIdRegex() {
  static const std::regex re(R"(^(.+:\d+))");
  return re;
}

// kimi_k2.py:179-188 (_extract_tool_id_and_name). Returns {native_id, name} for
// a header, or nullopt when it does not match `.+:\d+` (the call is dropped).
struct ParsedHeader {
  std::string id;
  std::string name;
};
std::optional<ParsedHeader> ParseHeader(const std::string& raw_header) {
  const std::string header = Strip(raw_header);
  std::smatch m;
  if (!std::regex_search(header, m, ToolIdRegex())) return std::nullopt;
  ParsedHeader out;
  out.id = m[1].str();
  // name = id.split(":")[0].removeprefix("functions.") (kimi_k2.py:187-188).
  const std::size_t colon = out.id.find(':');
  std::string name =
      (colon == std::string::npos) ? out.id : out.id.substr(0, colon);
  static const std::string kPrefix = "functions.";
  if (name.compare(0, kPrefix.size(), kPrefix) == 0) {
    name = name.substr(kPrefix.size());
  }
  if (name.empty()) return std::nullopt;  // kimi_k2.py:197 (no tool name -> drop)
  out.name = std::move(name);
  return out;
}

}  // namespace

void KimiK2ToolParser::adjust_request(ChatCompletionRequest& request) const {
  const bool has_tools =
      request.tools.has_value() && !request.tools->empty();
  const bool choice_none =
      request.tool_choice.has_value() && request.tool_choice->mode == "none";
  if (has_tools && !choice_none) request.skip_special_tokens = false;
}

ExtractedToolCallInformation KimiK2ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string section_begin = kToolCallsSectionBeginToken;
  const std::size_t section_pos = model_output.find(section_begin);
  // No section marker -> plain content (kimi_k2.py has no tool events).
  if (section_pos == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    const std::string call_begin = kToolCallBeginToken;
    const std::string call_end = kToolCallEndToken;
    const std::string arg_begin = kToolCallArgumentBeginToken;
    const std::string section_end = kToolCallsSectionEndToken;

    std::vector<ToolCall> tool_calls;
    std::size_t pos = section_pos + section_begin.size();
    for (;;) {
      const std::size_t ts = model_output.find(call_begin, pos);
      if (ts == std::string::npos) break;
      const std::size_t header_start = ts + call_begin.size();
      const std::size_t as = model_output.find(arg_begin, header_start);
      if (as == std::string::npos) break;  // no argument marker -> incomplete
      const std::string header =
          model_output.substr(header_start, as - header_start);
      const std::optional<ParsedHeader> parsed = ParseHeader(header);

      const std::size_t args_start = as + arg_begin.size();
      const std::size_t te = model_output.find(call_end, args_start);
      const std::size_t se = model_output.find(section_end, args_start);

      // A call closes on <|tool_call_end|> (next call may follow) OR directly on
      // <|tool_calls_section_end|> (kimi_k2.py:118-125); the section end also
      // stops the whole scan. An unterminated tail (max_tokens) runs to EOS.
      std::string args_body;
      std::size_t next_pos = model_output.size();
      const bool closed_by_call_end =
          te != std::string::npos && (se == std::string::npos || te < se);
      if (closed_by_call_end) {
        args_body = model_output.substr(args_start, te - args_start);
        next_pos = te + call_end.size();
      } else if (se != std::string::npos) {
        args_body = model_output.substr(args_start, se - args_start);
      } else {
        args_body = model_output.substr(args_start);  // truncated tail
      }

      if (parsed.has_value()) {
        ToolCall tc;
        tc.id = parsed->id;  // DEVIATION 2: native id, not make_tool_call_id().
        tc.type = "function";
        tc.function.name = parsed->name;
        const std::string args = Strip(args_body);
        tc.function.arguments = args.empty() ? "{}" : args;  // kimi_k2.py:237
        tool_calls.push_back(std::move(tc));
      }

      if (!closed_by_call_end) break;  // section-closed or truncated -> stop
      pos = next_pos;
    }

    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    // Content before the section; whitespace-only prefix dropped to None
    // (kimi_k2.py:144), non-whitespace kept unstripped (kimi_k2.py:145).
    const std::string content = model_output.substr(0, section_pos);
    if (!IsWhitespaceOnly(content)) info.content = content;
    return info;

  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> KimiK2ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/,
    const ChatCompletionRequest& /*request*/) {
  try {
    const std::string section_begin = kToolCallsSectionBeginToken;
    const std::string call_begin = kToolCallBeginToken;
    const std::string call_end = kToolCallEndToken;
    const std::string arg_begin = kToolCallArgumentBeginToken;
    const std::string section_end = kToolCallsSectionEndToken;

    // 1. Leading content = text before the section marker; hold back a partial
    // (possibly mid-marker) section-begin suffix so no marker byte leaks.
    std::optional<std::string> content;
    {
      const std::size_t pos = current_text.find(section_begin);
      const std::size_t sendable_idx =
          (pos == std::string::npos)
              ? current_text.size() - PartialTagOverlap(current_text, section_begin)
              : pos;
      if (sendable_idx > sent_content_idx_) {
        content = current_text.substr(sent_content_idx_,
                                      sendable_idx - sent_content_idx_);
        sent_content_idx_ = sendable_idx;
      }
    }

    std::vector<DeltaToolCall> deltas;
    const std::size_t section_pos = current_text.find(section_begin);
    if (section_pos != std::string::npos) {
      std::size_t pos = section_pos + section_begin.size();
      std::size_t index = 0;
      for (;;) {
        const std::size_t ts = current_text.find(call_begin, pos);
        if (ts == std::string::npos) break;
        const std::size_t header_start = ts + call_begin.size();
        const std::size_t as = current_text.find(arg_begin, header_start);
        if (as == std::string::npos) break;  // header not terminated yet -> wait
        const std::string header =
            current_text.substr(header_start, as - header_start);
        const std::optional<ParsedHeader> parsed = ParseHeader(header);

        const std::size_t args_start = as + arg_begin.size();
        const std::size_t te = current_text.find(call_end, args_start);
        const std::size_t se = current_text.find(section_end, args_start);
        bool complete = false;
        bool section_closed = false;
        std::size_t next_pos = current_text.size();
        std::string args_body;
        const bool closed_by_call_end =
            te != std::string::npos && (se == std::string::npos || te < se);
        if (closed_by_call_end) {
          complete = true;
          args_body = current_text.substr(args_start, te - args_start);
          next_pos = te + call_end.size();
        } else if (se != std::string::npos) {
          complete = true;
          section_closed = true;
          args_body = current_text.substr(args_start, se - args_start);
        } else {
          args_body = current_text.substr(args_start);
          // Hold back a partial trailing marker (a prefix of either closing
          // marker) so no marker byte leaks into an argument diff.
          const std::size_t ov = std::max(PartialTagOverlap(args_body, call_end),
                                          PartialTagOverlap(args_body, section_end));
          if (ov) args_body = args_body.substr(0, args_body.size() - ov);
        }

        // Ensure the per-index slot exists (a dropped/invalid header still
        // consumes a slot so the tool_index stays aligned with the engine).
        if (index >= prev_tool_call_arr.size()) {
          prev_tool_call_arr.push_back(nlohmann::json::object());
          streamed_args_for_tool.emplace_back();
        }

        if (parsed.has_value()) {
          // name-first: emit the native id + name exactly once (DEVIATION 2).
          if (!prev_tool_call_arr[index].contains("name")) {
            prev_tool_call_arr[index]["name"] = parsed->name;
            DeltaToolCall d;
            d.index = static_cast<int>(index);
            d.type = "function";
            d.id = parsed->id;
            d.function.name = parsed->name;
            deltas.push_back(std::move(d));
          }

          // argument diff: stream only the newly-arrived suffix. On completion
          // an empty payload becomes "{}" (kimi_k2.py:237), matching non-stream.
          std::string target = Strip(args_body);
          if (complete && target.empty()) target = "{}";
          const std::string& streamed = streamed_args_for_tool[index];
          if (target.size() > streamed.size() &&
              target.compare(0, streamed.size(), streamed) == 0) {
            std::string diff = target.substr(streamed.size());
            streamed_args_for_tool[index] = std::move(target);
            if (!diff.empty()) {
              DeltaToolCall d;
              d.index = static_cast<int>(index);
              d.function.arguments = std::move(diff);
              deltas.push_back(std::move(d));
            }
          }
        }

        if (!complete) break;      // partial tail (or truncated) -> wait/stop
        if (section_closed) break;  // section closed -> no more calls
        pos = next_pos;
        ++index;
      }
    }

    if ((content.has_value() && !content->empty()) || !deltas.empty()) {
      DeltaMessage msg;
      if (content.has_value() && !content->empty()) msg.content = content;
      if (!deltas.empty()) msg.tool_calls = std::move(deltas);
      return msg;
    }
    return std::nullopt;

  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
