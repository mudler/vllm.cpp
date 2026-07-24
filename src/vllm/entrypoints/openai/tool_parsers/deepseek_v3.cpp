// Ported from: vllm/tool_parsers/deepseekv3_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// Python str.rstrip() over ASCII whitespace.
bool IsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
std::string Rstrip(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0 && IsSpace(s[e - 1])) --e;
  return s.substr(0, e);
}

// tool_parsers/utils.py:42 (partial_tag_overlap): length of the longest prefix
// of `tag` that matches a suffix of `text` (0 when none). Lets the streaming
// content path hold back a partial (mid-byte) OUTER-begin marker so no broken
// UTF-8 is emitted as content when a marker is split across a delta boundary.
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k >= 1; --k) {
    if (text.compare(text.size() - k, k, tag, 0, k) == 0) return k;
  }
  return 0;
}

}  // namespace

// deepseekv3_tool_parser.py:49-51 — the per-call extraction regex. The markers
// are literal multi-byte UTF-8 byte strings; std::regex matches them byte-for-
// byte. `.` (ECMAScript, like Python re without DOTALL) excludes newlines, which
// matches upstream — every captured group is single-line in the DeepSeek format.
const std::regex& DeepSeekV3ToolParser::tool_call_pattern() const {
  static const std::regex re(
      R"(<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>)");
  return re;
}

// deepseekv3_tool_parser.py:101-110 — build a ToolCall from one match. Group 1 is
// the type (e.g. "function"), group 2 the name, group 3 the raw JSON arguments
// STRING (stored verbatim, not re-serialized — the client json.loads it).
ToolCall DeepSeekV3ToolParser::tool_call_from_match(
    const std::smatch& match) const {
  ToolCall tc;
  tc.id = make_tool_call_id();  // upstream ToolCall.id default_factory.
  tc.type = match[1].str();
  tc.function.name = match[2].str();
  tc.function.arguments = match[3].str();
  return tc;
}

// deepseekv3_tool_parser.py:53-59 — the streaming per-region parse, reworked to
// string ops. `region` is the text between <｜tool▁call▁begin｜> and its
// <｜tool▁call▁end｜> (or the unterminated tail). Layout: TYPE<｜tool▁sep｜>NAME
// \n```json\nARGS[\n```]. Returns name=nullopt until the NAME line is complete.
DeepSeekV3ToolParser::ParsedCall DeepSeekV3ToolParser::parse_region(
    const std::string& region) const {
  ParsedCall out;
  const std::string sep = kToolSepToken;
  const std::size_t sep_pos = region.find(sep);
  if (sep_pos == std::string::npos) return out;  // name not yet available

  const std::string after = region.substr(sep_pos + sep.size());
  const std::size_t nl = after.find('\n');
  if (nl == std::string::npos) return out;  // NAME line not terminated yet
  out.name = after.substr(0, nl);

  static const std::string kFence = "```json\n";
  const std::size_t fp = after.find(kFence);
  if (fp == std::string::npos) return out;  // args not started (empty)

  std::string args = after.substr(fp + kFence.size());
  // Cut the closing ``` fence when present (complete call); otherwise, while
  // streaming, hold back a PARTIAL closing fence suffix ("`" / "``") so the
  // fence backticks never leak into an argument diff. Then strip the trailing
  // newline so the streamed args grow monotonically and the final value equals
  // the non-streaming capture group exactly.
  const std::size_t close = args.find("```");
  if (close != std::string::npos) {
    args = args.substr(0, close);
  } else {
    const std::size_t ov = PartialTagOverlap(args, "```");
    if (ov) args = args.substr(0, args.size() - ov);
  }
  out.arguments = Rstrip(args);
  return out;
}

ExtractedToolCallInformation DeepSeekV3ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // deepseekv3_tool_parser.py:86-90 — sanity check; avoid unnecessary work.
  if (model_output.find(kToolCallsBeginToken) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    // deepseekv3_tool_parser.py:98-110 — findall over the per-call regex.
    std::vector<ToolCall> tool_calls;
    for (auto it = std::sregex_iterator(
             model_output.begin(), model_output.end(), tool_call_pattern());
         it != std::sregex_iterator(); ++it) {
      tool_calls.push_back(tool_call_from_match(*it));
    }

    // deepseekv3_tool_parser.py:112-117 — content is the text before the OUTER
    // begin marker; None when empty. NOTE (matches the upstream xfail): when the
    // begin marker is present but NO call matches (malformed), tools_called stays
    // true with an EMPTY tool_calls list.
    const std::string content = model_output.substr(
        0, model_output.find(kToolCallsBeginToken));
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (!content.empty()) info.content = content;
    return info;

  } catch (const std::exception&) {
    // deepseekv3_tool_parser.py:119-123 — on ANY error, fall back to plain text.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> DeepSeekV3ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/,
    const ChatCompletionRequest& /*request*/) {
  // See DEVIATION 3 (deepseek_v3.h): full-current_text re-parse + diff.
  try {
    // Leading content = text before the OUTER begin marker; hold back a partial
    // (possibly mid-byte) begin-marker suffix so we never emit broken UTF-8.
    std::optional<std::string> content;
    {
      const std::string begin = kToolCallsBeginToken;
      const std::size_t pos = current_text.find(begin);
      std::size_t sendable_idx =
          (pos == std::string::npos)
              ? current_text.size() - PartialTagOverlap(current_text, begin)
              : pos;
      if (sendable_idx > sent_content_idx_) {
        content = current_text.substr(sent_content_idx_,
                                      sendable_idx - sent_content_idx_);
        sent_content_idx_ = sendable_idx;
      }
    }

    // Re-extract every <｜tool▁call▁begin｜>…<｜tool▁call▁end｜> region (or the
    // unterminated tail) from the full current_text.
    const std::string call_begin = kToolCallBeginToken;
    const std::string call_end = kToolCallEndToken;
    std::vector<DeltaToolCall> deltas;
    std::size_t pos = 0;
    std::size_t index = 0;
    for (;;) {
      const std::size_t s = current_text.find(call_begin, pos);
      if (s == std::string::npos) break;
      const std::size_t region_start = s + call_begin.size();
      const std::size_t e = current_text.find(call_end, region_start);
      std::string region;
      if (e != std::string::npos) {
        region = current_text.substr(region_start, e - region_start);
        pos = e + call_end.size();
      } else {
        region = current_text.substr(region_start);
        pos = current_text.size();  // last region; loop ends next iteration
      }
      const ParsedCall call = parse_region(region);

      if (index >= prev_tool_call_arr.size()) {
        prev_tool_call_arr.push_back(nlohmann::json::object());
        streamed_args_for_tool.emplace_back();
      }

      // name-first: emit the tool name exactly once.
      if (!prev_tool_call_arr[index].contains("name")) {
        if (!call.name.has_value()) break;  // can't advance until name arrives
        prev_tool_call_arr[index]["name"] = *call.name;
        DeltaToolCall d;
        d.index = static_cast<int>(index);
        d.type = "function";
        d.id = make_tool_call_id();
        d.function.name = *call.name;
        deltas.push_back(std::move(d));
      }

      // argument diff: stream only the newly-arrived suffix.
      const std::string& streamed = streamed_args_for_tool[index];
      if (call.arguments.size() > streamed.size() &&
          call.arguments.compare(0, streamed.size(), streamed) == 0) {
        std::string diff = call.arguments.substr(streamed.size());
        streamed_args_for_tool[index] = call.arguments;
        if (!diff.empty()) {
          DeltaToolCall d;
          d.index = static_cast<int>(index);
          d.function.arguments = std::move(diff);
          deltas.push_back(std::move(d));
        }
      }
      if (e == std::string::npos) break;
      ++index;
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
