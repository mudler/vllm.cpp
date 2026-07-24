// Ported from: vllm/tool_parsers/lfm2_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/lfm2.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

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

std::string Lstrip(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && IsWs(s[b])) ++b;
  return s.substr(b);
}

std::string ReplaceAll(std::string s, char from, char to) {
  for (char& c : s) {
    if (c == from) c = to;
  }
  return s;
}

const std::string kStart = Lfm2ToolParser::kToolCallStart;
const std::string kEnd = Lfm2ToolParser::kToolCallEnd;

// lfm2_tool_parser.py:97-106 (_strip_echo). Drop any orphan <|tool_call_end|>
// (and the preceding echoed body) from trailing content: everything through the
// LAST such orphan is model garbage, not user content.
std::string StripEcho(const std::string& raw_after) {
  const std::size_t last = raw_after.rfind(kEnd);
  if (last != std::string::npos) return raw_after.substr(last + kEnd.size());
  return raw_after;
}

bool StartsWith(const std::string& s, char c) { return !s.empty() && s[0] == c; }

}  // namespace

ExtractedToolCallInformation Lfm2ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // lfm2_tool_parser.py:108-142 (_extract_tool_call_text) inlined.
  const std::size_t start_idx = model_output.find(kStart);
  if (start_idx == std::string::npos) {
    // No sentinel -> plain content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  std::string tool_text;
  std::optional<std::string> content;
  const std::size_t end_idx = model_output.find(kEnd, start_idx);
  if (end_idx == std::string::npos) {
    // Incomplete: treat everything after start as tool call.
    tool_text = model_output.substr(start_idx + kStart.size());
    const std::string content_before = Strip(model_output.substr(0, start_idx));
    if (!content_before.empty()) content = content_before;
  } else {
    tool_text = model_output.substr(start_idx + kStart.size(),
                                    end_idx - (start_idx + kStart.size()));
    const std::string content_before = Strip(model_output.substr(0, start_idx));
    const std::string content_after =
        Strip(StripEcho(model_output.substr(end_idx + kEnd.size())));
    std::vector<std::string> parts;
    if (!content_before.empty()) parts.push_back(content_before);
    if (!content_after.empty()) parts.push_back(content_after);
    if (!parts.empty()) {
      std::string joined;
      for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) joined += "\n";
        joined += parts[i];
      }
      content = joined;
    }
  }

  tool_text = Strip(tool_text);

  // lfm2_tool_parser.py:156-193 - regex gate (r"\[.*\]$") + ast parse collapse
  // into the strict parse_call_list. On any failure: WHOLE output as content.
  const std::optional<std::vector<PyCall>> calls =
      pythonic_core::parse_call_list(tool_text);
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
  info.content = content;  // may be nullopt
  return info;
}

std::optional<DeltaMessage> Lfm2ToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // lfm2_tool_parser.py:206-207 - start sentinel not seen yet -> plain content.
  if (current_text.find(kStart) == std::string::npos) {
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }

  // lfm2_tool_parser.py:214-219 - leading content (before the start sentinel)
  // that arrived in THIS delta and has not yet been streamed.
  std::string leading_content;
  if (previous_text.find(kStart) == std::string::npos) {
    const std::size_t start_idx = current_text.find(kStart);
    leading_content =
        current_text.substr(previous_text.size(), start_idx - previous_text.size());
  }

  const bool has_end_in_current = current_text.find(kEnd) != std::string::npos;
  const bool has_end_in_previous = previous_text.find(kEnd) != std::string::npos;

  // lfm2_tool_parser.py:233-250 - trailing content (after the first end
  // sentinel) not yet streamed, with echo suppression.
  std::string trailing_content;
  if (has_end_in_current) {
    const std::size_t end_idx = current_text.find(kEnd) + kEnd.size();
    const std::string full_trailing = current_text.substr(end_idx);
    const std::string stripped_trailing = StripEcho(full_trailing);
    std::string final_trailing;
    if (stripped_trailing == full_trailing) {
      // No second end token yet - possibly mid-echo.
      const std::string lstripped = Lstrip(full_trailing);
      if (StartsWith(lstripped, '[') || StartsWith(lstripped, '<')) {
        final_trailing = trailing_emitted_;  // suspect echo; hold off.
      } else {
        final_trailing = full_trailing;
      }
    } else {
      final_trailing = stripped_trailing;
    }
    if (final_trailing.size() >= trailing_emitted_.size() &&
        final_trailing.compare(0, trailing_emitted_.size(), trailing_emitted_) == 0) {
      trailing_content = final_trailing.substr(trailing_emitted_.size());
    }
    trailing_emitted_ = final_trailing;
  }

  // lfm2_tool_parser.py:254-257 - tools already parsed in a prior delta: just
  // stream any newly-arrived trailing content.
  if (has_end_in_current && !prev_tool_call_arr.empty() && has_end_in_previous) {
    DeltaMessage m;
    m.content = trailing_content;  // "" when nothing new
    return m;
  }

  // lfm2_tool_parser.py:260-263 - the pythonic text between start and end.
  std::string tool_text = current_text.substr(current_text.find(kStart) + kStart.size());
  const std::size_t te = tool_text.find(kEnd);
  if (te != std::string::npos) tool_text = tool_text.substr(0, te);

  // lfm2_tool_parser.py:265-271 (_content_only_or_none).
  auto content_only_or_none = [&]() -> std::optional<DeltaMessage> {
    const std::string combined = leading_content + trailing_content;
    if (combined.empty()) return std::nullopt;
    DeltaMessage m;
    m.content = combined;
    return m;
  };

  try {
    // lfm2_tool_parser.py:274-277.
    const std::optional<std::pair<std::string, std::string>> mv =
        pythonic_core::make_valid_python(tool_text);
    if (!mv.has_value()) return content_only_or_none();
    const std::string& added_text = mv->second;

    const std::optional<std::vector<PyCall>> calls =
        pythonic_core::parse_call_list(mv->first);
    if (!calls.has_value()) return content_only_or_none();

    std::vector<DeltaToolCall> tool_deltas;
    for (std::size_t index = 0; index < calls->size(); ++index) {
      if (static_cast<int>(index) < current_tool_id) continue;  // current_tool_index
      current_tool_id = static_cast<int>(index);
      if (streamed_args_for_tool.size() == index) {
        streamed_args_for_tool.emplace_back();
      }

      // lfm2_tool_parser.py:299-308. Same withheld-suffix bookkeeping as
      // pythonic (tool_text keeps the outer "[...]" list brackets): ")]" / [:-2].
      const bool new_call_complete = (index < calls->size() - 1) ||
                                     (added_text.find(")]") == std::string::npos);
      if (new_call_complete) ++current_tool_id;

      std::string withheld_suffix;
      if (!new_call_complete) {
        withheld_suffix = added_text.substr(0, added_text.size() - 2);  // [:-2]
        if (added_text[added_text.size() - 2] == ')') withheld_suffix += '}';
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

    // lfm2_tool_parser.py:324-325 - HACK: mark prev_tool_call_arr non-empty so
    // the serving layer sets finish_reason=tool_calls AND so the early-return
    // branch above fires on the next end-carrying delta.
    if (!tool_deltas.empty() && prev_tool_call_arr.empty()) {
      prev_tool_call_arr.push_back(
          nlohmann::json{{"arguments", nlohmann::json::object()}});
    }

    const std::string combined_content = leading_content + trailing_content;

    if (!tool_deltas.empty() || !combined_content.empty()) {
      DeltaMessage m;
      if (!combined_content.empty()) m.content = combined_content;
      m.tool_calls = std::move(tool_deltas);
      return m;
    }
    if (added_text.empty() && current_tool_id > 0) {
      DeltaMessage m;
      m.content = "";
      return m;
    }
    return std::nullopt;
  } catch (const pythonic_core::ToolDeltaError&) {
    return content_only_or_none();
  } catch (const std::exception&) {
    return content_only_or_none();
  }
}

}  // namespace vllm::entrypoints::openai
