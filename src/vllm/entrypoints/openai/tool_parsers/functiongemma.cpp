// Ported from: vllm/tool_parsers/functiongemma_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/functiongemma.h"

#include <cctype>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

const std::string kStart = FunctionGemmaToolParser::kToolCallStart;
const std::string kEnd = FunctionGemmaToolParser::kToolCallEnd;
const std::string kEsc = "<escape>";

bool IsWs(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
bool IsWordChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
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

std::size_t CountOccurrences(const std::string& hay, const std::string& needle) {
  if (needle.empty()) return 0;
  std::size_t n = 0;
  std::size_t p = 0;
  while ((p = hay.find(needle, p)) != std::string::npos) {
    ++n;
    p += needle.size();
  }
  return n;
}

// functiongemma_tool_parser.py:70-74 - json.loads(value) else keep raw string.
ojson DecodeValue(const std::string& value) {
  try {
    return ojson::parse(value);
  } catch (const std::exception&) {
    return ojson(value);
  }
}

// A tool call recovered by the tool_call_regex analogue. `complete` mirrors the
// regex alternation: true means the fully-closed alt1 matched
// (<start_function_call>call:NAME{ARGS}<end_function_call>), false the trailing
// alt2 (<start_function_call>call:NAME{ARGS...  with no closing tag yet).
struct FgCall {
  std::string name;
  std::string args;
  bool complete = false;
};

// functiongemma_tool_parser.py:42-46 (tool_call_regex.findall) reproduced with a
// non-greedy substring scan. Both alternatives require the literal prefix
// "<start_function_call>call:", a \w+ name, then "{".
std::vector<FgCall> FindAllCalls(const std::string& text) {
  std::vector<FgCall> out;
  const std::string close = "}" + kEnd;  // first "}<end_function_call>"
  std::size_t pos = 0;
  while (true) {
    const std::size_t s = text.find(kStart, pos);
    if (s == std::string::npos) break;
    std::size_t p = s + kStart.size();
    if (text.compare(p, 5, "call:") != 0) {
      pos = s + kStart.size();
      continue;
    }
    p += 5;  // past "call:"
    const std::size_t name_start = p;
    while (p < text.size() && IsWordChar(text[p])) ++p;
    if (p == name_start) {  // \w+ needs >= 1 char
      pos = s + kStart.size();
      continue;
    }
    const std::string name = text.substr(name_start, p - name_start);
    if (p >= text.size() || text[p] != '{') {  // need "{"
      pos = s + kStart.size();
      continue;
    }
    const std::size_t args_start = p + 1;
    const std::size_t c = text.find(close, args_start);
    if (c != std::string::npos) {
      out.push_back({name, text.substr(args_start, c - args_start), true});
      pos = c + close.size();
    } else {
      out.push_back({name, text.substr(args_start), false});
      pos = text.size();
    }
  }
  return out;
}

}  // namespace

ojson FunctionGemmaToolParser::ParseArguments(const std::string& args_str) const {
  // functiongemma_tool_parser.py:62-76 (_parse_arguments) via an arg_regex
  // analogue: (\w+):<escape>(.*?)<escape> findall, non-greedy value.
  ojson obj = ojson::object();
  if (args_str.empty()) return obj;
  const std::string colon_esc = ":" + kEsc;  // ":<escape>"
  std::size_t pos = 0;
  while (pos < args_str.size()) {
    const std::size_t c = args_str.find(colon_esc, pos);
    if (c == std::string::npos) break;
    // key = maximal run of word chars ending immediately before the ':'.
    std::size_t k_start = c;
    while (k_start > 0 && IsWordChar(args_str[k_start - 1])) --k_start;
    const std::string key = args_str.substr(k_start, c - k_start);
    if (key.empty()) {
      pos = c + colon_esc.size();
      continue;
    }
    const std::size_t val_start = c + colon_esc.size();
    const std::size_t val_end = args_str.find(kEsc, val_start);
    if (val_end == std::string::npos) break;  // non-greedy needs a closing <escape>
    const std::string value = args_str.substr(val_start, val_end - val_start);
    obj[key] = DecodeValue(value);
    pos = val_end + kEsc.size();
  }
  return obj;
}

ExtractedToolCallInformation FunctionGemmaToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // functiongemma_tool_parser.py:91-94 - no start token -> plain content.
  if (model_output.find(kStart) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    const std::vector<FgCall> matches = FindAllCalls(model_output);
    if (matches.empty()) {
      return ExtractedToolCallInformation{false, {}, model_output};
    }

    std::vector<ToolCall> tool_calls;
    for (const FgCall& m : matches) {
      if (m.name.empty()) continue;  // upstream: skip empty func_name.
      const ojson arguments = ParseArguments(m.args);
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = m.name;
      tc.function.arguments = arguments.dump();
      tool_calls.push_back(std::move(tc));
    }

    if (tool_calls.empty()) {
      return ExtractedToolCallInformation{false, {}, model_output};
    }

    // functiongemma_tool_parser.py:126-129 - content before the first start
    // token, stripped; None when empty.
    const std::size_t content_end = model_output.find(kStart);
    std::optional<std::string> content;
    if (content_end > 0 && content_end != std::string::npos) {
      const std::string c = Strip(model_output.substr(0, content_end));
      if (!c.empty()) content = c;
    }

    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    info.content = content;
    return info;
  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::string FunctionGemmaToolParser::BufferDeltaText(const std::string& delta_text) {
  // functiongemma_tool_parser.py:147-165 (_buffer_delta_text).
  const std::string combined = buffered_delta_text_ + delta_text;

  auto ends_with = [](const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };

  if (ends_with(combined, kStart) || ends_with(combined, kEnd)) {
    buffered_delta_text_.clear();
    return combined;
  }

  for (const std::string* tag : {&kStart, &kEnd}) {
    for (std::size_t i = 1; i < tag->size(); ++i) {
      if (ends_with(combined, tag->substr(0, i))) {
        buffered_delta_text_ = combined.substr(combined.size() - i);
        return combined.substr(0, combined.size() - i);
      }
    }
  }

  buffered_delta_text_.clear();
  return combined;
}

std::optional<DeltaMessage> FunctionGemmaToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text_raw, const ChatCompletionRequest& /*request*/) {
  // functiongemma_tool_parser.py:177-183. Buffer partial tags, then rebuild the
  // current text from the (buffered) delta (upstream ignores the passed
  // current_text and reconstructs it the same way).
  const std::string delta_text = BufferDeltaText(delta_text_raw);
  const std::string current_text = previous_text + delta_text;

  if (current_text.find(kStart) == std::string::npos) {
    if (!delta_text.empty()) {
      DeltaMessage m;
      m.content = delta_text;
      return m;
    }
    return std::nullopt;
  }

  try {
    const std::size_t start_count = CountOccurrences(current_text, kStart);
    const std::size_t end_count = CountOccurrences(current_text, kEnd);
    const std::size_t prev_start_count = CountOccurrences(previous_text, kStart);
    const std::size_t prev_end_count = CountOccurrences(previous_text, kEnd);

    // functiongemma_tool_parser.py:195-201 - starting a new function call.
    if (start_count > prev_start_count && start_count > end_count) {
      ++current_tool_id;
      current_tool_name_sent_ = false;
      streamed_args_for_tool.emplace_back();
      prev_tool_call_arr.push_back(nlohmann::json::object());
      return std::nullopt;
    }

    // functiongemma_tool_parser.py:204-268 - in the middle of a function call.
    if (start_count > end_count) {
      const std::size_t last_start = current_text.rfind(kStart);
      const std::string partial_call =
          current_text.substr(last_start + kStart.size());
      if (StartsWith(partial_call, "call:")) {
        const std::string func_part = partial_call.substr(5);
        const std::size_t brace = func_part.find('{');
        if (brace != std::string::npos) {
          const std::string func_name = func_part.substr(0, brace);
          const std::string args_part = func_part.substr(brace + 1);

          if (!current_tool_name_sent_ && !func_name.empty()) {
            current_tool_name_sent_ = true;
            prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
                nlohmann::json{{"name", func_name},
                               {"arguments", nlohmann::json::object()}};
            DeltaToolCall tc;
            tc.index = current_tool_id;
            tc.type = "function";
            tc.id = make_tool_call_id();
            tc.function.name = func_name;
            DeltaMessage m;
            m.tool_calls = std::vector<DeltaToolCall>{std::move(tc)};
            return m;
          }

          if (current_tool_name_sent_ && !args_part.empty()) {
            const ojson current_args = ParseArguments(args_part);
            if (!current_args.empty()) {
              const std::string current_args_json = current_args.dump();
              const std::string& prev_streamed =
                  streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)];
              if (current_args_json.size() > prev_streamed.size()) {
                const std::string diff =
                    current_args_json.substr(prev_streamed.size());
                streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] =
                    current_args_json;
                prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)]
                                  ["arguments"] =
                    nlohmann::json::parse(current_args_json);
                DeltaToolCall tc;
                tc.index = current_tool_id;
                tc.function.arguments = diff;
                DeltaMessage m;
                m.tool_calls = std::vector<DeltaToolCall>{std::move(tc)};
                return m;
              }
            }
          }
        }
      }
      return std::nullopt;
    }

    // functiongemma_tool_parser.py:271-306 - function call just ended.
    if (end_count > prev_end_count) {
      if (current_tool_id >= 0 &&
          static_cast<std::size_t>(current_tool_id) < prev_tool_call_arr.size()) {
        const std::vector<FgCall> all_calls = FindAllCalls(current_text);
        ojson args = ojson::object();
        if (static_cast<std::size_t>(current_tool_id) < all_calls.size()) {
          const FgCall& m = all_calls[static_cast<std::size_t>(current_tool_id)];
          if (m.complete) {  // match[0] truthy (fully-closed call)
            args = ParseArguments(m.args);
            prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)]
                              ["arguments"] = nlohmann::json::parse(args.dump());
          }
        }
        if (!args.empty()) {
          const std::string args_json = args.dump();
          const std::string& prev_streamed =
              streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)];
          if (args_json.size() > prev_streamed.size()) {
            const std::string diff = args_json.substr(prev_streamed.size());
            streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] =
                args_json;
            DeltaToolCall tc;
            tc.index = current_tool_id;
            tc.function.arguments = diff;
            DeltaMessage m;
            m.tool_calls = std::vector<DeltaToolCall>{std::move(tc)};
            return m;
          }
        }
      }
      return std::nullopt;
    }

    if (!delta_text.empty()) {
      DeltaMessage m;
      m.content = delta_text;
      return m;
    }
    return std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
