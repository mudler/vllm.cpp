// Ported from: vllm/tool_parsers/mistral_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/mistral.h"

#include <cstddef>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// mistral_tool_parser.py:52 (ALPHANUMERIC) + :76-80
// (MistralToolCall.generate_random_id). Mistral tool-call ids must be
// alphanumeric with a length of 9 (D5).
std::string GenerateMistralToolCallId() {
  static constexpr char kAlnum[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<int> dist(0, 61);  // 26+26+10 - 1
  std::string id(9, '0');
  for (char& c : id) c = kAlnum[dist(rng)];
  return id;
}

// Python str.strip() over ASCII whitespace.
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
// str.replace(old, new) - replace ALL occurrences.
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

// str.split(sep) - Python semantics (a leading/trailing sep yields an empty
// piece; N separators yield N+1 pieces).
std::vector<std::string> Split(const std::string& s, const std::string& sep) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t pos = s.find(sep, start);
    if (pos == std::string::npos) {
      out.push_back(s.substr(start));
      break;
    }
    out.push_back(s.substr(start, pos - start));
    start = pos + sep.size();
  }
  return out;
}

// Extract the top-level bracketed array `[ ... ]` starting at (or after leading
// whitespace of) `text`, honouring JSON string/escape rules so braces/brackets
// inside strings do not affect nesting. Returns the substring covering the
// balanced array; on success `end_pos` is the index one past the closing ']'.
// Returns nullopt when `text` (after leading whitespace) does not start with
// '[' or the array never closes.
std::optional<std::string> ExtractBalancedArray(const std::string& text,
                                                 std::size_t* end_pos) {
  std::size_t i = 0;
  while (i < text.size() && IsSpace(text[i])) ++i;
  if (i >= text.size() || text[i] != '[') return std::nullopt;
  const std::size_t begin = i;
  int depth = 0;
  bool in_string = false, escape = false;
  for (; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '[' || c == '{') {
      ++depth;
    } else if (c == ']' || c == '}') {
      --depth;
      if (depth == 0) {
        *end_pos = i + 1;
        return text.substr(begin, i + 1 - begin);
      }
    }
  }
  return std::nullopt;  // unterminated
}

// json.loads(raw)[k] where the value is re-serialized to a compact JSON string
// (D4). Builds {name, arguments} entries. Throws on malformed JSON or a missing
// "name" (matching upstream's dict indexing which propagates a KeyError).
std::vector<std::pair<std::string, std::string>> ParseArrayToolCalls(
    const std::string& array_json) {
  const nlohmann::json arr = nlohmann::json::parse(array_json);
  std::vector<std::pair<std::string, std::string>> out;
  for (const auto& tc : arr) {
    const std::string name = tc.at("name").get<std::string>();
    // json.dumps(tool_call.get("arguments", {}), ensure_ascii=False)
    const std::string args =
        tc.contains("arguments") ? tc.at("arguments").dump() : "{}";
    out.emplace_back(name, args);
  }
  return out;
}

// mistral_tool_parser.py:136 (tool_call_regex = re.compile(r"\[{.*}\]",
// re.DOTALL)). `.` under DOTALL matches newlines; ECMAScript `.` does not, so
// [\s\S] emulates it. Greedy, as upstream.
const std::regex& ToolCallArrayRegex() {
  static const std::regex re(R"(\[\{[\s\S]*\}\])");
  return re;
}

}  // namespace

MistralToolParser::MistralToolParser(bool is_pre_v11)
    : is_pre_v11_(is_pre_v11) {}

// mistral_tool_parser.py:265 (extract_tool_calls).
ExtractedToolCallInformation MistralToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  const std::string bot = kBotToken;

  // mistral_tool_parser.py:290-293 - no [BOT] => plain text response.
  if (model_output.find(bot) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  // mistral_tool_parser.py:295-297.
  const std::vector<std::string> parts = Split(model_output, bot);
  const std::string content = parts[0];
  std::vector<std::string> raw_tool_calls(parts.begin() + 1, parts.end());

  // Collected {name, arguments-string} pairs (mistral_tool_parser.py:364-373).
  std::vector<std::pair<std::string, std::string>> tool_calls;

  if (!is_pre_v11_) {
    // >= v11: content[BOT]name1{args1}[BOT]name2{args2}
    // mistral_tool_parser.py:300-314.
    for (const std::string& raw_tool_call : raw_tool_calls) {
      if (raw_tool_call.find('{') == std::string::npos) continue;
      const std::size_t end_name = raw_tool_call.find('{');
      std::string tool_name = raw_tool_call.substr(0, end_name);
      const std::string args = raw_tool_call.substr(end_name);
      // HF tokenizers may include [ARGS] in the text.
      tool_name = ReplaceAll(tool_name, "[ARGS]", "");
      tool_calls.emplace_back(tool_name, args);
    }
  } else {
    // < v11: content[BOT] [{tool_call1}, {tool_call2}]
    // mistral_tool_parser.py:317-362.
    if (raw_tool_calls.size() != 1) {
      throw std::runtime_error(
          "Only one BOT token should have been outputted, but got " +
          model_output);
    }
    const std::string stringified = Strip(raw_tool_calls[0]);

    bool parsed = false;
    // raw_decode: parse the first valid JSON array value, ignoring trailing
    // tokens the model may emit after the array (mistral_tool_parser.py:324-328).
    std::size_t end_pos = 0;
    if (const std::optional<std::string> arr =
            ExtractBalancedArray(stringified, &end_pos)) {
      try {
        tool_calls = ParseArrayToolCalls(*arr);
        parsed = true;
      } catch (const std::exception&) {
        parsed = false;
      }
    }

    if (!parsed) {
      // Regex fallback: \[{.*}\] anywhere in the string
      // (mistral_tool_parser.py:329-345).
      std::smatch m;
      bool fallback_ok = false;
      if (std::regex_search(stringified, m, ToolCallArrayRegex())) {
        try {
          tool_calls = ParseArrayToolCalls(m[0].str());
          fallback_ok = true;
        } catch (const std::exception&) {
          fallback_ok = false;
        }
      }
      if (!fallback_ok) {
        // mistral_tool_parser.py:345-351.
        return ExtractedToolCallInformation{false, {}, stringified};
      }
    }
  }

  // mistral_tool_parser.py:364-379 - assemble MistralToolCalls; content is None
  // when it is blank after stripping.
  ExtractedToolCallInformation info;
  info.tools_called = true;
  for (const auto& [name, args] : tool_calls) {
    ToolCall tc;
    tc.id = GenerateMistralToolCallId();
    tc.type = "function";
    tc.function.name = name;
    tc.function.arguments = args;  // already a JSON string
    info.tool_calls.push_back(std::move(tc));
  }
  if (!Strip(content).empty()) info.content = content;
  return info;
}

// mistral_tool_parser.py:381 (extract_tool_calls_streaming).
std::optional<DeltaMessage> MistralToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  const std::string bot = kBotToken;
  // D2: text-only [BOT] detection over current_text.
  const bool has_bot_token = current_text.find(bot) != std::string::npos;
  if (!has_bot_token) {
    // mistral_tool_parser.py:394-397 - not a tool call yet; stream as content.
    // pre-v11 re-parse (D3) tracks how much leading content has already been
    // streamed so the [BOT]-transition delta does not re-emit it: every char of
    // current_text is leading content while no [BOT] has appeared.
    if (is_pre_v11_) pre11_content_sent_ = current_text.size();
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }
  try {
    if (is_pre_v11_) return extract_pre_v11_streaming(current_text);
    return extract_v11_streaming(delta_text);
  } catch (const std::exception&) {
    // mistral_tool_parser.py:411-413 - on error, drop the delta.
    return std::nullopt;
  }
}

// mistral_tool_parser.py:415 (_extract_tool_calls_streaming). v11 name-first.
std::optional<DeltaMessage> MistralToolParser::extract_v11_streaming(
    const std::string& delta_text_in) {
  const std::string bot = kBotToken;
  std::string delta_text = delta_text_in;
  std::string additional_content;

  if (streaming_state_ == StreamingState::WAITING_FOR_TOOL_START) {
    // mistral_tool_parser.py:426-434.
    if (delta_text.find(bot) == std::string::npos) {
      DeltaMessage m;
      m.content = delta_text;
      return m;
    }
    if (delta_text.rfind(bot, 0) != 0) {  // not startswith(bot)
      const std::vector<std::string> pieces = Split(delta_text, bot);
      additional_content += pieces[0];
      // bot_token + "".join(pieces[1:])
      std::string rejoined = bot;
      for (std::size_t i = 1; i < pieces.size(); ++i) {
        rejoined += pieces[i];
        if (i + 1 < pieces.size()) rejoined += bot;
      }
      delta_text = rejoined;
    }
  }

  const std::vector<DeltaToolCall> delta_tool_calls =
      generate_delta_tool_call(delta_text);

  // mistral_tool_parser.py:437-452.
  if (additional_content.empty() && delta_tool_calls.empty()) {
    if (streaming_state_ == StreamingState::PARSING_ARGUMENTS ||
        streaming_state_ == StreamingState::PARSING_ARGUMENTS_COMPLETED ||
        streaming_state_ == StreamingState::TOOL_COMPLETE ||
        streaming_state_ == StreamingState::ALL_TOOLS_COMPLETE) {
      // Empty DeltaMessage so finish_reason gets set.
      return DeltaMessage{};
    }
    return std::nullopt;
  }

  DeltaMessage delta;
  if (!additional_content.empty()) delta.content = additional_content;
  if (!delta_tool_calls.empty()) delta.tool_calls = delta_tool_calls;
  return delta;
}

// mistral_tool_parser.py:462 (_generate_delta_tool_call).
std::vector<DeltaToolCall> MistralToolParser::generate_delta_tool_call(
    std::string delta_text) {
  const std::string bot = kBotToken;
  if (delta_text.empty()) return {};

  std::optional<std::string> delta_function_name;
  std::optional<std::string> tool_id;

  // mistral_tool_parser.py:467-475 - a new [BOT] starts a new tool.
  if (streaming_state_ != StreamingState::PARSING_NAME &&
      streaming_state_ != StreamingState::PARSING_ARGUMENTS &&
      delta_text.rfind(bot, 0) == 0) {  // startswith(bot)
    current_tool_id += 1;
    streamed_args_for_tool.emplace_back();
    prev_tool_call_arr.push_back(nlohmann::json::object());
    streaming_state_ = StreamingState::PARSING_NAME;
    delta_text.erase(0, bot.size());  // replace(bot, "", 1)
  }

  if (streaming_state_ == StreamingState::PARSING_NAME) {
    if (!current_tool_name_.has_value()) current_tool_name_ = "";
    // mistral_tool_parser.py:481-495 - the name ends where the args '{' starts.
    const std::size_t brace = delta_text.find('{');
    if (brace != std::string::npos) {
      tool_id = GenerateMistralToolCallId();
      delta_function_name = delta_text.substr(0, brace);
      *current_tool_name_ += *delta_function_name;
      *current_tool_name_ = ReplaceAll(*current_tool_name_, "[ARGS]", "");
      prev_tool_call_arr[current_tool_id]["name"] = *current_tool_name_;
      delta_text = delta_text.substr(delta_function_name->size());
      streaming_state_ = StreamingState::PARSING_ARGUMENTS;
    } else {
      // Wait for the name to complete before sending it.
      *current_tool_name_ += delta_text;
      return {};
    }
  }

  if (streaming_state_ == StreamingState::PARSING_ARGUMENTS) {
    // mistral_tool_parser.py:496-525.
    std::optional<std::string> next_function_text;
    std::string delta_arguments;
    if (delta_text.find(bot) != std::string::npos) {
      // current tool call is over
      delta_arguments = Split(delta_text, bot)[0];
      next_function_text = delta_text.substr(delta_arguments.size());
      streaming_state_ = StreamingState::TOOL_COMPLETE;
    } else {
      delta_arguments = delta_text;
    }
    streamed_args_for_tool[current_tool_id] += delta_arguments;
    prev_tool_call_arr[current_tool_id]["arguments"] =
        streamed_args_for_tool[current_tool_id];

    std::vector<DeltaToolCall> ret;
    const bool name_truthy =
        current_tool_name_.has_value() && !current_tool_name_->empty();
    if (name_truthy || !delta_arguments.empty()) {
      DeltaToolCall d;
      d.index = current_tool_id;
      d.type = "function";
      d.id = tool_id;
      // DeltaFunctionCall(...).model_dump(exclude_none=True): name only when
      // present, arguments always (even "").
      if (current_tool_name_.has_value()) d.function.name = current_tool_name_;
      d.function.arguments = delta_arguments;
      ret.push_back(std::move(d));
      current_tool_name_.reset();
    }
    if (next_function_text.has_value() && !next_function_text->empty()) {
      std::vector<DeltaToolCall> more =
          generate_delta_tool_call(*next_function_text);
      ret.insert(ret.end(), std::make_move_iterator(more.begin()),
                 std::make_move_iterator(more.end()));
    }
    return ret;
  }

  return {};  // should not happen
}

// pre-v11 streaming (D3): re-parse current_text + diff against sent state.
std::optional<DeltaMessage> MistralToolParser::extract_pre_v11_streaming(
    const std::string& current_text) {
  const std::string bot = kBotToken;
  const std::size_t bot_pos = current_text.find(bot);
  // has_bot guaranteed by the caller.

  std::string new_content;

  // Leading content before [BOT] (streamed once, incrementally).
  const std::string leading = current_text.substr(0, bot_pos);
  if (leading.size() > pre11_content_sent_) {
    new_content += leading.substr(pre11_content_sent_);
    pre11_content_sent_ = leading.size();
  }

  const std::string after_bot = current_text.substr(bot_pos + bot.size());

  // Locate the top-level array; find its complete inner objects.
  std::vector<DeltaToolCall> tool_call_deltas;
  std::size_t array_end = 0;
  bool array_closed = false;
  {
    std::size_t i = 0;
    while (i < after_bot.size() && IsSpace(after_bot[i])) ++i;
    if (i < after_bot.size() && after_bot[i] == '[') {
      ++i;  // enter the array
      int depth = 0;  // brace depth of the current inner object
      bool in_string = false, escape = false;
      std::size_t obj_start = std::string::npos;
      std::size_t tool_index = 0;
      for (; i < after_bot.size(); ++i) {
        const char c = after_bot[i];
        if (in_string) {
          if (escape) {
            escape = false;
          } else if (c == '\\') {
            escape = true;
          } else if (c == '"') {
            in_string = false;
          }
          continue;
        }
        if (c == '"') {
          in_string = true;
        } else if (c == '{') {
          if (depth == 0) obj_start = i;
          ++depth;
        } else if (c == '}') {
          --depth;
          if (depth == 0 && obj_start != std::string::npos) {
            // A complete inner object [obj_start, i].
            const std::string obj = after_bot.substr(obj_start, i + 1 - obj_start);
            const std::size_t idx = tool_index++;
            if (idx >= prev_tool_call_arr.size()) {
              prev_tool_call_arr.push_back(nlohmann::json::object());
              streamed_args_for_tool.emplace_back();
            }
            if (!prev_tool_call_arr[idx].contains("name")) {
              try {
                const nlohmann::json parsed = nlohmann::json::parse(obj);
                const std::string name = parsed.at("name").get<std::string>();
                const std::string args = parsed.contains("arguments")
                                             ? parsed.at("arguments").dump()
                                             : "{}";
                prev_tool_call_arr[idx]["name"] = name;
                prev_tool_call_arr[idx]["arguments"] = args;
                streamed_args_for_tool[idx] = args;
                DeltaToolCall d;
                d.index = static_cast<int>(idx);
                d.type = "function";
                d.id = GenerateMistralToolCallId();
                d.function.name = name;
                d.function.arguments = args;
                tool_call_deltas.push_back(std::move(d));
                if (static_cast<int>(idx) > current_tool_id)
                  current_tool_id = static_cast<int>(idx);
              } catch (const std::exception&) {
                // Malformed object: leave unsent (drop, like upstream's guard).
              }
            }
            obj_start = std::string::npos;
          }
        } else if (c == ']' && depth == 0) {
          array_closed = true;
          array_end = i + 1;  // index within after_bot, one past ']'
          break;
        }
      }
    }
  }

  // Trailing content after the closing ']' (streamed once, incrementally).
  if (array_closed) {
    const std::string trailing = after_bot.substr(array_end);
    if (trailing.size() > pre11_trailing_sent_) {
      new_content += trailing.substr(pre11_trailing_sent_);
      pre11_trailing_sent_ = trailing.size();
    }
  }

  if (!new_content.empty() || !tool_call_deltas.empty()) {
    DeltaMessage msg;
    if (!new_content.empty()) msg.content = new_content;
    if (!tool_call_deltas.empty()) msg.tool_calls = std::move(tool_call_deltas);
    return msg;
  }
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
