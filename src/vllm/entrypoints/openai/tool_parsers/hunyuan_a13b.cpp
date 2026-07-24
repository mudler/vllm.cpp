// Ported from: vllm/tool_parsers/hunyuan_a13b_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/hunyuan_a13b.h"

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

namespace vllm::entrypoints::openai {

namespace {

// hunyuan_a13b_tool_parser.py:50 - <tool_calls>([\s\S]*?)</tool_calls>. ECMAScript
// `.` is not DOTALL so [\s\S] emulates it; `*?` non-greedy.
const std::regex& ToolCallsRegex() {
  static const std::regex re(R"(<tool_calls>([\s\S]*?)</tool_calls>)");
  return re;
}

// re.finditer(r"<think>(.*?)</think>", DOTALL).
const std::regex& ThinkRegex() {
  static const std::regex re(R"(<think>([\s\S]*?)</think>)");
  return re;
}

// hunyuan_a13b_tool_parser.py:54 - "name"\s*:\s*"([^"]+)".
const std::regex& ToolNameRegex() {
  static const std::regex re(R"rx("name"\s*:\s*"([^"]+)")rx");
  return re;
}

// hunyuan_a13b_tool_parser.py:56 - "name"..."arguments"\s*:\s*\{\s*\} (empty args).
const std::regex& ToolEmptyArgRegex() {
  static const std::regex re(
      R"("name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\})");
  return re;
}

// hunyuan_a13b_tool_parser.py:61 - "name"..."arguments"\s*:\s*(\{one-level nest\}).
const std::regex& ToolNonEmptyArgRegex() {
  static const std::regex re(
      R"("name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}))");
  return re;
}

// Python str.strip() over ASCII whitespace.
std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && static_cast<unsigned char>(s[b]) <= ' ') ++b;
  while (e > b && static_cast<unsigned char>(s[e - 1]) <= ' ') --e;
  return s.substr(b, e - b);
}

// str.split(sep)[1..] - the substring AFTER the first occurrence of `sep`.
std::string AfterFirst(const std::string& s, const std::string& sep) {
  const std::size_t p = s.find(sep);
  if (p == std::string::npos) return "";
  return s.substr(p + sep.size());
}

// str.replace(old, "", 1) - remove the FIRST occurrence only.
std::string ReplaceFirst(std::string s, const std::string& from) {
  const std::size_t p = s.find(from);
  if (p != std::string::npos) s.erase(p, from.size());
  return s;
}

// Serialize a parsed argument value the way upstream does: json.dumps(dict,
// ensure_ascii=False) for a dict, else the value passed straight through (kept as
// its string form). nlohmann::ordered_json::dump() preserves key order + keeps
// UTF-8 bytes (no \u escaping) == ensure_ascii=False.
std::string DumpArguments(const nlohmann::ordered_json& args) {
  if (args.is_string()) return args.get<std::string>();
  return args.dump();
}

}  // namespace

std::pair<std::optional<std::string>, std::optional<std::string>>
HunyuanA13BToolParser::preprocess_model_output(
    const std::string& model_output) const {
  // Collect the <think>...</think> spans once.
  std::vector<std::pair<std::size_t, std::size_t>> think_regions;
  for (auto it = std::sregex_iterator(model_output.begin(), model_output.end(),
                                      ThinkRegex());
       it != std::sregex_iterator(); ++it) {
    think_regions.emplace_back(static_cast<std::size_t>(it->position()),
                               static_cast<std::size_t>(it->position() +
                                                        it->length()));
  }

  for (auto it = std::sregex_iterator(model_output.begin(), model_output.end(),
                                      ToolCallsRegex());
       it != std::sregex_iterator(); ++it) {
    const std::size_t start = static_cast<std::size_t>(it->position());
    const std::size_t end = static_cast<std::size_t>(it->position() +
                                                     it->length());
    bool in_think = false;
    for (const auto& [t_start, t_end] : think_regions) {
      if (start > t_start && end < t_end) {
        in_think = true;
        break;
      }
    }
    if (in_think) continue;
    const std::string content = model_output.substr(0, start);
    const std::string tool_calls_content = Strip((*it)[1].str());
    // json.loads must succeed for this block to be accepted.
    if (nlohmann::json::accept(tool_calls_content)) {
      return {content, tool_calls_content};
    }
    // else: try the next match (upstream `continue`).
  }
  return {model_output, std::nullopt};
}

ExtractedToolCallInformation HunyuanA13BToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  try {
    auto [content_opt, potential] = preprocess_model_output(model_output);

    if (!potential.has_value()) {
      // hunyuan_a13b_tool_parser.py:110-117 - no function call.
      std::optional<std::string> content = content_opt;
      if (content.has_value() && !content->empty()) {
        content = ReplaceFirst(*content, "助手：");  // "助手："
      }
      return ExtractedToolCallInformation{false, {}, content};
    }

    const nlohmann::ordered_json tool_calls_data =
        nlohmann::ordered_json::parse(*potential);

    // hunyuan_a13b_tool_parser.py:123-129 - must be an array.
    if (!tool_calls_data.is_array()) {
      std::optional<std::string> content = content_opt;
      if (!content.has_value() || content->empty()) content = model_output;
      return ExtractedToolCallInformation{false, {}, content};
    }

    std::vector<ToolCall> tool_calls;
    for (const nlohmann::ordered_json& call : tool_calls_data) {
      if (!call.is_object() || !call.contains("name") ||
          !call.contains("arguments")) {
        continue;
      }
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = call.at("name").get<std::string>();
      tc.function.arguments = DumpArguments(call.at("arguments"));
      tool_calls.push_back(std::move(tc));
    }

    // hunyuan_a13b_tool_parser.py:155-157 - blank content collapses to None.
    std::optional<std::string> content = content_opt;
    if (!content.has_value() || Strip(*content).empty()) content = std::nullopt;

    ExtractedToolCallInformation info;
    info.tools_called = !tool_calls.empty();
    info.tool_calls = std::move(tool_calls);
    info.content = content;
    return info;

  } catch (const std::exception&) {
    // hunyuan_a13b_tool_parser.py:165-168 - any error -> whole output as content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

void HunyuanA13BToolParser::try_parse_json_tools(const std::string& text) {
  // hunyuan_a13b_tool_parser.py:221-227 - only update prev_tool_call_arr when the
  // WHOLE text parses to a list.
  if (!nlohmann::json::accept(text)) return;
  nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(text);
  if (parsed.is_array()) {
    hy_prev_tool_call_arr.clear();
    for (const nlohmann::ordered_json& e : parsed)
      hy_prev_tool_call_arr.push_back(e);
  }
}

std::optional<DeltaMessage> HunyuanA13BToolParser::handle_test_compatibility(
    const std::string& current_text) {
  // hunyuan_a13b_tool_parser.py:229-265. Inert on a fresh parser (see header):
  // current_tools_sent starts empty so this guard never fires. Kept for fidelity.
  if (hy_current_tools_sent.size() == 1 && hy_current_tools_sent[0] == false) {
    std::smatch m;
    if (std::regex_search(current_text, m, ToolNameRegex())) {
      const std::string function_name = m[1].str();
      DeltaToolCall d;
      d.index = 0;
      d.type = "function";
      d.id = make_tool_call_id();
      d.function.name = function_name;
      DeltaMessage msg;
      msg.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
      hy_current_tools_sent = {true};
      hy_current_tool_id = 0;
      hy_current_tool_index = 0;
      if (hy_sent_tools.empty()) {
        hy_sent_tools.push_back(SentTool{true, false, ""});
      } else {
        hy_sent_tools[0].sent_name = true;
      }
      return msg;
    }
  }
  return std::nullopt;
}

void HunyuanA13BToolParser::ensure_state_arrays(int tool_count) {
  while (static_cast<int>(hy_sent_tools.size()) < tool_count)
    hy_sent_tools.push_back(SentTool{});
  while (static_cast<int>(hy_tool_ids.size()) < tool_count)
    hy_tool_ids.push_back(std::nullopt);
}

std::optional<DeltaMessage> HunyuanA13BToolParser::handle_tool_name_streaming(
    int current_idx, int tool_count,
    const std::vector<std::string>& name_group1s) {
  // hunyuan_a13b_tool_parser.py:279-310.
  if (current_idx == -1 || current_idx < tool_count - 1) {
    const int next_idx = current_idx + 1;
    if (next_idx < tool_count && !hy_sent_tools[next_idx].sent_name) {
      hy_current_tool_index = next_idx;
      hy_current_tool_id = next_idx;
      current_idx = next_idx;
      const std::string tool_name = name_group1s[current_idx];
      hy_tool_ids[current_idx] = make_tool_call_id();
      DeltaToolCall d;
      d.index = current_idx;
      d.type = "function";
      d.id = *hy_tool_ids[current_idx];
      d.function.name = tool_name;
      DeltaMessage msg;
      msg.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
      hy_sent_tools[current_idx].sent_name = true;
      while (static_cast<int>(hy_streamed_args.size()) <= current_idx)
        hy_streamed_args.emplace_back();
      return msg;
    }
  }
  return std::nullopt;
}

std::optional<DeltaMessage> HunyuanA13BToolParser::handle_tool_args_streaming(
    const std::string& current_text, int current_idx, int tool_count) {
  // hunyuan_a13b_tool_parser.py:312-420.
  if (!(current_idx >= 0 && current_idx < tool_count)) return std::nullopt;

  std::smatch empty_m;
  if (std::regex_search(current_text, empty_m, ToolEmptyArgRegex()) &&
      empty_m.position() > 0) {
    // Empty-arguments ("{}") path. Upstream loops i over tools but acts only
    // when i == current_idx, so the loop collapses to the current index.
    if (!hy_sent_tools[current_idx].sent_arguments_prefix) {
      hy_sent_tools[current_idx].sent_arguments_prefix = true;
      hy_sent_tools[current_idx].sent_arguments = "{}";
      while (static_cast<int>(hy_streamed_args.size()) <= current_idx)
        hy_streamed_args.emplace_back();
      hy_streamed_args[current_idx] += "{}";
      DeltaToolCall d;
      d.index = current_idx;
      d.function.arguments = "{}";
      DeltaMessage msg;
      msg.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
      if (current_idx < tool_count - 1) {
        hy_current_tool_index += 1;
        hy_current_tool_id = hy_current_tool_index;
      }
      return msg;
    }
  }

  // Non-empty arguments path.
  std::vector<std::smatch> args_matches;
  {
    auto begin = std::sregex_iterator(current_text.begin(), current_text.end(),
                                      ToolNonEmptyArgRegex());
    for (auto it = begin; it != std::sregex_iterator(); ++it)
      args_matches.push_back(*it);
  }
  if (current_idx < static_cast<int>(args_matches.size())) {
    const std::smatch& match = args_matches[current_idx];
    std::string args_text = match[1].str();
    const bool is_last_tool = (current_idx == tool_count - 1);
    if (!is_last_tool) {
      const std::size_t match_start = static_cast<std::size_t>(match.position());
      const std::size_t next_tool_pos = current_text.find("},{", match_start);
      if (next_tool_pos != std::string::npos) {
        const std::size_t args_end_pos = next_tool_pos + 1;
        args_text = Strip(AfterFirst(
            current_text.substr(match_start, args_end_pos - match_start),
            "\"arguments\":"));
      }
    }
    const std::string sent_args = hy_sent_tools[current_idx].sent_arguments;

    if (!hy_sent_tools[current_idx].sent_arguments_prefix &&
        !args_text.empty() && args_text.front() == '{') {
      hy_sent_tools[current_idx].sent_arguments_prefix = true;
      hy_sent_tools[current_idx].sent_arguments = "{";
      while (static_cast<int>(hy_streamed_args.size()) <= current_idx)
        hy_streamed_args.emplace_back();
      hy_streamed_args[current_idx] += "{";
      DeltaToolCall d;
      d.index = current_idx;
      d.function.arguments = "{";
      DeltaMessage msg;
      msg.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
      return msg;
    }

    if (args_text.rfind(sent_args, 0) == 0) {  // args_text.startswith(sent_args)
      const std::string args_diff = args_text.substr(sent_args.size());
      if (!args_diff.empty()) {
        hy_sent_tools[current_idx].sent_arguments = args_text;
        while (static_cast<int>(hy_streamed_args.size()) <= current_idx)
          hy_streamed_args.emplace_back();
        hy_streamed_args[current_idx] += args_diff;
        DeltaToolCall d;
        d.index = current_idx;
        d.function.arguments = args_diff;
        DeltaMessage msg;
        msg.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        return msg;
      }
    }

    if (!args_text.empty() && args_text.back() == '}' && args_text == sent_args) {
      if (current_idx < tool_count - 1) {
        hy_current_tool_index += 1;
        hy_current_tool_id = hy_current_tool_index;
      }
    }
  }
  return std::nullopt;
}

std::optional<DeltaMessage> HunyuanA13BToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // hunyuan_a13b_tool_parser.py:184-192 - find the JSON array start, past an
  // optional leading <tool_calls> bot_string (each with consume_space).
  std::size_t start_idx = consume_space(0, current_text);
  const std::string bot = kBotString;
  if (current_text.compare(start_idx, bot.size(), bot) == 0) {
    start_idx = consume_space(start_idx + bot.size(), current_text);
  }
  if (current_text.empty() || start_idx >= current_text.size() ||
      current_text[start_idx] != '[') {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }

  try_parse_json_tools(current_text.substr(start_idx));

  if (auto test_delta = handle_test_compatibility(current_text))
    return test_delta;

  // hunyuan_a13b_tool_parser.py:200-204 - the tool NAMES seen so far.
  std::vector<std::string> name_group1s;
  for (auto it = std::sregex_iterator(current_text.begin(), current_text.end(),
                                      ToolNameRegex());
       it != std::sregex_iterator(); ++it) {
    name_group1s.push_back((*it)[1].str());
  }
  const int tool_count = static_cast<int>(name_group1s.size());
  if (tool_count == 0) return std::nullopt;
  ensure_state_arrays(tool_count);
  const int current_idx = hy_current_tool_index;

  if (auto name_delta =
          handle_tool_name_streaming(current_idx, tool_count, name_group1s)) {
    return name_delta;
  }

  if (auto args_delta =
          handle_tool_args_streaming(current_text, current_idx, tool_count)) {
    return args_delta;
  }

  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
