// Ported from: vllm/tool_parsers/xlam_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/xlam.h"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
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
std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && IsSpace(s[b])) ++b;
  while (e > b && IsSpace(s[e - 1])) --e;
  return s.substr(b, e - b);
}
bool StartsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
bool EndsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
bool Contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}
bool ValidJson(const std::string& s) { return nlohmann::json::accept(s); }

// xlam_tool_parser.py:163 - f"call_{idx}_{random_uuid()}" (32 hex chars). The
// only requirement clients rely on is a stable, unique, >16-char id.
std::string GenXlamId(std::size_t idx) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t hi = dist(rng);
  const uint64_t lo = dist(rng);
  std::ostringstream oss;
  oss << "call_" << idx << "_" << std::hex << std::setfill('0') << std::setw(16)
      << hi << std::setw(16) << lo;
  return oss.str();
}

// xlam_tool_parser.py:50-55 - the json code-block patterns + thinking pattern.
const std::vector<std::regex>& CodeBlockPatterns() {
  static const std::vector<std::regex> pats = {
      std::regex(R"(```(?:json)?\s*([\s\S]*?)```)"),
      std::regex(R"(\[TOOL_CALLS\]([\s\S]*?)(?=\n|$))"),
      std::regex(R"(<tool_call>([\s\S]*?)</tool_call>)"),
  };
  return pats;
}
const std::regex& ThinkingPattern() {
  static const std::regex re(R"(</think>([\s\S]*))");
  return re;
}

// re.findall(pattern, text) for a single-capture-group pattern -> list of group1.
std::vector<std::string> FindAll(const std::regex& re, const std::string& text) {
  std::vector<std::string> out;
  for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
       it != std::sregex_iterator(); ++it) {
    out.push_back((*it)[1].str());
  }
  return out;
}

// json.dumps of a parsed "arguments" value, or the raw string when it is not a
// dict (xlam_tool_parser.py:167-171 / :476-480).
std::string ArgsToString(const nlohmann::ordered_json& args) {
  if (args.is_object()) return args.dump();
  if (args.is_string()) return args.get<std::string>();
  return args.dump();
}

}  // namespace

std::pair<std::optional<std::string>, std::optional<std::string>>
xLAMToolParser::preprocess_model_output(const std::string& model_output) const {
  // xlam_tool_parser.py:72-92 - a </think> block whose tail is (or contains) JSON.
  std::smatch tm;
  if (std::regex_search(model_output, tm, ThinkingPattern())) {
    const std::size_t start = static_cast<std::size_t>(tm.position(0));
    const std::string content =
        Strip(model_output.substr(0, start + std::string("</think>").size()));
    const std::string thinking_content = Strip(tm[1].str());
    if (ValidJson(thinking_content)) {
      return {content, thinking_content};
    }
    for (const std::regex& pat : CodeBlockPatterns()) {
      for (const std::string& json_str : FindAll(pat, thinking_content)) {
        if (ValidJson(json_str)) return {content, json_str};
      }
    }
    // No JSON inside the thinking block -> fall through (upstream does NOT return).
  }

  // xlam_tool_parser.py:95-105 - a JSON code block anywhere in the output.
  for (const std::regex& pat : CodeBlockPatterns()) {
    for (const std::string& json_str : FindAll(pat, model_output)) {
      if (ValidJson(json_str)) {
        const std::string content =
            Strip(std::regex_replace(model_output, pat, std::string()));
        return {content, json_str};
      }
    }
  }

  // xlam_tool_parser.py:107-119 - the whole output is (or looks like) a JSON list.
  if (StartsWith(Strip(model_output), "[")) {
    if (ValidJson(model_output)) {
      return {std::nullopt, model_output};
    }
    if (Contains(model_output, "{") && Contains(model_output, "name") &&
        Contains(model_output, "arguments")) {
      return {std::nullopt, model_output};
    }
  }

  // xlam_tool_parser.py:121-122 - no tool calls found.
  return {model_output, std::nullopt};
}

ExtractedToolCallInformation xLAMToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  try {
    auto [content, potential] = preprocess_model_output(model_output);

    // xlam_tool_parser.py:134-137 - nothing that looks like tool calls.
    if (!potential.has_value() || potential->empty()) {
      ExtractedToolCallInformation info;
      info.content = content;
      return info;
    }

    // xlam_tool_parser.py:140 - parse the potential tool calls.
    const nlohmann::ordered_json data = nlohmann::ordered_json::parse(*potential);

    // xlam_tool_parser.py:143-149 - must be an array.
    if (!data.is_array()) {
      ExtractedToolCallInformation info;
      info.content = content.has_value() ? content : std::optional<std::string>(model_output);
      return info;
    }

    // xlam_tool_parser.py:151-174.
    std::vector<ToolCall> tool_calls;
    std::size_t idx = 0;
    for (const nlohmann::ordered_json& call : data) {
      if (!call.is_object() || !call.contains("name") ||
          !call.contains("arguments")) {
        ++idx;
        continue;
      }
      ToolCall tc;
      tc.id = GenXlamId(idx);
      tc.type = "function";
      tc.function.name = call.at("name").get<std::string>();
      tc.function.arguments = ArgsToString(call.at("arguments"));
      tool_calls.push_back(std::move(tc));
      ++idx;
    }

    // xlam_tool_parser.py:176-180.
    ExtractedToolCallInformation info;
    info.tools_called = !tool_calls.empty();
    info.tool_calls = std::move(tool_calls);
    info.content = content;
    return info;

  } catch (const std::exception&) {
    // xlam_tool_parser.py:182-186 - any error -> whole output as content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> xLAMToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // xlam_tool_parser.py:203-233 - decide whether we are inside a tool-call block.
  const std::string stripped = Strip(current_text);
  auto [pre_content, pre_tools] = preprocess_model_output(current_text);

  const bool has_potential_json_block =
      Contains(current_text, "```json") || Contains(current_text, "```\n[") ||
      Contains(current_text, "[TOOL_CALLS]") ||
      Contains(current_text, "<tool_call>");

  const bool is_tool_call_block =
      StartsWith(stripped, "[") || StartsWith(stripped, "<tool_call>") ||
      StartsWith(stripped, "[TOOL_CALLS]") ||
      Contains(current_text, "</think>[") || pre_tools.has_value() ||
      (has_potential_json_block && Contains(current_text, "\"name\"") &&
       Contains(current_text, "\"arguments\""));

  if (!is_tool_call_block) {
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }

  try {
    // xlam_tool_parser.py:311-327 - pick the text to search over.
    std::string search_text = pre_tools.has_value() ? *pre_tools : current_text;
    if (!pre_tools.has_value() && has_potential_json_block) {
      static const std::regex code_extract(R"(```(?:json)?\s*([\s\S]*?)(?:```|$))");
      std::smatch jm;
      if (std::regex_search(current_text, jm, code_extract)) {
        const std::string potential_json = Strip(jm[1].str());
        if (StartsWith(potential_json, "[") &&
            Contains(potential_json, "\"name\"") &&
            Contains(potential_json, "\"arguments\"")) {
          search_text = potential_json;
        }
      }
    }

    // xlam_tool_parser.py:330-344 - complete tool names, else wait.
    static const std::regex name_pattern(R"rx("name"\s*:\s*"([^"]+)")rx");
    std::vector<std::string> name_matches = FindAll(name_pattern, search_text);
    const int tool_count = static_cast<int>(name_matches.size());
    if (tool_count == 0) {
      return std::nullopt;  // partial-or-absent name -> nothing to emit yet.
    }

    // xlam_tool_parser.py:347-357 - grow the per-tool state arrays.
    while (static_cast<int>(sent_tools_.size()) < tool_count)
      sent_tools_.emplace_back();
    while (static_cast<int>(tool_ids_.size()) < tool_count)
      tool_ids_.emplace_back();

    int current_idx = current_tool_index_;

    // xlam_tool_parser.py:363-402 - move to / announce the next tool by name.
    if (current_idx == -1 || current_idx < tool_count - 1) {
      const int next_idx = current_idx + 1;
      if (next_idx < tool_count && !sent_tools_[next_idx].sent_name) {
        current_tool_index_ = next_idx;
        current_tool_id = next_idx;
        current_idx = next_idx;
        const std::string tool_name = name_matches[current_idx];
        const std::string tool_id = GenXlamId(static_cast<std::size_t>(current_idx));
        tool_ids_[current_idx] = tool_id;
        DeltaToolCall d;
        d.index = current_idx;
        d.type = "function";
        d.id = tool_id;
        d.function.name = tool_name;
        sent_tools_[current_idx].sent_name = true;
        current_tool_name_sent = true;
        while (static_cast<int>(streamed_args_for_tool.size()) <= current_idx)
          streamed_args_for_tool.emplace_back();
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        return m;
      }
    }

    // xlam_tool_parser.py:405-553 - stream the current tool's arguments.
    if (current_idx >= 0 && current_idx < tool_count) {
      const std::size_t cur = static_cast<std::size_t>(current_idx);

      // Empty-arguments case: "arguments": {} .
      static const std::regex empty_args_pattern(
          R"("name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{\s*\})");
      std::smatch em;
      if (std::regex_search(search_text, em, empty_args_pattern) &&
          em.position(0) > 0) {
        if (!sent_tools_[cur].sent_arguments_prefix) {
          sent_tools_[cur].sent_arguments_prefix = true;
          sent_tools_[cur].sent_arguments = "{}";
          while (static_cast<int>(streamed_args_for_tool.size()) <= current_idx)
            streamed_args_for_tool.emplace_back();
          streamed_args_for_tool[cur] += "{}";
          DeltaToolCall d;
          d.index = current_idx;
          d.function.arguments = "{}";
          if (current_idx < tool_count - 1) {
            current_tool_index_ += 1;
            current_tool_id = current_tool_index_;
          }
          DeltaMessage m;
          m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
          return m;
        }
      }

      // Non-empty arguments: capture the balanced object (one nesting level).
      static const std::regex args_pattern(
          R"("name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\}))");
      std::vector<std::string> args_matches = FindAll(args_pattern, search_text);

      if (current_idx < static_cast<int>(args_matches.size())) {
        std::string args_text = args_matches[cur];

        // xlam_tool_parser.py:466-483 - for parallel calls, re-extract the exact
        // arguments for THIS tool from the fully-parsed structure when possible.
        if (tool_count > 1) {
          try {
            const nlohmann::ordered_json parsed =
                nlohmann::ordered_json::parse(search_text);
            if (parsed.is_array() && current_idx < static_cast<int>(parsed.size())) {
              const nlohmann::ordered_json& cur_tool = parsed[cur];
              if (cur_tool.is_object() && cur_tool.contains("arguments")) {
                args_text = ArgsToString(cur_tool.at("arguments"));
              } else {
                args_text = "{}";
              }
            }
          } catch (const std::exception&) {
            // Fall back to the regex-extracted args_text.
          }
        }

        const std::string sent_args = sent_tools_[cur].sent_arguments;

        // xlam_tool_parser.py:491-516 - emit the opening brace once.
        if (!sent_tools_[cur].sent_arguments_prefix && StartsWith(args_text, "{")) {
          sent_tools_[cur].sent_arguments_prefix = true;
          sent_tools_[cur].sent_arguments = "{";
          while (static_cast<int>(streamed_args_for_tool.size()) <= current_idx)
            streamed_args_for_tool.emplace_back();
          streamed_args_for_tool[cur] += "{";
          DeltaToolCall d;
          d.index = current_idx;
          d.function.arguments = "{";
          DeltaMessage m;
          m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
          return m;
        }

        // xlam_tool_parser.py:518-544 - emit the newly-arrived argument chars.
        if (StartsWith(args_text, sent_args)) {
          const std::string args_diff = args_text.substr(sent_args.size());
          if (!args_diff.empty()) {
            sent_tools_[cur].sent_arguments = args_text;
            while (static_cast<int>(streamed_args_for_tool.size()) <= current_idx)
              streamed_args_for_tool.emplace_back();
            streamed_args_for_tool[cur] += args_diff;
            DeltaToolCall d;
            d.index = current_idx;
            d.function.arguments = args_diff;
            DeltaMessage m;
            m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
            return m;
          }
        }

        // xlam_tool_parser.py:546-553 - this tool is done; advance next time.
        if (EndsWith(args_text, "}") && args_text == sent_args) {
          if (current_idx < tool_count - 1) {
            current_tool_index_ += 1;
            current_tool_id = current_tool_index_;
          }
        }
      }
    }

    // xlam_tool_parser.py:555-556 - nothing new to stream.
    return std::nullopt;

  } catch (const std::exception&) {
    // xlam_tool_parser.py:558-561 - on error, fall back to the delta as content.
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }
}

}  // namespace vllm::entrypoints::openai
