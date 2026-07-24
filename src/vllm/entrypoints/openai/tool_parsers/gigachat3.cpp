// Ported from: vllm/tool_parsers/gigachat3_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/gigachat3.h"

#include <cstddef>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// gigachat3_tool_parser.py:28-31 - the function-call header + JSON tail.
const std::regex& FuncRegex() {
  static const std::regex re(
      R"((?:function call<\|role_sep\|>\n|<\|function_call\|>)([\s\S]*))");
  return re;
}

// gigachat3_tool_parser.py:33-36 - content up to the first separator.
const std::regex& ContentRegex() {
  static const std::regex re(
      R"(^([\s\S]*?)(?:<\|message_sep\|>|<\|function_call\|>))");
  return re;
}

// gigachat3_tool_parser.py:38-46 - name + arguments.
const std::regex& NameRegex() {
  static const std::regex re(R"rx("name"\s*:\s*"([^"]*)")rx");
  return re;
}
const std::regex& ArgsRegex() {
  static const std::regex re(R"("arguments"\s*:\s*([\s\S]*))");
  return re;
}

std::string Strip(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && static_cast<unsigned char>(s[b]) <= ' ') ++b;
  while (e > b && static_cast<unsigned char>(s[e - 1]) <= ' ') --e;
  return s.substr(b, e - b);
}

std::string RStrip(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0 && static_cast<unsigned char>(s[e - 1]) <= ' ') --e;
  return s.substr(0, e);
}

bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool StartsWith(const std::string& s, const std::string& pre) {
  return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

}  // namespace

ExtractedToolCallInformation GigaChat3ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  std::string out = model_output;
  // gigachat3_tool_parser.py:74-75 - peel a trailing </s>.
  if (EndsWith(RStrip(out), "</s>")) {
    out = out.substr(0, out.rfind("</s>"));
  }

  std::optional<nlohmann::ordered_json> function_call;
  std::smatch mf;
  if (std::regex_search(out, mf, FuncRegex())) {
    try {
      nlohmann::ordered_json fc = nlohmann::ordered_json::parse(mf[1].str());
      if (fc.is_object() && fc.contains("name") && fc.contains("arguments")) {
        if (!fc.at("arguments").is_object()) {
          function_call = std::nullopt;  // arguments not a dict -> reject.
        } else {
          function_call = fc;
        }
      } else {
        function_call = std::nullopt;
      }
    } catch (const nlohmann::json::parse_error&) {
      // gigachat3_tool_parser.py:89-94 - JSON error -> plain content.
      return ExtractedToolCallInformation{false, {}, out};
    }
  }

  std::smatch mc;
  std::string content =
      std::regex_search(out, mc, ContentRegex()) ? mc[1].str() : out;

  if (!function_call.has_value()) {
    ExtractedToolCallInformation info;
    info.tools_called = false;
    if (!content.empty()) info.content = content;
    return info;
  }

  const std::string name = function_call->at("name").get<std::string>();
  const std::string args = function_call->at("arguments").dump();

  ToolCall tc;
  tc.id = make_tool_call_id();
  tc.type = "function";
  tc.function.name = name;
  tc.function.arguments = args;

  ExtractedToolCallInformation info;
  info.tools_called = true;
  info.tool_calls = std::vector<ToolCall>{std::move(tc)};
  if (!content.empty()) info.content = content;
  return info;
}

std::optional<DeltaMessage> GigaChat3ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  std::optional<std::string> content;
  std::optional<std::string> func_name;
  std::optional<std::string> cur_args;

  std::smatch mf;
  const bool has_func = std::regex_search(current_text, mf, FuncRegex());

  if (!tool_started_) {
    std::smatch mc;
    if (std::regex_search(delta_text, mc, ContentRegex())) {
      content = mc[1].str();
      end_content_ = true;
    } else if (!end_content_) {
      content = delta_text;
    }
    if (has_func) tool_started_ = true;
    if (content.has_value() && !content->empty()) {
      DeltaMessage m;
      m.content = content;
      return m;
    }
  }

  if (!has_func) return std::nullopt;

  const std::string json_tail = Strip(mf[1].str());
  std::smatch nm;
  if (std::regex_search(json_tail, nm, NameRegex())) func_name = nm[1].str();
  std::smatch am;
  if (std::regex_search(json_tail, am, ArgsRegex())) {
    std::string ca = Strip(am[1].str());
    if (EndsWith(ca, "</s>")) ca = ca.substr(0, ca.size() - 4);
    if (EndsWith(ca, "}")) {  // last '}' may close the outer JSON object.
      const std::string candidate = Strip(ca.substr(0, ca.size() - 1));
      if (nlohmann::json::accept(candidate)) ca = candidate;
    }
    cur_args = ca;
  }

  if (!tool_name_sent_) {
    if (!func_name.has_value() || func_name->empty()) return std::nullopt;
    tool_name_sent_ = true;
    tool_id_ = make_tool_call_id();
    prev_name_ = *func_name;
    DeltaToolCall d;
    d.index = 0;
    d.id = tool_id_;
    d.type = "function";
    d.function.name = func_name;
    DeltaMessage m;
    m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
    return m;
  }

  if (!cur_args.has_value()) return std::nullopt;

  const std::string prev_args = prev_arguments_str_set_ ? prev_arguments_str_ : "";
  std::string delta_args;
  if (prev_args.empty()) {
    delta_args = *cur_args;
  } else if (StartsWith(*cur_args, prev_args)) {
    delta_args = cur_args->substr(prev_args.size());
  } else {
    return std::nullopt;
  }
  if (delta_args.empty()) return std::nullopt;

  prev_arguments_str_ = *cur_args;
  prev_arguments_str_set_ = true;
  if (streamed_args_for_tool_.empty()) streamed_args_for_tool_.emplace_back();
  streamed_args_for_tool_[0] = *cur_args;

  DeltaToolCall d;
  d.index = 0;
  d.function.arguments = delta_args;
  DeltaMessage m;
  m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
  return m;
}

}  // namespace vllm::entrypoints::openai
