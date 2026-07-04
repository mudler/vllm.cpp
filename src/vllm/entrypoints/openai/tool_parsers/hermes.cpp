// Ported from: vllm/tool_parsers/hermes_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/hermes.h"

#include <regex>
#include <string>
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

}  // namespace vllm::entrypoints::openai
