// Ported from: vllm/tool_parsers/phi4mini_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/phi4_mini.h"

#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

// phi4mini_tool_parser.py:63 - re.search(r"functools\[(.*?)\]", output, DOTALL).
// ECMAScript `.` does not match newlines, so [\s\S] emulates re.DOTALL; the
// non-greedy `*?` matches the shortest bracket body.
const std::regex& FunctoolsRegex() {
  static const std::regex re(R"(functools\[([\s\S]*?)\])");
  return re;
}

}  // namespace

ExtractedToolCallInformation Phi4MiniJsonToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // phi4mini_tool_parser.py:63-70 - no `functools[...]` match -> plain content.
  std::smatch matches;
  if (!std::regex_search(model_output, matches, FunctoolsRegex())) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  try {
    // phi4mini_tool_parser.py:73-85 - the inner json.loads is wrapped in its own
    // try/except that only LOGS on failure, leaving function_call_arr empty. We
    // reproduce that: a parse error here yields an empty array (NOT a fallback to
    // content), which is why malformed `functools[...]` still reports
    // tools_called=True with an empty tool list (upstream xfail_nonstreaming).
    nlohmann::ordered_json function_call_arr = nlohmann::ordered_json::array();
    try {
      const std::string json_content = "[" + matches[1].str() + "]";
      function_call_arr = nlohmann::ordered_json::parse(json_content);
    } catch (const std::exception&) {
      // Swallowed like upstream's inner except json.JSONDecodeError.
    }

    // phi4mini_tool_parser.py:87-103 - build one ToolCall per entry. A missing
    // "name" (or missing BOTH "arguments" and "parameters") throws here and is
    // caught by the OUTER except below -> fall back to the whole output.
    std::vector<ToolCall> tool_calls;
    for (const nlohmann::ordered_json& raw : function_call_arr) {
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = raw.at("name").get<std::string>();
      // arguments are JSON but stored AS A STRING (json.dumps ensure_ascii=False);
      // prefer "arguments", else "parameters".
      const nlohmann::ordered_json& args =
          raw.contains("arguments") ? raw.at("arguments") : raw.at("parameters");
      tc.function.arguments = args.dump();
      tool_calls.push_back(std::move(tc));
    }

    // phi4mini_tool_parser.py:106-109 - tools_called is hardcoded True (even for
    // an empty list), content is None.
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    return info;

  } catch (const std::exception&) {
    // phi4mini_tool_parser.py:111-114 - any error -> whole output as content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> Phi4MiniJsonToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& /*delta_text*/, const ChatCompletionRequest& /*request*/) {
  // phi4mini_tool_parser.py:116-126 - streaming is unimplemented; returns None.
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
