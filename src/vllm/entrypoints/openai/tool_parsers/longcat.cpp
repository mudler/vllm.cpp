// Ported from: vllm/tool_parsers/longcat_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/longcat.h"

#include <regex>
#include <string>

namespace vllm::entrypoints::openai {

// Out-of-line anchor (key function) for LongcatToolParser's vtable.
LongcatToolParser::~LongcatToolParser() = default;

const std::string& LongcatToolParser::tool_call_start() const {
  static const std::string tok = kToolCallStartToken;
  return tok;
}

const std::string& LongcatToolParser::tool_call_end() const {
  static const std::string tok = kToolCallEndToken;
  return tok;
}

// longcat_tool_parser.py:18-22:
//   tool_call_regex = re.compile(
//       r"<longcat_tool_call>(.*?)</longcat_tool_call>"
//       r"|<longcat_tool_call>(.*)", re.DOTALL)
// Same two-capture shape as Hermes with the swapped tag; `[\s\S]` emulates the
// re.DOTALL `.` (ECMAScript `.` excludes newlines).
const std::regex& LongcatToolParser::tool_call_pattern() const {
  static const std::regex re(
      R"(<longcat_tool_call>([\s\S]*?)</longcat_tool_call>)"
      R"(|<longcat_tool_call>([\s\S]*))");
  return re;
}

}  // namespace vllm::entrypoints::openai
