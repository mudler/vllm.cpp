// Ported from: vllm/tool_parsers/deepseekv4_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v4.h"

#include <string>

namespace vllm::entrypoints::openai {

// Out-of-line anchor (key function) for DeepSeekV4ToolParser's vtable.
DeepSeekV4ToolParser::~DeepSeekV4ToolParser() = default;

// deepseekv4_tool_parser.py:15-16 — swap the wrapper tokens; everything else
// (schema coercion, name-first streaming, argument diffing, wrapper repair) is
// inherited from DeepSeekV32ToolParser unchanged.
const std::string& DeepSeekV4ToolParser::tool_call_start() const {
  static const std::string tok = kToolCallStartToken;
  return tok;
}
const std::string& DeepSeekV4ToolParser::tool_call_end() const {
  static const std::string tok = kToolCallEndToken;
  return tok;
}

}  // namespace vllm::entrypoints::openai
