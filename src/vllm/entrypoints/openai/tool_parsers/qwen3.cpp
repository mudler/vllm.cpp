// Ported from: vllm/tool_parsers/qwen3_engine_tool_parser.py @ e24d1b24
//
// Qwen3ToolParser is a behavior-identical subclass of HermesToolParser (the
// gate model Qwen3.6-35B emits the Hermes `<tool_call>{json}</tool_call>`
// format — see qwen3.h for the cited format decision). All behavior is
// inherited; this translation unit exists to mirror the upstream file layout
// and to anchor the class's vtable/type_info to one object file.
#include "vllm/entrypoints/openai/tool_parsers/qwen3.h"

namespace vllm::entrypoints::openai {

// Out-of-line anchor (key function) for Qwen3ToolParser's vtable.
Qwen3ToolParser::~Qwen3ToolParser() = default;

}  // namespace vllm::entrypoints::openai
