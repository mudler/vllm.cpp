// Ported from: vllm/tool_parsers/deepseekv4_tool_parser.py @ e24d1b24
//
// DeepSeekV4ToolParser — a 17-LOC DeepSeekV32ToolParser subclass. V4 keeps the
// V3.2 DSML invoke/parameter grammar VERBATIM but wraps the tool-call block in
// <｜DSML｜tool_calls> instead of <｜DSML｜function_calls>. The C++ port mirrors
// the upstream layout: override ONLY the two virtual wrapper accessors that
// DeepSeekV32ToolParser exposes; every other behaviour (schema coercion,
// name-first streaming, argument diffing, wrapper repair) is inherited unchanged.
// This is the same override-the-wrapper pattern longcat.{h,cpp} uses over hermes.
//
// DETECTION NOTE: probe <｜DSML｜tool_calls>. It is NOT a substring of the v3.2
// marker <｜DSML｜function_calls> (they diverge right after "<｜DSML｜"), and
// NEITHER shares a substring with the deepseek_v3 marker <｜tool▁calls▁begin｜>
// (which uses U+2581 ▁ and carries no "DSML"). See the detection notes reported
// with this port.
#pragma once

#include <string>

#include "vllm/entrypoints/openai/tool_parsers/deepseek_v32.h"

namespace vllm::entrypoints::openai {

// The DeepSeek-V4 DSML parser. Behaviourally identical to DeepSeekV32ToolParser
// except for the wrapper tokens.
class DeepSeekV4ToolParser : public DeepSeekV32ToolParser {
 public:
  DeepSeekV4ToolParser() = default;
  // Out-of-line key function (defined in deepseek_v4.cpp) — anchors the vtable.
  ~DeepSeekV4ToolParser() override;

  // deepseekv4_tool_parser.py:15-16 — the swapped wrapper tokens.
  static constexpr const char* kToolCallStartToken = "<｜DSML｜tool_calls>";
  static constexpr const char* kToolCallEndToken = "</｜DSML｜tool_calls>";

 protected:
  const std::string& tool_call_start() const override;
  const std::string& tool_call_end() const override;
};

}  // namespace vllm::entrypoints::openai
