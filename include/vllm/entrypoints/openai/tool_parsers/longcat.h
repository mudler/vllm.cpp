// Ported from: vllm/tool_parsers/longcat_tool_parser.py @ e24d1b24
//
// LongcatFlashToolParser — a 22-LOC Hermes subclass (longcat_tool_parser.py:11)
// that swaps ONLY the wrapper tag from `<tool_call>` to `<longcat_tool_call>`
// (LongCat-Flash emits `<longcat_tool_call>{json}</longcat_tool_call>`). Every
// other behaviour — the name-first streaming, the argument diffing, the
// graceful malformed-JSON fallback — is inherited verbatim from HermesToolParser
// (upstream inherits from Hermes2ProToolParser). The C++ port therefore mirrors
// the upstream layout: override the three virtual accessors HermesToolParser
// exposes for the wrapper tokens + extraction regex, and nothing else.
//
// DETECTION NOTE: probe `<longcat_tool_call>`. Verified against the upstream
// LongCat-Flash-Chat chat template — it wraps tool calls exclusively in
// <longcat_tool_call> and never emits the bare Hermes <tool_call>; `<tool_call>`
// is not a substring of `<longcat_tool_call>`, so a hermes-first check will not
// mis-detect standard output. A loose/substring auto-detector should still test
// the more specific `<longcat_tool_call>` marker before `<tool_call>`. See the
// detection notes in abstract.cpp.
#pragma once

#include <regex>
#include <string>

#include "vllm/entrypoints/openai/tool_parsers/hermes.h"

namespace vllm::entrypoints::openai {

// The LongCat-Flash Hermes-format parser. Behaviourally identical to
// HermesToolParser except for the wrapper tag.
class LongcatToolParser : public HermesToolParser {
 public:
  LongcatToolParser() = default;
  // Out-of-line key function (defined in longcat.cpp) — anchors the vtable.
  ~LongcatToolParser() override;

  // longcat_tool_parser.py:15-16 — the swapped wrapper tokens.
  static constexpr const char* kToolCallStartToken = "<longcat_tool_call>";
  static constexpr const char* kToolCallEndToken = "</longcat_tool_call>";

 protected:
  const std::string& tool_call_start() const override;
  const std::string& tool_call_end() const override;
  const std::regex& tool_call_pattern() const override;
};

}  // namespace vllm::entrypoints::openai
