// Ported from: vllm/tool_parsers/deepseekv31_tool_parser.py @ e24d1b24
//
// DeepSeekV31ToolParser — DeepSeek-V3.1 tool-call format. Same OUTER/INNER
// marker byte strings as V3 (<｜tool▁calls▁begin｜> / <｜tool▁call▁begin｜> /
// <｜tool▁sep｜> / …), but the per-call layout drops the ```json fence and puts
// the arguments DIRECTLY after the separator:
//   <｜tool▁call▁begin｜>NAME<｜tool▁sep｜>ARGS<｜tool▁call▁end｜>
// and the type is always "function" (there is no captured type group). Because
// only the per-call SHAPE differs, this is a thin subclass of DeepSeekV3ToolParser
// overriding the three virtual seams (regex / match->ToolCall / streaming region
// parse); the wrapper-scanning, content hold-back and name-first argument-diff
// streaming are inherited verbatim.
//
// DEVIATIONS: identical to deepseek_v3.h (token-id detection -> text find; vocab
// dropped; streaming reworked to full-current_text re-parse + diff). See that
// header for the full list.
#pragma once

#include <optional>
#include <regex>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v3.h"

namespace vllm::entrypoints::openai {

class DeepSeekV31ToolParser : public DeepSeekV3ToolParser {
 public:
  DeepSeekV31ToolParser() = default;
  ~DeepSeekV31ToolParser() override;

 protected:
  // deepseekv31_tool_parser.py:46-48 — the V3.1 per-call regex (name, args); no
  // fence, no type group.
  const std::regex& tool_call_pattern() const override;
  // type is hard-coded "function"; group 1 = name, group 2 = arguments.
  ToolCall tool_call_from_match(const std::smatch& match) const override;
  // Streaming region parse: NAME<｜tool▁sep｜>ARGS (args = everything after the
  // separator, with a partial <｜tool▁call▁end｜> suffix held back for UTF-8
  // safety on the unterminated tail).
  ParsedCall parse_region(const std::string& region) const override;
};

}  // namespace vllm::entrypoints::openai
