// Ported from: vllm/tool_parsers/pythonic_tool_parser.py @ e24d1b24
//
// PythonicToolParser - the `[func(arg=val, ...), ...]` Python-call-literal
// format emitted by Llama 3.2 / Llama 4 models. Ports both the NON-STREAMING
// extract_tool_calls and the STREAMING incremental parse
// extract_tool_calls_streaming.
//
// DEVIATIONS from upstream (all documented here + at the helper sites in
// pythonic.cpp):
//   - Upstream parses the call literals with Python's `ast` module
//     (ast.parse + ast.Call/Constant/Dict/List/Name walking in
//     tool_parsers/utils.py:handle_single_tool + get_parameter_value). There is
//     no `ast` in C++, so we hand-write a strict recursive-descent parser for
//     the pythonic call grammar (dotted identifiers, keyword args, and the
//     Python literals strings / ints / floats / True|False|None /
//     true|false|null / lists / dicts, nested). It is STRICT: it rejects
//     anything upstream's ast+get_parameter_value would reject (non-literal
//     args, bare identifiers other than the True|False|None|true|false|null
//     name-literals, set literals `{a, b}`, mismatched brackets). One concrete
//     consequence, matching upstream: a NEGATIVE numeric literal (`x=-5`) parses
//     in Python to an ast.UnaryOp, which is NOT an ast.Constant, so
//     get_parameter_value RAISES - upstream rejects it. We likewise reject a
//     leading '-' on a numeric value.
//   - Upstream's `TOOL_CALL_REGEX.match(...)` pre-gate (a coarse shape check run
//     before ast.parse) is collapsed into the strict parse: a text that is not a
//     non-empty `[call, ...]` list-of-calls simply fails to parse and is
//     returned as plain content, which is the identical net behavior (both the
//     regex-miss and the ast-failure branches return
//     tools_called=false / content=model_output). The Python `regex` timeout
//     path (VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS) has no C++ analogue and is
//     dropped; the graceful-fallback behavior it guards is preserved.
//   - Arguments are re-serialized with nlohmann::ordered_json::dump() (compact:
//     no space after ':'/','), whereas upstream json.dumps() uses ", "/": "
//     separators. The JSON is semantically identical (clients json.loads it);
//     only inter-token whitespace differs. ordered_json preserves the model's
//     key insertion order (Python dicts + json.dumps do the same).
//   - The abstract streaming seam is TEXT-only (see abstract.h): the three
//     token-id spans upstream's signature carries are dropped (the pythonic
//     parse never reads them).
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class PythonicToolParser : public ToolParser {
 public:
  PythonicToolParser() = default;

  // pythonic_tool_parser.py:69 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // pythonic_tool_parser.py:117 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 protected:
  // The format-agnostic cores, shared with Llama4PythonicToolParser (which
  // strips its <|python_start|>/<|python_end|> markers then delegates here).
  // `clean_text` is the call-literal list text with any model markers removed.
  ExtractedToolCallInformation ExtractClean(const std::string& clean_text);
  std::optional<DeltaMessage> StreamClean(const std::string& current_clean);
};

}  // namespace vllm::entrypoints::openai
