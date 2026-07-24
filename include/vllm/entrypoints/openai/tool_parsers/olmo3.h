// Ported from: vllm/tool_parsers/olmo3_tool_parser.py @ e24d1b24
//
// Olmo3PythonicToolParser (registry name "olmo3") - Olmo 3 emits tool calls as
// NEWLINE-separated pythonic call literals wrapped in a
// <function_calls>...</function_calls> tag:
//     <function_calls>get_weather(city='SF')\nget_time()</function_calls>
// Upstream copies the pythonic streaming body and reuses the shared utils.py
// helpers; we mirror that by REUSING pythonic_core (parse_call_list,
// make_valid_python, compute_tool_delta) rather than duplicating the grammar.
// Only the olmo3-specific pre/post-processing lives here:
//   - strip the <function_calls> wrapper,
//   - join the newline-separated calls with ", " and wrap them in a [...] list,
//   - the withheld-suffix bookkeeping operates on the UN-wrapped make_valid_python
//     result, so it uses added_text[:-1] / ")" (vs pythonic's [:-2] / ")]" ,
//     which account for pythonic's outer list brackets).
//
// DEVIATIONS from upstream (all inherited from the pythonic core + the text-only
// seam; see pythonic_core.h):
//   - Python `ast` -> the strict recursive-descent grammar in pythonic_core.
//   - Arguments re-serialized compact (no space after ':'/',').
//   - The regex TOOL_CALL_REGEX pre-gate + its VLLM_TOOL_PARSE_REGEX_TIMEOUT_
//     SECONDS timeout path collapse into the strict parse: a text that is not a
//     valid list-of-calls is returned as plain content, the identical net
//     behavior to both the regex-miss and the timeout-fallback branches.
//   - The abstract streaming seam is TEXT-only: the three token-id spans
//     upstream's signature carries are dropped (never read by this parse).
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Olmo3ToolParser : public ToolParser {
 public:
  Olmo3ToolParser() = default;

  // olmo3_tool_parser.py:71 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // olmo3_tool_parser.py:132 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;
};

}  // namespace vllm::entrypoints::openai
