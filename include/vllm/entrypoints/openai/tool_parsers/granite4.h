// Ported from: vllm/tool_parsers/granite4_tool_parser.py @ e24d1b24
//
// Granite4ToolParser (name "granite4") - for the IBM Granite 4.0 models. Format:
// interleaved text and `<tool_call> {json} </tool_call>` blocks, each block a
// bare {"name","arguments"} object. Unlike "granite"/"granite-20b-fc" this parser
// does NO partial-JSON parsing: it buffers via a look-ahead until a block CLOSES,
// then json.loads the complete block and emits the whole tool call in one delta.
//
// DEVIATIONS from upstream (all text-only reworks):
//   - streaming TOKEN-ID span params dropped (text-only), per the abstract/hermes
//     convention.
//   - adjust_request() (which sets skip_special_tokens=False so the special
//     `<tool_call>` tokens survive to the parser) is NOT ported: our serving seam
//     is pure text and never strips special tokens before the parser, so the
//     tokens are already present. Documented no-op.
//   - the `regex` module's partial-match search (start_regex.search(..., partial=
//     True)) becomes a literal find + partial_tag_overlap (the `<tool_call>` /
//     `</tool_call>` markers are literal strings with no regex metacharacters).
#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Granite4ToolParser : public ToolParser {
 public:
  Granite4ToolParser() = default;

  // granite4_tool_parser.py:92 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // granite4_tool_parser.py:205 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // granite4_tool_parser.py:58-59 - the block delimiters.
  static constexpr const char* kTcStart = "<tool_call>";
  static constexpr const char* kTcEnd = "</tool_call>";

 private:
  // granite4_tool_parser.py:150 (_tool_extraction_step). Consumes one buffered
  // `delta`, returning {done, content, tc_text}: `done` is false while there may
  // be more already-buffered content to drain (look_ahead_ holds the remainder).
  std::tuple<bool, std::string, std::string> tool_extraction_step(
      const std::string& delta);

  // granite4_tool_parser.py:55-56 - the streaming carry-over buffer + in-block
  // flag (state NOT present on the base ToolParser).
  std::string look_ahead_;
  bool in_tc_ = false;

  // granite4_tool_parser.py:51 (self.prev_tool_call_arr): the record of parsed
  // tool-call objects. Kept as ordered_json (author key order) here rather than
  // on the base ToolParser (whose nlohmann::json backing SORTS keys).
  std::vector<nlohmann::ordered_json> granite4_prev_tool_call_arr_;
};

}  // namespace vllm::entrypoints::openai
