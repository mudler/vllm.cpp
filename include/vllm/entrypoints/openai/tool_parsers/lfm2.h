// Ported from: vllm/tool_parsers/lfm2_tool_parser.py @ e24d1b24
//
// Lfm2ToolParser (registry name "lfm2") - LiquidAI LFM2 / LFM2.5 models emit
// PYTHONIC tool calls (a `[func(arg=val), ...]` list) wrapped in the special
// sentinels <|tool_call_start|> ... <|tool_call_end|>:
//     <|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>
//     The weather in Paris is sunny.
// The list body is the SAME pythonic grammar the Llama pythonic parser uses, so
// this port REUSES pythonic_core (parse_call_list, make_valid_python,
// compute_tool_delta) - exactly as olmo3 does - and layers only the LFM2-specific
// marker stripping, leading/trailing-content handling, and echo suppression on
// top. LFM2 frequently ECHOES the call body again after the first
// <|tool_call_end|>, capped with a second <|tool_call_end|>; that echo must never
// leak as assistant content, in either the non-streaming or streaming path.
//
// MAJOR DEVIATION (vocab -> text rework):
//   Upstream reads the tokenizer VOCAB (self.tool_call_start/end_token_id) purely
//   to (a) refuse construction when the sentinels are absent from the tokenizer,
//   and (b) via adjust_request, force skip_special_tokens=False so the engine
//   preserves the sentinels in the decoded text. Our abstract.h seam is TEXT-only
//   (no tokenizer, no token-id spans, no adjust_request hook - see abstract.h),
//   so BOTH are dropped: the parser operates directly on the decoded text and
//   assumes the sentinels are present (the serving layer owns skip_special_tokens
//   the same way it does for every other special-marker dialect here, e.g. jamba,
//   internlm, granite). No behaviour that this text-only seam can observe depends
//   on the dropped vocab wiring.
//
// Other (shared) deviations, all inherited from the pythonic core + the seam:
//   - Python `ast` -> the strict recursive-descent grammar in pythonic_core.
//   - The TOOL_CALL_REGEX (r"\[.*\]$", DOTALL) pre-gate + its
//     VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS timeout path collapse into the strict
//     parse_call_list: text that is not a `[call, ...]` list is returned as plain
//     content, the identical net behaviour to both the regex-miss and the
//     timeout-fallback branches.
//   - Arguments re-serialized compact (no space after ':'/','); json.loads-equal.
//   - The abstract streaming seam is TEXT-only: the three token-id spans upstream
//     carries are dropped (this parse never reads them).
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Lfm2ToolParser : public ToolParser {
 public:
  Lfm2ToolParser() = default;

  // lfm2_tool_parser.py:144 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // lfm2_tool_parser.py:195 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // lfm2_tool_parser.py:33-34 - the sentinel tokens.
  static constexpr const char* kToolCallStart = "<|tool_call_start|>";
  static constexpr const char* kToolCallEnd = "<|tool_call_end|>";

 private:
  // lfm2_tool_parser.py:72 (self._trailing_emitted). Trailing content already
  // streamed to the client; used to suppress LFM2's echo of the call body after
  // the first <|tool_call_end|> while still letting legitimate prose through.
  std::string trailing_emitted_;
};

}  // namespace vllm::entrypoints::openai
