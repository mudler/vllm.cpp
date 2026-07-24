// Ported from: vllm/tool_parsers/step3_tool_parser.py @ e24d1b24
//
// Step3ToolParser - the StepFun step3 tool-call format. A tool-call block is
// wrapped in the OUTER markers <｜tool_calls_begin｜> … <｜tool_calls_end｜> and
// each call is <｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="x">
// <steptml:parameter name="k">v</steptml:parameter>…</steptml:invoke>
// <｜tool_call_end｜>. NOTE: the markers use the FULLWIDTH vertical bar (U+FF5C,
// UTF-8 EF BD 9C), NOT ASCII '|'; the word separators are ASCII underscores.
// These are DISTINCT byte strings from the DeepSeek markers (which put U+2581
// ▁ between words). They are multi-byte UTF-8 and are treated here as opaque
// byte strings (std::string::find), never indexed per character.
//
// FIDELITY NOTE (bug-for-bug): upstream step3 is knowingly buggy - the Python
// test config (tests/tool_parsers/test_step3_tool_parser.py) xfails almost all
// non-streaming cases and the parallel/reconstruction streaming cases. Two root
// causes are preserved verbatim here so the port matches the reference exactly:
//   1. NON-STREAMING requires a `function<｜tool_sep｜>` prefix INSIDE every
//      call (DeepSeek layout), but the real step3 format never emits it, so the
//      non-streaming extract yields NO tool calls for genuine step3 output.
//   2. STREAMING never consumes the inter-call <｜tool_sep｜>, so a parallel
//      stream wedges after the first call. Both behaviours are reproduced.
//
// DEVIATIONS from upstream (all reworks of the token-id/vocab machinery to pure
// text, per the TEXT-ONLY abstract.h seam):
//   1. VOCAB / TOKENIZER DROPPED. Upstream __init__ takes a tokenizer (only used
//      by adjust_request to flip skip_special_tokens); the C++ ToolParser ABC is
//      tokenizer-free, so adjust_request has no analogue and is dropped.
//   2. TOOLS via REQUEST, not constructor. Upstream reads self.tools (set at
//      construction) in _cast_arguments; the C++ seam carries the tools on the
//      per-call ChatCompletionRequest, so we read request.tools instead.
//   3. STREAMING uses the accumulated `current_text` (upstream drives off the
//      same current_text via self.position); token-id params are dropped.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Step3ToolParser : public ToolParser {
 public:
  Step3ToolParser() = default;

  // step3_tool_parser.py:255 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // step3_tool_parser.py:115 (extract_tool_calls_streaming). The cursor-based
  // stateful streaming parse driven off the accumulated current_text.
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // step3_tool_parser.py:41-45 - the marker byte strings (fullwidth ｜, ASCII
  // underscores). Copied EXACTLY from the upstream source.
  static constexpr const char* kToolCallsBegin = "<｜tool_calls_begin｜>";
  static constexpr const char* kToolCallsEnd = "<｜tool_calls_end｜>";
  static constexpr const char* kToolCallBegin = "<｜tool_call_begin｜>";
  static constexpr const char* kToolCallEnd = "<｜tool_call_end｜>";
  static constexpr const char* kToolSep = "<｜tool_sep｜>";

 private:
  // step3_tool_parser.py:50-52 - explicit streaming state flags (in addition to
  // the base ToolParser members current_tool_id / current_tool_name_sent /
  // prev_tool_call_arr, which upstream inherits from ToolParser too).
  std::size_t position_ = 0;
  bool tool_block_started_ = false;
  bool tool_block_finished_ = false;
};

}  // namespace vllm::entrypoints::openai
