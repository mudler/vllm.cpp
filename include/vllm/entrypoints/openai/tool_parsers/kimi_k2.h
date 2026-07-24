// REIMPLEMENTED FROM THE WIRE FORMAT of vllm/parser/kimi_k2.py @ e24d1b24
// (KimiK2Parser + kimi_k2_config, a declarative token-id ParserEngine). This is
// NOT a mechanical port of that engine: upstream drives a state machine off the
// vocab TOKEN IDS of the five Kimi special markers and emits DeltaToolCall events
// from a generic ParserEngine. This file reimplements the SAME on-the-wire tool
// grammar on the TEXT seam (abstract.h is text-only), held to the upstream tests
// (tests/tool_parsers/test_kimi_k2_tool_parser.py, all 32 cases ported to
// test_kimi_k2.cpp), the same way deepseek_v3.cpp reimplements the DeepSeek
// marker grammar.
//
// WIRE FORMAT (kimi_k2.py:5-14 docstring + the terminals/transitions table at
// kimi_k2.py:43-140). A tool-call section is:
//   <|tool_calls_section_begin|>
//     <|tool_call_begin|>functions.NAME:INDEX <|tool_call_argument_begin|>{JSON args}<|tool_call_end|>
//     ...one <|tool_call_begin|>…<|tool_call_end|> per call…
//   <|tool_calls_section_end|>
// The header between <|tool_call_begin|> and <|tool_call_argument_begin|> is
// Kimi's NATIVE tool-call id (kimi_k2.py:12-13). It is `.strip()`-ed and matched
// against _TOOL_ID_RE = `(?P<id>.+:\d+)` (kimi_k2.py:49): the id is the whole
// match (e.g. `functions.get_weather:0`); the tool NAME is the text before the
// FIRST ':' with a leading `functions.` removed (kimi_k2.py:186-188). A header
// that does not match `.+:\d+` (e.g. `functions.invalid.0`) yields no name and
// the whole call is DROPPED (kimi_k2.py:197-200). The args between
// <|tool_call_argument_begin|> and the closing marker are `.strip() or "{}"`
// (kimi_k2.py:236-237) and stored VERBATIM (never re-serialized, never validated
// - malformed JSON is still returned, test_invalid_json_still_extracted). A call
// may be closed by <|tool_call_end|> (next call follows) OR directly by
// <|tool_calls_section_end|> (kimi_k2.py:118-125, TOOL_ARGS on either terminal
// emits TOOL_CALL_END). Content is the text BEFORE the section; a whitespace-only
// prefix is dropped to None (kimi_k2.py:144 drop_whitespace_only_content_before_
// tools) but non-whitespace content is kept UNSTRIPPED (kimi_k2.py:145
// strip_content_whitespace_with_tools=False). Text after the section is dropped.
//
// DEVIATIONS from the upstream engine (all reworks of the token-id machinery to
// pure text, per the TEXT-ONLY abstract.h seam):
//   1. TOKEN-ID STATE MACHINE -> TEXT SCAN. Upstream matches the five markers by
//      their vocab TOKEN IDS across previous/current/delta token-id spans and
//      drives ParserState transitions. We have no token ids: the non-streaming
//      extract does a single left-to-right std::string::find scan of the marker
//      byte strings, and streaming re-scans the FULL current_text and diffs, the
//      same text-only rework deepseek_v3.cpp documents. A marker split across a
//      delta boundary is simply "not found yet" (we wait); a partial marker
//      SUFFIX in emittable text (leading content, or a streaming argument tail)
//      is held back via PartialTagOverlap so no marker byte and no broken split
//      ever leaks as content/arguments (the markers are ASCII multi-char, so a
//      byte-chunked stream slices them mid-marker - covered by the tests).
//   2. NATIVE-ID CONVENTION vs make_tool_call_id. Unlike every other family here
//      (which stamp ToolCall.id = make_tool_call_id() => "chatcmpl-tool-<uuid>"),
//      Kimi carries its OWN id: ToolCall.id / DeltaToolCall.id are set to the
//      native `functions.NAME:INDEX` header string (kimi_k2.py:203-204,213 assign
//      slot.id = tool_id). This is REQUIRED (test_native_id_extracted,
//      test_multi_turn_native_id_continuity, and the streaming id asserts). We do
//      NOT call make_tool_call_id() for Kimi.
//   3. VOCAB / TOKENIZER / REASONING DROPPED. Upstream __init__ looks the markers
//      up in self.vocab and the engine also owns <think> reasoning splitting
//      (kimi_k2.py:41-42,239-285). The C++ ToolParser ABC is tokenizer-free and
//      reasoning is a SEPARATE reasoning-parser seam, so this file does tool-call
//      extraction ONLY; the reasoning terminals/transitions are not modeled here.
//   4. adjust_request. Upstream KimiK2ToolParser.adjust_request (the thin adapter,
//      tool_parsers/kimi_k2_tool_parser.py) forces skip_special_tokens=False when
//      tools are active. abstract.h has no adjust_request hook, so it is added as
//      a public method on this class ONLY (documented, tested) rather than widened
//      onto the base ABC.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class KimiK2ToolParser : public ToolParser {
 public:
  KimiK2ToolParser() = default;

  // kimi_k2.py wire format, non-streaming (the engine's batched extract).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // kimi_k2.py wire format, streaming - full-current_text re-scan + diff (see
  // DEVIATION 1). Name-first (with the native id), then argument diffs.
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // kimi_k2_tool_parser.py adjust_request (DEVIATION 4): when tools are active
  // (tools present AND tool_choice != "none"), Kimi's markers are special tokens
  // that MUST survive detokenization, so skip_special_tokens is forced false.
  // Mutates `request` in place.
  void adjust_request(ChatCompletionRequest& request) const;

  // The five marker byte strings (kimi_k2.py:43-47). ASCII, multi-char.
  static constexpr const char* kToolCallsSectionBeginToken =
      "<|tool_calls_section_begin|>";
  static constexpr const char* kToolCallsSectionEndToken =
      "<|tool_calls_section_end|>";
  static constexpr const char* kToolCallBeginToken = "<|tool_call_begin|>";
  static constexpr const char* kToolCallEndToken = "<|tool_call_end|>";
  static constexpr const char* kToolCallArgumentBeginToken =
      "<|tool_call_argument_begin|>";

 private:
  // hermes/deepseek-style streamed content cursor: how much leading content (the
  // text before the section) has already been sent. prev_tool_call_arr tracks
  // whether each index's name was emitted; streamed_args_for_tool the args sent.
  std::size_t sent_content_idx_ = 0;
};

}  // namespace vllm::entrypoints::openai
