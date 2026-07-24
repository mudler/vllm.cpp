// Ported from: vllm/tool_parsers/deepseekv3_tool_parser.py @ e24d1b24
//
// DeepSeekV3ToolParser — the DeepSeek-V3/R1 tool-call format. A tool-call block
// is wrapped in the OUTER markers <｜tool▁calls▁begin｜> … <｜tool▁calls▁end｜>
// and each call is <｜tool▁call▁begin｜>TYPE<｜tool▁sep｜>NAME\n```json\nARGS\n```
// <｜tool▁call▁end｜>. NOTE: those markers use the FULLWIDTH vertical bar (U+FF5C,
// UTF-8 EF BD 9C) and the LOWER-ONE-EIGHTH-block-style U+2581 (UTF-8 E2 96 81),
// NOT ASCII '|'/'_'. They are multi-byte UTF-8 and are treated here as opaque
// byte strings (std::string::find), never indexed per character.
//
// DEVIATIONS from upstream (all reworks of the token-id/vocab machinery to pure
// text, per the TEXT-ONLY abstract.h seam):
//   1. TOKEN-ID DETECTION -> TEXT FIND. Upstream gates on
//      `tool_calls_start_token_id not in current_token_ids` (needs the vocab id
//      of the marker). We gate on `model_output.find(<｜tool▁calls▁begin｜>)`.
//   2. VOCAB / TOKENIZER DROPPED. Upstream's __init__ looks the four marker
//      tokens up in self.vocab and raises if absent; that whole block is removed
//      (the C++ ToolParser ABC is tokenizer-free — see abstract.h).
//   3. STREAMING ALGORITHM REWORKED. Upstream streaming counts marker TOKEN IDS
//      across the previous/current/delta token-id spans
//      (deepseekv3_tool_parser.py:147-154) and drives an incremental state
//      machine off those counts + `len(delta_token_ids)`. Since we have no token
//      ids, streaming is reworked to the FULL-current_text re-parse + diff that
//      the existing hermes.cpp C++ port already uses: each call re-extracts every
//      <｜tool▁call▁begin｜>…<｜tool▁call▁end｜> region from current_text, emits the
//      tool name once (name-first) then the growing-argument diff. This is
//      text-only and UTF-8-safe (a marker split across a delta boundary is simply
//      not yet found, and a partial OUTER-begin suffix is held back), and its
//      reconstruction is identical to the non-streaming extract by construction.
#pragma once

#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class DeepSeekV3ToolParser : public ToolParser {
 public:
  DeepSeekV3ToolParser() = default;

  // deepseekv3_tool_parser.py:81 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // deepseekv3_tool_parser.py:125 (extract_tool_calls_streaming) — reworked to a
  // full-current_text re-parse + diff (see DEVIATION 3 above).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // deepseekv3_tool_parser.py:43-47 — the marker byte strings (fullwidth ｜ +
  // U+2581). Copied EXACTLY from the upstream source.
  static constexpr const char* kToolCallsBeginToken = "<｜tool▁calls▁begin｜>";
  static constexpr const char* kToolCallsEndToken = "<｜tool▁calls▁end｜>";
  static constexpr const char* kToolCallBeginToken = "<｜tool▁call▁begin｜>";
  static constexpr const char* kToolCallEndToken = "<｜tool▁call▁end｜>";
  static constexpr const char* kToolSepToken = "<｜tool▁sep｜>";

 protected:
  // Per-region extraction of (type, name, args). Virtual so DeepSeekV31 (which
  // has no ```json fence and hard-codes type="function") overrides ONLY the
  // per-region shape while reusing the wrapper-scanning + streaming logic.
  struct ParsedCall {
    std::optional<std::string> name;  // nullopt until the name has fully arrived
    std::string arguments;            // may be partial while streaming
    std::string type = "function";
  };
  // The non-streaming per-call regex (deepseekv3_tool_parser.py:49-51). Applied
  // over the whole output; each match yields one complete tool call.
  virtual const std::regex& tool_call_pattern() const;
  // Build a ToolCall from one regex match of tool_call_pattern().
  virtual ToolCall tool_call_from_match(const std::smatch& match) const;
  // Parse ONE region (the text between <｜tool▁call▁begin｜> and its
  // <｜tool▁call▁end｜>, or the unterminated tail) for streaming.
  virtual ParsedCall parse_region(const std::string& region) const;

 private:
  // hermes-style streamed state: how much leading content was sent + the args
  // already streamed per tool index (base ToolParser::streamed_args_for_tool is
  // reused for the latter; prev_tool_call_arr tracks whether the name was sent).
  std::size_t sent_content_idx_ = 0;
};

}  // namespace vllm::entrypoints::openai
