// REIMPLEMENTED FROM WIRE FORMAT: vllm/parser/gemma4.py +
// vllm/tool_parsers/gemma4_engine_tool_parser.py + vllm/tool_parsers/
// gemma4_utils.py @ e24d1b24.
//
// Gemma4 (google/gemma-4-it) tool-call surface. Upstream backs this with a
// TOKEN-ID parser ENGINE (ParserEngine + gemma4_config's state machine, driven
// by the vocab's <|tool_call>/<tool_call|>/<|channel>/<channel|> special-token
// IDs). We do NOT port that engine; we reimplement the WIRE FORMAT on the text
// seam, derived from the grammar's terminals + arg_converter and held to the
// upstream test suite (tests/tool_parsers/test_gemma4_tool_parser.py, 54 cases).
//
// WIRE FORMAT (from gemma4_config terminals + _gemma4_arg_converter):
//   <|tool_call>call:NAME{ARGS}<tool_call|>
// where ARGS is Gemma4's own compact key:value dialect (NOT JSON):
//   - string values are wrapped in the delimiter <|"|> ... <|"|>
//       location:<|"|>San Francisco<|"|>
//   - bare values (numbers / bools / null) are unwrapped, kept as raw strings by
//       the arg parser and later TYPE-COERCED against the tool's JSON schema
//       (the engine's _fix_arg_types step): count:42  flag:true  param:null
//   - nested objects {..} and arrays [..] recurse with the same dialect
//   - keys may themselves be <|"|>-wrapped (stripped)
// NAME may contain hyphens / dots (the engine's TOOL_NAME state runs to the
// opening '{', it is not \w+).
//
// DEVIATIONS from upstream (all forced by the TEXT-ONLY, tokenizer-free
// abstract.h seam, and documented at their sites in gemma4.cpp):
//   1. TOKEN-ID ENGINE -> text scan. The grammar's four token_id_terminals
//      (<|tool_call>, <tool_call|>, <|channel>, <channel|>) are recognized as
//      TEXT substrings, not vocab IDs. The gemma4 tool-call surface is fully
//      determined by these literals, so the text seam reproduces the identical
//      extraction the engine's state machine yields for the tool-call path.
//   2. REASONING (<|channel>thought..<channel|>) is a REASONING-parser concern,
//      not a tool concern; this class handles ONLY tool calls (the 54-case tool
//      suite exercises no reasoning). Content is the text before the first
//      <|tool_call>. Matches the seam's other tool parsers (functiongemma,
//      step3p5): reasoning lives in a separate reasoning parser.
//   3. SCHEMA TYPE COERCION (the engine's _fix_arg_types /
//      coerce_to_schema_type, vllm/tool_parsers/utils.py) is reproduced here as
//      a local helper because the tool suite REQUIRES it (bare true/42/3.14 ->
//      bool/int/float per the tool schema). Tools arrive on the per-call
//      request (not the constructor) per the seam.
//   4. TOKEN IDs DROPPED from the streaming signature (the text-only seam does
//      not carry them).
//   5. STREAMING emits the tool NAME as soon as the '{' delimits it and defers
//      the (coerced) ARGUMENTS as a single chunk at the call's close. The 54
//      upstream streaming cases assert only the CONCATENATED arguments' validity
//      + no duplication / no leaked partial delimiters / correct types, none
//      require mid-call argument deltas; deferring the args to the close is the
//      simplest reconstruction that satisfies every case and sidesteps the
//      prefix-stability hazards of nested/empty partial objects.
//   6. ordered_json throughout: json.dumps preserves dict INSERTION order and
//      the argument reconstruction only round-trips with stable key order.
#ifndef VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_GEMMA4_H_
#define VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_GEMMA4_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Gemma4ToolParser : public ToolParser {
 public:
  Gemma4ToolParser() = default;

  // gemma4_engine_tool_parser.py (extract_tool_calls, via ParserEngine).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // gemma4_engine_tool_parser.py (extract_tool_calls_streaming, via
  // ParserEngine's incremental parse).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // gemma4.py terminals - the literals the wire format is built from.
  static constexpr const char* kToolCallStart = "<|tool_call>";
  static constexpr const char* kToolCallEnd = "<tool_call|>";
  static constexpr const char* kCallPrefix = "call:";
  static constexpr const char* kStringDelim = "<|\"|>";

  // gemma4.py:_parse_gemma4_args - the Gemma4 compact key:value dialect ->
  // an object of RAW (string / nested / list) values (no schema coercion yet).
  // Exposed for the direct helper unit tests (TestParseGemma4Args).
  static nlohmann::ordered_json ParseArgs(const std::string& args_str,
                                          bool partial = false);

  // gemma4.py:_parse_gemma4_array - the array counterpart (TestParseGemma4Array).
  static nlohmann::ordered_json ParseArray(const std::string& arr_str,
                                           bool partial = false);

 private:
  // Streaming state (mirrors the engine's per-request slot bookkeeping).
  int current_tool_id_ = -1;
  std::vector<bool> name_sent_;
  std::vector<bool> args_sent_;
  std::size_t streamed_content_len_ = 0;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_GEMMA4_H_
