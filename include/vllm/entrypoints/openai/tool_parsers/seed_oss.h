// REIMPLEMENTED FROM WIRE FORMAT: vllm/parser/seed_oss.py (SeedOssParser) +
// vllm/parser/qwen3.py (the shared Qwen3 XML grammar it inherits) +
// vllm/tool_parsers/seed_oss_engine_tool_parser.py @ e24d1b24.
//
// ByteDance Seed-OSS. Upstream backs it with the TOKEN-ID parser ENGINE
// (SeedOssParser subclasses Qwen3Parser: same transition table +
// _qwen3_arg_converter, only the four wrapper token STRINGS differ). We do NOT
// port the engine; we reimplement the WIRE FORMAT on the text seam, derived
// from the grammar's terminals + _qwen3_arg_converter, held to the upstream
// tests (tests/parser/engine/test_seed_oss.py + the shared-grammar behaviour in
// tests/parser/engine/test_qwen3.py).
//
// WIRE FORMAT (seed_oss's four overridden wrappers around the Qwen3 XML body):
//   <seed:tool_call>
//   <function=NAME>
//   <parameter=KEY>VALUE</parameter>
//   ...
//   </function>
//   </seed:tool_call>
// <function=..> and <parameter=..> are byte-identical to Qwen3 (only <think>/
// <tool_call> gain the `seed:` prefix). Per _qwen3_arg_converter every VALUE is
// a STRING (value.strip()); the parameter body is closed by </parameter> OR by
// the lookahead to the next <parameter= (a missing </parameter> does not merge
// two params). A malformed <function=NAME</function> header (no closing '>')
// must not drop sibling calls (regression #46314).
//
// DEVIATIONS from upstream (all forced by the TEXT-ONLY, tokenizer-free
// abstract.h seam, documented at their sites in seed_oss.cpp):
//   1. TOKEN-ID ENGINE -> text scan. The four token_id_terminals
//      (<seed:think>, </seed:think>, <seed:tool_call>, </seed:tool_call>) plus
//      <function=>/<parameter=> are recognized as TEXT substrings, not vocab
//      IDs. The seed_oss tool surface is fully determined by these literals.
//   2. REASONING (<seed:think>..</seed:think>) is a REASONING-parser concern;
//      this class handles ONLY tool calls (the seed_oss tool tests feed already
//      reasoning-stripped text to the tool parser). Content is the text before
//      the first tool marker.
//   3. NO SCHEMA TYPE COERCION. _qwen3_arg_converter emits every value as a
//      STRING; the engine's separate _fix_arg_types step (schema coercion) is
//      NOT modelled here. The seed_oss tool suite exercises only string values,
//      and the Qwen3 shared grammar keeps values as strings when NO tool schema
//      is supplied (test_qwen3.py::test_various_data_types). Documented; every
//      value stays the raw stripped string.
//   4. TOKEN IDs DROPPED from the streaming signature (text-only seam).
//   5. ordered_json throughout: json.dumps preserves dict INSERTION order and
//      the argument reconstruction only round-trips with stable key order.
#ifndef VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_SEED_OSS_H_
#define VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_SEED_OSS_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class SeedOssToolParser : public ToolParser {
 public:
  SeedOssToolParser() = default;

  // seed_oss_engine_tool_parser.py (extract_tool_calls, via ParserEngine).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // seed_oss_engine_tool_parser.py (extract_tool_calls_streaming, via
  // ParserEngine's incremental parse).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // seed_oss.py - the four wrapper overrides + the shared Qwen3 body tokens.
  static constexpr const char* kToolCallStart = "<seed:tool_call>";
  static constexpr const char* kToolCallEnd = "</seed:tool_call>";
  static constexpr const char* kFuncPrefix = "<function=";
  static constexpr const char* kFuncEnd = "</function>";
  static constexpr const char* kParamPrefix = "<parameter=";
  static constexpr const char* kParamEnd = "</parameter>";

  // qwen3.py:_qwen3_arg_converter - the <parameter=..>..</parameter> body ->
  // an object of STRING values. Exposed for direct arg-converter unit tests.
  static nlohmann::ordered_json ParseParameters(const std::string& body,
                                                bool partial = false);

 private:
  // Streaming state (mirrors the engine's per-request slot bookkeeping).
  int current_tool_id_ = -1;
  std::vector<bool> name_sent_;
  std::vector<bool> args_sent_;
  std::size_t streamed_content_len_ = 0;
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_TOOL_PARSERS_SEED_OSS_H_
