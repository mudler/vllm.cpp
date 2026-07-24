// Ported from: vllm/tool_parsers/llama_tool_parser.py @ e24d1b24
//
// LlamaToolParser (upstream class Llama3JsonToolParser) - the JSON tool-call
// format used by Llama 3.x / 4 with the tool_chat_template_llama.jinja
// template, registered under BOTH names `llama3_json` and `llama4_json`. The
// model emits either a bare `{"name": ..., "parameters"|"arguments": {...}}`
// object or the same object prefixed by the `<|python_tag|>` bot token, and may
// emit several such objects separated by `"; "`. Ports the NON-STREAMING
// extract_tool_calls and the STREAMING extract_tool_calls_streaming.
//
// NOTE (parameters -> arguments): the Llama template emits the argument object
// under EITHER key; upstream maps `parameters` onto `arguments` (preferring
// `arguments` when both would be present). We mirror this exactly.
//
// DEVIATIONS from upstream (all documented at the helper sites in llama.cpp):
//   - Non-streaming: upstream drives a `re.compile(r"\{")` finditer +
//     json.JSONDecoder().raw_decode() to walk each top-level object. nlohmann
//     has no raw_decode (parse-one-value-return-end-index), so we hand-write a
//     string/escape-aware brace matcher to delimit each object, then parse it
//     with nlohmann::ordered_json (order-preserving, matching Python dict +
//     json.dumps order). Arguments are re-serialized compact (whitespace-only
//     difference vs json.dumps ", "/": " - see hermes.cpp / pythonic.h).
//   - Streaming: upstream uses `partial_json_parser` (incremental JSON with an
//     Allow bitmask that withholds partial STRING values until the tool name is
//     sent) + find_common_prefix. We reimplement that primitive as a tolerant
//     recursive JSON parser (`partial_json_loads`) with the same
//     allow-partial-string switch, plus a faithful port of find_common_prefix /
//     is_complete_json. Upstream ships NO streaming unit tests for this parser;
//     our streaming tests pin the observable behavior (name-first delta, then
//     monotonic argument-fragment deltas that reconstruct the compact JSON).
//   - The abstract streaming seam is TEXT-only (abstract.h): the token-id spans
//     and the `bot_token_id` vocab lookup (used only to fail construction when
//     the tokenizer lacks the token) are dropped; the bot token is matched as
//     literal text.
//   - The Python `regex` timeout path (VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS)
//     has no C++ analogue; the graceful plain-content fallback it guards is
//     preserved (malformed input -> content, tools_called=false).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class LlamaToolParser : public ToolParser {
 public:
  LlamaToolParser() = default;

  // llama_tool_parser.py:77 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // llama_tool_parser.py:170 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // llama_tool_parser.py:48 - the bot token the Llama template may prepend.
  static constexpr const char* kBotToken = "<|python_tag|>";

 private:
  // Streaming state. current_tool_id / current_tool_name_sent /
  // streamed_args_for_tool live on the base ToolParser. The base's
  // prev_tool_call_arr is nlohmann::json; the llama streaming diff needs the
  // ORDER-preserving ordered_json (arguments are serialized in the model's key
  // order), so we keep our own ordered copy here (mirrors
  // Llama3JsonToolParser.prev_tool_call_arr).
  std::vector<nlohmann::ordered_json> prev_tool_call_arr_;
};

}  // namespace vllm::entrypoints::openai
