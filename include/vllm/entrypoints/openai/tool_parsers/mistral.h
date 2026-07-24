// Ported from: vllm/tool_parsers/mistral_tool_parser.py @ e24d1b24
//
// MistralToolParser - the Mistral tool-call wire formats. [BOT] is the literal
// "[TOOL_CALLS]" marker; content precedes the first [BOT]:
//   - v11+ (name-first, the "legacy" per-name form):
//       content[BOT]name1{args1}[BOT]name2{args2}
//       e.g. [TOOL_CALLS]add{"a": 3.5, "b": 4}
//   - pre-v11 (a single bracketed JSON array):
//       content[BOT] [{tool_call1}, {tool_call2}]
//       e.g. [TOOL_CALLS] [{"name": "add", "arguments":{"a": 3.5, "b": 4}}]
//
// Ports the NON-STREAMING extract_tool_calls (mistral_tool_parser.py:265) and
// the STREAMING incremental extract_tool_calls_streaming
// (mistral_tool_parser.py:381) for BOTH forms.
//
// DEVIATIONS from upstream (this is a TEXT-ONLY seam; upstream reads the
// tokenizer + the previous/current/delta token-id spans):
//  D1. Form selection. Upstream sets self._is_pre_v11 from the tokenizer at
//      construction (mistral version < 11, or "[ARGS]" absent from an HF vocab:
//      mistral_tool_parser.py:87-93,128). With no tokenizer, the form is an
//      explicit constructor flag `is_pre_v11` (default false == the modern v11
//      name-first form, which get_tool_parser("mistral") returns). One instance
//      parses ONE form, exactly like upstream (one parser per model/tokenizer).
//  D2. [BOT] detection. Upstream tests bot_token_id in the token-id spans OR the
//      literal in text (mistral_tool_parser.py:391-393,569). Text-only: we test
//      the literal "[TOOL_CALLS]". Content/argument splitting is already
//      string-based upstream, so this is faithful.
//  D3. pre-v11 streaming. Upstream drives an ijson coroutine + a _split_delta
//      state machine over the raw byte deltas (mistral_tool_parser.py:529-776);
//      those per-token delta boundaries are tokenizer/ijson-specific and not
//      reconstructible from text alone. We instead RE-PARSE the accumulated
//      current_text each call and diff against what was already sent (the same
//      strategy hermes.cpp uses): each tool is emitted as one name+arguments
//      delta once its JSON object is fully received; leading/trailing text
//      streams as content. The OBSERVABLE cadence is identical (name exactly
//      once, argument text concatenates to the full JSON string, index
//      increments per tool, content preserved).
//  D4. arguments whitespace. The pre-v11 form re-serializes arguments with
//      nlohmann dump() (compact) where upstream uses json.dumps default
//      (", "/": " separators): semantically identical JSON, whitespace differs
//      (the same deviation hermes.cpp records). The v11 form keeps the raw
//      argument substring verbatim, matching upstream exactly.
//  D5. tool-call id. Mistral ids are 9-char alphanumeric
//      (MistralToolCall.generate_random_id, mistral_tool_parser.py:76-80), NOT
//      the shared chatcmpl-tool-<uuid> scheme; ported as such.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class MistralToolParser : public ToolParser {
 public:
  // D1: `is_pre_v11` selects the wire form. Default false == modern v11
  // name-first form (get_tool_parser("mistral") uses the default).
  explicit MistralToolParser(bool is_pre_v11 = false);

  // mistral_tool_parser.py:265 (extract_tool_calls). Non-streaming; stateless.
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // mistral_tool_parser.py:381 (extract_tool_calls_streaming). Stateful
  // incremental parse; dispatches on the form (D1/D3).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // mistral_tool_parser.py:134 - the tool-call marker ("bot" == begin-of-tool).
  static constexpr const char* kBotToken = "[TOOL_CALLS]";

  // mistral_tool_parser.py:57 (StreamingState). Only the v11 name-first subset
  // is reached by the v11 streaming path; the rest are kept for shape fidelity.
  enum class StreamingState {
    WAITING_FOR_TOOL_START,
    PARSING_NAME,
    PARSING_ARGUMENTS,
    PARSING_ARGUMENTS_COMPLETED,
    TOOL_COMPLETE,
    ALL_TOOLS_COMPLETE,
  };

 private:
  // mistral_tool_parser.py:415 (_extract_tool_calls_streaming): the v11
  // name-first delta parse (delta_text-based, faithful).
  std::optional<DeltaMessage> extract_v11_streaming(const std::string& delta_text);
  // mistral_tool_parser.py:462 (_generate_delta_tool_call).
  std::vector<DeltaToolCall> generate_delta_tool_call(std::string delta_text);

  // pre-v11 streaming: re-parse current_text + diff (D3).
  std::optional<DeltaMessage> extract_pre_v11_streaming(
      const std::string& current_text);

  bool is_pre_v11_;

  // v11 streaming state (mistral_tool_parser.py:120-127).
  StreamingState streaming_state_ = StreamingState::WAITING_FOR_TOOL_START;
  std::optional<std::string> current_tool_name_;  // the pending name to emit

  // pre-v11 streaming state (D3): leading/trailing content already streamed.
  std::size_t pre11_content_sent_ = 0;
  std::size_t pre11_trailing_sent_ = 0;
};

}  // namespace vllm::entrypoints::openai
