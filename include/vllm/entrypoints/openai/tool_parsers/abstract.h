// Ported from: vllm/tool_parsers/abstract_tool_parser.py @ e24d1b24
//
// The ToolParser ABC + the ExtractedToolCallInformation result shape
// (upstream: vllm/entrypoints/openai/engine/protocol.py:338) + the
// ToolParserManager factory (get_tool_parser). Covers the NON-STREAMING
// extract_tool_calls (Task 2) AND the STREAMING extract_tool_calls_streaming
// (Task 3 — the stateful incremental parse).
//
// DEVIATIONS from upstream shape (all T0-scoped):
//   - The upstream ToolParser.__init__ takes a tokenizer (+ tools). For the
//     gate model's Hermes path the tokenizer is NEVER read (it is only used for
//     is_mistral_tokenizer detection and, in the pythonic parsers, the vocab).
//     We therefore give the ABC a default ctor and drop the tokenizer wiring.
//   - The upstream streaming signature also takes the previous/current/delta
//     TOKEN-ID spans (abstract_tool_parser.py:201-217). The Hermes streaming
//     parse is TEXT-only (it never reads the token ids), so we drop those three
//     params to keep the serving seam small — documented deviation.
//   - ToolParserManager's lazy/plugin registry collapses to a small hand-wired
//     get_tool_parser() factory over the two T0 formats.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai {

// Ported from: vllm/entrypoints/openai/engine/protocol.py:338
// (ExtractedToolCallInformation). Per the OpenAI spec, content AND tool_calls
// can both be present (rare); `content` is None when there is no text before
// the first tool-call block.
struct ExtractedToolCallInformation {
  bool tools_called = false;
  std::vector<ToolCall> tool_calls;
  std::optional<std::string> content;
};

// Ported from: vllm/tool_parsers/abstract_tool_parser.py:43 (ToolParser).
// Abstract base; derived classes implement extract_tool_calls.
class ToolParser {
 public:
  ToolParser() = default;
  virtual ~ToolParser() = default;

  ToolParser(const ToolParser&) = delete;
  ToolParser& operator=(const ToolParser&) = delete;

  // Ported from: abstract_tool_parser.py:187 (extract_tool_calls). Extract tool
  // calls from a COMPLETE model-generated string (non-streaming). Stateless.
  virtual ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) = 0;

  // Ported from: abstract_tool_parser.py:201 (extract_tool_calls_streaming). The
  // INCREMENTAL streaming parse: given the accumulated `previous_text`, the new
  // `current_text` (= previous_text + delta_text) and the `delta_text` fragment,
  // return the DeltaMessage to emit (plain content before the first tool call,
  // or one/more DeltaToolCall entries: the name-first chunk then argument
  // deltas), or nullopt when there is nothing new to send yet. STATEFUL — the
  // parser instance carries the per-request streaming state (prev_tool_call_arr,
  // streamed_args_for_tool, …), so ONE parser must live for the whole stream.
  // Default: NotImplementedError-equivalent (abstract_tool_parser.py:218).
  virtual std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text, const ChatCompletionRequest& request);

 protected:
  // Streaming state (abstract_tool_parser.py:78-92 __init__). Held per request.
  //   prev_tool_call_arr: the tool call being parsed, one dict per index (we use
  //     a JSON object per entry — keys "name"/"arguments" set as they arrive).
  //   current_tool_id / current_tool_name_sent: the index of the tool currently
  //     being streamed + whether its name was already emitted (unused by Hermes,
  //     which tracks completion via prev_tool_call_arr; kept for shape fidelity).
  //   streamed_args_for_tool: per-index, the argument text already sent.
  std::vector<nlohmann::json> prev_tool_call_arr;
  int current_tool_id = -1;
  bool current_tool_name_sent = false;
  std::vector<std::string> streamed_args_for_tool;
};

// Ported from: vllm/tool_parsers/abstract_tool_parser.py:235
// (ToolParserManager.get_tool_parser). Returns a fresh parser for `name`, or
// nullptr when the name is not one of the T0-registered formats (upstream
// raises KeyError). Registered names: "hermes", "qwen3", "mistral".
// raises KeyError). Registered names: "hermes", "qwen3", "granite", "granite4",
// "granite-20b-fc".
std::unique_ptr<ToolParser> get_tool_parser(const std::string& name);

// Ported from: vllm/entrypoints/chat_utils.py:1964 (make_tool_call_id). Returns
// a "chatcmpl-tool-<uuid>" id (random_uuid() => uuid4().hex). Uniqueness only.
std::string make_tool_call_id();

}  // namespace vllm::entrypoints::openai
