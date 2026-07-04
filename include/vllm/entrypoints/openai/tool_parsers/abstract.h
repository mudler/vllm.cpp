// Ported from: vllm/tool_parsers/abstract_tool_parser.py @ e24d1b24
//
// The ToolParser ABC + the ExtractedToolCallInformation result shape
// (upstream: vllm/entrypoints/openai/engine/protocol.py:338) + the
// ToolParserManager factory (get_tool_parser). T0 scope is the NON-STREAMING
// extract_tool_calls path; the streaming method (extract_tool_calls_streaming)
// is Task 3 and is only declared here as a marked stub.
//
// DEVIATIONS from upstream shape (all T0-scoped):
//   - The upstream ToolParser.__init__ takes a tokenizer (+ tools). For the
//     gate model's non-streaming Hermes path the tokenizer is NEVER read
//     (it is only used for is_mistral_tokenizer detection and, in streaming,
//     the vocab / special-token ids). We therefore give the ABC a default
//     ctor and defer the tokenizer wiring to Task 3 (streaming).
//   - ToolParserManager's lazy/plugin registry collapses to a small hand-wired
//     get_tool_parser() factory over the two T0 formats.
#pragma once

#include <memory>
#include <string>
#include <vector>

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

  // TODO(M3.3 Task 3): extract_tool_calls_streaming (abstract_tool_parser.py:201)
  // — the incremental streaming parse (DeltaToolCall chunks). Not ported here.
};

// Ported from: vllm/tool_parsers/abstract_tool_parser.py:235
// (ToolParserManager.get_tool_parser). Returns a fresh parser for `name`, or
// nullptr when the name is not one of the T0-registered formats (upstream
// raises KeyError). Registered names: "hermes", "qwen3".
std::unique_ptr<ToolParser> get_tool_parser(const std::string& name);

// Ported from: vllm/entrypoints/chat_utils.py:1964 (make_tool_call_id). Returns
// a "chatcmpl-tool-<uuid>" id (random_uuid() => uuid4().hex). Uniqueness only.
std::string make_tool_call_id();

}  // namespace vllm::entrypoints::openai
