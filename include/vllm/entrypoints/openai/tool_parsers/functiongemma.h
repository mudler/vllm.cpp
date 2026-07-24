// Ported from: vllm/tool_parsers/functiongemma_tool_parser.py @ e24d1b24
//
// FunctionGemmaToolParser (registry name "functiongemma") - Google's
// FunctionGemma (google/functiongemma-270m-it). Tool calls are wrapped in
// <start_function_call> ... <end_function_call> with a bespoke inner grammar:
//     <start_function_call>call:NAME{KEY:<escape>VALUE<escape>,...}<end_function_call>
// Each argument VALUE is JSON-decoded when it parses (numbers, bools, JSON
// literals) and otherwise kept as a raw string. Ports the NON-STREAMING
// extract_tool_calls and the STREAMING incremental state machine.
//
// DEVIATIONS from upstream (all per the TEXT-ONLY, tokenizer-free abstract.h
// seam):
//   - The two upstream regexes (tool_call_regex with its two alternatives, and
//     arg_regex, both DOTALL) are reproduced with hand-written non-greedy
//     substring scans - identical match semantics for this tag grammar, without
//     std::regex escaping of the `<`/`{` literals or ECMAScript's non-DOTALL `.`.
//   - VALUE decode: upstream `json.loads(value)` -> nlohmann::ordered_json::parse
//     with the same fall-back-to-raw-string on failure.
//   - Arguments re-serialized compact (no space after ':'/','); json.loads-equal.
//   - No adjust_request hook + no vocab wiring: upstream's adjust_request only
//     forces skip_special_tokens=False (a serving/tokenizer concern the text-only
//     seam does not model), and the parser reads no token ids. Both are dropped;
//     the parse operates directly on the decoded text.
//   - The abstract streaming seam is TEXT-only: the three token-id spans upstream
//     carries are dropped (this parse never reads them).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class FunctionGemmaToolParser : public ToolParser {
 public:
  FunctionGemmaToolParser() = default;

  // functiongemma_tool_parser.py:86 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // functiongemma_tool_parser.py:167 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // functiongemma_tool_parser.py:38-39 - the wrapper tokens.
  static constexpr const char* kToolCallStart = "<start_function_call>";
  static constexpr const char* kToolCallEnd = "<end_function_call>";

  // functiongemma_tool_parser.py:62 (_parse_arguments). Exposed for the direct
  // unit tests upstream runs against this helper (TestParseArguments).
  nlohmann::ordered_json ParseArguments(const std::string& args_str) const;

  // functiongemma_tool_parser.py:147 (_buffer_delta_text). Exposed for the direct
  // unit tests upstream runs against this helper (TestBufferDeltaText). Buffers
  // any trailing partial-tag suffix (returned only once the tag completes).
  std::string BufferDeltaText(const std::string& delta_text);

  // Test-visible view of the withheld partial-tag buffer (upstream
  // self.buffered_delta_text).
  const std::string& buffered_delta_text() const { return buffered_delta_text_; }

 private:
  // Streaming state (functiongemma_tool_parser.py:55-60).
  bool current_tool_name_sent_ = false;
  std::string buffered_delta_text_;
};

}  // namespace vllm::entrypoints::openai
