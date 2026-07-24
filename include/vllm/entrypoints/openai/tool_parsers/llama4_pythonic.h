// Ported from: vllm/tool_parsers/llama4_pythonic_tool_parser.py @ e24d1b24
//
// Llama4PythonicToolParser - the Llama 4 variant of the pythonic call format,
// wrapped in `<|python_start|>[...]<|python_end|>` markers. Upstream duplicates
// the whole PythonicToolParser body and differs ONLY by stripping those two
// markers before parsing; we mirror that reuse by subclassing
// PythonicToolParser and delegating to its (marker-agnostic) ExtractClean /
// StreamClean cores after the strip. All pythonic deviations (see pythonic.h)
// apply here unchanged.
#pragma once

#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/pythonic.h"

namespace vllm::entrypoints::openai {

class Llama4PythonicToolParser : public PythonicToolParser {
 public:
  Llama4PythonicToolParser() = default;

  // llama4_pythonic_tool_parser.py:67 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // llama4_pythonic_tool_parser.py:122 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // llama4_pythonic_tool_parser.py:74-78 - the wrapper markers.
  static constexpr const char* kPythonStartToken = "<|python_start|>";
  static constexpr const char* kPythonEndToken = "<|python_end|>";
};

}  // namespace vllm::entrypoints::openai
