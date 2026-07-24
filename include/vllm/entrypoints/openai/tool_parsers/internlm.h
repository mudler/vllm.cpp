// Ported from: vllm/tool_parsers/internlm2_tool_parser.py @ e24d1b24
//
// Internlm2ToolParser (name "internlm") - for the InternLM2 models. Format:
//   <|action_start|><|plugin|>{"name":..,"parameters":{..}}<|action_end|>
// A single tool call wrapped in special markers (NO parallel calls). Ports both
// the NON-STREAMING extract_tool_calls and the STREAMING incremental parse.
//
// DEVIATIONS from upstream shape (all text-only-seam scoped):
//   - upstream's streaming signature takes previous/current/delta TOKEN-ID spans;
//     the internlm streaming parse is TEXT-only, so those params are dropped (the
//     same deviation the abstract/hermes/granite ports already document).
//   - upstream's partial_json_parser.loads -> our partial_json_loads (utils).
//   - upstream extract_tool_calls has NO try/except around json.loads, so a
//     malformed action body raises (upstream xfail_nonstreaming records exactly
//     this). We keep that behaviour: extract_tool_calls MAY throw on bad JSON.
//   - upstream reads request.tools to validate the tool name, but that branch is
//     DEAD CODE (it builds an ExtractedToolCallInformation without returning it),
//     so we drop it entirely (documented no-op).
#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class Internlm2ToolParser : public ToolParser {
 public:
  Internlm2ToolParser() = default;

  // internlm2_tool_parser.py:197 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // internlm2_tool_parser.py:57 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // internlm2_tool_parser.py:67,76 - the special markers.
  static constexpr const char* kActionStart = "<|action_start|>";
  static constexpr const char* kPluginStart = "<|action_start|><|plugin|>";
  static constexpr const char* kActionEnd = "<|action_end|>";

 private:
  // internlm2_tool_parser.py:37 (self.position) - how much of current_text has
  // already been consumed as leading content.
  std::size_t position_ = 0;
  // Insertion-ordered history of the single parsed tool call (upstream
  // self.prev_tool_call_arr; ordered so serialized argument key order is stable).
  std::optional<nlohmann::ordered_json> internlm_prev_tool_call_;
};

}  // namespace vllm::entrypoints::openai
