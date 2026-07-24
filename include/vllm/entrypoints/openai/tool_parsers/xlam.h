// Ported from: vllm/tool_parsers/xlam_tool_parser.py @ e24d1b24
//
// xLAMToolParser (name "xlam") - for the Salesforce Llama-xLAM function-calling
// models. It accepts a JSON array of {"name","arguments"} tool calls in ANY of
// several envelopes:
//   - a bare JSON list:                 [{"name":..,"arguments":{..}}]
//   - a fenced code block:              ```json\n[..]\n```
//   - a [TOOL_CALLS]-prefixed list:     [TOOL_CALLS][..]
//   - <tool_call>..</tool_call> tags:   <tool_call>[..]</tool_call>
//   - after a </think> reasoning block: <think>..</think>[..]
// Ports the preprocess_model_output splitter, the NON-STREAMING
// extract_tool_calls, and the STREAMING incremental parse.
//
// DEVIATIONS from upstream shape:
//   - streaming token-id params dropped (text-only seam; same deviation the
//     other ports document).
//   - upstream carries a test-only `current_tools_sent` attribute that a white-
//     box unit test pokes to force a name emission; that hack is NOT part of the
//     serving path and is omitted here (documented; the corresponding upstream
//     unit test is not ported - see the test file).
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class xLAMToolParser : public ToolParser {
 public:
  xLAMToolParser() = default;

  // xlam_tool_parser.py:64 (preprocess_model_output). Returns {content,
  // potential_tool_calls_json}; either element may be unset (upstream None).
  std::pair<std::optional<std::string>, std::optional<std::string>>
  preprocess_model_output(const std::string& model_output) const;

  // xlam_tool_parser.py:124 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // xlam_tool_parser.py:188 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // xlam_tool_parser.py:58-62 (self.streaming_state) - per-tool streaming
  // bookkeeping.
  struct SentTool {
    bool sent_name = false;
    bool sent_arguments_prefix = false;
    std::string sent_arguments;
  };
  int current_tool_index_ = -1;              // streaming_state.current_tool_index
  std::vector<std::string> tool_ids_;        // streaming_state.tool_ids
  std::vector<SentTool> sent_tools_;         // streaming_state.sent_tools
};

}  // namespace vllm::entrypoints::openai
