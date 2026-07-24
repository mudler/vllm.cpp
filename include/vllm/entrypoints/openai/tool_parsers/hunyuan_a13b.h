// Ported from: vllm/tool_parsers/hunyuan_a13b_tool_parser.py @ e24d1b24
//
// HunyuanA13BToolParser (name "hunyuan_a13b") - for Tencent Hunyuan A13B. Format:
//   <tool_calls>[{"name":..,"arguments":{..}}, ...]</tool_calls>
// a JSON ARRAY of tool calls wrapped in the <tool_calls>...</tool_calls> markers,
// with two wrinkles the plain jamba format lacks: (a) a <tool_calls> block that
// sits INSIDE a <think>...</think> region is IGNORED (the model is still
// reasoning), and (b) the leading bot_string "<tool_calls>" is consumed with
// consume_space in the streaming path. Ports the NON-STREAMING extract_tool_calls
// (preprocess + JSON array parse) and the STREAMING incremental parse (a
// regex-over-the-full-text state machine, NOT a partial-JSON differ).
//
// DEVIATIONS from upstream shape (all documented):
//   - Upstream __init__ takes a tokenizer; it is NEVER read (hunyuan is a pure
//     text format), so the ctor drops it (same as every sibling in this seam).
//   - The streaming signature drops the previous/current/delta TOKEN-ID spans
//     (the streaming parse is TEXT-only; it re-scans current_text each call).
//   - Tool-call ids: upstream uses f"call_{random_uuid()}" (non-streaming) and
//     f"call_{idx}_{random_uuid()}" / f"chatcmpl-tool-{random_uuid()}"
//     (streaming). We emit make_tool_call_id() ("chatcmpl-tool-<hex>") in every
//     case - only uniqueness is load-bearing, and no test pins the prefix.
//   - _handle_test_compatibility keys off self.current_tools_sent, which upstream
//     initializes EMPTY and never populates on the normal path (its guard
//     `len(self.current_tools_sent) > 0` is therefore always False on a fresh
//     parser). We keep the branch for shape fidelity; it stays inert.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class HunyuanA13BToolParser : public ToolParser {
 public:
  HunyuanA13BToolParser() = default;

  // hunyuan_a13b_tool_parser.py:100 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // hunyuan_a13b_tool_parser.py:170 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // hunyuan_a13b_tool_parser.py:65 - the leading bot_string.
  static constexpr const char* kBotString = "<tool_calls>";

 private:
  // hunyuan_a13b_tool_parser.py:68-72 (streaming_state["sent_tools"] entries).
  struct SentTool {
    bool sent_name = false;
    bool sent_arguments_prefix = false;
    std::string sent_arguments;
  };

  // preprocess_model_output (hunyuan_a13b_tool_parser.py:74). Returns
  // {content, tool_calls_body}: tool_calls_body is nullopt when no VALID
  // <tool_calls> block outside <think> was found.
  std::pair<std::optional<std::string>, std::optional<std::string>>
  preprocess_model_output(const std::string& model_output) const;

  // Streaming helpers (mirroring the upstream private methods).
  void try_parse_json_tools(const std::string& text);
  std::optional<DeltaMessage> handle_test_compatibility(
      const std::string& current_text);
  void ensure_state_arrays(int tool_count);
  std::optional<DeltaMessage> handle_tool_name_streaming(
      int current_idx, int tool_count,
      const std::vector<std::string>& name_group1s);
  std::optional<DeltaMessage> handle_tool_args_streaming(
      const std::string& current_text, int current_idx, int tool_count);

  // Streaming state (hunyuan_a13b_tool_parser.py:39-72). hy_* prefixes avoid
  // colliding with the base ToolParser members (which hunyuan does not use).
  int hy_current_tool_id = -1;
  std::vector<std::string> hy_streamed_args;
  std::vector<bool> hy_current_tools_sent;  // inert (see header note)
  std::vector<nlohmann::ordered_json> hy_prev_tool_call_arr;
  int hy_current_tool_index = -1;
  std::vector<std::optional<std::string>> hy_tool_ids;
  std::vector<SentTool> hy_sent_tools;
};

}  // namespace vllm::entrypoints::openai
