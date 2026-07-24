// Ported from: vllm/tool_parsers/hy_v3_tool_parser.py @ e24d1b24
//
// HYV3ToolParser (registry name "hy_v3") - Tencent Hunyuan v3 tool format. A
// suffix-parametrized XML dialect:
//     <tool_calls{S}> <tool_call{S}> NAME <tool_sep{S}>
//         <arg_key{S}>K</arg_key{S}> <arg_value{S}>V</arg_value{S}> ...
//     </tool_call{S}> </tool_calls{S}>
// where {S} is a per-tokenizer suffix (`token_suffix`, empty for the base
// tokenizer). Argument VALUES are coerced against the tool's JSON-Schema
// (bool -> int -> number -> json array/object -> string). Streaming sends the
// tool name first, then argument JSON incrementally, streaming pure-string
// values character-by-character and withholding the closing brace until the
// </tool_call> tag is seen.
//
// DEVIATIONS from upstream (all documented at their sites in hy_v3.cpp):
//   - The abstract streaming seam is TEXT-only (abstract.h). Upstream gates the
//     stream on TOKEN IDS (`tool_calls_start_token_id not in current_token_ids`)
//     and requires the tokenizer's vocab to contain the tag tokens; we rework
//     every token-id / vocab check to the equivalent TEXT check
//     (`tool_calls_start_token not in current_text`). No tokenizer is consulted.
//   - The upstream regex driver (tool_call_regex / tool_call_portion_regex /
//     func_args_regex, all DOTALL) is reproduced with hand-written non-greedy
//     substring scans - identical match semantics for the tag grammar, no
//     std::regex escaping of the suffix-parametrized literals.
//   - safe_literal_eval (Python ast.literal_eval) -> pythonic_core::parse_literal.
//   - The streaming JSON snapshot is built manually (as upstream does) but with
//     nlohmann compact value dumps; snapshot and final use ONE serializer so the
//     incremental prefixes stay byte-consistent. Whitespace-only difference vs
//     json.dumps (", "/": "); json.loads-equivalent.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class HYV3ToolParser : public ToolParser {
 public:
  // Upstream reads `token_suffix` from the tokenizer's init_kwargs; the base
  // tokenizer has none (suffix == ""). Exposed as a ctor argument so the
  // suffix-parametrized variants can be constructed directly.
  explicit HYV3ToolParser(std::string suffix = "");

  // hy_v3_tool_parser.py:371 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // hy_v3_tool_parser.py:407 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  std::optional<DeltaMessage> ExtractStreamingIncremental(
      const std::optional<DeltaMessage>& name_delta,
      const ChatCompletionRequest& request);
  void ResetStreamingToolState();

  std::string suffix_;
  std::string tool_calls_start_token_;
  std::string tool_calls_end_token_;
  std::string tool_call_start_token_;
  std::string tool_call_end_token_;
  std::string tool_sep_token_;
  std::string arg_key_start_token_;
  std::string arg_key_end_token_;
  std::string arg_value_start_token_;
  std::string arg_value_end_token_;

  // Streaming state (hy_v3_tool_parser.py:257-315).
  std::string buffer_;
  std::optional<std::string> streaming_tool_name_;
  nlohmann::ordered_json completed_args_ = nlohmann::ordered_json::object();
  std::optional<std::string> current_arg_key_;
  bool current_arg_is_string_ = false;
  std::size_t streamed_json_len_ = 0;
  std::string current_tool_call_id_;
};

}  // namespace vllm::entrypoints::openai
