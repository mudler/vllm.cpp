// Ported from: vllm/tool_parsers/apertus_tool_parser.py @ e24d1b24
//
// ApertusToolParser (name "apertus") - for Swiss AI Apertus models. Format:
//   <|tools_prefix|>[{"function_name": {"arg1": "value1", ...}}, ...]<|tools_suffix|>
// a JSON ARRAY of SINGLE-KEY dicts (each {name: args}) sandwiched between the
// <|tools_prefix|> / <|tools_suffix|> special tokens (the suffix is optional for
// truncated outputs). Ports the NON-STREAMING extract_tool_calls and the
// STREAMING incremental parse (a partial-JSON prefix-diff differ with a
// partial-tag buffer so a special token split across chunks is not leaked).
//
// DEVIATIONS from upstream shape (all documented):
//   - Upstream __init__ requires a tokenizer (only to assert it is non-None and,
//     via adjust_request, to force skip_special_tokens=False so the markers reach
//     the parser). Our text seam surfaces the markers unconditionally, so the ctor
//     drops the tokenizer + the adjust_request hook entirely.
//   - The streaming signature drops the previous/current/delta TOKEN-ID spans -
//     the whole Apertus streaming parse is already TEXT-only (it buffers partial
//     TAG TEXT, never token ids).
//   - Upstream isolates a complete leading JSON value with json.JSONDecoder().
//     raw_decode (parse-prefix, return end index) and falls back to
//     partial_json_loads on JSONDecodeError. We have no raw_decode, so RawDecode()
//     (in the .cpp) reproduces it with partial_json_loads + is_complete_json: a
//     COMPLETE leading value yields {value, end_idx}; a truncated one yields
//     nullopt so the caller takes the partial-parse fallback. Byte-for-byte the
//     same branch selection.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class ApertusToolParser : public ToolParser {
 public:
  ApertusToolParser() = default;

  // apertus_tool_parser.py:176 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // apertus_tool_parser.py:273 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // apertus_tool_parser.py:43-44 - the special tokens.
  static constexpr const char* kToolsPrefix = "<|tools_prefix|>";
  static constexpr const char* kToolsSuffix = "<|tools_suffix|>";

 private:
  // apertus_tool_parser.py:136 (_buffer_delta_text) - hold back a chunk that ends
  // with a partial prefix/suffix tag until the next chunk clarifies it.
  std::string buffer_delta_text(const std::string& delta_text);
  // apertus_tool_parser.py:323 (_extract_streaming).
  std::optional<DeltaMessage> extract_streaming(const std::string& current_text,
                                                const std::string& delta_text);
  // apertus_tool_parser.py:404 (_parse_and_diff_json).
  std::vector<DeltaToolCall> parse_and_diff_json(const std::string& json_str,
                                                 bool is_final);
  // apertus_tool_parser.py:460 (_emit_tool_name).
  void emit_tool_name(const nlohmann::ordered_json& parsed, int index,
                      std::vector<DeltaToolCall>& tool_calls);
  // apertus_tool_parser.py:491 (_get_tool_diff).
  std::optional<DeltaToolCall> get_tool_diff(const nlohmann::ordered_json& parsed,
                                             int index, bool is_final);

  // Streaming state (apertus_tool_parser.py:103-114 _reset_streaming_state).
  std::string buffered_delta_text_;
  int current_tool_id_ = -1;
  bool current_tool_name_sent_ = false;
  std::vector<std::string> streamed_args_for_tool_;
};

}  // namespace vllm::entrypoints::openai
