// Ported from: vllm/tool_parsers/minicpm5xml_tool_parser.py @ e24d1b24
//
// MiniCPM5XMLToolParser (registry name "minicpm5") - MiniCPM5 emits tool calls
// as an XML dialect:
//     <function name="get_weather">
//       <param name="city">Shanghai</param>
//       <param name="date">2024-06-27</param>
//     </function>
// Argument VALUES are coerced against the tool's JSON-Schema, OpenAI-style
// {"properties": {...}} / {"arguments": {...}} wrappers are unwrapped, unknown
// params are ignored, required params are enforced, and a table of MiniCPM5
// alias tool names is mapped onto the exposed schema. Streaming emits the tool
// name first, then argument JSON incrementally, withholding the closing brace
// until </function> is seen.
//
// DEVIATIONS from upstream (all documented at their sites in minicpm5.cpp):
//   - No lxml / xml.etree. Upstream tries an ElementTree parse first and only
//     falls back to a REGEX extraction when that fails; we implement the regex
//     path directly (it handles every in-scope input identically - CDATA,
//     collapsed tags, missing-name params, duplicates, wrappers). Malformed XML
//     that ET would nonetheless parse is out of scope for the ported cases.
//   - The abstract streaming seam is TEXT-only (abstract.h): the three token-id
//     spans upstream's signature carries are dropped (never read). adjust_request
//     (skip_special_tokens=False) is a serving-layer concern outside this seam
//     and is not modeled here.
//   - safe_literal_eval (Python ast.literal_eval) -> pythonic_core::parse_literal
//     for the single-quoted OpenAI-style wrappers json.loads cannot decode.
//   - Argument JSON re-serialized compact (no space after ':'/','); the streaming
//     diff uses ONE serializer so incremental prefixes stay byte-consistent.
//     json.loads-equivalent.
#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class MiniCPM5ToolParser : public ToolParser {
 public:
  MiniCPM5ToolParser() = default;

  // minicpm5xml_tool_parser.py:491 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // minicpm5xml_tool_parser.py:726 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  void ResetStreamState();
  std::optional<DeltaMessage> ProcessCompleteBlockStreaming(
      const std::string& block, const ChatCompletionRequest& request);
  std::optional<DeltaMessage> ProcessPartialBlockStreaming(
      const std::string& block, const ChatCompletionRequest& request);
  DeltaMessage StartToolCall(const std::string& func_name);
  std::optional<DeltaMessage> EmitToolArgsDelta(int tool_index,
                                                const std::string& args_json,
                                                bool is_complete);

  // minicpm5xml_tool_parser.py:472 (_processed_len). Per-request stream cursor.
  std::size_t processed_len_ = 0;
};

}  // namespace vllm::entrypoints::openai
