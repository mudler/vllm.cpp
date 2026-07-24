// Ported from: vllm/tool_parsers/poolside_v1_tool_parser.py @ e24d1b24
//
// PoolsideV1ToolParser (registry name "poolside_v1") - a GLM-4-derived dialect.
// Tool calls are wrapped in a BARE <tool_call> block (like Hermes) but the body
// is a custom name + <arg_key>/<arg_value> XML grammar:
//     <tool_call>NAME
//     <arg_key>K</arg_key>
//     <arg_value>V</arg_value>
//     ...
//     </tool_call>
// STRING-typed argument values are kept VERBATIM (whitespace is significant, e.g.
// code / file bodies) and streamed CHARACTER-BY-CHARACTER; every other type is
// stripped then deserialized (json.loads -> python-literal -> raw string). The
// streaming path is a full incremental state machine that emits the tool name
// first, then the argument JSON built up fragment by fragment.
//
// DEVIATIONS from upstream (all per the TEXT-ONLY, tokenizer-free abstract.h
// seam):
//   - No vocab / token ids. Upstream reads the tokenizer vocab only to look up
//     the (never-behaviourally-used) start/end token ids and to force
//     skip_special_tokens=False via adjust_request. The seam models neither a
//     tokenizer nor adjust_request (a serving concern), so both are dropped; the
//     parse runs directly on the decoded text.
//   - The three upstream regexes (func_call_regex, func_detail_regex,
//     func_arg_regex, all DOTALL) are reproduced with hand-written non-greedy
//     substring scans - identical match semantics for this tag grammar.
//   - safe_literal_eval (ast.literal_eval) -> pythonic_core::parse_literal.
//   - partial_json_parser (the tolerant JSON completer used ONLY to fill the
//     internal prev_tool_call_arr["arguments"] snapshot) -> a best-effort full
//     parse: it never feeds an emitted delta (deltas are built from the manually
//     assembled streamed_args_for_tool), so dropping the partial completion does
//     not change any streamed output; only the internal snapshot is coarser.
//   - Arguments re-serialized compact (no space after ':'/','); json.loads-equal.
//   - Responses-API paths: the seam has only ChatCompletionRequest. The
//     required/named-choice adjust_request behaviour is out of scope (no
//     adjust_request hook), and the "wants logprobs" branch maps
//     request.is_include_output_logprobs() -> ChatCompletionRequest.logprobs.
//   - The abstract streaming seam is TEXT-only: the three token-id spans upstream
//     carries are dropped (this parse never reads them).
#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class PoolsideV1ToolParser : public ToolParser {
 public:
  PoolsideV1ToolParser() = default;

  // poolside_v1_tool_parser.py:188 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // poolside_v1_tool_parser.py:248 (extract_tool_calls_streaming).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // --- streaming state machine helpers (poolside_v1_tool_parser.py:455-606) ---
  void EnsureToolState();
  void BeginToolCall();
  void FinishToolCall();
  void RevertLastToolCallState();
  DeltaToolCall& GetOrCreateDelta(std::map<int, DeltaToolCall>& pending);
  void UpdateToolName(std::map<int, DeltaToolCall>& pending,
                      const std::string& tool_name);
  void UpdateToolArgs(std::map<int, DeltaToolCall>& pending,
                      const std::string& fragment);
  std::optional<std::string> AppendArgFragment(const std::string& key,
                                               const std::string& raw_val);
  std::optional<std::string> CloseArgsIfNeeded();

  // Whether `arg_name` on `tool_name` is declared JSON-Schema "string".
  bool IsStringType(const std::string& tool_name, const std::string& arg_name,
                    const ChatCompletionRequest& request) const;

  // Grammar tokens (poolside_v1_tool_parser.py:68-73).
  static constexpr const char* kToolCallStart = "<tool_call>";
  static constexpr const char* kToolCallEnd = "</tool_call>";
  static constexpr const char* kArgKeyStart = "<arg_key>";
  static constexpr const char* kArgKeyEnd = "</arg_key>";
  static constexpr const char* kArgValStart = "<arg_value>";
  static constexpr const char* kArgValEnd = "</arg_value>";

  // Streaming state (poolside_v1_tool_parser.py:93-103).
  std::string buffer_;
  bool in_tool_call_ = false;
  std::optional<std::string> current_tool_name_;
  std::optional<std::string> pending_key_;
  bool streaming_string_value_ = false;
  std::vector<std::string> tool_call_ids_;
  std::vector<bool> args_started_;
  std::vector<bool> args_closed_;
  std::vector<std::set<std::string>> seen_keys_;
};

}  // namespace vllm::entrypoints::openai
