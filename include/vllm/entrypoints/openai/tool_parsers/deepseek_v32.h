// Ported from: vllm/tool_parsers/deepseekv32_tool_parser.py @ e24d1b24
//
// DeepSeekV32ToolParser — the DeepSeek-V3.2 "DSML" tool-call format. A tool-call
// block is wrapped in <｜DSML｜function_calls> … </｜DSML｜function_calls>; each
// call is <｜DSML｜invoke name="NAME"> … </｜DSML｜invoke> and each argument is
// <｜DSML｜parameter name="KEY" string="true|false">VALUE</｜DSML｜parameter>.
// NOTE: the markers use the FULLWIDTH vertical bar ｜ (U+FF5C, UTF-8 EF BD 9C),
// NOT ASCII '|'. "DSML" and the tag words are plain ASCII. The markers are
// treated as OPAQUE byte strings (std::string::find / literal regex atoms),
// never indexed per character — a marker split across a delta boundary is simply
// "not yet found", and any partial-marker suffix is held back byte-wise so no
// broken UTF-8 leaks as content.
//
// Unlike deepseek_v3 (raw ```json args), the DSML format carries EACH argument
// as a separately-typed string; the parser reconstructs the JSON arguments
// object using the tool's JSON-Schema (`string="true"` keeps the literal string,
// `string="false"` coerces via the schema type). This is the richest tool parser
// in the registry (49 upstream unit cases). deepseek_v4 subclasses this parser,
// swapping ONLY the wrapper tokens (see deepseek_v4.h) — mirroring the upstream
// v4<-v32 subclass relationship exactly as longcat.{h,cpp} subclasses hermes.
//
// DEVIATIONS from upstream (reworks of the token-id/vocab machinery + the
// tokenizer-carried tool list, per the TEXT-ONLY abstract.h seam):
//   1. TOKEN-ID DETECTION -> TEXT FIND. Upstream `adjust_request` flips
//      skip_special_tokens so the DSML markers survive detokenization; our seam
//      is text-only (the markers arrive as literal bytes), so there is no
//      adjust_request — detection is a plain model_output.find(start-marker).
//   2. VOCAB / TOKENIZER DROPPED. Upstream __init__ requires a tokenizer and
//      raises without one; the C++ ToolParser ABC is tokenizer-free (abstract.h).
//   3. self.tools -> request.tools. Upstream captures the tool list at __init__
//      (self.tools) and drives schema coercion off it. The text-only seam has no
//      constructor tools, so we read request.tools at call time and cache a
//      pointer (tools_) for the duration of extract_tool_calls /
//      extract_tool_calls_streaming — behaviourally identical, and it lets a
//      factory-constructed parser coerce types too.
//   4. EOS finalizer gate. Upstream returns a sentinel DeltaMessage(content="")
//      when `not delta_text and delta_token_ids and self.prev_tool_call_arr`
//      (an empty text delta carrying an EOS/closing token). The seam has no token
//      ids, so we gate on `delta_text.empty() && !prev_tool_call_arr.empty()`.
//   5. _generate_tool_call_id keeps the upstream STREAMING id shape
//      "call_" + 24 hex chars (NOT the "chatcmpl-tool-" default_factory used by
//      the non-streaming ToolCall). Both are uniqueness-only.
//   6. structural_tag_model / get_structural_tag (grammar path) is NOT part of
//      the tool-parser seam and is omitted (recommended markers reported instead).
//   7. STRING-VALUE UTF-8 HOLD-BACK. Upstream streams already-decoded str, so the
//      incremental string escaper never sees a partial code point. The text-only
//      seam is BYTE-level (a multi-byte value can split across a delta), so the
//      string-mode incremental emit holds back a trailing incomplete UTF-8
//      sequence (Utf8SafeLen in the .cpp) — the value reassembles byte-identically
//      on the next delta. raw/buffered modes need no hold-back (bytes concatenate).
#pragma once

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

class DeepSeekV32ToolParser : public ToolParser {
 public:
  DeepSeekV32ToolParser() = default;
  // Out-of-line key function (defined in deepseek_v32.cpp) — anchors the vtable
  // and lets deepseek_v4 subclass cleanly.
  ~DeepSeekV32ToolParser() override;

  // deepseekv32_tool_parser.py:180 (extract_tool_calls).
  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // deepseekv32_tool_parser.py:526 (extract_tool_calls_streaming) — the stateful
  // DSML buffer state machine, reworked to the text-only seam (DEVIATIONS 3-5).
  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // deepseekv32_tool_parser.py:54-55 — the wrapper marker byte strings (fullwidth
  // ｜ U+FF5C). Copied EXACTLY from the upstream source.
  static constexpr const char* kToolCallStartToken = "<｜DSML｜function_calls>";
  static constexpr const char* kToolCallEndToken = "</｜DSML｜function_calls>";

 protected:
  // The wrapper tokens, exposed virtually so deepseek_v4 overrides ONLY these two
  // (its invoke/parameter grammar is identical). Mirrors the longcat<-hermes port.
  virtual const std::string& tool_call_start() const;
  virtual const std::string& tool_call_end() const;

 private:
  // A raw DSML parameter as scanned from the markup: (value, string_attr) keyed
  // by name, kept in insertion order (Python dict order) so the rebuilt JSON
  // arguments object preserves parameter order.
  using RawParam =
      std::pair<std::string, std::pair<std::string, std::string>>;  // name,(val,attr)

  // deepseekv32_tool_parser.py:232 (_reset_streaming_state).
  void reset_streaming_state();

  // deepseekv32_tool_parser.py:171 (_get_param_config): the tool's `properties`
  // schema object, or {} when the name/tools are absent.
  nlohmann::json get_param_config(const std::optional<std::string>& fn) const;

  // deepseekv32_tool_parser.py:153 (_convert_params_with_schema) +
  // :131 (_repair_param_dict). Raw params -> the typed JSON arguments object.
  nlohmann::ordered_json convert_params_with_schema(
      const std::string& fn, const std::vector<RawParam>& raw) const;

  // Streaming argument-emission helpers (deepseekv32_tool_parser.py:248-393).
  void add_tool_call_delta(std::map<int, DeltaToolCall>& deltas, int index,
                           const std::optional<std::string>& call_id,
                           const std::optional<std::string>& call_type,
                           const std::optional<std::string>& name,
                           const std::optional<std::string>& arguments);
  void begin_streaming_tool_call(const std::string& name,
                                 std::map<int, DeltaToolCall>& deltas);
  void append_param_prefix(std::map<int, DeltaToolCall>& deltas, int index,
                           const std::string& key, bool as_string);
  void append_json_param_value(std::map<int, DeltaToolCall>& deltas, int index,
                               const std::string& key,
                               const nlohmann::ordered_json& value);
  std::vector<std::string> param_types_for_name(const std::string& name) const;
  bool should_buffer_wrapper_param(const std::string& name) const;
  void finish_buffered_param(std::map<int, DeltaToolCall>& deltas, int index);
  void close_streaming_tool_call(std::map<int, DeltaToolCall>& deltas);
  void process_streaming_buffer(std::vector<std::string>& content_parts,
                                std::map<int, DeltaToolCall>& deltas);

  // request.tools for the in-flight call (DEVIATION 3). Non-owning; valid only
  // for the duration of one extract_tool_calls[_streaming] invocation.
  const std::vector<ChatCompletionToolsParam>* tools_ = nullptr;

  // Streaming state (deepseekv32_tool_parser.py:63-74). prev_tool_call_arr and
  // streamed_args_for_tool are inherited from ToolParser (abstract.h).
  int current_tool_index_ = 0;
  std::string buffer_;
  bool in_tool_calls_ = false;
  std::optional<int> active_tool_index_;
  std::optional<std::string> active_tool_name_;
  std::optional<std::string> active_param_name_;
  std::optional<std::string> active_param_string_attr_;
  std::optional<std::string> active_param_mode_;  // wrapper|buffered|string|raw
  std::vector<std::string> active_param_parts_;
  std::vector<bool> args_started_;
};

}  // namespace vllm::entrypoints::openai
