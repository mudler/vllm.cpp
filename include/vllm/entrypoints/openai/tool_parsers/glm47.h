// Reimplemented from the WIRE FORMAT of: vllm/tool_parsers/glm47_moe_tool_parser
// .py @ e24d1b24 (registry names "glm45" + "glm47"), whose upstream parser is the
// declarative TOKEN-ID ParserEngine (vllm/parser/glm47_moe.py + the engine under
// vllm/parser/engine/). This is NOT a port of that engine: it reimplements GLM's
// XML tool-call WIRE SURFACE directly on the text-only abstract.h seam, held to
// the upstream fidelity tests (tests/tool_parsers/test_glm47_moe_tool_parser.py,
// 12 cases + test_glm4_moe_tool_parser.py, 6 cases).
//
// WIRE FORMAT (glm47_moe.py docstring + the engine terminals/transitions):
//     <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
//   - NAME runs from just after <tool_call> to the FIRST <arg_key> or </tool_call>
//     (engine TOOL_NAME leaves only on ARG_KEY_START / TOOL_END), then is STRIPPED
//     (glm47_moe.py `_emit_name_delta` / `_handle_tool_end` .strip()). A trailing
//     newline after the name is thus absorbed into NAME and stripped away.
//   - Arguments are `<arg_key>K</arg_key>\s*<arg_value>V</arg_value>` pairs
//     (_ARG_RE, DOTALL). The KEY is STRIPPED; the VALUE is kept RAW (whitespace is
//     significant - _glm47_arg_converter does `params[key.strip()] = value`).
//   - A call may have ZERO arguments (NAME directly followed by </tool_call>).
//   - Text before the first <tool_call> is CONTENT; when a tool is called it is
//     .strip()-ed and dropped when empty (engine strip_content_whitespace_with_
//     tools=True default) -> content is None for empty / whitespace-only text.
//
// ARGUMENT TYPING. The GLM arg_converter emits RAW STRING values; the ENGINE then
// coerces each against its JSON-Schema type (parser_engine.py _fix_arg_types ->
// coerce_to_schema_type, priority null>int>number>bool>object>array>string). We
// reproduce that with the shared utils.h coerce_arg_value(): a `number`-typed
// "42" -> 42, `boolean` "true" -> true, `string` stays the raw (whitespace-
// preserving) string, an untyped/unlisted key stays a string. This is a DIFFERENT
// coercion from hy_v3/poolside's json.loads->literal->raw `_deserialize`; the
// <arg_key>/<arg_value> tag SURFACE is shared with those two (and reuses the same
// scanning shape) but each is held to ITS OWN upstream tests.
//
// DEVIATIONS from upstream (all text-seam scoped):
//   - No tokenizer / token ids. The engine gates reasoning + tool starts on token
//     ids; the tool-call adapter (adapters.py) starts the engine in CONTENT state,
//     so the <think> reasoning split is NOT part of this tool seam (a separate
//     reasoning parser owns it). We parse the decoded text starting from CONTENT.
//   - regex (_ARG_RE / _PARTIAL_ARG_RE, DOTALL) -> hand-written non-greedy
//     substring scans with identical match semantics for the tag grammar.
//   - Streaming: the engine streams pure-string arg values character-by-character
//     (stream_arg_deltas). We stream at PER-ARGUMENT granularity instead - each
//     completed <arg_value> emits its full coerced `"key": value` fragment, and
//     the object opens/closes with `{`/`}`. The CONCATENATION of the emitted
//     argument fragments is byte-identical valid JSON (what every upstream test
//     asserts via json.loads of the joined deltas); only the intra-value chunk
//     boundary differs, which no test observes. This sidesteps the prefix-
//     instability that per-tick re-coercion would introduce.
//   - Arguments re-serialized compact (nlohmann dump: no space after ':'/','),
//     json.loads-equal (the convention already used across this repo's parsers).
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

class Glm47ToolParser : public ToolParser {
 public:
  Glm47ToolParser() = default;

  // Wire markers (glm47_moe.py:37-42).
  static constexpr const char* kToolCallStart = "<tool_call>";
  static constexpr const char* kToolCallEnd = "</tool_call>";
  static constexpr const char* kArgKeyStart = "<arg_key>";
  static constexpr const char* kArgKeyEnd = "</arg_key>";
  static constexpr const char* kArgValStart = "<arg_value>";
  static constexpr const char* kArgValEnd = "</arg_value>";

  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  void BeginToolCall();
  void FinishToolCall();
  void EnsureToolState();
  void RevertLastToolCallState();
  DeltaToolCall& GetOrCreateDelta(std::map<int, DeltaToolCall>& pending);
  void UpdateToolName(std::map<int, DeltaToolCall>& pending,
                      const std::string& tool_name);
  std::optional<std::string> AppendArgFragment(const std::string& key,
                                               const std::string& raw_value,
                                               const ChatCompletionRequest& request);
  std::optional<std::string> CloseArgsIfNeeded();

  // Streaming state.
  std::string buffer_;
  bool in_tool_call_ = false;
  std::optional<std::string> current_tool_name_;
  std::optional<std::string> pending_key_;
  std::vector<std::string> tool_call_ids_;
  std::vector<bool> args_started_;
  std::vector<bool> args_closed_;
  std::vector<std::set<std::string>> seen_keys_;
};

}  // namespace vllm::entrypoints::openai
