// Reimplemented from the WIRE FORMAT of: vllm/tool_parsers/minimax_m2_tool_parser
// .py @ e24d1b24 (registry name "minimax_m2"), whose upstream parser is the
// declarative TOKEN-ID ParserEngine (vllm/parser/minimax_m2.py + the engine under
// vllm/parser/engine/). This is NOT a port of that engine: it reimplements
// MiniMax-M2's XML tool-call WIRE SURFACE directly on the text-only abstract.h
// seam, held to the upstream fidelity tests (tests/tool_parsers/
// test_minimax_m2_tool_parser.py, 19 cases - all STREAMING).
//
// WIRE FORMAT (minimax_m2.py docstring + the engine terminals/transitions):
//     <minimax:tool_call><invoke name="get_weather">
//     <parameter name="city">Seattle</parameter>
//     </invoke></minimax:tool_call>
//   - A `<minimax:tool_call>` ... `</minimax:tool_call>` WRAPPER holds one or more
//     `<invoke name="NAME"> ... </invoke>` blocks; EACH invoke becomes one tool
//     call (its own sequential index).
//   - The invoke/parameter NAME attribute may be double-quoted (`name="X"`),
//     single-quoted (`name='X'`) or unquoted (`name=X`), closed by `">` / `'>` /
//     `>` respectively (engine INVOKE_PREFIX_* / NAME_END_* terminals). The name
//     is STRIPPED.
//   - Arguments are `<parameter name="K">V</parameter>` tags. The VALUE is
//     STRIPPED (`_minimax_m2_arg_converter` does `params[name] = value.strip()`),
//     unlike GLM which keeps arg values raw. The regex value terminator is
//     `</parameter>` OR a lookahead to the next `<parameter name=` (a missing
//     close tag before the next parameter still closes the value).
//   - Text before the first `<minimax:tool_call>` is CONTENT (streamed verbatim);
//     between/after invokes inside the wrapper, incidental text is ignored.
//
// ARGUMENT TYPING. As with GLM, the family arg_converter emits RAW (here stripped)
// STRING values and the ENGINE coerces each against its JSON-Schema type
// (parser_engine.py _fix_arg_types -> coerce_to_schema_type). We reproduce that
// with the shared utils.h coerce_arg_value(): anyOf[string,null] "null" -> null
// but "Alice" -> "Alice"; an enum/string "none" or "nil" stays the STRING (only
// the literal "null" coerces to null); an anyOf[object,null] JSON body parses to
// an object; an untyped/unlisted key stays a string ("days" -> "5").
//
// STRUCTURAL TAGS: nullopt. Upstream registers a `minimax` structural-tag builder
// (_minimax_tool_tags) whose per-parameter content uses xgrammar's `minimax_xml`
// style - it renders the JSON schema AS `<parameter name=..>..` XML, NOT a JSON
// args object. The flat native structural tag (begin-literal + JSON content_schema
// + end-literal) cannot express that per-argument XML, exactly the deepseek_v32 /
// GLM per-arg-XML precedent - so no family builder is added and every tool_choice
// mode yields nullopt (documented in structural_tags.h; asserted in
// test_structural_tags.cpp).
//
// DETECTION: NOT added to detect.cpp (per the task constraint). `<minimax:tool_call>`
// is a distinctive marker - it does NOT contain the bare Hermes `<tool_call>`
// substring, and it is disjoint from MiniMax-M3's namespaced start token
// `]<]minimax[>[<tool_call>` (neither contains the other) - so it COULD be a clean
// template row, but minimax_m2 stays an EXPLICIT-ONLY family (select via
// --tool-call-parser minimax_m2) for now.
//
// DEVIATIONS from upstream (all text-seam scoped):
//   - No tokenizer / token ids. The tool-call adapter starts the engine in CONTENT
//     state, so the <think> reasoning split (config initial_state=REASONING) is
//     owned by a separate reasoning parser, not this tool seam.
//   - regex (_PARAM_RE / _PARTIAL_PARAM_RE, DOTALL) -> hand-written non-greedy
//     substring scans with identical match semantics.
//   - Streaming granularity is PER-ARGUMENT (see glm47.h): each completed
//     <parameter> emits its full coerced `"key": value` fragment; the object
//     opens/closes with `{`/`}`. The CONCATENATION of the emitted fragments is
//     byte-identical valid JSON (what every upstream test asserts via json.loads
//     of the joined deltas). Arguments serialized compact, json.loads-equal.
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

class MinimaxM2ToolParser : public ToolParser {
 public:
  MinimaxM2ToolParser() = default;

  // Wire markers (minimax_m2.py:30-42).
  static constexpr const char* kToolCallStart = "<minimax:tool_call>";
  static constexpr const char* kToolCallEnd = "</minimax:tool_call>";
  static constexpr const char* kInvokePrefix = "<invoke name=";
  static constexpr const char* kInvokeEnd = "</invoke>";
  static constexpr const char* kParamPrefix = "<parameter name=";
  static constexpr const char* kParamEnd = "</parameter>";

  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  // Streaming region within the parse.
  enum class Region { kContent, kWrapper, kArgs };

  void BeginToolCall();
  void FinishToolCall();
  void EnsureToolState();
  DeltaToolCall& GetOrCreateDelta(std::map<int, DeltaToolCall>& pending);
  void UpdateToolName(std::map<int, DeltaToolCall>& pending,
                      const std::string& tool_name);
  std::optional<std::string> AppendArgFragment(const std::string& key,
                                               const std::string& raw_value,
                                               const ChatCompletionRequest& request);
  std::optional<std::string> CloseArgsIfNeeded();
  void AppendArgsDelta(std::map<int, DeltaToolCall>& pending,
                       const std::string& frag);

  // Streaming state.
  std::string buffer_;
  Region region_ = Region::kContent;
  std::optional<std::string> current_tool_name_;
  std::vector<std::string> tool_call_ids_;
  std::vector<bool> args_started_;
  std::vector<bool> args_closed_;
  std::vector<std::set<std::string>> seen_keys_;
};

}  // namespace vllm::entrypoints::openai
