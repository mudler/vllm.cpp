// REIMPLEMENTED-FROM-WIRE-FORMAT (not a line port).
//
// The Qwen3-Coder XML tool-call format, registered upstream under the names
// "qwen3_coder", "qwen3_xml" and "mimo" - all three resolve to
// Qwen3EngineToolParser (vllm/tool_parsers/qwen3_engine_tool_parser.py @
// e24d1b24), which is a thin subclass of Qwen3ParserToolAdapter backed by the
// TOKEN-ID scanning parser engine (vllm/parser/engine/*). This C++ parser does
// NOT port that engine. It reimplements the SAME WIRE FORMAT as a text parser on
// the existing text-only, tokenizer-free ToolParser seam (abstract.h), and is
// held to the upstream extraction tests
// (tests/tool_parsers/test_qwen3coder_tool_parser.py), which are text-in /
// extractions-out and fully portable.
//
// GRAMMAR SOURCE (the exact XML surface). vllm/parser/qwen3.py documents and
// encodes the grammar:
//
//     <tool_call>
//     <function=NAME>
//     <parameter=KEY>
//     VALUE
//     </parameter>
//     ...
//     </function>
//     </tool_call>
//
//   - The outer <tool_call>/</tool_call> wrapper is OPTIONAL for extraction:
//     qwen3.py has a CONTENT -> FUNC_PREFIX fallback transition, so a bare
//     <function=...> starts a call with no preceding <tool_call> (the
//     fallback_no_tags / missing_opening_tag tests).
//   - Parameter bodies are matched by qwen3.py's _PARAM_RE, whose value capture
//     is terminated by EITHER </parameter> OR the lookahead of the next
//     <parameter=. That lookahead is what lets a MISSING </parameter> still
//     parse (the missing-closing-tag tests): the value runs up to the next
//     parameter open. We additionally terminate a trailing value at </function>
//     (the last parameter of a call with no closing </parameter>).
//   - qwen3.py's TOOL_NAME -> FUNC_END transition (a </function> reached while
//     still reading the name, i.e. no closing '>') means a malformed
//     <function=NAME  (no '>') is dropped, not crashed (the no_gt / none-filtered
//     regressions, PR #36774). We drop a function whose name open has a '<'
//     before its '>'.
//   - Each raw parameter value is `.strip()`ed (qwen3.py _qwen3_arg_converter,
//     `value.strip()`).
//
// TYPED VALUES. The upstream converter emits `.strip()`ed STRINGS and the
// parameter values are TYPED against the tool's JSON schema (the same behaviour
// the fidelity tests pin: int/float/bool/object/array parsed, strings kept raw).
// We coerce per the resolved schema type, similar to step3p5's coercion:
//   - string-like            -> raw string value (never re-serialized/escaped).
//   - integer/number/boolean/object/array/null (any non-string schema type) ->
//     nlohmann::json::parse(value); on a parse failure we DEGRADE to the raw
//     string (graceful, matching the tolerant upstream converter).
//   - no schema / unknown type -> raw string (the seam carries tools on the
//     REQUEST, see DEVIATION 2; when a request lists no tools every value stays
//     a string, exactly the untyped converter output).
// Schema type resolution understands Pydantic-v2 nullable shapes: a `type` given
// as an ARRAY (["integer","null"]) and `anyOf`/`oneOf` unions both resolve to
// their FIRST non-"null" member type (the anyof / type-as-array tests).
//
// DEVIATIONS from upstream (knowing, documented):
//   1. ENGINE DROPPED. Upstream scans TOKEN IDS through a state-machine parser
//      engine (vllm/parser/engine/*) with an incremental lexer over the four
//      wrapper token strings. This is a pure TEXT parser: it scans the decoded
//      string surface directly. The wire format - not the token scan - is the
//      contract, and the text surface reproduces every extraction test.
//   2. TOOLS via REQUEST, not constructor. Upstream constructs the parser with
//      `tools=...` and reads self.tools; the C++ seam carries tools on the
//      per-call ChatCompletionRequest (same as step3p5 DEVIATION 3), so each
//      call re-points type resolution at request.tools. The upstream tests set
//      tools at construction AND pass a tools-less request; the ported C++ tests
//      put the tools on the request (the only place the seam exposes them).
//   3. TOKEN IDS DROPPED from the streaming signature (same as abstract.h): the
//      text-only seam passes previous/current/delta TEXT only.
//   4. STREAMING is a DIFF over the accumulated text, not an incremental token
//      lexer. On every streaming call the whole `current_text` is re-parsed and
//      only the NEW content / tool headers / appended-argument bytes are emitted.
//      Argument bytes are built so successive snapshots are PREFIX-EXTENSIONS
//      (`{"k": v`, then `, "k2": v2`, then the closing `}` only once the function
//      closes), so the per-call arguments deltas concatenate to exactly one valid
//      JSON document - the property the streaming tests assert. A parameter is
//      only emitted once its value terminator is seen (so a value never has to be
//      revised), which the upstream engine's per-parameter ARG_VALUE_CHUNK events
//      also guarantee at coarser granularity.
//   5. ORDERED json (nlohmann::ordered_json): json.dumps preserves dict INSERTION
//      order and the streaming prefix-extension only holds with stable key order.
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

namespace qwen3_coder_detail {
class Qwen3CoderStreamParser;  // defined in qwen3_coder.cpp
}  // namespace qwen3_coder_detail

// The Qwen3-Coder / qwen3_xml / mimo XML tool-call parser (text reimplementation
// of the qwen3.py wire format). One instance carries the per-request streaming
// state, so a single parser must live for the whole stream.
class Qwen3CoderToolParser : public ToolParser {
 public:
  Qwen3CoderToolParser();
  ~Qwen3CoderToolParser() override;

  ExtractedToolCallInformation extract_tool_calls(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  std::optional<DeltaMessage> extract_tool_calls_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

 private:
  std::unique_ptr<qwen3_coder_detail::Qwen3CoderStreamParser> parser_;
};

}  // namespace vllm::entrypoints::openai
