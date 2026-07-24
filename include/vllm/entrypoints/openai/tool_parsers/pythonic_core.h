// Shared pythonic-call-grammar core, extracted from
// vllm/tool_parsers/pythonic_tool_parser.py + vllm/tool_parsers/utils.py
// @ e24d1b24 (handle_single_tool, get_parameter_value, _ast_callable_dotted_name,
// make_valid_python, compute_tool_delta, safe_literal_eval).
//
// This is the strict recursive-descent stand-in for Python's `ast` module that
// both PythonicToolParser (the `[func(arg=val), ...]` list format) and
// Olmo3ToolParser (the `<function_calls>[func(arg=val)]</function_calls>`
// wrapper format) build on. Upstream olmo3_tool_parser.py copies the pythonic
// streaming body and reuses the shared utils.py helpers; we mirror that by
// hoisting the grammar + helpers here rather than duplicating them into
// olmo3.cpp.
//
// DEVIATIONS from upstream (also documented at the call sites):
//   - No Python `ast`. The grammar is a hand-written strict recursive-descent
//     parser: it rejects anything ast + get_parameter_value would reject
//     (non-literal args, bare identifiers other than True/False/None +
//     true/false/null, set literals `{a, b}`, mismatched brackets, a leading
//     '-' on a numeric literal — upstream's ast.UnaryOp is not an ast.Constant).
//   - make_valid_python here performs ONLY the bracket-closing + heuristic gates
//     (utils.py:405-462). It does NOT run the final ast.parse + no-Set
//     validation (utils.py:479-491); each caller re-parses the returned
//     candidate with parse_call_list (which itself rejects set literals and
//     non-call elements), so the net accept/reject decision is identical while
//     the shared helper stays format-agnostic.
//   - Argument JSON is re-serialized with nlohmann::ordered_json::dump()
//     (compact: no space after ':'/','), whereas upstream json.dumps() emits
//     ", "/": ". Semantically identical; key insertion order preserved.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"

namespace vllm::entrypoints::openai::pythonic_core {

// A parsed pythonic call: the (dotted) function name + its keyword arguments
// re-materialized as an ordered JSON object (insertion order preserved).
struct PyCall {
  std::string name;
  nlohmann::ordered_json arguments;  // always an object
};

// Thrown by compute_tool_delta on the withheld-suffix mismatch (upstream
// compute_tool_delta raises ValueError; callers catch it -> emit no delta).
struct ToolDeltaError : std::runtime_error {
  ToolDeltaError() : std::runtime_error("tool delta withheld-suffix mismatch") {}
};

// pythonic_tool_parser.py handle-single-tool path: parse the WHOLE text as
// `[ call (, call)* ]` (>= 1 call), then require only trailing whitespace.
// Returns nullopt on any malformed / non-call-literal input.
std::optional<std::vector<PyCall>> parse_call_list(const std::string& text);

// utils.py:36 (safe_literal_eval) analogue: parse the WHOLE text as a single
// Python literal (string / number / list / dict / True|False|None +
// true|false|null). Returns nullopt on any failure. Used by the XML dialects
// (minicpm5 / hy_v3) to decode single-quoted wrappers json.loads cannot.
std::optional<nlohmann::ordered_json> parse_literal(const std::string& text);

// utils.py:405 (make_valid_python). Close all open brackets/quotes to make a
// partial pythonic expression parseable. Returns (completed_text, added_suffix)
// or nullopt when the text is too incomplete to complete meaningfully (mid
// parameter-name / mid dict-key / etc.). Does NOT run the final AST validation
// (see DEVIATIONS above): the caller validates by re-parsing the candidate.
std::optional<std::pair<std::string, std::string>> make_valid_python(
    const std::string& text);

// utils.py:658 (compute_tool_delta). Returns the incremental DeltaToolCall, or
// nullopt when nothing new. Throws ToolDeltaError on the withheld-suffix
// mismatch (upstream raises ValueError -> caught -> None).
std::optional<DeltaToolCall> compute_tool_delta(const std::string& previously_sent,
                                                const std::string& call_id,
                                                const std::string& call_name,
                                                const std::string& call_args,
                                                int index,
                                                const std::string& withheld_suffix);

}  // namespace vllm::entrypoints::openai::pythonic_core
