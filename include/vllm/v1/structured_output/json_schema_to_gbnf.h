// vllm.cpp ORIGINAL component (§9 deviation) — NOT a 1:1 upstream port.
//
// JSON-Schema -> GBNF lowering feeding the M3.4 Task-4 native grammar engine.
// Upstream vLLM delegates JSON-schema constraint to xgrammar's
// json_schema_to_ebnf; since our engine is a from-scratch GBNF matcher, we port
// the SEMANTICS of a JSON-schema -> EBNF conversion (llama.cpp's
// json-schema-to-grammar is the canonical reference for the supported subset)
// into a GBNF string the Task-4 GbnfParser accepts (literals, char classes,
// `* + ? {m,n}`, alternation, rule refs, groups).
//
// SUPPORTED SUBSET (correctness-grade — the emitted grammar matches ONLY JSON
// conforming to the schema; on an unsupported construct it throws loudly rather
// than silently mis-constraining):
//   - type: object / array / string / number / integer / boolean / null, and a
//     `type` given as an array (a union -> alternation),
//   - object `properties` + `required` (required keys always present, optional
//     keys present-or-absent, in a FIXED key order — see the DEVIATIONS note),
//     a property-less object -> the generic any-object rule,
//   - array `items` (a homogeneous element schema); an item-less array -> the
//     generic any-array rule,
//   - `enum` (an alternation of the JSON-serialized values),
//   - `const` (the single JSON-serialized value),
//   - `anyOf` / `oneOf` (top-level OR nested): a GBNF alternation of the
//     sub-schemas — `root ::= (schemaA | schemaB | ...)`. Powers
//     tool_choice=required over multiple tools. `oneOf`'s exclusivity is
//     APPROXIMATED as a union (overlapping alternatives are both accepted);
//     exact for the disjoint tool-call schemas it is used for,
//   - string / number / integer / boolean / null primitives,
//   - nested objects / arrays (recursive rule generation),
//   - JSON inter-token whitespace flexibility (an optional-whitespace rule
//     between every structural token).
//
// DEVIATIONS / LIMITS (recorded — schema features NOT constrained at T0; the
// grammar stays a strict subset so output is always schema-valid):
//   - Object key ORDER is fixed to the parsed schema's `properties` iteration
//     order (nlohmann::json sorts object keys lexicographically). A conforming
//     JSON with keys in a different order is not matched; the constrained decode
//     simply emits keys in this canonical order (still schema-valid). This
//     mirrors llama.cpp's default fixed-order object grammar.
//   - `additionalProperties` is treated as CLOSED for objects that declare
//     `properties` (only the declared keys are emitted). A schema allowing extra
//     keys is thus constrained more tightly than it strictly permits — the
//     output remains valid.
//   - string `pattern` / `format`, numeric `minimum` / `maximum` /
//     `multipleOf`, array `minItems` / `maxItems` / tuple `items`, `$ref`,
//     `allOf`, `not` are NOT constrained (ignored where harmless; a
//     `$ref`/`allOf`/`not`/`false` schema throws). `anyOf`/`oneOf` ARE now
//     lowered (see the supported subset above). Documented, not silent.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::v1 {

// Lower a JSON Schema (already parsed) into a GBNF grammar string with a `root`
// rule matching exactly the JSON values conforming to `schema`. Throws
// std::runtime_error on an unsupported schema construct (never returns a grammar
// that would accept schema-INVALID JSON for the supported subset).
std::string JsonSchemaToGbnf(const nlohmann::json& schema);

// The permissive any-JSON grammar (OpenAI response_format `json_object`): a
// `root` rule matching any well-formed JSON value (object / array / string /
// number / boolean / null, arbitrarily nested).
std::string JsonObjectGbnf();

// Wrap a tool-call OBJECT schema into a GBNF grammar whose `root` emits the
// literal Hermes/Qwen `<tool_call>` wrapper around the schema-constrained JSON:
//
//   root ::= "<tool_call>\n" <schema-constrained JSON object> "\n</tool_call>"
//
// Used for tool_choice="required" / a named function: upstream vLLM forces the
// literal `<tool_call>` wrapper via an xgrammar StructuralTag
// (tool_parsers/structural_tag_registry.py:213-234 — begin `<tool_call>\n{"name":
// "`, end `}\n</tool_call>`) so the constrained decode is WRAPPED and the same
// Hermes regex parser (`<tool_call>(.*?)</tool_call>`, DOTALL) extracts it. Our
// json-only GBNF path emitted BARE JSON with no wrapper, so the parser's
// `find("<tool_call>")` guard dropped the forced call. This restores the wrapper
// IN THE GRAMMAR: the emitted output is `<tool_call>\n{...}\n</tool_call>`, which
// BOTH extract_tool_calls (non-stream) and extract_tool_calls_streaming extract.
//
// Implementation: lower `tool_call_object_schema` via JsonSchemaToGbnf, rename
// its generated `root` rule to an inner rule, and prepend a new `root` that wraps
// it in the `<tool_call>\n` … `\n</tool_call>` literals. The grammar REJECTS bare
// JSON (root's first byte must be `<`). Route it through structured_outputs.grammar
// (the kGrammar native compile path) — NOT structured_outputs.json.
std::string WrapSchemaAsToolCallGbnf(
    const nlohmann::json& tool_call_object_schema);

// One structural-tag body (M3.3b Task 2): a literal `begin`, a `content_schema`
// (a JSON schema, or boolean `true` == any JSON value), and a literal `end`.
// Mirrors vLLM's xgrammar TagFormat (begin / content=JSONSchemaFormat / end;
// tool_parsers/structural_tag_registry.py:224-231). `begin`/`end` are raw byte
// literals (NOT JSON — they carry the tag delimiters, e.g. `<tool_call>`).
struct StructuralTagBody {
  std::string begin;
  nlohmann::json content_schema;  // a schema object, or boolean `true`
  std::string end;
};

// Build a GBNF grammar whose `root` matches the structural-tag bodies (M3.3b
// Task 2 — the native STRUCTURAL_TAG compile; §9 original, the SEAM is 1:1 with
// vLLM but the emitted grammar is backend-private). Each tag lowers to
//   tagI ::= "<begin>" <content> "<end>"
// where <content> is the schema lowered via JsonSchemaToGbnf (or the any-JSON
// `value` rule when content_schema is `true`). MULTIPLE tags combine into an
// alternation. `stop_after_first` (vLLM TagsWithSeparatorFormat.stop_after_first,
// structural_tag_registry.py:260) => `root ::= (tag0 | tag1 | ...)` (exactly one
// tag); otherwise (at_least_one) => `root ::= (tag0 | ...)+` (one-or-more). The
// caller decides lazy vs forced (a lazy grammar wraps this GBNF with the
// triggers; a forced grammar compiles it directly). Throws on an empty tag list
// or a `false`/unsupported content schema.
std::string StructuralTagToGbnf(const std::vector<StructuralTagBody>& tags,
                                bool stop_after_first);

}  // namespace vllm::v1
