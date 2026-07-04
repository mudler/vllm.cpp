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

}  // namespace vllm::v1
