// vllm.cpp ORIGINAL component (§9 deviation). See json_schema_to_gbnf.h for the
// supported subset, deviations and limits.
//
// The lowering builds a GBNF string in two parts:
//   1. a fixed PRIMITIVES block (ws / string / number / integer / boolean /
//      null / value / object / array / member) — the JSON grammar the Task-4
//      GbnfParser consumes;
//   2. a `root ::= <expr>` plus any schema-specific rules (closed objects,
//      typed arrays, enum / const alternations) generated recursively.
#include "vllm/v1/structured_output/json_schema_to_gbnf.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::v1 {
namespace {

// The JSON primitive rules, in GBNF the Task-4 parser accepts. `ws` is an
// optional-whitespace rule (correctness-grade: accepts any JSON whitespace,
// including none, between structural tokens). `string` allows any non-control,
// non-quote, non-backslash byte (so arbitrary UTF-8 passes) plus the standard
// JSON escapes. `number` is the full JSON number grammar; `integer` its
// integer-only restriction. `value` / `object` / `array` are the generic
// (any-JSON) rules, referenced by json_object and by untyped sub-values.
const char* kPrimitivesBlock = R"GBNF(
ws ::= [ \t\n\r]*
hex ::= [0-9a-fA-F]
strchar ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
string ::= "\"" strchar* "\""
integer ::= "-"? ("0" | [1-9] [0-9]*)
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
value ::= object | array | string | number | boolean | null
member ::= string ws ":" ws value
object ::= "{" ws "}" | "{" ws member (ws "," ws member)* ws "}"
array ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
)GBNF";

// A JSON value (string / number / bool / null / object / array) -> a GBNF string
// literal matching the value's exact serialized bytes. dump() produces the
// canonical JSON serialization; we re-escape it for a GBNF double-quoted literal
// (escaping " and \ and control bytes so the literal matches byte-for-byte).
std::string JsonValueToGbnfLiteral(const nlohmann::json& value) {
  const std::string serialized = value.dump();
  std::string out = "\"";
  for (const char ch : serialized) {
    const auto b = static_cast<unsigned char>(ch);
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (b < 0x20) {
      // A control byte: emit a \xHH GBNF escape (the parser decodes \x).
      static const char* kHex = "0123456789abcdef";
      out += "\\x";
      out += kHex[(b >> 4) & 0xF];
      out += kHex[b & 0xF];
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}

// The recursive converter. Accumulates schema-specific rules in rules_ (name,
// body) and returns, for any schema node, a GBNF *element* string (a rule-ref
// name, a primitive name, a quoted literal, or a parenthesized group) that the
// caller can drop straight into a sequence.
class Converter {
 public:
  std::string Convert(const nlohmann::json& schema) {
    const std::string root_expr = Visit(schema);
    std::string out = kPrimitivesBlock;
    out += "\nroot ::= ";
    out += root_expr;
    out += "\n";
    for (const auto& rule : rules_) {
      out += rule.first;
      out += " ::= ";
      out += rule.second;
      out += "\n";
    }
    return out;
  }

 private:
  // Register a fresh named rule with the given body; return its name. Names are
  // "r<N>" — disjoint from every primitive rule name (none of which start "r"
  // followed by a digit) and from "root".
  std::string NewRule(std::string body) {
    std::string name = "r" + std::to_string(counter_++);
    rules_.emplace_back(name, std::move(body));
    return name;
  }

  // The GBNF element for a single (non-union, non-enum, non-const) type name.
  std::string VisitType(const std::string& type, const nlohmann::json& schema) {
    if (type == "object") return VisitObject(schema);
    if (type == "array") return VisitArray(schema);
    if (type == "string") return "string";
    if (type == "integer") return "integer";
    if (type == "number") return "number";
    if (type == "boolean") return "boolean";
    if (type == "null") return "null";
    throw std::runtime_error(
        "JsonSchemaToGbnf: unsupported JSON-schema type '" + type + "'");
  }

  std::string Visit(const nlohmann::json& schema) {
    // A boolean schema: true == any value, false == nothing (unsupported).
    if (schema.is_boolean()) {
      if (schema.get<bool>()) return "value";
      throw std::runtime_error(
          "JsonSchemaToGbnf: a `false` schema (matches nothing) is unsupported");
    }
    if (!schema.is_object()) {
      // A non-object, non-boolean schema node (e.g. an empty/absent schema) ->
      // any JSON value.
      return "value";
    }
    // Unsupported combinators: fail loud rather than mis-constrain.
    for (const char* key : {"$ref", "anyOf", "oneOf", "allOf", "not"}) {
      if (schema.contains(key)) {
        throw std::runtime_error(
            std::string("JsonSchemaToGbnf: unsupported schema keyword '") + key +
            "'");
      }
    }
    if (schema.contains("const")) {
      return JsonValueToGbnfLiteral(schema["const"]);
    }
    if (schema.contains("enum")) {
      const nlohmann::json& e = schema["enum"];
      if (!e.is_array() || e.empty()) {
        throw std::runtime_error(
            "JsonSchemaToGbnf: `enum` must be a non-empty array");
      }
      std::string body;
      bool first = true;
      for (const auto& v : e) {
        if (!first) body += " | ";
        first = false;
        body += JsonValueToGbnfLiteral(v);
      }
      return "(" + body + ")";
    }
    if (schema.contains("type")) {
      const nlohmann::json& t = schema["type"];
      if (t.is_string()) {
        return VisitType(t.get<std::string>(), schema);
      }
      if (t.is_array()) {
        // A union of types -> alternation.
        if (t.empty()) {
          throw std::runtime_error("JsonSchemaToGbnf: empty `type` array");
        }
        std::string body;
        bool first = true;
        for (const auto& tv : t) {
          if (!tv.is_string()) {
            throw std::runtime_error(
                "JsonSchemaToGbnf: `type` array entries must be strings");
          }
          if (!first) body += " | ";
          first = false;
          body += VisitType(tv.get<std::string>(), schema);
        }
        return "(" + body + ")";
      }
      throw std::runtime_error(
          "JsonSchemaToGbnf: `type` must be a string or array of strings");
    }
    // No type / enum / const: infer from structure, else any value.
    if (schema.contains("properties")) return VisitObject(schema);
    if (schema.contains("items")) return VisitArray(schema);
    return "value";
  }

  std::string VisitObject(const nlohmann::json& schema) {
    if (!schema.contains("properties")) {
      // No declared properties -> any JSON object.
      return "object";
    }
    const nlohmann::json& props = schema["properties"];
    if (!props.is_object()) {
      throw std::runtime_error(
          "JsonSchemaToGbnf: `properties` must be an object");
    }
    if (props.empty()) return "object";

    // Which keys are required (a JSON array of names).
    std::vector<std::string> required;
    if (schema.contains("required")) {
      const nlohmann::json& req = schema["required"];
      if (!req.is_array()) {
        throw std::runtime_error(
            "JsonSchemaToGbnf: `required` must be an array");
      }
      for (const auto& r : req) {
        if (r.is_string()) required.push_back(r.get<std::string>());
      }
    }
    auto is_required = [&](const std::string& key) {
      for (const std::string& r : required) {
        if (r == key) return true;
      }
      return false;
    };

    // Per property (in schema/key order): the "member" element
    //   "\"<key>\"" ws ":" ws <value-expr>
    // and whether it is required.
    struct Prop {
      std::string member_expr;
      bool required;
    };
    std::vector<Prop> ordered;
    for (auto it = props.begin(); it != props.end(); ++it) {
      const std::string key_literal =
          JsonValueToGbnfLiteral(nlohmann::json(it.key()));
      const std::string value_expr = Visit(it.value());
      Prop p;
      p.member_expr = key_literal + " ws \":\" ws " + value_expr;
      p.required = is_required(it.key());
      ordered.push_back(std::move(p));
    }

    // Build, from the tail forward, two rule families over the ordered members:
    //   first[i] : members i.. rendered with NO leading separator (the object's
    //              opening `"{" ws` precedes the first emitted member),
    //   rest[i]  : members i.. rendered when a member was ALREADY emitted (each
    //              emitted member here is prefixed by `ws "," ws`).
    // A required member is always emitted; an optional member may be skipped.
    // This yields any ordered subset of the optional keys (required always
    // present) with correct comma placement. `""` (an empty literal) is the
    // sentinel for "nothing more" (matches the empty string).
    const std::size_t n = ordered.size();
    std::vector<std::string> first(n + 1);
    std::vector<std::string> rest(n + 1);
    first[n] = "\"\"";
    rest[n] = "\"\"";
    for (std::size_t k = n; k-- > 0;) {
      const std::string& mem = ordered[k].member_expr;
      const std::string next_first = first[k + 1];
      const std::string next_rest = rest[k + 1];
      std::string first_body;
      std::string rest_body;
      if (ordered[k].required) {
        first_body = mem + " " + next_rest;
        rest_body = "ws \",\" ws " + mem + " " + next_rest;
      } else {
        first_body = mem + " " + next_rest + " | " + next_first;
        rest_body = "ws \",\" ws " + mem + " " + next_rest + " | " + next_rest;
      }
      first[k] = NewRule(std::move(first_body));
      rest[k] = NewRule(std::move(rest_body));
    }

    const std::string body =
        "\"{\" ws " + first[0] + " ws \"}\"";
    return NewRule(body);
  }

  std::string VisitArray(const nlohmann::json& schema) {
    if (!schema.contains("items")) {
      // No item schema -> any JSON array.
      return "array";
    }
    const nlohmann::json& items = schema["items"];
    if (items.is_array()) {
      // Tuple validation (positional items) is not constrained at T0 -> any
      // array. (Documented limit; still schema-valid output would need per-index
      // rules; deferred.)
      return "array";
    }
    const std::string item_expr = Visit(items);
    const std::string body = "\"[\" ws \"]\" | \"[\" ws " + item_expr +
                             " (ws \",\" ws " + item_expr + ")* ws \"]\"";
    return NewRule(body);
  }

  std::vector<std::pair<std::string, std::string>> rules_;
  int counter_ = 0;
};

}  // namespace

std::string JsonSchemaToGbnf(const nlohmann::json& schema) {
  Converter converter;
  return converter.Convert(schema);
}

std::string JsonObjectGbnf() {
  // The permissive any-JSON grammar: the primitives block + `root ::= value`.
  std::string out = kPrimitivesBlock;
  out += "\nroot ::= value\n";
  return out;
}

}  // namespace vllm::v1
