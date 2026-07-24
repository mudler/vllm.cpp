// Ported from: vllm/tool_parsers/utils.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

namespace {

bool IsJsonSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

// json.dumps(obj, ensure_ascii=False). nlohmann dump() is compact (no space
// after ':'/','); upstream CPython emits ", "/": ". The JSON is semantically
// identical (clients json.loads it) - whitespace-only deviation, matching the
// convention already established in hermes.cpp.
std::string DumpArgs(const nlohmann::ordered_json& v) { return v.dump(); }

// Python truthiness of a parsed "arguments" value: None / "" / {} / [] / 0 /
// false are falsy (granite guards arg streaming with `if cur_arguments:`).
bool JsonTruthy(const nlohmann::ordered_json& v) {
  if (v.is_null()) return false;
  if (v.is_string()) return !v.get<std::string>().empty();
  if (v.is_object() || v.is_array()) return !v.empty();
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number_float()) return v.get<double>() != 0.0;
  if (v.is_number()) return v.get<int64_t>() != 0;
  return true;
}

// ─── Tolerant JSON parser (stands in for partial_json_parser; see utils.h) ────
//
// Recursive-descent parser that COMPLETES a truncated document: on hitting the
// end of input inside a container it closes it, and inside an object it drops
// the trailing key/value pair that has not fully arrived. `allow_partial_str_`
// mirrors Allow.STR - when false an unterminated string value is dropped.
struct NeedMore {};   // an incomplete token that must be dropped by its parent
struct Malformed {};  // nothing parseable at this position

class TolerantParser {
 public:
  TolerantParser(const std::string& s, bool allow_partial_str)
      : s_(s), allow_partial_str_(allow_partial_str) {}

  nlohmann::ordered_json Parse() {
    SkipWs();
    if (pos_ >= s_.size()) throw Malformed{};
    return ParseValue();
  }
  std::size_t pos() const { return pos_; }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;
  bool allow_partial_str_;

  void SkipWs() {
    while (pos_ < s_.size() && IsJsonSpace(s_[pos_])) ++pos_;
  }

  nlohmann::ordered_json ParseValue() {
    SkipWs();
    if (pos_ >= s_.size()) throw NeedMore{};
    const char c = s_[pos_];
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') return ParseString(/*as_key=*/false);
    if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();
    if (c == 't' || c == 'f' || c == 'n') return ParseKeyword();
    throw Malformed{};
  }

  nlohmann::ordered_json ParseString(bool as_key) {
    const std::size_t start = pos_;  // at opening quote
    ++pos_;
    std::string partial;
    bool closed = false;
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (c == '\\') {
        if (pos_ + 1 >= s_.size()) {  // dangling backslash -> truncated
          ++pos_;
          break;
        }
        const char e = s_[pos_ + 1];
        switch (e) {  // best-effort unescape for the PARTIAL fallback only
          case 'n': partial += '\n'; break;
          case 't': partial += '\t'; break;
          case 'r': partial += '\r'; break;
          case 'b': partial += '\b'; break;
          case 'f': partial += '\f'; break;
          default: partial += e; break;  // \" \\ \/ and (approx) \uXXXX
        }
        pos_ += 2;
        continue;
      }
      if (c == '"') {
        ++pos_;
        closed = true;
        break;
      }
      partial += c;
      ++pos_;
    }
    if (closed) {
      // Complete token: let nlohmann do the authoritative unescaping (\uXXXX,
      // surrogate pairs, raw UTF-8 all handled correctly).
      return nlohmann::ordered_json::parse(s_.substr(start, pos_ - start));
    }
    if (as_key) throw NeedMore{};                 // an unterminated key -> drop
    if (allow_partial_str_) return nlohmann::ordered_json(partial);
    throw NeedMore{};                             // Allow.STR off -> drop
  }

  nlohmann::ordered_json ParseNumber() {
    const std::size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
    std::string num = s_.substr(start, pos_ - start);
    // Trim trailing incomplete exponents/decimals (e.g. "12." / "1e") until it
    // parses, mirroring the completer's best-effort numeric handling.
    while (!num.empty()) {
      if (nlohmann::json::accept(num)) return nlohmann::ordered_json::parse(num);
      num.pop_back();
    }
    throw NeedMore{};
  }

  nlohmann::ordered_json ParseKeyword() {
    static const std::pair<const char*, nlohmann::ordered_json> kws[] = {
        {"true", nlohmann::ordered_json(true)},
        {"false", nlohmann::ordered_json(false)},
        {"null", nlohmann::ordered_json(nullptr)}};
    for (const auto& kw : kws) {
      const std::string word = kw.first;
      if (s_.compare(pos_, word.size(), word) == 0) {
        pos_ += word.size();
        return kw.second;
      }
    }
    // A truncated prefix that runs to end-of-input completes to its keyword
    // (partial_json_parser turns "tru" -> true).
    for (const auto& kw : kws) {
      const std::string word = kw.first;
      const std::size_t rem = s_.size() - pos_;
      if (rem > 0 && rem < word.size() &&
          word.compare(0, rem, s_, pos_, rem) == 0) {
        pos_ = s_.size();
        return kw.second;
      }
    }
    throw Malformed{};
  }

  nlohmann::ordered_json ParseArray() {
    nlohmann::ordered_json arr = nlohmann::ordered_json::array();
    ++pos_;  // consume '['
    for (;;) {
      SkipWs();
      if (pos_ >= s_.size()) return arr;             // truncated -> complete
      if (s_[pos_] == ']') {
        ++pos_;
        return arr;
      }
      nlohmann::ordered_json v;
      try {
        v = ParseValue();
      } catch (const NeedMore&) {
        return arr;  // incomplete trailing element dropped
      }
      arr.push_back(std::move(v));
      SkipWs();
      if (pos_ >= s_.size()) return arr;
      if (s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (s_[pos_] == ']') {
        ++pos_;
        return arr;
      }
      return arr;  // unexpected -> stop and complete
    }
  }

  nlohmann::ordered_json ParseObject() {
    nlohmann::ordered_json obj = nlohmann::ordered_json::object();
    ++pos_;  // consume '{'
    for (;;) {
      SkipWs();
      if (pos_ >= s_.size()) return obj;
      if (s_[pos_] == '}') {
        ++pos_;
        return obj;
      }
      if (s_[pos_] != '"') return obj;  // trailing junk / bad key -> complete
      std::string key;
      try {
        key = ParseString(/*as_key=*/true).get<std::string>();
      } catch (const NeedMore&) {
        return obj;  // incomplete key -> drop the pair
      }
      SkipWs();
      if (pos_ >= s_.size()) return obj;   // no colon yet -> drop pair
      if (s_[pos_] != ':') return obj;
      ++pos_;                              // consume ':'
      SkipWs();
      if (pos_ >= s_.size()) return obj;   // no value yet -> drop pair
      nlohmann::ordered_json v;
      try {
        v = ParseValue();
      } catch (const NeedMore&) {
        return obj;  // incomplete value -> drop pair
      }
      obj[key] = std::move(v);
      SkipWs();
      if (pos_ >= s_.size()) return obj;
      if (s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (s_[pos_] == '}') {
        ++pos_;
        return obj;
      }
      return obj;  // unexpected -> stop and complete
    }
  }
};

}  // namespace

std::size_t consume_space(std::size_t i, const std::string& s) {
  while (i < s.size() && IsJsonSpace(s[i])) ++i;
  return i;
}

std::string find_common_prefix(const std::string& a, const std::string& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return a.substr(0, i);
}

std::string find_common_suffix(const std::string& a, const std::string& b) {
  // utils.py:78 - grow the shared suffix while both chars match AND the char is
  // NOT alphanumeric (so it never eats a partially-streamed word/number).
  std::string suffix;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 1; i <= n; ++i) {
    const char ca = a[a.size() - i];
    const char cb = b[b.size() - i];
    const bool alnum = std::isalnum(static_cast<unsigned char>(ca)) != 0;
    if (ca == cb && !alnum) {
      suffix.insert(suffix.begin(), ca);
    } else {
      break;
    }
  }
  return suffix;
}

std::string extract_intermediate_diff(const std::string& curr,
                                      const std::string& old_in) {
  // utils.py:96. The Python uses reversed-string .replace(...,1) to strip the
  // LAST occurrence of the suffix, and .replace(prefix,"",1) to strip the FIRST
  // occurrence of the prefix.
  const std::string suffix = find_common_suffix(curr, old_in);

  std::string old_stripped = old_in;
  if (!suffix.empty()) {
    const std::size_t p = old_stripped.rfind(suffix);
    if (p != std::string::npos) old_stripped.erase(p, suffix.size());
  }
  const std::string prefix = find_common_prefix(curr, old_stripped);

  std::string diff = curr;
  if (!suffix.empty()) {
    const std::size_t p = diff.rfind(suffix);
    if (p != std::string::npos) diff.erase(p, suffix.size());
  }
  if (!prefix.empty()) {
    const std::size_t p = diff.find(prefix);
    if (p != std::string::npos) diff.erase(p, prefix.size());
  }
  return diff;
}

bool is_complete_json(const std::string& s) { return nlohmann::json::accept(s); }

std::pair<nlohmann::ordered_json, std::size_t> partial_json_loads(
    const std::string& s, bool allow_partial_str) {
  TolerantParser p(s, allow_partial_str);
  nlohmann::ordered_json v;
  try {
    v = p.Parse();
  } catch (const NeedMore&) {
    throw MalformedPartialJson{};
  } catch (const Malformed&) {
    throw MalformedPartialJson{};
  }
  return {std::move(v), p.pos()};
}

std::optional<DeltaMessage> granite_stream_emit(
    std::vector<nlohmann::ordered_json>& prev_tool_call_arr, int& current_tool_id,
    bool& current_tool_name_sent, std::vector<std::string>& streamed_args_for_tool,
    const std::vector<nlohmann::ordered_json>& tool_call_arr,
    const std::vector<bool>& is_complete) {
  // granite_tool_parser.py:150-153 - nothing streamed yet (only brackets).
  if (tool_call_arr.empty()) return std::nullopt;

  // granite_tool_parser.py:154 - current_tool_call = tool_call_arr[current_tool_id];
  // Python negative index (-1 initially) selects the last element.
  const auto norm = [&](int id) -> const nlohmann::ordered_json& {
    const int n = static_cast<int>(tool_call_arr.size());
    const int idx = id < 0 ? n + id : id;
    return tool_call_arr[static_cast<std::size_t>(idx)];
  };

  std::optional<DeltaMessage> delta;  // None

  // granite_tool_parser.py:157-191 - a new tool has appeared in the array.
  if (static_cast<int>(tool_call_arr.size()) > current_tool_id + 1) {
    if (current_tool_id >= 0) {
      const nlohmann::ordered_json& current_tool_call = norm(current_tool_id);
      if (current_tool_call.contains("arguments") &&
          JsonTruthy(current_tool_call.at("arguments"))) {
        const std::string cur_args_json = DumpArgs(current_tool_call.at("arguments"));
        const std::size_t sent =
            streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)].size();
        const std::string argument_diff =
            sent < cur_args_json.size() ? cur_args_json.substr(sent) : std::string();
        DeltaToolCall d;
        d.index = current_tool_id;
        d.function.arguments = argument_diff;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] +=
            argument_diff;
      }
    }
    current_tool_id = static_cast<int>(tool_call_arr.size()) - 1;
    current_tool_name_sent = false;
    streamed_args_for_tool.emplace_back();
    return delta;  // NOTE: prev_tool_call_arr intentionally NOT updated here.
  }

  // granite_tool_parser.py:193-210 - send the tool name once, when available.
  if (!current_tool_name_sent) {
    const nlohmann::ordered_json& current_tool_call = norm(current_tool_id);
    if (current_tool_call.contains("name") &&
        JsonTruthy(current_tool_call.at("name"))) {
      const std::string function_name = current_tool_call.at("name").get<std::string>();
      DeltaToolCall d;
      d.index = current_tool_id;
      d.type = "function";
      d.id = make_tool_call_id();
      d.function.name = function_name;
      DeltaMessage m;
      m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
      delta = std::move(m);
      current_tool_name_sent = true;
    }
  } else {
    // granite_tool_parser.py:212-246 - same tool, stream the argument diff.
    const nlohmann::ordered_json& current_tool_call = norm(current_tool_id);
    if (current_tool_call.contains("arguments") &&
        JsonTruthy(current_tool_call.at("arguments"))) {
      const std::size_t sent =
          streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)].size();
      const std::string cur_args_json = DumpArgs(current_tool_call.at("arguments"));

      std::optional<std::string> argument_diff;
      if (is_complete[static_cast<std::size_t>(current_tool_id)]) {
        argument_diff = sent < cur_args_json.size() ? cur_args_json.substr(sent)
                                                    : std::string();
      } else if (static_cast<std::size_t>(current_tool_id) < prev_tool_call_arr.size() &&
                 prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)].contains(
                     "arguments") &&
                 JsonTruthy(prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)]
                                .at("arguments"))) {
        const std::string prev_args_json = DumpArgs(
            prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)].at("arguments"));
        if (cur_args_json != prev_args_json) {
          const std::string prefix = find_common_prefix(prev_args_json, cur_args_json);
          argument_diff = sent < prefix.size() ? prefix.substr(sent) : std::string();
        }
      }

      if (argument_diff.has_value()) {
        DeltaToolCall d;
        d.index = current_tool_id;
        d.function.arguments = *argument_diff;
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] +=
            *argument_diff;
      }
    }
  }

  // granite_tool_parser.py:248 - self.prev_tool_call_arr = tool_call_arr.
  prev_tool_call_arr = tool_call_arr;
  return delta;
}

// ── Schema-aware type coercion (utils.h) ────────────────────────────────────

namespace {

// Trim leading/trailing ASCII whitespace (Python str.strip()).
std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsJsonSpace(s[b])) ++b;
  while (e > b && IsJsonSpace(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// utils.py:544-568 (_TYPE_ALIASES) + coerce_to_schema_type's key normalization
// (t.strip().lower()). Maps alias type strings onto the seven JSON-Schema types.
std::string NormalizeTypeKey(const std::string& raw) {
  std::string t = Strip(raw);
  for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  static const std::vector<std::pair<std::string, std::string>> kAliases = {
      {"str", "string"},     {"text", "string"},    {"varchar", "string"},
      {"char", "string"},    {"enum", "string"},    {"int", "integer"},
      {"int32", "integer"},  {"int64", "integer"},  {"uint", "integer"},
      {"uint32", "integer"}, {"uint64", "integer"}, {"long", "integer"},
      {"short", "integer"},  {"unsigned", "integer"}, {"float", "number"},
      {"float32", "number"}, {"float64", "number"}, {"double", "number"},
      {"bool", "boolean"},   {"dict", "object"},    {"arr", "array"},
      {"list", "array"},     {"sequence", "array"},
  };
  for (const auto& kv : kAliases) {
    if (t == kv.first) return kv.second;
  }
  return t;
}

// Whether a parsed JSON value is free of non-finite floats (inf/nan), recursing
// into arrays/objects. json.dumps of Infinity/NaN is invalid JSON, so upstream
// (utils.py _is_json_finite) rejects such values back to the raw string.
bool IsJsonFinite(const nlohmann::ordered_json& v) {
  if (v.is_number_float()) {
    const double d = v.get<double>();
    return std::isfinite(d);
  }
  if (v.is_array() || v.is_object()) {
    for (auto it = v.begin(); it != v.end(); ++it) {
      if (!IsJsonFinite(it.value())) return false;
    }
  }
  return true;
}

// int(value) / float(value) accept surrounding ASCII whitespace but no interior
// junk. Strict decimal parse of the stripped string (optional sign).
std::optional<long long> StrictInt(const std::string& value) {
  const std::string t = Strip(value);
  if (t.empty()) return std::nullopt;
  std::size_t idx = 0;
  try {
    const long long v = std::stoll(t, &idx);
    if (idx != t.size()) return std::nullopt;
    return v;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<double> StrictFloat(const std::string& value) {
  const std::string t = Strip(value);
  if (t.empty()) return std::nullopt;
  std::size_t idx = 0;
  try {
    const double v = std::stod(t, &idx);
    if (idx != t.size()) return std::nullopt;
    return v;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace

std::set<std::string> extract_types_from_schema(const nlohmann::json& schema) {
  if (!schema.is_object()) return {"string"};
  std::set<std::string> types;

  if (schema.contains("type")) {
    const nlohmann::json& tv = schema.at("type");
    if (tv.is_string()) {
      types.insert(tv.get<std::string>());
    } else if (tv.is_array()) {
      for (const nlohmann::json& t : tv) {
        if (t.is_string()) types.insert(t.get<std::string>());
      }
    }
  }

  if (schema.contains("enum") && schema.at("enum").is_array() &&
      !schema.at("enum").empty()) {
    for (const nlohmann::json& value : schema.at("enum")) {
      if (value.is_null()) {
        types.insert("null");
      } else if (value.is_boolean()) {
        types.insert("boolean");
      } else if (value.is_number_integer() || value.is_number_unsigned()) {
        types.insert("integer");
      } else if (value.is_number_float()) {
        types.insert("number");
      } else if (value.is_string()) {
        types.insert("string");
      } else if (value.is_array()) {
        types.insert("array");
      } else if (value.is_object()) {
        types.insert("object");
      }
    }
  }

  for (const char* choice_field : {"anyOf", "oneOf", "allOf"}) {
    if (schema.contains(choice_field) && schema.at(choice_field).is_array()) {
      for (const nlohmann::json& choice : schema.at(choice_field)) {
        const std::set<std::string> sub = extract_types_from_schema(choice);
        types.insert(sub.begin(), sub.end());
      }
    }
  }

  if (types.empty()) return {"string"};
  return types;
}

nlohmann::ordered_json coerce_to_schema_type(const std::string& value,
                                             const std::set<std::string>& types) {
  std::set<std::string> normalized;
  for (const std::string& t : types) normalized.insert(NormalizeTypeKey(t));

  // Priority: null > integer > number > boolean > object > array > string.
  static const char* kPriority[] = {"null",    "integer", "number", "boolean",
                                    "object",  "array",   "string"};
  for (const char* candidate : kPriority) {
    if (normalized.find(candidate) == normalized.end()) continue;
    const std::string cand(candidate);
    if (cand == "null") {
      std::string lower = value;
      for (char& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (lower == "null") return nlohmann::ordered_json(nullptr);
      continue;
    }
    if (cand == "string") {
      return nlohmann::ordered_json(value);
    }
    if (cand == "integer") {
      const std::optional<long long> i = StrictInt(value);
      if (i.has_value()) return nlohmann::ordered_json(static_cast<std::int64_t>(*i));
      continue;
    }
    if (cand == "number") {
      const std::optional<double> d = StrictFloat(value);
      if (!d.has_value() || !std::isfinite(*d)) continue;
      // val if val != int(val) else int(val).
      if (*d == static_cast<double>(static_cast<long long>(*d))) {
        return nlohmann::ordered_json(static_cast<std::int64_t>(*d));
      }
      return nlohmann::ordered_json(*d);
    }
    if (cand == "boolean") {
      std::string lower = Strip(value);
      for (char& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (lower == "true" || lower == "1") return nlohmann::ordered_json(true);
      if (lower == "false" || lower == "0") return nlohmann::ordered_json(false);
      continue;
    }
    if (cand == "object" || cand == "array") {
      try {
        nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(value);
        if (IsJsonFinite(parsed)) return parsed;
      } catch (const std::exception&) {
        // fall through to next candidate.
      }
      continue;
    }
  }

  // Fallback: json.loads(value), rejecting non-finite results back to the string.
  try {
    nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(value);
    if (!IsJsonFinite(parsed)) return nlohmann::ordered_json(value);
    return parsed;
  } catch (const std::exception&) {
    return nlohmann::ordered_json(value);
  }
}

nlohmann::json find_tool_properties(
    const std::optional<std::vector<ChatCompletionToolsParam>>& tools,
    const std::string& tool_name) {
  if (!tools.has_value()) return nlohmann::json::object();
  for (const ChatCompletionToolsParam& tool : *tools) {
    if (tool.function.name != tool_name) continue;
    if (!tool.function.parameters.has_value()) return nlohmann::json::object();
    const nlohmann::json& params = *tool.function.parameters;
    if (!params.is_object() || !params.contains("properties")) {
      return nlohmann::json::object();
    }
    const nlohmann::json& props = params.at("properties");
    if (!props.is_object()) return nlohmann::json::object();
    return props;
  }
  return nlohmann::json::object();
}

nlohmann::ordered_json coerce_arg_value(const std::string& raw_value,
                                        const nlohmann::json& properties,
                                        const std::string& key) {
  if (properties.is_object() && properties.contains(key) &&
      properties.at(key).is_object()) {
    return coerce_to_schema_type(raw_value,
                                 extract_types_from_schema(properties.at(key)));
  }
  return nlohmann::ordered_json(raw_value);
}

}  // namespace vllm::entrypoints::openai
