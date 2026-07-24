// REIMPLEMENTED FROM WIRE FORMAT. See gemma4.h for the format spec + deviations.
#include "vllm/entrypoints/openai/tool_parsers/gemma4.h"

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

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

const std::string kStart = Gemma4ToolParser::kToolCallStart;   // <|tool_call>
const std::string kEnd = Gemma4ToolParser::kToolCallEnd;       // <tool_call|>
const std::string kCall = Gemma4ToolParser::kCallPrefix;       // call:
const std::string kDelim = Gemma4ToolParser::kStringDelim;     // <|"|>
const std::size_t kDelimLen = kDelim.size();

bool IsWs(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsWs(s[b])) ++b;
  while (e > b && IsWs(s[e - 1])) --e;
  return s.substr(b, e - b);
}

bool StartsWith(const std::string& s, const std::string& p, std::size_t at = 0) {
  return s.size() >= at + p.size() && s.compare(at, p.size(), p) == 0;
}

bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// gemma4.py:_strip_partial_delim - strip a trailing partial STRING_DELIM prefix
// from *value* so a split "<|\"|>" delimiter never leaks into the streamed diff.
std::string StripPartialDelim(const std::string& value) {
  for (std::size_t k = kDelimLen; k >= 1; --k) {
    if (EndsWith(value, kDelim.substr(0, k))) {
      return value.substr(0, value.size() - k);
    }
    if (k == 1) break;
  }
  return value;
}

// gemma4.py:_parse_gemma4_args (matched at kDelim positions). Forward decl for
// the mutual object/array recursion.
ojson ParseGemmaArgs(const std::string& s, bool partial);
ojson ParseGemmaArray(const std::string& s, bool partial);

// Whether s[i:] begins with the STRING_DELIM.
bool DelimAt(const std::string& s, std::size_t i) {
  return i + kDelimLen <= s.size() && s.compare(i, kDelimLen, kDelim) == 0;
}

ojson ParseGemmaArgs(const std::string& args_str, bool partial) {
  ojson result = ojson::object();
  if (args_str.empty() || Strip(args_str).empty()) return result;

  const std::string& s = args_str;
  const std::size_t n = s.size();
  std::size_t i = 0;
  while (i < n) {
    while (i < n && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' || s[i] == '\t'))
      ++i;
    if (i >= n) break;

    const std::size_t key_start = i;
    while (i < n && s[i] != ':') ++i;
    if (i >= n) break;
    std::string key = Strip(s.substr(key_start, i - key_start));
    if (key.size() >= 2 * kDelimLen && StartsWith(key, kDelim) &&
        EndsWith(key, kDelim)) {
      key = key.substr(kDelimLen, key.size() - 2 * kDelimLen);
    }
    ++i;  // past ':'

    if (i >= n) {
      if (!partial) result[key] = "";
      break;
    }
    while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t')) ++i;
    if (i >= n) {
      if (!partial) result[key] = "";
      break;
    }

    if (DelimAt(s, i)) {
      i += kDelimLen;
      const std::size_t val_start = i;
      const std::size_t end_pos = s.find(kDelim, i);
      if (end_pos == std::string::npos) {
        std::string value = s.substr(val_start);
        if (partial) value = StripPartialDelim(value);
        result[key] = value;
        break;
      }
      result[key] = s.substr(val_start, end_pos - val_start);
      i = end_pos + kDelimLen;
    } else if (s[i] == '{') {
      int depth = 1;
      const std::size_t obj_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (DelimAt(s, i)) {
          i += kDelimLen;
          const std::size_t nd = s.find(kDelim, i);
          i = (nd == std::string::npos) ? n : nd + kDelimLen;
          continue;
        }
        if (s[i] == '{')
          ++depth;
        else if (s[i] == '}')
          --depth;
        ++i;
      }
      if (depth > 0) {
        result[key] = ParseGemmaArgs(s.substr(obj_start, i - obj_start), true);
      } else {
        result[key] = ParseGemmaArgs(s.substr(obj_start, (i - 1) - obj_start), false);
      }
    } else if (s[i] == '[') {
      int depth = 1;
      const std::size_t arr_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (DelimAt(s, i)) {
          i += kDelimLen;
          const std::size_t nd = s.find(kDelim, i);
          i = (nd == std::string::npos) ? n : nd + kDelimLen;
          continue;
        }
        if (s[i] == '[')
          ++depth;
        else if (s[i] == ']')
          --depth;
        ++i;
      }
      if (depth > 0) {
        result[key] = ParseGemmaArray(s.substr(arr_start, i - arr_start), true);
      } else {
        result[key] =
            ParseGemmaArray(s.substr(arr_start, (i - 1) - arr_start), false);
      }
    } else {
      const std::size_t val_start = i;
      while (i < n && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
      if (partial && i >= n) break;  // may be incomplete (type-unstable)
      if (i == val_start) break;     // no progress: malformed, abort
      std::string raw_val = Strip(s.substr(val_start, i - val_start));
      if (partial && !raw_val.empty() && raw_val.back() == '.') break;
      result[key] = raw_val;
    }
  }
  return result;
}

ojson ParseGemmaArray(const std::string& arr_str, bool partial) {
  ojson items = ojson::array();
  const std::string& s = arr_str;
  const std::size_t n = s.size();
  std::size_t i = 0;
  while (i < n) {
    while (i < n && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' || s[i] == '\t'))
      ++i;
    if (i >= n) break;

    if (DelimAt(s, i)) {
      i += kDelimLen;
      const std::size_t end_pos = s.find(kDelim, i);
      if (end_pos == std::string::npos) {
        items.push_back(s.substr(i));
        break;
      }
      items.push_back(s.substr(i, end_pos - i));
      i = end_pos + kDelimLen;
    } else if (s[i] == '{') {
      int depth = 1;
      const std::size_t obj_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (DelimAt(s, i)) {
          i += kDelimLen;
          const std::size_t nd = s.find(kDelim, i);
          i = (nd == std::string::npos) ? n : nd + kDelimLen;
          continue;
        }
        if (s[i] == '{')
          ++depth;
        else if (s[i] == '}')
          --depth;
        ++i;
      }
      if (depth > 0) {
        items.push_back(ParseGemmaArgs(s.substr(obj_start, i - obj_start), true));
      } else {
        items.push_back(
            ParseGemmaArgs(s.substr(obj_start, (i - 1) - obj_start), false));
      }
    } else if (s[i] == '[') {
      int depth = 1;
      const std::size_t sub_start = i + 1;
      ++i;
      while (i < n && depth > 0) {
        if (DelimAt(s, i)) {
          i += kDelimLen;
          const std::size_t nd = s.find(kDelim, i);
          i = (nd == std::string::npos) ? n : nd + kDelimLen;
          continue;
        }
        if (s[i] == '[')
          ++depth;
        else if (s[i] == ']')
          --depth;
        ++i;
      }
      if (depth > 0) {
        items.push_back(ParseGemmaArray(s.substr(sub_start, i - sub_start), true));
      } else {
        items.push_back(
            ParseGemmaArray(s.substr(sub_start, (i - 1) - sub_start), false));
      }
    } else {
      const std::size_t val_start = i;
      while (i < n && s[i] != ',' && s[i] != ']') ++i;
      if (partial && i >= n) break;
      if (i == val_start) break;  // no progress: malformed
      std::string raw_val = Strip(s.substr(val_start, i - val_start));
      if (partial && !raw_val.empty() && raw_val.back() == '.') break;
      items.push_back(raw_val);
    }
  }
  return items;
}

// gemma4.py:_gemma4_arg_converter - strip a trailing '}', parse, dump to JSON.
// Returns the RAW (uncoerced, string-valued) JSON.
std::string ArgConverter(const std::string& raw_args, bool partial) {
  std::string text = Strip(raw_args);
  if (!text.empty() && text.back() == '}') text.pop_back();
  ojson parsed = ParseGemmaArgs(text, partial);
  return parsed.dump();
}

// ── Schema-aware type coercion (vllm/tool_parsers/utils.py) ──────────────────
// Reproduced locally: the gemma4 tool suite REQUIRES coercing bare true/42/3.14
// to the tool schema's bool/int/number (the engine's _fix_arg_types step).

std::string NormalizeType(const std::string& t) {
  std::string k = t;
  for (char& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  // strip surrounding whitespace
  k = Strip(k);
  static const std::vector<std::pair<std::string, std::string>> aliases = {
      {"str", "string"},   {"text", "string"},   {"varchar", "string"},
      {"char", "string"},  {"enum", "string"},   {"int", "integer"},
      {"int32", "integer"},{"int64", "integer"}, {"uint", "integer"},
      {"uint32", "integer"},{"uint64", "integer"},{"long", "integer"},
      {"short", "integer"},{"unsigned", "integer"},{"float", "number"},
      {"float32", "number"},{"float64", "number"},{"double", "number"},
      {"bool", "boolean"}, {"dict", "object"},   {"arr", "array"},
      {"list", "array"},   {"sequence", "array"}};
  for (const auto& [a, b] : aliases)
    if (k == a) return b;
  return k;
}

// utils.py:extract_types_from_schema - collect type strings from a schema
// (type/enum/anyOf/oneOf/allOf); default ["string"].
std::set<std::string> ExtractTypes(const ojson& schema) {
  std::set<std::string> types;
  if (!schema.is_object()) return {"string"};
  if (schema.contains("type")) {
    const ojson& tv = schema["type"];
    if (tv.is_string())
      types.insert(tv.get<std::string>());
    else if (tv.is_array())
      for (const auto& t : tv)
        if (t.is_string()) types.insert(t.get<std::string>());
  }
  if (schema.contains("enum") && schema["enum"].is_array() &&
      !schema["enum"].empty()) {
    for (const auto& v : schema["enum"]) {
      if (v.is_null())
        types.insert("null");
      else if (v.is_boolean())
        types.insert("boolean");
      else if (v.is_number_integer() || v.is_number_unsigned())
        types.insert("integer");
      else if (v.is_number_float())
        types.insert("number");
      else if (v.is_string())
        types.insert("string");
      else if (v.is_array())
        types.insert("array");
      else if (v.is_object())
        types.insert("object");
    }
  }
  for (const char* field : {"anyOf", "oneOf", "allOf"}) {
    if (schema.contains(field) && schema[field].is_array()) {
      for (const auto& choice : schema[field]) {
        std::set<std::string> sub = ExtractTypes(choice);
        types.insert(sub.begin(), sub.end());
      }
    }
  }
  if (types.empty()) return {"string"};
  return types;
}

bool AllFinite(const ojson& v) {
  if (v.is_number_float()) return std::isfinite(v.get<double>());
  if (v.is_array())
    for (const auto& e : v)
      if (!AllFinite(e)) return false;
  if (v.is_object())
    for (const auto& e : v)
      if (!AllFinite(e)) return false;
  return true;
}

// utils.py:coerce_to_schema_type - try types in priority order (null > integer >
// number > boolean > object > array > string); fall back to the raw string
// (or a bare json.loads) when nothing coerces.
ojson CoerceToSchemaType(const std::string& value,
                         const std::set<std::string>& raw_types) {
  std::set<std::string> types;
  for (const auto& t : raw_types) types.insert(NormalizeType(t));

  auto has = [&](const char* t) { return types.count(t) > 0; };

  static const char* priority[] = {"null",   "integer", "number", "boolean",
                                    "object", "array",   "string"};
  for (const char* cand : priority) {
    if (!has(cand)) continue;
    const std::string c = cand;
    if (c == "null") {
      std::string lower = value;
      for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      if (lower == "null") return nullptr;
      continue;
    }
    if (c == "string") return value;
    if (c == "integer") {
      try {
        std::size_t pos = 0;
        long long iv = std::stoll(value, &pos);
        if (pos == value.size()) return iv;
      } catch (...) {
      }
      continue;
    }
    if (c == "number") {
      try {
        std::size_t pos = 0;
        double dv = std::stod(value, &pos);
        if (pos != value.size()) continue;
        if (!std::isfinite(dv)) continue;
        if (dv == static_cast<double>(static_cast<long long>(dv)))
          return static_cast<long long>(dv);
        return dv;
      } catch (...) {
      }
      continue;
    }
    if (c == "boolean") {
      std::string lv = Strip(value);
      for (char& ch : lv) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      if (lv == "true" || lv == "1") return true;
      if (lv == "false" || lv == "0") return false;
      continue;
    }
    if (c == "object" || c == "array") {
      ojson parsed = ojson::parse(value, nullptr, false);
      if (parsed.is_discarded()) continue;
      if (AllFinite(parsed)) return parsed;
      continue;
    }
  }
  ojson parsed = ojson::parse(value, nullptr, false);
  if (parsed.is_discarded()) return value;
  if (!AllFinite(parsed)) return value;
  return parsed;
}

// utils.py:find_tool_properties - the "properties" map of the named tool, or {}.
ojson FindToolProperties(const ChatCompletionRequest& request,
                         const std::string& tool_name) {
  if (!request.tools.has_value()) return ojson::object();
  for (const ChatCompletionToolsParam& tool : *request.tools) {
    if (tool.type != "function") continue;
    if (tool.function.name != tool_name) continue;
    if (!tool.function.parameters.has_value()) return ojson::object();
    const nlohmann::json& params = *tool.function.parameters;
    if (params.is_object() && params.contains("properties") &&
        params["properties"].is_object()) {
      return ojson(params["properties"]);
    }
    return ojson::object();
  }
  return ojson::object();
}

bool CoerceDict(ojson& args, const ojson& properties);  // fwd

// parser_engine.py:_coerce_value - coerce one value per its schema (recurse into
// nested objects/arrays). Returns whether it changed; mutates *value*.
bool CoerceValue(ojson& value, const ojson& schema) {
  if (value.is_string()) {
    ojson coerced = CoerceToSchemaType(value.get<std::string>(), ExtractTypes(schema));
    if (coerced != value) {
      value = coerced;
      return true;
    }
    return false;
  }
  if (value.is_object()) {
    if (schema.is_object() && schema.contains("properties") &&
        schema["properties"].is_object()) {
      return CoerceDict(value, schema["properties"]);
    }
    return false;
  }
  if (value.is_array()) {
    if (schema.is_object() && schema.contains("items") &&
        schema["items"].is_object()) {
      bool changed = false;
      for (auto& item : value) {
        if (CoerceValue(item, schema["items"])) changed = true;
      }
      return changed;
    }
    return false;
  }
  return false;
}

bool CoerceDict(ojson& args, const ojson& properties) {
  bool changed = false;
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (!properties.contains(it.key())) continue;
    const ojson& prop = properties[it.key()];
    if (!prop.is_object()) continue;
    if (CoerceValue(it.value(), prop)) changed = true;
  }
  return changed;
}

// parser_engine.py:_fix_arg_types - coerce the RAW args JSON against the named
// tool's schema; no-op when there is no schema for the tool.
std::string FixArgTypes(const std::string& args_json, const std::string& func_name,
                        const ChatCompletionRequest& request) {
  if (!request.tools.has_value() || request.tools->empty() || func_name.empty())
    return args_json;
  ojson args = ojson::parse(args_json, nullptr, false);
  if (args.is_discarded() || !args.is_object()) return args_json;
  ojson properties = FindToolProperties(request, func_name);
  if (!properties.is_object() || properties.empty()) return args_json;
  if (CoerceDict(args, properties)) return args.dump();
  return args_json;
}

// ── Wire-format block scan ────────────────────────────────────────────────────
// One <|tool_call>call:NAME{ARGS}<tool_call|> block recovered from the text.
struct GemmaCall {
  std::string name;
  std::string args_text;   // raw args after '{' (may keep a trailing '}')
  bool has_name = false;   // '{' seen -> NAME is delimited/known
  bool complete = false;   // closing <tool_call|> seen
};

// Split *text* into (content-outside-blocks, blocks). content excludes every
// block; a trailing partial <|tool_call> prefix is buffered off (never emitted).
void ScanBlocks(const std::string& text, std::string& content_out,
                std::vector<GemmaCall>& calls_out) {
  content_out.clear();
  calls_out.clear();
  std::size_t pos = 0;
  const std::size_t n = text.size();
  while (pos < n) {
    const std::size_t s = text.find(kStart, pos);
    if (s == std::string::npos) {
      content_out += text.substr(pos);
      break;
    }
    content_out += text.substr(pos, s - pos);
    const std::size_t inner_start = s + kStart.size();
    const std::size_t e = text.find(kEnd, inner_start);
    GemmaCall call;
    std::string inner;
    if (e == std::string::npos) {
      inner = text.substr(inner_start);
      call.complete = false;
      pos = n;  // in-progress block runs to the end
    } else {
      inner = text.substr(inner_start, e - inner_start);
      call.complete = true;
      pos = e + kEnd.size();
    }
    if (StartsWith(inner, kCall)) {
      const std::string rest = inner.substr(kCall.size());
      const std::size_t brace = rest.find('{');
      if (brace != std::string::npos) {
        call.name = rest.substr(0, brace);
        call.args_text = rest.substr(brace + 1);
        call.has_name = true;
      } else {
        call.name = rest;  // partial: no '{' yet
        call.has_name = false;
      }
    }
    calls_out.push_back(std::move(call));
  }

  // Buffer a trailing partial <|tool_call> start-tag prefix out of content so a
  // split start tag never leaks (mirrors the engine never emitting a partial
  // special token as text).
  for (std::size_t k = kStart.size() - 1; k >= 1; --k) {
    if (EndsWith(content_out, kStart.substr(0, k))) {
      content_out.erase(content_out.size() - k);
      break;
    }
    if (k == 1) break;
  }
}

}  // namespace

nlohmann::ordered_json Gemma4ToolParser::ParseArgs(const std::string& args_str,
                                                   bool partial) {
  return ParseGemmaArgs(args_str, partial);
}

nlohmann::ordered_json Gemma4ToolParser::ParseArray(const std::string& arr_str,
                                                    bool partial) {
  return ParseGemmaArray(arr_str, partial);
}

ExtractedToolCallInformation Gemma4ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  // No start marker -> plain content (unmodified).
  if (model_output.find(kStart) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
  try {
    std::string content;
    std::vector<GemmaCall> calls;
    ScanBlocks(model_output, content, calls);

    std::vector<ToolCall> tool_calls;
    for (const GemmaCall& c : calls) {
      if (!c.has_name || c.name.empty()) continue;
      const std::string raw = ArgConverter(c.args_text, /*partial=*/false);
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = c.name;
      tc.function.arguments = FixArgTypes(raw, c.name, request);
      tool_calls.push_back(std::move(tc));
    }

    if (tool_calls.empty()) {
      return ExtractedToolCallInformation{false, {}, model_output};
    }

    // Content = text before the first <|tool_call>, stripped; None when empty.
    std::optional<std::string> content_opt;
    const std::size_t first = model_output.find(kStart);
    if (first != std::string::npos && first > 0) {
      const std::string c = Strip(model_output.substr(0, first));
      if (!c.empty()) content_opt = c;
    }

    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    info.content = content_opt;
    return info;
  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> Gemma4ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/, const ChatCompletionRequest& request) {
  try {
    std::string content;
    std::vector<GemmaCall> calls;
    ScanBlocks(current_text, content, calls);

    DeltaMessage msg;
    std::vector<DeltaToolCall> deltas;

    for (std::size_t i = 0; i < calls.size(); ++i) {
      const GemmaCall& c = calls[i];
      if (static_cast<int>(i) > current_tool_id_) {
        current_tool_id_ = static_cast<int>(i);
      }
      if (name_sent_.size() <= i) name_sent_.resize(i + 1, false);
      if (args_sent_.size() <= i) args_sent_.resize(i + 1, false);

      // Emit the NAME once it is delimited by '{'.
      if (c.has_name && !c.name.empty() && !name_sent_[i]) {
        name_sent_[i] = true;
        DeltaToolCall tc;
        tc.index = static_cast<int>(i);
        tc.id = make_tool_call_id();
        tc.type = "function";
        tc.function.name = c.name;
        deltas.push_back(std::move(tc));
      }

      // Emit the coerced ARGUMENTS as one chunk at the call's close.
      if (c.complete && c.has_name && !c.name.empty() && !args_sent_[i]) {
        args_sent_[i] = true;
        const std::string raw = ArgConverter(c.args_text, /*partial=*/false);
        const std::string fixed = FixArgTypes(raw, c.name, request);
        DeltaToolCall tc;
        tc.index = static_cast<int>(i);
        tc.function.arguments = fixed;
        deltas.push_back(std::move(tc));
      }
    }

    // Emit any newly-available plain content (before/between/after blocks).
    if (content.size() > streamed_content_len_) {
      const std::string new_content = content.substr(streamed_content_len_);
      streamed_content_len_ = content.size();
      if (!new_content.empty()) msg.content = new_content;
    }

    if (!deltas.empty()) msg.tool_calls = std::move(deltas);
    if (msg.content.has_value() || msg.tool_calls.has_value()) return msg;
    return std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
