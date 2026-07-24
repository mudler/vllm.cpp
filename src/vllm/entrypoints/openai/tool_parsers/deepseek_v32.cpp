// Ported from: vllm/tool_parsers/deepseekv32_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/deepseek_v32.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using nlohmann::json;
using nlohmann::ordered_json;

// ─── small string helpers ────────────────────────────────────────────────────

bool IsAsciiSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
// Python str.lstrip() over ASCII whitespace: count of leading space chars.
std::size_t LStripCount(const std::string& s) {
  std::size_t i = 0;
  while (i < s.size() && IsAsciiSpace(s[i])) ++i;
  return i;
}
std::string Strip(const std::string& s) {
  std::size_t a = 0, b = s.size();
  while (a < b && IsAsciiSpace(s[a])) ++a;
  while (b > a && IsAsciiSpace(s[b - 1])) --b;
  return s.substr(a, b - a);
}
std::string Lower(const std::string& s) {
  std::string r = s;
  for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return r;
}
bool StartsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// tool_parsers/utils.py:42 (partial_tag_overlap): length of the longest prefix
// of `tag` that matches a suffix of `text`. BYTE-based (like deepseek_v3.cpp) —
// this is what makes the text-only rework UTF-8 safe: a marker split across a
// delta boundary leaves a partial-byte suffix that is held back, never emitted.
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k >= 1; --k) {
    if (text.compare(text.size() - k, k, tag, 0, k) == 0) return k;
  }
  return 0;
}

// Largest prefix length <= n of `s` that does not end in the middle of a
// multi-byte UTF-8 sequence. DEVIATION from upstream: Python streams already-
// decoded str (whole code points), so json.dumps never sees a partial code
// point. Our byte-level seam CAN split a multi-byte value across a delta
// boundary, so the incremental string-escape path holds back a trailing
// incomplete sequence (the value reassembles exactly on the next delta).
std::size_t Utf8SafeLen(const std::string& s, std::size_t n) {
  if (n > s.size()) n = s.size();
  std::size_t i = n;
  while (i > 0) {
    const unsigned char c = static_cast<unsigned char>(s[i - 1]);
    if ((c & 0xC0) == 0x80) {  // UTF-8 continuation byte — keep walking back
      --i;
      continue;
    }
    const std::size_t need = c < 0x80              ? 1
                             : (c & 0xE0) == 0xC0  ? 2
                             : (c & 0xF0) == 0xE0  ? 3
                             : (c & 0xF8) == 0xF0  ? 4
                                                   : 1;  // invalid lead byte
    if (n - (i - 1) >= need) return n;  // the trailing code point is complete
    return i - 1;                       // incomplete — cut before its lead byte
  }
  return n;
}

bool RegexMatchStart(const std::string& s, std::smatch& m, const std::regex& re) {
  // Anchor the match at the start of the buffer (Python re.Pattern.match).
  return std::regex_search(s, m, re, std::regex_constants::match_continuous);
}

// utils.py:176 (_json_escape_string_content): json.dumps(text)[1:-1]. nlohmann
// dump() defaults to ensure_ascii=false, so UTF-8 bytes survive unescaped —
// matching json.dumps(..., ensure_ascii=False).
std::string JsonEscapeStringContent(const std::string& text) {
  const std::string dumped = json(text).dump();
  return dumped.substr(1, dumped.size() - 2);
}

// A streaming tool-call id: upstream _generate_tool_call_id emits
// f"call_{uuid4().hex[:24]}" (uniqueness only).
std::string GenStreamId() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t a = dist(rng), b = dist(rng);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << a << std::setw(16) << b;
  return "call_" + oss.str().substr(0, 24);
}

// ─── the four DSML sub-marker regexes (identical for v3.2 and v4) ─────────────
// The fullwidth ｜ (U+FF5C) bytes are literal in the raw-string atoms; ECMAScript
// std::regex matches them byte-for-byte (`[\s\S]` emulates Python re.DOTALL `.`).

// NOTE: a custom raw-string delimiter (re"...") is required — these patterns
// contain the byte sequence )" (e.g. ([^"]+)" ), which would otherwise close a
// plain R"(...)" literal prematurely.
const std::regex& InvokeCompleteRegex() {
  static const std::regex re(
      R"re(<｜DSML｜invoke\s+name="([^"]+)"\s*>([\s\S]*?)</｜DSML｜invoke>)re");
  return re;
}
const std::regex& ParameterCompleteRegex() {
  static const std::regex re(
      R"re(<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>([\s\S]*?)</｜DSML｜parameter>)re");
  return re;
}
const std::regex& InvokeStartRegex() {
  static const std::regex re(R"re(<｜DSML｜invoke\s+name="([^"]+)"\s*>)re");
  return re;
}
const std::regex& ParameterStartRegex() {
  static const std::regex re(
      R"re(<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>)re");
  return re;
}

// ─── schema-driven coercion (utils.py) ────────────────────────────────────────

std::optional<ordered_json> TryParseJson(const std::string& s) {
  try {
    return ordered_json::parse(s);
  } catch (...) {
    return std::nullopt;
  }
}

// utils.py:149 (_is_json_finite): whether json.dumps(obj, allow_nan=False) would
// succeed — i.e. no non-finite float anywhere in the value.
bool IsJsonFinite(const ordered_json& j) {
  if (j.is_number_float()) return std::isfinite(j.get<double>());
  if (j.is_array()) {
    for (const auto& e : j)
      if (!IsJsonFinite(e)) return false;
    return true;
  }
  if (j.is_object()) {
    for (auto it = j.begin(); it != j.end(); ++it)
      if (!IsJsonFinite(it.value())) return false;
    return true;
  }
  return true;
}

// Python int(str): optional surrounding whitespace, optional +/-, ASCII digits.
std::optional<long long> PyInt(const std::string& s) {
  const std::string t = Strip(s);
  if (t.empty()) return std::nullopt;
  std::size_t i = 0;
  if (t[i] == '+' || t[i] == '-') ++i;
  if (i >= t.size()) return std::nullopt;
  for (std::size_t j = i; j < t.size(); ++j)
    if (!std::isdigit(static_cast<unsigned char>(t[j]))) return std::nullopt;
  try {
    return std::stoll(t);
  } catch (...) {
    return std::nullopt;  // out-of-range: fall through (upstream int() is bignum)
  }
}

// Python float(str): the WHOLE trimmed token must parse (no trailing garbage).
std::optional<double> PyFloat(const std::string& s) {
  const std::string t = Strip(s);
  if (t.empty()) return std::nullopt;
  const char* c = t.c_str();
  char* end = nullptr;
  const double v = std::strtod(c, &end);
  if (end != c + t.size()) return std::nullopt;
  return v;
}

// utils.py:544 (_TYPE_ALIASES).
const std::map<std::string, std::string>& TypeAliases() {
  static const std::map<std::string, std::string> m = {
      {"str", "string"},   {"text", "string"},   {"varchar", "string"},
      {"char", "string"},  {"enum", "string"},   {"int", "integer"},
      {"int32", "integer"},{"int64", "integer"}, {"uint", "integer"},
      {"uint32", "integer"},{"uint64", "integer"},{"long", "integer"},
      {"short", "integer"},{"unsigned", "integer"},{"float", "number"},
      {"float32", "number"},{"float64", "number"},{"double", "number"},
      {"bool", "boolean"}, {"dict", "object"},   {"arr", "array"},
      {"list", "array"},   {"sequence", "array"}};
  return m;
}

// utils.py:498 (extract_types_from_schema). Order of the result is irrelevant —
// coerce_to_schema_type re-normalizes into a set and walks a fixed priority.
std::vector<std::string> ExtractTypesFromSchema(const json& schema) {
  if (!schema.is_object()) return {"string"};
  std::set<std::string> types;
  if (schema.contains("type")) {
    const auto& tv = schema["type"];
    if (tv.is_string()) {
      types.insert(tv.get<std::string>());
    } else if (tv.is_array()) {
      for (const auto& t : tv)
        if (t.is_string()) types.insert(t.get<std::string>());
    }
  }
  if (schema.contains("enum") && schema["enum"].is_array() &&
      !schema["enum"].empty()) {
    for (const auto& v : schema["enum"]) {
      if (v.is_null()) types.insert("null");
      else if (v.is_boolean()) types.insert("boolean");
      else if (v.is_number_integer() || v.is_number_unsigned()) types.insert("integer");
      else if (v.is_number_float()) types.insert("number");
      else if (v.is_string()) types.insert("string");
      else if (v.is_array()) types.insert("array");
      else if (v.is_object()) types.insert("object");
    }
  }
  for (const char* cf : {"anyOf", "oneOf", "allOf"}) {
    if (schema.contains(cf) && schema[cf].is_array()) {
      for (const auto& choice : schema[cf]) {
        auto sub = ExtractTypesFromSchema(choice);
        types.insert(sub.begin(), sub.end());
      }
    }
  }
  if (types.empty()) return {"string"};
  return {types.begin(), types.end()};
}

// utils.py:571 (coerce_to_schema_type). Returns an ordered_json so parsed
// object/array values keep insertion order for the re-serialized arguments.
ordered_json CoerceToSchemaType(const std::string& value,
                                const std::vector<std::string>& schema_types) {
  std::set<std::string> normalized;
  for (const auto& t : schema_types) {
    const std::string key = Lower(Strip(t));
    auto it = TypeAliases().find(key);
    normalized.insert(it != TypeAliases().end() ? it->second : key);
  }
  static const char* kPriority[] = {"null",    "integer", "number", "boolean",
                                    "object",  "array",   "string"};
  for (const char* cand : kPriority) {
    if (!normalized.count(cand)) continue;
    const std::string c = cand;
    if (c == "null") {
      if (Lower(value) == "null") return ordered_json(nullptr);
      continue;
    }
    if (c == "string") return ordered_json(value);
    if (c == "integer") {
      auto i = PyInt(value);
      if (i) return ordered_json(*i);
      continue;
    }
    if (c == "number") {
      auto f = PyFloat(value);
      if (!f) continue;
      if (!std::isfinite(*f)) continue;
      const double v = *f;
      if (v == std::trunc(v) && v >= -9.007199254740992e15 &&
          v <= 9.007199254740992e15) {
        return ordered_json(static_cast<long long>(v));
      }
      return ordered_json(v);
    }
    if (c == "boolean") {
      const std::string lv = Lower(Strip(value));
      if (lv == "true" || lv == "1") return ordered_json(true);
      if (lv == "false" || lv == "0") return ordered_json(false);
      continue;
    }
    if (c == "object" || c == "array") {
      auto parsed = TryParseJson(value);
      if (!parsed) continue;
      if (IsJsonFinite(*parsed)) return *parsed;
      continue;
    }
  }
  // Fallback: best-effort json.loads, else the raw string.
  auto parsed = TryParseJson(value);
  if (!parsed) return ordered_json(value);
  if (!IsJsonFinite(*parsed)) return ordered_json(value);
  return *parsed;
}

// utils.py:184 (find_tool_properties): the tool's `properties` schema, or {}.
json FindToolProperties(const std::vector<ChatCompletionToolsParam>* tools,
                        const std::string& name) {
  if (!tools) return json::object();
  for (const auto& t : *tools) {
    if (t.function.name != name) continue;
    if (t.function.parameters.has_value() && t.function.parameters->is_object()) {
      const auto& p = *t.function.parameters;
      if (p.contains("properties") && p["properties"].is_object())
        return p["properties"];
    }
    return json::object();
  }
  return json::object();
}

// deepseekv32_tool_parser.py:131 (_repair_param_dict): unwrap a lone
// "arguments"/"input" wrapper that is not itself a schema field.
ordered_json RepairParamDict(const ordered_json& param_dict,
                             const json& param_config) {
  std::set<std::string> allowed;
  if (param_config.is_object())
    for (auto it = param_config.begin(); it != param_config.end(); ++it)
      allowed.insert(it.key());
  for (const char* wrapper : {"arguments", "input"}) {
    if (!(param_dict.is_object() && param_dict.size() == 1 &&
          param_dict.contains(wrapper)))
      continue;
    if (allowed.count(wrapper)) continue;
    ordered_json inner = param_dict[wrapper];
    if (inner.is_string()) {
      auto parsed = TryParseJson(inner.get<std::string>());
      if (!parsed) return param_dict;  // json.loads failed -> keep as-is
      inner = *parsed;
    }
    if (inner.is_object()) {
      bool subset = true;
      for (auto it = inner.begin(); it != inner.end(); ++it)
        if (!allowed.count(it.key())) {
          subset = false;
          break;
        }
      if (subset) return inner;
    }
  }
  return param_dict;
}

// _can_stream_raw_param (deepseekv32_tool_parser.py:338): only object/array
// values may stream raw; scalars/unions buffer so streaming and non-streaming
// share the same coercion fallback.
bool CanStreamRawParam(const std::vector<std::string>& types) {
  for (const auto& t : types)
    if (t != "object" && t != "array") return false;
  return true;
}

}  // namespace

// ─── wrapper-token accessors (deepseek_v4 overrides these two) ────────────────

DeepSeekV32ToolParser::~DeepSeekV32ToolParser() = default;

const std::string& DeepSeekV32ToolParser::tool_call_start() const {
  static const std::string tok = kToolCallStartToken;
  return tok;
}
const std::string& DeepSeekV32ToolParser::tool_call_end() const {
  static const std::string tok = kToolCallEndToken;
  return tok;
}

// ─── schema helpers bound to the parser's cached tool list ────────────────────

json DeepSeekV32ToolParser::get_param_config(
    const std::optional<std::string>& fn) const {
  if (!fn.has_value() || fn->empty() || !tools_) return json::object();
  return FindToolProperties(tools_, *fn);
}

ordered_json DeepSeekV32ToolParser::convert_params_with_schema(
    const std::string& fn, const std::vector<RawParam>& raw) const {
  const json param_config = FindToolProperties(tools_, fn);
  ordered_json converted = ordered_json::object();
  for (const auto& rp : raw) {
    const std::string& name = rp.first;
    const std::string& value = rp.second.first;
    const std::string& string_attr = rp.second.second;
    if (string_attr == "true") {
      converted[name] = value;  // keep literal string despite schema
      continue;
    }
    const json field =
        param_config.contains(name) ? param_config[name] : json::object();
    converted[name] = CoerceToSchemaType(value, ExtractTypesFromSchema(field));
  }
  return RepairParamDict(converted, param_config);
}

// ─── non-streaming ────────────────────────────────────────────────────────────

ExtractedToolCallInformation DeepSeekV32ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  tools_ = request.tools.has_value() ? &*request.tools : nullptr;

  const std::string& start = tool_call_start();
  // deepseekv32_tool_parser.py:187 — quick check.
  if (model_output.find(start) == std::string::npos)
    return ExtractedToolCallInformation{false, {}, model_output};

  try {
    const std::string& end = tool_call_end();
    std::vector<ToolCall> tool_calls;

    // findall over start(.*?)end (non-greedy), then invoke findall per block.
    std::size_t pos = 0;
    while (true) {
      const std::size_t s = model_output.find(start, pos);
      if (s == std::string::npos) break;
      const std::size_t cs = s + start.size();
      const std::size_t e = model_output.find(end, cs);
      if (e == std::string::npos) break;  // no complete block
      const std::string block = model_output.substr(cs, e - cs);
      pos = e + end.size();

      for (auto it = std::sregex_iterator(block.begin(), block.end(),
                                          InvokeCompleteRegex());
           it != std::sregex_iterator(); ++it) {
        const std::smatch& im = *it;
        const std::string invoke_name = im[1].str();
        const std::string invoke_content = im[2].str();

        // parse params (deepseekv32_tool_parser.py:123 _parse_invoke_params).
        std::vector<RawParam> raw;
        for (auto pit = std::sregex_iterator(invoke_content.begin(),
                                             invoke_content.end(),
                                             ParameterCompleteRegex());
             pit != std::sregex_iterator(); ++pit) {
          const std::smatch& pm = *pit;
          raw.emplace_back(pm[1].str(),
                           std::make_pair(pm[3].str(), pm[2].str()));
        }
        const ordered_json params = convert_params_with_schema(invoke_name, raw);

        ToolCall tc;
        tc.id = make_tool_call_id();  // upstream ToolCall.id default_factory.
        tc.type = "function";
        tc.function.name = invoke_name;
        tc.function.arguments = params.dump();
        tool_calls.push_back(std::move(tc));
      }
    }

    if (tool_calls.empty())
      return ExtractedToolCallInformation{false, {}, model_output};

    const std::size_t first = model_output.find(start);
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    if (first > 0) info.content = model_output.substr(0, first);
    return info;

  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

// ─── streaming emission helpers ───────────────────────────────────────────────

void DeepSeekV32ToolParser::add_tool_call_delta(
    std::map<int, DeltaToolCall>& deltas, int index,
    const std::optional<std::string>& call_id,
    const std::optional<std::string>& call_type,
    const std::optional<std::string>& name,
    const std::optional<std::string>& arguments) {
  if (arguments.has_value() && !arguments->empty())
    streamed_args_for_tool[static_cast<std::size_t>(index)] += *arguments;

  auto it = deltas.find(index);
  if (it == deltas.end()) {
    DeltaToolCall d;
    d.index = index;
    d.id = call_id;
    d.type = call_type;
    if (name.has_value()) d.function.name = name;
    if (arguments.has_value()) d.function.arguments = arguments;
    deltas.emplace(index, std::move(d));
    return;
  }
  DeltaToolCall& d = it->second;
  if (call_id.has_value()) d.id = call_id;
  if (call_type.has_value()) d.type = call_type;
  if (name.has_value()) d.function.name = name;
  if (arguments.has_value())
    d.function.arguments = d.function.arguments.value_or("") + *arguments;
}

void DeepSeekV32ToolParser::begin_streaming_tool_call(
    const std::string& name, std::map<int, DeltaToolCall>& deltas) {
  const int index = current_tool_index_;
  current_tool_index_++;
  active_tool_index_ = index;
  active_tool_name_ = name;
  prev_tool_call_arr.push_back(json{{"name", name}, {"arguments", json::object()}});
  streamed_args_for_tool.emplace_back();
  args_started_.push_back(false);
  add_tool_call_delta(deltas, index, GenStreamId(), std::string("function"), name,
                      std::string(""));
}

void DeepSeekV32ToolParser::append_param_prefix(
    std::map<int, DeltaToolCall>& deltas, int index, const std::string& key,
    bool as_string) {
  const std::string prefix =
      args_started_[static_cast<std::size_t>(index)] ? "," : "{";
  args_started_[static_cast<std::size_t>(index)] = true;
  std::string arguments = prefix + json(key).dump() + ":";
  if (as_string) arguments += "\"";
  add_tool_call_delta(deltas, index, std::nullopt, std::nullopt, std::nullopt,
                      arguments);
}

void DeepSeekV32ToolParser::append_json_param_value(
    std::map<int, DeltaToolCall>& deltas, int index, const std::string& key,
    const ordered_json& value) {
  append_param_prefix(deltas, index, key, /*as_string=*/false);
  add_tool_call_delta(deltas, index, std::nullopt, std::nullopt, std::nullopt,
                      value.dump());
}

std::vector<std::string> DeepSeekV32ToolParser::param_types_for_name(
    const std::string& name) const {
  const json pc = get_param_config(active_tool_name_);
  if (pc.contains(name) && pc[name].is_object())
    return ExtractTypesFromSchema(pc[name]);
  return {"string"};
}

bool DeepSeekV32ToolParser::should_buffer_wrapper_param(
    const std::string& name) const {
  if (!active_tool_index_.has_value() ||
      args_started_[static_cast<std::size_t>(*active_tool_index_)])
    return false;
  const json pc = get_param_config(active_tool_name_);
  const bool pc_nonempty = pc.is_object() && !pc.empty();
  return pc_nonempty && (name == "arguments" || name == "input") &&
         !pc.contains(name);
}

void DeepSeekV32ToolParser::finish_buffered_param(
    std::map<int, DeltaToolCall>& deltas, int index) {
  std::string raw_value;
  for (const auto& part : active_param_parts_) raw_value += part;
  std::vector<RawParam> one{
      {*active_param_name_, {raw_value, *active_param_string_attr_}}};
  const ordered_json converted =
      convert_params_with_schema(active_tool_name_.value_or(""), one);
  for (auto it = converted.begin(); it != converted.end(); ++it)
    append_json_param_value(deltas, index, it.key(), it.value());
}

void DeepSeekV32ToolParser::close_streaming_tool_call(
    std::map<int, DeltaToolCall>& deltas) {
  if (!active_tool_index_.has_value()) return;
  const int index = *active_tool_index_;
  const std::string suffix =
      args_started_[static_cast<std::size_t>(index)] ? "}" : "{}";
  add_tool_call_delta(deltas, index, std::nullopt, std::nullopt, std::nullopt,
                      suffix);
  try {
    const json parsed =
        json::parse(streamed_args_for_tool[static_cast<std::size_t>(index)]);
    prev_tool_call_arr[static_cast<std::size_t>(index)] =
        json{{"name", active_tool_name_ ? json(*active_tool_name_) : json(nullptr)},
             {"arguments", parsed}};
  } catch (...) {
    // Failed to finalize DSML streaming tool call; leave prev entry as-is.
  }
  active_tool_index_.reset();
  active_tool_name_.reset();
  active_param_name_.reset();
  active_param_string_attr_.reset();
  active_param_mode_.reset();
  active_param_parts_.clear();
}

// deepseekv32_tool_parser.py:395 (_process_streaming_buffer).
void DeepSeekV32ToolParser::process_streaming_buffer(
    std::vector<std::string>& content_parts,
    std::map<int, DeltaToolCall>& deltas) {
  static const std::string parameter_end_token = "</｜DSML｜parameter>";
  static const std::string invoke_end_token = "</｜DSML｜invoke>";
  const std::string start_tok = tool_call_start();
  const std::string end_tok = tool_call_end();

  while (true) {
    if (!in_tool_calls_) {
      const std::size_t start_idx = buffer_.find(start_tok);
      if (start_idx == std::string::npos) {
        const std::size_t overlap = PartialTagOverlap(buffer_, start_tok);
        const std::size_t sendable_idx = buffer_.size() - overlap;
        if (sendable_idx > 0) {
          content_parts.push_back(buffer_.substr(0, sendable_idx));
          buffer_ = buffer_.substr(sendable_idx);
        }
        return;
      }
      if (start_idx > 0) {
        content_parts.push_back(buffer_.substr(0, start_idx));
        buffer_ = buffer_.substr(start_idx);
        continue;
      }
      buffer_ = buffer_.substr(start_tok.size());
      in_tool_calls_ = true;
      continue;
    }

    if (!active_tool_index_.has_value()) {
      const std::size_t stripped = LStripCount(buffer_);
      if (stripped) {
        buffer_ = buffer_.substr(stripped);
        continue;
      }
      if (StartsWith(buffer_, end_tok)) {
        buffer_ = buffer_.substr(end_tok.size());
        in_tool_calls_ = false;
        continue;
      }
      std::smatch m;
      if (!RegexMatchStart(buffer_, m, InvokeStartRegex())) return;
      const std::size_t consumed = static_cast<std::size_t>(m[0].length());
      const std::string name = m[1].str();
      buffer_ = buffer_.substr(consumed);
      begin_streaming_tool_call(name, deltas);
      continue;
    }

    const int index = *active_tool_index_;

    if (active_param_mode_.has_value()) {
      const std::size_t end_pos = buffer_.find(parameter_end_token);
      if (end_pos != std::string::npos) {
        const std::string raw_content = buffer_.substr(0, end_pos);
        buffer_ = buffer_.substr(end_pos + parameter_end_token.size());
        const std::string mode = *active_param_mode_;
        if (mode == "wrapper" || mode == "buffered") {
          active_param_parts_.push_back(raw_content);
          finish_buffered_param(deltas, index);
        } else if (mode == "string") {
          const std::string arguments =
              JsonEscapeStringContent(raw_content) + "\"";
          add_tool_call_delta(deltas, index, std::nullopt, std::nullopt,
                              std::nullopt, arguments);
        } else {  // "raw"
          add_tool_call_delta(deltas, index, std::nullopt, std::nullopt,
                              std::nullopt, raw_content);
        }
        active_param_name_.reset();
        active_param_string_attr_.reset();
        active_param_mode_.reset();
        active_param_parts_.clear();
        continue;
      }

      const std::size_t overlap =
          PartialTagOverlap(buffer_, parameter_end_token);
      std::size_t safe_len = buffer_.size() - overlap;
      // String mode escapes the fragment incrementally, so it must not end
      // mid-code-point (see Utf8SafeLen). raw/buffered append bytes verbatim and
      // reassemble on concatenation, so they do not need the hold-back.
      if (*active_param_mode_ == "string")
        safe_len = Utf8SafeLen(buffer_, safe_len);
      if (safe_len > 0) {
        const std::string raw_content = buffer_.substr(0, safe_len);
        buffer_ = buffer_.substr(safe_len);
        const std::string mode = *active_param_mode_;
        if (mode == "wrapper" || mode == "buffered") {
          active_param_parts_.push_back(raw_content);
        } else if (mode == "string") {
          add_tool_call_delta(deltas, index, std::nullopt, std::nullopt,
                              std::nullopt, JsonEscapeStringContent(raw_content));
        } else {  // "raw"
          add_tool_call_delta(deltas, index, std::nullopt, std::nullopt,
                              std::nullopt, raw_content);
        }
      }
      return;
    }

    const std::size_t stripped = LStripCount(buffer_);
    if (stripped) {
      buffer_ = buffer_.substr(stripped);
      continue;
    }

    if (StartsWith(buffer_, invoke_end_token)) {
      buffer_ = buffer_.substr(invoke_end_token.size());
      close_streaming_tool_call(deltas);
      continue;
    }

    std::smatch m;
    if (!RegexMatchStart(buffer_, m, ParameterStartRegex())) return;
    const std::size_t consumed = static_cast<std::size_t>(m[0].length());
    const std::string name = m[1].str();
    const std::string string_attr = m[2].str();
    buffer_ = buffer_.substr(consumed);
    active_param_name_ = name;
    active_param_string_attr_ = string_attr;

    if (should_buffer_wrapper_param(name)) {
      active_param_mode_ = "wrapper";
      continue;
    }
    if (string_attr == "true") {
      append_param_prefix(deltas, index, name, /*as_string=*/true);
      active_param_mode_ = "string";
      continue;
    }
    const std::vector<std::string> param_types = param_types_for_name(name);
    if (!CanStreamRawParam(param_types)) {
      active_param_mode_ = "buffered";
      continue;
    }
    append_param_prefix(deltas, index, name, /*as_string=*/false);
    active_param_mode_ = "raw";
  }
}

void DeepSeekV32ToolParser::reset_streaming_state() {
  current_tool_index_ = 0;
  buffer_.clear();
  in_tool_calls_ = false;
  active_tool_index_.reset();
  active_tool_name_.reset();
  active_param_name_.reset();
  active_param_string_attr_.reset();
  active_param_mode_.reset();
  active_param_parts_.clear();
  prev_tool_call_arr.clear();
  streamed_args_for_tool.clear();
  args_started_.clear();
}

std::optional<DeltaMessage> DeepSeekV32ToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  tools_ = request.tools.has_value() ? &*request.tools : nullptr;

  // First chunk of a new stream — reset state from any prior request.
  if (previous_text.empty()) reset_streaming_state();

  buffer_ += delta_text;
  std::vector<std::string> content_parts;
  std::map<int, DeltaToolCall> deltas;
  process_streaming_buffer(content_parts, deltas);

  if (!content_parts.empty() || !deltas.empty()) {
    DeltaMessage msg;
    std::string content;
    for (const auto& c : content_parts) content += c;
    if (!content.empty()) msg.content = content;  // "".join(...) or None
    if (!deltas.empty()) {
      std::vector<DeltaToolCall> v;
      v.reserve(deltas.size());
      for (auto& kv : deltas) v.push_back(std::move(kv.second));
      msg.tool_calls = std::move(v);
    }
    return msg;
  }

  // DEVIATION 4 (deepseek_v32.h): text-only EOS finalizer. Upstream gates on
  // `not delta_text and delta_token_ids and self.prev_tool_call_arr`; the seam
  // has no token ids, so we gate on an empty delta with a started tool array.
  if (delta_text.empty() && !prev_tool_call_arr.empty()) {
    DeltaMessage msg;
    msg.content = std::string("");
    return msg;
  }
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
