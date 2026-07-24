// Ported from: vllm/tool_parsers/hy_v3_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/hy_v3.h"

#include <array>
#include <cctype>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/tool_parsers/pythonic_core.h"

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

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

std::string ToLower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// hy_v3_tool_parser.py:35-49 (_TYPE_ALIASES).
std::string NormalizeType(const std::string& raw_type) {
  static const std::array<std::pair<const char*, const char*>, 13> kAliases = {{
      {"str", "string"},    {"text", "string"},  {"varchar", "string"},
      {"char", "string"},   {"enum", "string"},  {"bool", "boolean"},
      {"binary", "boolean"},{"int", "integer"},  {"float", "number"},
      {"double", "number"}, {"list", "array"},   {"dict", "object"},
      {"map", "object"},
  }};
  for (const auto& kv : kAliases) {
    if (raw_type == kv.first) return kv.second;
  }
  const std::string lower = ToLower(raw_type);
  static const std::array<const char*, 5> kIntPrefixes = {
      "int", "uint", "long", "short", "unsigned"};
  static const std::array<const char*, 2> kNumPrefixes = {"num", "float"};
  for (const char* p : kIntPrefixes) {
    if (lower.rfind(p, 0) == 0) return "integer";
  }
  for (const char* p : kNumPrefixes) {
    if (lower.rfind(p, 0) == 0) return "number";
  }
  return raw_type;
}

// hy_v3_tool_parser.py:84-98 (_get_arg_schema).
ojson GetArgSchema(const std::string& function_name, const std::string& arg_key,
                   const std::optional<std::vector<ChatCompletionToolsParam>>& tools) {
  if (!tools.has_value()) return ojson::object();
  for (const ChatCompletionToolsParam& tool : *tools) {
    if (tool.function.name == function_name) {
      if (!tool.function.parameters.has_value()) return ojson::object();
      const nlohmann::json& params = *tool.function.parameters;
      if (!params.is_object() || !params.contains("properties")) return ojson::object();
      const nlohmann::json& props = params["properties"];
      if (!props.is_object() || !props.contains(arg_key)) return ojson::object();
      return ojson(props[arg_key]);
    }
  }
  return ojson::object();
}

// hy_v3_tool_parser.py:100-125 (_get_schema_options).
std::vector<ojson> GetSchemaOptions(const ojson& arg_schema) {
  if (arg_schema.contains("type")) {
    const ojson& type_val = arg_schema["type"];
    if (type_val.is_array()) {
      std::vector<ojson> out;
      for (const ojson& t : type_val) out.push_back(ojson{{"type", t}});
      return out;
    }
    return {arg_schema};
  }
  if (arg_schema.contains("anyOf")) {
    std::vector<ojson> out;
    for (const ojson& s : arg_schema["anyOf"]) out.push_back(s);
    return out;
  }
  if (arg_schema.contains("oneOf")) {
    std::vector<ojson> out;
    for (const ojson& s : arg_schema["oneOf"]) out.push_back(s);
    return out;
  }
  return {ojson{{"type", "string"}}};
}

// hy_v3_tool_parser.py:127-133 (_get_types).
std::set<std::string> GetTypes(const ojson& arg_schema) {
  std::set<std::string> types;
  for (const ojson& s : GetSchemaOptions(arg_schema)) {
    std::string t = "string";
    if (s.contains("type") && s["type"].is_string()) t = s["type"].get<std::string>();
    types.insert(NormalizeType(t));
  }
  types.erase("null");
  return types;
}

bool IsOnlyStringType(const std::string& function_name, const std::string& arg_key,
                      const std::optional<std::vector<ChatCompletionToolsParam>>& tools) {
  const ojson schema = GetArgSchema(function_name, arg_key, tools);
  const std::set<std::string> types = GetTypes(schema);
  return types.size() == 1 && *types.begin() == "string";
}

std::optional<bool> TryParseBool(const std::string& value) {
  const std::string lower = ToLower(value);
  if (lower == "true") return true;
  if (lower == "false") return false;
  return std::nullopt;
}

std::optional<long long> TryParseInt(const std::string& value) {
  const std::string t = Strip(value);
  if (t.empty()) return std::nullopt;
  std::size_t idx = 0;
  try {
    const long long v = std::stoll(t, &idx);
    if (idx != t.size()) return std::nullopt;  // trailing junk -> Python int() fails
    return v;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// hy_v3_tool_parser.py:169-190 (_try_parse_wildcard_number).
std::optional<ojson> TryParseWildcardNumber(const std::string& value) {
  const bool as_float = value.find('.') != std::string::npos ||
                        value.find('e') != std::string::npos ||
                        value.find('E') != std::string::npos;
  const std::string t = Strip(value);
  if (t.empty()) return std::nullopt;
  std::size_t idx = 0;
  try {
    if (as_float) {
      const double v = std::stod(t, &idx);
      if (idx != t.size()) return std::nullopt;
      return ojson(v);
    }
    const long long v = std::stoll(t, &idx);
    if (idx != t.size()) return std::nullopt;
    return ojson(static_cast<std::int64_t>(v));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

// hy_v3_tool_parser.py:192-203 (_deserialize): json.loads then safe_literal_eval.
ojson Deserialize(const std::string& value) {
  ojson parsed;
  try {
    parsed = ojson::parse(value);
    return parsed;
  } catch (const std::exception&) {
    // fall through
  }
  const std::optional<ojson> lit = pythonic_core::parse_literal(value);
  if (lit.has_value()) return *lit;
  return ojson(value);
}

// hy_v3_tool_parser.py:205-252 (_parse_value).
ojson ParseValue(const std::string& value, const std::string& function_name,
                 const std::string& arg_key,
                 const std::optional<std::vector<ChatCompletionToolsParam>>& tools) {
  const ojson schema = GetArgSchema(function_name, arg_key, tools);
  const std::set<std::string> types = GetTypes(schema);

  if (types.count("boolean")) {
    const std::optional<bool> b = TryParseBool(value);
    if (b.has_value()) return ojson(*b);
  }
  if (types.count("integer")) {
    const std::optional<long long> i = TryParseInt(value);
    if (i.has_value()) return ojson(static_cast<std::int64_t>(*i));
  }
  if (types.count("number")) {
    const std::optional<ojson> n = TryParseWildcardNumber(value);
    if (n.has_value()) return *n;
  }
  // types - {string, boolean, integer, number} non-empty -> try json.loads.
  bool has_other = false;
  for (const std::string& t : types) {
    if (t != "string" && t != "boolean" && t != "integer" && t != "number") {
      has_other = true;
      break;
    }
  }
  if (has_other) {
    try {
      return ojson::parse(value);
    } catch (const std::exception&) {
      // fall through
    }
  }
  if (types.count("string")) return ojson(value);
  return Deserialize(value);
}

// Non-greedy scan for <start>(.*?)<end> beginning at/after `from`. Returns the
// inner text + the index just past <end>, or nullopt.
struct TagMatch {
  std::string inner;
  std::size_t end;  // index past the closing tag
};
std::optional<TagMatch> ScanBetween(const std::string& text, std::size_t from,
                                    const std::string& open, const std::string& close,
                                    std::size_t* open_at = nullptr) {
  const std::size_t s = text.find(open, from);
  if (s == std::string::npos) return std::nullopt;
  const std::size_t inner_start = s + open.size();
  const std::size_t e = text.find(close, inner_start);
  if (e == std::string::npos) return std::nullopt;
  if (open_at != nullptr) *open_at = s;
  return TagMatch{text.substr(inner_start, e - inner_start), e + close.size()};
}

}  // namespace

HYV3ToolParser::HYV3ToolParser(std::string suffix) : suffix_(std::move(suffix)) {
  tool_calls_start_token_ = "<tool_calls" + suffix_ + ">";
  tool_calls_end_token_ = "</tool_calls" + suffix_ + ">";
  tool_call_start_token_ = "<tool_call" + suffix_ + ">";
  tool_call_end_token_ = "</tool_call" + suffix_ + ">";
  tool_sep_token_ = "<tool_sep" + suffix_ + ">";
  arg_key_start_token_ = "<arg_key" + suffix_ + ">";
  arg_key_end_token_ = "</arg_key" + suffix_ + ">";
  arg_value_start_token_ = "<arg_value" + suffix_ + ">";
  arg_value_end_token_ = "</arg_value" + suffix_ + ">";
}

void HYV3ToolParser::ResetStreamingToolState() {
  streaming_tool_name_ = std::nullopt;
  completed_args_ = ojson::object();
  current_arg_key_ = std::nullopt;
  current_arg_is_string_ = false;
  streamed_json_len_ = 0;
}

namespace {

// Extract all fully-closed <arg_key>K</arg_key> (ws) <arg_value>V</arg_value>
// pairs from `text`, mirroring func_args_regex.findall/finditer. Each entry
// carries the end index (just past </arg_value>) for the tail computation.
struct ArgKV {
  std::string key;
  std::string value;
  std::size_t end;
};

}  // namespace

ExtractedToolCallInformation HYV3ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  // hy_v3_tool_parser.py:376-397. Sanity gate on the tool_calls start token.
  if (model_output.find(tool_calls_start_token_) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  // hy_v3_tool_parser.py:326-369 (_extract_tool_calls). Collect (name, args)
  // tuples: complete <tool_call>..<tool_sep>..</tool_call> blocks first, then a
  // trailing unterminated portion (start..sep..end-of-string).
  std::vector<std::pair<std::string, std::string>> tuples;
  {
    std::size_t pos = 0;
    std::size_t last_call_end = std::string::npos;
    for (;;) {
      const std::size_t start = model_output.find(tool_call_start_token_, pos);
      if (start == std::string::npos) break;
      const std::size_t name_start = start + tool_call_start_token_.size();
      const std::size_t sep = model_output.find(tool_sep_token_, name_start);
      if (sep == std::string::npos) break;
      const std::size_t args_start = sep + tool_sep_token_.size();
      const std::size_t end = model_output.find(tool_call_end_token_, args_start);
      if (end == std::string::npos) break;
      tuples.emplace_back(model_output.substr(name_start, sep - name_start),
                          model_output.substr(args_start, end - args_start));
      last_call_end = end + tool_call_end_token_.size();
      pos = last_call_end;
    }
    if (!tuples.empty()) {
      // remaining = model_output.split(end_token)[-1].
      const std::string remaining = model_output.substr(last_call_end);
      std::size_t open_at = 0;
      const std::optional<TagMatch> m =
          ScanBetween(remaining, 0, tool_call_start_token_, tool_sep_token_, &open_at);
      if (m.has_value()) {
        const std::size_t args_start =
            open_at + tool_call_start_token_.size() + m->inner.size() + tool_sep_token_.size();
        tuples.emplace_back(m->inner, remaining.substr(args_start));
      }
    } else {
      std::size_t open_at = 0;
      const std::optional<TagMatch> m =
          ScanBetween(model_output, 0, tool_call_start_token_, tool_sep_token_, &open_at);
      if (m.has_value()) {
        const std::size_t args_start =
            open_at + tool_call_start_token_.size() + m->inner.size() + tool_sep_token_.size();
        tuples.emplace_back(m->inner, model_output.substr(args_start));
      }
    }
  }

  std::vector<ToolCall> tool_calls;
  for (const auto& [raw_name, raw_args] : tuples) {
    const std::string function_name = Strip(raw_name);
    const std::string function_args = Strip(raw_args);
    ojson arg_dict = ojson::object();
    // func_args_regex.findall over the args text.
    std::size_t p = 0;
    for (;;) {
      std::size_t k_open = 0;
      const std::optional<TagMatch> key_m =
          ScanBetween(function_args, p, arg_key_start_token_, arg_key_end_token_, &k_open);
      if (!key_m.has_value()) break;
      // require \s* then arg_value_start immediately after </arg_key>.
      const std::size_t gap_start = key_m->end;
      const std::size_t v_open = function_args.find(arg_value_start_token_, gap_start);
      if (v_open == std::string::npos) break;
      if (Strip(function_args.substr(gap_start, v_open - gap_start)) != "") {
        p = key_m->end;  // non-ws between: this key unpaired, advance.
        continue;
      }
      const std::size_t val_start = v_open + arg_value_start_token_.size();
      const std::size_t val_end = function_args.find(arg_value_end_token_, val_start);
      if (val_end == std::string::npos) break;
      const std::string key = key_m->inner;
      const std::string value = function_args.substr(val_start, val_end - val_start);
      arg_dict[key] = ParseValue(value, function_name, key, request.tools);
      p = val_end + arg_value_end_token_.size();
    }
    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = function_name;
    tc.function.arguments = arg_dict.dump();
    tool_calls.push_back(std::move(tc));
  }

  const std::size_t s_index = model_output.find(tool_calls_start_token_);
  const std::string content =
      s_index != std::string::npos ? model_output.substr(0, s_index) : model_output;
  ExtractedToolCallInformation info;
  info.tools_called = true;
  info.tool_calls = std::move(tool_calls);
  if (!content.empty()) info.content = content;
  return info;
}

std::optional<DeltaMessage> HYV3ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  // hy_v3_tool_parser.py:417-419. DEVIATION: text detection replaces the
  // token-id gate (tool_calls_start_token_id not in current_token_ids).
  if (current_text.find(tool_calls_start_token_) == std::string::npos) {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }

  // hy_v3_tool_parser.py:421-429. Buffer, splitting off any preceding content.
  if (delta_text.find(tool_calls_start_token_) != std::string::npos) {
    const std::size_t at = delta_text.find(tool_calls_start_token_);
    // text_parts[-1] is everything after the LAST occurrence; text_parts[0] the
    // text before the FIRST. For our grammar there is at most one per delta.
    const std::size_t last = delta_text.rfind(tool_calls_start_token_);
    buffer_ += delta_text.substr(last + tool_calls_start_token_.size());
    const std::string before = delta_text.substr(0, at);
    if (!before.empty()) {
      DeltaMessage msg;
      msg.content = before;
      return msg;
    }
    // else fall through to process buffer.
  } else {
    buffer_ += delta_text;
  }

  // hy_v3_tool_parser.py:431-437. On the closing "</tool_call></tool_calls>"
  // (adjacent), ensure the buffer carries the end tokens.
  if (current_text.find(tool_call_end_token_ + tool_calls_end_token_) != std::string::npos &&
      buffer_.find(tool_call_end_token_) == std::string::npos) {
    buffer_ += tool_call_end_token_ + tool_calls_end_token_;
  }

  std::string cur_text = buffer_;

  const std::size_t start_idx = cur_text.find(tool_call_start_token_);
  if (start_idx == std::string::npos && !streaming_tool_name_.has_value()) {
    buffer_.clear();
    return std::nullopt;
  }

  // === Phase 1: detect tool name (emit when tool_sep is seen) ===
  std::optional<DeltaMessage> name_delta;
  if (!streaming_tool_name_.has_value()) {
    const std::size_t sep_idx = cur_text.find(tool_sep_token_);
    if (sep_idx == std::string::npos) {
      buffer_ = cur_text.substr(start_idx);
      return std::nullopt;
    }
    const std::size_t name_start = start_idx + tool_call_start_token_.size();
    const std::string tool_name = Strip(cur_text.substr(name_start, sep_idx - name_start));
    streaming_tool_name_ = tool_name;
    buffer_ = cur_text.substr(sep_idx + tool_sep_token_.size());

    current_tool_id += 1;
    current_tool_call_id_ = make_tool_call_id();
    DeltaMessage msg;
    DeltaToolCall tc;
    tc.index = current_tool_id;
    tc.id = current_tool_call_id_;
    tc.type = "function";
    tc.function.name = tool_name;
    msg.tool_calls = std::vector<DeltaToolCall>{tc};
    name_delta = msg;

    if (buffer_.find(tool_call_end_token_) == std::string::npos) {
      return name_delta;
    }
    // else fall through: buffer already carries a complete tool call.
  }

  // === Phase 2: incremental argument streaming ===
  return ExtractStreamingIncremental(name_delta, request);
}

std::optional<DeltaMessage> HYV3ToolParser::ExtractStreamingIncremental(
    const std::optional<DeltaMessage>& name_delta,
    const ChatCompletionRequest& request) {
  // hy_v3_tool_parser.py:499-655.
  const std::string buf = buffer_;
  const bool is_complete = buf.find(tool_call_end_token_) != std::string::npos;

  std::string args_text;
  std::string remaining;
  if (is_complete) {
    const std::size_t end_idx = buf.find(tool_call_end_token_);
    args_text = buf.substr(0, end_idx);
    remaining = buf.substr(end_idx + tool_call_end_token_.size());
  } else {
    args_text = buf;
  }

  // Scan all fully-closed kv pairs; record last closed end for tail detection.
  std::size_t last_closed_end = 0;
  {
    std::size_t p = 0;
    for (;;) {
      std::size_t k_open = 0;
      const std::optional<TagMatch> key_m =
          ScanBetween(args_text, p, arg_key_start_token_, arg_key_end_token_, &k_open);
      if (!key_m.has_value()) break;
      const std::size_t gap_start = key_m->end;
      const std::size_t v_open = args_text.find(arg_value_start_token_, gap_start);
      if (v_open == std::string::npos) break;
      if (Strip(args_text.substr(gap_start, v_open - gap_start)) != "") {
        p = key_m->end;
        continue;
      }
      const std::size_t val_start = v_open + arg_value_start_token_.size();
      const std::size_t val_end = args_text.find(arg_value_end_token_, val_start);
      if (val_end == std::string::npos) break;
      const std::string key = Strip(key_m->inner);
      const std::string value = args_text.substr(val_start, val_end - val_start);
      if (!completed_args_.contains(key)) {
        completed_args_[key] =
            ParseValue(value, streaming_tool_name_.value_or(""), key, request.tools);
      }
      last_closed_end = val_end + arg_value_end_token_.size();
      p = last_closed_end;
    }
  }

  // Detect partial (unclosed) kv at the tail.
  const std::string tail = args_text.substr(last_closed_end);
  std::optional<std::string> partial_key;
  std::optional<std::string> partial_value;
  {
    const std::size_t ak_start = tail.find(arg_key_start_token_);
    if (ak_start != std::string::npos) {
      const std::size_t ak_end =
          tail.find(arg_key_end_token_, ak_start + arg_key_start_token_.size());
      if (ak_end != std::string::npos) {
        partial_key = Strip(tail.substr(ak_start + arg_key_start_token_.size(),
                                        ak_end - (ak_start + arg_key_start_token_.size())));
        current_arg_key_ = partial_key;
        current_arg_is_string_ =
            IsOnlyStringType(streaming_tool_name_.value_or(""), *partial_key, request.tools);
        const std::size_t av_start = tail.find(arg_value_start_token_, ak_end);
        if (av_start != std::string::npos) {
          const std::size_t val_content_start = av_start + arg_value_start_token_.size();
          if (current_arg_is_string_) {
            partial_value = tail.substr(val_content_start);
          }
        }
      } else {
        current_arg_key_ = std::nullopt;
        current_arg_is_string_ = false;
      }
    }
  }

  // Build the current JSON snapshot manually. ONE serializer is used for both
  // the snapshot and the final json so incremental prefixes stay consistent.
  auto build_object = [](const ojson& obj) {
    std::string out = "{";
    bool first = true;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
      if (!first) out += ", ";
      out += ojson(it.key()).dump();
      out += ": ";
      out += it.value().dump();
      first = false;
    }
    out += "}";
    return out;
  };

  std::string snapshot;
  {
    std::string body;
    bool first = true;
    for (auto it = completed_args_.begin(); it != completed_args_.end(); ++it) {
      if (!first) body += ", ";
      body += ojson(it.key()).dump();
      body += ": ";
      body += it.value().dump();
      first = false;
    }
    if (partial_key.has_value() && partial_value.has_value()) {
      if (!first) body += ", ";
      // Open string value: build WITHOUT the closing quote so the prefix stays
      // stable as the value grows (upstream escapes \\, ", \n, \r, \t).
      std::string escaped;
      for (char c : *partial_value) {
        switch (c) {
          case '\\': escaped += "\\\\"; break;
          case '"': escaped += "\\\""; break;
          case '\n': escaped += "\\n"; break;
          case '\r': escaped += "\\r"; break;
          case '\t': escaped += "\\t"; break;
          default: escaped += c; break;
        }
      }
      body += ojson(*partial_key).dump();
      body += ": \"";
      body += escaped;
    }
    snapshot = "{" + body + "}";
  }

  std::optional<std::string> argument_diff;
  if (is_complete) {
    const std::string final_json = build_object(completed_args_);
    if (streamed_json_len_ < final_json.size()) {
      argument_diff = final_json.substr(streamed_json_len_);
    }
    streamed_json_len_ = final_json.size();

    prev_tool_call_arr.push_back(nlohmann::json{
        {"name", streaming_tool_name_.value_or("")},
        {"arguments", nlohmann::json::parse(final_json)}});
    streamed_args_for_tool.push_back(final_json);

    ResetStreamingToolState();
    buffer_ = remaining;
  } else {
    const std::size_t end = snapshot.size() - 1;  // exclude trailing "}"
    if (end > streamed_json_len_) {
      argument_diff = snapshot.substr(streamed_json_len_, end - streamed_json_len_);
      streamed_json_len_ = end;
    }
  }

  if (name_delta.has_value() && argument_diff.has_value()) {
    // hy_v3_tool_parser.py:635-649. Reuse the name carried by name_delta (the
    // tool name emitted this step) so name + first args diff ship together.
    std::optional<std::string> nd_name;
    if (name_delta->tool_calls.has_value() && !name_delta->tool_calls->empty()) {
      nd_name = name_delta->tool_calls->front().function.name;
    }
    DeltaMessage msg;
    DeltaToolCall tc;
    tc.index = current_tool_id;
    tc.id = current_tool_call_id_;
    tc.type = "function";
    tc.function.name = nd_name;
    tc.function.arguments = *argument_diff;
    msg.tool_calls = std::vector<DeltaToolCall>{tc};
    return msg;
  }
  if (name_delta.has_value()) {
    return name_delta;
  }
  if (argument_diff.has_value()) {
    DeltaMessage msg;
    DeltaToolCall tc;
    tc.index = current_tool_id;
    tc.function.arguments = *argument_diff;
    msg.tool_calls = std::vector<DeltaToolCall>{tc};
    return msg;
  }
  return std::nullopt;
}

}  // namespace vllm::entrypoints::openai
