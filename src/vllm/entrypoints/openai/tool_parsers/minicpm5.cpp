// Ported from: vllm/tool_parsers/minicpm5xml_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/minicpm5.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/tool_parsers/pythonic_core.h"

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

// SentencePiece/GPT-style decoders may emit U+0120 (Ġ) / U+010A (Ċ), whose
// UTF-8 encodings are 0xC4 0xA0 and 0xC4 0x8A.
const char kTokenizerSpace[] = "\xC4\xA0";
const char kTokenizerNewline[] = "\xC4\x8A";

const char kStartToken[] = "<function";
const char kEndToken[] = "</function>";

bool Contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

std::string ReplaceAll(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  std::size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

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

// minicpm5xml_tool_parser.py:56-71 (_normalize_model_output).
std::string NormalizeModelOutput(const std::string& text) {
  if (!Contains(text, kTokenizerSpace) && !Contains(text, kTokenizerNewline) &&
      !Contains(text, "<functionname=") && !Contains(text, "<paramname=")) {
    return text;
  }
  std::string out = ReplaceAll(text, kTokenizerSpace, " ");
  out = ReplaceAll(out, kTokenizerNewline, "\n");
  out = ReplaceAll(out, "<functionname=", "<function name=");
  out = ReplaceAll(out, "<paramname=", "<param name=");
  return out;
}

// minicpm5xml_tool_parser.py:74-79 (_strip_thinking_content).
std::string StripThinkingContent(const std::string& text) {
  const std::string kThink = "</think>";
  const std::size_t at = text.rfind(kThink);
  if (at == std::string::npos) return text;
  std::string visible = text.substr(at + kThink.size());
  // lstrip
  std::size_t b = 0;
  while (b < visible.size() && IsWs(visible[b])) ++b;
  visible = visible.substr(b);
  return visible.empty() ? text : visible;
}

// Tool schema maps derived from request.tools.
struct ToolMaps {
  std::set<std::string> tool_names;
  std::map<std::string, std::set<std::string>> allowed_props;
  std::map<std::string, std::set<std::string>> required;
  std::map<std::string, const ChatCompletionToolsParam*> name_to_tool;
};

ToolMaps BuildToolMaps(
    const std::optional<std::vector<ChatCompletionToolsParam>>& tools) {
  ToolMaps m;
  if (!tools.has_value()) return m;
  for (const ChatCompletionToolsParam& tool : *tools) {
    const std::string& name = tool.function.name;
    if (name.empty()) continue;
    m.name_to_tool[name] = &tool;
    m.tool_names.insert(name);
    std::set<std::string> props;
    std::set<std::string> req;
    if (tool.function.parameters.has_value() && tool.function.parameters->is_object()) {
      const nlohmann::json& params = *tool.function.parameters;
      if (params.contains("properties") && params["properties"].is_object()) {
        for (auto it = params["properties"].begin(); it != params["properties"].end();
             ++it) {
          props.insert(it.key());
        }
      }
      if (params.contains("required") && params["required"].is_array()) {
        for (const auto& r : params["required"]) {
          if (r.is_string()) req.insert(r.get<std::string>());
        }
      }
    }
    m.allowed_props[name] = std::move(props);
    m.required[name] = std::move(req);
  }
  return m;
}

// minicpm5xml_tool_parser.py:113-121 (_parse_arguments): json.loads then
// safe_literal_eval. Returns {value, ok}.
std::pair<ojson, bool> ParseArguments(const std::string& json_value) {
  try {
    return {ojson::parse(json_value), true};
  } catch (const std::exception&) {
    // fall through to literal eval
  }
  const std::optional<ojson> lit = pythonic_core::parse_literal(json_value);
  if (lit.has_value()) return {*lit, true};
  return {ojson(json_value), false};
}

// minicpm5xml_tool_parser.py:124-135 (_get_argument_type).
std::optional<std::string> GetArgumentType(const std::string& func_name,
                                           const std::string& arg_key,
                                           const ToolMaps& maps) {
  const auto it = maps.name_to_tool.find(func_name);
  if (it == maps.name_to_tool.end()) return std::nullopt;
  const ChatCompletionToolsParam* tool = it->second;
  if (!tool->function.parameters.has_value()) return std::nullopt;
  const nlohmann::json& params = *tool->function.parameters;
  if (!params.is_object() || !params.contains("properties")) return std::nullopt;
  const nlohmann::json& props = params["properties"];
  if (!props.is_object() || !props.contains(arg_key)) return std::nullopt;
  const nlohmann::json& prop = props[arg_key];
  if (!prop.is_object() || !prop.contains("type") || !prop["type"].is_string()) {
    return std::nullopt;
  }
  return prop["type"].get<std::string>();
}

// minicpm5xml_tool_parser.py:138-152 (_coerce_argument_value).
ojson CoerceArgumentValue(const std::string& func_name, const std::string& arg_key,
                          const ojson& value, const ToolMaps& maps) {
  const std::optional<std::string> arg_type = GetArgumentType(func_name, arg_key, maps);
  if (arg_type.has_value() && *arg_type == "string") {
    if (value.is_string()) return value;
    return ojson(value.dump());
  }
  if (value.is_string()) {
    const auto [parsed, ok] = ParseArguments(value.get<std::string>());
    (void)ok;
    return parsed;
  }
  return value;
}

// minicpm5xml_tool_parser.py:155-195 (_add_argument). Returns false if the
// argument violates the tool schema.
bool AddArgument(const std::string& func_name, const std::string& key,
                 const std::string& val_text, ojson& arguments,
                 std::set<std::string>& seen_keys,
                 const std::set<std::string>& allowed_props, const ToolMaps& maps) {
  const bool is_wrapper = (key == "properties" || key == "arguments");
  if (is_wrapper && !allowed_props.empty() && allowed_props.count(key) == 0) {
    const auto [parsed_val, ok] = ParseArguments(val_text);
    if (!ok || !parsed_val.is_object()) return false;
    bool added = false;
    for (auto it = parsed_val.begin(); it != parsed_val.end(); ++it) {
      const std::string& wrapped_key = it.key();
      if (allowed_props.count(wrapped_key) == 0) continue;
      if (seen_keys.count(wrapped_key)) return false;
      seen_keys.insert(wrapped_key);
      arguments[wrapped_key] =
          CoerceArgumentValue(func_name, wrapped_key, it.value(), maps);
      added = true;
    }
    return added;
  }

  if (!allowed_props.empty() && allowed_props.count(key) == 0) return true;  // ignore extra
  if (seen_keys.count(key)) return false;
  seen_keys.insert(key);
  arguments[key] = CoerceArgumentValue(func_name, key, ojson(val_text), maps);
  return true;
}

// Truthy in the Python sense (used by the alias table's `a or b`).
bool Truthy(const ojson& v) {
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_string()) return !v.get<std::string>().empty();
  if (v.is_number_integer()) return v.get<long long>() != 0;
  if (v.is_number_float()) return v.get<double>() != 0.0;
  if (v.is_number_unsigned()) return v.get<unsigned long long>() != 0;
  if (v.is_array() || v.is_object()) return !v.empty();
  return true;
}

// args.get(a) or args.get(b): first present-and-truthy value.
std::optional<ojson> GetOr(const ojson& args, std::initializer_list<const char*> keys) {
  for (const char* k : keys) {
    if (args.contains(k) && Truthy(args[k])) return args[k];
  }
  return std::nullopt;
}

// minicpm5xml_tool_parser.py:198-273 (_normalize_alias_tool_call).
std::pair<std::string, ojson> NormalizeAliasToolCall(const std::string& func_name,
                                                     const ojson& arguments,
                                                     const std::set<std::string>& tool_names) {
  auto has = [&](const char* n) { return tool_names.count(n) != 0; };
  if (tool_names.count(func_name)) return {func_name, arguments};

  if (func_name == "get_details_by_phone" && has("get_customer_by_phone")) {
    const std::optional<ojson> phone = GetOr(arguments, {"phone_number", "phone"});
    if (phone.has_value()) return {"get_customer_by_phone", ojson{{"phone_number", *phone}}};
  }
  if (func_name == "get_details_by_name" && has("get_customer_by_name")) {
    const std::optional<ojson> full_name = GetOr(arguments, {"full_name", "name"});
    const std::optional<ojson> dob = GetOr(arguments, {"date_of_birth", "dob"});
    ojson mapped = ojson::object();
    if (full_name.has_value()) mapped["full_name"] = *full_name;
    if (dob.has_value()) {
      mapped["date_of_birth"] = *dob;
      mapped["dob"] = *dob;
    }
    if (!mapped.empty()) return {"get_customer_by_name", mapped};
  }
  if ((func_name == "get_line_details" || func_name == "get_line_status" ||
       func_name == "get_roaming_status") &&
      has("get_details_by_id")) {
    const std::optional<ojson> detail_id = GetOr(arguments, {"line_id", "id"});
    if (detail_id.has_value()) return {"get_details_by_id", ojson{{"id", *detail_id}}};
  }
  if (func_name == "get_plan_details" && has("get_details_by_id")) {
    const std::optional<ojson> plan_id = GetOr(arguments, {"plan_id", "id"});
    if (plan_id.has_value()) return {"get_details_by_id", ojson{{"id", *plan_id}}};
  }
  if (func_name == "enable_roaming" && has("toggle_roaming")) {
    const std::optional<ojson> line_id = GetOr(arguments, {"line_id", "id"});
    if (line_id.has_value())
      return {"toggle_roaming", ojson{{"line_id", *line_id}, {"enabled", true}}};
  }
  if (func_name == "disable_roaming" && has("toggle_roaming")) {
    const std::optional<ojson> line_id = GetOr(arguments, {"line_id", "id"});
    if (line_id.has_value())
      return {"toggle_roaming", ojson{{"line_id", *line_id}, {"enabled", false}}};
  }
  if ((func_name == "add_refueled_data" || func_name == "add_data_to_line") &&
      has("refuel_data")) {
    const std::optional<ojson> line_id = GetOr(arguments, {"line_id", "id"});
    const std::optional<ojson> amount =
        GetOr(arguments, {"amount_gb", "gb", "gb_amount", "amount"});
    ojson mapped = ojson::object();
    if (arguments.contains("customer_id")) mapped["customer_id"] = arguments["customer_id"];
    if (line_id.has_value()) mapped["line_id"] = *line_id;
    if (amount.has_value()) mapped["amount_gb"] = *amount;
    if (!mapped.empty()) return {"refuel_data", mapped};
  }
  return {func_name, arguments};
}

const std::regex& FuncNameRegex() {
  static const std::regex re(R"(<function\s+name=['"]([^'"]+)['"][^>]*>)");
  return re;
}
const std::regex& ParamWithNameRegex() {
  static const std::regex re(R"(<param\s+name=['"]([^'"]+)['"]>([\s\S]*?)</param>)");
  return re;
}
const std::regex& ParamMissingNameRegex() {
  static const std::regex re(R"(<param(?![^>]*\bname=)[^>]*>)");
  return re;
}
const std::regex& FuncBlockRegex() {
  static const std::regex re(R"(<function[\s\S]*?</function>)");
  return re;
}

std::string StripCData(std::string val_text) {
  const std::string open = "<![CDATA[";
  const std::string close = "]]>";
  if (val_text.size() >= open.size() + close.size() &&
      val_text.compare(0, open.size(), open) == 0 &&
      val_text.compare(val_text.size() - close.size(), close.size(), close) == 0) {
    val_text = val_text.substr(open.size(), val_text.size() - open.size() - close.size());
  }
  return val_text;
}

// minicpm5xml_tool_parser.py:316-429 (_parse_function_block), regex branch only.
// Returns {name, parameters} on success, nullopt otherwise.
std::optional<std::pair<std::string, ojson>> ParseFunctionBlock(const std::string& block,
                                                                const ToolMaps& maps) {
  std::string func_name;
  ojson arguments = ojson::object();
  bool param_invalid = false;

  std::smatch mfn;
  if (std::regex_search(block, mfn, FuncNameRegex())) {
    func_name = Strip(mfn[1].str());
  }
  bool has_invalid_param = std::regex_search(block, ParamMissingNameRegex());

  std::set<std::string> seen_keys;
  const std::set<std::string> empty_props;
  const auto ap_it = maps.allowed_props.find(func_name);
  const std::set<std::string>& allowed_props =
      ap_it != maps.allowed_props.end() ? ap_it->second : empty_props;

  if (!has_invalid_param) {
    auto begin = std::sregex_iterator(block.begin(), block.end(), ParamWithNameRegex());
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      const std::string key = Strip((*it)[1].str());
      std::string val_text = (*it)[2].str();
      val_text = StripCData(val_text);
      val_text = Strip(val_text);
      if (!AddArgument(func_name, key, val_text, arguments, seen_keys, allowed_props, maps)) {
        has_invalid_param = true;
        break;
      }
    }
  }
  if (has_invalid_param) {
    arguments = ojson::object();
    param_invalid = true;
  }
  const bool parsed_ok = !func_name.empty();

  if (func_name.empty() || param_invalid) return std::nullopt;

  auto [mapped_name, mapped_args] =
      NormalizeAliasToolCall(func_name, arguments, maps.tool_names);
  if (maps.tool_names.count(mapped_name) == 0) return std::nullopt;

  const auto req_it = maps.required.find(mapped_name);
  if (req_it != maps.required.end() && !req_it->second.empty()) {
    for (const std::string& rp : req_it->second) {
      if (!mapped_args.contains(rp)) return std::nullopt;
    }
  }
  if (!parsed_ok) return std::nullopt;
  return std::make_pair(mapped_name, mapped_args);
}

// minicpm5xml_tool_parser.py:432-458 (_parse_partial_params).
ojson ParsePartialParams(const std::string& block, const std::string& func_name,
                         const ToolMaps& maps) {
  ojson arguments = ojson::object();
  std::set<std::string> seen_keys;
  const std::set<std::string> empty_props;
  const auto ap_it = maps.allowed_props.find(func_name);
  const std::set<std::string>& allowed_props =
      ap_it != maps.allowed_props.end() ? ap_it->second : empty_props;

  auto begin = std::sregex_iterator(block.begin(), block.end(), ParamWithNameRegex());
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    const std::string key = Strip((*it)[1].str());
    if (key.empty() || seen_keys.count(key)) continue;
    std::string val_text = (*it)[2].str();
    val_text = StripCData(val_text);
    val_text = Strip(val_text);
    AddArgument(func_name, key, val_text, arguments, seen_keys, allowed_props, maps);
  }
  return arguments;
}

// utils.py:42 (partial_tag_overlap).
std::size_t PartialTagOverlap(const std::string& text, const std::string& tag) {
  const std::size_t max_check = std::min(tag.size() - 1, text.size());
  for (std::size_t k = max_check; k > 0; --k) {
    if (text.size() >= k && text.compare(text.size() - k, k, tag, 0, k) == 0) {
      return k;
    }
  }
  return 0;
}

// minicpm5xml_tool_parser.py:82-86 (_streaming_args_snapshot).
std::string StreamingArgsSnapshot(const std::string& args_json, bool is_complete) {
  if (is_complete || args_json.empty() || args_json.back() != '}') return args_json;
  return args_json.substr(0, args_json.size() - 1);
}

// minicpm5xml_tool_parser.py:89-110 (_streaming_args_diff).
std::optional<std::string> StreamingArgsDiff(std::string prev_args,
                                             const std::string& args_json,
                                             bool is_complete) {
  if (prev_args == "{}") prev_args = "";
  const std::string target = StreamingArgsSnapshot(args_json, is_complete);
  if (prev_args.empty()) {
    if (target.empty()) return std::nullopt;
    return target;
  }
  if (target == prev_args) return std::nullopt;
  if (target.size() >= prev_args.size() && target.compare(0, prev_args.size(), prev_args) == 0) {
    const std::string diff = target.substr(prev_args.size());
    if (diff.empty()) return std::nullopt;
    return diff;
  }
  if (!prev_args.empty() && prev_args.back() == '}') {
    const std::string prev_open = prev_args.substr(0, prev_args.size() - 1);
    if (target.size() >= prev_open.size() &&
        target.compare(0, prev_open.size(), prev_open) == 0) {
      const std::string diff = target.substr(prev_open.size());
      if (diff.empty()) return std::nullopt;
      return diff;
    }
  }
  return std::nullopt;
}

}  // namespace

void MiniCPM5ToolParser::ResetStreamState() {
  processed_len_ = 0;
  current_tool_id = -1;
  current_tool_name_sent = false;
  prev_tool_call_arr.clear();
  streamed_args_for_tool.clear();
}

ExtractedToolCallInformation MiniCPM5ToolParser::extract_tool_calls(
    const std::string& model_output_in, const ChatCompletionRequest& request) {
  const std::string model_output = NormalizeModelOutput(model_output_in);
  if (!Contains(model_output, kStartToken)) {
    return ExtractedToolCallInformation{false, {}, StripThinkingContent(model_output)};
  }

  const ToolMaps maps = BuildToolMaps(request.tools);

  std::vector<ToolCall> tool_calls;
  std::string normal_parts;
  std::size_t last_end = 0;

  auto begin = std::sregex_iterator(model_output.begin(), model_output.end(), FuncBlockRegex());
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    const std::smatch& match = *it;
    const std::size_t mstart = static_cast<std::size_t>(match.position(0));
    const std::size_t mend = mstart + static_cast<std::size_t>(match.length(0));
    if (mstart > last_end) normal_parts += model_output.substr(last_end, mstart - last_end);

    const std::string block = match.str(0);
    const std::optional<std::pair<std::string, ojson>> parsed = ParseFunctionBlock(block, maps);
    if (parsed.has_value()) {
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = parsed->first;
      tc.function.arguments = parsed->second.dump();
      tool_calls.push_back(std::move(tc));
    } else {
      normal_parts += block;
    }
    last_end = mend;
  }
  if (last_end < model_output.size()) normal_parts += model_output.substr(last_end);

  const std::string content = StripThinkingContent(Strip(normal_parts));
  const bool tools_called = !tool_calls.empty();
  ExtractedToolCallInformation info;
  info.tools_called = tools_called;
  info.tool_calls = std::move(tool_calls);
  if (!tools_called) info.content = content;
  return info;
}

DeltaMessage MiniCPM5ToolParser::StartToolCall(const std::string& func_name) {
  current_tool_id += 1;
  current_tool_name_sent = true;
  while (static_cast<int>(streamed_args_for_tool.size()) <= current_tool_id) {
    streamed_args_for_tool.emplace_back();
  }
  while (static_cast<int>(prev_tool_call_arr.size()) <= current_tool_id) {
    prev_tool_call_arr.emplace_back(nlohmann::json::object());
  }
  prev_tool_call_arr[current_tool_id] = nlohmann::json{{"name", func_name}};
  DeltaMessage msg;
  DeltaToolCall tc;
  tc.index = current_tool_id;
  tc.id = make_tool_call_id();
  tc.type = "function";
  tc.function.name = func_name;
  msg.tool_calls = std::vector<DeltaToolCall>{tc};
  return msg;
}

std::optional<DeltaMessage> MiniCPM5ToolParser::EmitToolArgsDelta(
    int tool_index, const std::string& args_json, bool is_complete) {
  const std::string prev_args =
      tool_index < static_cast<int>(streamed_args_for_tool.size())
          ? streamed_args_for_tool[tool_index]
          : std::string();
  const std::optional<std::string> arg_diff =
      StreamingArgsDiff(prev_args, args_json, is_complete);
  if (!arg_diff.has_value()) return std::nullopt;
  while (static_cast<int>(streamed_args_for_tool.size()) <= tool_index) {
    streamed_args_for_tool.emplace_back();
  }
  streamed_args_for_tool[tool_index] = prev_args + *arg_diff;
  DeltaMessage msg;
  DeltaToolCall tc;
  tc.index = tool_index;
  tc.function.arguments = *arg_diff;
  msg.tool_calls = std::vector<DeltaToolCall>{tc};
  return msg;
}

std::optional<DeltaMessage> MiniCPM5ToolParser::ProcessCompleteBlockStreaming(
    const std::string& block, const ChatCompletionRequest& request) {
  const ToolMaps maps = BuildToolMaps(request.tools);
  const std::optional<std::pair<std::string, ojson>> parsed = ParseFunctionBlock(block, maps);
  if (!parsed.has_value()) {
    DeltaMessage msg;
    msg.content = block;
    return msg;
  }
  const std::string args_json = parsed->second.dump();
  const std::string& func_name = parsed->first;

  if (!current_tool_name_sent) {
    current_tool_id += 1;
    const int tool_index = current_tool_id;
    while (static_cast<int>(streamed_args_for_tool.size()) <= tool_index) {
      streamed_args_for_tool.emplace_back();
    }
    while (static_cast<int>(prev_tool_call_arr.size()) <= tool_index) {
      prev_tool_call_arr.emplace_back(nlohmann::json::object());
    }
    streamed_args_for_tool[tool_index] = args_json;
    prev_tool_call_arr[tool_index] =
        nlohmann::json{{"name", func_name}, {"arguments", parsed->second}};
    DeltaMessage msg;
    DeltaToolCall tc;
    tc.index = tool_index;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = func_name;
    tc.function.arguments = args_json;
    msg.tool_calls = std::vector<DeltaToolCall>{tc};
    return msg;
  }

  const int tool_index = current_tool_id;
  prev_tool_call_arr[tool_index]["arguments"] = parsed->second;
  const std::optional<DeltaMessage> delta = EmitToolArgsDelta(tool_index, args_json, true);
  current_tool_name_sent = false;
  if (delta.has_value()) return delta;
  DeltaMessage msg;
  msg.content = "";
  return msg;
}

std::optional<DeltaMessage> MiniCPM5ToolParser::ProcessPartialBlockStreaming(
    const std::string& block, const ChatCompletionRequest& request) {
  const ToolMaps maps = BuildToolMaps(request.tools);
  if (std::regex_search(block, ParamMissingNameRegex())) return std::nullopt;

  std::smatch mfn;
  if (!std::regex_search(block, mfn, FuncNameRegex())) return std::nullopt;
  const std::string func_name = Strip(mfn[1].str());
  if (maps.tool_names.count(func_name) == 0) return std::nullopt;

  if (!current_tool_name_sent) return StartToolCall(func_name);

  const ojson arguments = ParsePartialParams(block, func_name, maps);
  if (arguments.empty()) return std::nullopt;
  const std::string args_json = arguments.dump();
  prev_tool_call_arr[current_tool_id]["arguments"] = arguments;
  return EmitToolArgsDelta(current_tool_id, args_json, false);
}

std::optional<DeltaMessage> MiniCPM5ToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& current_text_in,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  const std::string current_text = NormalizeModelOutput(current_text_in);
  if (previous_text.empty()) ResetStreamState();

  if (!Contains(current_text, kStartToken)) {
    if (processed_len_ < current_text.size()) {
      const std::string content = current_text.substr(processed_len_);
      processed_len_ = current_text.size();
      if (content.empty()) return std::nullopt;
      DeltaMessage msg;
      msg.content = content;
      return msg;
    }
    return std::nullopt;
  }

  auto begin = std::sregex_iterator(current_text.begin(), current_text.end(), FuncBlockRegex());
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    const std::smatch& match = *it;
    const std::size_t mstart = static_cast<std::size_t>(match.position(0));
    const std::size_t mend = mstart + static_cast<std::size_t>(match.length(0));
    if (mend <= processed_len_) continue;
    if (mstart > processed_len_) {
      const std::string gap = current_text.substr(processed_len_, mstart - processed_len_);
      processed_len_ = mstart;
      if (gap.empty()) return std::nullopt;
      DeltaMessage msg;
      msg.content = gap;
      return msg;
    }
    const std::string block = match.str(0);
    const std::optional<DeltaMessage> delta = ProcessCompleteBlockStreaming(block, request);
    processed_len_ = mend;
    if (delta.has_value()) return delta;
  }

  std::string remainder = current_text.substr(processed_len_);
  if (remainder.empty()) {
    if (delta_text.empty() && current_tool_id >= 0 && !current_tool_name_sent) {
      DeltaMessage msg;
      msg.content = "";
      return msg;
    }
    return std::nullopt;
  }

  const std::size_t func_idx = remainder.find(kStartToken);
  if (func_idx != std::string::npos && func_idx > 0) {
    const std::string gap = remainder.substr(0, func_idx);
    processed_len_ += func_idx;
    DeltaMessage msg;
    msg.content = gap;
    return msg;
  }
  if (func_idx == std::string::npos) {
    if (PartialTagOverlap(remainder, kStartToken) != 0) return std::nullopt;
    processed_len_ = current_text.size();
    if (remainder.empty()) return std::nullopt;
    DeltaMessage msg;
    msg.content = remainder;
    return msg;
  }

  std::string partial_block = remainder.substr(func_idx);
  if (Contains(partial_block, kEndToken)) {
    const std::size_t end_idx = partial_block.rfind(kEndToken);
    const std::string complete_block =
        partial_block.substr(0, end_idx + std::string(kEndToken).size());
    const std::optional<DeltaMessage> delta = ProcessCompleteBlockStreaming(complete_block, request);
    processed_len_ += func_idx + complete_block.size();
    if (delta.has_value()) return delta;
    partial_block = partial_block.substr(end_idx + std::string(kEndToken).size());
    if (Strip(partial_block).empty()) return std::nullopt;
    const std::size_t next = partial_block.find(kStartToken);
    if (next == std::string::npos) {
      processed_len_ = current_text.size();
      if (!partial_block.empty()) {
        DeltaMessage msg;
        msg.content = partial_block;
        return msg;
      }
      return std::nullopt;
    }
    partial_block = partial_block.substr(next);
  }

  return ProcessPartialBlockStreaming(partial_block, request);
}

}  // namespace vllm::entrypoints::openai
