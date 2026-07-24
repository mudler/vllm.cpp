// Ported from: vllm/tool_parsers/poolside_v1_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/poolside_v1.h"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

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
bool EndsWith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// poolside_v1_tool_parser.py:105-117 (_deserialize): json.loads then
// safe_literal_eval, then the raw string.
ojson Deserialize(const std::string& value) {
  try {
    return ojson::parse(value);
  } catch (const std::exception&) {
    // fall through
  }
  const std::optional<ojson> lit = pythonic_core::parse_literal(value);
  if (lit.has_value()) return *lit;
  return ojson(value);
}

// poolside_v1_tool_parser.py:119-128 (_json_escape_string_content):
// json.dumps(s, ensure_ascii=False)[1:-1].
std::string JsonEscape(const std::string& s) {
  if (s.empty()) return "";
  const std::string dumped = ojson(s).dump();  // includes surrounding quotes
  return dumped.substr(1, dumped.size() - 2);
}

}  // namespace

bool PoolsideV1ToolParser::IsStringType(const std::string& tool_name,
                                        const std::string& arg_name,
                                        const ChatCompletionRequest& request) const {
  // poolside_v1_tool_parser.py:130-150 (_is_string_type). properties[arg].type
  // == "string".
  if (!request.tools.has_value()) return false;
  for (const ChatCompletionToolsParam& tool : *request.tools) {
    if (tool.function.name != tool_name) continue;
    if (!tool.function.parameters.has_value()) return false;
    const nlohmann::json& params = *tool.function.parameters;
    if (!params.is_object() || !params.contains("properties")) return false;
    const nlohmann::json& props = params.at("properties");
    if (!props.is_object() || !props.contains(arg_name)) return false;
    const nlohmann::json& arg = props.at(arg_name);
    if (!arg.is_object() || !arg.contains("type")) return false;
    const nlohmann::json& type = arg.at("type");
    return type.is_string() && type.get<std::string>() == "string";
  }
  return false;
}

ExtractedToolCallInformation PoolsideV1ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  // poolside_v1_tool_parser.py:193 - func_call_regex.findall: every complete
  // <tool_call>..</tool_call> block.
  const std::string start = kToolCallStart;
  const std::string end = kToolCallEnd;

  std::vector<ToolCall> tool_calls;
  try {
    std::size_t pos = 0;
    while (true) {
      const std::size_t s = model_output.find(start, pos);
      if (s == std::string::npos) break;
      const std::size_t inner_start = s + start.size();
      const std::size_t e = model_output.find(end, inner_start);
      if (e == std::string::npos) break;  // only complete blocks
      const std::string inner = model_output.substr(inner_start, e - inner_start);
      pos = e + end.size();

      // poolside_v1_tool_parser.py:78-80 (func_detail_regex): name =
      // \s*([^\n<]+?)\s* up to the first '\n' or '<'; args = from first <arg_key>.
      std::size_t np = 0;
      while (np < inner.size() && IsWs(inner[np])) ++np;
      std::size_t ne = np;
      while (ne < inner.size() && inner[ne] != '\n' && inner[ne] != '<') ++ne;
      const std::string tc_name = Strip(inner.substr(np, ne - np));

      ojson arg_dct = ojson::object();
      const std::size_t ak = inner.find(kArgKeyStart);
      if (ak != std::string::npos) {
        const std::string tc_args = inner.substr(ak);
        // poolside_v1_tool_parser.py:81-83 (func_arg_regex):
        // <arg_key>(K)</arg_key>\s*<arg_value>(V)</arg_value> findall.
        std::size_t p = 0;
        while (true) {
          const std::size_t k_open = tc_args.find(kArgKeyStart, p);
          if (k_open == std::string::npos) break;
          const std::size_t k_content = k_open + std::string(kArgKeyStart).size();
          const std::size_t k_close = tc_args.find(kArgKeyEnd, k_content);
          if (k_close == std::string::npos) break;
          const std::size_t gap_start = k_close + std::string(kArgKeyEnd).size();
          const std::size_t v_open = tc_args.find(kArgValStart, gap_start);
          if (v_open == std::string::npos) break;
          if (!Strip(tc_args.substr(gap_start, v_open - gap_start)).empty()) {
            p = gap_start;  // non-ws between key/value: this key unpaired.
            continue;
          }
          const std::size_t v_content = v_open + std::string(kArgValStart).size();
          const std::size_t v_close = tc_args.find(kArgValEnd, v_content);
          if (v_close == std::string::npos) break;
          const std::string key = Strip(tc_args.substr(k_content, k_close - k_content));
          const std::string value = tc_args.substr(v_content, v_close - v_content);
          // poolside_v1_tool_parser.py:210-216 - string kept verbatim; else
          // strip + deserialize.
          if (IsStringType(tc_name, key, request)) {
            arg_dct[key] = ojson(value);
          } else {
            arg_dct[key] = Deserialize(Strip(value));
          }
          p = v_close + std::string(kArgValEnd).size();
        }
      }

      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = tc_name;
      tc.function.arguments = arg_dct.dump();
      tool_calls.push_back(std::move(tc));
    }
  } catch (const std::exception&) {
    // poolside_v1_tool_parser.py:228-232 - any error -> whole output as content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  // poolside_v1_tool_parser.py:234-246.
  if (!tool_calls.empty()) {
    const std::size_t s = model_output.find(start);
    std::optional<std::string> content;
    if (s != std::string::npos) {
      const std::string c = model_output.substr(0, s);
      if (!Strip(c).empty()) content = c;
    }
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    info.content = content;
    return info;
  }
  return ExtractedToolCallInformation{false, {}, model_output};
}

// ---------------------------------------------------------------------------
// Streaming state machine.
// ---------------------------------------------------------------------------

void PoolsideV1ToolParser::EnsureToolState() {
  const std::size_t need = static_cast<std::size_t>(current_tool_id) + 1;
  while (tool_call_ids_.size() < need) tool_call_ids_.push_back(make_tool_call_id());
  while (streamed_args_for_tool.size() < need) streamed_args_for_tool.emplace_back();
  while (prev_tool_call_arr.size() < need)
    prev_tool_call_arr.push_back(nlohmann::json::object());
  while (args_started_.size() < need) args_started_.push_back(false);
  while (args_closed_.size() < need) args_closed_.push_back(false);
  while (seen_keys_.size() < need) seen_keys_.emplace_back();
}

void PoolsideV1ToolParser::BeginToolCall() {
  if (current_tool_id == -1) {
    current_tool_id = 0;
  } else {
    current_tool_id += 1;
  }
  EnsureToolState();
  current_tool_name_sent = false;
  current_tool_name_ = std::nullopt;
  pending_key_ = std::nullopt;
  streaming_string_value_ = false;
  in_tool_call_ = true;
}

void PoolsideV1ToolParser::FinishToolCall() {
  in_tool_call_ = false;
  current_tool_name_ = std::nullopt;
  pending_key_ = std::nullopt;
  streaming_string_value_ = false;
}

void PoolsideV1ToolParser::RevertLastToolCallState() {
  if (current_tool_id < 0) return;
  tool_call_ids_.pop_back();
  streamed_args_for_tool.pop_back();
  prev_tool_call_arr.pop_back();
  args_started_.pop_back();
  args_closed_.pop_back();
  seen_keys_.pop_back();
  current_tool_id -= 1;
}

DeltaToolCall& PoolsideV1ToolParser::GetOrCreateDelta(
    std::map<int, DeltaToolCall>& pending) {
  const int idx = current_tool_id;
  auto it = pending.find(idx);
  if (it == pending.end()) {
    DeltaToolCall d;
    d.index = idx;
    it = pending.emplace(idx, std::move(d)).first;
  }
  return it->second;
}

void PoolsideV1ToolParser::UpdateToolName(std::map<int, DeltaToolCall>& pending,
                                          const std::string& tool_name) {
  prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
      nlohmann::json{{"name", current_tool_name_.value_or("")},
                     {"arguments", nlohmann::json::object()}};
  DeltaToolCall& delta = GetOrCreateDelta(pending);
  delta.id = tool_call_ids_[static_cast<std::size_t>(current_tool_id)];
  delta.type = "function";
  delta.function.name = tool_name;
  if (!delta.function.arguments.has_value()) delta.function.arguments = "";
}

void PoolsideV1ToolParser::UpdateToolArgs(std::map<int, DeltaToolCall>& pending,
                                          const std::string& fragment) {
  // Best-effort fill of the internal snapshot (upstream _complete_json_prefix +
  // partial_json_parser). Only prev_tool_call_arr is affected; the emitted delta
  // is the raw fragment.
  const std::string& streamed =
      streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)];
  try {
    prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)]["arguments"] =
        nlohmann::json::parse(streamed);
  } catch (const std::exception&) {
    // partial: leave the previous snapshot in place.
  }
  DeltaToolCall& delta = GetOrCreateDelta(pending);
  if (!delta.function.arguments.has_value()) delta.function.arguments = "";
  *delta.function.arguments += fragment;
}

std::optional<std::string> PoolsideV1ToolParser::AppendArgFragment(
    const std::string& key_in, const std::string& raw_val) {
  const std::string key = Strip(key_in);
  if (key.empty()) return std::nullopt;
  std::set<std::string>& seen = seen_keys_[static_cast<std::size_t>(current_tool_id)];
  if (seen.count(key)) return std::nullopt;

  const ojson val_obj = Deserialize(raw_val);
  const std::string key_json = ojson(key).dump();
  const std::string val_json = val_obj.dump();

  std::string fragment;
  const auto ci = static_cast<std::size_t>(current_tool_id);
  if (!args_started_[ci]) {
    fragment = "{" + key_json + ": " + val_json;
    args_started_[ci] = true;
  } else {
    fragment = ", " + key_json + ": " + val_json;
  }

  seen.insert(key);
  streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] += fragment;
  return fragment;
}

std::optional<std::string> PoolsideV1ToolParser::CloseArgsIfNeeded() {
  auto idx = static_cast<std::size_t>(current_tool_id);
  if (args_closed_[idx]) return std::nullopt;
  args_closed_[idx] = true;
  std::string fragment;
  if (!args_started_[idx]) {
    fragment = "{}";
    streamed_args_for_tool[idx] = fragment;
  } else {
    fragment = "}";
    streamed_args_for_tool[idx] += fragment;
  }
  return fragment;
}

std::optional<DeltaMessage> PoolsideV1ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  // poolside_v1_tool_parser.py:258-259 - tools disabled -> pass delta through.
  const bool tools_enabled =
      request.tools.has_value() && !request.tools->empty() &&
      (!request.tool_choice.has_value() || request.tool_choice->mode != "none");
  if (!tools_enabled) {
    if (delta_text.empty()) return std::nullopt;
    DeltaMessage m;
    m.content = delta_text;
    return m;
  }

  buffer_ += delta_text;

  std::map<int, DeltaToolCall> pending;
  std::optional<std::string> content;

  const std::string start = kToolCallStart;
  const std::string end = kToolCallEnd;
  const std::string ak_start = kArgKeyStart;
  const std::string ak_end = kArgKeyEnd;
  const std::string av_start = kArgValStart;
  const std::string av_end = kArgValEnd;

  while (true) {
    if (!in_tool_call_) {
      const std::size_t start_idx = buffer_.find(start);
      if (start_idx == std::string::npos) {
        // poolside_v1_tool_parser.py:271-282 - partial start token at end?
        bool matched = false;
        for (std::size_t i = 1; i < start.size(); ++i) {
          if (EndsWith(buffer_, start.substr(0, i))) {
            const std::string out = buffer_.substr(0, buffer_.size() - i);
            buffer_ = buffer_.substr(buffer_.size() - i);
            if (!out.empty()) content = content.value_or("") + out;
            matched = true;
            break;
          }
        }
        if (!matched) {
          const std::string out = buffer_;
          buffer_.clear();
          if (!out.empty()) content = content.value_or("") + out;
        }
        break;
      }
      if (start_idx > 0) {
        content = content.value_or("") + buffer_.substr(0, start_idx);
        buffer_ = buffer_.substr(start_idx);
      }
      buffer_ = buffer_.substr(start.size());
      BeginToolCall();
      continue;
    }

    // poolside_v1_tool_parser.py:294-319 - parse the tool name.
    if (!current_tool_name_sent) {
      const std::size_t nl = buffer_.find('\n');
      const std::size_t akp = buffer_.find(ak_start);
      const std::size_t endp = buffer_.find(end);
      std::size_t cut = std::string::npos;
      for (std::size_t cand : {nl, akp, endp}) {
        if (cand != std::string::npos && (cut == std::string::npos || cand < cut))
          cut = cand;
      }
      if (cut == std::string::npos) break;
      const std::string tool_name = Strip(buffer_.substr(0, cut));
      if (tool_name.empty() && cut == endp) {
        buffer_ = buffer_.substr(endp + end.size());
        FinishToolCall();
        RevertLastToolCallState();
        continue;
      }
      if (cut == nl) {
        buffer_ = buffer_.substr(nl + 1);
      } else {
        buffer_ = buffer_.substr(cut);
      }
      current_tool_name_ = tool_name;
      current_tool_name_sent = true;
      UpdateToolName(pending, tool_name);
      continue;
    }

    // poolside_v1_tool_parser.py:323-352 - incremental string value streaming.
    if (streaming_string_value_) {
      const std::size_t val_end = buffer_.find(av_end);
      if (val_end != std::string::npos) {
        const std::string raw_content = buffer_.substr(0, val_end);
        buffer_ = buffer_.substr(val_end + av_end.size());
        streaming_string_value_ = false;
        pending_key_ = std::nullopt;
        const std::string frag = JsonEscape(raw_content) + "\"";
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] += frag;
        UpdateToolArgs(pending, frag);
        continue;
      }
      std::size_t safe_len = buffer_.size();
      for (std::size_t i = 1; i < av_end.size(); ++i) {
        if (EndsWith(buffer_, av_end.substr(0, i))) {
          safe_len = buffer_.size() - i;
          break;
        }
      }
      if (safe_len > 0) {
        const std::string to_emit = buffer_.substr(0, safe_len);
        buffer_ = buffer_.substr(safe_len);
        const std::string escaped = JsonEscape(to_emit);
        if (!escaped.empty()) {
          streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] += escaped;
          UpdateToolArgs(pending, escaped);
        }
      }
      break;
    }

    // poolside_v1_tool_parser.py:355-402 - a pending key awaits its value.
    if (pending_key_.has_value()) {
      const std::size_t val_pos = buffer_.find(av_start);
      if (val_pos == std::string::npos) break;
      if (val_pos > 0) buffer_ = buffer_.substr(val_pos);

      const std::string key = Strip(pending_key_.value_or(""));
      const bool is_string =
          IsStringType(current_tool_name_.value_or(""), key, request);

      if (is_string) {
        buffer_ = buffer_.substr(av_start.size());
        std::set<std::string>& seen =
            seen_keys_[static_cast<std::size_t>(current_tool_id)];
        if (seen.count(key)) {
          pending_key_ = std::nullopt;
          continue;
        }
        seen.insert(key);
        const std::string key_json = ojson(key).dump();
        std::string frag;
        const auto ci = static_cast<std::size_t>(current_tool_id);
        if (!args_started_[ci]) {
          frag = "{" + key_json + ": \"";
          args_started_[ci] = true;
        } else {
          frag = ", " + key_json + ": \"";
        }
        streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)] += frag;
        streaming_string_value_ = true;
        UpdateToolArgs(pending, frag);
        continue;
      }

      // Non-string: wait for the complete value.
      const std::size_t val_end = buffer_.find(av_end);
      if (val_end == std::string::npos) break;
      const std::string raw_val =
          Strip(buffer_.substr(av_start.size(), val_end - av_start.size()));
      buffer_ = buffer_.substr(val_end + av_end.size());
      pending_key_ = std::nullopt;
      const std::optional<std::string> frag = AppendArgFragment(key, raw_val);
      if (frag.has_value()) UpdateToolArgs(pending, *frag);
      continue;
    }

    // poolside_v1_tool_parser.py:404-442 - parse the next arg key, or close.
    const std::size_t end_pos = buffer_.find(end);
    const std::size_t key_pos = buffer_.find(ak_start);
    if (end_pos != std::string::npos &&
        (key_pos == std::string::npos || end_pos < key_pos)) {
      buffer_ = buffer_.substr(end_pos + end.size());
      const std::optional<std::string> frag = CloseArgsIfNeeded();
      if (current_tool_name_.has_value()) {
        try {
          const nlohmann::json args_dict = nlohmann::json::parse(
              streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)]);
          prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
              nlohmann::json{{"name", *current_tool_name_}, {"arguments", args_dict}};
        } catch (const std::exception&) {
          // leave snapshot as-is on partial/invalid JSON.
        }
      }
      FinishToolCall();
      if (frag.has_value()) UpdateToolArgs(pending, *frag);
      continue;
    }

    if (key_pos == std::string::npos) break;
    if (key_pos > 0) buffer_ = buffer_.substr(key_pos);
    const std::size_t key_end = buffer_.find(ak_end);
    if (key_end == std::string::npos) break;
    const std::string key = buffer_.substr(ak_start.size(), key_end - ak_start.size());
    buffer_ = buffer_.substr(key_end + ak_end.size());
    pending_key_ = key;
    continue;
  }

  std::vector<DeltaToolCall> tool_calls;
  for (auto& kv : pending) tool_calls.push_back(std::move(kv.second));

  if (!content.has_value() && tool_calls.empty()) {
    // poolside_v1_tool_parser.py:445-452 - DEVIATION: Responses
    // is_include_output_logprobs() -> ChatCompletionRequest.logprobs.
    if (request.logprobs) {
      DeltaMessage m;
      m.content = "";
      return m;
    }
    return std::nullopt;
  }
  DeltaMessage m;
  m.content = content;
  m.tool_calls = std::move(tool_calls);
  return m;
}

}  // namespace vllm::entrypoints::openai
