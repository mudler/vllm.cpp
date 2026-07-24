// Reimplemented from the WIRE FORMAT of glm47_moe_tool_parser.py @ e24d1b24.
// See glm47.h for the wire format, the argument-typing note and the deviations.
#include "vllm/entrypoints/openai/tool_parsers/glm47.h"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/utils.h"

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

}  // namespace

// ── Non-streaming ───────────────────────────────────────────────────────────

ExtractedToolCallInformation Glm47ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  const std::string start = kToolCallStart;
  const std::string end = kToolCallEnd;
  const std::string ak_start = kArgKeyStart;
  const std::string ak_end = kArgKeyEnd;
  const std::string av_start = kArgValStart;
  const std::string av_end = kArgValEnd;

  // No tool call at all: the whole output is content (glm47 test_no_tool_call).
  if (model_output.find(start) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  std::vector<ToolCall> tool_calls;
  std::size_t pos = 0;
  while (true) {
    const std::size_t s = model_output.find(start, pos);
    if (s == std::string::npos) break;
    const std::size_t inner_start = s + start.size();
    const std::size_t e = model_output.find(end, inner_start);
    if (e == std::string::npos) break;  // only fully-closed blocks
    const std::string inner = model_output.substr(inner_start, e - inner_start);
    pos = e + end.size();

    // NAME: from block start up to the first <arg_key> or end-of-block.
    const std::size_t akp = inner.find(ak_start);
    const std::string name = Strip(
        akp == std::string::npos ? inner : inner.substr(0, akp));

    const nlohmann::json properties =
        find_tool_properties(request.tools, name);

    // Arguments: <arg_key>K</arg_key>\s*<arg_value>V</arg_value> pairs.
    ojson args = ojson::object();
    std::size_t p = 0;
    while (true) {
      const std::size_t k_open = inner.find(ak_start, p);
      if (k_open == std::string::npos) break;
      const std::size_t k_content = k_open + ak_start.size();
      const std::size_t k_close = inner.find(ak_end, k_content);
      if (k_close == std::string::npos) break;
      const std::size_t gap_start = k_close + ak_end.size();
      const std::size_t v_open = inner.find(av_start, gap_start);
      if (v_open == std::string::npos) break;
      if (!Strip(inner.substr(gap_start, v_open - gap_start)).empty()) {
        p = gap_start;  // non-ws between key and value: unpaired key, skip.
        continue;
      }
      const std::size_t v_content = v_open + av_start.size();
      const std::size_t v_close = inner.find(av_end, v_content);
      if (v_close == std::string::npos) break;
      const std::string key = Strip(inner.substr(k_content, k_close - k_content));
      const std::string value = inner.substr(v_content, v_close - v_content);
      // _glm47_arg_converter keeps only the FIRST occurrence semantics via a
      // dict; a repeated key overwrites (dict assignment). Mirror with []=.
      args[key] = coerce_arg_value(value, properties, key);
      p = v_close + av_end.size();
    }

    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = name;
    tc.function.arguments = args.dump();
    tool_calls.push_back(std::move(tc));
  }

  ExtractedToolCallInformation info;
  info.tools_called = !tool_calls.empty();
  info.tool_calls = std::move(tool_calls);
  // Content = text before the first <tool_call>; .strip()-ed when a tool was
  // called and dropped to None when empty (engine strip_content_whitespace_
  // with_tools default).
  const std::size_t s0 = model_output.find(start);
  std::string content = s0 != std::string::npos ? model_output.substr(0, s0)
                                                 : model_output;
  if (info.tools_called) {
    content = Strip(content);
    if (!content.empty()) info.content = content;
  } else if (!content.empty()) {
    info.content = content;
  }
  return info;
}

// ── Streaming ───────────────────────────────────────────────────────────────

void Glm47ToolParser::EnsureToolState() {
  const std::size_t need = static_cast<std::size_t>(current_tool_id) + 1;
  while (tool_call_ids_.size() < need) tool_call_ids_.push_back(make_tool_call_id());
  while (streamed_args_for_tool.size() < need) streamed_args_for_tool.emplace_back();
  while (prev_tool_call_arr.size() < need)
    prev_tool_call_arr.push_back(nlohmann::json::object());
  while (args_started_.size() < need) args_started_.push_back(false);
  while (args_closed_.size() < need) args_closed_.push_back(false);
  while (seen_keys_.size() < need) seen_keys_.emplace_back();
}

void Glm47ToolParser::BeginToolCall() {
  current_tool_id = current_tool_id == -1 ? 0 : current_tool_id + 1;
  EnsureToolState();
  current_tool_name_sent = false;
  current_tool_name_ = std::nullopt;
  pending_key_ = std::nullopt;
  in_tool_call_ = true;
}

void Glm47ToolParser::FinishToolCall() {
  in_tool_call_ = false;
  current_tool_name_ = std::nullopt;
  pending_key_ = std::nullopt;
}

void Glm47ToolParser::RevertLastToolCallState() {
  if (current_tool_id < 0) return;
  tool_call_ids_.pop_back();
  streamed_args_for_tool.pop_back();
  prev_tool_call_arr.pop_back();
  args_started_.pop_back();
  args_closed_.pop_back();
  seen_keys_.pop_back();
  current_tool_id -= 1;
}

DeltaToolCall& Glm47ToolParser::GetOrCreateDelta(
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

void Glm47ToolParser::UpdateToolName(std::map<int, DeltaToolCall>& pending,
                                     const std::string& tool_name) {
  prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
      nlohmann::json{{"name", tool_name}, {"arguments", nlohmann::json::object()}};
  DeltaToolCall& delta = GetOrCreateDelta(pending);
  delta.id = tool_call_ids_[static_cast<std::size_t>(current_tool_id)];
  delta.type = "function";
  delta.function.name = tool_name;
  if (!delta.function.arguments.has_value()) delta.function.arguments = "";
}

std::optional<std::string> Glm47ToolParser::AppendArgFragment(
    const std::string& key_in, const std::string& raw_value,
    const ChatCompletionRequest& request) {
  const std::string key = Strip(key_in);
  if (key.empty()) return std::nullopt;
  std::set<std::string>& seen = seen_keys_[static_cast<std::size_t>(current_tool_id)];
  if (seen.count(key)) return std::nullopt;

  const nlohmann::json properties =
      find_tool_properties(request.tools, current_tool_name_.value_or(""));
  const ojson val = coerce_arg_value(raw_value, properties, key);
  const std::string key_json = ojson(key).dump();
  const std::string val_json = val.dump();

  const auto ci = static_cast<std::size_t>(current_tool_id);
  std::string frag;
  if (!args_started_[ci]) {
    frag = "{" + key_json + ": " + val_json;
    args_started_[ci] = true;
  } else {
    frag = ", " + key_json + ": " + val_json;
  }
  seen.insert(key);
  streamed_args_for_tool[ci] += frag;
  return frag;
}

std::optional<std::string> Glm47ToolParser::CloseArgsIfNeeded() {
  const auto ci = static_cast<std::size_t>(current_tool_id);
  if (args_closed_[ci]) return std::nullopt;
  args_closed_[ci] = true;
  std::string frag;
  if (!args_started_[ci]) {
    frag = "{}";
    streamed_args_for_tool[ci] = frag;
  } else {
    frag = "}";
    streamed_args_for_tool[ci] += frag;
  }
  return frag;
}

std::optional<DeltaMessage> Glm47ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& request) {
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
        // Hold a trailing prefix of the start token (it may complete next tick);
        // everything else is plain content.
        bool matched = false;
        for (std::size_t i = start.size() - 1; i >= 1; --i) {
          if (EndsWith(buffer_, start.substr(0, i))) {
            const std::string out = buffer_.substr(0, buffer_.size() - i);
            buffer_ = buffer_.substr(buffer_.size() - i);
            if (!out.empty()) content = content.value_or("") + out;
            matched = true;
            break;
          }
        }
        if (!matched) {
          if (!buffer_.empty()) content = content.value_or("") + buffer_;
          buffer_.clear();
        }
        break;
      }
      if (start_idx > 0) {
        content = content.value_or("") + buffer_.substr(0, start_idx);
      }
      buffer_ = buffer_.substr(start_idx + start.size());
      BeginToolCall();
      continue;
    }

    // Tool NAME: terminates at the first <arg_key> or </tool_call>.
    if (!current_tool_name_sent) {
      const std::size_t akp = buffer_.find(ak_start);
      const std::size_t endp = buffer_.find(end);
      std::size_t cut = std::string::npos;
      if (akp != std::string::npos) cut = akp;
      if (endp != std::string::npos && (cut == std::string::npos || endp < cut))
        cut = endp;
      if (cut == std::string::npos) break;  // wait for a terminator
      const std::string name = Strip(buffer_.substr(0, cut));
      if (name.empty() && cut == endp) {
        // Empty <tool_call></tool_call>: no call. Unwind the reserved slot.
        buffer_ = buffer_.substr(endp + end.size());
        FinishToolCall();
        RevertLastToolCallState();
        continue;
      }
      buffer_ = buffer_.substr(cut);  // keep the terminator for the arg loop.
      current_tool_name_ = name;
      current_tool_name_sent = true;
      UpdateToolName(pending, name);
      continue;
    }

    // Close the call, or parse the next complete <arg_key>/<arg_value> pair.
    const std::size_t end_pos = buffer_.find(end);
    const std::size_t key_pos = buffer_.find(ak_start);
    if (end_pos != std::string::npos &&
        (key_pos == std::string::npos || end_pos < key_pos)) {
      buffer_ = buffer_.substr(end_pos + end.size());
      const std::optional<std::string> frag = CloseArgsIfNeeded();
      if (current_tool_name_.has_value()) {
        try {
          prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
              nlohmann::json{{"name", *current_tool_name_},
                             {"arguments", nlohmann::json::parse(
                                  streamed_args_for_tool[static_cast<std::size_t>(
                                      current_tool_id)])}};
        } catch (const std::exception&) {
          // leave the snapshot as-is on partial/invalid JSON.
        }
      }
      FinishToolCall();
      if (frag.has_value()) {
        DeltaToolCall& delta = GetOrCreateDelta(pending);
        if (!delta.function.arguments.has_value()) delta.function.arguments = "";
        *delta.function.arguments += *frag;
      }
      continue;
    }

    if (key_pos == std::string::npos) break;  // wait for <arg_key> or </tool_call>
    const std::size_t k_content = key_pos + ak_start.size();
    const std::size_t k_close = buffer_.find(ak_end, k_content);
    if (k_close == std::string::npos) break;
    const std::size_t gap_start = k_close + ak_end.size();
    const std::size_t v_open = buffer_.find(av_start, gap_start);
    if (v_open == std::string::npos) break;
    if (!Strip(buffer_.substr(gap_start, v_open - gap_start)).empty()) {
      buffer_ = buffer_.substr(gap_start);  // unpaired key, drop it.
      continue;
    }
    const std::size_t v_content = v_open + av_start.size();
    const std::size_t v_close = buffer_.find(av_end, v_content);
    if (v_close == std::string::npos) break;  // value still streaming.
    const std::string key = buffer_.substr(k_content, k_close - k_content);
    const std::string value = buffer_.substr(v_content, v_close - v_content);
    buffer_ = buffer_.substr(v_close + av_end.size());
    const std::optional<std::string> frag = AppendArgFragment(key, value, request);
    if (frag.has_value()) {
      DeltaToolCall& delta = GetOrCreateDelta(pending);
      if (!delta.function.arguments.has_value()) delta.function.arguments = "";
      *delta.function.arguments += *frag;
    }
    continue;
  }

  std::vector<DeltaToolCall> tool_calls;
  for (auto& kv : pending) tool_calls.push_back(std::move(kv.second));

  if (!content.has_value() && tool_calls.empty()) return std::nullopt;
  DeltaMessage m;
  m.content = content;
  if (!tool_calls.empty()) m.tool_calls = std::move(tool_calls);
  return m;
}

}  // namespace vllm::entrypoints::openai
