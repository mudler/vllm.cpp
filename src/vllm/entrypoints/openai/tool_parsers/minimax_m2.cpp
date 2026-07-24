// Reimplemented from the WIRE FORMAT of minimax_m2_tool_parser.py @ e24d1b24.
// See minimax_m2.h for the wire format, the argument-typing note and deviations.
#include "vllm/entrypoints/openai/tool_parsers/minimax_m2.h"

#include <cstddef>
#include <initializer_list>
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

// Earliest non-npos position among the candidates.
std::size_t Earliest(std::initializer_list<std::size_t> xs) {
  std::size_t best = std::string::npos;
  for (std::size_t x : xs) {
    if (x != std::string::npos && (best == std::string::npos || x < best)) best = x;
  }
  return best;
}

struct NameMatch {
  std::string name;   // stripped
  std::size_t end;    // index just past the closing '>'
};

// _PARAM_RE / INVOKE name attribute: given `text` and `p` pointing at the first
// char AFTER `name=`, match ("[^"]*"|'[^']*'|[^>\s]+) then \s* then '>'. Returns
// nullopt when the attribute has not fully arrived yet (streaming) or is
// malformed (no closing '>').
std::optional<NameMatch> ParseNameAttr(const std::string& text, std::size_t p) {
  if (p >= text.size()) return std::nullopt;
  const char c = text[p];
  std::string name;
  std::size_t after_name = std::string::npos;
  if (c == '"' || c == '\'') {
    const std::size_t q = text.find(c, p + 1);
    if (q == std::string::npos) return std::nullopt;
    name = text.substr(p + 1, q - (p + 1));
    after_name = q + 1;
  } else {
    std::size_t e = p;
    while (e < text.size() && text[e] != '>' && !IsWs(text[e])) ++e;
    if (e == p) return std::nullopt;      // [^>\s]+ needs at least one char
    if (e >= text.size()) return std::nullopt;  // '>' / ws not arrived yet
    name = text.substr(p, e - p);
    after_name = e;
  }
  std::size_t g = after_name;
  while (g < text.size() && IsWs(text[g])) ++g;
  if (g >= text.size()) return std::nullopt;  // '>' not arrived yet
  if (text[g] != '>') return std::nullopt;    // malformed close
  return NameMatch{Strip(name), g + 1};
}

}  // namespace

// ── Non-streaming ───────────────────────────────────────────────────────────

ExtractedToolCallInformation MinimaxM2ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  const std::string start = kToolCallStart;
  const std::string inv_prefix = kInvokePrefix;
  const std::string inv_end = kInvokeEnd;
  const std::string par_prefix = kParamPrefix;
  const std::string par_end = kParamEnd;

  const std::size_t s0 = model_output.find(start);
  if (s0 == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  std::vector<ToolCall> tool_calls;
  std::size_t pos = s0 + start.size();
  while (true) {
    const std::size_t inv = model_output.find(inv_prefix, pos);
    if (inv == std::string::npos) break;
    const std::optional<NameMatch> nm =
        ParseNameAttr(model_output, inv + inv_prefix.size());
    if (!nm.has_value()) break;
    const std::size_t ie = model_output.find(inv_end, nm->end);
    if (ie == std::string::npos) break;
    const std::string args_text = model_output.substr(nm->end, ie - nm->end);
    pos = ie + inv_end.size();

    const nlohmann::json properties =
        find_tool_properties(request.tools, nm->name);

    // _PARAM_RE over args_text: <parameter name=K>V(?:</parameter>|(?=<parameter
    // name=)). Value STRIPPED, coerced against the schema.
    ojson args = ojson::object();
    std::size_t p = 0;
    while (true) {
      const std::size_t par = args_text.find(par_prefix, p);
      if (par == std::string::npos) break;
      const std::optional<NameMatch> pn =
          ParseNameAttr(args_text, par + par_prefix.size());
      if (!pn.has_value()) break;
      const std::size_t val_start = pn->end;
      const std::size_t close = args_text.find(par_end, val_start);
      const std::size_t next = args_text.find(par_prefix, val_start);
      const std::size_t term = Earliest({close, next});
      std::size_t val_end;
      std::size_t advance;
      if (term == std::string::npos) {
        // partial tail (no close, no next): whole remainder is the value.
        val_end = args_text.size();
        advance = args_text.size();
      } else if (term == close) {
        val_end = close;
        advance = close + par_end.size();
      } else {
        val_end = next;
        advance = next;  // leave the next <parameter for the following iteration
      }
      const std::string value =
          Strip(args_text.substr(val_start, val_end - val_start));
      if (!pn->name.empty()) {
        args[pn->name] = coerce_arg_value(value, properties, pn->name);
      }
      p = advance;
    }

    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = nm->name;
    tc.function.arguments = args.dump();
    tool_calls.push_back(std::move(tc));
  }

  ExtractedToolCallInformation info;
  info.tools_called = !tool_calls.empty();
  info.tool_calls = std::move(tool_calls);
  std::string content = model_output.substr(0, s0);
  if (info.tools_called) {
    content = Strip(content);
    if (!content.empty()) info.content = content;
  } else if (!content.empty()) {
    info.content = content;
  }
  return info;
}

// ── Streaming ───────────────────────────────────────────────────────────────

void MinimaxM2ToolParser::EnsureToolState() {
  const std::size_t need = static_cast<std::size_t>(current_tool_id) + 1;
  while (tool_call_ids_.size() < need) tool_call_ids_.push_back(make_tool_call_id());
  while (streamed_args_for_tool.size() < need) streamed_args_for_tool.emplace_back();
  while (prev_tool_call_arr.size() < need)
    prev_tool_call_arr.push_back(nlohmann::json::object());
  while (args_started_.size() < need) args_started_.push_back(false);
  while (args_closed_.size() < need) args_closed_.push_back(false);
  while (seen_keys_.size() < need) seen_keys_.emplace_back();
}

void MinimaxM2ToolParser::BeginToolCall() {
  current_tool_id = current_tool_id == -1 ? 0 : current_tool_id + 1;
  EnsureToolState();
  current_tool_name_ = std::nullopt;
}

void MinimaxM2ToolParser::FinishToolCall() { current_tool_name_ = std::nullopt; }

DeltaToolCall& MinimaxM2ToolParser::GetOrCreateDelta(
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

void MinimaxM2ToolParser::UpdateToolName(std::map<int, DeltaToolCall>& pending,
                                         const std::string& tool_name) {
  prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
      nlohmann::json{{"name", tool_name}, {"arguments", nlohmann::json::object()}};
  DeltaToolCall& delta = GetOrCreateDelta(pending);
  delta.id = tool_call_ids_[static_cast<std::size_t>(current_tool_id)];
  delta.type = "function";
  delta.function.name = tool_name;
  if (!delta.function.arguments.has_value()) delta.function.arguments = "";
}

std::optional<std::string> MinimaxM2ToolParser::AppendArgFragment(
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

std::optional<std::string> MinimaxM2ToolParser::CloseArgsIfNeeded() {
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

void MinimaxM2ToolParser::AppendArgsDelta(std::map<int, DeltaToolCall>& pending,
                                          const std::string& frag) {
  DeltaToolCall& delta = GetOrCreateDelta(pending);
  if (!delta.function.arguments.has_value()) delta.function.arguments = "";
  *delta.function.arguments += frag;
}

std::optional<DeltaMessage> MinimaxM2ToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  buffer_ += delta_text;

  std::map<int, DeltaToolCall> pending;
  std::optional<std::string> content;

  const std::string start = kToolCallStart;
  const std::string end = kToolCallEnd;
  const std::string inv_prefix = kInvokePrefix;
  const std::string inv_end = kInvokeEnd;
  const std::string par_prefix = kParamPrefix;
  const std::string par_end = kParamEnd;

  while (true) {
    if (region_ == Region::kContent) {
      const std::size_t start_idx = buffer_.find(start);
      if (start_idx == std::string::npos) {
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
      region_ = Region::kWrapper;
      continue;
    }

    if (region_ == Region::kWrapper) {
      const std::size_t inv = buffer_.find(inv_prefix);
      const std::size_t endw = buffer_.find(end);
      if (endw != std::string::npos && (inv == std::string::npos || endw < inv)) {
        buffer_ = buffer_.substr(endw + end.size());
        region_ = Region::kContent;
        continue;
      }
      if (inv == std::string::npos) break;  // wait for <invoke or </minimax:tool_call>
      const std::optional<NameMatch> nm =
          ParseNameAttr(buffer_, inv + inv_prefix.size());
      if (!nm.has_value()) break;  // name attribute still arriving
      BeginToolCall();
      current_tool_name_ = nm->name;
      buffer_ = buffer_.substr(nm->end);
      UpdateToolName(pending, nm->name);
      region_ = Region::kArgs;
      continue;
    }

    // region_ == kArgs
    const std::size_t par = buffer_.find(par_prefix);
    const std::size_t invend = buffer_.find(inv_end);
    if (invend != std::string::npos && (par == std::string::npos || invend < par)) {
      buffer_ = buffer_.substr(invend + inv_end.size());
      const std::optional<std::string> frag = CloseArgsIfNeeded();
      if (current_tool_name_.has_value()) {
        try {
          prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)] =
              nlohmann::json{{"name", *current_tool_name_},
                             {"arguments", nlohmann::json::parse(
                                  streamed_args_for_tool[static_cast<std::size_t>(
                                      current_tool_id)])}};
        } catch (const std::exception&) {
          // leave snapshot as-is.
        }
      }
      FinishToolCall();
      region_ = Region::kWrapper;
      if (frag.has_value()) AppendArgsDelta(pending, *frag);
      continue;
    }
    if (par == std::string::npos) break;  // wait for <parameter or </invoke>

    const std::optional<NameMatch> pn =
        ParseNameAttr(buffer_, par + par_prefix.size());
    if (!pn.has_value()) break;  // parameter name still arriving
    const std::size_t val_start = pn->end;
    const std::size_t close = buffer_.find(par_end, val_start);
    const std::size_t next = buffer_.find(par_prefix, val_start);
    const std::size_t invend2 = buffer_.find(inv_end, val_start);
    const std::size_t term = Earliest({close, next, invend2});
    if (term == std::string::npos) break;  // value still streaming
    const std::string value = Strip(buffer_.substr(val_start, term - val_start));
    if (term == close) {
      buffer_ = buffer_.substr(close + par_end.size());
    } else {
      buffer_ = buffer_.substr(term);  // leave next <parameter / </invoke>
    }
    const std::optional<std::string> frag =
        AppendArgFragment(pn->name, value, request);
    if (frag.has_value()) AppendArgsDelta(pending, *frag);
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
