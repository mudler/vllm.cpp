// REIMPLEMENTED-FROM-WIRE-FORMAT. See qwen3_coder.h for the grammar source
// (vllm/parser/qwen3.py @ e24d1b24), the typed-value contract and every knowing
// deviation. This is NOT a line port of the upstream token-id parser engine.
#include "vllm/entrypoints/openai/tool_parsers/qwen3_coder.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

constexpr const char* kToolCallStart = "<tool_call>";
constexpr const char* kFunctionStart = "<function=";
constexpr const char* kFunctionEnd = "</function>";
constexpr const char* kParameterStart = "<parameter=";
constexpr const char* kParameterEnd = "</parameter>";
constexpr std::size_t kFunctionStartLen = 10;   // len("<function=")
constexpr std::size_t kParameterStartLen = 11;  // len("<parameter=")
constexpr std::size_t kParameterEndLen = 12;    // len("</parameter>")

bool IsSpaceCh(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

// qwen3.py _qwen3_arg_converter uses `value.strip()`.
std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsSpaceCh(s[b])) ++b;
  while (e > b && IsSpaceCh(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// The earliest index of a tool-call marker (<tool_call> or <function=), or npos.
std::size_t FindFirstMarker(const std::string& text) {
  const std::size_t a = text.find(kToolCallStart);
  const std::size_t b = text.find(kFunctionStart);
  if (a == std::string::npos) return b;
  if (b == std::string::npos) return a;
  return std::min(a, b);
}

// Whether `s` is a (possibly empty) prefix of one of the two begin markers -
// used to hold back a partial marker at the tail of the streamed content.
bool IsMarkerPrefix(const std::string& s) {
  const std::string tc = kToolCallStart;
  const std::string fn = kFunctionStart;
  return tc.rfind(s, 0) == 0 || fn.rfind(s, 0) == 0;
}

// {content_end, marker_found}: how much of `text` is safe to emit as leading
// content, and whether a complete begin marker has appeared. When no complete
// marker is present we hold back the longest tail that could still grow into one.
std::pair<std::size_t, bool> ContentBoundary(const std::string& text) {
  const std::size_t first = FindFirstMarker(text);
  if (first != std::string::npos) return {first, true};
  const std::size_t max_hold = std::min<std::size_t>(text.size(), 11);
  for (std::size_t k = max_hold; k >= 1; --k) {
    if (IsMarkerPrefix(text.substr(text.size() - k))) {
      return {text.size() - k, false};
    }
  }
  return {text.size(), false};
}

// ── typed-value coercion (qwen3_coder.h "TYPED VALUES") ─────────────────────
bool IsStringType(const std::string& t) {
  return t == "string" || t == "str" || t == "text" || t == "varchar" ||
         t == "char" || t == "enum";
}

// Resolve a JSON-schema fragment to its scalar/container type, understanding the
// Pydantic-v2 nullable shapes: a `type` array (["integer","null"]) and
// `anyOf`/`oneOf` unions both resolve to their FIRST non-"null" member type.
std::string ResolveType(const nlohmann::json& schema) {
  if (!schema.is_object()) return "";
  const auto type_it = schema.find("type");
  if (type_it != schema.end()) {
    if (type_it->is_string()) return type_it->get<std::string>();
    if (type_it->is_array()) {
      for (const auto& e : *type_it) {
        if (e.is_string() && e.get<std::string>() != "null") {
          return e.get<std::string>();
        }
      }
      return "";
    }
  }
  for (const char* key : {"anyOf", "oneOf"}) {
    const auto it = schema.find(key);
    if (it != schema.end() && it->is_array()) {
      for (const auto& member : *it) {
        const std::string mt = ResolveType(member);
        if (!mt.empty() && mt != "null") return mt;
      }
    }
  }
  return "";
}

// The schema type of `param` under `func_name` in the request tools, or "" when
// the tools / schema do not describe it (then the value stays a raw string,
// mirroring the untyped upstream converter output for a tools-less request).
std::string ParamType(const std::vector<ChatCompletionToolsParam>* tools,
                      const std::string& func_name,
                      const std::string& param) {
  if (tools == nullptr) return "";
  for (const ChatCompletionToolsParam& tool : *tools) {
    if (tool.type != "function" || tool.function.name != func_name) continue;
    if (!tool.function.parameters.has_value()) return "";
    const nlohmann::json& params = *tool.function.parameters;
    if (!params.is_object()) return "";
    const auto props = params.find("properties");
    if (props != params.end() && props->is_object()) {
      const auto p = props->find(param);
      if (p != props->end() && p->is_object()) return ResolveType(*p);
      return "";
    }
    const auto p = params.find(param);
    if (p != params.end() && p->is_object()) return ResolveType(*p);
    return "";
  }
  return "";
}

// Coerce the stripped raw value by its resolved schema type. Strings are kept
// raw (never re-serialized); every non-string type is JSON-parsed, degrading to
// the raw string on a parse failure.
ojson CoerceValue(const std::string& value, const std::string& type) {
  if (type.empty() || IsStringType(type)) return ojson(value);
  ojson parsed;
  try {
    parsed = ojson::parse(value);
  } catch (const std::exception&) {
    return ojson(value);
  }
  return parsed;
}

// ── the shared XML surface parse ────────────────────────────────────────────
struct ParsedCall {
  std::string name;
  bool closed = false;
  std::vector<std::pair<std::string, ojson>> params;
};

// Parse every tool call from `text` starting at `region_start` (the first begin
// marker). Only calls whose <function=NAME> name resolves are returned; a
// malformed name open (a '<' before its '>') is dropped, and a call whose name
// is not yet closed stops the scan (streaming: wait for more text).
std::vector<ParsedCall> ParseCalls(
    const std::string& text, std::size_t region_start,
    const std::vector<ChatCompletionToolsParam>* tools) {
  std::vector<ParsedCall> calls;
  std::size_t pos = region_start;
  while (true) {
    const std::size_t fpos = text.find(kFunctionStart, pos);
    if (fpos == std::string::npos) break;
    const std::size_t next_f = text.find(kFunctionStart, fpos + kFunctionStartLen);
    const std::size_t region_end =
        (next_f == std::string::npos) ? text.size() : next_f;

    const std::size_t name_start = fpos + kFunctionStartLen;
    const std::size_t gt = text.find('>', name_start);
    const std::size_t lt = text.find('<', name_start);
    if (gt == std::string::npos || gt >= region_end) {
      break;  // name not closed yet (always the last call) -> wait.
    }
    if (lt != std::string::npos && lt < gt) {
      pos = region_end;  // malformed <function=NAME (no '>') -> drop, skip on.
      continue;
    }
    const std::string name = Strip(text.substr(name_start, gt - name_start));
    if (name.empty()) {
      pos = region_end;
      continue;
    }

    ParsedCall call;
    call.name = name;
    const std::size_t fend = text.find(kFunctionEnd, gt);
    call.closed = (fend != std::string::npos && fend < region_end);
    const std::size_t body_end = call.closed ? fend : region_end;

    std::size_t ppos = gt + 1;
    while (true) {
      const std::size_t pstart = text.find(kParameterStart, ppos);
      if (pstart == std::string::npos || pstart >= body_end) break;
      const std::size_t key_start = pstart + kParameterStartLen;
      const std::size_t pgt = text.find('>', key_start);
      const std::size_t plt = text.find('<', key_start);
      if (pgt == std::string::npos || pgt >= body_end) break;  // tag incomplete.
      if (plt != std::string::npos && plt < pgt) break;  // malformed param tag.
      const std::string key = Strip(text.substr(key_start, pgt - key_start));
      const std::size_t val_start = pgt + 1;

      // Value terminator: earliest of </parameter>, the next <parameter= (the
      // missing-</parameter> lookahead), or </function> (a trailing value).
      const std::size_t e_pe = text.find(kParameterEnd, val_start);
      const std::size_t e_np = text.find(kParameterStart, val_start);
      std::size_t best = std::string::npos;
      int kind = 0;  // 1 = </parameter>, 2 = next <parameter=, 3 = </function>.
      auto consider = [&](std::size_t p, int k) {
        if (p != std::string::npos && p < body_end &&
            (best == std::string::npos || p < best)) {
          best = p;
          kind = k;
        }
      };
      consider(e_pe, 1);
      consider(e_np, 2);
      if (best == std::string::npos && call.closed) {
        best = fend;  // FUNC_END is only the terminator for the last param.
        kind = 3;
      }
      if (best == std::string::npos) break;  // value still streaming -> wait.

      const std::string raw = text.substr(val_start, best - val_start);
      call.params.emplace_back(key, CoerceValue(Strip(raw),
                                                ParamType(tools, name, key)));
      if (kind == 1) {
        ppos = best + kParameterEndLen;
      } else if (kind == 2) {
        ppos = best;  // re-enter on the next <parameter=.
      } else {
        break;  // </function> reached: the call body is done.
      }
    }
    calls.push_back(std::move(call));
    pos = region_end;
  }
  return calls;
}

// The arguments JSON for a call, built so that successive streaming snapshots are
// PREFIX-EXTENSIONS: `{"k": v`, then `, "k2": v2`, then the closing `}` once the
// function closes. A header-only snapshot (no complete params, not closed) is the
// empty string.
std::string BuildArgs(const ParsedCall& call) {
  if (call.params.empty() && !call.closed) return "";
  std::string s = "{";
  for (std::size_t i = 0; i < call.params.size(); ++i) {
    if (i > 0) s += ", ";
    s += ojson(call.params[i].first).dump();
    s += ": ";
    s += call.params[i].second.dump();
  }
  if (call.closed) s += "}";
  return s;
}

}  // namespace

namespace qwen3_coder_detail {

// The per-request streaming state: a diff over the accumulated text.
class Qwen3CoderStreamParser {
 public:
  void reset() {
    content_emitted_ = 0;
    states_.clear();
  }

  void set_tools(const std::vector<ChatCompletionToolsParam>* tools) {
    tools_ = tools;
  }

  std::optional<DeltaMessage> feed(const std::string& current_text) {
    DeltaMessage out;
    bool has_payload = false;

    const auto [content_end, marker_found] = ContentBoundary(current_text);
    if (content_end > content_emitted_) {
      std::string chunk =
          current_text.substr(content_emitted_, content_end - content_emitted_);
      content_emitted_ = content_end;
      if (!chunk.empty()) {
        out.content = std::move(chunk);
        has_payload = true;
      }
    }

    if (marker_found) {
      const std::size_t first = FindFirstMarker(current_text);
      const std::vector<ParsedCall> calls =
          ParseCalls(current_text, first, tools_);
      std::vector<DeltaToolCall> tcs;
      for (std::size_t i = 0; i < calls.size(); ++i) {
        if (i >= states_.size()) {
          TState fresh;
          fresh.id = make_tool_call_id();
          states_.push_back(std::move(fresh));
        }
        TState& st = states_[i];
        DeltaToolCall d;
        d.index = static_cast<int>(i);
        bool changed = false;
        if (!st.name_emitted) {
          d.id = st.id;
          d.type = "function";
          d.function.name = calls[i].name;
          st.name_emitted = true;
          changed = true;
        }
        const std::string target = BuildArgs(calls[i]);
        if (target.size() > st.args_committed.size()) {
          d.function.arguments = target.substr(st.args_committed.size());
          st.args_committed = target;
          changed = true;
        }
        if (changed) tcs.push_back(std::move(d));
      }
      if (!tcs.empty()) {
        out.tool_calls = std::move(tcs);
        has_payload = true;
      }
    }

    if (!has_payload) return std::nullopt;
    return out;
  }

 private:
  struct TState {
    std::string id;
    bool name_emitted = false;
    std::string args_committed;
  };
  const std::vector<ChatCompletionToolsParam>* tools_ = nullptr;
  std::size_t content_emitted_ = 0;
  std::vector<TState> states_;
};

}  // namespace qwen3_coder_detail

Qwen3CoderToolParser::Qwen3CoderToolParser()
    : parser_(std::make_unique<qwen3_coder_detail::Qwen3CoderStreamParser>()) {}

Qwen3CoderToolParser::~Qwen3CoderToolParser() = default;

ExtractedToolCallInformation Qwen3CoderToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  const std::vector<ChatCompletionToolsParam>* tools =
      request.tools.has_value() ? &*request.tools : nullptr;

  const std::size_t first = FindFirstMarker(model_output);
  if (first == std::string::npos) {
    // No tool-call surface: the whole output is content.
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  const std::vector<ParsedCall> calls = ParseCalls(model_output, first, tools);

  std::vector<ToolCall> tool_calls;
  for (const ParsedCall& call : calls) {
    ojson args = ojson::object();
    for (const auto& [key, value] : call.params) args[key] = value;
    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = call.name;
    tc.function.arguments = args.dump();
    tool_calls.push_back(std::move(tc));
  }

  ExtractedToolCallInformation info;
  info.tools_called = !tool_calls.empty();
  info.tool_calls = std::move(tool_calls);
  const std::string content = model_output.substr(0, first);
  if (!content.empty()) info.content = content;
  return info;
}

std::optional<DeltaMessage> Qwen3CoderToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& current_text,
    const std::string& /*delta_text*/, const ChatCompletionRequest& request) {
  if (previous_text.empty()) parser_->reset();
  parser_->set_tools(request.tools.has_value() ? &*request.tools : nullptr);
  return parser_->feed(current_text);
}

}  // namespace vllm::entrypoints::openai
