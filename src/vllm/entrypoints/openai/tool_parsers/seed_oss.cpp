// REIMPLEMENTED FROM WIRE FORMAT. See seed_oss.h for the format spec + deviations.
#include "vllm/entrypoints/openai/tool_parsers/seed_oss.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

const std::string kTcStart = SeedOssToolParser::kToolCallStart;   // <seed:tool_call>
const std::string kFunc = SeedOssToolParser::kFuncPrefix;         // <function=
const std::string kFuncEnd = SeedOssToolParser::kFuncEnd;         // </function>
const std::string kParam = SeedOssToolParser::kParamPrefix;       // <parameter=
const std::string kParamEnd = SeedOssToolParser::kParamEnd;       // </parameter>

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

std::size_t MinPos(std::size_t a, std::size_t b) {
  return (a < b) ? a : b;
}

// One <function=NAME>..</function> unit recovered from the text.
struct SeedCall {
  std::string name;
  std::string body;       // raw <parameter=..> body between '>' and </function>
  bool has_name = false;  // NAME is delimited (closing '>' or malformed close)
  bool complete = false;  // </function> seen
};

// qwen3.py:_qwen3_arg_converter - parse the <parameter=KEY>VALUE</parameter>
// body into an object of STRING values. A param is closed by </parameter> OR by
// the lookahead to the next <parameter= (missing </parameter> does not merge two
// params). In partial mode a trailing unterminated <parameter=KEY>VALUE is
// emitted too (_PARTIAL_PARAM_RE); in non-partial mode it is dropped (_PARAM_RE
// requires a close).
ojson ParseParams(const std::string& body, bool partial) {
  ojson params = ojson::object();
  const std::size_t n = body.size();
  std::size_t pos = 0;
  while (pos < n) {
    const std::size_t pp = body.find(kParam, pos);
    if (pp == std::string::npos) break;
    const std::size_t after = pp + kParam.size();
    const std::size_t gt = body.find('>', after);
    if (gt == std::string::npos) break;  // no '>' yet: nothing to bind
    const std::string name = body.substr(after, gt - after);
    const std::size_t val_start = gt + 1;

    const std::size_t pe = body.find(kParamEnd, val_start);
    const std::size_t np = body.find(kParam, val_start);
    if (pe != std::string::npos && (np == std::string::npos || pe <= np)) {
      // Closed by </parameter>.
      params[name] = Strip(body.substr(val_start, pe - val_start));
      pos = pe + kParamEnd.size();
    } else if (np != std::string::npos) {
      // Closed by lookahead to the next <parameter=.
      params[name] = Strip(body.substr(val_start, np - val_start));
      pos = np;
    } else {
      // Unterminated trailing param: only emitted in partial (streaming) mode.
      if (partial) params[name] = Strip(body.substr(val_start));
      break;
    }
  }
  return params;
}

// Extract the sequence of <function=..> units + the leading content. content is
// the text before the first tool marker; a trailing partial start-marker prefix
// is buffered off so a split tag never leaks.
void ScanFunctions(const std::string& text, std::string& content_out,
                   std::vector<SeedCall>& calls_out) {
  content_out.clear();
  calls_out.clear();

  const std::size_t first_tc = text.find(kTcStart);
  const std::size_t first_fn = text.find(kFunc);
  const std::size_t first = MinPos(first_tc, first_fn);

  if (first == std::string::npos) {
    content_out = text;
    // Buffer a trailing partial start-marker prefix (<seed:tool_call> / <function=).
    std::size_t buf = 0;
    for (const std::string* m : {&kTcStart, &kFunc}) {
      for (std::size_t k = m->size() - 1; k >= 1; --k) {
        if (EndsWith(content_out, m->substr(0, k))) {
          if (k > buf) buf = k;
          break;
        }
        if (k == 1) break;
      }
    }
    if (buf > 0) content_out.erase(content_out.size() - buf);
    return;
  }
  content_out = text.substr(0, first);

  // Scan every <function=..> from the first marker onward.
  std::size_t pos = first;
  while (pos < text.size()) {
    const std::size_t fp = text.find(kFunc, pos);
    if (fp == std::string::npos) break;
    const std::size_t after = fp + kFunc.size();
    const std::size_t gt = text.find('>', after);
    const std::size_t fe = text.find(kFuncEnd, after);

    SeedCall call;
    if (gt == std::string::npos || (fe != std::string::npos && fe < gt)) {
      // Malformed header (no closing '>' before </function>): #46314. The name
      // runs to </function>; the call ends there with empty args, and the scan
      // continues so sibling calls are NOT dropped.
      const std::size_t name_end = (fe != std::string::npos) ? fe : text.size();
      call.name = text.substr(after, name_end - after);
      call.has_name = (fe != std::string::npos);
      call.complete = (fe != std::string::npos);
      pos = (fe != std::string::npos) ? fe + kFuncEnd.size() : text.size();
      if (call.has_name) calls_out.push_back(std::move(call));
      continue;
    }

    call.name = text.substr(after, gt - after);
    call.has_name = true;
    const std::size_t body_start = gt + 1;
    const std::size_t fe2 = text.find(kFuncEnd, body_start);
    if (fe2 != std::string::npos) {
      call.body = text.substr(body_start, fe2 - body_start);
      call.complete = true;
      pos = fe2 + kFuncEnd.size();
    } else {
      // In-progress: body runs to the next <function= / </seed:tool_call> / end.
      const std::size_t next_fn = text.find(kFunc, body_start);
      const std::size_t next_tc = text.find(kTcStart, body_start);
      std::size_t body_end = MinPos(next_fn, next_tc);
      if (body_end == std::string::npos) body_end = text.size();
      call.body = text.substr(body_start, body_end - body_start);
      call.complete = false;
      pos = body_end;
    }
    calls_out.push_back(std::move(call));
  }
}

}  // namespace

nlohmann::ordered_json SeedOssToolParser::ParseParameters(const std::string& body,
                                                          bool partial) {
  return ParseParams(body, partial);
}

ExtractedToolCallInformation SeedOssToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // No function marker -> plain content (unmodified).
  if (model_output.find(kFunc) == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
  try {
    std::string content;
    std::vector<SeedCall> calls;
    ScanFunctions(model_output, content, calls);

    std::vector<ToolCall> tool_calls;
    for (const SeedCall& c : calls) {
      if (!c.has_name) continue;
      const ojson params = ParseParams(c.body, /*partial=*/false);
      ToolCall tc;
      tc.id = make_tool_call_id();
      tc.type = "function";
      tc.function.name = c.name;
      tc.function.arguments = params.dump();
      tool_calls.push_back(std::move(tc));
    }

    if (tool_calls.empty()) {
      return ExtractedToolCallInformation{false, {}, model_output};
    }

    const std::string stripped = Strip(content);
    std::optional<std::string> content_opt;
    if (!stripped.empty()) content_opt = stripped;

    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    info.content = content_opt;
    return info;
  } catch (const std::exception&) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }
}

std::optional<DeltaMessage> SeedOssToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& /*delta_text*/, const ChatCompletionRequest& /*request*/) {
  try {
    std::string content;
    std::vector<SeedCall> calls;
    ScanFunctions(current_text, content, calls);

    DeltaMessage msg;
    std::vector<DeltaToolCall> deltas;

    for (std::size_t i = 0; i < calls.size(); ++i) {
      const SeedCall& c = calls[i];
      if (static_cast<int>(i) > current_tool_id_) current_tool_id_ = static_cast<int>(i);
      if (name_sent_.size() <= i) name_sent_.resize(i + 1, false);
      if (args_sent_.size() <= i) args_sent_.resize(i + 1, false);

      if (c.has_name && !name_sent_[i]) {
        name_sent_[i] = true;
        DeltaToolCall tc;
        tc.index = static_cast<int>(i);
        tc.id = make_tool_call_id();
        tc.type = "function";
        tc.function.name = c.name;
        deltas.push_back(std::move(tc));
      }

      // Emit the (string-valued) arguments as one chunk at the function's close.
      if (c.complete && c.has_name && !args_sent_[i]) {
        args_sent_[i] = true;
        const ojson params = ParseParams(c.body, /*partial=*/false);
        DeltaToolCall tc;
        tc.index = static_cast<int>(i);
        tc.function.arguments = params.dump();
        deltas.push_back(std::move(tc));
      }
    }

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
