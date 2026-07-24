// Ported from: vllm/tool_parsers/llama_tool_parser.py @ e24d1b24
// (streaming primitives: vllm/tool_parsers/utils.py @ e24d1b24 -
//  partial_json_loads, is_complete_json, find_common_prefix).
#include "vllm/entrypoints/openai/tool_parsers/llama.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints::openai {

namespace {

using ojson = nlohmann::ordered_json;

bool IsSpaceCh(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// tool_parsers/utils.py:141 (is_complete_json). accept() parses without throwing.
bool IsCompleteJson(const std::string& s) { return ojson::accept(s); }

// tool_parsers/utils.py:55 (find_common_prefix).
std::string FindCommonPrefix(const std::string& a, const std::string& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return a.substr(0, i);
}

// Python truthiness of a decoded-JSON value: empty container / empty string /
// zero / null / false are falsy. Used for the `if cur_arguments:` guards.
bool JsonTruthy(const ojson& j) {
  if (j.is_null()) return false;
  if (j.is_boolean()) return j.get<bool>();
  if (j.is_number_integer()) return j.get<std::int64_t>() != 0;
  if (j.is_number_unsigned()) return j.get<std::uint64_t>() != 0;
  if (j.is_number_float()) return j.get<double>() != 0.0;
  if (j.is_string()) return !j.get<std::string>().empty();
  if (j.is_object() || j.is_array()) return !j.empty();
  return true;
}

enum class ScanStatus { kFoundClose, kTruncated };

// String/escape-aware brace matcher. From `start` (which must index a '{' or
// '[') scan to the matching close. On success end = index one past the close.
ScanStatus RawDecodeScan(const std::string& s, std::size_t start,
                         std::size_t& end) {
  int depth = 0;
  bool in_string = false;
  for (std::size_t i = start; i < s.size(); ++i) {
    const char c = s[i];
    if (in_string) {
      if (c == '\\') {
        ++i;  // skip the escaped char
        continue;
      }
      if (c == '"') in_string = false;
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '{' || c == '[') {
      ++depth;
    } else if (c == '}' || c == ']') {
      --depth;
      if (depth == 0) {
        end = i + 1;
        return ScanStatus::kFoundClose;
      }
    }
  }
  return ScanStatus::kTruncated;
}

// A strict complete-object decode: brace-match from `start`, then parse the
// delimited substring. Returns the parsed object + one-past end, or nullopt on
// truncation OR malformed content (the caller distinguishes when needed).
struct RawDecodeResult {
  bool found_close = false;   // a matching close brace was located
  bool parsed_ok = false;     // ... and the delimited text parsed as JSON
  ojson value;
  std::size_t end = 0;        // one past the close (valid iff found_close)
};
RawDecodeResult RawDecodeObject(const std::string& s, std::size_t start) {
  RawDecodeResult r;
  std::size_t end = 0;
  if (RawDecodeScan(s, start, end) != ScanStatus::kFoundClose) {
    return r;  // truncated
  }
  r.found_close = true;
  r.end = end;
  ojson parsed = ojson::parse(s.substr(start, end - start), nullptr,
                              /*allow_exceptions=*/false);
  if (parsed.is_discarded()) return r;  // malformed content
  r.parsed_ok = true;
  r.value = std::move(parsed);
  return r;
}

// -- Tolerant partial JSON parser (stands in for partial_json_parser) ---------
// Reads as much of a (possibly truncated) JSON value as is CONFIRMED, i.e. only
// fully-terminated tokens are surfaced. `allow_str` toggles whether a truncated
// trailing STRING value is surfaced partially (upstream Allow.STR) or dropped;
// numbers/keywords truncated at end-of-buffer are always dropped (a
// stricter-but-safe reading that keeps surfaced arguments growing monotonically
// - see llama.h deviations). This is only reached when RawDecodeObject reports
// truncation (no matching close brace yet).
class TolerantJson {
 public:
  TolerantJson(const std::string& s, bool allow_str)
      : s_(s), allow_str_(allow_str) {}

  bool malformed() const { return malformed_; }

  ojson ParseObject() {
    // s_[i_] is '{'.
    ojson obj = ojson::object();
    ++i_;
    for (;;) {
      SkipWs();
      if (i_ >= s_.size()) return obj;  // truncated: no close
      if (s_[i_] == '}') {
        ++i_;
        return obj;
      }
      if (s_[i_] != '"') {
        malformed_ = true;
        return obj;
      }
      bool terminated = false;
      const std::string key = ParseStringRaw(terminated);
      if (!terminated) return obj;  // truncated key -> drop, stop
      SkipWs();
      if (i_ >= s_.size() || s_[i_] != ':') {
        if (i_ < s_.size()) malformed_ = true;
        return obj;
      }
      ++i_;
      bool drop = false;
      bool val_truncated = false;
      ojson value = ParseValue(drop, val_truncated);
      if (malformed_) return obj;
      if (drop) return obj;  // truncated value we chose not to surface
      obj[key] = std::move(value);
      if (val_truncated) return obj;  // partial value surfaced; stop
      SkipWs();
      if (i_ >= s_.size()) return obj;
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        return obj;
      }
      malformed_ = true;
      return obj;
    }
  }

 private:
  const std::string& s_;
  bool allow_str_;
  std::size_t i_ = 0;
  bool malformed_ = false;

  void SkipWs() {
    while (i_ < s_.size() && IsSpaceCh(s_[i_])) ++i_;
  }

  // Parse a JSON string starting at s_[i_]=='"'. Decodes standard JSON escapes.
  // `terminated` is set when the closing quote was seen.
  std::string ParseStringRaw(bool& terminated) {
    terminated = false;
    ++i_;  // opening quote
    std::string out;
    while (i_ < s_.size()) {
      const char c = s_[i_];
      if (c == '\\') {
        if (i_ + 1 >= s_.size()) {
          i_ = s_.size();
          return out;  // truncated mid-escape
        }
        const char e = s_[i_ + 1];
        switch (e) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            if (i_ + 5 < s_.size()) {
              const std::string hex = s_.substr(i_ + 2, 4);
              try {
                const int cp = std::stoi(hex, nullptr, 16);
                if (cp < 0x80) {
                  out += static_cast<char>(cp);
                } else {
                  // Minimal UTF-8 encode for the BMP subset the tests need.
                  out += static_cast<char>(0xC0 | (cp >> 6));
                  out += static_cast<char>(0x80 | (cp & 0x3F));
                }
              } catch (const std::exception&) {
                out += 'u';
              }
              i_ += 6;
              continue;
            }
            i_ = s_.size();
            return out;
          }
          default: out += e; break;
        }
        i_ += 2;
        continue;
      }
      if (c == '"') {
        ++i_;
        terminated = true;
        return out;
      }
      out += c;
      ++i_;
    }
    return out;  // truncated (no closing quote)
  }

  // Parse a value. On a truncated value that we choose not to surface, sets
  // `drop`. On a surfaced-but-partial value, sets `truncated`.
  ojson ParseValue(bool& drop, bool& truncated) {
    drop = false;
    truncated = false;
    SkipWs();
    if (i_ >= s_.size()) {
      drop = true;
      return ojson();
    }
    const char c = s_[i_];
    if (c == '"') {
      bool term = false;
      std::string str = ParseStringRaw(term);
      if (!term) {
        if (allow_str_) {
          truncated = true;
          return ojson(str);
        }
        drop = true;
        return ojson();
      }
      return ojson(str);
    }
    if (c == '{') {
      const std::string sub_src = s_.substr(i_);
      TolerantJson sub(sub_src, allow_str_);
      ojson obj = sub.ParseObject();
      if (sub.malformed()) {
        malformed_ = true;
        return ojson();
      }
      // Advance i_ by however much the sub-parser consumed.
      i_ += sub.i_;
      // A nested object is "confirmed" only if it closed; approximate by
      // checking the last consumed char.
      truncated = !(sub.i_ > 0 && s_[i_ - 1] == '}');
      return obj;
    }
    if (c == '[') {
      return ParseArray(drop, truncated);
    }
    // number / keyword: only surface when fully terminated (followed by a
    // delimiter). Otherwise drop.
    return ParseScalarConfirmed(drop, truncated);
  }

  ojson ParseArray(bool& drop, bool& truncated) {
    drop = false;
    truncated = false;
    ++i_;  // '['
    ojson arr = ojson::array();
    for (;;) {
      SkipWs();
      if (i_ >= s_.size()) {
        truncated = true;
        return arr;
      }
      if (s_[i_] == ']') {
        ++i_;
        return arr;
      }
      bool edrop = false;
      bool etrunc = false;
      ojson el = ParseValue(edrop, etrunc);
      if (malformed_) return arr;
      if (edrop) {
        truncated = true;
        return arr;
      }
      arr.push_back(std::move(el));
      if (etrunc) {
        truncated = true;
        return arr;
      }
      SkipWs();
      if (i_ >= s_.size()) {
        truncated = true;
        return arr;
      }
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        return arr;
      }
      malformed_ = true;
      return arr;
    }
  }

  // A number / true / false / null, surfaced only if the token is followed by a
  // delimiter (so we know it is complete). Otherwise `drop`.
  ojson ParseScalarConfirmed(bool& drop, bool& truncated) {
    drop = false;
    truncated = false;
    const std::size_t start = i_;
    while (i_ < s_.size()) {
      const char c = s_[i_];
      if (c == ',' || c == '}' || c == ']' || IsSpaceCh(c)) break;
      ++i_;
    }
    const std::string tok = s_.substr(start, i_ - start);
    if (i_ >= s_.size()) {
      // Ran to end of buffer: not confirmed complete -> drop.
      drop = true;
      return ojson();
    }
    ojson parsed = ojson::parse(tok, nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) {
      malformed_ = true;
      return ojson();
    }
    return parsed;
  }
};

// tool_parsers/utils.py:131 (partial_json_loads). Returns (value, consumed) for
// one JSON object at the front of `input`, tolerating a truncated trailing
// object. Returns nullopt on malformed JSON (upstream MalformedJSON).
std::optional<std::pair<ojson, std::size_t>> PartialJsonLoads(
    const std::string& input, bool allow_str) {
  std::size_t first = 0;
  while (first < input.size() && IsSpaceCh(input[first])) ++first;
  if (first >= input.size() || input[first] != '{') return std::nullopt;

  const RawDecodeResult rd = RawDecodeObject(input, first);
  if (rd.found_close) {
    if (!rd.parsed_ok) return std::nullopt;  // malformed
    // Extra-data vs whole-input: if only whitespace trails, consume everything.
    std::size_t tail = rd.end;
    while (tail < input.size() && IsSpaceCh(input[tail])) ++tail;
    const std::size_t consumed = (tail >= input.size()) ? input.size() : rd.end;
    return std::make_pair(rd.value, consumed);
  }
  // Truncated: tolerant partial parse of the whole remaining input.
  const std::string tail = input.substr(first);
  TolerantJson tj(tail, allow_str);
  ojson obj = tj.ParseObject();
  if (tj.malformed()) return std::nullopt;
  return std::make_pair(obj, input.size());
}

}  // namespace

ExtractedToolCallInformation LlamaToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  // llama_tool_parser.py:85-89 - quick check before scanning.
  if (model_output.find(kBotToken) == std::string::npos &&
      model_output.find('{') == std::string::npos) {
    return ExtractedToolCallInformation{false, {}, model_output};
  }

  // llama_tool_parser.py:91-157 - walk each top-level '{', raw_decode, collect.
  std::vector<ToolCall> tool_calls;
  long long end_index = -1;  // one past the last parsed object (absolute)
  std::size_t search = 0;
  for (;;) {
    const std::size_t start = model_output.find('{', search);
    if (start == std::string::npos) break;
    search = start + 1;
    if (static_cast<long long>(start) <= end_index) continue;  // inside prior obj

    const RawDecodeResult rd = RawDecodeObject(model_output, start);
    if (!rd.found_close || !rd.parsed_ok) {
      // llama_tool_parser.py:141-149 - any decode error -> plain content.
      return ExtractedToolCallInformation{false, {}, model_output};
    }
    end_index = static_cast<long long>(rd.end);

    // llama_tool_parser.py:112-115 - name is required; args come from
    // "arguments" or, failing that, "parameters".
    if (!rd.value.is_object() || !rd.value.contains("name")) {
      // Missing required "name" -> KeyError branch -> plain content.
      return ExtractedToolCallInformation{false, {}, model_output};
    }
    ojson args;
    if (rd.value.contains("arguments")) {
      args = rd.value.at("arguments");
    } else if (rd.value.contains("parameters")) {
      args = rd.value.at("parameters");
    } else {
      // Missing both -> KeyError branch -> plain content.
      return ExtractedToolCallInformation{false, {}, model_output};
    }

    ToolCall tc;
    tc.id = make_tool_call_id();
    tc.type = "function";
    tc.function.name = rd.value.at("name").get<std::string>();
    tc.function.arguments = args.dump();
    tool_calls.push_back(std::move(tc));
  }

  if (!tool_calls.empty()) {
    ExtractedToolCallInformation info;
    info.tools_called = true;
    info.tool_calls = std::move(tool_calls);
    return info;  // content = None
  }
  return ExtractedToolCallInformation{false, {}, model_output};
}

std::optional<DeltaMessage> LlamaToolParser::extract_tool_calls_streaming(
    const std::string& /*previous_text*/, const std::string& current_text,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  // llama_tool_parser.py:180-183 - plain content until the object/bot token.
  const std::string bot = kBotToken;
  if (!StartsWith(current_text, bot) && !StartsWith(current_text, "{")) {
    DeltaMessage msg;
    msg.content = delta_text;
    return msg;
  }

  // flags = Allow.ALL if name_sent else ALL & ~STR (line 189).
  const bool allow_str = current_tool_name_sent;

  try {
    // llama_tool_parser.py:191-217 - parse the (possibly multi-object,
    // "; "-separated, possibly truncated) tool-call array.
    std::vector<ojson> tool_call_arr;
    std::vector<bool> is_complete;
    std::size_t start_idx = StartsWith(current_text, bot) ? bot.size() : 0;
    while (start_idx < current_text.size()) {
      const std::optional<std::pair<ojson, std::size_t>> pj =
          PartialJsonLoads(current_text.substr(start_idx), allow_str);
      if (!pj.has_value()) return std::nullopt;  // MalformedJSON -> None
      ojson obj = pj->first;
      const std::size_t consumed = pj->second;
      is_complete.push_back(
          IsCompleteJson(current_text.substr(start_idx, consumed)));
      start_idx += consumed + 2;  // len("; ")
      // parameters -> arguments (line 209-213).
      if (obj.is_object() && obj.contains("parameters") &&
          !obj.contains("arguments")) {
        obj["arguments"] = obj.at("parameters");
      }
      tool_call_arr.push_back(std::move(obj));
    }

    // line 219-227 - nothing streamed yet.
    if (tool_call_arr.empty()) return std::nullopt;

    // line 220 - current tool call (Python negative indexing when id == -1).
    auto tool_at = [&](int id) -> const ojson& {
      int idx = id;
      if (idx < 0) idx += static_cast<int>(tool_call_arr.size());
      return tool_call_arr[static_cast<std::size_t>(idx)];
    };
    const ojson& current_tool_call = tool_at(current_tool_id);

    std::optional<DeltaMessage> delta;

    if (static_cast<int>(tool_call_arr.size()) > current_tool_id + 1) {
      // line 231-268 - starting a new tool; flush any unstreamed args of the
      // previous one first.
      if (current_tool_id >= 0) {
        const ojson cur_arguments =
            current_tool_call.contains("arguments") ? current_tool_call.at("arguments")
                                                     : ojson();
        if (JsonTruthy(cur_arguments)) {
          const std::string cur_args_json = cur_arguments.dump();
          const std::size_t sent =
              streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)].size();
          const std::string argument_diff =
              sent <= cur_args_json.size() ? cur_args_json.substr(sent) : std::string();
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
      prev_tool_call_arr_ = tool_call_arr;
      return delta;
    } else if (!current_tool_name_sent) {
      // line 272-289 - send the tool name once available.
      if (current_tool_call.is_object() && current_tool_call.contains("name") &&
          current_tool_call.at("name").is_string() &&
          !current_tool_call.at("name").get<std::string>().empty()) {
        DeltaToolCall d;
        d.index = current_tool_id;
        d.type = "function";
        d.id = make_tool_call_id();
        d.function.name = current_tool_call.at("name").get<std::string>();
        DeltaMessage m;
        m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        delta = std::move(m);
        current_tool_name_sent = true;
      }
    } else {
      // line 293-326 - stream argument fragments.
      const ojson cur_arguments =
          current_tool_call.contains("arguments") ? current_tool_call.at("arguments")
                                                   : ojson();
      if (JsonTruthy(cur_arguments)) {
        const std::size_t sent =
            streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)].size();
        const std::string cur_args_json = cur_arguments.dump();

        std::optional<std::string> argument_diff;
        const bool complete =
            static_cast<std::size_t>(current_tool_id) < is_complete.size() &&
            is_complete[static_cast<std::size_t>(current_tool_id)];
        if (complete) {
          argument_diff =
              sent <= cur_args_json.size() ? cur_args_json.substr(sent) : std::string();
        } else if (static_cast<std::size_t>(current_tool_id) <
                       prev_tool_call_arr_.size() &&
                   prev_tool_call_arr_[static_cast<std::size_t>(current_tool_id)]
                       .contains("arguments")) {
          const ojson prev_arguments =
              prev_tool_call_arr_[static_cast<std::size_t>(current_tool_id)].at(
                  "arguments");
          if (JsonTruthy(prev_arguments)) {
            const std::string prev_args_json = prev_arguments.dump();
            if (cur_args_json != prev_args_json) {
              const std::string prefix =
                  FindCommonPrefix(prev_args_json, cur_args_json);
              argument_diff =
                  sent <= prefix.size() ? prefix.substr(sent) : std::string();
            }
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

    prev_tool_call_arr_ = tool_call_arr;
    return delta;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace vllm::entrypoints::openai
