// Ported from: vllm/tool_parsers/utils.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/utils.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/tool_parsers/abstract.h"

namespace vllm::entrypoints::openai {

namespace {

bool IsJsonSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

// json.dumps(obj, ensure_ascii=False). nlohmann dump() is compact (no space
// after ':'/','); upstream CPython emits ", "/": ". The JSON is semantically
// identical (clients json.loads it) - whitespace-only deviation, matching the
// convention already established in hermes.cpp.
std::string DumpArgs(const nlohmann::ordered_json& v) { return v.dump(); }

// Python truthiness of a parsed "arguments" value: None / "" / {} / [] / 0 /
// false are falsy (granite guards arg streaming with `if cur_arguments:`).
bool JsonTruthy(const nlohmann::ordered_json& v) {
  if (v.is_null()) return false;
  if (v.is_string()) return !v.get<std::string>().empty();
  if (v.is_object() || v.is_array()) return !v.empty();
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number_float()) return v.get<double>() != 0.0;
  if (v.is_number()) return v.get<int64_t>() != 0;
  return true;
}

// ─── Tolerant JSON parser (stands in for partial_json_parser; see utils.h) ────
//
// Recursive-descent parser that COMPLETES a truncated document: on hitting the
// end of input inside a container it closes it, and inside an object it drops
// the trailing key/value pair that has not fully arrived. `allow_partial_str_`
// mirrors Allow.STR - when false an unterminated string value is dropped.
struct NeedMore {};   // an incomplete token that must be dropped by its parent
struct Malformed {};  // nothing parseable at this position

class TolerantParser {
 public:
  TolerantParser(const std::string& s, bool allow_partial_str)
      : s_(s), allow_partial_str_(allow_partial_str) {}

  nlohmann::ordered_json Parse() {
    SkipWs();
    if (pos_ >= s_.size()) throw Malformed{};
    return ParseValue();
  }
  std::size_t pos() const { return pos_; }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;
  bool allow_partial_str_;

  void SkipWs() {
    while (pos_ < s_.size() && IsJsonSpace(s_[pos_])) ++pos_;
  }

  nlohmann::ordered_json ParseValue() {
    SkipWs();
    if (pos_ >= s_.size()) throw NeedMore{};
    const char c = s_[pos_];
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') return ParseString(/*as_key=*/false);
    if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();
    if (c == 't' || c == 'f' || c == 'n') return ParseKeyword();
    throw Malformed{};
  }

  nlohmann::ordered_json ParseString(bool as_key) {
    const std::size_t start = pos_;  // at opening quote
    ++pos_;
    std::string partial;
    bool closed = false;
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (c == '\\') {
        if (pos_ + 1 >= s_.size()) {  // dangling backslash -> truncated
          ++pos_;
          break;
        }
        const char e = s_[pos_ + 1];
        switch (e) {  // best-effort unescape for the PARTIAL fallback only
          case 'n': partial += '\n'; break;
          case 't': partial += '\t'; break;
          case 'r': partial += '\r'; break;
          case 'b': partial += '\b'; break;
          case 'f': partial += '\f'; break;
          default: partial += e; break;  // \" \\ \/ and (approx) \uXXXX
        }
        pos_ += 2;
        continue;
      }
      if (c == '"') {
        ++pos_;
        closed = true;
        break;
      }
      partial += c;
      ++pos_;
    }
    if (closed) {
      // Complete token: let nlohmann do the authoritative unescaping (\uXXXX,
      // surrogate pairs, raw UTF-8 all handled correctly).
      return nlohmann::ordered_json::parse(s_.substr(start, pos_ - start));
    }
    if (as_key) throw NeedMore{};                 // an unterminated key -> drop
    if (allow_partial_str_) return nlohmann::ordered_json(partial);
    throw NeedMore{};                             // Allow.STR off -> drop
  }

  nlohmann::ordered_json ParseNumber() {
    const std::size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
          c == '+' || c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
    std::string num = s_.substr(start, pos_ - start);
    // Trim trailing incomplete exponents/decimals (e.g. "12." / "1e") until it
    // parses, mirroring the completer's best-effort numeric handling.
    while (!num.empty()) {
      if (nlohmann::json::accept(num)) return nlohmann::ordered_json::parse(num);
      num.pop_back();
    }
    throw NeedMore{};
  }

  nlohmann::ordered_json ParseKeyword() {
    static const std::pair<const char*, nlohmann::ordered_json> kws[] = {
        {"true", nlohmann::ordered_json(true)},
        {"false", nlohmann::ordered_json(false)},
        {"null", nlohmann::ordered_json(nullptr)}};
    for (const auto& kw : kws) {
      const std::string word = kw.first;
      if (s_.compare(pos_, word.size(), word) == 0) {
        pos_ += word.size();
        return kw.second;
      }
    }
    // A truncated prefix that runs to end-of-input completes to its keyword
    // (partial_json_parser turns "tru" -> true).
    for (const auto& kw : kws) {
      const std::string word = kw.first;
      const std::size_t rem = s_.size() - pos_;
      if (rem > 0 && rem < word.size() &&
          word.compare(0, rem, s_, pos_, rem) == 0) {
        pos_ = s_.size();
        return kw.second;
      }
    }
    throw Malformed{};
  }

  nlohmann::ordered_json ParseArray() {
    nlohmann::ordered_json arr = nlohmann::ordered_json::array();
    ++pos_;  // consume '['
    for (;;) {
      SkipWs();
      if (pos_ >= s_.size()) return arr;             // truncated -> complete
      if (s_[pos_] == ']') {
        ++pos_;
        return arr;
      }
      nlohmann::ordered_json v;
      try {
        v = ParseValue();
      } catch (const NeedMore&) {
        return arr;  // incomplete trailing element dropped
      }
      arr.push_back(std::move(v));
      SkipWs();
      if (pos_ >= s_.size()) return arr;
      if (s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (s_[pos_] == ']') {
        ++pos_;
        return arr;
      }
      return arr;  // unexpected -> stop and complete
    }
  }

  nlohmann::ordered_json ParseObject() {
    nlohmann::ordered_json obj = nlohmann::ordered_json::object();
    ++pos_;  // consume '{'
    for (;;) {
      SkipWs();
      if (pos_ >= s_.size()) return obj;
      if (s_[pos_] == '}') {
        ++pos_;
        return obj;
      }
      if (s_[pos_] != '"') return obj;  // trailing junk / bad key -> complete
      std::string key;
      try {
        key = ParseString(/*as_key=*/true).get<std::string>();
      } catch (const NeedMore&) {
        return obj;  // incomplete key -> drop the pair
      }
      SkipWs();
      if (pos_ >= s_.size()) return obj;   // no colon yet -> drop pair
      if (s_[pos_] != ':') return obj;
      ++pos_;                              // consume ':'
      SkipWs();
      if (pos_ >= s_.size()) return obj;   // no value yet -> drop pair
      nlohmann::ordered_json v;
      try {
        v = ParseValue();
      } catch (const NeedMore&) {
        return obj;  // incomplete value -> drop pair
      }
      obj[key] = std::move(v);
      SkipWs();
      if (pos_ >= s_.size()) return obj;
      if (s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (s_[pos_] == '}') {
        ++pos_;
        return obj;
      }
      return obj;  // unexpected -> stop and complete
    }
  }
};

}  // namespace

std::size_t consume_space(std::size_t i, const std::string& s) {
  while (i < s.size() && IsJsonSpace(s[i])) ++i;
  return i;
}

std::string find_common_prefix(const std::string& a, const std::string& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return a.substr(0, i);
}

bool is_complete_json(const std::string& s) { return nlohmann::json::accept(s); }

std::pair<nlohmann::ordered_json, std::size_t> partial_json_loads(
    const std::string& s, bool allow_partial_str) {
  TolerantParser p(s, allow_partial_str);
  nlohmann::ordered_json v;
  try {
    v = p.Parse();
  } catch (const NeedMore&) {
    throw MalformedPartialJson{};
  } catch (const Malformed&) {
    throw MalformedPartialJson{};
  }
  return {std::move(v), p.pos()};
}

std::optional<DeltaMessage> granite_stream_emit(
    std::vector<nlohmann::ordered_json>& prev_tool_call_arr, int& current_tool_id,
    bool& current_tool_name_sent, std::vector<std::string>& streamed_args_for_tool,
    const std::vector<nlohmann::ordered_json>& tool_call_arr,
    const std::vector<bool>& is_complete) {
  // granite_tool_parser.py:150-153 - nothing streamed yet (only brackets).
  if (tool_call_arr.empty()) return std::nullopt;

  // granite_tool_parser.py:154 - current_tool_call = tool_call_arr[current_tool_id];
  // Python negative index (-1 initially) selects the last element.
  const auto norm = [&](int id) -> const nlohmann::ordered_json& {
    const int n = static_cast<int>(tool_call_arr.size());
    const int idx = id < 0 ? n + id : id;
    return tool_call_arr[static_cast<std::size_t>(idx)];
  };

  std::optional<DeltaMessage> delta;  // None

  // granite_tool_parser.py:157-191 - a new tool has appeared in the array.
  if (static_cast<int>(tool_call_arr.size()) > current_tool_id + 1) {
    if (current_tool_id >= 0) {
      const nlohmann::ordered_json& current_tool_call = norm(current_tool_id);
      if (current_tool_call.contains("arguments") &&
          JsonTruthy(current_tool_call.at("arguments"))) {
        const std::string cur_args_json = DumpArgs(current_tool_call.at("arguments"));
        const std::size_t sent =
            streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)].size();
        const std::string argument_diff =
            sent < cur_args_json.size() ? cur_args_json.substr(sent) : std::string();
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
    return delta;  // NOTE: prev_tool_call_arr intentionally NOT updated here.
  }

  // granite_tool_parser.py:193-210 - send the tool name once, when available.
  if (!current_tool_name_sent) {
    const nlohmann::ordered_json& current_tool_call = norm(current_tool_id);
    if (current_tool_call.contains("name") &&
        JsonTruthy(current_tool_call.at("name"))) {
      const std::string function_name = current_tool_call.at("name").get<std::string>();
      DeltaToolCall d;
      d.index = current_tool_id;
      d.type = "function";
      d.id = make_tool_call_id();
      d.function.name = function_name;
      DeltaMessage m;
      m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
      delta = std::move(m);
      current_tool_name_sent = true;
    }
  } else {
    // granite_tool_parser.py:212-246 - same tool, stream the argument diff.
    const nlohmann::ordered_json& current_tool_call = norm(current_tool_id);
    if (current_tool_call.contains("arguments") &&
        JsonTruthy(current_tool_call.at("arguments"))) {
      const std::size_t sent =
          streamed_args_for_tool[static_cast<std::size_t>(current_tool_id)].size();
      const std::string cur_args_json = DumpArgs(current_tool_call.at("arguments"));

      std::optional<std::string> argument_diff;
      if (is_complete[static_cast<std::size_t>(current_tool_id)]) {
        argument_diff = sent < cur_args_json.size() ? cur_args_json.substr(sent)
                                                    : std::string();
      } else if (static_cast<std::size_t>(current_tool_id) < prev_tool_call_arr.size() &&
                 prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)].contains(
                     "arguments") &&
                 JsonTruthy(prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)]
                                .at("arguments"))) {
        const std::string prev_args_json = DumpArgs(
            prev_tool_call_arr[static_cast<std::size_t>(current_tool_id)].at("arguments"));
        if (cur_args_json != prev_args_json) {
          const std::string prefix = find_common_prefix(prev_args_json, cur_args_json);
          argument_diff = sent < prefix.size() ? prefix.substr(sent) : std::string();
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

  // granite_tool_parser.py:248 - self.prev_tool_call_arr = tool_call_arr.
  prev_tool_call_arr = tool_call_arr;
  return delta;
}

}  // namespace vllm::entrypoints::openai
