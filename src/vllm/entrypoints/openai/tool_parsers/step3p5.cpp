// Ported from: vllm/tool_parsers/step3p5_tool_parser.py @ e24d1b24
#include "vllm/entrypoints/openai/tool_parsers/step3p5.h"

#include <cctype>
#include <cstddef>
#include <optional>
#include <regex>
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
std::string Strip(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && IsSpaceCh(s[b])) ++b;
  while (e > b && IsSpaceCh(s[e - 1])) --e;
  return s.substr(b, e - b);
}
std::string Lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

// The two closed sets step3p5 keys type handling off of.
bool IsStringType(const std::string& t) {
  return t == "string" || t == "str" || t == "text" || t == "varchar" ||
         t == "char" || t == "enum";
}
bool IsBoolType(const std::string& t) {
  return t == "boolean" || t == "bool" || t == "binary";
}

// json.dumps(value, ensure_ascii=False). nlohmann dump() keeps UTF-8 by default.
std::string JsonDumps(const ojson& v) { return v.dump(); }

// _escape_xml_special_chars (step3p5_tool_parser.py:220). Order matters: & first.
std::string EscapeXml(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out += c;
    }
  }
  return out;
}

// The inverse expat performs when delivering CharacterData. &amp; resolved last.
std::string UnescapeXml(std::string s) {
  auto replace_all = [](std::string& str, const std::string& from,
                        const std::string& to) {
    std::size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
      str.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  replace_all(s, "&lt;", "<");
  replace_all(s, "&gt;", ">");
  replace_all(s, "&quot;", "\"");
  replace_all(s, "&apos;", "'");
  replace_all(s, "&amp;", "&");
  return s;
}

// ── Python literal evaluator (ast.literal_eval subset) ──────────────────────
// Handles None/True/False, ints, floats, single/double-quoted strings, lists,
// tuples and dicts, into ordered_json. Sets ok=false on any failure.
class PyLiteralParser {
 public:
  explicit PyLiteralParser(const std::string& s) : s_(s) {}

  bool Eval(ojson& out) {
    SkipWs();
    if (!ParseValue(out)) return false;
    SkipWs();
    return i_ == s_.size();  // no trailing junk (literal_eval semantics)
  }

 private:
  const std::string& s_;
  std::size_t i_ = 0;

  void SkipWs() {
    while (i_ < s_.size() && IsSpaceCh(s_[i_])) ++i_;
  }
  bool Peek(char c) const { return i_ < s_.size() && s_[i_] == c; }

  bool ParseValue(ojson& out) {
    SkipWs();
    if (i_ >= s_.size()) return false;
    char c = s_[i_];
    if (c == '{') return ParseDict(out);
    if (c == '[') return ParseSeq(out, ']');
    if (c == '(') return ParseSeq(out, ')');
    if (c == '\'' || c == '"') {
      std::string str;
      if (!ParseString(str)) return false;
      out = str;
      return true;
    }
    return ParseAtom(out);
  }

  bool ParseString(std::string& out) {
    char q = s_[i_++];
    std::string res;
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) return false;
        char e = s_[i_++];
        switch (e) {
          case 'n': res += '\n'; break;
          case 't': res += '\t'; break;
          case 'r': res += '\r'; break;
          case '\\': res += '\\'; break;
          case '\'': res += '\''; break;
          case '"': res += '"'; break;
          case '0': res += '\0'; break;
          default: res += e; break;
        }
      } else if (c == q) {
        out = res;
        return true;
      } else {
        res += c;
      }
    }
    return false;  // unterminated
  }

  bool ParseAtom(ojson& out) {
    std::size_t start = i_;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (IsSpaceCh(c) || c == ',' || c == '}' || c == ']' || c == ')' ||
          c == ':') {
        break;
      }
      ++i_;
    }
    std::string tok = s_.substr(start, i_ - start);
    if (tok.empty()) return false;
    if (tok == "None") {
      out = nullptr;
      return true;
    }
    if (tok == "True") {
      out = true;
      return true;
    }
    if (tok == "False") {
      out = false;
      return true;
    }
    // number
    try {
      if (tok.find('.') != std::string::npos ||
          tok.find('e') != std::string::npos ||
          tok.find('E') != std::string::npos) {
        std::size_t pos = 0;
        double d = std::stod(tok, &pos);
        if (pos != tok.size()) return false;
        out = d;
        return true;
      }
      std::size_t pos = 0;
      long long v = std::stoll(tok, &pos);
      if (pos != tok.size()) return false;
      out = v;
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  bool ParseSeq(ojson& out, char close) {
    out = ojson::array();
    ++i_;  // consume the opening bracket
    SkipWs();
    if (Peek(close)) {
      ++i_;
      return true;
    }
    for (;;) {
      ojson v;
      if (!ParseValue(v)) return false;
      out.push_back(std::move(v));
      SkipWs();
      if (Peek(',')) {
        ++i_;
        SkipWs();
        if (Peek(close)) {  // trailing comma (esp. 1-tuples)
          ++i_;
          return true;
        }
        continue;
      }
      if (Peek(close)) {
        ++i_;
        return true;
      }
      return false;
    }
  }

  bool ParseDict(ojson& out) {
    out = ojson::object();
    ++i_;  // consume {
    SkipWs();
    if (Peek('}')) {
      ++i_;
      return true;
    }
    for (;;) {
      SkipWs();
      // key: string / number / bool / None -> stringify to JSON object key.
      std::string key;
      if (Peek('\'') || Peek('"')) {
        if (!ParseString(key)) return false;
      } else {
        ojson kv;
        if (!ParseAtom(kv)) return false;
        if (kv.is_string()) {
          key = kv.get<std::string>();
        } else {
          key = kv.dump();  // Python str() of the key
        }
      }
      SkipWs();
      if (!Peek(':')) return false;
      ++i_;  // consume :
      ojson val;
      if (!ParseValue(val)) return false;
      out[key] = std::move(val);
      SkipWs();
      if (Peek(',')) {
        ++i_;
        SkipWs();
        if (Peek('}')) {
          ++i_;
          return true;
        }
        continue;
      }
      if (Peek('}')) {
        ++i_;
        return true;
      }
      return false;
    }
  }
};

bool SafeLiteralEval(const std::string& text, ojson& out) {
  PyLiteralParser p(text);
  return p.Eval(out);
}

}  // namespace

namespace step3p5_detail {

// The full streaming XML state machine (step3p5_tool_parser.py:30-1363).
class StreamingXmlToolCallParser {
 public:
  StreamingXmlToolCallParser() { reset_streaming_state(); }

  void set_tools(const std::vector<ChatCompletionToolsParam>* tools) {
    tools_ = tools;
  }

  // step3p5_tool_parser.py:48 (reset_streaming_state).
  void reset_streaming_state() {
    deltas_.clear();
    tool_call_index_ = 0;
    current_call_id_.reset();
    last_completed_call_id_.reset();
    current_function_name_.reset();
    current_function_open_ = false;
    parameters_ = ojson::object();
    current_param_name_.reset();
    current_param_value_.clear();
    current_param_value_converted_.clear();
    current_param_is_first_ = false;
    should_emit_end_newline_ = false;
    start_quote_emitted_ = false;
    streaming_buffer_.clear();
    last_processed_pos_ = 0;
    text_content_buffer_.clear();
    pre_inside_parameter_ = false;
    pre_param_buffer_.clear();
    pre_current_param_name_.reset();
    defer_current_parameter_ = false;
    deferred_param_raw_value_.clear();
    xml_recreate();
  }

  // step3p5_tool_parser.py:82 (parse_single_streaming_chunks).
  DeltaMessage parse_single_streaming_chunks(const std::string& xml_chunk) {
    const std::size_t initial_delta_count = deltas_.size();
    const std::optional<std::string> entry_call_id = current_call_id_;
    const int entry_tool_call_index = tool_call_index_;

    streaming_buffer_ += xml_chunk;

    const bool found_elements = process_complete_xml_elements();

    std::optional<std::string> fallback_call_id;
    if (entry_call_id.has_value()) {
      if (current_call_id_ == entry_call_id &&
          tool_call_index_ == entry_tool_call_index) {
        fallback_call_id = entry_call_id;
      }
    } else if (current_call_id_.has_value() &&
               tool_call_index_ == entry_tool_call_index + 1) {
      fallback_call_id = current_call_id_;
    }

    const std::string function_end = kFunctionEnd;
    const std::string tool_call_end = kToolCallEnd;

    if (found_elements) {
      // Complete elements found: patch up any missed end events.
      if (fallback_call_id.has_value() &&
          xml_chunk.find(function_end) != std::string::npos) {
        const bool has_function_close = AnyDeltaMatches(
            initial_delta_count, [&](const DeltaToolCall& tc) {
              return tc.id == fallback_call_id && tc.function.arguments &&
                     (*tc.function.arguments == "}" ||
                      *tc.function.arguments == "{}");
            });
        if (!has_function_close) {
          if (current_param_name_.has_value()) end_element("parameter");
          if (current_function_name_.has_value()) end_element("function");
        }
      }
      if (fallback_call_id.has_value() &&
          xml_chunk.find(tool_call_end) != std::string::npos) {
        const bool has_toolcall_close = AnyDeltaMatches(
            initial_delta_count, [&](const DeltaToolCall& tc) {
              return tc.type == "function" && tc.function.arguments &&
                     *tc.function.arguments == "" && tc.id == fallback_call_id;
            });
        if (!has_toolcall_close) {
          if (current_param_name_.has_value()) end_element("parameter");
          if (current_function_name_.has_value()) end_element("function");
          end_element("tool_call");
        }
      }
      return merge_new_deltas(initial_delta_count);
    }

    // No complete elements.
    if (!text_content_buffer_.empty() && tool_call_index_ == 0) {
      DeltaMessage m;
      m.content = text_content_buffer_;
      emit_delta(m);
      text_content_buffer_.clear();
      return m;
    }

    if (fallback_call_id.has_value() &&
        (xml_chunk.find(function_end) != std::string::npos ||
         xml_chunk.find(tool_call_end) != std::string::npos)) {
      if (current_param_name_.has_value()) end_element("parameter");
      if (xml_chunk.find(function_end) != std::string::npos &&
          current_function_name_.has_value()) {
        end_element("function");
      }
      if (xml_chunk.find(tool_call_end) != std::string::npos) {
        end_element("tool_call");
      }
      return merge_new_deltas(initial_delta_count);
    }

    return DeltaMessage{};  // content=None
  }

 private:
  // ── marker literals (step3p5_tool_parser.py:41-46) ────────────────────────
  static constexpr const char* kToolCallStart = "<tool_call>";
  static constexpr const char* kToolCallEnd = "</tool_call>";
  static constexpr const char* kFunctionStart = "<function=";
  static constexpr const char* kFunctionEnd = "</function>";
  static constexpr const char* kParameterStart = "<parameter=";
  static constexpr const char* kParameterEnd = "</parameter>";

  // ── streaming state ───────────────────────────────────────────────────────
  const std::vector<ChatCompletionToolsParam>* tools_ = nullptr;
  std::vector<DeltaMessage> deltas_;
  int tool_call_index_ = 0;
  std::optional<std::string> current_call_id_;
  std::optional<std::string> last_completed_call_id_;
  std::optional<std::string> current_function_name_;
  bool current_function_open_ = false;
  ojson parameters_ = ojson::object();
  std::optional<std::string> current_param_name_;
  std::string current_param_value_;
  std::string current_param_value_converted_;
  bool current_param_is_first_ = false;
  bool should_emit_end_newline_ = false;
  bool start_quote_emitted_ = false;
  std::string streaming_buffer_;
  std::size_t last_processed_pos_ = 0;
  std::string text_content_buffer_;
  bool pre_inside_parameter_ = false;
  std::string pre_param_buffer_;
  std::optional<std::string> pre_current_param_name_;
  bool defer_current_parameter_ = false;
  std::string deferred_param_raw_value_;

  // ── minimal XML emitter state (expat surrogate) ───────────────────────────
  std::string xml_pending_;  // buffered character data (buffer_text=True)

  void xml_recreate() { xml_pending_.clear(); }

  void emit_delta(DeltaMessage d) { deltas_.push_back(std::move(d)); }

  // Build a tool-call delta (index/id/type always set; name optional; arguments
  // always present as a string).
  DeltaMessage tool_delta(std::optional<std::string> name,
                          std::string arguments) {
    DeltaMessage m;
    DeltaToolCall d;
    d.index = tool_call_index_ - 1;
    d.id = current_call_id_;
    d.type = "function";
    d.function.name = std::move(name);
    d.function.arguments = std::move(arguments);
    m.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
    return m;
  }

  template <typename Pred>
  bool AnyDeltaMatches(std::size_t initial_count, Pred pred) {
    for (std::size_t i = initial_count; i < deltas_.size(); ++i) {
      if (!deltas_[i].tool_calls.has_value()) continue;
      for (const DeltaToolCall& tc : *deltas_[i].tool_calls) {
        if (pred(tc)) return true;
      }
    }
    return false;
  }

  // ── type inference / coercion (step3p5_tool_parser.py:1191-1331) ──────────
  std::string get_param_type(const std::string& param_name) {
    if (tools_ == nullptr || !current_function_name_.has_value()) {
      return "string";
    }
    for (const ChatCompletionToolsParam& tool : *tools_) {
      if (tool.type != "function" ||
          tool.function.name != *current_function_name_) {
        continue;
      }
      if (!tool.function.parameters.has_value()) return "string";
      const nlohmann::json& params = *tool.function.parameters;
      if (params.is_object() && params.contains("properties")) {
        const nlohmann::json& properties = params["properties"];
        if (properties.is_object() && properties.contains(param_name) &&
            properties[param_name].is_object()) {
          return repair_param_type(GetTypeString(properties[param_name]));
        }
      } else if (params.is_object() && params.contains(param_name)) {
        const nlohmann::json& pc = params[param_name];
        if (pc.is_object()) return repair_param_type(GetTypeString(pc));
      }
      break;
    }
    return "string";
  }

  static std::string GetTypeString(const nlohmann::json& prop) {
    if (prop.contains("type") && prop["type"].is_string()) {
      return prop["type"].get<std::string>();
    }
    return "string";
  }

  static std::string repair_param_type(const std::string& t) {
    if (IsStringType(t) || t.rfind("int", 0) == 0 || t.rfind("uint", 0) == 0 ||
        t.rfind("long", 0) == 0 || t.rfind("short", 0) == 0 ||
        t.rfind("unsigned", 0) == 0 || t.rfind("num", 0) == 0 ||
        t.rfind("float", 0) == 0 || IsBoolType(t) || t == "object" ||
        t == "array" || t == "arr" || t == "sequence" || t.rfind("dict", 0) == 0 ||
        t.rfind("list", 0) == 0) {
      return t;
    }
    return "string";
  }

  ojson convert_param_value(const std::string& param_value_in,
                            const std::string& param_type_in) {
    if (Lower(param_value_in) == "null") return ojson(nullptr);
    const std::string t = Lower(Strip(param_type_in));
    const std::string& v = param_value_in;
    if (IsStringType(t)) return ojson(v);
    if (t.rfind("int", 0) == 0 || t.rfind("uint", 0) == 0 ||
        t.rfind("long", 0) == 0 || t.rfind("short", 0) == 0 ||
        t.rfind("unsigned", 0) == 0) {
      try {
        std::size_t pos = 0;
        long long iv = std::stoll(v, &pos);
        if (pos == v.size()) return ojson(iv);
      } catch (const std::exception&) {
      }
      return ojson(v);  // degrade to string
    }
    if (t.rfind("num", 0) == 0 || t.rfind("float", 0) == 0) {
      try {
        std::size_t pos = 0;
        double fv = std::stod(v, &pos);
        if (pos == v.size()) {
          const long long as_int = static_cast<long long>(fv);
          if (fv - static_cast<double>(as_int) != 0.0) return ojson(fv);
          return ojson(as_int);
        }
      } catch (const std::exception&) {
      }
      return ojson(v);
    }
    if (IsBoolType(t)) return ojson(Lower(v) == "true");
    return ojson(v);  // object/array/etc: raw string passthrough
  }

  std::string convert_for_json_streaming(const ojson& value,
                                         const std::string& param_type) {
    if (value.is_null()) return "";
    if (value.is_string() && value.get<std::string>().empty()) return "";
    if (IsStringType(param_type)) {
      const std::string dumped = JsonDumps(value);  // "..."
      if (dumped.size() >= 2) return dumped.substr(1, dumped.size() - 2);
      return dumped;
    }
    if (!value.is_string()) return JsonDumps(value);
    return value.get<std::string>();
  }

  // ── tool / parameter name validation (for the tag-fix fallbacks) ──────────
  bool validate_function_name(const std::string& func_name) {
    if (tools_ == nullptr) return false;
    for (const ChatCompletionToolsParam& tool : *tools_) {
      if (tool.type == "function" && tool.function.name == func_name) {
        return true;
      }
    }
    return false;
  }

  bool validate_parameter_name(const std::string& param_name) {
    if (tools_ == nullptr || !current_function_name_.has_value()) return true;
    for (const ChatCompletionToolsParam& tool : *tools_) {
      if (tool.type == "function" &&
          tool.function.name == *current_function_name_) {
        if (!tool.function.parameters.has_value()) return true;
        const nlohmann::json& params = *tool.function.parameters;
        if (params.is_object()) {
          const nlohmann::json& properties =
              params.contains("properties") ? params["properties"] : params;
          return properties.is_object() && properties.contains(param_name);
        }
        break;
      }
    }
    return true;
  }

  // ── malformed-tag repair (step3p5_tool_parser.py:322-425) ─────────────────
  std::string fix_missing_equals_in_function_tag(std::string chunk) {
    if (chunk.find("<function=") != std::string::npos) return chunk;
    static const std::regex p1(R"(<function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*>)");
    std::smatch m;
    if (std::regex_search(chunk, m, p1)) {
      const std::string func_name = Strip(m[1].str());
      if (!func_name.empty() && validate_function_name(func_name)) {
        chunk.replace(m.position(0), m.length(0), "<function=" + func_name + ">");
        return chunk;
      }
    }
    static const std::regex p2(R"(<function([a-zA-Z_][a-zA-Z0-9_]*)\s*>)");
    if (std::regex_search(chunk, m, p2)) {
      const std::string func_name = Strip(m[1].str());
      if (!func_name.empty() && validate_function_name(func_name)) {
        chunk.replace(m.position(0), m.length(0), "<function=" + func_name + ">");
        return chunk;
      }
    }
    return chunk;
  }

  std::string fix_incomplete_tag_in_chunk(std::string chunk) {
    chunk = fix_missing_equals_in_function_tag(chunk);
    for (const std::string& tag_type : {std::string("parameter"),
                                        std::string("function")}) {
      const std::string pattern = "<" + tag_type + "=";
      if (chunk.find(pattern) == std::string::npos) continue;
      const std::size_t start_idx = chunk.find(pattern);
      const std::string after_tag = chunk.substr(start_idx);
      const std::size_t gt_pos = after_tag.find('>');
      const std::size_t lt_pos = after_tag.find('<', pattern.size());
      const bool well_formed =
          gt_pos != std::string::npos &&
          (lt_pos == std::string::npos || gt_pos < lt_pos) &&
          after_tag.substr(0, gt_pos).find(pattern) != std::string::npos;
      if (well_formed) continue;

      const std::string content = chunk.substr(start_idx + pattern.size());
      std::size_t end_pos = content.size();
      for (std::size_t i = 0; i < content.size(); ++i) {
        if (content[i] == ' ' || content[i] == '\n' || content[i] == '<') {
          end_pos = i;
          break;
        }
      }
      std::string tag_name = content.substr(0, end_pos);
      if (tag_name.empty()) continue;
      if (tag_name.rfind(tag_type + "=", 0) == 0) {
        tag_name = tag_name.substr(tag_type.size() + 1);
      }
      while (!tag_name.empty()) {
        char last = tag_name.back();
        if (std::isalnum(static_cast<unsigned char>(last)) || last == '-' ||
            last == '_') {
          break;
        }
        tag_name.pop_back();
      }
      if (tag_name.empty()) continue;
      if (tag_type == "parameter" && !validate_parameter_name(tag_name)) {
        continue;
      }
      const std::string from = "<" + tag_type + "=" + content.substr(0, end_pos);
      const std::string to = "<" + tag_type + "=" + tag_name + ">";
      const std::size_t fpos = chunk.find(from);
      if (fpos != std::string::npos) chunk.replace(fpos, from.size(), to);
    }
    return chunk;
  }

  // ── element skip / find (step3p5_tool_parser.py:467-588) ──────────────────
  bool should_skip_element(const std::string& element) {
    if (element.rfind(kToolCallStart, 0) == 0 ||
        element.rfind(kFunctionStart, 0) == 0 ||
        element.rfind(kParameterStart, 0) == 0) {
      return false;
    }
    if (!current_call_id_.has_value() && !element.empty()) {
      text_content_buffer_ += element;
      return true;
    }
    if (current_call_id_.has_value()) return false;
    return element.empty();
  }

  // Returns {element, end_pos}; element empty-optional means "wait".
  std::pair<std::optional<std::string>, std::size_t> find_next_complete_element(
      std::size_t start_pos) {
    const std::string buffer = streaming_buffer_.substr(start_pos);
    if (buffer.empty()) return {std::nullopt, start_pos};

    if (buffer[0] == '<') {
      const std::string first_line = buffer.substr(0, buffer.find('\n'));
      const bool is_incomplete_param =
          buffer.rfind("<parameter=", 0) == 0 &&
          first_line.find('>') == std::string::npos;
      const bool is_incomplete_func =
          buffer.rfind("<function=", 0) == 0 &&
          first_line.find('>') == std::string::npos;

      if (is_incomplete_param || is_incomplete_func) {
        const std::string closing =
            is_incomplete_param ? "</parameter>" : "</function>";
        const std::size_t closing_pos = buffer.find(closing);
        if (closing_pos != std::string::npos) {
          const std::size_t stop = closing_pos + closing.size();
          return {buffer.substr(0, stop), start_pos + stop};
        }
      }

      const std::size_t tag_end = buffer.find('<', 1);
      const std::size_t tag_end2 = buffer.find('>', 1);
      if (tag_end != std::string::npos && tag_end2 != std::string::npos) {
        if (tag_end < tag_end2) {
          return {buffer.substr(0, tag_end), start_pos + tag_end};
        }
        return {buffer.substr(0, tag_end2 + 1), start_pos + tag_end2 + 1};
      }
      if (tag_end != std::string::npos) {
        return {buffer.substr(0, tag_end), start_pos + tag_end};
      }
      if (tag_end2 != std::string::npos) {
        return {buffer.substr(0, tag_end2 + 1), start_pos + tag_end2 + 1};
      }
      // No '<' or '>' after position 0.
      if (!current_call_id_.has_value()) {
        if (std::string(kToolCallStart).rfind(buffer, 0) == 0) {
          return {std::nullopt, start_pos};  // maybe start of <tool_call>
        }
        if (buffer.rfind("<function=", 0) == 0 ||
            std::string("<function=").rfind(buffer, 0) == 0) {
          return {std::nullopt, start_pos};  // maybe start of <function=
        }
        return {buffer, start_pos + buffer.size()};  // treat as text
      }
      return {std::nullopt, start_pos};  // wait for more data
    }

    const std::size_t next_tag_pos = buffer.find('<');
    if (next_tag_pos != std::string::npos) {
      return {buffer.substr(0, next_tag_pos), start_pos + next_tag_pos};
    }
    return {buffer, start_pos + buffer.size()};
  }

  // ── the main element loop (step3p5_tool_parser.py:241-320) ────────────────
  bool process_complete_xml_elements() {
    bool found_any = false;
    while (last_processed_pos_ < streaming_buffer_.size()) {
      auto [element, end_pos] = find_next_complete_element(last_processed_pos_);
      if (!element.has_value()) break;

      if (should_skip_element(*element)) {
        last_processed_pos_ = end_pos;
        continue;
      }

      const std::string preprocessed = preprocess_xml_chunk(*element);
      const std::string trimmed = Strip(preprocessed);

      if (((trimmed.rfind("<tool_call>", 0) == 0 ||
            trimmed.rfind("<function name=", 0) == 0)) &&
          tool_call_index_ == 0 && !text_content_buffer_.empty()) {
        DeltaMessage m;
        m.content = text_content_buffer_;
        emit_delta(m);
        text_content_buffer_.clear();
      }

      if (trimmed.rfind("<tool_call>", 0) == 0 && tool_call_index_ > 0 &&
          current_call_id_.has_value() && current_function_name_.has_value()) {
        if (current_param_name_.has_value()) end_element("parameter");
        if (current_function_open_) end_element("function");
        DeltaMessage final_delta;
        DeltaToolCall d;
        d.index = tool_call_index_ - 1;
        d.id = current_call_id_;
        d.type = "function";
        d.function.arguments = std::string("");
        final_delta.tool_calls = std::vector<DeltaToolCall>{std::move(d)};
        emit_delta(final_delta);
        reset_xml_parser_after_tool_call();
      }

      xml_parse(preprocessed);
      found_any = true;

      last_processed_pos_ = end_pos;
    }
    return found_any;
  }

  // ── preprocessing (step3p5_tool_parser.py:654-785) ────────────────────────
  std::string preprocess_xml_chunk(const std::string& chunk_in) {
    std::string chunk = chunk_in;
    bool is_tool_call = false;
    if (chunk.rfind(kToolCallStart, 0) == 0 ||
        chunk.rfind(kToolCallEnd, 0) == 0) {
      is_tool_call = true;
    }
    static const std::regex func_prefix(R"(^<function[a-zA-Z_])");
    if (chunk.rfind(kFunctionStart, 0) == 0 || chunk.rfind(kFunctionEnd, 0) == 0 ||
        chunk.rfind("<function ", 0) == 0 ||
        std::regex_search(chunk, func_prefix)) {
      is_tool_call = true;
    }
    if (chunk.rfind(kParameterStart, 0) == 0 ||
        chunk.rfind(kParameterEnd, 0) == 0) {
      is_tool_call = true;
    }

    if (current_call_id_.has_value() || chunk.rfind("<function", 0) == 0 ||
        chunk.rfind("<parameter", 0) == 0) {
      chunk = fix_incomplete_tag_in_chunk(chunk);
    }

    static const std::regex func_eq(R"(<function=([^>]+)>)");
    static const std::regex param_eq(R"(<parameter=([^>]+)>)");
    std::string processed = std::regex_replace(chunk, func_eq, R"(<function name="$1">)");
    processed = std::regex_replace(processed, param_eq, R"(<parameter name="$1">)");

    const std::string original_chunk = chunk;

    if (pre_inside_parameter_) {
      if (processed.rfind("</parameter>", 0) == 0) {
        const std::string body_text = pre_param_buffer_;
        defer_current_parameter_ = true;
        deferred_param_raw_value_ = body_text;
        pre_inside_parameter_ = false;
        pre_param_buffer_.clear();
        pre_current_param_name_.reset();
        return EscapeXml(body_text) + "</parameter>";
      }
      if (pre_param_buffer_.empty()) {
        std::string param_type =
            pre_current_param_name_.has_value()
                ? get_param_type(*pre_current_param_name_)
                : std::string("string");
        const bool is_object_type = (param_type == "object");
        const bool is_complex_type =
            (param_type == "array" || param_type == "arr" ||
             param_type == "sequence" || param_type.rfind("dict", 0) == 0 ||
             param_type.rfind("list", 0) == 0);
        const bool has_container_hint =
            original_chunk.find('[') != std::string::npos ||
            original_chunk.find('{') != std::string::npos ||
            original_chunk.find('(') != std::string::npos;
        bool need_defer = false;
        if (is_complex_type) {
          need_defer = true;
        } else if (is_object_type && has_container_hint &&
                   original_chunk.find('\'') != std::string::npos) {
          need_defer = true;
        }
        if (!need_defer) {
          pre_inside_parameter_ = false;
          return EscapeXml(original_chunk);
        }
      }
      pre_param_buffer_ += original_chunk;
      return "";
    }

    if (processed.rfind("<parameter name=", 0) == 0) {
      static const std::regex pname(R"RE(<parameter name="([^"]+)">)RE");
      std::smatch m;
      if (std::regex_search(processed, m, pname)) {
        pre_current_param_name_ = m[1].str();
      }
      pre_inside_parameter_ = true;
      pre_param_buffer_.clear();
      return processed;
    }

    if (!is_tool_call) {
      processed = EscapeXml(processed);
    }
    return processed;
  }

  // ── minimal XML event emitter (expat surrogate) ───────────────────────────
  void xml_flush_chardata() {
    if (xml_pending_.empty()) return;
    const std::string data = UnescapeXml(xml_pending_);
    xml_pending_.clear();
    char_data(data);
  }

  void xml_parse(const std::string& fragment) {
    std::size_t i = 0;
    while (i < fragment.size()) {
      if (fragment[i] == '<') {
        const std::size_t gt = fragment.find('>', i);
        if (gt == std::string::npos) {
          // Malformed: no closing '>' — treat remainder as text.
          xml_pending_ += fragment.substr(i);
          return;
        }
        const std::string tag = fragment.substr(i, gt - i + 1);
        xml_flush_chardata();
        dispatch_tag(tag);
        i = gt + 1;
      } else {
        const std::size_t next = fragment.find('<', i);
        const std::size_t stop =
            next == std::string::npos ? fragment.size() : next;
        xml_pending_ += fragment.substr(i, stop - i);
        i = stop;
      }
    }
  }

  void dispatch_tag(const std::string& tag) {
    if (tag.size() < 2) return;
    if (tag[1] == '/') {
      std::string name = tag.substr(2, tag.size() - 3);
      name = Strip(name);
      end_element(name);
      return;
    }
    std::string inner = tag.substr(1, tag.size() - 2);
    bool self_close = false;
    if (!inner.empty() && inner.back() == '/') {
      self_close = true;
      inner.pop_back();
    }
    // element name up to first whitespace.
    std::size_t np = 0;
    while (np < inner.size() && !IsSpaceCh(inner[np])) ++np;
    const std::string name = inner.substr(0, np);
    // parse attrs: key="value"
    std::vector<std::pair<std::string, std::string>> attrs;
    static const std::regex attr_re(R"RE((\w+)\s*=\s*"([^"]*)")RE");
    const std::string rest = inner.substr(np);
    for (auto it = std::sregex_iterator(rest.begin(), rest.end(), attr_re);
         it != std::sregex_iterator(); ++it) {
      attrs.emplace_back((*it)[1].str(), UnescapeXml((*it)[2].str()));
    }
    start_element(name, attrs);
    if (self_close) end_element(name);
  }

  static std::optional<std::string> attr_get(
      const std::vector<std::pair<std::string, std::string>>& attrs,
      const std::string& key) {
    for (const auto& [k, v] : attrs) {
      if (k == key) return v;
    }
    return std::nullopt;
  }

  std::optional<std::string> extract_function_name(
      const std::string& name,
      const std::vector<std::pair<std::string, std::string>>& attrs) {
    auto a = attr_get(attrs, "name");
    if (a.has_value()) return a;
    const std::size_t eq = name.find('=');
    if (eq != std::string::npos && name.substr(0, eq) == "function") {
      return name.substr(eq + 1);
    }
    return std::nullopt;
  }

  std::optional<std::string> extract_parameter_name(
      const std::string& name,
      const std::vector<std::pair<std::string, std::string>>& attrs) {
    auto a = attr_get(attrs, "name");
    if (a.has_value()) return a;
    const std::size_t eq = name.find('=');
    if (eq != std::string::npos && name.substr(0, eq) == "parameter") {
      return name.substr(eq + 1);
    }
    return std::nullopt;
  }

  // ── auto-close (step3p5_tool_parser.py:791-814) ───────────────────────────
  void auto_close_open_parameter_if_needed(const std::string& incoming_tag) {
    if (current_param_name_.has_value()) end_element("parameter");
    if ((incoming_tag == "function" || incoming_tag == "tool_call") &&
        current_function_name_.has_value()) {
      end_element("function");
    }
    if (incoming_tag == "tool_call" && current_call_id_.has_value()) {
      end_element("tool_call");
    }
  }

  // ── start element (step3p5_tool_parser.py:816-904) ────────────────────────
  void start_element(
      const std::string& name,
      const std::vector<std::pair<std::string, std::string>>& attrs) {
    if (name == "root") return;

    if (name == "tool_call") {
      auto_close_open_parameter_if_needed("tool_call");
      parameters_ = ojson::object();
      current_call_id_ = make_tool_call_id();
      current_param_is_first_ = true;
      tool_call_index_ += 1;
    } else if (name.rfind("function", 0) == 0) {
      if (!current_call_id_.has_value()) start_element("tool_call", {});
      auto_close_open_parameter_if_needed("function");
      std::optional<std::string> function_name =
          extract_function_name(name, attrs);
      current_function_name_ = function_name;
      current_function_open_ = true;
      if (function_name.has_value()) {
        emit_delta(tool_delta(function_name, ""));
      }
    } else if (name.rfind("parameter", 0) == 0) {
      auto_close_open_parameter_if_needed("parameter");
      std::optional<std::string> param_name =
          extract_parameter_name(name, attrs);
      current_param_name_ = param_name;
      current_param_value_.clear();
      current_param_value_converted_.clear();
      start_quote_emitted_ = false;

      if (param_name.has_value()) {
        if (parameters_.empty()) {
          emit_delta(tool_delta(std::nullopt, "{\"" + *param_name + "\": "));
          current_param_is_first_ = true;
        } else {
          emit_delta(tool_delta(std::nullopt, ", \"" + *param_name + "\": "));
          current_param_is_first_ = false;
        }
      }
    }
  }

  // ── character data (step3p5_tool_parser.py:906-979) ───────────────────────
  void char_data(const std::string& data_in) {
    std::string data = data_in;
    if (data.empty() || !current_param_name_.has_value()) return;

    if (defer_current_parameter_) {
      std::string original = data;
      if (should_emit_end_newline_) {
        original = "\n" + original;
        should_emit_end_newline_ = false;
      }
      if (!original.empty() && original.back() == '\n') {
        should_emit_end_newline_ = true;
        original.pop_back();
      }
      current_param_value_ += original;
      return;
    }

    const std::string param_type = get_param_type(*current_param_name_);

    if (current_param_value_.empty() && data.rfind('\n', 0) == 0) {
      data = data.substr(1);
    }

    if (IsStringType(param_type) && !start_quote_emitted_) {
      emit_delta(tool_delta(std::nullopt, "\""));
      start_quote_emitted_ = true;
    }

    if (data.empty()) return;

    std::string original = data;
    if (should_emit_end_newline_) {
      original = "\n" + original;
      should_emit_end_newline_ = false;
    }
    if (!original.empty() && original.back() == '\n') {
      should_emit_end_newline_ = true;
      original.pop_back();
    }
    current_param_value_ += original;

    const ojson converted = convert_param_value(current_param_value_, param_type);
    const std::string output_data = convert_for_json_streaming(converted, param_type);
    std::string delta_data;
    if (output_data.size() > current_param_value_converted_.size()) {
      delta_data = output_data.substr(current_param_value_converted_.size());
    }
    current_param_value_converted_ = output_data;
    emit_delta(tool_delta(std::nullopt, delta_data));
  }

  // ── end element (step3p5_tool_parser.py:981-1154) ─────────────────────────
  void end_element(const std::string& name) {
    if (name == "root") return;

    if ((name.rfind("function", 0) == 0 || name == "tool_call") &&
        current_param_name_.has_value()) {
      auto_close_open_parameter_if_needed("");
    }

    if (name.rfind("parameter", 0) == 0 && current_param_name_.has_value()) {
      const std::string param_name = *current_param_name_;
      const std::string param_value = current_param_value_;

      if (defer_current_parameter_) {
        std::string raw_text =
            !deferred_param_raw_value_.empty() ? deferred_param_raw_value_
                                               : param_value;
        std::string output_arguments;
        ojson parsed_value;
        std::string raw_for_parse =
            should_emit_end_newline_ ? raw_text + "\n" : raw_text;
        ojson literal;
        if (SafeLiteralEval(raw_for_parse, literal)) {
          parsed_value = literal;
          output_arguments = JsonDumps(literal);
        } else {
          parsed_value = ojson(raw_text);
          output_arguments = JsonDumps(ojson(raw_text));
        }
        emit_delta(tool_delta(std::nullopt, output_arguments));

        should_emit_end_newline_ = false;
        parameters_[param_name] = parsed_value;
        current_param_name_.reset();
        current_param_value_.clear();
        current_param_value_converted_.clear();
        start_quote_emitted_ = false;
        defer_current_parameter_ = false;
        deferred_param_raw_value_.clear();
        return;
      }

      const std::string param_type = get_param_type(param_name);
      const ojson converted_value = convert_param_value(param_value, param_type);

      if (IsStringType(param_type)) {
        if (param_value.empty() && !start_quote_emitted_) {
          emit_delta(tool_delta(std::nullopt, "\"\""));
        } else {
          emit_delta(tool_delta(std::nullopt, "\""));
        }
      }

      should_emit_end_newline_ = false;
      parameters_[param_name] = converted_value;
      current_param_name_.reset();
      current_param_value_.clear();
      current_param_value_converted_.clear();
      start_quote_emitted_ = false;
    } else if (name.rfind("function", 0) == 0) {
      if (!parameters_.empty()) {
        emit_delta(tool_delta(std::nullopt, "}"));
      } else {
        emit_delta(tool_delta(std::nullopt, "{}"));
      }
      current_function_open_ = false;
      current_function_name_.reset();
    } else if (name == "tool_call") {
      if (current_function_open_) {
        if (current_param_name_.has_value()) end_element("parameter");
        end_element("function");
      }
      emit_delta(tool_delta(std::nullopt, ""));

      if (!Strip(text_content_buffer_).empty()) {
        DeltaMessage m;
        m.content = text_content_buffer_;
        emit_delta(m);
      }

      reset_xml_parser_after_tool_call();
    }
  }

  // ── reset after a tool_call (step3p5_tool_parser.py:1333-1363) ────────────
  void reset_xml_parser_after_tool_call() {
    xml_recreate();
    if (current_call_id_.has_value()) last_completed_call_id_ = current_call_id_;
    current_call_id_.reset();
    current_function_name_.reset();
    current_function_open_ = false;
    parameters_ = ojson::object();
    current_param_name_.reset();
    current_param_value_.clear();
    current_param_value_converted_.clear();
    current_param_is_first_ = false;
    should_emit_end_newline_ = false;
    start_quote_emitted_ = false;
    text_content_buffer_.clear();
    pre_inside_parameter_ = false;
    pre_param_buffer_.clear();
    pre_current_param_name_.reset();
    defer_current_parameter_ = false;
    deferred_param_raw_value_.clear();
  }

  // ── delta merge (step3p5_tool_parser.py:590-652) ──────────────────────────
  DeltaMessage merge_new_deltas(std::size_t initial_count) {
    if (deltas_.size() <= initial_count) return DeltaMessage{};
    if (deltas_.size() - initial_count == 1) return deltas_[initial_count];

    std::vector<DeltaToolCall> merged_tool_calls;
    std::string merged_content;

    for (std::size_t di = initial_count; di < deltas_.size(); ++di) {
      const DeltaMessage& delta = deltas_[di];
      if (delta.content.has_value()) merged_content += *delta.content;
      if (!delta.tool_calls.has_value()) continue;
      for (const DeltaToolCall& tc : *delta.tool_calls) {
        DeltaToolCall* existing = nullptr;
        for (DeltaToolCall& e : merged_tool_calls) {
          if (e.id == tc.id) {
            existing = &e;
            break;
          }
        }
        if (existing != nullptr) {
          if (tc.function.name.has_value() && !tc.function.name->empty()) {
            existing->function.name = tc.function.name;
          }
          if (tc.function.arguments.has_value()) {
            if (!existing->function.arguments.has_value()) {
              existing->function.arguments = std::string("");
            }
            *existing->function.arguments += *tc.function.arguments;
          }
          if (tc.type.has_value()) existing->type = tc.type;
        } else {
          merged_tool_calls.push_back(tc);
        }
      }
    }

    DeltaMessage out;
    if (!merged_content.empty()) out.content = merged_content;
    if (!merged_tool_calls.empty()) out.tool_calls = std::move(merged_tool_calls);
    return out;
  }
};

}  // namespace step3p5_detail

// ── the public parser ───────────────────────────────────────────────────────
Step3p5ToolParser::Step3p5ToolParser()
    : parser_(std::make_unique<step3p5_detail::StreamingXmlToolCallParser>()) {}

Step3p5ToolParser::~Step3p5ToolParser() = default;

ExtractedToolCallInformation Step3p5ToolParser::extract_tool_calls(
    const std::string& model_output, const ChatCompletionRequest& request) {
  parser_->reset_streaming_state();
  parser_->set_tools(request.tools.has_value() ? &*request.tools : nullptr);
  DeltaMessage result = parser_->parse_single_streaming_chunks(model_output);

  if (!result.tool_calls.has_value() || result.tool_calls->empty()) {
    return ExtractedToolCallInformation{false, {}, result.content};
  }

  std::vector<ToolCall> tool_calls;
  for (const DeltaToolCall& tc : *result.tool_calls) {
    if (tc.function.name.has_value() && !tc.function.name->empty()) {
      ToolCall t;
      t.id = tc.id.value_or("");
      t.type = tc.type.value_or("function");
      t.function.name = *tc.function.name;
      t.function.arguments = tc.function.arguments.value_or("");
      tool_calls.push_back(std::move(t));
    }
  }

  ExtractedToolCallInformation info;
  info.tools_called = !tool_calls.empty();
  info.tool_calls = std::move(tool_calls);
  info.content = result.content;
  return info;
}

std::optional<DeltaMessage> Step3p5ToolParser::extract_tool_calls_streaming(
    const std::string& previous_text, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& request) {
  if (previous_text.empty()) {
    parser_->reset_streaming_state();
    parser_->set_tools(request.tools.has_value() ? &*request.tools : nullptr);
  } else {
    // Keep the tools pointer fresh (request outlives the call).
    parser_->set_tools(request.tools.has_value() ? &*request.tools : nullptr);
  }

  // DEVIATION 2: no token ids -> an empty delta contributes nothing.
  if (delta_text.empty()) return std::nullopt;

  DeltaMessage result = parser_->parse_single_streaming_chunks(delta_text);
  if (!result.content.has_value() &&
      (!result.tool_calls.has_value() || result.tool_calls->empty())) {
    return std::nullopt;
  }
  return result;
}

}  // namespace vllm::entrypoints::openai
