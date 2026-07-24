// Shared pythonic-call-grammar core. See pythonic_core.h for provenance +
// deviations. Extracted verbatim from the former anonymous-namespace helpers of
// pythonic.cpp so PythonicToolParser and Olmo3ToolParser share one grammar.
#include "vllm/entrypoints/openai/tool_parsers/pythonic_core.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vllm::entrypoints::openai::pythonic_core {

namespace {

using ojson = nlohmann::ordered_json;

// Thrown by the recursive-descent parser on any malformed / non-literal input.
// Mirrors the union of Python's SyntaxError (ast.parse) and UnexpectedAstError
// (get_parameter_value / handle_single_tool).
struct ParseError {};

bool IsSpaceCh(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
bool IsAlpha(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool IsDigit(char c) { return c >= '0' && c <= '9'; }
bool IsIdentChar(char c) { return IsAlpha(c) || IsDigit(c) || c == '_'; }

// Strict recursive-descent parser for the pythonic call grammar. Stands in for
// upstream's Python `ast` (utils.py:handle_single_tool + get_parameter_value).
class PythonicParser {
 public:
  explicit PythonicParser(const std::string& text) : s_(text) {}

  // Top level: parse the WHOLE input as `[ call (, call)* ]` (>= 1 call), then
  // require only trailing whitespace. Mirrors module.body[0].value being an
  // ast.List all of whose elts are ast.Call.
  std::vector<PyCall> ParseCallList() {
    SkipWs();
    Expect('[');
    SkipWs();
    std::vector<PyCall> calls;
    if (Peek() == ']') {
      // Empty list: upstream's TOOL_CALL_REGEX requires >= 1 call, so `[]` is
      // never a tool call. Reject.
      throw ParseError{};
    }
    for (;;) {
      calls.push_back(ParseCall());
      SkipWs();
      const char c = Peek();
      if (c == ',') {
        ++pos_;
        SkipWs();
        if (Peek() == ']') break;  // trailing comma
        continue;
      }
      if (c == ']') break;
      throw ParseError{};
    }
    Expect(']');
    SkipWs();
    if (pos_ != s_.size()) throw ParseError{};  // trailing junk
    return calls;
  }

  // Parse the WHOLE input as a single Python literal (safe_literal_eval).
  ojson ParseLiteralTop() {
    SkipWs();
    ojson value = ParseValue();
    SkipWs();
    if (pos_ != s_.size()) throw ParseError{};  // trailing junk
    return value;
  }

 private:
  const std::string& s_;
  std::size_t pos_ = 0;

  char Peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  void SkipWs() {
    while (pos_ < s_.size() && IsSpaceCh(s_[pos_])) ++pos_;
  }
  void Expect(char c) {
    if (Peek() != c) throw ParseError{};
    ++pos_;
  }

  // [a-zA-Z][a-zA-Z0-9_]* (letter-start; matches the regex gate's identifier
  // class). Used for function-name segments and keyword-argument names.
  std::string ParseIdentifier() {
    if (!IsAlpha(Peek())) throw ParseError{};
    const std::size_t start = pos_;
    while (pos_ < s_.size() && IsIdentChar(s_[pos_])) ++pos_;
    return s_.substr(start, pos_ - start);
  }

  // A (possibly dotted) call target: `a` or `a.b.c` (utils.py:
  // _ast_callable_dotted_name). The dotted form is preserved in the name.
  std::string ParseDottedName() {
    std::string name = ParseIdentifier();
    while (Peek() == '.') {
      ++pos_;
      name += '.';
      name += ParseIdentifier();
    }
    return name;
  }

  PyCall ParseCall() {
    SkipWs();
    PyCall call;
    call.name = ParseDottedName();
    call.arguments = ojson::object();
    SkipWs();
    Expect('(');
    SkipWs();
    if (Peek() == ')') {
      ++pos_;
      return call;
    }
    for (;;) {
      SkipWs();
      const std::string arg_name = ParseIdentifier();
      SkipWs();
      Expect('=');
      SkipWs();
      call.arguments[arg_name] = ParseValue();
      SkipWs();
      const char c = Peek();
      if (c == ',') {
        ++pos_;
        SkipWs();
        if (Peek() == ')') break;  // trailing comma
        continue;
      }
      if (c == ')') break;
      throw ParseError{};
    }
    Expect(')');
    return call;
  }

  // A Python literal (get_parameter_value): string / number / list / dict /
  // True|False|None (+ the json-style name literals true|false|null).
  ojson ParseValue() {
    SkipWs();
    const char c = Peek();
    if (c == '\'' || c == '"') return ParseString();
    if (c == '[') return ParseList();
    if (c == '{') return ParseDict();
    if (IsDigit(c)) return ParseNumber();
    if (IsAlpha(c)) return ParseNameLiteral();
    // Leading '-' (ast.UnaryOp) and everything else is rejected, matching
    // get_parameter_value.
    throw ParseError{};
  }

  // Python string literal in either quote style. Decodes Python escape
  // sequences into the raw value; nlohmann re-escapes on dump (so `\'` -> ' and
  // `\"` -> " in the value, exactly as Python's str literal evaluation).
  ojson ParseString() {
    const char quote = s_[pos_];
    ++pos_;
    std::string out;
    while (pos_ < s_.size()) {
      const char ch = s_[pos_];
      if (ch == '\\') {
        if (pos_ + 1 >= s_.size()) throw ParseError{};  // dangling escape
        const char esc = s_[pos_ + 1];
        switch (esc) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'v': out += '\v'; break;
          case 'a': out += '\a'; break;
          case '0': out += '\0'; break;
          case '\\': out += '\\'; break;
          case '\'': out += '\''; break;
          case '"': out += '"'; break;
          // Python keeps the backslash for an unrecognized escape.
          default:
            out += '\\';
            out += esc;
            break;
        }
        pos_ += 2;
        continue;
      }
      if (ch == quote) {
        ++pos_;
        return ojson(out);
      }
      out += ch;
      ++pos_;
    }
    throw ParseError{};  // unterminated string
  }

  ojson ParseNumber() {
    const std::size_t start = pos_;
    bool is_float = false;
    while (pos_ < s_.size() && IsDigit(s_[pos_])) ++pos_;
    if (Peek() == '.') {
      is_float = true;
      ++pos_;
      while (pos_ < s_.size() && IsDigit(s_[pos_])) ++pos_;
    }
    if (Peek() == 'e' || Peek() == 'E') {
      is_float = true;
      ++pos_;
      if (Peek() == '+' || Peek() == '-') ++pos_;
      if (!IsDigit(Peek())) throw ParseError{};
      while (pos_ < s_.size() && IsDigit(s_[pos_])) ++pos_;
    }
    const std::string tok = s_.substr(start, pos_ - start);
    if (tok.empty()) throw ParseError{};
    try {
      if (is_float) return ojson(std::stod(tok));
      return ojson(static_cast<std::int64_t>(std::stoll(tok)));
    } catch (const std::exception&) {
      throw ParseError{};
    }
  }

  ojson ParseList() {
    Expect('[');
    SkipWs();
    ojson arr = ojson::array();
    if (Peek() == ']') {
      ++pos_;
      return arr;
    }
    for (;;) {
      arr.push_back(ParseValue());
      SkipWs();
      const char c = Peek();
      if (c == ',') {
        ++pos_;
        SkipWs();
        if (Peek() == ']') break;
        continue;
      }
      if (c == ']') break;
      throw ParseError{};
    }
    Expect(']');
    return arr;
  }

  // A dict literal. Keys must be literals (Python ast.Constant); we accept
  // string keys (the only kind the models emit) and numeric keys (stringified,
  // as json.dumps would). A `{a, b}` set literal fails here (no ':'), matching
  // upstream's rejection of ast.Set.
  ojson ParseDict() {
    Expect('{');
    SkipWs();
    ojson obj = ojson::object();
    if (Peek() == '}') {
      ++pos_;
      return obj;
    }
    for (;;) {
      SkipWs();
      std::string key;
      const char kc = Peek();
      if (kc == '\'' || kc == '"') {
        key = ParseString().get<std::string>();
      } else if (IsDigit(kc)) {
        const ojson num = ParseNumber();
        key = num.dump();
      } else {
        throw ParseError{};
      }
      SkipWs();
      Expect(':');
      SkipWs();
      obj[key] = ParseValue();
      SkipWs();
      const char c = Peek();
      if (c == ',') {
        ++pos_;
        SkipWs();
        if (Peek() == '}') break;
        continue;
      }
      if (c == '}') break;
      throw ParseError{};
    }
    Expect('}');
    return obj;
  }

  ojson ParseNameLiteral() {
    const std::size_t start = pos_;
    while (pos_ < s_.size() && IsIdentChar(s_[pos_])) ++pos_;
    const std::string id = s_.substr(start, pos_ - start);
    if (id == "True" || id == "true") return ojson(true);
    if (id == "False" || id == "false") return ojson(false);
    if (id == "None" || id == "null") return ojson(nullptr);
    // Any other bare identifier is not a literal -> reject.
    throw ParseError{};
  }
};

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsWs(char c) { return IsSpaceCh(c); }

std::string Rstrip(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0 && IsWs(s[e - 1])) --e;
  return s.substr(0, e);
}

}  // namespace

std::optional<std::vector<PyCall>> parse_call_list(const std::string& text) {
  try {
    PythonicParser p(text);
    return p.ParseCallList();
  } catch (const ParseError&) {
    return std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<nlohmann::ordered_json> parse_literal(const std::string& text) {
  try {
    PythonicParser p(text);
    return p.ParseLiteralTop();
  } catch (const ParseError&) {
    return std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<std::pair<std::string, std::string>> make_valid_python(
    const std::string& text_in) {
  std::vector<char> bracket_stack;
  for (std::size_t index = 0; index < text_in.size(); ++index) {
    const char ch = text_in[index];
    if (ch == '[' || ch == '(' || ch == '{') {
      bracket_stack.push_back(ch);
    } else if (ch == ']') {
      if (bracket_stack.empty() || bracket_stack.back() != '[') return std::nullopt;
      bracket_stack.pop_back();
    } else if (ch == ')') {
      if (bracket_stack.empty() || bracket_stack.back() != '(') return std::nullopt;
      bracket_stack.pop_back();
    } else if (ch == '}') {
      if (bracket_stack.empty() || bracket_stack.back() != '{') return std::nullopt;
      bracket_stack.pop_back();
    } else if (ch == '\'' || ch == '"') {
      if (!bracket_stack.empty() && bracket_stack.back() == ch) {
        if (index > 0 && text_in[index - 1] == '\\') {
          // escaped quote: not a closer
        } else {
          bracket_stack.pop_back();
        }
      } else if (!bracket_stack.empty() &&
                 (bracket_stack.back() == '\'' || bracket_stack.back() == '"')) {
        // inside a string of the other quote style: literal char
      } else {
        bracket_stack.push_back(ch);
      }
    }
  }
  // Note: upstream raises UnexpectedAstError on mismatched brackets; we return
  // nullopt above (the caller treats a raised/None result identically -> no
  // delta this step).

  std::string text = Rstrip(text_in);
  if (!text.empty() && (text.back() == '=' || text.back() == ':')) {
    return std::nullopt;
  }
  if (!bracket_stack.empty() && bracket_stack.back() == '{') {
    const std::size_t brace = text.rfind('{');
    const std::string trailing = brace == std::string::npos ? std::string()
                                                            : text.substr(0, brace);
    const std::size_t num_keys =
        std::count(trailing.begin(), trailing.end(), ':');
    const std::size_t num_values =
        std::count(trailing.begin(), trailing.end(), ',');
    if (num_keys <= num_values) return std::nullopt;
  }
  if (!bracket_stack.empty() && bracket_stack.back() == '(') {
    const std::size_t paren = text.rfind('(');
    const std::string trailing = paren == std::string::npos ? std::string()
                                                           : text.substr(0, paren);
    const std::size_t num_names =
        std::count(trailing.begin(), trailing.end(), '=');
    const std::size_t num_vals =
        std::count(trailing.begin(), trailing.end(), ',');
    if (num_names <= num_vals) return std::nullopt;
  }
  if (!text.empty() && text.back() == ',') {
    text.pop_back();
  }
  if (!bracket_stack.empty() && bracket_stack.back() == '[' &&
      !EndsWith(text, "[") && !EndsWith(text, ")")) {
    return std::nullopt;
  }

  std::string added_text;
  for (auto it = bracket_stack.rbegin(); it != bracket_stack.rend(); ++it) {
    switch (*it) {
      case '[': added_text += ']'; break;
      case '(': added_text += ')'; break;
      case '{': added_text += '}'; break;
      case '\'': added_text += '\''; break;
      case '"': added_text += '"'; break;
      default: break;
    }
  }

  return std::make_pair(text + added_text, added_text);
}

std::optional<DeltaToolCall> compute_tool_delta(const std::string& previously_sent,
                                                const std::string& call_id,
                                                const std::string& call_name,
                                                const std::string& call_args,
                                                int index,
                                                const std::string& withheld_suffix) {
  std::string new_args = call_args;
  if (!withheld_suffix.empty()) {
    if (!EndsWith(new_args, withheld_suffix)) throw ToolDeltaError{};
    new_args = new_args.substr(0, new_args.size() - withheld_suffix.size());
  }
  if (previously_sent.empty()) {
    DeltaToolCall d;
    d.id = call_id;
    d.type = "function";
    d.index = index;
    d.function.name = call_name;
    d.function.arguments = new_args;
    return d;
  }
  if (new_args.size() <= previously_sent.size()) return std::nullopt;
  std::string arg_diff = new_args.substr(previously_sent.size());
  if (arg_diff.empty()) return std::nullopt;
  DeltaToolCall d;
  d.index = index;
  d.function.arguments = std::move(arg_diff);
  return d;
}

}  // namespace vllm::entrypoints::openai::pythonic_core
