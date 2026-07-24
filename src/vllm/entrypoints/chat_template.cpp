// Ported from: vllm/entrypoints/chat_utils.py @ e24d1b24 (see chat_template.h
// for the deviation note: this minja-subset Jinja engine is an ORIGINAL
// component — vLLM delegates to transformers' Python Jinja2, which we cannot
// depend on at runtime. Whitespace policy mirrors transformers'
// ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True) @
// transformers/utils/chat_template_utils.py:474).
#include "vllm/entrypoints/chat_template.h"

#include "vllm/model_executor/model_loader/gguf_reader.h"

#include <cctype>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::entrypoints {
namespace {

// Transformers renders chat templates with these two block-whitespace flags on.
constexpr bool kTrimBlocks = true;
constexpr bool kLstripBlocks = true;

// ─── Runtime value ───────────────────────────────────────────────────────────
struct Value {
  enum class Kind {
    kUndefined,
    kNull,
    kBool,
    kInt,
    kString,
    kList,
    kObject,
    kJson  // opaque JSON subtree (tool schemas); rendered via `| tojson`.
  };
  Kind kind = Kind::kUndefined;
  bool b = false;
  long long i = 0;
  std::string s;
  std::shared_ptr<std::vector<Value>> list;
  std::shared_ptr<std::map<std::string, Value>> obj;
  std::shared_ptr<nlohmann::json> json;

  static Value Undefined() { return Value{}; }
  static Value Null() {
    Value v;
    v.kind = Kind::kNull;
    return v;
  }
  static Value Bool(bool x) {
    Value v;
    v.kind = Kind::kBool;
    v.b = x;
    return v;
  }
  static Value Int(long long x) {
    Value v;
    v.kind = Kind::kInt;
    v.i = x;
    return v;
  }
  static Value Str(std::string x) {
    Value v;
    v.kind = Kind::kString;
    v.s = std::move(x);
    return v;
  }
  static Value List(std::vector<Value> x) {
    Value v;
    v.kind = Kind::kList;
    v.list = std::make_shared<std::vector<Value>>(std::move(x));
    return v;
  }
  static Value Object(std::map<std::string, Value> x) {
    Value v;
    v.kind = Kind::kObject;
    v.obj = std::make_shared<std::map<std::string, Value>>(std::move(x));
    return v;
  }
  static Value Json(nlohmann::json x) {
    Value v;
    v.kind = Kind::kJson;
    v.json = std::make_shared<nlohmann::json>(std::move(x));
    return v;
  }
};

bool Truthy(const Value& v) {
  switch (v.kind) {
    case Value::Kind::kUndefined:
    case Value::Kind::kNull:
      return false;
    case Value::Kind::kBool:
      return v.b;
    case Value::Kind::kInt:
      return v.i != 0;
    case Value::Kind::kString:
      return !v.s.empty();
    case Value::Kind::kList:
      return !v.list->empty();
    case Value::Kind::kObject:
      return true;
    case Value::Kind::kJson:
      // Python truthiness of the JSON value (null/empty container → false).
      return !v.json->is_null() &&
             !((v.json->is_object() || v.json->is_array() ||
                v.json->is_string()) &&
               v.json->empty());
  }
  return false;
}

std::string ToOutput(const Value& v, const std::string& where);

bool Equals(const Value& a, const Value& b) {
  if (a.kind != b.kind) {
    // int/bool cross-compare like Python (True==1); keep minimal: only same kind.
    return false;
  }
  switch (a.kind) {
    case Value::Kind::kUndefined:
    case Value::Kind::kNull:
      return true;
    case Value::Kind::kBool:
      return a.b == b.b;
    case Value::Kind::kInt:
      return a.i == b.i;
    case Value::Kind::kString:
      return a.s == b.s;
    default:
      return false;  // list/object identity not needed by chat templates
  }
}

// ─── Error helpers ───────────────────────────────────────────────────────────
std::string LineCol(const std::string& src, std::size_t pos) {
  int line = 1, col = 1;
  for (std::size_t k = 0; k < pos && k < src.size(); ++k) {
    if (src[k] == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
  }
  std::ostringstream os;
  os << line << ':' << col;
  return os.str();
}

[[noreturn]] void Fail(const std::string& msg) {
  throw ChatTemplateError(msg);
}

// ─── Expression AST ──────────────────────────────────────────────────────────
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;
struct Expr {
  enum class K { kLit, kVar, kMember, kIndex, kBin, kUnary, kMethod, kFilter };
  K k;
  Value lit;                    // kLit
  std::string name;             // kVar / kMember(name) / kMethod(name) / kBin,kUnary(op)
  ExprPtr a, b;                 // operands / object / index
  std::vector<ExprPtr> args;    // kMethod
};

// ─── Statement AST ───────────────────────────────────────────────────────────
struct Stmt;
using StmtPtr = std::shared_ptr<Stmt>;
struct IfBranch {
  ExprPtr cond;
  std::vector<StmtPtr> body;
};
struct Stmt {
  enum class K { kText, kOutput, kFor, kIf, kSet } k;
  std::string text;                 // kText
  ExprPtr expr;                     // kOutput / kSet(value) / kFor(iterable)
  std::string var;                  // kFor / kSet variable
  std::vector<StmtPtr> body;        // kFor body
  std::vector<IfBranch> branches;   // kIf
  std::vector<StmtPtr> elsebody;    // kIf else
};

// ─── Expression lexer ────────────────────────────────────────────────────────
struct ETok {
  enum class T { kName, kString, kInt, kOp, kEnd } t = T::kEnd;
  std::string s;   // name / op text / decoded string
  long long i = 0;
};

class ExprLexer {
 public:
  ExprLexer(const std::string& src, const std::string& where)
      : src_(src), where_(where) {}

  ETok Next() {
    while (p_ < src_.size() &&
           (src_[p_] == ' ' || src_[p_] == '\t' || src_[p_] == '\n' ||
            src_[p_] == '\r')) {
      ++p_;
    }
    if (p_ >= src_.size()) return ETok{ETok::T::kEnd, "", 0};
    char c = src_[p_];
    if (c == '\'' || c == '"') return LexString(c);
    if (std::isdigit(static_cast<unsigned char>(c))) return LexInt();
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') return LexName();
    return LexOp();
  }

 private:
  ETok LexString(char q) {
    ++p_;  // opening quote
    std::string out;
    while (p_ < src_.size() && src_[p_] != q) {
      char c = src_[p_++];
      if (c == '\\' && p_ < src_.size()) {
        char e = src_[p_++];
        switch (e) {
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          case '\\': out.push_back('\\'); break;
          case '\'': out.push_back('\''); break;
          case '"': out.push_back('"'); break;
          default: out.push_back(e); break;
        }
      } else {
        out.push_back(c);
      }
    }
    if (p_ >= src_.size()) Fail("unterminated string literal at " + where_);
    ++p_;  // closing quote
    return ETok{ETok::T::kString, std::move(out), 0};
  }

  ETok LexInt() {
    std::size_t s = p_;
    while (p_ < src_.size() &&
           std::isdigit(static_cast<unsigned char>(src_[p_]))) {
      ++p_;
    }
    ETok t;
    t.t = ETok::T::kInt;
    t.i = std::stoll(src_.substr(s, p_ - s));
    return t;
  }

  ETok LexName() {
    std::size_t s = p_;
    while (p_ < src_.size() &&
           (std::isalnum(static_cast<unsigned char>(src_[p_])) ||
            src_[p_] == '_')) {
      ++p_;
    }
    return ETok{ETok::T::kName, src_.substr(s, p_ - s), 0};
  }

  ETok LexOp() {
    static const char* kTwo[] = {"==", "!="};
    for (const char* op : kTwo) {
      if (src_.compare(p_, 2, op) == 0) {
        p_ += 2;
        return ETok{ETok::T::kOp, op, 0};
      }
    }
    char c = src_[p_++];
    std::string one(1, c);
    // Recognized single-char operators/punctuation.
    if (std::string("()[].,+-~|").find(c) == std::string::npos) {
      Fail("unexpected character '" + one + "' in expression at " + where_);
    }
    return ETok{ETok::T::kOp, one, 0};
  }

  const std::string& src_;
  const std::string& where_;
  std::size_t p_ = 0;
};

// ─── Expression parser (precedence: or < and < not < cmp < add/concat < unary
//     < postfix < primary) ──────────────────────────────────────────────────
class ExprParser {
 public:
  ExprParser(const std::string& src, const std::string& where)
      : lex_(src, where), where_(where) {
    cur_ = lex_.Next();
  }

  ExprPtr Parse() {
    ExprPtr e = ParseOr();
    if (cur_.t != ETok::T::kEnd) {
      Fail("trailing tokens in expression at " + where_);
    }
    return e;
  }

 private:
  const ETok& Cur() const { return cur_; }
  void Advance() { cur_ = lex_.Next(); }
  bool IsOp(const char* o) const {
    return cur_.t == ETok::T::kOp && cur_.s == o;
  }
  bool IsKw(const char* k) const {
    return cur_.t == ETok::T::kName && cur_.s == k;
  }

  ExprPtr ParseOr() {
    ExprPtr l = ParseAnd();
    while (IsKw("or")) {
      Advance();
      ExprPtr r = ParseAnd();
      l = Bin("or", l, r);
    }
    return l;
  }
  ExprPtr ParseAnd() {
    ExprPtr l = ParseNot();
    while (IsKw("and")) {
      Advance();
      ExprPtr r = ParseNot();
      l = Bin("and", l, r);
    }
    return l;
  }
  ExprPtr ParseNot() {
    if (IsKw("not")) {
      Advance();
      ExprPtr e = ParseNot();
      auto n = std::make_shared<Expr>();
      n->k = Expr::K::kUnary;
      n->name = "not";
      n->a = e;
      return n;
    }
    return ParseCompare();
  }
  ExprPtr ParseCompare() {
    ExprPtr l = ParseAddConcat();
    for (;;) {
      if (IsOp("==") || IsOp("!=")) {
        std::string op = cur_.s;
        Advance();
        l = Bin(op, l, ParseAddConcat());
      } else if (IsKw("in")) {
        Advance();
        l = Bin("in", l, ParseAddConcat());
      } else if (IsKw("not")) {
        // "not in" comparison.
        Advance();
        if (!IsKw("in")) Fail("expected 'in' after 'not' at " + where_);
        Advance();
        l = Bin("notin", l, ParseAddConcat());
      } else {
        break;
      }
    }
    return l;
  }
  ExprPtr ParseAddConcat() {
    ExprPtr l = ParseUnary();
    while (IsOp("+") || IsOp("-") || IsOp("~")) {
      std::string op = cur_.s;
      Advance();
      l = Bin(op, l, ParseUnary());
    }
    return l;
  }
  ExprPtr ParseUnary() {
    if (IsOp("-")) {
      Advance();
      auto n = std::make_shared<Expr>();
      n->k = Expr::K::kUnary;
      n->name = "neg";
      n->a = ParseUnary();
      return n;
    }
    return ParsePostfix();
  }
  ExprPtr ParsePostfix() {
    ExprPtr e = ParsePrimary();
    for (;;) {
      if (IsOp(".")) {
        Advance();
        if (cur_.t != ETok::T::kName) Fail("expected name after '.' at " + where_);
        std::string name = cur_.s;
        Advance();
        if (IsOp("(")) {
          // Method call: strip / lstrip / rstrip (+ optional char arg).
          Advance();
          std::vector<ExprPtr> args;
          if (!IsOp(")")) {
            args.push_back(ParseOr());
            while (IsOp(",")) {
              Advance();
              args.push_back(ParseOr());
            }
          }
          if (!IsOp(")")) Fail("expected ')' after method args at " + where_);
          Advance();
          auto m = std::make_shared<Expr>();
          m->k = Expr::K::kMethod;
          m->a = e;
          m->name = name;
          m->args = std::move(args);
          e = m;
        } else {
          auto m = std::make_shared<Expr>();
          m->k = Expr::K::kMember;
          m->a = e;
          m->name = name;
          e = m;
        }
      } else if (IsOp("[")) {
        Advance();
        ExprPtr idx = ParseOr();
        if (!IsOp("]")) Fail("expected ']' in subscript at " + where_);
        Advance();
        auto m = std::make_shared<Expr>();
        m->k = Expr::K::kIndex;
        m->a = e;
        m->b = idx;
        e = m;
      } else if (IsOp("|")) {
        // Jinja filter. Only `tojson` is supported (M3.3 tool rendering); any
        // other filter is a loud error, as before.
        Advance();
        if (cur_.t != ETok::T::kName) {
          Fail("expected filter name after '|' at " + where_);
        }
        const std::string fname = cur_.s;
        Advance();
        if (IsOp("(")) {
          Fail("filter arguments are not supported (minja-subset) at " + where_);
        }
        if (fname != "tojson") {
          Fail("filter '|" + fname +
               "' is not supported (minja-subset) at " + where_);
        }
        auto m = std::make_shared<Expr>();
        m->k = Expr::K::kFilter;
        m->a = e;
        m->name = fname;
        e = m;
      } else if (IsOp("(")) {
        Fail("function calls are not supported (minja-subset) at " + where_);
      } else {
        break;
      }
    }
    return e;
  }
  ExprPtr ParsePrimary() {
    if (IsOp("(")) {
      Advance();
      ExprPtr e = ParseOr();
      if (!IsOp(")")) Fail("expected ')' at " + where_);
      Advance();
      return e;
    }
    if (cur_.t == ETok::T::kString) {
      auto e = Lit(Value::Str(cur_.s));
      Advance();
      return e;
    }
    if (cur_.t == ETok::T::kInt) {
      auto e = Lit(Value::Int(cur_.i));
      Advance();
      return e;
    }
    if (cur_.t == ETok::T::kName) {
      const std::string& n = cur_.s;
      if (n == "true" || n == "True") {
        Advance();
        return Lit(Value::Bool(true));
      }
      if (n == "false" || n == "False") {
        Advance();
        return Lit(Value::Bool(false));
      }
      if (n == "none" || n == "None") {
        Advance();
        return Lit(Value::Null());
      }
      if (n == "and" || n == "or" || n == "not" || n == "in") {
        Fail("unexpected keyword '" + n + "' at " + where_);
      }
      auto e = std::make_shared<Expr>();
      e->k = Expr::K::kVar;
      e->name = n;
      Advance();
      return e;
    }
    Fail("unexpected token in expression at " + where_);
  }

  static ExprPtr Lit(Value v) {
    auto e = std::make_shared<Expr>();
    e->k = Expr::K::kLit;
    e->lit = std::move(v);
    return e;
  }
  static ExprPtr Bin(const std::string& op, ExprPtr a, ExprPtr b) {
    auto e = std::make_shared<Expr>();
    e->k = Expr::K::kBin;
    e->name = op;
    e->a = std::move(a);
    e->b = std::move(b);
    return e;
  }

  ExprLexer lex_;
  const std::string& where_;
  ETok cur_;
};

ExprPtr ParseExpression(const std::string& src, const std::string& where) {
  ExprParser p(src, where);
  return p.Parse();
}

// ─── Template lexer (tags + whitespace control) ──────────────────────────────
struct Tok {
  enum class T { kText, kOutput, kBlock } t;
  std::string text;   // literal text / inner tag source
  std::size_t pos;    // offset in template source (for error messages)
};

std::string RstripAll(const std::string& s) {
  std::size_t e = s.size();
  while (e > 0) {
    char c = s[e - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' ||
        c == '\v') {
      --e;
    } else {
      break;
    }
  }
  return s.substr(0, e);
}
std::string LstripAll(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size()) {
    char c = s[b];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' ||
        c == '\v') {
      ++b;
    } else {
      break;
    }
  }
  return s.substr(b);
}
std::string StripOneLeadingNewline(const std::string& s) {
  if (s.rfind("\r\n", 0) == 0) return s.substr(2);
  if (s.rfind("\n", 0) == 0) return s.substr(1);
  return s;
}
// lstrip_blocks: strip the trailing run of spaces/tabs that begins a line (i.e.
// immediately follows a newline or the segment start), just before a block tag.
std::string LstripBlocksTail(const std::string& s) {
  std::size_t i = s.size();
  while (i > 0 && (s[i - 1] == ' ' || s[i - 1] == '\t')) --i;
  if (i == 0 || s[i - 1] == '\n' || s[i - 1] == '\r') return s.substr(0, i);
  return s;
}
std::string TrimSpaces(const std::string& s) { return LstripAll(RstripAll(s)); }

// Find `close` (e.g. "%}") from `start`, skipping over quoted string literals so
// a `}}`/`%}` inside a string does not close the tag. Returns npos if absent.
std::size_t FindTagClose(const std::string& src, std::size_t start,
                         const char* close) {
  std::size_t n = src.size();
  char quote = 0;
  for (std::size_t k = start; k < n; ++k) {
    char c = src[k];
    if (quote) {
      if (c == '\\') {
        ++k;  // skip escaped char
      } else if (c == quote) {
        quote = 0;
      }
      continue;
    }
    if (c == '\'' || c == '"') {
      quote = c;
      continue;
    }
    if (src.compare(k, 2, close) == 0) return k;
  }
  return std::string::npos;
}

std::vector<Tok> Lex(const std::string& src) {
  std::vector<Tok> out;
  std::size_t n = src.size();
  std::size_t i = 0;
  bool right_trim_prev = false;   // previous tag: -%} / -}} -> lstrip-all next
  bool trim_block_prev = false;   // previous block tag + trim_blocks -> drop \n

  while (i < n) {
    // Find the next tag opener.
    std::size_t open = std::string::npos;
    for (std::size_t k = i; k + 1 < n; ++k) {
      if (src[k] == '{' &&
          (src[k + 1] == '{' || src[k + 1] == '%' || src[k + 1] == '#')) {
        open = k;
        break;
      }
    }

    std::string raw = src.substr(i, (open == std::string::npos ? n : open) - i);
    if (right_trim_prev) {
      raw = LstripAll(raw);
    } else if (trim_block_prev) {
      raw = StripOneLeadingNewline(raw);
    }

    if (open == std::string::npos) {
      if (!raw.empty()) out.push_back({Tok::T::kText, raw, i});
      break;
    }

    char kind = src[open + 1];  // '{' output, '%' block, '#' comment
    bool left_trim = (open + 2 < n && src[open + 2] == '-');
    bool is_block = (kind == '%');
    bool is_comment = (kind == '#');

    if (left_trim) {
      raw = RstripAll(raw);
    } else if ((is_block || is_comment) && kLstripBlocks) {
      raw = LstripBlocksTail(raw);
    }
    if (!raw.empty()) out.push_back({Tok::T::kText, raw, i});

    const char* close = (kind == '{') ? "}}" : (kind == '%') ? "%}" : "#}";
    std::size_t body_start = open + 2 + (left_trim ? 1 : 0);
    std::size_t close_pos = FindTagClose(src, body_start, close);
    if (close_pos == std::string::npos) {
      Fail("unterminated tag opened at " + LineCol(src, open));
    }
    bool right_trim = (close_pos > body_start && src[close_pos - 1] == '-');
    std::size_t body_end = close_pos - (right_trim ? 1 : 0);
    std::string body = TrimSpaces(src.substr(body_start, body_end - body_start));

    if (kind == '{') {
      out.push_back({Tok::T::kOutput, body, open});
    } else if (kind == '%') {
      out.push_back({Tok::T::kBlock, body, open});
    }
    // comments emit nothing.

    right_trim_prev = right_trim;
    trim_block_prev = ((is_block || is_comment) && kTrimBlocks && !right_trim);
    i = close_pos + 2;
  }
  return out;
}

// ─── Statement parser ────────────────────────────────────────────────────────
std::pair<std::string, std::string> SplitKeyword(const std::string& body) {
  std::size_t p = 0;
  while (p < body.size() &&
         (std::isalnum(static_cast<unsigned char>(body[p])) || body[p] == '_')) {
    ++p;
  }
  return {body.substr(0, p), TrimSpaces(body.substr(p))};
}

class StmtParser {
 public:
  StmtParser(std::vector<Tok> toks, const std::string& src)
      : toks_(std::move(toks)), src_(src) {}

  std::vector<StmtPtr> ParseTop() {
    std::vector<StmtPtr> body = ParseSeq({});
    if (idx_ < toks_.size()) {
      auto [kw, rest] = SplitKeyword(toks_[idx_].text);
      (void)rest;
      Fail("unexpected '{% " + kw + " %}' at " +
           LineCol(src_, toks_[idx_].pos));
    }
    return body;
  }

 private:
  bool AtTerminator(const std::vector<std::string>& terms) {
    if (idx_ >= toks_.size() || toks_[idx_].t != Tok::T::kBlock) return false;
    auto [kw, rest] = SplitKeyword(toks_[idx_].text);
    (void)rest;
    for (const std::string& t : terms) {
      if (kw == t) return true;
    }
    return false;
  }

  std::vector<StmtPtr> ParseSeq(const std::vector<std::string>& terms) {
    std::vector<StmtPtr> body;
    while (idx_ < toks_.size()) {
      const Tok& tk = toks_[idx_];
      if (tk.t == Tok::T::kText) {
        auto s = std::make_shared<Stmt>();
        s->k = Stmt::K::kText;
        s->text = tk.text;
        body.push_back(s);
        ++idx_;
        continue;
      }
      if (tk.t == Tok::T::kOutput) {
        auto s = std::make_shared<Stmt>();
        s->k = Stmt::K::kOutput;
        s->expr = ParseExpression(tk.text, LineCol(src_, tk.pos));
        body.push_back(s);
        ++idx_;
        continue;
      }
      // Block token.
      if (AtTerminator(terms)) return body;
      auto [kw, rest] = SplitKeyword(tk.text);
      const std::string where = LineCol(src_, tk.pos);
      if (kw == "for") {
        body.push_back(ParseFor(rest, where));
      } else if (kw == "if") {
        body.push_back(ParseIf(rest, where));
      } else if (kw == "set") {
        body.push_back(ParseSet(rest, where));
      } else if (kw == "endfor" || kw == "endif" || kw == "elif" ||
                 kw == "else" || kw == "endset" || kw == "endmacro" ||
                 kw == "endblock") {
        Fail("unexpected '{% " + kw + " %}' at " + where);
      } else {
        Fail("unsupported statement '{% " + kw +
             " %}' (minja-subset) at " + where);
      }
    }
    return body;
  }

  StmtPtr ParseFor(const std::string& rest, const std::string& where) {
    // rest = "VAR in EXPR"
    auto [var, after] = SplitKeyword(rest);
    if (var.empty()) Fail("malformed for-loop variable at " + where);
    auto [kw, iter] = SplitKeyword(after);
    if (kw != "in") {
      Fail("expected 'in' in for-loop (tuple-unpacking / for-if unsupported) at " +
           where);
    }
    auto s = std::make_shared<Stmt>();
    s->k = Stmt::K::kFor;
    s->var = var;
    s->expr = ParseExpression(iter, where);
    ++idx_;  // consume the {% for %} token
    s->body = ParseSeq({"endfor"});
    if (!AtTerminator({"endfor"})) Fail("missing '{% endfor %}' for loop at " + where);
    ++idx_;  // consume endfor
    return s;
  }

  StmtPtr ParseIf(const std::string& rest, const std::string& where) {
    auto s = std::make_shared<Stmt>();
    s->k = Stmt::K::kIf;
    IfBranch first;
    first.cond = ParseExpression(rest, where);
    ++idx_;  // consume {% if %}
    first.body = ParseSeq({"elif", "else", "endif"});
    s->branches.push_back(std::move(first));
    for (;;) {
      if (AtTerminator({"elif"})) {
        auto [kw, cond_src] = SplitKeyword(toks_[idx_].text);
        (void)kw;
        const std::string w = LineCol(src_, toks_[idx_].pos);
        ++idx_;  // consume elif
        IfBranch br;
        br.cond = ParseExpression(cond_src, w);
        br.body = ParseSeq({"elif", "else", "endif"});
        s->branches.push_back(std::move(br));
      } else if (AtTerminator({"else"})) {
        ++idx_;  // consume else
        s->elsebody = ParseSeq({"endif"});
        break;
      } else {
        break;
      }
    }
    if (!AtTerminator({"endif"})) Fail("missing '{% endif %}' at " + where);
    ++idx_;  // consume endif
    return s;
  }

  StmtPtr ParseSet(const std::string& rest, const std::string& where) {
    // rest = "VAR = EXPR"
    auto [var, after] = SplitKeyword(rest);
    if (var.empty()) Fail("malformed set target at " + where);
    std::string a = TrimSpaces(after);
    if (a.empty() || a[0] != '=') {
      Fail("expected '=' in set (namespace/block-set unsupported) at " + where);
    }
    auto s = std::make_shared<Stmt>();
    s->k = Stmt::K::kSet;
    s->var = var;
    s->expr = ParseExpression(TrimSpaces(a.substr(1)), where);
    ++idx_;
    return s;
  }

  std::vector<Tok> toks_;
  const std::string& src_;
  std::size_t idx_ = 0;
};

// ─── Evaluator ───────────────────────────────────────────────────────────────
std::string ToOutput(const Value& v, const std::string& where) {
  switch (v.kind) {
    case Value::Kind::kUndefined:
      return "";  // jinja default Undefined renders empty
    case Value::Kind::kNull:
      return "None";
    case Value::Kind::kBool:
      return v.b ? "True" : "False";
    case Value::Kind::kInt:
      return std::to_string(v.i);
    case Value::Kind::kString:
      return v.s;
    default:
      Fail("cannot render list/object value at " + where);
  }
}

using Env = std::map<std::string, Value>;

Value Eval(const ExprPtr& e, Env& env, const std::string& where);

// Wrap a JSON node as a Value: scalars become native Values (so string compares
// / truthiness work); containers stay opaque kJson (rendered via `| tojson`).
Value JsonToValue(const nlohmann::json& j) {
  if (j.is_null()) return Value::Null();
  if (j.is_boolean()) return Value::Bool(j.get<bool>());
  if (j.is_number_integer() || j.is_number_unsigned()) {
    return Value::Int(j.get<long long>());
  }
  if (j.is_string()) return Value::Str(j.get<std::string>());
  return Value::Json(j);  // float / object / array
}

// Convert a Value back to JSON for the `tojson` filter.
nlohmann::json ValueToJson(const Value& v, const std::string& where) {
  switch (v.kind) {
    case Value::Kind::kUndefined:
    case Value::Kind::kNull:
      return nullptr;
    case Value::Kind::kBool:
      return v.b;
    case Value::Kind::kInt:
      return v.i;
    case Value::Kind::kString:
      return v.s;
    case Value::Kind::kJson:
      return *v.json;
    case Value::Kind::kList: {
      nlohmann::json arr = nlohmann::json::array();
      for (const Value& item : *v.list) arr.push_back(ValueToJson(item, where));
      return arr;
    }
    case Value::Kind::kObject: {
      nlohmann::json o = nlohmann::json::object();
      for (const auto& [k, val] : *v.obj) o[k] = ValueToJson(val, where);
      return o;
    }
  }
  Fail("cannot serialize value at " + where);
}

Value EvalMember(const Value& obj, const std::string& name) {
  if (obj.kind == Value::Kind::kObject) {
    auto it = obj.obj->find(name);
    if (it != obj.obj->end()) return it->second;
  }
  if (obj.kind == Value::Kind::kJson && obj.json->is_object()) {
    auto it = obj.json->find(name);
    if (it != obj.json->end()) return JsonToValue(*it);
  }
  return Value::Undefined();
}

std::string StripChars(const std::string& s, const std::string* chars, bool left,
                       bool right) {
  auto is_strip = [&](char c) {
    if (chars) return chars->find(c) != std::string::npos;
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
  };
  std::size_t b = 0, en = s.size();
  if (left) {
    while (b < en && is_strip(s[b])) ++b;
  }
  if (right) {
    while (en > b && is_strip(s[en - 1])) --en;
  }
  return s.substr(b, en - b);
}

Value EvalBin(const std::string& op, const ExprPtr& le, const ExprPtr& re,
              Env& env, const std::string& where) {
  if (op == "and") {
    Value l = Eval(le, env, where);
    return Truthy(l) ? Eval(re, env, where) : l;
  }
  if (op == "or") {
    Value l = Eval(le, env, where);
    return Truthy(l) ? l : Eval(re, env, where);
  }
  Value l = Eval(le, env, where);
  Value r = Eval(re, env, where);
  if (op == "==") return Value::Bool(Equals(l, r));
  if (op == "!=") return Value::Bool(!Equals(l, r));
  if (op == "~") return Value::Str(ToOutput(l, where) + ToOutput(r, where));
  if (op == "+") {
    if (l.kind == Value::Kind::kString && r.kind == Value::Kind::kString) {
      return Value::Str(l.s + r.s);
    }
    if (l.kind == Value::Kind::kInt && r.kind == Value::Kind::kInt) {
      return Value::Int(l.i + r.i);
    }
    Fail("'+' requires two strings or two ints at " + where);
  }
  if (op == "-") {
    if (l.kind == Value::Kind::kInt && r.kind == Value::Kind::kInt) {
      return Value::Int(l.i - r.i);
    }
    Fail("'-' requires two ints at " + where);
  }
  if (op == "in" || op == "notin") {
    bool found = false;
    if (r.kind == Value::Kind::kString) {
      if (l.kind != Value::Kind::kString) {
        Fail("'in' on a string needs a string LHS at " + where);
      }
      found = r.s.find(l.s) != std::string::npos;
    } else if (r.kind == Value::Kind::kList) {
      for (const Value& item : *r.list) {
        if (Equals(item, l)) {
          found = true;
          break;
        }
      }
    } else if (r.kind == Value::Kind::kObject) {
      found = l.kind == Value::Kind::kString &&
              r.obj->find(l.s) != r.obj->end();
    } else {
      Fail("'in' requires a string/list/object RHS at " + where);
    }
    return Value::Bool(op == "in" ? found : !found);
  }
  Fail("unknown operator '" + op + "' at " + where);
}

Value Eval(const ExprPtr& e, Env& env, const std::string& where) {
  switch (e->k) {
    case Expr::K::kLit:
      return e->lit;
    case Expr::K::kVar: {
      auto it = env.find(e->name);
      if (it != env.end()) return it->second;
      return Value::Undefined();
    }
    case Expr::K::kMember:
      return EvalMember(Eval(e->a, env, where), e->name);
    case Expr::K::kIndex: {
      Value obj = Eval(e->a, env, where);
      Value idx = Eval(e->b, env, where);
      if (obj.kind == Value::Kind::kList && idx.kind == Value::Kind::kInt) {
        long long n = static_cast<long long>(obj.list->size());
        long long k = idx.i < 0 ? idx.i + n : idx.i;
        if (k < 0 || k >= n) Fail("list index out of range at " + where);
        return (*obj.list)[static_cast<std::size_t>(k)];
      }
      if (obj.kind == Value::Kind::kObject && idx.kind == Value::Kind::kString) {
        return EvalMember(obj, idx.s);
      }
      if (obj.kind == Value::Kind::kString && idx.kind == Value::Kind::kInt) {
        long long n = static_cast<long long>(obj.s.size());
        long long k = idx.i < 0 ? idx.i + n : idx.i;
        if (k < 0 || k >= n) Fail("string index out of range at " + where);
        return Value::Str(std::string(1, obj.s[static_cast<std::size_t>(k)]));
      }
      Fail("unsupported subscript at " + where);
    }
    case Expr::K::kUnary: {
      Value a = Eval(e->a, env, where);
      if (e->name == "not") return Value::Bool(!Truthy(a));
      if (e->name == "neg") {
        if (a.kind != Value::Kind::kInt) Fail("unary '-' on non-int at " + where);
        return Value::Int(-a.i);
      }
      Fail("unknown unary operator at " + where);
    }
    case Expr::K::kBin:
      return EvalBin(e->name, e->a, e->b, env, where);
    case Expr::K::kMethod: {
      Value obj = Eval(e->a, env, where);
      if (obj.kind != Value::Kind::kString) {
        Fail("method '." + e->name + "()' only supported on strings at " + where);
      }
      if (e->name == "strip" || e->name == "lstrip" || e->name == "rstrip") {
        std::string chars;
        const std::string* cp = nullptr;
        if (!e->args.empty()) {
          Value a = Eval(e->args[0], env, where);
          if (a.kind != Value::Kind::kString) {
            Fail("strip() argument must be a string at " + where);
          }
          chars = a.s;
          cp = &chars;
        }
        bool left = e->name != "rstrip";
        bool right = e->name != "lstrip";
        return Value::Str(StripChars(obj.s, cp, left, right));
      }
      Fail("unsupported method '." + e->name + "()' (minja-subset) at " + where);
    }
    case Expr::K::kFilter: {
      Value a = Eval(e->a, env, where);
      if (e->name == "tojson") {
        // Jinja `tojson`: serialize the value to a compact JSON string
        // (ensure_ascii=False equivalent — nlohmann emits UTF-8 directly).
        return Value::Str(ValueToJson(a, where).dump());
      }
      Fail("unsupported filter '|" + e->name + "' (minja-subset) at " + where);
    }
  }
  Fail("internal: bad expr node");
}

Value MakeLoop(std::size_t index0, std::size_t length) {
  std::map<std::string, Value> m;
  long long i0 = static_cast<long long>(index0);
  long long len = static_cast<long long>(length);
  m["index0"] = Value::Int(i0);
  m["index"] = Value::Int(i0 + 1);
  m["revindex"] = Value::Int(len - i0);
  m["revindex0"] = Value::Int(len - i0 - 1);
  m["first"] = Value::Bool(index0 == 0);
  m["last"] = Value::Bool(index0 + 1 == length);
  m["length"] = Value::Int(len);
  return Value::Object(std::move(m));
}

void Exec(const std::vector<StmtPtr>& body, Env& env, std::string& out,
          const std::string& where);

void ExecStmt(const StmtPtr& s, Env& env, std::string& out,
              const std::string& where) {
  switch (s->k) {
    case Stmt::K::kText:
      out += s->text;
      return;
    case Stmt::K::kOutput:
      out += ToOutput(Eval(s->expr, env, where), where);
      return;
    case Stmt::K::kSet:
      env[s->var] = Eval(s->expr, env, where);
      return;
    case Stmt::K::kFor: {
      Value it = Eval(s->expr, env, where);
      if (it.kind != Value::Kind::kList) {
        Fail("for-loop target is not a list at " + where);
      }
      // Save shadowed bindings.
      bool had_var = env.count(s->var) != 0;
      Value saved_var = had_var ? env[s->var] : Value::Undefined();
      bool had_loop = env.count("loop") != 0;
      Value saved_loop = had_loop ? env["loop"] : Value::Undefined();
      const std::size_t len = it.list->size();
      for (std::size_t k = 0; k < len; ++k) {
        env[s->var] = (*it.list)[k];
        env["loop"] = MakeLoop(k, len);
        Exec(s->body, env, out, where);
      }
      if (had_var) {
        env[s->var] = saved_var;
      } else {
        env.erase(s->var);
      }
      if (had_loop) {
        env["loop"] = saved_loop;
      } else {
        env.erase("loop");
      }
      return;
    }
    case Stmt::K::kIf: {
      for (const IfBranch& br : s->branches) {
        if (Truthy(Eval(br.cond, env, where))) {
          Exec(br.body, env, out, where);
          return;
        }
      }
      Exec(s->elsebody, env, out, where);
      return;
    }
  }
}

void Exec(const std::vector<StmtPtr>& body, Env& env, std::string& out,
          const std::string& where) {
  for (const StmtPtr& s : body) ExecStmt(s, env, out, where);
}

}  // namespace

// ─── Public API ──────────────────────────────────────────────────────────────
namespace {
// Rebuild the OpenAI tool JSON object the request carried, so `{{ tool | tojson
// }}` renders exactly what transformers' apply_chat_template(tools=...) sees:
//   {"type": <t>, "function": {"name": .., "description"?: .., "parameters"?: ..}}
nlohmann::json ToolToJson(const openai::ChatCompletionToolsParam& t) {
  nlohmann::json fn = nlohmann::json::object();
  fn["name"] = t.function.name;
  if (t.function.description.has_value()) {
    fn["description"] = *t.function.description;
  }
  if (t.function.parameters.has_value()) {
    fn["parameters"] = *t.function.parameters;
  }
  return nlohmann::json{{"type", t.type}, {"function", std::move(fn)}};
}
}  // namespace

std::string apply_chat_template(
    const std::string& template_str,
    const std::vector<openai::ChatMessage>& messages, bool add_generation_prompt,
    const std::string& bos_token, const std::string& eos_token,
    const std::vector<openai::ChatCompletionToolsParam>& tools) {
  // Parse.
  std::vector<Tok> toks = Lex(template_str);
  StmtParser parser(std::move(toks), template_str);
  std::vector<StmtPtr> program = parser.ParseTop();

  // Build the render context.
  Env env;
  std::vector<Value> msg_values;
  msg_values.reserve(messages.size());
  for (const openai::ChatMessage& m : messages) {
    std::map<std::string, Value> obj;
    obj["role"] = Value::Str(m.role);
    obj["content"] =
        m.content.has_value() ? Value::Str(*m.content) : Value::Null();
    msg_values.push_back(Value::Object(std::move(obj)));
  }
  env["messages"] = Value::List(std::move(msg_values));
  env["add_generation_prompt"] = Value::Bool(add_generation_prompt);
  env["bos_token"] = Value::Str(bos_token);
  env["eos_token"] = Value::Str(eos_token);
  // `tools` — a list of opaque JSON tool objects for the template's tool branch
  // (`{% if tools %}` / `{% for tool in tools %}{{ tool | tojson }}`). Empty list
  // when no tools (falsy), matching transformers passing tools=None.
  std::vector<Value> tool_values;
  tool_values.reserve(tools.size());
  for (const openai::ChatCompletionToolsParam& t : tools) {
    tool_values.push_back(Value::Json(ToolToJson(t)));
  }
  env["tools"] = Value::List(std::move(tool_values));

  std::string out;
  Exec(program, env, out, "<template>");
  return out;
}

openai::ChatPromptFn MakeChatTemplatePromptFn(std::string template_str,
                                              std::string bos_token,
                                              std::string eos_token) {
  return [tmpl = std::move(template_str), bos = std::move(bos_token),
          eos = std::move(eos_token)](
             const std::vector<openai::ChatMessage>& messages,
             bool add_generation_prompt,
             const std::vector<openai::ChatCompletionToolsParam>& tools) {
    return apply_chat_template(tmpl, messages, add_generation_prompt, bos, eos,
                               tools);
  };
}

std::string LoadChatTemplateFromConfig(
    const std::string& tokenizer_config_path) {
  std::ifstream f(tokenizer_config_path, std::ios::binary);
  if (!f) {
    throw ChatTemplateError("cannot open tokenizer_config.json: " +
                            tokenizer_config_path);
  }
  nlohmann::json doc;
  try {
    f >> doc;
  } catch (const std::exception& e) {
    throw ChatTemplateError(std::string("failed to parse tokenizer_config.json: ") +
                            e.what());
  }
  auto it = doc.find("chat_template");
  if (it == doc.end() || it->is_null()) {
    throw ChatTemplateError("tokenizer_config.json has no 'chat_template': " +
                            tokenizer_config_path);
  }
  if (it->is_string()) return it->get<std::string>();
  // List-of-{name,template} form: pick "default", else the first.
  if (it->is_array()) {
    const nlohmann::json* chosen = nullptr;
    for (const auto& entry : *it) {
      if (entry.is_object() && entry.value("name", std::string()) == "default") {
        chosen = &entry;
        break;
      }
    }
    if (!chosen && !it->empty()) chosen = &it->front();
    if (chosen && chosen->contains("template") &&
        (*chosen)["template"].is_string()) {
      return (*chosen)["template"].get<std::string>();
    }
  }
  throw ChatTemplateError("unrecognized 'chat_template' shape in " +
                          tokenizer_config_path);
}

std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  try {
    const vllm::GgufFile gguf = vllm::GgufFile::Open(gguf_path);
    const vllm::GgufValue* kv = gguf.FindKv("tokenizer.chat_template");
    if (kv == nullptr) {
      throw ChatTemplateError("gguf has no 'tokenizer.chat_template': " +
                              gguf_path);
    }
    const std::string* tmpl = std::get_if<std::string>(&kv->v);
    if (tmpl == nullptr || tmpl->empty()) {
      throw ChatTemplateError(
          "gguf 'tokenizer.chat_template' is not a non-empty string: " +
          gguf_path);
    }
    return *tmpl;
  } catch (const ChatTemplateError&) {
    throw;
  } catch (const std::exception& e) {
    // GgufFile::Open throws std::runtime_error on any malformation.
    throw ChatTemplateError(std::string("cannot read gguf chat template: ") +
                            e.what());
  }
}

}  // namespace vllm::entrypoints
