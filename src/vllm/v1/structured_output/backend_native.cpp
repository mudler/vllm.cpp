// vllm.cpp ORIGINAL component (§9 deviation). See backend_native.h for scope.
//
// This file implements, from scratch:
//   1. a GBNF/EBNF parser  -> a byte-level rule table (NativeCompiledGrammar),
//   2. a regex -> GBNF lowering and a choice -> GBNF lowering,
//   3. a stack-based push-down FSM matcher over the rule table,
//   4. a token-byte trie built once at construction, and
//   5. fill_bitmask as a single DFS over (trie x FSM state) — sub-O(vocab).
#include "vllm/v1/structured_output/backend_native.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/tokenizer/unicode_data.h"

namespace vllm::v1 {
namespace {

// ===========================================================================
// The compiled byte-level grammar.
//
// Every terminal element (kByte) matches EXACTLY ONE byte. A multi-byte
// codepoint literal is lowered to a sequence of kByte elements; a char class is
// one kByte element (a set of byte ranges, optionally negated) when purely
// ASCII, else a synthetic alternation sub-rule. Rules are element vectors;
// alternates are separated by kAlt; each rule ends with kEnd. Positions into
// this table drive the FSM.
// ===========================================================================
enum class ET : uint8_t { kEnd, kAlt, kRuleRef, kByte };

struct Elem {
  ET type = ET::kEnd;
  int32_t rule = -1;  // kRuleRef target
  bool negated = false;
  // kByte: inclusive byte ranges. Empty + negated=false is never emitted.
  std::vector<std::pair<uint8_t, uint8_t>> ranges;

  bool matches(uint8_t b) const {
    bool in = false;
    for (const auto& r : ranges) {
      if (b >= r.first && b <= r.second) {
        in = true;
        break;
      }
    }
    return negated ? !in : in;
  }
};

}  // namespace

// Defined at namespace scope (forward-declared in the header).
struct NativeCompiledGrammar {
  std::vector<std::vector<Elem>> rules;
  int32_t start_rule = -1;
};

namespace {

// ---------------------------------------------------------------------------
// GBNF / EBNF parser.
//
// Grammar:  rule ::= name "::=" alternates newline
//           alternates ::= sequence ("|" sequence)*
//           sequence ::= element*
//           element ::= string | charclass | group | "." | name | element post
//           post ::= "*" | "+" | "?" | "{" m ["," [n]] "}"
// Names: [a-zA-Z][a-zA-Z0-9-_]*. Comments: '#' to end of line. The rule named
// "root" is the entry; a synthetic start rule wraps it so the FSM init reuses
// the RULE_REF alternate-enumeration path.
// ---------------------------------------------------------------------------
class GbnfParser {
 public:
  explicit GbnfParser(std::string_view text) : s_(text) {}

  NativeCompiledGrammar Parse() {
    NativeCompiledGrammar g;
    // Reserve slot 0 for the synthetic start rule (filled in at the end).
    g.rules.emplace_back();
    grammar_ = &g;
    SkipSpaceAndNewlines();
    while (pos_ < s_.size()) {
      ParseRule();
      SkipSpaceAndNewlines();
    }
    auto root_it = rule_ids_.find("root");
    if (root_it == rule_ids_.end()) {
      throw std::runtime_error("native grammar: no 'root' rule defined");
    }
    // Validate every referenced rule got a definition.
    for (const auto& [name, id] : rule_ids_) {
      if (id >= static_cast<int32_t>(g.rules.size()) || g.rules[id].empty()) {
        throw std::runtime_error("native grammar: rule '" + name +
                                 "' referenced but not defined");
      }
    }
    // Synthetic start rule: start ::= root  (RULE_REF root, END).
    g.rules[0] = {MakeRuleRef(root_it->second), MakeEnd()};
    g.start_rule = 0;
    return g;
  }

 private:
  static Elem MakeEnd() { return Elem{ET::kEnd, -1, false, {}}; }
  static Elem MakeAlt() { return Elem{ET::kAlt, -1, false, {}}; }
  static Elem MakeRuleRef(int32_t r) { return Elem{ET::kRuleRef, r, false, {}}; }
  static Elem MakeByteRange(uint8_t lo, uint8_t hi, bool negated = false) {
    Elem e;
    e.type = ET::kByte;
    e.negated = negated;
    e.ranges.push_back({lo, hi});
    return e;
  }

  int32_t RuleId(const std::string& name) {
    auto it = rule_ids_.find(name);
    if (it != rule_ids_.end()) return it->second;
    const int32_t id = static_cast<int32_t>(grammar_->rules.size());
    grammar_->rules.emplace_back();  // empty until defined
    rule_ids_[name] = id;
    return id;
  }

  int32_t NewSyntheticRule() {
    const int32_t id = static_cast<int32_t>(grammar_->rules.size());
    grammar_->rules.emplace_back();
    return id;
  }

  char Peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  char PeekAt(size_t off) const {
    return pos_ + off < s_.size() ? s_[pos_ + off] : '\0';
  }
  bool Eof() const { return pos_ >= s_.size(); }

  void SkipInlineSpace() {
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\r') {
        ++pos_;
      } else if (c == '#') {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
      } else {
        break;
      }
    }
  }

  void SkipSpaceAndNewlines() {
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        ++pos_;
      } else if (c == '#') {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
      } else {
        break;
      }
    }
  }

  std::string ParseName() {
    const size_t start = pos_;
    if (Eof() || !(std::isalpha(static_cast<unsigned char>(Peek())) ||
                   Peek() == '_')) {
      throw std::runtime_error("native grammar: expected a rule name");
    }
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
        ++pos_;
      } else {
        break;
      }
    }
    return std::string(s_.substr(start, pos_ - start));
  }

  void ParseRule() {
    const std::string name = ParseName();
    SkipInlineSpace();
    if (!(Peek() == ':' && PeekAt(1) == ':' && PeekAt(2) == '=')) {
      throw std::runtime_error("native grammar: expected '::=' after rule '" +
                               name + "'");
    }
    pos_ += 3;
    SkipInlineSpace();
    const int32_t id = RuleId(name);
    if (!grammar_->rules[id].empty()) {
      throw std::runtime_error("native grammar: rule '" + name +
                               "' defined twice");
    }
    std::vector<Elem> elems;
    ParseAlternates(elems);
    elems.push_back(MakeEnd());
    grammar_->rules[id] = std::move(elems);
  }

  void ParseAlternates(std::vector<Elem>& out) {
    ParseSequence(out);
    SkipInlineSpace();
    while (Peek() == '|') {
      ++pos_;
      out.push_back(MakeAlt());
      SkipInlineSpace();
      ParseSequence(out);
      SkipInlineSpace();
    }
  }

  void ParseSequence(std::vector<Elem>& out) {
    for (;;) {
      SkipInlineSpace();
      const char c = Peek();
      if (Eof() || c == '|' || c == ')' || c == '\n') break;
      // last_sym_start marks where the element(s) for THIS item begin, so a
      // following postfix operator can move exactly that item into a sub-rule.
      const size_t last_sym_start = out.size();
      if (c == '"') {
        ParseStringLiteral(out);
      } else if (c == '[') {
        ParseCharClass(out);
      } else if (c == '(') {
        ++pos_;
        std::vector<Elem> sub;
        ParseAlternates(sub);
        sub.push_back(MakeEnd());
        SkipInlineSpace();
        if (Peek() != ')') {
          throw std::runtime_error("native grammar: expected ')'");
        }
        ++pos_;
        const int32_t id = NewSyntheticRule();
        grammar_->rules[id] = std::move(sub);
        out.push_back(MakeRuleRef(id));
      } else if (c == '.') {
        ++pos_;
        out.push_back(MakeByteRange(0x00, 0xFF));  // any byte (MVP)
      } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        const std::string name = ParseName();
        out.push_back(MakeRuleRef(RuleId(name)));
      } else {
        throw std::runtime_error(std::string("native grammar: unexpected '") +
                                 c + "'");
      }
      ApplyPostfix(out, last_sym_start);
    }
  }

  // Reads a single escaped char inside a string literal or char class, returning
  // its codepoint. `pos_` is positioned at the escape's backslash.
  uint32_t ParseEscape() {
    ++pos_;  // consume backslash
    if (Eof()) throw std::runtime_error("native grammar: trailing backslash");
    const char e = s_[pos_++];
    switch (e) {
      case 'n':
        return '\n';
      case 'r':
        return '\r';
      case 't':
        return '\t';
      case '\\':
        return '\\';
      case '"':
        return '"';
      case '\'':
        return '\'';
      case '[':
        return '[';
      case ']':
        return ']';
      case '.':
        return '.';
      case 'x':
        return ParseHex(2);
      case 'u':
        return ParseHex(4);
      case 'U':
        return ParseHex(8);
      default:
        return static_cast<uint32_t>(static_cast<unsigned char>(e));
    }
  }

  uint32_t ParseHex(int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i) {
      if (Eof()) throw std::runtime_error("native grammar: bad \\x/\\u escape");
      const char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        throw std::runtime_error("native grammar: non-hex in \\x/\\u escape");
      }
    }
    return v;
  }

  // Appends one kByte element per UTF-8 byte of `cp`.
  void EmitCodepointBytes(uint32_t cp, std::vector<Elem>& out) {
    std::string bytes;
    tok::EncodeUtf8(cp, bytes);
    for (const char ch : bytes) {
      const auto b = static_cast<uint8_t>(ch);
      out.push_back(MakeByteRange(b, b));
    }
  }

  void ParseStringLiteral(std::vector<Elem>& out) {
    ++pos_;  // opening quote
    while (!Eof() && Peek() != '"') {
      uint32_t cp;
      if (Peek() == '\\') {
        cp = ParseEscape();
      } else {
        size_t p = pos_;
        cp = tok::DecodeUtf8(s_, p);
        pos_ = p;
      }
      EmitCodepointBytes(cp, out);
    }
    if (Peek() != '"') {
      throw std::runtime_error("native grammar: unterminated string literal");
    }
    ++pos_;  // closing quote
  }

  // A char class -> one kByte element (ASCII-only, fast path) OR a synthetic
  // alternation rule (when it contains a non-ASCII codepoint).
  void ParseCharClass(std::vector<Elem>& out) {
    ++pos_;  // '['
    bool negated = false;
    if (Peek() == '^') {
      negated = true;
      ++pos_;
    }
    std::vector<std::pair<uint8_t, uint8_t>> ascii_ranges;
    std::vector<uint32_t> multibyte_cps;  // non-ASCII single codepoints
    while (!Eof() && Peek() != ']') {
      uint32_t lo;
      if (Peek() == '\\') {
        lo = ParseEscape();
      } else {
        size_t p = pos_;
        lo = tok::DecodeUtf8(s_, p);
        pos_ = p;
      }
      uint32_t hi = lo;
      if (Peek() == '-' && PeekAt(1) != ']' && PeekAt(1) != '\0') {
        ++pos_;  // '-'
        if (Peek() == '\\') {
          hi = ParseEscape();
        } else {
          size_t p = pos_;
          hi = tok::DecodeUtf8(s_, p);
          pos_ = p;
        }
      }
      if (lo <= 0x7F && hi <= 0x7F) {
        ascii_ranges.push_back({static_cast<uint8_t>(lo),
                                static_cast<uint8_t>(hi)});
      } else if (lo == hi) {
        multibyte_cps.push_back(lo);
      } else {
        throw std::runtime_error(
            "native grammar: non-ASCII char-class RANGE is unsupported "
            "(single non-ASCII chars are ok)");
      }
    }
    if (Peek() != ']') {
      throw std::runtime_error("native grammar: unterminated char class");
    }
    ++pos_;  // ']'

    if (multibyte_cps.empty()) {
      Elem e;
      e.type = ET::kByte;
      e.negated = negated;
      e.ranges = std::move(ascii_ranges);
      if (e.ranges.empty()) {
        throw std::runtime_error("native grammar: empty char class");
      }
      out.push_back(std::move(e));
      return;
    }
    if (negated) {
      throw std::runtime_error(
          "native grammar: negated char class with non-ASCII is unsupported");
    }
    // Synthetic rule: alt of (the ASCII byte-set) and each non-ASCII codepoint.
    const int32_t id = NewSyntheticRule();
    std::vector<Elem> body;
    bool first = true;
    if (!ascii_ranges.empty()) {
      Elem e;
      e.type = ET::kByte;
      e.ranges = ascii_ranges;
      body.push_back(std::move(e));
      first = false;
    }
    for (const uint32_t cp : multibyte_cps) {
      if (!first) body.push_back(MakeAlt());
      first = false;
      EmitCodepointBytes(cp, body);
    }
    body.push_back(MakeEnd());
    grammar_->rules[id] = std::move(body);
    out.push_back(MakeRuleRef(id));
  }

  // Rewrites the last symbol [start,end) per a trailing */+/?/{m,n} into a
  // synthetic recursive rule, then replaces it with a single RULE_REF.
  void ApplyPostfix(std::vector<Elem>& out, size_t start) {
    for (;;) {
      const char c = Peek();
      if (c == '*' || c == '+' || c == '?') {
        ++pos_;
        RewriteRepeat(out, start, c == '+' ? 1 : 0,
                      c == '?' ? 1 : -1);  // *:0..inf +:1..inf ?:0..1
      } else if (c == '{') {
        int lo = 0, hi = 0;
        ParseBraceCount(lo, hi);
        RewriteRepeat(out, start, lo, hi);
      } else {
        break;
      }
    }
  }

  void ParseBraceCount(int& lo, int& hi) {
    ++pos_;  // '{'
    SkipInlineSpace();
    lo = ReadInt();
    SkipInlineSpace();
    if (Peek() == ',') {
      ++pos_;
      SkipInlineSpace();
      if (Peek() == '}') {
        hi = -1;  // {m,} -> m..inf
      } else {
        hi = ReadInt();
      }
    } else {
      hi = lo;  // {m} -> exactly m
    }
    SkipInlineSpace();
    if (Peek() != '}') {
      throw std::runtime_error("native grammar: expected '}' in {m,n}");
    }
    ++pos_;
  }

  int ReadInt() {
    if (!std::isdigit(static_cast<unsigned char>(Peek()))) {
      throw std::runtime_error("native grammar: expected a number in {m,n}");
    }
    int v = 0;
    while (std::isdigit(static_cast<unsigned char>(Peek()))) {
      v = v * 10 + (s_[pos_] - '0');
      ++pos_;
    }
    return v;
  }

  // Replaces out[start..] (the "moved" symbol) with a RULE_REF implementing
  // `moved` repeated in [lo, hi] (hi == -1 means unbounded). Built by expansion:
  //   lo mandatory copies, then either an unbounded tail (moved)* or (hi-lo)
  //   optional copies.
  void RewriteRepeat(std::vector<Elem>& out, size_t start, int lo, int hi) {
    std::vector<Elem> moved(out.begin() + static_cast<std::ptrdiff_t>(start),
                            out.end());
    out.resize(start);
    if (lo < 0 || (hi >= 0 && hi < lo)) {
      throw std::runtime_error("native grammar: invalid repetition count");
    }
    // star ::= moved star |            (zero or more)
    auto make_star = [&]() -> int32_t {
      const int32_t id = NewSyntheticRule();
      std::vector<Elem> body = moved;
      body.push_back(MakeRuleRef(id));
      body.push_back(MakeAlt());
      body.push_back(MakeEnd());
      grammar_->rules[id] = std::move(body);
      return id;
    };
    // opt ::= moved tail |             (an optional copy chaining `tail`)
    auto make_opt_chain = [&](int32_t tail) -> int32_t {
      const int32_t id = NewSyntheticRule();
      std::vector<Elem> body = moved;
      if (tail >= 0) body.push_back(MakeRuleRef(tail));
      body.push_back(MakeAlt());
      body.push_back(MakeEnd());
      grammar_->rules[id] = std::move(body);
      return id;
    };

    // Build the tail: the part after the `lo` mandatory copies.
    int32_t tail = -1;  // -1 => nothing further required
    if (hi < 0) {
      tail = make_star();  // unbounded
    } else {
      // (hi - lo) nested optionals.
      for (int i = 0; i < hi - lo; ++i) {
        tail = make_opt_chain(tail);
      }
    }

    if (lo == 0) {
      if (tail < 0) {
        // {0} or {0,0}: matches empty. Represent as an empty synthetic rule.
        const int32_t id = NewSyntheticRule();
        grammar_->rules[id] = {MakeEnd()};
        out.push_back(MakeRuleRef(id));
      } else {
        out.push_back(MakeRuleRef(tail));
      }
      return;
    }
    // `lo` mandatory copies, then the tail.
    const int32_t id = NewSyntheticRule();
    std::vector<Elem> body;
    for (int i = 0; i < lo; ++i) {
      body.insert(body.end(), moved.begin(), moved.end());
    }
    if (tail >= 0) body.push_back(MakeRuleRef(tail));
    body.push_back(MakeEnd());
    grammar_->rules[id] = std::move(body);
    out.push_back(MakeRuleRef(id));
  }

  std::string_view s_;
  size_t pos_ = 0;
  NativeCompiledGrammar* grammar_ = nullptr;
  std::unordered_map<std::string, int32_t> rule_ids_;
};

// ---------------------------------------------------------------------------
// Regex -> GBNF lowering (correctness-grade for common constructs). Emits a
// `root ::= <translated>` GBNF string, then reuses GbnfParser. Supported:
// literals, escapes (\d \w \s \D \W \S \n \r \t \. and \-escaped metachars),
// char classes [..] / [^..] / ranges, '.', groups '(...)' incl. (?:...), '|',
// quantifiers * + ? {m} {m,} {m,n}. Anchors ^ $ are dropped (whole-string match
// is implied by the constrained decode). Backreferences / lookaround throw.
// ---------------------------------------------------------------------------
class RegexToGbnf {
 public:
  explicit RegexToGbnf(std::string_view pattern) : s_(pattern) {}

  std::string Convert() {
    std::string body = ParseAlternation();
    if (pos_ != s_.size()) {
      throw std::runtime_error("native regex: unexpected trailing input");
    }
    return "root ::= " + body;
  }

 private:
  char Peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  bool Eof() const { return pos_ >= s_.size(); }

  std::string ParseAlternation() {
    std::string result = ParseConcat();
    while (Peek() == '|') {
      ++pos_;
      result += " | ";
      result += ParseConcat();
    }
    return result;
  }

  std::string ParseConcat() {
    std::string out;
    bool first = true;
    while (!Eof() && Peek() != '|' && Peek() != ')') {
      std::string atom = ParseAtomWithQuantifier();
      if (atom.empty()) continue;  // dropped anchor
      if (!first) out += ' ';
      out += atom;
      first = false;
    }
    if (out.empty()) out = "\"\"";  // empty alternative -> empty literal
    return out;
  }

  std::string ParseAtomWithQuantifier() {
    std::string atom = ParseAtom();
    if (atom.empty()) return atom;  // anchor -> nothing to quantify
    const char c = Peek();
    if (c == '*' || c == '+' || c == '?') {
      ++pos_;
      atom = "(" + atom + ")" + std::string(1, c);
      // A possessive/lazy modifier ('?' or '+' after the quantifier) is treated
      // as greedy (semantics-equivalent for acceptance).
      if (Peek() == '?' || Peek() == '+') ++pos_;
    } else if (c == '{') {
      atom = "(" + atom + ")" + ParseBrace();
      if (Peek() == '?') ++pos_;
    }
    return atom;
  }

  std::string ParseBrace() {
    std::string out = "{";
    ++pos_;  // '{'
    while (!Eof() && Peek() != '}') out += s_[pos_++];
    if (Peek() != '}') throw std::runtime_error("native regex: unbalanced {");
    ++pos_;
    out += "}";
    return out;
  }

  std::string ParseAtom() {
    const char c = Peek();
    if (c == '(') {
      ++pos_;
      if (Peek() == '?') {  // (?:...) non-capturing and the like
        ++pos_;
        if (Peek() == ':') {
          ++pos_;
        } else {
          throw std::runtime_error(
              "native regex: lookaround/named groups unsupported");
        }
      }
      std::string inner = ParseAlternation();
      if (Peek() != ')') throw std::runtime_error("native regex: unbalanced (");
      ++pos_;
      return "(" + inner + ")";
    }
    if (c == '[') {
      return ParseClass();
    }
    if (c == '.') {
      ++pos_;
      return "[^\\n]";  // any byte except newline (MVP)
    }
    if (c == '^' || c == '$') {
      ++pos_;  // drop anchor
      return "";
    }
    if (c == '\\') {
      return ParseEscapeAtom();
    }
    // Literal char -> a quoted GBNF string with the char escaped.
    ++pos_;
    return "\"" + EscapeGbnfChar(c) + "\"";
  }

  std::string ParseEscapeAtom() {
    ++pos_;  // backslash
    if (Eof()) throw std::runtime_error("native regex: trailing backslash");
    const char e = s_[pos_++];
    switch (e) {
      case 'd':
        return "[0-9]";
      case 'D':
        return "[^0-9]";
      case 'w':
        return "[a-zA-Z0-9_]";
      case 'W':
        return "[^a-zA-Z0-9_]";
      case 's':
        return "[ \\t\\n\\r]";
      case 'S':
        return "[^ \\t\\n\\r]";
      case 'n':
        return "\"\\n\"";
      case 'r':
        return "\"\\r\"";
      case 't':
        return "\"\\t\"";
      default:
        // An escaped literal / metachar.
        return "\"" + EscapeGbnfChar(e) + "\"";
    }
  }

  // A regex [..] class maps almost 1:1 to a GBNF [..] class; we re-emit it,
  // passing escapes through and normalizing \d/\w/\s inside the class to ranges.
  std::string ParseClass() {
    std::string out = "[";
    ++pos_;  // '['
    if (Peek() == '^') {
      out += '^';
      ++pos_;
    }
    while (!Eof() && Peek() != ']') {
      if (Peek() == '\\') {
        ++pos_;
        if (Eof()) throw std::runtime_error("native regex: trailing backslash");
        const char e = s_[pos_++];
        switch (e) {
          case 'd':
            out += "0-9";
            break;
          case 'w':
            out += "a-zA-Z0-9_";
            break;
          case 's':
            out += " \\t\\n\\r";
            break;
          case 'n':
            out += "\\n";
            break;
          case 'r':
            out += "\\r";
            break;
          case 't':
            out += "\\t";
            break;
          case ']':
            out += "\\]";
            break;
          case '\\':
            out += "\\\\";
            break;
          default:
            out += e;
            break;
        }
      } else {
        const char ch = s_[pos_++];
        if (ch == '\\') {
          out += "\\\\";
        } else {
          out += ch;
        }
      }
    }
    if (Peek() != ']') throw std::runtime_error("native regex: unbalanced [");
    ++pos_;
    out += "]";
    return out;
  }

  static std::string EscapeGbnfChar(char c) {
    switch (c) {
      case '"':
        return "\\\"";
      case '\\':
        return "\\\\";
      case '\n':
        return "\\n";
      case '\r':
        return "\\r";
      case '\t':
        return "\\t";
      default:
        return std::string(1, c);
    }
  }

  std::string_view s_;
  size_t pos_ = 0;
};

// choice_as_grammar (mirrors utils.py:490): a JSON array of strings ->
// `root ::= "a" | "b" | ...` with " and \ escaped.
std::string ChoiceToGbnf(const std::string& choice_spec) {
  nlohmann::json arr = nlohmann::json::parse(choice_spec);
  if (!arr.is_array() || arr.empty()) {
    throw std::runtime_error("native choice: spec must be a non-empty JSON "
                             "array of strings");
  }
  std::string grammar = "root ::= ";
  bool first = true;
  for (const auto& item : arr) {
    if (!item.is_string()) {
      throw std::runtime_error("native choice: array items must be strings");
    }
    if (!first) grammar += " | ";
    first = false;
    grammar += '"';
    for (const char c : item.get<std::string>()) {
      if (c == '"' || c == '\\') grammar += '\\';
      grammar += c;
    }
    grammar += '"';
  }
  return grammar;
}

// ===========================================================================
// The push-down FSM over the compiled grammar.
//
// State = a set of stacks; a stack is a list of positions (rule, idx), the back
// being where we are, the rest the return positions. advance_stack expands
// rule refs (and their alternates) until every stack's top is a byte terminal
// (or the stack is empty == a complete derivation).
// ===========================================================================
using Pos = std::pair<int32_t, int32_t>;  // (rule, elem index)
using Stack = std::vector<Pos>;
using State = std::vector<Stack>;

constexpr int kMaxAdvanceDepth = 4000;  // guards left-recursive grammars

inline bool IsSeqEnd(const Elem& e) {
  return e.type == ET::kEnd || e.type == ET::kAlt;
}

void AdvanceStack(const NativeCompiledGrammar& g, const Stack& stack,
                  std::set<Stack>& out, int depth) {
  if (depth > kMaxAdvanceDepth) {
    throw std::runtime_error(
        "native grammar: advance depth exceeded (left recursion?)");
  }
  if (stack.empty()) {
    out.insert(stack);  // a complete derivation
    return;
  }
  const Pos pos = stack.back();
  const Elem& elem = g.rules[static_cast<size_t>(pos.first)]
                            [static_cast<size_t>(pos.second)];
  if (elem.type == ET::kByte) {
    out.insert(stack);  // waiting for a byte
    return;
  }
  // elem.type == kRuleRef (kEnd/kAlt are never a stack top).
  Stack base(stack.begin(), stack.end() - 1);
  const Pos cont{pos.first, pos.second + 1};
  const bool has_cont =
      !IsSeqEnd(g.rules[static_cast<size_t>(cont.first)]
                       [static_cast<size_t>(cont.second)]);
  const int32_t ref = elem.rule;
  const auto& rref = g.rules[static_cast<size_t>(ref)];
  int32_t i = 0;
  for (;;) {
    Stack ns = base;
    if (has_cont) ns.push_back(cont);
    if (!IsSeqEnd(rref[static_cast<size_t>(i)])) ns.push_back({ref, i});
    AdvanceStack(g, ns, out, depth + 1);
    while (!IsSeqEnd(rref[static_cast<size_t>(i)])) ++i;
    if (rref[static_cast<size_t>(i)].type == ET::kAlt) {
      ++i;  // next alternate
    } else {
      break;  // kEnd
    }
  }
}

State InitState(const NativeCompiledGrammar& g) {
  std::set<Stack> acc;
  Stack start{{g.start_rule, 0}};
  AdvanceStack(g, start, acc, 0);
  return State(acc.begin(), acc.end());
}

State AcceptByte(const NativeCompiledGrammar& g, const State& state,
                 uint8_t byte) {
  std::set<Stack> acc;
  for (const Stack& stack : state) {
    if (stack.empty()) continue;  // completed; consumes no more bytes
    const Pos pos = stack.back();
    const Elem& elem = g.rules[static_cast<size_t>(pos.first)]
                              [static_cast<size_t>(pos.second)];
    if (elem.type != ET::kByte || !elem.matches(byte)) continue;
    Stack base(stack.begin(), stack.end() - 1);
    const Pos next{pos.first, pos.second + 1};
    if (!IsSeqEnd(g.rules[static_cast<size_t>(next.first)]
                        [static_cast<size_t>(next.second)])) {
      base.push_back(next);
    }
    AdvanceStack(g, base, acc, 0);
  }
  return State(acc.begin(), acc.end());
}

// The union of acceptable bytes from `state` (for the trie DFS). Returns a
// 256-entry presence table; also reports whether ANY terminal is present.
struct ByteAcceptSet {
  std::array<bool, 256> allowed{};
  bool any = false;
};

ByteAcceptSet AcceptableBytes(const NativeCompiledGrammar& g,
                              const State& state) {
  ByteAcceptSet r;
  for (const Stack& stack : state) {
    if (stack.empty()) continue;
    const Pos pos = stack.back();
    const Elem& elem = g.rules[static_cast<size_t>(pos.first)]
                              [static_cast<size_t>(pos.second)];
    if (elem.type != ET::kByte) continue;
    for (int b = 0; b < 256; ++b) {
      if (!r.allowed[static_cast<size_t>(b)] &&
          elem.matches(static_cast<uint8_t>(b))) {
        r.allowed[static_cast<size_t>(b)] = true;
        r.any = true;
      }
    }
  }
  return r;
}

inline bool IsAccepting(const State& state) {
  for (const Stack& stack : state) {
    if (stack.empty()) return true;
  }
  return false;
}

// Fully matched == accepting AND no stack can consume another byte (all stacks
// empty). Then the grammar is terminated (only EOS may follow).
inline bool IsFullyMatched(const State& state) {
  if (state.empty()) return false;
  for (const Stack& stack : state) {
    if (!stack.empty()) return false;
  }
  return true;
}

// ===========================================================================
// The token-byte trie.
// ===========================================================================
struct TrieNode {
  // Sorted (byte -> child index) edges. Small per node; keeps memory O(sum of
  // token lengths) rather than 256 pointers/node.
  std::vector<std::pair<uint8_t, int32_t>> children;
  std::vector<int32_t> token_ids;  // tokens ending exactly here

  int32_t child(uint8_t b) const {
    for (const auto& e : children) {
      if (e.first == b) return e.second;
    }
    return -1;
  }
};

struct TokenByteTrie {
  std::vector<TrieNode> nodes;  // node 0 is the root

  TokenByteTrie() { nodes.emplace_back(); }

  void Insert(std::string_view bytes, int32_t token_id) {
    int32_t cur = 0;
    for (const char ch : bytes) {
      const auto b = static_cast<uint8_t>(ch);
      int32_t next = nodes[static_cast<size_t>(cur)].child(b);
      if (next < 0) {
        next = static_cast<int32_t>(nodes.size());
        nodes.emplace_back();
        nodes[static_cast<size_t>(cur)].children.push_back({b, next});
        // keep edges sorted for deterministic DFS order
        auto& ch_edges = nodes[static_cast<size_t>(cur)].children;
        std::sort(ch_edges.begin(), ch_edges.end());
      }
      cur = next;
    }
    nodes[static_cast<size_t>(cur)].token_ids.push_back(token_id);
  }
};

}  // namespace

// The shared, once-built engine context (forward-declared in the header).
struct NativeBackendShared {
  TokenByteTrie trie;
  // id -> raw bytes, for every trie-inserted (grammar-matchable) token. Used by
  // accept_tokens/validate_tokens to recover a token's bytes (the trie is keyed
  // by bytes, so it cannot map an id back to bytes).
  std::unordered_map<int32_t, std::string> token_bytes;
  std::vector<int32_t> stop_token_ids;  // allowed only at an accepting state
  int vocab_size = 0;
};

// ===========================================================================
// NativeGrammar
// ===========================================================================
struct NativeGrammar::Snapshot {
  State state;
  bool done = false;  // reached by consuming EOS/stop; nothing may follow
};

NativeGrammar::NativeGrammar(std::shared_ptr<const NativeBackendShared> shared,
                             std::shared_ptr<const NativeCompiledGrammar> grammar)
    : shared_(std::move(shared)), grammar_(std::move(grammar)) {
  history_.push_back(Snapshot{InitState(*grammar_), false});
}

NativeGrammar::~NativeGrammar() = default;

bool NativeGrammar::accept_tokens(const std::string& /*request_id*/,
                                  const std::vector<int32_t>& tokens) {
  for (const int32_t token : tokens) {
    const Snapshot& cur = history_.back();
    if (cur.done) return false;  // nothing follows EOS

    const auto& stops = shared_->stop_token_ids;
    if (std::find(stops.begin(), stops.end(), token) != stops.end()) {
      if (!IsAccepting(cur.state)) return false;
      history_.push_back(Snapshot{cur.state, true});
      continue;
    }

    // A regular token: recover its raw bytes and advance the FSM byte-by-byte.
    // A token with no byte representation (added/special/hole) is not matchable.
    auto it = shared_->token_bytes.find(token);
    if (it == shared_->token_bytes.end()) return false;

    State s = cur.state;
    bool ok = true;
    for (const char ch : it->second) {
      s = AcceptByte(*grammar_, s, static_cast<uint8_t>(ch));
      if (s.empty()) {
        ok = false;
        break;
      }
    }
    if (!ok) return false;
    history_.push_back(Snapshot{std::move(s), false});
  }
  return true;
}

std::vector<int32_t> NativeGrammar::validate_tokens(
    const std::vector<int32_t>& tokens) {
  std::vector<int32_t> accepted;
  Snapshot cur = history_.back();  // copy; do not mutate the real state
  for (const int32_t token : tokens) {
    if (cur.done) break;
    const auto& stops = shared_->stop_token_ids;
    if (std::find(stops.begin(), stops.end(), token) != stops.end()) {
      if (!IsAccepting(cur.state)) break;
      cur.done = true;
      accepted.push_back(token);
      continue;
    }
    auto it = shared_->token_bytes.find(token);
    if (it == shared_->token_bytes.end()) break;
    State s = cur.state;
    bool ok = true;
    for (const char ch : it->second) {
      s = AcceptByte(*grammar_, s, static_cast<uint8_t>(ch));
      if (s.empty()) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
    cur.state = std::move(s);
    accepted.push_back(token);
  }
  return accepted;
}

void NativeGrammar::rollback(int num_tokens) {
  if (num_tokens <= 0) return;
  const int n = std::min<int>(num_tokens,
                              static_cast<int>(history_.size()) - 1);
  history_.erase(history_.end() - n, history_.end());
}

bool NativeGrammar::is_terminated() {
  const Snapshot& cur = history_.back();
  return cur.done || IsFullyMatched(cur.state);
}

void NativeGrammar::reset() {
  history_.clear();
  history_.push_back(Snapshot{InitState(*grammar_), false});
}

void NativeGrammar::fill_bitmask(TokenBitmask& bitmask, int batch_index) {
  last_fill_visited_nodes_ = 0;
  const int num_words = bitmask.num_words;
  const std::size_t base =
      static_cast<std::size_t>(batch_index) * static_cast<std::size_t>(num_words);
  // Clear this row first (the manager reuses the buffer across steps and
  // fill_bitmask only SETS allowed bits).
  for (int w = 0; w < num_words; ++w) {
    bitmask.data[base + static_cast<std::size_t>(w)] = 0;
  }

  const Snapshot& cur = history_.back();
  auto set_bit = [&](int32_t tid) {
    if (tid < 0 || tid >= shared_->vocab_size) return;
    const std::size_t word = base + static_cast<std::size_t>(tid >> 5);
    bitmask.data[word] |=
        static_cast<int32_t>(1u << (static_cast<uint32_t>(tid) & 31u));
  };

  if (cur.done) {
    // EOS already consumed: the derivation is complete and NOTHING more may be
    // emitted, so leave the whole row clear (all-forbidden). This keeps
    // fill_bitmask and accept_tokens in EXACT agreement — accept_tokens([EOS])
    // (and any token) returns false in the done state, so fill must allow none.
    // (The manager never fills a terminated row anyway — !is_terminated() guard —
    // so this branch is defensive; agreement matters for the differential test.)
    return;
  }

  // THE BYTE-ALIGNMENT CORE: one DFS over (trie x FSM state). At each trie node
  // reachable under the grammar, tokens ending there are allowed; descend only
  // along bytes the grammar accepts from the current sub-state.
  const TokenByteTrie& trie = shared_->trie;
  struct Frame {
    int32_t node;
    State state;
  };
  std::vector<Frame> stack;
  stack.push_back(Frame{0, cur.state});
  while (!stack.empty()) {
    Frame frame = std::move(stack.back());
    stack.pop_back();
    ++last_fill_visited_nodes_;
    const TrieNode& node = trie.nodes[static_cast<size_t>(frame.node)];
    for (const int32_t tid : node.token_ids) set_bit(tid);
    if (node.children.empty()) continue;
    const ByteAcceptSet acc = AcceptableBytes(*grammar_, frame.state);
    if (!acc.any) continue;
    for (const auto& edge : node.children) {
      if (!acc.allowed[static_cast<size_t>(edge.first)]) continue;
      State ns = AcceptByte(*grammar_, frame.state, edge.first);
      if (ns.empty()) continue;
      stack.push_back(Frame{edge.second, std::move(ns)});
    }
  }

  // EOS/stop tokens are allowed iff the grammar is at an accepting state.
  if (IsAccepting(cur.state)) {
    for (const int32_t sid : shared_->stop_token_ids) set_bit(sid);
  }
}

// ===========================================================================
// NativeStructuredOutputBackend
// ===========================================================================
NativeStructuredOutputBackend::NativeStructuredOutputBackend(
    const tok::Tokenizer& tokenizer, int vocab_size,
    std::vector<int32_t> stop_token_ids) {
  auto shared = std::make_shared<NativeBackendShared>();
  shared->vocab_size = vocab_size;

  // Stop tokens: caller-supplied, else the tokenizer's EOS (when present).
  if (stop_token_ids.empty()) {
    const int32_t eos = tokenizer.EosId();
    if (eos >= 0) shared->stop_token_ids.push_back(eos);
  } else {
    shared->stop_token_ids = std::move(stop_token_ids);
  }

  // Excluded ids: every added token (special OR not — added tokens carry literal
  // content, not byte-mapped bytes) and every stop token. These are never
  // grammar-matchable mid-derivation.
  std::unordered_set<int32_t> excluded;
  for (const auto& at : tokenizer.AddedTokens()) excluded.insert(at.id);
  for (const int32_t sid : shared->stop_token_ids) excluded.insert(sid);

  // Build the token-byte trie + the id->bytes table (used by accept_tokens).
  for (int32_t id = 0; id < vocab_size; ++id) {
    if (!tokenizer.HasToken(id)) continue;  // vocab hole
    if (excluded.count(id) != 0) continue;
    std::string raw;
    try {
      raw = tok::UnmapUnicodeToBytes(tokenizer.TokenText(id));
    } catch (const std::exception&) {
      continue;  // not a byte-level token (defensive)
    }
    if (raw.empty()) continue;
    shared->trie.Insert(raw, id);
    shared->token_bytes[id] = std::move(raw);
  }

  shared_ = std::move(shared);
}

NativeStructuredOutputBackend::~NativeStructuredOutputBackend() = default;

int NativeStructuredOutputBackend::vocab_size() const {
  return shared_->vocab_size;
}

std::unique_ptr<StructuredOutputGrammar>
NativeStructuredOutputBackend::compile_grammar(
    StructuredOutputOptions request_type, const std::string& grammar_spec) {
  std::string gbnf;
  switch (request_type) {
    case StructuredOutputOptions::kGrammar:
      gbnf = grammar_spec;
      break;
    case StructuredOutputOptions::kChoice:
      gbnf = ChoiceToGbnf(grammar_spec);
      break;
    case StructuredOutputOptions::kRegex:
      gbnf = RegexToGbnf(grammar_spec).Convert();
      break;
    case StructuredOutputOptions::kJson:
    case StructuredOutputOptions::kJsonObject:
      throw std::runtime_error(
          "native backend: JSON / json_object (JSON-schema -> GBNF) is M3.4 "
          "Task 5; not yet implemented");
    case StructuredOutputOptions::kStructuralTag:
      throw std::runtime_error(
          "native backend: STRUCTURAL_TAG is deferred (see backend_types.h)");
  }
  auto compiled = std::make_shared<NativeCompiledGrammar>(
      GbnfParser(gbnf).Parse());
  return std::make_unique<NativeGrammar>(shared_, std::move(compiled));
}

TokenBitmask NativeStructuredOutputBackend::allocate_token_bitmask(
    int max_num_seqs) {
  TokenBitmask m;
  m.num_seqs = max_num_seqs;
  m.num_words = BitmaskWordsForVocab(shared_->vocab_size);
  m.data.assign(static_cast<std::size_t>(max_num_seqs) *
                    static_cast<std::size_t>(m.num_words),
                0);
  return m;
}

void NativeStructuredOutputBackend::destroy() {}

std::function<std::unique_ptr<StructuredOutputBackend>()>
MakeNativeBackendFactory(const tok::Tokenizer& tokenizer, int vocab_size,
                         std::vector<int32_t> stop_token_ids) {
  return [&tokenizer, vocab_size, stop_token_ids]() {
    return std::make_unique<NativeStructuredOutputBackend>(
        tokenizer, vocab_size, stop_token_ids);
  };
}

}  // namespace vllm::v1
