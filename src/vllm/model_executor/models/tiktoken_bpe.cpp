// tiktoken vocabulary + BPE. See tiktoken_bpe.h for the upstream anchors.
#include "vllm/model_executor/models/tiktoken_bpe.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace vllm {
namespace models {
namespace tiktoken {
namespace {

int Base64Value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

std::string Base64Decode(const std::string& in) {
  std::string out;
  int acc = 0;
  int bits = 0;
  for (const char c : in) {
    if (c == '=') break;
    const int v = Base64Value(c);
    if (v < 0) {
      throw std::runtime_error("tiktoken: not base64: '" + in + "'");
    }
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((acc >> bits) & 0xFF));
    }
  }
  return out;
}

// Decode one UTF-8 codepoint; returns its byte length, 0 on a malformed lead.
int DecodeUtf8(const std::string& s, size_t i, uint32_t* cp) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) { *cp = c; return 1; }
  if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
    *cp = ((c & 0x1FU) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3FU);
    return 2;
  }
  if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
    *cp = ((c & 0x0FU) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3FU) << 6) |
          (static_cast<unsigned char>(s[i + 2]) & 0x3FU);
    return 3;
  }
  if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
    *cp = ((c & 0x07U) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3FU) << 12) |
          ((static_cast<unsigned char>(s[i + 2]) & 0x3FU) << 6) |
          (static_cast<unsigned char>(s[i + 3]) & 0x3FU);
    return 4;
  }
  *cp = c;
  return 1;
}

// The NARROW classifier. Each entry is a range this table decides CONFIDENTLY;
// anything outside them sets `exact` false rather than being guessed at.
struct Range { uint32_t lo, hi; };

constexpr Range kLetters[] = {
    {0x41, 0x5A}, {0x61, 0x7A},            // Latin A-Z a-z
    {0xC0, 0x24F},                          // Latin-1 supplement + extended
    {0x370, 0x3FF}, {0x400, 0x4FF},         // Greek, Cyrillic
    {0x3040, 0x309F}, {0x30A0, 0x30FF},     // hiragana, katakana
    {0x3400, 0x4DBF}, {0x4E00, 0x9FFF},     // CJK ideographs
    {0xF900, 0xFAFF},                       // CJK compatibility
    {0x20000, 0x2A6DF},                     // CJK extension B
};
constexpr Range kNumbers[] = {{0x30, 0x39}};
// Ranges the table knows it does NOT decide: everything else is "unknown", but
// pure ASCII punctuation and whitespace ARE decided, since the pattern's third
// and fourth alternatives cover them exactly.
constexpr Range kDecidedOther[] = {{0x20, 0x2F}, {0x3A, 0x40}, {0x5B, 0x60},
                                   {0x7B, 0x7E}, {0x09, 0x0D}};

bool In(const Range* table, size_t n, uint32_t cp) {
  for (size_t i = 0; i < n; ++i) {
    if (cp >= table[i].lo && cp <= table[i].hi) return true;
  }
  return false;
}

bool IsLetter(uint32_t cp) { return In(kLetters, sizeof(kLetters) / sizeof(Range), cp); }
bool IsNumber(uint32_t cp) { return In(kNumbers, sizeof(kNumbers) / sizeof(Range), cp); }
bool IsSpace(uint32_t cp) { return cp == 0x20 || (cp >= 0x09 && cp <= 0x0D); }
bool Decided(uint32_t cp) {
  return IsLetter(cp) || IsNumber(cp) ||
         In(kDecidedOther, sizeof(kDecidedOther) / sizeof(Range), cp);
}

}  // namespace

Ranks LoadRanks(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("tiktoken: cannot open '" + path + "'");
  }
  Ranks ranks;
  std::string line;
  int64_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string b64;
    int64_t rank = -1;
    if (!(ss >> b64 >> rank) || rank < 0) {
      throw std::runtime_error("tiktoken: malformed entry at line " +
                               std::to_string(lineno) + " of '" + path + "'");
    }
    ranks[Base64Decode(b64)] = rank;
  }
  if (ranks.empty()) {
    throw std::runtime_error("tiktoken: '" + path + "' held no entries");
  }
  return ranks;
}

std::vector<int64_t> BytePairEncode(const std::string& piece, const Ranks& ranks) {
  if (piece.empty()) return {};
  if (piece.size() == 1) {
    const auto it = ranks.find(piece);
    if (it == ranks.end()) {
      throw std::runtime_error("tiktoken: byte not in the vocabulary");
    }
    return {it->second};
  }

  // tiktoken's algorithm: hold the split points, repeatedly merge the adjacent
  // pair with the LOWEST rank. Merging by first-match instead of lowest rank
  // still produces tokens, just different ones.
  constexpr int64_t kNone = std::numeric_limits<int64_t>::max();
  std::vector<std::pair<size_t, int64_t>> parts;
  parts.reserve(piece.size() + 1);
  auto rank_at = [&](size_t start, size_t end) -> int64_t {
    const auto it = ranks.find(piece.substr(start, end - start));
    return it == ranks.end() ? kNone : it->second;
  };
  for (size_t i = 0; i < piece.size(); ++i) {
    const int64_t r = (i + 2 <= piece.size()) ? rank_at(i, i + 2) : kNone;
    parts.emplace_back(i, r);
  }
  parts.emplace_back(piece.size(), kNone);

  while (parts.size() > 1) {
    int64_t best = kNone;
    size_t at = 0;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      if (parts[i].second < best) {
        best = parts[i].second;
        at = i;
      }
    }
    if (best == kNone) break;
    parts.erase(parts.begin() + static_cast<ptrdiff_t>(at) + 1);
    parts[at].second = (at + 2 < parts.size())
                           ? rank_at(parts[at].first, parts[at + 2].first)
                           : kNone;
    if (at > 0) {
      parts[at - 1].second = (at + 1 < parts.size())
                                 ? rank_at(parts[at - 1].first, parts[at + 1].first)
                                 : kNone;
    }
  }

  std::vector<int64_t> out;
  for (size_t i = 0; i + 1 < parts.size(); ++i) {
    const std::string tok = piece.substr(parts[i].first, parts[i + 1].first - parts[i].first);
    const auto it = ranks.find(tok);
    if (it == ranks.end()) {
      throw std::runtime_error("tiktoken: merged token is not in the vocabulary");
    }
    out.push_back(it->second);
  }
  return out;
}

std::vector<std::string> Pretokenize(const std::string& text, bool* exact) {
  if (exact != nullptr) *exact = true;
  std::vector<std::string> pieces;
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = 0;
    int len = DecodeUtf8(text, i, &cp);
    // NOT checked here. Each branch below re-checks every codepoint it consumes,
    // including this first one, so a check at the top is a second copy of the
    // same rule. Mutation testing is what surfaced it: deleting this line kept
    // every gate green, which is the signature of a redundant check rather than
    // a weak test.

    // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`, `\s+`.
    size_t start = i;
    bool leading_space = false;
    if (IsSpace(cp) && cp == 0x20 && i + len < text.size()) {
      uint32_t next = 0;
      const int nlen = DecodeUtf8(text, i + static_cast<size_t>(len), &next);
      (void)nlen;
      if (!IsSpace(next)) {
        leading_space = true;
        i += static_cast<size_t>(len);
        len = DecodeUtf8(text, i, &cp);
      }
    }
    (void)leading_space;

    if (IsSpace(cp)) {
      while (i < text.size()) {
        uint32_t c2 = 0;
        const int l2 = DecodeUtf8(text, i, &c2);
        if (!IsSpace(c2)) break;
        if (exact != nullptr && !Decided(c2)) *exact = false;
        i += static_cast<size_t>(l2);
      }
    } else if (IsLetter(cp)) {
      while (i < text.size()) {
        uint32_t c2 = 0;
        const int l2 = DecodeUtf8(text, i, &c2);
        if (!IsLetter(c2)) break;
        if (exact != nullptr && !Decided(c2)) *exact = false;
        i += static_cast<size_t>(l2);
      }
    } else if (IsNumber(cp)) {
      while (i < text.size()) {
        uint32_t c2 = 0;
        const int l2 = DecodeUtf8(text, i, &c2);
        if (!IsNumber(c2)) break;
        if (exact != nullptr && !Decided(c2)) *exact = false;
        i += static_cast<size_t>(l2);
      }
    } else {
      while (i < text.size()) {
        uint32_t c2 = 0;
        const int l2 = DecodeUtf8(text, i, &c2);
        if (IsSpace(c2) || IsLetter(c2) || IsNumber(c2)) break;
        if (exact != nullptr && !Decided(c2)) *exact = false;
        i += static_cast<size_t>(l2);
      }
    }
    pieces.push_back(text.substr(start, i - start));
  }
  return pieces;
}

std::vector<int64_t> Encode(const std::string& text, const Ranks& ranks, bool* exact) {
  std::vector<int64_t> out;
  for (const std::string& piece : Pretokenize(text, exact)) {
    const std::vector<int64_t> ids = BytePairEncode(piece, ranks);
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

}  // namespace tiktoken
}  // namespace models
}  // namespace vllm
