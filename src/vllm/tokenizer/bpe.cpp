// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE (GPT-2 bytes_to_unicode bijection + merge-ranked pair merging).
#include "vllm/tokenizer/bpe.h"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "vllm/tokenizer/unicode_data.h"

namespace vllm::tok {
namespace {

constexpr bool IsPrintableByte(int b) {
  return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
         (b >= 0xAE && b <= 0xFF);
}

// GPT-2 bytes_to_unicode: printable bytes keep their codepoint; the 68
// remaining bytes get 0x100 + n in increasing byte order, so every mapped
// codepoint is < 0x144.
constexpr uint32_t kMappedEnd = 0x144;

struct Tables {
  std::array<uint32_t, 256> byte_to_cp{};
  std::array<int16_t, kMappedEnd> cp_to_byte{};

  Tables() {
    cp_to_byte.fill(-1);
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
      const uint32_t cp =
          IsPrintableByte(b) ? static_cast<uint32_t>(b) : 0x100 + n++;
      byte_to_cp[static_cast<size_t>(b)] = cp;
      cp_to_byte[cp] = static_cast<int16_t>(b);
    }
  }
};

const Tables& GetTables() {
  static const Tables t;
  return t;
}

}  // namespace

uint32_t ByteToUnicode(uint8_t b) { return GetTables().byte_to_cp[b]; }

int32_t UnicodeToByte(uint32_t cp) {
  if (cp >= kMappedEnd) return -1;
  return GetTables().cp_to_byte[cp];
}

std::string MapBytesToUnicode(std::string_view raw) {
  std::string out;
  out.reserve(raw.size() * 2);
  for (const char c : raw) {
    EncodeUtf8(ByteToUnicode(static_cast<uint8_t>(c)), out);
  }
  return out;
}

std::string UnmapUnicodeToBytes(std::string_view mapped) {
  std::string out;
  out.reserve(mapped.size());
  size_t pos = 0;
  while (pos < mapped.size()) {
    const uint32_t cp = DecodeUtf8(mapped, pos);
    const int32_t b = UnicodeToByte(cp);
    if (b < 0) {
      throw std::runtime_error(
          "tokenizer: codepoint U+" + std::to_string(cp) +
          " is not in the byte-level alphabet (not a byte-level BPE token "
          "string)");
    }
    out.push_back(static_cast<char>(b));
  }
  return out;
}

std::string MergeKey(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back(' ');  // never occurs inside a mapped-alphabet symbol
  key.append(right);
  return key;
}

void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks) {
  // Repeatedly merge the lowest-ranked adjacent pair; leftmost wins ties
  // (strict < keeps the first best). O(n^2) scan; pretokens are tiny.
  while (symbols.size() >= 2) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_i = symbols.size();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = ranks.find(MergeKey(symbols[i], symbols[i + 1]));
      if (it != ranks.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i == symbols.size()) break;  // no mergeable pair left
    symbols[best_i] += symbols[best_i + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
  }
}

std::vector<std::string> BpeSplit(std::string_view mapped_pretoken,
                                  const MergeRanks& ranks) {
  // Start from single-codepoint symbols.
  std::vector<std::string> symbols;
  size_t pos = 0;
  while (pos < mapped_pretoken.size()) {
    const size_t begin = pos;
    (void)DecodeUtf8(mapped_pretoken, pos);
    symbols.emplace_back(mapped_pretoken.substr(begin, pos - begin));
  }
  BpeMerge(symbols, ranks);
  return symbols;
}

}  // namespace vllm::tok
