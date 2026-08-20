// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE (GPT-2 bytes_to_unicode bijection + merge-ranked pair merging).
#include "vllm/tokenizer/bpe.h"

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

namespace {

// Packs an identifier pair into the merge table's key. Mirrors upstream's
// `Pair = (u32, u32)` (`tokenizers/src/models/bpe/mod.rs:9`); one 64-bit
// integer replaces the "left<SP>right" string the old table needed.
constexpr uint64_t PairKey(MergeRanks::SymbolId left,
                           MergeRanks::SymbolId right) {
  return (static_cast<uint64_t>(left) << 32) | static_cast<uint64_t>(right);
}

}  // namespace

MergeRanks::SymbolId MergeRanks::Intern(std::string_view symbol) {
  const auto it = ids_.find(symbol);
  if (it != ids_.end()) return it->second;
  const SymbolId id = static_cast<SymbolId>(texts_.size());
  // kNoSymbol is reserved, so a table this large is a corrupt checkpoint
  // rather than a limit worth raising.
  if (id == kNoSymbol) {
    throw std::runtime_error("tokenizer: merge table names too many symbols");
  }
  const auto [pos, inserted] = ids_.emplace(std::string(symbol), id);
  (void)inserted;
  texts_.push_back(&pos->first);
  return id;
}

bool MergeRanks::Insert(std::string_view left, std::string_view right,
                        int32_t rank) {
  // Upstream interns all three tokens at load and refuses when any is absent
  // from the vocabulary (`model.rs:174-192`). The vocabulary rule lives in
  // `src/vllm/tokenizer/tokenizer.cpp::InsertMerge`, which owns the vocabulary;
  // this function owns the table.
  std::string merged;
  merged.reserve(left.size() + right.size());
  merged.append(left);
  merged.append(right);  // prefix_len is 0 for us; see the header
  const SymbolId left_id = Intern(left);
  const SymbolId right_id = Intern(right);
  const SymbolId new_id = Intern(merged);
  return merges_.emplace(PairKey(left_id, right_id), Entry{rank, new_id})
      .second;
}

MergeRanks::SymbolId MergeRanks::Find(std::string_view symbol) const {
  const auto it = ids_.find(symbol);
  return it == ids_.end() ? kNoSymbol : it->second;
}

const std::string& MergeRanks::Text(SymbolId id) const {
  return *texts_.at(id);
}

const MergeRanks::Entry* MergeRanks::Find(SymbolId left, SymbolId right) const {
  if (left == kNoSymbol || right == kNoSymbol) return nullptr;
  const auto it = merges_.find(PairKey(left, right));
  return it == merges_.end() ? nullptr : &it->second;
}

void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks) {
  // Repeatedly merge the lowest-ranked adjacent pair; leftmost wins ties
  // (strict < keeps the first best).
  //
  // STILL the O(n^2) rescan. This revision moves only the REPRESENTATION: the
  // probe is an identifier pair rather than a freshly built "left<SP>right"
  // string. Removing a constant from a quadratic leaves a quadratic, and
  // `.agents/specs/bpe-quadratic-merge.md` §Design says so explicitly so that
  // nobody mistakes this step for the fix. The heap of `Word::merge_all`
  // (`tokenizers/src/models/bpe/word.rs:162-250`) is what removes the n^2 term,
  // and it needs this representation to express upstream's `new_id` staleness
  // test (`word.rs:197-205`).
  if (symbols.size() < 2) return;
  std::vector<MergeRanks::SymbolId> ids;
  ids.reserve(symbols.size());
  for (const std::string& s : symbols) ids.push_back(ranks.Find(s));

  while (symbols.size() >= 2) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_i = 0;
    const MergeRanks::Entry* best = nullptr;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const MergeRanks::Entry* e = ranks.Find(ids[i], ids[i + 1]);
      if (e != nullptr && e->rank < best_rank) {
        best_rank = e->rank;
        best_i = i;
        best = e;
      }
    }
    if (best == nullptr) break;  // no mergeable pair left
    // `Text(new_id)` IS `symbols[best_i] + symbols[best_i + 1]`: the identifier
    // was interned on that concatenation. The assignment therefore replaces a
    // string append, and at W3 it disappears from the loop entirely.
    symbols[best_i] = ranks.Text(best->new_id);
    ids[best_i] = best->new_id;
    const auto at = static_cast<std::ptrdiff_t>(best_i) + 1;
    symbols.erase(symbols.begin() + at);
    ids.erase(ids.begin() + at);
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
