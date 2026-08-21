// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE (GPT-2 bytes_to_unicode bijection + merge-ranked pair merging).
#include "vllm/tokenizer/bpe.h"

#include <algorithm>
#include <array>
#include <cstddef>
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

namespace {

// Mirrors `Symbol` (`tokenizers/src/models/bpe/word.rs:37-43`). `len` is the
// byte length of the symbol's text, and `len == 0` TAGS A REMOVED SYMBOL rather
// than erasing it (`word.rs:210`), so no index in the queue ever shifts. The
// final sweep drops them all at once (`word.rs:249`).
struct Symbol {
  MergeRanks::SymbolId id;  // upstream's `c`
  int32_t prev;             // -1 when there is none
  int32_t next;             // -1 when there is none
  uint32_t len;             // 0 == removed
};

// Mirrors `Merge` (`word.rs:7-12`): a candidate merge at a position, carrying
// the rank that orders it and the `new_id` its staleness test compares.
struct Merge {
  uint32_t pos;
  int32_t rank;
  MergeRanks::SymbolId new_id;
};

// Mirrors `Ord for Merge` (`word.rs:28-35`), which INVERTS both comparisons so
// that the max-heap yields the LOWEST rank first and the LOWEST position on a
// tie. That is the same leftmost-wins rule the old `strict <` scan had, and it
// is stated here because a heap that silently reverses a tie is exactly the
// change a token gate over ordinary text would not catch.
//
// Returns true when `a` sorts BELOW `b`, i.e. `a` is the worse candidate.
constexpr bool MergeBelow(const Merge& a, const Merge& b) {
  if (a.rank != b.rank) return a.rank > b.rank;
  return a.pos > b.pos;
}

// A 4-ary max-heap, mirroring upstream's `dary_heap::QuaternaryHeap`
// (`word.rs:3,163`). The ORDER is decided by `MergeBelow` alone, so any correct
// heap yields the same sequence: two entries compare equal only when their rank
// and position are both equal, and rank is the merge's index in the
// checkpoint's merge list, so equal rank means the same pair and therefore the
// same `new_id`. Fully identical entries are interchangeable. The arity is
// therefore a constant, not a semantic — it is mirrored because upstream's is
// the constant this design was measured against.
class QuaternaryHeap {
 public:
  void Reserve(size_t n) { v_.reserve(n); }
  bool Empty() const { return v_.empty(); }

  void Push(const Merge& m) {
    v_.push_back(m);
    SiftUp(v_.size() - 1);
  }

  Merge Pop() {
    const Merge top = v_.front();
    v_.front() = v_.back();
    v_.pop_back();
    if (!v_.empty()) SiftDown(0);
    return top;
  }

 private:
  void SiftUp(size_t i) {
    while (i > 0) {
      const size_t parent = (i - 1) / 4;
      if (!MergeBelow(v_[parent], v_[i])) break;
      std::swap(v_[parent], v_[i]);
      i = parent;
    }
  }

  void SiftDown(size_t i) {
    const size_t n = v_.size();
    for (;;) {
      const size_t first = 4 * i + 1;
      if (first >= n) break;
      const size_t last = std::min(first + 4, n);
      size_t best = first;
      for (size_t c = first + 1; c < last; ++c) {
        if (MergeBelow(v_[best], v_[c])) best = c;
      }
      if (!MergeBelow(v_[i], v_[best])) break;
      std::swap(v_[i], v_[best]);
      i = best;
    }
  }

  std::vector<Merge> v_;
};

}  // namespace

void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks) {
  // Mirrors `Word::merge_all` (`word.rs:162-250`). A linked list of symbols and
  // a heap of candidate merges: pop the best candidate, apply it, and push only
  // the two pairs the merge creates. O(n log n) merges of O(1) work each,
  // against the O(n) rescan this replaces.
  //
  // Upstream also has a WORD CACHE (`model.rs:475-496`), which is deliberately
  // NOT ported: it stores only sequences shorter than `MAX_LENGTH = 256`
  // (`tokenizers/src/utils/cache.rs:10`), so it never touches the regime this
  // row exists for. `.agents/specs/bpe-quadratic-merge.md` §Scope keeps it out
  // and records that adding it is a separate, measurable question.
  //
  // Upstream's `dropout` argument is absent here because we implement no
  // dropout; its only effect is the `skip` list, which is always empty without
  // it (`word.rs:184-186`).
  const size_t n = symbols.size();
  if (n < 2) return;

  std::vector<Symbol> syms;
  syms.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    // A symbol no merge names interns to kNoSymbol, and no merge can apply to
    // it. That is the whole cost of the string probe the old loop paid per
    // candidate, paid once per symbol here.
    syms.push_back(Symbol{ranks.Find(symbols[i]),
                          static_cast<int32_t>(i) - 1,
                          i + 1 < n ? static_cast<int32_t>(i) + 1 : -1,
                          static_cast<uint32_t>(symbols[i].size())});
  }

  // Seed once from every adjacent pair (`word.rs:163-178`).
  QuaternaryHeap queue;
  queue.Reserve(n);
  for (size_t i = 0; i + 1 < n; ++i) {
    const MergeRanks::Entry* e = ranks.Find(syms[i].id, syms[i + 1].id);
    if (e != nullptr) {
      queue.Push(Merge{static_cast<uint32_t>(i), e->rank, e->new_id});
    }
  }

  while (!queue.Empty()) {
    const Merge top = queue.Pop();
    Symbol& left = syms[top.pos];

    // THE THREE VALIDATIONS. A stale candidate is skipped when it is popped,
    // never removed when it goes stale, because the heap cannot find it.
    if (left.len == 0) continue;    // left symbol was removed (`word.rs:187`)
    if (left.next == -1) continue;  // no right neighbour (`word.rs:191`)
    const size_t next_pos = static_cast<size_t>(left.next);
    const Symbol right = syms[next_pos];
    // The third test compares `new_id`, NOT the pair (`word.rs:197-205`):
    // `merges.get(&target_new_pair).is_none_or(|(_, new_id)| *new_id != top.new_id)`.
    // A table with two distinct pairs mapping to one `new_id` accepts the entry
    // where a pair comparison would reject it, and — the case that matters — a
    // table where the pair now at this position produces a DIFFERENT token
    // rejects the entry where a pair comparison would apply it with the wrong
    // `new_id`. `tests/vllm/test_bpe.cpp` builds exactly that table.
    const MergeRanks::Entry* e = ranks.Find(left.id, right.id);
    if (e == nullptr || e->new_id != top.new_id) continue;

    // merge_with (`word.rs:44-50`), then tag the right part removed
    // (`word.rs:210`).
    left.id = top.new_id;
    left.len += right.len;
    left.next = right.next;
    syms[next_pos].len = 0;

    // Update `prev` on the new `next` (`word.rs:212-215`).
    if (right.next > -1 && static_cast<size_t>(right.next) < syms.size()) {
      syms[static_cast<size_t>(right.next)].prev = static_cast<int32_t>(top.pos);
    }

    // The two pairs the merge creates, and only those: with the previous
    // symbol (`word.rs:217-231`) and with the next (`word.rs:233-244`).
    if (left.prev >= 0) {
      const size_t prev = static_cast<size_t>(left.prev);
      const MergeRanks::Entry* pe = ranks.Find(syms[prev].id, left.id);
      if (pe != nullptr) {
        queue.Push(Merge{static_cast<uint32_t>(prev), pe->rank, pe->new_id});
      }
    }
    if (left.next >= 0 && static_cast<size_t>(left.next) < syms.size()) {
      const MergeRanks::Entry* ne =
          ranks.Find(left.id, syms[static_cast<size_t>(left.next)].id);
      if (ne != nullptr) {
        queue.Push(Merge{top.pos, ne->rank, ne->new_id});
      }
    }
  }

  // Filter out the removed symbols (`word.rs:249`), and write the surviving
  // text back. A symbol that merged takes its text from the table: `len` grew,
  // and `Text(id)` IS the concatenation the identifier was interned on. A
  // symbol that did not merge keeps the string it arrived with, so the whole
  // loop above built no string at all.
  size_t out = 0;
  for (size_t i = 0; i < n; ++i) {
    if (syms[i].len == 0) continue;
    if (syms[i].len != symbols[i].size()) {
      symbols[out] = ranks.Text(syms[i].id);
    } else if (out != i) {
      symbols[out] = std::move(symbols[i]);
    }
    ++out;
  }
  symbols.resize(out);
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
