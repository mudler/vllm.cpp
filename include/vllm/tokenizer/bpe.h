// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE: the GPT-2 bytes_to_unicode bijection and the merge-ranked BPE loop
// that HF's BPE model applies per pretoken.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vllm::tok {

// GPT-2 bytes_to_unicode bijection: printable bytes 0x21-0x7E, 0xA1-0xAC and
// 0xAE-0xFF map to their own codepoint; the remaining 68 bytes map, in
// increasing byte order, to 0x100 + n (n = 0..67). Token strings in a
// byte-level BPE vocab are UTF-8 encodings of these mapped codepoints.
uint32_t ByteToUnicode(uint8_t b);

// Reverse of ByteToUnicode; -1 when `cp` is not in the bijection's image.
int32_t UnicodeToByte(uint32_t cp);

// Raw bytes -> UTF-8 string over the mapped alphabet (each input byte becomes
// one mapped codepoint).
std::string MapBytesToUnicode(std::string_view raw);

// Inverse of MapBytesToUnicode. Throws std::runtime_error if the string
// contains a codepoint outside the bijection's image (i.e. it is not a
// byte-level token string).
std::string UnmapUnicodeToBytes(std::string_view mapped);

// The merge table, keyed on a pair of symbol IDENTIFIERS.
//
// Mirrors HF `tokenizers` 0.22.2: `type Pair = (u32, u32)`
// (`tokenizers/src/models/bpe/mod.rs:9`) and
// `pub type MergeMap = AHashMap<Pair, (u32, u32)>`
// (`tokenizers/src/models/bpe/model.rs:19`), built once at load
// (`model.rs:174-192`). The value is `(rank, new_id)`: the merge's index in
// the checkpoint's merge list, and the identifier of the concatenation.
//
// It replaces `unordered_map<std::string, int32_t>` keyed on "left<SP>right",
// which had no upstream counterpart and forced one key string per probe.
// Interning is not a tuning step taken beside the heap of `BpeMerge`; it is
// forced BY it. Upstream's heap entry carries `new_id` (`word.rs:8-12`) and
// its staleness test compares `new_id` (`word.rs:197-205`), and a merge step
// that has to rebuild a `std::string` to name a pair cannot express that test.
//
// ONE ADAPTATION, stated because it is the only place this diverges from
// upstream's shape. Upstream interns to VOCABULARY ids, because the model owns
// the vocabulary. This table owns no vocabulary, so it assigns its own dense
// identifiers in order of first appearance among the merge entries. The two
// numberings differ; the EQUIVALENCE they induce does not. Upstream's `new_id`
// is the vocabulary id of the concatenated string, so two merges share a
// `new_id` exactly when they produce the same string -- and here they share an
// identifier under exactly the same condition, because the identifier is keyed
// on that string. `src/vllm/tokenizer/tokenizer.cpp::InsertMerge` still applies
// upstream's `MergeTokenOutOfVocabulary` rule (`model.rs:180-189`) against the
// real vocabulary at load, which is where the behaviour lives.
//
// A symbol no merge names has no identifier (`kNoSymbol`). That is not a
// limitation: if a merge named it, inserting that merge would have interned it.
// So `kNoSymbol` means "no merge can apply here", which is the answer a string
// probe would have taken a hash of a freshly built key to reach.
class MergeRanks {
 public:
  using SymbolId = uint32_t;
  // Reserved: no merge names this symbol. `Tokenizer::EncodePlainSp` relies on
  // it for the unk sentinel, which is deliberately not a real symbol.
  static constexpr SymbolId kNoSymbol = static_cast<SymbolId>(-1);

  // Upstream's `(u32 rank, u32 new_id)` (`model.rs:19`).
  struct Entry {
    int32_t rank;     // the merge's index in the checkpoint's merge list
    SymbolId new_id;  // the identifier of `left + right`
  };

  // Interns `left`, `right` and their concatenation and records the merge at
  // `rank`. Returns false when this pair is already present; the loader fails
  // loud on that, so the table never silently keeps one of two ranks.
  //
  // The concatenation is `left + right` whole. Upstream writes
  // `format!("{}{}", a, &b[prefix_len..])` (`model.rs:169-173,186`) with
  // `prefix_len = continuing_subword_prefix.len()`, and
  // `src/vllm/tokenizer/tokenizer.cpp::FromHfJson` already REFUSES a non-empty
  // `continuing_subword_prefix` at load, so `prefix_len` is 0 on every
  // checkpoint we accept and the term is inert. Do not drop that refusal
  // without restoring this term.
  bool Insert(std::string_view left, std::string_view right, int32_t rank);

  // The identifier of a symbol string, or `kNoSymbol` when no merge names it.
  SymbolId Find(std::string_view symbol) const;

  // The symbol string of an identifier. `id` must be a real identifier.
  const std::string& Text(SymbolId id) const;

  // The merge for an adjacent identifier pair, or nullptr when there is none.
  const Entry* Find(SymbolId left, SymbolId right) const;

  size_t size() const { return merges_.size(); }
  bool empty() const { return merges_.empty(); }
  // Distinct symbols named by the table. Diagnostic; no behaviour reads it.
  size_t SymbolCount() const { return texts_.size(); }

 private:
  // Heterogeneous lookup, so `Find(std::string_view)` builds no string.
  struct SvHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };

  SymbolId Intern(std::string_view symbol);

  std::unordered_map<std::string, SymbolId, SvHash, std::equal_to<>> ids_;
  // id -> the key string inside `ids_`. `unordered_map` never invalidates a
  // pointer to an element, so these stay valid across rehash.
  std::vector<const std::string*> texts_;
  // (left << 32) | right -> Entry. One 64-bit key, no string.
  std::unordered_map<uint64_t, Entry> merges_;
};

// The merge loop of BpeSplit, factored out so a pre-seeded symbol list can be
// merged directly. Repeatedly merges the adjacent pair with the lowest rank
// (leftmost on ties) until no adjacent pair is in `ranks`; mutates `symbols`
// in place. Mirrors HF `tokenizers` `Word::merge_all`
// (`tokenizers/src/models/bpe/word.rs:162-250`). The SentencePiece path uses
// this after building initial symbols with byte-fallback substitution (HF BPE
// constructs the Word — including the byte-fallback decomposition of unknown
// characters — BEFORE running merges).
void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks);

// Applies BPE to one pretoken already in the mapped alphabet: start from
// single-codepoint symbols, repeatedly merge the adjacent pair with the
// lowest rank (leftmost on ties) until no adjacent pair is in `ranks`.
// Returns the final symbol strings (concatenation == mapped_pretoken).
std::vector<std::string> BpeSplit(std::string_view mapped_pretoken,
                                  const MergeRanks& ranks);

}  // namespace vllm::tok
