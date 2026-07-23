// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE: the GPT-2 bytes_to_unicode bijection and the merge-ranked BPE loop
// that HF's BPE model applies per pretoken.
#pragma once

#include <cstdint>
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

// Merge table: "left<SP>right" -> rank (lower merges first). A single 0x20
// separator is unambiguous because the mapped alphabet never contains a raw
// space (0x20 maps to U+0120).
using MergeRanks = std::unordered_map<std::string, int32_t>;

std::string MergeKey(std::string_view left, std::string_view right);

// The merge loop of BpeSplit, factored out so a pre-seeded symbol list can be
// merged directly. Repeatedly merges the adjacent pair with the lowest rank
// (leftmost on ties) until no adjacent pair is in `ranks`; mutates `symbols`
// in place. The SentencePiece path uses this after building initial symbols
// with byte-fallback substitution (HF BPE constructs the Word — including the
// byte-fallback decomposition of unknown characters — BEFORE running merges).
void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks);

// Applies BPE to one pretoken already in the mapped alphabet: start from
// single-codepoint symbols, repeatedly merge the adjacent pair with the
// lowest rank (leftmost on ties) until no adjacent pair is in `ranks`.
// Returns the final symbol strings (concatenation == mapped_pretoken).
std::vector<std::string> BpeSplit(std::string_view mapped_pretoken,
                                  const MergeRanks& ranks);

}  // namespace vllm::tok
