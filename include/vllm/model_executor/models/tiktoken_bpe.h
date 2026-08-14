// A tiktoken vocabulary reader and BPE encoder (#634).
//
// IndexTTS-2.5 ships `multilingual_zh_ja_yue_char_del.tiktoken` and NO
// `tokenizer.json`, so nothing in this tree could turn its text into tokens --
// the constraint first recorded for Kimi-Linear. This is that reader.
//
// The file is one `base64(token bytes) rank` pair per line, 58836 of them for
// this checkpoint. Encoding is tiktoken's own two steps:
//
//   1. PRETOKENIZE with the pattern upstream passes to `tiktoken.Encoding`
//      (`indextts/utils/tokenizer.py:215`), the standard GPT-2 one:
//        's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//   2. BYTE-PAIR ENCODE each piece independently, repeatedly merging the
//      adjacent pair with the LOWEST rank.
//
// `\p{L}` and `\p{N}` are Unicode CATEGORIES and `std::regex` has no support for
// them, so the classifier here is an explicit range table rather than a library
// call. It is deliberately NARROW: it covers what this vocabulary targets --
// Latin, the CJK ideographs, kana, and the decimal digits -- and
// `PretokenizeIsExact` reports whether every codepoint in a string fell inside
// a range the table decides confidently. A caller that needs a guarantee checks
// it rather than discovering a silent mis-split later.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {
namespace models {
namespace tiktoken {

// token bytes -> rank, as the file stores them.
using Ranks = std::map<std::string, int64_t>;

// Parse the `base64 rank` lines. Throws std::runtime_error naming the line on
// a malformed entry rather than skipping it: a silently dropped token shifts
// nothing visible and changes every encoding that would have used it.
Ranks LoadRanks(const std::string& path);

// tiktoken's `byte_pair_encode` over ONE pretokenized piece.
std::vector<int64_t> BytePairEncode(const std::string& piece, const Ranks& ranks);

// Split on the pattern above. `exact` is set false when any codepoint fell
// outside the ranges the classifier decides confidently.
std::vector<std::string> Pretokenize(const std::string& text, bool* exact);

// Pretokenize then byte-pair encode each piece.
std::vector<int64_t> Encode(const std::string& text, const Ranks& ranks, bool* exact);

}  // namespace tiktoken
}  // namespace models
}  // namespace vllm
