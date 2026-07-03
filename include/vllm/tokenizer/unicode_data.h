// GENERATED FILE — do not edit by hand.  (declarations)
// Generator:  tools/gen_unicode_data.py
// Regenerate: python3 tools/gen_unicode_data.py
// Unicode data version (Python unicodedata.unidata_version): 15.0.0
// Category ranges: 1563; whitespace ranges: 10.
// Semantics mirror HF tokenizers byte-level BPE: categories are the major
// Unicode general-category classes (unassigned -> kOther); IsWhitespace is
// python str.isspace().
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace vllm::tok {

// Major Unicode general-category class. kOther is returned for unassigned
// codepoints (general category Cn) and for values beyond U+10FFFF.
enum class UCat : uint8_t {
  kOther = 0,
  kLetter = 1,     // L*
  kNumber = 2,     // N*
  kSeparator = 3,  // Z*
  kMark = 4,       // M*
  kPunct = 5,      // P*
  kSymbol = 6,     // S*
  kControl = 7,    // C* except Cn
};

UCat Category(uint32_t cp);

// python str.isspace() semantics (per HF byte-level pretokenization):
// includes 0x1C-0x1F, NEL (0x85) and NBSP (0xA0); excludes ZWSP (0x200B),
// Mongolian vowel separator (0x180E) and BOM (0xFEFF).
bool IsWhitespace(uint32_t cp);

// Decodes the UTF-8 codepoint starting at `pos` and advances `pos` past it.
// Invalid input (bad lead byte, truncated or non-continuation tail, overlong
// form, surrogate, > U+10FFFF) yields U+FFFD and advances exactly one byte.
// Precondition: pos < s.size(); otherwise returns U+FFFD, pos unchanged.
uint32_t DecodeUtf8(std::string_view s, size_t& pos);

// Appends the UTF-8 encoding of `cp` to `out`. Invalid codepoints
// (surrogates, > U+10FFFF) encode as U+FFFD.
void EncodeUtf8(uint32_t cp, std::string& out);

}  // namespace vllm::tok
