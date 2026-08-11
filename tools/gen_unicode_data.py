#!/usr/bin/env python3
"""Generate src/vllm/tokenizer/unicode_data.cpp + include/vllm/tokenizer/unicode_data.h.

Emits merged (start, end, cat) codepoint ranges for Unicode general-category
lookup, a letter-CASE table, and a whitespace table with python str.isspace()
semantics, all binary-searched at runtime. Category mapping (major class of
unicodedata.category):

    L* -> kLetter, N* -> kNumber, Z* -> kSeparator, M* -> kMark,
    P* -> kPunct,  S* -> kSymbol, C* -> kControl,
    except Cn (unassigned) which is omitted from the table so lookup misses
    fall through to kOther.

The letter-case table splits L* three ways (Lu+Lt | Ll | Lm+Lo). It exists
because two pretokenizer patterns we implement -- SplitPattern::kTekken and
SplitPattern::kGpt4o -- name the MINOR letter categories rather than \\p{L},
and both use the SAME pair of classes:
    [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]  and  [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]
Lu and Lt always occur together in those regexes, as do Lm and Lo, so three
classes are enough and the table stays small.

ONE table, not two. Tekken (#168) and GPT-4o (#347) landed concurrently and
each proposed its own three-way cut of L*, named LetterCase{kUpper, kLower,
kCaseless} and ULetterSub{kLuLt, kLl, kLmLo}. Those are the SAME partition:
the relabelling kUpper<->kLuLt, kLower<->kLl, kCaseless<->kLmLo is a bijection
over all 0x110000 codepoints, checked exhaustively when the two were merged.
Emitting both would be 1861 duplicated ranges that can only ever drift apart,
so LetterCase is kept (it landed first) and the GPT-4o scanner reads it too.

Whitespace is enumerated directly with str.isspace() over every codepoint
(covers 0x85/0xA0 and friends exactly as Python does) rather than hand-listed.

Regenerate with:  python3 tools/gen_unicode_data.py
The generated files are committed; rerun only when bumping the Unicode data
version (i.e. the Python used to generate) and commit the diff.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = REPO_ROOT / "include" / "vllm" / "tokenizer" / "unicode_data.h"
SOURCE_PATH = REPO_ROOT / "src" / "vllm" / "tokenizer" / "unicode_data.cpp"

MAX_CP = 0x110000  # exclusive

# Must match the UCat enum in the generated header.
CAT_NAMES = [
    "kOther",      # 0
    "kLetter",     # 1
    "kNumber",     # 2
    "kSeparator",  # 3
    "kMark",       # 4
    "kPunct",      # 5
    "kSymbol",     # 6
    "kControl",    # 7
]
MAJOR_TO_CAT = {"L": 1, "N": 2, "Z": 3, "M": 4, "P": 5, "S": 6, "C": 7}


def category_index(cp: int) -> int:
    gc = unicodedata.category(chr(cp))
    if gc == "Cn":  # unassigned: omit from table -> lookup miss -> kOther
        return 0
    return MAJOR_TO_CAT[gc[0]]


def build_category_ranges() -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    run_start, run_cat = None, 0
    for cp in range(MAX_CP + 1):  # +1 sentinel iteration flushes the last run
        cat = category_index(cp) if cp < MAX_CP else -1
        if cat != run_cat:
            if run_cat != 0 and run_start is not None:
                ranges.append((run_start, cp - 1, run_cat))
            run_start, run_cat = cp, cat
    return ranges


# Letter CASE class, needed by the Tekken and GPT-4o pre-tokenizers, whose two
# letter alternatives are BOTH [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and
# [\p{Ll}\p{Lm}\p{Lo}\p{M}] (see SplitPattern::kTekken / ::kGpt4o -- the two
# regexes differ in their quantifiers, digit grouping and contraction handling,
# but not in these two classes). The major-class table above collapses every L*
# into kLetter, which cannot express that split. This is deliberately a SECOND,
# narrow table rather than a re-cut of the category table: every existing
# consumer of UCat keeps the exact bytes it has today.
#   1 = kUpper    (Lu, Lt) -- upper class only
#   2 = kLower    (Ll)     -- lower class only
#   3 = kCaseless (Lm, Lo) -- BOTH classes
# Non-letters are omitted, so a lookup miss means "not a cased letter".
LCASE_NAMES = ["kNotLetter", "kUpper", "kLower", "kCaseless"]
LETTER_SUBCAT_TO_LCASE = {"Lu": 1, "Lt": 1, "Ll": 2, "Lm": 3, "Lo": 3}


def letter_case_index(cp: int) -> int:
    return LETTER_SUBCAT_TO_LCASE.get(unicodedata.category(chr(cp)), 0)


def build_letter_case_ranges() -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    run_start, run_cls = None, 0
    for cp in range(MAX_CP + 1):  # +1 sentinel iteration flushes the last run
        cls = letter_case_index(cp) if cp < MAX_CP else -1
        if cls != run_cls:
            if run_cls != 0 and run_start is not None:
                ranges.append((run_start, cp - 1, run_cls))
            run_start, run_cls = cp, cls
    return ranges


def build_whitespace_ranges() -> list[tuple[int, int]]:
    cps = [cp for cp in range(MAX_CP) if chr(cp).isspace()]
    ranges: list[tuple[int, int]] = []
    for cp in cps:
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], cp)
        else:
            ranges.append((cp, cp))
    return ranges


def banner(what: str, n_cat: int, n_ws: int, n_lc: int) -> str:
    return f"""\
// GENERATED FILE — do not edit by hand.  ({what})
// Generator:  tools/gen_unicode_data.py
// Regenerate: python3 tools/gen_unicode_data.py
// Unicode data version (Python unicodedata.unidata_version): {unicodedata.unidata_version}
// Category ranges: {n_cat}; whitespace ranges: {n_ws}; letter-case ranges: {n_lc}.
// Semantics mirror HF tokenizers byte-level BPE: categories are the major
// Unicode general-category classes (unassigned -> kOther); IsWhitespace is
// python str.isspace().
"""


HEADER_BODY = """\
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

// CASE class of a letter. Needed by SplitPattern::kTekken and ::kGpt4o, whose
// letter alternatives are BOTH [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}] and
// [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]; UCat collapses all of L* into kLetter and
// cannot express that. Deliberately a separate narrow table, so UCat and every
// existing consumer of it are unchanged. kNotLetter for anything outside L*.
// Category(cp) == UCat::kLetter is exactly LetterCaseOf(cp) != kNotLetter;
// this table splits that same set three ways. Marks are NOT in it -- \\p{M} is
// not a letter subcategory, and both regexes above get it from Category().
enum class LetterCase : uint8_t {
  kNotLetter = 0,
  kUpper = 1,     // Lu, Lt -- the UPPER class only
  kLower = 2,     // Ll     -- the LOWER class only
  kCaseless = 3,  // Lm, Lo -- present in BOTH classes
};

LetterCase LetterCaseOf(uint32_t cp);

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
"""

SOURCE_PROLOGUE = """\
#include "vllm/tokenizer/unicode_data.h"

#include <algorithm>
#include <iterator>

namespace vllm::tok {

namespace {

struct CatRange {
  uint32_t start;
  uint32_t end;  // inclusive
  uint8_t cat;   // static_cast<uint8_t>(UCat)
};

struct WsRange {
  uint32_t start;
  uint32_t end;  // inclusive
};

struct LCaseRange {
  uint32_t start;
  uint32_t end;  // inclusive
  uint8_t cls;   // static_cast<uint8_t>(LetterCase)
};
"""

SOURCE_EPILOGUE = """\
}  // namespace

UCat Category(uint32_t cp) {
  const CatRange* first = std::begin(kCategoryRanges);
  const CatRange* last = std::end(kCategoryRanges);
  const CatRange* it = std::upper_bound(
      first, last, cp,
      [](uint32_t v, const CatRange& r) { return v < r.start; });
  if (it == first) return UCat::kOther;
  --it;
  if (cp <= it->end) return static_cast<UCat>(it->cat);
  return UCat::kOther;
}

LetterCase LetterCaseOf(uint32_t cp) {
  const LCaseRange* first = std::begin(kLetterCaseRanges);
  const LCaseRange* last = std::end(kLetterCaseRanges);
  const LCaseRange* it = std::upper_bound(
      first, last, cp,
      [](uint32_t v, const LCaseRange& r) { return v < r.start; });
  if (it == first) return LetterCase::kNotLetter;
  --it;
  if (cp <= it->end) return static_cast<LetterCase>(it->cls);
  return LetterCase::kNotLetter;
}

bool IsWhitespace(uint32_t cp) {
  const WsRange* first = std::begin(kWhitespaceRanges);
  const WsRange* last = std::end(kWhitespaceRanges);
  const WsRange* it = std::upper_bound(
      first, last, cp,
      [](uint32_t v, const WsRange& r) { return v < r.start; });
  if (it == first) return false;
  --it;
  return cp <= it->end;
}

uint32_t DecodeUtf8(std::string_view s, size_t& pos) {
  constexpr uint32_t kReplacement = 0xFFFD;
  if (pos >= s.size()) return kReplacement;
  const unsigned char c0 = static_cast<unsigned char>(s[pos]);
  if (c0 < 0x80) {
    pos += 1;
    return c0;
  }
  size_t len;
  uint32_t cp;
  uint32_t min_cp;
  if ((c0 & 0xE0) == 0xC0) {
    len = 2;
    cp = c0 & 0x1Fu;
    min_cp = 0x80;
  } else if ((c0 & 0xF0) == 0xE0) {
    len = 3;
    cp = c0 & 0x0Fu;
    min_cp = 0x800;
  } else if ((c0 & 0xF8) == 0xF0) {
    len = 4;
    cp = c0 & 0x07u;
    min_cp = 0x10000;
  } else {
    pos += 1;  // lone continuation byte or invalid lead (0xF8..0xFF)
    return kReplacement;
  }
  if (pos + len > s.size()) {
    pos += 1;  // truncated sequence
    return kReplacement;
  }
  for (size_t i = 1; i < len; ++i) {
    const unsigned char c = static_cast<unsigned char>(s[pos + i]);
    if ((c & 0xC0) != 0x80) {
      pos += 1;  // non-continuation tail byte
      return kReplacement;
    }
    cp = (cp << 6) | (c & 0x3Fu);
  }
  if (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
    pos += 1;  // overlong form, surrogate, or beyond Unicode range
    return kReplacement;
  }
  pos += len;
  return cp;
}

void EncodeUtf8(uint32_t cp, std::string& out) {
  if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
    out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
    out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else {
    out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
    out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
    out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  }
}

}  // namespace vllm::tok
"""


def format_rows(rows: list[str], per_line: int) -> str:
    lines = []
    for i in range(0, len(rows), per_line):
        lines.append("    " + " ".join(rows[i : i + per_line]))
    return "\n".join(lines)


def main() -> int:
    cat_ranges = build_category_ranges()
    ws_ranges = build_whitespace_ranges()
    lc_ranges = build_letter_case_ranges()

    header = (
        banner("declarations", len(cat_ranges), len(ws_ranges), len(lc_ranges))
        + HEADER_BODY
    )

    cat_rows = [f"{{0x{s:X}, 0x{e:X}, {c}}}," for s, e, c in cat_ranges]
    ws_rows = [f"{{0x{s:X}, 0x{e:X}}}," for s, e in ws_ranges]
    lc_rows = [f"{{0x{s:X}, 0x{e:X}, {c}}}," for s, e, c in lc_ranges]
    legend = "  // cat legend: " + " ".join(
        f"{i}={n}" for i, n in enumerate(CAT_NAMES) if i != 0
    )
    lc_legend = "  // letter-case legend: " + " ".join(
        f"{i}={n}" for i, n in enumerate(LCASE_NAMES) if i != 0
    )
    source = (
        banner(
            "tables + utf8 helpers", len(cat_ranges), len(ws_ranges), len(lc_ranges)
        )
        + SOURCE_PROLOGUE
        + "\n"
        + legend
        + "\n"
        + f"constexpr CatRange kCategoryRanges[{len(cat_ranges)}] = {{\n"
        + format_rows(cat_rows, 4)
        + "\n};\n\n"
        + f"constexpr WsRange kWhitespaceRanges[{len(ws_ranges)}] = {{\n"
        + format_rows(ws_rows, 6)
        + "\n};\n\n"
        + lc_legend
        + "\n"
        + f"constexpr LCaseRange kLetterCaseRanges[{len(lc_ranges)}] = {{\n"
        + format_rows(lc_rows, 4)
        + "\n};\n\n"
        + SOURCE_EPILOGUE
    )

    HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEADER_PATH.write_text(header, encoding="utf-8")
    SOURCE_PATH.write_text(source, encoding="utf-8")
    print(
        f"wrote {HEADER_PATH.relative_to(REPO_ROOT)} and "
        f"{SOURCE_PATH.relative_to(REPO_ROOT)}: "
        f"{len(cat_ranges)} category ranges, {len(ws_ranges)} whitespace "
        f"ranges, {len(lc_ranges)} letter-case ranges, "
        f"unidata {unicodedata.unidata_version}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
