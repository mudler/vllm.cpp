// vllm.cpp original (tokenizer); semantics mirror HF tokenizers' Split
// pre-tokenizer (behavior=Isolated) with the byte-level BPE split regexes.
//
// Authoritative patterns (verbatim from tokenizer.json "pre_tokenizer"):
//
// kQwen2 — unsloth/Qwen3.6-27B-NVFP4 tokenizer.json (fetched 2026-07-03):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// kQwen2Classic — classic Qwen2/Qwen3 (e.g. Qwen/Qwen3-0.6B, Qwen3-Coder-30B):
//   the same pattern WITHOUT \p{M} in the letter-run and punct classes:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   We implement BOTH: kQwen2 (\p{M}-aware) and kQwen2Classic differ only on
//   combining marks adjacent to letter/punctuation runs (kQwen2Classic treats
//   a mark like ordinary punct-run text, exactly as Llama-3 does, but keeps the
//   single-codepoint \p{N} number grouping). The first ADDITIVE model
//   (Qwen3-0.6B, MODEL-TEXT-qwen3) forced kQwen2Classic in.
//
// kLlama3 — PROVISIONAL: no Llama-3 checkpoint exists in the DGX HF cache to
// read verbatim (checked 2026-07-03); pattern from public knowledge of
// meta-llama/Meta-Llama-3-8B tokenizer.json (cl100k-style):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// The scanner mirrors the regex alternation: at each position the rules are
// tried in order and the first that matches wins (leftmost-alternation
// semantics); the match is emitted as a byte span and scanning resumes at its
// end. Every position matches some rule (regex-space -> rule 5/6/7; letter ->
// rule 2; number -> rule 3; mark -> rule 2 run for Qwen, rule 4 class for
// Llama-3; everything else, incl. non-space controls and U+FFFD from invalid
// UTF-8, is in rule 4's negated class), so Split's "unmatched text between
// matches" case never occurs and the spans tile the input exactly.
//
// Engine notes (oracle-verified against HF tokenizers 'onig' regex engine,
// tools/gen_pretok_goldens.py):
//  * `\s` is Unicode White_Space, NOT python str.isspace(): U+001C..U+001F
//    are isspace-only and behave as rule-4 punctuation for the regex
//    (oracle: ".\x1c." stays a single punct run).
//  * `(?i:...)` uses Unicode simple case folding: besides ASCII letters,
//    U+017F (LATIN SMALL LETTER LONG S) folds to 's' and matches the 's
//    contraction (oracle: "a'ſx" -> ["a", "'ſ", "x"]). No other
//    non-ASCII simple fold hits these letters; multi-char full folds
//    (ss -> ß etc.) do not apply.

#include "vllm/tokenizer/pretokenizer.h"

#include <cstdint>

#include "vllm/tokenizer/unicode_data.h"

namespace vllm::tok {
namespace {

struct Cp {
  uint32_t cp;
  size_t end;  // byte offset just past this codepoint
};

Cp DecodeAt(std::string_view s, size_t pos) {
  size_t p = pos;
  const uint32_t cp = DecodeUtf8(s, p);
  return {cp, p};
}

// Regex `\s` (onig): Unicode White_Space. IsWhitespace is python
// str.isspace(); the sets differ exactly on U+001C..U+001F (isspace-only).
bool IsRegexSpace(uint32_t cp) {
  return IsWhitespace(cp) && !(cp >= 0x1C && cp <= 0x1F);
}

bool IsNewlineByte(char c) { return c == '\r' || c == '\n'; }

char AsciiLower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

// All Match* helpers return the byte offset just past the match, or 0 if the
// rule does not match at `pos` (0 is never a valid match end since matches
// are non-empty). Precondition: pos < t.size().

// Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
size_t MatchContraction(std::string_view t, size_t pos) {
  if (t[pos] != '\'' || pos + 1 >= t.size()) return 0;
  // U+017F (UTF-8 C5 BF) simple-case-folds to 's'.
  if (pos + 3 <= t.size() && static_cast<unsigned char>(t[pos + 1]) == 0xC5 &&
      static_cast<unsigned char>(t[pos + 2]) == 0xBF) {
    return pos + 3;
  }
  const char c1 = AsciiLower(t[pos + 1]);
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
  if (pos + 2 >= t.size()) return 0;
  const char c2 = AsciiLower(t[pos + 2]);
  if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
      (c1 == 'l' && c2 == 'l')) {
    return pos + 3;
  }
  return 0;
}

// Rule 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   (Qwen3.6; marks_in_run=true)
//         [^\r\n\p{L}\p{N}]?\p{L}+          (Llama-3; marks_in_run=false)
// One optional prefix codepoint (anything but \r, \n, letters, numbers),
// then a maximal run of letters (and marks, for Qwen). When the first
// codepoint is itself a run character, taking it as prefix vs. run start
// yields the same match end, so we always treat it as run start.
size_t MatchLetterRun(std::string_view t, size_t pos, bool marks_in_run) {
  const auto is_run = [marks_in_run](uint32_t cp) {
    const UCat c = Category(cp);
    return c == UCat::kLetter || (marks_in_run && c == UCat::kMark);
  };
  const Cp c0 = DecodeAt(t, pos);
  size_t run = pos;
  if (!is_run(c0.cp)) {
    const UCat cat = Category(c0.cp);
    if (c0.cp == U'\r' || c0.cp == U'\n' || cat == UCat::kLetter ||
        cat == UCat::kNumber || c0.end >= t.size()) {
      return 0;
    }
    if (!is_run(DecodeAt(t, c0.end).cp)) return 0;
    run = c0.end;  // prefix consumed
  }
  size_t p = run;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!is_run(c.cp)) break;
    p = c.end;
  }
  return p;
}

// Rule 3: \p{N} (Qwen, max_digits=1) | \p{N}{1,3} (Llama-3, max_digits=3;
// greedy, so long digit runs split into groups of three from the left).
size_t MatchNumbers(std::string_view t, size_t pos, int max_digits) {
  size_t p = pos;
  int count = 0;
  while (p < t.size() && count < max_digits) {
    const Cp c = DecodeAt(t, p);
    if (Category(c.cp) != UCat::kNumber) break;
    p = c.end;
    ++count;
  }
  return count > 0 ? p : 0;
}

// Rule 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` (Qwen3.6; marks_excluded=true)
//         ` ?[^\s\p{L}\p{N}]+[\r\n]*`      (Llama-3; marks_excluded=false)
// Optional single literal ASCII space, then >=1 codepoints that are not
// regex-space/letters/numbers (nor marks, for Qwen), then any trailing \r/\n
// bytes. If only the space matched, the rule fails as a whole.
size_t MatchPunctRun(std::string_view t, size_t pos, bool marks_excluded) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (IsRegexSpace(c.cp)) break;
    const UCat cat = Category(c.cp);
    if (cat == UCat::kLetter || cat == UCat::kNumber ||
        (marks_excluded && cat == UCat::kMark)) {
      break;
    }
    p = c.end;
  }
  if (p == run_begin) return 0;
  while (p < t.size() && IsNewlineByte(t[p])) ++p;
  return p;
}

// Rule 5: \s*[\r\n]+
// Backtracking semantics: greedy \s* over the whitespace run, then [\r\n]+
// backtracks to the LAST \r/\n in the run, consuming exactly it. Net effect:
// match [pos, last_newline_in_run + 1) iff the run contains a newline;
// trailing non-newline whitespace is left for later rules.
size_t MatchWsNewlines(std::string_view t, size_t pos) {
  size_t p = pos;
  size_t after_last_nl = 0;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!IsRegexSpace(c.cp)) break;
    if (c.cp == U'\r' || c.cp == U'\n') after_last_nl = c.end;
    p = c.end;
  }
  return after_last_nl;  // 0 when the run has no \r/\n (or no run at all)
}

// Rule 6: \s+(?!\S)
// Whitespace run not followed by a non-space: the full run matches at end of
// input; otherwise the last whitespace codepoint is left attached to the
// following token (it becomes e.g. the ` ?` prefix of rules 2/4), so the
// match is the run minus its last codepoint — failing entirely when the run
// is a single codepoint.
size_t MatchWsNotBeforeNonSpace(std::string_view t, size_t pos) {
  size_t p = pos;
  size_t last_begin = pos;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!IsRegexSpace(c.cp)) break;
    last_begin = p;
    p = c.end;
  }
  if (p == pos) return 0;      // no whitespace at all
  if (p == t.size()) return p; // (?!\S) holds at end of input
  return last_begin == pos ? 0 : last_begin;
}

// Rule 7: \s+  (maximal whitespace run; the catch-all for e.g. a single
// space before a digit, which no earlier rule absorbs).
size_t MatchWs(std::string_view t, size_t pos) {
  size_t p = pos;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!IsRegexSpace(c.cp)) break;
    p = c.end;
  }
  return p > pos ? p : 0;
}

// ---------------------------------------------------------------------------
// GPT-2 rules. The ORIGINAL byte-level BPE split, applied implicitly by HF's
// ByteLevel pre-tokenizer when `use_regex: true` and no explicit Split
// component is present (facebook/opt-125m, gpt2, and friends):
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Four of the six alternatives differ from the Qwen/Llama-3 family above, so
// they get their own matchers rather than extra flags on the existing ones.

// GPT-2 rule 1: 's|'t|'re|'ve|'m|'ll|'d — CASE-SENSITIVE (no `(?i:)` wrapper),
// so ASCII lowercase only and no U+017F long-s fold.
size_t MatchGpt2Contraction(std::string_view t, size_t pos) {
  if (t[pos] != '\'' || pos + 1 >= t.size()) return 0;
  const char c1 = t[pos + 1];
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
  if (pos + 2 >= t.size()) return 0;
  const char c2 = t[pos + 2];
  if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
      (c1 == 'l' && c2 == 'l')) {
    return pos + 3;
  }
  return 0;
}

// GPT-2 rules 2 and 3: ` ?\p{L}+` and ` ?\p{N}+`.
// An optional single literal ASCII space, then a MAXIMAL run of the wanted
// category. Note both differences vs the Qwen form: the prefix is a plain space
// (not "any non-letter/number codepoint"), and digit runs are unbounded rather
// than grouped.
size_t MatchGpt2CategoryRun(std::string_view t, size_t pos, UCat want) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (Category(c.cp) != want) break;
    p = c.end;
  }
  return p == run_begin ? 0 : p;
}

// GPT-2 rule 4: ` ?[^\s\p{L}\p{N}]+` — as MatchPunctRun but with NO trailing
// `[\r\n]*` (GPT-2 has no newline-absorbing tail and no `\s*[\r\n]+` rule, so
// newlines fall through to the two whitespace rules).
size_t MatchGpt2PunctRun(std::string_view t, size_t pos) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (IsRegexSpace(c.cp)) break;
    const UCat cat = Category(c.cp);
    if (cat == UCat::kLetter || cat == UCat::kNumber) break;
    p = c.end;
  }
  return p == run_begin ? 0 : p;
}

std::vector<std::pair<size_t, size_t>> PretokenizeGpt2(std::string_view text) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = MatchGpt2Contraction(text, pos);
    if (end == 0) end = MatchGpt2CategoryRun(text, pos, UCat::kLetter);
    if (end == 0) end = MatchGpt2CategoryRun(text, pos, UCat::kNumber);
    if (end == 0) end = MatchGpt2PunctRun(text, pos);
    if (end == 0) end = MatchWsNotBeforeNonSpace(text, pos);
    if (end == 0) end = MatchWs(text, pos);
    if (end == 0) end = DecodeAt(text, pos).end;  // forward-progress guarantee
    spans.emplace_back(pos, end);
    pos = end;
  }
  return spans;
}

// ---------------------------------------------------------------------------
// DeepSeek rules (DeepSeek-V2 / V2-Lite / V3). MLA campaign W8.
//
// This family is NOT another alternation regex. Its tokenizer.json declares a HF
// `Sequence` PIPELINE of seven pre-tokenizers, and HF applies them in order,
// each one further splitting the pieces the previous one produced:
//
//   0. Split(Regex("[\r\n]"),        Isolated)
//   1. Split(Regex("\s?[<cased letters>]+"), Isolated)
//   2. Split(Regex("\s?[<ascii + fullwidth + CJK punctuation>]+"), Isolated)
//   3. Split(Regex("\s+$"),          Isolated)
//   4. Split(Regex("[<CJK ideographs + Hangul>]+"), Isolated)
//   5. Digits(individual_digits=true)
//   6. ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false)
//
// Two consequences worth stating, because they are what make a single-pass
// alternation scanner the WRONG shape here:
//  * stage 2's class `[!-/:-~…]+` spans 0x3A-0x7E, which CONTAINS A-Z and a-z.
//    It is only correct because stage 1 already isolated the letter runs. Order
//    is load-bearing, so the implementation is a genuine pipeline.
//  * stage 3's `\s+$` anchors to the end of the PIECE. `$` in onig is
//    end-of-LINE, not end-of-string — but stage 0 has already isolated every
//    \r and \n into its own piece, so no piece reaching stage 3 contains a
//    newline and the two readings coincide. (Recorded rather than relied on
//    silently.)
//
// The classes are ENUMERATED codepoint ranges in the checkpoint, not `\p{...}`
// properties, so they are transcribed here verbatim from
// deepseek-ai/DeepSeek-V2-Lite tokenizer.json (snapshot 604d5664, read
// 2026-07-22) and the tokenizer loader compares the checkpoint's regex strings
// against the same verbatim patterns before selecting this family — a DeepSeek
// variant that ships different ranges is REFUSED loudly instead of being
// mis-tokenized.

// Stage 1: "\s?[A-Za-zµÀ-Ö…]+" — 87 ranges, ascending and disjoint as shipped.
constexpr uint32_t kDsLetterRanges[][2] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00B5, 0x00B5}, {0x00C0, 0x00D6},
    {0x00D8, 0x00F6}, {0x00F8, 0x01BA}, {0x01BC, 0x01BF}, {0x01C4, 0x0293},
    {0x0295, 0x02AF}, {0x0370, 0x0373}, {0x0376, 0x0376}, {0x0377, 0x0377},
    {0x037B, 0x037D}, {0x037F, 0x037F}, {0x0386, 0x0386}, {0x0388, 0x038A},
    {0x038C, 0x038C}, {0x038E, 0x03A1}, {0x03A3, 0x03F5}, {0x03F7, 0x0481},
    {0x048A, 0x052F}, {0x0531, 0x0556}, {0x10A0, 0x10C5}, {0x13A0, 0x13F5},
    {0x13F8, 0x13FD}, {0x1C90, 0x1CBA}, {0x1CBD, 0x1CBF}, {0x1D00, 0x1D2B},
    {0x1D6B, 0x1D77}, {0x1D79, 0x1D9A}, {0x1E00, 0x1F15}, {0x1F18, 0x1F1D},
    {0x1F20, 0x1F45}, {0x1F48, 0x1F4D}, {0x1F50, 0x1F57}, {0x1F59, 0x1F59},
    {0x1F5B, 0x1F5B}, {0x1F5D, 0x1F5D}, {0x1F5F, 0x1F7D}, {0x1F80, 0x1FB4},
    {0x1FB6, 0x1FBC}, {0x1FBE, 0x1FBE}, {0x1FC2, 0x1FC4}, {0x1FC6, 0x1FCC},
    {0x1FD0, 0x1FD3}, {0x1FD6, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FF4},
    {0x1FF6, 0x1FFC}, {0x2102, 0x2102}, {0x2107, 0x2107}, {0x210A, 0x2113},
    {0x2115, 0x2115}, {0x2119, 0x211D}, {0x2124, 0x2124}, {0x2126, 0x2126},
    {0x2128, 0x2128}, {0x212A, 0x212D}, {0x212F, 0x2134}, {0x2139, 0x2139},
    {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E}, {0x2183, 0x2183},
    {0x2184, 0x2184}, {0x2C00, 0x2C7B}, {0x2C7E, 0x2CE4}, {0x2CEB, 0x2CEE},
    {0x2CF2, 0x2CF2}, {0x2CF3, 0x2CF3}, {0xA640, 0xA66D}, {0xA680, 0xA69B},
    {0xA722, 0xA76F}, {0xA771, 0xA787}, {0xA78B, 0xA78E}, {0xAB70, 0xABBF},
    {0xFB00, 0xFB06}, {0xFB13, 0xFB17}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
    {0x10400, 0x1044F}, {0x104B0, 0x104D3}, {0x104D8, 0x104FB},
    {0x10C80, 0x10CB2}, {0x10CC0, 0x10CF2}, {0x118A0, 0x118DF},
    {0x1E900, 0x1E943},
};

// Stage 2: "\s?[!-/:-~！-／：-～‘-‟　-。]+", sorted here (the checkpoint lists the
// two General-Punctuation/CJK ranges after the fullwidth ones).
constexpr uint32_t kDsPunctRanges[][2] = {
    {0x0021, 0x002F}, {0x003A, 0x007E}, {0x2018, 0x201F},
    {0x3000, 0x3002}, {0xFF01, 0xFF0F}, {0xFF1A, 0xFF5E},
};

// Stage 4: "[一-龥ࠀ-一가-퟿]+" = {0x4E00-0x9FA5, 0x0800-0x4E00, 0xAC00-0xD7FF}.
// The first two OVERLAP at 0x4E00 and are contiguous, so they are merged here
// into 0x0800-0x9FA5; the raw shipped form is recorded above.
constexpr uint32_t kDsCjkRanges[][2] = {
    {0x0800, 0x9FA5},
    {0xAC00, 0xD7FF},
};

template <size_t N>
bool InRanges(uint32_t cp, const uint32_t (&ranges)[N][2]) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (cp < ranges[mid][0]) {
      hi = mid;
    } else if (cp > ranges[mid][1]) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

using Span = std::pair<size_t, size_t>;

// One HF `Split(behavior=Isolated)` stage. For every current piece, scan left to
// right; at each position ask `match` for the end of a match STARTING THERE
// (0 = no match, and matches are never empty). A match is emitted as its own
// piece and the unmatched text before it is emitted as a piece too — that is
// exactly what "Isolated" means. Pieces with no match at all pass through whole.
template <typename Match>
void ApplySplitIsolated(std::string_view text, std::vector<Span>& pieces,
                        Match match) {
  std::vector<Span> out;
  out.reserve(pieces.size());
  for (const Span& piece : pieces) {
    size_t gap_begin = piece.first;
    size_t pos = piece.first;
    while (pos < piece.second) {
      const size_t end = match(text, pos, piece.second);
      if (end == 0) {
        pos = DecodeAt(text, pos).end;
        continue;
      }
      if (pos > gap_begin) out.emplace_back(gap_begin, pos);
      out.emplace_back(pos, end);
      pos = end;
      gap_begin = pos;
    }
    if (piece.second > gap_begin) out.emplace_back(gap_begin, piece.second);
  }
  pieces.swap(out);
}

// `Digits(individual_digits=true)`: HF splits on the digit predicate with
// Isolated behaviour, so EACH digit codepoint becomes its own piece and the
// non-digit stretches between them stay whole. Rust's `char::is_numeric()` is
// the Unicode N category — our UCat::kNumber.
void ApplyDigitsIndividual(std::string_view text, std::vector<Span>& pieces) {
  std::vector<Span> out;
  out.reserve(pieces.size());
  for (const Span& piece : pieces) {
    size_t gap_begin = piece.first;
    size_t pos = piece.first;
    while (pos < piece.second) {
      const Cp c = DecodeAt(text, pos);
      if (Category(c.cp) == UCat::kNumber) {
        if (pos > gap_begin) out.emplace_back(gap_begin, pos);
        out.emplace_back(pos, c.end);
        gap_begin = c.end;
      }
      pos = c.end;
    }
    if (piece.second > gap_begin) out.emplace_back(gap_begin, piece.second);
  }
  pieces.swap(out);
}

// Stage 0: `[\r\n]` — a single newline codepoint.
size_t MatchDsNewline(std::string_view t, size_t pos, size_t /*piece_end*/) {
  return IsNewlineByte(t[pos]) ? pos + 1 : 0;
}

// Stages 1/2: `\s?[CLASS]+` — one OPTIONAL whitespace codepoint (any Unicode
// White_Space, not just ASCII space), then a maximal run of class members.
template <size_t N>
size_t MatchDsWsClassRun(std::string_view t, size_t pos, size_t piece_end,
                         const uint32_t (&ranges)[N][2]) {
  size_t p = pos;
  const Cp c0 = DecodeAt(t, p);
  if (c0.end <= piece_end && IsRegexSpace(c0.cp) && !InRanges(c0.cp, ranges)) {
    p = c0.end;
  }
  const size_t run_begin = p;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !InRanges(c.cp, ranges)) break;
    p = c.end;
  }
  return p == run_begin ? 0 : p;
}

// Stage 3: `\s+$` — a whitespace run that reaches the END of the piece.
size_t MatchDsTrailingWs(std::string_view t, size_t pos, size_t piece_end) {
  size_t p = pos;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !IsRegexSpace(c.cp)) break;
    p = c.end;
  }
  return (p > pos && p == piece_end) ? p : 0;
}

// Stage 4: `[CLASS]+` — a maximal class run with NO whitespace prefix.
template <size_t N>
size_t MatchDsClassRun(std::string_view t, size_t pos, size_t piece_end,
                       const uint32_t (&ranges)[N][2]) {
  size_t p = pos;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !InRanges(c.cp, ranges)) break;
    p = c.end;
  }
  return p == pos ? 0 : p;
}

std::vector<Span> PretokenizeDeepSeek(std::string_view text) {
  std::vector<Span> pieces;
  if (text.empty()) return pieces;
  pieces.emplace_back(0, text.size());
  ApplySplitIsolated(text, pieces, MatchDsNewline);
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsWsClassRun(t, p, e, kDsLetterRanges);
  });
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsWsClassRun(t, p, e, kDsPunctRanges);
  });
  ApplySplitIsolated(text, pieces, MatchDsTrailingWs);
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsClassRun(t, p, e, kDsCjkRanges);
  });
  ApplyDigitsIndividual(text, pieces);
  // Stage 6 is ByteLevel(use_regex=false): it maps bytes, it does not split.
  return pieces;
}

}  // namespace

std::vector<std::pair<size_t, size_t>> Pretokenize(std::string_view text,
                                                   SplitPattern pattern) {
  // DeepSeek is a Sequence PIPELINE, not an alternation (see above).
  if (pattern == SplitPattern::kDeepSeek) return PretokenizeDeepSeek(text);
  // GPT-2's alternation differs in four of six rules, so it runs its own
  // scanner rather than threading more flags through the Qwen/Llama-3 one.
  if (pattern == SplitPattern::kGpt2) return PretokenizeGpt2(text);
  // \p{M} awareness (letter runs absorb marks; punct runs exclude them) is
  // UNIQUE to the Qwen3.6 regex. Classic Qwen2/Qwen3 and Llama-3 both treat
  // marks like ordinary punct-run codepoints. Number grouping is single-digit
  // for BOTH Qwen variants; only Llama-3 groups \p{N}{1,3}.
  const bool marks_aware = pattern == SplitPattern::kQwen2;
  const int max_digits = pattern == SplitPattern::kLlama3 ? 3 : 1;
  std::vector<std::pair<size_t, size_t>> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = MatchContraction(text, pos);
    if (end == 0) end = MatchLetterRun(text, pos, /*marks_in_run=*/marks_aware);
    if (end == 0) end = MatchNumbers(text, pos, max_digits);
    if (end == 0) end = MatchPunctRun(text, pos, /*marks_excluded=*/marks_aware);
    if (end == 0) end = MatchWsNewlines(text, pos);
    if (end == 0) end = MatchWsNotBeforeNonSpace(text, pos);
    if (end == 0) end = MatchWs(text, pos);
    // Unreachable (the rules cover every codepoint class), but guarantee
    // forward progress rather than looping if the tables ever change.
    if (end == 0) end = DecodeAt(text, pos).end;
    spans.emplace_back(pos, end);
    pos = end;
  }
  return spans;
}

}  // namespace vllm::tok
