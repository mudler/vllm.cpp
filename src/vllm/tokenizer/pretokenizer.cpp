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

}  // namespace

std::vector<std::pair<size_t, size_t>> Pretokenize(std::string_view text,
                                                   SplitPattern pattern) {
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
