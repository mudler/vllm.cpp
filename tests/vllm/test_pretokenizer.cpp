// vllm.cpp original (tokenizer); semantics mirror HF tokenizers' Split
// pre-tokenizer. Golden expectations below were produced by running the real
// HF tokenizers regex engine (NOT hand-guessed):
//   scp tools/gen_pretok_goldens.py dgx.casa:/tmp/ &&
//   ssh dgx.casa '~/venvs/vllm-oracle/bin/python /tmp/gen_pretok_goldens.py'
//     > tests/vllm/pretokenizer_goldens.inc
// The inline expected pieces in the TEST_CASEs are copied from that same
// oracle output (they also appear in the golden table; duplicated here as
// readable documentation of the split behavior).
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/tokenizer/pretokenizer.h"

using vllm::tok::Pretokenize;
using vllm::tok::SplitPattern;

namespace {

// Runs Pretokenize and checks the structural property on the way: spans are
// contiguous, non-empty, non-overlapping, and cover the input exactly.
std::vector<std::string> Pieces(std::string_view text, SplitPattern p) {
  const auto spans = Pretokenize(text, p);
  std::vector<std::string> out;
  size_t prev = 0;
  for (const auto& [b, e] : spans) {
    REQUIRE(b == prev);
    REQUIRE(e > b);
    REQUIRE(e <= text.size());
    out.emplace_back(text.substr(b, e - b));
    prev = e;
  }
  REQUIRE(prev == text.size());
  return out;
}

std::vector<std::string> QwenPieces(std::string_view t) {
  return Pieces(t, SplitPattern::kQwen2);
}
std::vector<std::string> LlamaPieces(std::string_view t) {
  return Pieces(t, SplitPattern::kLlama3);
}
std::vector<std::string> ClassicPieces(std::string_view t) {
  return Pieces(t, SplitPattern::kQwen2Classic);
}
std::vector<std::string> Gpt2Pieces(std::string_view t) {
  return Pieces(t, SplitPattern::kGpt2);
}

using V = std::vector<std::string>;

struct PretokGolden {
  std::string_view input;
  std::vector<std::string_view> qwen;
  std::vector<std::string_view> llama;
};

// sizeof-based so embedded NUL bytes survive.
#define SV(lit) std::string_view(lit, sizeof(lit) - 1)
#include "pretokenizer_goldens.inc"
#undef SV

}  // namespace

TEST_CASE("basic word split") {
  CHECK(QwenPieces("Hello world") == V{"Hello", " world"});
  CHECK(LlamaPieces("Hello world") == V{"Hello", " world"});
}

TEST_CASE("contraction + double space + trailing newlines") {
  // "I'm  fine\n\n": 'm is a contraction; of the two spaces the first is a
  // \s+(?!\S) match and the second sticks to "fine" (rule 2 prefix); the
  // newline run is one token.
  CHECK(QwenPieces("I'm  fine\n\n") == V{"I", "'m", " ", " fine", "\n\n"});
}

TEST_CASE("digits: Qwen splits singly, Llama-3 groups up to three") {
  CHECK(QwenPieces("x123") == V{"x", "1", "2", "3"});
  CHECK(LlamaPieces("x123") == V{"x", "123"});
  // Greedy {1,3}: groups of three from the left.
  CHECK(LlamaPieces("1234567") == V{"123", "456", "7"});
  CHECK(QwenPieces("a 1") == V{"a", " ", "1"});  // rule 7 catches the space
}

TEST_CASE("CJK letters take an optional space prefix") {
  CHECK(QwenPieces(" \xE4\xBD\xA0\xE5\xA5\xBD") ==
        V{" \xE4\xBD\xA0\xE5\xA5\xBD"});  // " 你好"
}

TEST_CASE("punct run between letters") {
  CHECK(QwenPieces("a...b") == V{"a", "...", "b"});
}

TEST_CASE("kQwen2Classic: combining marks split off the letter run (unlike kQwen2)") {
  // The ONLY behavioral difference between kQwen2 (\p{M}-aware, Qwen3.6) and
  // kQwen2Classic (classic Qwen2/Qwen3, e.g. Qwen3-0.6B) is \p{M} handling:
  // kQwen2 folds a combining mark into the adjacent letter run; kQwen2Classic
  // treats it as ordinary punct-run text. "e" + U+0301 (combining acute):
  const std::string em = "e\xCC\x81";
  CHECK(QwenPieces(em) == V{"e\xCC\x81"});         // one \p{M}-aware run
  CHECK(ClassicPieces(em) == V{"e", "\xCC\x81"});  // classic: mark splits off
  // Number grouping is single-codepoint for BOTH Qwen variants (only Llama-3
  // groups \p{N}{1,3}); the plain ASCII split is otherwise identical to kQwen2.
  CHECK(ClassicPieces("x123") == V{"x", "1", "2", "3"});
  CHECK(ClassicPieces("Hello world") == V{"Hello", " world"});
}

TEST_CASE("punct run absorbs trailing newlines") {
  CHECK(QwenPieces("hi!!\n\nok") == V{"hi", "!!\n\n", "ok"});
  CHECK(QwenPieces("foo !!! bar?\r\n") == V{"foo", " !!!", " bar", "?\r\n"});
}

TEST_CASE("tabs and spaces mix") {
  CHECK(QwenPieces("\tfoo \t bar  ") ==
        V{"\tfoo", " \t", " bar", "  "});
}

TEST_CASE("trailing spaces match to end of input") {
  CHECK(QwenPieces("trailing   ") == V{"trailing", "   "});
}

TEST_CASE("whitespace before non-space leaves last space to next token") {
  CHECK(QwenPieces("a  b   c") == V{"a", " ", " b", "  ", " c"});
  CHECK(QwenPieces("  \n \n  x\n") == V{"  \n \n", " ", " x", "\n"});
}

TEST_CASE("contractions are case-insensitive and unconditional") {
  // Note " 'd": at the space, rule 4 (` ?punct+`) wins with " '" before the
  // contraction rule ever gets a chance at the apostrophe (leftmost match).
  CHECK(QwenPieces("I'M I'll DON'T can'tt 'd 'vex") ==
        V{"I", "'M", " I", "'ll", " DON", "'T", " can", "'t", "t", " '", "d",
          " '", "vex"});
  // U+017F simple-case-folds to 's' in onig (?i:...).
  CHECK(QwenPieces("a'\xC5\xBF" "b") == V{"a", "'\xC5\xBF", "b"});
}

TEST_CASE("combining marks: Qwen letter run includes \\p{M}, Llama-3 not") {
  // "word" + U+0301 (combining acute, UTF-8 CC 81) then " " + U+0301 + "word"
  const std::string s = "word\xCC\x81 \xCC\x81word";
  CHECK(QwenPieces(s) == V{"word\xCC\x81", " \xCC\x81word"});
  // Llama-3: marks are in rule 4's punct class, so the trailing mark splits
  // off and " \xCC\x81" forms a space+punct run, leaving "word" bare.
  CHECK(LlamaPieces(s) == V{"word", "\xCC\x81", " \xCC\x81", "word"});
}

TEST_CASE("goldens: scanner matches the HF tokenizers oracle") {
  for (const PretokGolden& g : kPretokGoldens) {
    CAPTURE(g.input);
    const auto q = QwenPieces(g.input);
    const auto l = LlamaPieces(g.input);
    REQUIRE(q.size() == g.qwen.size());
    for (size_t i = 0; i < q.size(); ++i) CHECK(q[i] == g.qwen[i]);
    REQUIRE(l.size() == g.llama.size());
    for (size_t i = 0; i < l.size(); ++i) CHECK(l[i] == g.llama[i]);
  }
}

namespace {

// Deterministic xorshift PRNG so failures reproduce.
struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  uint64_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  uint32_t Below(uint32_t n) { return static_cast<uint32_t>(Next() % n); }
};

}  // namespace

TEST_CASE("reconstruction property over random strings (ASCII + UTF-8 + "
          "invalid bytes)") {
  static constexpr std::string_view kSnippets[] = {
      "a", "Z", "9", " ", "\t", "\n", "\r", "'", "'s", ".", "!?", "_",
      "\xC3\xA9",          // é
      "\xCC\x81",          // U+0301 combining acute
      "\xE4\xBD\xA0",      // 你
      "\xF0\x9F\x99\x82",  // 🙂
      "\xC2\xA0",          // NBSP
      "\xE3\x80\x80",      // U+3000 ideographic space
      "\xC5\xBF",          // U+017F long s
      "\x1C", "\x00",      // isspace-vs-White_Space edge, NUL
      "\x80", "\xFF", "\xE9\xA7",  // invalid UTF-8 (lone cont., bad lead,
                                   // truncated 3-byte)
  };
  Rng rng;
  for (int iter = 0; iter < 2000; ++iter) {
    std::string s;
    const uint32_t n = rng.Below(24);
    for (uint32_t i = 0; i < n; ++i) {
      const auto& sn = kSnippets[rng.Below(sizeof(kSnippets) /
                                           sizeof(kSnippets[0]))];
      s.append(sn.data(), sn.size());
    }
    for (SplitPattern p : {SplitPattern::kQwen2, SplitPattern::kLlama3}) {
      // Pieces() REQUIREs contiguity/cover; also re-join to be explicit.
      const auto pieces = Pieces(s, p);
      std::string joined;
      for (const auto& piece : pieces) joined += piece;
      REQUIRE(joined == s);
    }
  }
}

TEST_CASE("empty and single-char inputs") {
  CHECK(Pretokenize("", SplitPattern::kQwen2).empty());
  CHECK(QwenPieces(" ") == V{" "});
  CHECK(QwenPieces("\n") == V{"\n"});
  CHECK(QwenPieces("'") == V{"'"});
  CHECK(QwenPieces("a") == V{"a"});
  CHECK(QwenPieces("7") == V{"7"});
}

// ---------------------------------------------------------------------------
// kGpt2 — the ORIGINAL GPT-2 byte-level split, added by the OPT
// (`OPTForCausalLM`) bring-up. OPT's tokenizer.json carries no explicit Split
// component; it sets ByteLevel `use_regex: true`, which applies:
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// EVERY expectation below is oracle-generated, NOT hand-guessed: produced by
// HF tokenizers itself via
//   tok.pre_tokenizer.pre_tokenize_str(case)
// on ~/models/opt-125m-bf16-st/tokenizer.json (run on dgx with the vllm-oracle
// venv), with the byte-level alphabet un-mapped back to raw bytes
// (Ġ -> ' ', Ċ -> '\n', č -> '\r', ĉ -> '\t').

TEST_CASE("kGpt2: basic words, and the four rules that differ from Qwen/Llama-3") {
  CHECK(Gpt2Pieces("Hello world") == V{"Hello", " world"});

  // (1) Digit runs are UNBOUNDED `\p{N}+` — not Qwen's single codepoint and not
  // Llama-3's groups of three.
  CHECK(Gpt2Pieces("abc123def4567890") == V{"abc", "123", "def", "4567890"});
  CHECK(Gpt2Pieces("The year 2024 had 365 days.") ==
        V{"The", " year", " 2024", " had", " 365", " days", "."});

  // (2) The letter-run prefix is a plain ` ?`, not `[^\r\n\p{L}\p{N}]?`, so a
  // punctuation character before a word does NOT get absorbed into the word.
  CHECK(Gpt2Pieces("punct!!!between???letters") ==
        V{"punct", "!!!", "between", "???", "letters"});

  // (3) There is NO `\s*[\r\n]+` rule and NO `[\r\n]*` punct tail: newlines
  // fall through to the two whitespace rules.
  CHECK(Gpt2Pieces("tabs\tand\nnewlines\r\n here") ==
        V{"tabs", "\t", "and", "\n", "newlines", "\r\n", " here"});
  CHECK(Gpt2Pieces("\n\n\n") == V{"\n\n\n"});

  // (4) Contractions are CASE-SENSITIVE (no `(?i:)` wrapper): 't matches, 'S
  // does not, and an apostrophe not directly after a letter run is punctuation.
  CHECK(Gpt2Pieces("don't stop  believing\n\n") ==
        V{"don", "'t", " stop", " ", " believing", "\n\n"});
  CHECK(Gpt2Pieces("It'S a 'Test' 'll 'RE") ==
        V{"It", "'", "S", " a", " '", "Test", "'", " '", "ll", " '", "RE"});
}

TEST_CASE("kGpt2: whitespace runs, marks and non-ASCII") {
  // `\s+(?!\S)` leaves the LAST whitespace codepoint attached to the following
  // token; a run at end of input matches whole.
  CHECK(Gpt2Pieces("  leading and trailing   ") ==
        V{" ", " leading", " and", " trailing", "   "});
  CHECK(Gpt2Pieces("x  \n  y") == V{"x", "  \n ", " y"});

  // Combining marks are NOT part of the letter run (GPT-2 has no \p{M}
  // awareness); they land in the punct-run class. "áb" here is DECOMPOSED
  // (a + U+0301).
  CHECK(Gpt2Pieces("a\u0301b combining") == V{"a", "\u0301", "b", " combining"});

  // CJK are letters, so they form one run behind the optional space prefix.
  CHECK(Gpt2Pieces("CJK \u4f60\u597d\u4e16\u754c test") ==
        V{"CJK", " \u4f60\u597d\u4e16\u754c", " test"});
}

TEST_CASE("kGpt2: empty and single-char inputs tile exactly") {
  CHECK(Pretokenize("", SplitPattern::kGpt2).empty());
  CHECK(Gpt2Pieces(" ") == V{" "});
  CHECK(Gpt2Pieces("\n") == V{"\n"});
  CHECK(Gpt2Pieces("'") == V{"'"});
  CHECK(Gpt2Pieces("a") == V{"a"});
  CHECK(Gpt2Pieces("7") == V{"7"});
}
