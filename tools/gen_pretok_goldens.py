#!/usr/bin/env python3
"""Generate tests/vllm/pretokenizer_goldens.inc from the HF tokenizers oracle.

Runs the exact Split regexes (see src/vllm/tokenizer/pretokenizer.cpp) through
huggingface/tokenizers (Rust onig engine) and emits the resulting pieces as a
C++ table. The C++ scanner in pretokenizer.cpp must reproduce these pieces
byte-for-byte.

Usage (needs the `tokenizers` package; the vllm-oracle venv on dgx has it, and
any host with `pip install tokenizers` will do — the regexes are literals here,
no checkpoint is read):
  python3 tools/gen_pretok_goldens.py > tests/vllm/pretokenizer_goldens.inc
"""

import random
import sys

from tokenizers import Regex
from tokenizers.pre_tokenizers import Split

# Verbatim from unsloth/Qwen3.6-27B-NVFP4 tokenizer.json (pre_tokenizer.
# pretokenizers[0].pattern.Regex, behavior=Isolated, invert=false).
QWEN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Llama-3 family (cl100k-style). PROVISIONAL: taken from public knowledge of
# meta-llama/Meta-Llama-3-8B tokenizer.json (no Llama checkpoint in the DGX HF
# cache to verify against as of 2026-07-03).
LLAMA3 = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Tekken (Mistral). Verbatim from mistralai/Mistral-Nemo-Instruct-2407
# tokenizer.json (pre_tokenizer.pretokenizers[0].pretokenizers[0].pattern.Regex,
# behavior=Isolated, invert=false); the ByteLevel sibling carries
# use_regex=false, so this Split is the whole word-splitting rule.
#
# Structurally this is tiktoken's o200k_base pat_str with exactly two edits:
# the optional `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` group is absent from both letter
# alternatives, and numbers are single-codepoint `\p{N}` rather than
# `\p{N}{1,3}`. Everything from the punct run onward is byte-identical to
# o200k_base. Note the two CASE-AWARE letter alternatives, which no other
# pattern in this file has: an uppercase/caseless run followed by a lowercase
# run, then the same pair with the quantifiers swapped. Alternation is ORDERED,
# so the first is tried first at every position.
TEKKEN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# GPT-4o / o200k family (llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT4O; GGUF pre names
# "gpt-4o", "llama4", "kanana2", "talkie"). VERBATIM from
# /mnt/nas_share/checkpoints/muse-glimmer-30b/tokenizer.json
# (pre_tokenizer.pretokenizers[0].pattern.Regex, read 2026-08-11) — byte-equal
# to the string llama.cpp records as "original regex from tokenizer.json" at
# src/llama-vocab.cpp:432 @ 153d324bcf.
GPT4O = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

DETERMINISTIC = [
    "",
    "Hello world",
    "I'm  fine\n\n",
    "x123",
    " 你好",
    "a...b",
    "\tfoo \t bar  ",
    "trailing   ",
    "I'M I'll DON'T can'tt 'd 'vex",
    "'sX x's ''ll '",
    "''''",
    "1234567 12 345.67 0",
    "hi!!\n\nok",
    "foo !!! bar?\r\n",
    "(hello -world _mix2",
    "a  b   c",
    "a 1 b2",
    "é́ x ́",
    "́abc ́́",
    "café résumé",
    "日本語123テスト",
    "Привет мир",
    "\U0001f600\U0001f600 hi \U0001f44dok",
    "  \n \n  x\n",
    "\r\n\r\n",
    "   a b",
    "　你 好　",
    "don't stop believin'",
    ".\x1c. \x1c\x1d\n",
    "a\x00b\x7f",
    "3.14e10",
    "x+y=z; //comment\n",
    " ",
    "\n",
    "'",
    "  leading",
    "\t\t\t",
    "١٢٣٤",  # Arabic-Indic digits (Nd)
    "½¾7",  # vulgar fractions (No) + Nd
    "​​word",  # ZWSP is NOT \s for onig
    "a'ſb it'ſ",  # U+017F LATIN SMALL LETTER LONG S (case-fold probe)
    "word́ ́word",  # combining mark boundaries (qwen vs llama differ)
    "mixed  \nws",
    "a \x1c",
    "12345",
    "1234 x 123456789",
    "  12",
    "tab\tnum\t9",
    # Tekken's case-aware letter split: an uppercase run ends a piece when a
    # lowercase run follows. Qwen/Llama-3 keep each of these as ONE letter run,
    # so these rows discriminate the new pattern from both existing ones.
    "HelloWorld fooBarBaz",
    "McDonald iOS ABCdef",
    "ABC A1b2",
    # Lt (titlecase, U+01C5) and Lu-with-dot (U+0130) must both count as the
    # UPPER class; ſ is Ll despite looking uppercase-ish.
    "ǅabc İstanbul aſb",
    # Marks are in Tekken's letter classes but NOT excluded from its punct run
    # (unlike Qwen, which excludes them from both) -- the combination that
    # forces marks_in_run and marks_excluded apart.
    "é́X word́ ́word",
    # '/' is in Tekken's punct-run TRAILING class ([\r\n/]*, vs [\r\n]* for
    # Qwen/Llama-3). Only observable when '/' follows a newline that follows a
    # punct run -- a bare "a/b" absorbs '/' into the run itself and cannot see
    # the difference.
    "!!\n/b",
    "a.\n//y",
    "x!\n/",
    # GPT-4o-specific structure (each of these separates it from kLlama3):
    "abcDEF HTMLParser camelCaseWord",   # lower->upper splits, upper run does not
    "don't DON'T It's o'clock",          # contraction is a word SUFFIX here
    "path/to/file.txt a//b !/x",         # `/` in the punct-run tail
    "a!\n/b  x!\r\n/y  end.\n/usr/bin",   # `/` AFTER a newline: tail-only case
    "\u00e9\u00c9 \u00c9\u00e9 \u00c9\u00c9\u00e9 caf\u00e9",  # Ll/Lu boundaries
    "e\u0301X X\u0301e \u0301abc",         # combining marks are in BOTH classes
    "\u02b0ello A\u02b0B",                # Lm sits in both classes
    "\u01c5\u01c4\u01c6 \u01c4x",             # Lt (title case) is in A only
    "'s 'S x's X'S ' 's",                # apostrophe with no preceding word
    "\u4f60\u597dABC\u4f60\u597d",             # Lo neighbours a Lu run
    # \p{M} INSIDE the UPPER letter class (issue #369). Both kTekken and
    # kGpt4o put marks in BOTH letter classes, but nothing above could tell:
    # with a LOWERCASE base
    # ("e\u0301X", "word\u0301") the mark is absorbed by the lower class and the
    # upper class's \p{M} branch never decides anything. It takes an UPPERCASE
    # base, a combining mark, and then MORE upper/caseless letters -- the mark
    # has to keep the upper run alive across itself. Found by mutation: dropping
    # \p{M} from the upper class left every other row in this file byte-identical
    # and every gate green, while changing the split on 1957 of 36104 fuzz
    # inputs. These rows are what makes that mutant die.
    "A\u0301Aa A\u0301A\u0301 A\u0301\u0308B",
    "\u00c9CHO A\u0301\u0359\u01c5x",
    "HTM\u0139Parser",
]

ASCII_POOL = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \t\n\r'\".,!?;:()[]{}<>-_+=*/\\|@#$%^&~`"
)
UNI_POOL = (
    "éüß́̈你好日本テス"
    "Абв١٢½€ 　 ​"
    "\U0001f600\U0001f44dſẞ\x1c\x00\x7f\x0b\x0c\x85"
)


def random_strings(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    pool = ASCII_POOL + UNI_POOL
    out = []
    for _ in range(count):
        n = rng.randint(1, 30)
        if rng.random() < 0.5:
            out.append("".join(rng.choice(ASCII_POOL) for _ in range(n)))
        else:
            out.append("".join(rng.choice(pool) for _ in range(n)))
    return out


def cxx_bytes(s: str) -> str:
    """Escape every byte as a 3-digit octal escape (unambiguous, NUL-safe)."""
    b = s.encode("utf-8")
    return '"' + "".join("\\%03o" % x for x in b) + '"'


def pieces(pt: Split, s: str) -> list[str]:
    got = [p for p, _ in pt.pre_tokenize_str(s)]
    assert "".join(got) == s, (s, got)
    return got


def main() -> None:
    qwen = Split(Regex(QWEN), behavior="isolated", invert=False)
    llama = Split(Regex(LLAMA3), behavior="isolated", invert=False)
    tekken = Split(Regex(TEKKEN), behavior="isolated", invert=False)
    gpt4o = Split(Regex(GPT4O), behavior="isolated", invert=False)
    cases = DETERMINISTIC + random_strings(seed=42, count=60)

    w = sys.stdout.write
    w("// GENERATED FILE — do not edit by hand.\n")
    w("// Generator: tools/gen_pretok_goldens.py (HF tokenizers oracle; see\n")
    w("// the usage comment there for the exact command). Included by\n")
    w("// tests/vllm/test_pretokenizer.cpp.\n")
    import tokenizers

    w("// tokenizers version: %s\n" % tokenizers.__version__)
    w("// clang-format off\n")
    w("static const PretokGolden kPretokGoldens[] = {\n")
    for s in cases:
        q = pieces(qwen, s)
        l = pieces(llama, s)
        t = pieces(tekken, s)
        g = pieces(gpt4o, s)
        # rstrip: a trailing backslash in a // comment splices lines (GCC even
        # splices across "backslash then whitespace then newline").
        w("  {  // %s\n" % ascii(s)[1:-1][:90].rstrip("\\ "))
        w("    SV(%s),\n" % cxx_bytes(s))
        w("    {%s},\n" % ", ".join("SV(%s)" % cxx_bytes(p) for p in q))
        w("    {%s},\n" % ", ".join("SV(%s)" % cxx_bytes(p) for p in l))
        w("    {%s},\n" % ", ".join("SV(%s)" % cxx_bytes(p) for p in t))
        w("    {%s},\n" % ", ".join("SV(%s)" % cxx_bytes(p) for p in g))
        w("  },\n")
    w("};\n")
    w("// clang-format on\n")


if __name__ == "__main__":
    main()
