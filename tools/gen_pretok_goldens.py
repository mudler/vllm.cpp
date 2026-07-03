#!/usr/bin/env python3
"""Generate tests/vllm/pretokenizer_goldens.inc from the HF tokenizers oracle.

Runs the exact Split regexes (see src/vllm/tokenizer/pretokenizer.cpp) through
huggingface/tokenizers (Rust onig engine) and emits the resulting pieces as a
C++ table. The C++ scanner in pretokenizer.cpp must reproduce these pieces
byte-for-byte.

Usage (needs the `tokenizers` package; the vllm-oracle venv on dgx has it):
  scp tools/gen_pretok_goldens.py dgx.casa:/tmp/ && \
  ssh dgx.casa '~/venvs/vllm-oracle/bin/python /tmp/gen_pretok_goldens.py' \
    > tests/vllm/pretokenizer_goldens.inc
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
        # rstrip: a trailing backslash in a // comment splices lines (GCC even
        # splices across "backslash then whitespace then newline").
        w("  {  // %s\n" % ascii(s)[1:-1][:90].rstrip("\\ "))
        w("    SV(%s),\n" % cxx_bytes(s))
        w("    {%s},\n" % ", ".join("SV(%s)" % cxx_bytes(p) for p in q))
        w("    {%s},\n" % ", ".join("SV(%s)" % cxx_bytes(p) for p in l))
        w("  },\n")
    w("};\n")
    w("// clang-format on\n")


if __name__ == "__main__":
    main()
