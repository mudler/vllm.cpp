#!/usr/bin/env python3
"""SPEC-BPE-QUADRATIC-MERGE (#1365) — the BPE equivalence oracle.

`.agents/specs/bpe-quadratic-merge.md` §Tests to port item 1 asks for a corpus
that reaches the LONG-PRETOKEN regime, which every gate in the tree misses: the
committed 64-entry Qwen3.6 corpus has a longest pretoken of 54 bytes and the
defect starts two orders of magnitude above it.

**The recorded identifiers come from HF `tokenizers`, not from our own output.**
That is the whole point of the file. A baseline captured from the code under
test is a change detector: it proves the answer did not move, never that the
answer is right. Recording our own ids would make "it cannot go green on a
faster wrong answer" false, and this row replaces a merge algorithm.

Both committed goldens are encoded, because the two arms are different code:

  - `tests/parity/goldens/tokenizer_qwen36/tokenizer.json` — byte-level, split
    by regex, reached through `Tokenizer::EncodePlain`.
  - `tests/parity/goldens/tokenizer_mistral/tokenizer.json` — SentencePiece,
    `"split": false` in its own `pre_tokenizer`, so the WHOLE prompt is one
    word, reached through `Tokenizer::EncodePlainSp`. This is the arm that pays
    the quadratic cost on ordinary English, and the arm the prototype evidence
    of the spec did not cover.

Per entry it records, for each golden:

  ids          = encode(text, add_special_tokens=False).ids
  ids_special  = encode(text, add_special_tokens=True).ids

which pair with `Tokenizer::Encode` and `Tokenizer::EncodeWithSpecialTokens`.
`EncodeWithSpecialTokens` is the function the request path actually calls
(`src/vllm/v1/engine/input_processor.cpp::process_inputs`), so both are checked.

Run, from the repository root, with `tokenizers` 0.22.2 on the path (the version
the pinned `transformers` resolves — see `.agents/oracles/transformers.md`):

    python3 tools/parity/dump_bpe_equivalence.py

It rewrites `tests/parity/goldens/bpe_equivalence/encodings.json` in place and
records the `tokenizers` version and both tokenizer.json sha256 values beside
the ids, so a later reader can tell whether the capture still describes the
files the C++ test loads.
"""
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDENS = REPO / "tests/parity/goldens"
OUT_DIR = GOLDENS / "bpe_equivalence"

ARMS = {
    "qwen36": GOLDENS / "tokenizer_qwen36/tokenizer.json",
    "mistral": GOLDENS / "tokenizer_mistral/tokenizer.json",
}

# ---------------------------------------------------------------------------
# Group A — the existing 64-entry Qwen3.6 parity corpus, verbatim.
#
# It is carried so this binary is a strict superset of the corpus that was in
# the tree when the defect shipped, and so a regression on ordinary text is
# caught here as well as in test_tokenizer_parity.
# ---------------------------------------------------------------------------
def group_a():
    doc = json.loads((GOLDENS / "tokenizer_qwen36/encodings.json").read_text())
    return [(f"existing/{i:02d}", e["text"]) for i, e in enumerate(doc["entries"])]


# ---------------------------------------------------------------------------
# Group B — real multilingual prose. The spec asks for Chinese, Japanese, Thai,
# Lao and Khmer specifically: the pretokenizer's rule-2 letter run does not stop
# at a word boundary in a script that does not write one, so ordinary prose in
# these scripts is already a long single pretoken.
# ---------------------------------------------------------------------------
GROUP_B = {
    "prose/zh": "人工智能的发展正在改变我们的生活方式，从医疗诊断到交通运输，从教育学习到科学研究，各个领域都在经历深刻的变革。" * 6,
    "prose/ja": "自然言語処理の研究はここ数年で大きく進展し、機械翻訳や文章要約などの応用が実用化されています。" * 6,
    "prose/th": "การพัฒนาปัญญาประดิษฐ์กำลังเปลี่ยนแปลงโลกของเราอย่างรวดเร็ว" * 6,
    "prose/lo": "ການພັດທະນາເຕັກໂນໂລຊີກຳລັງປ່ຽນແປງໂລກ" * 8,
    "prose/km": "បច្ចេកវិទ្យាព័ត៌មានកំពុងផ្លាស់ប្ដូរពិភពលោក" * 8,
}

# ---------------------------------------------------------------------------
# Group C — the failing regime. A run of ONE character class is one pretoken
# under five of the seven pretokenizer rules (spec §"Five rules are unbounded"),
# and the WHOLE prompt on the SentencePiece arm.
#
# 2,048 bytes is what the suite can afford at W1, where these run against the
# UNCHANGED O(n^2) code. The spec's own tables are at 8,192 and 65,535 bytes,
# which cost 0.5 s and 25-45 s per input per arm on that code — a red capture,
# not a corpus a gate can carry.
# ---------------------------------------------------------------------------
GROUP_C = {
    "run/a": "a" * 2048,             # rule 2, letter run
    "run/space": " " * 2048,         # rule 7, whitespace
    "run/newline": "\n" * 2048,      # rule 5, \s*[\r\n]+ — the rule #1365 missed
    "run/tilde": "~" * 2048,         # rule 4, punctuation run
    "run/cjk": "的" * 682,       # rule 2, non-ASCII letter run
    "run/thai": "ก" * 682,      # rule 2, Thai letter run
    "run/tab": "\t" * 2048,          # rule 7 via a non-space whitespace
}

# ---------------------------------------------------------------------------
# Group D — the SentencePiece arm on ordinary English, which is the input a
# user actually sends and the case the spec measures at 2,507x against HF.
# ---------------------------------------------------------------------------
PANGRAM = "The quick brown fox jumps over the lazy dog. "


def repeat_to(unit: str, size: int) -> str:
    out = (unit * (size // len(unit) + 1))[:size]
    return out


GROUP_D = {
    "english/1000": repeat_to(PANGRAM, 1000),
    "english/2000": repeat_to(PANGRAM, 2000),
    "english/8000": repeat_to(PANGRAM, 8000),
}

# ---------------------------------------------------------------------------
# Group E — the SHAPE of #1365's prompt 2, which is the request the row was
# opened for: 8,034 bytes that pretokenize to ONE pretoken.
#
# THE SIX LITERAL #1365 PROMPTS ARE NOT REPRODUCIBLE FROM THIS TREE. They were
# read out of `out/bench-20260819T035148Z/`, a `vllm bench serve --dataset-name
# random` run on a leased GB10 against a checkpoint that is not committed; the
# raw files are not in the repository and the generator samples token ids from
# that checkpoint's vocabulary. So this entry reproduces the PROPERTY that made
# prompt 2 the outlier — 8,034 bytes, one pretoken, no whitespace or class
# boundary anywhere in it — rather than its bytes, and says so here rather than
# claiming a provenance it does not have.
#
# The letters are drawn by a fixed 32-bit LCG so the entry is deterministic;
# the text is stored literally in the golden, so the C++ side never re-derives
# it and the two sides cannot disagree about the generator.
# ---------------------------------------------------------------------------
def lcg_letters(n: int, seed: int = 1365) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    state = seed
    out = []
    for _ in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        out.append(alphabet[(state >> 16) % len(alphabet)])
    return "".join(out)


# ---------------------------------------------------------------------------
# The cost inputs of §Tests to port item 2, recorded separately because the
# corpus cannot carry them.
#
# Each is 65,536 bytes that pretokenize to ONE word, on ONE of the two callers.
# On the UNCHANGED code these cost tens of seconds of one core each, which is
# the defect; on the fixed code they cost tens of milliseconds. The ids are
# recorded so the cost assertion cannot go green on a faster WRONG answer: the
# test compares the full id vector before it looks at the clock.
#
# The text is a repeat rule rather than 128 KB of literal JSON. The id
# comparison is what verifies that the C++ side built the same bytes — a
# different string produces different ids and reds the case.
# ---------------------------------------------------------------------------
COST = [
    {"name": "mistral/english/65536", "arm": "mistral",
     "repeat_unit": PANGRAM, "bytes": 65536},
    {"name": "qwen36/a/65536", "arm": "qwen36",
     "repeat_unit": "a", "bytes": 65536},
]


def corpus():
    entries = list(group_a())
    entries += sorted(GROUP_B.items())
    entries += sorted(GROUP_C.items())
    entries += sorted(GROUP_D.items())
    entries.append(("issue1365/prompt2-shape", lcg_letters(8034)))
    return entries


def main() -> int:
    import tokenizers
    from tokenizers import Tokenizer

    if tokenizers.__version__ != "0.22.2":
        print(f"WARNING: tokenizers {tokenizers.__version__}, spec pins 0.22.2",
              file=sys.stderr)

    loaded = {}
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2
        loaded[arm] = (Tokenizer.from_file(str(path)),
                       hashlib.sha256(path.read_bytes()).hexdigest())

    entries = []
    for name, text in corpus():
        rec = {"name": name, "text": text}
        for arm, (tok, _) in loaded.items():
            rec[f"{arm}_ids"] = [int(x) for x in
                                 tok.encode(text, add_special_tokens=False).ids]
            rec[f"{arm}_ids_special"] = [
                int(x) for x in tok.encode(text, add_special_tokens=True).ids]
        entries.append(rec)

    cost = []
    for c in COST:
        text = repeat_to(c["repeat_unit"], c["bytes"])
        tok = loaded[c["arm"]][0]
        rec = dict(c)
        rec["ids"] = [int(x) for x in
                      tok.encode(text, add_special_tokens=False).ids]
        cost.append(rec)

    doc = {
        "oracle": {
            "tokenizers": tokenizers.__version__,
            "generator": "tools/parity/dump_bpe_equivalence.py",
            "spec": ".agents/specs/bpe-quadratic-merge.md",
            "issue": 1365,
            "arms": {arm: {"path": str(path.relative_to(REPO)),
                           "tokenizer_json_sha256": loaded[arm][1]}
                     for arm, path in ARMS.items()},
        },
        "entries": entries,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "encodings.json"
    # One entry per line. `indent=1` over id arrays this long produces a file
    # nobody can diff; one line per entry keeps a changed entry to one changed
    # line, which is what a reviewer of a re-capture actually reads.
    body = ",\n  ".join(json.dumps(e, ensure_ascii=False) for e in entries)
    out.write_text(
        '{\n "oracle": '
        + json.dumps(doc["oracle"], ensure_ascii=False, indent=2)
        + ',\n "cost": [\n  '
        + ",\n  ".join(json.dumps(c, ensure_ascii=False) for c in cost)
        + ' \n ],\n "entries": [\n  '
        + body
        + "\n ]\n}\n",
        encoding="utf-8")
    total = sum(len(e["text"].encode()) for e in entries)
    print(f"wrote {out}: {len(entries)} entries, {total} corpus bytes, "
          f"tokenizers {tokenizers.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
