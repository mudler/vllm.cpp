#!/usr/bin/env python3
"""Tokenizer parity oracle for the GPT-4o / o200k family (issue #347).

Same job as dump_tokenizer.py -- run the REAL HF `tokenizers` library over
tests/parity/goldens/tokenizer_muse_glimmer/corpus.txt and commit the ids --
with one difference: it does NOT copy the checkpoint's tokenizer.json into the
golden dir verbatim. Muse Glimmer's is 28 MB, most of it a 439802-entry merge
list of which this corpus can reach a few thousand.

Instead it writes a file with the vocabulary intact and a CORPUS-CLOSED MERGE
LIST: the 439802 merges (7.6 MB) are cut to those that can possibly fire while
encoding this corpus, which is where nearly all the size is. That criterion is
not a guess:

  * a BPE merge (l, r) can only fire while encoding a pretoken w if the string
    l+r occurs in w, so keeping every merge whose joined form is a substring of
    some corpus pretoken keeps a superset of the merges that actually fire;
  * keeping a merge that never fires cannot change the result, because the full
    list contains it too, at the same relative rank.

The 200000-entry vocabulary is kept WHOLE and unmodified. That is deliberate:
HF `tokenizers` places added tokens by vocabulary size, so trimming the vocab
silently renumbers every `<|...|>` token (measured: `<|begin_of_text|>` moved
from 200000 to 2104), and the ids in encodings.json would stop being the
model's ids.

The reduction is checked, not trusted: this script loads the reduced file back
into HF `tokenizers` and asserts it reproduces the FULL tokenizer's ids for
every corpus entry, and it records the full file's sha256 in encodings.json so
the fixture is always traceable to the artifact it came from.

The reduced fixture is what CI checks. The full artifact is checked separately
and by hand, against the GGUF itself:

    build-cpu/examples/tokenize \
        "$CHECKPOINT_ROOT/muse-glimmer-30b-gguf/muse-glimmer-30B-kquant-17gb.gguf" \
        tests/parity/goldens/tokenizer_muse_glimmer/corpus.txt > /tmp/gguf_cpp.txt
    python3 tools/parity/verify_tokenizer_gguf.py \
        "$CHECKPOINT_ROOT/muse-glimmer-30b/tokenizer.json" \
        tests/parity/goldens/tokenizer_muse_glimmer/corpus.txt > /tmp/hf_py.txt
    diff /tmp/hf_py.txt /tmp/gguf_cpp.txt   # expect no output

Regenerate:
    CHECKPOINT_ROOT=... python3 tools/parity/dump_tokenizer_gpt4o.py
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys

from dump_tokenizer import read_corpus

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests/parity/goldens/tokenizer_muse_glimmer"

# The checkpoint root comes from `CHECKPOINT_ROOT` (`.env`), never from a
# literal here. The literal this replaced named `/mnt/nas_share`, which sits on
# the ephemeral root overlay of the gate box's immutable OS and disappeared at a
# reboot (issue #1073); `.agents/environment.md` records the live location and
# the reason. With the variable unset, `--tokenizer-json` is required, so the
# tool refuses by name instead of reading a path nobody declared.
_CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT") or ""
DEFAULT_TOKENIZER_JSON = (
    pathlib.Path(_CHECKPOINT_ROOT) / "muse-glimmer-30b" / "tokenizer.json"
    if _CHECKPOINT_ROOT
    else None
)
DEFAULT_LABEL = "meta/muse-glimmer-30b (HF snapshot on the NAS)"


def merge_pair(m) -> tuple[str, str]:
    """tokenizers writes merges either as ["l", "r"] or as "l r"."""
    if isinstance(m, list):
        return m[0], m[1]
    left, _, right = m.partition(" ")
    return left, right


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-json", type=pathlib.Path,
                    default=DEFAULT_TOKENIZER_JSON,
                    required=DEFAULT_TOKENIZER_JSON is None,
                    help="the checkpoint's tokenizer.json; defaults to "
                         "$CHECKPOINT_ROOT/muse-glimmer-30b/tokenizer.json")
    ap.add_argument("--golden-dir", type=pathlib.Path, default=GOLDEN_DIR)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    args = ap.parse_args()

    import tokenizers

    full_bytes = args.tokenizer_json.read_bytes()
    sha = hashlib.sha256(full_bytes).hexdigest()
    full_doc = json.loads(full_bytes)
    full_tok = tokenizers.Tokenizer.from_file(str(args.tokenizer_json))

    corpus = read_corpus(args.golden_dir / "corpus.txt")

    # Reference ids: the FULL tokenizer, exactly as the model ships.
    entries = []
    for text in corpus:
        ids = full_tok.encode(text, add_special_tokens=False).ids
        rt = full_tok.decode(ids, skip_special_tokens=False)
        if rt != text:
            print(f"DECODE ROUND-TRIP FAILED for {text!r}: got {rt!r}",
                  file=sys.stderr)
            return 1
        entries.append({"text": text, "ids": ids})

    # ---- corpus-closed subset -------------------------------------------
    # Pretokens in the byte-mapped alphabet, straight from the checkpoint's own
    # pre_tokenizer (Split(GPT-4o regex) then ByteLevel(use_regex=false)).
    pretokens = set()
    for text in corpus:
        for piece, _ in full_tok.pre_tokenizer.pre_tokenize_str(text):
            pretokens.add(piece)

    full_vocab = full_doc["model"]["vocab"]
    keep_merges = [
        m for m in full_doc["model"]["merges"]
        if any("".join(merge_pair(m)) in piece for piece in pretokens)
    ]

    reduced = dict(full_doc)
    reduced["model"] = dict(full_doc["model"])
    reduced["model"]["merges"] = keep_merges  # vocab stays whole -- see above
    # NOTE no provenance key is added to the fixture itself: HF `tokenizers`
    # rejects unknown top-level fields in tokenizer.json. The provenance lives
    # in encodings.json ("oracle"), which is written next to it.

    golden = args.golden_dir
    golden.mkdir(parents=True, exist_ok=True)
    reduced_path = golden / "tokenizer.json"
    # Compact: this file is a fixture, not something to read by eye.
    reduced_path.write_text(
        json.dumps(reduced, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8")

    # ---- the subset argument, verified rather than trusted ---------------
    subset_tok = tokenizers.Tokenizer.from_file(str(reduced_path))
    bad = 0
    for entry in entries:
        got = subset_tok.encode(entry["text"], add_special_tokens=False).ids
        if got != entry["ids"]:
            bad += 1
            print(f"SUBSET MISMATCH for {entry['text']!r}:\n"
                  f"  full   {entry['ids']}\n  subset {got}", file=sys.stderr)
        rt = subset_tok.decode(entry["ids"], skip_special_tokens=False)
        if rt != entry["text"]:
            bad += 1
            print(f"SUBSET DECODE MISMATCH for {entry['text']!r}: {rt!r}",
                  file=sys.stderr)
    if bad:
        print(f"{bad} subset mismatches -- fixture NOT written", file=sys.stderr)
        reduced_path.unlink()
        return 1

    doc = {
        "oracle": {
            "tokenizers": tokenizers.__version__,
            "tokenizer_json": args.label,
            "tokenizer_json_sha256": sha,
            "tokenizer_json_path": str(args.tokenizer_json),
            "pre_tokenizer": "GPT-4o / o200k (GGUF tokenizer.ggml.pre=llama4)",
            "add_special_tokens": False,
            "fixture": (
                "tokenizer.json in this directory is a CORPUS-CLOSED SUBSET of "
                "the file above (see tools/parity/dump_tokenizer_gpt4o.py); it "
                "reproduces these ids exactly under HF tokenizers."
            ),
            "regenerate": "python3 tools/parity/dump_tokenizer_gpt4o.py",
        },
        "entries": entries,
    }
    out = golden / "encodings.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} ({len(entries)} entries, decode round-trip OK)")
    print(f"wrote {reduced_path} "
          f"({len(full_vocab)} vocab entries kept whole, "
          f"{len(keep_merges)} merges of "
          f"{len(full_doc['model']['merges'])}, "
          f"{reduced_path.stat().st_size / 1e6:.1f} MB, subset check OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
