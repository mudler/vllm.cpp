#!/usr/bin/env python3
"""Tokenizer parity oracle: dumps golden encodings for the parity corpus.

Runs on dgx.casa in the oracle venv (HF `tokenizers` is a vllm dependency):

    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
      tools/parity/dump_tokenizer.py'

Loads the unsloth Qwen3.6-27B snapshot's tokenizer.json, copies it into the
golden dir (the C++ test loads THAT copy, so oracle and test read identical
bytes), encodes every corpus entry with add_special_tokens=False, checks the
decode round-trip, and writes encodings.json.

The corpus line format (comments, escapes, empty line) is documented at the
top of corpus.txt; read_corpus() here is the reference implementation, also
imported by verify_tokenizer_gguf.py and mirrored in
examples/tokenize/main.cpp.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests/parity/goldens/tokenizer_qwen36"
DEFAULT_TOKENIZER_JSON = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots"
    / "890bdef7a42feba6d83b6e17a03315c694112f2a/tokenizer.json"
)

_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}


def unescape(line: str) -> str:
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            if i + 1 >= len(line) or line[i + 1] not in _ESCAPES:
                raise ValueError(f"bad escape at col {i} in corpus line: {line!r}")
            out.append(_ESCAPES[line[i + 1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def read_corpus(path: pathlib.Path) -> list[str]:
    """Corpus entries: '#' lines skipped, empty line = empty string,
    \\n/\\r/\\t/\\\\ escapes decoded. The file's trailing newline does not
    create an extra entry."""
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise ValueError(f"{path} must end with a newline")
    lines = raw.split("\n")[:-1]
    return [unescape(l) for l in lines if not l.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-json", type=pathlib.Path,
                    default=DEFAULT_TOKENIZER_JSON)
    ap.add_argument("--golden-dir", type=pathlib.Path, default=GOLDEN_DIR)
    # Provenance string recorded in encodings.json. Defaulted to the original
    # Qwen3.6 oracle so the existing regeneration command is unchanged; the
    # DeepSeek-V2 corpus (MLA campaign W8) passes its own.
    ap.add_argument("--label", default="unsloth/Qwen3.6-27B-NVFP4 @ 890bdef7")
    args = ap.parse_args()

    import tokenizers

    golden = args.golden_dir
    golden.mkdir(parents=True, exist_ok=True)
    dst = golden / "tokenizer.json"
    if not dst.exists() or (dst.read_bytes() != args.tokenizer_json.read_bytes()):
        shutil.copyfile(args.tokenizer_json, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()

    tok = tokenizers.Tokenizer.from_file(str(dst))
    entries = []
    for text in read_corpus(golden / "corpus.txt"):
        ids = tok.encode(text, add_special_tokens=False).ids
        rt = tok.decode(ids, skip_special_tokens=False)
        if rt != text:
            print(f"DECODE ROUND-TRIP FAILED for {text!r}: got {rt!r}",
                  file=sys.stderr)
            return 1
        entries.append({"text": text, "ids": ids})

    doc = {
        "oracle": {
            "tokenizers": tokenizers.__version__,
            "tokenizer_json": args.label,
            "tokenizer_json_sha256": sha,
            "add_special_tokens": False,
            "regenerate": "~/venvs/vllm-oracle/bin/python tools/parity/dump_tokenizer.py",
        },
        "entries": entries,
    }
    out = golden / "encodings.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} ({len(entries)} entries, decode round-trip OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
