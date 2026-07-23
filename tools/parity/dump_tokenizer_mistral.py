#!/usr/bin/env python3
"""SentencePiece (Metaspace + byte-fallback) tokenizer parity oracle.

Dumps golden encodings for the Mistral-7B-v0.3 tokenizer — the first
SentencePiece-BPE tokenizer in the tree (a `Metaspace` pre_tokenizer replacing
spaces with U+2581, a byte-fallback vocab of `<0xNN>` tokens, and a
`TemplateProcessing` post-processor prepending BOS=1). The C++
`Tokenizer::FromHfJson` must reproduce these ids byte-for-byte.

Runs against the pinned vLLM 0.25.0 oracle's HF `tokenizers` on dgx.casa
(SACRED goldens) OR locally with any recent `tokenizers` (it matches, since
vLLM tokenizes through the identical HF backend):

    ssh dgx.casa 'cd ~/work/vllm.cpp && ~/venvs/vllm-oracle/bin/python \
      tools/parity/dump_tokenizer_mistral.py'

Copies the tokenizer.json into the golden dir (the C++ test loads THAT copy, so
oracle and test parse identical bytes) and writes encodings.json. Per corpus
entry it records:
  - ids           = encode(text, add_special_tokens=False)   (no BOS)
  - ids_special   = encode(text, add_special_tokens=True)     (BOS=1 prepended)
  - decode        = decode(ids, skip_special_tokens=False)    (round-trip text)

The corpus deliberately stresses the Metaspace edges (leading/trailing/repeated
spaces, a literal U+2581, the empty string), the byte-fallback triggers (emoji,
rare Unicode, control bytes, CJK/Hangul), and the added/special-token
interaction (BOS/EOS, [INST]/[/INST], a special token mid-string — where the
"first" prepend must NOT re-fire), plus the 16 paged-engine battery prompts so
the ENCODE goldens double-check the SACRED gate's prompt tokenization.
"""
import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO / "tests/parity/goldens/tokenizer_mistral"
DEFAULT_TOKENIZER_JSON = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots"
    / "caa1feb0e54d415e2df31207e5f4e273e33509b1/tokenizer.json"
)

# The 16 paged-engine battery prompts (MUST match
# tests/parity/test_mistral_paged_engine.cpp::Prompts()).
BATTERY = [
    "The capital of France is",
    "Once upon a time,",
    "In the beginning God created",
    "The quick brown fox jumps over",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The theory of relativity was developed by",
    "To be or not to be, that is",
    "The largest planet in our solar system is",
    "Machine learning is a subfield of",
    "The mitochondria is the powerhouse of",
    "Roses are red, violets are",
    "The first president of the United States was",
    "E equals m c",
    "A journey of a thousand miles begins with",
    "The chemical symbol for gold is",
]

# Metaspace / byte-fallback / special-token edge corpus.
EDGES = [
    "",                       # empty -> [] / [BOS]
    " ",                      # single space
    "  ",                     # two spaces
    "   ",                    # three spaces (▁▁▁)
    "hello",                  # no leading space -> prepend ▁
    " hello",                 # leading space already ▁, no double prepend
    "  hello",                # two leading spaces
    "Hello world",            # interior space
    "Hello, world!",          # punctuation
    "multiple   spaces",      # repeated interior spaces
    "end ",                   # trailing space
    "▁already",          # literal U+2581 in input, no double prepend
    "café",                   # in-vocab non-ASCII
    "\U0001F600",             # emoji (in vocab)
    "\U0001F600\U0001F600",   # repeated emoji
    "\U0001F1FA\U0001F1F8",   # regional-indicator pair (flag)
    "ℤ",                 # rare Unicode (ℤ) -> byte fallback <0xE2><0x84><0xA4>
    "\tTab",                  # tab -> byte fallback <0x09>
    "a\nb",                   # newline -> byte fallback <0x0A>
    "\x00",                   # NUL -> byte fallback <0x00>
    "\x1f",                   # US control (in vocab as a char)
    "日本語",     # 日本語 (CJK)
    "한국어test",  # 한국어test (Hangul + ASCII)
    "Hello[INST]world",       # special token mid-string; "world" NOT prepended
    "Hello [INST] world",     # spaces around a special token
    "[INST]hi[/INST]",        # leading special token; "hi" NOT prepended
    "a</s>b",                 # EOS mid-string
    "<s>hi",                  # leading BOS literal
    "The quick brown fox.",   # sanity
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-json", type=pathlib.Path,
                    default=DEFAULT_TOKENIZER_JSON)
    ap.add_argument("--golden-dir", type=pathlib.Path, default=GOLDEN_DIR)
    ap.add_argument("--label", default="mistralai/Mistral-7B-v0.3 @ caa1feb0")
    args = ap.parse_args()

    import tokenizers
    from tokenizers import Tokenizer

    golden = args.golden_dir
    golden.mkdir(parents=True, exist_ok=True)
    dst = golden / "tokenizer.json"
    if not dst.exists() or (dst.read_bytes() != args.tokenizer_json.read_bytes()):
        dst.write_bytes(args.tokenizer_json.read_bytes())
        print(f"copied tokenizer.json -> {dst}")

    tok = Tokenizer.from_file(str(dst))

    entries = []
    for text in list(EDGES) + list(BATTERY):
        e0 = tok.encode(text, add_special_tokens=False)
        e1 = tok.encode(text, add_special_tokens=True)
        decoded = tok.decode(e0.ids, skip_special_tokens=False)
        # Round-trip sanity: SentencePiece decode reproduces the input for
        # inputs with no leading whitespace (Metaspace's Strip removes exactly
        # one leading space, so " hello" decodes to "hello" by design).
        entries.append({
            "text": text,
            "ids": [int(x) for x in e0.ids],
            "ids_special": [int(x) for x in e1.ids],
            "decode": decoded,
        })

    doc = {
        "label": args.label,
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "bos_id": 1,
        "entries": entries,
    }
    out = golden / "encodings.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    print(f"wrote {out} ({len(entries)} entries); tokenizers {tokenizers.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
