#!/usr/bin/env python3
"""HF-tokenizers side of the GGUF tokenizer e2e check (dgx-only, not CI).

Prints one line per corpus entry: space-separated token ids, or "EMPTY" for
an entry that encodes to zero ids — the exact output format of
examples/tokenize/main.cpp. Diff the two on dgx.casa:

    cd ~/work/vllm.cpp
    build/examples/tokenize \
        ~/work/apex/qwen36_35b/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
        tests/parity/goldens/tokenizer_qwen36/corpus.txt > /tmp/gguf_cpp.txt
    ~/venvs/vllm-oracle/bin/python tools/parity/verify_tokenizer_gguf.py \
        tests/parity/goldens/tokenizer_qwen36/tokenizer.json \
        tests/parity/goldens/tokenizer_qwen36/corpus.txt > /tmp/hf_py.txt
    diff /tmp/hf_py.txt /tmp/gguf_cpp.txt   # expect no output

(The APEX GGUF ships the same Qwen3.6 vocab as the HF snapshot; Task 4
verified the GGUF vocab arrays faithful.)
"""
import sys
import pathlib

from dump_tokenizer import read_corpus


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <tokenizer.json> <corpus.txt>",
              file=sys.stderr)
        return 2
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(sys.argv[1])
    for text in read_corpus(pathlib.Path(sys.argv[2])):
        ids = tok.encode(text, add_special_tokens=False).ids
        print(" ".join(map(str, ids)) if ids else "EMPTY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
