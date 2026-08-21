#!/usr/bin/env python3
"""The HF `tokenizers` arm of the BPE encode-cost comparison (#1365).

`tools/bench/bpe_encode_cost.cpp` times OUR `Tokenizer::Encode`. This times HF
`tokenizers` on the same input through the same `tokenizer.json` bytes, so the
two are comparable: identical case units, identical repeat-and-truncate rule,
min-of-k, and the 1/5/15-minute load average printed beside every row.

WHY IT IS COMMITTED. The same reason its C++ sibling is. A recipe nobody can
execute is not a recipe, and this row's history is three rounds of certified
constants that no reviewer could re-derive. The harness is committed and the
numbers are not.

WHAT ITS OUTPUT IS. A SESSION READING, never a bound. Do not copy a number out
of it without the load beside it, and do not treat one as reproducible on
another box, another load, or another day. Say what k was.

HF `tokenizers` is not a registered oracle in its own right; it is the code the
pinned `transformers` executes, and `.agents/oracles/transformers.md` is the pin
that selects the version. This program prints the version it actually imported.

RUN, from the repository root:

    python3 tools/bench/bpe_encode_cost_hf.py \\
        tests/parity/goldens/tokenizer_mistral/tokenizer.json \\
        --case english --sizes 1000,8000 --repeats 5
"""
import argparse
import pathlib
import sys
import time

# One shape per case, matching `kCases` in tools/bench/bpe_encode_cost.cpp.
CASES = {
    "english": "The quick brown fox jumps over the lazy dog. ",
    "a": "a",
    "space": " ",
    "newline": "\n",
    "tilde": "~",
    "cjk": "的",
}


def load_average() -> str:
    """The 1/5/15-minute load average. A figure with no load beside it is the
    defect this file exists to stop, so an unreadable /proc says so."""
    try:
        one, five, fifteen = pathlib.Path("/proc/loadavg").read_text().split()[:3]
    except (OSError, ValueError):
        return "UNKNOWN"
    return f"{one}/{five}/{fifteen}"


def repeat_to(unit: str, size: int) -> str:
    """Repeat and truncate to exactly `size` bytes on a codepoint boundary."""
    out = unit * (size // len(unit.encode()) + 2)
    encoded = out.encode()[:size]
    while encoded:
        try:
            return encoded.decode()
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokenizer_json", type=pathlib.Path)
    ap.add_argument("--case", action="append", dest="cases", choices=sorted(CASES))
    ap.add_argument("--sizes", default="1000,8000")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2

    import tokenizers
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer_json))
    print("# bpe_encode_cost_hf -- SESSION READING, NOT A BOUND")
    print(f"# tokenizers {tokenizers.__version__}")
    print(f"# tokenizer: {args.tokenizer_json}")
    print(f"# repeats k={args.repeats}, reporting MIN over k")
    print("case\tbytes\tids\tk\tmin_ms\tmax_ms\tload_before\tload_after")
    for name in args.cases or ["english"]:
        for size in (int(x) for x in args.sizes.split(",")):
            text = repeat_to(CASES[name], size)
            before = load_average()
            lo = hi = None
            ids: list[int] = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                ids = tok.encode(text, add_special_tokens=False).ids
                ms = (time.perf_counter() - start) * 1000.0
                lo = ms if lo is None else min(lo, ms)
                hi = ms if hi is None else max(hi, ms)
            print(f"{name}\t{len(text.encode())}\t{len(ids)}\t{args.repeats}\t"
                  f"{lo:.3f}\t{hi:.3f}\t{before}\t{load_average()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
