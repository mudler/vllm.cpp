#!/usr/bin/env python3
"""Build a variadic prompt corpus from pinned sources.

The corpus is a pure function of (source bytes, seed, weights, count). Run this
twice with the same arguments on two machines and you get the same file, which
is what lets a published length histogram be reproduced rather than believed.

Four bands, one composition rule each. `docs/benchmarks/variadic-load-methodology.md`
carries the reasoning; this file carries the rules.

  S   short question      GSM8K `question`, verbatim
  M   code completion     HumanEval `prompt`, verbatim
  L   prose summary       a contiguous block of sonnet lines, under an instruction
  XL  long code review    k HumanEval prompts concatenated, under an instruction

Output is the ShareGPT shape the head-to-head client already reads, with a
`band` key added. A manifest beside it records every source sha256, the seed,
the weights, and the realised CHARACTER lengths. Token lengths are NOT recorded
here, because this script has no tokenizer and the published histogram is read
back from each server's own `usage.prompt_tokens`.
"""
import argparse
import hashlib
import json
import random
import sys

# Targets are in characters, because this script has no tokenizer. The
# methodology document states the token bands these produced when measured.
L_TARGET_CHARS = 3200      # about 800 prompt tokens of English verse
XL_TARGET_CHARS = 9000     # about 3000 prompt tokens of Python

L_INSTRUCTION = (
    "Read the following passage and write a short prose summary of it. "
    "Say what it is about, in your own words.\n\n")
XL_INSTRUCTION = (
    "Below are several Python function signatures with their docstrings. "
    "For each one, say in a single sentence what the function is supposed to "
    "do, and name the edge case its docstring leaves undefined.\n\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path, field):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)[field])
    return out


def band_counts(count, weights):
    """Deterministic split, largest-remainder, so the counts sum to `count`."""
    raw = {k: count * w for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    short = count - sum(base.values())
    order = sorted(weights, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:short]:
        base[k] += 1
    return base


def build(args):
    rng = random.Random(args.seed)
    gsm = load_jsonl(args.gsm8k, "question")
    he = load_jsonl(args.humaneval, "prompt")
    with open(args.sonnet, encoding="utf-8") as f:
        sonnet = [ln.rstrip("\n") for ln in f if ln.strip()]

    weights = {"S": args.weight_s, "M": args.weight_m,
               "L": args.weight_l, "XL": args.weight_xl}
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-9:
        print(f"ERROR: weights sum to {total_w}, not 1.0", file=sys.stderr)
        return 2
    counts = band_counts(args.count, weights)

    items = []

    # S: one real short question, verbatim.
    for q in rng.sample(gsm, counts["S"]):
        items.append(("S", q))

    # M: one real HumanEval prompt, verbatim. This is the predecessor's whole
    # workload, kept so the two runs share a band.
    for p in rng.sample(he, counts["M"]):
        items.append(("M", p))

    # L: a CONTIGUOUS block of lines, so two L prompts overlap only where the
    # blocks overlap. A shared prefix would be measuring a prefix cache.
    mean_line = sum(len(ln) for ln in sonnet) / len(sonnet)
    block_mid = max(1, min(len(sonnet), round(L_TARGET_CHARS / mean_line)))
    # Jitter the block length per prompt. A fixed block gives the whole band one
    # length, which is the opposite of what this corpus is for.
    lo_b = max(1, round(block_mid * 0.75))
    hi_b = min(len(sonnet), round(block_mid * 1.25))
    for _ in range(counts["L"]):
        block = rng.randint(lo_b, hi_b)
        start = rng.randrange(0, max(1, len(sonnet) - block + 1))
        body = "\n".join(sonnet[start:start + block])
        items.append(("L", L_INSTRUCTION + body))

    # XL: k problems concatenated. k is drawn per prompt so the band has an
    # internal spread rather than one length repeated.
    mean_he = sum(len(p) for p in he) / len(he)
    k_mid = max(2, round(XL_TARGET_CHARS / mean_he))
    for _ in range(counts["XL"]):
        k = rng.randint(max(2, k_mid - 2), k_mid + 2)
        body = "\n\n".join(rng.sample(he, min(k, len(he))))
        items.append(("XL", XL_INSTRUCTION + body))

    # One shuffle, so the band order is fixed and identical for every arm and
    # every rung. Two arms that see different orders are not one workload.
    rng.shuffle(items)

    out = [{"id": f"{band}-{i:04d}", "band": band,
            "conversations": [{"from": "human", "value": text},
                              {"from": "gpt", "value": ""}]}
           for i, (band, text) in enumerate(items)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    per_band = {}
    for band, text in items:
        per_band.setdefault(band, []).append(len(text))
    manifest = {
        "generator": "benchmarks/variadic/build_corpus.py",
        "seed": args.seed,
        "count": args.count,
        "weights": weights,
        "band_counts": counts,
        "sources": {
            "gsm8k": {"path": args.gsm8k, "sha256": sha256_of(args.gsm8k),
                      "rows": len(gsm)},
            "humaneval": {"path": args.humaneval,
                          "sha256": sha256_of(args.humaneval), "rows": len(he)},
            "sonnet": {"path": args.sonnet, "sha256": sha256_of(args.sonnet),
                       "lines": len(sonnet)},
        },
        "rules": {
            "L_block_lines_mid": block_mid,
            "L_block_lines_range": [lo_b, hi_b],
            "L_target_chars": L_TARGET_CHARS,
            "XL_k_mid": k_mid,
            "XL_target_chars": XL_TARGET_CHARS,
        },
        "realised_chars": {
            b: {"n": len(v), "min": min(v), "median": sorted(v)[len(v) // 2],
                "max": max(v), "mean": sum(v) / len(v)}
            for b, v in sorted(per_band.items())
        },
        "corpus_sha256": sha256_of(args.out),
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsm8k", required=True, help="GSM8K test.jsonl")
    ap.add_argument("--humaneval", required=True, help="HumanEval.jsonl")
    ap.add_argument("--sonnet", required=True, help="vLLM benchmarks/sonnet.txt")
    ap.add_argument("--count", type=int, default=136)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-s", type=float, default=0.35)
    ap.add_argument("--weight-m", type=float, default=0.40)
    ap.add_argument("--weight-l", type=float, default=0.15)
    ap.add_argument("--weight-xl", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
