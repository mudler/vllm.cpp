#!/usr/bin/env python3
"""Build a variadic prompt corpus from pinned sources.

The corpus is a pure function of (source bytes, seed, weights, count). Run this
twice with the same arguments on two machines and you get the same file, which
is what lets a published length histogram be reproduced rather than believed.

Five bands, one composition rule each. `docs/benchmarks/variadic-load-methodology.md`
carries the reasoning; this file carries the rules.

  S    short question     GSM8K `question`, verbatim
  M    code completion    HumanEval `prompt`, verbatim
  L    prose summary      a contiguous block of sonnet lines, under an instruction
  XL   long code review   k HumanEval prompts concatenated, under an instruction
  XXL  as XL, longer      the same rule, sized at the served context

`XXL` carries no weight by default, so an invocation that does not name it
produces the same corpus bytes as the four-band predecessor did. Ask for it with
`--weights S=0.2,M=0.2,L=0.2,XL=0.2,XXL=0.2`.

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

# XXL is sized against the context the server actually serves, not against a
# round number, and it is sized to FIT rather than to hit 7000 exactly. A band
# that overruns the context voids itself under `G-FITS` and costs the lease it
# was measured on, so the margin is bought here and not argued for later.
#
# Where the ratio comes from. `docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/`
# measured the SAME 144 prompts twice: `corpus-manifest.json` holds their
# characters, and `corpus-token-histogram.md` holds their prompt tokens under
# the target checkpoint's own tokenizer. Over the `XL` band, which is the band
# XXL copies its composition rule from, that is 9011 characters against 2804
# tokens at the mean, 9572 against 2928 at p50, 7217 against 2237 at the
# minimum, and 10048 against 3233 at the maximum: 3.21, 3.27, 3.23 and 3.11
# characters per prompt token.
#
# Use the CODE band's ratio, not a blended one. The English bands render at a
# higher ratio on the same pairing -- `L` is 3361 characters against 878
# tokens, or 3.83 -- so a figure averaged over the whole corpus reads high and
# flatters a band that is entirely Python. XXL draws from HumanEval, so 3.21 is
# the applicable number and the 3.4 an earlier draft of this comment used was a
# blend.
#
# Size. 21000 / 3.21 = about 6540 prompt tokens, which is about 2.2 times the
# `XL` band's realised median of 2928 tokens and well past the 3.3k every
# published band stops at.
#
# Ceiling. The band is a sum of k whole problems with k drawn from
# [k_mid - 2, k_mid + 2], so it overshoots its target by the k jitter plus the
# sampling spread of the draws. XL realised 11044 characters at the top of the
# 192-prompt corpus, 22.7% over its 9000 target, at k_mid = 20: 10 points of
# that is the jitter, 2/20, and the remaining 12.7 points is the sampling
# spread. Both terms shrink with k. The jitter is 2/k_mid, and the spread of a
# sum of k draws grows as sqrt(k) while the sum grows as k, so it falls as
# 1/sqrt(k). At the k_mid = 47 that 21000 characters gives on the pinned
# HumanEval, that is 2/47 + 12.7% * sqrt(20/47) = 4.26% + 8.28% = 12.54% over
# target, which is the 1.125 the test below pins. (An earlier draft of this
# comment displayed the two terms rounded, as 4.3% + 8.3% = 12.5%; those
# rounded terms add to 12.6%, not to 12.5%. The constant was always computed
# from the unrounded terms and does not move.) The expected ceiling is
# therefore about 23600 characters. Convert that at
# 3.11, the ratio the TOP of the XL band realised, because the ceiling is the
# longest prompt and not the average one: about 7600 prompt tokens. At the mean
# ratio of 3.21 it is about 7360.
#
# Headroom. The served configuration is `--max-model-len 8192` with
# `max_tokens: 192`, and 57 tokens are allowed for the chat template, which
# leaves 8192 - 192 - 57 = 7943 tokens for the prompt body. The headroom is
# therefore about 346 tokens at the adverse 3.11 pairing and about 583 at the
# mean. Every published pairing leaves the band inside the budget.
#
# The template allowance is MEASURED, not assumed.
# `docs/benchmarks/qwen38-27b-exl3-variadic-gb10.md` ran this corpus through
# both engines, each rendering its own chat template, and read the realised
# counts back from every server's own `usage.prompt_tokens`. The corpus
# measures 26 to 3233 tokens with the target's own tokenizer, and the served
# histogram runs 78 to 3290: +52 at the corpus minimum and +57 at the `XL`
# top. 57 is the larger of the two measured deltas, so the pin is conservative
# at its own boundary; 50 was not, because it admits a target of 21977
# characters whose ceiling is 7950 served tokens.
#
# What that page does NOT settle is the second tokenizer. Each engine applies
# its own template and its own tokenizer, and the published run found the two
# rendered the SAME counts, but one agreeing run is not a guarantee for a band
# no published run has built.
#
# So this headroom is not a licence to trust the number. `G-FITS` in
# `.agents/specs/bench-qwen38-exl3-longctx.md` reads every realised
# `usage.prompt_tokens` back from BOTH servers and voids the band rather than
# publishing a truncation. `tests/scripts/test_variadic_harness.py`
# `test_the_xxl_target_fits_the_served_context` pins the constant below against
# that budget with no GPU, so a target that cannot fit is refused before a
# lease is spent on it.
XXL_TARGET_CHARS = 21000

BANDS = ("S", "M", "L", "XL", "XXL")
DEFAULT_WEIGHTS = "S=0.35,M=0.40,L=0.15,XL=0.10"

L_INSTRUCTION = (
    "Read the following passage and write a short prose summary of it. "
    "Say what it is about, in your own words.\n\n")
# XXL reuses XL_INSTRUCTION deliberately: the composition rule is XL's,
# unchanged, so that length is the only variable that differs between them.
XL_INSTRUCTION = (
    "Below are several Python function signatures with their docstrings. "
    "For each one, say in a single sentence what the function is supposed to "
    "do, and name the edge case its docstring leaves undefined.\n\n")


def parse_weights(text):
    """`S=0.35,M=0.40,...` -> {band: share}. Refuses, never rounds.

    Every defect here is silent if it is tolerated: an unknown band name would
    drop its mass, a share that does not sum to 1.0 would change the realised
    count, and both would publish a histogram that the manifest describes
    wrongly. So each one is an error that names the offending text.
    """
    weights = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"{pair!r} is not BAND=SHARE")
        band, _, share = pair.partition("=")
        band = band.strip()
        if band not in BANDS:
            raise argparse.ArgumentTypeError(
                f"unknown band {band!r}; known bands are "
                + ", ".join(BANDS))
        if band in weights:
            raise argparse.ArgumentTypeError(f"band {band!r} given twice")
        try:
            value = float(share)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"share {share!r} for band {band!r} is not a number") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise argparse.ArgumentTypeError(
                f"share {share!r} for band {band!r} is not finite")
        if value < 0.0:
            raise argparse.ArgumentTypeError(
                f"share {value} for band {band!r} is negative")
        weights[band] = value
    if not weights:
        raise argparse.ArgumentTypeError("no bands given")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise argparse.ArgumentTypeError(
            f"weights sum to {total}, not 1.0")
    return weights


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

    weights = args.weights
    counts = band_counts(args.count, weights)

    items = []

    # S: one real short question, verbatim.
    for q in rng.sample(gsm, counts.get("S", 0)):
        items.append(("S", q))

    # M: one real HumanEval prompt, verbatim. This is the predecessor's whole
    # workload, kept so the two runs share a band.
    for p in rng.sample(he, counts.get("M", 0)):
        items.append(("M", p))

    # L: a CONTIGUOUS block of lines, so two L prompts overlap only where the
    # blocks overlap. A shared prefix would be measuring a prefix cache.
    mean_line = sum(len(ln) for ln in sonnet) / len(sonnet)
    block_mid = max(1, min(len(sonnet), round(L_TARGET_CHARS / mean_line)))
    # Jitter the block length per prompt. A fixed block gives the whole band one
    # length, which is the opposite of what this corpus is for.
    lo_b = max(1, round(block_mid * 0.75))
    hi_b = min(len(sonnet), round(block_mid * 1.25))
    for _ in range(counts.get("L", 0)):
        block = rng.randint(lo_b, hi_b)
        start = rng.randrange(0, max(1, len(sonnet) - block + 1))
        body = "\n".join(sonnet[start:start + block])
        items.append(("L", L_INSTRUCTION + body))

    # XL: k problems concatenated. k is drawn per prompt so the band has an
    # internal spread rather than one length repeated.
    mean_he = sum(len(p) for p in he) / len(he)
    k_mid = max(2, round(XL_TARGET_CHARS / mean_he))
    for _ in range(counts.get("XL", 0)):
        k = rng.randint(max(2, k_mid - 2), k_mid + 2)
        body = "\n\n".join(rng.sample(he, min(k, len(he))))
        items.append(("XL", XL_INSTRUCTION + body))

    # XXL: the same rule at the served context. It draws from `rng` only when
    # it was asked for, which is what keeps the four-band default byte-exact.
    k_mid_xxl = max(2, round(XXL_TARGET_CHARS / mean_he))
    for _ in range(counts.get("XXL", 0)):
        k = rng.randint(max(2, k_mid_xxl - 2), k_mid_xxl + 2)
        body = "\n\n".join(rng.sample(he, min(k, len(he))))
        items.append(("XXL", XL_INSTRUCTION + body))

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
            "XXL_k_mid": k_mid_xxl,
            "XXL_target_chars": XXL_TARGET_CHARS,
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
    ap.add_argument(
        "--weights", type=parse_weights, default=DEFAULT_WEIGHTS,
        metavar="S=0.35,M=0.40,L=0.15,XL=0.10",
        help="band shares, comma separated, summing to 1.0. Bands: "
             + ", ".join(BANDS) + ". A band that is not named gets no items. "
             "The default is the four-band distribution the predecessor run "
             "published.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
