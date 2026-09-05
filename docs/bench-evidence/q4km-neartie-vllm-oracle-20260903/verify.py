#!/usr/bin/env python3
"""#2809 -- recount the four §12.2 conjuncts from the harness's per-step record.

The harness aggregated while it held the engine open. This reads only the
recorded per-step numbers afterwards, does the counting again, and refuses to
agree with the harness's own `summary` until every figure matches. It shares no
code with the harness. It does share the per-step data, which IS the
measurement and is the thing being reported.

Three things it checks that a recount alone would not:

1. **Stream identity.** Each scored stream is rebuilt from the per-step
   `our_tok` fields and compared to the committed artifact it is supposed to
   be. A harness that scored some other stream would recount consistently and
   still be measuring the wrong thing.
2. **The negative control discriminates.** A corrupted step must FAIL loudly.
   An instrument that cannot fail cannot pass.
3. **The published llama.cpp table reproduces.** The free-running divergence of
   our arm against llama.cpp `b10451` is recomputed from the same two committed
   streams and checked against
   `qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`. That table is NOT
   rescored by this row; it is re-derived, so that a scorer which agrees with
   everything is caught.

Usage, from the repository root:

    python3 docs/bench-evidence/q4km-neartie-vllm-oracle-20260903/verify.py
"""
import gzip
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STREAMS = os.path.join(ROOT, "docs/bench-evidence/oracle-vllm-gfx1151-20260903")

BAND_NATS = 0.5
ARGMAX_EPS = 1e-9

PROMPTS = [
    "The capital city of France is",
    "The three primary colors are",
    "Water boils at a temperature of",
    "The Pythagorean theorem states that",
    "In 1969, humans first walked on",
    "A prime number is a natural number",
]
PROMPTS_SHA256 = "c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e"
OURS_SHA256 = "8b542c718fd38721d5dd3286a77c91ed30ab495b0c604783b9a2681fcc1ad107"

# The published figures this document reports, restated here so that the script
# fails when the document and the data disagree, in either direction.
PUBLISHED = {
    "compiled": {"ours": (0, 0.0, 0, "PASS"),
                 "self": (3, 0.125, 3, "FAIL"),
                 "llamacpp": (1, 0.125, 1, "FAIL")},
    "eager": {"ours": (4, 0.25, 4, "FAIL"),
              "self": (6, 0.125, 6, "FAIL"),
              "llamacpp": (3, 0.125, 3, "FAIL")},
}
# The v2 token gate's per-prompt table: prompt -> (first diff step, ours, oracle).
V2_TABLE = {1: (45, 303, 1521), 3: (45, 25, 393), 5: (32, 16, 15)}
V2_DIVERGENT_PROMPTS = [1, 3, 5]

CLAIMS = 0
MISMATCHES = []


def check(label, got, want):
    global CLAIMS
    CLAIMS += 1
    if got != want:
        MISMATCHES.append(label)
        print(f"  MISMATCH  {label}: got {got!r} want {want!r}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_maybe_gz(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path) as fh:
        return json.load(fh)


def read_llamacpp(path):
    gen = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("GEN_IDS "):
                f = line.split()
                gen[int(f[1])] = [int(x) for x in f[2:]]
    return [gen[i] for i in sorted(gen)]


def read_records(path):
    with open(path) as fh:
        return [r["gen_ids"] for r in json.load(fh)["records"]]


def stream_from(per_step):
    return [[st["our_tok"] for st in pr["steps"]] for pr in per_step]


def recount(per_step):
    """The four conjuncts, recomputed from top_lp and our_lp alone."""
    n_div = n_tok_mismatch = n_exact_tie = 0
    worst, worst_at, over_band = 0.0, None, []
    for pr in per_step:
        for st in pr["steps"]:
            gap = st["top_lp"] - st["our_lp"]
            assert abs(gap - st["gap"]) < 1e-12, ("recorded gap is not top_lp - our_lp", st)
            argmax = gap <= ARGMAX_EPS
            assert argmax == st["argmax"], ("argmax flag inconsistent", st)
            tok_match = int(st["top_tok"]) == int(st["our_tok"])
            assert tok_match == st["tok_match"], ("tok_match flag inconsistent", st)
            if not tok_match:
                n_tok_mismatch += 1
                if argmax:
                    n_exact_tie += 1
            if not argmax:
                n_div += 1
                if gap > worst:
                    worst, worst_at = gap, (pr["prompt"], st["step"])
                if gap > BAND_NATS:
                    over_band.append([pr["prompt"], st["step"], st["our_tok"], gap])
    result = "PASS" if not over_band else "FAIL"
    four = (result == "PASS" and n_div == 0 and not over_band and worst <= BAND_NATS)
    return {"n_divergent": n_div, "n_token_mismatch": n_tok_mismatch,
            "n_exact_tie": n_exact_tie, "worst": worst, "worst_at": worst_at,
            "over_band": len(over_band), "result": result,
            "four": "PASS" if four else "FAIL",
            "total": sum(len(p["steps"]) for p in per_step)}


def main():
    print("== inputs, and the identity of each ==")
    ours_path = os.path.join(STREAMS, "ours_gen_ids_1.json")
    check("ours_gen_ids_1.json sha256", sha256_file(ours_path), OURS_SHA256)
    body = "".join(p + "\n" for p in PROMPTS).encode()
    check("prompts_sha256 recomputed from the six prompt strings",
          hashlib.sha256(body).hexdigest(), PROMPTS_SHA256)

    ours = json.load(open(ours_path))
    lcpp = read_llamacpp(os.path.join(STREAMS, "oracle_hip.txt"))

    for cfg in ("eager", "compiled"):
        path = os.path.join(HERE, f"neartie-{cfg}.json.gz")
        d = load_json_maybe_gz(path)
        print(f"\n== oracle vLLM {cfg} -- {os.path.basename(path)} ==")
        check(f"{cfg}: config field", d["config"], cfg)
        check(f"{cfg}: enforce_eager", d["enforce_eager"], cfg == "eager")
        check(f"{cfg}: vllm_version", d["vllm_version"], "0.26.0.dev0+g5559679229")
        check(f"{cfg}: prompts_sha256", d["prompts_sha256"], PROMPTS_SHA256)
        check(f"{cfg}: band_nats", d["band_nats"], BAND_NATS)
        check(f"{cfg}: argmax_eps", d["argmax_eps"], ARGMAX_EPS)
        check(f"{cfg}: topk", d["topk"], 20)

        # 1. the harness scored the streams the spec pinned, and no others
        check(f"{cfg}: scored 'ours' == committed ours_gen_ids_1.json",
              stream_from(d["per_step"]["ours"]), ours)
        check(f"{cfg}: scored 'llamacpp' == committed oracle_hip.txt GEN_IDS",
              stream_from(d["per_step"]["llamacpp"]), lcpp)
        check(f"{cfg}: scored 'self' == committed tokens-gguf-{cfg}.json",
              stream_from(d["per_step"]["self"]),
              read_records(os.path.join(STREAMS, f"tokens-gguf-{cfg}.json")))

        for name in ("ours", "self", "llamacpp", "control_corrupt"):
            r = recount(d["per_step"][name])
            s = d["summary"][name]
            check(f"{cfg}/{name}: total_steps", r["total"], s["total_steps"])
            check(f"{cfg}/{name}: n_divergent", r["n_divergent"], s["n_divergent"])
            check(f"{cfg}/{name}: worst_gap_nats", round(r["worst"], 12),
                  round(s["worst_gap_nats"], 12))
            check(f"{cfg}/{name}: over_band_failures", r["over_band"],
                  len(s["over_band_failures"]))
            check(f"{cfg}/{name}: band-only result", r["result"], s["result"])
            check(f"{cfg}/{name}: four_conjuncts", r["four"], s["four_conjuncts"])
            check(f"{cfg}/{name}: n_token_mismatch", r["n_token_mismatch"],
                  s["n_token_mismatch"])
            check(f"{cfg}/{name}: n_exact_tie_mismatch", r["n_exact_tie"],
                  s["n_exact_tie_mismatch"])
            if name == "control_corrupt":
                # 2. the instrument discriminates, or nothing above means anything
                check(f"{cfg}/control: four_conjuncts is FAIL", r["four"], "FAIL")
                check(f"{cfg}/control: band result is FAIL", r["result"], "FAIL")
                check(f"{cfg}/control: worst_gap above 1 nat", r["worst"] > 1.0, True)
            else:
                n_div, worst, n_tm, four = PUBLISHED[cfg][name]
                check(f"{cfg}/{name}: PUBLISHED n_divergent", r["n_divergent"], n_div)
                check(f"{cfg}/{name}: PUBLISHED worst_gap", round(r["worst"], 6), worst)
                check(f"{cfg}/{name}: PUBLISHED n_token_mismatch",
                      r["n_token_mismatch"], n_tm)
                check(f"{cfg}/{name}: PUBLISHED four_conjuncts", r["four"], four)
                check(f"{cfg}/{name}: denominator is 288", r["total"], 288)

    # 3. the v2 token-gate table re-derives from the same two committed streams
    print("\n== the published v2 llama.cpp table, re-derived (NOT rescored) ==")
    diverged = []
    for i in range(6):
        first = next((s for s in range(48) if ours[i][s] != lcpp[i][s]), None)
        if first is not None:
            diverged.append(i)
            check(f"v2 row p{i}: (first diff, ours, oracle)",
                  (first, ours[i][first], lcpp[i][first]), V2_TABLE[i])
    check("v2 GENERATION_DIVERGENCES prompt set", diverged, V2_DIVERGENT_PROMPTS)

    print(f"\nCLAIMS_CHECKED = {CLAIMS}")
    print(f"MISMATCHES     = {len(MISMATCHES)}")
    for m in MISMATCHES:
        print("   ", m)
    return 1 if MISMATCHES else 0


if __name__ == "__main__":
    sys.exit(main())
