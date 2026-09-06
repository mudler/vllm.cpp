#!/usr/bin/env python3
"""Fold the #2944 vllm.cpp arm legs. N COMES FROM THE DESIGN, never from a tally.

The design (rounds, token counts, repeat, cold runs) is passed in. This script
counts nothing: it looks for exactly the legs the design says exist and reports
the ones it could not read as MISSING rather than shrinking N to fit.

Two definitions live here and they are never mixed:

  whole_completion  what `vllm-cli` prints as tok_s -- completion_tokens over the
                    wall time of the whole `vllm_complete()` call, prompt
                    included. This is the SAME quantity as vLLM's generate() leg.
  decode_derived    DERIVED, from the slope between two token counts. It is the
                    only quantity that may sit beside `llama-bench -p 0`, which
                    is pure decode. PR #2940 records what happens otherwise: a
                    meaningless 1.814x from dividing two different quantities.
"""
import argparse
import json
import pathlib
import re
import statistics
import sys

# Carried from the published survey and the landed denominator. NOT re-measured
# by this run, and labelled as constants wherever they appear.
REFERENCE = {
    "llamacpp_b10451_decode_tok_s": 12.219,
    "llamacpp_b10451_decode_tok_s_landed_denominator": 12.233,
    "vllm_5559679229_whole_completion_tok_s": 6.734,
    "vllm_5559679229_decode_derived_tok_s": 11.056,
    "source": "docs/benchmarks/qwen38-27b-q4km-gfx1151.md (PR #2940) and "
              "docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md",
    "note": "constants carried from earlier evidence; no leg of this run re-measured them",
}
TOKEN_GATE = {
    "state": "FAIL",
    "vllmcpp_vs_llamacpp_divergent_prompts": "3 of 6",
    "vllmcpp_vs_vllm_compiled_divergent_prompts": "5 of 6",
    "every_divergence": "a near-tie at about 0.125 nats, one bf16 ULP",
    "deterministic_denominator": "none established on this path",
    "note": "these counts are constants carried from earlier evidence, not measured here",
}

RUN_RE = re.compile(
    r"run=(\d+)/(\d+) finish_reason=(\S+) prompt_tokens=(\d+) "
    r"completion_tokens=(\d+) secs=([0-9.]+) tok_s=([0-9.]+)")


def median(xs):
    return statistics.median(xs) if xs else None


def spread_pct(xs):
    m = median(xs)
    if not xs or not m:
        return None
    return 100.0 * (max(xs) - min(xs)) / m


def read_rc(p):
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return None


def clock_window(path):
    if not path.exists():
        return None
    busy, sclk, n = [], [], 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        if r.get("busy_percent") is not None:
            busy.append(r["busy_percent"])
        if r.get("sclk_mhz") is not None:
            sclk.append(r["sclk_mhz"])
    if not n:
        return None
    out = {"samples": n}
    for name, xs in (("busy_percent", busy), ("sclk_mhz", sclk)):
        if xs:
            out[name] = {"mean": round(statistics.fmean(xs), 1),
                         "min": min(xs), "max": max(xs), "n": len(xs)}
    return out


def classify(err_text, rc):
    # A refusal is not a board fault. Order matters and mirrors the job script.
    if "no kernel for op" in err_text:
        return "OPREFUSED"
    if "Memory access fault" in err_text:
        return "MEMFAULT"
    if "GPU Hang" in err_text:
        return "GPUHANG"
    if rc == 124:
        return "TIMEOUT"
    if rc == 0:
        return "OK"
    return "OTHER_rc%s" % rc


def leg(ev, tag, cold_runs):
    err = ev / ("%s.err" % tag)
    rc = read_rc(ev / ("%s.rc" % tag))
    out = {"tag": tag, "rc": rc, "clock": clock_window(ev / ("clock-%s.jsonl" % tag))}
    if not err.exists():
        out["class"] = "MISSING_CAPTURE"
        out["read"] = False
        return out
    text = err.read_text(errors="replace")
    out["class"] = classify(text, rc)
    out["read"] = True
    reps = [{"run": int(m.group(1)), "of": int(m.group(2)),
             "finish_reason": m.group(3),
             "prompt_tokens": int(m.group(4)),
             "completion_tokens": int(m.group(5)),
             "secs": float(m.group(6)), "tok_s": float(m.group(7))}
            for m in RUN_RE.finditer(text)]
    out["repetitions"] = reps
    # `grep -c` semantics without the trap: both spellings of the tier banner.
    out["reference_tier_lines"] = len(re.findall(r"reference[- ]tier", text))
    kept = [r for r in reps if r["run"] > cold_runs]
    out["discarded_cold"] = cold_runs
    if kept:
        out["kept_completions"] = len(kept)
        out["whole_completion_tok_s"] = median([r["tok_s"] for r in kept])
        out["whole_completion_secs"] = median([r["secs"] for r in kept])
        out["completion_tokens"] = median([r["completion_tokens"] for r in kept])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=pathlib.Path, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--ngen-a", type=int, required=True)
    ap.add_argument("--ngen-b", type=int, required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--cold-runs", type=int, required=True)
    a = ap.parse_args()

    design = {"rounds": a.rounds, "ngen_a": a.ngen_a, "ngen_b": a.ngen_b,
              "repeat": a.repeat, "cold_runs": a.cold_runs,
              "legs_by_design": a.rounds * 2,
              "n_source": "the design, passed in; this script tallies nothing"}

    legs = {}
    for ngen in (a.ngen_a, a.ngen_b):
        legs[ngen] = [leg(a.evidence, "n%d-r%d" % (ngen, r), a.cold_runs)
                      for r in range(1, a.rounds + 1)]

    result = {"design": design, "token_gate": TOKEN_GATE, "reference": REFERENCE,
              "quantity_of_our_figure": (
                  "whole completion (vllm-cli times the whole vllm_complete() call, "
                  "prompt included). NOT decode. Comparable to vLLM's generate() leg; "
                  "NOT comparable to llama-bench -p 0."),
              "legs": {str(k): v for k, v in legs.items()}, "arms": {}}

    faulted_total = 0
    for ngen, ls in legs.items():
        ok = [x for x in ls if x.get("class") == "OK" and "whole_completion_tok_s" in x]
        faulted = [x for x in ls if x.get("class") != "OK"]
        faulted_total += len(faulted)
        tok = [x["whole_completion_tok_s"] for x in ok]
        secs = [x["whole_completion_secs"] for x in ok]
        result["arms"]["n%d" % ngen] = {
            "legs_by_design": a.rounds,
            "legs_completed": len(ok),
            "fault_rate": "%d of %d" % (len(faulted), a.rounds),
            "fault_classes": sorted({x.get("class") for x in faulted}),
            "median_whole_completion_tok_s": median(tok),
            "leg_spread_pct_of_median": spread_pct(tok),
            "per_leg_whole_completion_tok_s": tok,
            "median_whole_completion_secs": median(secs),
            "per_leg_secs": secs,
        }

    result["fault_rate_all_legs"] = "%d of %d" % (faulted_total, design["legs_by_design"])

    # --- the DERIVED decode figure, per round, exactly as the survey derived vLLM's ---
    slopes, fixed = [], []
    rows = []
    for r in range(1, a.rounds + 1):
        la = next((x for x in legs[a.ngen_a] if x["tag"] == "n%d-r%d" % (a.ngen_a, r)), None)
        lb = next((x for x in legs[a.ngen_b] if x["tag"] == "n%d-r%d" % (a.ngen_b, r)), None)
        if not la or not lb:
            continue
        ta = la.get("whole_completion_secs")
        tb = lb.get("whole_completion_secs")
        if ta is None or tb is None or tb <= ta:
            continue
        slope = (a.ngen_b - a.ngen_a) / (tb - ta)
        overhead = ta - a.ngen_a / slope
        slopes.append(slope)
        fixed.append(overhead)
        rows.append({"round": r, "t_a_s": ta, "t_b_s": tb,
                     "slope_tok_s": slope, "implied_fixed_cost_s": overhead})
    result["decode_derived"] = {
        "method": ("slope between two token counts on the SAME prompt in the same "
                   "lease; removes whatever fixed cost each call pays. DERIVED, "
                   "not a direct decode reading."),
        "rows": rows,
        "pairs_by_design": a.rounds,
        "pairs_usable": len(rows),
        "median_decode_tok_s_derived": median(slopes),
        "spread_pct_of_median": spread_pct(slopes),
        "median_implied_fixed_cost_s": median(fixed),
    }

    # --- ratios, each stated with the definition it is taken on ---
    ratios = {}
    wc = result["arms"].get("n%d" % a.ngen_a, {}).get("median_whole_completion_tok_s")
    dd = result["decode_derived"]["median_decode_tok_s_derived"]
    if wc:
        ratios["like_for_like_whole_completion__vllm_over_vllmcpp"] = {
            "value": REFERENCE["vllm_5559679229_whole_completion_tok_s"] / wc,
            "numerator": "vLLM 6.734 tok/s whole completion (carried constant)",
            "denominator": "vllm.cpp %.3f tok/s whole completion (measured here)" % wc,
            "definition": "both are whole completion; this ratio is like-for-like",
        }
    if dd:
        ratios["like_for_like_decode__llamacpp_over_vllmcpp_derived"] = {
            "value": REFERENCE["llamacpp_b10451_decode_tok_s"] / dd,
            "numerator": "llama.cpp 12.219 tok/s decode (carried constant, llama-bench -p 0)",
            "denominator": "vllm.cpp %.3f tok/s decode, DERIVED from the slope" % dd,
            "definition": "decode against decode; our side is DERIVED and labelled so",
        }
    if wc:
        ratios["REFUSED__llamacpp_decode_over_vllmcpp_whole_completion"] = (
            "not computed: dividing pure decode by whole completion compares two "
            "different quantities. PR #2940 records that exact mistake emitting 1.814x.")
    result["ratios"] = ratios

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
