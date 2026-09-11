#!/usr/bin/env python3
"""Fold the multi-engine survey's legs into one result (#2497).

N COMES FROM THE DESIGN. Every arm's leg count is passed in with --rounds and
the fold refuses to infer one by counting anything. The reason is re-emission,
not `tee`: a job log carries each leg's figure in its own step line and again
inside the RESULT.json the job echoes back, so a grep tally over it answers
whatever the grep term happens to hit. On the oracle-only run of 2026-09-02 that
tally read 13 against six legs.

The fold computes each arm's own figures and the ratios between them. The ratio
is admissible here only because a recorded developer decision ratified this
survey (developer-preferences.md, 2026-09-04); every ratio is emitted beside
both absolutes and beside the correctness status, never alone.
"""
import argparse
import json
import pathlib
import re
import statistics

ARMS = ("vllmcpp", "llamacpp", "llamacli", "vllm")


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
    """Mean/min/max of the busy and sclk samples the leg's own sampler wrote."""
    if not path.exists():
        return None
    busy, sclk, n = [], [], 0
    for line in path.read_text().splitlines():
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


def leg_vllmcpp(ev, tag, cold_runs):
    """vllm-cli prints one `run=i/n ... tok_s=` line per completion."""
    err = ev / f"{tag}.err"
    if not err.exists():
        return None
    reps = []
    for m in re.finditer(
        r"run=(\d+)/(\d+) finish_reason=(\S+) prompt_tokens=(\d+) "
        r"completion_tokens=(\d+) secs=([0-9.]+) tok_s=([0-9.]+)",
        err.read_text(errors="replace"),
    ):
        reps.append({"run": int(m.group(1)), "of": int(m.group(2)),
                     "finish_reason": m.group(3),
                     "prompt_tokens": int(m.group(4)),
                     "completion_tokens": int(m.group(5)),
                     "secs": float(m.group(6)), "tok_s": float(m.group(7))})
    if not reps:
        return None
    kept = [r for r in reps if r["run"] > cold_runs]
    if not kept:
        return None
    return {"repetitions": reps, "kept": kept,
            "discarded_cold": cold_runs,
            "tok_s": median([r["tok_s"] for r in kept])}


def leg_llamacpp(ev, tag):
    """llama-bench's own avg_ts, straight out of its JSON. No re-derivation."""
    p = ev / f"{tag}.json"
    if not p.exists():
        return None
    try:
        recs = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not recs:
        return None
    r = recs[0]
    return {"avg_ts": r.get("avg_ts"), "stddev_ts": r.get("stddev_ts"),
            "samples_ts": r.get("samples_ts"), "n_gen": r.get("n_gen"),
            "n_gpu_layers": r.get("n_gpu_layers"), "backend": r.get("backend"),
            "tok_s": r.get("avg_ts")}


LLAMA_PERF = {
    "prompt_eval": re.compile(
        r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens"),
    "eval": re.compile(
        r"\beval time =\s*([0-9.]+) ms /\s*(\d+) runs\s*\(\s*[0-9.]+ ms per token,\s*([0-9.]+) tokens per second"),
    "total": re.compile(r"total time =\s*([0-9.]+) ms /\s*(\d+) tokens"),
}


def leg_llamacli(ev, tag):
    """llama.cpp through its ORDINARY request path, so the end-to-end column has
    a llama.cpp row measured the same way as the other two engines' rows."""
    p = ev / f"{tag}.err"
    if not p.exists():
        return None
    txt = p.read_text(errors="replace")
    m_eval = LLAMA_PERF["eval"].search(txt)
    if not m_eval:
        return None
    eval_ms, eval_runs, eval_tps = (float(m_eval.group(1)), int(m_eval.group(2)),
                                    float(m_eval.group(3)))
    out = {"eval_ms": eval_ms, "eval_runs": eval_runs,
           "decode_tok_s": eval_tps, "tok_s": eval_tps}
    m_pp = LLAMA_PERF["prompt_eval"].search(txt)
    if m_pp:
        out["prompt_eval_ms"] = float(m_pp.group(1))
        out["prompt_tokens"] = int(m_pp.group(2))
        # End to end over the same object vllm-cli and the vLLM leg time: the
        # prompt plus the generation, model load excluded on every side.
        e2e_ms = out["prompt_eval_ms"] + eval_ms
        out["e2e_ms"] = e2e_ms
        out["e2e_tok_s"] = (eval_runs + 1) / (e2e_ms / 1000.0) if e2e_ms > 0 else None
    m_tot = LLAMA_PERF["total"].search(txt)
    if m_tot:
        out["total_ms"] = float(m_tot.group(1))
        out["total_tokens"] = int(m_tot.group(2))
    return out


def leg_vllm(ev, tag, cold_runs):
    p = ev / f"{tag}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    reps = d.get("runs") or []
    kept = [r for r in reps if r["run"] > cold_runs]
    if not kept:
        return None
    ngen = d.get("n_gen", 64)
    one = d.get("one_token_secs")
    secs = [r["secs"] for r in kept]
    ms = median(secs)
    decode_only = None
    if one is not None and ms is not None and ms > one and ngen > 1:
        decode_only = (ngen - 1) / (ms - one)
    # The SLOPE between two ordinary requests of NGEN and 2*NGEN tokens. Both
    # points are on the same path, so every fixed per-request cost cancels and no
    # special case enters. This is the decode figure to trust; the one-token
    # subtraction above is the cross-check.
    slope_decode = None
    dbl = [r["secs"] for r in (d.get("double_runs") or [])]
    md = median(dbl)
    if md is not None and ms is not None and md > ms:
        slope_decode = ngen / (md - ms)
    return {"repetitions": reps, "kept": kept, "discarded_cold": cold_runs,
            "enforce_eager": d.get("enforce_eager"),
            "gpu_memory_utilization": d.get("gpu_memory_utilization"),
            "max_model_len": d.get("max_model_len"),
            "config_fallback_used": d.get("config_fallback_used"),
            "load_secs": d.get("load_secs"),
            "one_token_secs": one,
            "one_token_runs": d.get("one_token_runs"),
            "double_runs": d.get("double_runs"),
            "tok_s_decode_only_one_token": decode_only,
            "tok_s_decode_only_slope": slope_decode,
            "tok_s": median([r["tok_s"] for r in kept])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=pathlib.Path, required=True)
    ap.add_argument("--rounds", type=int, required=True,
                    help="legs per arm, DECLARED by the design")
    ap.add_argument("--cold-runs", type=int, default=1)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    ev = args.evidence
    arms = [a for a in args.arms.split(",") if a]

    result = {"design": {"rounds": args.rounds, "arms": arms,
                         "cold_runs": args.cold_runs,
                         "legs_per_arm_by_design": args.rounds},
              "arms": {}, "faults": [], "status": "MEASURED", "reasons": []}

    for arm in arms:
        legs, faulted = [], []
        for r in range(1, args.rounds + 1):
            tag = f"{arm}-r{r}"
            rc = read_rc(ev / f"{tag}.rc")
            errtxt = ""
            e = ev / f"{tag}.err"
            if e.exists():
                errtxt = e.read_text(errors="replace")
            hang = bool(re.search(r"GPU Hang|HW Exception|Memory access fault", errtxt))
            if arm == "vllmcpp":
                fig = leg_vllmcpp(ev, tag, args.cold_runs)
            elif arm == "llamacpp":
                fig = leg_llamacpp(ev, tag)
            elif arm == "llamacli":
                fig = leg_llamacli(ev, tag)
            else:
                fig = leg_vllm(ev, tag, args.cold_runs)
            entry = {"leg": r, "tag": tag, "rc": rc, "board_fault": hang,
                     "clock": clock_window(ev / f"clock-{tag}.jsonl")}
            if arm == "vllmcpp":
                n = errtxt.count("[vt reference-tier]")
                entry["reference_tier_notices"] = n if e.exists() else "UNREAD"
            if fig is None or rc not in (0,) or hang:
                entry["usable"] = False
                if fig is not None:
                    entry["figure"] = fig
                faulted.append(entry)
                result["faults"].append(
                    {"arm": arm, "leg": r, "rc": rc, "board_fault": hang,
                     "figure_parsed": fig is not None})
            else:
                entry["usable"] = True
                entry.update(fig)
                legs.append(entry)
        tok = [l["tok_s"] for l in legs if l.get("tok_s") is not None]
        reps = []
        for l in legs:
            if arm == "llamacli":
                reps.append(l["tok_s"])
            elif arm == "llamacpp":
                reps.extend(l.get("samples_ts") and
                            [l["n_gen"] / s for s in l["samples_ts"]] or [])
            else:
                reps.extend([r["tok_s"] for r in l.get("kept", [])])
        result["arms"][arm] = {
            "legs_by_design": args.rounds,
            "legs_completed": len(legs),
            "legs_faulted": len(faulted),
            "board_fault_rate": f"{len(faulted)}/{args.rounds}",
            "median_tok_s": median(tok),
            "mean_tok_s": statistics.fmean(tok) if tok else None,
            "min_leg_tok_s": min(tok) if tok else None,
            "max_leg_tok_s": max(tok) if tok else None,
            "leg_spread_pct_of_median": spread_pct(tok),
            "repetition_median_tok_s": median(reps),
            "repetition_min_tok_s": min(reps) if reps else None,
            "repetition_max_tok_s": max(reps) if reps else None,
            "repetition_count": len(reps),
            "legs": legs,
            "faulted_legs": faulted,
        }
        if len(legs) == 0:
            result["status"] = "INCOMPLETE"
            result["reasons"].append(f"{arm}: no usable leg")
        elif len(legs) < args.rounds:
            result["reasons"].append(
                f"{arm}: {len(legs)} of {args.rounds} legs usable")

    meds = {a: result["arms"][a]["median_tok_s"] for a in arms}
    ratios = {}
    for a in arms:
        for b in arms:
            if a != b and meds[a] and meds[b]:
                ratios[f"{a}_over_{b}"] = {
                    "ratio": meds[a] / meds[b],
                    "numerator_tok_s": meds[a],
                    "denominator_tok_s": meds[b],
                }
    result["ratios"] = ratios
    # The caveat travels WITH the numbers, in the machine-readable artifact too,
    # so no consumer of this file can lift a ratio out of it without it.
    result["correctness"] = {
        "token_gate": "FAIL",
        "vllmcpp_vs_llamacpp_b10451": "3 of 6 prompts divergent",
        "vllmcpp_vs_vllm_compiled": "5 of 6 prompts divergent",
        "vllm_compiled_vs_llamacpp_b10451": "3 of 6 prompts divergent",
        "vllm_eager_vs_llamacpp_b10451": "4 of 6 prompts divergent",
        "deterministic_denominator": "NONE ESTABLISHED on this path",
        "note": "This is a SURVEY, not a parity claim. No ratio here is a gated "
                "result. Ratified by the recorded developer decision of "
                "2026-09-04 in .agents/developer-preferences.md.",
    }
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["status"] == "MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
