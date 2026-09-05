#!/usr/bin/env python3
"""Fold one llama.cpp denominator run into a RESULT.json.

It states, in its own output and in words, WHAT it compared against WHAT:
every derived number carries the population it came from, and the clock
instrument names itself as ad-hoc rather than as the committed helper.

N comes from --legs, which is the DESIGN. It is never derived by counting log
lines: `grep -c 'avg_ts' job.log` reads 13 against six legs, because the per-leg
echo, this script's own `population` string and the whole RESULT.json all land in
the same log. Re-emission is the mechanism; `tee -a` writes each line once.

But a declared N is only half a guard, so --legs is checked in BOTH directions
against the legs actually present in --evidence. Lowering it must not drop one.

THIS SCRIPT WRITES RESULT.json INTO --evidence. Point it at a scratch copy when
testing, or it overwrites the committed artifact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--evidence", type=pathlib.Path, required=True)
ap.add_argument("--legs", type=int, required=True, help="leg count from the DESIGN")
ap.add_argument("--busy-threshold", type=int, default=90)
args = ap.parse_args()

E = args.evidence


def clock_window(path: pathlib.Path, threshold: int) -> dict:
    if not path.is_file():
        return {"error": f"no sampler output at {path.name}"}
    samples = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def fold(rows: list[dict], label: str) -> dict:
        sclks = [int(r["sclk_mhz"]) for r in rows if r.get("sclk_mhz") is not None]
        if not sclks:
            return {"window": label, "samples": len(rows), "sclk_mhz": None,
                    "note": "no active sclk value in this window"}
        med = statistics.median(sclks)
        return {
            "window": label,
            "samples": len(rows),
            "sclk_mhz": {
                "min": min(sclks),
                "median": med,
                "mean": round(statistics.fmean(sclks), 1),
                "max": max(sclks),
                "spread_pct": round(100.0 * (max(sclks) - min(sclks)) / med, 1),
            },
        }

    busy = [r for r in samples
            if r.get("busy_percent") is not None and int(r["busy_percent"]) >= threshold]
    return {
        "instrument": "AD-HOC /sys/class/drm sampler carried beside the job; "
                      "NOT tools/bench/gpu_clock_state.py, which reads NVIDIA fields "
                      "and does not run on this board (#2381 owns the in-tree gap)",
        "interval_s": 0.25,
        "boot_ids": sorted({str(r.get("boot_id")) for r in samples}),
        "whole_window": fold(samples, f"whole leg, including the 17 GiB model load"),
        "compute_window": fold(busy, f"gpu_busy_percent >= {threshold}"),
    }


legs = []
for i in range(1, args.legs + 1):
    row: dict = {"leg": i, "order": i}
    jpath = E / f"leg{i}.json"
    rcpath = E / f"leg{i}.rc"
    row["rc"] = int(rcpath.read_text().strip()) if rcpath.is_file() else None
    if jpath.is_file():
        try:
            tests = json.loads(jpath.read_text())
        except json.JSONDecodeError as exc:
            row["error"] = f"llama-bench JSON did not parse: {exc}"
            tests = []
        for t in tests:
            row["avg_ts"] = t.get("avg_ts")
            row["stddev_ts"] = t.get("stddev_ts")
            row["samples_ts"] = t.get("samples_ts")
            row["n_gen"] = t.get("n_gen")
            row["n_prompt"] = t.get("n_prompt")
            row["n_gpu_layers"] = t.get("n_gpu_layers")
            row["n_threads"] = t.get("n_threads")
            row["backends"] = t.get("backends")
            row["build_commit"] = t.get("build_commit")
            row["build_number"] = t.get("build_number")
            row["cpu_info"] = t.get("cpu_info")
            row["gpu_info"] = t.get("gpu_info")
            row["model_type"] = t.get("model_type")
            row["model_size"] = t.get("model_size")
            row["flash_attn"] = t.get("flash_attn")
            row["type_k"] = t.get("type_k")
            row["type_v"] = t.get("type_v")
    else:
        row["error"] = "no llama-bench JSON for this leg"
    row["clock"] = clock_window(E / f"clock-leg{i}.jsonl", args.busy_threshold)
    legs.append(row)

usable = [l for l in legs if l.get("avg_ts") is not None and l.get("rc") == 0]
values = [float(l["avg_ts"]) for l in usable]
reps: list[float] = []
for l in usable:
    reps.extend(float(v) for v in (l.get("samples_ts") or []))

reasons = []
# The refusal must run in BOTH directions, and against what is ON DISK.
#
# A `len(legs) != args.legs` test sat here and was structurally unreachable:
# `legs` is built by a loop of exactly `args.legs` iterations, so no mutation
# could ever break it. It also left the dangerous direction open. `--legs 7`
# refused, because the seventh leg produced no usable figure -- but `--legs 5`
# folded five legs, ignored `leg6.*` entirely and reported MEASURED with rc 0.
# A leg could be DROPPED by lowering the declared design, which is the one
# failure a leg-count guard exists to stop.
#
# So the directory is enumerated and compared against the design. This is a real
# function of the tree: lowering `--legs` now names the legs it would have
# dropped, and raising it names the legs it cannot find.
found: set[int] = set()
for pattern in ("leg*.json", "leg*.rc", "clock-leg*.jsonl"):
    for p in E.glob(pattern):
        m = re.fullmatch(r"(?:clock-)?leg(\d+)\.(?:json|rc|jsonl)", p.name)
        if m:
            found.add(int(m.group(1)))
on_disk = sorted(found)
declared = list(range(1, args.legs + 1))
undeclared = [i for i in on_disk if i not in declared]
absent = [i for i in declared if i not in on_disk]
if undeclared:
    reasons.append(
        f"design declares {args.legs} legs, but the evidence directory also holds "
        f"leg(s) {undeclared}; a leg may not be dropped by lowering --legs"
    )
if absent:
    reasons.append(
        f"design declares {args.legs} legs and the evidence directory holds "
        f"nothing at all for leg(s) {absent}"
    )
if len(usable) != args.legs:
    reasons.append(f"{args.legs - len(usable)} of {args.legs} legs produced no usable figure")
if len(usable) < 2:
    reasons.append("a denominator may not be reported from one leg")

summary = None
if values:
    med = statistics.median(values)
    summary = {
        "population": f"one avg_ts per leg, {len(values)} legs, "
                      "each the mean of that leg's own repetitions",
        "leg_count": len(values),
        "median_tok_s": round(med, 3),
        "mean_tok_s": round(statistics.fmean(values), 3),
        "min_tok_s": round(min(values), 3),
        "max_tok_s": round(max(values), 3),
        "spread_pct_of_median": round(100.0 * (max(values) - min(values)) / med, 3),
        "repetition_count": len(reps),
        "repetition_median_tok_s": round(statistics.median(reps), 3) if reps else None,
        "repetition_min_tok_s": round(min(reps), 3) if reps else None,
        "repetition_max_tok_s": round(max(reps), 3) if reps else None,
    }

result = {
    "what_this_is": "llama.cpp b10451 decode on gfx1151, ONE engine. "
                    "A denominator, measured alone.",
    "what_this_is_not": "It is not a comparison. No vllm.cpp figure was produced "
                        "by this run and no ratio is computed here: that arm's "
                        "declared token gate reads FAIL at 3 of 6 "
                        "(docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md) "
                        "and AGENTS.md Gates admits no performance result from it.",
    "status": "MEASURED" if not reasons else "INCOMPLETE",
    "reasons": reasons,
    "legs_by_design": args.legs,
    "legs_folded": len(legs),
    "denominator": summary,
    "clock_attribution": "AD-HOC. Every clock figure below comes from the carried "
                         "sysfs sampler, not from the committed helper (#2381).",
    "legs": legs,
}

(E / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if not reasons else 3)
