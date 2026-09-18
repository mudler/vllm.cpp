#!/usr/bin/env python3
"""Rank ROCm kernel time over the decode-only windows of a rocprofv3 trace.

WHY THIS EXISTS. `VT_OP_PROVIDER_STATS` counts which provider was selected, not
how long it ran, so nothing in this tree could say where a ROCm decode step goes.
On CUDA `nsys` answers that question directly. This is the ROCm equivalent.

HOW IT WINDOWS. Greedy decode emits exactly one sampler dispatch per generated
token, so consecutive sampler dispatches delimit one decode step. Everything
before the first mark is model load and prefill and is dropped by construction,
and the leading steps are dropped because our arm stages lazily on the first
generation. Nothing is added to the profiled process: the binary that is traced
is byte-identical to the binary that is not.

HOW IT REFUSES. The window is self-validating. A correct segmentation gives every
decode step the SAME dispatch count; a ragged one means the marker is not
once-per-token and the ranking would be meaningless. That refuses (exit 3) rather
than printing a table. An inverted duration refuses too (exit 4), because that is
the one failure rocprofiler-sdk's timestamp adjustment is documented to leave
behind (#3040) and a negative row must never be summed into a share.

WHAT IT CANNOT SEE. Kernel execution only. Host time is whatever the step wall
exceeds the kernel-busy sum by, and the tool prints that percentage rather than
attributing the gap to kernels. It cannot see overlap either: durations are summed
per name, and the printed occupancy is what tells a reader whether one queue was
in fact serial.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import statistics
import sys

# rocprofiler-sdk 1.1.0 adjusts inverted timestamps instead of failing
# (profiling_time.hpp:82) and does not mark which rows it touched. The committed
# gfx1151 capture warned on 62 of 85,737 rows. The aggregate error is bounded by
# assuming every adjusted row was maximally wrong, which is what
# `swap_bound_fraction` reports.
ADJUSTMENT_NOTE = (
    "rocprofiler-sdk adjusts inverted timestamps silently (#3040); "
    "worst-case aggregate error is reported as swap_bound_fraction"
)

_TEMPLATE = re.compile(r"(?:vt::rocm::)?(?:\(anonymous namespace\)::)?([A-Za-z_]\w*)\s*<")
_PLAIN = re.compile(r"([A-Za-z_]\w*)\s*\(")


def short_name(name: str) -> str:
    """Collapse a mangled dispatch name to the kernel identifier."""
    match = _TEMPLATE.search(name)
    if match:
        return match.group(1)
    match = _PLAIN.search(name)
    return match.group(1) if match else name.strip()


def read_trace(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("Kernel_Name")]
    if not rows:
        raise ValueError(f"{path}: no KERNEL_DISPATCH rows")
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    return rows


def durations(rows: list[dict]) -> list[int]:
    return [int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in rows]


def segment(rows: list[dict], sampler: str, skip_first: int) -> list[tuple[int, int]]:
    """Return the retained [start, end) decode-step slices.

    Raises when the marker cannot delimit steps, because a table built on a bad
    window is worse than no table.
    """
    marks = [i for i, r in enumerate(rows) if sampler in r["Kernel_Name"]]
    if len(marks) < skip_first + 3:
        raise LookupError(
            f"only {len(marks)} '{sampler}' dispatches; need at least "
            f"{skip_first + 3} to drop {skip_first} and still bound 2 steps"
        )
    return [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)][skip_first:]


def rank(rows: list[dict], segments: list[tuple[int, int]]) -> dict:
    per: dict[str, list[int]] = collections.defaultdict(list)
    for start, stop in segments:
        for row in rows[start:stop]:
            per[short_name(row["Kernel_Name"])].append(
                int(row["End_Timestamp"]) - int(row["Start_Timestamp"]))
    steps = len(segments)
    total = sum(sum(v) for v in per.values())
    walls = [int(rows[segments[i + 1][0]]["Start_Timestamp"])
             - int(rows[segments[i][0]]["Start_Timestamp"])
             for i in range(steps - 1)]
    table = []
    for name, values in sorted(per.items(), key=lambda kv: -sum(kv[1])):
        table.append({
            "kernel": name,
            "share_pct": 100.0 * sum(values) / total,
            "total_ms": sum(values) / 1e6,
            "per_step": len(values) / steps,
            "mean_us": statistics.mean(values) / 1e3,
            "min_us": min(values) / 1e3,
            "max_us": max(values) / 1e3,
        })
    return {
        "steps": steps,
        "dispatches_per_step": sorted({b - a for a, b in segments}),
        "kernel_busy_ms_per_step": total / steps / 1e6,
        "step_wall_ms_median": statistics.median(walls) / 1e6,
        "step_wall_ms_min": min(walls) / 1e6,
        "step_wall_ms_max": max(walls) / 1e6,
        "occupancy_pct": 100.0 * (total / steps) / statistics.median(walls),
        "total_ms": total / 1e6,
        "table": table,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", help="rocprofv3 *_kernel_trace.csv[.gz]")
    parser.add_argument("--sampler", default="ArgmaxK",
                        help="the once-per-token dispatch that delimits a decode step")
    parser.add_argument("--skip-first", type=int, default=1,
                        help="leading decode steps to discard (lazy staging)")
    parser.add_argument("--top", type=int, default=25)
    # Default None, NOT 0. A missing count and a count of zero are different
    # claims, and printing 0.00% for "you did not tell me" would read as "#3040
    # did not fire here" -- a silent zero is how an instrument reports success at
    # the wrong question.
    parser.add_argument("--swap-warnings", type=int, default=None,
                        help="count of rocprofv3 timestamp-adjustment warnings on "
                             "this run, from its stderr; bounds the #3040 error")
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)

    rows = read_trace(args.trace)
    all_d = durations(rows)
    inverted = sum(1 for d in all_d if d < 0)
    if inverted:
        print(f"REFUSED: {inverted} of {len(rows)} rows have End < Start. "
              f"{ADJUSTMENT_NOTE}. A negative row must not be summed into a share.",
              file=sys.stderr)
        return 4
    try:
        segments = segment(rows, args.sampler, args.skip_first)
    except LookupError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    sizes = sorted({b - a for a, b in segments})
    if len(sizes) != 1:
        print(f"REFUSED: decode steps are ragged, dispatch counts {sizes[:8]}. "
              f"'{args.sampler}' is not once per token, so the window is not a step.",
              file=sys.stderr)
        return 3

    result = rank(rows, segments)
    marks = [i for i, r in enumerate(rows) if args.sampler in r["Kernel_Name"]]
    result["trace"] = args.trace
    result["sampler"] = args.sampler
    result["skipped_steps"] = args.skip_first
    result["excluded_load_prefill_rows"] = marks[0]
    # The worst case if every adjusted row actually cost the longest duration seen.
    # The CSV does not mark the adjusted rows, so this is assumed, not inspected.
    result["max_duration_us"] = max(all_d) / 1e3
    result["swap_warnings"] = args.swap_warnings
    result["swap_bound_fraction"] = (
        (args.swap_warnings * max(all_d)) / sum(all_d)
        if args.swap_warnings is not None and sum(all_d) else None)
    result["note"] = ADJUSTMENT_NOTE

    print(f"trace                 : {args.trace}")
    print(f"sampler marks         : {len(marks)} ('{args.sampler}')")
    print(f"decode steps ranked   : {result['steps']} "
          f"(dropped {args.skip_first} leading; "
          f"{result['excluded_load_prefill_rows']} load+prefill rows excluded)")
    print(f"dispatches per step   : {sizes}   <- one value validates the window")
    print(f"step wall ms          : median {result['step_wall_ms_median']:.2f}  "
          f"min {result['step_wall_ms_min']:.2f}  max {result['step_wall_ms_max']:.2f}")
    print(f"kernel busy per step  : {result['kernel_busy_ms_per_step']:.2f} ms "
          f"({result['occupancy_pct']:.1f}% of step wall; the remainder is HOST time)")
    if args.swap_warnings is None:
        print("#3040 swap warnings   : NOT SUPPLIED -- the #3040 error bound is "
              "UNKNOWN for this trace. Pass --swap-warnings N, counted from the "
              "run's own stderr. This is not a claim that none fired.")
    else:
        print(f"#3040 swap warnings   : {args.swap_warnings} of {len(rows)} rows "
              f"({100.0 * args.swap_warnings / len(rows):.4f}%); worst-case share of "
              f"the kernel budget {100.0 * result['swap_bound_fraction']:.2f}% "
              f"(every adjusted row assumed to cost the trace maximum, "
              f"{result['max_duration_us'] / 1e3:.3f} ms)")
    print(f"note                  : {ADJUSTMENT_NOTE}")
    print()
    header = (f"{'kernel':<40}{'share%':>8}{'total ms':>11}{'n/step':>8}"
              f"{'mean us':>10}{'min us':>9}{'max us':>10}")
    print(header)
    print("-" * len(header))
    for entry in result["table"][:args.top]:
        print(f"{entry['kernel'][:40]:<40}{entry['share_pct']:>7.2f}%"
              f"{entry['total_ms']:>11.1f}{entry['per_step']:>8.1f}"
              f"{entry['mean_us']:>10.1f}{entry['min_us']:>9.1f}{entry['max_us']:>10.1f}")
    print("-" * len(header))
    print(f"{'TOTAL':<40}{100.0:>7.2f}%{result['total_ms']:>11.1f}"
          f"{sum(e['per_step'] for e in result['table']):>8.1f}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=1)
        print(f"\njson written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
