#!/usr/bin/env python3
"""Attribute a cross-tier delta profile to layer families, and calibrate it.

ROCM-TIER-DIVERGENCE (#2590).
Spec: .agents/specs/rocm-tier-hidden-state-bisect.md section 6.

`tier-hidden-delta.py` prints WHERE the two tiers differ. This says WHETHER that
difference is a defect or the arithmetic, and it answers with two computations
rather than an impression.

1. THE RANDOM-WALK TEST. If each layer contributes an independent rounding
   difference, `rel_l2` squared grows LINEARLY in the layer index and the
   per-layer increment is roughly constant. The increment is reported per layer
   family, so a family that contributes disproportionately names itself. A single
   defective op appears as one layer whose increment dwarfs its family's, which a
   ratio to the previous layer cannot show at small `rel_l2` (early layers ramp
   by 2x on rounding alone).

2. THE bf16 PREDICTION. Both tiers carry the residual stream in bf16, whose unit
   roundoff is 2^-8. Round-to-nearest gives an RMS relative error of eps/sqrt(3)
   per store; two INDEPENDENT roundings differ by sqrt(2) times that; over L
   layers a random walk accumulates as sqrt(L). That number is printed beside the
   measurement, because "the tiers differ by 2.6%" means nothing until it is next
   to what they MUST differ by.

STEPS AFTER THE FIRST DIVERGENCE ARE EXCLUDED, and this is not a detail. Once the
two tiers emit different tokens they consume different inputs, so every later
step compares two different computations. Pass --first-divergence; the tool
refuses to guess it, and reports what the excluded steps looked like so a reader
can see the exclusion was needed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys


def die(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True,
                    help="the --json output of tier-hidden-delta.py")
    ap.add_argument("--first-divergence", type=int, required=True,
                    help="the step at which the two tiers first emit different "
                         "tokens; steps AFTER it are excluded as incomparable")
    ap.add_argument("--stage", default="stream",
                    help="which stage to profile (default: stream = hidden+res)")
    ap.add_argument("--residual-bits", type=int, default=8,
                    help="mantissa bits of the residual store, including the "
                         "implicit one (bf16 = 8)")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as fh:
        blob = json.load(fh)
    rows = [r for r in blob["rows"] if r["stage"] == args.stage]
    if not rows:
        die(f"no rows with stage {args.stage!r} in {args.json}")

    keep = [r for r in rows if r["step"] <= args.first_divergence]
    drop = [r for r in rows if r["step"] > args.first_divergence]
    steps = sorted({r["step"] for r in keep})
    layers = sorted({r["layer"] for r in keep})
    print("== tier-delta-attribution ==")
    print(f"source={args.json}  A={blob['label_a']} B={blob['label_b']}  "
          f"stage={args.stage}")
    print(f"comparable steps = 0..{args.first_divergence} ({len(steps)}); "
          f"excluded as post-divergence = {len({r['step'] for r in drop})}")
    if drop:
        emb = [r for r in drop if r["layer"] == min(layers)]
        if emb:
            print(f"  the exclusion is load-bearing: at layer {min(layers)} the "
                  f"EXCLUDED steps reach max_abs "
                  f"{max(r['max_abs'] for r in emb):.4g}, against "
                  f"{max((r['max_abs'] for r in keep if r['layer'] == min(layers)), default=0):.4g} "
                  f"on the comparable ones")

    by = {}
    for r in keep:
        by.setdefault(r["step"], {})[r["layer"]] = r

    # The pre-layer snapshot, if present, is the input both tiers were handed.
    if min(layers) < 0:
        e = [by[s][min(layers)] for s in by if min(layers) in by[s]]
        nz = [r for r in e if r["max_abs"] != 0.0]
        print(f"\ninput (layer {min(layers)}): {len(e) - len(nz)} of {len(e)} steps "
              f"BIT-IDENTICAL; max_abs over all = {max(r['max_abs'] for r in e):.4g}")

    print("\n== the random-walk test: increment of rel_l2^2 per layer ==")
    inc = {}
    for s in by:
        ls = sorted(k for k in by[s] if k >= 0)
        prev = by[s].get(min(layers), {"rel_l2": 0.0})["rel_l2"]
        prev = prev if math.isfinite(prev) else 0.0
        for l in ls:
            cur = by[s][l]["rel_l2"]
            if not math.isfinite(cur):
                continue
            inc.setdefault(by[s][l]["family"], []).append(cur * cur - prev * prev)
            prev = cur
    total_var = 0.0
    print(f"{'family':>8} {'layers':>7} {'median inc':>13} {'mean inc':>13} "
          f"{'sqrt(median)':>13} {'family var':>13} {'share':>7}")
    counts = {f: len({l for s in by for l in by[s]
                      if l >= 0 and by[s][l]["family"] == f}) for f in inc}
    var = {f: counts[f] * st.median(v) for f, v in inc.items()}
    total_var = sum(max(x, 0.0) for x in var.values())
    for f in sorted(inc):
        v = inc[f]
        share = (max(var[f], 0.0) / total_var * 100) if total_var > 0 else float("nan")
        print(f"{f:>8} {counts[f]:>7} {st.median(v):13.4e} {st.mean(v):13.4e} "
              f"{math.sqrt(abs(st.median(v))):13.4e} {var[f]:13.4e} {share:6.1f}%")
    print(f"  predicted final rel_l2 from these increments: "
          f"{math.sqrt(total_var):.4e}")

    last = max(layers)
    fin = [by[s][last]["rel_l2"] for s in by if last in by[s]]
    print(f"\n== the measurement at the last layer ({last}) ==")
    print(f"  rel_l2  median={st.median(fin):.4e}  min={min(fin):.4e}  "
          f"max={max(fin):.4e}  n={len(fin)}")

    eps = 2.0 ** (-args.residual_bits)
    per_store = eps / math.sqrt(3.0) * math.sqrt(2.0)
    nlayers = len([l for l in layers if l >= 0])
    pred = per_store * math.sqrt(nlayers)
    print(f"\n== the bf16 prediction, for one store per layer ==")
    print(f"  unit roundoff 2^-{args.residual_bits} = {eps:.4e}; RMS relative "
          f"rounding error eps/sqrt(3) = {eps / math.sqrt(3):.4e}")
    print(f"  two INDEPENDENT roundings differ by sqrt(2)x that = "
          f"{per_store:.4e} per store")
    print(f"  a random walk over {nlayers} layers reaches {pred:.4e}")
    print(f"  MEASURED / PREDICTED = {st.median(fin) / pred:.3f}")
    print(f"\n  A ratio near 1 says the two tiers differ by what two independent "
          f"{args.residual_bits}-bit\n  residual streams MUST differ by, and that "
          f"no op is contributing more than its\n  own rounding. A ratio far above "
          f"1 says something else is adding error, and the\n  family table above "
          f"says which family it is in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
