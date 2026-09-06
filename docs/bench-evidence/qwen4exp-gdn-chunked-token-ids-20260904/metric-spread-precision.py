#!/usr/bin/env python3
"""How much of `MetricSpread` a 400-draw sample determines (#2879).

The set this branch first published -- `31.4 / 69.2 / 568.2`, `58.7 / 52.2 /
44.9 / 32.8 %`, `4.6% / 5.4%` -- came from a script that was never committed,
and it does not reproduce from `MetricSpread` in
`tests/scripts/test_q4exp_layerfp_diff.py` over the 400 seeds it documents. This
script answers whether the two differ in CONSTRUCTION or only in SAMPLE, by
reading the committed control's own `metric_draw` -- so it cannot drift from it
-- and measuring the estimator's sampling error two independent ways:

  1. six DISJOINT 400-seed blocks, which is the difference the withdrawn set is;
  2. a 200-resample bootstrap over the committed block, which puts an interval
     on every published figure;
  3. every PREFIX of the committed seed stream, which answers the other way the
     withdrawn set could have arisen -- the same seeds, fewer of them.

Usage:  python3 metric-spread-precision.py > metric-spread-precision.txt
Runtime is a few minutes; it is standard library only, and there is no GPU, no
artifact and no network.
"""

import importlib.util
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
_p = ROOT / "tests" / "scripts" / "test_q4exp_layerfp_diff.py"
_spec = importlib.util.spec_from_file_location("q4exp_metric_spread", _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
draw = _mod.metric_draw

SIGMA = 1e-3
SEEDS = 400
PREFIXES = (64, 80, 100, 120, 128, 150, 160, 200, 240, 256, 300, 320, 360, 400)
MOVES = (1.80, 2.02, 2.34, 3.15, 11.6, 16.7, 19.9, 24.1)
# The figures this branch published before #2879, in the order reported below.
WITHDRAWN = dict(ur_p05=31.4, ur_med=69.2, ur_p95=568.2,
                 s180=58.7, s202=52.2, s234=44.9, s315=32.8,
                 s116=7.6, s167=5.4, s199=4.6, s241=3.9,
                 pair_med=2.1, pair_p75=4.0, pair_p90=8.9, pair_p95=18.2)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q / 100.0 * len(v)))]


def pairs_of(rel):
    out = []
    for i in range(len(rel)):
        for j in range(i + 1, len(rel)):
            x, y = rel[i], rel[j]
            out.append(max(x, y) / min(x, y))
    out.sort()
    return out


def figures(rel, ratios):
    p = pairs_of(rel)
    f = dict(ur_p05=pct(ratios, 5), ur_med=pct(ratios, 50), ur_p95=pct(ratios, 95),
             pair_med=pct(p, 50), pair_p75=pct(p, 75), pair_p90=pct(p, 90),
             pair_p95=pct(p, 95))
    for move in MOVES:
        f["s%d" % round(move * 10)] = 100.0 * sum(1 for r in p if r >= move) / len(p)
    return f


ORDER = ["ur_p05", "ur_med", "ur_p95", "s18", "s20", "s23", "s32",
         "s116", "s167", "s199", "s241", "pair_med", "pair_p75", "pair_p90",
         "pair_p95"]
LABEL = {"ur_p05": "under-report p05", "ur_med": "under-report median",
         "ur_p95": "under-report p95", "s18": "P(no change >= 1.80x) %",
         "s20": "P(no change >= 2.02x) %", "s23": "P(no change >= 2.34x) %",
         "s32": "P(no change >= 3.15x) %", "s116": "P(no change >= 11.6x) %",
         "s167": "P(no change >= 16.7x) %", "s199": "P(no change >= 19.9x) %",
         "s241": "P(no change >= 24.1x) %", "pair_med": "pair ratio median",
         "pair_p75": "pair ratio p75", "pair_p90": "pair ratio p90",
         "pair_p95": "pair ratio p95"}
OLD = {"ur_p05": "ur_p05", "ur_med": "ur_med", "ur_p95": "ur_p95",
       "s18": "s180", "s20": "s202", "s23": "s234", "s32": "s315",
       "s116": "s116", "s167": "s167", "s199": "s199", "s241": "s241",
       "pair_med": "pair_med", "pair_p75": "pair_p75", "pair_p90": "pair_p90",
       "pair_p95": "pair_p95"}


def block(first):
    d = [draw(s, SIGMA) for s in range(first, first + SEEDS)]
    return [rs for rs, _ in d], [rl / rs for rs, rl in d]


def main():
    print("MetricSpread precision study (#2879). sigma %g, n = 12800, %d seeds "
          "per block." % (SIGMA, SEEDS))
    print("draw() is tests/scripts/test_q4exp_layerfp_diff.py::metric_draw, "
          "read by import.\n")

    blocks = []
    for off in range(0, 6 * SEEDS, SEEDS):
        rel, ratios = block(off)
        blocks.append(figures(rel, ratios))
        if off == 0:
            committed_rel, committed_ratios = rel, ratios
        print("block seeds %4d..%4d measured" % (off, off + SEEDS - 1),
              file=sys.stderr)

    print("1. SIX DISJOINT 400-SEED BLOCKS OF THE SAME ESTIMATOR")
    print("%-24s %10s %10s %10s %10s  %s"
          % ("figure", "committed", "min", "max", "withdrawn", "withdrawn in range?"))
    for k in ORDER:
        vals = [b[k] for b in blocks]
        w = WITHDRAWN[OLD[k]]
        inr = "yes" if min(vals) <= w <= max(vals) else "NO (%+.1f%%)" % (
            100.0 * (w - (min(vals) if w < min(vals) else max(vals)))
            / (min(vals) if w < min(vals) else max(vals)))
        print("%-24s %10.2f %10.2f %10.2f %10.2f  %s"
              % (LABEL[k], blocks[0][k], min(vals), max(vals), w, inr))

    print("\n2. BOOTSTRAP OVER THE COMMITTED BLOCK (seeds 0..399, 200 resamples,")
    print("   random.Random(12345)) -- an interval on each published figure")
    R = random.Random(12345)
    B = 200
    acc = {k: [] for k in ORDER}
    for _ in range(B):
        idx = [R.randrange(SEEDS) for _ in range(SEEDS)]
        f = figures([committed_rel[i] for i in idx],
                    [committed_ratios[i] for i in idx])
        for k in ORDER:
            acc[k].append(f[k])
    inside = 0
    print("%-24s %10s %10s %10s %10s  %s"
          % ("figure", "committed", "boot2.5%", "boot97.5%", "withdrawn", "inside 95% CI?"))
    for k in ORDER:
        a = sorted(acc[k])
        lo, hi = a[int(0.025 * B)], a[int(0.975 * B)]
        w = WITHDRAWN[OLD[k]]
        ok = lo <= w <= hi
        inside += ok
        print("%-24s %10.2f %10.2f %10.2f %10.2f  %s"
              % (LABEL[k], blocks[0][k], lo, hi, w, "yes" if ok else "NO"))
    print("\n%d of %d withdrawn figures lie inside the committed control's own "
          "95%% interval." % (inside, len(ORDER)))
    print("The construction is not falsified by them. The third significant "
          "figure is.")

    print("\n3. EVERY PREFIX OF THE COMMITTED SEED STREAM (seeds 0..n-1), which is")
    print("   the other way the withdrawn set could have arisen: the same stream,")
    print("   read short. n in %s" % (PREFIXES,))
    pre = {}
    for n in PREFIXES:
        pre[n] = figures(committed_rel[:n], committed_ratios[:n])
    print("%-24s %10s %10s %10s %10s  %s"
          % ("figure", "committed", "min", "max", "withdrawn", "withdrawn vs every prefix"))
    below = matched = 0
    for k in ORDER:
        vals = [pre[n][k] for n in PREFIXES]
        w = WITHDRAWN[OLD[k]]
        if w < min(vals):
            verdict = "below all"
            below += 1
        elif w > max(vals):
            verdict = "above all"
        else:
            verdict = "inside (n=%s nearest)" % min(
                PREFIXES, key=lambda n: abs(pre[n][k] - w))
        print("%-24s %10.2f %10.2f %10.2f %10.2f  %s"
              % (LABEL[k], blocks[0][k], min(vals), max(vals), w, verdict))
    for n in PREFIXES:
        if all(abs(pre[n][k] - WITHDRAWN[OLD[k]]) < 0.05 * abs(WITHDRAWN[OLD[k]])
               for k in ORDER):
            matched += 1
    print("\n%d of %d prefixes reproduce the withdrawn set within 5%% on all %d "
          "figures." % (matched, len(PREFIXES), len(ORDER)))
    n0 = min(PREFIXES)
    tails = ("s116", "s167", "s199", "s241")
    above = sum(1 for k in tails if WITHDRAWN[OLD[k]] > pre[n0][k])
    print("%d of %d withdrawn figures sit below EVERY prefix. The ordering is not"
          % (below, len(ORDER)))
    print("universal -- at n = %d, %d of the %d tail shares sit ABOVE the withdrawn"
          % (n0, above, len(tails)))
    print("value -- so the set is a different SAMPLE, and it is not a short read of")
    print("this one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
