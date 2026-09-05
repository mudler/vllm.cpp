#!/usr/bin/env python3
"""Per-step cross-tier logit delta, against each tier's own decision margin.

ROCM-TIER-DIVERGENCE (#2590).
Spec: .agents/specs/rocm-tier-hidden-state-bisect.md section 6, third reading.

Reads the `VT_DUMP_LOGITS` dumps (#2534) two tiers wrote for the same prompt and
reports, per decode step, how far apart their pre-sampler logits are AND how far
each tier's own top-1 sits above its own top-2.

WHY BOTH NUMBERS. A cross-tier delta decides a token only when it is of the size
of the margin it has to cross. "The tiers differ by 0.5 in logits" is neither
large nor small until it sits beside a margin of 0.13. Reporting the delta alone
is the mistake this file exists to prevent, so the margin is not optional output.

INPUT is the layout `job/oracle_logits.cpp` and the runner's dump both write: one
`ours_<req>.f32` per sequence, steps concatenated, `n_steps * n_vocab`
little-endian f32, beside an `ours_<req>.ids.txt` sidecar of `step argmax` lines.
The vocabulary size is derived from the sidecar's line count rather than assumed,
and a file whose length is not a whole multiple of it is refused: a dump silently
read at the wrong vocab produces a plausible table over reinterpreted bytes.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def die(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load(directory: str, label: str):
    f32 = sorted(glob.glob(os.path.join(directory, "ours_*.f32")))
    ids = sorted(glob.glob(os.path.join(directory, "ours_*.ids.txt")))
    if not f32:
        die(f"arm {label}: no ours_*.f32 in {directory!r}. A dump that wrote "
            f"nothing looks exactly like a dump that was switched off.")
    if not ids:
        die(f"arm {label}: no ours_*.ids.txt sidecar in {directory!r}; without it "
            f"the vocabulary size and the alignment are both unverifiable.")
    with open(ids[0], encoding="utf-8") as fh:
        sidecar = [int(l.split()[1]) for l in fh if l.strip()]
    raw = np.fromfile(f32[0], dtype="<f4")
    if not sidecar:
        die(f"arm {label}: the ids sidecar is empty")
    if raw.size % len(sidecar) != 0:
        die(f"arm {label}: {os.path.basename(f32[0])} holds {raw.size} floats, "
            f"which is not a whole multiple of the sidecar's {len(sidecar)} "
            f"steps; the vocabulary size cannot be derived and reading it at a "
            f"guessed one would produce a plausible table over reinterpreted "
            f"bytes")
    vocab = raw.size // len(sidecar)
    return raw.reshape(len(sidecar), vocab), sidecar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--steps", type=int, default=None,
                    help="print only the first N steps (all are still compared)")
    ap.add_argument("--last-comparable-step", type=int, default=None,
                    help="the last step at which both tiers were fed the same "
                         "prefix. AFTER the first divergence they are not, so "
                         "every later step compares two different contexts and "
                         "belongs in no statistic. Rows past it are still "
                         "printed, and marked, so the exclusion is visible.")
    args = ap.parse_args()

    a, ida = load(args.a, args.label_a)
    b, idb = load(args.b, args.label_b)
    if a.shape[1] != b.shape[1]:
        die(f"vocabulary sizes differ: {a.shape[1]} on {args.label_a} against "
            f"{b.shape[1]} on {args.label_b}")
    n = min(len(a), len(b))
    last = n - 1 if args.last_comparable_step is None else args.last_comparable_step

    print("== tier-logit-delta ==")
    print(f"A = {args.label_a}  dir={args.a}  steps={len(a)}")
    print(f"B = {args.label_b}  dir={args.b}  steps={len(b)}")
    print(f"vocab={a.shape[1]} (derived from the ids sidecar); comparing {n} steps")
    print(f"delta is A minus B; `margin` is each tier's own top1 - top2, which is "
          f"what its argmax\nactually turned on -- a delta matters only against it")
    print()
    print(f"{'step':>5} {'max_abs':>12} {'rms':>12} {'am_' + args.label_a:>12} "
          f"{'am_' + args.label_b:>12} {'margin_' + args.label_a:>14} "
          f"{'margin_' + args.label_b:>14} {'flip':>5}")
    flips = []
    margins = {}
    deltas = {}
    for s in range(n):
        ra = a[s].astype(np.float64)
        rb = b[s].astype(np.float64)
        d = ra - rb
        ia = np.argpartition(-ra, 1)[:2]
        ia = ia[np.argsort(-ra[ia])]
        ib = np.argpartition(-rb, 1)[:2]
        ib = ib[np.argsort(-rb[ib])]
        ma = float(ra[ia[0]] - ra[ia[1]])
        mb = float(rb[ib[0]] - rb[ib[1]])
        margins[s] = (ma, mb, min(ma, mb))
        deltas[s] = float(np.abs(d).max())
        flip = int(ia[0]) != int(ib[0])
        if flip:
            flips.append((s, int(ia[0]), int(ib[0])))
        if args.steps is None or s < args.steps or flip:
            print(f"{s:>5} {np.abs(d).max():12.6e} {float(np.sqrt((d * d).mean())):12.6e} "
                  f"{int(ia[0]):>12} {int(ib[0]):>12} "
                  f"{ra[ia[0]] - ra[ia[1]]:14.6f} {rb[ib[0]] - rb[ib[1]]:14.6f} "
                  f"{'YES' if flip else '':>5}"
                  + ("   (post-divergence: NOT comparable)" if s > last else ""))
    print()
    print(f"argmax flips: {len(flips)} of {n}"
          + ("" if not flips else "  " + ", ".join(
              f"step {s}: {x} vs {y}" for s, x, y in flips)))

    # THE CALIBRATION, and the only line here that answers a question. A step is
    # AT RISK when the smaller of the two tiers' own margins is below that step's
    # cross-tier max_abs delta: the tiers disagree by more than the decision was
    # won by. A flip outside that set would mean something other than the delta
    # moved the argmax; a step inside it that did not flip is ordinary, because
    # the delta lands on 248320 logits and only sometimes on the right two.
    comparable = [s for s in range(n) if s <= last]
    at_risk = {s for s in comparable if margins[s][2] < deltas[s]}
    flipped = {s for s, _, _ in flips if s <= last}
    print(f"comparable steps: {len(comparable)} of {n}"
          + ("" if last == n - 1 else f" (0..{last}; the rest are post-divergence)"))
    print(f"steps AT RISK (min margin < that step's max_abs delta): "
          f"{len(at_risk)} of {len(comparable)}")
    print(f"  of those, flipped: {len(at_risk & flipped)}")
    outside = sorted(flipped - at_risk)
    print(f"  flips OUTSIDE the at-risk set: {len(outside)}"
          + ("" if not outside else f" -- steps {outside}; a flip the delta "
             f"cannot account for is a finding, not a rounding"))

    # The ALIGNMENT control the #2534 spec requires, restated here because a
    # delta between two dumps that describe different contexts is meaningless.
    bad = [i for i in range(min(len(ida), len(idb), n))
           if ida[i] != int(np.argmax(a[i]))]
    print(f"alignment {args.label_a}: dumped argmax == sidecar on "
          f"{n - len(bad)} of {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
