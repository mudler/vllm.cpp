#!/usr/bin/env python3
"""Compare two `VT_Q4EXP_LAYER_FP` fingerprints without collapsing the layer axis.

WHY THIS TOOL EXISTS (#2877). The differ inlined in
`docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904/run2-job.sh` split
each token on `'='` to build its key. The tap prints the layer as `L%+03lld` --
`L+00`, with NO `=` -- so `f.get('L')` was `None` on every row, the key collapsed
to `(step, None, tag)`, and `if key in rows: continue` kept only the FIRST
occurrence. The instrument printed 437 taps per step and 1311 over three steps;
the comparator compared 42, which is the 14 distinct tags times 3 steps. Layers
1..47 were discarded in silence, and the summary's `L00` label was a hardcoded
string rather than a field read from the row.

That job script is EVIDENCE and is left byte-for-byte as the record of what ran.
This is the tool a later wave should use instead.

THREE THINGS IT DOES THAT THE INLINE ONE DID NOT.

1. It PARSES the layer, and it REFUSES a duplicate key rather than deduplicating
   one. A repeated `(step, layer, tag)` means the format changed or the parse is
   wrong; silently keeping the first is how 1311 taps read as 42.

2. It asserts its own COUNTED PROPERTY. `taps=N END` closes every fingerprinted
   step, and N is CUMULATIVE -- one counter that `LayerFpEndStep` never resets --
   so the rows parsed for steps 0..s must equal the N that step s declares. A
   three-step run of a 437-tap forward prints 437, 874, 1311. A comparator that
   parsed nothing prints an empty table, and an empty table reads like two
   agreeing arms. Rows past the last `END` are refused separately: a capture cut
   off mid-step is short in a way the cumulative count cannot see.

3. It says what `rel(sumabs)` IS WORTH, on every run, in the output.
   `rel(sumabs) = | S|a| - S|b| | / max` is a DIFFERENCE OF NORMS, not a norm of
   differences. Its zero means "the two tensors have equal L1 norm", not "the two
   tensors are equal", and it is not monotone in divergence. `S|x|` is
   sign-INSENSITIVE, so a zero-mean perturbation -- which is every reassociation
   and rounding difference -- cancels at `O(sqrt(n))`.

   THE UNDER-REPORT IS A DISTRIBUTION, NOT A CONSTANT. The first version of this
   file quoted one seed draw ("122.7x at sigma 1e-3, 229.8x at sigma 1e-4") to
   four significant figures. At this tap's real size (`n = 12800`), over the 400
   seeds of `MetricSpread` in `tests/scripts/test_q4exp_layerfp_diff.py`, which
   asserts every figure in this docstring:

     perturbation             p05    median      p95
     sign(a)-aligned         1.00      1.00     1.00   <- the positive control
     zero-mean, sigma 1e-3     34        75      770
     zero-mean, sigma 1e-4     48       140     1500
     zero-mean, sigma 1e-5     48       140     1300

   TWO SIGNIFICANT FIGURES IS WHAT 400 DRAWS BUY, and the control carries the
   demonstration: the next disjoint 400 seeds give a median of 80 rather than 75
   and a p95 of 574 rather than 770. Read the p95 column as an order of
   magnitude. There is no sigma dependence in the linear regime -- 1e-4 and 1e-5
   agree, and both approach `sqrt(2n/pi)/|z| = 90.3/|z|` for a standard normal
   `z`, median 134. `sqrt(n) = 113` is not "the observed factor"; nothing
   observed equals it.

   WHAT THIS COSTS A READER IS THE SPREAD, NOT THE MEDIAN. Hold the TRUE
   divergence FIXED at sigma 1e-3 and vary only the perturbation's sign structure:
   `rel(sumabs)` spans 21x p05..p95 while the true divergence spans 1.06x. Its
   end-to-end span is not a figure at all -- one draw sets it, and it moves by
   more than 4x between seed blocks. Two readings OF THE SAME true divergence
   differ by a median 2.1x, by 4.2x at p75, 11x at p90 and 24x at p95. So a ratio
   between two `rel(sumabs)` numbers is worth what that says and no more: NO
   CHANGE AT ALL produces a ratio at least as large as 1.80x, 2.02x, 2.34x and
   3.15x in 59%, 52%, 45% and 33% of draws, and one at least as large as 16.7x or
   19.9x in 7% and 6%. This file first called the first four "the 59th, 52nd,
   45th and 33rd percentile", which states the complement and inverts the
   ranking: 3.15x sits at the 67th percentile of no change, not the 33rd -- it is
   the LEAST ordinary of the four, and still an ordinary reading. Whole percent
   is the last digit 400 draws hold, and the tail rows move by up to a factor of
   two between seed blocks.

   THE BOUND IS MODEL-DEPENDENT, AND IT DOES NOT FAIL CONSERVATIVELY. The table
   models the perturbation as i.i.d. zero-mean against a Gaussian signal, which is
   the premise under which the metric is being used. AT SIGMA 1e-3, held at the
   same total perturbation energy, a MULTIPLICATIVE rounding-like perturbation is
   read WORSE on every column -- median under-report 110x, pair p95 41x,
   `P(no change >= 19.9x)` 8%, against the dense model's 75x, 24x and 6% -- while
   a SPARSE one, 16 elements of 12800 like a top-k flip, is read almost in full at
   a median 2.9x.

   ONE OF THOSE THREE COLUMNS REVERSES BELOW SIGMA 1e-3, AND THIS ROW READS BELOW
   IT. The multiplicative model is scale-INVARIANT -- 110x at 1e-3, 1e-4 and 1e-5
   alike -- while the dense one loses the second-order term that holds its
   denominator up as sigma falls, and reaches 140x. So in the LINEAR regime the
   DENSE model is the one read worse on magnitude, which is the opposite of the
   sigma 1e-3 ordering; the two SPREAD columns do not reverse, and multiplicative
   stays worse on both at every sigma. The regime is not academic here: this
   control's median `rel(sumabs)` is 1.9e-05 at sigma 1e-4 and 1.9e-06 at sigma
   1e-5, and the `1.772e-05` and `1.062e-06` the row's own reading argues over
   are at or below the first of those -- the LINEAR regime, on the reversed side
   of this table -- while sigma 1e-3 draws 3.5e-04. Read "multiplicative is
   worse" as a sigma 1e-3 statement. Which of the three the real tensors carry is
   UNMEASURED. This bounds the METRIC's resolution and is not a significance test
   on the real tensors.

   So this tool also prints `head_dmax`, the exact elementwise
   `max|v_i(a) - v_i(b)|` over the four `v=` values the tap already emits. That is
   a genuine DIFFERENCE norm and it cannot cancel -- but it samples 4 of 12800
   elements, so it is a floor and a witness, never a magnitude. A non-zero
   `head_dmax` PROVES the tensors differ; a zero one proves nothing.

   A per-arm axis that estimates `||a-b||` over the WHOLE tensor needs the tap to
   emit a sign-SENSITIVE aggregate -- fixed-seed random projections `S w_i x_i`,
   whose difference `S w_i (a_i - b_i)` is a linear functional of the difference
   itself. That is a change to `LayerFp` in the model forward path, so it needs its
   own spec, red-first test and fresh review. It is OWED, not done here (#2877).
"""

from __future__ import annotations

import argparse
import re
import sys

# `q4fp step=%lld L%+03lld tag=%-10s dtype=%-4s dev=%d n=%lld nonfinite=%lld
#  maxabs=%.9g sumabs=%.9g v=%.9g,%.9g,%.9g,%.9g`
_LAYER = re.compile(r"^L[+-]\d+$")


class ParseError(RuntimeError):
    pass


def parse(path):
    """Return (rows, order, taps_declared). Refuses a collapsed or duplicate key."""
    rows, order, declared = {}, [], {}
    layer_seen = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, ln in enumerate(fh, 1):
            if not ln.startswith("q4fp "):
                continue
            toks = ln.split()
            if " taps=" in ln and ln.rstrip().endswith("END"):
                f = dict(t.split("=", 1) for t in toks if "=" in t)
                declared[f["step"]] = int(f["taps"])
                continue
            f = {}
            layer = None
            for t in toks:
                if _LAYER.match(t):
                    layer = int(t[1:])
                    layer_seen += 1
                elif "=" in t:
                    k, v = t.split("=", 1)
                    f[k] = v
            if "step" not in f or "tag" not in f:
                continue
            if layer is None:
                raise ParseError(
                    "%s:%d: no L<layer> field on a tap line. The tap prints "
                    "`L%%+03lld`; a parser that splits on '=' never sees it, and "
                    "that is the #2877 defect this tool exists to refuse."
                    % (path, lineno))
            key = (int(f["step"]), layer, f["tag"])
            if key in rows:
                raise ParseError(
                    "%s:%d: duplicate key %r. A repeated (step, layer, tag) means "
                    "the format changed or the parse is wrong. Deduplicating it "
                    "silently is what made 1311 taps read as 42 (#2877)."
                    % (path, lineno, key))
            rows[key] = f
            order.append(key)
    if not rows:
        raise ParseError("%s: parsed ZERO tap rows. An empty comparison reads "
                         "like two agreeing arms; it is not one." % path)
    if layer_seen != len(rows):
        raise ParseError("%s: %d rows but %d layer fields." % (path, len(rows), layer_seen))
    return rows, order, declared


def check_counted_property(path, rows, declared):
    """`taps=N END` closes each step, and N is a RUNNING TOTAL, not a per-step count.

    `LayerFp` does `++s.taps` on a counter that `LayerFpEndStep` never resets
    (`src/vllm/model_executor/models/qwen4_exp_forward.cpp:153`), so a real
    three-step fingerprint of a 437-tap forward prints `taps=437`, `taps=874`,
    `taps=1311` -- 437 taps EACH, declared cumulatively. Reading N as a per-step
    count refuses every genuine run at step 1, which is what the first version of
    this file did: its committed tests all fingerprint ONE step, where cumulative
    and per-step are the same number, so nothing could fail.

    Returns `(step, declared_cumulative, parsed_cumulative, ok)` per closed step.
    The single-step case still reads `(0, N, N, True)`.
    """
    out = []
    seen = 0
    for step in sorted(declared, key=int):
        s = int(step)
        seen += sum(1 for k in rows if k[0] == s)
        out.append((s, declared[step], seen, seen == declared[step]))
    return out


def unclosed_steps(rows, declared):
    """Steps that printed taps and no `taps=N END`: a truncated capture.

    The cumulative check above can only speak for steps the instrument closed.
    Rows past the last `END` are invisible to it, and a capture cut off mid-step
    is exactly the case where an empty or short comparison reads like agreement.
    """
    closed = {int(k) for k in declared}
    return sorted({k[0] for k in rows} - closed)


def rel_sumabs(a, b):
    """DIFFERENCE OF NORMS. Read the module docstring before using this number."""
    a, b = float(a), float(b)
    m = max(abs(a), abs(b))
    return 0.0 if m == 0.0 else abs(a - b) / m


def head_dmax(a, b):
    """Exact elementwise max|a_i - b_i| over the 4 `v=` values. A witness, not a magnitude."""
    try:
        va = [float(x) for x in a["v"].split(",")]
        vb = [float(x) for x in b["v"].split(",")]
    except (KeyError, ValueError):
        return None
    if len(va) != len(vb):
        return None
    return max(abs(x - y) for x, y in zip(va, vb))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--label", default="BASE vs OTHER")
    ap.add_argument("--step", type=int, default=None, help="print only this step")
    ap.add_argument("--tag", default=None, help="print only this tag")
    ap.add_argument("--top", type=int, default=20, help="rows to print (0 = all)")
    a = ap.parse_args(argv)

    try:
        A, order, dA = parse(a.base)
        B, _, dB = parse(a.other)
    except ParseError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2

    print("=== %s ===" % a.label)
    print("rows: base=%d other=%d   distinct layers: base=%d other=%d"
          % (len(A), len(B), len({k[1] for k in A}), len({k[1] for k in B})))
    for path, rows, decl in ((a.base, A, dA), (a.other, B, dB)):
        name = path.split("/")[-1]
        prev = 0
        for step, want, got, ok in check_counted_property(path, rows, decl):
            print("COUNTED PROPERTY %-28s step=%d taps=%d (cumulative; +%d this "
                  "step) parsed=%d %s"
                  % (name, step, want, want - prev, got, "OK" if ok else "MISMATCH"))
            prev = want
            if not ok:
                print("REFUSED: the comparator did not parse every tap the "
                      "instrument printed. `taps=` is a RUNNING TOTAL over the "
                      "process (qwen4_exp_forward.cpp:153 never resets it), so "
                      "step %d declares %d taps SINCE STEP 0 and %d parsed."
                      % (step, want, got), file=sys.stderr)
                return 2
        open_steps = unclosed_steps(rows, decl)
        if open_steps:
            print("REFUSED: %s has taps at step(s) %s that no `taps=N END` "
                  "closes. The capture is truncated, and a short comparison "
                  "reads like agreement." % (name, open_steps), file=sys.stderr)
            return 2

    print()
    print("rel_sumabs is a DIFFERENCE OF NORMS: zero means EQUAL L1 NORM, not equal")
    print("tensors. It cancels a zero-mean perturbation at O(sqrt(n)), and by a factor")
    print("that is a DISTRIBUTION, not a constant: at this tap's n=12800 the median")
    print("under-report is 75x (sigma 1e-3) to 140x (sigma 1e-4), p05..p95 34..1500,")
    print("over 400 seeds of MetricSpread in tests/scripts/test_q4exp_layerfp_diff.py.")
    print("READ RATIOS ACCORDINGLY: at a FIXED true divergence two readings differ by a")
    print("median 2.1x, 11x at p90 and 24x at p95, so a ratio in the low TENS between two")
    print("of these numbers ranks nothing, in either direction. head_dmax is an exact")
    print("elementwise difference over the 4 emitted values: non-zero PROVES the tensors")
    print("differ, zero proves nothing. A whole-tensor difference norm is OWED (#2877).")
    print()

    sel = [k for k in order
           if (a.step is None or k[0] == a.step) and (a.tag is None or k[2] == a.tag)]
    scored, missing = [], 0
    for k in sel:
        if k not in B:
            missing += 1
            continue
        scored.append((rel_sumabs(A[k]["sumabs"], B[k]["sumabs"]), k))
    scored.sort(reverse=True)

    print("%-5s %-5s %-10s %-6s %14s %14s %11s %11s"
          % ("step", "L", "tag", "dtype", "sumabs_base", "sumabs_other",
             "rel_sumabs", "head_dmax"))
    for r, k in (scored if a.top == 0 else scored[:a.top]):
        hd = head_dmax(A[k], B[k])
        print("%-5d %-5d %-10s %-6s %14s %14s %11.3e %11s"
              % (k[0], k[1], k[2], A[k]["dtype"], A[k]["sumabs"], B[k]["sumabs"],
                 r, ("%.3e" % hd) if hd is not None else "n/a"))

    nz = sum(1 for r, _ in scored if r > 0.0)
    hz = sum(1 for _, k in scored if (head_dmax(A[k], B[k]) or 0.0) > 0.0)
    print()
    print("--- %s SUMMARY ---" % a.label)
    print("TAPS COMPARED                 : %d  (of %d parsed in base)" % (len(scored), len(A)))
    print("MISSING IN OTHER              : %d" % missing)
    print("TAPS WITH rel(sumabs) != 0    : %d" % nz)
    print("TAPS WITH head_dmax != 0      : %d   <- a DIFFERENCE norm; these PROVE a difference" % hz)
    if scored:
        r, k = scored[0]
        print("LARGEST rel(sumabs)           : %.6e at step=%d L%+03d %s" % (r, k[0], k[1], k[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
