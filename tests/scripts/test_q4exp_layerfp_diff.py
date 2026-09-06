#!/usr/bin/env python3
"""The layer axis survives the comparator, and the legacy loader is why it did not.

#2877. `scripts/q4exp-layerfp-diff.py` replaces the differ inlined in
`docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904/run2-job.sh`, which
collapsed 1311 tap lines to 42 because it built its key by splitting each token on
`'='` while the tap prints the layer as `L%+03lld` -- `L+00`, with no `=`.

`LegacyLoaderReproduction` runs the committed loader VERBATIM on lines built from
the instrument's exact `printf`, and asserts it collapses. That is the red. It is
the contrast that makes the counted property below mean something: without it, "48
rows" is a number with nothing to fail against.

`RepairedLoader` asserts the counted property -- 48 synthetic per-layer rows load
as 48, with 48 distinct layers -- and that the tool REFUSES rather than
deduplicates when a key really does repeat, and REFUSES a fingerprint whose
`taps=N END` disagrees with the rows parsed.

`MetricHonesty` pins the thing the numbers are worth: `rel_sumabs` is a difference
of NORMS and reads ZERO on two tensors that differ in every element, while
`head_dmax` -- an exact elementwise difference over the emitted values -- does not.

`MetricSpread` says HOW MUCH that costs, and it PUBLISHES. It draws every figure
the documents quote FROM IT over `range(400)`, and its `test_the_PUBLISHER_*`
cases READ
`scripts/q4exp-layerfp-diff.py` (its docstring AND its runtime stdout),
`docs/USAGE.md`, the row's evidence file and the two specs off disk and compare
each figure those documents print against the value drawn here, rendered by the
one rounding function all of them use. So a reader reproduces the documents by
running this file, and a document cannot move a digit without reddening a case
that names the surface and the figure. #2879: the set those documents first
carried came from an ad-hoc script that was never committed and did not reproduce
from this control -- and the control could not see that, because until #2879
every figure it asserted was a literal typed into this file.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import unittest
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "q4exp-layerfp-diff.py"
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

_spec = importlib.util.spec_from_file_location("q4exp_layerfp_diff", TOOL)
diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diff)


def tap_line(step, il, tag, sumabs, v=(0.0, 0.0, 0.0, 0.0)):
    """The instrument's EXACT format, src/vllm/model_executor/models/qwen4_exp_forward.cpp."""
    return ("q4fp step=%d L%+03d tag=%-10s dtype=%-4s dev=%d n=%d "
            "nonfinite=%d maxabs=%.9g sumabs=%.9g v=%.9g,%.9g,%.9g,%.9g\n"
            % (step, il, tag, "bf16", 0, 12800, 0, 1.0, sumabs, v[0], v[1], v[2], v[3]))


def fingerprint(path, layers=48, tags=("blk",), sumabs=lambda il, tag: 100.0, v=None,
                steps=1):
    """Write `steps` fingerprinted forwards, exactly as the instrument prints them.

    `taps=` is CUMULATIVE. `LayerFp` does `++s.taps` on a counter that
    `LayerFpEndStep` never resets (`qwen4_exp_forward.cpp:153`), so a real
    three-step run of a 437-tap forward closes its steps with `taps=437`,
    `taps=874`, `taps=1311` -- which is what the committed
    `run2-results.txt` records. `steps=1` is the degenerate case where cumulative
    and per-step are the SAME number, and it is the only case the first version of
    this file could express.
    """
    lines, taps = [], 0
    for step in range(steps):
        for il in range(layers):
            for tag in tags:
                vv = v(il, tag) if v else (0.0, 0.0, 0.0, 0.0)
                lines.append(tap_line(step, il, tag, sumabs(il, tag), vv))
                taps += 1
        lines.append("q4fp step=%d taps=%d END\n" % (step, taps))
    path.write_text("".join(lines), encoding="utf-8")
    return taps


# The loader as committed in run2-job.sh, copied VERBATIM. Do not repair it here:
# its whole job in this file is to fail. Its unclosed `open()` is part of what was
# committed, so the ResourceWarning is filtered rather than the line changed.
warnings.filterwarnings("ignore", category=ResourceWarning)


def legacy_load(p):
    rows, order = {}, []
    for ln in open(p):
        if not ln.startswith('q4fp ') or ' taps=' in ln:
            continue
        f = {}
        for tok in ln.split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                f[k] = v
        key = (f.get('step'), f.get('L'), f.get('tag'))
        if key in rows:
            continue
        rows[key] = f
        order.append(key)
    return rows, order


class LegacyLoaderReproduction(unittest.TestCase):
    """THE RED: the committed loader collapses 48 layers to 1, silently."""

    def test_legacy_collapses_the_layer_axis(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            taps = fingerprint(p, layers=48, sumabs=lambda il, tag: 100.0 + il)
            self.assertEqual(taps, 48)
            rows, order = legacy_load(p)
            self.assertEqual(len(rows), 1, "the #2877 defect: 48 taps must collapse to 1")
            self.assertEqual(order[0][1], None, "the layer field must never parse")
            self.assertEqual(float(rows[order[0]]["sumabs"]), 100.0,
                             "first-wins dedup keeps layer 0 and drops 1..47")

    def test_legacy_probe_form_can_report_48(self):
        """POSITIVE CONTROL: the same loader on an `L=` spelling gives 48 rows,
        so the 1 above is the defect and not a dead probe."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text("".join(
                "q4fp step=0 L=%+03d tag=blk dtype=bf16 dev=0 n=1 nonfinite=0 "
                "maxabs=1 sumabs=%.9g v=0,0,0,0\n" % (il, 100.0 + il)
                for il in range(48)), encoding="utf-8")
            rows, _ = legacy_load(p)
            self.assertEqual(len(rows), 48)


class RepairedLoader(unittest.TestCase):
    """THE GREEN, as a COUNTED PROPERTY: 48 rows and 48 distinct layers."""

    def test_counted_property_48_rows_48_layers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            fingerprint(p, layers=48, sumabs=lambda il, tag: 100.0 + il)
            rows, order, declared = diff.parse(p)
            self.assertEqual(len(rows), 48)
            self.assertEqual(len({k[1] for k in rows}), 48)
            self.assertEqual(declared["0"], 48)
            checks = diff.check_counted_property(str(p), rows, declared)
            self.assertEqual(checks, [(0, 48, 48, True)])

    def test_refuses_a_duplicate_key_instead_of_deduplicating(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text(tap_line(0, 0, "blk", 1.0) + tap_line(0, 0, "blk", 2.0),
                         encoding="utf-8")
            with self.assertRaises(diff.ParseError) as cm:
                diff.parse(p)
            self.assertIn("duplicate key", str(cm.exception))

    def test_refuses_a_line_with_no_layer_field(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text("q4fp step=0 tag=blk dtype=bf16 dev=0 n=1 nonfinite=0 "
                         "maxabs=1 sumabs=1 v=0,0,0,0\n", encoding="utf-8")
            with self.assertRaises(diff.ParseError) as cm:
                diff.parse(p)
            self.assertIn("no L<layer> field", str(cm.exception))

    def test_refuses_an_empty_parse(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text("nothing here\n", encoding="utf-8")
            with self.assertRaises(diff.ParseError) as cm:
                diff.parse(p)
            self.assertIn("ZERO tap rows", str(cm.exception))

    def test_cli_compares_every_layer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=48, sumabs=lambda il, tag: 100.0 + il)
            fingerprint(b, layers=48, sumabs=lambda il, tag: 100.0 + il + 0.001)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b), "--top", "0"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("TAPS COMPARED                 : 48", r.stdout)
            self.assertIn("distinct layers: base=48 other=48", r.stdout)
            self.assertIn("DIFFERENCE OF NORMS", r.stdout)

    def test_cli_refuses_a_counted_property_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=48, sumabs=lambda il, tag: 1.0)
            # 48 rows but the instrument declares 60: the comparator must refuse.
            b.write_text("".join(tap_line(0, il, "blk", 1.0) for il in range(48))
                         + "q4fp step=0 taps=60 END\n", encoding="utf-8")
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("did not parse every tap", r.stderr)


class CumulativeTapCounter(unittest.TestCase):
    """`taps=N END` is a RUNNING TOTAL, and reading it per-step refuses real data.

    THE RED THIS CLASS EXISTS FOR. Every case above fingerprints ONE step, where
    the cumulative count and the per-step count are the same number, so the first
    version of the tool -- which subtracted nothing -- passed all eleven of them
    and still exited 2 on every genuine multi-step run. The committed evidence
    reads `q4fp step=0 taps=437 END q4fp step=1 taps=874 END q4fp step=2
    taps=1311 END` (`run2-results.txt:12`), and the tool refused it.

    The instrument is the authority for the shape, not this file:
    `src/vllm/model_executor/models/qwen4_exp_forward.cpp:153` increments
    `s.taps` per tap and `LayerFpEndStep` prints it without resetting.
    """

    def _three_steps(self, td, name, bump=0.0):
        p = pathlib.Path(td) / name
        fingerprint(p, layers=3, steps=3, sumabs=lambda il, tag: 100.0 + il + bump)
        return p

    def test_the_evidence_shape_is_cumulative_not_per_step(self):
        """The fixture reproduces 437/874/1311's ARITHMETIC at 3 layers x 3 steps."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._three_steps(td, "fp.txt")
            declared = [ln.split("taps=")[1].split()[0]
                        for ln in p.read_text(encoding="utf-8").splitlines()
                        if " taps=" in ln]
            self.assertEqual(declared, ["3", "6", "9"],
                             "the instrument declares a running total, not 3,3,3")

    def test_counted_property_accepts_a_real_multi_step_fingerprint(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = self._three_steps(td, "fp.txt")
            rows, _, declared = diff.parse(p)
            self.assertEqual(len(rows), 9)
            checks = diff.check_counted_property(str(p), rows, declared)
            self.assertEqual(checks, [(0, 3, 3, True), (1, 6, 6, True), (2, 9, 9, True)],
                             "step 1 declares 6 taps SINCE STEP 0, and 6 have been parsed")

    def test_cli_does_not_refuse_a_real_multi_step_fingerprint(self):
        """THE WHOLE FINDING: the tool exited 2 on every genuine run."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = self._three_steps(td, "a.txt")
            b = self._three_steps(td, "b.txt", bump=0.001)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b), "--top", "0"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("did not parse every tap", r.stderr)
            self.assertIn("TAPS COMPARED                 : 9", r.stdout)
            self.assertIn("(cumulative; +3 this step)", r.stdout)

    def test_a_short_step_is_still_refused_under_the_cumulative_reading(self):
        """The repair must not become 'accept anything'. A missing tap still reds."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = self._three_steps(td, "a.txt")
            b = pathlib.Path(td) / "b.txt"
            # Steps 0 and 2 complete; step 1 prints two of its three taps.
            lines = []
            for step, present in ((0, 3), (1, 2), (2, 3)):
                for il in range(present):
                    lines.append(tap_line(step, il, "blk", 100.0 + il))
                lines.append("q4fp step=%d taps=%d END\n" % (step, 3 * (step + 1)))
            b.write_text("".join(lines), encoding="utf-8")
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("did not parse every tap", r.stderr)
            self.assertIn("RUNNING TOTAL", r.stderr)

    def test_taps_past_the_last_END_are_refused_as_a_truncated_capture(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "fp.txt"
            p.write_text(tap_line(0, 0, "blk", 1.0)
                         + "q4fp step=0 taps=1 END\n"
                         + tap_line(1, 0, "blk", 1.0), encoding="utf-8")
            rows, _, declared = diff.parse(p)
            self.assertEqual(diff.unclosed_steps(rows, declared), [1])
            r = subprocess.run([sys.executable, str(TOOL), str(p), str(p)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("truncated", r.stderr)


def sig(x, n=2):
    """Round to `n` significant figures.

    Every figure `MetricSpread` publishes is a Monte Carlo estimate over
    `MetricSpread.SEEDS` draws, and
    `test_a_second_seed_block_moves_every_figure_a_third_digit_would_claim`
    measures what that many draws determine. It is not a third digit.
    """
    import math
    if x == 0.0:
        return 0.0
    return round(x, -(int(math.ceil(math.log10(abs(x)))) - n))


def published(x, digits=2, decimals=0, sci=False):
    """Render a drawn value EXACTLY as the documents round it. ONE function.

    The documents quote two significant figures, because that is what
    `MetricSpread.SEEDS` draws determine. `digits=None` is the escape for the few
    figures a document quotes RAW as a CONTRAST -- the second seed block's `80`
    against `75`, the whole-percent shares, and the closed-form asymptotes --
    where sig-rounding would erase the difference the figure is quoted to show.

    The `test_the_PUBLISHER_*` cases render through this and compare the result to
    what each surface prints, so a document and this control cannot move apart. A
    rendered string is typed below only in an ESTIMATOR pin -- the `want =` tuples,
    and the three `rel(sumabs)` medians at the end of
    `test_the_MAGNITUDE_column_of_that_table_REVERSES_below_sigma_1e_3` -- which
    reds when the draw moves. No document-site assertion path carries one, and a
    copy there is what would be #2879 again, one level up.
    """
    v = float(x) if digits is None else sig(x, digits)
    return ("%.*e" if sci else "%.*f") % (decimals, v)


_DRAWS = {}


def metric_draw(seed, sigma, mode="dense", n=12800, sigma_a=0.0382):
    """One draw of the model under which `rel(sumabs)` is being read.

    Returns `(rel_sumabs, true_l1)`, both divided by the same `max(S|a|, S|b|)`,
    so a RATIO between them is independent of that denominator.

    `mode` selects the perturbation, and it is the load-bearing choice -- see
    `test_the_bound_is_MODEL_DEPENDENT_and_this_says_which_way`. All four hold
    the same total perturbation energy (RMS `sigma`) apart from `aligned`, which
    is the positive control:

      aligned   `e_i = sign(a_i) * |N(0, sigma)|` -- cannot cancel, so the two
                measures must agree and the ratio must read 1.
      dense     `e_i ~ N(0, sigma)` i.i.d. -- a reassociation or rounding
                difference spread over every element.
      mult      `e_i = a_i * N(0, sigma/sigma_a)` -- rounding-like, proportional
                to the value it perturbs.
      sparse    16 of `n` elements carry the whole perturbation,
                `e_i ~ N(0, sigma*sqrt(n/16))` -- like a top-k flip.

    Cached, because the published table, the pair statistics and the
    model-dependence table all read the same draws.
    """
    import math
    import random
    key = (seed, sigma, mode, n, sigma_a)
    hit = _DRAWS.get(key)
    if hit is not None:
        return hit
    r = random.Random(seed)
    a = [r.gauss(0.0, sigma_a) for _ in range(n)]
    if mode == "aligned":
        e = [math.copysign(abs(r.gauss(0.0, sigma)), x) for x in a]
    elif mode == "dense":
        e = [r.gauss(0.0, sigma) for _ in range(n)]
    elif mode == "mult":
        k = sigma / sigma_a
        e = [x * r.gauss(0.0, k) for x in a]
    elif mode == "sparse":
        nz = 16
        c = sigma * math.sqrt(float(n) / nz)
        e = [0.0] * n
        for i in r.sample(range(n), nz):
            e[i] = r.gauss(0.0, c)
    else:
        raise ValueError("unknown perturbation mode %r" % (mode,))
    sa = sum(abs(x) for x in a)
    sb = sum(abs(x + y) for x, y in zip(a, e))
    m = max(sa, sb)
    # (the committed metric, the honest one)
    out = (abs(sa - sb) / m, sum(abs(y) for y in e) / m)
    _DRAWS[key] = out
    return out


class MetricSpread(unittest.TestCase):
    """WHAT A RATIO BETWEEN TWO `rel(sumabs)` NUMBERS IS WORTH, AND WHO SAYS SO.

    The first version of this file's tool quoted ONE seed draw -- "122.7x at
    sigma 1e-3 and 229.8x at sigma 1e-4" -- to four significant figures. It is a
    distribution, its median is not those numbers, and its spread is the part
    that decides whether any ratio in the #2877 reading means anything.

    THIS CLASS IS THE PUBLISHER, IN TWO LAYERS. The `test_the_PUBLISHED_*` cases
    pin what this ESTIMATOR draws over `range(SEEDS)`. The `test_the_PUBLISHER_*`
    cases then read `scripts/q4exp-layerfp-diff.py` -- its docstring and the
    stdout of an actual run -- `docs/USAGE.md`,
    `docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md`, the two
    specs and this file's own docstrings OFF DISK, and compare every figure those
    surfaces quote FROM THIS CONTROL against the drawn value, rendered by
    `published()`, the one rounding function. What they do not cover is a figure
    no draw here produces -- a measured `rel(sumabs)` off the real run, or a count
    from the committed precision study. A reader reproduces the documents by
    running this file, and neither side can move without the other.

    THE SECOND LAYER IS WHAT #2879 WAS ABOUT, AND IT DID NOT EXIST UNTIL #2879.
    The set this branch first published -- `31.4 / 69.2 / 568.2`, `58.7 / 52.2 /
    44.9 / 32.8 %`, `4.6% / 5.4%` -- came from an ad-hoc script that was never
    committed, and it does not reproduce here. Every one of those figures is a
    valid draw of THIS estimator (a disjoint 400-seed block of this same control
    gives `32.5 / 80.3 / 573.8` and `3.7% / 4.5%`); what was not valid was
    quoting three and four significant figures for it. The digits below are
    rounded to two. The case that would have caught the difference is the
    exact-equality publisher, and only because it reads the documents:
    `test_a_second_seed_block_moves_every_figure_a_third_digit_would_claim`
    measures the estimator's imprecision and compares it to NO document, so it
    could not have caught a document quoting a sample this control never drew.

    Hermetic: standard library only, fixed seeds, no artifact and no GPU. It
    models the perturbation against a Gaussian signal at the committed
    `L00 moe` scale (`n = 12800`, `sum|x| ~ 390`), which is the premise under
    which `rel(sumabs)` is being read. It bounds the METRIC's resolution. It is
    not a significance test on the real tensors.
    """

    N = 12800
    SIGMA_A = 0.0382  # sum|a| ~ 390 over n = 12800
    SEEDS = 400       # the seed set EVERY published figure is drawn over
    SIGMA = 1e-3      # the sigma every published SPREAD figure is drawn at

    def _draw(self, seed, sigma, mode="dense"):
        return metric_draw(seed, sigma, mode, self.N, self.SIGMA_A)

    def _rel(self, mode="dense", sigma=None, first=0, count=None):
        sigma = self.SIGMA if sigma is None else sigma
        count = self.SEEDS if count is None else count
        return [self._draw(s, sigma, mode)[0]
                for s in range(first, first + count)]

    @staticmethod
    def _pairs(rel):
        """Every unordered pair of readings OF THE SAME true divergence."""
        out = []
        for i in range(len(rel)):
            for j in range(i + 1, len(rel)):
                x, y = rel[i], rel[j]
                out.append(max(x, y) / min(x, y))
        out.sort()
        return out

    @staticmethod
    def _share(pairs, move):
        return 100.0 * sum(1 for r in pairs if r >= move) / len(pairs)

    @staticmethod
    def _pct(values, q):
        v = sorted(values)
        return v[min(len(v) - 1, int(q / 100.0 * len(v)))]

    def test_positive_control_a_sign_aligned_perturbation_reads_1x(self):
        """Where the two measures MUST agree they do, so the gap below is real."""
        for seed in range(8):
            rs, rl = self._draw(seed, 1e-3, "aligned")
            self.assertAlmostEqual(rl / rs, 1.0, places=6)

    def test_the_under_report_is_a_distribution_and_not_122x(self):
        draws = [self._draw(s, 1e-3) for s in range(64)]
        ratios = sorted(rl / rs for rs, rl in draws)
        median = ratios[len(ratios) // 2]
        self.assertGreater(median, 20.0, "a zero-mean perturbation is heavily under-read")
        self.assertLess(median, 400.0)
        # The point of the case: p05..p95 spans an order of magnitude, so no
        # single figure -- 122.7x, 229.8x or this median -- is a constant.
        self.assertGreater(self._pct(ratios, 95) / self._pct(ratios, 5), 10.0)

    def test_at_a_FIXED_true_divergence_the_metric_still_spans_an_order_of_magnitude(self):
        draws = [self._draw(s, 1e-3) for s in range(64)]
        rel_sumabs = [rs for rs, _ in draws]
        true_l1 = [rl for _, rl in draws]
        self.assertLess(max(true_l1) / min(true_l1), 1.1,
                        "the TRUE divergence is held fixed across these seeds")
        self.assertGreater(max(rel_sumabs) / min(rel_sumabs), 20.0,
                           "the committed metric is not, on the same divergence")

    def test_a_ratio_of_two_readings_in_the_LOW_TENS_ranks_nothing(self):
        """The consequence for #2877: 1.80x, 2.02x and 3.15x are NO CHANGE.

        Pairs drawn from readings of the SAME true divergence. If the moves the
        reading argues over are ordinary values of this ratio, the reading cannot
        order them -- in either direction.
        """
        pairs = self._pairs(self._rel(count=64))
        for move in (1.80, 2.02, 3.15):
            share = self._share(pairs, move)
            self.assertGreater(share, 20.0,
                               "%.2fx is an ordinary reading of an UNCHANGED "
                               "divergence (%.0f%% of pairs reach it)"
                               % (move, share))
        self.assertGreater(self._pct(pairs, 95), 8.0,
                           "two readings of one divergence differ by ~an order of "
                           "magnitude at p95")

    # ------------------------------------------------------------------
    # THE PUBLISHED FIGURES. Everything below is quoted in the documents.
    # ------------------------------------------------------------------

    def test_the_PUBLISHED_under_report_table_is_what_this_control_draws(self):
        """The table in the tool docstring and the evidence file, cell for cell."""
        want = ((1e-3, "dense", 34.0, 75.0, 770.0),
                (1e-4, "dense", 48.0, 140.0, 1500.0),
                (1e-5, "dense", 48.0, 140.0, 1300.0),
                (1e-3, "aligned", 1.0, 1.0, 1.0))
        for sigma, mode, p05, med, p95 in want:
            draws = [self._draw(s, sigma, mode) for s in range(self.SEEDS)]
            ratios = [rl / rs for rs, rl in draws]
            got = (sig(self._pct(ratios, 5)), sig(self._pct(ratios, 50)),
                   sig(self._pct(ratios, 95)))
            self.assertEqual(got, (p05, med, p95),
                             "sigma %g / %s row of the published table" % (sigma, mode))

    def test_the_PUBLISHED_no_change_shares_are_what_this_control_draws(self):
        """`P(no change produces a ratio at least this large)`, whole percent.

        Whole percent because that is the last digit 400 draws hold: the four
        tail rows move by up to a factor of two between seed blocks, which the
        documents say beside them.
        """
        pairs = self._pairs(self._rel())
        want = ((1.80, 59), (2.02, 52), (2.34, 45), (3.15, 33),
                (11.6, 9), (16.7, 7), (19.9, 6), (24.1, 5))
        for move, share in want:
            self.assertEqual(round(self._share(pairs, move)), share,
                             "P(no change >= %.2fx)" % move)

    def test_the_PUBLISHED_spread_figures_are_what_this_control_draws(self):
        """The pair quantiles and the two spans quoted beside them."""
        rel = self._rel()
        true_l1 = [self._draw(s, self.SIGMA)[1] for s in range(self.SEEDS)]
        pairs = self._pairs(rel)
        self.assertEqual((sig(self._pct(pairs, 50)), sig(self._pct(pairs, 75)),
                          sig(self._pct(pairs, 90)), sig(self._pct(pairs, 95))),
                         (2.1, 4.2, 11.0, 24.0), "pair median / p75 / p90 / p95")
        self.assertEqual(sig(self._pct(rel, 95) / self._pct(rel, 5)), 21.0,
                         "rel(sumabs) p05..p95 span")
        self.assertEqual(round(max(true_l1) / min(true_l1), 2), 1.06,
                         "the TRUE divergence over the same 400 draws")

    def test_a_second_seed_block_moves_every_figure_a_third_digit_would_claim(self):
        """WHY THE DIGITS ABOVE STOP AT TWO. NOT what would have caught #2879.

        This case compares the estimator to ITSELF on a second seed block. It
        measures how much of a figure 400 draws determine, which is why two
        significant figures is the ceiling. It could not have caught #2879,
        because it reads no document: the `test_the_PUBLISHER_*` cases below are
        the ones that do.

        Seeds 400..799 are as valid a 400-draw sample of this estimator as seeds
        0..399. Their figures differ by more than the third digit the branch
        first published, which is the whole reason those digits did not
        reproduce. The end-to-end span of `rel(sumabs)` is worse than imprecise:
        it is set by the single smallest draw and moves by more than 4x between
        blocks, so no digit of it is publishable at all.
        """
        a, b = self._rel(), self._rel(first=self.SEEDS)
        ra = [self._draw(s, self.SIGMA)[1] / self._draw(s, self.SIGMA)[0]
              for s in range(self.SEEDS)]
        rb = [self._draw(s, self.SIGMA)[1] / self._draw(s, self.SIGMA)[0]
              for s in range(self.SEEDS, 2 * self.SEEDS)]
        med_a, med_b = self._pct(ra, 50), self._pct(rb, 50)
        self.assertGreater(max(med_a, med_b) / min(med_a, med_b), 1.05,
                           "the median under-report moves %.1f -> %.1f between "
                           "two equally valid 400-seed blocks" % (med_a, med_b))
        pa, pb = self._pairs(a), self._pairs(b)
        sa, sb = self._share(pa, 19.9), self._share(pb, 19.9)
        self.assertGreater(max(sa, sb) / min(sa, sb), 1.5,
                           "P(no change >= 19.9x) moves %.1f%% -> %.1f%%" % (sa, sb))
        qa, qb = self._pct(pa, 95), self._pct(pb, 95)
        self.assertGreater(max(qa, qb) / min(qa, qb), 1.5,
                           "the pair p95 moves %.0fx -> %.0fx" % (qa, qb))
        ea, eb = max(a) / min(a), max(b) / min(b)
        self.assertGreater(max(ea, eb) / min(ea, eb), 4.0,
                           "the end-to-end span moves %.0fx -> %.0fx, so it is "
                           "not a figure" % (ea, eb))
        # The body of the distribution is where the reading actually lives, and
        # it does hold to a point: this is why 59/52/45/33 are publishable.
        self.assertLess(abs(self._share(pa, 1.80) - self._share(pb, 1.80)), 2.0)

    def test_the_bound_is_MODEL_DEPENDENT_and_this_says_which_way(self):
        """The premise is load-bearing, and it does not fail conservatively.

        AT SIGMA 1e-3, held at the same total perturbation energy, a
        MULTIPLICATIVE (rounding-like) perturbation is read WORSE than the
        i.i.d. dense one on all three columns -- it carries no second-order term
        to hold the denominator away from zero, so both the under-report and the
        pair spread grow. A SPARSE one is read almost in full: its median
        under-report is single digits rather than 75x. Its pair spread is barely
        better, and that column is confounded anyway, because 16 non-zero
        elements let the TRUE divergence span 3.2x across the same seeds where
        the dense model holds it to 1.06x.
        """
        got = {}
        for mode in ("dense", "mult", "sparse"):
            rel = self._rel(mode=mode)
            ratios = [self._draw(s, self.SIGMA, mode)[1] /
                      self._draw(s, self.SIGMA, mode)[0]
                      for s in range(self.SEEDS)]
            pairs = self._pairs(rel)
            got[mode] = (sig(self._pct(ratios, 50)), sig(self._pct(pairs, 95)),
                         round(self._share(pairs, 19.9)))
        self.assertEqual(got["dense"], (75.0, 24.0, 6))
        self.assertEqual(got["mult"], (110.0, 41.0, 8))
        self.assertEqual(got["sparse"], (2.9, 19.0, 5))
        true_l1 = [self._draw(s, self.SIGMA, "sparse")[1] for s in range(self.SEEDS)]
        self.assertEqual(sig(max(true_l1) / min(true_l1)), 3.2,
                         "the sparse model does NOT hold the true divergence fixed")

    def test_the_MAGNITUDE_column_of_that_table_REVERSES_below_sigma_1e_3(self):
        """"Multiplicative is worse" is a SIGMA 1e-3 statement, not a general one.

        The multiplicative model is scale-INVARIANT: its perturbation is
        `x * N(0, sigma/sigma_a)`, so the three columns `figures()` returns --
        the median under-report, the pair p95 and the tail share, every one of
        them a RATIO or a share of ratios -- are the same at 1e-3, 1e-4 and 1e-5.
        Its `rel(sumabs)` is NOT: that scales exactly tenfold with sigma, and the
        scale then divides back out of each ratio taken from it. The dense model
        does not even hold the ratios -- as sigma falls it loses the
        second-order term that holds its denominator up, and its median
        under-report rises from 75x to 140x, PAST the multiplicative model's
        110x. So in the linear regime the DENSE model is the one read worse on
        magnitude.

        This matters for the row that quotes the table, not only for the table.
        The readings #2877 argues over are `1.772e-05` and `1.062e-06`, and this
        control's median `rel(sumabs)` is 1.9e-05 at sigma 1e-4 and 1.9e-06 at
        sigma 1e-5 against 3.5e-04 at sigma 1e-3: the row reads in the LINEAR
        regime, on the reversed side of the table it cites.

        The two SPREAD columns do NOT reverse, which is why the paragraph's
        conclusion survives its own qualification: multiplicative stays worse on
        both at every sigma.
        """
        def figures(mode, sigma):
            ratios = [self._draw(s, sigma, mode)[1] / self._draw(s, sigma, mode)[0]
                      for s in range(self.SEEDS)]
            pairs = self._pairs(self._rel(mode=mode, sigma=sigma))
            return (sig(self._pct(ratios, 50)), sig(self._pct(pairs, 95)),
                    round(self._share(pairs, 19.9)))

        mult = {s: figures("mult", s) for s in (1e-3, 1e-4, 1e-5)}
        dense = {s: figures("dense", s) for s in (1e-3, 1e-4, 1e-5)}
        self.assertEqual(len({v for v in mult.values()}), 1,
                         "the multiplicative model is scale-invariant: %r" % (mult,))
        self.assertGreater(mult[1e-3][0], dense[1e-3][0],
                           "at sigma 1e-3 the multiplicative model is read worse")
        for sigma in (1e-4, 1e-5):
            self.assertGreater(
                dense[sigma][0], mult[sigma][0],
                "at sigma %g the DENSE model is read worse on magnitude "
                "(%.0fx against %.0fx), which is the reversal the documents must "
                "carry" % (sigma, dense[sigma][0], mult[sigma][0]))
        for sigma in (1e-3, 1e-4, 1e-5):
            self.assertGreater(mult[sigma][1], dense[sigma][1],
                               "the pair p95 column does not reverse at %g" % sigma)
            self.assertGreater(mult[sigma][2], dense[sigma][2],
                               "the tail-share column does not reverse at %g" % sigma)
        # The regime the row actually reads in, as the documents state it.
        self.assertEqual(
            [published(self._pct(self._rel(sigma=s), 50), decimals=1, sci=True)
             for s in (1e-3, 1e-4, 1e-5)],
            ["3.5e-04", "1.9e-05", "1.9e-06"],
            "the sigma each of the row's own readings corresponds to")

    def test_the_tool_reports_the_spread_and_not_a_single_constant(self):
        """Every run's stdout must carry what the cases above measured.

        The first version printed `~122x under-report measured at this tap's
        n=12800` -- one seed draw, presented as the property of the tap.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=2, steps=2)
            fingerprint(b, layers=2, steps=2)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("DISTRIBUTION, not a constant", r.stdout)
            self.assertIn("p05..p95", r.stdout)
            self.assertNotIn("~122x", r.stdout,
                             "one seed draw must not be printed as the tap's property")

    # ------------------------------------------------------------------
    # THE PUBLISHER. Everything below READS a surface off disk (or runs the
    # tool and reads its stdout) and compares what it PRINTS against the value
    # drawn above, rendered by `published()`. No rendered figure is typed here.
    # ------------------------------------------------------------------

    _FIGURES = {}

    _LABEL = {
        "u3p05": "the sigma 1e-3 dense p05 under-report",
        "u3med": "the sigma 1e-3 dense MEDIAN under-report",
        "u3p95": "the sigma 1e-3 dense p95 under-report",
        "u4p05": "the sigma 1e-4 dense p05 under-report",
        "u4med": "the sigma 1e-4 dense MEDIAN under-report",
        "u4p95": "the sigma 1e-4 dense p95 under-report",
        "u5p05": "the sigma 1e-5 dense p05 under-report",
        "u5med": "the sigma 1e-5 dense MEDIAN under-report",
        "u5p95": "the sigma 1e-5 dense p95 under-report",
        "u1p05": "the sign-aligned positive control p05",
        "u1med": "the sign-aligned positive control median",
        "u1p95": "the sign-aligned positive control p95",
        "pair50": "the pair-ratio median at a fixed true divergence",
        "pair75": "the pair-ratio p75",
        "pair90": "the pair-ratio p90",
        "pair95": "the pair-ratio p95",
        "relspan": "the rel(sumabs) p05..p95 span",
        "truespan": "the TRUE divergence span over the same draws",
        "s180": "P(no change >= 1.80x)",
        "s202": "P(no change >= 2.02x)",
        "s234": "P(no change >= 2.34x)",
        "s315": "P(no change >= 3.15x)",
        "s1160": "P(no change >= 11.6x)",
        "s1670": "P(no change >= 16.7x)",
        "s1990": "P(no change >= 19.9x)",
        "s2410": "P(no change >= 24.1x)",
        "pct315": "the percentile of no change that 3.15x sits at",
        "mdDmed": "the dense median under-report, model-dependence table",
        "mdDp95": "the dense pair p95, model-dependence table",
        "mdDs199": "the dense P(no change >= 19.9x), model-dependence table",
        "mdMmed": "the MULTIPLICATIVE median under-report at sigma 1e-3",
        "mdMp95": "the multiplicative pair p95 at sigma 1e-3",
        "mdMs199": "the multiplicative P(no change >= 19.9x) at sigma 1e-3",
        "mdSmed": "the SPARSE median under-report",
        "mdSp95": "the sparse pair p95",
        "mdSs199": "the sparse P(no change >= 19.9x)",
        "mdMmed4": "the multiplicative median under-report at sigma 1e-4",
        "mdMmed5": "the multiplicative median under-report at sigma 1e-5",
        "mdMp954": "the multiplicative pair p95 at sigma 1e-4",
        "mdMp955": "the multiplicative pair p95 at sigma 1e-5",
        "mdMs1994": "the multiplicative P(no change >= 19.9x) at sigma 1e-4",
        "mdMs1995": "the multiplicative P(no change >= 19.9x) at sigma 1e-5",
        "sparsespan": "the TRUE divergence span under the sparse model",
        "b2p05_1": "the second seed block's p05 under-report",
        "b2med": "the second seed block's median under-report",
        "b2med_1": "the second seed block's median under-report, one decimal",
        "b2p95": "the second seed block's p95 under-report",
        "b2p95_1": "the second seed block's p95 under-report, one decimal",
        "b2s199": "the second seed block's P(no change >= 19.9x)",
        "b2s199_1": "the second seed block's P(no change >= 19.9x), one decimal",
        "b2s167_1": "the second seed block's P(no change >= 16.7x), one decimal",
        "asym": "the closed-form asymptote sqrt(2n/pi)",
        "asymmed": "the median of the closed-form asymptote",
        "sqrtn": "sqrt(n)",
        "relmed3": "the median rel(sumabs) drawn at sigma 1e-3",
        "relmed4": "the median rel(sumabs) drawn at sigma 1e-4",
        "relmed5": "the median rel(sumabs) drawn at sigma 1e-5",
    }

    def _figures(self):
        """Draw every figure the documents quote, rendered as they round it."""
        if MetricSpread._FIGURES:
            return MetricSpread._FIGURES
        import math
        import statistics

        def under_report(sigma, mode="dense", first=0):
            d = [self._draw(s, sigma, mode)
                 for s in range(first, first + self.SEEDS)]
            return [rl / rs for rs, rl in d]

        f = {}
        for tag, sigma, mode, dec in (("u3", 1e-3, "dense", 0),
                                      ("u4", 1e-4, "dense", 0),
                                      ("u5", 1e-5, "dense", 0),
                                      ("u1", 1e-3, "aligned", 2)):
            r = under_report(sigma, mode)
            for q, name in ((5, "p05"), (50, "med"), (95, "p95")):
                f[tag + name] = published(self._pct(r, q), decimals=dec)

        rel = self._rel()
        pairs = self._pairs(rel)
        f["pair50"] = published(self._pct(pairs, 50), decimals=1)
        f["pair75"] = published(self._pct(pairs, 75), decimals=1)
        f["pair90"] = published(self._pct(pairs, 90))
        f["pair95"] = published(self._pct(pairs, 95))
        f["relspan"] = published(self._pct(rel, 95) / self._pct(rel, 5))
        true_l1 = [self._draw(s, self.SIGMA)[1] for s in range(self.SEEDS)]
        f["truespan"] = published(max(true_l1) / min(true_l1), digits=3, decimals=2)
        for move in (1.80, 2.02, 2.34, 3.15, 11.6, 16.7, 19.9, 24.1):
            f["s%d" % round(move * 100)] = published(self._share(pairs, move),
                                                     digits=None)
        # The complement, DERIVED from the share rather than quoted beside it:
        # calling 3.15x "the 33rd percentile" inverts the ranking (#2877).
        f["pct315"] = published(100.0 - float(f["s315"]), digits=None)

        for mode, key, dec in (("dense", "D", 0), ("mult", "M", 0),
                               ("sparse", "S", 1)):
            p = self._pairs(self._rel(mode=mode))
            f["md%smed" % key] = published(self._pct(under_report(self.SIGMA, mode), 50),
                                           decimals=dec)
            f["md%sp95" % key] = published(self._pct(p, 95))
            f["md%ss199" % key] = published(self._share(p, 19.9), digits=None)
        # The multiplicative model is scale-INVARIANT and the dense one is not,
        # which is what reverses the magnitude column below sigma 1e-3.
        for sigma, key in ((1e-4, "4"), (1e-5, "5")):
            p = self._pairs(self._rel(mode="mult", sigma=sigma))
            f["mdMmed" + key] = published(
                self._pct(under_report(sigma, "mult"), 50))
            f["mdMp95" + key] = published(self._pct(p, 95))
            f["mdMs199" + key] = published(self._share(p, 19.9), digits=None)
        sparse_l1 = [self._draw(s, self.SIGMA, "sparse")[1]
                     for s in range(self.SEEDS)]
        f["sparsespan"] = published(max(sparse_l1) / min(sparse_l1), decimals=1)

        rb = under_report(self.SIGMA, "dense", first=self.SEEDS)
        pb = self._pairs(self._rel(first=self.SEEDS))
        f["b2p05_1"] = published(self._pct(rb, 5), digits=None, decimals=1)
        f["b2med"] = published(self._pct(rb, 50), digits=None)
        f["b2med_1"] = published(self._pct(rb, 50), digits=None, decimals=1)
        f["b2p95"] = published(self._pct(rb, 95), digits=None)
        f["b2p95_1"] = published(self._pct(rb, 95), digits=None, decimals=1)
        f["b2s199"] = published(self._share(pb, 19.9), digits=None)
        f["b2s199_1"] = published(self._share(pb, 19.9), digits=None, decimals=1)
        f["b2s167_1"] = published(self._share(pb, 16.7), digits=None, decimals=1)

        f["asym"] = published(math.sqrt(2.0 * self.N / math.pi),
                              digits=None, decimals=1)
        f["asymmed"] = published(math.sqrt(2.0 * self.N / math.pi)
                                 / statistics.NormalDist().inv_cdf(0.75),
                                 digits=None)
        f["sqrtn"] = published(math.sqrt(float(self.N)), digits=None)
        for sigma, key in ((1e-3, "3"), (1e-4, "4"), (1e-5, "5")):
            f["relmed" + key] = published(self._pct(self._rel(sigma=sigma), 50),
                                          decimals=1, sci=True)

        # The documents say the multiplicative model draws "the same" triple at
        # 1e-3, 1e-4 and 1e-5. That word is a claim, so it is checked here rather
        # than trusted, and the sites below may then quote any one of the three.
        for a, b in (("mdMmed", "mdMmed4"), ("mdMmed", "mdMmed5"),
                     ("mdMp95", "mdMp954"), ("mdMp95", "mdMp955"),
                     ("mdMs199", "mdMs1994"), ("mdMs199", "mdMs1995")):
            assert f[a] == f[b], ("the multiplicative model is not "
                                  "scale-invariant at two significant figures: "
                                  "%s=%s vs %s=%s" % (a, f[a], b, f[b]))
        missing = set(f) ^ set(self._LABEL)
        assert not missing, "figure/label table disagree: %s" % sorted(missing)
        MetricSpread._FIGURES = f
        return f

    @staticmethod
    def _flatten(text):
        """One line, one space, no blockquote markers.

        A published figure sits inside a sentence that the author wrapped, or a
        Markdown table row inside a blockquote. Flattening lets one pattern match
        the sentence rather than the line it happens to break on.
        """
        lines = [re.sub(r"^\s*>\s?", "", ln) for ln in text.splitlines()]
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    @staticmethod
    def _read(relpath):
        """Read a SURFACE off disk, at its committed path. Never from memory."""
        return MetricSpread._flatten((ROOT / relpath).read_text(encoding="utf-8"))

    def _publishes(self, surface, text, sites):
        """Every site in `sites` must state its figures as this control draws them."""
        fig = self._figures()
        for what, pattern, keys in sites:
            m = re.search(pattern, text)
            self.assertIsNotNone(
                m,
                "%s: the publisher cannot find where this surface states %s.\n"
                "  pattern: %s\n"
                "The surface either stopped publishing that figure or was "
                "reworded around it. A document and this control move together "
                "or not at all (#2879)." % (surface, what, pattern))
            self.assertEqual(len(m.groups()), len(keys),
                             "site table bug for %s in %s" % (what, surface))
            for key, got in zip(keys, m.groups()):
                self.assertEqual(
                    got, fig[key],
                    "%s publishes %s as `%s`.\n"
                    "  This control draws %s over range(%d) and renders it `%s`.\n"
                    "One of the two is wrong; #2879 is what happens when nobody "
                    "checks." % (surface, what, got, self._LABEL[key],
                                 self.SEEDS, fig[key]))

    _USAGE_SITES = (
        ("the under-report median and p05..p95",
         r"MEDIAN (\d+(?:\.\d+)?)x-(\d+(?:\.\d+)?)x with a p05\.\.p95 of (\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)",
         ("u3med", "u4med", "u3p05", "u4p95")),
        ("the pair median and p95",
         r"two readings differ by a median \*\*(\d+(?:\.\d+)?)x\*\* and by "
         r"\*\*(\d+(?:\.\d+)?)x\*\* at p95",
         ("pair50", "pair95")),
        ("where the 19.9x and 16.7x readings sit",
         r"Those two ratios sit at (\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)% of the metric",
         ("s1990", "s1670")),
        ("the four no-change shares",
         r"1\.80x, 2\.02x, 2\.34x and 3\.15x in (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)% "
         r"and (\d+(?:\.\d+)?)% of draws",
         ("s180", "s202", "s234", "s315")),
        ("the percentile 3.15x actually sits at",
         r"3\.15x sits at the \*\*(\d+(?:\.\d+)?)th\*\* percentile",
         ("pct315",)),
    )

    _TOOL_DOC_SITES = (
        ("the positive-control row of the under-report table",
         r"sign\(a\)-aligned (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)",
         ("u1p05", "u1med", "u1p95")),
        ("the sigma 1e-3 row of the under-report table",
         r"zero-mean, sigma 1e-3 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)",
         ("u3p05", "u3med", "u3p95")),
        ("the sigma 1e-4 row of the under-report table",
         r"zero-mean, sigma 1e-4 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)",
         ("u4p05", "u4med", "u4p95")),
        ("the sigma 1e-5 row of the under-report table",
         r"zero-mean, sigma 1e-5 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)",
         ("u5p05", "u5med", "u5p95")),
        ("the second seed block, quoted as the contrast",
         r"the next disjoint 400 seeds give a median of (\d+(?:\.\d+)?) rather than "
         r"(\d+(?:\.\d+)?) and a p95 of (\d+(?:\.\d+)?) rather than (\d+(?:\.\d+)?)",
         ("b2med", "u3med", "b2p95", "u3p95")),
        ("the closed-form asymptote and sqrt(n)",
         r"sqrt\(2n/pi\)/\|z\| = (\d+(?:\.\d+)?)/\|z\|` for a standard normal "
         r"`z`, median (\d+(?:\.\d+)?)\. `sqrt\(n\) = (\d+(?:\.\d+)?)`",
         ("asym", "asymmed", "sqrtn")),
        ("the two spans",
         r"spans (\d+(?:\.\d+)?)x p05\.\.p95 while the true divergence spans (\d+(?:\.\d+)?)x",
         ("relspan", "truespan")),
        ("the four pair quantiles",
         r"differ by a median (\d+(?:\.\d+)?)x, by (\d+(?:\.\d+)?)x at p75, (\d+(?:\.\d+)?)x at p90 "
         r"and (\d+(?:\.\d+)?)x at p95",
         ("pair50", "pair75", "pair90", "pair95")),
        ("the eight no-change shares",
         r"3\.15x in (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)% of draws, and "
         r"one at least as large as 16\.7x or 19\.9x in (\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)%",
         ("s180", "s202", "s234", "s315", "s1670", "s1990")),
        ("the percentile 3.15x actually sits at",
         r"3\.15x sits at the (\d+(?:\.\d+)?)th percentile of no change",
         ("pct315",)),
        ("the multiplicative model at sigma 1e-3",
         r"median under-report (\d+(?:\.\d+)?)x, pair p95 (\d+(?:\.\d+)?)x, "
         r"`P\(no change >= 19\.9x\)` (\d+(?:\.\d+)?)%, against the dense model's "
         r"(\d+(?:\.\d+)?)x, (\d+(?:\.\d+)?)x and (\d+(?:\.\d+)?)%",
         ("mdMmed", "mdMp95", "mdMs199", "mdDmed", "mdDp95", "mdDs199")),
        ("the sparse model",
         r"is read almost in full at a median (\d+(?:\.\d+)?)x",
         ("mdSmed",)),
        ("the multiplicative model's scale invariance",
         r"scale-INVARIANT -- (\d+(?:\.\d+)?)x at 1e-3, 1e-4 and 1e-5 alike",
         ("mdMmed",)),
        ("what the dense model reaches in the linear regime",
         r"as sigma falls, and reaches (\d+(?:\.\d+)?)x",
         ("u4med",)),
        ("the regime this row's own readings sit in",
         r"median `rel\(sumabs\)` is (\d+(?:\.\d+)?e[-+]?\d+) at sigma 1e-4 and (\d+(?:\.\d+)?e[-+]?\d+) at "
         r"sigma 1e-5, and the `1\.772e-05` and `1\.062e-06` the row's own reading "
         r"argues over are at or below the first of those -- the LINEAR regime, on "
         r"the reversed side of this table -- while sigma 1e-3 draws "
         r"(\d+(?:\.\d+)?e[-+]?\d+)",
         ("relmed4", "relmed5", "relmed3")),
    )

    _STDOUT_SITES = (
        ("the under-report median and p05..p95",
         r"under-report is (\d+(?:\.\d+)?)x \(sigma 1e-3\) to (\d+(?:\.\d+)?)x \(sigma 1e-4\), "
         r"p05\.\.p95 (\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)",
         ("u3med", "u4med", "u3p05", "u4p95")),
        ("the pair quantiles",
         r"two readings differ by a median (\d+(?:\.\d+)?)x, (\d+(?:\.\d+)?)x at p90 and "
         r"(\d+(?:\.\d+)?)x at p95",
         ("pair50", "pair90", "pair95")),
    )

    _EVIDENCE_SITES = (
        ("the positive-control row of the under-report table",
         r"\| aligned with `sign\(a\)` \| (\d+(?:\.\d+)?) \| \*\*(\d+(?:\.\d+)?)\*\* \| (\d+(?:\.\d+)?) \|",
         ("u1p05", "u1med", "u1p95")),
        ("the sigma 1e-3 row of the under-report table",
         r"\| zero-mean, sigma 1e-3 \| (\d+(?:\.\d+)?) \| \*\*(\d+(?:\.\d+)?)\*\* \| (\d+(?:\.\d+)?) \|",
         ("u3p05", "u3med", "u3p95")),
        ("the sigma 1e-4 row of the under-report table",
         r"\| zero-mean, sigma 1e-4 \| (\d+(?:\.\d+)?) \| \*\*(\d+(?:\.\d+)?)\*\* \| (\d+(?:\.\d+)?) \|",
         ("u4p05", "u4med", "u4p95")),
        ("the sigma 1e-5 row of the under-report table",
         r"\| zero-mean, sigma 1e-5 \| (\d+(?:\.\d+)?) \| \*\*(\d+(?:\.\d+)?)\*\* \| (\d+(?:\.\d+)?) \|",
         ("u5p05", "u5med", "u5p95")),
        ("the closed-form asymptote",
         r"`sqrt\(2n/pi\)/\|z\| = (\d+(?:\.\d+)?)/\|z\|` for a standard normal `z`, "
         r"median \*\*(\d+(?:\.\d+)?)\*\*",
         ("asym", "asymmed")),
        ("the second seed block, quoted as the contrast",
         r"median of \*\*(\d+(?:\.\d+)?)\*\* rather than (\d+(?:\.\d+)?) and a p95 of "
         r"\*\*(\d+(?:\.\d+)?)\*\* rather than (\d+(?:\.\d+)?)",
         ("b2med", "u3med", "b2p95", "u3p95")),
        ("the two spans",
         r"spans \*\*(\d+(?:\.\d+)?)x\*\* p05\.\.p95 while the true divergence spans "
         r"(\d+(?:\.\d+)?)x",
         ("relspan", "truespan")),
        ("the four pair quantiles",
         r"differ by a median \*\*(\d+(?:\.\d+)?)x\*\*, (\d+(?:\.\d+)?)x at p75, (\d+(?:\.\d+)?)x at "
         r"p90 and \*\*(\d+(?:\.\d+)?)x\*\* at p95",
         ("pair50", "pair75", "pair90", "pair95")),
        ("the 24.1x row of the no-change table",
         r"\| 24\.1x \| [^|]+\| (\d+(?:\.\d+)?)% \|", ("s2410",)),
        ("the 19.9x row of the no-change table",
         r"\| 19\.9x \| [^|]+\| (\d+(?:\.\d+)?)% \|", ("s1990",)),
        ("the 16.7x row of the no-change table",
         r"\| 16\.7x \| [^|]+\| (\d+(?:\.\d+)?)% \|", ("s1670",)),
        ("the 11.6x row of the no-change table",
         r"\| 11\.6x \| [^|]+\| (\d+(?:\.\d+)?)% \|", ("s1160",)),
        ("the 3.15x row of the no-change table",
         r"\| 3\.15x \| [^|]+\| \*\*(\d+(?:\.\d+)?)%\*\* \|", ("s315",)),
        ("the 2.34x row of the no-change table",
         r"\| 2\.34x \| [^|]+\| \*\*(\d+(?:\.\d+)?)%\*\* \|", ("s234",)),
        ("the 2.02x row of the no-change table",
         r"\| 2\.02x \| [^|]+\| \*\*(\d+(?:\.\d+)?)%\*\* \|", ("s202",)),
        ("the 1.80x row of the no-change table",
         r"\| 1\.80x \| [^|]+\| \*\*(\d+(?:\.\d+)?)%\*\* \|", ("s180",)),
        ("how far the tail shares move between blocks",
         r"`P\(>= 19\.9x\)` reads (\d+(?:\.\d+)?)% on the committed block and (\d+(?:\.\d+)?)% "
         r"on the next one",
         ("s1990", "b2s199")),
        ("the share this line first truncated",
         r"truncating (\d+(?:\.\d+)?)% where every other share on this row is rounded",
         ("b2s199_1",)),
        ("the dense row of the model-dependence table",
         r"\| i\.i\.d\. dense \| (\d+(?:\.\d+)?)x \| (\d+(?:\.\d+)?)x \| (\d+(?:\.\d+)?)% \|",
         ("mdDmed", "mdDp95", "mdDs199")),
        ("the multiplicative row of the model-dependence table",
         r"\| multiplicative, proportional to `a` \(rounding-like\) \| (\d+(?:\.\d+)?)x "
         r"\| (\d+(?:\.\d+)?)x \| (\d+(?:\.\d+)?)% \|",
         ("mdMmed", "mdMp95", "mdMs199")),
        ("the sparse row of the model-dependence table",
         r"\| sparse, 16 elements of 12800 \(like a top-k flip\) \| "
         r"\*\*(\d+(?:\.\d+)?)x\*\* \| (\d+(?:\.\d+)?)x \| (\d+(?:\.\d+)?)% \|",
         ("mdSmed", "mdSp95", "mdSs199")),
        ("the sparse model read against the dense one",
         r"a median (\d+(?:\.\d+)?)x under-report where the dense model gives (\d+(?:\.\d+)?)x",
         ("mdSmed", "mdDmed")),
        ("the multiplicative model's scale invariance",
         r"the control draws the same (\d+(?:\.\d+)?)x, (\d+(?:\.\d+)?)x and (\d+(?:\.\d+)?)% at "
         r"sigma 1e-3, 1e-4 and 1e-5",
         ("mdMmed4", "mdMp954", "mdMs1994")),
        ("what the dense model reaches in the linear regime",
         r"median under-report rises from (\d+(?:\.\d+)?)x to \*\*(\d+(?:\.\d+)?)x\*\*",
         ("mdDmed", "u4med")),
        ("the regime this row's own readings sit in",
         r"median `rel\(sumabs\)` is \*\*(\d+(?:\.\d+)?e[-+]?\d+)\*\* at sigma 1e-4 and "
         r"\*\*(\d+(?:\.\d+)?e[-+]?\d+)\*\* at sigma 1e-5, .{0,140}? where sigma 1e-3 draws "
         r"\*\*(\d+(?:\.\d+)?e[-+]?\d+)\*\*",
         ("relmed4", "relmed5", "relmed3")),
        ("the confounded sparse pair column",
         r"let the TRUE divergence span (\d+(?:\.\d+)?)x across the same seeds where "
         r"the dense model holds it to (\d+(?:\.\d+)?)x",
         ("sparsespan", "truespan")),
    )

    _MIRROR_SPEC_SITES = (
        ("the under-report median and p05..p95",
         r"the median is (\d+(?:\.\d+)?)x \(sigma 1e-3\) to (\d+(?:\.\d+)?)x \(sigma 1e-4\), "
         r"p05\.\.p95 (\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)",
         ("u3med", "u4med", "u3p05", "u4p95")),
        ("the pair median and p95",
         r"two readings differ by a median \*\*(\d+(?:\.\d+)?)x\*\* and \*\*(\d+(?:\.\d+)?)x\*\* "
         r"at p95",
         ("pair50", "pair95")),
        ("where the 19.9x and 16.7x readings sit",
         r"19\.9x and 16\.7x sit at (\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)% of the metric",
         ("s1990", "s1670")),
        ("the four no-change shares",
         r"produces 1\.80x, 2\.02x, 2\.34x and 3\.15x in \*\*(\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, "
         r"(\d+(?:\.\d+)?)% and (\d+(?:\.\d+)?)%\*\* of draws",
         ("s180", "s202", "s234", "s315")),
        ("the percentile 3.15x actually sits at",
         r"3\.15x sits at the \*\*(\d+(?:\.\d+)?)th\*\* percentile",
         ("pct315",)),
    )

    _FLASH_SPEC_SITES = (
        ("the under-report median and p05..p95",
         r"a median of (\d+(?:\.\d+)?)x at sigma 1e-3 and (\d+(?:\.\d+)?)x at sigma 1e-4 with "
         r"a p05\.\.p95 of (\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)",
         ("u3med", "u4med", "u3p05", "u4p95")),
        ("the closed-form asymptote",
         r"`sqrt\(2n/pi\)/\|z\| = (\d+(?:\.\d+)?)/\|z\|`, median (\d+(?:\.\d+)?)",
         ("asym", "asymmed")),
        ("the rel(sumabs) span",
         r"Over 400 seeds that span is \*\*(\d+(?:\.\d+)?)x\*\* p05\.\.p95",
         ("relspan",)),
        ("the pair quantiles",
         r"differ by a median (\d+(?:\.\d+)?)x, (\d+(?:\.\d+)?)x at p90 and (\d+(?:\.\d+)?)x at p95",
         ("pair50", "pair90", "pair95")),
    )

    _SELF_SITES = (
        ("the second seed block, quoted three ways in this class's own docstring",
         r"gives `(\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?)` and `(\d+(?:\.\d+)?)% / (\d+(?:\.\d+)?)%`",
         ("b2p05_1", "b2med_1", "b2p95_1", "b2s199_1", "b2s167_1")),
    )

    def test_the_PUBLISHER_reproduces_docs_USAGE_md(self):
        """`docs/USAGE.md` is PUBLIC, and every figure in its cell is drawn here."""
        self._publishes("docs/USAGE.md", self._read("docs/USAGE.md"),
                        self._USAGE_SITES)

    def test_the_PUBLISHER_reproduces_the_tool_docstring(self):
        """The table and prose a reader gets from `--help` and from the file."""
        self._publishes("the docstring of scripts/q4exp-layerfp-diff.py",
                        self._flatten(diff.__doc__), self._TOOL_DOC_SITES)

    def test_the_PUBLISHER_reproduces_the_tools_RUNTIME_STDOUT(self):
        """The surface a USER sees. A docstring nobody opens is not it.

        The tool prints its own summary of this control on every run. That text
        is a published figure like any other, and until #2879 nothing read it:
        the older case asserted three substrings and would pass with every number
        in the paragraph wrong.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            a = pathlib.Path(td) / "a.txt"
            b = pathlib.Path(td) / "b.txt"
            fingerprint(a, layers=2, steps=2)
            fingerprint(b, layers=2, steps=2)
            r = subprocess.run([sys.executable, str(TOOL), str(a), str(b)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        self._publishes("the stdout of scripts/q4exp-layerfp-diff.py",
                        self._flatten(r.stdout), self._STDOUT_SITES)

    def test_the_PUBLISHER_reproduces_the_evidence_file(self):
        """The row's evidence file: three tables and the prose around them."""
        self._publishes(
            "docs/bench-evidence/qwen4exp-gdn-chunked-token-ids-20260904.md",
            self._read("docs/bench-evidence/"
                       "qwen4exp-gdn-chunked-token-ids-20260904.md"),
            self._EVIDENCE_SITES)

    def test_the_PUBLISHER_reproduces_both_specs(self):
        """Both owning specs quote the same set, in their own words."""
        self._publishes(".agents/specs/gdn-chunked-mirror.md",
                        self._read(".agents/specs/gdn-chunked-mirror.md"),
                        self._MIRROR_SPEC_SITES)
        self._publishes(".agents/specs/qwen4-exp-flash-next.md",
                        self._read(".agents/specs/qwen4-exp-flash-next.md"),
                        self._FLASH_SPEC_SITES)

    def test_the_PUBLISHER_reproduces_this_files_own_docstrings(self):
        """This file quotes the second block too, and it is a surface as well."""
        self._publishes("tests/scripts/test_q4exp_layerfp_diff.py",
                        self._flatten(pathlib.Path(__file__).read_text(
                            encoding="utf-8")),
                        self._SELF_SITES)

    def test_the_PUBLISHER_is_NOT_VACUOUS_on_a_moved_digit_or_a_lost_anchor(self):
        """The positive control for every case above.

        A publisher that silently matched nothing reads exactly like a publisher
        that matched everything, which is the #2879 defect one level up. Both
        failure modes are exercised here on a COPY of a real surface, mutated
        over the SPAN the site pattern captured rather than over a literal typed
        here: move the digit, and delete the anchor.
        """
        fig = self._figures()
        real = self._read("docs/USAGE.md")
        what, pattern, keys = self._USAGE_SITES[3]
        m = re.search(pattern, real)
        self.assertIsNotNone(m, "the site under test no longer matches: %s" % what)
        self.assertEqual(m.group(4), fig["s315"])

        bumped = published(float(fig["s315"]) + 1.0, digits=None)
        moved = real[:m.start(4)] + bumped + real[m.end(4):]
        self.assertNotEqual(len(moved), 0)
        self.assertTrue(moved != real, "the digit mutation did not apply")
        with self.assertRaises(AssertionError) as caught:
            self._publishes("docs/USAGE.md (one digit moved)", moved,
                            self._USAGE_SITES)
        self.assertIn("P(no change >= 3.15x)", str(caught.exception))
        self.assertIn(bumped, str(caught.exception))

        cut = real[:m.start()] + real[m.end():]
        self.assertTrue(cut != real, "the anchor mutation did not apply")
        with self.assertRaises(AssertionError) as gone:
            self._publishes("docs/USAGE.md (anchor deleted)", cut,
                            self._USAGE_SITES)
        self.assertIn("cannot find where this surface states", str(gone.exception))


class MetricHonesty(unittest.TestCase):
    """rel_sumabs is a difference of NORMS. head_dmax is a difference."""

    def test_rel_sumabs_reads_zero_on_two_wholly_different_tensors(self):
        # Equal L1 norms, opposite signs: every element differs, rel_sumabs == 0.
        self.assertEqual(diff.rel_sumabs("390.0", "390.0"), 0.0)

    def test_head_dmax_sees_what_rel_sumabs_cannot(self):
        a = {"v": "1.0,2.0,3.0,4.0", "sumabs": "10"}
        b = {"v": "-1.0,-2.0,-3.0,-4.0", "sumabs": "10"}
        self.assertEqual(diff.rel_sumabs(a["sumabs"], b["sumabs"]), 0.0,
                         "the norms are equal, so the committed metric reads agreement")
        self.assertEqual(diff.head_dmax(a, b), 8.0,
                         "an elementwise difference cannot cancel")

    def test_head_dmax_positive_control_is_zero_on_identical_values(self):
        a = {"v": "1.0,2.0,3.0,4.0"}
        self.assertEqual(diff.head_dmax(a, dict(a)), 0.0)


if __name__ == "__main__":
    unittest.main()
