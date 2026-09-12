#!/usr/bin/env python3
"""Mutation checks for the DeepSeek-V4 vision parity harness.

Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411.

`tools/parity/dsv4v_w6_compare.py` judges a parity leg and exits on its verdict,
and the four drivers beside it record a step per leg and read those steps back.
Both claims were made before and neither held: the comparator returned PASS on
data that should fail, and the readback could not tell a step that never ran
from one that passed. This file PERFORMS each failure and fails if the harness
shrugs it off.

Every case below was RED against the harness as it stood at eb6009f6c or at
6e55cc113:

  a zero image row on both sides   VERDICT PASS, exit 0, while 99 other rows
                                   were 50% off
  a stage 100x wrong               VERDICT PASS, exit 0
  a stage absent entirely          VERDICT PASS, exit 0
  a stage shape-mismatched         VERDICT PASS, exit 0
  a profile with no `judged` key   VERDICT DIAGNOSTIC, exit 0, on 50%-off data
  a steps file missing a step      failed_steps=0, exit 0
  an empty steps file              failed_steps=0, exit 0
  a profile missing ANY OTHER
    judging key                    VERDICT PASS, exit 0, on 50%-off data
  an absent `cells` stage          VERDICT PASS, exit 0, though both the module
                                   docstring and the bounds file say its
                                   presence is reported
  the f32 arm's `input` stage
    100x wrong                     VERDICT PASS, exit 0, with no BOUND line
  a crash in the device probe      filed as an expected refusal, RC=0
  a failed W6 run                  printed its `### W6_*_DONE` banner anyway

THE FIXTURE ROWS MUST BE PAIRWISE DISTINCT IN DIRECTION. A first version built
them from a smooth ramp, which made every image row near-parallel (cosine
0.9999988 between DIFFERENT rows) while rounding to bf16 moves each element by
about 0.4%. The rounding noise swamped the angular separation, the argmax in
best_match() became arbitrary, and the identity-permutation condition failed on
every dataset including the clean one -- so each case exited 1 for a reason that
had nothing to do with what it meant to test. Independent pseudo-random
directions separate by far more than the rounding noise. That fixture defect is
now also a PROPERTY THE COMPARATOR CHECKS, because the same degeneracy is a
false GREEN on genuinely permuted output: `_block(mode="parallel")` below builds
the degenerate shape on purpose and the comparator must refuse to read an
ordering claim off it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / "tools/parity"
COMPARE = PARITY / "dsv4v_w6_compare.py"
BOUNDS = PARITY / "dsv4v_w6_bounds.json"

_spec = importlib.util.spec_from_file_location("dsv4v_w6_compare", COMPARE)
assert _spec is not None and _spec.loader is not None
CMP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CMP)

COLS = 16


class _LCG:
    """Deterministic, so a failure here is reproducible rather than a flake."""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF

    def unit(self) -> float:
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF

    def direction(self, n: int) -> list[float]:
        return [CMP.bf16(2.0 * self.unit() - 1.0) for _ in range(n)]


def _write(path: Path, rows: list[list[float]]) -> None:
    n, c = len(rows), len(rows[0])
    with open(path, "wb") as handle:
        handle.write(struct.pack("<ii", n, c))
        for row in rows:
            handle.write(struct.pack("<%df" % c, *row))


def _block(outdir: Path, tag: str, lead_pad: int, mode: str,
           factor: float = 1.02) -> None:
    """Write a token block. `mode` selects the defect under test."""
    types, _cell = CMP.layout(lead_pad, 10, 10)
    rng = _LCG(20260912)
    if mode == "drift":
        factor = 1.5
    ours, ref = [], []
    seen = 0
    for i, kind in enumerate(types):
        if kind == "IMAGE":
            if mode == "parallel":
                # THE DEGENERATE SHAPE, on purpose: a smooth ramp makes every
                # image row near-parallel to every other, so the argmax that
                # carries the ordering claim is decided by bf16 rounding.
                base = [CMP.bf16(0.5 + 0.01 * seen + 0.001 * j)
                        for j in range(COLS)]
            else:
                base = rng.direction(COLS)
            if mode == "zero" and seen == 0:
                # A row that is ZERO ON BOTH SIDES. It sits at image index 0
                # deliberately: best_match maps a zero row to argmax 0, so
                # anywhere else the identity-permutation condition would fail
                # and mask the bound this case exists to test.
                base = [0.0] * COLS
                ours_row = [0.0] * COLS
            else:
                ours_row = [CMP.bf16(v * factor) for v in base]
            seen += 1
        else:
            base = [CMP.bf16(0.25 + 0.001 * ((i + j) % 11)) for j in range(COLS)]
            ours_row = list(base)
        ours.append(ours_row)
        ref.append(base)
    _write(outdir / ("ours-%s-block.f32" % tag), ours)
    _write(outdir / ("oracle-%s-block.f32" % tag), ref)


def _stage(outdir: Path, tag: str, stage: str, rows: int, factor: float,
           shape_bad: bool = False) -> None:
    rng = _LCG(777 + len(stage))
    ref = [rng.direction(COLS) for _ in range(rows)]
    ours = [[CMP.bf16(v * factor) for v in row] for row in ref]
    if shape_bad:
        ours = ours[: max(1, rows // 2)]
    _write(outdir / ("ours-%s-%s.f32" % (tag, stage)), ours)
    _write(outdir / ("oracle-%s-%s.f32" % (tag, stage)), ref)


def _dataset(outdir: Path, mode: str = "clean", tag: str = "lp0",
             factor: float = 1.02) -> Path:
    """A faithful leg: a block plus the three stage dumps a real run writes."""
    outdir.mkdir(parents=True, exist_ok=True)
    _block(outdir, tag, 0, mode, factor)
    _stage(outdir, tag, "input", 64, 1.0)        # exactly bf16(oracle)
    _stage(outdir, tag, "vit", 64, factor)       # inside the recorded bound
    _stage(outdir, tag, "cells", 100, factor)
    return outdir


def _geometry_rows(kind: str, count: int, dim: int,
                   rng: _LCG) -> list[list[float]]:
    """Image rows with a chosen DIRECTIONAL geometry, at a realistic width.

    `shared-<eps>` gives every cell one global component plus `eps` of its own
    detail, which is what a photograph's sky or wall looks like to a cosine.
    `flat-<eps>` puts 20 such cells in an otherwise independent image, because
    `min_best_margin` is a MIN and one flat pair is enough to decide a run.
    """
    if kind == "random":
        return [rng.direction(dim) for _ in range(count)]
    name, eps = kind.rsplit("-", 1)
    shared, scale = rng.direction(dim), float(eps)
    rows = []
    for index in range(count):
        detail = rng.direction(dim)
        flat = name == "shared" or (name == "flat" and index < 20)
        rows.append([CMP.bf16(shared[j] + scale * detail[j]) for j in range(dim)]
                    if flat else detail)
    return rows


def _photo_dataset(outdir: Path, kind: str, dim: int = 1280,
                   factor: float = 1.02) -> Path:
    """A CLEAN leg whose image rows carry a photograph-shaped geometry.

    The width is the point. The mutation fixture is 16-dimensional pseudo-random
    rows -- the least photograph-like geometry available, measuring a margin of
    0.229 -- so it was never evidence for any margin constant.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    types, _cell = CMP.layout(0, 10, 10)
    rng = _LCG(20260912)
    image = _geometry_rows(kind, 100, dim, rng)
    ours, ref, seen = [], [], 0
    for i, kindof in enumerate(types):
        if kindof == "IMAGE":
            base = image[seen]
            seen += 1
            row = [CMP.bf16(v * factor) for v in base]
        else:
            base = [CMP.bf16(0.25 + 0.001 * ((i + j) % 11)) for j in range(dim)]
            row = list(base)
        ours.append(row)
        ref.append(base)
    _write(outdir / "ours-lp0-block.f32", ours)
    _write(outdir / "oracle-lp0-block.f32", ref)
    _stage(outdir, "lp0", "input", 64, 1.0)
    _stage(outdir, "lp0", "vit", 64, factor)
    _stage(outdir, "lp0", "cells", 100, factor)
    return outdir


def _run(directory: Path, tag: str = "lp0", script: Path = COMPARE,
         lead_pad: int = 0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), str(directory), tag, str(lead_pad), "10", "10"],
        capture_output=True, text=True)


def _with_bounds(tmp: Path, mutate, name: str = "mutated") -> Path:
    """A copy of the comparator whose bounds file has been mutated beside it."""
    home = tmp / name
    home.mkdir(parents=True, exist_ok=True)
    shutil.copy(COMPARE, home / COMPARE.name)
    data = json.loads(BOUNDS.read_text())
    mutate(data)
    (home / BOUNDS.name).write_text(json.dumps(data, indent=1))
    return home / COMPARE.name


class ComparatorVerdict(unittest.TestCase):
    """The comparator must not report success over a failure."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dsv4v-compare-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_clean_leg_passes(self) -> None:
        """The repair must not turn a good run red."""
        done = _run(_dataset(self.tmp / "clean"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("VERDICT PASS", done.stdout)

    def test_the_f32_leg_passes(self) -> None:
        """The other judged profile, on data inside its tighter bound."""
        done = _run(_dataset(self.tmp / "f32", tag="f32", factor=1.005), tag="f32")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("VERDICT PASS", done.stdout)

    def test_zero_row_cannot_disable_the_bounds(self) -> None:
        """One row zero on BOTH sides used to make every bound a no-op.

        stats() returns nan for that row, the mean over rows is nan, and both
        `nan > 0.049` and `nan < 0.998` are False. Measured before the repair:
        VERDICT PASS and exit 0, with the other 99 rows 50% off.
        """
        done = _run(_dataset(self.tmp / "zero", mode="zero"))
        self.assertNotEqual(done.returncode, 0,
                            "a degenerate row still reports success:\n" + done.stdout)
        self.assertIn("VERDICT FAIL", done.stdout)
        self.assertIn("degenerate", done.stdout)
        self.assertIn("NOT FINITE", done.stdout)

    def test_the_degenerate_row_rule_is_load_bearing(self) -> None:
        """Deleting the degenerate-row refusal alone must turn something red.

        It cannot be shown through a data file: stats() returns nan for ANY
        zero-norm row, so on file data the degenerate rule and the isfinite
        guard always fire together and the case above is carried by whichever
        remains. Judging a REPORT directly separates them, and this is the case
        that reds when the `max_degenerate_rows` refusal in judge() is removed
        while every other rule stands.
        """
        bounds = json.loads(BOUNDS.read_text())
        report = {
            "sentinels": {k: {"bf16_of_oracle_exact": True, "max_abs": 0.0}
                          for k in ("START", "END", "NEWLINE", "PAD")},
            "permutation": {"identity_is_best": 100, "of": 100,
                            "min_best_cos": 0.99, "min_best_margin": 0.5,
                            # The margin is judged against the DATASET's own
                            # bf16 rounding scale, so a report that carries a
                            # margin must carry the scale it is judged against.
                            "bf16_rounding_scale": 4e-06},
            "image_rows": {"rows": 100, "mean_rel_l2": 0.02, "mean_cos": 0.999,
                           "degenerate_rows": 3, "degenerate_row_index": [1, 2, 3]},
            "input": {"bf16_of_oracle_exact": True},
            "vit": {"mean_rel_l2": 0.02, "mean_cos": 0.999, "degenerate_rows": 0},
            "cells": {"rows": 100},
        }
        verdict, bad, _notes = CMP.judge(report, "lp0", bounds)
        self.assertEqual(verdict, "FAIL", bad)
        self.assertTrue(any("degenerate" in line for line in bad), bad)
        # THE CONTROL: the identical report with no degenerate row passes, so
        # the verdict above is that rule and nothing else.
        report["image_rows"]["degenerate_rows"] = 0
        report["image_rows"]["degenerate_row_index"] = []
        self.assertEqual(CMP.judge(report, "lp0", bounds)[0], "PASS")

    def test_drifted_leg_still_fails(self) -> None:
        """The control: the same 50%-off data with no zero row."""
        done = _run(_dataset(self.tmp / "drift", mode="drift"))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("EXCEEDS the recorded bound", done.stdout)

    def test_absent_stage_is_not_a_pass(self) -> None:
        """A stage that never ran is not a stage that passed."""
        directory = _dataset(self.tmp / "novit")
        os.remove(directory / "ours-lp0-vit.f32")
        os.remove(directory / "oracle-lp0-vit.f32")
        done = _run(directory)
        self.assertNotEqual(done.returncode, 0,
                            "an absent stage still reports success:\n" + done.stdout)
        self.assertIn("ABSENT", done.stdout)

    def test_wrong_stage_is_caught(self) -> None:
        directory = _dataset(self.tmp / "badvit")
        _stage(directory, "lp0", "vit", 64, 100.0)
        done = _run(directory)
        self.assertNotEqual(done.returncode, 0,
                            "a 100x-wrong stage still reports success:\n" + done.stdout)
        self.assertIn("stage 'vit' mean_rel_l2", done.stdout)

    def test_shape_mismatched_stage_is_caught(self) -> None:
        directory = _dataset(self.tmp / "badinput")
        _stage(directory, "lp0", "input", 64, 1.0, shape_bad=True)
        done = _run(directory)
        self.assertNotEqual(done.returncode, 0,
                            "a shape-mismatched stage still reports success:\n" + done.stdout)
        self.assertIn("SHAPE MISMATCH", done.stdout)

    def test_missing_judged_key_is_an_error(self) -> None:
        """An incomplete profile must not downgrade a judged leg."""
        def drop(data):
            del data["profiles"]["shipped_bf16"]["judged"]

        script = _with_bounds(self.tmp, drop)
        done = _run(_dataset(self.tmp / "d1", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("VERDICT ERROR", done.stdout)

    def test_every_judging_key_must_be_declared(self) -> None:
        """THE CLASS, not the instance.

        `judged` was hardened first and every other judging key kept the same
        shape: read with `profile.get(...)`, silently unbounded when absent.
        Measured on 50%-off data, tag lp0: dropping `mean_rel_l2_max` exited 0
        PASS, dropping it with `mean_cos_min` exited 0 PASS, and a profile
        holding only `judged` exited 0 PASS. Each key is dropped here on its own
        and the run must ERROR rather than judge the leg without that bound.
        """
        for key in sorted(CMP.PROFILE_KEYS):
            with self.subTest(key=key):
                def drop(data, key=key):
                    del data["profiles"]["shipped_bf16"][key]

                script = _with_bounds(self.tmp, drop, name="drop-" + key)
                done = _run(_dataset(self.tmp / ("k-" + key), mode="drift"),
                            script=script)
                self.assertEqual(done.returncode, 4,
                                 "dropping %r was judged anyway:\n%s"
                                 % (key, done.stdout))
                self.assertIn("does not DECLARE", done.stdout)

    def test_dropping_every_key_but_judged_is_an_error(self) -> None:
        """The measured worst case: a profile holding `judged` alone passed."""
        def strip(data):
            data["profiles"]["shipped_bf16"] = {"judged": True}

        script = _with_bounds(self.tmp, strip)
        done = _run(_dataset(self.tmp / "bare", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("VERDICT ERROR", done.stdout)

    def test_a_mistyped_judging_key_is_an_error(self) -> None:
        """A typo is not a bound, and it used to read as one."""
        def typo(data):
            profile = data["profiles"]["shipped_bf16"]
            profile["mean_rel_l2_mx"] = profile.pop("mean_rel_l2_max")

        script = _with_bounds(self.tmp, typo)
        done = _run(_dataset(self.tmp / "typo", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("unknown key", done.stdout)

    def test_a_wrongly_typed_bound_is_an_error(self) -> None:
        def wrong(data):
            data["profiles"]["shipped_bf16"]["mean_rel_l2_max"] = True

        script = _with_bounds(self.tmp, wrong)
        done = _run(_dataset(self.tmp / "typed", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("which is not a number", done.stdout)

    def test_a_null_bound_must_name_its_reason(self) -> None:
        """Unapplying a bound is allowed. Doing it silently is not."""
        def blank(data):
            data["profiles"]["shipped_bf16"]["mean_cos_min"] = None

        script = _with_bounds(self.tmp, blank)
        done = _run(_dataset(self.tmp / "null1", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("without naming a reason", done.stdout)

    def test_a_declared_null_bound_is_honoured_and_reported(self) -> None:
        def blank(data):
            profile = data["profiles"]["shipped_bf16"]
            profile["mean_cos_min"] = None
            profile["unbounded"]["mean_cos_min"] = "declared by this test"

        script = _with_bounds(self.tmp, blank)
        done = _run(_dataset(self.tmp / "null2"), script=script)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("mean_cos: NOT BOUNDED (declared null)", done.stdout)

    def test_every_stage_key_must_be_declared(self) -> None:
        for key in sorted(CMP.STAGE_KEYS):
            with self.subTest(key=key):
                def drop(data, key=key):
                    del data["profiles"]["shipped_bf16"]["stages"]["vit"][key]

                script = _with_bounds(self.tmp, drop, name="stage-" + key)
                done = _run(_dataset(self.tmp / ("s-" + key), mode="drift"),
                            script=script)
                self.assertEqual(done.returncode, 4, done.stdout)
                self.assertIn("does not DECLARE", done.stdout)

    def test_every_stage_must_be_declared(self) -> None:
        """THE MEMBERSHIP of `stages`, not the keys inside one rule.

        `PROFILE_KEYS` forced the `stages` KEY to exist and constrained nothing
        about what was IN it, so deleting a whole stage RULE was the one way
        left to drop a bound while declaring nothing -- every other judging key
        had to be written or the run was ERROR. Measured on a leg whose vit file
        was 100x wrong: dropping the `vit` rule printed `stage 'vit': NOT
        REQUIRED by this profile`, `VERDICT PASS` and exited 0, while the same
        data with the rule present exited 1 at 9900.3609%.
        """
        for stage in CMP.STAGES:
            with self.subTest(stage=stage):
                def drop(data, stage=stage):
                    del data["profiles"]["shipped_bf16"]["stages"][stage]

                script = _with_bounds(self.tmp, drop, name="norule-" + stage)
                done = _run(_dataset(self.tmp / ("r-" + stage), mode="drift"),
                            script=script)
                self.assertEqual(done.returncode, 4,
                                 "a dropped stage RULE was judged by nothing:\n"
                                 + done.stdout)
                self.assertIn("does not DECLARE stage", done.stdout)

    def test_an_emptied_stages_map_is_an_error(self) -> None:
        """The measured worst case: every stage rule gone at once."""
        def strip(data):
            data["profiles"]["shipped_bf16"]["stages"] = {}

        script = _with_bounds(self.tmp, strip)
        done = _run(_dataset(self.tmp / "nostages", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("does not DECLARE stage", done.stdout)

    def test_a_stage_declared_absent_must_name_its_reason(self) -> None:
        """Dropping a stage is allowed. Doing it silently is not."""
        def blank(data):
            data["profiles"]["shipped_bf16"]["stages"]["vit"] = None

        script = _with_bounds(self.tmp, blank)
        done = _run(_dataset(self.tmp / "nullvit", mode="drift"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("declared null without naming a reason", done.stdout)

    def test_a_declared_absent_stage_is_honoured_and_reported(self) -> None:
        """The CLI profile's real shape: that oracle writes no stage dumps."""
        def blank(data):
            profile = data["profiles"]["shipped_bf16"]
            profile["stages"]["vit"] = None
            profile["unbounded"]["stages.vit"] = "declared by this test"

        script = _with_bounds(self.tmp, blank)
        directory = _dataset(self.tmp / "nullvit2")
        os.remove(directory / "ours-lp0-vit.f32")
        os.remove(directory / "oracle-lp0-vit.f32")
        done = _run(directory, script=script)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn(
            "stage 'vit': NOT REQUIRED (declared null): declared by this test",
            done.stdout)

    def test_a_diagnostic_only_stage_must_still_be_present(self) -> None:
        """`diagnostic_only` used to skip the presence check as well.

        Measured at 6e55cc113 for `shipped_bf16`: the `cells` files removed
        exited 0 PASS, against a bounds file that says its presence is reported.
        """
        directory = _dataset(self.tmp / "nocells")
        os.remove(directory / "ours-lp0-cells.f32")
        os.remove(directory / "oracle-lp0-cells.f32")
        done = _run(directory)
        self.assertNotEqual(done.returncode, 0,
                            "an absent diagnostic stage passed:\n" + done.stdout)
        self.assertIn("stage 'cells' is ABSENT", done.stdout)

    def test_a_diagnostic_only_stage_is_reported_not_judged(self) -> None:
        """The declared behaviour, so the silence is deliberate and visible."""
        directory = _dataset(self.tmp / "bigcells")
        _stage(directory, "lp0", "cells", 100, 100.0)
        done = _run(directory)
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("stage 'cells': PRESENT and reported, magnitude NOT judged",
                      done.stdout)

    def test_the_f32_input_stage_is_presence_only_and_says_so(self) -> None:
        """Finding 3: it was bounded by nothing and the file implied otherwise.

        Measured at 6e55cc113: an `input` stage 100x wrong on the f32 leg exited
        0 PASS with no BOUND line at all, while a `vit` stage 100x wrong exited
        1. The bounds file now declares each of that stage's magnitude keys as
        null with its reason, so the run SAYS the stage is presence-only, and
        presence is enforced.
        """
        directory = _dataset(self.tmp / "f32in", tag="f32", factor=1.005)
        _stage(directory, "f32", "input", 64, 100.0)
        done = _run(directory, tag="f32")
        self.assertEqual(done.returncode, 0, done.stdout)
        self.assertIn("stage 'input' mean_rel_l2: NOT BOUNDED (declared null)",
                      done.stdout)
        os.remove(directory / "ours-f32-input.f32")
        os.remove(directory / "oracle-f32-input.f32")
        gone = _run(directory, tag="f32")
        self.assertNotEqual(gone.returncode, 0, gone.stdout)
        self.assertIn("stage 'input' is ABSENT", gone.stdout)

    def test_near_parallel_rows_cannot_carry_the_ordering_claim(self) -> None:
        """The permutation condition needs the reference rows to be separable.

        Measured: a smooth ramp separates DIFFERENT rows by 7e-7 in cosine while
        rounding a row to bf16 moves each element by about 0.4%, so the argmax
        is decided by noise. That reads as a false red on clean data and as a
        false GREEN on genuinely permuted output, and the comparator asserted
        nothing about it.
        """
        done = _run(_dataset(self.tmp / "parallel", mode="parallel"))
        self.assertNotEqual(done.returncode, 0,
                            "an arbitrary argmax passed as ordering evidence:\n"
                            + done.stdout)
        self.assertIn("min_best_margin", done.stdout)
        self.assertIn("not separable enough", done.stdout)

    def test_a_photographic_geometry_is_not_a_separability_failure(self) -> None:
        """THE FALSE RED THE WITHDRAWN CONSTANT 0.01 WOULD HAVE FIRED.

        `min_best_margin` is a MIN over the cells, so one flat pair decides a
        run, and flat regions are what a photograph is full of. Measured with
        the shipped `best_match()` at D=1280: a shared global component with 10%
        and 5% per-cell detail gives margins 0.00834 and 0.00210, and a 20-cell
        flat region gives 0.00875 and 0.00221. All four are UNDER the withdrawn
        0.01, and in every one the identity was still best for 100 of 100 rows,
        so the constant would have red a dataset whose ordering claim was right.
        The bound is now this dataset's own bf16 rounding scale.
        """
        for kind in ("random", "shared-0.10", "shared-0.05",
                     "flat-0.10", "flat-0.05"):
            with self.subTest(geometry=kind):
                directory = _photo_dataset(self.tmp / ("photo-" + kind), kind)
                done = _run(directory)
                self.assertEqual(
                    done.returncode, 0,
                    "a correct photographic geometry was red:\n" + done.stdout)
                permutation = json.loads(
                    (directory / "report-lp0.json").read_text())["permutation"]
                # The ordering claim the margin guards HOLDS in every one of
                # these, which is what makes a red on them a false one.
                self.assertEqual(permutation["identity_is_best"], 100)
                self.assertGreater(permutation["min_best_margin"],
                                   permutation["bf16_rounding_scale"])
                if kind.endswith("0.05"):
                    # RED-BEFORE, executably: these two sit under the constant.
                    self.assertLess(permutation["min_best_margin"], 0.01)

    def test_the_margin_is_reported_on_a_clean_leg(self) -> None:
        directory = _dataset(self.tmp / "margin")
        done = _run(directory)
        self.assertEqual(done.returncode, 0, done.stdout)
        permutation = json.loads(
            (directory / "report-lp0.json").read_text())["permutation"]
        # The bound is DERIVED from this dataset, so the report must carry the
        # scale it was judged against, that scale must be a real quantity, and
        # the margin must clear it. Asserting a CONSTANT here is what the 0.01
        # bound did, and it is what made four correct photographic geometries
        # red while their ordering claim held.
        self.assertGreater(permutation["bf16_rounding_scale"], 0.0)
        self.assertGreater(permutation["min_best_margin"],
                           permutation["bf16_rounding_scale"])

    def test_the_run_says_what_it_judged(self) -> None:
        """`.agents/verification.md`: an instrument states what it measured.

        The output was `VERDICT PASS tag=lp0 profile=shipped_bf16` and nothing
        else, so a stage that went unjudged looked exactly like one that passed.
        """
        done = _run(_dataset(self.tmp / "narrate"))
        self.assertEqual(done.returncode, 0, done.stdout)
        for line in ("JUDGED profile 'shipped_bf16', judged",
                     "JUDGED sentinels: all 4 kinds required exactly bf16(oracle)",
                     "JUDGED image_rows mean_rel_l2: bound <= 0.049",
                     "JUDGED image_rows mean_cos: bound >= 0.998",
                     "JUDGED stage 'input' bf16_of_oracle_exact",
                     "JUDGED stage 'vit' mean_rel_l2: bound <= 0.0307",
                     "JUDGED stage 'cells': PRESENT and reported",
                     # THE OUTPUT MUST SAY WHAT KIND OF BOUND THIS IS. It read
                     # `margin bound >= 0.01` with nothing saying the number was
                     # declared rather than measured, unlike every `NOT BOUNDED
                     # (declared null)` line beside it.
                     "JUDGED permutation: best-match margin bound > ",
                     "DERIVED from this run's own rows"):
            self.assertIn(line, done.stdout)

    def test_undefined_profile_is_an_error(self) -> None:
        def repoint(data):
            data["tag_rules"] = [["lp0", "no_such_profile"]] + data["tag_rules"]

        script = _with_bounds(self.tmp, repoint)
        done = _run(_dataset(self.tmp / "d2"), script=script)
        self.assertEqual(done.returncode, 4, done.stdout)
        self.assertIn("does not define", done.stdout)

    def test_unmatched_tag_stays_unjudged(self) -> None:
        """The prior repair's guarantee must survive this one."""
        directory = _dataset(self.tmp / "unk")
        shutil.copy(directory / "ours-lp0-block.f32", directory / "ours-zz-block.f32")
        shutil.copy(directory / "oracle-lp0-block.f32", directory / "oracle-zz-block.f32")
        done = _run(directory, tag="zz")
        self.assertEqual(done.returncode, 3, done.stdout)
        self.assertIn("VERDICT UNJUDGED", done.stdout)


# The recorded shape of the W7-CUDA run this row already measured on `thor`:
# `ctest -R 'deepseek_v4|clip_mmproj_gguf'` reported 24 of 27 passed
# (`.agents/specs/deepseek-v4-flash-vision.md:1370`), the three failures being
# the ones the spec attributes at :1006-1018, :1019-1024 and :994-1004.
CTEST_RECORDED = """\
The following tests FAILED:
\t  7 - test_deepseek_v4_mm_reach (Failed)
\t 11 - test_deepseek_v4_mm_chat (Failed)
\t 19 - test_serve_deepseek_v4_mm (Timeout)
Errors while running CTest
"""
# The dev_attn refusal as `deepseek_v4.cpp:1325` emits it, quoted by the spec at
# :1377-1383.
DEV_ATTN_RECORDED = (
    "deepseek-v4 attention: layer 0 runs the DEVICE decode kernel at "
    "sliding_window 128. ... Refused by name; the windowed device kernel is "
    "owed by issue #2411. Unset VT_V4_DEVICE_ATTN to take the host arm\n")

# Each driver's readback block is self-contained: it reads only $OUT (and
# $RUN_CLI in the parity driver) and runs entirely on the steps file and the
# artefacts the job left beside it. Extracting it from its own `EXPECTED=` line
# to the end of the file -- plus the shipped `refusal_recorded` function where
# the driver defines one -- and executing THAT is what makes these cases test
# the shipped block rather than a copy of its logic.
DRIVERS = {
    "dsv4v_w6_parity.sh": {
        "steps": "steps.txt", "extra": [], "banner": "W6_PARITY_DONE",
        "failed": "W6_PARITY_FAILED", "nonzero": {}, "artefacts": {}},
    "dsv4v_w6_floor.sh": {
        "steps": "floor-steps.txt", "extra": [], "banner": "W6_FLOOR_DONE",
        "failed": "W6_FLOOR_FAILED", "nonzero": {}, "artefacts": {}},
    "dsv4v_w6_f32.sh": {
        "steps": "f32-steps.txt", "extra": [], "banner": "W6_F32_DONE",
        "failed": "W6_F32_FAILED", "nonzero": {}, "artefacts": {}},
    "dsv4v_w7_cuda.sh": {
        "steps": "steps.txt",
        "extra": ["compare_cuda_lp0", "compare_cuda_lp1",
                  "compare_cuda_lp2", "compare_cuda_lp3"],
        "banner": "W7_CUDA_DONE", "failed": "W7_CUDA_FAILED",
        # The outcomes the spec RECORDS for this box, which a bare
        # "any non-zero fails" rule would have turned into a red run.
        "nonzero": {"ctest_cuda": 8, "dev_attn_on": 1},
        "artefacts": {"ctest-cuda.log": CTEST_RECORDED,
                      "dev-attn-on.log": DEV_ATTN_RECORDED}},
}


def _readback_block(driver: str) -> str:
    text = (PARITY / driver).read_text()
    prefix = ""
    if "refusal_recorded() {" in text:
        start = text.index("refusal_recorded() {")
        prefix = text[start:text.index("\n}\n", start) + 3]
    return prefix + text[text.index("EXPECTED="):]


class DriverStepReadback(unittest.TestCase):
    """A step that never ran must not be indistinguishable from one that passed."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dsv4v-steps-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _expected(self, driver: str) -> list[str]:
        text = (PARITY / driver).read_text()
        match = re.search(r'EXPECTED="([^"]*)"', text)
        assert match, driver
        return match.group(1).split() + DRIVERS[driver]["extra"]

    def _recorded(self, driver: str) -> list[str]:
        """The step lines of a run that went exactly as the record says."""
        nonzero = DRIVERS[driver]["nonzero"]
        return ["%s RC=%d" % (s, nonzero.get(s, 0)) for s in self._expected(driver)]

    def _run(self, driver: str, lines: list[str],
             artefacts: dict | None = None) -> subprocess.CompletedProcess:
        block = self.tmp / (driver + ".readback")
        block.write_text(_readback_block(driver))
        out = self.tmp / driver
        out.mkdir(exist_ok=True)
        (out / DRIVERS[driver]["steps"]).write_text(
            "".join(line + "\n" for line in lines))
        for name, body in (DRIVERS[driver]["artefacts"]
                           if artefacts is None else artefacts).items():
            (out / name).write_text(body)
        env = dict(os.environ, OUT=str(out), RUN_CLI="0")
        return subprocess.run(["bash", str(block)], capture_output=True,
                              text=True, env=env)

    def test_the_recorded_run_passes(self) -> None:
        """The repair must not red the run this row already measured."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                done = self._run(driver, self._recorded(driver))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(DRIVERS[driver]["banner"], done.stdout)
                self.assertNotIn(DRIVERS[driver]["failed"], done.stdout)

    def test_a_missing_expected_step_fails(self) -> None:
        """`awk '!/ RC=0$/' | wc -l` counted an ABSENT step as zero failures."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                done = self._run(driver, self._recorded(driver)[1:])
                self.assertNotEqual(done.returncode, 0,
                                    "a step that never ran passed:\n" + done.stdout)
                self.assertIn("MISSING", done.stdout)

    def test_an_empty_steps_file_fails(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                done = self._run(driver, [])
                self.assertNotEqual(done.returncode, 0,
                                    "an empty steps file passed:\n" + done.stdout)

    def test_a_failing_step_fails(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                lines = self._recorded(driver)
                first = self._expected(driver)[0]
                self.assertNotIn(first, DRIVERS[driver]["nonzero"])
                lines[0] = "%s RC=7" % first
                done = self._run(driver, lines)
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertIn("FAILING STEPS", done.stdout)

    def test_the_done_banner_is_unreachable_on_a_failed_run(self) -> None:
        """EXECUTED, not asserted on the text.

        `dsv4v_w6_parity.sh`, `_floor.sh` and `_f32.sh` each printed their
        `### W6_*_DONE` banner and THEN exited 1, so a log grep for the banner
        read a failed run as a finished one. The previous version of this case
        checked only that the string `[ "$BAD" -eq 0 ]` appeared in the file,
        which all four drivers satisfied while three of them still printed it.
        """
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                lines = self._recorded(driver)
                lines[0] = "%s RC=7" % self._expected(driver)[0]
                done = self._run(driver, lines)
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertNotIn(DRIVERS[driver]["banner"], done.stdout)
                self.assertIn(DRIVERS[driver]["failed"], done.stdout)

    def test_a_malformed_line_fails(self) -> None:
        """The W7-CUDA driver used to write a bare `cpu_control identical`."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                done = self._run(driver,
                                 self._recorded(driver) + ["cpu_control identical"])
                self.assertNotEqual(done.returncode, 0,
                                    "an unparsable step line passed:\n" + done.stdout)
                self.assertIn("MALFORMED", done.stdout)


class W7StepClassification(unittest.TestCase):
    """A recorded expectation must not become an excuse for any outcome."""

    DRIVER = "dsv4v_w7_cuda.sh"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dsv4v-w7-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.helper = DriverStepReadback("test_the_recorded_run_passes")
        self.helper.tmp = self.tmp

    def _run(self, steps: dict, artefacts: dict) -> subprocess.CompletedProcess:
        lines = ["%s RC=%d" % (s, steps.get(s, 0))
                 for s in self.helper._expected(self.DRIVER)]
        return self.helper._run(self.DRIVER, lines, artefacts)

    def test_the_attributed_ctest_failures_are_accepted(self) -> None:
        done = self._run({"ctest_cuda": 8, "dev_attn_on": 1},
                         DRIVERS[self.DRIVER]["artefacts"])
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ATTRIBUTED by the spec", done.stdout)

    def test_an_unattributed_ctest_failure_fails(self) -> None:
        log = CTEST_RECORDED.replace("test_deepseek_v4_mm_chat",
                                     "test_cuda_deepseek_v4")
        done = self._run({"ctest_cuda": 8, "dev_attn_on": 1},
                         {"ctest-cuda.log": log,
                          "dev-attn-on.log": DEV_ATTN_RECORDED})
        self.assertNotEqual(done.returncode, 0,
                            "an unattributed suite failure passed:\n" + done.stdout)
        self.assertIn("has NOT attributed it", done.stdout)

    def test_a_ctest_failure_with_no_named_test_fails(self) -> None:
        done = self._run({"ctest_cuda": 8, "dev_attn_on": 1},
                         {"ctest-cuda.log": "Segmentation fault\n",
                          "dev-attn-on.log": DEV_ATTN_RECORDED})
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("UNEXPLAINED ctest_cuda", done.stdout)

    def test_a_ctest_failure_with_no_log_fails(self) -> None:
        done = self._run({"ctest_cuda": 8, "dev_attn_on": 1},
                         {"dev-attn-on.log": DEV_ATTN_RECORDED})
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("no ctest-cuda.log", done.stdout)

    def test_the_refusal_is_classified_from_the_full_log(self) -> None:
        """The prefiltered excerpt DROPS two of the three refusal families.

        `dev-attn-refusal.txt` is `grep -iE 'sliding_window|image span|DEVICE
        decode|2411|refus' dev-attn-on.log | head -20`, and the classifier was
        applied to THAT FILE. Measured: a log holding `DeepSeek-V4 vision
        compute dtype must be bf16` classifies rc=0 as the full log and rc=1 as
        the filtered file, which is 0 BYTES, and `DeepseekV4 DEVICE forward
        (W7-device) not implemented` does the same. So a real product refusal
        became `UNEXPLAINED dev_attn_on`, reached `### FAILING STEPS` and failed
        the job -- on the one leg that runs on every thor lease.
        """
        for message in ("DeepSeek-V4 vision compute dtype must be bf16\n",
                        "DeepseekV4 DEVICE forward (W7-device) not implemented\n",
                        "DeepSeek-V4 vision qkv weight has the wrong dtype\n"):
            with self.subTest(message=message[:44]):
                done = self._run(
                    {"ctest_cuda": 8, "dev_attn_on": 1},
                    {"ctest-cuda.log": CTEST_RECORDED,
                     "dev-attn-on.log": message,
                     # Exactly what the shipped prefilter leaves behind: nothing.
                     "dev-attn-refusal.txt": ""})
                self.assertEqual(
                    done.returncode, 0,
                    "a real refusal was read as a crash:\n" + done.stdout)
                self.assertIn("the recorded refusal is in the log", done.stdout)

    def test_a_clean_ctest_run_is_accepted(self) -> None:
        done = self._run({"ctest_cuda": 0, "dev_attn_on": 1},
                         DRIVERS[self.DRIVER]["artefacts"])
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_dev_attn_passing_falsifies_the_record(self) -> None:
        """The POINT of that step is that the path refuses."""
        done = self._run({"ctest_cuda": 8, "dev_attn_on": 0},
                         DRIVERS[self.DRIVER]["artefacts"])
        self.assertNotEqual(done.returncode, 0,
                            "a refusal that stopped firing passed:\n" + done.stdout)
        self.assertIn("FALSIFIES the record", done.stdout)

    def test_dev_attn_failing_without_the_refusal_fails(self) -> None:
        done = self._run({"ctest_cuda": 8, "dev_attn_on": 1},
                         {"ctest-cuda.log": CTEST_RECORDED,
                          "dev-attn-on.log": "Segmentation fault\n"})
        self.assertNotEqual(done.returncode, 0,
                            "a crash was filed as the refusal:\n" + done.stdout)
        self.assertIn("UNEXPLAINED dev_attn_on", done.stdout)


class RefusalClassifier(unittest.TestCase):
    """The shipped `refusal_recorded` function, executed on real message text.

    It was `grep -iE 'refus|unsupported|share one device|must be'`, and `must be`
    matches ordinary assertion and exception text: the two CRASH lines below were
    both filed as expected refusals with RC=0.
    """

    ACCEPT = (
        "terminate called: DeepSeek-V4 vision weights must share one device",
        "DeepSeek-V4 vision compute dtype must be bf16",
        "DeepSeek-V4 vision patch dtype must equal model dtype",
        DEV_ATTN_RECORDED,
        "DeepseekV4 DEVICE forward (W7-device) not implemented - the tiny-config",
        # THE FAMILIES THE `.*must ` ANCHOR REJECTED. Counted in
        # deepseek_v4_vision.cpp: 61 distinct "DeepSeek-V4 vision*" literals, of
        # which 23 matched that anchor and 38 did not -- every ValidateTensor
        # label at :103-146 and every overflow refusal. A probe leg refused by
        # one of these was recorded `_unexplained`, reached ### FAILING STEPS
        # and failed the job, which is a false red on a real product refusal.
        "DeepSeek-V4 vision qkv weight has the wrong dtype",
        "DeepSeek-V4 vision aligner w1 bias has no storage",
        "DeepSeek-V4 vision final norm weight has the wrong shape",
        "DeepSeek-V4 vision patch count overflow",
        "DeepSeek-V4 vision RoPE cache size overflow",
        "DeepSeek-V4 vision block count does not match depth",
    )
    REJECT = (
        "Assertion failed: n must be positive",
        "terminate called after throwing an instance of 'std::out_of_range': "
        "vector index must be less than size",
        "Segmentation fault",
        "CUDA error: an illegal memory access was encountered",
        "unsupported thing happened somewhere else",
    )

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dsv4v-refusal-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        text = (PARITY / "dsv4v_w7_cuda.sh").read_text()
        start = text.index("refusal_recorded() {")
        self.fn = text[start:text.index("\n}\n", start) + 3]

    def _classify(self, message: str) -> int:
        log = self.tmp / "probe.log"
        log.write_text(message + "\n")
        script = self.tmp / "classify.sh"
        script.write_text(self.fn + '\nrefusal_recorded "$1"\n')
        return subprocess.run(["bash", str(script), str(log)]).returncode

    def test_a_product_refusal_is_recorded(self) -> None:
        for message in self.ACCEPT:
            with self.subTest(message=message[:40]):
                self.assertEqual(self._classify(message), 0)

    def test_a_crash_is_not_a_refusal(self) -> None:
        for message in self.REJECT:
            with self.subTest(message=message[:40]):
                self.assertNotEqual(
                    self._classify(message), 0,
                    "a crash would be filed as the expected refusal")


class DriverContract(unittest.TestCase):
    """The properties every driver must keep to be able to fail at all."""

    def test_every_driver_parses(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                done = subprocess.run(["bash", "-n", str(PARITY / driver)],
                                      capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)

    def test_every_driver_sets_pipefail(self) -> None:
        """Without it `cmd | tee f; step name $?` records TEE's status."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                text = (PARITY / driver).read_text()
                self.assertRegex(text, r"(?m)^set -uo pipefail$")

    def test_every_load_bearing_control_records_a_step(self) -> None:
        """The byte-for-byte controls the file headers call load-bearing.

        Each `cmp` used to print a message and record nothing, so a build that
        did NOT reproduce the recorded block still ended green.
        """
        for driver, marker in (("dsv4v_w6_floor.sh", "oracle_reproducible"),
                               ("dsv4v_w6_f32.sh", "control_identical"),
                               ("dsv4v_w7_cuda.sh", "cpu_control")):
            with self.subTest(driver=driver):
                self.assertIn("step %s" % marker, (PARITY / driver).read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
