#!/usr/bin/env python3
"""Mutation checks for the DeepSeek-V4 vision parity harness.

Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411.

`tools/parity/dsv4v_w6_compare.py` judges a parity leg and exits on its verdict,
and the four drivers beside it record a step per leg and read those steps back.
Both claims were made before and neither held: the comparator returned PASS on
data that should fail, and the readback could not tell a step that never ran
from one that passed. This file PERFORMS each failure and fails if the harness
shrugs it off.

Every case below was RED against the harness as it stood at eb6009f6c:

  a zero image row on both sides   VERDICT PASS, exit 0, while 99 other rows
                                   were 50% off
  a stage 100x wrong               VERDICT PASS, exit 0
  a stage absent entirely          VERDICT PASS, exit 0
  a stage shape-mismatched         VERDICT PASS, exit 0
  a profile with no `judged` key   VERDICT DIAGNOSTIC, exit 0, on 50%-off data
  a steps file missing a step      failed_steps=0, exit 0
  an empty steps file              failed_steps=0, exit 0

THE FIXTURE ROWS MUST BE PAIRWISE DISTINCT IN DIRECTION. A first version built
them from a smooth ramp, which made every image row near-parallel (cosine
0.9999988 between DIFFERENT rows) while rounding to bf16 moves each element by
about 0.4%. The rounding noise swamped the angular separation, the argmax in
best_match() became arbitrary, and the identity-permutation condition failed on
every dataset including the clean one -- so each case exited 1 for a reason that
had nothing to do with what it meant to test. Independent pseudo-random
directions separate by far more than the rounding noise.
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


def _block(outdir: Path, tag: str, lead_pad: int, mode: str) -> None:
    """Write a token block. `mode` selects the defect under test."""
    types, _cell = CMP.layout(lead_pad, 10, 10)
    rng = _LCG(20260912)
    factor = 1.02 if mode == "clean" else 1.5
    ours, ref = [], []
    seen = 0
    for i, kind in enumerate(types):
        if kind == "IMAGE":
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


def _dataset(outdir: Path, mode: str = "clean", tag: str = "lp0") -> Path:
    """A faithful leg: a block plus the three stage dumps a real run writes."""
    outdir.mkdir(parents=True, exist_ok=True)
    _block(outdir, tag, 0, mode)
    _stage(outdir, tag, "input", 64, 1.0)     # exactly bf16(oracle)
    _stage(outdir, tag, "vit", 64, 1.02)      # inside the recorded 3.07%
    _stage(outdir, tag, "cells", 100, 1.02)
    return outdir


def _run(directory: Path, tag: str = "lp0", script: Path = COMPARE,
         lead_pad: int = 0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), str(directory), tag, str(lead_pad), "10", "10"],
        capture_output=True, text=True)


def _with_bounds(tmp: Path, mutate) -> Path:
    """A copy of the comparator whose bounds file has been mutated beside it."""
    home = tmp / "mutated"
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


# Each driver's readback block is self-contained: it reads only $OUT (and
# $RUN_CLI in the parity driver) and runs entirely on the steps file. Extracting
# it from its own `EXPECTED=` line to the end of the file and executing THAT is
# what makes these cases test the shipped block rather than a copy of its logic.
DRIVERS = {
    "dsv4v_w6_parity.sh": ("steps.txt", []),
    "dsv4v_w6_floor.sh": ("floor-steps.txt", []),
    "dsv4v_w6_f32.sh": ("f32-steps.txt", []),
    "dsv4v_w7_cuda.sh": ("steps.txt",
                         ["compare_cuda_lp0", "compare_cuda_lp1",
                          "compare_cuda_lp2", "compare_cuda_lp3"]),
}


class DriverStepReadback(unittest.TestCase):
    """A step that never ran must not be indistinguishable from one that passed."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dsv4v-steps-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _block_and_steps(self, driver: str) -> tuple[Path, list[str]]:
        text = (PARITY / driver).read_text()
        start = text.index("EXPECTED=")
        block = self.tmp / (driver + ".readback")
        block.write_text(text[start:])
        match = re.search(r'EXPECTED="([^"]*)"', text)
        assert match, driver
        expected = match.group(1).split() + DRIVERS[driver][1]
        return block, expected

    def _run(self, driver: str, lines: list[str]) -> subprocess.CompletedProcess:
        block, _ = self._block_and_steps(driver)
        out = self.tmp / driver
        out.mkdir(exist_ok=True)
        (out / DRIVERS[driver][0]).write_text("".join(l + "\n" for l in lines))
        env = dict(os.environ, OUT=str(out), RUN_CLI="0")
        return subprocess.run(["bash", str(block)], capture_output=True,
                              text=True, env=env)

    def test_complete_run_passes(self) -> None:
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                _, expected = self._block_and_steps(driver)
                done = self._run(driver, ["%s RC=0" % s for s in expected])
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_a_missing_expected_step_fails(self) -> None:
        """`awk '!/ RC=0$/' | wc -l` counted an ABSENT step as zero failures."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                _, expected = self._block_and_steps(driver)
                done = self._run(driver, ["%s RC=0" % s for s in expected[1:]])
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
                _, expected = self._block_and_steps(driver)
                lines = ["%s RC=0" % s for s in expected]
                lines[-1] = "%s RC=7" % expected[-1]
                done = self._run(driver, lines)
                self.assertNotEqual(done.returncode, 0, done.stdout)
                self.assertIn("FAILING STEPS", done.stdout)

    def test_a_malformed_line_fails(self) -> None:
        """The W7-CUDA driver used to write a bare `cpu_control identical`."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                _, expected = self._block_and_steps(driver)
                lines = ["%s RC=0" % s for s in expected] + ["cpu_control identical"]
                done = self._run(driver, lines)
                self.assertNotEqual(done.returncode, 0,
                                    "an unparsable step line passed:\n" + done.stdout)
                self.assertIn("MALFORMED", done.stdout)


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

    def test_the_done_banner_is_guarded(self) -> None:
        """A DONE banner must not be reachable when a step failed."""
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                text = (PARITY / driver).read_text()
                self.assertIn('[ "$BAD" -eq 0 ]', text)

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
