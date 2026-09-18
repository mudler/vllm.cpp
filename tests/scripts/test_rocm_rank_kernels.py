#!/usr/bin/env python3
"""Behavior tests for the ROCm decode kernel-time ranking tool.

Every case here is about a way the tool must REFUSE, or about a window boundary
it must honour. A ranking tool that prints a plausible table from a bad window is
the worst possible outcome, because nothing downstream can tell.

The real-trace cases read the committed gfx1151 capture, so they need no GPU.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/rocm-rank-kernels.py"
REAL = (ROOT / "docs/bench-evidence/strix-kernel-trace-3015-20260907"
        / "diag-kernel-control-asaj36ta--trace--e5367aefafa8--1_kernel_trace.csv.gz")

COLUMNS = ["Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp"]


def load_tool():
    spec = importlib.util.spec_from_file_location("rocm_rank_kernels", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_trace(path: Path, rows: list[tuple[str, int, int]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for name, start, end in rows:
            writer.writerow(["KERNEL_DISPATCH", name, start, end])


def synthetic(steps: int, per_step: int, *, load_rows: int = 7,
              sampler: str = "ArgmaxK", ragged_at: int | None = None,
              invert_at: int | None = None) -> list[tuple[str, int, int]]:
    """A trace shaped like a decode: load rows, then `steps` identical steps.

    Each step is the sampler dispatch followed by `per_step - 1` body dispatches.
    `Hot` is deliberately the most expensive kernel so a ranking has a known top.
    """
    rows: list[tuple[str, int, int]] = []
    clock = 1_000_000
    for i in range(load_rows):
        rows.append(("LoadK", clock, clock + 500))
        clock += 1_000
    for step in range(steps):
        rows.append((sampler, clock, clock + 300))
        clock += 1_000
        body = per_step - 1
        if ragged_at is not None and step == ragged_at:
            body += 1
        for j in range(body):
            name = "Hot" if j % 2 == 0 else "Cold"
            cost = 10_000 if name == "Hot" else 1_000
            rows.append((name, clock, clock + cost))
            clock += cost + 100
    if invert_at is not None:
        name, start, end = rows[invert_at]
        rows[invert_at] = (name, end, start)
    return rows


class Refusals(unittest.TestCase):
    """Each case names the guarantee it convicts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_tool(self, rows, *args) -> tuple[int, str, str]:
        path = self.dir / "t.csv"
        write_trace(path, rows)
        out, err = io.StringIO(), io.StringIO()
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = self.tool.main([str(path), *args])
        finally:
            sys.stdout, sys.stderr = old
        return code, out.getvalue(), err.getvalue()

    def test_G_REFUSE_MARKS_a_sampler_that_never_fires_is_refused(self) -> None:
        code, _, err = self.run_tool(synthetic(6, 9), "--sampler", "Absent")
        self.assertEqual(code, 2, err)
        self.assertIn("only 0 'Absent' dispatches", err)

    def test_G_REFUSE_MARKS_too_few_steps_to_bound_is_refused(self) -> None:
        # 3 marks, skipping 1, leaves 1 step: not enough to bound a wall time.
        code, _, err = self.run_tool(synthetic(3, 9), "--skip-first", "1")
        self.assertEqual(code, 2, err)

    def test_G_REFUSE_RAGGED_an_uneven_step_is_refused_not_ranked(self) -> None:
        code, out, err = self.run_tool(synthetic(6, 9, ragged_at=3))
        self.assertEqual(code, 3, err)
        self.assertIn("ragged", err)
        self.assertNotIn("share%", out)

    def test_G_SWAP_UNKNOWN_an_unsupplied_warning_count_is_not_reported_as_zero(self) -> None:
        # A missing count and a count of zero are different claims. Printing
        # "0.00%" for "you did not tell me" would read as "#3040 did not fire
        # here", which is an instrument succeeding at the wrong question.
        code, out, err = self.run_tool(synthetic(6, 9))
        self.assertEqual(code, 0, err)
        self.assertIn("NOT SUPPLIED", out)
        self.assertNotIn("0.0000%", out)

    def test_G_SWAP_BOUND_a_supplied_count_produces_the_worst_case_share(self) -> None:
        code, out, err = self.run_tool(synthetic(6, 9), "--swap-warnings", "2")
        self.assertEqual(code, 0, err)
        self.assertIn("swap warnings", out)
        self.assertNotIn("NOT SUPPLIED", out)

    def test_G_REFUSE_INVERTED_a_negative_duration_is_refused(self) -> None:
        # This is #3040's documented failure: the SDK adjusts inverted stamps and
        # does not say which. One that survives must never enter a share.
        code, out, err = self.run_tool(synthetic(6, 9, invert_at=20))
        self.assertEqual(code, 4, err)
        self.assertIn("End < Start", err)
        self.assertNotIn("share%", out)


class Windows(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()

    def test_G_EXCLUDE_LOAD_rows_before_the_first_mark_are_not_ranked(self) -> None:
        # LoadK is the most numerous kernel in the file and must not appear at all:
        # a window that leaked prefill in would rank it.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            write_trace(path, synthetic(6, 9, load_rows=40))
            rows = self.tool.read_trace(str(path))
            segments = self.tool.segment(rows, "ArgmaxK", 1)
            result = self.tool.rank(rows, segments)
        self.assertNotIn("LoadK", [e["kernel"] for e in result["table"]])
        self.assertEqual(result["table"][0]["kernel"], "Hot")

    def test_G_SKIP_FIRST_discards_exactly_the_requested_leading_steps(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            write_trace(path, synthetic(10, 9))
            rows = self.tool.read_trace(str(path))
            kept = {n: len(self.tool.segment(rows, "ArgmaxK", n)) for n in (0, 1, 3)}
        # 10 marks give 9 inter-mark segments; each skip removes exactly one.
        self.assertEqual(kept, {0: 9, 1: 8, 3: 6})

    def test_G_SHARES_sum_to_one_hundred_percent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.csv"
            write_trace(path, synthetic(8, 9))
            rows = self.tool.read_trace(str(path))
            result = self.tool.rank(rows, self.tool.segment(rows, "ArgmaxK", 1))
        self.assertAlmostEqual(sum(e["share_pct"] for e in result["table"]), 100.0, places=6)


class CommittedGfx1151Trace(unittest.TestCase):
    """The real capture. These numbers are what narrows #3040."""

    @classmethod
    def setUpClass(cls) -> None:
        if not REAL.exists():
            raise unittest.SkipTest(f"committed trace absent: {REAL}")
        cls.tool = load_tool()
        cls.rows = cls.tool.read_trace(str(REAL))

    def test_the_capture_has_no_inverted_or_zero_length_row(self) -> None:
        d = self.tool.durations(self.rows)
        self.assertEqual(len(d), 85737)
        self.assertEqual(sum(1 for x in d if x < 0), 0)
        self.assertEqual(sum(1 for x in d if x == 0), 0)

    def test_ArgmaxK_delimits_sixty_four_tokens_into_identical_steps(self) -> None:
        segments = self.tool.segment(self.rows, "ArgmaxK", 0)
        self.assertEqual(len(segments), 63)
        self.assertEqual({b - a for a, b in segments}, {1330})

    def test_the_swap_bound_is_the_documented_two_point_seven_five_percent(self) -> None:
        d = self.tool.durations(self.rows)
        bound = (62 * max(d)) / sum(d)
        self.assertAlmostEqual(bound * 100.0, 2.75, delta=0.05)

    def test_kernel_busy_is_a_plausible_share_of_the_step_wall(self) -> None:
        segments = self.tool.segment(self.rows, "ArgmaxK", 1)
        result = self.tool.rank(self.rows, segments)
        # A trace of garbage timestamps would not land inside a physical band.
        self.assertGreater(result["occupancy_pct"], 80.0)
        self.assertLess(result["occupancy_pct"], 100.5)


if __name__ == "__main__":
    unittest.main()
