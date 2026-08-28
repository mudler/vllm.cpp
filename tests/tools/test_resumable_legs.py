"""The leg ledger's rules, gated on the CPU.

Every rule here was learned by losing a measurement to it on 2026-08-28: three
crashes killed the same experiment (#545), two A/B sequences self-invalidated on
a terminal control, and two comparisons were built across a reboot before anyone
noticed. These rules decide whether a GPU run's output means anything, so they
must not need a GPU to check -- the polarity `gpu_clock_state.py` chose.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from tools.bench.resumable_legs import (
    append_leg,
    fold,
    plan,
    read_ledger,
    remaining,
    terminal_check,
)


def leg(arm: str, boot: str = "aaaa", **kw: object) -> dict[str, object]:
    rec: dict[str, object] = {"arm": arm, "boot_id": boot}
    rec.update(kw)
    return rec


class PlanTest(unittest.TestCase):
    def test_interleaves_and_appends_a_terminal_control(self) -> None:
        # NOT AABB: a block of one arm then the other measures the hour as much
        # as the change, which is how two sequences self-invalidated.
        self.assertEqual(plan(["on", "off"], 3),
                         ["on", "off", "on", "off", "on", "off", "on"])

    def test_a_single_arm_gets_no_control(self) -> None:
        self.assertEqual(plan(["only"], 2), ["only", "only"])

    def test_zero_legs_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            plan(["on", "off"], 0)


class ResumeTest(unittest.TestCase):
    def test_returns_only_the_owed_legs(self) -> None:
        order = plan(["on", "off"], 2)          # on off on off on
        done = [leg("on"), leg("off"), leg("on")]
        self.assertEqual(remaining(order, done), ["off", "on"])

    def test_a_torn_tail_from_a_crash_is_dropped_not_fatal(self) -> None:
        # A crash mid-append leaves half a line. The ledger must still be
        # readable, or the leg in flight takes every completed leg with it.
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "legs.jsonl"
            append_leg(p, leg("on", fwd=31.4))
            with p.open("a", encoding="utf-8") as fh:
                fh.write('{"arm": "off", "boot_i')
            got = read_ledger(p)
            self.assertEqual(len(got), 1)
            self.assertEqual(got[0]["arm"], "on")

    def test_a_leg_without_its_boot_is_refused(self) -> None:
        # #545: a reading with no boot cannot be attributed later, which is
        # exactly how two comparisons were built across a reboot.
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "legs.jsonl"
            with self.assertRaises(ValueError) as ctx:
                append_leg(p, {"arm": "on", "fwd": 1.0})
            self.assertIn("boot", str(ctx.exception))


class FoldTest(unittest.TestCase):
    def test_REFUSES_across_a_boot_change(self) -> None:
        done = [leg("on", "aaaa", fwd=31.0), leg("on", "aaaa", fwd=31.4),
                leg("off", "bbbb", fwd=35.2), leg("off", "bbbb", fwd=35.0)]
        out = fold(done, "fwd")
        self.assertFalse(out["admissible"])
        self.assertTrue(any("boot" in r for r in out["reasons"]))
        self.assertEqual(out["boots"], ["aaaa", "bbbb"])

    def test_refuses_an_arm_with_one_leg(self) -> None:
        done = [leg("on", fwd=31.0), leg("on", fwd=31.4), leg("off", fwd=35.2)]
        out = fold(done, "fwd")
        self.assertFalse(out["admissible"])
        self.assertTrue(any("leg" in r for r in out["reasons"]))

    def test_reports_medians_and_spread_when_admissible(self) -> None:
        done = [leg("on", fwd=31.5), leg("off", fwd=35.2),
                leg("on", fwd=31.1), leg("off", fwd=35.0)]
        out = fold(done, "fwd")
        self.assertTrue(out["admissible"], out["reasons"])
        self.assertAlmostEqual(out["arms"]["on"]["median"], 31.3)
        self.assertEqual(out["arms"]["off"]["n"], 2)
        self.assertLess(out["arms"]["on"]["spread_pct"], 2.0)

    def test_an_absent_metric_is_a_refusal_not_an_empty_pass(self) -> None:
        out = fold([leg("on", fwd=1.0), leg("on", fwd=2.0)], "tok")
        self.assertFalse(out["admissible"])
        self.assertTrue(any("tok" in r for r in out["reasons"]))


class TerminalControlTest(unittest.TestCase):
    def test_catches_a_drifting_box(self) -> None:
        # The real shape: the opening arm read 51.64 and the SAME arm read 36.44
        # at the end, so nothing between them was comparable.
        done = [leg("y", tok=51.64), leg("n", tok=35.58), leg("y", tok=36.44)]
        out = terminal_check(done, "tok", tolerance_pct=6.0)
        self.assertTrue(out["checked"])
        self.assertFalse(out["ok"])
        self.assertGreater(out["drift_pct"], 25)

    def test_passes_when_the_control_matches(self) -> None:
        # The L2 run: fwd 31.48 then 31.15 on the same arm, 1.1% apart.
        done = [leg("on", fwd=31.48), leg("off", fwd=35.19), leg("on", fwd=31.15)]
        out = terminal_check(done, "fwd", tolerance_pct=6.0)
        self.assertTrue(out["checked"])
        self.assertTrue(out["ok"])
        self.assertLess(out["drift_pct"], 2.0)

    def test_says_so_when_there_is_no_control(self) -> None:
        done = [leg("on", fwd=31.4), leg("off", fwd=35.2)]
        out = terminal_check(done, "fwd", tolerance_pct=6.0)
        self.assertFalse(out["checked"])
        self.assertIn("no terminal control", out["reason"])


if __name__ == "__main__":
    unittest.main()
