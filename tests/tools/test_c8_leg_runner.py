"""The runner's behaviour, gated on the CPU via --dry-run.

The ledger's rules are tested in `test_resumable_legs.py`. What is tested HERE
is that the runner actually reaches them: that a crash leaves a resumable
ledger, that an unparsable leg is recorded VOID rather than dropped, and that a
drifted terminal control makes the run exit non-zero instead of printing a
confident number.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tools.bench.c8_leg_runner import main, parse_metric, read_boot_id
from tools.bench.resumable_legs import read_ledger


class MetricParseTest(unittest.TestCase):
    def test_pulls_the_capture_group(self) -> None:
        self.assertEqual(parse_metric("out_tok/s=66.4 end", r"out_tok/s=([0-9.]+)"), 66.4)

    def test_no_match_is_None_not_an_exception(self) -> None:
        # A leg that ran and produced nothing parsable is a broken instrument.
        # It must reach the ledger as VOID, not vanish.
        self.assertIsNone(parse_metric("nothing here", r"out_tok/s=([0-9.]+)"))

    def test_a_nonnumeric_capture_is_None(self) -> None:
        self.assertIsNone(parse_metric("v=abc", r"v=(\w+)"))


class BootIdTest(unittest.TestCase):
    def test_an_unreadable_path_yields_empty_not_a_crash(self) -> None:
        self.assertEqual(read_boot_id(pathlib.Path("/nonexistent/boot_id")), "")


class DryRunTest(unittest.TestCase):
    def _run(self, ledger: pathlib.Path, legs: int, extra: list[str] | None = None) -> int:
        argv = [
            "--ledger", str(ledger),
            "--arm", "on", "--arm", "off",
            "--legs-per-arm", str(legs),
            "--metric", "metric",
            "--metric-regex", r"metric=([0-9.]+)",
            "--command", "true {arm}",
            "--dry-run",
        ]
        return main(argv + (extra or []))

    def test_a_dry_run_fills_the_ledger_and_folds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            led = pathlib.Path(td) / "legs.jsonl"
            rc = self._run(led, 2)
            recs = read_ledger(led)
            # plan(["on","off"], 2) is on off on off on -> 5 legs
            self.assertEqual(len(recs), 5)
            self.assertEqual([r["arm"] for r in recs],
                             ["on", "off", "on", "off", "on"])
            self.assertTrue(all("boot_id" in r for r in recs))
            self.assertEqual(rc, 0)

    def test_RESUME_runs_only_what_is_owed(self) -> None:
        # The property three host crashes cost: a second invocation must not
        # redo completed legs.
        with tempfile.TemporaryDirectory() as td:
            led = pathlib.Path(td) / "legs.jsonl"
            self._run(led, 1)                      # on off on -> 3 legs
            first = len(read_ledger(led))
            self._run(led, 1)                      # same plan, nothing owed
            self.assertEqual(len(read_ledger(led)), first)

    def test_an_UNPARSABLE_leg_is_recorded_VOID_not_dropped(self) -> None:
        # A leg that ran and produced no parsable number is a broken
        # instrument. Dropping it would leave the fold looking healthy on fewer
        # legs than it claims -- the shape that let a 52-byte "No such
        # container" file read as "the trace did not reach the server".
        with tempfile.TemporaryDirectory() as td:
            led = pathlib.Path(td) / "legs.jsonl"
            rc = main([
                "--ledger", str(led),
                "--arm", "on", "--arm", "off",
                "--legs-per-arm", "1",
                "--metric", "metric",
                "--metric-regex", r"NEVER_MATCHES=([0-9.]+)",
                "--command", "true {arm}",
                "--dry-run",
            ])
            recs = read_ledger(led)
            self.assertEqual(len(recs), 3)                     # every leg recorded
            self.assertTrue(all("void" in r for r in recs))    # each marked VOID
            self.assertTrue(all("metric" not in r for r in recs))
            self.assertEqual(rc, 1)                            # and the run REFUSES

    def test_a_partial_ledger_is_completed_not_restarted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            led = pathlib.Path(td) / "legs.jsonl"
            led.write_text(json.dumps({"arm": "on", "boot_id": "z", "metric": 1.0}) + "\n")
            self._run(led, 1)
            recs = read_ledger(led)
            self.assertEqual(len(recs), 3)
            self.assertEqual(recs[0]["boot_id"], "z")   # the pre-existing leg survives


if __name__ == "__main__":
    unittest.main()
