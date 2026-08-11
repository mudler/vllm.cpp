#!/usr/bin/env python3
"""Tests for scripts/now.py, the derived live-position digest.

The digest replaced a per-row table in .agents/NOW.md that every row-advancing
PR was required to write (ENG-NOW-DERIVED, #374). The whole benefit is that
nobody writes it, so the properties that matter are: it renders from the per-row
records, it renders WITHOUT a network, and an unreachable remote degrades to
REMOTE_UNVERIFIED rather than to a silent empty result.

That last one is the trap worth naming. "No claims" and "could not ask" look
identical in a rendered table, and a digest that quietly shows an empty roster
when the network is down is worse than the file it replaced -- it reads as
authoritative.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]

SPEC = importlib.util.spec_from_file_location("now_render", ROOT / "scripts/now.py")
assert SPEC is not None and SPEC.loader is not None
now = importlib.util.module_from_spec(SPEC)
sys.modules["now_render"] = now
SPEC.loader.exec_module(now)


class OfflineFirst(unittest.TestCase):
    def test_renders_without_a_network(self) -> None:
        text = now.render(offline=True)
        self.assertIn("Live rows", text)
        self.assertIn("REMOTE_UNVERIFIED", text)

    def test_offline_still_lists_every_live_row(self) -> None:
        """The rows come from local files, so losing the network loses nothing
        but the PR column."""
        rows = now.live_rows()
        self.assertGreater(len(rows), 0, "no SPIKE/ACTIVE rows found at all")
        text = now.render(offline=True)
        for row in rows[:5]:
            self.assertIn(f"`{row['id']}`", text)

    def test_an_unreachable_remote_degrades_rather_than_raises(self) -> None:
        with mock.patch.object(
            now.subprocess, "run", side_effect=OSError("no gh on PATH")
        ):
            prs, degraded = now.open_prs()
        self.assertEqual(prs, {})
        self.assertIsNotNone(degraded)
        self.assertIn("REMOTE_UNVERIFIED", degraded)

    def test_a_failing_gh_is_not_read_as_no_prs(self) -> None:
        """Unknown is neither absence nor success."""
        failed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="could not resolve host\n"
        )
        with mock.patch.object(now.subprocess, "run", return_value=failed):
            prs, degraded = now.open_prs()
        self.assertEqual(prs, {})
        self.assertIn("REMOTE_UNVERIFIED", degraded)

    def test_unparseable_output_degrades_too(self) -> None:
        garbage = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr=""
        )
        with mock.patch.object(now.subprocess, "run", return_value=garbage):
            prs, degraded = now.open_prs()
        self.assertEqual(prs, {})
        self.assertIn("REMOTE_UNVERIFIED", degraded)

    def test_the_degraded_banner_reaches_the_reader(self) -> None:
        """A degrade nobody sees is the silent-empty failure with extra steps."""
        with mock.patch.object(now, "open_prs", return_value=({}, "REMOTE_UNVERIFIED: x")):
            text = now.render()
        self.assertIn("REMOTE_UNVERIFIED", text)


class DerivedFromPerRowRecords(unittest.TestCase):
    def test_rows_come_from_the_matrices_with_their_state(self) -> None:
        rows = {r["id"]: r for r in now.live_rows()}
        self.assertTrue(rows)
        for row in rows.values():
            self.assertIn(row["state"], {"SPIKE", "ACTIVE"})

    def test_a_spec_now_line_is_picked_up(self) -> None:
        self.assertIn(
            "W2",
            now.spec_now("now-derived.md"),
            "this row's own spec carries a `## Now`, so it must render",
        )

    def test_a_spec_without_a_now_section_yields_nothing(self) -> None:
        self.assertEqual(now.spec_now("does-not-exist-at-all.md"), "")

    def test_claims_are_read_from_the_per_claim_directory(self) -> None:
        """The one-file-per-claim shape from #364 is the claim source."""
        probe = ROOT / ".agents/claims/CLAIM-NOW-RENDER-PROBE.md"
        probe.write_text(
            "# CLAIM-NOW-RENDER-PROBE\n\n"
            "| Claim | Row IDs |\n|---|---|\n"
            "| `CLAIM-NOW-RENDER-PROBE` | `ENG-NOW-DERIVED` |\n",
            encoding="utf-8",
        )
        try:
            self.assertIn("CLAIM-NOW-RENDER-PROBE", now.claims().get("ENG-NOW-DERIVED", []))
        finally:
            probe.unlink()

    def test_nothing_is_read_from_the_authored_file(self) -> None:
        """The digest must not depend on .agents/NOW.md for any row.

        If it did, the shared file would be back in the loop and the lock with
        it -- which is the entire defect this row removes.
        """
        rows = now.live_rows()
        with mock.patch.object(now, "AGENTS", ROOT / ".agents"):
            self.assertEqual([r["id"] for r in now.live_rows()], [r["id"] for r in rows])
        text = now.render(offline=True)
        self.assertNotIn("Live claims", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
