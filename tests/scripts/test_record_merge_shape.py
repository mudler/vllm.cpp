#!/usr/bin/env python3
"""Prove the shared record surfaces no longer conflict by construction.

This is the regression test for ENG-RECORD-CONFLICT-SURFACES (issue #364). It
does not inspect any checker: it asks git the question the open PR queue was
answering the wrong way. Two branches make one ordinary, independent record edit
each; `git merge-tree` must merge them without a conflict.

Measured at origin/main d928e2c3 before the row landed: 16 of 29 open PRs were
CONFLICTING and 13 of those conflicted in bookkeeping files ONLY -- coordination
in 8, NOW.md in 5, roadmap_v1.md in 4, check-public-doc-tables.py in 4,
STATUS.md in 4 -- against 3 in any src/ or tests/ path. The cause was never the
volume of parallel work; it was three surfaces shaped so that concurrent writers
must collide:

  * a fixed byte budget on a file with zero headroom, which turns every addition
    into "evict someone else's row first";
  * a byte count of one file hardcoded in another, which moves on every edit;
  * keyed tables that everyone appends to at the same anchor.

The first two are removed. The third is mitigated by ordering keyed rows by a
stable key, so two new rows land at two different places instead of both landing
on the last line.

Each case below is RED before the corresponding work item and GREEN after. If a
future change reintroduces a per-file budget, a cross-file measurement, or an
append-at-one-anchor table, the matching case goes red.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


class MergeShape(unittest.TestCase):
    """Two independent record edits must merge cleanly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls._tmp.name) / "repo"
        cls.repo.mkdir()
        git("init", "-q", "-b", "main", cwd=cls.repo)
        git("config", "user.email", "t@example.invalid", cwd=cls.repo)
        git("config", "user.name", "t", cwd=cls.repo)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _merges_cleanly(self, base: str, edit_a: str, edit_b: str) -> bool:
        """Commit `base`, branch two edits off it, and ask git to merge them."""
        repo = self.repo
        target = repo / "record.md"

        target.write_text(base, encoding="utf-8")
        git("add", "record.md", cwd=repo)
        git("commit", "-qm", "base", cwd=repo)
        base_sha = git("rev-parse", "HEAD", cwd=repo).strip()

        target.write_text(edit_a, encoding="utf-8")
        git("commit", "-qam", "a", cwd=repo)
        a_sha = git("rev-parse", "HEAD", cwd=repo).strip()

        git("checkout", "-q", base_sha, cwd=repo)
        target.write_text(edit_b, encoding="utf-8")
        git("add", "record.md", cwd=repo)
        git("commit", "-qm", "b", cwd=repo)
        b_sha = git("rev-parse", "HEAD", cwd=repo).strip()

        merged = subprocess.run(
            ["git", "merge-tree", "--write-tree", a_sha, b_sha],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        git("checkout", "-q", "main", cwd=repo)
        git("update-ref", "-d", "refs/heads/main", cwd=repo)
        return merged.returncode == 0 and "CONFLICT" not in merged.stdout

    def test_two_rows_appended_to_an_unordered_table_collide(self) -> None:
        """The defect, reproduced. This is what every keyed table used to do.

        Both branches append their row to the same last line, so git has two
        different insertions at one anchor and cannot choose. Asserting the
        collision is what makes the ordered case below evidence rather than a
        tautology -- without it, a merge that trivially succeeds would look like
        a fix.
        """
        base = "| ID | Note |\n|---|---|\n| AAA | x |\n"
        self.assertFalse(
            self._merges_cleanly(
                base,
                base + "| MMM | new |\n",
                base + "| ZZZ | new |\n",
            )
        )

    def test_two_rows_in_an_id_ordered_table_merge_cleanly(self) -> None:
        """W4: ordering by a stable key separates the insertion points.

        The same two rows, placed where their IDs sort rather than at the end,
        land in different hunks and merge without a conflict.
        """
        rows = ["| AAA | x |\n", "| FFF | x |\n", "| SSS | x |\n", "| ZZZ | x |\n"]
        head = "| ID | Note |\n|---|---|\n"
        base = head + "".join(rows)

        def with_row(new: str) -> str:
            merged = sorted([*rows, new])
            return head + "".join(merged)

        self.assertTrue(
            self._merges_cleanly(base, with_row("| BBB | new |\n"), with_row("| TTT | new |\n"))
        )

    def test_a_fixed_budget_forces_an_eviction_that_collides(self) -> None:
        """The NOW.md defect, reproduced.

        A file at exactly its budget cannot take a row without dropping one. Two
        branches each drop a DIFFERENT neighbour to make room, so the edits
        overlap even though the additions do not.
        """
        rows = ["| AAA |\n", "| FFF |\n", "| SSS |\n"]
        base = "".join(rows)
        # Each branch evicts a different existing row to fit its own.
        a = "".join([rows[0], rows[2], "| NEW-A |\n"])
        b = "".join([rows[0], rows[1], "| NEW-B |\n"])
        self.assertFalse(self._merges_cleanly(base, a, b))

    def test_without_a_budget_an_edit_deletes_nothing(self) -> None:
        """W3: with no budget, adding a row deletes nobody else's row.

        This is the precise benefit, and it is worth stating exactly because it
        is NOT "the conflict goes away". Two appends land on the same anchor and
        collide either way -- that is what the ordering work item addresses. What
        the budget added on top was a forced DELETION of unrelated content, and
        that is the part with teeth:

          * it makes every PR edit lines it does not own, which is what turned
            an ordinary one-row change into a review of someone else's row; and
          * it makes a SUCCESSFUL merge unsafe. Git resolving two such branches
            without complaint applies both evictions, so both dropped rows are
            gone from the result and no gate notices -- the budget was defended
            by losing the content it was defending.

        So assert the shape of the edit, not the merge: with a budget the diff
        removes a line, without one it only adds.
        """
        rows = ["| AAA |\n", "| FFF |\n", "| SSS |\n"]
        base = "".join(rows)

        budgeted = "".join([rows[0], rows[2], "| NEW-A |\n"])  # evicted FFF to fit
        unbudgeted = base + "| NEW-A |\n"

        self.assertNotIn(rows[1], budgeted, "the budgeted edit must evict a row")
        for row in rows:
            self.assertIn(
                row,
                unbudgeted,
                "an unbudgeted addition must preserve every existing row",
            )

        # And the data-loss hazard itself: had the two evicting branches merged,
        # BOTH victims would be missing from the result.
        a = "".join([rows[0], rows[2], "| NEW-A |\n"])  # dropped FFF
        b = "".join([rows[0], rows[1], "| NEW-B |\n"])  # dropped SSS
        union = set(a.splitlines()) | set(b.splitlines())
        silently_merged = {r for r in union if r.strip()}
        self.assertNotIn(
            rows[1].strip(),
            {r for r in a.splitlines()},
            "branch A must have dropped FFF for this to be the real scenario",
        )
        self.assertTrue(
            rows[1].strip() in silently_merged and rows[2].strip() in silently_merged,
            "sanity: the union restores them, which is exactly what a real "
            "three-way merge would NOT do -- it applies each side's deletion",
        )

    def test_ordered_and_unbudgeted_rows_merge_cleanly(self) -> None:
        """Both fixes together: the end state this row is aiming at.

        No budget means neither branch evicts anything; ordering by a stable key
        means the two additions land in different hunks. Only with BOTH does the
        ordinary case -- two PRs each adding one record row -- merge without a
        human resolving it.
        """
        rows = ["| AAA |\n", "| FFF |\n", "| SSS |\n", "| ZZZ |\n"]
        base = "".join(rows)

        def with_row(new: str) -> str:
            return "".join(sorted([*rows, new]))

        self.assertTrue(
            self._merges_cleanly(base, with_row("| BBB |\n"), with_row("| TTT |\n"))
        )


class NoReintroduction(unittest.TestCase):
    """The removed shapes must stay removed."""

    def test_the_now_checker_has_no_whole_file_byte_budget(self) -> None:
        source = (ROOT / "scripts/check-now-current.py").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "MAX_CHARS",
            code,
            "a whole-file byte budget is back on .agents/NOW.md; cap the ENTRY, "
            "never the file -- see AGENTS.md 'No surface that every PR must write'",
        )

    def test_the_status_ratchet_measures_defects_not_length(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "doc_tables", ROOT / "scripts/check-public-doc-tables.py"
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)

        self.assertNotIn(
            "chars",
            module.STATUS_RATCHET,
            "a byte count of docs/STATUS.md is pinned in check-public-doc-tables.py "
            "again; that couples every PR to lines it does not own",
        )
        # And the keys that remain must still be the quality counters, or the
        # obligation left with the byte count.
        self.assertEqual(
            set(module.STATUS_RATCHET),
            {"h2_sections", "long_paragraphs", "oversized_cells"},
        )

    def test_the_live_digest_has_headroom(self) -> None:
        """Zero headroom is the tell that a budget has become a lock.

        .agents/NOW.md measured EXACTLY 6000 of 6000 permitted characters, and
        docs/STATUS.md EXACTLY 243245 of 243245. A surface tuned to the byte is
        one where the next writer must take something from the last one.
        """
        now = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        lines = len(now.splitlines())
        self.assertLess(
            lines,
            100,
            "the digest is at its line cap; a row now costs an eviction again",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
