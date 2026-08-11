#!/usr/bin/env python3
"""Mutation tests for scripts/check-now-current.py.

NOW.md is the one file a cold session is guaranteed to read, so its budget is
the thing that stops it decaying back into a status log. Every guard below is
proved by a mutation that breaks the digest and asserts the checker rejects it.

Freshness coupling ("NOW.md must move when the live position moves") is NOT
tested here because it is not enforced here: check-doc-checkpoint.py owns it,
and requiring NOW.md on a lifecycle change is tested in test_doc_checkpoint.py.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "check_now_current", ROOT / "scripts/check-now-current.py"
)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


VALID = """# NOW

<!-- now-updated: 2026-08-09 -->

## Live claims

- none

## Current gate

- none

## Next actions

- none
"""


class StructureContract(unittest.TestCase):
    def test_valid_digest_has_no_errors(self):
        self.assertEqual(checker.structure_errors(VALID), [])

    def test_missing_stamp_is_rejected(self):
        text = VALID.replace("<!-- now-updated: 2026-08-09 -->\n", "")
        self.assertTrue(
            any("freshness stamp" in e for e in checker.structure_errors(text))
        )

    def test_malformed_stamp_is_rejected(self):
        text = VALID.replace("2026-08-09", "yesterday")
        self.assertTrue(
            any("freshness stamp" in e for e in checker.structure_errors(text))
        )

    def test_each_required_heading_is_load_bearing(self):
        for heading in checker.REQUIRED_HEADINGS:
            with self.subTest(heading=heading):
                text = VALID.replace(f"## {heading.capitalize()}", "## Removed")
                errors = checker.structure_errors(text)
                self.assertTrue(
                    any(heading in e for e in errors),
                    f"removing '{heading}' was not caught: {errors}",
                )

    def test_line_budget_is_enforced(self):
        text = VALID + "\n".join(f"- entry {i}" for i in range(checker.MAX_LINES))
        self.assertTrue(
            any("line budget" in e for e in checker.structure_errors(text))
        )

    def test_a_row_costs_only_itself(self):
        """The defect ENG-RECORD-CONFLICT-SURFACES (#364) removed.

        RED-BEFORE: the tracked digest measured EXACTLY the old 6000-char cap,
        so adding one row put it over and the PR had to EVICT another row --
        someone else's -- to land. That read-modify-write on a shared global is
        what made NOW.md conflict in 5 of 16 conflicting open PRs, and it made a
        CLEAN merge unsafe: both evictions and both additions would apply.

        A row must now cost one line and nothing else.
        """
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        self.assertEqual(checker.structure_errors(live), [])
        # RETARGETED by ENG-NOW-DERIVED (#374): this used to add a per-row TABLE
        # row, which is now refused outright -- rows live in the row's own spec
        # and are rendered by scripts/now.py. The obligation under test is
        # unchanged and still the point: adding authored content must not force
        # evicting somebody else's.
        added = "- A new authored note about the current gate.\n"
        self.assertEqual(checker.structure_errors(live + added), [])

    def test_two_concurrent_rows_do_not_couple(self):
        """Two PRs adding a row each must both pass, and so must their union."""
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        # Authored lines, not row-table rows -- see the note above (#374).
        a = "- Operator note A.\n"
        b = "- Operator note B.\n"
        self.assertEqual(checker.structure_errors(live + a), [])
        self.assertEqual(checker.structure_errors(live + b), [])
        self.assertEqual(checker.structure_errors(live + a + b), [])

    def test_the_line_budget_still_bounds_the_page(self):
        """Removing the byte cap must not have left the page unbounded.

        MAX_LINES is what carries the obligation now, so mutate the LIVE digest
        past it and require the failure -- otherwise the removal quietly took
        the whole size gate with it.
        """
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        grown = live + "\n".join(f"- entry {i}" for i in range(checker.MAX_LINES))
        self.assertTrue(any("line budget" in e for e in checker.structure_errors(grown)))

    def test_one_oversized_entry_is_rejected(self):
        text = VALID + "- " + ("x" * (checker.MAX_ENTRY_CHARS + 1)) + "\n"
        errors = checker.structure_errors(text)
        self.assertTrue(
            any("over the" in e and "character budget" in e for e in errors), errors
        )

    def test_budgets_are_small_enough_to_be_read_in_full(self):
        """A budget nobody could read in one pass is not a budget."""
        self.assertLessEqual(checker.MAX_LINES, 120)
        # The byte cap is gone (#364); the line cap is what bounds the read.


class RepositoryDigest(unittest.TestCase):
    def test_tracked_now_is_in_budget(self):
        text = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        self.assertEqual(checker.structure_errors(text), [])


class DerivedLivePosition(unittest.TestCase):
    """NOW.md may not carry a per-row table again (ENG-NOW-DERIVED, #374).

    The table left because requiring it made this file a surface every
    row-advancing PR had to write -- a lock, and 5 of the 16 conflicting open
    PRs at d928e2c3. Removing it once is not enough: the decay path is someone
    re-adding "just one row", after which it is a status log again and every PR
    is back in it. The SHAPE is enforced, not just today's state.
    """

    def test_a_per_row_table_row_is_rejected(self):
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        self.assertEqual(checker.structure_errors(live), [])
        regrown = live + "\n| `KERNEL-ALPHA` | ACTIVE | next thing |\n"
        errors = checker.structure_errors(regrown)
        self.assertTrue(
            any("per-row table row is back" in e for e in errors),
            f"a re-added row must be caught; got {errors}",
        )

    def test_ordinary_prose_tables_still_pass(self):
        """The guard keys on a backticked stable ID, not on tables as such.

        The authored half legitimately uses tables. If this goes red the guard
        has become a ban on tables and will simply be worked around.
        """
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        prose = live + "\n| Surface | Changes when |\n|---|---|\n| STATUS | a row moves |\n"
        self.assertEqual(checker.structure_errors(prose), [])

    def test_the_guard_is_load_bearing(self):
        """MUTATION: without the pattern the regrown table passes."""
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        regrown = live + "\n| `KERNEL-ALPHA` | ACTIVE | next thing |\n"
        with mock.patch.object(checker, "ROW_TABLE_LINE", re.compile(r"(?!)")):
            self.assertEqual(
                [e for e in checker.structure_errors(regrown) if "per-row" in e],
                [],
                "the mutation must disarm the guard, or this proves nothing",
            )
        self.assertTrue(
            any("per-row" in e for e in checker.structure_errors(regrown))
        )

    def test_the_live_file_no_longer_lists_rows(self):
        live = (ROOT / ".agents/NOW.md").read_text(encoding="utf-8")
        self.assertNotIn("| `SPEC-DSPARK`", live)
        self.assertIn("scripts/now.py", live)


if __name__ == "__main__":
    unittest.main()
