#!/usr/bin/env python3
"""Unit and mutation checks for W2-W4.

scripts/claim-view.py       - the generated, PR-derived claim view
scripts/ready-for-helper.py - the pickable queue
scripts/check-pr-size.py    - the reviewability cap
"""

from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


view = _load("claim_view", "scripts/claim-view.py")
ready = _load("ready_for_helper", "scripts/ready-for-helper.py")
prsize = _load("pr_size", "scripts/check-pr-size.py")

KNOWN = {"ENG-FOO", "KV-BAR"}


def block(stamp: str, rows: str = "| `ENG-FOO` | #7 | draft | someone | 2026-08-04 |") -> str:
    return "\n".join([
        view.BEGIN,
        f"<!-- claim-view:generated {stamp} -->",
        "",
        "| Row | PR | State | Agent | Updated |",
        "|---|---|---|---|---|",
        rows,
        "",
        view.END,
    ])


class ClaimView(unittest.TestCase):
    def test_valid_block_passes(self) -> None:
        today = time.strftime("%Y-%m-%d")
        self.assertEqual(view.check_errors(block(today), KNOWN), [])

    def test_missing_block_is_rejected(self) -> None:
        errors = view.check_errors("# coordination\n\nno block here", KNOWN)
        self.assertTrue(any("missing the claim view" in e for e in errors))

    def test_missing_stamp_is_rejected(self) -> None:
        text = block(time.strftime("%Y-%m-%d")).replace(
            f"<!-- claim-view:generated {time.strftime('%Y-%m-%d')} -->", ""
        )
        self.assertTrue(any("no <!-- claim-view:generated" in e
                            for e in view.check_errors(text, KNOWN)))

    def test_stale_view_is_rejected(self) -> None:
        """A claim with no live PR behind it must EXPIRE, not rot."""
        old = time.strftime("%Y-%m-%d", time.localtime(time.time() - 60 * 86400))
        errors = view.check_errors(block(old), KNOWN)
        self.assertTrue(any("TTL" in e for e in errors))

    def test_unknown_row_is_rejected(self) -> None:
        text = block(time.strftime("%Y-%m-%d"),
                     "| `NOT-A-ROW` | #7 | draft | a | 2026-08-04 |")
        self.assertTrue(any("unknown row" in e for e in view.check_errors(text, KNOWN)))

    def test_render_only_counts_row_branches(self) -> None:
        rendered = view.render(
            [
                {"number": 1, "headRefName": "row/ENG-FOO", "isDraft": True,
                 "author": {"login": "a"}, "updatedAt": "2026-08-04T00:00:00Z"},
                {"number": 2, "headRefName": "feat/unrelated", "isDraft": False,
                 "author": {"login": "b"}, "updatedAt": "2026-08-04T00:00:00Z"},
            ],
            "2026-08-04",
        )
        self.assertIn("ENG-FOO", rendered)
        self.assertNotIn("unrelated", rendered)

    def test_empty_state_renders_a_valid_block(self) -> None:
        rendered = view.render([], "2026-08-04")
        self.assertIn(view.BEGIN, rendered)
        self.assertIn(view.END, rendered)

    def test_live_repository_view_is_valid(self) -> None:
        self.assertEqual(view.main.__module__, "claim_view")
        text = (ROOT / ".agents/coordination.md").read_text(encoding="utf-8")
        self.assertEqual(view.check_errors(text, view.known_row_ids()), [])


class ReadyForHelper(unittest.TestCase):
    def test_queue_is_computable_and_offered_rows_are_valid(self) -> None:
        self.assertEqual(ready.main.__module__, "ready_for_helper")
        pickable, _ = ready.queue()
        reserved = ready.reserved_rows()
        for row in pickable:
            self.assertEqual(ready.evaluate(row, reserved), [], row.item_id)

    def test_a_reserved_row_is_not_offered(self) -> None:
        pickable, _ = ready.queue()
        if not pickable:
            self.skipTest("queue is empty; nothing to reserve")
        row = pickable[0]
        self.assertIn("already claimed by an open PR",
                      ready.evaluate(row, {row.item_id}))

    def test_only_ready_or_inventoried_states_are_pickable(self) -> None:
        self.assertEqual(ready.PICKABLE_STATES, {"READY", "INVENTORIED"})


class PrSize(unittest.TestCase):
    def test_record_paths_are_exempt(self) -> None:
        for path in (".agents/state.md", "docs/STATUS.md", "scripts/x.py",
                     "tests/scripts/test_x.py", ".github/workflows/ci.yml"):
            self.assertTrue(path.startswith(prsize.EXEMPT_PREFIXES), path)

    def test_feature_paths_are_counted(self) -> None:
        for path in ("src/vllm/a.cpp", "include/vt/b.h", "tests/vt/c.cpp"):
            self.assertFalse(path.startswith(prsize.EXEMPT_PREFIXES), path)

    def test_cap_is_a_real_number(self) -> None:
        self.assertGreater(prsize.MAX_FEATURE_LINES, 0)


if __name__ == "__main__":
    unittest.main()
