#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-state-order.py and the sorter.

The mutation that matters is the real one: entries that a union merge left in
non-chronological order, which is how the "append; newest last" contract broke
without anyone noticing.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


order = _load("state_order", "scripts/check-state-order.py")
sorter = _load("state_sorter", "scripts/sort-state-tail.py")


def entry(title: str, stamp: str, body: str = "Body text.") -> str:
    return f"## {title}\n<!-- state: {stamp} -->\n\n{body}\n"


def log(*entries: str) -> str:
    return "# State log\n\nFrozen history.\n\n" + order.MARKER + "\n\n" + "\n".join(
        entries
    )


class Baseline(unittest.TestCase):
    def test_ordered_log_passes(self) -> None:
        text = log(
            entry("First", "2026-08-01"),
            entry("Second", "2026-08-02T09:30"),
            entry("Third", "2026-08-02T14:00"),
        )
        self.assertEqual(order.entry_errors(text), [])

    def test_same_day_entries_are_allowed(self) -> None:
        text = log(entry("A", "2026-08-01"), entry("B", "2026-08-01"))
        self.assertEqual(order.entry_errors(text), [])


class Mutations(unittest.TestCase):
    def test_out_of_order_entry_is_rejected(self) -> None:
        """The historical failure: a 07-27 entry appended after a 07-30 one."""
        text = log(entry("Later", "2026-07-30"), entry("Earlier", "2026-07-27"))
        errors = order.entry_errors(text)
        self.assertTrue(any("is anchored" in e and "follows" in e for e in errors))

    def test_out_of_order_by_time_is_rejected(self) -> None:
        text = log(entry("A", "2026-08-01T18:00"), entry("B", "2026-08-01T09:00"))
        self.assertNotEqual(order.entry_errors(text), [])

    def test_missing_anchor_is_rejected(self) -> None:
        text = log(entry("Anchored", "2026-08-01")) + "\n## Bare entry\n\nBody.\n"
        errors = order.entry_errors(text)
        self.assertTrue(any("has no" in e and "anchor" in e for e in errors))

    def test_missing_marker_is_rejected(self) -> None:
        errors = order.entry_errors("# State log\n\n## Entry\n\nBody.\n")
        self.assertTrue(any("enforcement marker" in e for e in errors))

    def test_orphan_anchor_is_rejected(self) -> None:
        text = log(entry("A", "2026-08-01")) + "\n<!-- state: 2026-08-02 -->\n"
        errors = order.entry_errors(text)
        self.assertTrue(any("not attached to an entry heading" in e for e in errors))

    def test_malformed_anchor_does_not_count_as_an_anchor(self) -> None:
        text = log("## Entry\n<!-- state: 08/01/2026 -->\n\nBody.\n")
        self.assertNotEqual(order.entry_errors(text), [])


class Sorter(unittest.TestCase):
    def test_sorting_repairs_an_interleaved_merge(self) -> None:
        text = log(entry("Later", "2026-07-30"), entry("Earlier", "2026-07-27"))
        enforced = order.split_at_marker(text)
        assert enforced is not None
        entries, problems = sorter.parse_entries(enforced)
        self.assertEqual(problems, [])
        titles = [e.title for e in sorted(entries, key=lambda e: e.key)]
        self.assertEqual(titles, ["Earlier", "Later"])

    def test_sorting_never_drops_an_entry(self) -> None:
        text = log(
            entry("C", "2026-08-03"), entry("A", "2026-08-01"), entry("B", "2026-08-02")
        )
        enforced = order.split_at_marker(text)
        assert enforced is not None
        entries, _ = sorter.parse_entries(enforced)
        self.assertEqual(len(entries), 3)
        self.assertEqual(
            sorted(e.title for e in entries), ["A", "B", "C"]
        )

    def test_unanchored_entry_blocks_sorting(self) -> None:
        enforced = order.split_at_marker(
            log(entry("A", "2026-08-01")) + "\n## Bare\n\nBody.\n"
        )
        assert enforced is not None
        _, problems = sorter.parse_entries(enforced)
        self.assertTrue(any("cannot sort" in p for p in problems))


class LiveTree(unittest.TestCase):
    def test_repository_state_log_is_ordered(self) -> None:
        self.assertEqual(order.main(), 0)


if __name__ == "__main__":
    unittest.main()
