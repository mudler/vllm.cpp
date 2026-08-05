#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-now-current.py.

NOW.md only works if it is short and true. The mutations therefore cover both
failure directions: a digest that grew back into a status log, and a change that
moved what is live without refreshing the digest.
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


now = _load("now_current", "scripts/check-now-current.py")


VALID = "\n".join(
    [
        "# NOW",
        "",
        "<!-- now-updated: 2026-08-04 -->",
        "",
        "## Live claims",
        "",
        "| Claim | State | Next |",
        "|---|---|---|",
        "| Thing | ACTIVE | Run the gate |",
        "",
        "## Current gate",
        "",
        "Token-exact against the pinned oracle, then every speed axis.",
        "",
        "## Next actions",
        "",
        "1. Do the next thing.",
    ]
)


class Baseline(unittest.TestCase):
    def test_valid_digest_passes(self) -> None:
        self.assertEqual(now.structure_errors(VALID), [])

    def test_headings_are_case_insensitive(self) -> None:
        self.assertEqual(now.structure_errors(VALID.replace("## Live claims", "## LIVE CLAIMS")), [])


class StructureMutations(unittest.TestCase):
    def test_missing_stamp_is_rejected(self) -> None:
        text = VALID.replace("<!-- now-updated: 2026-08-04 -->", "")
        self.assertTrue(any("freshness stamp" in e for e in now.structure_errors(text)))

    def test_malformed_stamp_is_rejected(self) -> None:
        text = VALID.replace("2026-08-04 -->", "yesterday -->")
        self.assertTrue(any("freshness stamp" in e for e in now.structure_errors(text)))

    def test_each_required_heading_is_enforced(self) -> None:
        for heading in now.REQUIRED_HEADINGS:
            text = VALID.replace(f"## {heading.capitalize()}", "## Something else")
            with self.subTest(heading=heading):
                self.assertTrue(
                    any(heading in e for e in now.structure_errors(text)),
                    f"dropping '{heading}' was not rejected",
                )

    def test_overlong_digest_is_rejected(self) -> None:
        text = VALID + "\n" + "\n".join(f"- line {i}" for i in range(now.MAX_LINES))
        self.assertTrue(any("line budget" in e for e in now.structure_errors(text)))

    def test_oversized_entry_is_rejected(self) -> None:
        text = VALID + "\n- " + "x" * (now.MAX_ENTRY_CHARS + 1)
        self.assertTrue(
            any("character budget" in e for e in now.structure_errors(text))
        )


class FreshnessMutations(unittest.TestCase):
    def test_state_append_without_digest_refresh_is_rejected(self) -> None:
        errors = now.freshness_errors({".agents/state.md", "src/vllm/thing.cpp"})
        self.assertTrue(any("did not" in e for e in errors))

    def test_state_append_with_digest_refresh_passes(self) -> None:
        self.assertEqual(
            now.freshness_errors({".agents/state.md", ".agents/NOW.md"}), []
        )

    def test_unrelated_change_is_not_forced_to_refresh(self) -> None:
        self.assertEqual(now.freshness_errors({"README.md"}), [])


class LiveTree(unittest.TestCase):
    def test_repository_digest_is_valid(self) -> None:
        self.assertTrue(now.NOW.exists(), f"{now.NOW_PATH} is missing")
        self.assertEqual(
            now.structure_errors(now.NOW.read_text(encoding="utf-8")), []
        )


if __name__ == "__main__":
    unittest.main()


class ReorderIsNotAnAppend(unittest.TestCase):
    """A repaired interleave moves no entry, so it owes no NOW.md refresh."""

    def test_reorder_only_is_exempt(self) -> None:
        self.assertEqual(
            now.freshness_errors({".agents/state.md"}, entries_changed=False), []
        )

    def test_a_real_append_still_requires_the_refresh(self) -> None:
        self.assertTrue(
            now.freshness_errors({".agents/state.md"}, entries_changed=True)
        )

    def test_default_still_demands_the_refresh(self) -> None:
        self.assertTrue(now.freshness_errors({".agents/state.md"}))
