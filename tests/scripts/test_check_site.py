#!/usr/bin/env python3
"""Mutation tests for scripts/check-site.py.

Each test copies the real tree into a scratch directory, breaks exactly one
invariant, and asserts the checker reports it. Reading the checker is not
evidence that it catches anything; breaking the thing it claims to catch is.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-site.py"


def run_in(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(tree / "scripts" / "check-site.py")],
        capture_output=True,
        text=True,
        cwd=tree,
    )


class SiteGuardTests(unittest.TestCase):
    def scratch(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        tree = tmp / "repo"
        (tree / "scripts").mkdir(parents=True)
        shutil.copy(CHECKER, tree / "scripts" / "check-site.py")
        # Only what the checker reads: the top-level docs and the nav file.
        # Copying docs/bench-evidence and docs/superpowers would make every test
        # noticeably slower for no coverage.
        (tree / "docs").mkdir()
        for doc in (ROOT / "docs").glob("*.md"):
            shutil.copy(doc, tree / "docs" / doc.name)
        (tree / "website" / "data").mkdir(parents=True)
        shutil.copy(
            ROOT / "website" / "data" / "nav.yaml",
            tree / "website" / "data" / "nav.yaml",
        )
        return tree

    def test_the_shipped_tree_is_clean(self) -> None:
        result = run_in(self.scratch())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nav in bijection", result.stdout)

    def test_a_doc_without_an_h1_is_caught(self) -> None:
        tree = self.scratch()
        target = tree / "docs" / "USAGE.md"
        body = target.read_text().split("\n", 1)[1]
        target.write_text("Using vllm.cpp\n" + body)
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("USAGE.md", result.stderr)
        self.assertIn("H1", result.stderr)

    def test_a_doc_missing_from_nav_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        kept = [
            line
            for line in nav.read_text().splitlines(keepends=True)
            if "ROCM.md" not in line and "ROCm backend" not in line
        ]
        nav.write_text("".join(kept))
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ROCM.md", result.stderr)
        self.assertIn("absent from", result.stderr)

    def test_a_nav_entry_with_no_file_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        nav.write_text(nav.read_text() + "  - file: GHOST.md\n    label: Ghost\n")
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("GHOST.md", result.stderr)
        self.assertIn("dead sidebar link", result.stderr)

    def test_a_duplicated_nav_entry_is_caught(self) -> None:
        tree = self.scratch()
        nav = tree / "website" / "data" / "nav.yaml"
        nav.write_text(nav.read_text() + "  - file: ROCM.md\n    label: ROCm again\n")
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("more than once", result.stderr)

    def test_a_missing_nav_file_fails_closed(self) -> None:
        tree = self.scratch()
        (tree / "website" / "data" / "nav.yaml").unlink()
        result = run_in(tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
