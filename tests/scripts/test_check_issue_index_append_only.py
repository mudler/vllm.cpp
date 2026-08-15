#!/usr/bin/env python3
"""The append-only gate, mutated against real Git history.

`check-agent-record.py` sees one tree, so it cannot tell an appended row from
an edited one. Only a range can. These cases build actual repositories rather
than stubbing the diff, because the thing under test IS the diff.

The SKIP case matters as much as the failing ones. A range check with no base
resolves to nothing, and nothing must never read as success.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-issue-index-append-only.py"
INDEX = ".agents/issue-index.md"

PREAMBLE = "# Issue index\n\n| Issue | Row | Title | Kind |\n|---:|---|---|---|\n"


def row(number: int, owner: str = "`BACKEND-ROCM`") -> str:
    return (
        f"| [#{number}](https://github.com/mudler/vllm.cpp/issues/{number})"
        f" | {owner} | title | bug |\n"
    )


class AppendOnlyRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / ".agents").mkdir(parents=True)
        (self.tmp / "scripts").mkdir(parents=True)
        shutil.copy(CHECKER, self.tmp / "scripts" / CHECKER.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")
        self.write(PREAMBLE + row(1) + row(2))
        self.git("add", "-A")
        self.git("commit", "-qm", "base")
        self.git("branch", "base-ref")

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=self.tmp, capture_output=True, text=True, check=True
        )

    def write(self, text: str) -> None:
        (self.tmp / INDEX).write_text(text, encoding="utf-8")

    def commit(self, message: str) -> None:
        self.git("add", "-A")
        self.git("commit", "-qm", message)

    def run_check(self, base: str = "base-ref") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.tmp / "scripts" / CHECKER.name),
             "--base", base, "--head", "HEAD"],
            cwd=self.tmp, capture_output=True, text=True, check=False,
        )

    def test_appending_a_row_passes(self) -> None:
        self.write(PREAMBLE + row(1) + row(2) + row(3))
        self.commit("append")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", result.stdout)

    def test_deleting_a_row_fails(self) -> None:
        self.write(PREAMBLE + row(1))
        self.commit("delete a row")
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("append-only", result.stdout)
        self.assertIn("#2", result.stdout)

    def test_editing_a_row_in_place_fails(self) -> None:
        """The `FIXED IN FLOW` annotation this policy retired. A union merge
        duplicates an edited row rather than merging it, so an annotation is
        exactly as damaging as a deletion."""
        self.write(PREAMBLE + row(1) + row(2).replace("title", "title FIXED IN FLOW"))
        self.commit("annotate a row")
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("append-only", result.stdout)

    def test_editing_the_preamble_fails(self) -> None:
        self.write(PREAMBLE.replace("# Issue index", "# Issues") + row(1) + row(2))
        self.commit("edit the preamble")
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_an_unresolvable_base_skips_and_says_so(self) -> None:
        """Unknown is not absence and it is not success. A silent PASS here
        would mean every fork PR, where the base often does not resolve,
        reported green on a check that never ran."""
        result = self.run_check(base="origin/does-not-exist")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("SKIP", result.stdout)
        self.assertNotIn("OK:", result.stdout)

    def test_an_untouched_index_passes(self) -> None:
        (self.tmp / "unrelated.txt").write_text("x", encoding="utf-8")
        self.commit("unrelated change")
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
