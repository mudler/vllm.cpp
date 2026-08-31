#!/usr/bin/env python3
"""`scripts/agent-issue-index.py` -- the derived issue index (#2290).

Two properties carry this module, and both are about REFUSING rather than
writing. A snapshot the gates read is only as good as its worst failure: a
partial one is indistinguishable from a complete one at the reader, so every
degraded path must leave the previous file alone and say why.

The shape half is here rather than in the reader's suite on purpose. This module
WRITES the table `check-agent-record.ISSUE_ROW` parses, so the shape belongs to
the writer; gating it at the reader would test the generator through a proxy,
which is what the retired `IssueIndexTableShape` did.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "agent_issue_index", ROOT / "scripts/agent-issue-index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The reader's regex, copied so a drift between writer and reader is RED here.
# Importing it would hide the drift: the point is that two files agree, and a
# shared import proves only that one file agrees with itself.
ISSUE_ROW = re.compile(
    r"^\|\s*\[#(\d+)\]\((https://github\.com/[^)]+/issues/(\d+))\)\s*\|"
    r"\s*(?:`([A-Z0-9][A-Za-z0-9_.-]*)`|—)\s*\|"
)


def completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class RefusesToWriteAPartialSnapshot(unittest.TestCase):
    """Every degraded path leaves the existing file untouched and says why.

    This is the whole safety argument. `now.py` established the contract in #374
    and this mirrors it: unknown is neither absence nor success, so a refresh
    that cannot complete must not leave a shorter table behind for a gate to
    read as the truth.
    """

    def setUp(self) -> None:
        self.mod = load_module()

    def assert_refused(self, run_result, pattern: str) -> None:
        with mock.patch.object(self.mod, "SNAPSHOT") as snapshot:
            with mock.patch.object(subprocess, "run", return_value=run_result):
                code, message = self.mod.refresh()
        self.assertEqual(code, 3, message)
        self.assertRegex(message, pattern)
        snapshot.write_text.assert_not_called()

    def test_a_failed_gh_call_writes_nothing(self) -> None:
        self.assert_refused(completed(1, stderr="gh: not authenticated"),
                            r"REMOTE_UNVERIFIED")

    def test_unparseable_output_writes_nothing(self) -> None:
        self.assert_refused(completed(0, stdout="not json at all"),
                            r"unparseable")

    def test_a_non_array_payload_writes_nothing(self) -> None:
        # `gh` printing an object, or `null`, must not read as "no issues".
        self.assert_refused(completed(0, stdout='{"message":"Not Found"}'),
                            r"no issue array")

    def test_zero_open_issues_writes_nothing(self) -> None:
        # A repository with no open issue and a query that silently matched
        # nothing are identical here, and one of them is a bug.
        self.assert_refused(completed(0, stdout="[]"), r"zero open issues")

    def test_a_missing_gh_binary_is_reported_not_raised(self) -> None:
        with mock.patch.object(self.mod, "SNAPSHOT") as snapshot:
            with mock.patch.object(subprocess, "run", side_effect=OSError("no gh")):
                code, message = self.mod.refresh()
        self.assertEqual(code, 3)
        self.assertRegex(message, r"REMOTE_UNVERIFIED: OSError")
        snapshot.write_text.assert_not_called()

    def test_a_timeout_is_reported_not_raised(self) -> None:
        with mock.patch.object(self.mod, "SNAPSHOT") as snapshot:
            with mock.patch.object(
                subprocess, "run",
                side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=60),
            ):
                code, message = self.mod.refresh()
        self.assertEqual(code, 3)
        self.assertRegex(message, r"REMOTE_UNVERIFIED")
        snapshot.write_text.assert_not_called()


class RendersTheShapeTheReaderParses(unittest.TestCase):
    """The written table is what `check-agent-record.ISSUE_ROW` accepts."""

    def setUp(self) -> None:
        self.mod = load_module()

    def issues(self, *entries):
        return [
            {"number": n, "title": t, "body": b, "labels": [{"name": l} for l in ls]}
            for n, t, b, ls in entries
        ]

    def rows(self, text: str) -> list[re.Match]:
        return [
            ISSUE_ROW.match(line)
            for line in text.splitlines()
            if line.startswith("| [#")
        ]

    def test_every_rendered_row_matches_the_readers_regex(self) -> None:
        text = self.mod.render(self.issues(
            (7, "a title", "Row: `BACKEND-ROCM`\n\nbody", ["bug"]),
            (9, "another", "no row line here", []),
        ))
        matches = self.rows(text)
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(matches), "the reader's regex rejected a rendered row")

    def test_a_row_line_becomes_the_owner_cell(self) -> None:
        text = self.mod.render(self.issues(
            (7, "t", "Row: `BACKEND-ROCM`", ["bug"]),
        ))
        self.assertEqual(self.rows(text)[0].group(4), "BACKEND-ROCM")

    def test_a_body_with_no_row_line_renders_the_dash(self) -> None:
        text = self.mod.render(self.issues((7, "t", "nothing here", [])))
        self.assertIsNone(self.rows(text)[0].group(4))

    def test_an_explicit_dash_is_the_same_as_no_line(self) -> None:
        text = self.mod.render(self.issues((7, "t", "Row: -", [])))
        self.assertIsNone(self.rows(text)[0].group(4))

    def test_a_none_body_does_not_raise(self) -> None:
        # `gh` returns null for an empty body, and a crash here would take the
        # whole refresh down over one issue nobody filled in.
        text = self.mod.render([{"number": 7, "title": "t", "body": None, "labels": []}])
        self.assertIsNone(self.rows(text)[0].group(4))

    def test_a_pipe_in_a_title_cannot_break_the_table(self) -> None:
        # An unescaped pipe splits the row and the reader sees a malformed line
        # or, worse, a shifted owner cell. #1033 was this failure in the tracked
        # index; a generated table must not reintroduce it.
        text = self.mod.render(self.issues((7, "a | b | c", "Row: `X`", [])))
        match = self.rows(text)[0]
        self.assertIsNotNone(match)
        self.assertEqual(match.group(4), "X")

    def test_a_newline_in_a_title_cannot_break_the_table(self) -> None:
        text = self.mod.render(self.issues((7, "one\ntwo", "Row: `X`", [])))
        self.assertEqual(len(self.rows(text)), 1)

    def test_the_number_and_its_url_always_agree(self) -> None:
        # The reader gates this; the writer must never be the one that breaks it.
        text = self.mod.render(self.issues((7, "t", "", []), (12, "u", "", [])))
        for match in self.rows(text):
            self.assertEqual(match.group(1), match.group(3))

    def test_rows_are_sorted_so_a_refresh_is_stable(self) -> None:
        text = self.mod.render(self.issues(
            (12, "b", "", []), (7, "a", "", []), (9, "c", "", []),
        ))
        self.assertEqual([m.group(1) for m in self.rows(text)], ["7", "9", "12"])

    def test_the_label_becomes_the_kind_column(self) -> None:
        text = self.mod.render(self.issues((7, "t", "", ["bug"])))
        self.assertTrue(text.rstrip().endswith("| bug |"))

    def test_an_unlabelled_issue_gets_a_kind_rather_than_an_empty_cell(self) -> None:
        text = self.mod.render(self.issues((7, "t", "", [])))
        self.assertTrue(text.rstrip().endswith("| task |"))


class SnapshotStateIsReportedNeverAssumed(unittest.TestCase):
    """Absence and age both come back with a reason attached."""

    def setUp(self) -> None:
        self.mod = load_module()

    def test_an_absent_snapshot_reports_the_refresh_command(self) -> None:
        with mock.patch.object(self.mod, "SNAPSHOT", ROOT / "does-not-exist.md"):
            text, reason = self.mod.snapshot_state()
        self.assertIsNone(text)
        self.assertRegex(reason, r"agent-issue-index\.py --refresh")

    def test_a_stale_snapshot_returns_its_text_AND_a_reason(self) -> None:
        # Both halves matter: a stale index is still better than none, and a
        # caller that got text without a reason would gate on it silently.
        import time
        with mock.patch.object(self.mod, "STALE_SECONDS", -1):
            path = ROOT / ".agents/issue-index.generated.md"
            if not path.is_file():
                self.skipTest("no snapshot in this tree to age")
            text, reason = self.mod.snapshot_state()
        self.assertIsNotNone(text)
        self.assertRegex(reason, r"old; run")


class BackfillIsIdempotentAndSafe(unittest.TestCase):
    """The one-time migration. It made ~369 writes, so it had to be re-runnable."""

    def setUp(self) -> None:
        self.mod = load_module()

    def test_legacy_rows_skips_a_row_whose_link_names_another_issue(self) -> None:
        # The retired index had no gate against this, and writing a WRONG row
        # into someone's issue body is not repairable by re-running.
        source = ROOT / "does-not-matter.md"
        text = (
            "| [#7](https://github.com/mudler/vllm.cpp/issues/9) | `WRONG` | t | bug |\n"
            "| [#8](https://github.com/mudler/vllm.cpp/issues/8) | `RIGHT` | t | bug |\n"
        )
        with mock.patch.object(Path, "is_file", lambda self: True):
            with mock.patch.object(Path, "read_text", lambda self, encoding=None: text):
                mapping = self.mod.legacy_rows(source)
        self.assertEqual(mapping, {"8": "RIGHT"})

    def test_an_issue_that_already_has_a_row_line_is_not_rewritten(self) -> None:
        # Idempotence: a resumed run must not prepend a second Row line.
        mapping = {"7": "SOME-ROW"}
        issues = [{"number": 7, "title": "t", "body": "Row: `SOME-ROW`\n\nx", "labels": []}]
        with mock.patch.object(self.mod, "legacy_rows", lambda: mapping):
            with mock.patch.object(self.mod, "fetch", lambda limit=1000: (issues, None)):
                code, message = self.mod.backfill(apply=False)
        self.assertEqual(code, 0)
        self.assertRegex(message, r"0 to write, 1 already carry")

    def test_a_dry_run_makes_no_call_to_gh_issue_edit(self) -> None:
        mapping = {"7": "SOME-ROW"}
        issues = [{"number": 7, "title": "t", "body": "no row line", "labels": []}]
        with mock.patch.object(self.mod, "legacy_rows", lambda: mapping):
            with mock.patch.object(self.mod, "fetch", lambda limit=1000: (issues, None)):
                with mock.patch.object(subprocess, "run") as run:
                    code, message = self.mod.backfill(apply=False)
        run.assert_not_called()
        self.assertEqual(code, 0)
        self.assertRegex(message, r"DRY RUN: 1 to write")

    def test_the_existing_body_is_preserved_verbatim_under_the_row_line(self) -> None:
        mapping = {"7": "SOME-ROW"}
        body = "## Problem\n\nsomething **with** markup\n"
        issues = [{"number": 7, "title": "t", "body": body, "labels": []}]
        with mock.patch.object(self.mod, "legacy_rows", lambda: mapping):
            with mock.patch.object(self.mod, "fetch", lambda limit=1000: (issues, None)):
                with mock.patch.object(
                    subprocess, "run", return_value=completed(0)
                ) as run:
                    with mock.patch.object(self.mod.time, "sleep", lambda _: None):
                        code, _ = self.mod.backfill(apply=True)
        self.assertEqual(code, 0)
        written = run.call_args[0][0][-1]
        self.assertTrue(written.startswith("Row: `SOME-ROW`\n\n"))
        self.assertTrue(written.endswith(body))

    def test_a_failed_edit_is_reported_and_the_run_continues(self) -> None:
        # Resumability: one refusal must not abandon the other 368.
        mapping = {"7": "A", "8": "B"}
        issues = [
            {"number": 7, "title": "t", "body": "x", "labels": []},
            {"number": 8, "title": "u", "body": "y", "labels": []},
        ]
        results = [completed(1, stderr="denied"), completed(0)]
        with mock.patch.object(self.mod, "legacy_rows", lambda: mapping):
            with mock.patch.object(self.mod, "fetch", lambda limit=1000: (issues, None)):
                with mock.patch.object(subprocess, "run", side_effect=results):
                    with mock.patch.object(self.mod.time, "sleep", lambda _: None):
                        code, message = self.mod.backfill(apply=True)
        self.assertEqual(code, 1)
        self.assertRegex(message, r"wrote 1 of 2")
        self.assertRegex(message, r"FAILED \(re-run to resume\)")

    def test_a_degraded_fetch_makes_no_edits(self) -> None:
        with mock.patch.object(self.mod, "legacy_rows", lambda: {"7": "A"}):
            with mock.patch.object(
                self.mod, "fetch", lambda limit=1000: (None, "REMOTE_UNVERIFIED: x")
            ):
                with mock.patch.object(subprocess, "run") as run:
                    code, message = self.mod.backfill(apply=True)
        run.assert_not_called()
        self.assertEqual(code, 3)


class FilingAnIssueNoLongerCollides(unittest.TestCase):
    """The campaign's actual claim, executable.

    The measurement that opened this row -- 16 of 21 open pull requests
    CONFLICTING, four of them on the index alone -- is a property of a SHARED
    APPEND TARGET, not of the index's content. These two cases pin the mechanism
    in a scratch repository so the argument cannot rot into folklore: two branches
    that each append to one file collide, and two that each write their own file
    do not.

    `merge.union.driver=false` throughout, because that is what GITHUB does. A
    merge run WITH the driver reproduces the local false green this row exists to
    remove: it resolves cleanly on a developer's machine while the forge reports
    CONFLICTING and never schedules CI at all (#883, #2248).
    """

    def setUp(self) -> None:
        self.tmp = __import__("tempfile").TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git("init", "-q", ".")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "T")
        (self.repo / "seed").write_text("seed\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "seed")
        self.base = self.git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True,
            stderr=subprocess.DEVNULL,
        )

    def branch_writing(self, name: str, path: str, content: str) -> str:
        self.git("checkout", "-q", "-B", name, self.base)
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Append when the file already exists, which is what an index row was.
        existing = target.read_text() if target.exists() else ""
        target.write_text(existing + content)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"{name} files an issue")
        return self.git("rev-parse", "HEAD").strip()

    def conflicts(self, a: str, b: str) -> int:
        out = subprocess.run(
            ["git", "-C", str(self.repo), "-c", "merge.union.driver=false",
             "merge-tree", "--write-tree", a, b],
            capture_output=True, text=True,
        ).stdout
        return sum(1 for line in out.splitlines() if line.startswith("CONFLICT"))

    def test_the_old_shape_collides(self) -> None:
        """A shared append target. This is the control, and it must be RED-ish.

        Without it, the green below proves nothing: a merge that cannot conflict
        for an unrelated reason would pass the same assertion.
        """
        (self.repo / "index.md").write_text("| row 0 |\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "the shared index")
        self.base = self.git("rev-parse", "HEAD").strip()
        a = self.branch_writing("a", "index.md", "| row A |\n")
        b = self.branch_writing("b", "index.md", "| row B |\n")
        self.assertGreater(
            self.conflicts(a, b), 0,
            "the control did not collide, so the case below proves nothing",
        )

    def test_the_new_shape_does_not(self) -> None:
        """Each change carries its own file, and the index is not in the tree."""
        a = self.branch_writing("a", "specs/a.md", "spec A\n")
        b = self.branch_writing("b", "specs/b.md", "spec B\n")
        self.assertEqual(
            self.conflicts(a, b), 0,
            "two independent filings still collide; the lock was not removed",
        )


class TheRetiredIndexStaysRetired(unittest.TestCase):
    """The row's whole point. If the tracked file returns, so does the lock."""

    def test_the_tracked_index_is_gone(self) -> None:
        self.assertFalse(
            (ROOT / ".agents/issue-index.md").exists(),
            "the tracked issue index is back; it is a surface every PR writes",
        )

    def test_the_archive_kept_its_content(self) -> None:
        archive = ROOT / ".agents/completed/issue-index.md"
        self.assertTrue(archive.is_file(), "the archive must keep the provenance")
        self.assertGreater(
            archive.read_text(encoding="utf-8").count("| [#"), 500,
            "the archive lost rows in the move",
        )

    def test_the_union_driver_is_gone_from_gitattributes(self) -> None:
        # The driver is what made the shape look safe locally while the forge
        # conflicted anyway. Leaving it would invite the next shared file.
        text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertNotIn("merge=union", text)

    def test_the_snapshot_is_ignored_so_no_pull_request_can_write_it(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".agents/issue-index.generated.md", text)


if __name__ == "__main__":
    unittest.main()
