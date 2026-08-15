#!/usr/bin/env python3
"""Tests for the commit writing-style checker."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-commit-style.py"
SPEC = importlib.util.spec_from_file_location("check_commit_style", CHECKER)
assert SPEC is not None and SPEC.loader is not None
check_commit_style = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_commit_style)

PROTOCOL_BLOCK = """FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Codex:GPT-5 [Codex]
"""


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


class RepositoryTest(unittest.TestCase):
    def new_repo(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repo = Path(temporary.name)
        git(repo, "init", "-b", "main")
        git(repo, "config", "user.name", "Commit Style Test")
        git(repo, "config", "user.email", "commit-style@example.com")
        return repo

    def commit(self, repo: Path, subject: str, body: str | None) -> str:
        command = ["commit", "--allow-empty", "-m", subject]
        if body is not None:
            command.extend(("-m", body))
        git(repo, *command)
        return git(repo, "rev-parse", "HEAD")


class CommitMessageContract(unittest.TestCase):
    def test_valid_message(self) -> None:
        message = "policy: check commit prose\n\nExplain why this policy needs a gate.\n"
        self.assertEqual(check_commit_style.validate_commit_message(message), [])

    def test_subject_period_fails_independently(self) -> None:
        message = "policy: check commit prose.\n\nExplain why this policy needs a gate.\n"
        self.assertEqual(
            check_commit_style.validate_commit_message(message),
            ["subject must not end in a period"],
        )

    def test_empty_authored_body_fails_independently(self) -> None:
        message = f"policy: check commit prose\n\n{PROTOCOL_BLOCK}"
        self.assertEqual(
            check_commit_style.validate_commit_message(message),
            ["authored body must not be empty"],
        )

    def test_both_failures_are_reported(self) -> None:
        message = f"policy: check commit prose.\n\n{PROTOCOL_BLOCK}"
        self.assertEqual(
            check_commit_style.validate_commit_message(message),
            [
                "subject must not end in a period",
                "authored body must not be empty",
            ],
        )

    def test_protocol_paragraph_and_trailers_do_not_count_as_authored_prose(self) -> None:
        message = f"policy: check commit prose\n\n{PROTOCOL_BLOCK}"
        self.assertIn(
            "authored body must not be empty",
            check_commit_style.validate_commit_message(message),
        )
        message = (
            "policy: check commit prose\n\n"
            "The policy needs a reason that the diff cannot show.\n\n"
            f"{PROTOCOL_BLOCK}"
        )
        self.assertEqual(check_commit_style.validate_commit_message(message), [])

    def test_measured_sample_shape_reports_23_empty_bodies_and_no_periods(self) -> None:
        messages = [
            f"policy: measured sample {index}\n\nReason {index}.\n"
            for index in range(177)
        ]
        messages.extend(
            f"policy: measured sample {index}\n" for index in range(177, 200)
        )
        failures = [
            error
            for message in messages
            for error in check_commit_style.validate_commit_message(message)
        ]
        self.assertEqual(failures.count("authored body must not be empty"), 23)
        self.assertEqual(failures.count("subject must not end in a period"), 0)


class RangeContract(RepositoryTest):
    def test_range_names_each_failure(self) -> None:
        repo = self.new_repo()
        base = self.commit(repo, "policy: establish the base", "Explain the base.")
        no_body = self.commit(repo, "policy: omit the body", None)
        period = self.commit(repo, "policy: end the subject.", "Explain the period.")

        failures = check_commit_style.validate_range(
            repo, base, period, cutover=None
        )

        self.assertEqual(len(failures), 2)
        self.assertTrue(any(no_body[:12] in failure for failure in failures))
        self.assertTrue(any(period[:12] in failure for failure in failures))
        self.assertTrue(any("authored body" in failure for failure in failures))
        self.assertTrue(any("end in a period" in failure for failure in failures))

    def test_merge_commits_are_excluded(self) -> None:
        repo = self.new_repo()
        base = self.commit(repo, "policy: establish the base", "Explain the base.")
        git(repo, "switch", "-c", "feature")
        self.commit(repo, "policy: add the feature", "Explain the feature.")
        git(repo, "switch", "main")
        self.commit(repo, "policy: advance main", "Explain main.")
        git(repo, "merge", "--no-ff", "feature", "-m", "Merge feature.")
        head = git(repo, "rev-parse", "HEAD")

        self.assertEqual(
            check_commit_style.validate_range(repo, base, head, cutover=None), []
        )

    def test_cutover_excludes_ancestors_and_checks_the_cutover(self) -> None:
        repo = self.new_repo()
        base = self.commit(repo, "policy: establish the base", "Explain the base.")
        legacy = self.commit(repo, "policy: old style.", None)
        cutover = self.commit(repo, "policy: enforce style.", None)

        failures = check_commit_style.validate_range(
            repo, base, cutover, cutover=cutover
        )

        self.assertEqual(len(failures), 2)
        self.assertTrue(all(cutover[:12] in failure for failure in failures))
        self.assertTrue(all(legacy[:12] not in failure for failure in failures))

    def test_revision_and_ancestry_errors_fail_closed(self) -> None:
        repo = self.new_repo()
        root = self.commit(repo, "policy: establish the root", "Explain the root.")
        git(repo, "switch", "-c", "left")
        left = self.commit(repo, "policy: advance left", "Explain left.")
        git(repo, "switch", "-c", "right", root)
        right = self.commit(repo, "policy: advance right", "Explain right.")

        cases = (
            ("missing", right, None, "revision"),
            (left, right, None, "range base must be an ancestor"),
            (root, right, left, "cutover must be reachable"),
        )
        for base, head, cutover, error in cases:
            with self.subTest(base=base, head=head, cutover=cutover):
                with self.assertRaisesRegex(ValueError, error):
                    check_commit_style.validate_range(
                        repo, base, head, cutover=cutover
                    )

    def test_commit_incomparable_with_cutover_fails_closed(self) -> None:
        repo = self.new_repo()
        base = self.commit(repo, "policy: establish the base", "Explain the base.")
        git(repo, "switch", "-c", "feature")
        self.commit(repo, "policy: add the feature", "Explain the feature.")
        git(repo, "switch", "main")
        cutover = self.commit(repo, "policy: set the cutover", "Explain it.")
        git(repo, "merge", "--no-ff", "feature", "-m", "Merge feature")
        head = git(repo, "rev-parse", "HEAD")

        with self.assertRaisesRegex(ValueError, "incomparable with cutover"):
            check_commit_style.validate_range(repo, base, head, cutover=cutover)

    def test_cli_exits_nonzero_and_names_every_failure(self) -> None:
        repo = self.new_repo()
        scripts = repo / "scripts"
        scripts.mkdir()
        shutil.copy2(CHECKER, scripts / CHECKER.name)
        base = self.commit(repo, "policy: establish the base", "Explain the base.")
        no_body = self.commit(repo, "policy: omit the body", None)
        period = self.commit(repo, "policy: end the subject.", "Explain the period.")

        result = subprocess.run(
            [
                sys.executable,
                str(scripts / CHECKER.name),
                "--range",
                f"{base}..{period}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(no_body[:12], result.stderr)
        self.assertIn(period[:12], result.stderr)
        self.assertIn("authored body must not be empty", result.stderr)
        self.assertIn("subject must not end in a period", result.stderr)


if __name__ == "__main__":
    unittest.main()
