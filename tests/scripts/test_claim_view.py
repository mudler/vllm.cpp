#!/usr/bin/env python3
"""Behavior checks for live, PR-derived helper claims."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


view = _load("claim_view", "scripts/claim-view.py")

KNOWN = {"ENG-FOO", "KV-BAR", "internal-policy-optimization-1"}
EXPECTED = {"repository": "owner/repo", "base": "HEAD"}


def pr(number: int, task: str, *, state: str = "OPEN", head: str | None = None) -> dict:
    return {
        "number": number,
        "state": state,
        "headRefName": head if head is not None else f"row/{task}",
        "isDraft": True,
        "title": task,
        "author": {"login": "helper"},
        "headRepository": {"nameWithOwner": "owner/repo"},
    }



class AgentRecordDynamicLoader(unittest.TestCase):
    def test_checker_loads_without_scripts_on_sys_path(self) -> None:
        checker = ROOT / "scripts/check-agent-record.py"
        scripts = checker.parent.resolve()
        original_path = sys.path[:]
        original_issue_records = sys.modules.pop("issue_records", None)
        module_name = "agent_record_without_scripts_path"
        try:
            sys.path[:] = [
                entry
                for entry in sys.path
                if Path(entry or ".").resolve() != scripts
            ]
            spec = importlib.util.spec_from_file_location(module_name, checker)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            errors = []
            rows = module.parse_claim_rows(ROOT / ".agents/engine-matrix.md", errors)
            self.assertEqual(errors, [])
            self.assertTrue(rows)
        finally:
            sys.modules.pop(module_name, None)
            if original_issue_records is None:
                sys.modules.pop("issue_records", None)
            else:
                sys.modules["issue_records"] = original_issue_records
            sys.path[:] = original_path


class KnownTasks(unittest.TestCase):
    def test_shared_adapter_reads_canonical_matrix_ids(self) -> None:
        ids = view.known_task_ids(ROOT)
        self.assertIn("ENG-ASYNC-SCHED", ids)
        self.assertNotIn("NOT-A-REAL-TASK", ids)

    def test_duplicate_ids_within_or_across_matrices_are_rejected_with_locations(self) -> None:
        class DuplicateRow:
            def __init__(self, path: Path, line_no: int) -> None:
                self.item_id = "ENG-DUP"
                self.path = path
                self.line_no = line_no

        first = ROOT / ".agents/engine-matrix.md"
        second = ROOT / ".agents/feature-matrix.md"
        fake = mock.Mock()
        fake.MATRIX_PATHS = (first, second)
        fake.parse_claim_rows.side_effect = (
            [DuplicateRow(first, 10), DuplicateRow(first, 11)],
            [DuplicateRow(second, 20)],
        )
        with mock.patch.object(view, "_load_record", return_value=fake):
            with self.assertRaisesRegex(
                ValueError,
                r"duplicate task ID ENG-DUP.*engine-matrix\.md:10.*engine-matrix\.md:11.*feature-matrix\.md:20",
            ):
                view.known_task_ids(ROOT)


class LiveClaimValidation(unittest.TestCase):
    def test_one_well_formed_open_claim_passes(self) -> None:
        self.assertEqual(
            view.validate_live_claims([pr(7, "ENG-FOO")], KNOWN, EXPECTED), []
        )

    def test_nonempty_claim_input_requires_expected_repository_authority(self) -> None:
        errors = view.validate_live_claims([pr(7, "ENG-FOO")], KNOWN)
        self.assertEqual(errors, ["nonempty PR input lacks expected repository identity"])

    def test_every_claim_must_match_the_expected_repository(self) -> None:
        candidate = pr(7, "ENG-FOO")
        candidate["headRepository"] = {"nameWithOwner": "attacker/repo"}
        errors = view.validate_live_claims([candidate], KNOWN, EXPECTED)
        self.assertEqual(
            errors,
            ["PR #7 repository 'attacker/repo' does not match expected 'owner/repo'"],
        )
        unrelated = pr(8, "ENG-FOO", head="feature/not-a-claim")
        unrelated["headRepository"] = {"nameWithOwner": "attacker/repo"}
        self.assertEqual(
            view.validate_live_claims([unrelated], KNOWN, EXPECTED),
            ["PR #8 repository 'attacker/repo' does not match expected 'owner/repo'"],
        )

    def test_duplicate_task_claims_are_rejected_deterministically(self) -> None:
        errors = view.validate_live_claims(
            [pr(9, "ENG-FOO"), pr(7, "ENG-FOO")], KNOWN, EXPECTED
        )
        self.assertEqual(
            errors,
            ["task ENG-FOO has duplicate open claims: PR #7, PR #9"],
        )

    def test_unknown_task_is_rejected(self) -> None:
        errors = view.validate_live_claims([pr(7, "UNKNOWN")], KNOWN, EXPECTED)
        self.assertEqual(errors, ["PR #7 claims unknown task UNKNOWN"])

    def test_noncanonical_and_closed_row_heads_are_rejected(self) -> None:
        cases = (
            (pr(7, "ENG-FOO", head="row/KV-BAR"), {"task_id": "ENG-FOO"}, "wrong head"),
            (pr(7, "ENG-FOO", state="CLOSED"), None, "is CLOSED"),
            (pr(7, "ENG-FOO", state="MERGED"), None, "is MERGED"),
        )
        for candidate, expected, message in cases:
            with self.subTest(message=message):
                authority = {**EXPECTED, **(expected or {})}
                errors = view.validate_live_claims([candidate], KNOWN, authority)
                self.assertTrue(any(message in error for error in errors), errors)

    def test_expected_claim_must_exist_and_match_number_and_head(self) -> None:
        expected = {"task_id": "ENG-FOO", "head": "row/ENG-FOO", "number": 8}
        errors = view.validate_live_claims(
            [pr(7, "ENG-FOO")], KNOWN, {**EXPECTED, **expected}
        )
        self.assertTrue(any("expected PR #8" in error for error in errors), errors)

        missing = view.validate_live_claims(
            [], KNOWN, {**EXPECTED, "task_id": "ENG-FOO"}
        )
        self.assertEqual(missing, ["expected live claim for task ENG-FOO is missing"])

    def test_malformed_or_ambiguous_pr_identity_is_rejected(self) -> None:
        malformed = pr(7, "ENG-FOO")
        malformed["state"] = "open"
        duplicate_number = [pr(7, "ENG-FOO"), pr(7, "KV-BAR")]
        errors = view.validate_live_claims(
            [malformed, *duplicate_number], KNOWN, EXPECTED
        )
        self.assertTrue(any("malformed state" in error for error in errors), errors)
        self.assertTrue(any("PR number #7 is ambiguous" in error for error in errors), errors)

    def test_row_claim_requires_unambiguous_repository_identity(self) -> None:
        candidate = pr(7, "ENG-FOO")
        candidate["headRepository"] = None
        errors = view.validate_live_claims([candidate], KNOWN, EXPECTED)
        self.assertEqual(
            errors,
            ["PR #7 has malformed head repository identity"],
        )


class ClaimViewCli(unittest.TestCase):
    def test_local_check_rejects_a_committed_snapshot(self) -> None:
        text = "# Coordination\n\n<!-- claim-view:begin -->\nold\n<!-- claim-view:end -->\n"
        self.assertEqual(
            view.local_errors(text),
            ["committed claim-view snapshots are forbidden; live PR state is authoritative"],
        )

    def test_live_fixture_is_offline_and_green(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "prs.json"
            fixture.write_text(
                json.dumps({"expected": EXPECTED, "prs": [pr(7, "ENG-FOO")]}),
                encoding="utf-8",
            )
            with mock.patch.object(view, "known_task_ids", return_value=KNOWN):
                self.assertEqual(view.main(["--check-live", "--pr-json", str(fixture)]), 0)

    def test_remote_failure_has_distinct_nonzero_result(self) -> None:
        failure = view.RemoteUnverified("gh unavailable")
        stderr = io.StringIO()
        with mock.patch.object(view, "fetch_prs", side_effect=failure), redirect_stderr(stderr):
            rc = view.main(["--check-live"])
        self.assertEqual(rc, view.REMOTE_UNVERIFIED_EXIT)
        self.assertIn("REMOTE_UNVERIFIED", stderr.getvalue())

    def test_modes_are_explicit(self) -> None:
        with self.assertRaises(SystemExit):
            view.main([])

    def test_deprecated_check_alias_is_local_and_never_fetches(self) -> None:
        with mock.patch.object(view, "fetch_prs", side_effect=AssertionError("network")):
            self.assertEqual(view.main(["--check"]), 0)

    def test_deprecated_check_rejects_live_options(self) -> None:
        with self.assertRaises(SystemExit):
            view.main(["--check", "--pr-json", "claims.json"])

    def test_fixture_requires_closed_authority_envelope_and_rejects_duplicates(self) -> None:
        bad_payloads = (
            json.dumps([pr(7, "ENG-FOO")]),
            '{"expected":{"repository":"owner/repo"},"prs":[],"prs":[]}',
            '{"expected":{"repository":"owner/repo","surprise":true},"prs":[]}',
            '{"expected":{},"prs":[]}',
            '{"expected":{"repository":"owner/repo"},"prs":[],"extra":0}',
            '{"expected":{"repository":"owner/repo","number":true},"prs":[]}',
            '{"expected":{"repository":"owner/repo","task_id":7},"prs":[]}',
            '{"expected":{"repository":"owner/repo","head":"main"},"prs":[]}',
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "prs.json"
            for payload in bad_payloads:
                with self.subTest(payload=payload):
                    fixture.write_text(payload, encoding="utf-8")
                    with self.assertRaises(view.RemoteUnverified):
                        view.load_pr_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
