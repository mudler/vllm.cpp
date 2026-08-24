#!/usr/bin/env python3
"""Behavior and negative tests for readiness and integration gates."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ready = load("ready_gate_test", ROOT / "scripts/agent-ready.py")
integration = load("integration_gate_test", ROOT / "scripts/agent-integration.py")


EXPECTED = {
    "repository": "owner/repo",
    "base": "main",
    "base_oid": "b" * 40,
    "head_branch": "row/POLICY-1",
    "head_sha": "a" * 40,
}


def pr(**changes):
    value = {
        "number": 128,
        "state": "OPEN",
        "isDraft": True,
        "headRefName": EXPECTED["head_branch"],
        "headRefOid": EXPECTED["head_sha"],
        "headRepository": {"nameWithOwner": EXPECTED["repository"]},
        "baseRefName": EXPECTED["base"],
        "baseRefOid": EXPECTED["base_oid"],
        "statusCheckRollup": [
            {"name": "policy", "status": "COMPLETED", "conclusion": "SUCCESS"}
        ],
        "reviewDecision": "APPROVED",
        "reviews": [],
    }
    value.update(changes)
    return value


def payload(*prs):
    return {"expected": dict(EXPECTED), "prs": list(prs or (pr(),))}


class AgentGateBootstrapTests(unittest.TestCase):
    def test_gate_entry_points_exist(self) -> None:
        """Removing either required gate entry point must fail the suite."""
        self.assertTrue((ROOT / "scripts/agent-ready.py").is_file())
        self.assertTrue((ROOT / "scripts/agent-integration.py").is_file())

    def test_ci_trailer_gate_uses_event_head_not_synthetic_checkout_head(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn('head="$PR_HEAD"', workflow)
        self.assertIn('head="$PUSH_HEAD"', workflow)
        self.assertIn('--range "$base..$head"', workflow)
        self.assertNotIn('--range "$base..HEAD"', workflow)

    def test_ci_role_suite_uses_exact_event_range_not_detached_head(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        role_step = workflow.split(
            "- name: Agent role machinery and role discipline", 1
        )[1].split("- name: Claim view, helper queue and PR reviewability", 1)[0]
        self.assertIn('head="$PR_HEAD"', role_step)
        self.assertIn('head="$PUSH_HEAD"', role_step)
        # The base selection moved out of this step and into
        # `scripts/ci-walk-base.py` (#1809). What this case is about does not
        # change: the base comes from the EVENT payload and never from the
        # runner's checkout. Both event values still reach the resolver, the
        # resolver's output is the base, and the lane split it applies is
        # asserted by EXECUTING it rather than by matching a string that no
        # longer exists.
        self.assertIn('base="$(python3 scripts/ci-walk-base.py', role_step)
        self.assertIn('--pr-base "${PR_BASE:-}"', role_step)
        self.assertIn('--push-base "${PUSH_BASE:-}"', role_step)
        for detached in ('base="HEAD', '--base "HEAD', 'base="$(git '):
            self.assertNotIn(detached, role_step, "the base must not come from the checkout")
        self.assertIn('pending_args=(--pending-pr-head "$PR_HEAD")', role_step)
        self.assertIn('--base "$base" --head "$head"', role_step)
        resolver = [
            sys.executable,
            str(ROOT / "scripts/ci-walk-base.py"),
            "--head", "b" * 40,
            "--pr-base", "a" * 40,
            "--push-base", "c" * 40,
            "--last-green", "d" * 40,
            "--floor", "",
            "--repo", str(ROOT),
        ]
        self.assertEqual(
            subprocess.check_output(
                [*resolver, "--event", "pull_request"], text=True
            ).strip(),
            "a" * 40,
            "the pull request lane must base on the pull request event's base",
        )
        self.assertEqual(
            subprocess.check_output([*resolver, "--event", "push"], text=True).strip(),
            "d" * 40,
            "the push lane must base on the last gated commit",
        )


class ReadyContractTests(unittest.TestCase):
    def test_exact_live_pr_and_green_ci_pass(self) -> None:
        self.assertEqual(ready.ready_errors(payload(), EXPECTED), [])

    def test_fixture_identity_cannot_override_local_authority(self) -> None:
        forged = payload()
        forged["expected"]["repository"] = "attacker/repo"
        self.assertIn("fixture identity", ready.ready_errors(forged, EXPECTED)[0])

    def test_wrong_remote_identity_or_lifecycle_fails(self) -> None:
        mutations = (
            {"headRepository": {"nameWithOwner": "attacker/repo"}},
            {"state": "CLOSED"},
            {"headRefOid": "c" * 40},
            {"baseRefName": "release"},
            {"baseRefOid": "d" * 40},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assertTrue(ready.ready_errors(payload(pr(**mutation)), EXPECTED))

    def test_duplicate_or_missing_claim_fails(self) -> None:
        self.assertTrue(ready.ready_errors(payload(pr(), pr(number=129)), EXPECTED))
        self.assertTrue(ready.ready_errors({"expected": EXPECTED, "prs": []}, EXPECTED))

    def test_pending_red_missing_or_malformed_ci_fails(self) -> None:
        rollups = (
            [],
            [{"name": "ci", "status": "IN_PROGRESS", "conclusion": ""}],
            [{"name": "ci", "status": "COMPLETED", "conclusion": "FAILURE"}],
            ["green"],
        )
        for rollup in rollups:
            with self.subTest(rollup=rollup):
                self.assertTrue(
                    ready.ready_errors(payload(pr(statusCheckRollup=rollup)), EXPECTED)
                )

    def test_fixture_rejects_unknown_and_duplicate_json_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pr.json"
            path.write_text('{"expected":{},"prs":[],"extra":0}', encoding="utf-8")
            with self.assertRaises(ready.RemoteUnverified):
                ready.load_payload(path)
            path.write_text('{"expected":{},"prs":[],"prs":[]}', encoding="utf-8")
            with self.assertRaises(ready.RemoteUnverified):
                ready.load_payload(path)

    def test_missing_fixture_is_remote_unverified(self) -> None:
        with self.assertRaises(ready.RemoteUnverified):
            ready.load_payload(Path("/definitely/missing/pr.json"))

    def test_live_query_failure_is_remote_unverified(self) -> None:
        failed = subprocess.CompletedProcess(["gh"], 1, stdout="", stderr="offline")
        with mock.patch.object(ready.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(ready.RemoteUnverified, "offline"):
                ready.query_remote(EXPECTED)


class IntegrationContractTests(unittest.TestCase):
    def test_review_requires_explicit_approval(self) -> None:
        self.assertEqual(integration.review_errors(payload(), EXPECTED), [])
        for decision in ("", "CHANGES_REQUESTED", None):
            with self.subTest(decision=decision):
                self.assertTrue(
                    integration.review_errors(payload(pr(reviewDecision=decision)), EXPECTED)
                )

    def test_second_snapshot_must_remain_ready_after_first_gate_passes(self) -> None:
        stale = payload(
            pr(
                headRefOid="c" * 40,
                statusCheckRollup=[
                    {"name": "policy", "status": "COMPLETED", "conclusion": "FAILURE"}
                ],
            )
        )
        with (
            mock.patch.object(integration, "run_ready", return_value=True),
            mock.patch.object(integration.ready, "local_expected", return_value=EXPECTED),
            mock.patch.object(integration.ready, "load_payload", return_value=stale),
            mock.patch.object(integration, "base_freshness_errors", return_value=[]),
            mock.patch.object(integration, "cutover_oid", return_value="d" * 40),
            mock.patch.object(
                integration.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            mock.patch.object(
                sys,
                "argv",
                ["agent-integration.py", "--base", "origin/main", "--pr-json", "pr.json"],
            ),
        ):
            self.assertEqual(integration.main(), 1)

    def test_trailer_range_starts_at_integration_base_not_cutover(self) -> None:
        completed = subprocess.CompletedProcess([], 0)
        with (
            mock.patch.object(integration, "run_ready", return_value=True),
            mock.patch.object(integration.ready, "local_expected", return_value=EXPECTED),
            mock.patch.object(integration.ready, "load_payload", return_value=payload()),
            mock.patch.object(integration, "base_freshness_errors", return_value=[]),
            mock.patch.object(integration, "cutover_oid", return_value="d" * 40),
            mock.patch.object(integration.subprocess, "run", return_value=completed) as run,
            mock.patch.object(
                sys,
                "argv",
                ["agent-integration.py", "--base", "origin/main", "--pr-json", "pr.json"],
            ),
        ):
            self.assertEqual(integration.main(), 0)
        command = run.call_args.args[0]
        self.assertIn("origin/main..HEAD", command)
        self.assertNotIn(("d" * 40) + "..HEAD", command)

    def test_cutover_and_base_must_be_reachable_and_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
            subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", repo, "config", "user.email", "test@example.com"], check=True)
            (repo / "one").write_text("one", encoding="utf-8")
            subprocess.run(["git", "-C", repo, "add", "one"], check=True)
            subprocess.run(["git", "-C", repo, "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", repo, "branch", "origin/main"], check=True)
            (repo / "two").write_text("two", encoding="utf-8")
            subprocess.run(["git", "-C", repo, "add", "two"], check=True)
            subprocess.run(["git", "-C", repo, "commit", "-qm", "head"], check=True)
            cutover = repo / ".agents/policy-cutover"
            cutover.parent.mkdir()
            cutover.write_text(base + "\n", encoding="utf-8")
            old_root, old_cutover = integration.ROOT, integration.CUTOVER_FILE
            integration.ROOT, integration.CUTOVER_FILE = repo, cutover
            try:
                self.assertEqual(integration.cutover_oid(), base)
                self.assertEqual(integration.base_freshness_errors("origin/main", base), [])
                cutover.write_text("not-a-commit\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    integration.cutover_oid()
                cutover.unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    integration.cutover_oid()
                self.assertTrue(integration.base_freshness_errors("origin/main", "f" * 40))
                subprocess.run(["git", "-C", repo, "checkout", "-q", "--orphan", "foreign"], check=True)
                subprocess.run(["git", "-C", repo, "rm", "-q", "-rf", "."], check=True)
                (repo / "foreign").write_text("foreign", encoding="utf-8")
                subprocess.run(["git", "-C", repo, "add", "foreign"], check=True)
                subprocess.run(["git", "-C", repo, "commit", "-qm", "foreign"], check=True)
                foreign = subprocess.check_output(
                    ["git", "-C", repo, "rev-parse", "HEAD"], text=True
                ).strip()
                subprocess.run(["git", "-C", repo, "checkout", "-q", "main"], check=True)
                cutover.write_text(foreign + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "ancestor"):
                    integration.cutover_oid()
            finally:
                integration.ROOT, integration.CUTOVER_FILE = old_root, old_cutover

    def test_matching_base_oid_is_rejected_when_not_head_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
            subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
            subprocess.run(
                ["git", "-C", repo, "config", "user.email", "test@example.com"],
                check=True,
            )
            (repo / "root").write_text("root", encoding="utf-8")
            subprocess.run(["git", "-C", repo, "add", "root"], check=True)
            subprocess.run(["git", "-C", repo, "commit", "-qm", "root"], check=True)
            subprocess.run(["git", "-C", repo, "checkout", "-qb", "unrelated-base"], check=True)
            (repo / "base").write_text("base", encoding="utf-8")
            subprocess.run(["git", "-C", repo, "add", "base"], check=True)
            subprocess.run(["git", "-C", repo, "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "HEAD"], text=True
            ).strip()
            subprocess.run(["git", "-C", repo, "checkout", "-q", "main"], check=True)
            (repo / "head").write_text("head", encoding="utf-8")
            subprocess.run(["git", "-C", repo, "add", "head"], check=True)
            subprocess.run(["git", "-C", repo, "commit", "-qm", "head"], check=True)

            old_root = integration.ROOT
            integration.ROOT = repo
            try:
                errors = integration.base_freshness_errors("unrelated-base", base)
            finally:
                integration.ROOT = old_root

            self.assertTrue(errors)
            self.assertNotIn(
                "local integration base does not match the current PR base SHA", errors
            )


if __name__ == "__main__":
    unittest.main()
