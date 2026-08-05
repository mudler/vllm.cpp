#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-role.py (W0) and
scripts/check-role-discipline.py (W1).

The behaviours that matter are the ones the protocol rests on: a second
self-declared operator must FAIL rather than race, a session sharing a checkout
must NOT inherit another session's role, a stale lock must be breakable but
never silently, and feature code must not reach main without a row/* PR.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
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


discipline = _load("role_discipline", "scripts/check-role-discipline.py")
ROLE_SCRIPT = ROOT / "scripts/agent-role.py"


def run_role(repo: Path, session: str, *args: str):
    env = dict(os.environ, VLLM_CPP_AGENT_SESSION=session)
    return subprocess.run(
        [sys.executable, str(ROLE_SCRIPT), *args],
        cwd=repo, env=env, capture_output=True, text=True,
    )


class RoleLifecycle(unittest.TestCase):
    """Exercised against a throwaway repo, never the real one."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "root", "--allow-empty"],
                       cwd=self.repo, check=True,
                       env=dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_undeclared_session_exits_3(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)

    def test_claim_then_resolve(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        out = run_role(self.repo, "a", "show")
        self.assertEqual(out.returncode, 0)
        self.assertIn("role=operator", out.stdout)

    def test_second_operator_is_refused(self) -> None:
        """The core mutual-exclusion guarantee."""
        run_role(self.repo, "a", "claim", "operator")
        second = run_role(self.repo, "b", "claim", "operator")
        self.assertEqual(second.returncode, 1)
        self.assertIn("already held", second.stderr)

    def test_other_session_does_not_inherit_the_role(self) -> None:
        """Sessions sharing one checkout must not read each other's marker."""
        run_role(self.repo, "a", "claim", "operator")
        other = run_role(self.repo, "b", "show")
        self.assertEqual(other.returncode, 3)
        self.assertIn("UNDECLARED", other.stdout)

    def test_helper_requires_a_row(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "claim", "helper").returncode, 2)
        ok = run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        self.assertEqual(ok.returncode, 0)
        self.assertIn("row=ENG-FOO", run_role(self.repo, "a", "show").stdout)

    def test_release_frees_the_lock_for_another_session(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        run_role(self.repo, "a", "release")
        self.assertEqual(run_role(self.repo, "b", "claim", "operator").returncode, 0)

    def test_stale_lock_is_broken_but_reported(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip()
        lock = Path(common) / "vllm-cpp-operator.lock"
        record = json.loads(lock.read_text())
        record["heartbeat"] = time.time() - (10 * 60 * 60)
        lock.write_text(json.dumps(record))
        took = run_role(self.repo, "b", "claim", "operator")
        self.assertEqual(took.returncode, 0)
        self.assertIn("STALE", took.stderr)  # broken, but never silently

    def test_operator_marker_without_lock_does_not_resolve(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip()
        (Path(common) / "vllm-cpp-operator.lock").unlink()
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)


class RoleDiscipline(unittest.TestCase):
    def test_feature_path_classification(self) -> None:
        for path in ("src/vllm/a.cpp", "include/vt/b.h", "tests/vt/c.cpp",
                     "CMakeLists.txt", "cmake/x.cmake"):
            self.assertTrue(discipline.is_feature_path(path), path)
        for path in ("scripts/check-x.py", "tests/scripts/test_x.py",
                     ".agents/state.md", "docs/STATUS.md",
                     ".github/workflows/ci.yml"):
            self.assertFalse(discipline.is_feature_path(path), path)

    def test_direct_feature_push_is_a_violation(self) -> None:
        problems = discipline.commit_violations(
            "abc1234", ["p1"], "perf: faster kernel", "", ["src/vllm/a.cpp"]
        )
        self.assertTrue(problems)
        self.assertIn("without a reviewed", problems[0])

    def test_row_pr_merge_is_accepted(self) -> None:
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["p1", "p2"],
                "Merge pull request #12 from mudler/row/ENG-FOO", "",
                ["src/vllm/a.cpp"]),
            [],
        )

    def test_squash_merge_with_pr_number_is_accepted(self) -> None:
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["p1"], "feat: thing (#12)", "", ["src/vllm/a.cpp"]),
            [],
        )

    def test_integration_only_commit_is_exempt(self) -> None:
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["p1"], "docs: record", "",
                ["scripts/check-x.py", ".agents/state.md", "docs/STATUS.md"]),
            [],
        )

    def test_mixed_commit_is_judged_on_its_feature_paths(self) -> None:
        self.assertTrue(
            discipline.commit_violations(
                "abc1234", ["p1"], "chore", "",
                ["docs/STATUS.md", "src/vllm/a.cpp"])
        )

    def test_enforcement_is_live_and_anchored_to_a_real_commit(self) -> None:
        """Enabled 2026-08-05. The cutover must be a commit that exists."""
        self.assertIsNotNone(discipline.ROLE_DISCIPLINE_SINCE)
        import subprocess
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{discipline.ROLE_DISCIPLINE_SINCE}^{{commit}}"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_the_cutover_commit_itself_is_exempt(self) -> None:
        """History created under the previous direct-push policy stays green."""
        self.assertTrue(discipline.enforced(discipline.ROLE_DISCIPLINE_SINCE))
        first = discipline.git("rev-list", "--max-parents=0", "HEAD").split()[0]
        self.assertFalse(discipline.enforced(first))

    def test_a_direct_feature_push_after_cutover_now_FAILS(self) -> None:
        """The whole point of enabling it: this is an error, not a report."""
        problems = discipline.commit_violations(
            "deadbee", ["p1"], "perf: hand-edit a kernel", "", ["src/vt/cuda/x.cu"])
        self.assertTrue(problems)
        self.assertTrue(discipline.enforced("HEAD"))

    def test_live_repository_is_reportable(self) -> None:
        self.assertEqual(discipline.main(), 0)


if __name__ == "__main__":
    unittest.main()
