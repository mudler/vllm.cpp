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
from unittest import mock


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
role = _load("agent_role", "scripts/agent-role.py")
ROLE_SCRIPT = ROOT / "scripts/agent-role.py"


def run_role(repo: Path, session: str, *args: str):
    env = dict(os.environ, VLLM_CPP_AGENT_SESSION=session)
    return subprocess.run(
        [sys.executable, str(ROLE_SCRIPT), *args],
        cwd=repo, env=env, capture_output=True, text=True,
    )


class _TempRepo:
    """A throwaway git repo per test. The real checkout is never touched."""

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

    def worktree(self, name: str) -> Path:
        """A real second worktree of the throwaway repo.

        A role keys on the worktree, so both surviving invariants -- one
        operator per repo, and helper isolation -- can only be proven with a
        genuine second worktree rather than a second session id.
        """
        path = self.repo / f".{name}"
        subprocess.run(["git", "worktree", "add", "-q", str(path), "-b", name],
                       cwd=self.repo, check=True, capture_output=True)
        return path


class RoleLifecycle(_TempRepo, unittest.TestCase):
    """Exercised against a throwaway repo, never the real one."""

    def test_undeclared_session_exits_3(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)

    def test_claim_then_resolve(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        out = run_role(self.repo, "a", "show")
        self.assertEqual(out.returncode, 0)
        self.assertIn("role=operator", out.stdout)

    def test_second_operator_is_refused(self) -> None:
        """The core mutual-exclusion guarantee.

        Stated across WORKTREES since the 2026-08-06 correction: a role keys on
        the worktree, so a second session in the SAME worktree is the same
        operator (idempotent), while the lock in the git common dir still
        refuses a genuinely different one.
        """
        run_role(self.repo, "a", "claim", "operator")
        second = run_role(self.worktree("rival"), "b", "claim", "operator")
        self.assertEqual(second.returncode, 1)
        self.assertIn("already held", second.stderr)

    def test_another_session_in_the_same_worktree_shares_the_role(self) -> None:
        """The accepted cost of keying on the worktree, made explicit.

        This test asserted the opposite until 2026-08-06. The session id was
        measured NOT to be stable across tool calls in a real harness, so
        requiring it made a declared role invisible one call later and turned
        --require-role default-on into an unpassable gate rather than a strict
        one. Isolation is preserved where it is real -- see
        test_helper_marker_does_not_leak_into_another_worktree -- and
        .agents/specs/session-onboarding.md records the trade.
        """
        run_role(self.repo, "a", "claim", "operator")
        other = run_role(self.repo, "b", "show")
        self.assertEqual(other.returncode, 0)
        self.assertIn("role=operator", other.stdout)

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
        # From ANOTHER worktree: re-claiming inside the worktree that already
        # holds the lock is the same operator and is idempotent, so a crashed
        # operator can only be displaced from somewhere else.
        took = run_role(self.worktree("successor"), "b", "claim", "operator")
        self.assertEqual(took.returncode, 0)
        self.assertIn("STALE", took.stderr)  # broken, but never silently

    def test_operator_marker_without_lock_does_not_resolve(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip()
        (Path(common) / "vllm-cpp-operator.lock").unlink()
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)


class WorktreeKeyedRole(_TempRepo, unittest.TestCase):
    """A role keys on the WORKTREE, not the session (user-directed 2026-08-06).

    `.agents/specs/session-onboarding.md`, "Correction: a role keys on the
    WORKTREE, not the session". Every test here dies if `resolve()` goes back to
    comparing `marker['session']` to the current process.
    """

    def test_helper_role_survives_a_new_session_id(self) -> None:
        # THE regression this correction exists to prevent: claim in one tool
        # call, resolve in the next, where the parent pid has already changed.
        self.assertEqual(
            run_role(self.repo, "call-1", "claim", "helper", "--row", "ENG-FOO").returncode,
            0,
        )
        later = run_role(self.repo, "call-2-different-pid", "show")
        self.assertEqual(later.returncode, 0)
        self.assertIn("role=helper", later.stdout)
        self.assertIn("row=ENG-FOO", later.stdout)

    def test_operator_role_survives_a_new_session_id(self) -> None:
        # The lock is what makes an operator an operator, so lock OWNERSHIP has
        # to key on the worktree too. Key only the marker and the operator alone
        # still dies at the call boundary, which is the failure that matters
        # most: the operator is the role that lands on main.
        run_role(self.repo, "call-1", "claim", "operator")
        later = run_role(self.repo, "call-2-different-pid", "show")
        self.assertEqual(later.returncode, 0)
        self.assertIn("role=operator", later.stdout)

    def test_resolve_ignores_the_marker_session(self) -> None:
        # The same pin driven through resolve() itself rather than the CLI: a
        # marker whose recorded session is NOT this process's must still
        # resolve, and the recorded session must survive as provenance.
        run_role(self.repo, "some-other-session", "claim", "helper", "--row", "ENG-BAR")
        marker = json.loads(
            (self.repo / ".git/vllm-cpp-agent-role").read_text(encoding="utf-8"))
        self.assertEqual(marker["session"], "some-other-session")

        saved_env = os.environ.get("VLLM_CPP_AGENT_SESSION")
        saved_cwd = os.getcwd()
        os.environ["VLLM_CPP_AGENT_SESSION"] = "a-completely-different-session"
        os.chdir(self.repo)
        try:
            state = role.resolve()
        finally:
            os.chdir(saved_cwd)
            if saved_env is None:
                os.environ.pop("VLLM_CPP_AGENT_SESSION", None)
            else:
                os.environ["VLLM_CPP_AGENT_SESSION"] = saved_env

        self.assertEqual(state["role"], "helper")
        self.assertEqual(state["row"], "ENG-BAR")
        self.assertEqual(state["declared_by"], "some-other-session")

    def test_one_operator_per_repo_holds_across_worktrees(self) -> None:
        # Keying on the worktree must not WIDEN the lock: it lives in the git
        # common dir, shared by every worktree, and that is the scope of "one
        # operator per repo".
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        second = run_role(self.worktree("rival"), "b", "claim", "operator")
        self.assertEqual(second.returncode, 1)
        self.assertIn("already held", second.stderr)

    def test_helper_marker_does_not_leak_into_another_worktree(self) -> None:
        # Isolation, asserted where it is now real. The SAME session id in
        # another worktree must resolve as undeclared: a helper materializes its
        # own worktree, so its marker cannot reach anyone else's.
        run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        out = run_role(self.worktree("elsewhere"), "a", "show")
        self.assertEqual(out.returncode, 3)
        self.assertIn("UNDECLARED", out.stdout)

    def _lock(self, repo: Path) -> Path:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo, text=True).strip()
        return Path(common) / "vllm-cpp-operator.lock"

    def _resolve_in(self, where: Path) -> dict:
        """resolve()'s OWN return value, read in `where`.

        The CLI prints a rendering; several keys of the resolved state never
        reach stdout, so a dict-level assertion is the only way to pin them.
        """
        saved = os.getcwd()
        os.chdir(where)
        try:
            return role.resolve()
        finally:
            os.chdir(saved)

    def test_reclaiming_your_own_lock_refreshes_the_heartbeat(self) -> None:
        # The idempotent branch REWRITES the record instead of passing. With
        # ownership keyed on the worktree a live operator is likelier to
        # re-claim than to heartbeat, and a lock that ages out while its owner
        # is alive gets broken by someone else.
        run_role(self.repo, "a", "claim", "operator")
        lock = self._lock(self.repo)
        record = json.loads(lock.read_text(encoding="utf-8"))
        record["heartbeat"] = time.time() - (10 * 60 * 60)
        lock.write_text(json.dumps(record), encoding="utf-8")

        run_role(self.repo, "b", "claim", "operator")
        self.assertGreater(
            json.loads(lock.read_text(encoding="utf-8"))["heartbeat"],
            time.time() - 60,
        )

    def test_a_legacy_lock_cannot_produce_two_operators(self) -> None:
        # A lock written BEFORE the 2026-08-06 correction carries no worktree,
        # so ownership falls back to its recorded session -- and two worktrees
        # under the same session id would both read it as theirs. Rewriting on
        # the idempotent branch stamps the worktree on, so the fallback heals
        # the first time it is used and only one worktree stays the operator.
        run_role(self.repo, "s1", "claim", "operator")
        lock = self._lock(self.repo)
        legacy = json.loads(lock.read_text(encoding="utf-8"))
        del legacy["worktree"]
        lock.write_text(json.dumps(legacy), encoding="utf-8")

        twin = self.worktree("twin")
        self.assertEqual(run_role(twin, "s1", "claim", "operator").returncode, 0)
        resolved = [
            run_role(self.repo, "s1", "show"),
            run_role(twin, "s1", "show"),
        ]
        operators = [r for r in resolved if r.returncode == 0 and "role=operator" in r.stdout]
        self.assertEqual(
            len(operators), 1,
            f"exactly one operator expected, got {[r.stdout for r in resolved]}")

    def test_an_operator_marker_beaten_to_the_lock_reports_the_lockout(self) -> None:
        # Keying lock OWNERSHIP on the worktree made this branch reachable from
        # a LIVE RIVAL, not only from a self-lost lock: another worktree can now
        # hold the lock while this one still carries an operator marker. Without
        # operator_held_by_other the state is indistinguishable from "never
        # declared" -- same JSON, same cmd_show output, same probe NOTE (none) --
        # and every one of those tells the session to `claim operator`, which
        # exits 1. Deleting the key from this branch must go red.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        self._lock(self.repo).unlink()
        rival = self.worktree("live-rival")
        self.assertEqual(run_role(rival, "b", "claim", "operator").returncode, 0)

        state = self._resolve_in(self.repo)
        self.assertIsNone(state["role"])
        self.assertEqual(
            state["reason"], "operator marker without a held lock; re-claim")
        # .get, so DROPPING the key fails on the value rather than erroring
        # on the lookup: a missing signal is the same defect as a false one.
        self.assertIs(state.get("operator_held_by_other"), True)
        # the CLI surface, which is what an agent actually reads
        shown = run_role(self.repo, "a", "show")
        self.assertEqual(shown.returncode, 3)
        self.assertIn("held by another live session", shown.stdout)

    def test_a_self_lost_lock_is_not_reported_as_a_rival(self) -> None:
        # The other side of the same key: with the lock simply GONE there is no
        # rival, so a blanket True would be a different lie. Hardcoding the key
        # either way now fails one of these two tests.
        run_role(self.repo, "a", "claim", "operator")
        self._lock(self.repo).unlink()
        state = self._resolve_in(self.repo)
        self.assertIsNone(state["role"])
        self.assertIs(state.get("operator_held_by_other"), False)
        self.assertNotIn("another live session", run_role(self.repo, "a", "show").stdout)

    def test_downgrading_out_of_operator_releases_the_lock(self) -> None:
        # `claim read-only` ("just looking") is the exact next command an
        # operator types, and leaving the lock behind is worse than holding no
        # role: heartbeat answers "not the operator; nothing to heartbeat", so
        # nothing renews it, and a second worktree is refused for the full 2h
        # TTL by a session that holds no role at all.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        self.assertTrue(self._lock(self.repo).exists())
        downgrade = run_role(self.repo, "a", "claim", "read-only")
        self.assertEqual(downgrade.returncode, 0, downgrade.stderr)
        self.assertFalse(self._lock(self.repo).exists())
        self.assertIn("role=read-only", run_role(self.repo, "a", "show").stdout)
        # the lock is genuinely free, not merely absent from this worktree
        successor = run_role(self.worktree("successor"), "b", "claim", "operator")
        self.assertEqual(successor.returncode, 0, successor.stderr)

    def test_downgrading_to_helper_releases_the_lock_too(self) -> None:
        # Same rule, the other non-operator answer: the release keys on "the
        # claimed role is not operator", not on read-only specifically.
        run_role(self.repo, "a", "claim", "operator")
        self.assertEqual(
            run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO").returncode, 0)
        self.assertFalse(self._lock(self.repo).exists())

    def test_a_downgrade_never_frees_ANOTHER_worktrees_lock(self) -> None:
        # The release must use the same ownership test `release` does. A
        # read-only claim in one worktree stealing the operator lock from
        # another would be the mutual-exclusion guarantee inverted.
        run_role(self.repo, "a", "claim", "operator")
        elsewhere = self.worktree("bystander")
        self.assertEqual(run_role(elsewhere, "b", "claim", "read-only").returncode, 0)
        self.assertTrue(self._lock(self.repo).exists())
        self.assertIn("role=operator", run_role(self.repo, "a", "show").stdout)

    def test_release_from_a_new_session_frees_the_lock(self) -> None:
        # Release has to cross the call boundary as well, or a lock taken in one
        # call is unreleasable in the next and wedges the repo until the TTL.
        run_role(self.repo, "call-1", "claim", "operator")
        run_role(self.repo, "call-2-different-pid", "release")
        taken = run_role(self.worktree("next"), "c", "claim", "operator")
        self.assertEqual(taken.returncode, 0)


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

    def test_githubs_synthetic_pr_merge_is_accepted(self) -> None:
        """`refs/pull/N/merge` names neither the branch nor the PR.

        GitHub builds it as "Merge <head> into <base>" and CI checks out exactly
        that commit, so before this every feature PR failed a gate about MAIN's
        history on a commit that never lands on main. The reviewed content is the
        SECOND parent: the PR head.
        """
        self.assertEqual(
            discipline.commit_violations(
                "abc1234", ["base", "head"],
                "Merge 01cf15a1 into 4cfeee13", "",
                ["src/vllm/a.cpp"],
                ("feat(videos): a thing\n\nbranch `row/SERVE-VIDEOS-OAI`.",)),
            [],
        )

    def test_a_merge_naming_no_row_anywhere_still_FAILS(self) -> None:
        """The hole the case above must not open: a merge of a NON-row branch."""
        problems = discipline.commit_violations(
            "abc1234", ["base", "head"],
            "Merge 01cf15a1 into 4cfeee13", "",
            ["src/vllm/a.cpp"],
            ("perf: hand-edit a kernel\n\nno branch, no PR",))
        self.assertTrue(problems)
        self.assertIn("without a reviewed", problems[0])
        # And with no merged-branch messages at all (a plain local merge).
        self.assertTrue(
            discipline.commit_violations(
                "abc1234", ["base", "head"], "Merge branch 'wip'", "",
                ["src/vllm/a.cpp"])
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

    def test_exact_pending_pr_range_is_reportable_in_any_checkout(self) -> None:
        head = discipline.git("rev-parse", "HEAD")
        base = discipline.git("rev-parse", "HEAD^")
        saved = sys.argv
        sys.argv = [
            saved[0],
            "--base",
            base,
            "--head",
            head,
            "--pending-pr-head",
            head,
        ]
        try:
            self.assertEqual(discipline.main(), 0)
        finally:
            sys.argv = saved

    def test_landed_detached_commit_remains_strict_without_pending_evidence(self) -> None:
        # The subject here is main()'s DECISION: a violation on a commit that
        # has landed, with no --pending-pr-head evidence, is strict (1), not a
        # REPORT (0). Feed it a fixed violation instead of relying on the real
        # HEAD to be one. It did rely on that, and the coupling was live: under
        # a pull_request event CI checks out the SYNTHETIC merge, whose
        # merged_messages are the PR's own commit bodies, so any PR whose
        # message cites an issue or PR number matched PR_REFERENCE, HEAD stopped
        # being a violation, main() returned 0, and this test failed for a
        # reason that had nothing to do with what it asserts.
        saved = sys.argv
        sys.argv = [saved[0], "--commit", "HEAD"]
        try:
            with mock.patch.object(discipline, "has_reached_main", return_value=True), \
                 mock.patch.object(discipline, "enforced", return_value=True), \
                 mock.patch.object(discipline, "inspect", return_value=["x: landed without a row PR"]):
                self.assertEqual(discipline.main(), 1)
        finally:
            sys.argv = saved

    def test_a_pr_number_in_the_body_does_not_decide_this_gate(self) -> None:
        """Regression: the case that made the test above fail in CI.

        A commit message that merely MENTIONS `#123` must not change main()'s
        landed-vs-pending decision. This pins the decision to the inputs it is
        about, so a message quoting PR numbers cannot flip the outcome again.
        """
        saved = sys.argv
        sys.argv = [saved[0], "--commit", "HEAD"]
        try:
            with mock.patch.object(discipline, "has_reached_main", return_value=True), \
                 mock.patch.object(discipline, "enforced", return_value=True), \
                 mock.patch.object(discipline, "inspect", return_value=["x: see (#157) and #174"]):
                self.assertEqual(discipline.main(), 1)
        finally:
            sys.argv = saved


class ReadOnlyAndModeTests(unittest.TestCase):
    def test_claimable_roles_stay_exactly_two(self):
        # read-only must never become a third claimable role: it takes no lock
        # and no worktree. CLAIMABLE_ROLES is the vocabulary a "may this session
        # write?" test is meant to key on; it has no consumer outside
        # agent-role.py and this suite today, so this pin protects the
        # constant's meaning rather than a live refusal.
        self.assertEqual(role.CLAIMABLE_ROLES, ("operator", "helper"))
        self.assertIn("read-only", role.DECLARABLE)
        self.assertNotIn("read-only", role.CLAIMABLE_ROLES)

    def test_read_only_is_declarable(self):
        self.assertIn("read-only", role.DECLARABLE)

    def test_the_roles_alias_is_not_widened(self):
        # ROLES is the alias a write-gating call site would import; no such
        # call site exists yet, so keeping it the CLAIMABLE pair is what stops
        # the first one from being born wrong. Mutating it to DECLARABLE leaves every other
        # assertion in this suite green, so the constraint that keeps read-only
        # out of "may this session write?" would be enforced by comment only.
        self.assertEqual(role.ROLES, role.CLAIMABLE_ROLES)
        self.assertNotIn("read-only", role.ROLES)

    def test_mode_defaults_to_interactive(self):
        # Headless is DECLARED, never inferred. Absent an explicit flag the
        # session is interactive.
        self.assertEqual(role.mode_from_marker({}), "interactive")
        self.assertEqual(role.mode_from_marker({"mode": "headless"}), "headless")
        self.assertEqual(role.mode_from_marker({"mode": "nonsense"}), "interactive")


class ReadOnlyAndModeResolved(_TempRepo, unittest.TestCase):
    """Drives resolve() itself, not only the pure helpers above.

    mode_from_marker() can be perfectly correct while resolve() never calls it:
    the key would simply be absent from the resolved state and a test that only
    exercised the helper would stay green. So these claim through the real CLI
    and read the mode back out of resolve()'s OWN return value.
    """

    def _resolve_as(self, session: str, where: Path | None = None) -> dict:
        cwd = os.getcwd()
        saved = os.environ.get("VLLM_CPP_AGENT_SESSION")
        os.chdir(where or self.repo)
        os.environ["VLLM_CPP_AGENT_SESSION"] = session
        try:
            return role.resolve()
        finally:
            os.chdir(cwd)
            if saved is None:
                del os.environ["VLLM_CPP_AGENT_SESSION"]
            else:
                os.environ["VLLM_CPP_AGENT_SESSION"] = saved

    def test_resolve_carries_a_declared_headless_mode(self) -> None:
        claimed = run_role(self.repo, "a", "claim", "read-only", "--headless")
        self.assertEqual(claimed.returncode, 0, claimed.stderr)
        state = self._resolve_as("a")
        self.assertEqual(state["role"], "read-only")
        self.assertEqual(state["mode"], "headless")

    def test_resolve_reports_interactive_unless_headless_was_declared(self) -> None:
        run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        self.assertEqual(self._resolve_as("a")["mode"], "interactive")
        # An UNDECLARED context is interactive too: silence is never headless.
        # Genuinely undeclared means another WORKTREE since the 2026-08-06
        # correction; another session id in THIS one resolves to the role that
        # was declared here.
        undeclared = self._resolve_as("b", where=self.worktree("undeclared"))
        self.assertIsNone(undeclared["role"])
        self.assertEqual(undeclared["mode"], "interactive")

    def test_read_only_takes_no_operator_lock(self) -> None:
        # The whole reason read-only exists: a session that only reads must not
        # hold the repo-wide operator lock, or it blocks a real operator.
        self.assertEqual(run_role(self.repo, "a", "claim", "read-only").returncode, 0)
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip()
        self.assertFalse((Path(common) / "vllm-cpp-operator.lock").exists())
        self.assertIn("role=read-only", run_role(self.repo, "a", "show").stdout)
        # ... and a real operator elsewhere is still free to take the lock.
        self.assertEqual(
            run_role(self.worktree("real-operator"), "b", "claim", "operator").returncode, 0)


if __name__ == "__main__":
    unittest.main()
