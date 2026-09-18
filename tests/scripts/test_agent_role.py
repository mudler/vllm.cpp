#!/usr/bin/env python3
"""Unit and mutation checks for scripts/agent-role.py (W0).

The behaviours that matter are the ones the protocol rests on: a coordinator
must be RECORDED and never refused (issue #285), a session sharing a checkout
must NOT inherit another session's role, and a stale record must be pruned
without blocking anyone.
"""

from __future__ import annotations

import argparse
import builtins
import concurrent.futures
import importlib.util
import io
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


role = _load("agent_role", "scripts/agent-role.py")
ROLE_SCRIPT = ROOT / "scripts/agent-role.py"


def run_role(repo: Path, session: str, *args: str):
    env = dict(os.environ, VLLM_CPP_AGENT_SESSION=session)
    return subprocess.run(
        [sys.executable, str(ROLE_SCRIPT), *args],
        cwd=repo, env=env, capture_output=True, text=True,
    )


# The publisher half of the OUTSIDE OBSERVER (see
# `test_a_concurrent_observer_never_sees_the_record_name_absent`). It runs in its
# own process so the observing test process can watch the published name with no
# knowledge of which I/O primitives the publish uses.
#
# 2000 publishes at ~0.16ms each (see the child loop below) put the window under
# observation at ~0.33s, against ~1us per `exists()` poll. The count is SIZED
# from measurement, not chosen for margin. MEASURED 2026-08-10 on a 20-core box
# at load average ~200, against the publish that only this half can catch
# (bytes first, unlink by basename, rename the temp in), 20 runs per cell:
#
#   publishes | multi-core | one core (taskset -c 3) | clean run, multi / one
#         200 |  16/20 RED |                6/20 RED | 0.25s / 0.23s
#         800 |  19/20 RED |               14/20 RED | 0.37s / 0.44s
#        2000 |  20/20 RED |               20/20 RED | 0.71s / 0.80s
#
# Unmutated stayed 0/3 RED at every count in both regimes. 2000 is the first
# count that is not probabilistic in EITHER regime, and it costs ~0.5s over 200
# on a suite that runs in ~5s -- which is why the single-core residual (issue 296)
# recorded is closed here rather than recorded again.
_OBSERVED_PUBLISHES = 2000

# A wedged child must FAIL this test, not hang the suite. The publish loop takes
# ~0.33s plus interpreter start; 60s is over two orders of magnitude of
# headroom, so reaching it means the child stopped making progress, and the test
# says so instead of spinning until someone kills the run.
_OBSERVER_DEADLINE_SECONDS = 60.0
_OBSERVED_WINDOW_SECONDS = 0.33

_OBSERVED_PUBLISH_LOOP = '''
import importlib.util, sys
from pathlib import Path

script, count = Path(sys.argv[1]), int(sys.argv[2])
spec = importlib.util.spec_from_file_location("agent_role", script)
role = importlib.util.module_from_spec(spec)
sys.modules["agent_role"] = role
spec.loader.exec_module(role)

# The publish's FILE sequence is the subject, so both `git rev-parse` calls on
# the publish path are hoisted out of the loop: `worktree_id` (via
# `record_path`) and `common_dir` (via `records_dir`). Caching only the first
# leaves the second, and MEASURED on a 20-core box at load average 184
# (2026-08-10) that is where nearly all the time was going:
#   * worktree_id cached only          1.467 ms/publish
#   * both cached                      0.163 ms/publish
#   * one bare `git rev-parse`         1.041 ms
# i.e. 89% of the window a half-cached child offers the observer is the
# subprocess it is not supposed to be measuring. Both answers are constant for
# one repo and `record_path`/`records_dir` are their only consumers here, so no
# path this test looks at changes -- it just stops being diluted.
worktree = role.worktree_id()
role.worktree_id = lambda: worktree
common = role.common_dir()
role.common_dir = lambda: common

sys.stdout.write("ready\\n")
sys.stdout.flush()
for _ in range(count):
    role.write_our_record()
'''


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

        A role keys on the worktree, so everything this suite proves --
        coordinator records being per-worktree, and helper isolation -- can only
        be proven with a genuine second worktree rather than a second session id.
        """
        path = self.repo / f".{name}"
        subprocess.run(["git", "worktree", "add", "-q", str(path), "-b", name],
                       cwd=self.repo, check=True, capture_output=True)
        return path

    def common(self) -> Path:
        return Path(subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=self.repo, text=True).strip())

    def records_dir(self) -> Path:
        """Where coordinator records live: shared by every worktree, never
        inside a work tree, and one file per worktree so no two claimants ever
        write the same path."""
        return self.common() / "vllm-cpp-operators"

    def record_files(self) -> list[Path]:
        directory = self.records_dir()
        return sorted(directory.glob("*.json")) if directory.is_dir() else []

    def records(self) -> list[dict]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.record_files()]

    def record_of(self, worktree: Path) -> dict:
        """The record whose worktree is `worktree`'s git dir. Fails if absent."""
        wanted = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=worktree, text=True).strip()
        for record in self.records():
            if record.get("worktree") == wanted:
                return record
        raise AssertionError(f"no coordinator record for {wanted}: {self.records()}")

    def backdate(self, worktree: Path, seconds: float) -> Path:
        """Age one worktree's record past the TTL, as a crashed session leaves it."""
        wanted = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=worktree, text=True).strip()
        for path in self.record_files():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("worktree") == wanted:
                record["heartbeat"] = time.time() - seconds
                path.write_text(json.dumps(record), encoding="utf-8")
                return path
        raise AssertionError(f"no coordinator record for {wanted}")


class RoleLifecycle(_TempRepo, unittest.TestCase):
    """Exercised against a throwaway repo, never the real one."""

    def test_undeclared_session_exits_3(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)

    def test_claim_then_resolve(self) -> None:
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        out = run_role(self.repo, "a", "show")
        self.assertEqual(out.returncode, 0)
        self.assertIn("role=operator", out.stdout)

    def test_a_second_coordinator_is_recorded_not_refused(self) -> None:
        """Issue #285, user-directed: the record must never refuse.

        This test asserted the OPPOSITE until 2026-08-10 -- a second worktree's
        `claim operator` exited 1 with "already held". An operator is a
        coordinator: it merges PRs and dispatches sub-agents into worktrees, and
        it never force-pushes `main`, so git's non-fast-forward refusal is the
        interlock and a JSON file in `.git/` never was one. The refusal cost two
        hours of blocked coordination every time a session died mid-flight.
        """
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        rival = self.worktree("rival")
        second = run_role(rival, "b", "claim", "operator")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotIn("already held", second.stderr)
        # Both are operators, and both are recorded -- neither displaced the other.
        for where in (self.repo, rival):
            self.assertIn("role=operator", run_role(where, "z", "show").stdout)
        self.assertEqual(len(self.record_files()), 2, self.records())

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

    def test_release_removes_this_worktrees_record(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        self.assertEqual(len(self.record_files()), 1)
        run_role(self.repo, "a", "release")
        self.assertEqual(self.record_files(), [])
        self.assertEqual(run_role(self.repo, "b", "claim", "operator").returncode, 0)

    def test_operator_marker_without_a_record_does_not_resolve(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        for path in self.record_files():
            path.unlink()
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

    def test_the_records_are_shared_by_every_worktree(self) -> None:
        # Keying on the worktree must not NARROW the records: they live in the
        # git common dir, shared by every worktree, which is what makes "who is
        # coordinating where" answerable from any of them.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        rival = self.worktree("rival")
        self.assertEqual(run_role(rival, "b", "claim", "operator").returncode, 0)
        # The same two records are visible from either side, and neither lives
        # in a work tree where a commit could pick it up.
        self.assertEqual(len(self.record_files()), 2)
        for where in (self.repo, rival):
            self.assertNotIn("vllm-cpp-operators", subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=where, text=True))

    def test_helper_marker_does_not_leak_into_another_worktree(self) -> None:
        # Isolation, asserted where it is now real. The SAME session id in
        # another worktree must resolve as undeclared: a helper materializes its
        # own worktree, so its marker cannot reach anyone else's.
        run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO")
        out = run_role(self.worktree("elsewhere"), "a", "show")
        self.assertEqual(out.returncode, 3)
        self.assertIn("UNDECLARED", out.stdout)

    def _legacy(self) -> Path:
        """The pre-#285 single-file lock. Still read, so a session that claimed
        before the change does not silently become UNDECLARED mid-flight."""
        return self.common() / "vllm-cpp-operator.lock"

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

    def test_reclaiming_your_own_record_refreshes_the_heartbeat(self) -> None:
        # A live coordinator is likelier to re-claim than to heartbeat, and a
        # record that ages past the TTL while its owner is alive drops out of
        # everyone else's display of who is working where.
        run_role(self.repo, "a", "claim", "operator")
        self.backdate(self.repo, 10 * 60 * 60)
        run_role(self.repo, "b", "claim", "operator")
        self.assertGreater(
            self.record_of(self.repo)["heartbeat"], time.time() - 60)

    def test_a_legacy_lock_file_is_adopted_as_this_worktrees_record(self) -> None:
        # A pre-#285 session holds the single-file lock. It must keep resolving
        # as operator -- turning a live coordinator UNDECLARED mid-flight fails
        # its next preflight -- and the next claim must heal the repo by moving
        # it into the records directory, so the legacy file cannot linger and be
        # counted twice.
        legacy = self._legacy()
        worktree = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=self.repo, text=True).strip()
        legacy.write_text(json.dumps({
            "session": "pre-285", "worktree": worktree,
            "claimed_at": time.time(), "heartbeat": time.time(),
            "host": "somewhere", "pid": 1}), encoding="utf-8")
        (Path(worktree) / "vllm-cpp-agent-role").write_text(
            json.dumps({"role": "operator", "session": "pre-285", "at": time.time()}),
            encoding="utf-8")

        self.assertIn("role=operator", run_role(self.repo, "pre-285", "show").stdout)
        self.assertEqual(run_role(self.repo, "later", "claim", "operator").returncode, 0)
        self.assertFalse(legacy.exists(), "the legacy lock file was not healed away")
        self.assertEqual(len(self.record_files()), 1, self.records())

    def test_an_operator_whose_record_vanished_is_told_to_re_claim(self) -> None:
        # The one path that still refuses to resolve: an operator marker with no
        # record of its own. It is REACHABLE -- a host cleanup deleted exactly
        # this file on 2026-08-10 -- and it must be distinguishable from "never
        # declared", because the remedy is `claim operator`, which now always
        # succeeds. A live coordinator elsewhere changes none of it.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        rival = self.worktree("live-rival")
        self.assertEqual(run_role(rival, "b", "claim", "operator").returncode, 0)
        self.record_of(self.repo)  # precondition: ours exists before we delete it
        for path in self.record_files():
            if json.loads(path.read_text(encoding="utf-8")).get("worktree") == \
                    subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"],
                                            cwd=self.repo, text=True).strip():
                path.unlink()

        state = self._resolve_in(self.repo)
        self.assertIsNone(state["role"])
        self.assertEqual(
            state["reason"],
            "operator marker without a coordinator record; re-claim")
        shown = run_role(self.repo, "a", "show")
        self.assertEqual(shown.returncode, 3)
        # No refusal language survives anywhere: the rival is reported as a
        # peer, never as a blocker, and re-claiming works on the spot.
        self.assertNotIn("cannot be the operator", shown.stdout + shown.stderr)
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        self.assertIn("role=operator", run_role(self.repo, "a", "show").stdout)

    def test_downgrading_out_of_operator_releases_the_record(self) -> None:
        # `claim read-only` ("just looking") is the exact next command an
        # operator types, and leaving the record behind is worse than holding no
        # role: heartbeat answers "not the operator; nothing to heartbeat", so
        # nothing renews it, and the display then shows a coordinator that is
        # not coordinating for the full 2h TTL.
        self.assertEqual(run_role(self.repo, "a", "claim", "operator").returncode, 0)
        self.assertEqual(len(self.record_files()), 1)
        downgrade = run_role(self.repo, "a", "claim", "read-only")
        self.assertEqual(downgrade.returncode, 0, downgrade.stderr)
        self.assertEqual(self.record_files(), [])
        self.assertIn("role=read-only", run_role(self.repo, "a", "show").stdout)

    def test_downgrading_to_helper_releases_the_record_too(self) -> None:
        # Same rule, the other non-operator answer: the release keys on "the
        # claimed role is not operator", not on read-only specifically.
        run_role(self.repo, "a", "claim", "operator")
        self.assertEqual(
            run_role(self.repo, "a", "claim", "helper", "--row", "ENG-FOO").returncode, 0)
        self.assertEqual(self.record_files(), [])

    def test_a_downgrade_never_removes_ANOTHER_worktrees_record(self) -> None:
        # The removal must use the same ownership test `release` does. A
        # read-only claim in one worktree erasing another worktree's record
        # would make the record lie about who is working where.
        run_role(self.repo, "a", "claim", "operator")
        elsewhere = self.worktree("bystander")
        self.assertEqual(run_role(elsewhere, "b", "claim", "read-only").returncode, 0)
        self.assertEqual(len(self.record_files()), 1)
        self.assertIn("role=operator", run_role(self.repo, "a", "show").stdout)

    def test_release_from_a_new_session_removes_the_record(self) -> None:
        # Release has to cross the call boundary as well, or a record written in
        # one call is unreleasable in the next and shows a phantom coordinator
        # until the TTL.
        run_role(self.repo, "call-1", "claim", "operator")
        run_role(self.repo, "call-2-different-pid", "release")
        self.assertEqual(self.record_files(), [])


class CoordinatorRecords(_TempRepo, unittest.TestCase):
    """Issue #285: the file records who is coordinating where; it never refuses.

    Everything here is stated across REAL worktrees, because a record keys on
    the worktree and one file per worktree is what makes concurrent claimants
    safe: two claims write two different paths, so neither can lose the other's
    record, and each publish is a rename over its own path.
    """

    def peers_shown(self, where: Path) -> str:
        shown = run_role(where, "watcher", "show")
        return shown.stdout

    def test_show_lists_the_other_live_coordinators(self) -> None:
        # The point of keeping the file at all: who, which worktree, and how
        # long since the heartbeat. Diagnosing a dead holder needed exactly this
        # on 2026-08-10 and the tool would not say it.
        run_role(self.repo, "primary", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "rival-session", "claim", "operator")
        rival_git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=rival, text=True).strip()

        shown = self.peers_shown(self.repo)
        self.assertIn("other coordinators", shown)
        self.assertIn(rival_git_dir, shown)          # which worktree
        self.assertIn("rival-session", shown)        # who
        self.assertIn("heartbeat", shown)            # how long since
        self.assertRegex(shown, r"\d+s ago")

    def test_show_never_lists_this_worktree_as_a_peer(self) -> None:
        # A record that counted itself would report a coordinator conflict with
        # nobody, which is how a record starts reading like a lock again.
        run_role(self.repo, "solo", "claim", "operator")
        own_git_dir = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=self.repo, text=True).strip()
        shown = run_role(self.repo, "solo", "show").stdout
        self.assertIn("role=operator", shown)
        self.assertNotIn("other coordinators", shown)
        self.assertNotIn(own_git_dir, shown)

    def test_release_removes_only_the_callers_record(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "b", "claim", "operator")

        run_role(self.repo, "a", "release")
        self.assertEqual(len(self.record_files()), 1, self.records())
        self.record_of(rival)  # the survivor is the OTHER worktree's
        self.assertIn("role=operator", run_role(rival, "b", "show").stdout)
        self.assertEqual(run_role(self.repo, "a", "show").returncode, 3)

    def test_a_stale_record_is_pruned_and_never_blocks(self) -> None:
        # A session killed mid-flight leaves a dead pid and a frozen heartbeat.
        # The TTL stays; what changes is the consequence -- the stale record
        # drops out of the display instead of refusing everyone for two hours.
        run_role(self.repo, "crashed", "claim", "operator")
        self.backdate(self.repo, 10 * 60 * 60)
        successor = self.worktree("successor")

        took = run_role(successor, "b", "claim", "operator")
        self.assertEqual(took.returncode, 0, took.stderr)
        self.assertNotIn("crashed", took.stdout + took.stderr)
        shown = run_role(successor, "b", "show").stdout
        self.assertIn("role=operator", shown)
        self.assertNotIn("other coordinators", shown)
        self.assertNotIn("crashed", shown)
        self.assertEqual(len(self.record_files()), 1, self.records())
        self.record_of(successor)

    def test_a_stale_record_is_hidden_before_anything_prunes_it(self) -> None:
        # Pruning is a WRITE, so it happens on claim and never in `show`:
        # agent-preflight.sh documents itself as never writing anything. The
        # display must therefore filter on the TTL itself, not rely on a
        # previous claim having cleaned up.
        run_role(self.repo, "crashed", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "live", "claim", "operator")
        self.backdate(self.repo, 10 * 60 * 60)

        shown = run_role(rival, "live", "show").stdout
        self.assertNotIn("crashed", shown)
        self.assertNotIn("other coordinators", shown)
        self.assertEqual(len(self.record_files()), 2, "show must not delete anything")

    def test_concurrent_claims_lose_no_record(self) -> None:
        # The representation exists for this case. Eight worktrees claim at
        # once; every one of them must end up recorded, and every record must be
        # readable JSON -- a shared file rewritten by eight writers loses some.
        worktrees = [self.worktree(f"coord{index}") for index in range(8)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda pair: run_role(pair[1], f"s{pair[0]}", "claim", "operator"),
                list(enumerate(worktrees))))
        for index, result in enumerate(results):
            self.assertEqual(result.returncode, 0, f"claim {index}: {result.stderr}")
        self.assertEqual(len(self.record_files()), 8, self.records())
        for worktree in worktrees:
            self.record_of(worktree)  # every one survived, none was overwritten
        # And every claimant sees the other seven.
        shown = run_role(worktrees[0], "s0", "show").stdout
        self.assertIn("other coordinators recorded: 7", shown)

    def test_a_claim_never_rewrites_another_worktrees_record(self) -> None:
        # The atomicity argument in one assertion: a writer only ever touches
        # the path derived from its OWN worktree, so a peer's bytes are
        # untouched. If claims ever shared a file this is the first thing to go.
        run_role(self.repo, "a", "claim", "operator")
        before = self.record_files()[0].read_bytes()
        rival = self.worktree("rival")
        run_role(rival, "b", "claim", "operator")
        run_role(rival, "b", "heartbeat")
        run_role(rival, "b", "release")
        self.assertEqual(self.record_files()[0].read_bytes(), before)

    def test_heartbeat_renews_only_this_worktrees_record(self) -> None:
        run_role(self.repo, "a", "claim", "operator")
        rival = self.worktree("rival")
        run_role(rival, "b", "claim", "operator")
        self.backdate(self.repo, 30 * 60)
        self.backdate(rival, 30 * 60)

        beat = run_role(rival, "b", "heartbeat")
        self.assertEqual(beat.returncode, 0, beat.stderr)
        self.assertGreater(self.record_of(rival)["heartbeat"], time.time() - 60)
        self.assertLess(self.record_of(self.repo)["heartbeat"], time.time() - 60)

    def test_a_record_is_never_written_inside_a_work_tree(self) -> None:
        # It must stay uncommittable. `git status` in the worktree that claimed
        # is the check that matters, because that is where a `git add` runs.
        run_role(self.repo, "a", "claim", "operator")
        self.assertTrue(self.records_dir().is_dir())
        self.assertEqual(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repo, text=True).strip(),
            "")


class RecordPublishAndBadInput(_TempRepo, unittest.TestCase):
    """How a record reaches its own path, and what a bad file in the directory
    does to `show`.

    `test_a_claim_never_rewrites_another_worktrees_record` pins the DISJOINT
    PATHS half of the concurrency argument: it fails on shared-path symptoms and
    says nothing about the publish itself. A fresh review (2026-08-10) replaced
    `write temp + os.replace` with an in-place, byte-at-a-time flushing write --
    deliberately torn publishes -- and all three suites stayed green, so the
    mechanism the module docstring calls atomic was pinned by nothing. The same
    review found the re-claim path unlinking its own record before rewriting it,
    and both defensive branches (`record_is_stale`'s unreadable heartbeat,
    `read_records`'s suffix filter) reachable by no test at all.

    A third round then showed both of those repairs still short. The re-claim
    test only ever started from a FRESH record, so it never reached the prune
    that runs first and unlinked our own record once it was stale -- the common
    case, since the TTL is two hours. And both publish tests survive
    `unlink(target); target.write_text(new)`: a new inode leaves the hardlink
    witness intact and no temp is left, but the NAME is transiently absent,
    which is neither the old record nor the new one.

    A fourth round (issue 296) found the same pattern once more, and stopped
    repeating it. Round 3's NAME watcher named four primitives, so a publish
    written `os.rename(target, aside)` then `open(target, "w")` escaped all 59
    tests while a concurrent `show` against the slowed window really did return
    `rc=3, role=UNDECLARED` -- MEASURED here: the pre-issue-296 suite is 59/59 GREEN
    under that mutation. Any watcher is escapable by one more primitive, so
    widening the list cannot be the whole answer. The pin is now two tests that
    fail differently: `_watch_publish` names more primitives and catches every
    one of them deterministically, and
    `test_a_concurrent_observer_never_sees_the_record_name_absent` enumerates
    nothing and can therefore catch the next primitive nobody thought of.
    """

    def test_a_publish_replaces_the_record_and_never_rewrites_it_in_place(self) -> None:
        # A hardlink is the inode a concurrent reader is holding. `os.replace`
        # publishes a NEW inode over the name, so the reader keeps seeing whole
        # old bytes; any in-place rewrite reaches through the link, which is
        # exactly how a reader gets half a record.
        run_role(self.repo, "a", "claim", "operator")
        published = self.record_files()[0]
        before = published.read_bytes()
        witness = published.with_name("witness-hardlink")
        os.link(published, witness)

        beat = run_role(self.repo, "a", "heartbeat")
        self.assertEqual(beat.returncode, 0, beat.stderr)
        self.assertNotEqual(published.read_bytes(), before, "nothing was published")
        self.assertEqual(
            witness.read_bytes(), before,
            "the record was rewritten IN PLACE: a reader holding the previous "
            "inode sees the new bytes, so it can observe a partial record")

    def test_a_publish_leaves_no_temporary_file_behind(self) -> None:
        # The other half of rename-publish: the temp file is CONSUMED by the
        # rename. A publish that copies instead leaves it, and a leftover temp
        # is a record-shaped file nobody owns.
        run_role(self.repo, "a", "claim", "operator")
        run_role(self.repo, "a", "heartbeat")
        run_role(self.repo, "a", "claim", "operator")
        residue = sorted(entry.name for entry in self.records_dir().iterdir()
                         if entry.suffix != ".json")
        self.assertEqual(residue, [], f"publish residue left behind: {residue}")

    def _killed_reclaim(self) -> None:
        """`claim operator` whose publish dies mid-flight. What survives is the
        point: whatever is on disk at that instant is what a killed session
        leaves behind."""
        saved = os.getcwd()
        os.chdir(self.repo)
        try:
            with mock.patch.object(role, "write_our_record",
                                   side_effect=RuntimeError("killed mid-publish")):
                with self.assertRaises(RuntimeError):
                    role.cmd_claim(argparse.Namespace(
                        role="operator", row=None, headless=False))
        finally:
            os.chdir(saved)

    def _watch_publish(self, publish) -> tuple[list[str], list[str]]:
        """Run `publish` with the record's name watched from INSIDE the process.

        Returns (written-while-absent, name-removals). Both must be empty.

        Two families can make the published name stop resolving: removing it
        (`unlink`/`remove`) and renaming it AWAY (`rename`/`replace` with the
        record as SOURCE -- `os.replace(temp, target)`, the shipped publish, has
        it as destination and never removes it). Content writes are watched too,
        because a create that finds the name missing proves it was absent an
        instant earlier whichever call removed it.

        This half is DETERMINISTIC and only ever as complete as the list below.
        Every round of review has fixed one residual here and created the next
        escape: the hardlink and residue pins fell to `unlink` + `write_text`,
        which fell to `os.rename` + `open`. That is why the list is not the whole
        pin -- see `test_a_concurrent_observer_never_sees_the_record_name_absent`,
        which enumerates nothing.
        """
        absent_when_writing: list[str] = []
        removed: list[str] = []
        real = {
            "write_text": Path.write_text,
            "path_unlink": Path.unlink,
            "path_rename": Path.rename,
            "path_replace": Path.replace,
            "os_unlink": os.unlink,
            "os_remove": os.remove,
            "os_rename": os.rename,
            "os_replace": os.replace,
            "os_open": os.open,
            "io_open": io.open,
        }

        saved = os.getcwd()
        os.chdir(self.repo)
        try:
            target = role.record_path()

            def writing(path) -> None:
                if not target.exists():
                    absent_when_writing.append(str(path))

            def removing(path) -> None:
                if Path(path) == target:
                    removed.append(str(path))

            def watched_write_text(path, *args, **kwargs):
                writing(path)
                return real["write_text"](path, *args, **kwargs)

            def watched_io_open(file, mode="r", *args, **kwargs):
                if any(character in mode for character in "wax+"):
                    writing(file)
                return real["io_open"](file, mode, *args, **kwargs)

            def watched_os_open(path, flags, *args, **kwargs):
                if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                    writing(path)
                return real["os_open"](path, flags, *args, **kwargs)

            def watched_path_unlink(path, *args, **kwargs):
                removing(path)
                return real["path_unlink"](path, *args, **kwargs)

            def watched_os_unlink(path, *args, **kwargs):
                removing(path)
                return real["os_unlink"](path, *args, **kwargs)

            def watched_os_remove(path, *args, **kwargs):
                removing(path)
                return real["os_remove"](path, *args, **kwargs)

            def watched_path_rename(path, *args, **kwargs):
                removing(path)
                return real["path_rename"](path, *args, **kwargs)

            def watched_path_replace(path, *args, **kwargs):
                removing(path)
                return real["path_replace"](path, *args, **kwargs)

            def watched_os_rename(source, *args, **kwargs):
                removing(source)
                return real["os_rename"](source, *args, **kwargs)

            def watched_os_replace(source, *args, **kwargs):
                removing(source)
                return real["os_replace"](source, *args, **kwargs)

            with mock.patch.object(Path, "write_text", watched_write_text), \
                    mock.patch.object(Path, "unlink", watched_path_unlink), \
                    mock.patch.object(Path, "rename", watched_path_rename), \
                    mock.patch.object(Path, "replace", watched_path_replace), \
                    mock.patch.object(os, "unlink", watched_os_unlink), \
                    mock.patch.object(os, "remove", watched_os_remove), \
                    mock.patch.object(os, "rename", watched_os_rename), \
                    mock.patch.object(os, "replace", watched_os_replace), \
                    mock.patch.object(os, "open", watched_os_open), \
                    mock.patch.object(io, "open", watched_io_open), \
                    mock.patch.object(builtins, "open", watched_io_open):
                publish()
        finally:
            os.chdir(saved)
        return absent_when_writing, removed

    def test_a_publish_never_leaves_the_record_NAME_absent(self) -> None:
        # The two tests above both survive `target.unlink(); target.write_text()`
        # (review mutation MINE-B, 2026-08-10): a fresh inode leaves the hardlink
        # witness reading the old bytes, and no temp file is left behind. What
        # that publish does do is make the NAME transiently absent, which is
        # neither "the old record" nor "the new one" -- a `show` landing in the
        # window reports an operator marker with no record and exits 3.
        #
        # So the NAME is watched rather than the bytes. The claim this test can
        # actually support is bounded by `_watch_publish`'s list: of the
        # primitives named there, none may remove the published name, and none
        # may write content while that name does not resolve. It holds for
        # temp + os.replace and fails for unlink-then-create and for
        # rename-aside-then-create. The record has to exist first -- "old or new,
        # never absent" says nothing about the first publish, which has no old.
        run_role(self.repo, "a", "claim", "operator")

        absent_when_writing, removed = self._watch_publish(role.write_our_record)

        self.assertEqual(
            removed, [],
            "the publish REMOVED the record name (unlink, or rename away); a "
            "reader in that window sees an operator marker with no record, not "
            f"the old record: {removed}")
        self.assertEqual(
            absent_when_writing, [],
            "the new bytes were written while the record name did not exist, so "
            f"the name was transiently absent: {absent_when_writing}")

    def test_a_concurrent_observer_never_sees_the_record_name_absent(self) -> None:
        # The other half, and the only half that needs no enumeration. A separate
        # process republishes this worktree's record `_OBSERVED_PUBLISHES` times
        # while THIS process does nothing but ask whether the published name
        # resolves. Any publish that removes the name -- by any primitive, named
        # or not, including whichever one escapes the watcher next -- is visible
        # from out here.
        #
        # It is deliberately ONE-SIDED: a hit proves the name was absent, a miss
        # proves nothing. That asymmetry is the point. The observer cannot go red
        # on correct code, because `os.replace` never lets the name stop
        # resolving no matter how the two processes are scheduled; scheduling can
        # only cost SENSITIVITY, never produce a false failure. So it is safe
        # under CI contention in the way a timing assertion never is.
        #
        # RE-MEASURED 2026-08-10 at this file's `_OBSERVED_PUBLISHES` on a
        # 20-core box already at load average 185 (multi-core) and 262 (both
        # processes pinned to one core with `taskset -c 3`), 10 runs per cell:
        #                                              multi-core  single core
        #   * `os.rename` + `open`                          10/10        9/10
        #   * `unlink` + `write_text`                       10/10       10/10
        #   * bytes first, unlink by BASENAME, rename the
        #     temp in -- the one shape NO watcher arm sees  10/10       10/10
        #   * byte-at-a-time IN-PLACE rewrite                0/10        0/10
        #   * unmutated                                      0/10        0/10
        #                                        (zero absent readings, both)
        # The third row is why this half exists at all: it satisfies the
        # hardlink and residue pins and trips no `_watch_publish` arm --
        # `writing` sees the name still present, and both removals arrive as a
        # relative basename that does not compare equal to the absolute target
        # -- so the observer is the ONLY test that fails on it. The fourth row
        # is the converse: this half is blind BY CONSTRUCTION to a publish that
        # never makes the name absent, which is
        # `test_a_publish_replaces_the_record_and_never_rewrites_it_in_place`'s
        # subject. The watcher above holds the line deterministically on every
        # primitive it names; this holds it on the ones nobody named. Neither
        # subsumes the other, and only the pair covers both.
        run_role(self.repo, "a", "claim", "operator")
        saved = os.getcwd()
        os.chdir(self.repo)
        try:
            target = role.record_path()
        finally:
            os.chdir(saved)
        self.assertTrue(target.exists(), "nothing was published to observe")
        # Non-vacuity, checked at the end: `absent == 0` also holds for a future
        # publish that stops touching this path at all, and the `exists()` guard
        # above only proves the path existed BEFORE. The record carries the
        # writer's pid and a fresh heartbeat, so requiring the bytes to change
        # proves the name the observer watched is the name that was republished.
        before = target.read_bytes()

        # Child output goes to FILES, never to a pipe nobody reads. An unread
        # `subprocess.PIPE` deadlocks as soon as the child fills the pipe buffer
        # (65536 bytes here), and this child forks `git rev-parse` per publish,
        # so a few hundred bytes of warning per publish from anyone's unrelated
        # change is enough to cross it. MEASURED 2026-08-10 against the shipped,
        # correct publish with the pipe in place: 391 bytes of child stderr per
        # publish never completed at all -- killed at 90s, rc=124, spinning a
        # core -- while 155 bytes/publish (~31KB, under the pipe) passed in
        # 0.41s. A file has no capacity to fill, so no volume of child output
        # can wedge the observer, and the text still survives for the failure
        # message. Everything below is additionally bounded by a wall-clock
        # deadline, so a child that stops making progress FAILS this test in
        # bounded time instead of hanging a suite everyone has to pass.
        logs = tempfile.TemporaryDirectory()
        self.addCleanup(logs.cleanup)
        out_path = Path(logs.name) / "publisher.out"
        err_path = Path(logs.name) / "publisher.err"
        deadline = time.monotonic() + _OBSERVER_DEADLINE_SECONDS

        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            publisher = subprocess.Popen(
                [sys.executable, "-c", _OBSERVED_PUBLISH_LOOP,
                 str(ROLE_SCRIPT), str(_OBSERVED_PUBLISHES)],
                cwd=self.repo, stdout=out, stderr=err)

        def wedged(what: str) -> None:
            publisher.kill()
            publisher.wait()
            self.fail(
                f"the publisher {what} within {_OBSERVER_DEADLINE_SECONDS:.0f}s "
                f"(~{_OBSERVER_DEADLINE_SECONDS / _OBSERVED_WINDOW_SECONDS:.0f}x "
                f"the measured window for {_OBSERVED_PUBLISHES} publishes); "
                f"killed. "
                f"stderr: {err_path.read_text(encoding='utf-8', errors='replace')[-2000:]!r}")

        try:
            # Import and both `git rev-parse` calls happen before "ready", so
            # the observed window is publishes and nothing else.
            while not out_path.read_bytes().startswith(b"ready\n"):
                if publisher.poll() is not None:
                    break  # died before signalling; the returncode check reports it
                if time.monotonic() > deadline:
                    wedged("never signalled ready")
            absent = 0
            polls = 0
            while publisher.poll() is None:
                polls += 1
                if not target.exists():
                    absent += 1
                # Every 4096 polls (~4ms of polling), so the deadline costs the
                # hot loop nothing measurable and still resolves to milliseconds.
                if polls % 4096 == 0 and time.monotonic() > deadline:
                    wedged("never finished publishing")
        finally:
            if publisher.poll() is None:
                publisher.kill()
            publisher.wait()
        errors = err_path.read_text(encoding="utf-8", errors="replace")

        self.assertEqual(publisher.returncode, 0, errors)
        # A floor for the observer having LOOKED at all, not a timing assertion:
        # the measured count is 148k-257k polls per run (multi-core and pinned
        # to one core alike, load average 312), so this is under 2% of it and
        # can only fire if the poll loop did not run.
        self.assertGreater(polls, _OBSERVED_PUBLISHES,
                           "the observer never got to look; sensitivity unknown")
        self.assertEqual(
            absent, 0,
            f"a concurrent observer saw the published record name NOT resolve "
            f"{absent} times in {polls} polls across {_OBSERVED_PUBLISHES} "
            "publishes: that window is an operator marker with no record")
        self.assertTrue(target.exists(), "the publishes left no record behind")
        self.assertNotEqual(
            target.read_bytes(), before,
            "the record is byte-identical to the one the claim wrote, so the "
            "publishes did not touch the name this test observed and "
            "`absent == 0` says nothing")

    def test_a_reclaim_never_unlinks_its_own_record(self) -> None:
        # Re-claim must REPLACE, never unlink-then-create. With the publish
        # killed mid-flight, the record from the previous claim has to survive
        # byte-for-byte -- an operator marker with no record is the one state
        # that still refuses to resolve, and it is what this change exists to
        # remove.
        #
        # Both ages, because they take DIFFERENT code paths and only the fresh
        # one was covered: `cmd_claim` prunes before it publishes, and a prune
        # that is not scoped to skip our own path unlinks our record whenever it
        # is the stale one. Stale is the COMMON case -- the TTL is two hours and
        # a session re-claims at the top of its next tool call -- and `resolve`
        # matches our own record with no staleness filter, so the state RESOLVES
        # FINE until the re-claim destroys it. One backdate is the whole
        # difference between the two legs.
        for age, backdate_by in (("fresh", None),
                                 ("stale", role.RECORD_TTL_SECONDS + 60)):
            with self.subTest(own_record=age):
                run_role(self.repo, "a", "claim", "operator")
                if backdate_by is not None:
                    self.backdate(self.repo, backdate_by)
                before = self.record_files()[0].read_bytes()
                # Whatever its age, this worktree resolves BEFORE the re-claim.
                # So any refusal afterwards was manufactured by the re-claim.
                self.assertEqual(run_role(self.repo, "a", "show").returncode, 0)

                self._killed_reclaim()

                self.assertEqual(
                    len(self.record_files()), 1,
                    "the re-claim unlinked this worktree's own record before "
                    f"republishing it: {self.records()}")
                self.assertEqual(self.record_files()[0].read_bytes(), before)
                # The state the whole change exists to remove: an operator
                # marker with no record, created out of one that resolved.
                self.assertEqual(run_role(self.repo, "a", "show").returncode, 0)

    def test_show_survives_a_corrupt_record_a_bad_heartbeat_and_a_stray_temp(self) -> None:
        # Nothing exercised either defensive branch. A record whose heartbeat is
        # unreadable must read as STALE, not raise out of `show`; a `.tmp` file
        # caught mid-publish must not parse as a coordinator. And `show` must
        # still write nothing: agent-preflight.sh documents itself as never
        # writing, and it calls exactly this path.
        run_role(self.repo, "a", "claim", "operator")
        directory = self.records_dir()
        (directory / "corrupt.json").write_text("{ not json at all", encoding="utf-8")
        (directory / "bad-heartbeat.json").write_text(json.dumps({
            "session": "bad-beat-session", "worktree": "/nowhere/.git",
            "claimed_at": "not-a-number", "heartbeat": "not-a-number",
            "host": "somewhere", "pid": 4321}), encoding="utf-8")
        (directory / ".half-published.999.tmp").write_text(json.dumps({
            "session": "stray-temp-session", "worktree": "/torn/.git",
            "claimed_at": time.time(), "heartbeat": time.time(),
            "host": "somewhere", "pid": 999}), encoding="utf-8")
        before = sorted((entry.name, entry.read_bytes()) for entry in directory.iterdir())

        shown = run_role(self.repo, "a", "show")

        self.assertEqual(shown.returncode, 0, shown.stdout + shown.stderr)
        self.assertIn("role=operator", shown.stdout)
        self.assertNotIn("Traceback", shown.stderr)
        # Neither bad file may become a coordinator: one is past no TTL it can
        # state, the other is a temp file mid-publish.
        self.assertNotIn("other coordinators", shown.stdout)
        self.assertNotIn("bad-beat-session", shown.stdout)
        self.assertNotIn("stray-temp-session", shown.stdout)
        self.assertEqual(
            sorted((entry.name, entry.read_bytes()) for entry in directory.iterdir()),
            before, "`show` wrote to the records directory")


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

    def test_read_only_writes_no_coordinator_record(self) -> None:
        # The whole reason read-only exists: a session that only reads is not
        # coordinating, so it must not appear in the record of who is.
        self.assertEqual(run_role(self.repo, "a", "claim", "read-only").returncode, 0)
        self.assertEqual(self.record_files(), [])
        self.assertIn("role=read-only", run_role(self.repo, "a", "show").stdout)
        # ... and a real coordinator elsewhere records itself normally.
        self.assertEqual(
            run_role(self.worktree("real-operator"), "b", "claim", "operator").returncode, 0)
        self.assertEqual(len(self.record_files()), 1)


if __name__ == "__main__":
    unittest.main()
