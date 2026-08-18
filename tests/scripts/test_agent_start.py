#!/usr/bin/env python3
"""Behavior and CLI checks for the universal session entrypoint."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/agent-start.py"


def load_start():
    if not SCRIPT.is_file():
        raise AssertionError("scripts/agent-start.py does not exist")
    spec = importlib.util.spec_from_file_location("agent_start", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_start"] = module
    spec.loader.exec_module(module)
    return module


def state(**changes):
    complete = {
        "role": None,
        "row": None,
        "mode": "interactive",
        "branch": "main",
        "worktree": "/repo/.git",
        "operator_peers": [],
        "reason": "undeclared",
        "env": "present",
        "env_missing": [],
        "queue": ["ENGINE-READY", "MODEL-READY"],
        "queue_error": None,
    }
    complete.update(changes)
    return complete


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = load_start()

    def render(self, route=None, intent=None, row=None, headless=False):
        return self.start.render_route(route or state(), intent, row, headless)

    def test_first_time_route_has_compact_ascii_welcome_and_actions(self):
        out = self.render()
        self.assertEqual(out.count("--- WELCOME: RELAY VERBATIM ---"), 1)
        self.assertEqual(out.count("--- END WELCOME ---"), 1)
        self.assertEqual(out.count("--- AGENT NEXT ACTIONS ---"), 1)
        self.assertEqual(out.count("--- END ACTIONS ---"), 1)
        self.assertIn("Helper", out)
        self.assertIn("one focused contribution", out.lower())
        self.assertIn("Operator", out)
        self.assertIn("long multi-agent campaign", out)
        self.assertIn("Looking", out)
        self.assertIn("read, review, or ask questions", out.lower())
        self.assertIn("Relay only the welcome block above verbatim", out)
        self.assertIn("ask what the contributor is here to do", out)
        self.assertIn("scripts/agent-role.py claim", out)
        self.assertIn("rerun scripts/agent-start.py", out)
        self.assertIn("scripts/agent-preflight.sh", out)
        self.assertTrue(out.isascii())
        self.assertLessEqual(max(map(len, out.splitlines())), 72)

    def test_welcome_constant_is_the_single_rendered_banner(self):
        out = self.render()
        welcome = out.split("--- WELCOME: RELAY VERBATIM ---\n", 1)[1].split(
            "\n--- END WELCOME ---", 1
        )[0]
        self.assertEqual(welcome, self.start.WELCOME)

    def test_explicit_operator_prints_exact_claim_and_suppresses_welcome(self):
        out = self.render(intent="operator")
        self.assertNotIn("WELCOME: RELAY", out)
        self.assertIn("scripts/agent-role.py claim operator", out)
        self.assertNotIn("--headless", out)
        self.assertIn("rerun scripts/agent-start.py", out)

    def test_explicit_headless_operator_propagates_only_explicit_mode(self):
        out = self.render(intent="operator", headless=True)
        self.assertIn("scripts/agent-role.py claim operator --headless", out)

    def test_explicit_read_only_prints_exact_claim(self):
        out = self.render(intent="read-only")
        self.assertNotIn("WELCOME: RELAY", out)
        self.assertIn("scripts/agent-role.py claim read-only", out)

    def test_helper_with_row_prints_exact_claim(self):
        out = self.render(intent="helper", row="ENGINE-FOCUSED")
        self.assertNotIn("WELCOME: RELAY", out)
        self.assertIn(
            "scripts/agent-role.py claim helper --row ENGINE-FOCUSED", out
        )

    def test_headless_helper_appends_mode_after_row(self):
        out = self.render(intent="helper", row="ENGINE-FOCUSED", headless=True)
        self.assertIn(
            "scripts/agent-role.py claim helper --row ENGINE-FOCUSED --headless",
            out,
        )

    def test_helper_without_row_shows_ready_queue_but_no_placeholder_claim(self):
        out = self.render(intent="helper")
        self.assertIn("identify or create one scoped row", out.lower())
        self.assertIn("ENGINE-READY", out)
        self.assertIn("MODEL-READY", out)
        self.assertNotIn("claim helper --row <ROW-ID>", out)
        self.assertNotIn("claim helper --row ENGINE-READY", out)
        self.assertNotIn("ROW-ID", out)

    def test_helper_without_row_distinguishes_unavailable_and_empty_queue(self):
        unavailable = self.render(
            state(queue=[], queue_error="record cannot be parsed"), intent="helper"
        )
        empty = self.render(state(queue=[], queue_error=None), intent="helper")
        self.assertIn("READY queue unavailable", unavailable)
        self.assertIn("record cannot be parsed", unavailable)
        self.assertIn("READY queue is empty", empty)

    def test_declared_roles_report_real_materialized_state_and_no_welcome(self):
        cases = (
            ("operator", None),
            ("helper", "ENGINE-FOCUSED"),
            ("read-only", None),
        )
        for role, row in cases:
            with self.subTest(role=role):
                out = self.render(
                    state(
                        role=role,
                        row=row,
                        mode="headless",
                        branch="row/real-branch",
                        worktree="/repo/.git/worktrees/real-worktree",
                        reason="declared",
                    )
                )
                self.assertNotIn("WELCOME: RELAY", out)
                self.assertIn(f"role: {role}", out)
                if row:
                    self.assertIn(f"row: {row}", out)
                self.assertIn("mode: headless", out)
                self.assertIn("branch: row/real-branch", out)
                self.assertIn("worktree: /repo/.git/worktrees/real-worktree", out)
                self.assertIn("confirm this inherited role fits", out.lower())
                self.assertIn("scripts/agent-preflight.sh", out)
                self.assertIn("printed .agents/NOW.md", out)
                self.assertIn(".agents/developer-preferences.md", out)
                self.assertIn(".agents/coordination.md", out)
                self.assertIn("structured state event anchors", out)

    def test_declared_marker_owns_mode_even_when_no_conflict(self):
        out = self.render(
            state(role="operator", mode="interactive", reason="declared"),
            intent="operator",
            headless=True,
        )
        self.assertIn("mode: interactive", out)
        self.assertNotIn("claim operator --headless", out)

    def test_conflicting_declared_intent_reports_mismatch_without_claim(self):
        out = self.render(
            state(role="helper", row="ENGINE-FOCUSED", reason="declared"),
            intent="operator",
        )
        self.assertNotIn("WELCOME: RELAY", out)
        self.assertIn("intent operator conflicts with declared role helper", out)
        self.assertIn("stop and obtain direction", out.lower())
        self.assertNotIn("scripts/agent-role.py claim operator", out)
        self.assertIn("No worktree or PR was created by this command", out)

    def test_a_recorded_peer_never_blocks_the_operator_claim(self):
        """Issue #285. This test asserted the OPPOSITE until 2026-08-10.

        It required "BLOCKED: the operator lock is held by another live
        worktree" and forbade printing the claim command. `claim operator` is
        now never refused, so that route sent a session away from a command
        that works. The peer is reported as status and the claim is still
        offered.
        """
        out = self.render(
            state(operator_peers=[{"worktree": "/repo/.git/worktrees/other"}]),
            intent="operator",
        )
        self.assertIn("other coordinators: 1 recorded", out)
        self.assertIn("scripts/agent-role.py claim operator", out)
        self.assertNotIn("BLOCKED", out)
        self.assertNotIn("known-failing", out)

    def test_first_time_route_reports_peers_without_withholding_a_role(self):
        out = self.render(
            state(operator_peers=[
                {"worktree": "/repo/.git/worktrees/a"},
                {"worktree": "/repo/.git/worktrees/b"},
            ])
        )
        self.assertIn("WELCOME: RELAY VERBATIM", out)
        self.assertIn("other coordinators: 2 recorded", out)
        # every role stays on the table, and nothing tells the agent to refuse
        self.assertIn("Use the matching scripts/agent-role.py claim command.", out)
        self.assertNotIn("BLOCKED", out)
        self.assertNotIn("report the conflict", out)

    def test_no_peers_renders_no_coordinator_line(self):
        # The other side of the same key: a hardcoded line would announce
        # coordinators that do not exist, which is how a record reads as a lock.
        self.assertNotIn("other coordinators", self.render(state()))

    def test_environment_reports_status_only_and_never_secret_values(self):
        secret = "TOP-SECRET-TOKEN"
        out = self.render(
            state(env="incomplete", env_missing=["VLLM_ORACLE", secret])
        )
        self.assertIn("environment: incomplete", out)
        self.assertNotIn("VLLM_ORACLE", out)
        self.assertNotIn(secret, out)
        unreadable = self.render(state(env="unreadable", env_missing=[secret]))
        missing = self.render(state(env="missing", env_missing=[secret]))
        self.assertIn("environment: unreadable", unreadable)
        self.assertIn("environment: missing", missing)
        self.assertNotIn(secret, unreadable + missing)


    def test_unresolved_environment_routes_to_ask_and_record(self):
        """Issue #1190. Status alone left the session with no next action.

        `environment: missing` was printed and nothing told the agent what to
        do about it, so the fallback in practice was a host name copied from a
        document. The route must send the agent to ASK and then RECORD.
        """
        for status in ("missing", "incomplete", "unreadable"):
            for route in (
                state(env=status, env_missing=["VLLM_ORACLE"]),
                state(env=status, env_missing=["VLLM_ORACLE"], role="operator",
                      reason="declared"),
            ):
                with self.subTest(status=status, role=route["role"]):
                    out = self.start.render_route(route, None, None, False)
                    self.assertIn("NO ENVIRONMENT", out)
                    self.assertIn(f".env is {status}", out)
                    self.assertIn("ask the developer", out.lower())
                    self.assertIn(
                        "scripts/agent-onboard.py --env-set KEY=VALUE", out
                    )
                    self.assertIn("never infer", out.lower())
                    # The unset key names are the caller's own state and can
                    # carry a secret, so the ask names none of them.
                    self.assertNotIn("VLLM_ORACLE", out)
                    self.assertTrue(out.isascii())
                    self.assertLessEqual(max(map(len, out.splitlines())), 72)

    def test_resolved_environment_adds_no_ask_route(self):
        # The other side of the key: a hardcoded block would tell a session
        # with a complete .env to go and ask for a value it already has.
        out = self.render(state(env="present"))
        self.assertNotIn("NO ENVIRONMENT", out)
        self.assertNotIn("--env-set", out)


class MainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = load_start()

    def run_main(self, argv, route=None):
        saved = self.start.onboard.probe
        self.start.onboard.probe = lambda: route or state()
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = self.start.main(argv)
        finally:
            self.start.onboard.probe = saved
        return code, stdout.getvalue(), stderr.getvalue()

    def test_valid_routes_exit_zero(self):
        for argv in (
            [],
            ["--intent", "operator"],
            ["--intent", "read-only"],
            ["--intent", "helper"],
            ["--intent", "helper", "--row", "ENGINE-FOCUSED"],
            ["--intent", "operator", "--headless"],
        ):
            with self.subTest(argv=argv):
                code, out, err = self.run_main(argv)
                self.assertEqual(code, 0, err)
                self.assertIn("--- AGENT NEXT ACTIONS ---", out)

    def test_invalid_flag_combinations_exit_nonzero(self):
        for argv in (
            ["--row", "ENGINE-FOCUSED"],
            ["--intent", "operator", "--row", "ENGINE-FOCUSED"],
            ["--intent", "read-only", "--row", "ENGINE-FOCUSED"],
            ["--headless"],
        ):
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertNotEqual(code, 0)
                self.assertIn("ERROR", err)

    def test_probe_failure_is_nonzero_and_actionable(self):
        def explode():
            raise RuntimeError("probe broke")

        saved = self.start.onboard.probe
        self.start.onboard.probe = explode
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = self.start.main([])
        finally:
            self.start.onboard.probe = saved
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("probe broke", stderr.getvalue())

    def test_real_cli_renders_branch_and_worktree_from_real_probe(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
        self.assertIn(f"branch: {branch}", result.stdout)
        expected = subprocess.check_output(
            ["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, text=True
        ).strip()
        self.assertIn(f"worktree: {expected}", result.stdout)


if __name__ == "__main__":
    unittest.main()
