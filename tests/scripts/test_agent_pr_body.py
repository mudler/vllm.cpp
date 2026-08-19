#!/usr/bin/env python3
"""The pre-merge read of a pull request body, offline and with a stubbed forge.

`squash_merge_commit_message = PR_BODY` makes a body the landed commit message.
`scripts/agent-pr-body.py` is the operator's local read of it before a merge
(#1263). Two properties are worth more than the rest and both are pinned here.

It DELEGATES. The contract lives in `scripts/check-commit-trailers.py` and this
tool must never grow a second copy of it, so one case reads the source and
refuses any trailer grammar of its own.

It CANNOT REACH THE FORGE from a test. Every case below drives the offline
`--body-file` door or a `gh` stub placed on PATH, so the suite runs on a runner
with no credentials and never depends on a live pull request.
"""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/agent-pr-body.py"

# The value that actually landed as the message of 281b4bc76c0e: the grammar is
# `AGENT:MODEL [TOOL]` and the bracketed tool is missing (#1262).
LANDED_MALFORMED = "AGENT:claude-opus-5 CLI"
FILLED = "AGENT:claude-opus-5 [CLI]"
# What the pull request template ships. Legal grammar, attributes nobody, and
# `--filled` is the only thing that rejects it.
PLACEHOLDER = "AGENT:MODEL [TOOL]"

EXIT_OK = 0
EXIT_CONTRACT = 1
EXIT_UNVERIFIED = 3


def body(assisted: str) -> str:
    return (
        "fix(ROW): the subject line of a body that will be squashed\n"
        "\n"
        "One paragraph of reason, because the reader of a commit already has\n"
        "the diff and lacks the why.\n"
        "\n"
        "FOLLOWING_AGENTS_PROTOCOL\n"
        "\n"
        "Following-Agents-Protocol: true\n"
        "AI-Assisted: true\n"
        f"Assisted-by: {assisted}\n"
    )


class ToolHarness(unittest.TestCase):
    """Run the tool as a process, exactly as an operator would."""

    def setUp(self) -> None:
        self.workspace = Path(tempfile.mkdtemp(prefix="agent-pr-body-"))
        self.addCleanup(shutil.rmtree, self.workspace, True)
        self.gh_calls = self.workspace / "gh-was-called"

    def run_tool(self, *args: str, gh: str | None = None):
        env = dict(os.environ)
        if gh is not None:
            stub_dir = self.workspace / "bin"
            stub_dir.mkdir(exist_ok=True)
            stub = stub_dir / "gh"
            stub.write_text(gh, encoding="utf-8")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)
            # PREPENDED, never replaced: the tool still needs `git` on PATH to
            # derive the repository, and a stripped PATH would make every case
            # here fail for a reason that has nothing to do with the forge.
            env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
        return subprocess.run(
            [sys.executable, str(TOOL), *args],
            cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def body_file(self, text: str) -> str:
        path = self.workspace / "body.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def gh_stub(self, *, stdout: str = "", stderr: str = "", code: int = 0) -> str:
        return (
            "#!/bin/sh\n"
            f"printf '%s' {json.dumps(stdout)} \n"
            f"printf '%s' {json.dumps(stderr)} >&2\n"
            f"touch {json.dumps(str(self.gh_calls))}\n"
            f"exit {code}\n"
        )


class OfflineContractTests(ToolHarness):
    def test_the_exact_landed_malformed_value_is_refused(self) -> None:
        """The bytes of 281b4bc76c0e, refused by a command an operator can run."""
        result = self.run_tool("--body-file", self.body_file(body(LANDED_MALFORMED)))
        self.assertEqual(result.returncode, EXIT_CONTRACT, result.stderr)
        self.assertIn(LANDED_MALFORMED, result.stdout + result.stderr)
        self.assertIn("malformed", result.stdout + result.stderr)

    def test_a_filled_body_is_accepted(self) -> None:
        """So the refusal above is a verdict and not a constant."""
        result = self.run_tool("--body-file", self.body_file(body(FILLED)))
        self.assertEqual(result.returncode, EXIT_OK, result.stderr)

    def test_the_template_placeholder_is_refused(self) -> None:
        """`--filled` is passed. This value is legal WITHOUT the flag."""
        result = self.run_tool("--body-file", self.body_file(body(PLACEHOLDER)))
        self.assertEqual(result.returncode, EXIT_CONTRACT, result.stderr)
        self.assertIn("placeholder", result.stdout + result.stderr)

    def test_a_body_with_no_trailer_paragraph_is_refused(self) -> None:
        """The whole contract applies, not one trailer of it."""
        result = self.run_tool("--body-file", self.body_file("Just a description.\n"))
        self.assertEqual(result.returncode, EXIT_CONTRACT, result.stderr)

    def test_an_unreadable_body_file_is_unverified_not_a_verdict(self) -> None:
        """A message that could not be read says nothing about a message."""
        result = self.run_tool("--body-file", str(self.workspace / "absent.txt"))
        self.assertEqual(result.returncode, EXIT_UNVERIFIED, result.stdout)
        self.assertIn("UNVERIFIED", result.stderr)

    def test_exactly_one_of_pr_and_body_file(self) -> None:
        for args in ((), ("--pr", "1", "--body-file", "x")):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_tool(*args).returncode, EXIT_OK)

    def test_the_pr_number_is_rejected_unless_it_is_a_positive_integer(self) -> None:
        """Nothing a shell would expand ever reaches the forge argument list."""
        # The stub keeps the case OFFLINE even when the guard is mutated away:
        # without it a broken validation reaches the real forge from a test.
        stub = self.gh_stub(stdout="{}", code=1)
        for number in ("abc", "1;rm -rf /", "-1", "0", "1 2"):
            with self.subTest(number=number):
                result = self.run_tool("--pr", number, gh=stub)
                self.assertNotEqual(result.returncode, EXIT_OK)
                self.assertNotIn("REMOTE_UNVERIFIED", result.stderr)
                self.assertFalse(self.gh_calls.exists(), "a bad number reached gh")


class FetchTests(ToolHarness):
    def test_a_fetched_body_is_held_to_the_same_contract(self) -> None:
        """The fetching half feeds the validating half, and nothing else does."""
        payload = json.dumps({"body": body(LANDED_MALFORMED)})
        result = self.run_tool("--pr", "1257", gh=self.gh_stub(stdout=payload))
        self.assertEqual(result.returncode, EXIT_CONTRACT, result.stderr)
        self.assertIn(LANDED_MALFORMED, result.stdout + result.stderr)
        self.assertTrue(self.gh_calls.exists(), "the stub was never invoked")

    def test_a_fetched_filled_body_passes(self) -> None:
        payload = json.dumps({"body": body(FILLED)})
        result = self.run_tool("--pr", "1257", gh=self.gh_stub(stdout=payload))
        self.assertEqual(result.returncode, EXIT_OK, result.stderr)

    def test_an_unreachable_forge_is_remote_unverified_and_never_a_verdict(self) -> None:
        """The defect this row exists for, in one case.

        The one-liner #1263 proposed pipes `gh` into the checker, which loses
        `gh`'s exit status and renders "could not resolve a pull request" as
        "the message is empty" -- a verdict about a body nobody read.
        """
        stub = self.gh_stub(
            stderr="GraphQL: Could not resolve to a PullRequest with the number of 999999",
            code=1,
        )
        result = self.run_tool("--pr", "999999", gh=stub)
        self.assertEqual(result.returncode, EXIT_UNVERIFIED, result.stdout)
        self.assertIn("REMOTE_UNVERIFIED", result.stderr)
        self.assertIn("Could not resolve", result.stderr)
        self.assertNotIn("the message is empty", result.stdout + result.stderr)

    def test_an_absent_gh_is_remote_unverified(self) -> None:
        stub = "#!/bin/sh\nexit 127\n"
        result = self.run_tool("--pr", "1", gh=stub)
        self.assertEqual(result.returncode, EXIT_UNVERIFIED, result.stdout)
        self.assertIn("REMOTE_UNVERIFIED", result.stderr)

    def test_malformed_forge_json_is_remote_unverified(self) -> None:
        for payload in ("not json", "[]", json.dumps({}), json.dumps({"body": None})):
            with self.subTest(payload=payload):
                result = self.run_tool("--pr", "1", gh=self.gh_stub(stdout=payload))
                self.assertEqual(
                    result.returncode, EXIT_UNVERIFIED, result.stdout
                )
                self.assertIn("REMOTE_UNVERIFIED", result.stderr)

    def test_an_empty_body_read_successfully_is_a_contract_failure(self) -> None:
        """Read and empty is NOT the same state as never read.

        An empty body lands a commit with no trailers at all, so it fails the
        contract. It is exit 1, and the unreachable forge above is exit 3.
        """
        payload = json.dumps({"body": ""})
        result = self.run_tool("--pr", "1", gh=self.gh_stub(stdout=payload))
        self.assertEqual(result.returncode, EXIT_CONTRACT, result.stdout)
        self.assertNotIn("REMOTE_UNVERIFIED", result.stderr)

    def test_gh_is_not_invoked_by_the_offline_door(self) -> None:
        """`--body-file` makes no network call, which is what makes it gateable."""
        stub = self.gh_stub(stdout="{}", code=1)
        result = self.run_tool(
            "--body-file", self.body_file(body(FILLED)), gh=stub
        )
        self.assertEqual(result.returncode, EXIT_OK, result.stderr)
        self.assertFalse(self.gh_calls.exists(), "the offline door reached gh")


class DelegationTests(unittest.TestCase):
    def test_the_rule_is_not_reimplemented(self) -> None:
        """One implementation of the contract, two callers of it.

        The other caller is `.github/workflows/ci.yml`. This tool must own no
        grammar, no vocabulary and no trailer parsing, or the two can drift.
        """
        source = TOOL.read_text(encoding="utf-8")
        self.assertIn("check-commit-trailers.py", source)
        self.assertIn("--message-file", source)
        self.assertIn("--filled", source)
        for owned_by_the_checker in (
            "Assisted-by",
            "AI-Assisted",
            "FOLLOWING_AGENTS_PROTOCOL",
            "interpret-trailers",
            "Signed-off-by",
        ):
            with self.subTest(literal=owned_by_the_checker):
                self.assertNotIn(owned_by_the_checker, source)

    def test_the_checker_it_delegates_to_exists(self) -> None:
        self.assertTrue((ROOT / "scripts/check-commit-trailers.py").is_file())

    def test_the_landing_procedure_names_the_command(self) -> None:
        """For a command a human types, the documented line IS the call site.

        Both literals are written out here rather than read from the file under
        test, so this cannot pass by agreeing with whatever the document says.
        """
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        workflow = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn("scripts/agent-pr-body.py --pr", agents)
        self.assertIn("scripts/agent-pr-body.py --pr", workflow)
        landing = agents.split("\n## Landing work", 1)[1].split("\n## ", 1)[0]
        self.assertIn("scripts/agent-pr-body.py", landing)

    def test_the_spec_table_names_exactly_these_cases(self) -> None:
        """The row's spec describes this suite, and a table goes stale silently.

        A recorded name drifts inside the pull request that records it, and a
        reader then reviews a table rather than a suite. The two sets are
        compared, not sampled, and neither file is the other's source: the spec
        is prose a human wrote and the names come from the loaded classes.
        """
        spec = (ROOT / ".agents/specs/gate-pr-body-trailers.md").read_text(
            encoding="utf-8"
        )
        section = spec.split("\n## Tests", 1)[1].split("\n## ", 1)[0]
        recorded = set(re.findall(r"`(test_[a-z0-9_]+)`", section))
        module = sys.modules[__name__]
        implemented = {
            name
            for value in vars(module).values()
            if isinstance(value, type) and issubclass(value, unittest.TestCase)
            for name in vars(value)
            if name.startswith("test_")
        }
        self.assertEqual(recorded, implemented)

    def test_the_suite_is_registered_where_gates_run(self) -> None:
        """A suite nothing runs is not a gate."""
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("test_agent_pr_body", preflight)
        self.assertIn("tests/scripts/test_agent_pr_body.py", ci)


if __name__ == "__main__":
    unittest.main()
