#!/usr/bin/env python3
"""Unit and behaviour checks for scripts/ci-walk-base.py.

The base of the diff-scoped walk used to be four byte-similar copies of inline
shell inside `.github/workflows/ci.yml`. One of them, `agent-record`'s, was
replayed by `test_main_baseline.py::AgentRecordDiffRangeTests` under a shim that
stubs every `python3` call: that pins WHICH checker is invoked and with WHICH
range string, and it cannot see the base rule. The other three were executed by
nothing. So the ratchet in #1809 could only be found by reading a job log.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts/ci-walk-base.py"
WORKFLOW = ROOT / ".github/workflows/ci.yml"

SPEC = importlib.util.spec_from_file_location("ci_walk_base", MODULE)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

ZERO = "0" * 40


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


class ScratchRepo:
    """A linear throwaway history, so ancestry questions have real answers."""

    def __init__(self, directory: Path) -> None:
        self.path = directory
        git(directory, "init", "--quiet", "--initial-branch=main", ".")
        git(directory, "config", "user.email", "test@example.invalid")
        git(directory, "config", "user.name", "Test")
        git(directory, "config", "commit.gpgsign", "false")

    def commit(self, name: str) -> str:
        (self.path / name).write_text(name, encoding="utf-8")
        git(self.path, "add", name)
        git(self.path, "commit", "--quiet", "-m", name)
        return git(self.path, "rev-parse", "HEAD")


class RepoCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="ci-walk-base-")
        self.addCleanup(self._tmp.cleanup)
        self.repo = ScratchRepo(Path(self._tmp.name))

    def resolve(self, **kwargs) -> str:
        base = mod.resolve_base(
            repo=self.repo.path,
            event=kwargs.pop("event", "push"),
            head=kwargs.pop("head", ""),
            pr_base=kwargs.pop("pr_base", ""),
            push_base=kwargs.pop("push_base", ""),
            last_green=kwargs.pop("last_green", ""),
        )
        self.assertFalse(kwargs, f"unexpected keyword arguments {sorted(kwargs)}")
        return base


class BaseResolutionTests(RepoCase):
    def test_last_green_is_kept(self) -> None:
        green = self.repo.commit("c1")
        head = self.repo.commit("c2")
        self.assertEqual(self.resolve(head=head, last_green=green), green)

    def test_before_is_used_when_no_run_was_green(self) -> None:
        before = self.repo.commit("c1")
        head = self.repo.commit("c2")
        self.assertEqual(self.resolve(head=head, push_base=before), before)

    def test_all_zero_before_is_returned_unchanged(self) -> None:
        head = self.repo.commit("c1")
        # The new-branch guard downstream turns this into a tip-only check.
        self.assertEqual(self.resolve(head=head, push_base=ZERO), ZERO)

    def test_a_before_the_history_no_longer_contains_is_returned_unchanged(self) -> None:
        head = self.repo.commit("c1")
        gone = "d" * 40
        self.assertEqual(self.resolve(head=head, push_base=gone), gone)

    def test_an_empty_base_stays_empty(self) -> None:
        head = self.repo.commit("c1")
        self.assertEqual(self.resolve(head=head), "")


class CancelledRunLosslessTests(RepoCase):
    """#822/#863: a cancelled run's commits must still be covered by a later run.

    The sequence: C1's run is green, C2 is pushed and its run is CANCELLED, C3 is
    pushed. The last successful run is still C1, so the C3 run must walk from C1
    and its range must contain C2.
    """

    def setUp(self) -> None:
        super().setUp()
        self.c1 = self.repo.commit("c1")
        self.c2 = self.repo.commit("c2")
        self.c3 = self.repo.commit("c3")

    def walked(self, base: str, head: str) -> list[str]:
        return git(self.repo.path, "rev-list", f"{base}..{head}").splitlines()

    def test_the_cancelled_commit_is_covered(self) -> None:
        base = self.resolve(head=self.c3, last_green=self.c1, push_base=self.c2)
        self.assertEqual(base, self.c1)
        self.assertIn(self.c2, self.walked(base, self.c3))

    def test_positive_control_the_naive_before_base_loses_the_cancelled_commit(self) -> None:
        # Without this case the one above could pass on a resolver that returned
        # anything at all: this is the failure it is asserting the ABSENCE of,
        # made to happen on purpose.
        self.assertNotIn(self.c2, self.walked(self.c2, self.c3))


class CommandLineTests(RepoCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(MODULE), "--repo", str(self.repo.path), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_it_prints_the_resolved_base(self) -> None:
        green = self.repo.commit("c1")
        head = self.repo.commit("c2")
        result = self.run_script(
            "--event", "push", "--head", head, "--last-green", green
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), green)


class WorkflowWiringTests(unittest.TestCase):
    """Every diff-scoped base in `ci.yml` goes through the script, or this reds.

    A resolver nothing calls resolves nothing. This is the reachability half:
    re-inlining the rule into the YAML, or dropping one call site, fails here.
    """

    def setUp(self) -> None:
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def test_the_three_diff_scoped_steps_call_the_resolver(self) -> None:
        calls = re.findall(r"scripts/ci-walk-base\.py", self.text)
        self.assertEqual(
            len(calls),
            3,
            "the three diff-scoped steps each resolve their base through the script",
        )

    def test_no_step_still_chooses_its_own_base(self) -> None:
        inlined = re.findall(r'base="\$\{LAST_GREEN:-\}"', self.text)
        self.assertEqual(inlined, [], "a step is choosing its base inline again")

    def test_the_resolver_suite_runs_on_a_lane(self) -> None:
        self.assertIn("tests/scripts/test_ci_walk_base.py", self.text)


if __name__ == "__main__":
    unittest.main()
