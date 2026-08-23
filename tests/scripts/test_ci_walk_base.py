#!/usr/bin/env python3
"""Unit and behaviour checks for scripts/ci-walk-base.py.

The base of the diff-scoped walk used to be four byte-similar copies of inline
shell inside `.github/workflows/ci.yml`. One of them, `agent-record`'s, was
replayed by `test_main_baseline.py::AgentRecordDiffRangeTests` under a shim that
stubs every `python3` call: that pins WHICH checker is invoked and with WHICH
range string, and it cannot see the base rule. The other three were executed by
nothing. So the ratchet in #1809 could only be found by reading a job log.

Two properties are load-bearing and are tested against a REAL throwaway
repository rather than a mock, because both are statements about ancestry:

  * a CANCELLED run stays lossless -- its commits are still covered by the next
    run. This is the property #822 and #863 bought, and the one a floor is most
    likely to break silently. `CancelledRunLosslessTests` replays the sequence
    and carries the naive `github.event.before` base as its POSITIVE CONTROL, so
    a test that stopped discriminating would fail rather than pass;
  * the floor CLAMPS the base from below and never substitutes for an unusable
    one, so the new-branch/force-push guard downstream keeps its behaviour.
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
FLOOR_FILE = ROOT / "scripts/ci-enforcement-floor.txt"
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
        warnings: list[str] = []
        base = mod.resolve_base(
            repo=self.repo.path,
            event=kwargs.pop("event", "push"),
            head=kwargs.pop("head", ""),
            pr_base=kwargs.pop("pr_base", ""),
            push_base=kwargs.pop("push_base", ""),
            last_green=kwargs.pop("last_green", ""),
            floor=kwargs.pop("floor", ""),
            warn=warnings.append,
        )
        self.assertFalse(kwargs, f"unexpected keyword arguments {sorted(kwargs)}")
        self.warnings = warnings
        return base


class FloorRecordTests(unittest.TestCase):
    """The floor record fails CLOSED. A broken record is never read as absent."""

    def write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(text)
        tmp.close()
        path = Path(tmp.name)
        self.addCleanup(path.unlink)
        return path

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        path = self.write(f"# a reason\n\n   \n{'a' * 40}\n")
        self.assertEqual(mod.read_floor(path), "a" * 40)

    def test_two_shas_are_refused(self) -> None:
        path = self.write(f"{'a' * 40}\n{'b' * 40}\n")
        with self.assertRaises(mod.FloorError):
            mod.read_floor(path)

    def test_empty_record_is_refused_rather_than_read_as_no_floor(self) -> None:
        path = self.write("# only a comment\n")
        with self.assertRaises(mod.FloorError):
            mod.read_floor(path)

    def test_short_sha_is_refused(self) -> None:
        path = self.write("bacb71109\n")
        with self.assertRaises(mod.FloorError):
            mod.read_floor(path)

    def test_uppercase_sha_is_refused(self) -> None:
        path = self.write("A" * 40 + "\n")
        with self.assertRaises(mod.FloorError):
            mod.read_floor(path)

    def test_missing_file_is_refused(self) -> None:
        with self.assertRaises(mod.FloorError):
            mod.read_floor(Path("/nonexistent/ci-enforcement-floor.txt"))


class RecordedFloorTests(unittest.TestCase):
    """The floor this repository actually carries is a real ancestor of HEAD."""

    def test_recorded_floor_parses(self) -> None:
        self.assertRegex(mod.read_floor(FLOOR_FILE), r"\A[0-9a-f]{40}\Z")

    def test_recorded_floor_is_an_ancestor_of_head(self) -> None:
        floor = mod.read_floor(FLOOR_FILE)
        self.assertTrue(
            mod.known(ROOT, floor), f"the recorded floor {floor} is not in this checkout"
        )
        self.assertTrue(
            mod.is_ancestor(ROOT, floor, "HEAD"),
            f"the recorded floor {floor} is not an ancestor of HEAD",
        )


class BaseResolutionTests(RepoCase):
    def test_last_green_ahead_of_the_floor_is_kept(self) -> None:
        floor = self.repo.commit("c1")
        green = self.repo.commit("c2")
        head = self.repo.commit("c3")
        self.assertEqual(self.resolve(head=head, last_green=green, floor=floor), green)
        self.assertEqual(self.warnings, [])

    def test_last_green_behind_the_floor_is_raised(self) -> None:
        green = self.repo.commit("c1")
        floor = self.repo.commit("c2")
        head = self.repo.commit("c3")
        self.assertEqual(self.resolve(head=head, last_green=green, floor=floor), floor)
        self.assertEqual(len(self.warnings), 1)

    def test_a_base_equal_to_the_floor_is_kept(self) -> None:
        floor = self.repo.commit("c1")
        head = self.repo.commit("c2")
        self.assertEqual(self.resolve(head=head, last_green=floor, floor=floor), floor)

    def test_before_is_used_when_no_run_was_green(self) -> None:
        floor = self.repo.commit("c1")
        before = self.repo.commit("c2")
        head = self.repo.commit("c3")
        self.assertEqual(self.resolve(head=head, push_base=before, floor=floor), before)

    def test_before_behind_the_floor_is_raised(self) -> None:
        before = self.repo.commit("c1")
        floor = self.repo.commit("c2")
        head = self.repo.commit("c3")
        self.assertEqual(self.resolve(head=head, push_base=before, floor=floor), floor)

    def test_all_zero_before_is_returned_unchanged(self) -> None:
        floor = self.repo.commit("c1")
        head = self.repo.commit("c2")
        # The new-branch guard downstream turns this into a tip-only check. The
        # floor must not quietly widen that into a range.
        self.assertEqual(self.resolve(head=head, push_base=ZERO, floor=floor), ZERO)

    def test_a_before_the_history_no_longer_contains_is_returned_unchanged(self) -> None:
        floor = self.repo.commit("c1")
        head = self.repo.commit("c2")
        gone = "d" * 40
        self.assertEqual(self.resolve(head=head, push_base=gone, floor=floor), gone)

    def test_an_empty_base_stays_empty(self) -> None:
        floor = self.repo.commit("c1")
        head = self.repo.commit("c2")
        self.assertEqual(self.resolve(head=head, floor=floor), "")

    def test_no_floor_leaves_the_base_alone(self) -> None:
        green = self.repo.commit("c1")
        head = self.repo.commit("c2")
        self.assertEqual(self.resolve(head=head, last_green=green, floor=""), green)

    def test_a_floor_absent_from_the_checkout_warns_and_changes_nothing(self) -> None:
        green = self.repo.commit("c1")
        head = self.repo.commit("c2")
        self.assertEqual(
            self.resolve(head=head, last_green=green, floor="e" * 40), green
        )
        self.assertEqual(len(self.warnings), 1)
        self.assertIn("not in this checkout", self.warnings[0])

    def test_a_floor_that_is_not_an_ancestor_of_head_changes_nothing(self) -> None:
        green = self.repo.commit("c1")
        head = self.repo.commit("c2")
        git(self.repo.path, "checkout", "--quiet", "-b", "side", green)
        sideways = self.repo.commit("s1")
        # `sideways` is newer by date and unreachable from `head`, which is the
        # case a date comparison gets wrong and ancestry gets right.
        self.assertEqual(
            self.resolve(head=head, last_green=green, floor=sideways), green
        )
        self.assertIn("not an ancestor", self.warnings[0])

    def test_a_floor_ahead_of_head_changes_nothing(self) -> None:
        green = self.repo.commit("c1")
        head = self.repo.commit("c2")
        ahead = self.repo.commit("c3")
        self.assertEqual(self.resolve(head=head, last_green=green, floor=ahead), green)
        self.assertIn("not an ancestor", self.warnings[0])

    def test_the_pull_request_lane_is_untouched_by_the_floor(self) -> None:
        pr_base = self.repo.commit("c1")
        floor = self.repo.commit("c2")
        head = self.repo.commit("c3")
        self.assertEqual(
            self.resolve(event="pull_request", head=head, pr_base=pr_base, floor=floor),
            pr_base,
        )
        self.assertEqual(self.warnings, [])


class CancelledRunLosslessTests(RepoCase):
    """#822/#863: a cancelled run's commits must still be covered by a later run.

    The sequence: C1's run is green, C2 is pushed and its run is CANCELLED, C3 is
    pushed. The last successful run is still C1, so the C3 run must walk from C1
    and its range must contain C2.
    """

    def setUp(self) -> None:
        super().setUp()
        self.c0 = self.repo.commit("c0")
        self.c1 = self.repo.commit("c1")
        self.c2 = self.repo.commit("c2")
        self.c3 = self.repo.commit("c3")

    def walked(self, base: str, head: str) -> list[str]:
        return git(self.repo.path, "rev-list", f"{base}..{head}").splitlines()

    def test_the_cancelled_commit_is_covered_with_a_floor_behind_the_base(self) -> None:
        base = self.resolve(head=self.c3, last_green=self.c1, push_base=self.c2, floor=self.c0)
        self.assertEqual(base, self.c1)
        self.assertIn(self.c2, self.walked(base, self.c3))

    def test_the_cancelled_commit_is_covered_with_the_floor_at_the_base(self) -> None:
        base = self.resolve(head=self.c3, last_green=self.c1, push_base=self.c2, floor=self.c1)
        self.assertEqual(base, self.c1)
        self.assertIn(self.c2, self.walked(base, self.c3))

    def test_positive_control_the_naive_before_base_loses_the_cancelled_commit(self) -> None:
        # Without this case the two above could pass on a resolver that returned
        # anything at all: this is the failure they are asserting the ABSENCE of,
        # made to happen on purpose.
        self.assertNotIn(self.c2, self.walked(self.c2, self.c3))

    def test_a_floor_advance_is_the_one_window_that_skips(self) -> None:
        # Stated rather than hidden. Advancing the floor past a cancelled run's
        # commits skips them, which is exactly the forgiveness a floor advance
        # asks for, and is why an advance is a reviewed commit that must name
        # what it forgives.
        base = self.resolve(head=self.c3, last_green=self.c1, push_base=self.c2, floor=self.c2)
        self.assertEqual(base, self.c2)
        self.assertNotIn(self.c2, self.walked(base, self.c3))


class CommandLineTests(RepoCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(MODULE), "--repo", str(self.repo.path), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_it_prints_the_raised_base(self) -> None:
        green = self.repo.commit("c1")
        floor = self.repo.commit("c2")
        head = self.repo.commit("c3")
        result = self.run_script(
            "--event", "push", "--head", head, "--last-green", green, "--floor", floor
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), floor)

    def test_an_unreadable_floor_record_exits_nonzero(self) -> None:
        head = self.repo.commit("c1")
        result = self.run_script(
            "--event", "push", "--head", head,
            "--floor-file", "/nonexistent/ci-enforcement-floor.txt",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("enforcement floor", result.stderr)

    def test_a_malformed_floor_argument_exits_nonzero(self) -> None:
        head = self.repo.commit("c1")
        result = self.run_script("--event", "push", "--head", head, "--floor", "nope")
        self.assertEqual(result.returncode, 2)


class WorkflowWiringTests(unittest.TestCase):
    """Every diff-scoped base in `ci.yml` goes through the script, or this reds.

    A resolver nothing calls resolves nothing. This is the reachability half:
    re-inlining the rule into the YAML, or dropping one call site, fails here.
    """

    def setUp(self) -> None:
        self.text = WORKFLOW.read_text(encoding="utf-8")

    def test_the_four_diff_scoped_steps_call_the_resolver(self) -> None:
        calls = re.findall(r"scripts/ci-walk-base\.py", self.text)
        self.assertEqual(
            len(calls),
            4,
            "the four diff-scoped steps each resolve their base through the script",
        )

    def test_no_step_still_chooses_its_own_base(self) -> None:
        inlined = re.findall(r'base="\$\{LAST_GREEN:-\}"', self.text)
        self.assertEqual(inlined, [], "a step is choosing its base inline again")

    def test_the_resolver_suite_runs_on_a_lane(self) -> None:
        self.assertIn("tests/scripts/test_ci_walk_base.py", self.text)


if __name__ == "__main__":
    unittest.main()
