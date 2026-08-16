#!/usr/bin/env python3
"""`scripts/agent-preflight.sh` never reports green over a block that did not run.

Row `GATE-PREFLIGHT-SKIP-REPORT`, issue #998, spec
`.agents/specs/gate-preflight-skip-report.md`.

The defect this suite pins: the trailer block was guarded by
`git merge-base --is-ancestor origin/main HEAD`, and when that guard was false
the block emitted NOTHING and the run still printed `All gates green.` and
exited 0. Two gates left the report without a word. It fired three times in one
session, twice because `origin/main` is a remote-tracking ref that every
worktree of one checkout shares, so another worktree's fetch moved it BETWEEN
the top of the run and the guard. On the third the `ok` count fell from 76 to 74
and the banner did not change.

That last number is the whole problem in one line. Nothing in the output
distinguished a run that checked the trailers from a run that did not, so the
only reader who could catch it was one who read the script.

## Why this suite executes the script instead of grepping it

Every assertion here could be spelled as a text match against the script, and
every one of them would then be satisfied by a REWRITE while missing an
OVERRIDE on a later line. `tests/scripts/test_agent_onboard.py` already learned
that lesson about `REQUIRE_ROLE` and switched to executing the script. The same
reasoning applies harder here, because what is under test is control flow across
four `git` calls, not the value of a variable.

## Why it can execute the script without recursing

`scripts/agent-preflight.sh` runs this very suite from its `SUITES` array, so a
nested full run would recurse without bound. `--role-only` exists for that
problem and is too narrow here, because the blocks under test are the last two
in the file.

So the run is pointed at a SCRATCH repository instead: the script is copied into
a temporary git repo and a stub `python3` that exits 0 is put on `PATH`. Every
`run` then reports `ok` without executing a checker, nothing re-enters this
suite, and the git-shaped control flow under test executes for real against refs
the test owns. `ROOT` in the script is derived from `BASH_SOURCE`, so the copy
makes the scratch repo the script's own tree.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"
SUITE_NAME = Path(__file__).stem

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Exits 0 for every checker and every suite. The script's `run()` reports `ok`.
INERT_PYTHON3 = "#!/bin/sh\nexit 0\n"

# Advances `refs/remotes/origin/main` on its FIRST call and then goes inert.
# This is what another worktree of the same checkout does when it fetches while
# a preflight is running, and it is occurrences 2 and 3 of #998 reproduced
# deterministically. `cd "$ROOT"` at the top of the script means the stub's cwd
# is the scratch repo, so the ref it moves is the one the script reads.
MOVING_PYTHON3 = """#!/bin/sh
if [ ! -e "$VLLM_TEST_MOVE_MARKER" ]; then
  : > "$VLLM_TEST_MOVE_MARKER"
  git update-ref refs/remotes/origin/main "$VLLM_TEST_MOVE_TO"
fi
exit 0
"""


class Report:
    """One preflight run, parsed into the three states it can report."""

    def __init__(self, returncode: int, text: str) -> None:
        self.returncode = returncode
        self.text = ANSI.sub("", text)
        lines = self.text.splitlines()
        self.ok = [line[7:] for line in lines if line.startswith("  ok   ")]
        self.fail = [line[7:] for line in lines if line.startswith("  FAIL ")]
        self.skip = [line[7:] for line in lines if line.startswith("  SKIP ")]
        self.headings = [line for line in lines if re.fullmatch(r"[A-Z][^\n]*:", line)]

    @property
    def green(self) -> bool:
        return "All gates green." in self.text

    def __str__(self) -> str:  # shown verbatim on any failure below
        return self.text


class PreflightHarness(unittest.TestCase):
    """A scratch repo with a known ref topology and an inert `python3`."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="preflight-skip-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "scripts").mkdir()
        (self.tmp / "bin").mkdir()
        shutil.copy2(PREFLIGHT, self.tmp / "scripts" / PREFLIGHT.name)
        self.script = self.tmp / "scripts" / PREFLIGHT.name
        self.set_python3(INERT_PYTHON3)

        self.git("init", "--quiet", ".")
        self.git("config", "user.email", "preflight@test.invalid")
        self.git("config", "user.name", "preflight test")
        self.git("config", "commit.gpgsign", "false")
        (self.tmp / "a").write_text("a\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "base")
        self.base = self.rev("HEAD")

        # A divergent commit that HEAD does NOT contain. This is what
        # `origin/main` looks like from a branch that is behind it.
        self.git("checkout", "--quiet", "-b", "divergent")
        (self.tmp / "b").write_text("b\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "divergent")
        self.divergent = self.rev("HEAD")

        self.git("checkout", "--quiet", "-B", "work", self.base)
        (self.tmp / "c").write_text("c\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "work")
        self.head = self.rev("HEAD")

        self.set_origin_main(self.base)

    # -- scratch repo helpers ------------------------------------------------

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.tmp, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def rev(self, revision: str) -> str:
        return self.git("rev-parse", revision)

    def set_origin_main(self, oid: str) -> None:
        self.git("update-ref", "refs/remotes/origin/main", oid)

    def unset_origin_main(self) -> None:
        self.git("update-ref", "-d", "refs/remotes/origin/main")

    def set_python3(self, body: str) -> None:
        stub = self.tmp / "bin" / "python3"
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)

    def preflight(self, *args: str, **env: str) -> Report:
        environment = dict(os.environ)
        environment["PATH"] = f"{self.tmp / 'bin'}{os.pathsep}{environment['PATH']}"
        environment.update(env)
        result = subprocess.run(
            ["bash", str(self.script), "--quiet", *args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        return Report(result.returncode, result.stdout + result.stderr)

    # -- preconditions -------------------------------------------------------

    def assert_ran_something(self, report: Report) -> None:
        """A run that reported nothing at all asserts nothing at all.

        The stub `python3` makes every gate cheap, and a broken harness makes
        every gate ABSENT. Both produce a short transcript with no `FAIL`, and
        only this check tells them apart. `assertEqual([], report.fail)` on its
        own is satisfied by a run that never started.
        """

        self.assertGreater(
            len(report.ok),
            40,
            f"harness precondition failed: only {len(report.ok)} gates reported "
            f"`ok`, so this run asserts nothing.\n{report}",
        )
        self.assertEqual([], report.fail, f"scratch run had failures:\n{report}")


class SkipIsReportedTests(PreflightHarness):
    def test_a_non_ancestor_base_reports_skip_and_no_green_banner(self) -> None:
        """RED before: prints `All gates green.` with the trailer block absent.

        Occurrence 1 of #998: a branch behind `main`. The two trailer gates were
        not run, nothing said so, and the banner claimed otherwise.
        """

        self.set_origin_main(self.divergent)
        report = self.preflight()
        self.assert_ran_something(report)

        self.assertTrue(
            any("commit-trailers" in line for line in report.skip),
            f"the trailer block did not report a SKIP:\n{report}",
        )
        self.assertTrue(
            any("commit-style" in line for line in report.skip),
            f"the style gate did not report a SKIP:\n{report}",
        )
        self.assertFalse(
            report.green,
            f"a run that skipped {len(report.skip)} gate(s) still printed the "
            f"green banner:\n{report}",
        )
        self.assertIn(
            "SKIPPED",
            report.text,
            f"the summary does not count the skipped gates:\n{report}",
        )
        # The reason has to travel with the skip, or the reader learns only that
        # something did not happen.
        self.assertIn(
            "ancestor",
            report.text,
            f"no reason printed beside the SKIP:\n{report}",
        )

    def test_the_ok_count_falls_only_with_a_reported_skip(self) -> None:
        """RED before: the count falls by two and there are zero SKIP lines.

        This is occurrence 3 exactly, and it is the case that makes a SILENT
        drop impossible rather than merely unlikely. Asserting the presence of a
        SKIP line elsewhere does not cover it: a future block could vanish the
        same way and every other case here would stay green.
        """

        self.set_origin_main(self.base)
        ancestor = self.preflight()
        self.assert_ran_something(ancestor)
        self.assertEqual([], ancestor.skip, f"the control run skipped:\n{ancestor}")

        self.set_origin_main(self.divergent)
        behind = self.preflight()
        self.assert_ran_something(behind)

        missing = len(ancestor.ok) - len(behind.ok)
        self.assertGreater(
            missing,
            0,
            "harness precondition failed: the two runs report the same number "
            f"of gates, so there is no drop to account for.\n{behind}",
        )
        self.assertEqual(
            missing,
            len(behind.skip),
            f"{missing} gate(s) disappeared from the report and "
            f"{len(behind.skip)} were reported as skipped. Every gate that "
            f"stops running has to say so.\n{behind}",
        )

    def test_a_base_that_moves_mid_run_does_not_change_the_verdict(self) -> None:
        """RED before: the trailer block vanishes and the run still says green.

        Occurrences 2 and 3. `origin/main` is shared by every worktree of one
        checkout, so the guard was true when the run started and false when it
        was evaluated. Pinning the base at the start is what fixes it, and the
        run has to name the SHA it pinned.
        """

        marker = self.tmp / "moved"
        self.set_python3(MOVING_PYTHON3)
        self.set_origin_main(self.base)
        report = self.preflight(
            VLLM_TEST_MOVE_MARKER=str(marker),
            VLLM_TEST_MOVE_TO=self.divergent,
        )
        self.assert_ran_something(report)

        # PRECONDITION, asserted and never skipped: if the stub never moved the
        # ref then this test reproduces nothing and must go red.
        self.assertTrue(
            marker.exists(),
            "precondition failed: the stub python3 never ran, so origin/main "
            f"never moved and this test asserts nothing.\n{report}",
        )
        self.assertEqual(
            self.divergent,
            self.rev("refs/remotes/origin/main"),
            f"precondition failed: origin/main did not move.\n{report}",
        )

        self.assertIn(
            self.base,
            report.text,
            "the run does not name the base SHA it gated against, so a reader "
            f"cannot tell which revision it compared to.\n{report}",
        )
        self.assertEqual(
            [],
            report.skip,
            "the base was pinned to an ancestor at the start of the run, so no "
            f"block should have been skipped when the ref moved:\n{report}",
        )
        self.assertTrue(
            any("commit-trailers" in line for line in report.ok),
            f"the trailer gate did not run against the pinned base:\n{report}",
        )
        self.assertTrue(report.green, f"the run earned the banner and withheld it:\n{report}")

    def test_an_unresolvable_base_reports_skip_for_both_range_blocks(self) -> None:
        """RED before: five gates vanish and the run says green.

        The same shape in the `Committed range` block. A checkout whose remote
        is not named `origin`, or one that has never fetched, resolves no
        `origin/main`. `AGENTS.md`: "Unknown is not absence or success."
        """

        self.unset_origin_main()
        report = self.preflight()
        self.assert_ran_something(report)

        for label in (
            "now-current range",
            "doc-checkpoint range",
            "issue-index append-only",
            "commit-trailers",
            "commit-style",
        ):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.skip),
                    f"{label} vanished without reporting a SKIP:\n{report}",
                )
        self.assertFalse(report.green, f"five skipped gates still printed green:\n{report}")


class TheBannerStaysReachableTests(PreflightHarness):
    def test_an_ancestor_base_still_earns_the_banner(self) -> None:
        """The control. A fix that suppresses the banner always cannot pass."""

        self.set_origin_main(self.base)
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"an ordinary run skipped a gate:\n{report}")
        self.assertTrue(report.green, f"an ordinary run lost the banner:\n{report}")
        self.assertEqual(0, report.returncode, f"an ordinary run exited non-zero:\n{report}")
        for label in ("commit-trailers", "commit-style"):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.ok),
                    f"{label} did not run on an ancestor base:\n{report}",
                )

    def test_an_empty_range_is_not_a_skip(self) -> None:
        """An empty input set is not an unexamined one.

        `HEAD` level with the base means the range gates have zero commits to
        read, and a verdict over zero commits is not withheld information. If
        this reported `SKIP`, the ordinary session-start run of a freshly cut
        branch would print a false alarm and lose the banner, and
        `AGENTS.md` is explicit that a gate firing on ordinary work is the
        defect. Without this case the obvious over-correction passes every
        other test in this file.
        """

        self.set_origin_main(self.head)
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"an empty range reported a SKIP:\n{report}")
        self.assertTrue(report.green, f"an empty range lost the banner:\n{report}")
        self.assertEqual(0, report.returncode, f"an empty range exited non-zero:\n{report}")

    def test_a_skipped_run_exits_zero_and_a_failing_run_does_not(self) -> None:
        """SKIP and FAIL are different facts and keep different exit codes.

        A branch behind `main` is ordinary work, so a skip does not fail the
        run. Widening exit 1 to mean "a gate did not run" would merge the two
        facts into one signal, which is the conflation this row removes.
        """

        self.set_origin_main(self.divergent)
        skipped = self.preflight()
        self.assert_ran_something(skipped)
        self.assertNotEqual([], skipped.skip, f"nothing was skipped:\n{skipped}")
        self.assertEqual(
            0,
            skipped.returncode,
            f"a skipped block failed the run:\n{skipped}",
        )

        # And a real failure still exits 1, so exit 0 above is a decision rather
        # than a script that cannot fail.
        self.set_python3("#!/bin/sh\nexit 1\n")
        failing = self.preflight()
        self.assertNotEqual(
            0,
            failing.returncode,
            f"every gate failed and the run exited 0:\n{failing}",
        )
        self.assertFalse(failing.green, f"a failing run printed the banner:\n{failing}")


class RegistrationTests(unittest.TestCase):
    def test_the_suite_runs_in_preflight_and_in_ci(self) -> None:
        """A suite wired into neither surface runs on no machine.

        `test_main_baseline.py` shipped in neither and its 24 tests ran nowhere,
        so both surfaces are asserted here and removing either is red.
        """

        preflight = PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn(SUITE_NAME, preflight, "not in the preflight SUITES array")
        self.assertIn(
            f"python3 tests/scripts/{SUITE_NAME}.py",
            CI.read_text(encoding="utf-8"),
            "CI does not run this suite",
        )


if __name__ == "__main__":
    unittest.main()
