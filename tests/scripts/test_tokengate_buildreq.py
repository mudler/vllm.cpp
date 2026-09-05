#!/usr/bin/env python3
"""The TOKENGATE job must install the PIN'S OWN build requirements.

`.agents/scripts/tokengate-e126687-job.sh` builds vLLM at
`e126687a9a828d513c01a07cd69f025f27d63280` with

    pip wheel --no-deps --no-build-isolation -w "$WORK/dist" .

and `--no-build-isolation` makes pip SKIP `pyproject.toml`'s `[build-system]
requires` ENTIRELY. Every build dependency therefore has to be in the venv
before that line runs, and nothing but this script puts it there.

WHAT THIS SUITE IS FOR. Job `efc30c74-005e-4e80-bc28-bd34f5b76b77` on `dgx:gpu0`
reached the build stage and died in 3 seconds with

    File "<string>", line 21, in <module>
    ModuleNotFoundError: No module named 'setuptools_rust'

`setup.py:21` at the pin is `from setuptools_rust.build import build_rust`. The
script's pre-install was a HAND-KEPT list -- `pip wheel setuptools setuptools_scm
ninja cmake numpy` -- and the pin's `requires` had moved past it: `packaging`,
`setuptools-rust` and `jinja2` were all absent. A lease, a queue position and a
cold ccache bought a three-second traceback.

WHAT THIS SUITE COMPARES AGAINST WHAT, because a test that does not narrate its
own wiring cannot be audited.

1. Structurally, over the committed script: the build stage installs
   `requirements/build/cuda.txt` FROM THE CLONED TREE, and it does so before the
   `--no-build-isolation` build. Sourcing the list from the tree is the fix; a
   list retyped into the harness is the defect, and it drifts silently.
2. Behaviourally, over the assertion the script embeds: the suite extracts that
   embedded checker and runs it against scratch trees it writes itself. The
   checker PARSES `[build-system] requires` out of the `pyproject.toml` it is
   handed -- it never carries a list of its own -- so the cases here are a
   distribution that is installed, one that is not, spellings that PEP 503
   normalises to the same name, and a tree whose `requires` cannot be read.
   The absent case is the mutation that matters: a checker that cannot go red
   over a missing build requirement measures nothing.
3. Three things the repair MUST NOT have moved: the ten staged-input
   `assert_sha` calls, the compute-capability guard that refuses any device but
   the GB10 the committed golden was captured on, and the exit-code map in which
   DRIFT is 7 and every instrument failure is something else. A drift that
   shares a code with a broken instrument is the one outcome this job exists to
   tell apart.
4. The four SHELL lines that turn the checker's red into a refusal. Case 2 above
   executes the checker and pins what it RETURNS; nothing there reaches the
   `BUILDREQ_RC=$?` / `-ne 0` / `exit 3` wiring that stops the job, and a
   checker whose red the build ignores is a log line, not a gate.

Nothing here needs a GPU, a lease, a network or a vLLM checkout: `bash`, the
standard library, and scratch directories, in well under a second.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".agents/scripts/tokengate-e126687-job.sh"

ASSERT_BEGIN = "# --- BUILD-REQUIRES ASSERTION begin ---"
ASSERT_END = "# --- BUILD-REQUIRES ASSERTION end ---"

# The mirror `pyproject.toml` names at the pin. The script installs THIS path
# out of the cloned tree; the suite only checks that it is the path installed,
# never what the file contains, because the file's contents belong to the pin.
BUILDREQ_RELPATH = "requirements/build/cuda.txt"

# The eight committed goldens plus the two staged scripts: ten `assert_sha`
# calls. Names, not hashes -- a re-staged golden legitimately changes a hash and
# must not have to change this file too, while a DELETED assertion is the defect
# this pins.
STAGED_ASSERTIONS = (
    "opt-oracle-capture.py",
    "tokengate-e126687-diff.py",
    "goldens-committed/greedy_ids.npy",
    "goldens-committed/greedy_dist.npy",
    "goldens-committed/p0_prompt.i32",
    "goldens-committed/p1_prompt.i32",
    "goldens-committed/p2_prompt.i32",
    "goldens-committed/p3_prompt.i32",
    "goldens-committed/p4_prompt.i32",
    "goldens-committed/p5_prompt.i32",
)


def script_lines() -> list[str]:
    return SCRIPT.read_text(encoding="utf-8").splitlines()


def first_index(lines: list[str], needle: str) -> int:
    for i, line in enumerate(lines):
        if needle in line:
            return i
    return -1


def first_match(lines: list[str], pattern: str) -> int:
    """Index of the first line matching `pattern`, or -1.

    Matched rather than searched for as a substring, because the script also
    NARRATES this path in a comment and a comment installs nothing.
    """
    rx = re.compile(pattern)
    for i, line in enumerate(lines):
        if rx.search(line) and not line.lstrip().startswith("#"):
            return i
    return -1


def embedded_checker() -> str:
    """Return the python program the script runs to assert its build requires.

    Extracted from between the sentinels so that the suite exercises the SAME
    bytes the leased worker runs, rather than a copy that can disagree with it.
    """
    lines = script_lines()
    begin = first_index(lines, ASSERT_BEGIN)
    end = first_index(lines, ASSERT_END)
    if begin < 0 or end < 0 or end < begin:
        raise AssertionError(
            "the build-requires assertion block is not delimited in "
            f"{SCRIPT}: begin={begin} end={end}"
        )
    block = lines[begin : end + 1]
    open_at = first_index(block, "<<'PY'")
    if open_at < 0:
        raise AssertionError("no quoted heredoc opens inside the assertion block")
    body: list[str] = []
    for line in block[open_at + 1 :]:
        if line == "PY":
            return "\n".join(body) + "\n"
        body.append(line)
    raise AssertionError("the assertion heredoc is never closed by a bare `PY`")


def an_installed_distribution() -> str:
    """A distribution name this interpreter can certainly resolve.

    Chosen from the live environment rather than named, so the positive cases
    do not depend on any particular lane having pip or setuptools installed.
    """
    names = sorted(
        {
            name
            for dist in md.distributions()
            if (name := dist.metadata["Name"])
        }
    )
    if not names:
        raise AssertionError(
            "this interpreter reports no installed distributions at all, so the "
            "positive cases cannot be posed"
        )
    # Prefer a name that PEP 503 normalisation actually changes, so the
    # spelling case below exercises the rule rather than restating it.
    dashed = [name for name in names if "-" in name]
    return dashed[0] if dashed else names[0]


def run_checker(requires: object) -> subprocess.CompletedProcess:
    """Run the extracted checker over a scratch tree with these `requires`.

    `requires is None` writes a `pyproject.toml` with no `[build-system]` table,
    which is the malformed-input case.
    """
    program = embedded_checker()
    with tempfile.TemporaryDirectory() as scratch:
        pyproject = Path(scratch) / "pyproject.toml"
        text = '[project]\nname = "fixture"\n'
        if requires is not None:
            text += "\n[build-system]\nrequires = %s\nbuild-backend = \"setuptools.build_meta\"\n" % json.dumps(
                list(requires)
            )
        pyproject.write_text(text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-", str(pyproject)],
            input=program,
            capture_output=True,
            text=True,
            timeout=120,
        )


class BuildRequiresAreInstalledFromTheTree(unittest.TestCase):
    """`--no-build-isolation` installs nothing, so this script has to."""

    def assignment_index(self) -> int:
        return first_match(
            script_lines(),
            r'^\s*BUILDREQ="\$WORK/vllm/' + re.escape(BUILDREQ_RELPATH) + r'"\s*$',
        )

    def install_index(self) -> int:
        return first_match(script_lines(), r'pip"\s+install\b[^\n]*-r\s+"\$BUILDREQ"')

    def test_the_build_stage_installs_the_trees_build_requirements(self) -> None:
        self.assertGreaterEqual(
            self.install_index(),
            0,
            "nothing in the build stage installs the pin's build requirements, "
            "so `--no-build-isolation` runs against whatever a hand-kept list "
            "happened to leave in the venv -- the efc30c74 failure exactly",
        )

    def test_the_requirements_come_from_the_cloned_tree_not_a_retyped_list(
        self,
    ) -> None:
        self.assertGreaterEqual(
            self.assignment_index(),
            0,
            "the build requirements must be read out of the CLONED tree at "
            f"$TARGET_SHA ($WORK/vllm/{BUILDREQ_RELPATH}); a list retyped into "
            "this harness drifts from the pin without saying so",
        )

    def test_a_missing_requirements_file_refuses_instead_of_building(self) -> None:
        lines = script_lines()
        guard = first_match(lines, r'if \[ ! -f "\$BUILDREQ" \]; then')
        self.assertGreaterEqual(
            guard,
            0,
            "an absent requirements file must refuse; pip install -r on a "
            "missing file is the one shape that could pass silently",
        )
        self.assertLess(guard, self.install_index())
        # ...and it has to REFUSE, not merely branch. Existence and order were
        # pinned here from the start; the exit was not, and `exit 3 -> exit 0`
        # on this branch survived the suite green.
        closed = -1
        for i in range(guard + 1, len(lines)):
            if re.match(r"^\s*fi\s*$", lines[i]):
                closed = i
                break
        self.assertGreater(closed, guard, "the missing-file guard's `if` never closes")
        body = "\n".join(lines[guard + 1 : closed])
        self.assertRegex(
            body,
            re.compile(r"^\s*exit 3\s*$", re.MULTILINE),
            "the missing-file branch no longer exits 3, so a clone without "
            "requirements/build/cuda.txt falls through into the "
            "--no-build-isolation build with the venv it happened to have",
        )

    def test_the_install_precedes_the_no_build_isolation_build(self) -> None:
        lines = script_lines()
        build_at = first_match(lines, r"pip\"\s+wheel\b[^\n]*--no-build-isolation")
        self.assertGreaterEqual(build_at, 0, "the wheel build vanished")
        install_at = self.install_index()
        self.assertGreaterEqual(install_at, 0, "no build-requirements install")
        self.assertLess(
            install_at,
            build_at,
            "the build requirements are installed AFTER the build that needs "
            "them, which is the same failure with a longer log",
        )
        self.assertLess(self.assignment_index(), install_at)


class TheEmbeddedAssertionDetectsAMissingRequirement(unittest.TestCase):
    def test_an_absent_distribution_is_named_and_reds(self) -> None:
        present = an_installed_distribution()
        absent = "vllmcpp-tokengate-fixture-absent-dist"
        proc = run_checker([present, absent + ">=1.9.0"])
        self.assertEqual(
            proc.returncode,
            1,
            "the checker accepted a tree whose build requires it cannot "
            f"satisfy; stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )
        self.assertIn(absent, proc.stdout, "the red does not name what is missing")

    def test_a_satisfied_tree_is_accepted(self) -> None:
        present = an_installed_distribution()
        proc = run_checker([present])
        self.assertEqual(
            proc.returncode,
            0,
            "the checker refuses a tree whose build requires ARE satisfied, so "
            f"its red says nothing; stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )
        self.assertIn(present, proc.stdout)

    def test_pep503_spellings_of_one_distribution_are_one_distribution(
        self,
    ) -> None:
        present = an_installed_distribution()
        variant = present.upper().replace("-", "_")
        proc = run_checker([variant + " >= 0.1 ; python_version >= '3.8'"])
        self.assertEqual(
            proc.returncode,
            0,
            "`setuptools_rust` and `setuptools-rust` are one distribution and "
            "pyproject and the requirements file do not agree on the spelling; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )

    def test_an_extra_or_a_version_is_not_read_as_part_of_the_name(self) -> None:
        present = an_installed_distribution()
        proc = run_checker([present + "[cu13]==0.1.12"])
        self.assertEqual(
            proc.returncode,
            0,
            "an extras marker was read into the distribution name, which turns "
            f"every pinned requirement into a false red; stdout={proc.stdout!r}",
        )

    def test_a_tree_with_no_build_system_requires_is_not_a_pass(self) -> None:
        proc = run_checker(None)
        self.assertNotIn(
            proc.returncode,
            (0, 1),
            "a pyproject the checker could not read must be an INSTRUMENT "
            "result, never a green and never a finding; "
            f"rc={proc.returncode} stdout={proc.stdout!r}",
        )


class TheRepairMovedNothingItMustNotMove(unittest.TestCase):
    def test_all_ten_staged_input_assertions_survive(self) -> None:
        lines = script_lines()
        calls = [ln.strip() for ln in lines if ln.strip().startswith("assert_sha ")]
        self.assertEqual(
            len(calls),
            len(STAGED_ASSERTIONS),
            "the staged-input integrity block changed size; "
            "`goldens-committed` IS the bar this job measures against",
        )
        for name in STAGED_ASSERTIONS:
            hit = [c for c in calls if name in c]
            self.assertEqual(
                len(hit), 1, f"exactly one assert_sha must cover {name}; got {hit}"
            )
            self.assertRegex(
                hit[0],
                r"\b[0-9a-f]{64}\b",
                f"the assertion for {name} carries no sha256 to compare against",
            )

    def test_the_compute_capability_guard_survives(self) -> None:
        """The device assertion is the reason this job is pinned to `dgx:gpu0`.

        The committed golden was captured on GB10. A candidate capture on any
        other silicon moves revision AND device at once and cannot attribute a
        difference, which is the whole argument for holding the box fixed.

        BOTH halves are pinned, because reading the capability and refusing on
        it are separate lines: delete the read alone and the guard compares an
        unset `CAP`; delete the guard alone and the read becomes a log line.
        """
        lines = script_lines()
        read_at = first_match(
            lines, r'^\s*CAP="\$\(\s*nvidia-smi\b.*\bcompute_cap\b.*\)"\s*$'
        )
        self.assertGreaterEqual(
            read_at,
            0,
            "nothing reads the device's compute capability out of nvidia-smi "
            "any more, so the guard below it compares an unset CAP and the job "
            "captures on whatever silicon the lease handed it",
        )
        guard_at = first_match(lines, r'^\s*if \[ "\$CAP" != "12\.1" \]; then\s*$')
        self.assertGreaterEqual(
            guard_at,
            0,
            "the compute-capability guard is gone; a capture on anything but "
            "GB10 (12.1) is not the measurement this job was asked for, and "
            "without the refusal it would carry the right label anyway",
        )
        self.assertLess(
            read_at, guard_at, "CAP is compared before anything reads it"
        )
        closed_at = -1
        for i in range(guard_at + 1, len(lines)):
            if re.match(r"^\s*fi\s*$", lines[i]):
                closed_at = i
                break
        self.assertGreater(closed_at, guard_at, "the device guard's `if` never closes")
        body = "\n".join(lines[guard_at + 1 : closed_at])
        self.assertRegex(
            body,
            re.compile(r"^\s*exit 2\s*$", re.MULTILINE),
            "the wrong-device branch no longer exits 2, the map's `wrong "
            "device: nothing was measured`; a guard that prints and continues "
            "produces the number it exists to prevent",
        )
        stage_at = first_match(lines, r'^\s*if \[ "\$STAGE" = "build" \]')
        self.assertGreaterEqual(stage_at, 0, "the build stage vanished")
        self.assertLess(
            closed_at,
            stage_at,
            "the device is checked after the build stage has started, so the "
            "refusal no longer means that nothing was measured",
        )

    def test_drift_keeps_an_exit_code_no_instrument_failure_uses(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        tail = text[text.rindex('case "${DIFF_RC:-0}" in') :]
        for want, why in (
            (r"^\s*0\)\s*exit 0 ;;$", "PASS is no longer exit 0"),
            (r"^\s*1\)\s*exit 7 ;;$", "DRIFT is no longer exit 7"),
            (r"^\s*\*\)\s*exit 8 ;;$", "a differ that could not compare is no longer exit 8"),
        ):
            self.assertRegex(tail, re.compile(want, re.MULTILINE), why)
        # Every `exit N` the script can take on an instrument or environment
        # failure. 7 may appear in none of them.
        instrument = {
            int(m)
            for m in re.findall(r"\bexit (\d+)\b", text[: text.rindex('case "${DIFF_RC:-0}" in')])
        }
        self.assertNotIn(
            7,
            instrument,
            "an instrument failure now exits 7, which is DRIFT's code; a "
            "finding and a broken harness would be indistinguishable to `rc`",
        )
        self.assertTrue(
            instrument,
            "no refusal path exits non-zero any more",
        )


class TheShellRefusesWhenTheEmbeddedAssertionReds(unittest.TestCase):
    """A checker that goes red and a script that builds anyway is not a gate.

    `TheEmbeddedAssertionDetectsAMissingRequirement` runs the checker and pins
    what it RETURNS. Nothing there reaches the shell that reads that return
    value, and the two layers fail independently: the checker can name every
    absent distribution correctly while `$?` is never read, the branch is never
    taken, or the refusal exits 0 into the build it was supposed to stop. Those
    lines are therefore pinned here, statically, over the committed script.
    """

    def assertion_block(self) -> list[str]:
        lines = script_lines()
        begin = first_index(lines, ASSERT_BEGIN)
        end = first_index(lines, ASSERT_END)
        self.assertGreaterEqual(begin, 0, "the assertion block's begin sentinel is gone")
        self.assertGreater(end, begin, "the assertion block's end sentinel is gone")
        return lines[begin : end + 1]

    def test_the_checker_is_handed_the_cloned_tree_not_a_snapshot(self) -> None:
        at = first_match(
            self.assertion_block(),
            r"""^\s*"\$WORK/venv/bin/python" - "\$WORK/vllm/pyproject\.toml" <<'PY'\s*$""",
        )
        self.assertGreaterEqual(
            at,
            0,
            "the checker is no longer handed $WORK/vllm/pyproject.toml -- the "
            "tree `git checkout $TARGET_SHA` produced -- read by the BUILD "
            "venv's own interpreter. Reading the requires from a snapshot "
            "instead of the clone is the drift this repair exists to remove, "
            "and resolving them in any interpreter but the one the build runs "
            "in answers a question nobody asked",
        )

    def test_the_status_the_refusal_tests_is_the_checkers_own(self) -> None:
        block = self.assertion_block()
        close_at = -1
        for i, line in enumerate(block):
            if line == "PY":
                close_at = i
                break
        else:
            self.fail("the checker heredoc is never closed by a bare `PY`")
        after = [ln for ln in block[close_at + 1 :] if ln.strip()]
        self.assertTrue(after, "nothing at all follows the checker")
        self.assertRegex(
            after[0].strip(),
            r"^BUILDREQ_RC=\$\?$",
            "the first statement after the checker is not `BUILDREQ_RC=$?`, so "
            "the value the refusal tests is not the checker's status: `$?` "
            "reads whatever ran last, and a literal makes the refusal below "
            "unreachable while leaving it in the file to read as a guard",
        )

    def test_a_red_checker_stops_the_job_with_the_prerequisites_code(self) -> None:
        block = self.assertion_block()
        guard_at = first_match(block, r'^\s*if \[ "\$BUILDREQ_RC" -ne 0 \]; then\s*$')
        self.assertGreaterEqual(
            guard_at,
            0,
            "nothing branches on BUILDREQ_RC, so a checker that has just named "
            "three absent build requirements is followed by the "
            "--no-build-isolation build regardless: the efc30c74 traceback, "
            "with a warning printed above it",
        )
        closed_at = -1
        for i in range(guard_at + 1, len(block)):
            if re.match(r"^\s*fi\s*$", block[i]):
                closed_at = i
                break
        self.assertGreater(closed_at, guard_at, "the refusal's `if` never closes")
        body = "\n".join(block[guard_at + 1 : closed_at])
        self.assertRegex(
            body,
            re.compile(r'^\s*sum "BUILDREQ_RC=1', re.MULTILINE),
            "the refusal no longer prints BUILDREQ_RC=1, which is the field a "
            "reader of the job log greps the SUM lines for",
        )
        self.assertRegex(
            body,
            re.compile(r"^\s*exit 3\s*$", re.MULTILINE),
            "the refusal no longer exits 3, the map's `staging or build "
            "prerequisites missing`; exit 0 walks into the build it was "
            "supposed to stop, and any other code reports the wrong class",
        )

    def test_the_whole_assertion_precedes_the_build_it_guards(self) -> None:
        self.assertion_block()  # both sentinels exist, so the index below is real
        lines = script_lines()
        build_at = first_match(lines, r"pip\"\s+wheel\b[^\n]*--no-build-isolation")
        self.assertGreaterEqual(build_at, 0, "the wheel build vanished")
        end_at = first_index(lines, ASSERT_END)
        self.assertGreaterEqual(end_at, 0, "the assertion block's end sentinel is gone")
        self.assertLess(
            end_at,
            build_at,
            "the build requirements are asserted AFTER the build that needs "
            "them, which is the same failure with a longer log",
        )


if __name__ == "__main__":
    unittest.main()
