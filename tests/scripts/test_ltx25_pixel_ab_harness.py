#!/usr/bin/env python3
"""The pixel A/B harness's own preconditions, exercised without a GPU.

`.agents/specs/ltx25-dit-attn-flash.md` section 10, #1612.

`scripts/ltx25-dit-attn-flash-pixel-ab.sh` spends a four-hour lease on
`dgx:gpu0`. Nothing in this repository could run it, so every guard it carries
was a guard nobody had ever seen fire -- and two of them did not:

  the memory precondition   phase [0] PRINTED `available 5` at +0s on
                            2026-08-22 and built anyway, into a lost worker at
                            +728s with no binary cached and nothing measured
                            (rc job 5fb9399f-4f4e-417c-adbd-4d741a2e18e4).
  the resume                three lost workers, and each one threw away every
                            arm that had already rendered, including a ~2 h
                            naive arm.

The shell functions those two rest on are extracted VERBATIM from the harness,
between its `# BEGIN pixab-helpers` and `# END pixab-helpers` markers, and run
here against a fabricated `/proc/meminfo` and fabricated arm directories. The
extraction is itself asserted, because a suite that silently found no block
would report success over nothing.

WHAT THIS CANNOT DO. It does not run the harness. Its wiring assertions at the
bottom read the file as TEXT, and a text assertion is a tripwire rather than a
proof: it catches the exact inversion that shipped and it would not catch a
rewrite that reintroduced the same defect in different words. The lease is the
only place the rest of that file executes, and that is stated here rather than
papered over.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Overridable ONLY so that the red-before run of this suite can be taken against
# an earlier revision of the harness. It defaults to the committed file.
HARNESS = Path(os.environ.get("PIXAB_HARNESS", ROOT / "scripts/ltx25-dit-attn-flash-pixel-ab.sh"))

BEGIN = "# BEGIN pixab-helpers"
END = "# END pixab-helpers"

MEM_GATE_REFUSED = 39
UNIT_GATE_FAILED = 44
UNIT_GATE_ABSENT = 45
ROUTING_BAD = 46
RESUMED_LIBRARY_DIFFERS = 47


def helper_block() -> str:
    text = HARNESS.read_text()
    if BEGIN not in text or END not in text:
        raise AssertionError(
            f"{HARNESS} carries no {BEGIN!r}/{END!r} block: there is nothing to exercise")
    return text.split(BEGIN, 1)[1].split(END, 1)[0]


def bash(snippet: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run the extracted helpers plus `snippet`, with the two externals they use."""
    prelude = 'set -u\nT0=$(date +%s)\nsay() { echo "[say]$*"; }\n'
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(["bash", "-c", prelude + helper_block() + "\n" + snippet],
                          capture_output=True, text=True, env=e)


def meminfo(path: Path, gib: float | None) -> Path:
    """A `/proc/meminfo` with, or deliberately without, a MemAvailable line."""
    lines = ["MemTotal:       124680000 kB\n"]
    if gib is not None:
        lines.append("MemAvailable:   %d kB\n" % int(gib * 1048576))
    lines.append("SwapTotal:              0 kB\n")
    path.write_text("".join(lines))
    return path


class HelperBlock(unittest.TestCase):
    def test_the_block_exists_and_holds_every_function_this_suite_runs(self) -> None:
        block = helper_block()
        for fn in ("mem_avail_gib()", "wait_for_memory()", "arm_is_complete()",
                   "routing_verdict()", "resumed_arm_library()"):
            self.assertIn(fn, block, f"{fn} left the extracted block: this suite would "
                                     f"then exercise nothing while still passing")


class MemAvailableReader(unittest.TestCase):
    """ONE reader, and phase [0b] and the render watchdog both call it.

    They read `/proc/meminfo` rather than `free`'s `available` column, so the
    start gate and the watchdog cannot disagree about what they measured.
    """

    def test_it_reads_memavailable_in_gib(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            mi = meminfo(Path(t) / "meminfo", 61.5)
            p = bash("mem_avail_gib", {"MEMINFO": str(mi)})
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "61.5")

    def test_it_is_empty_rather_than_zero_when_the_field_is_absent(self) -> None:
        """Empty and zero are different facts. Zero would trip a floor check as
        if the box were full; empty says the instrument did not read."""
        with tempfile.TemporaryDirectory() as t:
            mi = meminfo(Path(t) / "meminfo", None)
            p = bash("v=$(mem_avail_gib); echo \"[${v}]\"", {"MEMINFO": str(mi)})
        self.assertEqual(p.stdout.strip(), "[]")

    def test_it_is_empty_when_the_file_does_not_exist(self) -> None:
        p = bash("v=$(mem_avail_gib); echo \"[${v}]\"", {"MEMINFO": "/nonexistent/meminfo"})
        self.assertEqual(p.stdout.strip(), "[]")


class MemoryPrecondition(unittest.TestCase):
    """Phase [0b]: wait, then refuse, and never proceed on a full box.

    The floor is 60 GiB because the recorded 20260820 render at this geometry
    peaked at 79.503 GiB with a MemAvailable low-water of 40.13 GiB. A lease
    spent waiting and refusing costs a lease; a lease spent building into an
    out-of-memory kill costs the lease and leaves a record that cannot say why.
    """

    def test_it_proceeds_immediately_above_the_floor(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            mi = meminfo(Path(t) / "meminfo", 80.0)
            p = bash("wait_for_memory 60.0 1200 30; echo rc=$?", {"MEMINFO": str(mi)})
        self.assertIn("rc=0", p.stdout)
        self.assertIn("80.0 GiB >= floor 60.0 GiB", p.stdout)
        self.assertIn("after 0s", p.stdout)

    def test_it_refuses_below_the_floor_with_its_own_status(self) -> None:
        """The 2026-08-22 case, in numbers: 5 GiB available at +0s."""
        with tempfile.TemporaryDirectory() as t:
            mi = meminfo(Path(t) / "meminfo", 5.0)
            p = bash("wait_for_memory 60.0 0 1; echo rc=$?", {"MEMINFO": str(mi)})
        self.assertIn(f"rc={MEM_GATE_REFUSED}", p.stdout)
        out = p.stdout + p.stderr
        self.assertIn("5.0 GiB", out, "the refusal must name what it saw")
        self.assertIn("60.0 GiB start floor", out, "and the floor it wanted")
        self.assertIn("after 0s of waiting", out, "and how long it waited")
        self.assertIn("nothing was rendered", out)

    def test_it_waits_and_then_proceeds_when_the_box_recovers(self) -> None:
        """The reason this waits rather than failing at once: the previous
        tenant's memory is often still being reclaimed when the lease starts."""
        with tempfile.TemporaryDirectory() as t:
            mi = Path(t) / "meminfo"
            meminfo(mi, 5.0)

            def recover() -> None:
                time.sleep(1.5)
                meminfo(mi, 90.0)

            th = threading.Thread(target=recover)
            th.start()
            p = bash("wait_for_memory 60.0 20 1; echo rc=$?", {"MEMINFO": str(mi)})
            th.join()
        self.assertIn("rc=0", p.stdout)
        # It logged the low readings on the way, so a reader can tell a
        # recovering box from a flat one.
        self.assertIn("5.0 GiB < 60.0 GiB", p.stdout)
        self.assertIn("90.0 GiB >= floor 60.0 GiB", p.stdout)

    def test_an_unreadable_meminfo_is_a_refusal_and_never_a_pass(self) -> None:
        p = bash("wait_for_memory 60.0 0 1; echo rc=$?", {"MEMINFO": "/nonexistent/meminfo"})
        self.assertIn(f"rc={MEM_GATE_REFUSED}", p.stdout)
        self.assertIn("cannot read MemAvailable", p.stdout + p.stderr)


class ArmCompleteness(unittest.TestCase):
    """Phase [G]'s resume: an arm is reused only when it is COMPLETE.

    A partial arm is re-rendered from scratch rather than resumed mid-flight.
    """

    def _arm(self, root: Path, frames: int, audio: bool, audio_size: int = 64) -> Path:
        d = root / "arm"
        d.mkdir()
        for i in range(frames):
            (d / f"frame_{i:06d}.ppm").write_bytes(b"P6\n1 1\n255\n\0\0\0")
        if audio:
            (d / "audio.wav").write_bytes(b"\0" * audio_size)
        return d

    def _check(self, d: Path, want: int = 49) -> int:
        return bash(f'arm_is_complete "{d}" {want}; echo rc=$?').stdout.strip().split("=")[-1]

    def test_a_complete_arm_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            d = self._arm(Path(t), 49, True)
            self.assertEqual(self._check(d), "0")

    def test_a_short_arm_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            d = self._arm(Path(t), 48, True)
            self.assertEqual(self._check(d), "1")

    def test_a_long_arm_is_not_either(self) -> None:
        """More frames than asked for is a directory from another geometry, not
        a completed render of this one."""
        with tempfile.TemporaryDirectory() as t:
            d = self._arm(Path(t), 50, True)
            self.assertEqual(self._check(d), "1")

    def test_frames_without_audio_are_not_complete(self) -> None:
        """The DiT drives both streams and the comparison reads both, so an arm
        that lost its wav is an arm half of the verdict cannot be taken on."""
        with tempfile.TemporaryDirectory() as t:
            d = self._arm(Path(t), 49, False)
            self.assertEqual(self._check(d), "1")

    def test_an_empty_wav_is_not_audio(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            d = self._arm(Path(t), 49, True, audio_size=0)
            self.assertEqual(self._check(d), "1")

    def test_an_absent_directory_is_not_complete(self) -> None:
        """AND THE FRAME COUNT IS WHAT REFUSES IT. `arm_is_complete` opened with
        `[ -d "$d" ] || return 1`, and deleting that line left this whole suite
        green: the glob does not expand for a directory that is not there, so
        `ls ... | wc -l` reports 0 and the frame-count check returns 1 on its
        own. The observable behaviour was identical with the guard and without
        it, so it was a redundant guard rather than a defect, and it is gone.
        This case is written down so that the next reader adds it back
        deliberately or not at all."""
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._check(Path(t) / "never-rendered"), "1")

    def test_an_absent_directory_is_not_complete_even_at_zero_frames(self) -> None:
        """The one call shape where the frame count could not refuse an absent
        directory on its own: ask for zero frames and `0 = 0` holds. The wav
        check is what refuses it, so no argument makes an arm that was never
        rendered read as a completed one."""
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._check(Path(t) / "never-rendered", want=0), "1")


class Wiring(unittest.TestCase):
    """Eleven text tripwires on harness lines that only a lease executes.

    THEY STAND OVER SIX SURFACES, AND THE TWO NUMBERS ARE NOT THE SAME NUMBER.
    Phase [D]'s build-identity record carries one of the eleven (#1881).
    Phase [L] carries three of them by itself -- the exit wiring, the exit-3
    arm and the `*)` fallback -- the phase [I] call site carries two (the
    control wiring and the rule that only the PRIMARY pair is the run's exit
    status), the routing assertion carries two (the arm-A/control pairing and
    the three-sided op proof), and the phase [F] unit-gate refusal, the signal
    traps and the render loop's knob-selection line carry one each. The spec's
    section 10.8 counts SURFACES that never execute; this class counts TESTS.
    Both read "six" for one commit, which looked like two records agreeing and
    was two different sets landing on one number.

    THE ROUTING SURFACE IS NO LONGER TEXT-ONLY, AND THIS CLASS STILL ONLY READS
    IT. `test_every_rung_must_resolve_its_own_op_and_neither_other` below pins
    the three `want=` strings and nothing else; the PREDICATE those strings
    describe now runs in `RoutingVerdict`, against a fabricated `render.log`,
    because five mutations of it saw no red at all while it lived below the
    `# END pixab-helpers` marker. The count word on the first line above is the
    number of tests in THIS class, so it does not move.

    THE RENDER LOOP IS NO LONGER PINNED BY NOTHING, AND IT IS NOT PINNED EITHER.
    One line of it is: the branch that spells the fa2 arm's knob as `unset`
    rather than as an empty export. The watchdog poll, the memory floor and the
    timeout in the same loop are still executed nowhere a test can watch.

    Each pins a defect that shipped, in the words that shipped it. None is a
    proof: the lease is the only place these lines run.

    The count was "four" while five tests stood here, and the two the prose left
    out -- the phase [F] unit-gate refusal and the signal traps -- were absent
    from the spec's list of text-pinned guards as well. `TheDisclosureCounts-
    WhatIsThere` below now holds this number against the class, so the next
    tripwire cannot be added silently.
    """

    def setUp(self) -> None:
        self.text = HARNESS.read_text()

    def test_the_control_is_compared_against_the_arm_it_repeats(self) -> None:
        """`fa2-ctl` repeats FA2, so fa2 is arm A of every pair that uses it and
        --control-of says so. With `--a naive` the control was a second
        naive-vs-treatment comparison, which reads about the size of the
        treatment whatever the kernel did."""
        self.assertIn('--a "$OUT/fa2" --b "$OUT/naive" --control "$OUT/fa2-ctl" --control-of a',
                      self.text)
        self.assertIn("--label-a fa2 --label-b naive --label-control fa2-ctl", self.text)
        self.assertIn('--a "$OUT/fa2" --b "$OUT/flash" --control "$OUT/fa2-ctl" --control-of a',
                      self.text)

    def test_the_unset_arm_is_spelled_unset_and_never_an_empty_export(self) -> None:
        """FA-2 IS SELECTED BY ABSENCE, and the empty string is a REFUSAL.
        `ltx2_device.cpp` rejects `VLLM_LTX2_DIT_FLASH_ATTN=""` by name (#1751),
        so `export ...="$knob"` with an empty knob would abort the fa2 arm at its
        first DiT forward -- an hour into a four-hour lease, over a shell quoting
        choice. The branch is what makes an empty knob mean unset, and it is the
        same spelling `ltx25-dit-attn-fa2-hd128-ab.sh` uses."""
        self.assertIn(
            'if [ -n "$knob" ]; then export VLLM_LTX2_DIT_FLASH_ATTN="$knob"; '
            'else unset VLLM_LTX2_DIT_FLASH_ATTN; fi',
            self.text)
        self.assertNotIn('export VLLM_LTX2_DIT_FLASH_ATTN="$knob"\n', self.text)

    def test_every_rung_must_resolve_its_own_op_and_neither_other(self) -> None:
        """#1794 WAS A LABEL THAT LIED FOR MONTHS: an arm named `flash` ran FA-2,
        and only the announced op caught it. A two-sided proof over op18 and op21
        alone cannot see that here either -- an fa2 arm that fell through to flash
        shows op18=0 and op21=1, which the old naive/other `case` read as OK. Each
        rung is now named with its OWN op and the ABSENCE of the other two, and a
        knob outside the ladder is refused instead of being checked as flash."""
        self.assertIn("""n22=$(grep -cE 'op-provider.*op=22 device=1' "$log")""", self.text)
        for want in ("op18>=1 and op21==0 and op22==0",
                     "op21>=1 and op18==0 and op22==0",
                     "op22>=1 and op18==0 and op21==0"):
            self.assertIn(want, self.text)
        self.assertIn("is not one of 0, flash, or unset", self.text)

    def test_only_the_primary_comparison_is_the_runs_exit_status(self) -> None:
        """THREE COMPARISONS RUN AND ONE VERDICT LEAVES THE JOB. `PIXEL_RC` is
        assigned exactly once, from the fa2-vs-naive pipeline; each secondary pair
        captures its own status under its own name on the line immediately after
        its own pipeline, because `$PIPESTATUS` is clobbered by the next command.
        Three statuses reported as one would let a reader quote whichever agreed
        with them. The flash-vs-naive pair passes NO control: `fa2-ctl` repeats
        fa2, and offering it there is the inverted wiring the comment above the
        primary call exists to prevent."""
        self.assertEqual(self.text.count("PIXEL_RC=${PIPESTATUS[0]}"), 1)
        self.assertIn("FA2_FLASH_RC=${PIPESTATUS[0]}", self.text)
        self.assertIn("FLASH_NAIVE_RC=${PIPESTATUS[0]}", self.text)
        # COUNTING THE CAPTURE IS NOT PINNING IT, and a fresh review proved it on
        # this very test. `assertEqual(count, 1)` says the line occurs once and
        # says NOTHING about which pipeline it follows. Two mutations kept the
        # suite green: an `echo` inserted between the primary pipeline and the
        # capture, after which `PIXEL_RC` reads the ECHO's status and phase [L]
        # exits 0 on every failing pixel verdict; and the capture moved onto the
        # fa2-vs-flash pipeline, after which a SECONDARY comparison becomes the
        # run's verdict. `$PIPESTATUS` is clobbered by the next command, so
        # adjacency IS the guarantee, and it is asserted here rather than
        # described in the comment above the line.
        # LINE-ANCHORED, because a substring match is not an anchor. `assertIn`
        # on `python3 "$CMP" \\` is satisfied by `true python3 "$CMP" \\`, which
        # makes the tool's status unreachable and left this test green.
        started = re.search(
            r'(?m)^python3 "\$CMP" \\\n'
            r'  --a "\$OUT/fa2" --b "\$OUT/naive" --control "\$OUT/fa2-ctl" --control-of a',
            self.text)
        self.assertIsNotNone(
            started,
            "the primary comparison must START its own line and be `python3 \"$CMP\"` "
            "itself. A word in front of it -- `true`, `echo`, a wrapper -- leaves the "
            "adjacency assertion below satisfied while the verdict comes from something "
            "that is not the comparison tool.")
        i = started.start()
        j = self.text.index("PIXEL_RC=${PIPESTATUS[0]}", i)
        # EQUALITY, NOT A SEARCH. `assertRegex` with a `\\Z` anchor matches a
        # SUFFIX, so a decoy pipeline whose last line is byte-identical to the
        # primary's `--json ... | tee` line satisfies it while `$PIPESTATUS` then
        # reads the DECOY. That is the same substring-is-not-an-anchor defect the
        # line-anchored regex above exists to fix, one assertion later, and a
        # review found it here after the first repair.
        self.assertEqual(
            self.text[i:j],
            started.group(0) + ' \\\n'
            '  --label-a fa2 --label-b naive --label-control fa2-ctl \\\n'
            '  --json "$OUT/pixel-compare.json" 2>&1 | tee "$OUT/pixel-compare.txt"\n',
            "everything between the primary invocation and the capture must be the "
            "REST OF THAT ONE PIPELINE and nothing else. Any other command there -- "
            "including a second pipeline that ends in the same bytes -- clobbers "
            "$PIPESTATUS, and the job then exits on a status that is not the pixel "
            "verdict.")
        # THE CAPTURE IS ALSO THE ONLY ASSIGNMENT, and the re-review of the first
        # repair is why this line exists. Pinning what PRECEDES the capture leaves
        # the same defect reachable one line LATER: `PIXEL_RC=$?` on the next line
        # reads the ASSIGNMENT's status, which is always 0, and `PIXEL_RC=0`
        # anywhere before phase [L] does it in one word. Both stayed green against
        # the adjacency assertion alone.
        # EVERY OCCURRENCE OF THE NAME, not every line that starts with it. A
        # line-anchored count says nothing about `x=1; PIXEL_RC=0`,
        # `export PIXEL_RC=0`, `declare PIXEL_RC=0` or `printf -v PIXEL_RC 0`, and
        # a review demonstrated all four at runtime: each leaves the file with a
        # second write to the verdict and the counting assertion still reading 1.
        # So the rule is stated over the NAME: `PIXEL_RC` may appear as the single
        # capture, or preceded by `$` as a read, and nowhere else.
        writes = [m.start() for m in re.finditer(r"(?<!\$)\bPIXEL_RC\b", self.text)]
        capture = self.text.index("PIXEL_RC=${PIPESTATUS[0]}")
        stray = [w for w in writes if w != capture]
        self.assertEqual(
            stray, [],
            "PIXEL_RC must be WRITTEN exactly once, by the capture, and read only "
            "as $PIXEL_RC. Every other appearance of the bare name is a second "
            "write to the run's verdict, whatever syntax it uses: "
            + "; ".join(
                repr(self.text[w:self.text.index(chr(10), w)]) for w in stray[:5]))
        block = self.text.split("# (b) flash vs naive")[1].split("FLASH_NAIVE_RC=")[0]
        self.assertNotIn("--control", block,
                         "the flash-vs-naive pair has no control on this ladder and must "
                         "not be given one that repeats a different arm")
        self.assertIn("flash_vs_naive_control=none", self.text)

    def test_the_run_records_the_LIBRARY_and_not_only_the_launcher(self) -> None:
        """#1881. `ltx2-gen` is a 75 KB `main()`; every op this A/B measures is in
        `libvllm.so.0.0.3`. Run 1612-r3 (source `3e2961ef0`) and this ladder's
        first run (source `62cbae10d`) both recorded
        `binary_sha256=834cec557c16cf77...` while their libraries differed by
        6,372,624 bytes, because the launcher's translation unit did not change
        and its output is reproducible BY CONSTRUCTION. An identity that cannot
        see the change is not an identity."""
        self.assertIn(
            """LIBSHA=$(sha256sum "$BIN/libvllm.so.0.0.3" | awk '{print $1}')""",
            self.text)
        self.assertIn("library_sha256=$LIBSHA", self.text)
        self.assertIn('echo "[arm] library=$BIN/libvllm.so.0.0.3 sha256=$LIBSHA"',
                      self.text,
                      "every arm's own log must carry the library identity, because an "
                      "arm's log is what a later reader quotes")
        # THE FOURTH SITE. `LIBSHA` is computed, written to `PROVENANCE`, written
        # into every arm's log -- and PRINTED, which is the one a reader watching
        # the job sees first and the only one no assertion held. Deleting the
        # other three reds; deleting this one did not.
        self.assertIn(
            'say "                             libvllm.so.0.0.3=$LIBSHA'
            '  <- the one that carries the kernels"',
            self.text,
            "the run must PRINT the library hash beside the launcher hash. A "
            "reader who sees only `ltx2-gen=...` on stdout reads the identity "
            "that #1881 proved cannot see the change")

    def test_the_run_exits_with_the_pixel_verdict(self) -> None:
        """THE EXIT IS THE LAST LINE, and `assertIn` could not say so. A review of
        the repair above found the same substring-is-not-an-anchor defect here:
        wrapping this exact string in a never-taken `if` and adding a bare
        `exit 0` after it left the test green while the job exited 0 on every
        failing verdict. The exit must therefore start its own line and be the
        LAST statement in the file."""
        self.assertIn('PIXEL_RC=${PIPESTATUS[0]}', self.text)
        self.assertRegex(self.text, r'(?m)^exit "\$PIXEL_RC"$')
        tail = [l for l in self.text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        self.assertEqual(
            tail[-1], 'exit "$PIXEL_RC"',
            f"the last executable line of the harness must be the verdict exit; it "
            f"is {tail[-1]!r}. Anything after it can exit on a status that is not "
            "the pixel verdict.")

    def test_a_routing_failure_stops_the_run(self) -> None:
        self.assertIn(f"exit {ROUTING_BAD}", self.text)
        # Computed into a variable and tested OUTSIDE the pipeline: an `exit`
        # inside `case ... | tee` leaves the subshell, not the run.
        self.assertIn('if [ "$routing" = OK ]; then', self.text)

    def test_the_unit_gate_refuses_rather_than_reporting(self) -> None:
        self.assertIn(f"exit {UNIT_GATE_FAILED}", self.text)
        self.assertIn(f"exit {UNIT_GATE_ABSENT}", self.text)

    def test_a_degenerate_control_is_a_status_the_run_defines(self) -> None:
        """The comparison gained exit 3 -- the treatment passed and the control
        rendered no picture -- and phase [L]'s `case` has a `*)` arm that calls
        an unlisted status UNKNOWN. An exit this repository defines must not
        reach it, because "the comparison exited 3, which it does not define"
        reads as a harness defect rather than as the verdict it is."""
        self.assertIn("  3) say \"PIXEL VERDICT: CONTROL DEGENERATE", self.text)
        self.assertIn("do not publish a reading from this run", self.text)

    def test_an_unlisted_status_still_gets_a_verdict_line(self) -> None:
        """THE `*)` ARM ABOVE IS A GUARD AND NOTHING HELD IT. The test directly
        above cites it -- "phase [L]'s `case` has a `*)` arm that calls an
        unlisted status UNKNOWN" -- as the whole reason exit 3 needed its own
        arm, and deleting the `*)` line left this suite green at 22/22. Without
        it a status the comparison never defined (a `python3` that died on an
        import, a 137 from the OOM killer) prints no verdict at all: the run
        ends on DONE and exits on a number nobody named, which is the exact
        silence phase [L] was added to remove.
        """
        marker = 'case "$PIXEL_RC" in'
        self.assertIn(marker, self.text, "phase [L] no longer branches on the verdict")
        block = self.text.split(marker, 1)[1].split("esac", 1)[0]
        arm = re.search(r"^\s*\*\)\s*(say .*)$", block, re.M)
        self.assertIsNotNone(
            arm, "phase [L]'s `case` has no `*)` arm, so a status it does not "
                 "enumerate produces no verdict line at all")
        self.assertIn("PIXEL VERDICT: UNKNOWN", arm.group(1))
        self.assertIn("$PIXEL_RC", arm.group(1),
                      "the fallback must print the status it could not name")

    def test_the_heartbeat_is_reaped_on_a_lease_kill(self) -> None:
        """`rc` reclaiming a device sends SIGTERM, and a bash EXIT trap does not
        run for a signal with no handler of its own."""
        self.assertIn("trap cleanup EXIT", self.text)
        for sig, status in (("HUP", 129), ("INT", 130), ("TERM", 143)):
            self.assertIn(f"trap 'cleanup; exit {status}' {sig}", self.text)


class ThePhaseICommentDescribesTheToolItCalls(unittest.TestCase):
    """[I]'s comment is the only place a reader is told what the run's exit
    status means, and it went stale the moment [L] gained a fourth verdict.

    `12c880a52` added exit 3 -- the treatment passed and the CONTROL rendered no
    picture -- and it updated the header's EXIT STATUS block and phase [L]'s
    `case`. Phase [I]'s comment still enumerated "0 pass, 1 a threshold failed,
    2 an input could not be read", so a reader who trusted it would have read a
    3 as an undefined status, or worse, quoted `PIXEL_COMPARE_RC` as a pass
    because it was neither 1 nor 2. These two tests derive the statuses from
    the `case` that prints them, so the comment cannot fall behind the code
    again without a red.
    """

    def setUp(self) -> None:
        self.text = HARNESS.read_text()

    def _phase_i_comment(self) -> str:
        marker = 'say "=== [I] the pixel comparison ==="'
        self.assertIn(marker, self.text, "phase [I] is not in this harness")
        after = self.text.split(marker, 1)[1]
        self.assertIn('python3 "$CMP"', after, "phase [I] no longer calls the tool")
        return after.split('python3 "$CMP"', 1)[0]

    def _verdict_statuses(self) -> list[str]:
        """The statuses phase [L]'s `case` actually prints a verdict for."""
        found = sorted(set(re.findall(r'^\s*(\d+)\) say "PIXEL VERDICT:', self.text,
                                      re.M)), key=int)
        # Guard the instrument before trusting it: a regex that matched nothing
        # would make every assertion below vacuously true.
        self.assertEqual(found, ["0", "1", "2", "3"],
                         "phase [L] defines a different set of verdict statuses than this "
                         "gate was written against; update the comment in [I], the header "
                         "EXIT STATUS block and this line together")
        return found

    def test_the_phase_i_comment_names_every_status_the_verdict_defines(self) -> None:
        comment = self._phase_i_comment()
        for s in self._verdict_statuses():
            self.assertIn(s, comment,
                          f"phase [L] prints a verdict for exit {s} and phase [I]'s comment "
                          f"does not name it. The comment is what a reader trusts when they "
                          f"quote PIXEL_COMPARE_RC.")

    def test_the_phase_i_comment_refuses_both_non_pass_statuses_by_name(self) -> None:
        """2 and 3 are the two that are NOT failures and are NOT passes either,
        which is exactly the pair a reader mis-reads as a pass."""
        comment = self._phase_i_comment()
        self.assertIn("never a pass", comment)
        for s in ("2", "3"):
            self.assertRegex(comment, rf"(?s)never a pass.{{0,80}}\b{s}\b|\b{s}\b.{{0,80}}never a pass",
                             f"the comment must say that a {s} is never a pass")

    def test_the_phase_i_comment_names_where_the_comparison_tool_came_from(self) -> None:
        """`$CMP` is `$SRC/scripts/ltx25-render-compare.py`, and `$SRC` is
        unpacked in phase [B] from `$W/pixab-src.tar.gz` -- a tarball STAGED ON
        THE SHARE, not this checkout. A lease whose tarball predates the exit-3
        repair cannot return a 3 at all: a degenerate control exits 0 there. The
        comment must send the reader to `source_sha` rather than let them assume
        the tool is the one beside this comment."""
        comment = self._phase_i_comment()
        self.assertIn("pixab-src.tar.gz", comment)
        self.assertIn("source_sha", comment)



# A FABRICATED `render.log`, because the routing proof reads nothing else. The
# op-provider line is the shape `VT_OP_PROVIDER_STATS=1` prints once per resolved
# op, and `device=1` is kCUDA: a selection on another device is not this arm's
# selection, which is why the harness's grep carries the qualifier and why one
# case below fabricates `device=0`.
def render_log(path: Path, ops: dict[int, int], library: str | None = None) -> Path:
    lines = ["[arm] label=lbl knob=? tmo=1s\n"]
    if library is not None:
        lines.append(f"[arm] library=/root/pixabbin/libvllm.so.0.0.3 sha256={library}\n")
    for op, n in sorted(ops.items()):
        for i in range(n):
            lines.append(f"[vt op-provider] op={op} device=1 provider=cuda name=sel{i}\n")
    lines.append("[engine] step 1 last=1.234s\n")
    path.write_text("".join(lines))
    return path


SHA_A = "a" * 64
SHA_B = "b" * 64


class RoutingVerdict(unittest.TestCase):
    """THE PREDICATE, RUN. Not the sentence next to it.

    `Wiring.test_every_rung_must_resolve_its_own_op_and_neither_other` greps for
    the three `want=` strings, so it pins what the verdict SAYS it requires. A
    fresh review of #1871 applied five semantic mutations to the lines that
    decide it -- the fa2 rung accepting a flash fallback, the unknown-knob arm
    flipped from BAD to OK, the `op18>=1` and `op22>=1` floors muted to `>=0`,
    and the if/else wrapped back inside the `| tee` pipeline so that `exit 46`
    leaves a subshell -- and saw no red at all, because nothing in this
    repository executed `arm_report`.

    ONE OF THE FIVE IS INERT, and it was worth measuring rather than assuming.
    `&&` and `||` are left-associative and of equal precedence in bash, so the
    fa2 rung's `n22>=1 || n21>=1 && n18==0 && n21==0` still ends in `n21==0` and
    still refuses a flash fallback: `n18=0 n21=1 n22=0` reads BAD before and
    after. That mutation stays green here too, and correctly. Replacing the
    WHOLE condition is the form that accepts a fallback, and the `("", {21: 3})`
    row below reds on it.

    So the verdict moved ABOVE the `# END pixab-helpers` marker as
    `routing_verdict`, where the same extraction that already runs the memory
    gate can run it against a fabricated log. It needs no GPU, no checkpoint and
    no binary: `$log`, `$d`, grep, sed, sort and tee are its whole world.

    THE LAST CASE IN EVERY BAD ROW IS THE `exit`, NOT THE WORD. Each refusal
    asserts status 46 AND that the statement after the call never ran. A
    `ROUTING_BAD` printed by a function that returns 0 is #1794's defect exactly:
    a word in a log, four renders of one configuration, and a perfect match.
    """

    # (knob, ops, expected verdict). `""` is the fa2 rung: FA-2 is selected by
    # the ABSENCE of the variable, so the empty knob is a rung and not a typo.
    CASES = (
        ("",         {22: 3},         "OK"),
        ("",         {},              "BAD"),   # a log with no selection at all
        ("",         {21: 3},         "BAD"),   # the flash fallback #1751 was filed for
        ("",         {18: 3},         "BAD"),
        ("",         {22: 3, 21: 1},  "BAD"),   # routed AND fell back
        ("",         {22: 3, 18: 1},  "BAD"),
        ("flash",    {21: 3},         "OK"),
        ("flash",    {},              "BAD"),
        ("flash",    {22: 3},         "BAD"),   # the #1794 label that lied
        ("flash",    {21: 3, 18: 1},  "BAD"),
        ("flash",    {21: 3, 22: 1},  "BAD"),
        ("0",        {18: 3},         "OK"),
        ("0",        {},              "BAD"),
        ("0",        {21: 3},         "BAD"),
        ("0",        {22: 3},         "BAD"),
        ("0",        {18: 3, 22: 1},  "BAD"),
        ("1",        {21: 3},         "BAD"),   # #1751 retired `=1`; it is not a rung
        ("flash-ctl", {21: 3},        "BAD"),
    )

    def _run(self, knob: str, ops: dict[int, int], t: str):
        d = Path(t) / "arm"
        d.mkdir()
        log = render_log(d / "render.log", ops)
        p = bash("routing_verdict lbl %s %s %s\necho \"SURVIVED rc=$?\"" % (
            shlex.quote(knob), shlex.quote(str(d)), shlex.quote(str(log))))
        arm = (d / "ARM").read_text() if (d / "ARM").exists() else ""
        return p, arm

    def test_the_table_covers_every_rung_and_every_knob_the_ladder_sets(self) -> None:
        """Guard the instrument before trusting it. A table that lost its `0` row
        would leave `test_each_rung` green over a smaller experiment."""
        self.assertEqual({k for k, _, _ in self.CASES},
                         {"", "flash", "0", "1", "flash-ctl"})
        self.assertEqual(sorted({v for _, _, v in self.CASES}), ["BAD", "OK"])

    def test_each_rung_resolves_its_own_op_and_neither_other(self) -> None:
        for knob, ops, want in self.CASES:
            with self.subTest(knob=knob or "<unset>", ops=ops, want=want):
                with tempfile.TemporaryDirectory() as t:
                    p, arm = self._run(knob, ops, t)
                if want == "OK":
                    self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                    self.assertIn("ROUTING_OK=lbl", p.stdout)
                    self.assertIn("ROUTING_OK=lbl", arm)
                    self.assertIn("SURVIVED rc=0", p.stdout)
                else:
                    self.assertEqual(
                        p.returncode, ROUTING_BAD,
                        "a mis-routed arm must END THE RUN with %d, not print a word "
                        "and carry on: %s" % (ROUTING_BAD, p.stdout + p.stderr))
                    self.assertIn("ROUTING_BAD=lbl", p.stdout)
                    self.assertIn("ROUTING_BAD=lbl", arm)
                    self.assertNotIn(
                        "SURVIVED", p.stdout,
                        "the statement after the routing proof RAN, so `exit 46` left a "
                        "subshell rather than the run -- #1794's defect, in which "
                        "ROUTING_BAD was a word in a log and four renders went ahead")

    def test_a_selection_on_another_device_is_not_this_arms_selection(self) -> None:
        """`device=1` is kCUDA and the grep carries it. A host-side resolution of
        op=22 does not prove the CUDA arm routed, and dropping the qualifier would
        let one appear to."""
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "arm"
            d.mkdir()
            (d / "render.log").write_text(
                "[vt op-provider] op=22 device=0 provider=cpu name=sel0\n")
            p = bash("routing_verdict lbl '' %s %s\necho SURVIVED" % (
                shlex.quote(str(d)), shlex.quote(str(d / "render.log"))))
        self.assertEqual(p.returncode, ROUTING_BAD, p.stdout + p.stderr)
        self.assertIn("op22_fa2=0", p.stdout)

    def test_the_counts_are_printed_with_the_verdict(self) -> None:
        """The verdict line names what it saw, so a reader of an ARM file can
        recompute it. A bare OK/BAD would make the record unreadable."""
        with tempfile.TemporaryDirectory() as t:
            p, arm = self._run("flash", {21: 5, 19: 2}, t)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("op18_naive=0 op21_flash=5 op22_fa2=0", arm)
        self.assertIn("saw op18=0 op21=5 op22=0", arm)

    def test_an_unknown_knob_is_named_in_the_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            p, arm = self._run("1", {21: 3}, t)
        self.assertEqual(p.returncode, ROUTING_BAD, p.stdout + p.stderr)
        self.assertIn('knob "1" is not one of 0, flash, or unset', arm)


class ResumedArmLibrary(unittest.TestCase):
    """#1881, ONE LEVEL DOWN: a resumed arm was rendered by a DIFFERENT library.

    The binary cache lives on `$W`, which is CIFS; the arm directories live under
    `$OUT`. A lease that finds the arms but not the cache REBUILDS, and this tree
    is not byte-reproducible, so the rebuilt `libvllm.so.0.0.3` has a new sha256.
    The resume branch used to append only `timing_source` and never look at the
    arm's own `render.log`, so the run then compared arms produced by two
    different libraries while `PROVENANCE` recorded ONE `library_sha256`. That is
    exactly the reading #1881 removed from the launcher hash, reappearing across
    a lease boundary instead of across a build boundary.

    ABSENCE IS REFUSED TOO, and that is a decision rather than an oversight. An
    arm rendered before the `[arm] library=` line existed cannot be proved to
    carry this lease's library; unknown is not a match. The cost is re-rendering
    such an arm, and the message says so.
    """

    def _run(self, log_sha: str | None, want: str):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "arm"
            d.mkdir()
            log = render_log(d / "render.log", {22: 1}, library=log_sha)
            p = bash("resumed_arm_library lbl %s %s %s\necho \"SURVIVED rc=$?\"" % (
                shlex.quote(str(d)), shlex.quote(str(log)), shlex.quote(want)))
            return p, ((d / "ARM").read_text() if (d / "ARM").exists() else "")

    def test_a_resumed_arm_from_this_lease_is_accepted_and_says_so(self) -> None:
        p, arm = self._run(SHA_A, SHA_A)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("SURVIVED rc=0", p.stdout)
        self.assertIn("library_sha256=" + SHA_A, arm)

    def test_a_resumed_arm_from_another_library_ends_the_run(self) -> None:
        p, arm = self._run(SHA_B, SHA_A)
        self.assertEqual(
            p.returncode, RESUMED_LIBRARY_DIFFERS,
            "an arm rendered by another library must END THE RUN; comparing it "
            "against this lease's arms publishes a pixel delta between two "
            "binaries under one `library_sha256`: " + p.stdout + p.stderr)
        self.assertNotIn("SURVIVED", p.stdout)
        self.assertIn("library_sha256_mismatch=" + SHA_B, arm)
        self.assertIn(SHA_A, arm)

    def test_a_resumed_arm_that_cannot_say_ends_the_run(self) -> None:
        p, arm = self._run(None, SHA_A)
        self.assertEqual(p.returncode, RESUMED_LIBRARY_DIFFERS,
                         p.stdout + p.stderr)
        self.assertNotIn("SURVIVED", p.stdout)
        self.assertIn("library_sha256=unknown", arm)

    def test_the_resume_branch_calls_it_on_its_own_line(self) -> None:
        """The call site is the one part of this that only a lease runs. It must
        not sit inside a pipeline: `exit` there leaves the subshell, which is the
        defect `routing_verdict` was moved out of a `| tee` to remove."""
        text = HARNESS.read_text()
        self.assertRegex(
            text,
            r'(?m)^\s*resumed_arm_library "\$label" "\$d" "\$log" "\$LIBSHA"\s*$',
            "the resume branch must call the library check as a bare statement")


class TheVerdictCannotBeBypassed(unittest.TestCase):
    """FOUR MORE WAYS PAST THE EXIT STATUS, none of them a live defect.

    `6e7bcb3f2` pinned the last executable line and made `PIXEL_RC` writable
    exactly once. A fresh review of #1871 then walked past both four times, and
    the shipped file is correct in all four cases -- this class is coverage, not
    a repair:

      the EXIT trap outranks the last line. `exit 0` appended to `cleanup`, or a
      `trap 'exit 0' EXIT` anywhere above phase [L], makes every failing verdict
      exit 0 and leaves `exit "$PIXEL_RC"` in place, still last, still pinned.

      `eval "PIXEL""_RC=0"` writes the verdict through a name that does not
      appear in the file. THE HONEST FIX IS A BAN, not a stronger sweep: a sweep
      reads text, and the same trick survives every sweep in another spelling
      (`declare "PIX""EL_RC=0"`, a `source`d fragment). `eval` has no use in this
      harness, so it is refused outright, and the assignment builtins are refused
      a QUOTED target for the same reason. What remains uncovered is stated
      rather than papered over: only executing phase [L] proves the verdict, and
      that needs the lease.

      `CMP` is the tool whose status becomes the verdict, and nothing pinned its
      value. `CMP=$W/always-pass.py` before phase [I] makes every comparison
      pass while every assertion about the pipeline around it still holds.
    """

    def setUp(self) -> None:
        self.text = HARNESS.read_text()
        self.exec_lines = [l for l in self.text.splitlines()
                           if not l.lstrip().startswith("#")]

    def test_cleanup_reaps_the_heartbeat_and_does_nothing_else(self) -> None:
        defs = re.findall(r"(?m)^\s*cleanup\(\).*$", self.text)
        self.assertEqual(
            defs, ['cleanup() { kill "$HEARTBEAT" 2>/dev/null; }'],
            "`cleanup` runs on EXIT and on three signals, so ANY statement added "
            "to it runs at the end of every path this job takes. An `exit 0` "
            "there overrides the verdict exit while leaving it the last line of "
            "the file. It is a one-liner and it is pinned as one: " + repr(defs))

    def test_the_only_EXIT_trap_is_the_heartbeat_reaper(self) -> None:
        traps = [l.strip() for l in self.exec_lines if re.match(r"trap\s", l.strip())]
        self.assertEqual(
            [t for t in traps if "EXIT" in t], ["trap cleanup EXIT"],
            "an EXIT trap runs after the last line and its status REPLACES the "
            "one the script exited with, so a second one is a second verdict: "
            + repr(traps))
        self.assertEqual(len(traps), 4,
                         "the traps are EXIT plus HUP, INT and TERM: " + repr(traps))

    def test_the_comparison_tool_is_assigned_exactly_once(self) -> None:
        """Same rule as `PIXEL_RC`, over the name that PRODUCES it. Every read is
        `$CMP`; the bare name may appear only as the single assignment."""
        writes = [m.start() for m in re.finditer(r"(?<!\$)\bCMP\b", self.text)]
        first = self.text.index('CMP="$SRC/scripts/ltx25-render-compare.py"')
        stray = [w for w in writes if w != first]
        self.assertEqual(
            stray, [],
            "`CMP` names the tool whose exit status becomes the run's verdict. It "
            "is written once, in phase [B], where phase [B] also refuses an "
            "unreadable one. A second assignment anywhere is a second tool: "
            + "; ".join(repr(self.text[w:self.text.index(chr(10), w)])
                        for w in stray[:5]))

    def test_no_assignment_target_is_built_out_of_a_quoted_string(self) -> None:
        ASSIGNERS = r"(?:eval|declare|typeset|readonly|export|read|printf\s+-v)"
        offenders = []
        for line in self.exec_lines:
            m = re.match(r"\s*" + ASSIGNERS + r"\b(?P<rest>.*)$", line)
            if not m:
                continue
            if re.match(r"\s*eval\b", line):
                offenders.append(line.strip())
                continue
            rest = m.group("rest").strip()
            while rest.startswith("-"):
                rest = rest.split(None, 1)[1].strip() if " " in rest else ""
            if rest[:1] in ("'", '"'):
                offenders.append(line.strip())
        self.assertEqual(
            offenders, [],
            "`eval` is refused outright and an assignment builtin may not take a "
            "quoted target. `eval \"PIXEL\"\"_RC=0\"` writes the run's verdict "
            "through a name the sweep in `Wiring` cannot see, because the name is "
            "not in the file. This ban closes the demonstrated bypass; it is a "
            "tripwire and not a proof, since a text gate can never see a name "
            "built at run time: " + repr(offenders))

    def test_the_routing_proof_is_called_as_a_bare_statement(self) -> None:
        """`arm_report` calls it; it must not be piped. #1794's `exit` was
        unreachable precisely because it stood inside `case ... | tee`."""
        calls = [l.strip() for l in self.exec_lines
                 if l.strip().startswith("routing_verdict ")]
        self.assertEqual(calls, ['routing_verdict "$label" "$knob" "$d" "$log"'],
                         "the routing proof is called once, from `arm_report`, "
                         "outside any pipeline: " + repr(calls))


class TheDisclosureCountsWhatIsThere(unittest.TestCase):
    """The count of text tripwires is itself a claim, and it had drifted.

    `Wiring` said "the four call sites" while defining five tests, and the
    spec's section 10.8 and `## Owed` named four unexecuted things -- the render
    loop, the routing assertion, the phase [I] call site and the phase [L] exit
    -- while two further guards, the phase [F] unit-gate refusal and the signal
    traps, were text-only and in neither list. The error was in the safe
    direction: nothing claimed as EXECUTED was in fact only text-pinned. It is
    still a number in a document that no longer described the file beside it,
    which is the shape section 10.6 is about, and it is cheap to hold.
    """

    # WORD BOUNDARIES, because a substring match is not a word match and this
    # dict is read by substring in one of the gates below. "eleven" CONTAINS
    # "seven": with both keys present a plain `in` finds two count words and the
    # docstring gate refuses a docstring that names exactly one; with only
    # "seven" present it finds the wrong one and reports 7 against 11. The same
    # trap waits in "seventeen", "nineteen" and "eighteen". The lookup below is
    # therefore anchored on `\b`.
    WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12}

    SPEC = ROOT / ".agents/specs/ltx25-dit-attn-flash.md"

    def test_the_wiring_docstring_names_as_many_tripwires_as_it_defines(self) -> None:
        doc = Wiring.__doc__ or ""
        first = doc.split("\n")[0].lower()
        found = [w for w in self.WORDS if re.search(rf"\b{w}\b", first)]
        self.assertEqual(len(found), 1,
                         f"the first line of Wiring's docstring must name exactly one "
                         f"count word so this gate can read it; it names {found}")
        claimed = self.WORDS[found[0]]
        defined = len([m for m in dir(Wiring) if m.startswith("test_")])
        self.assertEqual(claimed, defined,
                         f"Wiring's docstring claims {claimed} tripwires and the class "
                         f"defines {defined}. Adding a tripwire without saying so leaves "
                         f"a count in a document that no longer describes the file.")

    def test_the_spec_counts_tripwire_TESTS_and_not_unexecuted_SURFACES(self) -> None:
        """TWO SIXES THAT COUNT DIFFERENT SETS ARE NOT A CROSS-CHECK. Section
        10.8 and `## Owed` count the harness lines that do NOT execute -- six of
        them, five text-pinned, one pinned by nothing -- while `Wiring` counts
        the TESTS that pin them. Both read "six" and the coincidence was read as
        agreement, so the tripwire the two lists were missing (phase [L]'s
        `case` arm for exit 3) was invisible in both. The spec now states the
        test count in its own words, and this gate holds that number against the
        class the way the docstring gate above holds the docstring's.
        """
        text = self.SPEC.read_text()
        found = re.findall(r"([A-Za-z]+) tripwire tests", text)
        self.assertGreaterEqual(
            len(found), 2,
            "section 10.8 and `## Owed` must each state how many tripwire TESTS "
            f"stand over the surfaces they list, in the form '<word> tripwire "
            f"tests', so the two counts cannot be mistaken for one; found {found}")
        words = {w.lower() for w in found}
        self.assertEqual(len(words), 1,
                         f"the spec states more than one tripwire-test count: {sorted(words)}")
        word = words.pop()
        self.assertIn(word, self.WORDS, f"{word!r} is not a count this gate can read")
        defined = len([m for m in dir(Wiring) if m.startswith("test_")])
        self.assertEqual(self.WORDS[word], defined,
                         f"the spec claims {self.WORDS[word]} tripwire tests and Wiring "
                         f"defines {defined}. A count in a document must describe the "
                         f"file beside it.")
        self.assertIn("`case` arm", text,
                      "the tripwire the two lists omitted is phase [L]'s `case` arm; the "
                      "spec must name it, not leave it inside a total")


if __name__ == "__main__":
    unittest.main()
