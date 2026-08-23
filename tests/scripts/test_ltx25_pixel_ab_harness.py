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
    def test_the_block_exists_and_holds_the_three_functions(self) -> None:
        block = helper_block()
        for fn in ("mem_avail_gib()", "wait_for_memory()", "arm_is_complete()"):
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
    """Seven text tripwires on harness lines that only a lease executes.

    THEY STAND OVER FIVE SURFACES, AND THE TWO NUMBERS ARE NOT THE SAME NUMBER.
    Phase [L] carries three of the seven by itself -- the exit wiring, the
    exit-3 arm and the `*)` fallback -- and the phase [I] call site, the routing
    assertion, the phase [F] unit-gate refusal and the signal traps carry one
    each. The spec's section 10.8 counts SURFACES that never execute; this class
    counts TESTS. Both read "six" for one commit, which looked like two records
    agreeing and was two different sets landing on one number.

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
        """`flash-ctl` repeats FLASH, so flash is arm A and --control-of says so.
        With `--a naive` the control was a second naive-vs-flash comparison."""
        self.assertIn('--a "$OUT/flash" --b "$OUT/naive" --control "$OUT/flash-ctl" --control-of a',
                      self.text)
        self.assertIn("--label-a flash --label-b naive --label-control flash-ctl", self.text)

    def test_the_run_exits_with_the_pixel_verdict(self) -> None:
        self.assertIn('PIXEL_RC=${PIPESTATUS[0]}', self.text)
        self.assertIn('exit "$PIXEL_RC"', self.text)

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

    WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}

    SPEC = ROOT / ".agents/specs/ltx25-dit-attn-flash.md"

    def test_the_wiring_docstring_names_as_many_tripwires_as_it_defines(self) -> None:
        doc = Wiring.__doc__ or ""
        found = [w for w in self.WORDS if w in doc.split("\n")[0].lower()]
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
