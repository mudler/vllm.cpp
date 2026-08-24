#!/usr/bin/env python3
"""The three LTX-2.5 A/B harnesses' memory watchdog, exercised without a GPU.

[#1734](https://github.com/mudler/vllm.cpp/issues/1734),
`.agents/specs/ltx25-dit-attn-fa2-hd128.md` section 8.11.

Both arms of the `LTX25-DIT-ATTN-FA2-HD128` A/B ended their summary with

    memavail low-water:  GiB

on a run whose watch files held a perfectly good 40.3 GiB. A MISSING measurement
printed as a field, in a harness that backs a published attention ratio, and a
blank where a number belongs reads as "measured, and fine".

The cause was the WRITER, not the reducer that printed it. `grep -c` prints `0`
AND exits 1 when it matches nothing, so `grep -c ... || echo 0` fired the
fallback on top of grep's own count and produced the two-line string `0\\n0`. The
poll's tab-separated record went out through `echo`, so 170 of that run's 186
records landed split across two lines, and the positional reducer's `$4` was
empty on every one of them. An empty string sorts first under `sort -n`.

The same idiom stood in three files in `scripts/`, written three ways, two of
them wrong. `ltx25-dit-attn-flash-ab.sh` carried the identical defect;
`ltx25-dit-attn-flash-pixel-ab.sh` had the working writer and the fragile
reducer. This suite therefore holds TWO independent guarantees and the fact that
there is ONE spelling of them:

  the count       is a single integer on one line, whatever the log is;
  the reducer     finds the reading by its KEY and never by a field number, and
                  says NO READINGS rather than printing nothing;
  the block       is byte-identical in all three harnesses.

WHAT THIS CANNOT DO. It does not run a render, and it does not need a GPU. The
poll body and the report line are pulled out of the committed harnesses by
UNIQUE regex and executed, so this is not a transcription of them -- but
everything around those lines is only reached inside a four-hour lease on
`dgx:gpu0`, and that is stated here rather than papered over.

The red-before run is taken by pointing `MEMWATCH_SCRIPTS` at a scratch copy of
a whole `scripts/` DIRECTORY, not at three files:

    git archive 27d8bfa70 scripts | tar -x -C /tmp/red
    MEMWATCH_SCRIPTS=/tmp/red/scripts python3 tests/scripts/test_ltx25_ab_memwatch.py

Three harnesses alone make `TheIdiomIsGoneFromEveryShellScript`'s floor red for
the wrong reason -- it cannot find `dspark-paired-e2e.sh` -- and a spurious red
inside a red-before run is indistinguishable from a real one.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Overridable ONLY so the red-before run can be taken against earlier harnesses.
SCRIPTS = Path(os.environ.get("MEMWATCH_SCRIPTS", ROOT / "scripts"))

HARNESSES = (
    "ltx25-dit-attn-fa2-hd128-ab.sh",
    "ltx25-dit-attn-flash-ab.sh",
    "ltx25-dit-attn-flash-pixel-ab.sh",
)

BEGIN = "# BEGIN memwatch-helpers"
END = "# END memwatch-helpers"

PIX_BEGIN = "# BEGIN pixab-helpers"
PIX_END = "# END pixab-helpers"


def harness(name: str) -> Path:
    p = SCRIPTS / name
    if not p.is_file():
        raise AssertionError(f"{p} is not a file: this suite would exercise nothing")
    return p


def helper_block(name: str) -> str:
    text = harness(name).read_text()
    if BEGIN not in text or END not in text:
        raise AssertionError(
            f"{harness(name)} carries no {BEGIN!r}/{END!r} block: there is nothing "
            f"to exercise, and #1734's idiom is loose in this file again")
    return text.split(BEGIN, 1)[1].split(END, 1)[0]


def bash(snippet: str, name: str = HARNESSES[0], env: dict[str, str] | None = None,
         prelude: str = "") -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update(env or {})
    body = "set -u\n" + prelude + helper_block(name) + "\n" + snippet
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True, env=e)


def only(pattern: str, text: str, what: str) -> str:
    """The one match, or an assertion. Existence is not the property wanted.

    Two near-identical poll bodies in one harness is exactly how a positional
    edit lands on the wrong one, so this refuses a second match as loudly as it
    refuses none.
    """
    found = re.findall(pattern, text, re.M)
    if len(found) != 1:
        raise AssertionError(f"{what}: expected exactly 1 match for {pattern!r}, "
                             f"found {len(found)}")
    return found[0]


class TheBlockIsOneSpelling(unittest.TestCase):
    """A block that must match byte-for-byte is what stops a fourth spelling.

    #1734 was not one bug: it was one idiom written three ways in one directory,
    and only a reader who opened all three files could see that two of them were
    wrong. Fixing two of three would have left the divergence that produced it.
    """

    def test_every_harness_carries_the_block(self) -> None:
        for name in HARNESSES:
            with self.subTest(name):
                self.assertIn("sample_count()", helper_block(name))
                self.assertIn("memavail_low_water()", helper_block(name))

    def test_the_three_blocks_are_byte_identical(self) -> None:
        blocks = {name: helper_block(name) for name in HARNESSES}
        first = blocks[HARNESSES[0]]
        for name, b in blocks.items():
            self.assertEqual(
                b, first,
                f"{name}'s memwatch block differs from {HARNESSES[0]}'s. One idiom, "
                f"one spelling: divergence here is what #1734 was.")

    # The exact bytes that shipped. Pinning the CODE and not prose about it
    # lets the block beside them explain what they did wrong.
    RETIRED_WRITER = """grep -c 'last=' "$log" 2>/dev/null || echo 0"""
    RETIRED_REDUCER = 'gsub(/memavail_gib=/,"",$4)'

    def test_the_retired_spellings_are_gone_from_every_harness(self) -> None:
        """The two forms that shipped, named so a revert cannot be silent."""
        for name in HARNESSES:
            text = harness(name).read_text()
            with self.subTest(name):
                self.assertNotIn(self.RETIRED_WRITER, text,
                                 "`grep -c` prints its own 0 AND exits 1, so the "
                                 "fallback fires on top of it; that is #1734's writer")
                self.assertNotIn(self.RETIRED_REDUCER, text,
                                 "the positional `$4` reducer is back")

    def test_the_two_instrument_properties_no_behaviour_can_reach(self) -> None:
        """`LC_ALL=C` and `\\b`, HELD AS TEXT, and the difference is stated.

        The `\\b` anchor has a real case beside it -- a record carrying
        `gpu_memavail_gib=` reports the wrong quantity without it -- so this is
        belt over braces there. `LC_ALL=C` has NOTHING else: removing it left
        this suite green, and the defect it prevents cannot be built on this
        box, because no non-C locale is installed and `sort -n` falls back to C.
        A guarantee whose only gate is a text tripwire is worth less than one
        that is executed, and writing that down is the alternative to letting
        the reader assume both are measured.
        """
        block = helper_block(HARNESSES[0])
        self.assertIn("LC_ALL=C sort -n", block,
                      "the reducer's sort is locale-sensitive again: under a locale "
                      "whose thousands separator is `.`, `sort -n` can read 1234.5 "
                      "as 12345, and this instrument reports a low-water mark")
        self.assertIn(r"'\bmemavail_gib=", block,
                      "the reducer's key lost its anchor, so a different key that "
                      "ends in these characters now reads as this one")

    def test_each_harness_calls_both_helpers_exactly_once(self) -> None:
        """A helper nothing calls is a class, not a capability."""
        for name in HARNESSES:
            text = harness(name).read_text()
            with self.subTest(name):
                only(r'^\s*n=\$\(sample_count "\$log"\)\s*$', text,
                     f"{name}'s poll must read its count through sample_count")
                only(r'^\s*echo "  memavail low-water: \$\(memavail_low_water .*$', text,
                     f"{name}'s report must reduce through memavail_low_water")


def swept_files() -> list[Path]:
    """Every shell file this sweep is responsible for.

    `git ls-files` rather than a glob: it is exact, it cannot descend into a
    build directory or `.git`, and it cannot miss `scripts/lmcache/`, `docker/`,
    `tools/bench/` or the extensionless `.githooks/pre-push` the way
    `scripts/*.sh` did. Under the MEMWATCH_SCRIPTS override -- which exists so
    the red-before run can point at an earlier revision -- there is no index to
    ask, so it walks that directory instead.

    TWO LIMITS, BOTH LATENT TODAY. The index is the population, so an UNTRACKED
    file carrying the idiom is invisible until `git add`. The filter is by
    extension plus the `.githooks/` case, so an extensionless shell script
    elsewhere would escape; a shebang scan of the tracked tree finds none.
    """
    if "MEMWATCH_SCRIPTS" in os.environ:
        return sorted(SCRIPTS.rglob("*.sh"))
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # Loud, and carrying git's own words: `check=True` alone reports a
        # status and hides the sentence that says whether this is a missing
        # binary or an archive export with no index.
        raise AssertionError(f"`git ls-files` failed in {ROOT} (rc={r.returncode}), so "
                             f"this sweep has no population: {r.stderr.strip()}")
    out = r.stdout
    names = [n for n in out.split("\0") if n]
    return sorted(ROOT / n for n in names
                  if n.endswith(".sh")
                  or (n.startswith(".githooks/") and not n.endswith(".md")))


class TheIdiomIsGoneFromEveryShellScript(unittest.TestCase):
    """The sweep, not the three files the issue and its triage named.

    #1734 is the FIFTH diagnosis of this one idiom in this tree.
    `scripts/cpu-x86-llamacpp-floor.sh` carries the third in a comment beside
    the removal -- "`pgrep -c` already prints 0 on no match and exits 1, so a
    `|| echo 0` fallback emits \"0\\n0\" and every numeric test that consumes it
    fails" -- and it did not stop the two LTX-2.5 harnesses from shipping it,
    because a comment in one file is not reachable from another.

    The sweep found a THIRD live instance the issue does not mention:
    `scripts/dspark-paired-e2e.sh`'s `settle()`, where the string reached
    `[ "$n" -eq 0 ]` and made the loop's only break unreachable, so a wait for
    the GPU to drain always spent its full 360 s budget
    ([#1791](https://github.com/mudler/vllm.cpp/issues/1791)).

    THIS IS A TRIPWIRE, NOT A PROOF. It reads text, and it catches this
    spelling. `-c` piped into something that swallows the status, or a `wc -l`
    with the same fallback, walks past it. It is here because the alternative
    to a cheap sweep is another comment in a sixth file.
    """

    # `<counting command> ... || echo <n>`, outside a comment. The counting
    # commands are the ones that print a count AND exit non-zero on no match.
    IDIOM = re.compile(r"(grep|pgrep)\s+-[a-zA-Z]*c\b[^|#]*\|\|\s*echo\b")

    def test_the_sweep_reads_something_before_it_reports_nothing(self) -> None:
        """A SWEEP THAT READ NO FILES REPORTS NO OFFENDERS AND EXITS 0.

        Measured: pointing the override at an empty directory made the sweep
        below the one green test in a suite whose other 21 cases fail loudly,
        because it is the only one that finds its input by walking rather than
        by name. The floor is stated as NAMES and not as a count, because a
        count in a gate is a number that drifts against the tree beside it.
        """
        swept = {p.name for p in swept_files()}
        self.assertTrue(swept, "the sweep read no files at all")
        for name in HARNESSES + ("dspark-paired-e2e.sh",):
            self.assertIn(name, swept,
                          f"the sweep did not reach {name}, which is one of the four "
                          f"files #1734 and #1791 were found in")
        if "MEMWATCH_SCRIPTS" in os.environ:
            return
        # AND ONE FILE THAT IS NOT IN `scripts/` AND IS NOT A `.sh`. All four
        # names above are `scripts/*.sh`, so without this the widening from
        # `SCRIPTS.glob("*.sh")` to the whole index is DONE and not GATED:
        # narrowing it back left this suite green. `.githooks/pre-push` is the
        # one file that fails both halves of the old rule, and it is a hook this
        # repository runs on every push.
        self.assertIn("pre-push", swept,
                      "the sweep no longer reaches `.githooks/pre-push`, so it is back "
                      "to reading only `scripts/*.sh` and the idiom can return "
                      "anywhere else in the tree")

    def test_no_shell_script_pairs_a_counting_grep_with_an_echo_fallback(self) -> None:
        offenders = []
        for path in swept_files():
            try:
                body = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(body.splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue  # prose about the idiom is how it gets explained
                if self.IDIOM.search(line):
                    rel = path.relative_to(ROOT) if ROOT in path.parents else path
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertEqual(
            offenders, [],
            "`grep -c`/`pgrep -c` print their own 0 AND exit 1 on no match, so an "
            "`|| echo` fallback runs ON TOP of that count and yields a two-line "
            "string. Every numeric test that consumes it errors rather than "
            "deciding, and every record that embeds it splits in two.\n  "
            + "\n  ".join(offenders))


class SampleCount(unittest.TestCase):
    """ONE integer, on ONE line, for every log the poll can be handed.

    All three harnesses put it in the tab-separated watch record, where a
    newline splits the record in two. TWO of the three also feed it to the
    sample cap's `[ "${n:-0}" -ge "$WANT_SAMPLES" ]`, which decides whether to
    SIGINT a render on a shared box and which a non-integer makes error rather
    than decide. The pixel harness has no cap: it renders to completion.
    """

    def _count(self, path: str) -> subprocess.CompletedProcess:
        return bash(f'sample_count "{path}"')

    def test_a_log_with_no_match_is_a_single_zero(self) -> None:
        """THE #1734 CASE. `grep -c` prints 0 and exits 1 on no match."""
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "engine.log"
            log.write_text("loading weights\nallocating kv\n")
            p = self._count(str(log))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout, "0\n",
                         "a zero count must be ONE line; two is the defect")

    def test_it_counts_the_matches_it_has(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "engine.log"
            log.write_text("".join(f"forward {i} last={i}.5s\n" for i in range(7)))
            p = self._count(str(log))
        self.assertEqual(p.stdout, "7\n")

    def test_an_absent_log_is_a_single_zero(self) -> None:
        """The first poll can fire before the engine has opened its log."""
        p = self._count("/nonexistent/engine.log")
        self.assertEqual(p.stdout, "0\n")

    def test_a_directory_is_a_single_zero(self) -> None:
        """GNU grep 3.11 answers a directory with `0` and status 2.

        THE FIRST VERSION OF THIS DOCSTRING SAID `grep` DISAGREED WITH ITSELF
        ACROSS TWO IMPLEMENTATIONS, and that was an instrument artifact: the
        "second grep" was a shell FUNCTION the authoring agent's terminal
        installs, not a binary. There is no `ugrep` on this box, and every
        `bash -c` -- which is how both this suite and the harnesses invoke it --
        resolves `/usr/bin/grep`. The case is kept because the status is 2 and a
        caller that read the status rather than the output would decide wrongly;
        the floor makes the output an integer regardless.
        """
        with tempfile.TemporaryDirectory() as t:
            p = self._count(t)
        self.assertEqual(p.stdout, "0\n")

    def test_an_unreadable_log_is_a_single_zero(self) -> None:
        """THE CASE THAT REALLY PRINTS NOTHING, and the one the floor is for.

        `grep -c` on a file it cannot open writes only to stderr and exits 2, so
        `$n` is the empty string. `head -1` cannot help here: there is no line
        to take. Deleting the `case` leaves this returning empty into an integer
        test. Skipped under a uid that can read anything.
        """
        if os.geteuid() == 0:
            self.skipTest("root reads a 000 file, so this case cannot be built here")
        with tempfile.TemporaryDirectory() as t:
            log = Path(t) / "engine.log"
            log.write_text("forward 0 last=1.0s\n")
            log.chmod(0o000)
            try:
                p = self._count(str(log))
            finally:
                log.chmod(0o644)
        self.assertEqual(p.stdout, "0\n")

    # The cap's own test, lifted out of the harness rather than retyped. A copy
    # here would go on passing after the real line changed.
    CAP = re.compile(r'^\s*if (\[ .*-ge "\$WANT_SAMPLES" \]); then stopped_by="sample-cap"',
                     re.M)

    def test_its_output_survives_the_integer_test_the_cap_makes(self) -> None:
        """The second consequence #1734 names: `[ "0\\n0" -ge 12 ]` does not
        return false, it returns 2 with `integer expression expected` on stderr.
        A guard whose job is to SIGINT a render on a shared box must not be able
        to error out instead of deciding.

        ONLY TWO OF THE THREE HARNESSES HAVE A SAMPLE CAP. The pixel harness
        renders to completion on purpose and stops on the memory floor or the
        timeout, so the cap is not a property of the shared block; it is a
        property of the two harnesses that reduce a median. This asserts which
        is which, so the asymmetry cannot be read as one of them having lost it.
        """
        capped = {n: self.CAP.findall(harness(n).read_text()) for n in HARNESSES}
        self.assertEqual([n for n, c in capped.items() if len(c) == 1],
                         list(HARNESSES[:2]),
                         f"the sample cap moved between harnesses: "
                         f"{ {n: len(c) for n, c in capped.items()} }")
        self.assertEqual(capped[HARNESSES[2]], [],
                         "the pixel harness gained a sample cap; it renders to "
                         "completion on purpose")

        for name in HARNESSES[:2]:
            test = capped[name][0]
            for content, expect in (("", "GO"), ("last=1s\n" * 20, "CAP")):
                with tempfile.TemporaryDirectory() as t:
                    log = Path(t) / "engine.log"
                    log.write_text(content)
                    p = bash(f'WANT_SAMPLES=12; n=$(sample_count "{log}"); '
                             f'if {test}; then echo CAP; else echo GO; fi', name=name)
                with self.subTest(harness=name, content=repr(content[:12])):
                    self.assertEqual(p.stderr, "",
                                     f"{name}'s cap errored instead of deciding")
                    self.assertEqual(p.returncode, 0)
                    self.assertEqual(p.stdout.strip(), expect)


class MemavailLowWater(unittest.TestCase):
    """The reducer is KEYED, and it can say that it read nothing."""

    def _tsv(self, root: Path, lines: list[str]) -> Path:
        p = root / "watch.tsv"
        p.write_text("".join(l + "\n" for l in lines))
        return p

    def _low(self, path: Path | str) -> str:
        p = bash(f'memavail_low_water "{path}"')
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout.strip()

    def test_it_reduces_a_well_formed_file(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), [
                "21:10:09\tflash\tsamples=3\tmemavail_gib=115.1",
                "21:10:14\tflash\tsamples=5\tmemavail_gib=40.3",
                "21:10:19\tflash\tsamples=8\tmemavail_gib=88.6",
            ])
            self.assertEqual(self._low(tsv), "40.3 GiB")

    def test_it_reduces_the_malformed_file_the_defect_produced(self) -> None:
        """THE SECOND GUARANTEE, AND IT IS INDEPENDENT OF THE FIRST. These are
        the bytes off the 20260822T203535Z run: 85 records split in two by the
        embedded newline, so `memavail_gib=` lands in field 2. The writer is
        repaired, and the reducer must still not be the reason a repaired writer
        was needed to read a file at all."""
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), [
                "21:10:09\tflash\tsamples=0",
                "0\tmemavail_gib=115.1",
                "21:10:14\tflash\tsamples=0",
                "0\tmemavail_gib=40.3",
                "21:16:22\tflash\tsamples=1\tmemavail_gib=88.6",
            ])
            self.assertEqual(self._low(tsv), "40.3 GiB",
                             "issue #1734 re-derived 40.3 GiB from exactly this shape")

    def test_it_does_not_depend_on_the_field_it_lands_in(self) -> None:
        """The positional reducer read `$4` and stripped the prefix off `$4`
        alone. A record that gains a column is not a reason to stop reporting
        memory."""
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), [
                "memavail_gib=51.2\t21:10:09\tflash\tsamples=3\tgpu_gib=60.0",
                "21:10:14\tflash\tmemavail_gib=12.5\tsamples=5\tgpu_gib=60.0",
            ])
            self.assertEqual(self._low(tsv), "12.5 GiB")

    def test_a_different_key_that_ends_in_this_one_is_not_this_one(self) -> None:
        """`memavail_gib=` unanchored also matches inside `gpu_memavail_gib=`.

        No harness writes that key today, so this pins the anchor rather than a
        defect that shipped. It is cheap and the docstring beside the function
        says "MATCHED ON ITS KEY", which is only true with the `\\b`.
        """
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), [
                "a\tb\tgpu_memavail_gib=2.0\tmemavail_gib=99.0",
                "a\tb\tgpu_memavail_gib=1.0\tmemavail_gib=40.3",
            ])
            self.assertEqual(self._low(tsv), "40.3 GiB")

    def test_it_sorts_numerically_and_not_as_text(self) -> None:
        """`sort` without `-n` puts 100.0 below 40.3, which is a low-water mark
        that reads three times better than the box ever was."""
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), [
                "a\tb\tc\tmemavail_gib=100.0",
                "a\tb\tc\tmemavail_gib=40.3",
                "a\tb\tc\tmemavail_gib=9.5",
            ])
            self.assertEqual(self._low(tsv), "9.5 GiB")

    def test_an_integer_reading_is_still_a_reading(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), ["a\tb\tc\tmemavail_gib=7"])
            self.assertEqual(self._low(tsv), "7 GiB")

    def test_a_file_with_no_reading_says_so_rather_than_printing_nothing(self) -> None:
        """THE SYMPTOM, AT ITS ROOT. A blank where a number belongs reads as
        "measured, and fine". It must be impossible for this to emit a bare
        unit, whatever it is handed."""
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), ["21:10:09\tflash\tsamples=0"])
            self.assertEqual(self._low(tsv), "NO READINGS")

    def test_an_empty_file_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._low(self._tsv(Path(t), [])), "NO READINGS")

    def test_an_absent_file_says_so(self) -> None:
        """An arm killed before its first poll leaves no watch file at all."""
        self.assertEqual(self._low("/nonexistent/watch.tsv"), "NO READINGS")

    def test_a_truncated_key_is_not_a_reading(self) -> None:
        """`memavail_gib=` with nothing after it is a poll that did not read,
        and reporting it as 0 would trip a reader into thinking the box filled."""
        with tempfile.TemporaryDirectory() as t:
            tsv = self._tsv(Path(t), ["a\tb\tc\tmemavail_gib="])
            self.assertEqual(self._low(tsv), "NO READINGS")


class ThePollAndTheReportAsCommitted(unittest.TestCase):
    """The end-to-end #1734 reproduction, over the harnesses' OWN lines.

    The poll body and the report line are pulled out of each committed harness
    by unique regex and executed here. Nothing below is a copy of them, so a
    harness that reverts to the old writer goes red on the symptom itself --
    a blank field -- and not merely on a text tripwire.

    The shape is the run that filed the issue: three polls during the ~7 minutes
    of model load, before the engine has emitted a single `last=` line, then two
    polls once samples exist.
    """

    # `n=...`, the MemAvailable read, and the record write: three consecutive
    # lines in every one of the three harnesses.
    POLL = r'^(\s*n=\$\(sample_count "\$log"\)\n.*\n.*memavail_gib=\$avail".*)$'
    REPORT = r'^(\s*echo "  memavail low-water: \$\(memavail_low_water .*)$'

    def _run(self, name: str) -> subprocess.CompletedProcess:
        text = harness(name).read_text()
        poll = only(self.POLL, text, f"{name}'s poll body")
        report = only(self.REPORT, text, f"{name}'s report line")
        self.assertIn("sample_count", poll)
        self.assertIn("memavail_gib=$avail", poll)

        t = tempfile.mkdtemp()
        meminfo = Path(t) / "meminfo"
        # 42257613 kB is 40.3 GiB: the value issue #1734 re-derived by hand from
        # the run's own watch files, so a pass here reproduces that number.
        meminfo.write_text("MemTotal:       124680000 kB\nMemAvailable:    42257613 kB\n")
        poll = poll.replace("/proc/meminfo", str(meminfo))

        # `mem_avail_gib` is pixel-ab's own reader, with its own suite; take it
        # from that harness rather than stubbing the thing under test.
        pre = ""
        pix = harness(name).read_text()
        if PIX_BEGIN in pix:
            pre = pix.split(PIX_BEGIN, 1)[1].split(PIX_END, 1)[0] + "\n"

        script = f'''
OUT={t}; d={t}; label=arm; log={t}/engine.log
{pre}
poll() {{
  local n avail
{poll}
}}
: > "$log"
poll; poll; poll
echo "step 0 last=6.236s" >> "$log"
poll; poll
tsvs=""
for f in "$OUT"/watch*.tsv; do [ -f "$f" ] && tsvs="$tsvs $f"; done
echo "---TSV---"
cat $tsvs
echo "---NF---"
awk -F'\\t' '{{print NF}}' $tsvs | sort -u | tr '\\n' ' '
echo
echo "---REPORT---"
{report}
'''
        return subprocess.run(["bash", "-c", "set -u\n" + helper_block(name) + "\n" + script],
                              capture_output=True, text=True, env={**os.environ,
                                                                   "MEMINFO": str(meminfo),
                                                                   "PATH": os.environ["PATH"]})

    def test_every_record_is_one_line_of_four_fields(self) -> None:
        for name in HARNESSES:
            p = self._run(name)
            with self.subTest(name):
                self.assertEqual(p.returncode, 0, p.stderr)
                nf = p.stdout.split("---NF---", 1)[1].split("\n", 2)[1].strip()
                self.assertEqual(nf, "4",
                                 f"{name} wrote a record that is not four tab-separated "
                                 f"fields; field counts seen: {nf!r}. #1734's run had "
                                 f"85 NF=3 and 85 NF=2 records against 16 good ones.")

    def test_the_report_prints_the_number_that_was_measured(self) -> None:
        for name in HARNESSES:
            p = self._run(name)
            report = p.stdout.split("---REPORT---", 1)[1].strip()
            with self.subTest(name):
                self.assertEqual(
                    report.splitlines()[0].strip(), "memavail low-water: 40.3 GiB",
                    f"{name} printed {report!r}. The run that filed #1734 printed\n"
                    f"  `  memavail low-water:  GiB`\nover watch files holding 40.3.")

    def test_the_report_can_never_print_a_bare_unit(self) -> None:
        """The exact bytes the defect emitted, refused by name."""
        for name in HARNESSES:
            p = self._run(name)
            report = p.stdout.split("---REPORT---", 1)[1]
            with self.subTest(name):
                self.assertNotIn("memavail low-water:  GiB", report)


if __name__ == "__main__":
    unittest.main()
