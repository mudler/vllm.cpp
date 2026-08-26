#!/usr/bin/env python3
"""The tower-skip RSS reporter, on fabricated inputs, across every verdict.

`scripts/mm/tower_skip_rss.sh` (#607 L3) cannot be exercised by CI: it needs a
56 G or an 8.3 G checkpoint, two Release builds and a quiet box. Its REPORTER
needs none of that -- it reads four `/usr/bin/time -v` files and four server
logs -- so the arithmetic, the VOID conditions and the pass/fail boundary are
gated here on synthetic inputs, for both declared model kinds.

Three properties this suite exists for, each of which was once absent:

* **Both pairs reach the verdict.** The run swaps the arm-to-binary assignment
  between its two pairs (A-B then B-A) so a binary-shaped bias `d` cannot
  masquerade as a saving. Until #607's L3 review the reporter read pair 1 only
  and exited on it, so the decorrelation was performed and thrown away.
  `PairDisagreementTests` builds exactly that input -- pair 1 `true + d` MET,
  pair 2 `true - d` FAILING -- and requires a FAILING verdict.
* **VOID outranks FAILING.** An unmeasured pair is not a measurement of a small
  saving, and a run with no second pair at all (the pre-repair on-disk shape) is
  VOID rather than a pass.
* **The threshold is per model.** Muse Glimmer's 7.161 GiB does not describe
  Qwen3-VL's 1.547 GiB tower, so a missing kind is REFUSED rather than
  defaulted, and the two declared byte counts are pinned against the spec that
  declares them.

One more, added after the harness shipped unable to build the binary it
measures:

* **The plan can produce a binary.** Everything above reads finished logs, so
  none of it could see that the script configured with
  `-DVLLM_CPP_BUILD_EXAMPLES=OFF` and then asked ninja for `vllm-server`, an
  `examples/` target -- `unknown target`, exit 4, no RSS, and 41/41 green.
  `--dry-run` prints the cmake/ninja/`run_arm` invocations the run would issue
  and asserts that CMake defines that target under those flags;
  `DryRunTests` runs it against the script AS COMMITTED, which is the case that
  reds on the defect, and against a scratch copy with the flag flipped back to
  OFF, which is the mutation proving the assertion has teeth.

Two more, added when the leased worker turned out not to be able to see
`/mnt/nas_share/checkpoints` at all:

* **The source must be local disk.** `--check-source` is the refusal on its own,
  so the `/workspace` prong is gateable without a lease and the CIFS prong is
  gateable on any host that has the NAS mounted -- skipped BY NAME where it does
  not, because an absent mount verifies nothing.
* **The copy is checked by its POSTCONDITION.** `--stage-check` compares the two
  trees file by file. `cp`'s exit status is not evidence: on this fleet a
  missing wrapper binary has already made a copy command print success and move
  nothing, which is the `EmptyDestination` case below.

Nothing here runs a server, a build, or a model.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/mm/tower_skip_rss.sh"
SPEC = ROOT / ".agents/specs/multimodal-track.md"

EXIT_MET = 0
EXIT_FAILING = 1
EXIT_ARGS = 2
EXIT_VOID = 3
EXIT_BUILD = 4
EXIT_STAGE = 6
EXIT_NOT_LOCAL = 7

# A CIFS/SMB mount to test the filesystem-type prong against. Absent on a box
# without the NAS, and the case is then SKIPPED by name rather than passing.
CIFS_CANDIDATES = ("/mnt/nas_share", "/mnt/media")

# The declarations under test, repeated here so a silent edit to either the
# script or the spec reds this suite instead of passing quietly. The key is the
# `--model-kind`; the value is the shell variable that carries it and the byte
# count, because asserting the NUMBER alone is not asserting the CONSTANT --
# `830695424` also appears twice in the header prose, so a substring search over
# the whole file stays green while the load-bearing assignment drifts.
ONDISK = {
    "muse-glimmer": ("MUSE_GLIMMER_TOWER_ONDISK_BYTES", 3843691520),
    "qwen3-vl": ("QWEN3_VL_TOWER_ONDISK_BYTES", 830695424),
}
FRACTION_PCT = 90

# The other two declared numbers, in the same shape. `MIN_SAVING_FRACTION_PCT`
# is also gated behaviourally, through every verdict below; `DEFAULT_ARM_DRIFT_PCT`
# is only printed, so this assertion is the whole of its gate.
SHARED_DECLARATIONS = {
    "MIN_SAVING_FRACTION_PCT": FRACTION_PCT,
    "DEFAULT_ARM_DRIFT_PCT": 2,
}

# The build the measurement runs, declared in the script and asserted here. The
# target is an `examples/` one, so the configure flags must turn `examples/` on.
NINJA_TARGET = "vllm-server"
EXAMPLES_FLAG = "-DVLLM_CPP_BUILD_EXAMPLES=ON"
SERVER_RELPATH = "examples/vllm-server"

SKIP_LINE = (
    "server: multimodal towers NOT loaded (every modality they serve is at "
    "limit 0): vision_tower\n"
)


def ondisk(kind: str) -> int:
    return ONDISK[kind][1]


def resident(kind: str) -> int:
    """The loader widens bf16 -> host f32 on both kinds (#1359)."""

    return ondisk(kind) * 2


def need(kind: str) -> int:
    """Integer arithmetic, matching the shell's `$(( a * pct / 100 ))`."""

    return resident(kind) * FRACTION_PCT // 100


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class Run:
    """A fabricated run directory: four `.time` files and four `.log` files."""

    def __init__(self, root: Path, kind: str) -> None:
        self.dir = root
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "model-kind").write_text(kind + "\n", encoding="utf-8")

    def pair(
        self,
        dtag: str,
        ltag: str,
        default_bytes: int,
        saving_bytes: int,
        *,
        skip_line_on_lmo: bool = True,
        skip_line_on_default: bool = False,
        omit_time_line: bool = False,
    ) -> None:
        lmo_bytes = default_bytes - saving_bytes
        self._time(dtag, default_bytes, omit=False)
        self._time(ltag, lmo_bytes, omit=omit_time_line)
        self._log(dtag, skip_line_on_default)
        self._log(ltag, skip_line_on_lmo)

    def _time(self, tag: str, size_bytes: int, *, omit: bool) -> None:
        body = [
            "\tCommand being timed: \"vllm-server\"\n",
            "\tUser time (seconds): 12.34\n",
        ]
        if not omit:
            # `/usr/bin/time -v` reports KILOBYTES; the reporter multiplies by
            # 1024, so a byte figure that is not a whole number of kilobytes
            # cannot be expressed and is not fabricated here.
            assert size_bytes % 1024 == 0, "fabricate whole kilobytes"
            body.append(
                "\tMaximum resident set size (kbytes): %d\n" % (size_bytes // 1024)
            )
        body.append("\tExit status: 0\n")
        (self.dir / (tag + ".time")).write_text("".join(body), encoding="utf-8")

    def _log(self, tag: str, skip_line: bool) -> None:
        body = "server: prefix caching disabled\n"
        if skip_line:
            body += SKIP_LINE
        body += "server: listening on 127.0.0.1:18607\n"
        (self.dir / (tag + ".log")).write_text(body, encoding="utf-8")

    def warmup(self, size_bytes: int) -> None:
        """The discarded page-cache leg.

        Same binary and same arm as `default`, run first and thrown away, so its
        peak RSS is one repeat of that cell rather than a third arm.
        """

        self._time("warmup", size_bytes, omit=False)
        self._log("warmup", False)

    def report(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return run("--report-only", str(self.dir), *extra)


def kb(n: int) -> int:
    """Round a byte count DOWN to a whole kilobyte, so it is expressible."""

    return (n // 1024) * 1024


class VerdictTests(unittest.TestCase):
    """MET, FAILING and both VOID shapes, for each declared kind."""

    def _met(self, kind: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind) + 8 * 1024**2)
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_MET, out.stdout + out.stderr)
            self.assertIn("RESULT: MET (first half, BOTH pairs)", out.stdout)
            self.assertIn("mean (estimator", out.stdout)

    def _failing(self, kind: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind) - 8 * 1024**2)
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_FAILING, out.stdout + out.stderr)
            self.assertIn("RESULT: FAILING", out.stdout)
            self.assertIn("Do not renegotiate the threshold", out.stdout)

    def _void_no_announcement(self, kind: str) -> None:
        """The skipping arm never SAID it skipped: measured, but not this change."""

        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind) + 8 * 1024**2)
            r.pair("default", "lmo", base, saving, skip_line_on_lmo=False)
            r.pair("default2", "lmo2", base, saving)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_VOID, out.stdout + out.stderr)
            self.assertIn("never reported a skipped tower", out.stdout)
            self.assertIn("RESULT: VOID — a pair is unmeasured", out.stdout)

    def _void_both_arms_skipped(self, kind: str) -> None:
        """Both arms skipped, so the difference between them is not the skip."""

        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind) + 8 * 1024**2)
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving, skip_line_on_default=True)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_VOID, out.stdout + out.stderr)
            self.assertIn("reported a skipped tower. The two arms", out.stdout)

    def _void_missing_instrument(self, kind: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind) + 8 * 1024**2)
            r.pair("default", "lmo", base, saving, omit_time_line=True)
            r.pair("default2", "lmo2", base, saving)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_VOID, out.stdout + out.stderr)
            self.assertIn("A missing instrument line is not a measurement", out.stdout)

    def _void_no_second_pair(self, kind: str) -> None:
        """The pre-repair on-disk shape: two logs, no swapped pair. Not a pass."""

        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind) + 8 * 1024**2)
            r.pair("default", "lmo", base, saving)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_VOID, out.stdout + out.stderr)
            self.assertIn("default2.time", out.stdout)

    def test_muse_glimmer_met(self) -> None:
        self._met("muse-glimmer")

    def test_muse_glimmer_failing(self) -> None:
        self._failing("muse-glimmer")

    def test_muse_glimmer_void_no_announcement(self) -> None:
        self._void_no_announcement("muse-glimmer")

    def test_muse_glimmer_void_both_arms_skipped(self) -> None:
        self._void_both_arms_skipped("muse-glimmer")

    def test_muse_glimmer_void_missing_instrument(self) -> None:
        self._void_missing_instrument("muse-glimmer")

    def test_muse_glimmer_void_no_second_pair(self) -> None:
        self._void_no_second_pair("muse-glimmer")

    def test_qwen3_vl_met(self) -> None:
        self._met("qwen3-vl")

    def test_qwen3_vl_failing(self) -> None:
        self._failing("qwen3-vl")

    def test_qwen3_vl_void_no_announcement(self) -> None:
        self._void_no_announcement("qwen3-vl")

    def test_qwen3_vl_void_both_arms_skipped(self) -> None:
        self._void_both_arms_skipped("qwen3-vl")

    def test_qwen3_vl_void_missing_instrument(self) -> None:
        self._void_missing_instrument("qwen3-vl")

    def test_qwen3_vl_void_no_second_pair(self) -> None:
        self._void_no_second_pair("qwen3-vl")


def kb_up(n: int) -> int:
    """Round a byte count UP to a whole kilobyte, so it is expressible."""

    return -(-n // 1024) * 1024


class BoundaryTests(unittest.TestCase):
    """The pass/fail boundary is `>=`, and equality is tested as equality.

    A saving is `peak(default) - peak(lmo)`, and `/usr/bin/time -v` reports
    kilobytes, so every expressible saving is a multiple of 1024. Whether the
    threshold itself can be HIT therefore depends on the kind:

    * `need("muse-glimmer") == 6918644736 == 6756489 KiB` exactly, so a saving
      equal to the threshold exists and `-ge` vs `-gt` is a real distinction
      this suite can make. It is made below, and it is the only case that reds
      when the comparison is loosened.
    * `need("qwen3-vl") == 1495251763` is not a whole kilobyte, so no run can
      land on it. The tightest expressible pair straddling it is asserted
      instead -- the next kilobyte up is MET, the next one down is FAILING --
      which is the strongest statement that input can support.

    An earlier version of this class added a kilobyte to a rounded-down
    threshold on both kinds, so its "exactly at the threshold" case sat ABOVE
    the threshold and `-ge` -> `-gt` left the whole suite green.
    """

    def _verdict(self, kind: str, saving: int) -> int:
        self.assertEqual(saving % 1024, 0, "an unexpressible saving cannot be run")
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving)
            return r.report().returncode

    def test_a_saving_equal_to_the_threshold_is_met(self) -> None:
        kind = "muse-glimmer"
        self.assertEqual(need(kind) % 1024, 0, "this kind's threshold is hittable")
        self.assertEqual(self._verdict(kind, need(kind)), EXIT_MET)

    def test_one_kilobyte_below_an_exactly_hittable_threshold_is_failing(self) -> None:
        kind = "muse-glimmer"
        self.assertEqual(self._verdict(kind, need(kind) - 1024), EXIT_FAILING)

    def test_the_unhittable_threshold_is_straddled_as_tightly_as_it_can_be(self) -> None:
        kind = "qwen3-vl"
        self.assertNotEqual(need(kind) % 1024, 0, "this kind's threshold is not hittable")
        self.assertEqual(self._verdict(kind, kb_up(need(kind))), EXIT_MET)
        self.assertEqual(self._verdict(kind, kb(need(kind))), EXIT_FAILING)


class PairDisagreementTests(unittest.TestCase):
    """A binary-shaped bias `d` must reach the verdict, not be reported beside it.

    The input is the one the swap exists to catch: `true == need`, and a bias
    `d` that pair 1 adds and pair 2 subtracts. Pair 1 alone is MET -- asserted
    below, because that is exactly what made the pre-repair reporter exit 0 --
    and the decorrelated verdict is FAILING.
    """

    def _biased(self, kind: str, tmp: Path) -> Run:
        r = Run(tmp / "run", kind)
        base = kb(40 * 1024**3)
        bias = kb(need(kind) // 5)
        r.pair("default", "lmo", base, kb(need(kind)) + bias)
        r.pair("default2", "lmo2", base, kb(need(kind)) - bias)
        return r

    def test_pair_one_alone_would_have_passed(self) -> None:
        """The premise: pair 1 of this run meets the threshold on its own."""

        for kind in ONDISK:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                r = self._biased(kind, Path(tmp))
                # Copy pair 1 over pair 2, which is what "read pair 1 and exit"
                # amounts to: a reporter that sees only the first assignment.
                for src, dst in (("default", "default2"), ("lmo", "lmo2")):
                    for ext in (".time", ".log"):
                        (r.dir / (dst + ext)).write_text(
                            (r.dir / (src + ext)).read_text(encoding="utf-8"),
                            encoding="utf-8",
                        )
                out = r.report()
                self.assertEqual(out.returncode, EXIT_MET, out.stdout + out.stderr)

    def test_disagreeing_pairs_do_not_silently_pass(self) -> None:
        for kind in ONDISK:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                r = self._biased(kind, Path(tmp))
                out = r.report()
                self.assertEqual(out.returncode, EXIT_FAILING, out.stdout + out.stderr)
                self.assertIn("pair 2 FAILING", out.stdout)
                self.assertIn("The two pairs DISAGREE", out.stdout)
                self.assertIn("binary-shaped bias", out.stdout)

    def test_the_mean_is_reported_and_is_not_the_gate(self) -> None:
        """The mean cancels `d` exactly; the verdict is still FAILING."""

        kind = "qwen3-vl"
        with tempfile.TemporaryDirectory() as tmp:
            r = self._biased(kind, Path(tmp))
            out = r.report()
            self.assertEqual(out.returncode, EXIT_FAILING, out.stdout + out.stderr)
            mean_line = [ln for ln in out.stdout.splitlines() if "mean (estimator" in ln]
            self.assertEqual(len(mean_line), 1, out.stdout)
            self.assertIn(str(kb(need(kind))), mean_line[0])


class KindResolutionTests(unittest.TestCase):
    """The threshold is per model, so a missing kind is refused, never defaulted."""

    def test_unknown_kind_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = run("--report-only", tmp, "--model-kind", "qwen9-vl")
            self.assertEqual(out.returncode, EXIT_ARGS, out.stdout + out.stderr)
            self.assertIn("unknown --model-kind", out.stderr)

    def test_report_only_without_a_kind_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = run("--report-only", tmp)
            self.assertEqual(out.returncode, EXIT_ARGS, out.stdout + out.stderr)
            self.assertIn("holds no 'model-kind' file", out.stderr)

    def test_the_kind_selects_the_threshold(self) -> None:
        """One run directory, two kinds, two verdicts.

        The saving is above Qwen3-VL's threshold and far below Muse Glimmer's,
        so reading it under the wrong kind is a wrong verdict rather than a
        rounding difference. That is why there is no default.
        """

        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", "qwen3-vl")
            base = kb(40 * 1024**3)
            saving = kb(need("qwen3-vl") + 8 * 1024**2)
            self.assertLess(saving, need("muse-glimmer"))
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving)
            self.assertEqual(r.report().returncode, EXIT_MET)
            self.assertEqual(
                r.report("--model-kind", "muse-glimmer").returncode, EXIT_FAILING
            )

    def test_an_explicit_kind_overrides_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", "muse-glimmer")
            out = r.report("--model-kind", "qwen3-vl")
            self.assertIn("model kind      qwen3-vl", out.stdout)


class DeclarationTests(unittest.TestCase):
    """Script and spec declare the same numbers, because both are required to.

    The script's assertions are anchored on the ASSIGNMENT LINE. A whole-file
    substring search for the digits is not a check on the constant: both on-disk
    figures also appear in the header prose that explains them, so
    `QWEN3_VL_TOWER_ONDISK_BYTES=830695425` -- one byte out, and every threshold
    derived from it wrong -- left this suite 41/41 green. The drift window that
    opened was roughly -1100 .. +280 B, because only a change large enough to
    move a printed GiB figure by a displayed digit would have been noticed.

    The spec is prose and carries no assignment, so its figures stay substring
    assertions: there the number in the sentence IS the declaration.
    """

    def _assignment(self, name: str, value: int) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        hits = re.findall(r"(?m)^%s=(\d+)" % re.escape(name), text)
        self.assertEqual(
            hits,
            [str(value)],
            "expected exactly one '%s=%d' assignment line in %s, found %r"
            % (name, value, SCRIPT.name, hits),
        )

    def test_script_carries_both_on_disk_figures(self) -> None:
        for kind, (name, value) in ONDISK.items():
            with self.subTest(kind=kind):
                self._assignment(name, value)

    def test_script_carries_the_shared_declarations(self) -> None:
        for name, value in SHARED_DECLARATIONS.items():
            with self.subTest(declaration=name):
                self._assignment(name, value)

    def test_spec_carries_both_on_disk_figures(self) -> None:
        text = SPEC.read_text(encoding="utf-8")
        for kind, (_name, value) in ONDISK.items():
            with self.subTest(kind=kind):
                self.assertIn(str(value), text)

    def test_spec_carries_both_resident_figures(self) -> None:
        text = SPEC.read_text(encoding="utf-8")
        for kind in ONDISK:
            with self.subTest(kind=kind):
                self.assertIn(str(resident(kind)), text)

    def test_the_widening_defect_is_named_where_the_thresholds_are(self) -> None:
        """A large saving is partly a large widening; both files must say so."""

        self.assertIn("#1359", SCRIPT.read_text(encoding="utf-8"))
        self.assertIn("1359", SPEC.read_text(encoding="utf-8"))


def _fstype(path: str) -> str:
    out = subprocess.run(
        ["stat", "-f", "-c", "%T", path], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else ""


class LocalSourceRefusalTests(unittest.TestCase):
    """Weights are read from local disk, or the run refuses to start."""

    def test_a_workspace_path_is_refused(self) -> None:
        out = run("--check-source", "/workspace/ckpt/muse-glimmer-30b")
        self.assertEqual(out.returncode, EXIT_NOT_LOCAL, out.stdout + out.stderr)
        self.assertIn("leased worker's CIFS mount", out.stderr)

    def test_the_workspace_root_itself_is_refused(self) -> None:
        out = run("--check-source", "/workspace")
        self.assertEqual(out.returncode, EXIT_NOT_LOCAL, out.stdout + out.stderr)

    def test_a_local_path_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = run("--check-source", tmp)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def test_a_path_that_merely_mentions_workspace_is_accepted(self) -> None:
        """The prong is a path PREFIX, not a substring: /tmp/workspace is local."""

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "workspace" / "ckpt"
            local.mkdir(parents=True)
            out = run("--check-source", str(local))
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)

    def _a_cifs_mount(self) -> str:
        for candidate in CIFS_CANDIDATES:
            if _fstype(candidate) in ("cifs", "smb", "smb2", "smb3"):
                return candidate
        self.skipTest(
            "no CIFS/SMB mount among %s on this host, so the filesystem-type "
            "prong is UNVERIFIED here rather than passing" % (CIFS_CANDIDATES,)
        )
        raise AssertionError("unreachable")

    def test_a_cifs_path_is_refused(self) -> None:
        out = run("--check-source", self._a_cifs_mount())
        self.assertEqual(out.returncode, EXIT_NOT_LOCAL, out.stdout + out.stderr)
        self.assertIn("filesystem", out.stderr)

    def test_a_not_yet_existing_path_on_a_cifs_mount_is_refused(self) -> None:
        """`stat -f` cannot answer for a path that is not there yet.

        It failed, `fstype` came out empty, and the path was ACCEPTED. Measured
        before the repair: `--check-source /mnt/nas_share` returned 7 while
        `--check-source /mnt/nas_share/no-such-dir` returned 0. This is not a
        corner: `check_source_is_local` is what clears `--stage-to`, and
        `--stage-to` NAMES A DIRECTORY THE RUN IS ABOUT TO CREATE, so the guard
        was blind in the one case it exists for -- a staging directory on the
        NAS would have been accepted, created, and the checkpoint copied onto
        CIFS. The filesystem of a path that does not exist is the filesystem of
        its nearest existing ancestor.
        """

        mount = self._a_cifs_mount()
        absent = os.path.join(mount, "no-such-dir-tower-skip-rss-gate")
        self.assertFalse(os.path.exists(absent), "fixture path must not exist")
        out = run("--check-source", absent)
        self.assertEqual(out.returncode, EXIT_NOT_LOCAL, out.stdout + out.stderr)
        self.assertIn("does not exist yet", out.stderr)
        self.assertIn(mount, out.stderr)

    def test_a_not_yet_existing_local_path_is_still_accepted(self) -> None:
        """The ancestor walk must not turn every absent path into a refusal."""

        with tempfile.TemporaryDirectory() as tmp:
            absent = Path(tmp) / "not" / "there" / "yet"
            out = run("--check-source", str(absent))
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


class StagePostconditionTests(unittest.TestCase):
    """The copy is judged by what arrived, never by what `cp` returned."""

    def _tree(self, root: Path, files: dict[str, int]) -> Path:
        for rel, size in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * size)
        return root

    def _check(self, src: Path, dst: Path) -> subprocess.CompletedProcess[str]:
        return run("--stage-check", str(src), str(dst))

    def test_an_identical_tree_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = {"config.json": 1200, "a/model-00001.safetensors": 4096}
            src = self._tree(Path(tmp) / "src", files)
            dst = self._tree(Path(tmp) / "dst", files)
            out = self._check(src, dst)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("every relative path and byte size matches", out.stdout)

    def test_an_empty_destination_fails(self) -> None:
        """The wrapper-binary case: the copy printed success and moved nothing."""

        with tempfile.TemporaryDirectory() as tmp:
            src = self._tree(Path(tmp) / "src", {"config.json": 1200})
            dst = Path(tmp) / "dst"
            dst.mkdir()
            out = self._check(src, dst)
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)
            self.assertIn("the two trees differ", out.stderr)

    def test_a_truncated_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._tree(Path(tmp) / "src", {"m.safetensors": 8192})
            dst = self._tree(Path(tmp) / "dst", {"m.safetensors": 8191})
            out = self._check(src, dst)
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)

    def test_a_missing_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._tree(
                Path(tmp) / "src", {"a.safetensors": 4096, "b.safetensors": 4096}
            )
            dst = self._tree(Path(tmp) / "dst", {"a.safetensors": 4096})
            out = self._check(src, dst)
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)

    def test_an_extra_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._tree(Path(tmp) / "src", {"a.safetensors": 4096})
            dst = self._tree(
                Path(tmp) / "dst", {"a.safetensors": 4096, "stray.bin": 16}
            )
            out = self._check(src, dst)
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)

    def test_a_file_at_the_wrong_relative_path_fails(self) -> None:
        """Same name, same size, wrong place. A total-bytes check would pass it."""

        with tempfile.TemporaryDirectory() as tmp:
            src = self._tree(Path(tmp) / "src", {"shards/a.safetensors": 4096})
            dst = self._tree(Path(tmp) / "dst", {"a.safetensors": 4096})
            out = self._check(src, dst)
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)

    def test_an_empty_source_fails_rather_than_passing_vacuously(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dst = Path(tmp) / "dst"
            src.mkdir()
            dst.mkdir()
            out = self._check(src, dst)
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)
            self.assertIn("no regular files at all", out.stderr)

    def test_a_missing_destination_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = self._tree(Path(tmp) / "src", {"config.json": 10})
            out = self._check(src, Path(tmp) / "nope")
            self.assertEqual(out.returncode, EXIT_STAGE, out.stdout + out.stderr)


class LegToLegTests(unittest.TestCase):
    """The spread has something to be read against, and it is not called a bias.

    `warmup` is a second `default` leg: same binary, same arm, same flags, run
    and discarded to warm the page cache. Its `.time` file was written and read
    by nothing. `|warmup - default|` is one repeat of one cell, so it is a
    LEG-TO-LEG figure -- cold, therefore an upper bound on run-to-run variation
    rather than the calibrated noise band `.agents/benchmarking.md` asks for.

    It is printed and never gated. The pass rule is unchanged and no spread
    threshold exists; what changed is that the FAILING message no longer says a
    bias of `spread / 2` "is in this run", when one leg per cell cannot tell a
    bias from variance.
    """

    def _run_with_warmup(self, kind: str, tmp: Path, warmup_delta: int) -> Run:
        r = Run(tmp / "run", kind)
        base = kb(40 * 1024**3)
        saving = kb(need(kind) + 8 * 1024**2)
        r.pair("default", "lmo", base, saving)
        r.pair("default2", "lmo2", base, saving)
        r.warmup(base + warmup_delta)
        return r

    def test_the_warmup_leg_is_reported_as_a_leg_to_leg_figure(self) -> None:
        delta = 3 * 1024**2
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_with_warmup("qwen3-vl", Path(tmp), delta)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_MET, out.stdout + out.stderr)
            line = [ln for ln in out.stdout.splitlines() if "leg-to-leg" in ln]
            self.assertEqual(len(line), 1, out.stdout)
            self.assertIn(str(delta), line[0])
            self.assertIn("UPPER BOUND", out.stdout)

    def test_the_figure_is_an_absolute_difference(self) -> None:
        """A warmup leg BELOW the default one is still a distance, not a sign."""

        delta = -3 * 1024**2
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run_with_warmup("qwen3-vl", Path(tmp), delta)
            out = r.report()
            line = [ln for ln in out.stdout.splitlines() if "leg-to-leg" in ln]
            self.assertEqual(len(line), 1, out.stdout)
            self.assertIn(str(abs(delta)), line[0])

    def test_a_run_without_a_warmup_leg_says_so(self) -> None:
        """Absence is reported, never rendered as a leg-to-leg spread of zero."""

        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", "qwen3-vl")
            base = kb(40 * 1024**3)
            saving = kb(need("qwen3-vl") + 8 * 1024**2)
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_MET, out.stdout + out.stderr)
            self.assertIn("NOT AVAILABLE", out.stdout)
            self.assertNotRegex(out.stdout, r"leg-to-leg \|warmup-default\|\s+\d")

    def test_the_spread_is_not_attributed_entirely_to_a_bias(self) -> None:
        kind = "qwen3-vl"
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            bias = kb(need(kind) // 5)
            r.pair("default", "lmo", base, kb(need(kind)) + bias)
            r.pair("default2", "lmo2", base, kb(need(kind)) - bias)
            out = r.report()
            self.assertEqual(out.returncode, EXIT_FAILING, out.stdout + out.stderr)
            # The label no longer asserts the spread IS twice a bias ...
            self.assertNotIn("spread = 2|d|", out.stdout)
            self.assertIn("spread |pair 1 - pair 2|", out.stdout)
            # ... and the prose names the alternative explanation.
            self.assertIn("run-to-run variance lands in the same spread", out.stdout)
            self.assertIn("IF the spread were", out.stdout)


class DryRunTests(unittest.TestCase):
    """The half of the harness that only ever runs under a lease, gated.

    Everything else in this file reads finished logs. None of it could see that
    the script configured two build directories with
    `-DVLLM_CPP_BUILD_EXAMPLES=OFF` and then ran `ninja vllm-server` -- and
    `vllm-server` is the OUTPUT_NAME of the `server` target in
    `examples/CMakeLists.txt`, which the root CMakeLists adds only under
    `if(VLLM_CPP_BUILD_EXAMPLES)`. `ninja` answered `unknown target
    'vllm-server'`, the run exited 4 at the first arm with no RSS in existence,
    and this suite was 41/41 green.

    `--dry-run` builds nothing and starts nothing. It resolves the kind, prints
    the cmake, ninja and `run_arm` invocations the run would issue -- out of the
    same variables the run itself uses, so it is the plan rather than a
    transcription of it -- and asserts that CMake defines the target under those
    flags. `test_the_committed_script_has_a_coherent_plan` is the case that reds
    on the defect; `test_flipping_the_examples_flag_back_off_is_caught` is the
    mutation that proves the assertion has teeth.
    """

    def test_the_committed_script_has_a_coherent_plan(self) -> None:
        out = run("--dry-run", "--model-kind", "qwen3-vl")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(EXAMPLES_FLAG, out.stdout)
        self.assertIn(SERVER_RELPATH, out.stdout)
        self.assertIn("Nothing was built", out.stdout)

    def test_the_plan_names_every_leg_the_run_issues(self) -> None:
        out = run("--dry-run", "--model-kind", "muse-glimmer")
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        for tag in ("warmup", "default", "lmo", "default2", "lmo2"):
            with self.subTest(leg=tag):
                self.assertIn("%s.time" % tag, out.stdout)
        # Each binary runs each arm exactly once, and only the two `lmo` legs
        # carry the flag under measurement.
        self.assertEqual(out.stdout.count("--language-model-only"), 2)
        self.assertIn("/health", out.stdout)
        self.assertIn("/v1/completions", out.stdout)

    def test_the_qwen3_vl_plan_runs_no_completion(self) -> None:
        out = run("--dry-run", "--model-kind", "qwen3-vl")
        self.assertNotIn("/v1/completions", out.stdout)
        self.assertIn("cannot run a completion", out.stdout)

    def test_a_dry_run_without_a_kind_is_refused(self) -> None:
        out = run("--dry-run")
        self.assertEqual(out.returncode, EXIT_ARGS, out.stdout + out.stderr)
        self.assertIn("--dry-run needs a model kind", out.stderr)

    def test_a_dry_run_configures_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "build"
            out = run(
                "--dry-run", "--model-kind", "qwen3-vl",
                "--build-dir-prefix", str(prefix),
            )
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [])

    def _script_with(self, tmp: Path, old: str, new: str) -> Path:
        """A scratch copy of the script with one substitution. Never the tree.

        The script resolves its repository as `dirname $0/../..`, so the copy is
        laid out the same way and the two CMakeLists files it reads are
        symlinked in unchanged. The mutation is in the SCRIPT; what it is
        checked against stays the tree's own CMake.
        """

        root = tmp / "scratch"
        (root / "scripts/mm").mkdir(parents=True)
        (root / "examples").mkdir()
        (root / "CMakeLists.txt").symlink_to(ROOT / "CMakeLists.txt")
        (root / "examples/CMakeLists.txt").symlink_to(ROOT / "examples/CMakeLists.txt")
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(text.count(old), 1, "mutation anchor is not unique: %r" % old)
        copy = root / "scripts/mm" / SCRIPT.name
        copy.write_text(text.replace(old, new, 1), encoding="utf-8")
        return copy

    def test_flipping_the_examples_flag_back_off_is_caught(self) -> None:
        """The mutation that is the defect, in a scratch copy: exit 4, named."""

        with tempfile.TemporaryDirectory() as tmp:
            copy = self._script_with(
                Path(tmp), EXAMPLES_FLAG, "-DVLLM_CPP_BUILD_EXAMPLES=OFF"
            )
            out = subprocess.run(
                ["bash", str(copy), "--dry-run", "--model-kind", "qwen3-vl"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(out.returncode, EXIT_BUILD, out.stdout + out.stderr)
            self.assertIn("cannot produce", out.stderr)
            self.assertIn("VLLM_CPP_BUILD_EXAMPLES", out.stderr)
            self.assertIn("unknown target", out.stderr)

    def test_dropping_the_flag_entirely_is_caught(self) -> None:
        """Absent is not ON. CMake's own default for the option is not read."""

        with tempfile.TemporaryDirectory() as tmp:
            copy = self._script_with(Path(tmp), " " + EXAMPLES_FLAG, "")
            out = subprocess.run(
                ["bash", str(copy), "--dry-run", "--model-kind", "qwen3-vl"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(out.returncode, EXIT_BUILD, out.stdout + out.stderr)
            self.assertIn("no -DVLLM_CPP_BUILD_EXAMPLES at all", out.stderr)

    def test_a_binary_path_that_disagrees_with_cmake_is_caught(self) -> None:
        """`run_arm` must look where CMake writes the target, not beside it."""

        with tempfile.TemporaryDirectory() as tmp:
            copy = self._script_with(
                Path(tmp),
                "SERVER_RELPATH=%s" % SERVER_RELPATH,
                "SERVER_RELPATH=%s" % NINJA_TARGET,
            )
            out = subprocess.run(
                ["bash", str(copy), "--dry-run", "--model-kind", "qwen3-vl"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(out.returncode, EXIT_BUILD, out.stdout + out.stderr)
            self.assertIn("where the legs would look for it", out.stderr)


class DryRunLiveQueryTests(unittest.TestCase):
    """The second prong: ask a CONFIGURED tree whether the target is in it.

    Static agreement between the flags and CMake's sources is what CI can check
    on a checkout with no toolchain. On a box that has already configured the
    build directories -- which the operator's box has, by the time this matters
    -- ninja can be asked directly. The fixtures below are hand-written
    `build.ninja` files, so this needs no cmake, no compiler and no build.
    """

    def setUp(self) -> None:
        if shutil.which("ninja") is None:
            self.skipTest("ninja is not on PATH, so the live prong is UNVERIFIED here")

    def _tree(self, root: Path, with_target: bool) -> None:
        root.mkdir(parents=True, exist_ok=True)
        body = "rule noop\n  command = true\nbuild examples/other: noop\n"
        if with_target:
            body += "build %s: phony examples/other\n" % NINJA_TARGET
        (root / "build.ninja").write_text(body, encoding="utf-8")

    def _dry_run(self, prefix: Path) -> subprocess.CompletedProcess[str]:
        return run(
            "--dry-run", "--model-kind", "qwen3-vl",
            "--build-dir-prefix", str(prefix),
        )

    def test_a_configured_tree_carrying_the_target_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "build"
            for arm in ("a", "b"):
                self._tree(Path(str(prefix) + "-" + arm), with_target=True)
            out = self._dry_run(prefix)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(out.stdout.count("IS a target in"), 2, out.stdout)

    def test_a_configured_tree_without_the_target_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "build"
            self._tree(Path(str(prefix) + "-a"), with_target=False)
            self._tree(Path(str(prefix) + "-b"), with_target=True)
            out = self._dry_run(prefix)
            self.assertEqual(out.returncode, EXIT_BUILD, out.stdout + out.stderr)
            self.assertIn("is not a target in the configured tree", out.stderr)

    def test_an_unconfigured_tree_is_skipped_by_name(self) -> None:
        """Absent is not verified. It says so rather than reading as a pass."""

        with tempfile.TemporaryDirectory() as tmp:
            out = self._dry_run(Path(tmp) / "never-configured")
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(out.stdout.count("live query         SKIPPED"), 2)


class WorkerMappingTests(unittest.TestCase):
    """The header records worker `/workspace` == local `/mnt/nas_share/rc`.

    Not discoverable from the client side, and it cost several probes, so it is
    written down where the invocation is -- and the invocation examples must not
    hand a leased worker a path it cannot resolve.
    """

    def test_the_header_documents_a_worker_visible_invocation(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("/workspace/ckpt/", text)
        self.assertIn("--stage-to", text)

    def test_no_example_invocation_uses_the_unreachable_path(self) -> None:
        for line in SCRIPT.read_text(encoding="utf-8").splitlines():
            if "--checkpoint /mnt/nas_share/checkpoints" in line:
                self.fail("example invocation uses a path no worker resolves: " + line)

    def test_the_spec_records_the_mount_mapping(self) -> None:
        text = SPEC.read_text(encoding="utf-8")
        self.assertIn("/mnt/nas_share/rc", text)
        self.assertIn("/workspace/ckpt", text)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
