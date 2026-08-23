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
EXIT_STAGE = 6
EXIT_NOT_LOCAL = 7

# A CIFS/SMB mount to test the filesystem-type prong against. Absent on a box
# without the NAS, and the case is then SKIPPED by name rather than passing.
CIFS_CANDIDATES = ("/mnt/nas_share", "/mnt/media")

# The declarations under test, repeated here so a silent edit to either the
# script or the spec reds this suite instead of passing quietly.
ONDISK = {
    "muse-glimmer": 3843691520,
    "qwen3-vl": 830695424,
}
FRACTION_PCT = 90

SKIP_LINE = (
    "server: multimodal towers NOT loaded (every modality they serve is at "
    "limit 0): vision_tower\n"
)


def resident(kind: str) -> int:
    """The loader widens bf16 -> host f32 on both kinds (#1359)."""

    return ONDISK[kind] * 2


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


class BoundaryTests(unittest.TestCase):
    """The pass/fail boundary is `>=`, on both kinds, and it is not fuzzy."""

    def _at(self, kind: str, delta_bytes: int) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            r = Run(Path(tmp) / "run", kind)
            base = kb(40 * 1024**3)
            saving = kb(need(kind)) + delta_bytes
            r.pair("default", "lmo", base, saving)
            r.pair("default2", "lmo2", base, saving)
            return r.report().returncode

    def test_exactly_at_the_threshold_is_met(self) -> None:
        # `kb(need(kind))` rounds DOWN, so add one kilobyte to land on or above.
        for kind in ONDISK:
            with self.subTest(kind=kind):
                self.assertEqual(self._at(kind, 1024), EXIT_MET)

    def test_one_kilobyte_short_is_failing(self) -> None:
        for kind in ONDISK:
            with self.subTest(kind=kind):
                self.assertEqual(self._at(kind, -1024), EXIT_FAILING)


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
    """Script and spec declare the same numbers, because both are required to."""

    def test_script_carries_both_on_disk_figures(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        for kind, value in ONDISK.items():
            with self.subTest(kind=kind):
                self.assertIn(str(value), text)

    def test_spec_carries_both_on_disk_figures(self) -> None:
        text = SPEC.read_text(encoding="utf-8")
        for kind, value in ONDISK.items():
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

    def test_a_cifs_path_is_refused(self) -> None:
        for candidate in CIFS_CANDIDATES:
            fs = _fstype(candidate)
            if fs in ("cifs", "smb", "smb2", "smb3"):
                out = run("--check-source", candidate)
                self.assertEqual(
                    out.returncode, EXIT_NOT_LOCAL, out.stdout + out.stderr
                )
                self.assertIn("filesystem", out.stderr)
                return
        self.skipTest(
            "no CIFS/SMB mount among %s on this host, so the filesystem-type "
            "prong is UNVERIFIED here rather than passing" % (CIFS_CANDIDATES,)
        )


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
