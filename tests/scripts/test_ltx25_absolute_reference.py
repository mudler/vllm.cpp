#!/usr/bin/env python3
"""`LTX25-ORACLE-ABSOLUTE` (#1854): the absolute panel is a GATE, and it fires.

#1854 was filed rather than closed with a proxy, and it is explicit about the
only admissible shape: "worse than the oracle on this statistic", **because that
is a comparison and not a convention**. Its own words for the alternative are
that "a proxy for perceptual quality that measures nothing is worse than a
declared gap", and that a hand-rolled statistic "would be the
`a-shape-valid-gate-passes-a-wrong-artefact` failure.

So this suite has two jobs, and the second is the one that is easy to skip.

FIRST, that the check FIRES. A bound wide enough to admit anything is a mute
switch, so the suite degrades a render in the way the statistic exists to
detect and requires a FAIL, on the same fixtures that pass clean.

SECOND, that the bound is the REFERENCE and not a number. Every assertion below
that touches a bound derives it from a reference the test itself built, and one
case changes the reference and requires the bound to move with it. A literal in
this file would be `a-transcription-cannot-gate-the-function-it-transcribes`
wearing a test's clothes: it would keep passing after `blockiness_bands` changed
meaning.

Two limits are stated rather than papered over. The 25 PPM frames of the real
#1864 render are NOT committed -- `SHA256SUMS` records their digests and says
they stay on the NAS -- so the only in-tree reference is the mp4, and the case
that reads it needs `ffmpeg`. It SKIPS loudly rather than passing quietly when
ffmpeg is absent, because a case that cannot run is not a case that passed.

No build, no GPU, no network, numpy only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/ltx25-render-compare.py"
GOLDENS = ROOT / "tests/parity/goldens/ltx2_oracle"
COMMITTED_MP4 = GOLDENS / "upstream-render.mp4"
COMMITTED_SUMS = GOLDENS / "SHA256SUMS"


# --- fixtures -----------------------------------------------------------------
def write_ppm(path: Path, a: np.ndarray) -> None:
    h, w, _ = a.shape
    path.write_bytes(b"P6\n%d %d\n255\n" % (w, h) + a.astype(np.uint8).tobytes())


def write_wav(path: Path, x: np.ndarray, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.astype("<i2").tobytes())


def textured(seed: int, h: int = 64, w: int = 96, n: int = 8) -> list[np.ndarray]:
    """A short clip with structure at every scale and real motion.

    Smooth gradients plus band-limited noise, translated frame to frame. It has
    to be textured rather than white noise: `blockiness_bands` divides the
    on-grid step by the off-grid step, and on white noise both are the same
    large number, so the ratio pins to 1.0 and the fixture could not be made
    blockier. It has to MOVE, because C0 refuses a still clip before any of this
    is reached.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (h + 16, w + 16, 3))
    # A cheap low-pass: repeated 2x2 box, which leaves energy at the scales an
    # 8x8 flatten can remove.
    for _ in range(3):
        base = 0.25 * (base + np.roll(base, 1, 0) + np.roll(base, 1, 1)
                       + np.roll(np.roll(base, 1, 0), 1, 1))
    base = base / (base.std() + 1e-9)
    yy, xx = np.mgrid[0:h + 16, 0:w + 16]
    ramp = (yy * 0.9 + xx * 1.3)[:, :, None]
    out = []
    for k in range(n):
        f = np.roll(np.roll(base, k * 2, axis=0), k * 3, axis=1) * 26.0 + ramp + 96.0
        out.append(np.clip(f[:h, :w], 0, 255).astype(np.uint8))
    return out


def flatten_blocks(a: np.ndarray, alpha: float, grid: int = 8) -> np.ndarray:
    """Move every `grid`x`grid` block a fraction `alpha` toward its own mean.

    The canonical block artefact, and the one a blockiness ratio exists to see.
    `alpha = 1.0` is the degenerate end: the off-grid step inside a block becomes
    zero, the ratio's DENOMINATOR collapses, and `blockiness_bands` returns 0.0.
    That is the case the band's lower edge is for.
    """
    f = a.astype(np.float64)
    h, w, _ = f.shape
    hh, ww = h // grid * grid, w // grid * grid
    t = f[:hh, :ww].reshape(hh // grid, grid, ww // grid, grid, 3).mean(axis=(1, 3))
    t = np.repeat(np.repeat(t, grid, axis=0), grid, axis=1)
    out = f.copy()
    out[:hh, :ww] = (1.0 - alpha) * f[:hh, :ww] + alpha * t
    return np.clip(out, 0, 255).astype(np.uint8)


def render_dir(base: Path, name: str, frames: list[np.ndarray], seed: int = 7) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        write_ppm(d / f"frame_{i:06d}.ppm", f)
    rng = np.random.default_rng(seed)
    write_wav(d / "audio.wav", rng.integers(-9000, 9000, (24000, 2)))
    return d


def sums_for(d: Path, out: Path) -> Path:
    """A digest list over a directory, in `sha256sum` form."""
    lines = ["# a digest list this test wrote for a reference it built"]
    for p in sorted(d.glob("frame_*.ppm")):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    out.write_text("\n".join(lines) + "\n")
    return out


def run(*args: str) -> tuple[int, str, dict | None]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
        jpath = jf.name
    p = subprocess.run([sys.executable, str(TOOL), *args, "--json", jpath],
                       capture_output=True, text=True)
    try:
        report = json.loads(Path(jpath).read_text())
    except (OSError, json.JSONDecodeError):
        report = None
    return p.returncode, p.stdout + p.stderr, report


def checks_of(report: dict) -> dict[str, bool]:
    return {c["name"]: c["pass"] for c in report["checks"]}


class ReferenceIdentity(unittest.TestCase):
    """T1 and its neighbours: what is admitted as a reference, and what is not."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ltx25-absref-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ref = render_dir(self.tmp, "ref", textured(20260827))
        self.sums = sums_for(self.ref, self.tmp / "SHA256SUMS")
        # A DIFFERENT SEED, and the reason is a defect this suite found in
        # itself. Built from the same seed, `ours` is byte-identical to `ref`,
        # so its digests ARE in the list and the case passed at exit 0 -- which
        # is the failure it is named after, staged by accident: a reference that
        # is the render under test admits itself. The fixture has to be a
        # DIFFERENT render for the refusal to be the thing under test.
        self.ours = render_dir(self.tmp, "ours", textured(31337))

    def test_a_reference_whose_digest_is_absent_is_refused_and_never_compared(self) -> None:
        """A reference a caller can point anywhere is not a reference.

        Pointed at the render under test the gate would pass by construction,
        which is `oracle-identity-must-be-asserted` exactly. The status matters
        as much as the refusal: EXIT_UNREADABLE (2), never EXIT_FAIL (1). A 1
        would say this render is worse than a reference that was never
        established, and a reader cannot tell that apart from the finding this
        gate exists to make.
        """
        rc, out, rep = run("--a", str(self.ours), "--reference", str(self.ours),
                           "--reference-sums", str(self.sums))
        self.assertEqual(rc, 2, out)
        self.assertIn("is not what SHA256SUMS records", out)
        self.assertIn("VERDICT UNREADABLE", out)
        self.assertIsNone(rep, "a refused reference must write NO report")

    def test_one_tampered_frame_is_enough_to_refuse_the_whole_reference(self) -> None:
        """Every frame's digest is checked, not the first or a sample of them."""
        victim = sorted(self.ref.glob("frame_*.ppm"))[-1]
        a = textured(999)[0]
        write_ppm(victim, a)
        rc, out, _ = run("--a", str(self.ours), "--reference", str(self.ref),
                         "--reference-sums", str(self.sums))
        self.assertEqual(rc, 2, out)
        self.assertIn(victim.name, out)

    def test_an_empty_digest_list_anchors_nothing_and_is_refused(self) -> None:
        empty = self.tmp / "EMPTY"
        empty.write_text("# nothing but a comment\n")
        rc, out, _ = run("--a", str(self.ours), "--reference", str(self.ref),
                         "--reference-sums", str(empty))
        self.assertEqual(rc, 2, out)
        self.assertIn("nothing anchors the reference", out)

    def test_neither_b_nor_reference_is_refused_by_the_parser(self) -> None:
        rc, out, _ = run("--a", str(self.ours))
        self.assertEqual(rc, 2, out)
        self.assertIn("--b or --reference is required", out)

    def test_a_control_without_b_is_refused_rather_than_ignored(self) -> None:
        """`--control` calibrates a delta between two arms. With no arm B there is
        no delta, and silently dropping the argument would leave a caller
        believing a noise floor was read."""
        rc, out, _ = run("--a", str(self.ours), "--reference", str(self.ref),
                         "--reference-sums", str(self.sums), "--control", str(self.ref))
        self.assertEqual(rc, 2, out)
        self.assertIn("no delta for it to calibrate", out)


class TheBoundIsTheReference(unittest.TestCase):
    """T6 and T12: the bound is recomputed from the reference, never written down."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ltx25-absref-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _reference(self, name: str, frames: list[np.ndarray]) -> tuple[Path, Path]:
        d = render_dir(self.tmp, name, frames)
        return d, sums_for(d, self.tmp / f"SUMS-{name}")

    def test_the_reported_bound_equals_the_references_own_per_frame_maximum(self) -> None:
        """Computed here from the reference's pixels, with the tool's own
        functions, and required to equal what the tool published. A literal on
        either side of this assertion would make it a tautology."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("rc_tool", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)

        frames = textured(4242)
        ref, sums = self._reference("ref", frames)
        ours = render_dir(self.tmp, "ours", textured(4242))
        rc, out, rep = run("--a", str(ours), "--reference", str(ref),
                           "--reference-sums", str(sums))
        self.assertEqual(rc, 0, out)

        want8 = max(float(tool.blockiness_bands(
            tool.luma(f).astype(np.float64), grid=tool.BLOCK_GRID).mean())
            for f in frames)
        got = rep["reference"]["bounds"]["blockiness_grid8"]
        self.assertAlmostEqual(got["frame_max"], want8, places=12)
        self.assertEqual(got["n"], len(frames))

    def test_the_ENFORCED_bound_moves_with_the_reference_not_only_the_reported_one(self) -> None:
        """THE HOLE A MUTATION FOUND, and the case that closes it.

        Replacing `lo, hi = b["frame_min"], b["frame_max"]` in `reference_checks`
        with the real oracle render's two literals left every other case in this
        file GREEN. The reported bound in the JSON still moved with the
        reference, because `reference_bounds` still computed it -- so the case
        below that reads the report could not see it -- and the pass/fail cases
        happened to agree with the literals on their fixtures. A gate enforcing a
        transcribed number while REPORTING a computed one is precisely
        `a-transcription-cannot-gate-the-function-it-transcribes`, and it is
        worse than an honest literal because the report vouches for it.

        So: ONE render, TWO references, OPPOSITE verdicts. The render is blockier
        than a clean reference tolerates and no blockier than a blocky reference
        tolerates. No number appears in this test at all, and any bound that is
        not read from the reference in hand gives the same answer twice.
        """
        ours_frames = [flatten_blocks(f, 0.30) for f in textured(77)]
        ours = render_dir(self.tmp, "ours-mid", ours_frames)
        clean_ref, clean_sums = self._reference("clean", textured(78))
        blocky_ref, blocky_sums = self._reference(
            "blocky", [flatten_blocks(f, 0.60) for f in textured(78)])

        rc_clean, out_clean, rep_clean = run(
            "--a", str(ours), "--label-a", "ours",
            "--reference", str(clean_ref), "--reference-sums", str(clean_sums))
        rc_blocky, out_blocky, rep_blocky = run(
            "--a", str(ours), "--label-a", "ours",
            "--reference", str(blocky_ref), "--reference-sums", str(blocky_sums))

        self.assertEqual(rc_clean, 1,
                         "against a CLEAN reference this render must be worse\n" + out_clean)
        self.assertFalse(checks_of(rep_clean)["absolute.ours.blockiness_grid8"])
        self.assertEqual(rc_blocky, 0,
                         "against a BLOCKY reference the same render must pass; if it "
                         "does not, the bound is not coming from the reference\n" + out_blocky)
        self.assertTrue(checks_of(rep_blocky)["absolute.ours.blockiness_grid8"])
        # And it is the SAME render on both sides, so nothing but the reference
        # can explain the two verdicts.
        self.assertEqual(rep_clean["absolute_quality"]["ours"]["blockiness_grid8"],
                         rep_blocky["absolute_quality"]["ours"]["blockiness_grid8"])

    def test_a_different_reference_moves_the_bound(self) -> None:
        """The proof that the bound is READ rather than stored. Two references,
        two bands, from one unchanged tool."""
        ours = render_dir(self.tmp, "ours", textured(1))
        ref_a, sums_a = self._reference("refa", textured(2))
        ref_b, sums_b = self._reference("refb",
                                        [flatten_blocks(f, 0.5) for f in textured(2)])
        _, _, rep_a = run("--a", str(ours), "--reference", str(ref_a),
                          "--reference-sums", str(sums_a))
        _, _, rep_b = run("--a", str(ours), "--reference", str(ref_b),
                          "--reference-sums", str(sums_b))
        hi_a = rep_a["reference"]["bounds"]["blockiness_grid8"]["frame_max"]
        hi_b = rep_b["reference"]["bounds"]["blockiness_grid8"]["frame_max"]
        self.assertGreater(hi_b, hi_a * 1.2,
                           "a blockier reference must yield a higher ceiling; if these "
                           "agree the bound is not coming from the reference")


class TheGateFires(unittest.TestCase):
    """T3, T4, T5: clean passes, blocky fails, and the degenerate end fails too."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ltx25-absref-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.frames = textured(20260827)
        self.ref = render_dir(self.tmp, "ref", self.frames)
        self.sums = sums_for(self.ref, self.tmp / "SHA256SUMS")

    def _run_on(self, frames: list[np.ndarray], name: str):
        d = render_dir(self.tmp, name, frames)
        return run("--a", str(d), "--reference", str(self.ref),
                   "--reference-sums", str(self.sums), "--label-a", "ours")

    def test_a_clean_render_passes_both_blockiness_checks(self) -> None:
        rc, out, rep = self._run_on(textured(20260828), "clean")
        got = checks_of(rep)
        for name in ("blockiness_grid8", "blockiness_grid32",
                     "blockiness_grid8_defined", "blockiness_grid32_defined"):
            self.assertTrue(got[f"absolute.ours.{name}"], f"{name}\n{out}")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["reading"], "NO_WORSE_THAN_ORACLE_ON_BLOCKINESS")

    def test_a_block_flattened_render_fails_and_the_run_exits_one(self) -> None:
        """THE CASE THE WHOLE ROW IS FOR. Same fixtures, same reference, one
        degradation: the 8x8 grid a blockiness ratio exists to detect."""
        rc, out, rep = self._run_on(
            [flatten_blocks(f, 0.35) for f in textured(20260828)], "blocky")
        got = checks_of(rep)
        self.assertFalse(got["absolute.ours.blockiness_grid8"], out)
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["reading"], "WORSE_THAN_ORACLE")
        self.assertIn("worse than the oracle on this statistic", out)

    def test_the_fully_flattened_render_fails_the_DEFINEDNESS_check(self) -> None:
        """THE MUTE SWITCH THE GUARD EXISTS TO CLOSE.

        `blockiness_bands` returns 0.0 when the off-grid denominator is zero,
        which is what a fully flat block grid produces -- the WORST artefact this
        statistic can be shown, reading as the SMALLEST possible value. It
        therefore clears the ceiling, and this case asserts that it does, so the
        hole is visible rather than implied. What catches it is the collapsed-band
        COUNT beside the ceiling.

        The count and not a second edge: a two-sided band was the first design,
        and `test_the_ENFORCED_bound_moves_with_the_reference...` is the case
        that found it failing a render for being BETTER than the reference.
        """
        rc, out, rep = self._run_on(
            [flatten_blocks(f, 1.0) for f in textured(20260828)], "flat")
        panel = rep["absolute_quality"]["ours"]
        got = checks_of(rep)
        self.assertEqual(panel["blockiness_grid8"], 0.0,
                         "the fixture must reach the degenerate reading, or this case "
                         "is not testing what it says it is")
        self.assertGreater(panel["blockiness_grid8_collapsed_bands"], 0)
        # THE HOLE, ASSERTED: the ceiling passes this render.
        self.assertTrue(got["absolute.ours.blockiness_grid8"],
                        "a ceiling cannot see a collapse; if this ever fails the "
                        "guard below is no longer the thing doing the work")
        # AND THE GUARD THAT CLOSES IT.
        self.assertFalse(got["absolute.ours.blockiness_grid8_defined"], out)
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["reading"], "WORSE_THAN_ORACLE")
        self.assertIn("denominator collapsing", out)

    def test_a_degenerate_REFERENCE_is_refused_rather_than_used(self) -> None:
        """The instrument's own precondition. A reference whose bands collapsed
        has a ceiling of 0.0, against which every render on earth fails -- a
        broken instrument reporting a code verdict. It is refused at
        EXIT_UNREADABLE, where a refused reference belongs."""
        flat = [flatten_blocks(f, 1.0) for f in textured(4)]
        bad = render_dir(self.tmp, "flatref", flat)
        bad_sums = sums_for(bad, self.tmp / "SUMS-flatref")
        rc, out, rep = run("--a", str(self.ref), "--label-a", "ours",
                           "--reference", str(bad), "--reference-sums", str(bad_sums))
        self.assertEqual(rc, 2, out)
        self.assertIn("off-grid denominator collapsed", out)
        self.assertIsNone(rep)

    def test_a_pure_noise_render_PASSES_and_that_is_the_gates_limit(self) -> None:
        """THE LIMIT, WRITTEN AS A TEST RATHER THAN AS A PARAGRAPH.

        Blockiness is a RATIO, so white noise has a huge step on the grid and an
        equally huge one off it and reads near 1.0 -- inside the reference's band.
        C0 does not catch it either: `ltx25-dit-attn-flash.md` §10.8 records that
        two identical sequences of pure noise clear all three C0 checks. So a
        render that is worthless in the way a reader most fears passes this gate,
        and the passing reading is named for the one thing it measured.

        This case asserts the PASS deliberately. A later reader who assumes the
        gate is broader than blockiness is contradicted by a red test if anyone
        ever narrows it here without saying so, and by this one if they assume it
        already is.
        """
        rng = np.random.default_rng(5)
        noise = [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(8)]
        rc, out, rep = self._run_on(noise, "noise")
        self.assertEqual(rc, 0, out)
        self.assertTrue(checks_of(rep)["absolute.ours.blockiness_grid8"], out)
        self.assertEqual(rep["reading"], "NO_WORSE_THAN_ORACLE_ON_BLOCKINESS")
        # And the statistic that WOULD have seen it is printed, unchecked, beside.
        self.assertGreater(rep["absolute_quality"]["ours"]["sharpness_mean"],
                           3.0 * 11.274,
                           "pure noise must read far above the real reference's "
                           "sharpness; if it does not, this case is not the "
                           "demonstration it claims to be")

    def test_a_degenerate_render_fails_C0_before_any_bound_is_read(self) -> None:
        """A blank clip has no off-grid step either, so its blockiness number is
        meaningless. C0 must catch it first, and the READING must say so."""
        flat = [np.full((64, 96, 3), 130, dtype=np.uint8) for _ in range(8)]
        rc, out, rep = self._run_on(flat, "blank")
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["reading"], "CONTENT_DEGENERATE")


class OneRenderOrTwo(unittest.TestCase):
    """T7 and T8: the single-render path judges only what it measured, and the
    two-render path is not touched by any of this."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ltx25-absref-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ref = render_dir(self.tmp, "ref", textured(11))
        self.sums = sums_for(self.ref, self.tmp / "SHA256SUMS")
        self.a = render_dir(self.tmp, "a", textured(12))
        self.b = render_dir(self.tmp, "b", textured(12, n=8))

    def test_without_b_no_identity_alignment_or_coherence_check_is_emitted(self) -> None:
        """The single-render path must not report a relative verdict it never
        computed. A `coherence` entry here would be an answer to a question no
        second render was supplied for."""
        rc, out, rep = run("--a", str(self.a), "--reference", str(self.ref),
                           "--reference-sums", str(self.sums))
        self.assertEqual(rc, 0, out)
        names = [c["name"] for c in rep["checks"]]
        for prefix in ("align.", "coherence.", "video.", "audio."):
            self.assertEqual([n for n in names if n.startswith(prefix)], [], prefix)
        self.assertIsNone(rep["identity_verdict"])
        self.assertIsNone(rep["control_verdict"])
        self.assertEqual(rep["mode"], "absolute_only")
        # A run with no scorer must still SAY that it measured no adherence, and
        # must not carry an adherence check into the table. The wording moved
        # when #2295 landed the scorer -- the declaration used to read "not
        # measured anywhere in this tree", which stopped being true -- and the
        # obligation did not: the 77-position bound is stated in both states.
        self.assertIn("PROMPT ADHERENCE IS NOT MEASURED IN THIS RUN", out)
        self.assertIn("77 positions", out)
        self.assertEqual([n for n in names if "adherence" in n], [])
        self.assertNotIn("adherence", rep)

    def test_a_reference_changes_no_arm_to_arm_check(self) -> None:
        """#1743's checks keep their names, their values and their verdicts.

        Compared as REPORTS rather than read off a diff, on the same fixtures, so
        a later edit that quietly moved one of them is red here.
        """
        _, out1, plain = run("--a", str(self.a), "--b", str(self.b))
        _, out2, withref = run("--a", str(self.a), "--b", str(self.b),
                               "--reference", str(self.ref),
                               "--reference-sums", str(self.sums))
        strip = lambda rep: [c for c in rep["checks"]
                             if not c["name"].startswith("absolute.")]
        self.assertEqual(strip(plain), strip(withref), out1 + out2)
        for key in ("video", "audio", "structural", "content", "identity_verdict",
                    "identity_failed", "treatment_verdict"):
            self.assertEqual(plain[key], withref[key], key)

    def test_without_a_reference_nothing_absolute_is_checked(self) -> None:
        """#1854 as filed, and it must stay reachable: no reference, no gate, and
        the tool says so in its own output."""
        _, out, rep = run("--a", str(self.a), "--b", str(self.b))
        self.assertIn("absolute quality: REPORTED, and NOT CHECKED", out)
        self.assertIsNone(rep["reference"])
        self.assertEqual([c["name"] for c in rep["checks"]
                          if c["name"].startswith("absolute.")], [])
        for panel in rep["absolute_quality"].values():
            self.assertFalse(panel["checked"])

    def test_with_b_and_a_reference_both_arms_are_judged(self) -> None:
        _, out, rep = run("--a", str(self.a), "--b", str(self.b),
                          "--label-a", "ours", "--label-b", "theirs",
                          "--reference", str(self.ref),
                          "--reference-sums", str(self.sums))
        names = {c["name"] for c in rep["checks"]}
        self.assertIn("absolute.ours.blockiness_grid8", names)
        self.assertIn("absolute.theirs.blockiness_grid8", names)


class TheCommittedReference(unittest.TestCase):
    """The in-tree #1864 artefact is admissible under the SHIPPED default.

    Nothing else in this suite touches the real reference: every other case
    builds its own so the mechanism is exercised without a 225 KB dependency.
    This case exists because a gate whose default digest list does not resolve,
    or whose one committed reference is refused by its own checker, is a gate
    that will be discovered broken inside a GPU lease.
    """

    @unittest.skipUnless(shutil.which("ffmpeg"),
                         "ffmpeg is absent, so the committed mp4 cannot be decoded; "
                         "this case did NOT pass, it did not run")
    def test_the_committed_mp4_is_admitted_by_the_default_digest_list(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="ltx25-absref-"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ours = render_dir(tmp, "ours", textured(3))
        _, out, rep = run("--a", str(ours), "--reference", str(COMMITTED_MP4))
        self.assertIsNotNone(rep, out)
        ref = rep["reference"]
        self.assertEqual(ref["form"], "mp4")
        self.assertEqual(ref["frames"], 25)
        self.assertEqual(Path(ref["sums"]), COMMITTED_SUMS)
        self.assertEqual(ref["source_sha256"],
                         hashlib.sha256(COMMITTED_MP4.read_bytes()).hexdigest())
        # The bound is the render's, not this test's: only that it exists, is
        # finite, and brackets the reference's own mean is asserted here, because
        # pinning its VALUE would put a transcription in the tree.
        for name in ("blockiness_grid8", "blockiness_grid32"):
            b = ref["bounds"][name]
            self.assertEqual(b["n"], 25)
            self.assertLess(b["frame_min"], b["mean"])
            self.assertLess(b["mean"], b["frame_max"])

    def test_the_default_digest_list_resolves_from_the_script_itself(self) -> None:
        """No ffmpeg needed. The path is built from the tool's own location, the
        idiom this repository uses so a checker run from any directory reads the
        same file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("rc_tool2", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        self.assertEqual(Path(tool.DEFAULT_REFERENCE_SUMS), COMMITTED_SUMS)
        self.assertTrue(COMMITTED_SUMS.is_file())
        sums = tool.parse_sha256sums(str(COMMITTED_SUMS))
        self.assertEqual(
            sums["upstream-render.mp4"],
            hashlib.sha256(COMMITTED_MP4.read_bytes()).hexdigest(),
            "the committed mp4 no longer matches its own committed digest")

    def test_only_blockiness_gates_and_the_rest_say_they_do_not(self) -> None:
        """The panel must keep declaring which half decides anything. #1854
        shipped it declaring itself unchecked and the declaration was the point;
        a half-gated panel that reported `checked: true` would tell a reader
        something this row did not measure."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("rc_tool3", TOOL)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        self.assertEqual(tuple(tool.REFERENCE_GATED),
                         ("blockiness_grid8", "blockiness_grid32"))
        self.assertIn("sharpness_mean", tool.REFERENCE_REPORTED)
        self.assertIn("clipped_fraction", tool.REFERENCE_REPORTED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
