#!/usr/bin/env python3
"""The pixel comparison's own discrimination proof.

`.agents/specs/ltx25-dit-attn-flash.md` section 10, #1612.

`scripts/ltx25-render-compare.py` is the substitute for a token gate on a model
that cannot have one: LTX-2.5 renders pixels, not symbols, so "the output is
the same" has to be a measurement rather than an equality. A tool that answers
that question is only worth its verdict if it FAILS on a difference that matters
and PASSES on one that does not, and neither half is provable by reading it.

So both halves are pinned here, on fabricated frames, with no NAS and no GPU:

  Discrimination     two identical renders read as bit-identical and every
                     threshold is then vacuous rather than passed; a +/-1
                     dither on 3% of samples -- the shape section 10.2 predicts
                     from bf16 rounding -- passes all four video checks with
                     headroom; ONE PIXEL of global horizontal shift fails ALL
                     FOUR, which is the calibration section 10.4 quotes and a
                     criterion that admitted it would not be a criterion. All
                     eight cells of section 10.4's "Population 2, GATED" table
                     are asserted here, which is what makes that word true.
  IdenticallyBroken  C0: two all-black renders differ by zero and score the
                     strongest pass every V check can give, so each arm is
                     judged on its own content first.
  Refusal            an input that cannot be read exits 2, and a threshold
                     failure still exits 1. A missing input is never a pass and
                     never a difference either.
  Audio              A1 and A2 disagree on a waveform that drifted in time,
                     which is the case A2 exists for and the only case that can
                     read it; absent audio is a failed check and never a pass.
  Control            which arm the control repeats is an argument, and the
                     control-to-treatment ratio section 10.5 selects on is
                     computed rather than eyeballed.
  ControlContent     the control is judged on its OWN content too. It was not,
                     so a control of one-colour frames read as a very large
                     noise floor and published section 10.5's STRONGER null. A
                     degenerate control is its own status, exit 3, because it is
                     a broken experiment and not a visible difference.
  SsimIsTheOneItNames  V3 names Wang et al. 2004 at 11x11 sigma=1.5 on Rec.601
                     luma, and every part of that name is pinned by a property
                     or a value: scikit-image is absent here and stays absent.

The fixtures are TEXTURED rather than flat. A flat image makes SSIM degenerate
and makes a one-pixel shift invisible, so a test built on one would pass while
proving nothing -- the shape this file exists to refuse.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/ltx25-render-compare.py"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_UNREADABLE = 2
EXIT_CONTROL_DEGENERATE = 3

W, H, FRAMES = 96, 64, 6


def write_ppm(path: Path, arr: np.ndarray) -> None:
    h, w, _ = arr.shape
    path.write_bytes(b"P6\n%d %d\n255\n" % (w, h) + arr.astype(np.uint8).tobytes())


def write_wav(path: Path, samples: np.ndarray, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.astype("<i2").tobytes())


def make_render(d: Path, rng: np.random.Generator, motion: int = 3) -> list[np.ndarray]:
    """A textured, MOVING sequence: SSIM and the temporal denominator both need one."""
    d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    base = (
        127
        + 60 * np.sin(xx / 4.0)
        + 40 * np.cos(yy / 3.0)
        + rng.integers(-20, 21, (H, W))
    )
    frames = []
    for i in range(FRAMES):
        shifted = np.roll(base, motion * i, axis=1)
        rgb = np.stack(
            [shifted, np.roll(shifted, 5, axis=0), np.roll(shifted, -5, axis=1)], axis=2
        )
        arr = np.clip(rgb, 0, 255).astype(np.uint8)
        write_ppm(d / f"frame_{i:06d}.ppm", arr)
        frames.append(arr)
    t = np.arange(4800)
    wav = np.stack([8000 * np.sin(t / 20.0), 8000 * np.sin(t / 31.0)], axis=1)
    write_wav(d / "audio.wav", wav)
    return frames


def run(*args: str) -> tuple[int, str, dict | None]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
        jpath = jf.name
    p = subprocess.run(
        [sys.executable, str(TOOL), *args, "--json", jpath],
        capture_output=True,
        text=True,
    )
    try:
        report = json.loads(Path(jpath).read_text())
    except (OSError, json.JSONDecodeError):
        report = None
    return p.returncode, p.stdout + p.stderr, report


def checks_of(report: dict) -> dict[str, bool]:
    return {c["name"]: c["pass"] for c in report["checks"]}


def module_float_constants() -> dict[str, float]:
    """The tool's module-level float constants, PARSED rather than grepped.

    A SUBSTRING MATCH IS NOT A PIN, and a fresh review proved it on this file:
    the anti-widening test used `assertIn`, so `DEFAULT_MAX_MEAN_ABS = 1.09` and
    `DEFAULT_MAX_TEMPORAL_RATIO = 0.109` both left the whole suite green. Both
    are MAXIMUM bounds, so appending a digit widens them, which is the exact
    move the test is named after.
    """
    import ast

    src = (ROOT / "scripts/ltx25-render-compare.py").read_text()
    out: dict[str, float] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name) and isinstance(node.value.value, float):
                    out[t.id] = node.value.value
    return out



# --- the structural criterion's own discrimination proof (#1743) --------------

def box_blur(a: np.ndarray) -> np.ndarray:
    """A 3x3 box blur: the canonical one-directional loss of detail."""
    f = a.astype(np.float64)
    out = np.zeros_like(f)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out += np.roll(np.roll(f, dy, axis=0), dx, axis=1)
    return np.clip(out / 9.0, 0, 255).astype(np.uint8)


def block_artefacts(a: np.ndarray, q: int = 8) -> np.ndarray:
    """Flatten every 8x8 block halfway toward its own mean: a block grid."""
    f = a.astype(np.float64)
    h, w, _ = f.shape
    hh, ww = h // q * q, w // q * q
    t = f[:hh, :ww].reshape(hh // q, q, ww // q, q, 3).mean(axis=(1, 3))
    t = np.repeat(np.repeat(t, q, axis=0), q, axis=1)
    out = f.copy()
    out[:hh, :ww] = 0.5 * f[:hh, :ww] + 0.5 * t
    return np.clip(out, 0, 255).astype(np.uint8)


def read_wav_arr(p: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(p), "rb") as w:
        n, ch, rate = w.getnframes(), w.getnchannels(), w.getframerate()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.int64), rate


class Structural(unittest.TestCase):
    """#1743, section 11. What the verdict rests on now, and why it is not a
    widened tolerance.

    Section 10.4's bounds all measure IDENTITY, and section 11.1 records what
    that cannot do: an identity bound reads the same on a change that DEGRADED
    the render and on a pipeline that is sensitive to any arithmetic at all. The
    frames on the share show both at once -- flash against naive at ONE build is
    6.414156 and naive against naive across TWO builds is 9.452407, so the
    LARGER divergence is the one nobody attributes to a defect.

    The replacement asserts something else entirely:

      correspondence  a perturbation of the arithmetic moves the picture, and
                      does not move it in TIME or in SPACE. Frame k must still
                      match frame k, the audio argmax must still be at lag 0,
                      and the spatial argmin must still be (0, 0). Each constant
                      is the exact point at which a correspondence is lost.

      incoherence     a reassociated sum makes two renders EXCHANGEABLE and a
                      defect makes one of them worse, so K = |sum of the
                      differences| / sum of |them| is 1 EXACTLY under any
                      one-directional degradation and near N^-1/2 without one.

    Every case below mutates a real guarantee rather than reading one. The five
    degradations must RED the check the spec names; the symmetric null and the
    bit-identical pair must PASS; and the constant `0.5` must be shown to carry
    no argument, because any other value in the open interval gives the same
    verdict on both populations.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rng = np.random.default_rng(20260822)
        self.a = root / "a"
        self.frames = make_render(self.a, rng)
        self.root = root

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- fixtures -------------------------------------------------------------
    def arm(self, name: str, frames: list[np.ndarray] | None = None,
            audio=None) -> Path:
        d = self.root / name
        d.mkdir(exist_ok=True)
        for i, f in enumerate(frames if frames is not None else self.frames):
            write_ppm(d / f"frame_{i:06d}.ppm", f)
        w, rate = read_wav_arr(self.a / "audio.wav")
        write_wav(d / "audio.wav", audio(w) if audio else w, rate)
        return d

    def dithered(self, name: str, seed: int) -> Path:
        r = np.random.default_rng(seed)
        out = []
        for arr in self.frames:
            noise = (r.random(arr.shape) < 0.03) * r.integers(-1, 2, arr.shape)
            out.append(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        return self.arm(name, out)

    def structural_failures(self, rep: dict) -> list[str]:
        return [c["name"] for c in rep["checks"]
                if c["judges"] == "treatment" and not c["pass"]]

    # -- the null -------------------------------------------------------------
    def test_a_symmetric_dither_pair_passes_every_structural_check(self) -> None:
        """THE FAITHFUL NULL, and it is not the fixture section 10.4 uses.

        That one builds `B = A + dither`: one arm is literally the other plus
        noise. This one draws the dither INDEPENDENTLY into both arms, which is
        what two trajectories that separated look like. Nothing here may fire.
        """
        rc, out, rep = run("--a", str(self.dithered("sa", 11)),
                           "--b", str(self.dithered("sb", 12)))
        self.assertEqual(self.structural_failures(rep), [], out)
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertEqual(rep["reading"], "SEPARATED, NOT DEGRADED", out)
        for name in ("sharpness", "blockiness", "motion"):
            self.assertLess(rep["structural"]["coherence"][name]["k"], 0.5, out)

    def test_a_bit_identical_pair_passes_every_structural_check(self) -> None:
        """The control. `flash-ctl` against `flash` came back bit-identical, and
        a criterion that cannot pass a repeat of the same render is measuring
        the machine rather than the change."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.arm("same")))
        self.assertEqual(self.structural_failures(rep), [], out)
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertEqual(rep["reading"], "BIT_IDENTICAL", out)
        for name in ("sharpness", "blockiness", "motion", "audio_rms"):
            self.assertEqual(rep["structural"]["coherence"][name]["k"], 0.0, out)
        self.assertEqual(rep["structural"]["frame_correspondence"]["off_diagonal_frames"], 0)

    # -- the five degradations ------------------------------------------------
    def test_a_blur_makes_the_sharpness_coherence_exactly_one(self) -> None:
        """EXACTLY one, not merely large. A blur takes detail out of every tile,
        so every term of the sum has the same sign and the numerator equals the
        denominator by algebra. That is the property the threshold rests on."""
        rc, out, rep = run("--a", str(self.a),
                           "--b", str(self.arm("blur", [box_blur(f) for f in self.frames])))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["reading"], "DIRECTIONAL", out)
        co = rep["structural"]["coherence"]["sharpness"]
        self.assertEqual(co["k"], 1.0, out)
        self.assertEqual(co["direction"], "a>b", out)
        self.assertIn("coherence.sharpness", self.structural_failures(rep))

    def test_block_artefacts_make_the_blockiness_coherence_exactly_one(self) -> None:
        rc, out, rep = run("--a", str(self.a),
                           "--b", str(self.arm("blk", [block_artefacts(f) for f in self.frames])))
        self.assertEqual(rc, EXIT_FAIL, out)
        co = rep["structural"]["coherence"]["blockiness"]
        self.assertEqual(co["k"], 1.0, out)
        self.assertEqual(co["direction"], "b>a", out)
        self.assertIn("coherence.blockiness", self.structural_failures(rep))

    def test_a_one_frame_offset_breaks_frame_correspondence_where_C0_cannot(self) -> None:
        """The frame-drop case, built so that C0 is BLIND to it.

        Deleting a frame and repeating the tail makes two frames equal, and C0's
        distinct-hash and zero-motion checks catch that on their own -- which
        would prove nothing about `align.frames`. So arm B is arm A advanced by
        exactly one frame, with every frame still distinct and every pair still
        moving. C0 passes, and the correspondence is broken: arm B's frame k is
        arm A's frame k+1, so the nearest match is off the diagonal."""
        n = len(self.frames)
        a = self.arm("a48", self.frames[:-1])
        b = self.arm("b48", self.frames[1:])
        rc, out, rep = run("--a", str(a), "--b", str(b))
        content = [c["name"] for c in rep["checks"]
                   if c["name"].startswith("content.") and not c["pass"]]
        self.assertEqual(content, [], f"C0 must be blind to this: {out}")
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["reading"], "MISALIGNED", out)
        fc = rep["structural"]["frame_correspondence"]
        self.assertGreater(fc["off_diagonal_frames"], 0, out)
        self.assertLessEqual(fc["worst_margin"], 1.0, out)
        self.assertIn("align.frames", self.structural_failures(rep))
        self.assertEqual(n - 1, len(self.frames) - 1)

    def test_a_desynced_track_moves_the_audio_argmax_off_zero(self) -> None:
        """Section 10.7 swept the lag by hand and read 0. Nothing checked it.
        A2's `0.999` cannot express it either: a drifted track and a different
        track both drag the correlation down at lag 0, which is the confusion
        A2 was registered to remove and could not."""
        rc, out, rep = run("--a", str(self.a),
                           "--b", str(self.arm("desync", audio=lambda w: np.roll(w, 480, axis=0))))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["reading"], "MISALIGNED", out)
        ac = rep["structural"]["audio_correspondence"]
        self.assertEqual(abs(ac["best_lag"]), 480, out)
        self.assertGreater(ac["r_at_best"], ac["r_at_zero"], out)
        self.assertIn("align.audio_lag", self.structural_failures(rep))

    def test_a_silenced_track_makes_the_audio_rms_coherence_exactly_one(self) -> None:
        rc, out, rep = run("--a", str(self.a),
                           "--b", str(self.arm("silent", audio=lambda w: np.zeros_like(w))))
        self.assertEqual(rc, EXIT_FAIL, out)
        co = rep["structural"]["coherence"]["audio_rms"]
        self.assertEqual(co["k"], 1.0, out)
        self.assertEqual(co["mean_b"], 0.0, out)
        self.assertIn("coherence.audio_rms", self.structural_failures(rep))

    def test_one_pixel_of_shift_is_refused_by_the_SPATIAL_check_and_by_nothing_else(self) -> None:
        """WHY `align.spatial` has to exist.

        Section 10.4 calibrates against one pixel of global horizontal shift and
        says a criterion that admitted it would not be a criterion. No coherence
        statistic can refuse it: a rigid translation removes no detail, adds no
        block grid, changes no motion energy and touches no audio, so every K
        stays near zero. It is refused at an argmin of (0, 1) instead of (0, 0),
        which is the same structural shape as the other two correspondences."""
        d = self.arm("shift", [np.roll(f, 1, axis=1) for f in self.frames])
        rc, out, rep = run("--a", str(self.a), "--b", str(d))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["reading"], "MISALIGNED", out)
        sc = rep["structural"]["spatial_correspondence"]
        self.assertEqual(sc["frames_off_origin"], sc["frames"], out)
        self.assertEqual(tuple(sc["worst_offset"]), (0, 1), out)
        self.assertEqual(self.structural_failures(rep), ["align.spatial"], out)
        for name in ("sharpness", "blockiness", "motion", "audio_rms"):
            self.assertLess(rep["structural"]["coherence"][name]["k"], 0.5,
                            f"a translation must not look DIRECTIONAL: {out}")


class TheConstantCarriesNoArgument(unittest.TestCase):
    """`0.5` is a constant in an open interval, and section 11.3 claims that ANY
    constant in it gives the same verdict on both populations. That claim is
    checkable, so it is checked rather than asserted: the same two fixtures are
    run at the two ends of the interval and the verdicts must not move.

    This is the difference between a structural check and a tuned one. A
    threshold that had been fitted to the case in front of it would flip when
    the constant moved, and a reader who suspects that this is a widened
    tolerance in disguise should read this class first.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        self.a = root / "a"
        self.frames = make_render(self.a, np.random.default_rng(20260822))
        base_wav, rate = read_wav_arr(self.a / "audio.wav")
        # THE DEGRADED POPULATION, degraded in the audio TOO. A fresh review
        # found that this class copied `audio.wav` byte-for-byte between arms,
        # so `coherence.audio_rms` was identically zero throughout it -- and
        # audio energy is the one statistic that fires on the real data, so the
        # class defending the constant never exercised the statistic the
        # constant decides. The track is attenuated 20%, which is a direction.
        self.blur = root / "blur"
        self.blur.mkdir()
        for i, f in enumerate(self.frames):
            write_ppm(self.blur / f"frame_{i:06d}.ppm", box_blur(f))
        write_wav(self.blur / "audio.wav", (base_wav * 0.8).astype(np.int64), rate)
        # THE SEPARATED POPULATION, separated in the audio too: independent
        # noise into each arm rather than one arm plus noise.
        self.n1, self.n2 = root / "n1", root / "n2"
        for d, seed in ((self.n1, 11), (self.n2, 12)):
            d.mkdir()
            r = np.random.default_rng(seed)
            for i, f in enumerate(self.frames):
                noise = (r.random(f.shape) < 0.03) * r.integers(-1, 2, f.shape)
                write_ppm(d / f"frame_{i:06d}.ppm",
                          np.clip(f.astype(np.int16) + noise, 0, 255))
            write_wav(d / "audio.wav",
                      base_wav + r.integers(-40, 41, base_wav.shape), rate)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_every_constant_in_the_interval_gives_the_same_two_verdicts(self) -> None:
        # The interval is OPEN at the bottom, and its lower end is set by the
        # statistic with the FEWEST terms: an incoherent difference concentrates
        # at N^-1/2 and fluctuates around it. In these fixtures that statistic is
        # blockiness at N=120, whose floor is 0.091 and whose observed null
        # realisation is a few times that, so a constant down at the floor sits
        # inside the null's own scatter rather than above it. At the production
        # geometry the smallest N is the audio at 376, a floor of 0.052, and the
        # three video statistics measure K between 0.009 and 0.033 there.
        for k in ("0.4", "0.5", "0.7", "0.9"):
            rc, out, rep = run("--a", str(self.a), "--b", str(self.blur),
                               "--max-coherence", k)
            self.assertEqual(rc, EXIT_FAIL, f"a blur passed at K<={k}: {out}")
            rc2, out2, rep2 = run("--a", str(self.n1), "--b", str(self.n2),
                                  "--max-coherence", k)
            self.assertEqual(rc2, EXIT_PASS,
                             f"a separated pair failed at K<={k}: {out2}")

    def test_the_audio_statistic_is_LIVE_in_this_class(self) -> None:
        """The guard on the repair above. If either fixture goes back to copying
        one wav into both arms, `coherence.audio_rms` returns to a constant zero
        and this class silently stops defending the constant on the only
        statistic that fires on the real frames."""
        _, out, deg = run("--a", str(self.a), "--b", str(self.blur))
        _, out2, sep = run("--a", str(self.n1), "--b", str(self.n2))
        self.assertEqual(deg["structural"]["coherence"]["audio_rms"]["k"], 1.0, out)
        sep_k = sep["structural"]["coherence"]["audio_rms"]["k"]
        self.assertGreater(sep_k, 0.0, f"the separated arms share one wav: {out2}")
        self.assertLess(sep_k, 0.5, out2)

    def test_the_concentration_of_the_net_is_reported(self) -> None:
        """A K built from ONE event and a K built from a shift in every term are
        the same number and are not the same evidence. The real audio result is
        the first kind, and a reader who runs the tool without opening the spec
        has to be able to see that. `top10%` is that column."""
        _, out, rep = run("--a", str(self.a), "--b", str(self.blur))
        self.assertIn("N COUNTS TERMS, NOT INDEPENDENT OBSERVATIONS", out)
        self.assertIn("top10%=", out)
        share = rep["structural"]["coherence"]["sharpness"]["net_share_top_decile"]
        self.assertIsNotNone(share)
        # A blur takes detail out of every tile, so the top tenth of the terms
        # carries only a modest share of the net: this is the SPREAD case.
        self.assertLess(share, 0.5, out)
        # And a uniform 20% attenuation of a constant-amplitude track is the
        # most spread direction there is: every window loses the same fraction,
        # so the top tenth carries about a tenth. The real 20260820 track reads
        # +0.989 on this column instead, which is how a reader tells "quieter
        # everywhere" from "quieter in the one passage that has any sound".
        a_share = rep["structural"]["coherence"]["audio_rms"]["net_share_top_decile"]
        self.assertAlmostEqual(a_share, 0.1, delta=0.06,
                               msg=f"a uniform attenuation should be SPREAD: {out}")

    def test_the_degraded_population_sits_at_the_closed_end(self) -> None:
        """The blur's K is 1.0, so no constant BELOW 1 can admit it. That is the
        algebraic half of the claim, and it is why the interval is open at 1."""
        _, out, rep = run("--a", str(self.a), "--b", str(self.blur))
        self.assertEqual(rep["structural"]["coherence"]["sharpness"]["k"], 1.0, out)


class TheRelocationIsVisible(unittest.TestCase):
    """Section 11.4 relocates six bounds out of the verdict and NOT out of the
    report. A reader must be able to see that the bar was moved, and see the old
    bounds still failing, rather than find them quietly absent.

    The failure this class refuses is the one #1668 names: a red gate made green
    by widening what it admits. Nothing here was widened, and the first test
    pins that to the byte.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        self.a = root / "a"
        self.frames = make_render(self.a, np.random.default_rng(20260822))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_no_identity_threshold_moved(self) -> None:
        """THE ANTI-WIDENING PIN. These six values are section 10.4's, and #1743
        changed WHICH checks decide the verdict and not what any of them admits.
        If a later change wants to move one, this test is what it has to argue
        with. It compares NUMBERS, because the substring form of it was itself
        widenable: see `module_float_constants`."""
        assigned = module_float_constants()
        for name, value in (("DEFAULT_MAX_MEAN_ABS", 1.0),
                            ("DEFAULT_MIN_PSNR_DB", 40.0),
                            ("DEFAULT_MIN_SSIM", 0.99),
                            ("DEFAULT_MAX_TEMPORAL_RATIO", 0.10),
                            ("DEFAULT_MIN_AUDIO_PSNR_DB", 40.0),
                            ("DEFAULT_MIN_AUDIO_CORR", 0.999)):
            self.assertIn(name, assigned, f"{name} is gone from the tool")
            self.assertEqual(assigned[name], value,
                             f"an identity threshold moved: {name} is "
                             f"{assigned[name]}, not {value}")

    def test_the_coherence_constant_cannot_be_moved_SILENTLY(self) -> None:
        """FOUND BY MUTATION, and it was green before this test existed.

        `TheConstantCarriesNoArgument` proves that 0.4, 0.5, 0.7 and 0.9 all give
        the same verdict on the two populations the criterion was built from, and
        that is exactly why it could not catch this: it passes an explicit
        `--max-coherence` every time, so the DEFAULT was free. Moving
        `DEFAULT_MAX_COHERENCE` from 0.5 to 0.99 left all 59 tests passing.

        The default is what a production run uses, and it IS load-bearing where
        a measured K is a PARTIAL direction rather than the algebraic 1. The real
        arm pair reads 0.674002 on audio energy, so 0.5 fires and 0.7 would not,
        and section 11.3 says so rather than pretending the constant is free
        everywhere. A value approaching 1 admits every partial direction and
        leaves only an exact algebraic one, which is the widening #1668 names.

        So the constant is pinned to the byte. Moving it is a change that has to
        argue with this test, which is the whole point."""
        self.assertEqual(module_float_constants().get("DEFAULT_MAX_COHERENCE"), 0.5,
                         "the coherence constant moved without an argument")

    def test_a_separated_pair_that_fails_every_identity_bound_still_exits_zero(self) -> None:
        """The relocation, demonstrated on the shape the share actually holds.

        Two independently perturbed renders, far enough apart that every
        identity bound fails, and structurally incoherent and aligned. The old
        criterion called that FAIL. The new one calls it a separation and says
        so, while printing all six failures under their own verdict."""
        arms = []
        for name, seed in (("p1", 21), ("p2", 22)):
            d = self.root / name
            d.mkdir()
            r = np.random.default_rng(seed)
            for i, f in enumerate(self.frames):
                noise = r.integers(-14, 15, f.shape)
                write_ppm(d / f"frame_{i:06d}.ppm",
                          np.clip(f.astype(np.int16) + noise, 0, 255))
            (d / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())
            arms.append(d)
        rc, out, rep = run("--a", str(arms[0]), "--b", str(arms[1]))
        self.assertEqual(rep["identity_verdict"], "DIFFERENT", out)
        for name in ("video.mean_abs", "video.psnr_min_db", "video.ssim_min"):
            self.assertIn(name, rep["identity_failed"], out)
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertEqual(rep["reading"], "SEPARATED, NOT DEGRADED", out)

    def test_the_report_says_the_bounds_were_relocated_and_prints_them(self) -> None:
        _, out, rep = run("--a", str(self.a), "--b", str(self.a))
        self.assertIn("IDENTITY bounds of section 10.4", out)
        self.assertIn("NONE of them decides the verdict any more", out)
        self.assertIn("IDENTITY IDENTICAL", out)
        self.assertIn("READING", out)
        judges = {c["name"]: c["judges"] for c in rep["checks"]}
        self.assertEqual(judges["video.bit_identical"], "identity")
        self.assertEqual(judges["align.spatial"], "treatment")
        self.assertEqual(judges["coherence.sharpness"], "treatment")

    def test_the_absolute_quality_panel_is_reported_and_never_checked(self) -> None:
        """Section 11.5 GAP 2, #1854. It is instrumentation, and a reader must be
        able to see that nothing fires on it. A panel that quietly became a gate
        would be the invented proxy the issue exists to refuse."""
        _, out, rep = run("--a", str(self.a), "--b", str(self.a))
        self.assertIn("absolute quality: REPORTED, and NOT CHECKED", out)
        names = [c["name"] for c in rep["checks"]]
        for lbl, panel in rep["absolute_quality"].items():
            self.assertFalse(panel["checked"])
            self.assertIsNotNone(panel["blockiness_grid8"])
            self.assertIsNotNone(panel["clipped_fraction"])
        self.assertEqual([n for n in names if "absolute" in n or "quality" in n], [])

class Discrimination(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rng = np.random.default_rng(20260822)
        self.a = root / "a"
        self.frames = make_render(self.a, rng)

        # SAME bytes, a second directory: two renders that agree.
        self.same = root / "same"
        make_render(self.same, np.random.default_rng(20260822))

        # DITHER: +/-1 on 3% of samples. Section 10.2 predicts single-ULP bf16
        # flips on 8.6e-05 to 3.7e-04 of attention outputs; 3% at the 8-bit
        # artefact is deliberately far MORE perturbation than that, so a pass
        # here is a pass with room.
        self.dither = root / "dither"
        self.dither.mkdir()
        drng = np.random.default_rng(7)
        for i, arr in enumerate(self.frames):
            noise = (drng.random(arr.shape) < 0.03) * drng.integers(-1, 2, arr.shape)
            write_ppm(self.dither / f"frame_{i:06d}.ppm",
                      np.clip(arr.astype(np.int16) + noise, 0, 255))
        (self.dither / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

        # STRUCTURE: one pixel of global horizontal shift.
        self.shift = root / "shift"
        self.shift.mkdir()
        for i, arr in enumerate(self.frames):
            write_ppm(self.shift / f"frame_{i:06d}.ppm", np.roll(arr, 1, axis=1))
        (self.shift / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_identical_renders_read_as_bit_identical(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.same))
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        self.assertEqual(rep["video"]["max_abs"], 0)
        self.assertEqual(rep["video"]["identical_frame_files"], FRAMES)
        # A bit-identical pair reports THAT, and does not report four thresholds
        # as passed. A reader must never mistake a vacuous bound for a read one.
        names = checks_of(rep)
        self.assertIn("video.bit_identical", names)
        self.assertNotIn("video.psnr_min_db", names)
        self.assertTrue(rep["audio"]["bit_identical"])
        self.assertEqual(rep["verdict"], "PASS")

    def test_dither_passes_every_video_check(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.dither))
        self.assertEqual(rc, EXIT_PASS, out)
        v, names = rep["video"], checks_of(rep)
        self.assertFalse(v["bit_identical"])
        for name in ("video.mean_abs", "video.psnr_min_db", "video.ssim_min",
                     "video.temporal_ratio"):
            self.assertTrue(names[name], f"{name} failed on a dither: {out}")
        # The delta is confined to 0 and 1: the shape "numerical noise" predicts.
        self.assertEqual(set(v["delta_histogram"]) - {"0", "1"}, set())
        self.assertLess(v["mean_abs"], 0.1)
        self.assertGreater(v["psnr_min_db"], 50.0)
        self.assertGreater(v["ssim_min"], 0.99)

    def test_one_pixel_shift_fails_all_four_video_checks(self) -> None:
        """The discrimination proof. All four, not one: a criterion that caught
        a global shift on only one axis would be one threshold with three
        decorations."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.shift))
        self.assertEqual(rc, EXIT_FAIL, out)
        names = checks_of(rep)
        for name in ("video.mean_abs", "video.psnr_min_db", "video.ssim_min",
                     "video.temporal_ratio"):
            self.assertFalse(names[name], f"{name} PASSED on a one-pixel shift: {out}")
        self.assertEqual(rep["verdict"], "FAIL")

    def test_the_dither_row_of_the_gated_table_is_pinned(self) -> None:
        """Section 10.4's second table is labelled "Population 2, GATED", and
        nothing gated it. `test_dither_passes_every_video_check` asserted
        `mean < 0.1`, `psnr > 50` and `ssim > 0.99`, which every one of those
        cells clears by two orders of magnitude, so a change to `make_render` --
        the seed, the motion step, the texture -- moved all four numbers with
        the suite green and the table silently wrong. Demonstrated: `motion`
        from 3 to 4 left 43 of 43 tests passing.

        These are the four cells the table quotes for the dither row, at the
        precision this file already pins SSIM to."""
        _, out, rep = run("--a", str(self.a), "--b", str(self.dither))
        v = rep["video"]
        self.assertAlmostEqual(v["mean_abs"], 0.01935944733796296, places=10, msg=out)
        self.assertAlmostEqual(v["psnr_min_db"], 65.04621554883309, places=6, msg=out)
        self.assertAlmostEqual(v["ssim_min"], 0.99998566099925, places=8, msg=out)
        self.assertAlmostEqual(v["temporal_ratio"], 0.0007008817611029302,
                               places=10, msg=out)

    def test_the_one_pixel_shift_row_of_the_gated_table_is_pinned(self) -> None:
        """The other half of the same table, and the more load-bearing half:
        section 10.4 argues the thresholds sit "between a dither and a single
        pixel of motion, nearer the dither" from exactly these four numbers, and
        `test_one_pixel_shift_fails_all_four_video_checks` asserted only that
        four booleans were False. A fixture that failed by a hair and one that
        fails by 4.5x the V4 bound are the same test to a boolean."""
        _, out, rep = run("--a", str(self.a), "--b", str(self.shift))
        v = rep["video"]
        self.assertAlmostEqual(v["mean_abs"], 16.611979166666668, places=10, msg=out)
        self.assertAlmostEqual(v["psnr_min_db"], 21.789085745489977, places=6, msg=out)
        self.assertAlmostEqual(v["ssim_min"], 0.7703600993461216, places=8, msg=out)
        self.assertAlmostEqual(v["temporal_ratio"], 0.44871294988741245,
                               places=10, msg=out)

    def test_temporal_ratio_is_normalised_by_arm_a_motion(self) -> None:
        """V4's denominator is the render's own frame-to-frame step, so the same
        absolute difference must read SMALLER against a faster-moving render.
        Without that, V4 is a constant wearing a ratio's name."""
        root = Path(self.tmp.name)
        slow, fast = root / "slow", root / "fast"
        make_render(slow, np.random.default_rng(3), motion=1)
        make_render(fast, np.random.default_rng(3), motion=12)
        for src, dst in ((slow, root / "slow_s"), (fast, root / "fast_s")):
            dst.mkdir()
            for p in sorted(src.glob("frame_*.ppm")):
                arr = np.frombuffer(
                    p.read_bytes().split(b"255\n", 1)[1], dtype=np.uint8
                ).reshape(H, W, 3)
                write_ppm(dst / p.name, np.clip(arr.astype(np.int16) + 2, 0, 255))
            (dst / "audio.wav").write_bytes((src / "audio.wav").read_bytes())
        _, _, slow_rep = run("--a", str(slow), "--b", str(root / "slow_s"))
        _, _, fast_rep = run("--a", str(fast), "--b", str(root / "fast_s"))
        self.assertAlmostEqual(slow_rep["video"]["mean_abs"],
                               fast_rep["video"]["mean_abs"], places=6)
        self.assertGreater(slow_rep["video"]["temporal_ratio"],
                           fast_rep["video"]["temporal_ratio"])

    def test_audio_divergence_fails_even_when_the_video_matches(self) -> None:
        """The DiT drives both streams. A comparison that only reads pixels
        would call a broken audio path identical."""
        root = Path(self.tmp.name)
        bad = root / "bad_audio"
        bad.mkdir()
        for p in sorted(self.a.glob("frame_*.ppm")):
            (bad / p.name).write_bytes(p.read_bytes())
        with wave.open(str(self.a / "audio.wav"), "rb") as f:
            n, rate = f.getnframes(), f.getframerate()
            raw = f.readframes(n)
        s = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, 2)
        write_wav(bad / "audio.wav", np.clip(s * 0.5, -32768, 32767), rate)
        rc, out, rep = run("--a", str(self.a), "--b", str(bad))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        self.assertFalse(checks_of(rep)["audio.psnr_db"], out)

    def test_control_arm_is_reported_separately(self) -> None:
        """The control's DIFFERENCE from its arm never enters the treatment
        verdict: it is the scale that verdict is read against, and a tool that
        folded it into the pass/fail would hide exactly the attribution it
        exists to supply.

        This used to assert that no check name contained "control" at all, which
        also pinned the hole beside it: the control's own CONTENT was unjudged,
        so a control of one-colour frames read as a very large noise floor and
        upgraded a pass to section 10.5's stronger null. What must stay outside
        the treatment verdict is the RATIO and the control-vs-arm comparison, and
        that is what is asserted here."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.dither),
                           "--control", str(self.same))
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertIn("control_video", rep)
        self.assertTrue(rep["control_video"]["bit_identical"], out)
        treatment = [c["name"] for c in rep["checks"] if c["judges"] == "treatment"]
        self.assertNotIn("control", " ".join(treatment))
        # Nothing registers the ratio or the control-vs-arm delta as a check, in
        # either list.
        for c in rep["checks"]:
            self.assertNotIn("ratio_mean_abs", c["name"])
            self.assertNotIn("control_video", c["name"])
            self.assertNotIn("control_ratio", c["name"])


class IdenticallyBroken(unittest.TestCase):
    """The hole every difference-only comparison has.

    Two all-black renders differ by zero, score infinite PSNR and SSIM 1.0, and
    would read as the strongest possible pass. A run that exited 0 having
    written frames that were all one colour has happened in this repository, so
    this is a recorded failure mode and not a hypothetical. Each arm is
    therefore judged on its own content BEFORE anything is subtracted.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _flat(self, d: Path, value: int = 0) -> None:
        d.mkdir(parents=True, exist_ok=True)
        arr = np.full((H, W, 3), value, dtype=np.uint8)
        for i in range(FRAMES):
            write_ppm(d / f"frame_{i:06d}.ppm", arr)
        t = np.arange(4800)
        write_wav(d / "audio.wav", np.stack([8000 * np.sin(t / 20.0)] * 2, axis=1))

    def test_two_all_black_renders_do_not_read_as_a_perfect_match(self) -> None:
        a, b = self.root / "a", self.root / "b"
        self._flat(a)
        self._flat(b)
        rc, out, rep = run("--a", str(a), "--b", str(b), "--label-a", "x", "--label-b", "y")
        # The difference really is nothing, and the tool says so honestly.
        self.assertTrue(rep["video"]["bit_identical"], out)
        # And it still FAILS, because neither arm rendered a picture.
        self.assertEqual(rc, EXIT_FAIL, out)
        names = checks_of(rep)
        self.assertFalse(names["content.x.not_uniform"], out)
        self.assertFalse(names["content.y.not_uniform"], out)
        self.assertFalse(names["content.x.motion"], out)
        self.assertFalse(names["content.x.distinct_frames"], out)

    def test_a_frozen_render_fails_on_motion_even_when_it_has_a_picture(self) -> None:
        """Textured but identical frames: a picture with nothing moving. The
        variance check passes and the motion check is what catches it, which is
        why both exist."""
        a = self.root / "a"
        make_render(a, np.random.default_rng(11))
        frozen = self.root / "frozen"
        frozen.mkdir()
        first = sorted(a.glob("frame_*.ppm"))[0].read_bytes()
        for i in range(FRAMES):
            (frozen / f"frame_{i:06d}.ppm").write_bytes(first)
        (frozen / "audio.wav").write_bytes((a / "audio.wav").read_bytes())
        rc, out, rep = run("--a", str(a), "--b", str(frozen),
                           "--label-a", "good", "--label-b", "frozen")
        self.assertEqual(rc, EXIT_FAIL, out)
        names = checks_of(rep)
        self.assertTrue(names["content.frozen.not_uniform"], out)
        self.assertFalse(names["content.frozen.motion"], out)
        self.assertFalse(names["content.frozen.distinct_frames"], out)
        self.assertTrue(names["content.good.motion"], out)

    def test_a_healthy_pair_passes_every_content_check(self) -> None:
        a, b = self.root / "a", self.root / "b"
        make_render(a, np.random.default_rng(13))
        make_render(b, np.random.default_rng(13))
        rc, out, rep = run("--a", str(a), "--b", str(b), "--label-a", "p", "--label-b", "q")
        self.assertEqual(rc, EXIT_PASS, out)
        names = checks_of(rep)
        for label in ("p", "q"):
            for check in ("not_uniform", "distinct_frames", "motion"):
                self.assertTrue(names[f"content.{label}.{check}"],
                                f"content.{label}.{check} failed on a healthy render: {out}")
        # C0 registers THREE checks per arm, not four. "frames written" was a
        # fourth that could never be False -- an arm with no frames is refused
        # at exit 2 before any check is built -- and a check that cannot fail is
        # a decoration in a table whose whole value is that each row can.
        self.assertNotIn("content.p.frames", names)
        self.assertNotIn("content.q.frames", names)


class Refusal(unittest.TestCase):
    """Exit 2 is a SEPARATE verdict from exit 1, and the separation is the point.

    Exit 1 means "the pixels differ", which is a statement about a render that
    happened. Exit 0 and exit 1 are both readings of a completed experiment. An
    input that cannot be read is not a reading at all, and a broken render that
    reported the same status as a divergent one would be indistinguishable from
    the finding this tool exists to make -- at exactly the moment nobody is
    looking closely, because a 1 is the expected answer for a while.

    The docstring promised this and only a missing top-level DIRECTORY did it.
    An empty arm, a frame-count mismatch, frames that do not correspond, a
    truncated frame and a 16-bit frame all exited 1 and wrote no JSON.
    """

    def test_missing_directory_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a = Path(t) / "a"
            make_render(a, np.random.default_rng(1))
            rc, out, _ = run("--a", str(a), "--b", str(Path(t) / "absent"))
        self.assertEqual(rc, EXIT_UNREADABLE, out)

    def test_an_empty_arm_directory_exits_two(self) -> None:
        """A render that wrote nothing. Every video statistic is undefined, so
        there is no comparison to fail: this is a refusal, not a verdict."""
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            b.mkdir()
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("no frame", out.lower())
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_frame_count_mismatch_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            next(iter(sorted(b.glob("frame_*.ppm")))).unlink()
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("frame count differs", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_frames_that_do_not_correspond_by_name_exit_two(self) -> None:
        """The pairing is `sorted()` against `sorted()`. Equal counts are not
        equal frames: an arm that dropped frame 000000 and gained a frame 000049
        pairs every index against its neighbour and reports the render's own
        motion as the arm-to-arm delta -- a large, plausible, entirely spurious
        number."""
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            first = sorted(b.glob("frame_*.ppm"))[0]
            first.rename(b / "frame_000099.ppm")
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("frame_000099.ppm", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_truncated_frame_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            p = sorted(b.glob("frame_*.ppm"))[2]
            p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2])
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("truncated", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_sixteen_bit_frame_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            p = sorted(b.glob("frame_*.ppm"))[1]
            p.write_bytes(b"P6\n%d %d\n65535\n" % (W, H) + b"\0" * (W * H * 6))
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIn("maxval", out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_geometry_mismatch_between_arms_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            make_render(a, np.random.default_rng(1))
            make_render(b, np.random.default_rng(1))
            p = sorted(b.glob("frame_*.ppm"))[0]
            write_ppm(p, np.zeros((H // 2, W, 3), dtype=np.uint8) + 40)
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_UNREADABLE, out)
        self.assertIsNone(rep, "a refused run must not write a verdict")

    def test_a_threshold_failure_is_still_exit_one(self) -> None:
        """The other half of the separation. Widening exit 2 to cover a
        divergent render would erase the finding instead of the ambiguity."""
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            frames = make_render(a, np.random.default_rng(1))
            b.mkdir()
            for i, arr in enumerate(frames):
                write_ppm(b / f"frame_{i:06d}.ppm", np.roll(arr, 1, axis=1))
            (b / "audio.wav").write_bytes((a / "audio.wav").read_bytes())
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["verdict"], "FAIL")

    def test_ppm_reader_reads_width_before_height(self) -> None:
        """A square fixture cannot see this, and every fixture in this file was
        square-ish. The header is `P6 W H maxval` and the array is (H, W, 3)."""
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ppm"
            px = bytes(range(7 * 3 * 3))
            p.write_bytes(b"P6\n7 3\n255\n" + px)
            arr = _load_tool().read_ppm(str(p))
        self.assertEqual(arr.shape, (3, 7, 3))
        self.assertEqual(tuple(int(v) for v in arr[0, 0]), (0, 1, 2))
        self.assertEqual(tuple(int(v) for v in arr[1, 0]), (21, 22, 23))

    def test_ppm_reader_refuses_a_maxval_it_cannot_scale(self) -> None:
        """Every threshold below is in 8-bit levels. A 16-bit PPM would make
        each one mean something else, so it is refused rather than rescaled."""
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ppm"
            p.write_bytes(b"P6\n2 2\n65535\n" + b"\0" * 24)
            mod = _load_tool()
            with self.assertRaises(ValueError):
                mod.read_ppm(str(p))

    def test_ppm_reader_refuses_a_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.ppm"
            p.write_bytes(b"P6\n8 8\n255\n" + b"\0" * 10)
            mod = _load_tool()
            with self.assertRaises(ValueError):
                mod.read_ppm(str(p))


class Audio(unittest.TestCase):
    """A1 and A2 are two checks and they were pinned by one fixture.

    The only divergent-audio fixture was a 0.5x scale, and Pearson r of a
    waveform against a scaled copy of itself is EXACTLY 1.0. So A2 could not
    fail on anything this suite owned: `DEFAULT_MIN_AUDIO_CORR` moved from 0.999
    to -1.0 stayed green, and deleting the check entirely stayed green. Section
    10.4 says A2 exists for "a waveform that has drifted in time", so the
    fixture that reads it is a waveform that has drifted in time.
    """

    def _video_only_pair(self, root: Path) -> tuple[Path, Path]:
        a, b = root / "a", root / "b"
        make_render(a, np.random.default_rng(17))
        b.mkdir()
        for p in sorted(a.glob("frame_*.ppm")):
            (b / p.name).write_bytes(p.read_bytes())
        return a, b

    def test_a_time_shifted_waveform_fails_correlation_while_psnr_passes(self) -> None:
        """The two checks DISAGREE on this fixture, and that is the measured
        fact rather than a hoped-for one: one sample of shift on a 20-sample
        radian scale is 0.05 rad, which is 41.28 dB (A1 passes, floor 40) and
        r = 0.99875 (A2 fails, floor 0.999). A tool holding only A1 would call
        this stream identical."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            a, b = self._video_only_pair(root)
            n = np.arange(4800)
            write_wav(a / "audio.wav",
                      np.stack([8000 * np.sin(n / 20.0), 8000 * np.cos(n / 20.0)], axis=1))
            write_wav(b / "audio.wav",
                      np.stack([8000 * np.sin((n + 1) / 20.0),
                                8000 * np.cos((n + 1) / 20.0)], axis=1))
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        names = checks_of(rep)
        self.assertTrue(names["audio.psnr_db"],
                        f"A1 was expected to PASS on a one-sample drift: {out}")
        self.assertFalse(names["audio.pearson_r"],
                         f"A2 was expected to FAIL on a one-sample drift: {out}")
        self.assertGreater(rep["audio"]["psnr_db"], 40.0)
        self.assertLess(rep["audio"]["psnr_db"], 42.0)
        self.assertLess(rep["audio"]["pearson_r"], 0.999)
        self.assertGreater(rep["audio"]["pearson_r"], 0.998)

    def test_absent_audio_is_a_failed_check_and_never_a_pass(self) -> None:
        """Two arms whose video agrees bit-for-bit and that wrote no audio at
        all. `present` must be False and the verdict must be FAIL: the DiT
        drives both streams, so a silent half is an unmeasured half."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            a, b = self._video_only_pair(root)
            (a / "audio.wav").unlink()
            rc, out, rep = run("--a", str(a), "--b", str(b))
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertTrue(rep["video"]["bit_identical"], out)
        self.assertFalse(rep["audio"]["present"], out)
        self.assertNotIn("comparable", rep["audio"])
        self.assertNotIn("bit_identical", rep["audio"])
        self.assertFalse(checks_of(rep)["audio.present"], out)
        self.assertEqual(rep["verdict"], "FAIL")


class Control(unittest.TestCase):
    """WHICH ARM the control repeats is now stated, not conventional.

    The tool computed the control as `compare_video(a, control, ...)` and said
    so only in a docstring. The harness passed `--a naive --b flash --control
    flash-ctl`, so the "run-to-run noise floor" was a SECOND naive-vs-flash
    comparison. It necessarily read about the same size as the treatment, and
    section 10.5's second branch would then have published "indistinguishable
    from run-to-run nondeterminism" whatever the kernel did. A silent convention
    a caller can invert is not a convention.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.a = root / "a"
        frames = make_render(self.a, np.random.default_rng(21))
        # b differs from a everywhere by ONE level, which every threshold
        # admits, so the pair is a PASS and the control is read on a passing
        # result -- the only case section 10.5 has two readings for. `ctl` is a
        # byte-for-byte repeat of b.
        self.b, self.ctl = root / "b", root / "ctl"
        for d in (self.b, self.ctl):
            d.mkdir()
            for i, arr in enumerate(frames):
                write_ppm(d / f"frame_{i:06d}.ppm", np.clip(arr.astype(np.int16) + 1, 0, 255))
            (d / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_control_of_b_compares_the_control_against_arm_b(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.b),
                           "--control", str(self.ctl), "--control-of", "b",
                           "--label-a", "naive", "--label-b", "flash",
                           "--label-control", "flash-ctl")
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertEqual(rep["control_of"], "b")
        self.assertEqual(rep["control_video"]["label_a"], "flash")
        self.assertTrue(rep["control_video"]["bit_identical"],
                        f"the control repeats FLASH and must be compared to it: {out}")
        self.assertIn("flash-ctl", out)
        # The block says, in words, which arm it was read against.
        self.assertRegex(out, r"control .*flash-ctl.* repeats arm B \(flash\)")

    def test_control_of_defaults_to_a_and_says_so(self) -> None:
        rc, out, rep = run("--a", str(self.a), "--b", str(self.b),
                           "--control", str(self.ctl),
                           "--label-a", "naive", "--label-b", "flash",
                           "--label-control", "flash-ctl")
        self.assertEqual(rep["control_of"], "a")
        self.assertEqual(rep["control_video"]["label_a"], "naive")
        self.assertFalse(rep["control_video"]["bit_identical"], out)
        self.assertRegex(out, r"control .*flash-ctl.* repeats arm A \(naive\)")

    def test_the_control_to_treatment_ratio_is_computed_not_eyeballed(self) -> None:
        """Section 10.5 selects between two OPPOSITE published verdicts on
        "control is 0" against "control comparable to the delta", and nothing
        computed that comparison. It is reported, never checked: it chooses
        between two readings, not between pass and fail."""
        rc, out, rep = run("--a", str(self.a), "--b", str(self.b),
                           "--control", str(self.ctl), "--control-of", "b")
        self.assertEqual(rc, EXIT_PASS, out)
        r = rep["control_ratio"]
        # The control repeats B and IS B, so its delta against B is zero and the
        # ratio is exactly 0: the "noise floor is zero" branch, arithmetically.
        self.assertEqual(r["control_mean_abs_luma"], 0.0)
        self.assertAlmostEqual(r["treatment_mean_abs_luma"],
                               rep["video"]["mean_abs_luma"], places=12)
        self.assertEqual(r["ratio_mean_abs_luma"], 0.0)
        self.assertEqual(r["ratio_mean_abs_rgb"], 0.0)
        self.assertIsNone(r["undefined"])
        self.assertIn("control/treatment", out)
        # And it is REPORTED, not gated: no check carries the RATIO, in either
        # list. This is narrower than it was, deliberately. It used to assert
        # that no check name contained "control" at all, which also pinned the
        # absence of the control's own CONTENT checks -- see
        # `ControlContent` -- and the control's content is now judged.
        for c in rep["checks"]:
            self.assertNotIn("control_ratio", c["name"])
            self.assertNotIn("ratio_mean_abs", c["name"])
        self.assertNotIn("control", " ".join(
            c["name"] for c in rep["checks"] if c["judges"] == "treatment"))

    def test_the_ratio_is_undefined_rather_than_a_division_by_zero(self) -> None:
        """The expected case for a bit-identical treatment, which is exactly
        what section 10.2 predicts this experiment will NOT see -- so it is the
        case a reader would meet only when something else went wrong."""
        root = Path(self.tmp.name)
        same = root / "same"
        same.mkdir()
        for p in sorted(self.a.glob("*")):
            (same / p.name).write_bytes(p.read_bytes())
        rc, out, rep = run("--a", str(self.a), "--b", str(same),
                           "--control", str(self.ctl))
        self.assertEqual(rc, EXIT_PASS, out)
        r = rep["control_ratio"]
        self.assertIsNone(r["ratio_mean_abs_luma"])
        self.assertIsNotNone(r["undefined"])
        self.assertGreater(r["control_mean_abs_luma"], 0.0)
        self.assertIn("undefined", out)


class ControlContent(unittest.TestCase):
    """The control is judged on its OWN content, and a broken one is its own status.

    Section 10.4's C0 rationale -- "a difference cannot tell two good renders
    from two identically broken ones" -- applies to the control-vs-arm
    comparison with identical force, and until this revision it was applied only
    to arms A and B. `arm_content(control)` was computed and PRINTED and never
    entered `checks`, so a control of six one-colour frames left the tool at
    exit 0 with the verdict PASS and `R = 112.77`, which section 10.5 reads as
    `R >= 0.5`: **indistinguishable from run-to-run nondeterminism**, the
    STRONGER of its two null readings. A control that rendered no picture at all
    therefore upgraded the published conclusion.

    A degenerate control is NOT "visibly different". Section 10.5 maps any
    failing check to that reading, and it is a finding about a change already on
    `main`; a control with no picture in it is a broken EXPERIMENT, and mapping
    one onto the other would replace one mis-reading with a second. So it is a
    THIRD status, exit 3, carrying its own verdict token, which is neither
    `PASS` nor `FAIL` and cannot be read as either.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.a = root / "a"
        frames = make_render(self.a, np.random.default_rng(29))
        # A healthy treatment pair: one level apart everywhere, which every
        # threshold admits. The control is read only on a PASS, so the fixture
        # has to be one.
        self.b = root / "b"
        self.b.mkdir()
        for i, arr in enumerate(frames):
            write_ppm(self.b / f"frame_{i:06d}.ppm", np.clip(arr.astype(np.int16) + 1, 0, 255))
        (self.b / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())
        # A healthy control: a byte-for-byte repeat of arm A, which is the arm
        # `--control-of a` says it repeats.
        self.ctl = root / "ctl"
        self.ctl.mkdir()
        for p in sorted(self.a.glob("*")):
            (self.ctl / p.name).write_bytes(p.read_bytes())
        # A DEGENERATE control: the same geometry, one colour, no picture.
        self.flat = root / "flat"
        self.flat.mkdir()
        flat = np.full((H, W, 3), 17, dtype=np.uint8)
        for i in range(FRAMES):
            write_ppm(self.flat / f"frame_{i:06d}.ppm", flat)
        (self.flat / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, control: Path):
        return run("--a", str(self.a), "--b", str(self.b), "--control", str(control),
                   "--control-of", "a", "--label-a", "flash", "--label-b", "naive",
                   "--label-control", "flash-ctl")

    def test_a_control_that_rendered_no_picture_is_not_a_pass(self) -> None:
        rc, out, rep = self._run(self.flat)
        self.assertEqual(rc, EXIT_CONTROL_DEGENERATE, out)
        self.assertEqual(rep["verdict"], "CONTROL_DEGENERATE", out)
        self.assertEqual(rep["treatment_verdict"], "PASS", out)
        self.assertEqual(rep["control_verdict"], "DEGENERATE", out)
        # The headline token is neither of the two a reader would act on.
        self.assertNotIn("VERDICT PASS", out)
        self.assertNotIn("VERDICT FAIL", out)
        self.assertIn("VERDICT CONTROL_DEGENERATE", out)

    def test_the_controls_own_content_checks_are_registered(self) -> None:
        _, out, rep = self._run(self.flat)
        names = checks_of(rep)
        for check in ("not_uniform", "distinct_frames", "motion"):
            self.assertIn(f"content.flash-ctl.{check}", names,
                          f"the control's {check} check was never built: {out}")
            self.assertFalse(names[f"content.flash-ctl.{check}"], out)
        # And they judge the CONTROL, not the treatment: the entry says which.
        judges = {c["name"]: c["judges"] for c in rep["checks"]}
        self.assertEqual(judges["content.flash-ctl.not_uniform"], "control")
        self.assertEqual(judges["content.flash.not_uniform"], "treatment")

    def test_the_ratio_is_marked_unreadable_when_the_control_is_degenerate(self) -> None:
        """`R` is still computed and still printed -- the arithmetic was never
        the defect -- and it carries a stated reason why it may not be read."""
        _, out, rep = self._run(self.flat)
        r = rep["control_ratio"]
        self.assertIsNotNone(r["ratio_mean_abs_luma"])
        self.assertGreater(r["ratio_mean_abs_luma"], 0.5,
                           "the fixture must land in section 10.5's stronger-null band")
        self.assertIsNotNone(r["unusable"], "R must say why it cannot be read")
        self.assertIn("not readable", out.lower())

    def test_a_healthy_control_still_passes_at_exit_zero(self) -> None:
        rc, out, rep = self._run(self.ctl)
        self.assertEqual(rc, EXIT_PASS, out)
        self.assertEqual(rep["verdict"], "PASS", out)
        self.assertEqual(rep["control_verdict"], "USABLE", out)
        names = checks_of(rep)
        for check in ("not_uniform", "distinct_frames", "motion"):
            self.assertTrue(names[f"content.flash-ctl.{check}"], out)
        self.assertIsNone(rep["control_ratio"]["unusable"], out)

    def test_a_treatment_failure_outranks_a_degenerate_control(self) -> None:
        """Exit 1 says "the two renders differ", which is established WITHOUT the
        control and is a finding about a change already on `main`. A broken
        control must not hide it, so exit 3 is reserved for the case where the
        treatment passed and only the READING of that pass is at stake."""
        root = Path(self.tmp.name)
        shift = root / "shift"
        shift.mkdir()
        for p in sorted(self.a.glob("frame_*.ppm")):
            arr = np.frombuffer(p.read_bytes().split(b"255\n", 1)[1],
                                dtype=np.uint8).reshape(H, W, 3)
            write_ppm(shift / p.name, np.roll(arr, 1, axis=1))
        (shift / "audio.wav").write_bytes((self.a / "audio.wav").read_bytes())
        rc, out, rep = run("--a", str(self.a), "--b", str(shift),
                           "--control", str(self.flat), "--control-of", "a",
                           "--label-a", "flash", "--label-b", "naive",
                           "--label-control", "flash-ctl")
        self.assertEqual(rc, EXIT_FAIL, out)
        self.assertEqual(rep["verdict"], "FAIL", out)
        self.assertEqual(rep["control_verdict"], "DEGENERATE", out)
        # Both facts are stated. Neither is inferred from the other.
        self.assertIn("CONTROL DEGENERATE", out)
        self.assertIn("VERDICT FAIL", out)

    def test_the_report_says_in_words_which_checks_decide_what(self) -> None:
        """`.agents/verification.md`: an instrument states what it compared
        against what, in its own output. A reader who cannot see which list
        drives the exit status cannot audit the one that does not."""
        _, out, _ = self._run(self.ctl)
        self.assertIn("these decide the verdict", out)
        self.assertIn("do NOT decide", out)
        self.assertIn("noise floor", out)


def _load_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ltx25_render_compare", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _texture(seed: int = 20260822) -> np.ndarray:
    """One textured plane, as float64 luma levels. The SSIM fixtures below are
    all derived from this one array so that every pinned number is reproducible
    from the seed printed here and nothing else."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 127 + 60 * np.sin(xx / 4.0) + 40 * np.cos(yy / 3.0) + rng.integers(-20, 21, (H, W))
    return np.clip(base, 0, 255).astype(np.uint8).astype(np.float64)


class Metrics(unittest.TestCase):
    def test_ssim_of_a_frame_with_itself_is_one(self) -> None:
        mod = _load_tool()
        rng = np.random.default_rng(5)
        a = rng.integers(0, 256, (H, W)).astype(np.float64)
        self.assertAlmostEqual(mod.ssim(a, a), 1.0, places=9)

    def test_psnr_of_zero_error_is_infinite_not_a_large_number(self) -> None:
        mod = _load_tool()
        self.assertEqual(mod.psnr_from_mse(0.0), float("inf"))

    def test_psnr_matches_its_definition(self) -> None:
        mod = _load_tool()
        # A uniform error of exactly 1 level: 20*log10(255) = 48.1308 dB.
        self.assertAlmostEqual(mod.psnr_from_mse(1.0), 48.13080361, places=6)


class SsimIsTheOneItNames(unittest.TestCase):
    """V3 names a specific SSIM, and the name is load-bearing.

    Section 10.4 says "Wang et al. 2004, 11x11 Gaussian sigma=1.5 on luma", and
    says it because scikit-image's 7x7 uniform default disagrees in the third
    decimal while a `0.99` gate reads to the sixth. That sentence was a claim: a
    test that only asked "is SSIM(a, a) == 1" stayed green with sigma at 3.0 or
    0.5, the window at 7 or 3, either stabilising constant at 0.5 or at 0, the
    second separable pass blurring the wrong axis or deleted outright, and the
    luma weights replaced by red-only or a flat third. Every one of those is a
    different metric wearing the same name.

    scikit-image is NOT available here and must stay absent, so the criterion is
    pinned by PROPERTIES and by values computed at this revision:

      the taps themselves      catches sigma and the window size directly
      the impulse response     catches either separable pass, because a
                               separable blur of a unit impulse IS outer(k, k)
      its two profiles         catches a second pass on the WRONG axis, which
                               leaves the response summing to 1 and asymmetric
      three fixture SSIMs      catches C1 and C2, which change no property above
                               and only move the number
    """

    def test_gaussian_taps_are_the_eleven_of_sigma_1_5(self) -> None:
        mod = _load_tool()
        k = mod._gauss1d()
        self.assertEqual(len(k), 11, "the window is 11 wide, not scikit-image's 7")
        expected = [
            0.00102838008447911, 0.007598758135239185, 0.03600077212843083,
            0.10936068950970002, 0.2130055377112537, 0.26601172486179436,
            0.2130055377112537, 0.10936068950970002, 0.03600077212843083,
            0.007598758135239185, 0.00102838008447911,
        ]
        np.testing.assert_allclose(k, expected, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(k.sum()), 1.0, places=15)

    def test_the_taps_fit_the_sigma_they_claim(self) -> None:
        """The second moment of the discrete taps. A truncated 11-tap Gaussian
        loses a little of the tail, so it reads 1.4978 rather than 1.5000 -- and
        that residual is itself the evidence the window is 11 rather than
        infinite."""
        mod = _load_tool()
        k = mod._gauss1d()
        r = np.arange(len(k), dtype=np.float64) - (len(k) - 1) / 2.0
        fitted = float(np.sqrt((r ** 2 * k).sum()))
        self.assertAlmostEqual(fitted, 1.4978283460942616, places=12)
        self.assertLess(abs(fitted - 1.5), 3e-3)

    def test_blur_of_a_unit_impulse_is_the_separable_outer_product(self) -> None:
        mod = _load_tool()
        k = mod._gauss1d()
        n = 31
        imp = np.zeros((n, n))
        imp[n // 2, n // 2] = 1.0
        out = mod._blur(imp, k)
        lo, hi = n // 2 - 5, n // 2 + 6
        np.testing.assert_allclose(out[lo:hi, lo:hi], np.outer(k, k), rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=12)
        # Nothing outside the 11x11 support: a wider window would spill here.
        support = np.zeros_like(out, dtype=bool)
        support[lo:hi, lo:hi] = True
        self.assertEqual(float(np.abs(out[~support]).max()), 0.0)

    def test_the_impulse_response_is_identical_along_both_axes(self) -> None:
        """The mutation this exists for: a second separable pass that blurs axis
        0 again instead of axis 1, or is deleted. Both leave the response
        summing to 1.0 and both leave SSIM(a, a) == 1, and both make the metric
        anisotropic -- which is what this reads."""
        mod = _load_tool()
        k = mod._gauss1d()
        n = 31
        imp = np.zeros((n, n))
        imp[n // 2, n // 2] = 1.0
        out = mod._blur(imp, k)
        np.testing.assert_allclose(out[n // 2, :], out[:, n // 2], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(out[n // 2, :], out[n // 2, ::-1], rtol=0.0, atol=1e-15)

    def test_ssim_of_a_one_pixel_shift_is_pinned(self) -> None:
        """The value section 10.4's fixture row quotes. It moves under every one
        of the ten mutations in this class's docstring."""
        mod = _load_tool()
        a = _texture()
        self.assertAlmostEqual(mod.ssim(a, np.roll(a, 1, axis=1)),
                               0.7029544035134992, places=8)

    def test_ssim_of_a_five_level_luminance_shift_is_pinned(self) -> None:
        """A pure luminance offset: the variance terms are identical on both
        sides, so the contrast and structure factors are exactly 1 and this
        number is the LUMINANCE factor alone. It is what pins C1."""
        mod = _load_tool()
        a = _texture()
        self.assertAlmostEqual(mod.ssim(a, np.clip(a + 5, 0, 255)),
                               0.9987425512854893, places=8)

    def test_ssim_of_a_contrast_scaled_pair_is_pinned(self) -> None:
        """The mirror of the test above: the means are identical and the
        variances are not, so this number moves with C2 and not with C1."""
        mod = _load_tool()
        a = _texture()
        c = np.clip(127 + 0.9 * (a - 127), 0, 255)
        self.assertAlmostEqual(mod.ssim(a, c), 0.9931050937084597, places=8)

    def test_luma_is_the_rec_601_triple(self) -> None:
        mod = _load_tool()
        px = lambda r, g, b: float(mod.luma(np.array([[[r, g, b]]], dtype=np.uint8))[0, 0])
        # Each primary on its own reads back its own weight, which is the triple
        # asserted rather than described: red-only, a flat third, and an R/B swap
        # each fail a different line here.
        self.assertAlmostEqual(px(255, 0, 0) / 255.0, 0.299, places=12)
        self.assertAlmostEqual(px(0, 255, 0) / 255.0, 0.587, places=12)
        self.assertAlmostEqual(px(0, 0, 255) / 255.0, 0.114, places=12)
        self.assertAlmostEqual(px(10, 20, 30), 0.299 * 10 + 0.587 * 20 + 0.114 * 30,
                               places=12)
        self.assertAlmostEqual(px(10, 20, 30), 18.15, places=12)


if __name__ == "__main__":
    unittest.main()
