#!/usr/bin/env python3
"""`LTX25-ADHERENCE-DETAIL-LOSS` (#2513): the instrument that decides whether the
render's smoothness CAUSES its adherence gap has to be able to give the wrong
answer, and these cases prove each of its parts can.

WHY A SUITE AT ALL FOR A DIAGNOSIS. This row publishes a NEGATIVE result: the
spectrum says our render is not smoother, the within-render correlation points
the wrong way, and blurring the reference does not reproduce the gap. A negative
result is exactly the shape a broken instrument produces for free. A radial
profile that returns noise, a correlation that returns zero because one input is
constant, and a blur that does not blur all read as "no effect". So each case
below feeds the function a signal whose answer is known in closed form, and
several of them assert the instrument moves in the direction the falsified
hypothesis would have needed. An instrument that cannot produce the positive
answer has not refuted it.

THE DIGEST GUARD IS THE FIRST CASE, not a footnote. This repository has gated
against the wrong checkpoint before, and the frames live on a soft-mounted CIFS
share where a read can truncate. `verify_ours` refuses a set whose concatenated
digest is not the one `ltx25-prompt-adherence.md` records for the frames W3
scored, so a measurement taken on some other render cannot be compared with that
verdict by accident.

Needs numpy and nothing else: no checkpoint, no frames, no network, no GPU. Every
signal is synthesised here.
"""
from __future__ import annotations

import hashlib
import io
import json
import importlib.util
import os
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - the lane installs it
    print("SKIP: numpy is not importable, and every case here computes over arrays")
    raise SystemExit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cmp = _load("ltx25_render_compare_for_detail",
             os.path.join("scripts", "ltx25-render-compare.py"))
D = _load("ltx25_detail_loss", os.path.join("scripts", "ltx25-adherence-detail-loss.py"))


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_ppm(path: str, a: np.ndarray) -> None:
    h, w, _ = a.shape
    with open(path, "wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        fh.write(np.ascontiguousarray(a.astype(np.uint8)).tobytes())


class IdentityRefusals(unittest.TestCase):
    """The set has to be the set W3 scored, or nothing may be compared to it."""

    def _dir(self, n: int = 3, seed: int = 0):
        d = tempfile.mkdtemp()
        rng = np.random.default_rng(seed)
        for i in range(n):
            write_ppm(os.path.join(d, f"frame_{i:06d}.ppm"),
                      rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
        return d

    def test_a_wrong_set_is_refused_and_names_the_recorded_digest(self):
        with self.assertRaises(D.UnreadableInput) as cm:
            D.verify_ours(self._dir())
        self.assertIn(D.OURS_SET_DIGEST, str(cm.exception))

    def test_the_recorded_digest_is_the_concatenated_form_and_not_a_hash_of_bytes(self):
        # The guard is only meaningful if it recomputes the SAME reduction the
        # spec published. Build a directory, compute the reduction by hand, and
        # require the checker to accept exactly that value -- so a checker that
        # hashed the frames some other way would fail here rather than silently
        # guard a different quantity.
        d = self._dir(seed=7)
        paths = D.frame_paths(d)
        lines = "".join(f"{D.sha256_file(p)}  {os.path.basename(p)}\n" for p in paths)
        want = hashlib.sha256(lines.encode("ascii")).hexdigest()
        saved = D.OURS_SET_DIGEST
        try:
            D.OURS_SET_DIGEST = want
            self.assertEqual(D.verify_ours(d)["set_digest"], want)
        finally:
            D.OURS_SET_DIGEST = saved

    def test_one_flipped_byte_in_one_frame_is_refused(self):
        d = self._dir(seed=11)
        paths = D.frame_paths(d)
        lines = "".join(f"{D.sha256_file(p)}  {os.path.basename(p)}\n" for p in paths)
        saved = D.OURS_SET_DIGEST
        try:
            D.OURS_SET_DIGEST = hashlib.sha256(lines.encode("ascii")).hexdigest()
            self.assertTrue(D.verify_ours(d)["verified"])
            with open(paths[-1], "r+b") as fh:
                fh.seek(-1, os.SEEK_END)
                b = fh.read(1)
                fh.seek(-1, os.SEEK_END)
                fh.write(bytes([b[0] ^ 0x01]))
            with self.assertRaises(D.UnreadableInput):
                D.verify_ours(d)
        finally:
            D.OURS_SET_DIGEST = saved

    def test_a_reference_frame_whose_digest_is_absent_is_unidentified(self):
        d = self._dir(n=2, seed=3)
        sums = os.path.join(d, "SUMS")
        with open(sums, "w", encoding="utf-8") as fh:
            fh.write("# only one of the two\n")
            p = D.frame_paths(d)[0]
            fh.write(f"{D.sha256_file(p)}  {os.path.basename(p)}\n")
        with self.assertRaises(D.UnreadableInput):
            D.verify_reference(d, sums)


class ToneFit(unittest.TestCase):
    """A gamma that IS there must be recovered, or the tone arm proves nothing."""

    def test_a_known_gamma_is_recovered_from_the_quantile_curve(self):
        # GREY, deliberately. The fit reads LUMA quantiles, and luma is a linear
        # mix of three channels while a gamma is not, so
        # `mix(c_i ** g) != mix(c_i) ** g` and a colour fixture would measure the
        # mixing bias rather than the recovery. That approximation is a real
        # limit of the tone arm and the report states it; this case isolates the
        # fit itself by handing it an image on which the two agree exactly.
        rng = np.random.default_rng(2)
        g = np.clip(rng.normal(0.45, 0.16, (48, 64, 1)), 0.02, 0.98)
        base = np.repeat(g, 3, axis=2)
        src = (255.0 * base).astype(np.uint8)
        dst = np.clip(np.rint(255.0 * 0.93 * (base ** 1.35)), 0, 255).astype(np.uint8)
        fit = D.fit_gain_gamma(D.luma_quantiles([src]), D.luma_quantiles([dst]))
        self.assertAlmostEqual(fit["gamma"], 1.35, delta=0.06)
        self.assertAlmostEqual(fit["gain"], 0.93, delta=0.05)

    def test_applying_the_fit_moves_the_level_toward_the_target(self):
        rng = np.random.default_rng(4)
        base = np.clip(rng.normal(0.5, 0.15, (48, 64, 3)), 0.02, 0.98)
        src = (255.0 * base).astype(np.uint8)
        dst = np.clip(np.rint(255.0 * 0.85 * (base ** 1.2)), 0, 255).astype(np.uint8)
        fit = D.fit_gain_gamma(D.luma_quantiles([src]), D.luma_quantiles([dst]))
        got = D.apply_gain_gamma(src, fit["gain"], fit["gamma"])
        before = abs(float(_cmp.luma(src).mean()) - float(_cmp.luma(dst).mean()))
        after = abs(float(_cmp.luma(got).mean()) - float(_cmp.luma(dst).mean()))
        self.assertLess(after, before / 4.0)

    def test_a_collapsed_render_cannot_be_fitted_and_says_so(self):
        flat = np.zeros(101)
        with self.assertRaises(D.UnreadableInput):
            D.fit_gain_gamma(flat, flat)


class Spectrum(unittest.TestCase):
    """A profile that cannot locate a frequency cannot localise a deficit."""

    @staticmethod
    def _sine(f: float, h: int = 96, w: int = 128) -> np.ndarray:
        x = np.arange(w)[None, :]
        return 128.0 + 100.0 * np.sin(2.0 * np.pi * f * x) * np.ones((h, 1))

    def test_a_pure_sinusoid_lands_in_the_bin_that_names_its_frequency(self):
        # RAW, because this case is about the radial BINNING and nothing else.
        # `periodic_component` deliberately injects a low-frequency counter-term
        # whose size scales with the boundary jump, and a high-frequency sinusoid
        # is almost all boundary jump, so under that convention the counter-term
        # outweighs the signal's own ring mean and the peak lands in bin 0. That
        # is the decomposition behaving correctly on a pathological fixture, not
        # a binning defect, and `test_the_periodic_counter_term_is_real` below
        # pins it as a known property rather than leaving it to be rediscovered.
        for f in (10 / 128, 24 / 128, 42 / 128):
            c, p = D.radial_power(self._sine(f), None, nbins=64, convention="raw")
            self.assertAlmostEqual(float(c[int(np.nanargmax(p))]), f, delta=0.012,
                                   msg=f"peak misplaced for f={f}")

    def test_the_periodic_counter_term_is_real_and_scales_with_the_boundary(self):
        # WHY THE REPORTED ESTIMATOR IS WELCH AND NOT THIS. `p = u - s` removes
        # the wrap step exactly, but `s` carries it into a low-frequency
        # counter-term, and the two renders do not share a boundary jump -- so
        # this convention is not neutral between them either.
        x = np.arange(128)[None, :]
        hi_f = 128.0 + 100.0 * np.sin(2 * np.pi * (24 / 128) * x) * np.ones((96, 1))
        c, p = D.radial_power(hi_f, None, nbins=64, convention="periodic")
        self.assertLess(float(c[int(np.nanargmax(p))]), 0.05,
                        "the counter-term no longer dominates; the note in "
                        "welch_psd's docstring and this row's Outcome would "
                        "need re-deriving")

    def test_the_dc_term_is_not_reported_as_signal(self):
        flat = np.full((96, 128), 200.0)
        _, p = D.radial_power(flat, None, nbins=32, convention="raw")
        self.assertLess(float(np.nansum(p)), 1e-6)

    def test_welch_never_sees_the_frame_border(self):
        # THE PROPERTY THE WHOLE CORRECTION RESTS ON. Paint a cliff into the
        # outermost rows and columns only. A whole-frame estimator's answer moves;
        # Welch's interior tiles must not see it at all.
        rng = np.random.default_rng(31)
        base = rng.integers(60, 190, (192, 320, 3), dtype=np.uint8)
        edged = base.copy()
        edged[0, :, :] = 0; edged[-1, :, :] = 255
        edged[:, 0, :] = 0; edged[:, -1, :] = 255
        r = D.tile_freq_grid(); hi = (r >= 0.20); nz = r > 0
        a = D.welch_psd([base])[0]; b = D.welch_psd([edged])[0]
        sa = float(a[hi].sum() / a[nz].sum()); sb = float(b[hi].sum() / b[nz].sum())
        self.assertAlmostEqual(sa, sb, places=9)
        # and the border DOES move a whole-frame estimator, or the case is vacuous
        pa = D._power(D.luma(base) - D.luma(base).mean())
        pb = D._power(D.luma(edged) - D.luma(edged).mean())
        rr = D._fgrid(192, 320); h2 = (rr >= 0.20); n2 = rr > 0
        self.assertNotAlmostEqual(float(pa[h2].sum() / pa[n2].sum()),
                                  float(pb[h2].sum() / pb[n2].sum()), places=4)

    def test_the_inset_is_a_TILE_STEP_and_not_merely_nonzero(self):
        # THE CASE ABOVE PAINTS ONE ROW AND ONE COLUMN, so it stays green with
        # an inset of a single pixel: it constrains `lo > 0`, not `lo` being an
        # inset that keeps whole tiles off the border. Paint a band TILE_STEP
        # deep instead. A tile placed at any offset below TILE_STEP overlaps it,
        # so this reds for `lo = 1` and passes only for the real inset.
        rng = np.random.default_rng(131)
        base = rng.integers(60, 190, (192, 320, 3), dtype=np.uint8)
        edged = base.copy()
        k = D.TILE_STEP
        edged[:k, :, :] = 0; edged[-k:, :, :] = 255
        edged[:, :k, :] = 0; edged[:, -k:, :] = 255
        r = D.tile_freq_grid(); hi = (r >= 0.20); nz = r > 0
        a = D.welch_psd([base])[0]; b = D.welch_psd([edged])[0]
        self.assertAlmostEqual(float(a[hi].sum() / a[nz].sum()),
                               float(b[hi].sum() / b[nz].sum()), places=9)
        # vacuity guard: the band DOES move a whole-frame estimator
        pa = D._power(D.luma(base) - D.luma(base).mean())
        pb = D._power(D.luma(edged) - D.luma(edged).mean())
        rr = D._fgrid(192, 320); h2 = (rr >= 0.20); n2 = rr > 0
        self.assertNotAlmostEqual(float(pa[h2].sum() / pa[n2].sum()),
                                  float(pb[h2].sum() / pb[n2].sum()), places=4)

    def test_welch_refuses_a_frame_too_small_to_hold_an_interior_tile(self):
        tiny = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(D.UnreadableInput):
            D.welch_psd([tiny])

    def test_blurring_moves_high_frequency_energy_DOWN(self):
        # The direction the smoothness hypothesis needed. If this case cannot
        # see it, the measured absence of it downstream means nothing.
        rng = np.random.default_rng(9)
        a = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        b = D.gaussian_blur(a, 1.2)
        c, pa = D.radial_power(_cmp.luma(a), None, nbins=64, convention="raw")
        _, pb = D.radial_power(_cmp.luma(b), None, nbins=64, convention="raw")
        fa = D.hf_fraction(c, np.array([pa]), 0.25)[0]
        fb = D.hf_fraction(c, np.array([pb]), 0.25)[0]
        self.assertLess(fb, fa / 2.0)

    def test_the_crossover_of_a_uniformly_scaled_copy_is_the_first_bin(self):
        # A pure SCALE difference must not read as a rolloff. Halving every
        # amplitude divides every bin by four, so the ratio is flat and below
        # one everywhere, and the crossover belongs at the bottom of the range.
        rng = np.random.default_rng(13)
        l = rng.normal(128.0, 30.0, (96, 128))
        c, p1 = D.radial_power(l, None, nbins=64, convention="raw")
        _, p2 = D.radial_power(l * 0.5, None, nbins=64, convention="raw")
        ratio = p2 / p1
        have = np.isfinite(ratio)
        self.assertTrue(np.all(ratio[have] < 1.0))
        below = have & (ratio < 1.0)
        cross = next(float(c[i]) for i in range(len(ratio))
                     if have[i] and np.all(below[i:][have[i:]]))
        # The first bin that HAS coefficients, not the first bin. At 96x128 the
        # lowest annulus is empty, and an empty bin must not push the crossover
        # up and read as a rolloff -- which is exactly the defect this case
        # found in the first version of the crossover rule.
        self.assertEqual(cross, float(c[int(np.argmax(have))]))


class PeriodicDecomposition(unittest.TestCase):
    """The repair that changed this row's headline, so it is the most heavily
    guarded thing in the file.

    A reviewer reproduced the high-band comparison with the raw convention and
    got the OPPOSITE SIGN. Both readings were the DFT's wrap step, which the two
    renders do not carry equally. These cases pin that the decomposition removes
    that step, leaves interior content alone, and actually changes the answer on
    a frame whose borders disagree -- because a no-op would read as "the
    convention did not matter" and put the withdrawn headline straight back.
    """

    @staticmethod
    def _ramp(h=64, w=96):
        # Borders disagree by construction: a horizontal ramp wraps as a cliff.
        return np.linspace(10.0, 240.0, w)[None, :] * np.ones((h, 1))

    def test_the_periodic_component_removes_the_wrap_step(self):
        u = self._ramp()
        p = D.periodic_component(u)
        before = float(np.abs(u[:, 0] - u[:, -1]).mean())
        after = float(np.abs(p[:, 0] - p[:, -1]).mean())
        self.assertGreater(before, 100.0)
        self.assertLess(after, before / 10.0)

    def test_an_already_periodic_image_is_left_alone(self):
        # A sinusoid at an integer number of cycles has no wrap DISCONTINUITY,
        # so the decomposition must be near the identity on it. A transform that
        # "cleans" such an image is altering content, not removing an artefact.
        #
        # The bound is ONE ORDINARY INTERIOR STEP and not a constant, because a
        # sampled sinusoid's last-to-first difference is not zero: the wrap runs
        # from sample N-1 to sample 0, which is one step like any other. A
        # tighter bound would be asserting the fixture is continuous when it is
        # merely periodic, and it failed here for exactly that reason.
        x = np.arange(96)[None, :]
        u = 128.0 + 40.0 * np.sin(2 * np.pi * 3 * x / 96) * np.ones((64, 1))
        step = float(np.abs(np.diff(u, axis=1)).mean())
        self.assertLess(float(np.abs(D.periodic_component(u) - u).max()), step)

    def test_it_changes_the_high_band_share_on_a_frame_with_a_wrap_step(self):
        # THE CASE THAT MATTERS. If this passed with a no-op decomposition the
        # withdrawn headline would come back silently.
        u = self._ramp() + np.random.default_rng(3).normal(0, 4.0, (64, 96))
        r = D._fgrid(64, 96)
        hi = (r >= 0.20) & (r < 0.71)
        tot = r > 0
        raw = D._power(u - u.mean())
        per = D._power(D.periodic_component(u))
        s_raw = float(raw[hi].sum() / raw[tot].sum())
        s_per = float(per[hi].sum() / per[tot].sum())
        self.assertGreater(s_per, 1.5 * s_raw,
                           f"decomposition barely moved the share: {s_raw} -> {s_per}")

    def test_the_wrap_statistic_ranks_a_cliff_above_a_seamless_frame(self):
        seam = np.repeat((self._ramp())[:, :, None], 3, axis=2).astype(np.uint8)
        x = np.arange(96)[None, :]
        smooth = 128.0 + 40.0 * np.sin(2 * np.pi * 3 * x / 96) * np.ones((64, 1))
        per = np.repeat(smooth[:, :, None], 3, axis=2).astype(np.uint8)
        a = D.wrap_discontinuity([seam, seam])
        b = D.wrap_discontinuity([per, per])
        self.assertGreater(a["jump_over_interior_step"],
                           10.0 * b["jump_over_interior_step"])

    def test_radial_power_rejects_nothing_and_the_conventions_differ(self):
        u = self._ramp() + np.random.default_rng(5).normal(0, 3.0, (64, 96))
        c, p_raw = D.radial_power(u, None, 32, "raw")
        _, p_per = D.radial_power(u, None, 32, "periodic")
        self.assertFalse(np.allclose(np.nan_to_num(p_raw), np.nan_to_num(p_per)))


class Bands(unittest.TestCase):
    """Mid against high, and the churn that separates detail from grain."""

    def test_a_mid_band_sinusoid_lands_in_the_mid_share_and_not_the_high_one(self):
        f = 0.09
        x = np.arange(320)[None, :]
        img = np.clip(128.0 + 90.0 * np.sin(2 * np.pi * f * x) * np.ones((192, 1)),
                      0, 255).astype(np.uint8)
        frames = [np.repeat(img[:, :, None], 3, axis=2)] * 3
        t = D.band_terms(frames)
        self.assertGreater(float(t["mid_fraction"].mean()), 0.5)
        self.assertLess(float(t["hi_fraction"].mean()), 0.05)

    def test_static_detail_churns_far_less_than_per_frame_grain(self):
        # THE DISCRIMINATOR. A fixed high-frequency texture repeated across
        # frames must churn near zero; independent noise per frame must churn
        # near or above one. If this case cannot tell them apart, the churn
        # number reported for the two renders decides nothing.
        rng = np.random.default_rng(5)
        tex = rng.integers(0, 256, (192, 320, 3), dtype=np.uint8)
        static = [tex.copy() for _ in range(6)]
        grain = [rng.integers(0, 256, (192, 320, 3), dtype=np.uint8) for _ in range(6)]
        self.assertLess(D.band_terms(static)["hi_temporal_churn"], 0.05)
        self.assertGreater(D.band_terms(grain)["hi_temporal_churn"], 1.0)

    def test_an_axis_aligned_nyquist_pattern_beats_its_own_radial_ring(self):
        # What a separable upsampling stage leaves behind, and the probe has to
        # see it as an AXIS excess rather than as ordinary fine texture.
        y = np.arange(192)[:, None]
        img = np.clip(128.0 + 60.0 * np.cos(np.pi * y) * np.ones((1, 320)),
                      0, 255).astype(np.uint8)
        frames = [np.repeat(img[:, :, None], 3, axis=2)] * 2
        a = D.nyquist_axes(frames)
        self.assertGreater(a["vertical_nyquist"], 20.0 * a["radial_ring_mean"])

    def test_a_smooth_image_with_broadband_content_has_no_axis_excess(self):
        # THE NOISE IS NOT DECORATION. A noiseless smooth fixture has essentially
        # nothing near Nyquist, so its ring mean is floating-point dust and ANY
        # ratio against it is meaningless -- the earlier version of this case
        # failed for that reason and not because the probe was wrong. Every real
        # frame carries broadband content, so the fixture does too, and the
        # assertion is then about the AXES against a populated ring.
        rng = np.random.default_rng(41)
        y = np.arange(192)[:, None]
        smooth = 128.0 + 90.0 * np.cos(2 * np.pi * y / 192) * np.ones((1, 320))
        img8 = np.clip(smooth + rng.normal(0, 6.0, (192, 320)), 0, 255).astype(np.uint8)
        img = np.repeat(img8[:, :, None], 3, axis=2)
        a = D.nyquist_axes([img, img])
        self.assertLess(a["vertical_nyquist"], 5.0 * a["radial_ring_mean"])

    def test_the_axis_probe_SEPARATES_a_lattice_pattern_from_ordinary_texture(self):
        # The comparative form, which is what the probe is actually used for:
        # ours against the reference. A probe that scored both alike would have
        # nothing to say about either.
        rng = np.random.default_rng(43)
        noise = rng.integers(0, 256, (192, 320, 3), dtype=np.uint8)
        y = np.arange(192)[:, None]
        lattice8 = np.clip(128.0 + 50.0 * np.cos(np.pi * y) * np.ones((1, 320))
                           + rng.normal(0, 20.0, (192, 320)), 0, 255).astype(np.uint8)
        lattice = np.repeat(lattice8[:, :, None], 3, axis=2)
        a = D.nyquist_axes([lattice, lattice])
        b = D.nyquist_axes([noise, noise])
        self.assertGreater(a["vertical_nyquist"] / a["radial_ring_mean"],
                           10.0 * b["vertical_nyquist"] / b["radial_ring_mean"])


class Correlation(unittest.TestCase):
    """The hypothesis test, and the two ways it can be vacuous."""

    def test_a_planted_positive_association_is_recovered(self):
        x = np.linspace(0.0, 1.0, 25)
        y = 3.0 * x + 0.5
        self.assertAlmostEqual(D.pearson(x, y), 1.0, places=9)
        self.assertAlmostEqual(D.spearman(x, y), 1.0, places=9)

    def test_a_planted_negative_association_is_recovered_with_its_SIGN(self):
        # The measured within-render coefficient is negative, which is the
        # opposite of what the hypothesis predicts. A `pearson` that lost the
        # sign would turn that refutation into a confirmation.
        x = np.linspace(0.0, 1.0, 25)
        self.assertAlmostEqual(D.pearson(x, -2.0 * x + 7.0), -1.0, places=9)
        self.assertAlmostEqual(D.spearman(x, -2.0 * x + 7.0), -1.0, places=9)

    def test_a_constant_input_returns_nan_and_never_zero(self):
        # A zero would read as "measured, no association". A constant input is
        # an input on which no association CAN be measured, and the two must not
        # print the same way.
        self.assertTrue(np.isnan(D.pearson(np.ones(25), np.arange(25.0))))

    def test_spearman_sees_a_monotone_relation_pearson_understates(self):
        x = np.linspace(0.1, 1.0, 25)
        y = np.exp(6.0 * x)
        self.assertAlmostEqual(D.spearman(x, y), 1.0, places=9)
        self.assertLess(D.pearson(x, y), 0.95)


class Interventions(unittest.TestCase):
    """The blur has to blur and the unsharp has to sharpen, in the statistic the
    report quotes. A no-op intervention reads as "detail does not matter"."""

    def test_the_blur_lowers_the_sharpness_statistic_monotonically(self):
        rng = np.random.default_rng(17)
        a = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        vals = [D.mean_sharpness([D.gaussian_blur(a, s)]) for s in (0.0, 0.5, 1.0, 2.0)]
        self.assertTrue(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
                        f"not monotone: {vals}")

    def test_a_zero_sigma_blur_is_the_identity_and_not_a_quiet_copy(self):
        rng = np.random.default_rng(19)
        a = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(D.gaussian_blur(a, 0.0), a))

    def test_the_unsharp_raises_the_sharpness_statistic(self):
        rng = np.random.default_rng(23)
        a = D.gaussian_blur(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), 1.0)
        self.assertGreater(D.mean_sharpness([D.unsharp(a, 1.0, 1.0)]),
                           D.mean_sharpness([a]))

    def test_the_blur_stays_inside_the_representable_range(self):
        rng = np.random.default_rng(29)
        a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        b = D.gaussian_blur(a, 1.5)
        self.assertEqual(b.dtype, np.uint8)
        self.assertGreaterEqual(int(b.min()), 0)
        self.assertLessEqual(int(b.max()), 255)



class ArmValidity(unittest.TestCase):
    """THE PRECONDITION IS ASKED OF EVERY ARM, not once of the reference.

    This row published `+1.9131` -- its decisive figure -- from a blurred arm on
    which the decoy `near:1` outranks the true prompt by 0.4036. The rule that
    forbids exactly that was already written down and already enforced, once, on
    the unblurred reference. So these cases are about the SCOPE of a check and
    not about its logic, and the mutation that kills them is deleting a
    `scorer_valid` field rather than loosening a threshold.
    """

    @staticmethod
    def _disc(true_score, decoy_scores):
        m = np.array([[true_score] + list(decoy_scores)], dtype=np.float64)
        labels = ["true"] + [f"near:{i}" for i in range(len(decoy_scores))]
        return _cmp.discrimination(m, labels)

    def test_an_arm_whose_decoy_outranks_the_true_prompt_is_INVALID(self):
        v = _cmp.adherence_arm_valid(self._disc(30.0, [30.5, 29.0]))
        self.assertFalse(v["valid"])
        self.assertEqual(v["argmax_label"], "near:0")
        self.assertIn("above the true prompt", v["reason"])

    def test_a_zero_margin_arm_is_INVALID_even_though_it_has_an_argmax(self):
        # numpy's argmax returns the FIRST maximal index, which is the true
        # prompt, so a bare `argmax == true` passes a scorer that returns one
        # number for every prompt.
        d = self._disc(7.0, [7.0, 7.0])
        self.assertTrue(d["true_first"], "the trap this case exists for")
        v = _cmp.adherence_arm_valid(d)
        self.assertFalse(v["valid"])
        self.assertIn("no separation", v["reason"])

    def test_a_separating_arm_is_valid_and_reports_its_margin(self):
        v = _cmp.adherence_arm_valid(self._disc(31.0, [20.0, 12.0]))
        self.assertTrue(v["valid"])
        self.assertAlmostEqual(v["margin"], 11.0)
        self.assertEqual(v["reason"], "")

    def test_S0_AND_the_arm_check_are_the_SAME_predicate(self):
        # Two copies of a refusal rule drift, and the copy that drifts is the
        # one nothing calls on the failing path. S0 must raise on exactly the
        # discriminations this returns invalid for, and on no others.
        #
        # This case asserts CONSISTENCY, not refusal: it compares the two sides
        # to each other, so a predicate disarmed to always-permissive moves both
        # sides together and this stays green. The absolute gate -- that S0
        # refuses a scorer a decoy outranks, whatever the other side does -- is
        # `tests/scripts/test_ltx25_prompt_adherence.py::Precondition`, which is
        # the case that reds under that mutation.
        cases = [(30.0, [30.5, 29.0]), (7.0, [7.0, 7.0]), (31.0, [20.0, 12.0]),
                 (31.0, [30.999999, 1.0]), (0.0, [-1.0])]
        for t, ds in cases:
            d = self._disc(t, ds)
            invalid = not _cmp.adherence_arm_valid(d)["valid"]
            try:
                _cmp.scorer_precondition(d)
                raised = False
            except _cmp.UnreadableInput:
                raised = True
            self.assertEqual(invalid, raised,
                             f"the two disagree on true={t} decoys={ds}")


class TheRunRecordsValidityOnEveryArm(unittest.TestCase):
    """P: `main()` itself, which is where findings 1 and 6 lived and where no
    case reached.

    The suite above this one constrains primitives. Every one of them passed
    while the run printed a rising CLIP delta from arms a decoy had won and
    labelled its high band with the crossover frequency instead of the band's
    own cut. So this drives the whole run on synthesised frames with a stub
    scorer, and the stub is built to reproduce the row's real shape: the
    reference separates comfortably, our unblurred frames separate narrowly, and
    blurring ours far enough takes the true prompt below a decoy exactly as it
    did on the real render.

    `discrimination`, `scorer_precondition` and `adherence_arm_valid` are the
    REAL ones. Only the checkpoint, the pin, the decoy list and the CLIP forward
    are stubbed, because those need a 598 MB pickle this lane does not have.
    """

    # true prompt: 30 + sharpness/4. Every decoy: a flat 35.0. Sharp frames beat
    # the decoys, blurred ones do not, which is the mechanism under test.
    DECOY_LEVEL = 35.0

    class _Scorer:
        def __init__(self, model_dir, pin):
            self.model_dir = model_dir

        def count_tokens(self, text):
            return len(text.split())

        def score(self, frames, prompts):
            out = np.empty((len(frames), len(prompts)), dtype=np.float64)
            for i, a in enumerate(frames):
                s = float(D.sharpness_map(D.luma(a)).mean())
                out[i, 0] = 30.0 + s / 4.0
                out[i, 1:] = TheRunRecordsValidityOnEveryArm.DECOY_LEVEL
            return out

    class _Shim:
        """The real module, with four loaders and the scorer replaced."""

        def __init__(self, real, scorer_cls):
            self._real, self._scorer_cls = real, scorer_cls

        def __getattr__(self, name):
            return getattr(self._real, name)

        ClipAdherenceScorer = property(lambda self: self._scorer_cls)

        def load_json_record(self, path, what):
            return {"text_context_positions": 77}

        def assert_scorer_identity(self, model_dir, pin):
            return {"repo": "stub", "files_verified": 0}

        def load_decoys(self, path):
            return [{"kind": "near", "text": "a grey wolf in a snowy forest"},
                    {"kind": "far", "text": "a city street at noon"}]

        def true_prompt(self, path):
            return "a red fox in a snowy pine forest"

        def refuse_overlong_prompts(self, prompts, count, limit):
            return None

    def _run(self, td):
        rng = np.random.default_rng(7)
        ref = [rng.integers(0, 256, (128, 160, 3), dtype=np.uint8) for _ in range(6)]
        ours = [D.gaussian_blur(a, 0.6) for a in ref]
        od, rd = os.path.join(td, "ours"), os.path.join(td, "ref")
        os.mkdir(od); os.mkdir(rd)
        for i, a in enumerate(ours):
            write_ppm(os.path.join(od, f"frame_{i:06d}.ppm"), a)
        sums = os.path.join(td, "SHA256SUMS")
        with open(sums, "w", encoding="utf-8") as fh:
            for i, a in enumerate(ref):
                q = os.path.join(rd, f"frame_{i:06d}.ppm")
                write_ppm(q, a)
                fh.write(f"{_sha(q)}  frame_{i:06d}.ppm\n")
        # The digest guard is real; point it at the set this case just wrote.
        lines = "".join(f"{_sha(os.path.join(od, n))}  {n}\n"
                        for n in sorted(os.listdir(od)))
        digest = hashlib.sha256(lines.encode("ascii")).hexdigest()
        out_json = os.path.join(td, "out.json")
        old_digest, old_cmp = D.OURS_SET_DIGEST, D._cmp
        D.OURS_SET_DIGEST = digest
        D._cmp = self._Shim(old_cmp, self._Scorer)
        try:
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = D.main(["--ours", od, "--reference", rd, "--sha256sums", sums,
                             "--adherence-model", td, "--blur-sigmas", "0.3",
                             "--json", out_json])
        finally:
            D.OURS_SET_DIGEST, D._cmp = old_digest, old_cmp
        self.assertEqual(rc, 0)
        with open(out_json, encoding="utf-8") as fh:
            return json.load(fh), buf.getvalue()

    def test_every_scored_arm_carries_a_validity_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            out, _ = self._run(td)
        arms = out["arm_validity"]["arms"]
        # One per scored arm: both unblurred renders, the reference sweep, our
        # four-sigma sweep, the tone rescore and the sharpen diagnostic.
        for want in ("reference_unblurred", "ours_unblurred", "ours_tone_matched",
                     "ours_unsharp_1.0", "reference_blur_sigma_0.30",
                     "ours_blur_sigma_0.30", "ours_blur_sigma_0.50",
                     "ours_blur_sigma_0.70", "ours_blur_sigma_1.00"):
            self.assertIn(want, arms, f"{want} was scored and not checked")
        for name, v in arms.items():
            self.assertIn("valid", v, name)
        # and the verdict is recorded ON the row a reader reads, not only here
        for row in out["ours_lowpass_ablation"]["sweep"]:
            self.assertIn("scorer_valid", row)
        for row in out["blur_ablation"]["sweep"]:
            self.assertIn("scorer_valid", row)

    def test_a_blurred_arm_a_decoy_WINS_is_marked_invalid_and_says_so(self):
        # THE DEFECT, REPRODUCED. The delta rises with sigma while the arm
        # producing it stops ranking the true prompt first. Both must be visible.
        with tempfile.TemporaryDirectory() as td:
            out, log = self._run(td)
        sweep = {r["sigma"]: r for r in out["ours_lowpass_ablation"]["sweep"]}
        self.assertTrue(sweep[0.3]["scorer_valid"]["valid"],
                        "the mildest arm must stay readable or the case is vacuous")
        for s in (0.5, 0.7, 1.0):
            self.assertFalse(sweep[s]["scorer_valid"]["valid"], f"sigma {s}")
            self.assertNotEqual(sweep[s]["argmax_label"], "true")
        self.assertEqual(out["ours_lowpass_ablation"]["readable_sigmas"], [0.3])
        self.assertIn("INVALID", log)

    def test_the_console_prints_the_columns_that_decide_the_delta(self):
        # The reference sweep always printed argmax, margin and wins. Our own
        # sweep computed all three, recorded all three, and printed none, so the
        # spec table copied from that console had no column that could show it.
        with tempfile.TemporaryDirectory() as td:
            _, log = self._run(td)
        for line in log.splitlines():
            if "OUR frames blurred" in line:
                self.assertIn("argmax", line)
                self.assertIn("margin", line)
                self.assertIn("wins", line)

    def test_the_high_band_line_names_the_band_it_measured(self):
        # It named `f_lo` -- the CROSSOVER -- for numbers `band_terms` cuts at
        # BAND_HI[0], and on the real frames those differ by 2.2x.
        with tempfile.TemporaryDirectory() as td:
            out, log = self._run(td)
        line = next(l for l in log.splitlines()
                    if l.startswith("[spectrum] energy"))
        self.assertIn(f"{D.BAND_HI[0]:.4f}", line)
        self.assertEqual(out["spectrum"]["hf_cut_cycles_per_pixel"], D.BAND_HI[0])
        cross = out["spectrum"]["crossover_cycles_per_pixel"]
        if cross is not None and abs(cross - D.BAND_HI[0]) > 1e-9:
            self.assertNotIn(f"{cross:.4f}", line)

    def test_the_four_conventions_are_correlated_PER_FRAME_and_recorded(self):
        # The row published a four-estimator correlation table while the harness
        # computed one correlation and `_conv_shares` returned whole-render
        # scalars. Two of that table's four figures existed in no code and no
        # artifact.
        with tempfile.TemporaryDirectory() as td:
            out, _ = self._run(td)
        for arm in ("ours", "reference"):
            by = out["correlation"][arm]["pearson_hf_vs_clip_by_convention"]
            self.assertEqual(set(by), {"raw", "hann", "periodic", "welch"})
            # welch is the reported one, so it must equal the headline exactly
            self.assertAlmostEqual(by["welch"],
                                   out["correlation"][arm]["pearson_hf_vs_clip"],
                                   places=12)
            n = len(out["per_frame_clip_true"][arm])
            for conv in by:
                self.assertEqual(
                    len(out["convention_sensitivity"][conv][f"{arm}_high_per_frame"]), n)
            # FOUR ESTIMATORS, OR THE TABLE IS ONE NUMBER FOUR TIMES. Feeding
            # every entry the same per-frame array leaves every assertion above
            # green, which is what a transcribed table looks like from the
            # inside, so the four must be measurably distinct.
            self.assertEqual(len({round(v, 9) for v in by.values()}), 4,
                             f"{arm}: the conventions returned fewer than four "
                             f"distinct coefficients: {by}")
            self.assertGreater(abs(by["raw"] - by["welch"]), 0.05,
                               f"{arm}: raw and welch must not agree; raw is the "
                               f"convention whose band delta reverses sign")
            # and the per-frame ARRAYS differ, not only their reductions
            arrs = [tuple(out["convention_sensitivity"][c][f"{arm}_high_per_frame"])
                    for c in ("raw", "hann", "periodic", "welch")]
            self.assertEqual(len(set(arrs)), 4, f"{arm}: duplicate per-frame arrays")

    def test_one_number_is_not_reported_twice_as_two(self):
        # `pearson_high_band_vs_clip` reduced the same array as
        # `pearson_hf_vs_clip` and shipped beside it, so the JSON showed two
        # agreeing coefficients where one measurement existed.
        with tempfile.TemporaryDirectory() as td:
            out, _ = self._run(td)
        for arm in ("ours", "reference"):
            self.assertNotIn("pearson_high_band_vs_clip", out["correlation"][arm])
            self.assertIn("pearson_mid_band_vs_clip", out["correlation"][arm])


class TheCommittedGoldenIsREAD(unittest.TestCase):
    """P: the 2980-line artifact had NO consumer -- no test, no script, no CI
    step, no CMake target -- so the file that falsifies the prose could disagree
    with it forever and nothing would notice. It disagreed.

    These cases are the consumer. Each figure below is quoted in
    `.agents/specs/ltx25-adherence-detail-loss.md`, `ltx25-oracle-absolute.md`,
    `ltx25-prompt-adherence.md` or `docs/USAGE.md`, and each is asserted against
    the committed JSON.

    What that binds is the GOLDEN against the constant written here: editing
    either of those two alone reds the suite. It does not bind the prose --
    editing a figure in a spec and nothing else reds nothing, because no case
    parses the Markdown. The constants below were hand-verified against the
    quoted prose when they were written, and moving a published figure means
    moving the spec and the constant in the same change.
    """

    GOLDEN = os.path.join(ROOT, "tests", "parity", "goldens",
                          "ltx25_detail_loss", "detail-loss.json")

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.GOLDEN):
            raise unittest.SkipTest(f"{cls.GOLDEN} is absent")
        with open(cls.GOLDEN, encoding="utf-8") as fh:
            cls.g = json.load(fh)

    def near(self, got, want, places=4, what=""):
        self.assertAlmostEqual(float(got), want, places=places, msg=what)

    def test_the_absolute_band_powers_the_records_publish(self):
        b = self.g["bands"]
        self.near(b["ours_mid_power_over_reference"], 1.0373, 4, "mid 1.0373x")
        self.near(b["ours_hi_power_over_reference"], 1.4031, 4, "high 1.4031x")
        self.near(b["ours_total_power_over_reference"], 1.1437, 4, "total 1.1437x")

    def test_the_four_convention_band_deltas(self):
        c = self.g["convention_sensitivity"]
        for conv, mid, hi in (("raw", -0.3115, -0.2525), ("hann", -0.0888, 0.7751),
                              ("periodic", -0.1183, 0.0578), ("welch", -0.0930, 0.2268)):
            self.near(c[conv]["mid_delta"], mid, 4, f"{conv} mid")
            self.near(c[conv]["high_delta"], hi, 4, f"{conv} high")

    def test_the_within_render_correlations_and_their_SIGNS(self):
        c = self.g["correlation"]
        self.near(c["ours"]["pearson_hf_vs_clip"], -0.6195, 4)
        self.near(c["reference"]["pearson_hf_vs_clip"], 0.2640, 4)
        self.near(c["ours"]["spearman_hf_vs_clip"], -0.5985, 4)
        self.near(c["reference"]["spearman_hf_vs_clip"], 0.2254, 4)
        # against SHARPNESS both are negative, so that axis carries no contrast
        # and the records must not present it as one
        self.assertLess(c["ours"]["pearson_sharpness_vs_clip"], 0.0)
        self.assertLess(c["reference"]["pearson_sharpness_vs_clip"], 0.0)
        self.near(c["ours"]["pearson_sharpness_vs_clip"], -0.4450, 4)
        self.near(c["reference"]["pearson_sharpness_vs_clip"], -0.1889, 4)
        # the mid band is POSITIVE on both sides: the sign flip is the high band
        self.assertGreater(c["ours"]["pearson_mid_band_vs_clip"], 0.0)
        self.assertGreater(c["reference"]["pearson_mid_band_vs_clip"], 0.0)
        self.near(c["ours"]["pearson_mid_band_vs_clip"], 0.3682, 4)
        self.near(c["reference"]["pearson_mid_band_vs_clip"], 0.4513, 4)

    def test_the_four_estimators_agree_about_OURS_and_NOT_about_the_reference(self):
        # "The within-render correlation survives every estimator" was published
        # from one computed coefficient and two figures that existed nowhere.
        # Computed per frame, the claim is half true, and the half that fails is
        # the half the sign flip rests on.
        by_o = self.g["correlation"]["ours"]["pearson_hf_vs_clip_by_convention"]
        by_r = self.g["correlation"]["reference"]["pearson_hf_vs_clip_by_convention"]
        self.assertEqual(set(by_o), {"raw", "hann", "periodic", "welch"})
        # OURS is robust: four estimators, one tight negative cluster
        for conv, want in (("raw", -0.5946), ("hann", -0.6097),
                           ("periodic", -0.6216), ("welch", -0.6195)):
            self.near(by_o[conv], want, 4, f"ours {conv}")
            self.assertLess(by_o[conv], -0.5, f"ours {conv} must stay clearly negative")
        self.assertLess(max(by_o.values()) - min(by_o.values()), 0.05,
                        "ours must be a tight cluster or 'robust' is the wrong word")
        # THE REFERENCE IS NOT. Raw reads essentially ZERO, so the positive
        # coefficient the sign flip needs is estimator-dependent and the records
        # must not present it as robust.
        for conv, want in (("raw", 0.0009), ("hann", 0.4514),
                           ("periodic", 0.1754), ("welch", 0.2640)):
            self.near(by_r[conv], want, 4, f"reference {conv}")
        self.assertLess(abs(by_r["raw"]), 0.01,
                        "the raw convention gives the reference NO association")
        self.assertGreater(max(by_r.values()) - min(by_r.values()), 0.4,
                           "the reference's spread is the finding; if it closed, "
                           "the prose that calls it estimator-dependent is stale")

    def test_the_reference_blur_sweep_HOLDS_the_precondition_throughout(self):
        # This is the leg the falsification now rests on, so its validity is
        # asserted rather than assumed.
        s = self.g["blur_ablation"]["sweep"]
        for row in s:
            self.assertEqual(row["argmax_label"], "true", f"sigma {row['sigma']}")
            self.assertGreater(row["margin_to_best_decoy"], 0.0)
        last = s[-1]
        self.near(last["sigma"], 2.0, 6)
        self.near(last["clip_delta_vs_unblurred"], -1.9687, 4)
        self.assertEqual(last["per_frame_true_wins"], 23)
        self.near(self.g["blur_ablation"]["observed_gap"], 2.7305, 4)
        self.assertLess(abs(last["clip_delta_vs_unblurred"]),
                        self.g["blur_ablation"]["observed_gap"],
                        "no achievable blur reproduces the gap")

    def test_our_own_blur_sweep_FAILS_the_precondition_above_sigma_0_3(self):
        # THE FINDING THAT WITHDREW THE DECISIVE ROW. `+1.9131` comes from an
        # arm where the decoy `near:1` outranks the true prompt by 0.4036.
        s = {r["sigma"]: r for r in self.g["ours_lowpass_ablation"]["sweep"]}
        self.assertEqual(s[0.3]["argmax_label"], "true")
        self.near(s[0.3]["delta_vs_ours"], -0.0029, 4)
        for sig, delta, margin in ((0.5, -0.4252, -0.0436), (0.7, 0.8305, -0.3660),
                                   (1.0, 1.9131, -0.4036)):
            self.assertNotEqual(s[sig]["argmax_label"], "true", f"sigma {sig}")
            self.near(s[sig]["delta_vs_ours"], delta, 4)
            self.near(s[sig]["margin_to_best_decoy"], margin, 4)
        # and the margin walks off monotonically, which is why no larger sigma
        # rescues it either
        margins = [s[k]["margin_to_best_decoy"] for k in (0.3, 0.5, 0.7, 1.0)]
        self.assertEqual(margins, sorted(margins, reverse=True))

    def test_tone_is_ruled_out_on_a_VALID_arm(self):
        t = self.g["tone"]["rescored_after_correction"]
        self.near(t["delta_vs_uncorrected"], 0.1846, 4)
        self.assertEqual(t["argmax_label"], "true")
        self.assertGreater(t["margin_to_best_decoy"], 0.0)
        self.assertEqual(t["frames_clearing_bound"], 9)
        self.near(self.g["tone"]["fit_ours_to_reference"]["gain"], 0.99144, 5)
        self.near(self.g["tone"]["fit_ours_to_reference"]["gamma"], 1.04654, 5)

    def test_the_unsharp_arm_is_VALID_and_is_the_fifth_leg(self):
        # The spec framed this arm's +0.4542 against the WITHDRAWN +1.9131 and
        # called it "a quarter of what the blur buys", under "only one of them
        # helps". Both halves divide by, or compare against, a disqualified
        # number, and the polarity is inverted: this is the arm of that pair
        # that holds its precondition. Read against the GAP it is a leg of the
        # falsification, so its figures are pinned here.
        d = self.g["sharpen_diagnostic"]
        self.assertEqual(d["argmax_label"], "true")
        self.assertTrue(d["scorer_valid"]["valid"])
        self.near(d["margin_to_best_decoy"], 0.6547, 4)
        self.assertEqual(d["per_frame_true_wins"], 19)
        self.near(d["delta_vs_ours"], 0.4542, 4)
        self.near(d["sharpness"], 19.0357, 4)
        gap = self.g["blur_ablation"]["observed_gap"]
        self.near(d["delta_vs_ours"] / gap, 0.166, 3)
        self.assertLess(d["delta_vs_ours"] / gap, 0.2,
                        "sharpening recovers a SIXTH of the gap; if this rose, "
                        "the prose that calls the hypothesis refuted is stale")
        # and it sharpens 69% past the reference's own sharpness while doing so
        ref_sharp = self.g["blur_ablation"]["reference_sharpness"]
        self.near(d["sharpness"] / ref_sharp - 1.0, 0.688, 3)
        # the arm the spec preferred is the INVALID one, from sigma 0.50 up
        arms = self.g["arm_validity"]["arms"]
        self.assertTrue(arms["ours_unsharp_1.0"]["valid"])
        self.assertFalse(arms["ours_blur_sigma_1.00"]["valid"])

    def test_the_sharpest_five_and_the_best_scoring_five_share_no_member(self):
        # FFT-free and reproducible from the per-frame table, which is why the
        # records now lean on it.
        rows = sorted(self.g["per_frame_table"], key=lambda r: -r["clip_true"])
        best5 = {r["frame"] for r in rows[:5]}
        sharp5 = set(self.g["sharpest_five"])
        self.assertEqual(best5 & sharp5, set())
        self.assertEqual(self.g["frames_clearing_bound"], [0, 1, 2, 12, 15, 16])

    def test_the_temporal_split_is_FFT_free_and_opposite_in_sign(self):
        t = self.g["correlation"]["temporal"]
        self.near(t["ours_r_frame_index_vs_clip"], -0.3351, 4)
        self.near(t["reference_r_frame_index_vs_clip"], 0.6987, 4)
        # the hf-GROWTH figures in the same paragraph were published as
        # +22.8%/+17.9% with the order reversed; they are +4.8%/+7.0%
        self.near(t["ours_hf_last5_over_first5"] - 1.0, 0.0481, 4)
        self.near(t["reference_hf_last5_over_first5"] - 1.0, 0.0698, 4)
        self.assertLess(t["ours_hf_last5_over_first5"],
                        t["reference_hf_last5_over_first5"],
                        "ours grows LESS, not more")

    def test_the_axis_to_ring_figures_that_withdrew_the_upsampler_lead(self):
        b = self.g["bands"]
        for arm, h, v in (("ours", 2.46, 2.22), ("reference", 2.23, 1.78)):
            a = b[f"{arm}_nyquist_axes"]
            self.near(a["horizontal_nyquist"] / a["radial_ring_mean"], h, 2, arm)
            self.near(a["vertical_nyquist"] / a["radial_ring_mean"], v, 2, arm)

    def test_the_crossover_and_the_wrap_step_the_correction_cites(self):
        self.near(self.g["spectrum"]["crossover_cycles_per_pixel"], 0.43359375, 8)
        w = self.g["wrap_discontinuity"]
        self.near(w["ours"]["top_bottom_jump"], 73.39, 2)
        self.near(w["reference"]["top_bottom_jump"], 49.48, 2)

    def test_the_scores_the_bound_and_n_equals_one(self):
        self.near(self.g["bound"], 35.9286, 4)
        self.near(self.g["blur_ablation"]["ours_clip_mean"], 35.2719, 4)
        self.near(self.g["blur_ablation"]["reference_clip_mean"], 38.0024, 4)
        self.assertEqual(self.g["n_ours_renders"], 1)
        self.near(self.g["ours_discrimination"]["margin"], 0.3370, 4)
        self.assertEqual(self.g["ours_discrimination"]["per_frame_true_wins"], 15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
