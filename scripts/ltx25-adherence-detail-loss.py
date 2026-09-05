#!/usr/bin/env python3
"""Does our LTX-2.5 render's SMOOTHNESS cause its prompt-adherence gap?

`LTX25-PROMPT-ADHERENCE` W3 measured our render at a mean CLIP score of 35.2719
against a bound of 36.0087 taken from the reference's own 25 frames, and
`ltx25-oracle-absolute.md` separately records our render as less sharp, less
blocky and less clipped than that reference. Those two readings are CONSISTENT
and nothing connects them. This script runs the measurement that would, and
`.agents/specs/ltx25-adherence-detail-loss.md` (#2513) is its spec.

FOUR THINGS, IN THIS ORDER, AND THE ORDER IS THE POINT.

  1. TONE FIRST. A global gain, a per-channel level, a gamma or a compressed
     dynamic range depresses sharpness, the clipped fraction and CLIP together
     with NO detail lost at all, and its repair is not a detail repair. So the
     luminance distributions are compared, a gain-and-gamma map is fitted
     between them by quantile, our frames are corrected with it, and the
     corrected frames are RESCORED. If the gap survives the correction, tone is
     not the explanation, and that is a measurement rather than an assumption.

  2. THE SPECTRUM, because "smoother" is a word and a spectrum is a number. A
     radially averaged power spectrum per frame on luma, windowed and unwindowed
     so the conclusion cannot rest on the window. The per-bin ratio says whether
     the difference is a uniform SCALE (which is item 1 again, in disguise), a
     high-frequency ROLLOFF, or one band.

  3. THE CORRELATION, which is the hypothesis test. If our low-detail frames are
     our low-scoring frames, the mechanism is supported. If they are not, the
     smoothness hypothesis is FALSIFIED. Reported with its n and its null, and
     never converted into a threshold.

  4. THE CAUSAL ABLATION, which is the only item that can establish a direction.
     Correlation within one render is an association. BLURRING THE REFERENCE is
     an intervention: if removing detail from upstream's own frames costs CLIP
     score at the rate our shortfall needs, detail loss is sufficient to produce
     the gap. If blurring the reference all the way down to OUR sharpness barely
     moves its score, then no amount of smoothness explains a 2.86 shortfall and
     the cause is elsewhere.

WHAT THIS SCRIPT MAY NOT DO. It does not touch the engine, it moves no bound,
and the sharpening arm at the end is a DIAGNOSTIC and not a proposal. Fitting an
output to raise a gate score is what AGENTS.md forbids; measuring what would
raise it is how you find out what is broken. The two are distinguished by
whether anything is landed, and this lands nothing.

n IS 1 ON OUR SIDE. The line `[ "$i" = 1 ] || rm -f "$D"/frame_*.ppm` in
`scripts/ltx25-render-confirm.sh` deletes renders 2 and 3 by design, so the
run-to-run spread of our own render does not exist in pixels. The line is quoted
rather than numbered because the number moved: `:505` at `592e224e7`, which
produced these frames, and `:551` on `origin/main` at `63889449c`. Every number
this script prints about our render is one render.

The CLIP scores come from the landed instrument itself: `ClipAdherenceScorer`
and the pin check are imported from `ltx25-render-compare.py` rather than
reimplemented, so these per-frame numbers are the same instrument that produced
the verdict. That tool publishes per-prompt reductions and not the per-frame
matrix, which is the only reason this file calls the scorer directly.

Usage:

    ltx25-adherence-detail-loss.py --ours <dir> --reference <dir> \
        --adherence-model <dir> --json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_cmp = __import__("ltx25-render-compare")

read_ppm = _cmp.read_ppm
frame_paths = _cmp.frame_paths
luma = _cmp.luma
sharpness_map = _cmp.sharpness_map
UnreadableInput = _cmp.UnreadableInput

EXIT_UNREADABLE = 2

# The digest `ltx25-prompt-adherence.md` records for the 25 frames W3 scored.
# Checked rather than trusted: this repository has gated against the wrong
# checkpoint before, and the share is CIFS and soft-mounted, so a streaming read
# can truncate a file without failing.
OURS_SET_DIGEST = "1166b28694001c52a6b5258804f1bb8f97ea2834dac5f16b5a9f5b48469d93ae"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ours(d: str) -> dict:
    """The concatenated-digest form the spec records, recomputed here."""
    paths = frame_paths(d)
    lines = "".join(f"{sha256_file(p)}  {os.path.basename(p)}\n" for p in paths)
    got = hashlib.sha256(lines.encode("ascii")).hexdigest()
    if got != OURS_SET_DIGEST:
        raise UnreadableInput(
            f"{d}: the 25 frames hash to {got}, not the {OURS_SET_DIGEST} that "
            f"ltx25-prompt-adherence.md records for the set W3 scored. These are "
            f"not those frames, so nothing measured on them compares to that "
            f"verdict")
    return {"frames": len(paths), "set_digest": got, "verified": True}


def verify_reference(d: str, sums_path: str) -> dict:
    """Every reference frame against its committed digest. All 25, or refuse."""
    want = {}
    with open(sums_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, name = line.partition("  ")
            name = os.path.basename(name.strip())
            if name.startswith("frame_") and name.endswith(".ppm"):
                want[name] = digest.strip()
    if not want:
        raise UnreadableInput(f"{sums_path}: lists no frame digests")
    paths = frame_paths(d)
    checked = 0
    for p in paths:
        n = os.path.basename(p)
        if n not in want:
            raise UnreadableInput(f"{n} is not in {sums_path}, so it is unidentified")
        got = sha256_file(p)
        if got != want[n]:
            raise UnreadableInput(f"{p}: sha256 {got}, committed {want[n]}")
        checked += 1
    if checked != len(want):
        raise UnreadableInput(
            f"{d}: {checked} frames against {len(want)} committed digests")
    return {"frames": checked, "digests_verified": checked, "verified": True}


# --- 1. TONE, and it runs first ---------------------------------------------
def tone_terms(frames: list[np.ndarray]) -> dict:
    """Per-frame level, spread, tails and range, per channel and on luma.

    A global tone difference is the explanation that costs nothing to test and
    invalidates everything downstream if it is true, which is why it is here and
    not at the end.
    """
    per = []
    for a in frames:
        f = a.astype(np.float64)
        l = luma(a)
        e = {"luma_mean": float(l.mean()), "luma_sd": float(l.std()),
             "luma_p05": float(np.percentile(l, 5)),
             "luma_p50": float(np.percentile(l, 50)),
             "luma_p95": float(np.percentile(l, 95)),
             "luma_min": float(l.min()), "luma_max": float(l.max()),
             "clipped_fraction": float(((a == 0) | (a == 255)).mean()),
             "sharpness": float(sharpness_map(l).mean())}
        for i, c in enumerate("rgb"):
            e[f"{c}_mean"] = float(f[..., i].mean())
            e[f"{c}_sd"] = float(f[..., i].std())
        per.append(e)
    keys = list(per[0].keys())
    return {"per_frame": per,
            "mean": {k: float(np.mean([p[k] for p in per])) for k in keys},
            "sd": {k: float(np.std([p[k] for p in per], ddof=1)) for k in keys}}


def luma_quantiles(frames: list[np.ndarray], n: int = 101) -> np.ndarray:
    """The pooled luminance quantile curve of a whole render.

    Two renders of one prompt are not pixel-aligned, so a per-pixel transfer
    function is meaningless and a quantile-to-quantile one is not. This is the
    curve a gain-and-gamma map is fitted between.
    """
    pool = np.concatenate([luma(a).reshape(-1) for a in frames])
    return np.percentile(pool, np.linspace(0.0, 100.0, n))


def fit_gain_gamma(src_q: np.ndarray, dst_q: np.ndarray) -> dict:
    """Least squares for `dst = 255 * a * (src/255) ** g`, fitted in log space.

    The interior quantiles only: a 0th or 100th percentile that has clipped
    carries no information about the transfer and would drag the fit.
    """
    s = np.clip(src_q / 255.0, 1e-4, 1.0)
    d = np.clip(dst_q / 255.0, 1e-4, 1.0)
    m = (src_q > 2.0) & (src_q < 253.0) & (dst_q > 2.0) & (dst_q < 253.0)
    if m.sum() < 8:
        raise UnreadableInput(
            "fewer than 8 usable quantiles: one render has collapsed to its "
            "endpoints and no transfer can be fitted through it")
    x, y = np.log(s[m]), np.log(d[m])
    g, la = np.polyfit(x, y, 1)
    a = float(np.exp(la))
    pred = 255.0 * a * (s[m] ** g)
    resid = float(np.sqrt(np.mean((pred - dst_q[m]) ** 2)))
    return {"gain": a, "gamma": float(g), "rms_residual_levels": resid,
            "quantiles_used": int(m.sum())}


def apply_gain_gamma(a: np.ndarray, gain: float, gamma: float) -> np.ndarray:
    """The fitted map, applied per channel and rounded back to 8 bits."""
    f = np.clip(a.astype(np.float64) / 255.0, 0.0, 1.0)
    out = 255.0 * gain * (f ** gamma)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --- 2. THE SPECTRUM ---------------------------------------------------------
def hann2d(h: int, w: int) -> np.ndarray:
    """A separable Hann window. KEPT AS A DIAGNOSTIC ARM ONLY -- see
    `periodic_component`, which is what every reported number now uses.

    A window suppresses the wrap step, but it also tapers roughly half the
    frame's area to near zero, so it measures the middle of the picture and
    weights the rest away. On these two renders that over-correction inflated
    the high-band ratio by a factor of 13 and produced a headline this row had
    to withdraw. It stays so the three conventions can be printed side by side.
    """
    return np.outer(np.hanning(h), np.hanning(w))


def periodic_component(u: np.ndarray) -> np.ndarray:
    """Moisan's periodic-plus-smooth decomposition, which this row uses for the
    radial PROFILE's shape and for nothing that decides anything.

    IT IS NOT THE ESTIMATOR ANY VERDICT COMES FROM. `welch_psd` is, and this
    docstring said the opposite of that one for as long as both existed: this
    one claimed to be "the ONLY spectrum this row reports from" while
    `welch_psd` claimed to be "what every reported spectral number in this row
    comes from". The code settles it -- `band_terms`, `nyquist_axes` and every
    band figure read `welch_psd`, and `out["spectrum"]["ours_mean_profile"]` is
    the only consumer of this function's output. Moisan removes the wrap step
    exactly but leaves a low-frequency counter-term that scales with the
    boundary jump, which the two renders do not share, so it is not border-free
    in the sense a band comparison needs.

    THE DEFECT IT REPAIRS, MEASURED RATHER THAN ASSERTED. The DFT treats a frame
    as periodic, so the jump from its last row to its first is a step the picture
    does not contain. A step's leakage falls as 1/f^2, which piles energy into
    the LOW bins and therefore shrinks every high-band SHARE through the
    denominator. It also lands on the fx and fy AXES, which is exactly where a
    separable-upsampling artefact would be read.

    Both failures were live here. Our frames' top-to-bottom wrap jump is 73.39
    against the reference's 49.48, on interior steps of 10.68 and 11.32 -- so the
    contamination is UNEQUAL between the two renders and does not cancel in a
    ratio. Measured on the same 25 frames a side, the high-band comparison reads
    -25.25% raw, +77.51% Hann-windowed, +5.78% here, and +22.68% under Welch.

    THE AXIS-TO-RING FIGURES THAT WITHDREW THE UPSAMPLER LEAD ARE WELCH'S, and
    they are in the committed JSON under `bands/*_nyquist_axes`: 2.46x/2.22x for
    ours against 2.23x/1.78x for the reference, down from the published Hann
    reading of 7.35x/7.27x against 4.86x/3.91x. An earlier version of this
    docstring gave that collapse as "7.3x-against-4.9x to 6.70x-against-5.74x",
    a periodic-convention pair that appears in no artifact this row commits and
    that no reader could recompute. One withdrawal, one derivation.

    The decomposition splits `u = p + s` where `s` is smooth and carries the
    boundary jump and `p` is periodic with no wrap discontinuity. It removes the
    artefact exactly, without a window and without touching interior content, so
    it costs neither area nor resolution.

    Moisan, "Periodic plus Smooth Image Decomposition", JMIV 2011.
    """
    h, w = u.shape
    v = np.zeros_like(u, dtype=np.float64)
    v[0, :] += u[-1, :] - u[0, :]
    v[-1, :] += u[0, :] - u[-1, :]
    v[:, 0] += u[:, -1] - u[:, 0]
    v[:, -1] += u[:, 0] - u[:, -1]
    i = np.arange(h)[:, None]
    j = np.arange(w)[None, :]
    den = 2.0 * np.cos(2.0 * np.pi * i / h) + 2.0 * np.cos(2.0 * np.pi * j / w) - 4.0
    den[0, 0] = 1.0
    ss = np.fft.fft2(v) / den
    ss[0, 0] = 0.0
    return np.real(np.fft.ifft2(np.fft.fft2(u) - ss))


TILE_SIZE = 64
TILE_STEP = 32


def welch_psd(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Per-frame Welch PSD over INTERIOR tiles. Every spectral number this row
    draws a conclusion from comes from here, and the reason is measured.

    The one spectral output that does NOT is the radial profile's shape in
    `out["spectrum"]["ours_mean_profile"]`, which is Moisan's periodic
    component and is kept for shape only. `periodic_component`'s own docstring
    once claimed the opposite of this sentence; the code decides, and it reads
    this function for `band_terms`, `nyquist_axes` and every band figure.

    THREE WHOLE-FRAME CONVENTIONS GAVE THREE ANSWERS ON THESE FRAMES, two of
    them with different signs. The high-band comparison reads -25.25% raw,
    +77.51% Hann-windowed and +5.78% periodic. Each is a different artefact:

      raw       The DFT treats the frame as periodic, so its wrap edge is a step
                the picture does not contain, and a step leaks as 1/f^2 into the
                LOW bins -- inflating the denominator and shrinking the high-band
                SHARE. Our frames' wrap jump is 73.39 against the reference's
                49.48 on near-identical interior steps, so the contamination is
                UNEQUAL and does not cancel in a ratio.
      hann      A whole-frame window kills that leak but tapers roughly half the
                frame's area to nothing, so it measures the middle of the picture
                and weights the rest away.
      periodic  Moisan's decomposition removes the wrap step exactly, but
                `p = u - s` leaves a low-frequency counter-term whose size scales
                with the boundary jump -- which, again, the two renders do not
                share. On a high-frequency test sinusoid that counter-term
                outweighs the signal's own ring mean.

    Welch tiling has none of those. Interior tiles never touch the frame border,
    so no wrap step can enter and no boundary counter-term is created; every tile
    is weighted equally, so there is no whole-frame taper bias; and the per-tile
    Hann only has to suppress each tile's own edges, which are interior content
    and not an invented discontinuity. It is the standard PSD estimator for
    exactly this reason.

    It trades frequency resolution for that: a 64-pixel tile resolves 1/64 rather
    than 1/320, which is ample for a band comparison and useless for locating a
    single line. This row compares bands.
    """
    h, w = luma(frames[0]).shape
    win = np.outer(np.hanning(TILE_SIZE), np.hanning(TILE_SIZE))
    lo = TILE_STEP
    ys = range(lo, h - TILE_SIZE - lo + 1, TILE_STEP)
    xs = range(lo, w - TILE_SIZE - lo + 1, TILE_STEP)
    if not len(list(ys)) or not len(list(xs)):
        raise UnreadableInput(
            f"a {h}x{w} frame holds no interior {TILE_SIZE}px tile at step "
            f"{TILE_STEP}, so no border-free spectrum can be estimated from it")
    out = []
    for a in frames:
        l = luma(a)
        acc, n = None, 0
        for y in range(lo, h - TILE_SIZE - lo + 1, TILE_STEP):
            for x in range(lo, w - TILE_SIZE - lo + 1, TILE_STEP):
                t = l[y:y + TILE_SIZE, x:x + TILE_SIZE]
                t = (t - t.mean()) * win
                f = np.fft.fftshift(np.fft.fft2(t))
                pw = f.real ** 2 + f.imag ** 2
                acc = pw if acc is None else acc + pw
                n += 1
        out.append(acc / n)
    return out


def tile_freq_grid() -> np.ndarray:
    fy = np.fft.fftshift(np.fft.fftfreq(TILE_SIZE))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(TILE_SIZE))[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def wrap_discontinuity(frames: list[np.ndarray]) -> dict:
    """How big the step is that the DFT's periodic assumption invents, against
    the frame's own mean interior step so it is a ratio and not a level.

    Reported because the two renders differ on it, which is what made the raw
    spectrum's verdict a property of the border rather than of the picture.
    """
    vj, hj, iv = [], [], []
    for a in frames:
        l = luma(a)
        vj.append(float(np.abs(l[0, :] - l[-1, :]).mean()))
        hj.append(float(np.abs(l[:, 0] - l[:, -1]).mean()))
        iv.append(0.5 * (float(np.abs(np.diff(l, axis=0)).mean())
                         + float(np.abs(np.diff(l, axis=1)).mean())))
    v, h_, i_ = float(np.mean(vj)), float(np.mean(hj)), float(np.mean(iv))
    return {"top_bottom_jump": v, "left_right_jump": h_,
            "mean_interior_step": i_, "jump_over_interior_step": (v + h_) / 2.0 / i_}


def radial_power(l: np.ndarray, window: np.ndarray | None, nbins: int = 64,
                 convention: str = "periodic") -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of one luma plane.

    Frequency is in cycles per pixel on each axis, so two renders of identical
    geometry share the bin grid exactly and no resampling enters the comparison.
    The DC bin is dropped: it is the frame's mean, which item 1 measures.

    `convention` selects how the wrap step is handled, and it is a parameter
    rather than a constant because this row got a headline wrong by leaving it
    implicit. `periodic` is the reported one. `raw` and `hann` are printed beside
    it so a reader sees the spread the choice is worth instead of trusting one.
    """
    if convention == "periodic":
        l = periodic_component(l)
    x = l - l.mean()
    if convention == "hann" and window is not None:
        x = x * window
    f = np.fft.fftshift(np.fft.fft2(x))
    p = (f.real ** 2 + f.imag ** 2) / float(x.size)
    h, w = l.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.digitize(r.reshape(-1), edges) - 1
    pv = p.reshape(-1)
    keep = (idx >= 0) & (idx < nbins) & (r.reshape(-1) > 0)
    sums = np.bincount(idx[keep], weights=pv[keep], minlength=nbins)
    cnts = np.bincount(idx[keep], minlength=nbins).astype(np.float64)
    prof = np.where(cnts > 0, sums / np.maximum(cnts, 1.0), np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


def spectra(frames: list[np.ndarray], nbins: int = 64) -> dict:
    """All three conventions, every run. `periodic` is what gets reported."""
    h, w = luma(frames[0]).shape
    win = hann2d(h, w)
    prof_p, prof_w, prof_u = [], [], []
    centers = None
    for a in frames:
        l = luma(a)
        c, pp = radial_power(l, None, nbins, "periodic")
        _, pw = radial_power(l, win, nbins, "hann")
        _, pu = radial_power(l, None, nbins, "raw")
        centers = c
        prof_p.append(pp)
        prof_w.append(pw)
        prof_u.append(pu)
    return {"centers": centers, "periodic": np.array(prof_p),
            "windowed": np.array(prof_w), "unwindowed": np.array(prof_u)}


def hf_fraction(centers: np.ndarray, prof: np.ndarray, f_lo: float) -> np.ndarray:
    """Share of each frame's spectral ENERGY at or above `f_lo`.

    Energy, not the mean profile: each bin's mean power is weighted by the count
    of coefficients that fall in it, which is what makes this a fraction of the
    frame's variance rather than a fraction of a curve.
    """
    edges = np.linspace(0.0, 0.5, prof.shape[1] + 1)
    # Annulus area is proportional to the coefficient count in the limit, and
    # using the geometric weight rather than the exact count keeps this a pure
    # function of the profile that a reader can recompute.
    wgt = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    good = np.isfinite(prof)
    e = np.where(good, prof, 0.0) * wgt[None, :]
    hi = centers >= f_lo
    return e[:, hi].sum(axis=1) / np.maximum(e.sum(axis=1), 1e-30)


# --- 2b. WHICH BAND, and whether the excess one is noise ---------------------
# The radial profile answers "is it a rolloff", and the answer here was NO, so
# these three ask the question that replaced it: WHICH band moved, in which
# direction, and does the band we hold MORE of behave like detail or like grain.
BAND_MID = (0.04, 0.14)   # object-scale structure at 320x192: 7 to 25 pixel periods
BAND_HI = (0.20, 0.71)    # everything from a 5-pixel period up to the corner


def _fgrid(h: int, w: int) -> np.ndarray:
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def _power(l: np.ndarray, window: np.ndarray | None = None) -> np.ndarray:
    x = l - l.mean()
    if window is not None:
        x = x * window
    f = np.fft.fftshift(np.fft.fft2(x))
    return (f.real ** 2 + f.imag ** 2) / float(x.size)


def band_terms(frames: list[np.ndarray]) -> dict:
    """Per-frame share of spectral energy in the mid and high bands, plus the
    TEMPORAL CHURN of the high band.

    Churn is the high-band energy of a consecutive-frame DIFFERENCE over the
    high-band energy of a frame. Detail that belongs to the scene survives from
    one frame to the next and churns little; per-frame grain does not survive
    and churns near unity. It is a discriminator between "we render more fine
    detail" and "we render more noise", and it is reported with both renders'
    values because only the COMPARISON means anything.
    """
    r = tile_freq_grid()
    mid_m = (r >= BAND_MID[0]) & (r < BAND_MID[1])
    hi_m = (r >= BAND_HI[0]) & (r < BAND_HI[1])
    tot_m = r > 0
    P = welch_psd(frames)
    mid = np.array([float(p[mid_m].sum() / p[tot_m].sum()) for p in P])
    hi = np.array([float(p[hi_m].sum() / p[tot_m].sum()) for p in P])
    # ABSOLUTE band power too. A share moves when EITHER band moves, so a share
    # alone cannot say which side changed, and on these renders the answer is
    # that the mid bands are equal and the high band and the total are not.
    mid_abs = np.array([float(p[mid_m].sum()) for p in P])
    hi_abs = np.array([float(p[hi_m].sum()) for p in P])
    tot_abs = np.array([float(p[tot_m].sum()) for p in P])
    D = welch_psd([np.clip(frames[i + 1].astype(np.int16)
                           - frames[i].astype(np.int16) + 128, 0, 255).astype(np.uint8)
                   for i in range(len(frames) - 1)])
    churn = (float(np.mean([p[hi_m].sum() for p in D]))
             / float(np.mean([p[hi_m].sum() for p in P])))
    return {"mid_fraction": mid, "hi_fraction": hi, "hi_temporal_churn": churn,
            "mid_power": mid_abs, "hi_power": hi_abs, "total_power": tot_abs}


def nyquist_axes(frames: list[np.ndarray]) -> dict:
    """Near-Nyquist energy ON the two axes against the radial ring around them.

    A separable upsampling stage -- transposed convolution, pixel shuffle, a
    nearest-neighbour rung -- puts energy at the sampling lattice's own
    frequencies, which lie on the fx and fy axes near 0.5 and NOT on the
    diagonal. A render whose fine structure is scene content spreads that ring
    evenly instead. So the axis-to-ring contrast separates an artefact of the
    decoder's geometry from a picture with texture in it, and an ASYMMETRY
    between the two axes says the stage is not isotropic.
    """
    fy = np.fft.fftshift(np.fft.fftfreq(TILE_SIZE))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(TILE_SIZE))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    # WELCH, and this probe is the whole reason it matters. The wrap step lands
    # ON these axes and our frames' step is the larger one, so a whole-frame
    # estimator reads a BORDER difference as a decoder signature. It did: the
    # raw and Hann conventions put the axis-to-ring contrast at roughly 7x for us
    # against 4-5x for the reference, and this row published a separable-upsampler
    # lead on it. Estimated border-free the contrast is 2.46x/2.22x against
    # 2.23x/1.78x, which is a weak difference, and the lead was withdrawn.
    P = np.mean(welch_psd(frames), axis=0)
    tol = 1.0 / TILE_SIZE * 1.01
    horiz = np.broadcast_to((np.abs(fy) < tol) & (np.abs(np.abs(fx) - 0.5) < tol), P.shape)
    vert = np.broadcast_to((np.abs(np.abs(fy) - 0.5) < tol) & (np.abs(fx) < tol), P.shape)
    corner = np.broadcast_to((np.abs(np.abs(fy) - 0.5) < tol)
                             & (np.abs(np.abs(fx) - 0.5) < tol), P.shape)
    ring = (r > 0.42) & (r < 0.52)
    return {"horizontal_nyquist": float(P[horiz].mean()),
            "vertical_nyquist": float(P[vert].mean()),
            "diagonal_corner": float(P[corner].mean()),
            "radial_ring_mean": float(P[ring].mean())}


# --- 3. CORRELATION ----------------------------------------------------------
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    d = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    return float((xc * yc).sum() / d) if d > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, dtype=np.float64)))
        return o.astype(np.float64)
    return pearson(rank(x), rank(y))


# --- 4. THE INTERVENTION -----------------------------------------------------
def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian on an RGB frame, edge-replicated.

    The intervention. It removes fine detail and changes nothing else about the
    frame's content, so a CLIP score measured after it is a measurement of what
    detail loss alone costs.
    """
    if sigma <= 0:
        return a.copy()
    rad = max(1, int(np.ceil(3.0 * sigma)))
    t = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-(t ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    f = a.astype(np.float64)
    for axis in (0, 1):
        f = np.apply_along_axis(
            lambda v: np.convolve(np.pad(v, rad, mode="edge"), k, mode="valid"),
            axis, f)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def unsharp(a: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """The symmetric diagnostic arm: add back what a blur would remove.

    A DIAGNOSTIC, not a proposal. It says whether the instrument responds to
    detail in the direction the hypothesis needs. Nothing in the engine does
    this and this row proposes that nothing should.
    """
    b = gaussian_blur(a, sigma).astype(np.float64)
    f = a.astype(np.float64)
    return np.clip(np.rint(f + amount * (f - b)), 0, 255).astype(np.uint8)


def mean_sharpness(frames: list[np.ndarray]) -> float:
    return float(np.mean([sharpness_map(luma(a)).mean() for a in frames]))


# --- the run -----------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--sha256sums", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests", "parity",
        "goldens", "ltx2_oracle", "SHA256SUMS"))
    ap.add_argument("--adherence-model", required=True)
    ap.add_argument("--blur-sigmas", default="0.3,0.5,0.7,1.0,1.5,2.0")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    out: dict = {"n_ours_renders": 1,
                 "n_ours_renders_note":
                     "the confirm harness retains only render 1, so no "
                     "run-to-run spread of our render exists in pixels"}

    # IDENTITY BEFORE PIXELS.
    out["ours_identity"] = verify_ours(args.ours)
    out["reference_identity"] = verify_reference(args.reference, args.sha256sums)
    ours = [read_ppm(p) for p in frame_paths(args.ours)]
    ref = [read_ppm(p) for p in frame_paths(args.reference)]
    print(f"[identity] ours 25 frames -> {out['ours_identity']['set_digest'][:16]}..., "
          f"reference {out['reference_identity']['digests_verified']} digests verified")

    # --- 1. TONE ---
    out["tone"] = {"ours": tone_terms(ours), "reference": tone_terms(ref)}
    qo, qr = luma_quantiles(ours), luma_quantiles(ref)
    out["tone"]["ours_luma_quantiles"] = [float(v) for v in qo]
    out["tone"]["reference_luma_quantiles"] = [float(v) for v in qr]
    fit = fit_gain_gamma(qo, qr)
    out["tone"]["fit_ours_to_reference"] = fit
    print(f"[tone] ours luma mean {out['tone']['ours']['mean']['luma_mean']:.4f} "
          f"sd {out['tone']['ours']['mean']['luma_sd']:.4f} | reference "
          f"{out['tone']['reference']['mean']['luma_mean']:.4f} / "
          f"{out['tone']['reference']['mean']['luma_sd']:.4f}")
    print(f"[tone] fitted ours->reference gain {fit['gain']:.5f} gamma "
          f"{fit['gamma']:.5f} rms residual {fit['rms_residual_levels']:.4f} levels")
    ours_tone = [apply_gain_gamma(a, fit["gain"], fit["gamma"]) for a in ours]
    out["tone"]["ours_tonematched"] = tone_terms(ours_tone)

    # --- 2. SPECTRUM ---
    so, sr = spectra(ours), spectra(ref)
    centers = so["centers"]
    prof_o = np.nanmean(so["periodic"], axis=0)
    prof_r = np.nanmean(sr["periodic"], axis=0)
    ratio = prof_o / prof_r
    out["spectrum"] = {
        "bin_centers_cycles_per_pixel": [float(v) for v in centers],
        "ours_mean_profile": [float(v) for v in prof_o],
        "reference_mean_profile": [float(v) for v in prof_r],
        "ratio_ours_over_reference": [float(v) for v in ratio],
        "convention": "periodic",
        "why_periodic": (
            "the DFT's wrap step leaks as 1/f^2 into the LOW bins and onto the "
            "fx/fy axes, and the two renders do not carry the same step, so the "
            "raw and Hann conventions each returned a different SIGN for the "
            "high band on these frames"),
        "ours_mean_profile_unwindowed":
            [float(v) for v in np.nanmean(so["unwindowed"], axis=0)],
        "reference_mean_profile_unwindowed":
            [float(v) for v in np.nanmean(sr["unwindowed"], axis=0)],
        "ours_mean_profile_hann":
            [float(v) for v in np.nanmean(so["windowed"], axis=0)],
        "reference_mean_profile_hann":
            [float(v) for v in np.nanmean(sr["windowed"], axis=0)],
    }
    # THE CROSSOVER, defined rather than eyeballed: the lowest bin above which
    # EVERY bin's ratio stays below 1. A uniform deficit has its crossover at
    # the first bin and is a scale difference; a rolloff has it in the middle.
    # A BIN WITH NO COEFFICIENTS IS NOT A BIN BELOW UNITY. An empty annulus
    # yields NaN, `NaN < 1.0` is False, and a bottom row of empty bins would
    # therefore push the crossover upward and read as a rolloff that is not
    # there. Empty bins carry no information and are skipped in both directions.
    have = np.isfinite(ratio)
    below = have & (ratio < 1.0)
    cross = None
    for i in range(len(ratio)):
        if not have[i]:
            continue
        tail = have[i:]
        if tail.any() and np.all(below[i:][tail]):
            cross = float(centers[i])
            break
    out["spectrum"]["crossover_cycles_per_pixel"] = cross
    print(f"[spectrum] ratio at f=0.05 {np.interp(0.05, centers, ratio):.4f}, "
          f"0.15 {np.interp(0.15, centers, ratio):.4f}, "
          f"0.25 {np.interp(0.25, centers, ratio):.4f}, "
          f"0.40 {np.interp(0.40, centers, ratio):.4f}; crossover {cross}")

    # THE CORRELATION'S OWN INPUT IS THE WELCH SHARE, not a radial-profile
    # reduction. The radial profile above is kept for shape, but every number
    # that enters a verdict comes from the border-free estimator.
    hf_o = band_terms(ours)["hi_fraction"]
    hf_r = band_terms(ref)["hi_fraction"]
    out["spectrum"]["hf_cut_cycles_per_pixel"] = float(BAND_HI[0])
    out["spectrum"]["hf_source"] = "welch interior tiles, band >= 0.20 c/px"
    out["spectrum"]["ours_hf_energy_fraction"] = [float(v) for v in hf_o]
    out["spectrum"]["reference_hf_energy_fraction"] = [float(v) for v in hf_r]
    out["spectrum"]["ours_hf_energy_fraction_mean"] = float(hf_o.mean())
    out["spectrum"]["reference_hf_energy_fraction_mean"] = float(hf_r.mean())
    # THE LABEL IS `BAND_HI`, WHICH IS THE BAND THESE NUMBERS ACTUALLY SPAN.
    # It said `f_lo` -- the CROSSOVER, 0.4336 on these frames -- for numbers
    # `band_terms` cuts at 0.20, so the console mislabelled its own band by 2.2x
    # while the JSON's `hf_cut_cycles_per_pixel` recorded 0.20 correctly.
    print(f"[spectrum] energy in the band [{BAND_HI[0]:.4f}, {BAND_HI[1]:.4f}) "
          f"c/px: ours {hf_o.mean():.6f}, reference {hf_r.mean():.6f}, "
          f"relative loss {1.0 - hf_o.mean() / hf_r.mean():+.4f}")

    # THE CONVENTION SPREAD, printed rather than chosen silently. A reviewer
    # reproduced this row's high band with the raw convention and got the
    # OPPOSITE SIGN; both readings were the border, not the picture. The three
    # numbers now ship side by side so that disagreement is visible in the
    # output instead of being discovered in review.
    # PER FRAME, not whole-render. The whole-render share is the share of the
    # MEAN spectrum, which is what a band comparison wants; a CORRELATION needs
    # one number per frame, and the row published a four-estimator correlation
    # table while this helper could only ever return two whole-render scalars.
    # Both come off the same power arrays here, so they cannot disagree about
    # which convention they are.
    def _conv_frame_powers(frames, conv):
        if conv == "welch":
            return welch_psd(frames), tile_freq_grid()
        h_, w_ = luma(frames[0]).shape
        win_ = hann2d(h_, w_)
        out_ = []
        for a in frames:
            l_ = luma(a)
            if conv == "periodic":
                l_ = periodic_component(l_)
            out_.append(_power(l_, win_ if conv == "hann" else None))
        return out_, _fgrid(h_, w_)

    def _conv_shares(frames, conv):
        """(whole-render mid, whole-render high, per-frame mid, per-frame high)."""
        P_, rr = _conv_frame_powers(frames, conv)
        mm = (rr >= BAND_MID[0]) & (rr < BAND_MID[1])
        hh = (rr >= BAND_HI[0]) & (rr < BAND_HI[1])
        tt = rr > 0
        if conv == "welch":
            acc = np.mean(P_, axis=0)
        else:
            acc = None
            for pw_ in P_:
                acc = pw_ if acc is None else acc + pw_
            acc = acc / len(P_)
        per_mid = np.array([float(p[mm].sum() / p[tt].sum()) for p in P_])
        per_hi = np.array([float(p[hh].sum() / p[tt].sum()) for p in P_])
        return (float(acc[mm].sum() / acc[tt].sum()),
                float(acc[hh].sum() / acc[tt].sum()), per_mid, per_hi)

    out["convention_sensitivity"] = {}
    per_frame_hi_by_conv: dict = {}
    for conv in ("raw", "hann", "periodic", "welch"):
        om, oh, o_pm, o_ph = _conv_shares(ours, conv)
        rm, rh, r_pm, r_ph = _conv_shares(ref, conv)
        per_frame_hi_by_conv[conv] = (o_ph, r_ph)
        out["convention_sensitivity"][conv] = {
            "ours_mid": om, "reference_mid": rm, "mid_delta": om / rm - 1.0,
            "ours_high": oh, "reference_high": rh, "high_delta": oh / rh - 1.0,
            "ours_high_per_frame": [float(v) for v in o_ph],
            "reference_high_per_frame": [float(v) for v in r_ph],
            "ours_mid_per_frame": [float(v) for v in o_pm],
            "reference_mid_per_frame": [float(v) for v in r_pm]}
        print(f"[conv] {conv:9s} mid {om:.6f}/{rm:.6f} ({om/rm-1:+.2%})  "
              f"high {oh:.6f}/{rh:.6f} ({oh/rh-1:+.2%})")
    out["wrap_discontinuity"] = {"ours": wrap_discontinuity(ours),
                                 "reference": wrap_discontinuity(ref)}
    wo, wr = out["wrap_discontinuity"]["ours"], out["wrap_discontinuity"]["reference"]
    print(f"[conv] wrap step over interior step: ours "
          f"{wo['jump_over_interior_step']:.2f}x, reference "
          f"{wr['jump_over_interior_step']:.2f}x -- UNEQUAL, so it does not "
          f"cancel in a ratio")

    bo, br = band_terms(ours), band_terms(ref)
    ao, ar = nyquist_axes(ours), nyquist_axes(ref)
    out["bands"] = {
        "mid_band_cycles_per_pixel": list(BAND_MID),
        "high_band_cycles_per_pixel": list(BAND_HI),
        "ours_mid_fraction_mean": float(bo["mid_fraction"].mean()),
        "reference_mid_fraction_mean": float(br["mid_fraction"].mean()),
        "ours_hi_fraction_mean": float(bo["hi_fraction"].mean()),
        "reference_hi_fraction_mean": float(br["hi_fraction"].mean()),
        "estimator": "welch interior tiles 64px/32px, hann per tile",
        "ours_mid_power_over_reference":
            float(bo["mid_power"].sum() / br["mid_power"].sum()),
        "ours_hi_power_over_reference":
            float(bo["hi_power"].sum() / br["hi_power"].sum()),
        "ours_total_power_over_reference":
            float(bo["total_power"].sum() / br["total_power"].sum()),
        "ours_hi_temporal_churn": bo["hi_temporal_churn"],
        "reference_hi_temporal_churn": br["hi_temporal_churn"],
        "ours_nyquist_axes": ao,
        "reference_nyquist_axes": ar,
    }
    print(f"[bands] mid {BAND_MID} share: ours {bo['mid_fraction'].mean():.6f} "
          f"reference {br['mid_fraction'].mean():.6f} "
          f"({bo['mid_fraction'].mean() / br['mid_fraction'].mean() - 1:+.2%})")
    print(f"[bands] high {BAND_HI} share: ours {bo['hi_fraction'].mean():.6f} "
          f"reference {br['hi_fraction'].mean():.6f} "
          f"({bo['hi_fraction'].mean() / br['hi_fraction'].mean() - 1:+.2%})")
    print(f"[bands] ABSOLUTE power ours/reference: mid "
          f"{out['bands']['ours_mid_power_over_reference']:.4f}x, high "
          f"{out['bands']['ours_hi_power_over_reference']:.4f}x, total "
          f"{out['bands']['ours_total_power_over_reference']:.4f}x")
    print(f"[bands] high-band temporal churn: ours {bo['hi_temporal_churn']:.4f} "
          f"reference {br['hi_temporal_churn']:.4f}")
    print(f"[bands] near-Nyquist axis/ring: ours "
          f"{ao['horizontal_nyquist']:.3f}/{ao['vertical_nyquist']:.3f} over ring "
          f"{ao['radial_ring_mean']:.3f}; reference "
          f"{ar['horizontal_nyquist']:.3f}/{ar['vertical_nyquist']:.3f} over ring "
          f"{ar['radial_ring_mean']:.3f}")

    # --- the instrument ---
    pin = _cmp.load_json_record(_cmp.DEFAULT_ADHERENCE_PIN, "the pinned scorer record")
    identity = _cmp.assert_scorer_identity(args.adherence_model, pin)
    decoys = _cmp.load_decoys(_cmp.DEFAULT_ADHERENCE_DECOYS)
    prompt = _cmp.true_prompt(_cmp.DEFAULT_ORACLE_MANIFEST)
    prompts = [prompt] + [d["text"] for d in decoys]
    labels = ["true"] + [f"{d['kind']}:{i}" for i, d in enumerate(decoys)]
    scorer = _cmp.ClipAdherenceScorer(args.adherence_model, pin)
    _cmp.refuse_overlong_prompts(prompts, scorer.count_tokens,
                                 int(pin.get("text_context_positions") or 0))
    out["scorer"] = identity
    out["prompt"] = prompt
    out["labels"] = labels

    # EVERY SCORED ARM IS CHECKED, NOT JUST THE REFERENCE. S0 raises on the
    # oracle's own frames because a scorer that fails there is broken. An
    # ablation arm cannot make the scorer broken, so it does not raise -- but a
    # blurred arm on which a DECOY outranks the true prompt has stopped
    # measuring prompt adherence just as completely, and its CLIP mean is a
    # number rather than a reading. This row published `+1.9131` from exactly
    # such an arm as its decisive result while S0 ran once, on the reference.
    def _arm(disc: dict, name: str) -> dict:
        v = _cmp.adherence_arm_valid(disc)
        if not v["valid"]:
            print(f"[arm] INVALID: {name} -- {v['reason']}. Nothing derived "
                  f"from this arm's CLIP mean is a reading of adherence")
        return v

    m_ref = scorer.score(ref, prompts)
    ref_disc = _cmp.discrimination(m_ref, labels)
    # S0 BEFORE ANY NUMBER OF OURS IS PUBLISHED, exactly as the landed tool does.
    out["s0"] = _cmp.scorer_precondition(ref_disc)
    m_ours = scorer.score(ours, prompts)
    ours_disc = _cmp.discrimination(m_ours, labels)
    out["reference_discrimination"] = ref_disc
    out["ours_discrimination"] = ours_disc
    ref_disc["scorer_valid"] = _arm(ref_disc, "reference, unblurred")
    ours_disc["scorer_valid"] = _arm(ours_disc, "ours, unblurred")
    clip_o = m_ours[:, 0]
    clip_r = m_ref[:, 0]
    out["per_frame_clip_true"] = {"ours": [float(v) for v in clip_o],
                                  "reference": [float(v) for v in clip_r]}
    bound = ref_disc["stats"][0]["frame_min"]
    out["bound"] = float(bound)
    print(f"[clip] S0 margin {out['s0']['margin']:+.4f}; ours mean "
          f"{clip_o.mean():.4f} vs bound {bound:.4f} "
          f"({(clip_o >= bound).sum()} of 25 frames clear it)")

    # --- 3. CORRELATION ---
    sharp_o = np.array([sharpness_map(luma(a)).mean() for a in ours])
    sharp_r = np.array([sharpness_map(luma(a)).mean() for a in ref])
    corr = {}
    for name, hf, cl, sh in (("ours", hf_o, clip_o, sharp_o),
                             ("reference", hf_r, clip_r, sharp_r)):
        corr[name] = {
            "n": int(len(cl)),
            "pearson_hf_vs_clip": pearson(hf, cl),
            "spearman_hf_vs_clip": spearman(hf, cl),
            "pearson_sharpness_vs_clip": pearson(sh, cl),
            "spearman_sharpness_vs_clip": spearman(sh, cl),
        }
    # `pearson_high_band_vs_clip` USED TO LIVE HERE AND IT WAS THE SAME NUMBER.
    # It reduced `band_terms(...)["hi_fraction"]`, which is exactly what `hf_o`
    # and `hf_r` above are, so the JSON carried one measurement twice under two
    # names and a reader saw two agreeing coefficients. The high band's
    # coefficient is `pearson_hf_vs_clip`; only the MID band is a second one.
    for name, b, cl in (("ours", bo, clip_o), ("reference", br, clip_r)):
        corr[name]["pearson_mid_band_vs_clip"] = pearson(b["mid_fraction"], cl)
    # THE FOUR ESTIMATORS, COMPUTED RATHER THAN TRANSCRIBED. The row published a
    # table claiming the within-render correlation "survives every estimator"
    # while the harness computed exactly one of them. Two of that table's four
    # figures appeared in no code and no artifact, and the raw convention -- the
    # one whose band delta reverses sign -- was absent from it entirely.
    for name, cl, sel in (("ours", clip_o, 0), ("reference", clip_r, 1)):
        corr[name]["pearson_hf_vs_clip_by_convention"] = {
            conv: pearson(per_frame_hi_by_conv[conv][sel], cl)
            for conv in ("raw", "hann", "periodic", "welch")}
        corr[name]["spearman_hf_vs_clip_by_convention"] = {
            conv: spearman(per_frame_hi_by_conv[conv][sel], cl)
            for conv in ("raw", "hann", "periodic", "welch")}
    hf_all = np.concatenate([hf_o, hf_r])
    cl_all = np.concatenate([clip_o, clip_r])
    sh_all = np.concatenate([sharp_o, sharp_r])
    corr["pooled"] = {
        "n": int(len(cl_all)),
        "pearson_hf_vs_clip": pearson(hf_all, cl_all),
        "spearman_hf_vs_clip": spearman(hf_all, cl_all),
        "pearson_sharpness_vs_clip": pearson(sh_all, cl_all),
        "spearman_sharpness_vs_clip": spearman(sh_all, cl_all),
        "caveat": ("the between-render difference dominates this pool, so it is "
                   "not independent evidence for a within-render mechanism"),
    }
    # THE TEMPORAL AXIS, which the per-frame table made visible and no earlier
    # item asked for. A render is 25 frames of one shot, so "which frame" is a
    # variable, and the two renders answer it differently: if one improves along
    # the clip while the other decays, the difference is not a property of the
    # spatial path at all.
    idx = np.arange(len(clip_o), dtype=np.float64)
    corr["temporal"] = {
        "n": int(len(idx)),
        "ours_r_frame_index_vs_clip": pearson(idx, clip_o),
        "reference_r_frame_index_vs_clip": pearson(idx, clip_r),
        "ours_r_frame_index_vs_hf": pearson(idx, hf_o),
        "reference_r_frame_index_vs_hf": pearson(idx, hf_r),
        "ours_clip_first5_minus_last5": float(clip_o[:5].mean() - clip_o[-5:].mean()),
        "reference_clip_first5_minus_last5":
            float(clip_r[:5].mean() - clip_r[-5:].mean()),
        "ours_hf_last5_over_first5": float(hf_o[-5:].mean() / hf_o[:5].mean()),
        "reference_hf_last5_over_first5": float(hf_r[-5:].mean() / hf_r[:5].mean()),
    }
    corr["null"] = ("no association; r = 0. 25 frames of one continuous shot are "
                    "not 25 independent samples, so the effective n is below 25 "
                    "and no p-value is quoted against an i.i.d. null")
    corr["is_a_gate"] = False
    out["correlation"] = corr
    print(f"[corr] within ours   r(hf,clip)={corr['ours']['pearson_hf_vs_clip']:+.4f} "
          f"rho={corr['ours']['spearman_hf_vs_clip']:+.4f} n=25")
    print(f"[corr] within ref    r(hf,clip)={corr['reference']['pearson_hf_vs_clip']:+.4f} "
          f"rho={corr['reference']['spearman_hf_vs_clip']:+.4f} n=25")
    print(f"[corr] pooled 50     r(hf,clip)={corr['pooled']['pearson_hf_vs_clip']:+.4f} "
          f"rho={corr['pooled']['spearman_hf_vs_clip']:+.4f} n=50")
    print(f"[corr] frame index vs clip: ours "
          f"{corr['temporal']['ours_r_frame_index_vs_clip']:+.4f}, reference "
          f"{corr['temporal']['reference_r_frame_index_vs_clip']:+.4f} -- our "
          f"render DECAYS along the clip where upstream's IMPROVES")

    # --- the per-frame table ---
    order_sharp = np.argsort(-sharp_o)
    clears = [int(i) for i in range(len(clip_o)) if clip_o[i] >= bound]
    out["per_frame_table"] = [
        {"frame": i, "clip_true": float(clip_o[i]),
         "clears_bound": bool(clip_o[i] >= bound),
         "sharpness": float(sharp_o[i]),
         "sharpness_rank": int(np.where(order_sharp == i)[0][0]) + 1,
         "hf_energy_fraction": float(hf_o[i]),
         "luma_mean": out["tone"]["ours"]["per_frame"][i]["luma_mean"]}
        for i in range(len(clip_o))]
    out["frames_clearing_bound"] = clears
    out["sharpest_five"] = [int(i) for i in order_sharp[:5]]

    # --- 4. THE INTERVENTION ---
    sig = [float(s) for s in args.blur_sigmas.split(",") if s.strip()]
    sweep = []
    base_sharp_r = mean_sharpness(ref)
    base_sharp_o = mean_sharpness(ours)
    for s in sig:
        blurred = [gaussian_blur(a, s) for a in ref]
        sm = mean_sharpness(blurred)
        sc = scorer.score(blurred, prompts)
        d = _cmp.discrimination(sc, labels)
        row = {"sigma": s, "sharpness": sm,
               "clip_true_mean": float(sc[:, 0].mean()),
               "clip_delta_vs_unblurred": float(sc[:, 0].mean() - clip_r.mean()),
               "argmax_label": d["argmax_label"],
               "margin_to_best_decoy": d["margin"],
               "per_frame_true_wins": d["per_frame_true_wins"],
               "scorer_valid": _arm(d, f"reference blurred sigma={s:.2f}")}
        sweep.append(row)
        print(f"[ablate] reference blurred sigma={s:.2f}: sharpness "
              f"{sm:.4f} (ours {base_sharp_o:.4f}), clip {row['clip_true_mean']:.4f} "
              f"({row['clip_delta_vs_unblurred']:+.4f}), argmax "
              f"{row['argmax_label']}, margin "
              f"{row['margin_to_best_decoy']:+.4f}, wins "
              f"{row['per_frame_true_wins']}/25, "
              f"{'READABLE' if row['scorer_valid']['valid'] else 'INVALID'}")
    out["blur_ablation"] = {
        "reference_sharpness": base_sharp_r,
        "ours_sharpness": base_sharp_o,
        "reference_clip_mean": float(clip_r.mean()),
        "ours_clip_mean": float(clip_o.mean()),
        "observed_gap": float(clip_r.mean() - clip_o.mean()),
        "sweep": sweep,
        "question": ("how much CLIP score does REMOVING detail from upstream's "
                     "own frames cost, at the sharpness our render actually has"),
    }
    # The sigma whose blurred reference matches OUR sharpness, by interpolation
    # over the sweep, and the CLIP cost the hypothesis therefore predicts.
    xs = np.array([r["sharpness"] for r in sweep])
    ys = np.array([r["clip_delta_vs_unblurred"] for r in sweep])
    if xs.size >= 2 and xs.min() <= base_sharp_o <= xs.max():
        o = np.argsort(xs)
        predicted = float(np.interp(base_sharp_o, xs[o], ys[o]))
    else:
        predicted = None
    out["blur_ablation"]["predicted_clip_cost_at_our_sharpness"] = predicted
    print(f"[ablate] blurring the reference to OUR sharpness "
          f"({base_sharp_o:.4f}) costs it {predicted} CLIP points; the observed "
          f"gap is {out['blur_ablation']['observed_gap']:+.4f}")

    # --- 4b. THE SYMMETRIC INTERVENTION, on OUR frames ---
    # The sweep above removes detail from the reference. This one removes our
    # EXCESS high-frequency energy from our own frames. If that energy is grain
    # the instrument is penalising, taking it away raises our score; if it is
    # detail, taking it away lowers it. Either answer is informative and neither
    # is a proposal: nothing in the engine does this.
    ours_lp = []
    for s_ in (0.3, 0.5, 0.7, 1.0):
        b = [gaussian_blur(a, s_) for a in ours]
        sc = scorer.score(b, prompts)
        d_ = _cmp.discrimination(sc, labels)
        ours_lp.append({"sigma": s_, "sharpness": mean_sharpness(b),
                        "clip_true_mean": float(sc[:, 0].mean()),
                        "delta_vs_ours": float(sc[:, 0].mean() - clip_o.mean()),
                        "frames_clearing_bound": int((sc[:, 0] >= bound).sum()),
                        "argmax_label": d_["argmax_label"],
                        "margin_to_best_decoy": d_["margin"],
                        "per_frame_true_wins": d_["per_frame_true_wins"],
                        "scorer_valid": _arm(d_, f"ours blurred sigma={s_:.2f}")})
        # THE COLUMNS THAT DECIDE WHETHER THE DELTA MEANS ANYTHING, PRINTED.
        # The reference sweep above has always shown argmax, margin and wins.
        # This arm computed and recorded all three and printed none of them, so
        # the console -- and the spec table copied from it -- showed a rising
        # delta on arms where a DECOY outranks the true prompt.
        print(f"[ablate] OUR frames blurred sigma={s_:.2f}: sharpness "
              f"{ours_lp[-1]['sharpness']:.4f}, clip "
              f"{ours_lp[-1]['clip_true_mean']:.4f} "
              f"({ours_lp[-1]['delta_vs_ours']:+.4f}), "
              f"{ours_lp[-1]['frames_clearing_bound']}/25 clear, argmax "
              f"{ours_lp[-1]['argmax_label']}, margin "
              f"{ours_lp[-1]['margin_to_best_decoy']:+.4f}, wins "
              f"{ours_lp[-1]['per_frame_true_wins']}/25, "
              f"{'READABLE' if ours_lp[-1]['scorer_valid']['valid'] else 'INVALID'}")
    out["ours_lowpass_ablation"] = {
        "question": ("does removing OUR excess high-frequency energy raise or "
                     "lower our score -- grain, or detail"),
        "readable_sigmas": [r_["sigma"] for r_ in ours_lp
                            if r_["scorer_valid"]["valid"]],
        "sweep": ours_lp}
    _bad = [r_ for r_ in ours_lp if not r_["scorer_valid"]["valid"]]
    if _bad:
        _sig = ", ".join("%.2f" % r_["sigma"] for r_ in _bad)
        print(f"[ablate] {len(_bad)} of {len(ours_lp)} rows of this sweep are "
              f"INVALID (sigma {_sig}): a decoy outranks the true prompt, so "
              f"their CLIP means are not adherence readings and no conclusion "
              f"may rest on them")

    # --- tone-corrected rescore, closing item 1 ---
    m_tone = scorer.score(ours_tone, prompts)
    d_tone = _cmp.discrimination(m_tone, labels)
    out["tone"]["rescored_after_correction"] = {
        "clip_true_mean": float(m_tone[:, 0].mean()),
        "delta_vs_uncorrected": float(m_tone[:, 0].mean() - clip_o.mean()),
        "frames_clearing_bound": int((m_tone[:, 0] >= bound).sum()),
        "argmax_label": d_tone["argmax_label"],
        "margin_to_best_decoy": d_tone["margin"],
        "per_frame_true_wins": d_tone["per_frame_true_wins"],
        "sharpness": mean_sharpness(ours_tone),
        "scorer_valid": _arm(d_tone, "ours, tone-matched"),
    }
    r = out["tone"]["rescored_after_correction"]
    print(f"[tone] our frames tone-matched to the reference score "
          f"{r['clip_true_mean']:.4f} ({r['delta_vs_uncorrected']:+.4f}), "
          f"{r['frames_clearing_bound']}/25 clear the bound, argmax "
          f"{r['argmax_label']}, margin {r['margin_to_best_decoy']:+.4f}, "
          f"{'READABLE' if r['scorer_valid']['valid'] else 'INVALID'}")

    # --- the symmetric diagnostic arm ---
    sharpened = [unsharp(a, 1.0, 1.0) for a in ours]
    m_sh = scorer.score(sharpened, prompts)
    d_sh = _cmp.discrimination(m_sh, labels)
    out["sharpen_diagnostic"] = {
        "what_it_is": ("a DIAGNOSTIC that asks whether the instrument responds "
                       "to added detail at all. It is not a repair and this row "
                       "proposes nothing that does it"),
        "sigma": 1.0, "amount": 1.0,
        "sharpness": mean_sharpness(sharpened),
        "clip_true_mean": float(m_sh[:, 0].mean()),
        "delta_vs_ours": float(m_sh[:, 0].mean() - clip_o.mean()),
        "frames_clearing_bound": int((m_sh[:, 0] >= bound).sum()),
        "argmax_label": d_sh["argmax_label"],
        "margin_to_best_decoy": d_sh["margin"],
        "per_frame_true_wins": d_sh["per_frame_true_wins"],
        "scorer_valid": _arm(d_sh, "ours, unsharp(1.0, 1.0)"),
    }
    s = out["sharpen_diagnostic"]
    print(f"[diag] unsharp(1.0, 1.0) on OUR frames: sharpness {s['sharpness']:.4f}, "
          f"clip {s['clip_true_mean']:.4f} ({s['delta_vs_ours']:+.4f}), "
          f"{s['frames_clearing_bound']}/25 clear the bound, argmax "
          f"{s['argmax_label']}, margin {s['margin_to_best_decoy']:+.4f}, "
          f"{'READABLE' if s['scorer_valid']['valid'] else 'INVALID'}")

    # THE ARM LEDGER, so a reader of the JSON never has to hunt for validity.
    out["arm_validity"] = {
        "rule": ("an arm's CLIP mean is a reading of prompt adherence only when "
                 "the scorer ranks the TRUE prompt first on that arm's own "
                 "frames, by a strictly positive margin -- the same predicate S0 "
                 "applies to the oracle's render, `adherence_arm_valid` in "
                 "ltx25-render-compare.py"),
        "why": ("this row published its decisive figure, +1.9131 at sigma 1.0, "
                "from an arm on which the decoy 'near:1' outranks the true "
                "prompt by 0.4036. S0 had run once, on the unblurred reference. "
                "A precondition asked of one arm is not a precondition of a run"),
        "arms": {},
    }
    _arms = out["arm_validity"]["arms"]
    _arms["reference_unblurred"] = ref_disc["scorer_valid"]
    _arms["ours_unblurred"] = ours_disc["scorer_valid"]
    _arms["ours_tone_matched"] = r["scorer_valid"]
    _arms["ours_unsharp_1.0"] = s["scorer_valid"]
    for _r in sweep:
        _arms["reference_blur_sigma_%.2f" % _r["sigma"]] = _r["scorer_valid"]
    for _r in ours_lp:
        _arms["ours_blur_sigma_%.2f" % _r["sigma"]] = _r["scorer_valid"]
    _n_bad = sum(1 for _v in _arms.values() if not _v["valid"])
    print(f"[arm] {len(_arms) - _n_bad} of {len(_arms)} scored arms are "
          f"READABLE; {_n_bad} are INVALID and nothing may be concluded from "
          f"them")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except UnreadableInput as exc:
        print(f"UNREADABLE INPUT: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_UNREADABLE)
