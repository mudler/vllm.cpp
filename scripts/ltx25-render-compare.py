#!/usr/bin/env python3
"""Compare two LTX-2.5 renders pixel-for-pixel and sample-for-sample.

A diffusion render has no token gate. There is no discrete output to hold
against a reference, so the correctness net every other model in this tree
leans on does not exist here. This is the substitute: two renders of the same
prompt, seed, geometry, checkpoint and binary, differing only in the knob under
test, compared on the bytes they actually wrote.

It exists because `LTX25-DIT-ATTN-FLASH` (#1549) moved the DiT self-attention
from `vt::Attention` to `vt::AttentionDenseFlash`, the two are NOT bit-identical
on CUDA, and nothing measured what that did to a picture (#1612). The same
question is owed for the FA-2 arm (#1551), whose divergence is larger, so this
tool takes the arm labels as arguments and hard-codes neither.

WHAT IT MEASURES, and why each one is here rather than a fourth statistic:

  identity     Byte equality of the frame files, then array equality. If the
               two arms are bit-identical there is nothing further to argue
               and every threshold below is vacuous. Report it FIRST so a
               reader never mistakes a passing bound for an unread one.

  |delta|      max, mean and the full histogram over 8-bit RGB. The histogram
               is not decoration: "within numerical noise" predicts a mass
               concentrated at 0 and 1, and a bimodal tail is the shape of a
               structural difference wearing a small mean.

  PSNR         The video-coding convention. 40 dB on 8-bit is the usual
               "visually lossless" line and it is a threshold this experiment
               did not choose for itself.

  SSIM         Wang et al. 2004, the ORIGINAL 11x11 Gaussian sigma=1.5 window
               on luma, not scikit-image's 7x7 uniform default. Stated because
               the two disagree in the third decimal and this gate reads that
               far.

  temporal     The self-calibrating one, and the only bound here derived from
               the render rather than from a convention. Mean absolute
               difference between ADJACENT FRAMES of arm A is the video's own
               frame-to-frame step. An arm-to-arm difference far below it is
               smaller than the motion the render is made of. A ratio, unlike
               a constant, does not need to be re-argued at another geometry.

  audio        The DiT drives both streams, so a video-only comparison would
               leave half the change unmeasured.

  correspondence  #1743, section 11.3. Does arm B's frame k still match arm A's
               frame k better than its neighbours do; is the audio lag that
               maximises the cross-correlation still 0; is the spatial offset
               that minimises the frame difference still (0, 0). An arithmetic
               change moves the picture; it does not move it in TIME or in
               SPACE, and each constant here is the exact point at which a
               correspondence is lost rather than a value anyone chose.

  incoherence  #1743, section 11.3, and it is what the exit status now rests
               on together with C0 and correspondence. For a one-sided quality
               statistic -- sharpness, blockiness, motion energy, audio energy
               -- `K = |sum of the per-tile differences| / sum of |them|` is 1
               EXACTLY when every term moves the same way, which is what a blur,
               a block grid or a silenced track does, and concentrates near
               `N^-1/2` when the signs are a fair coin, which is what two
               trajectories that separated do.

EVERY THRESHOLD ABOVE THE STRUCTURAL ONES IS RELOCATED, NOT WIDENED. Section
10.4's V1, V2, V3, V4, A1 and A2 keep their values byte-for-byte, keep their
computation and keep their printed line. What they lose is the exit status,
because they measure IDENTITY and section 11.1 records why identity cannot
separate a change that degraded the render from a pipeline that is sensitive to
any arithmetic at all: on the section 10.7 frames, flash-vs-naive at ONE build
reads 6.414156 and naive-vs-naive across builds reads 9.452407. They are printed
under an IDENTITY verdict of their own, which on those frames reads DIFFERENT.
Widening one of them to admit the swap is the failure #1668 names and it is not
what this file does.

USAGE
    ltx25-render-compare.py --a <dir> --b <dir> [--control <dir>] \
        [--control-of a|b] [--label-a naive] [--label-b flash] [--json out.json]

`--control` is a THIRD render that repeats ONE of the two arms with nothing
changed. It measures the noise floor: run-to-run nondeterminism of the same
binary and the same knob. Without it, an arm-to-arm delta cannot be attributed
to the knob rather than to the machine. With it, the attribution is arithmetic
-- if the control is zero, every bit of the A-vs-B delta is the knob; if the
control is the same size as the delta, the knob changed nothing the box does not
change on its own.

`--control-of` says WHICH ARM the control repeats, and it exists because the
answer used to be a silent convention. The control was always compared against
arm A, stated only here, and the harness called this tool with the control
repeating arm B -- so the "noise floor" was a second copy of the treatment
comparison, guaranteed to read about the same size as the delta it was supposed
to calibrate, and the null verdict below would have been published whatever the
kernel did. The argument is now explicit, it is recorded in the JSON as
`control_of`, and the control block prints the arm it was read against.

`control_ratio` is the number section 10.5 selects on: the control's mean
absolute difference over the treatment's, on luma, in the same units. It is
REPORTED and never checked, because it chooses between two readings of a passing
result rather than between passing and failing. It is `null`, with a reason,
when the treatment is bit-identical and the denominator is therefore zero.

THE CONTROL'S OWN CONTENT IS JUDGED, and that is separate from the ratio being
unchecked. C0 asks of each arm "is there a picture in it", and the same question
has to be asked of the control, for the same reason section 10.4 gives for arms
A and B: a difference cannot tell two good renders from two identically broken
ones. It was asked of A and B only. A control of six one-colour frames therefore
left this tool at exit 0, verdict PASS, with `R = 112.77` -- which section 10.5
reads as `R >= 0.5`, *indistinguishable from run-to-run nondeterminism*, the
STRONGER of its two null readings. A control that rendered nothing at all
upgraded the published conclusion, which is the section 10.6 class one step
removed.

Exit 0 when every threshold passes, 1 when one fails, 2 when the inputs cannot
be read, 3 when the treatment passed and the CONTROL failed its own content
checks. A missing input is never a pass, and it is never an exit 1 either: a
1 says "the two renders differ", which is a reading of an experiment that
happened, and a broken render that reported it would be indistinguishable from
the finding this tool exists to make. A 3 is not an exit 1 for the mirror-image
reason: a degenerate control is a broken EXPERIMENT, not a visible difference
between two renders, and section 10.5 maps exit 1 to the second of those. It is
not an exit 0 either, because the pass it would report is a pass nobody may
read. When a threshold fails AND the control is degenerate, the status is 1: the
renders differ, that is established without the control, and it is a finding
about a change already on `main` that a broken control must not hide.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import wave

import numpy as np

# --- the four exit statuses, which are four different statements --------------
EXIT_PASS = 0        # every threshold passed
EXIT_FAIL = 1        # a threshold failed: the renders differ, and by how much
EXIT_UNREADABLE = 2  # nothing was compared, because an input could not be read
EXIT_CONTROL_DEGENERATE = 3  # the treatment passed and the control has no picture in it


class UnreadableInput(ValueError):
    """An input this tool cannot read, as opposed to one it read and refused.

    It subclasses `ValueError` because `read_ppm` raised that before this class
    existed and callers outside `main()` still catch it. Everything that raises
    it leaves `main()` at `EXIT_UNREADABLE` with no JSON written: a report with
    no comparison in it is worse than no report.
    """

# --- the registered thresholds ------------------------------------------------
# These are DEFAULTS, and they are written here rather than passed at the call
# site so that the criterion is committed to the repository before any number is
# read against it. `.agents/specs/ltx25-dit-attn-flash.md` section 10.4 derives
# each one. Overriding one on the command line is legitimate for a different
# arm pair (#1551) and is recorded in the JSON as an override.
DEFAULT_MAX_MEAN_ABS = 1.0        # 8-bit levels, mean over every pixel/channel
DEFAULT_MIN_PSNR_DB = 40.0        # visually-lossless convention
DEFAULT_MIN_SSIM = 0.99           # per-frame minimum, not the mean
DEFAULT_MAX_TEMPORAL_RATIO = 0.10 # arm delta vs the render's own motion step
DEFAULT_MIN_AUDIO_PSNR_DB = 40.0
DEFAULT_MIN_AUDIO_CORR = 0.999


# --- PPM ----------------------------------------------------------------------
def read_ppm(path: str) -> np.ndarray:
    """Read a binary P6 PPM into an (H, W, 3) uint8 array.

    Written out rather than delegated because the only image library certain to
    be present in a leased worker is the one that ships with numpy, which is
    none. The parser is strict: a maxval other than 255 changes the meaning of
    every threshold below, so it refuses instead of rescaling silently.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(b"P6"):
        raise UnreadableInput(f"{path}: not a binary P6 PPM (starts {data[:2]!r})")
    # Header tokens: P6 width height maxval, with '#' comments allowed anywhere.
    tokens: list[bytes] = []
    i = 2
    while len(tokens) < 3:
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i : i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        j = i
        while j < len(data) and not data[j : j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    i += 1  # exactly one whitespace byte after maxval, per the format
    w, h, maxval = (int(t) for t in tokens)
    if maxval != 255:
        raise UnreadableInput(f"{path}: maxval {maxval}, expected 255")
    need = w * h * 3
    px = data[i : i + need]
    if len(px) != need:
        raise UnreadableInput(f"{path}: truncated, {len(px)} of {need} pixel bytes")
    return np.frombuffer(px, dtype=np.uint8).reshape(h, w, 3)


def frame_paths(d: str) -> list[str]:
    """Every `frame_*.ppm` in one arm, in name order, and NEVER an empty list.

    A directory with no frames is a render that did not happen, so it is refused
    here rather than carried forward as a comparison with zero terms. It used to
    reach a `content.<arm>.frames = False` check that could never fire -- the
    zero-frame arm raised out of `compare_video` first -- and exit 1, which is
    the status that says the two renders DIFFER.
    """
    names = sorted(n for n in os.listdir(d) if n.startswith("frame_") and n.endswith(".ppm"))
    if not names:
        raise UnreadableInput(f"{d}: no frame_*.ppm files, so this arm rendered nothing")
    return [os.path.join(d, n) for n in names]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- SSIM ---------------------------------------------------------------------
def _gauss1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    r = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(r ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _blur(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable convolution with reflect padding, numpy only."""
    pad = len(k) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * xp[i : i + x.shape[0], :]
    xp = np.pad(out, ((0, 0), (pad, pad)), mode="reflect")
    out2 = np.zeros_like(x)
    for i, w in enumerate(k):
        out2 += w * xp[:, i : i + x.shape[1]]
    return out2


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma, the plane SSIM is conventionally computed on."""
    f = rgb.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Wang et al. 2004 mean SSIM on luma, 11x11 Gaussian sigma=1.5, L=255."""
    k = _gauss1d()
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a, mu_b = _blur(a, k), _blur(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a = _blur(a * a, k) - mu_a2
    s_b = _blur(b * b, k) - mu_b2
    s_ab = _blur(a * b, k) - mu_ab
    num = (2 * mu_ab + c1) * (2 * s_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (s_a + s_b + c2)
    return float(np.mean(num / den))


def psnr_from_mse(mse: float, peak: float = 255.0) -> float:
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


# --- what each arm rendered, on its own -------------------------------------
def arm_content(d: str) -> dict:
    """Absolute content of ONE arm, independent of the other.

    THE HOLE THIS CLOSES. Every other measurement in this file is a DIFFERENCE,
    and a difference cannot tell two good renders from two identically broken
    ones. Two all-black renders differ by zero, score infinite PSNR and SSIM
    1.0, and would read as the strongest possible pass. A run that exited 0
    having written frames that were all one colour has happened in this
    repository, so this is a recorded failure mode and not a hypothetical.

    So each arm is also judged on its own: does it contain a picture, are the
    frames distinct, and does anything move. These are the checks
    `ltx25-fullmodel/job/verify_render.py` makes, computed here instead of
    shelled out to, because that file lives on a mutable path on a share and
    this one is committed per revision.
    """
    paths = frame_paths(d)  # refuses an empty arm; see EXIT_UNREADABLE above
    out: dict = {"dir": os.path.abspath(d), "frames": len(paths)}
    means, variances, hashes, adjacent = [], [], set(), []
    prev = None
    shape = None
    for p in paths:
        a = read_ppm(p)
        # ONE geometry per arm. Without this the adjacent-frame difference below
        # raises a numpy broadcast error, which is not an exit status this tool
        # defines and would leave the run with a traceback instead of a verdict.
        if shape is None:
            shape = a.shape
        elif a.shape != shape:
            raise UnreadableInput(
                f"{p}: frame shape {a.shape} differs from {shape} earlier in {d}")
        f = a.astype(np.float64)
        means.append(float(f.mean()))
        variances.append(float(f.var()))
        hashes.add(sha256_file(p))
        lu = luma(a)
        if prev is not None:
            adjacent.append(float(np.abs(lu - prev).mean()))
        prev = lu
    out["pixel_mean"] = float(np.mean(means))
    out["per_frame_var_min"] = float(np.min(variances))
    out["per_frame_mean_min"] = float(np.min(means))
    out["per_frame_mean_max"] = float(np.max(means))
    out["distinct_frame_hashes"] = len(hashes)
    out["adjacent_frame_mad_mean"] = float(np.mean(adjacent)) if adjacent else 0.0
    out["adjacent_frame_mad_min"] = float(np.min(adjacent)) if adjacent else 0.0
    out["zero_motion_pairs"] = int(sum(1 for m in adjacent if m == 0.0))
    # A frame whose variance is under this carries no picture. 1.0 in squared
    # 8-bit levels is a standard deviation of one level: below that, every pixel
    # in the frame is the same colour to within the artefact's own resolution.
    out["near_uniform_frames"] = int(sum(1 for v in variances if v < 1.0))
    return out


# --- video --------------------------------------------------------------------
def compare_video(dir_a: str, dir_b: str, label_a: str, label_b: str) -> dict:
    pa, pb = frame_paths(dir_a), frame_paths(dir_b)
    if len(pa) != len(pb):
        raise UnreadableInput(f"frame count differs ({len(pa)} vs {len(pb)})")
    # PAIRED BY NAME, not merely by position. `sorted()` against `sorted()` puts
    # index i of one arm against index i of the other, and equal counts are not
    # equal frames: an arm that lost `frame_000000.ppm` and gained a later one
    # pairs every frame against its neighbour, and then reports the render's own
    # frame-to-frame motion as the arm-to-arm delta. That is a large, plausible
    # and entirely spurious number, and nothing downstream can tell it from a
    # real divergence.
    na = [os.path.basename(p) for p in pa]
    nb = [os.path.basename(p) for p in pb]
    if na != nb:
        i = next(i for i, (x, y) in enumerate(zip(na, nb)) if x != y)
        only_a = sorted(set(na) - set(nb))[:4]
        only_b = sorted(set(nb) - set(na))[:4]
        raise UnreadableInput(
            f"frame names do not correspond at index {i}: "
            f"{label_a} has {na[i]}, {label_b} has {nb[i]}; "
            f"only in {label_a}: {only_a or 'none'}; only in {label_b}: {only_b or 'none'}"
        )

    res: dict = {"label_a": label_a, "label_b": label_b, "frames": len(pa), "per_frame": []}

    identical_files = 0
    total_sq = 0.0
    total_abs = 0.0
    total_n = 0
    hist = np.zeros(256, dtype=np.int64)
    prev_a: np.ndarray | None = None
    adjacent_mads: list[float] = []
    global_max = 0

    for idx, (fa, fb) in enumerate(zip(pa, pb)):
        ha, hb = sha256_file(fa), sha256_file(fb)
        same_file = ha == hb
        identical_files += int(same_file)
        A, B = read_ppm(fa), read_ppm(fb)
        if A.shape != B.shape:
            raise UnreadableInput(f"frame {idx} shape {A.shape} vs {B.shape}")
        d = np.abs(A.astype(np.int16) - B.astype(np.int16))
        hist += np.bincount(d.reshape(-1), minlength=256).astype(np.int64)
        mx = int(d.max())
        global_max = max(global_max, mx)
        mean_abs = float(d.mean())
        mse = float((d.astype(np.float64) ** 2).mean())
        total_sq += mse * d.size
        total_abs += float(d.sum())
        total_n += d.size
        la, lb = luma(A), luma(B)
        s = ssim(la, lb)
        if prev_a is not None:
            adjacent_mads.append(float(np.abs(la - prev_a).mean()))
        prev_a = la
        res["per_frame"].append(
            {
                "index": idx,
                "file_a": os.path.basename(fa),
                "sha_equal": same_file,
                "max_abs": mx,
                "mean_abs": mean_abs,
                "psnr_db": psnr_from_mse(mse),
                "ssim": s,
                "differing_pixels": int((d.sum(axis=2) > 0).sum()),
                "pixels": int(d.shape[0] * d.shape[1]),
            }
        )

    agg_mse = total_sq / total_n
    res["identical_frame_files"] = identical_files
    res["bit_identical"] = identical_files == len(pa)
    res["max_abs"] = global_max
    res["mean_abs"] = total_abs / total_n
    res["psnr_db"] = psnr_from_mse(agg_mse)
    res["rmse"] = math.sqrt(agg_mse)
    res["ssim_mean"] = float(np.mean([f["ssim"] for f in res["per_frame"]]))
    res["ssim_min"] = float(np.min([f["ssim"] for f in res["per_frame"]]))
    res["psnr_min_db"] = float(np.min([f["psnr_db"] for f in res["per_frame"]]))
    res["delta_histogram"] = {str(v): int(c) for v, c in enumerate(hist) if c}
    res["samples"] = int(total_n)
    # The self-calibrating denominator: arm A's own frame-to-frame step, on luma,
    # in the same 8-bit units as mean_abs above.
    res["adjacent_frame_mad_a"] = float(np.mean(adjacent_mads)) if adjacent_mads else None
    if res["adjacent_frame_mad_a"]:
        # mean_abs is over RGB, adjacent MAD over luma; recompute the numerator on
        # luma so the ratio divides like with like rather than nearly-like.
        res["temporal_ratio"] = None  # filled by the caller, which has the luma delta
    return res


def compare_video_luma_delta(dir_a: str, dir_b: str) -> float:
    """Mean |delta| on LUMA, the numerator of the temporal ratio."""
    pa, pb = frame_paths(dir_a), frame_paths(dir_b)
    tot, n = 0.0, 0
    for fa, fb in zip(pa, pb):
        la, lb = luma(read_ppm(fa)), luma(read_ppm(fb))
        tot += float(np.abs(la - lb).sum())
        n += la.size
    return tot / n


# --- audio --------------------------------------------------------------------
def read_wav(path: str) -> tuple[np.ndarray, int]:
    try:
        with wave.open(path, "rb") as w:
            n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(n)
    except wave.Error as exc:  # a file that is present and is not a wav
        raise UnreadableInput(f"{path}: not a readable wav ({exc})") from exc
    if sw != 2:
        raise UnreadableInput(f"{path}: sample width {sw}, expected 2 (16-bit PCM)")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return a.reshape(-1, ch), sr


def compare_audio(a_path: str, b_path: str) -> dict:
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return {"present": False, "reason": "one or both wav files absent"}
    A, sr_a = read_wav(a_path)
    B, sr_b = read_wav(b_path)
    out: dict = {"present": True, "sample_rate_a": sr_a, "sample_rate_b": sr_b,
                 "frames_a": int(A.shape[0]), "frames_b": int(B.shape[0]),
                 "sha_equal": sha256_file(a_path) == sha256_file(b_path)}
    if A.shape != B.shape or sr_a != sr_b:
        out["comparable"] = False
        return out
    out["comparable"] = True
    d = np.abs(A - B)
    peak = 32768.0
    mse = float((d ** 2).mean())
    out["max_abs_lsb"] = float(d.max())
    out["mean_abs_lsb"] = float(d.mean())
    out["max_abs_fs"] = float(d.max() / peak)
    out["rms_diff_fs"] = float(math.sqrt(mse) / peak)
    out["psnr_db"] = psnr_from_mse(mse, peak=peak)
    out["bit_identical"] = bool(out["sha_equal"] and d.max() == 0)
    fa, fb = A.reshape(-1), B.reshape(-1)
    if fa.std() > 0 and fb.std() > 0:
        out["pearson_r"] = float(np.corrcoef(fa, fb)[0, 1])
    else:
        out["pearson_r"] = None
    return out


# --- the structural criterion: correspondence and incoherence (#1743) ---------
# `.agents/specs/ltx25-dit-attn-flash.md` section 11.
#
# Everything above this line measures IDENTITY: how close two renders are to
# being the same picture. Section 10.7 answered that -- they are not close -- and
# section 11.1 records why that answer cannot decide anything. An identity bound
# reads the same on a change that DEGRADED the render and on a pipeline that is
# sensitive to any arithmetic at all, and the frames on the share show both:
# flash-vs-naive at one build is 6.414156 and naive-vs-naive across builds is
# 9.452407.
#
# So the verdict moves onto two properties that DO separate those populations,
# and neither is a tolerance:
#
#   CORRESPONDENCE  A perturbation of the arithmetic moves the picture. It does
#                   not move the picture in TIME or in SPACE. So arm B's frame k
#                   must still be the nearest of arm B's frames to arm A's frame
#                   k; the audio lag that maximises the cross-correlation must
#                   still be 0; and the spatial offset that minimises the frame
#                   difference must still be (0, 0). Each constant is the exact
#                   point at which the correspondence is lost, not a chosen
#                   value: a dropped frame, a desync and a translation each move
#                   the argmin by a whole index, sample or pixel.
#
#   INCOHERENCE     A reassociated sum makes two renders EXCHANGEABLE; a defect
#                   makes one of them worse. So for a one-sided quality
#                   statistic -- sharpness, blockiness, motion energy, audio
#                   energy -- the coherence ratio
#
#                       K = |SUM_k (s_k^A - s_k^B)| / SUM_k |s_k^A - s_k^B|
#
#                   is 1 EXACTLY when every term moves the same way, which is
#                   what a blur, a block grid or a silenced track does, and
#                   concentrates near N^-1/2 when the signs are a fair coin,
#                   which is what a separated trajectory does. Any constant
#                   ABOVE THE NULL'S OWN SCATTER and below 1 gives the same
#                   verdict on both populations. `N^-1/2` is where the null
#                   CONCENTRATES and not a bound on it: on the 96x64/6f
#                   fixtures the smallest-N statistic realises K = 0.22 against
#                   a floor of 0.09, and a constant at 0.2 flips that fixture's
#                   verdict. K is magnitude-weighted rather than a sign test, so
#                   a bias that is small against the per-tile variation does not
#                   fire it.

TILE = 16                  # pixels; per-tile terms are what make N large
BLOCK_GRID = 8             # the DCT block grid a codec artefact would sit on
ALT_GRID = 32              # LTX-2's own spatial compression factor
BLOCK_BAND = 8             # rows/columns per blockiness term; smaller than
                           # TILE so that the statistic with the FEWEST
                           # terms is not the one that sets the criterion's
                           # usable interval
AUDIO_WINDOW = 256         # samples; 5.3 ms at 48 kHz
ALIGN_FRAME_WINDOW = 2     # neighbouring frame indices searched
ALIGN_SPATIAL_WINDOW = 1   # pixels searched in each spatial direction
AUDIO_MAX_LAG = 2000       # samples; the sweep section 10.7 ran by hand
DEFAULT_MAX_COHERENCE = 0.5  # section 11.3: half the total variation


def _tile_mean(x: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Block means of a per-pixel map, dropping the ragged right/bottom edge."""
    h, w = x.shape
    th, tw = h // tile, w // tile
    if th < 1 or tw < 1:
        return np.array([[float(x.mean())]])
    return x[: th * tile, : tw * tile].reshape(th, tile, tw, tile).mean(axis=(1, 3))


def sharpness_map(l: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude of luma: the plane a blur removes."""
    g = np.zeros_like(l)
    g[:, 1:] += np.abs(np.diff(l, axis=1))
    g[1:, :] += np.abs(np.diff(l, axis=0))
    return g * 0.5


def blockiness_bands(l: np.ndarray, grid: int = BLOCK_GRID,
                     band: int = BLOCK_BAND) -> np.ndarray:
    """Ratio of the mean luma step ON the block grid to the mean step off it.

    One value per horizontal band and one per vertical band, concatenated. A
    render with no block structure sits near 1.0 because the grid has no
    special status in it; block artefacts raise the numerator in every band.
    """
    vals: list[float] = []
    for axis in (1, 0):
        d = np.abs(np.diff(l, axis=axis))
        steps = np.arange(1, l.shape[axis])
        on = (steps % grid) == 0
        if not on.any() or not (~on).any():
            continue
        # Band along the OTHER axis, so each band still sees the whole run of
        # steps and the ratio is computed over many columns rather than a few.
        length = l.shape[1 - axis]
        nb = max(1, length // band)
        for bi in range(nb):
            lo, hi = bi * band, (bi + 1) * band if bi + 1 < nb else length
            sl = d[lo:hi, :] if axis == 1 else d[:, lo:hi]
            num = float(sl[:, on].mean()) if axis == 1 else float(sl[on, :].mean())
            den = float(sl[:, ~on].mean()) if axis == 1 else float(sl[~on, :].mean())
            vals.append(num / den if den > 0 else 0.0)
    return np.asarray(vals, dtype=np.float64)


def arm_quality_terms(d: str) -> dict:
    """The one-sided quality statistics of ONE arm, as matched term vectors.

    Each entry is an array whose k-th element pairs with the k-th element of the
    other arm's array of the same name. Per tile and per frame, so that N is
    large enough that the incoherent null sits far below any constant in (0, 1).
    """
    paths = frame_paths(d)
    sharp, block, motion = [], [], []
    prev = None
    for p in paths:
        l = luma(read_ppm(p)).astype(np.float64)
        sharp.append(_tile_mean(sharpness_map(l)).reshape(-1))
        block.append(blockiness_bands(l))
        if prev is not None:
            motion.append(_tile_mean(np.abs(l - prev)).reshape(-1))
        prev = l
    return {
        "sharpness": np.concatenate(sharp) if sharp else np.zeros(0),
        "blockiness": np.concatenate(block) if block else np.zeros(0),
        "motion": np.concatenate(motion) if motion else np.zeros(0),
    }


def audio_rms_terms(path: str, window: int = AUDIO_WINDOW) -> np.ndarray:
    """Per-window RMS energy of a track. A silenced arm loses it in every one."""
    a, _ = read_wav(path)
    x = a.mean(axis=1)
    nw = len(x) // window
    if nw < 1:
        return np.zeros(0)
    return np.sqrt((x[: nw * window].reshape(nw, window) ** 2).mean(axis=1))


def audio_rms_terms_per_channel(path: str, window: int = AUDIO_WINDOW) -> dict[str, np.ndarray]:
    """The same per-window RMS, per CHANNEL instead of over the mono mean.

    `audio_rms_terms` averages the channels before it windows the track, and a
    mean is a cancellation: two channels that move in OPPOSITE directions leave a
    mono term that does not move at all, and two that move by different amounts
    leave one number that is neither. Section 11.9 measured that on the 1612-r3
    frames -- `flash` against `naive` reads `K = 0.756589` on channel 0 and
    `0.426780` on channel 1, so the mono `0.674002` is one channel's direction
    diluted by the other's, and channel 1 alone does not cross the criterion.

    THIS IS REPORTED AND IT IS NOT CHECKED. Section 11.3's ratified statistic is
    the mono term, and adding a channel to the CHECKED set widens the verdict,
    which is a criterion change and owes its own row, its own red-before evidence
    and its own mutation. Printing it costs nothing and makes the dilution
    visible; gating on it silently would move a bound this row is not entitled to
    move.
    """
    a, _ = read_wav(path)
    out: dict[str, np.ndarray] = {}
    for c in range(a.shape[1]):
        x = a[:, c]
        nw = len(x) // window
        out[f"ch{c}"] = (np.sqrt((x[: nw * window].reshape(nw, window) ** 2).mean(axis=1))
                         if nw >= 1 else np.zeros(0))
    return out


def _top_decile_share(d: np.ndarray) -> float | None:
    """Share of the NET difference carried by the largest tenth of the terms.

    `n` counts terms and not independent observations. A `K` built from one
    event spread over many windows and a `K` built from a shift present in every
    window are the same number and are not the same evidence, and the section
    11.8 audio result is the first kind: 98.9% of its net comes from a tenth of
    the windows, which is the render's single loud passage. Near 0.1 the
    direction is spread over the whole population. Outside [0, 1] the net is
    cancellation rather than a direction, which is what an incoherent K looks
    like from this angle and is not a defect in the statistic.
    """
    n = d.size
    if n == 0:
        return None
    net = float(d.sum())
    if net == 0.0:
        return None
    k = max(1, n // 10)
    order = np.argsort(-np.abs(d))
    return float(d[order[:k]].sum() / net)


def coherence(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """K = |sum of the differences| / sum of |the differences|.

    1 EXACTLY when every term moves the same way. Near N^-1/2 when the signs are
    a fair coin. The `hoeffding_p` beside it is CONTEXT and never the argument:
    it assumes the terms are independent, tiles within a frame are not, so it is
    optimistic. The gate rests on the algebraic 1 and not on a probability.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or a.size != b.size:
        return {"statistic": name, "n": int(min(a.size, b.size)), "k": None,
                "reason": f"no matched terms ({a.size} vs {b.size})"}
    d = a - b
    total = float(np.abs(d).sum())
    net = float(d.sum())
    if total == 0.0:
        return {"statistic": name, "n": int(d.size), "k": 0.0, "net": 0.0,
                "total": 0.0, "direction": "none", "mean_a": float(a.mean()),
                "mean_b": float(b.mean()), "hoeffding_p": None,
                "null_floor": 1.0 / math.sqrt(d.size) if d.size else None,
                "majority_fraction": 0.0,
                "net_share_top_decile": None,
                "reason": "every term is equal, so the difference has neither "
                          "magnitude nor direction"}
    ssq = float((d ** 2).sum())
    n_eff = (total ** 2) / ssq if ssq > 0 else 0.0
    sign = 1.0 if net > 0 else -1.0
    return {
        "statistic": name,
        "n": int(d.size),
        "k": abs(net) / total,
        # WHERE K SITS BETWEEN THE TWO POPULATIONS, so that a reader never has
        # to take the threshold on trust. `null_floor` is N^-1/2, which is where
        # an incoherent difference concentrates; 1.0 is where a one-directional
        # degradation sits by algebra. A K between them is a PARTIAL direction
        # and the constant IS load-bearing there. That state is real, it is
        # reported as what it is, and it is not defined away.
        "null_floor": 1.0 / math.sqrt(d.size) if d.size else None,
        "majority_fraction": float(np.mean(np.sign(d) == sign)),
        # HOW MANY TERMS ACTUALLY CARRY THE NET. See `_top_decile_share`.
        "net_share_top_decile": _top_decile_share(d),
        "net": net,
        "total": total,
        "direction": "a>b" if net > 0 else "b>a",
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "n_effective": n_eff,
        "hoeffding_p": min(1.0, 2.0 * math.exp(-0.25 * n_eff / 2.0)),
        "reason": None,
    }


def frame_correspondence(dir_a: str, dir_b: str,
                         window: int = ALIGN_FRAME_WINDOW) -> dict:
    """Is arm B's frame k still the nearest thing in arm B to arm A's frame k?

    The margin is `min over the neighbours / the diagonal`, and the check is
    `> 1`. The 1 is not chosen: below it the corresponding frame has stopped
    being the corresponding frame. This is what section 10.4's V4 was reaching
    for, with the denominator it derived and without the tenth it did not.

    THIS STATISTIC IS NOT SYMMETRIC IN A AND B, and a fresh review found the
    consequence: the neighbours searched are arm B's, against arm A's frame k,
    so swapping the arms gives a different worst margin. On the section 10.7
    frames it is 1.4230 at frame 25 with `--a flash --b naive` and 1.3796 at
    frame 28 the other way round. Both clear 1 and the verdict does not move,
    but a recorded margin has to name its arm order or it does not reproduce.
    """
    la = [luma(read_ppm(p)).astype(np.float32) for p in frame_paths(dir_a)]
    lb = [luma(read_ppm(p)).astype(np.float32) for p in frame_paths(dir_b)]
    n = len(la)
    if n < 2:
        return {"frames": n, "applicable": False, "worst_margin": None,
                "reason": "fewer than two frames, so there is no neighbour to "
                          "compare the diagonal against"}
    per: list[dict] = []
    for k in range(n):
        diag = float(np.abs(la[k] - lb[k]).mean())
        best_j, best_d = None, None
        for j in range(max(0, k - window), min(n, k + window + 1)):
            if j == k:
                continue
            dj = float(np.abs(la[k] - lb[j]).mean())
            if best_d is None or dj < best_d:
                best_j, best_d = j, dj
        margin = math.inf if diag == 0.0 else best_d / diag
        per.append({"index": k, "diagonal": diag, "nearest_other_index": best_j,
                    "nearest_other": best_d, "margin": margin})
    worst = min(per, key=lambda r: r["margin"])
    return {"frames": n, "applicable": True, "window": window,
            "worst_margin": worst["margin"], "worst_index": worst["index"],
            "worst_nearest_other_index": worst["nearest_other_index"],
            "off_diagonal_frames": int(sum(1 for r in per if r["margin"] <= 1.0)),
            "per_frame": per, "reason": None}


def spatial_correspondence(dir_a: str, dir_b: str,
                           window: int = ALIGN_SPATIAL_WINDOW) -> dict:
    """Does any spatial offset match the frames better than (0, 0) does?

    Section 10.4 calibrates its thresholds against ONE PIXEL of global
    horizontal shift and says a criterion that admitted it would not be a
    criterion. No coherence statistic can refuse it, because a rigid translation
    changes no quality at all. It is refused HERE, at an argmin that is not the
    origin. Every offset is scored on the same interior crop, so the comparison
    is not an artefact of which pixels each one can see.
    """
    pa, pb = frame_paths(dir_a), frame_paths(dir_b)
    w = window
    per: list[dict] = []
    for fa, fb in zip(pa, pb):
        la = luma(read_ppm(fa)).astype(np.float32)
        lb = luma(read_ppm(fb)).astype(np.float32)
        H, W = la.shape
        if H <= 2 * w + 1 or W <= 2 * w + 1:
            return {"applicable": False, "worst_offset": None,
                    "reason": f"frame {H}x{W} is too small for a +/-{w} search"}
        core = la[w:H - w, w:W - w]
        best = None
        at_origin = None
        for dy in range(-w, w + 1):
            for dx in range(-w, w + 1):
                sl = lb[w + dy:H - w + dy, w + dx:W - w + dx]
                d = float(np.abs(core - sl).mean())
                if dy == 0 and dx == 0:
                    at_origin = d
                if best is None or d < best[2]:
                    best = (dy, dx, d)
        per.append({"dy": best[0], "dx": best[1], "best": best[2],
                    "at_origin": at_origin})
    off = [r for r in per if (r["dy"], r["dx"]) != (0, 0)]
    return {"applicable": True, "window": w, "frames": len(per),
            "frames_off_origin": len(off),
            "worst_offset": (off[0]["dy"], off[0]["dx"]) if off else (0, 0),
            "per_frame": per, "reason": None}


def audio_correspondence(a_path: str, b_path: str,
                         max_lag: int = AUDIO_MAX_LAG) -> dict:
    """The lag that maximises the cross-correlation must be exactly 0.

    Section 10.7 ran this sweep by hand and read lag 0. It was never a check,
    and A2's `0.999` cannot express it: a track that drifted in time and a
    track that is different both drag the correlation down at lag 0.
    """
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        return {"applicable": False, "best_lag": None,
                "reason": "one or both wav files absent"}
    A, sra = read_wav(a_path)
    B, srb = read_wav(b_path)
    if sra != srb:
        return {"applicable": False, "best_lag": None,
                "reason": f"sample rates differ ({sra} vs {srb})"}
    x = A.mean(axis=1)
    y = B.mean(axis=1)
    x = x - x.mean()
    y = y - y.mean()
    if x.std() == 0.0 or y.std() == 0.0:
        # A track with no variation has no lag that maximises anything, so the
        # check cannot be evaluated and is reported as a failure rather than a
        # pass. Naming WHICH track is the point: two legitimately silent tracks
        # and one silenced arm reach this same line, and a reader has to be able
        # to tell them apart from the message alone.
        which = ("both tracks are" if x.std() == 0.0 and y.std() == 0.0
                 else ("the A track is" if x.std() == 0.0 else "the B track is"))
        return {"applicable": False, "best_lag": None,
                "a_is_constant": bool(x.std() == 0.0),
                "b_is_constant": bool(y.std() == 0.0),
                "reason": f"{which} constant, so no lag has a correlation and "
                          f"this check cannot be evaluated"}
    n = int(2 ** math.ceil(math.log2(max(len(x), len(y)) + max_lag + 1))) * 2
    cc = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(y, n)), n)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = cc[lags]  # negative indices wrap to the negative lags, as intended
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    best = int(lags[int(np.argmax(vals))])
    return {"applicable": True, "max_lag": max_lag, "best_lag": best,
            "r_at_best": float(vals.max() / norm) if norm > 0 else None,
            "r_at_zero": float(cc[0] / norm) if norm > 0 else None,
            "reason": None}


def absolute_quality(d: str, audio: str | None) -> dict:
    """REPORTED, and NOT CHECKED. Section 11.5 GAP 2, filed as #1854.

    Everything else in this file is a difference between two renders. This is
    the only block that is about ONE render in absolute terms, and no threshold
    over it means anything without either an oracle that renders LTX-2.5 or a
    pinned scoring model, neither of which exists in this tree. A blockiness
    ratio of 1.14 is not good or bad until something says what this VAE produces
    when it is working. So the numbers are printed for the next reader and none
    of them is a check. Inventing one would be a gate that passes a wrong
    artefact.
    """
    paths = frame_paths(d)
    sharp, b8, b32, clipped, total = [], [], [], 0, 0
    for p in paths:
        a = read_ppm(p)
        l = luma(a).astype(np.float64)
        sharp.append(float(sharpness_map(l).mean()))
        r8 = blockiness_bands(l, grid=BLOCK_GRID)
        r32 = blockiness_bands(l, grid=ALT_GRID)
        if r8.size:
            b8.append(float(r8.mean()))
        if r32.size:
            b32.append(float(r32.mean()))
        clipped += int(((a == 0) | (a == 255)).sum())
        total += int(a.size)
    out = {
        "sharpness_mean": float(np.mean(sharp)) if sharp else None,
        "blockiness_grid8": float(np.mean(b8)) if b8 else None,
        "blockiness_grid32": float(np.mean(b32)) if b32 else None,
        "clipped_fraction": clipped / total if total else None,
        "checked": False,
        "why_not_checked": "absolute render quality is not gateable in this tree "
                           "(#1854): prompt adherence needs a model and "
                           "artefact-freedom needs an absolute reference render",
    }
    if audio and os.path.exists(audio):
        t = audio_rms_terms(audio)
        out["audio_rms_mean"] = float(t.mean()) if t.size else None
        out["audio_rms_min"] = float(t.min()) if t.size else None
    return out


def structural_report(dir_a: str, dir_b: str, audio_name: str) -> dict:
    """Every structural measurement for ONE arm pair."""
    qa = arm_quality_terms(dir_a)
    qb = arm_quality_terms(dir_b)
    out: dict = {
        "tile": TILE,
        "coherence": {k: coherence(qa[k], qb[k], k) for k in sorted(qa)},
        "frame_correspondence": frame_correspondence(dir_a, dir_b),
        "spatial_correspondence": spatial_correspondence(dir_a, dir_b),
    }
    wa = os.path.join(dir_a, audio_name)
    wb = os.path.join(dir_b, audio_name)
    out["audio_correspondence"] = audio_correspondence(wa, wb)
    if os.path.exists(wa) and os.path.exists(wb):
        ta, tb = audio_rms_terms(wa), audio_rms_terms(wb)
        out["coherence"]["audio_rms"] = coherence(ta, tb, "audio_rms")
        # REPORTED, NOT CHECKED. It never reaches `checks`; see the docstring of
        # `audio_rms_terms_per_channel` for why widening the verdict is a
        # different row's work.
        ca, cb = audio_rms_terms_per_channel(wa), audio_rms_terms_per_channel(wb)
        out["audio_channel_coherence"] = {
            k: coherence(ca[k], cb[k], f"audio_rms_{k}")
            for k in sorted(ca) if k in cb}
    else:
        out["coherence"]["audio_rms"] = {
            "statistic": "audio_rms", "n": 0, "k": None,
            "reason": "one or both wav files absent"}
        out["audio_channel_coherence"] = {}
    return out


# --- main ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="arm A render directory (the reference)")
    ap.add_argument("--b", required=True, help="arm B render directory (the change under test)")
    ap.add_argument("--control", default=None,
                    help="a repeat of ONE arm, unchanged: the run-to-run noise floor")
    ap.add_argument("--control-of", choices=("a", "b"), default="a",
                    help="which arm --control repeats, and therefore which arm it is "
                         "compared against (default: a)")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--label-control", default="control")
    ap.add_argument("--audio-name", default="audio.wav")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-mean-abs", type=float, default=DEFAULT_MAX_MEAN_ABS)
    ap.add_argument("--min-psnr-db", type=float, default=DEFAULT_MIN_PSNR_DB)
    ap.add_argument("--min-ssim", type=float, default=DEFAULT_MIN_SSIM)
    ap.add_argument("--max-temporal-ratio", type=float, default=DEFAULT_MAX_TEMPORAL_RATIO)
    ap.add_argument("--min-audio-psnr-db", type=float, default=DEFAULT_MIN_AUDIO_PSNR_DB)
    ap.add_argument("--min-audio-corr", type=float, default=DEFAULT_MIN_AUDIO_CORR)
    ap.add_argument("--max-coherence", type=float, default=DEFAULT_MAX_COHERENCE,
                    help="section 11.3: the coherence ratio K above which the "
                         "difference has a DIRECTION rather than only a size")
    args = ap.parse_args()

    # ONE place turns an unreadable input into the status that says so. Every
    # refusal below raises rather than returning a number, so a new one cannot
    # be added that quietly reports `the renders differ` instead.
    try:
        return _compare(args)
    except UnreadableInput as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        print(f"VERDICT UNREADABLE (exit {EXIT_UNREADABLE}): nothing was compared",
              file=sys.stderr)
        return EXIT_UNREADABLE
    except OSError as exc:
        print(f"FATAL: cannot read an input: {exc}", file=sys.stderr)
        print(f"VERDICT UNREADABLE (exit {EXIT_UNREADABLE}): nothing was compared",
              file=sys.stderr)
        return EXIT_UNREADABLE


def _compare(args: argparse.Namespace) -> int:
    for d in (args.a, args.b) + ((args.control,) if args.control else ()):
        if not os.path.isdir(d):
            raise UnreadableInput(f"not a directory: {d}")

    report: dict = {
        "thresholds": {
            "max_mean_abs": args.max_mean_abs,
            "min_psnr_db": args.min_psnr_db,
            "min_ssim": args.min_ssim,
            "max_temporal_ratio": args.max_temporal_ratio,
            "min_audio_psnr_db": args.min_audio_psnr_db,
            "min_audio_corr": args.min_audio_corr,
            "max_coherence": args.max_coherence,
        },
        "inputs": {"a": os.path.abspath(args.a), "b": os.path.abspath(args.b),
                   "control": os.path.abspath(args.control) if args.control else None},
        "control_of": args.control_of if args.control else None,
    }

    # WHAT EACH ARM RENDERED, before anything is subtracted. Reported first
    # because a difference of zero between two broken renders is the strongest
    # possible pass on every other line in this report.
    report["content"] = {
        args.label_a: arm_content(args.a),
        args.label_b: arm_content(args.b),
    }
    if args.control:
        report["content"][args.label_control] = arm_content(args.control)

    v = compare_video(args.a, args.b, args.label_a, args.label_b)
    luma_delta = compare_video_luma_delta(args.a, args.b)
    v["mean_abs_luma"] = luma_delta
    if v["adjacent_frame_mad_a"]:
        v["temporal_ratio"] = luma_delta / v["adjacent_frame_mad_a"]
    report["video"] = v

    report["audio"] = compare_audio(
        os.path.join(args.a, args.audio_name), os.path.join(args.b, args.audio_name)
    )

    if args.control:
        # THE ARM THE CONTROL REPEATS, named by the caller. A control is only a
        # noise floor when it is compared against the arm it is a repeat OF.
        ctl_dir = args.a if args.control_of == "a" else args.b
        ctl_label = args.label_a if args.control_of == "a" else args.label_b
        c = compare_video(ctl_dir, args.control, ctl_label, args.label_control)
        c["mean_abs_luma"] = compare_video_luma_delta(ctl_dir, args.control)
        report["control_video"] = c
        report["control_audio"] = compare_audio(
            os.path.join(ctl_dir, args.audio_name), os.path.join(args.control, args.audio_name)
        )
        # THE NUMBER SECTION 10.5 SELECTS ON, computed rather than eyeballed.
        # It is reported and never checked: it chooses between two readings of a
        # PASSING result -- "the delta is entirely the kernel's" against "the
        # delta is what the box does on its own" -- and a pass/fail cannot
        # express a choice between two answers that are both answers.
        num, den = c["mean_abs_luma"], v["mean_abs_luma"]
        num_rgb, den_rgb = c["mean_abs"], v["mean_abs"]
        undefined = None
        if den <= 0.0:
            # The EXPECTED division by zero, and it has a name: a bit-identical
            # treatment. Section 10.2 predicts this experiment will not see one,
            # so a reader meets this line only when something else went wrong.
            undefined = (f"the {args.label_a} vs {args.label_b} delta is zero, "
                         f"so the ratio has no denominator")
        report["control_ratio"] = {
            "repeats": args.control_of,
            "repeats_label": ctl_label,
            "control_mean_abs_luma": num,
            "treatment_mean_abs_luma": den,
            "ratio_mean_abs_luma": (num / den) if den > 0.0 else None,
            "control_mean_abs_rgb": num_rgb,
            "treatment_mean_abs_rgb": den_rgb,
            "ratio_mean_abs_rgb": (num_rgb / den_rgb) if den_rgb > 0.0 else None,
            "undefined": undefined,
            # `undefined` is arithmetic: there was no denominator. `unusable` is
            # about the control itself: the ratio divides two real numbers and
            # still means nothing, because the numerator came from a render with
            # no picture in it. The verdict block below fills it in.
            "unusable": None,
        }

    # THE STRUCTURAL MEASUREMENTS (#1743, section 11.3), computed before any
    # check is built so that a reader of the JSON has the numbers whether or not
    # the corresponding check fired.
    report["structural"] = structural_report(args.a, args.b, args.audio_name)
    # AND THE ABSOLUTE PANEL, which is instrumentation and not a gate (#1854).
    report["absolute_quality"] = {
        lbl: absolute_quality(d, os.path.join(d, args.audio_name))
        for lbl, d in ((args.label_a, args.a), (args.label_b, args.b))
    }

    # --- verdict --------------------------------------------------------------
    # Every entry carries WHICH outcome it drives: `treatment` entries decide the
    # arm-A-vs-arm-B verdict and the exit status, `control` entries decide only
    # whether the control is a noise floor the ratio may be read against. One
    # list rather than two, so that a reader of `report["checks"]` sees every
    # judged thing in one place and can see which is which; and the field rather
    # than a naming convention, because a convention a caller can misread
    # silently is what section 10.6 is about.
    checks: list[tuple[str, bool, str, str]] = []

    def c0_checks(label: str, judges: str) -> None:
        """C0 for ONE render, judged on its own content before anything is
        subtracted. Three checks, not four: "frames written" used to be a fourth
        and it could never be False, because `frame_paths` refuses an empty
        directory at EXIT_UNREADABLE long before this runs, and a row that
        cannot fail is a decoration in a table whose entire value is that every
        row can."""
        c = report["content"][label]
        checks.append((f"content.{label}.not_uniform",
                       c["near_uniform_frames"] == 0,
                       f"near-uniform frames {c['near_uniform_frames']} == 0 "
                       f"(min per-frame variance {c['per_frame_var_min']:.3f})",
                       judges))
        checks.append((f"content.{label}.distinct_frames",
                       c["distinct_frame_hashes"] == c["frames"],
                       f"{c['distinct_frame_hashes']} distinct of {c['frames']}",
                       judges))
        checks.append((f"content.{label}.motion",
                       c["zero_motion_pairs"] == 0 and c["adjacent_frame_mad_mean"] > 0.0,
                       f"zero-motion pairs {c['zero_motion_pairs']}, "
                       f"mean adjacent MAD {c['adjacent_frame_mad_mean']:.4f}",
                       judges))

    # C0 FIRST, and it is not a formality. Everything after this line is a
    # DIFFERENCE, and every difference check passes vacuously when both arms are
    # equally broken. An arm that rendered nothing, rendered one colour, or
    # rendered the same frame 49 times fails HERE, where the failure is legible,
    # rather than passing silently as a perfect match.
    for label in (args.label_a, args.label_b):
        c0_checks(label, "treatment")
    # AND THE CONTROL, for the same reason and to a different outcome. The
    # control was the one render whose content nothing judged: it was computed,
    # printed, and never registered. A control of one-colour frames then read as
    # a very large noise floor, which is section 10.5's STRONGER null.
    if args.control:
        c0_checks(args.label_control, "control")

    # THE IDENTITY BOUNDS, RELOCATED AND NOT WIDENED (#1743, section 11.4).
    # Every one of them keeps its value, its computation and its printed line.
    # What it loses is the exit status, because it answers "are these two
    # renders the same picture" and section 11.1 records why that answer cannot
    # separate a change that degraded the render from a pipeline that is
    # sensitive to any arithmetic at all. They are judged, printed and recorded
    # under their own verdict, which on the section 10.7 frames reads DIFFERENT
    # and will keep reading DIFFERENT.
    if v["bit_identical"]:
        checks.append(("video.bit_identical", True, "every frame file sha256-equal",
                       "identity"))
    else:
        checks.append(
            ("video.mean_abs", v["mean_abs"] <= args.max_mean_abs,
             f"{v['mean_abs']:.6f} <= {args.max_mean_abs}", "identity")
        )
        checks.append(
            ("video.psnr_min_db", v["psnr_min_db"] >= args.min_psnr_db,
             f"{v['psnr_min_db']:.3f} >= {args.min_psnr_db}", "identity")
        )
        checks.append(
            ("video.ssim_min", v["ssim_min"] >= args.min_ssim,
             f"{v['ssim_min']:.6f} >= {args.min_ssim}", "identity")
        )
        if v.get("temporal_ratio") is not None:
            checks.append(
                ("video.temporal_ratio", v["temporal_ratio"] <= args.max_temporal_ratio,
                 f"{v['temporal_ratio']:.6f} <= {args.max_temporal_ratio}", "identity")
            )
        else:
            checks.append(("video.temporal_ratio", False, "no adjacent-frame denominator",
                           "identity"))

    a = report["audio"]
    if not a.get("present"):
        checks.append(("audio.present", False, a.get("reason", "absent"), "treatment"))
    elif not a.get("comparable"):
        checks.append(("audio.comparable", False, "shape or sample rate differs",
                       "treatment"))
    elif a.get("bit_identical"):
        checks.append(("audio.bit_identical", True, "wav sha256-equal", "identity"))
    else:
        checks.append(("audio.psnr_db", a["psnr_db"] >= args.min_audio_psnr_db,
                       f"{a['psnr_db']:.3f} >= {args.min_audio_psnr_db}", "identity"))
        checks.append(("audio.pearson_r", (a["pearson_r"] or 0.0) >= args.min_audio_corr,
                       f"{a['pearson_r']} >= {args.min_audio_corr}", "identity"))

    # THE STRUCTURAL CRITERION, and it is what decides the verdict now.
    # `align.*` asks whether the two renders still depict the same moments, the
    # same samples and the same places; `coherence.*` asks whether their
    # difference has a DIRECTION. Section 11.3 derives both, and neither carries
    # a borrowed constant: the alignment constants are the exact points at which
    # a correspondence is lost, and K is 1 by algebra under any one-directional
    # degradation.
    st = report["structural"]
    fc = st["frame_correspondence"]
    if fc["applicable"]:
        checks.append(("align.frames", fc["off_diagonal_frames"] == 0,
                       f"{fc['off_diagonal_frames']} frames whose nearest match in "
                       f"{args.label_b} is not the corresponding frame; worst margin "
                       f"{fc['worst_margin']:.4f} > 1 at frame {fc['worst_index']}",
                       "treatment"))
    else:
        checks.append(("align.frames", False, fc["reason"], "treatment"))
    sc = st["spatial_correspondence"]
    if sc["applicable"]:
        checks.append(("align.spatial", sc["frames_off_origin"] == 0,
                       f"{sc['frames_off_origin']} of {sc['frames']} frames match "
                       f"better at an offset other than (0, 0); worst "
                       f"{sc['worst_offset']}", "treatment"))
    else:
        checks.append(("align.spatial", False, sc["reason"], "treatment"))
    ac = st["audio_correspondence"]
    if ac["applicable"]:
        checks.append(("align.audio_lag", ac["best_lag"] == 0,
                       f"best lag {ac['best_lag']} samples == 0 "
                       f"(r {ac['r_at_best']:.6f} there, {ac['r_at_zero']:.6f} at 0)",
                       "treatment"))
    else:
        checks.append(("align.audio_lag", False, ac["reason"], "treatment"))
    for name in ("sharpness", "blockiness", "motion", "audio_rms"):
        co = st["coherence"][name]
        k = co.get("k")
        if k is None:
            checks.append((f"coherence.{name}", False, co.get("reason", "not computed"),
                           "treatment"))
            continue
        detail = (f"K {k:.6f} <= {args.max_coherence} over {co['n']} terms "
                  f"(net {co.get('net', 0.0):.6g} of total {co.get('total', 0.0):.6g}, "
                  f"direction {co.get('direction')}, "
                  f"means {co.get('mean_a'):.6g} / {co.get('mean_b'):.6g})")
        checks.append((f"coherence.{name}", k <= args.max_coherence, detail, "treatment"))

    report["checks"] = [{"name": n, "pass": p, "detail": d, "judges": j}
                        for n, p, d, j in checks]
    treatment = [c for c in checks if c[3] == "treatment"]
    control_c = [c for c in checks if c[3] == "control"]
    identity_c = [c for c in checks if c[3] == "identity"]
    ok = all(c[1] for c in treatment)
    report["treatment_verdict"] = "PASS" if ok else "FAIL"

    # THE IDENTITY VERDICT, which is a SEPARATE STATEMENT and not a gate. It is
    # printed and recorded so that the relocation of section 11.4 is visible in
    # every report: a reader who wants to know whether the two renders are the
    # same picture gets the answer, in the failing numbers, next to a verdict
    # that is about something else.
    report["identity_verdict"] = ("IDENTICAL" if all(c[1] for c in identity_c)
                                  else "DIFFERENT")
    report["identity_failed"] = [c[0] for c in identity_c if not c[1]]

    # THE READING, section 11.6, written before any number existed.
    c0_failed = [c[0] for c in treatment
                 if c[0].startswith("content.") and not c[1]]
    align_failed = [c[0] for c in treatment if c[0].startswith("align.") and not c[1]]
    coh_failed = [c[0] for c in treatment if c[0].startswith("coherence.") and not c[1]]
    other_failed = [c[0] for c in treatment if not c[1]
                    and c[0] not in c0_failed + align_failed + coh_failed]
    if c0_failed:
        reading = "CONTENT_DEGENERATE"
    elif other_failed:
        reading = "ARTEFACT_MISSING"
    elif align_failed:
        reading = "MISALIGNED"
    elif coh_failed:
        reading = "DIRECTIONAL"
    elif v["bit_identical"] and report["audio"].get("bit_identical", True):
        # BIT_IDENTICAL is about the BYTES, not about the identity bounds. A pair
        # that clears every relocated bound with headroom is still two different
        # renders, and calling that bit-identical would be the report telling a
        # reader something the frames do not say.
        reading = "BIT_IDENTICAL"
    else:
        reading = "SEPARATED, NOT DEGRADED"
    report["reading"] = reading

    # THE CONTROL'S OWN VERDICT, which is a separate statement about a separate
    # render. `USABLE` says the control has a picture in it and is therefore a
    # noise floor the ratio may be read against. `DEGENERATE` says it does not,
    # and then no reading of `R` is available at all -- section 10.5's four
    # branches all assume the control is a repeat of a render, and a repeat of
    # nothing is not one of them.
    control_failed = [c[0] for c in control_c if not c[1]]
    if not args.control:
        report["control_verdict"] = None
    elif control_failed:
        report["control_verdict"] = "DEGENERATE"
    else:
        report["control_verdict"] = "USABLE"

    degenerate_reason = None
    if report["control_verdict"] == "DEGENERATE":
        degenerate_reason = (
            f"the control {args.label_control} failed {len(control_failed)} of its own "
            f"{len(control_c)} content checks ({', '.join(control_failed)}), so it is "
            f"a repeat of no picture rather than a repeat of a render: it is NOT a "
            f"noise floor and section 10.5's R is not readable from this run")
        report["control_ratio"]["unusable"] = degenerate_reason

    # THE STATUS, and its precedence is an argument rather than an ordering.
    #   FAIL outranks a degenerate control: "the two renders differ" is
    #   established without the control at all, and section 10.5 calls that a
    #   finding about a change already on `main`. A broken control must not
    #   swallow it.
    #   A degenerate control outranks a PASS: the pass is real and the READING
    #   of it is what the control supplies, so a run that cannot be read must
    #   not exit with the status that says it may be.
    if not ok:
        report["verdict"] = "FAIL"
        status = EXIT_FAIL
    elif report["control_verdict"] == "DEGENERATE":
        report["verdict"] = "CONTROL_DEGENERATE"
        status = EXIT_CONTROL_DEGENERATE
    else:
        report["verdict"] = "PASS"
        status = EXIT_PASS

    # --- print ----------------------------------------------------------------
    print("=== what each arm rendered, before anything is subtracted ===")
    for label, c in report["content"].items():
        print(f"{label:12s} frames={c['frames']} distinct={c['distinct_frame_hashes']} "
              f"mean={c['pixel_mean']:.3f} min_var={c['per_frame_var_min']:.1f} "
              f"near_uniform={c['near_uniform_frames']} "
              f"adj_mad={c['adjacent_frame_mad_mean']:.4f} "
              f"zero_motion_pairs={c['zero_motion_pairs']}")
    print(f"=== {args.label_a} vs {args.label_b} ===")
    print(f"frames                 {v['frames']}")
    print(f"bit-identical frames   {v['identical_frame_files']}/{v['frames']}")
    print(f"max |delta| (8-bit)    {v['max_abs']}")
    print(f"mean |delta| RGB       {v['mean_abs']:.6f}")
    print(f"mean |delta| luma      {v['mean_abs_luma']:.6f}")
    print(f"RMSE                   {v['rmse']:.6f}")
    print(f"PSNR aggregate         {v['psnr_db']:.3f} dB   (worst frame {v['psnr_min_db']:.3f} dB)")
    print(f"SSIM mean              {v['ssim_mean']:.6f}   (worst frame {v['ssim_min']:.6f})")
    print(f"adjacent-frame MAD (A) {v['adjacent_frame_mad_a']}")
    print(f"temporal ratio         {v.get('temporal_ratio')}")
    print(f"|delta| histogram      {dict(list(report['video']['delta_histogram'].items())[:12])}")
    if "control_video" in report:
        c = report["control_video"]
        r = report["control_ratio"]
        arm = "A" if r["repeats"] == "a" else "B"
        # SAY WHICH ARM, in words, in the block itself. The convention used to be
        # silent and the harness inverted it.
        print(f"--- control ({args.label_control}): the noise floor ---")
        print(f"the control {args.label_control} repeats arm {arm} "
              f"({r['repeats_label']}), so it is compared against {r['repeats_label']}")
        print(f"bit-identical frames   {c['identical_frame_files']}/{c['frames']}")
        print(f"max |delta|            {c['max_abs']}   mean {c['mean_abs']:.6f}")
        print(f"PSNR                   {c['psnr_db']:.3f} dB   SSIM min {c['ssim_min']:.6f}")
        if r["ratio_mean_abs_luma"] is None:
            print(f"control/treatment      undefined: {r['undefined']}")
        else:
            print(f"control/treatment      {r['ratio_mean_abs_luma']:.6f} on luma "
                  f"({r['control_mean_abs_luma']:.6f} / "
                  f"{r['treatment_mean_abs_luma']:.6f}), "
                  f"{r['ratio_mean_abs_rgb']:.6f} on RGB")
        if r["unusable"]:
            print("                       REPORTED, and NOT READABLE:")
            print(f"                       {r['unusable']}")
        else:
            print("                       REPORTED, never checked: section 10.5 reads it")
    print("--- audio ---")
    for k in ("present", "comparable", "bit_identical", "max_abs_lsb", "max_abs_fs",
              "rms_diff_fs", "psnr_db", "pearson_r"):
        if k in a:
            print(f"{k:22s} {a[k]}")
    # WHICH LIST DRIVES WHAT, in the report's own words. `.agents/verification.md`
    # asks an instrument to state what it compared against what: a reader who
    # cannot see which checks decide the exit status cannot audit the ones that
    # do not, and the control's three checks sat outside every list for exactly
    # as long as nothing said so.
    # THE STRUCTURAL BLOCK, printed before the checks so that the numbers the
    # verdict rests on are visible whether or not a check fired on them.
    st = report["structural"]
    print("--- correspondence: do the two renders still line up in time and space ---")
    fc, sc2, ac2 = (st["frame_correspondence"], st["spatial_correspondence"],
                    st["audio_correspondence"])
    if fc["applicable"]:
        print(f"frame margin (worst)   {fc['worst_margin']:.4f} at frame "
              f"{fc['worst_index']}, nearest other index "
              f"{fc['worst_nearest_other_index']}; must be > 1")
    else:
        print(f"frame margin           not applicable: {fc['reason']}")
    if sc2["applicable"]:
        print(f"spatial argmin         {sc2['frames_off_origin']} of {sc2['frames']} "
              f"frames off (0, 0); worst {sc2['worst_offset']}")
    else:
        print(f"spatial argmin         not applicable: {sc2['reason']}")
    if ac2["applicable"]:
        print(f"audio best lag         {ac2['best_lag']} samples; must be 0")
    else:
        print(f"audio best lag         not applicable: {ac2['reason']}")
    print("--- incoherence: does the difference have a DIRECTION ---")
    print("K is 1 EXACTLY when every term moves the same way, and near N^-1/2 "
          "when the signs are a coin.")
    print("N COUNTS TERMS, NOT INDEPENDENT OBSERVATIONS. Read `top10%` before "
          "quoting a K: near 0.1 the")
    print("direction is spread over the whole population, near 1.0 it is ONE "
          "event and has to be named as one.")
    print("A top10% OUTSIDE [0, 1] means the net is cancellation rather than a "
          "direction, which is what an")
    print("incoherent K looks like from this angle and is not a defect in the "
          "statistic.")
    for name in ("sharpness", "blockiness", "motion", "audio_rms"):
        co = st["coherence"][name]
        if co.get("k") is None:
            print(f"{name:14s} not computed: {co.get('reason')}")
            continue
        share = co.get("net_share_top_decile")
        share_s = f"{share:+.3f}" if share is not None else "n/a"
        print(f"{name:14s} K={co['k']:.6f}  (incoherent floor "
              f"{co.get('null_floor'):.4f}, a full direction is 1.0)  N={co['n']}  "
              f"means {co.get('mean_a'):.6g} / {co.get('mean_b'):.6g}  "
              f"direction {co.get('direction')} in "
              f"{co.get('majority_fraction'):.3f} of terms  "
              f"top10%={share_s}  "
              f"hoeffding_p {co.get('hoeffding_p')}")
    print("--- audio coherence PER CHANNEL: REPORTED, and NOT CHECKED (section 11.9) ---")
    print("the checked audio term above is the MONO MEAN, and a mean cancels: two "
          "channels that move by")
    print("different amounts leave one number that is neither. Adding a channel to "
          "the CHECKED set widens")
    print("the verdict, which is a criterion change and owes its own row. These "
          "lines make the dilution")
    print("visible without moving a bound.")
    chco = st.get("audio_channel_coherence") or {}
    if not chco:
        print("  not computed: one or both wav files absent")
    for name, co in chco.items():
        if co.get("k") is None:
            print(f"  {name:10s} not computed: {co.get('reason')}")
            continue
        share = co.get("net_share_top_decile")
        share_s = f"{share:+.3f}" if share is not None else "n/a"
        print(f"  {name:10s} K={co['k']:.6f}  (incoherent floor "
              f"{co.get('null_floor'):.4f})  N={co['n']}  "
              f"means {co.get('mean_a'):.6g} / {co.get('mean_b'):.6g}  "
              f"direction {co.get('direction')} in "
              f"{co.get('majority_fraction'):.3f} of terms  top10%={share_s}")
    print("--- absolute quality: REPORTED, and NOT CHECKED (#1854) ---")
    print("no threshold over these means anything without an oracle that renders "
          "LTX-2.5 or a pinned scoring model, and this tree has neither")
    for lbl, q in report["absolute_quality"].items():
        print(f"{lbl:12s} sharpness={q['sharpness_mean']} "
              f"block8={q['blockiness_grid8']} block32={q['blockiness_grid32']} "
              f"clipped={q['clipped_fraction']} "
              f"audio_rms={q.get('audio_rms_mean')}")
    print("--- checks ---")
    print(f"  these decide the verdict: does the {args.label_b} render CORRESPOND "
          f"to the {args.label_a} render, and is their difference DIRECTIONAL")
    for n, p, d, j in treatment:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    if identity_c:
        # SECTION 11.4's RELOCATION, made visible in every report. These bounds
        # are unchanged, they are still computed, and they no longer decide the
        # exit status. On the section 10.7 frames this block reads DIFFERENT.
        print(f"  these are the IDENTITY bounds of section 10.4. Every value is "
              f"unchanged and NONE of them decides the verdict any more; section "
              f"11.4 records why each was relocated")
        for n, p, d, j in identity_c:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    if control_c:
        print(f"  these decide whether the control {args.label_control} is a noise floor "
              f"at all, and they do NOT decide the "
              f"{args.label_a} vs {args.label_b} verdict")
        for n, p, d, j in control_c:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    if degenerate_reason:
        print(f"CONTROL DEGENERATE: {degenerate_reason}")
    print(f"IDENTITY {report['identity_verdict']}: "
          f"{len(report['identity_failed'])} of {len(identity_c)} relocated bounds "
          f"fail ({', '.join(report['identity_failed']) or 'none'}). "
          f"This does NOT set the exit status (section 11.4).")
    print(f"READING {report['reading']} (section 11.6)")
    print(f"VERDICT {report['verdict']} (exit {status})")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
