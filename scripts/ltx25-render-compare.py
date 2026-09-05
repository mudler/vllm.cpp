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
    ltx25-render-compare.py --a <dir> --reference <mp4|dir> [--json out.json]
    ltx25-render-compare.py --a <dir> --reference <mp4> --adherence-model <dir>

`--reference` is the ABSOLUTE question, and #1854 is the issue that refused to
answer it until an oracle existed. It now does: #1864 pinned `ltx-2`, ran it on
the real bf16 checkpoints, and committed the render to
`tests/parity/goldens/ltx2_oracle/`. Passing it turns the 8-grid and 32-grid
blockiness ratios of the panel below from REPORTED into CHECKED, against a band
recomputed from that render's own frames on every run. Nothing about the bound is
written down here, and the two remaining panel statistics stay reported for
reasons `.agents/specs/ltx25-oracle-absolute.md` section 5 measures rather than
asserts.

`--b` is OPTIONAL when `--reference` is given, and the second form above is what
a single render uses. The absolute question is about ONE render; requiring a
second would make this tool demand a comparison it does not use, and both ways of
faking one are worse than the extra entry point. `_absolute_only` records them.

`--adherence-model` is #1854's OTHER half, and it is what makes a passing run say
the render depicts what was asked for. The scorer is CLIP -- upstream's own
choice, registered in vLLM as `CLIPEmbeddingModel` and used by vLLM-Omni's
accuracy suite to score prompt faithfulness for video -- and it is an INSTRUMENT
rather than an oracle: `ltx-2` still supplies the other side of every comparison,
and the checkpoint is pinned by revision AND sha256 in
`tests/parity/goldens/ltx25_adherence/scorer-pin.json`. Three checks, none of
them a threshold:

  S0  the scorer must rank the render's true prompt first ON THE REFERENCE, by a
      margin above zero, or the run exits UNREADABLE and publishes nothing. An
      instrument that has never failed is not known to be able to.
  S1  `ours_mean >= ref_frame_min`, the mirror of the blockiness bound, with every
      digit recomputed from the reference's own frames.
  S2  the argmax over the true prompt plus the committed decoys must be the true
      prompt. Its null is 1/(N+1) and is printed beside the verdict.

S2 COVERS THE BLIND SPOT THE BLOCKINESS BOUND DECLARES: a pure-noise render passes
C0 and passes both blockiness ratios, and it ranks the true prompt LAST.

WHAT THIS STILL DOES NOT ANSWER. **CLIP's text context is 77 positions**, so a
longer prompt is REFUSED and never truncated. The #1864 reference request fits at
17 tokens. #1854's own 70-word golden-retriever example needs at least 83 and
does not, so this gate answers the request the reference render was taken at and
CANNOT answer the one #1854 quotes. The tower is 224x224, so a 320x192 frame
reaches it as a resized centre crop. Frames are scored one at a time, so temporal
adherence -- "walks SLOWLY" -- is invisible to it, exactly as it is to
vLLM-Omni's own middle-frame scorer. Without `--adherence-model` a passing run
says only that the render is no worse than upstream's on two blockiness ratios at
one geometry.

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
import subprocess
import sys
import tempfile
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


def absolute_quality(d: str, audio: str | None, gated: bool = False) -> dict:
    """The panel. REPORTED, and CHECKED only where a reference supplies a bound.

    Everything else in this file is a difference between two renders. This is
    the only block that is about ONE render in absolute terms, and no threshold
    over it means anything without either an oracle that renders LTX-2.5 or a
    pinned scoring model, neither of which exists in this tree. A blockiness
    ratio of 1.14 is not good or bad until something says what this VAE produces
    when it is working. So the numbers are printed for the next reader and none
    of them is a check. Inventing one would be a gate that passes a wrong
    artefact.

    `gated` says a reference was supplied and the two blockiness ratios are now
    checked against it (`reference_checks`). The remaining three statistics stay
    reported EVEN THEN, and `checked_statistics` names which is which, because a
    bare `"checked": true` over a panel where half the entries decide nothing
    would tell a reader something the report does not mean.
    """
    paths = frame_paths(d)
    sharp, b8, b32, clipped, total = [], [], [], 0, 0
    # COLLAPSED BANDS, COUNTED. `blockiness_bands` returns 0.0 for a band whose
    # OFF-grid step is zero, which is what a fully flat block grid produces --
    # the worst artefact this statistic can be shown, reading as the smallest
    # possible value. A ceiling alone would pass it, so the count is carried out
    # of here and checked. It is a count and not a threshold: the ratio of two
    # non-negative means is 0.0 only when the numerator or the denominator has
    # collapsed, and neither happens to a render with a picture in it.
    zero8, zero32, bands8, bands32 = 0, 0, 0, 0
    for p in paths:
        a = read_ppm(p)
        l = luma(a).astype(np.float64)
        sharp.append(float(sharpness_map(l).mean()))
        r8 = blockiness_bands(l, grid=BLOCK_GRID)
        r32 = blockiness_bands(l, grid=ALT_GRID)
        if r8.size:
            b8.append(float(r8.mean()))
            zero8 += int((r8 == 0.0).sum()); bands8 += int(r8.size)
        if r32.size:
            b32.append(float(r32.mean()))
            zero32 += int((r32 == 0.0).sum()); bands32 += int(r32.size)
        clipped += int(((a == 0) | (a == 255)).sum())
        total += int(a.size)
    out = {
        "sharpness_mean": float(np.mean(sharp)) if sharp else None,
        "blockiness_grid8": float(np.mean(b8)) if b8 else None,
        "blockiness_grid32": float(np.mean(b32)) if b32 else None,
        "clipped_fraction": clipped / total if total else None,
        "blockiness_grid8_collapsed_bands": zero8,
        "blockiness_grid32_collapsed_bands": zero32,
        "blockiness_grid8_bands": bands8,
        "blockiness_grid32_bands": bands32,
        "checked": bool(gated),
        "checked_statistics": list(REFERENCE_GATED) if gated else [],
        "reported_statistics": (list(REFERENCE_REPORTED) if gated
                                else list(REFERENCE_GATED) + list(REFERENCE_REPORTED)),
        "why_not_checked": (
            "prompt adherence still needs a pinned scoring model, and sharpness, "
            "the clipped fraction and audio RMS have no bound the committed "
            "reference can supply (spec ltx25-oracle-absolute.md section 5)"
            if gated else
            "absolute render quality is not gateable in this tree "
            "(#1854): prompt adherence needs a model and "
            "artefact-freedom needs an absolute reference render"),
    }
    if audio and os.path.exists(audio):
        t = audio_rms_terms(audio)
        out["audio_rms_mean"] = float(t.mean()) if t.size else None
        out["audio_rms_min"] = float(t.min()) if t.size else None
    return out



# --- the absolute reference (#1854) -------------------------------------------
# EVERYTHING ABOVE THIS LINE IS A DIFFERENCE BETWEEN TWO RENDERS. This block is
# the one place a render is judged on its own, and #1854 is precise about the
# only shape that is admissible for it: "worse than the oracle on this
# statistic", **because that is a comparison and not a convention**. It filed
# itself rather than shipping a proxy, on the grounds that "a proxy for
# perceptual quality that measures nothing is worse than a declared gap".
#
# So no number below is written down. Every bound is RECOMPUTED from the
# reference render's own frames on each run. A transcribed bound could not
# survive a change to `blockiness_bands`, and comparing a new definition of a
# statistic against an old definition's recorded value is the failure
# `a-transcription-cannot-gate-the-function-it-transcribes` names.
#
# WHICH STATISTIC GATES, AND WHY IT IS THE ONLY ONE. Of the four panel
# statistics, blockiness is the one whose value is anchored by construction
# rather than by content: it is the ratio of the mean luma step ON the block
# grid to the mean step off it, so a render with no block structure sits near
# 1.0 whatever it depicts, because the grid has no special status in it. The
# reference confirms that empirically rather than by assertion -- its 25 frames
# read 1.042812 on grid 8 with per-frame values from 0.947454 to 1.143393, so
# its healthy excess over the null is SMALLER than its own scatter and its
# frames straddle 1.0.
#
# Sharpness, the clipped fraction and audio RMS stay REPORTED, and
# `.agents/specs/ltx25-oracle-absolute.md` section 5 gives the derivation that
# failed for each: sharpness has no structural null and our render is not the
# same picture as the reference; the clipped fraction is content-driven AND does
# not survive the committed mp4's yuv420p round trip (0.001650 to 0.001391, 16%
# relative); and the reference's `audio.wav` is not committed at all, so there
# is no bytes-exact audio reference in this tree to derive a bound from.
GOLDEN_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "tests", "parity", "goldens", "ltx2_oracle"))
DEFAULT_REFERENCE_SUMS = os.path.join(GOLDEN_DIR, "SHA256SUMS")

# The gated statistics, and the direction "worse" runs in. `higher_is_worse` is
# recorded rather than assumed because the LOWER edge of each band is not a
# quality claim and must not be read as one; see `reference_checks`.
REFERENCE_GATED = ("blockiness_grid8", "blockiness_grid32")
REFERENCE_REPORTED = ("sharpness_mean", "clipped_fraction",
                      "audio_rms_mean", "audio_rms_min")


def parse_sha256sums(path: str) -> dict[str, str]:
    """`name -> digest` out of a `sha256sum` file, comments and blanks dropped."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise UnreadableInput(f"{path}: cannot read the reference digest list ({exc})")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        digest, name = digest.strip(), name.strip()
        if len(digest) == 64 and name:
            out[name] = digest
    if not out:
        raise UnreadableInput(f"{path}: no sha256 lines, so nothing anchors the reference")
    return out


def load_reference(path: str, sums_path: str) -> tuple[dict, list[np.ndarray]]:
    """The oracle render, IDENTITY-ASSERTED BEFORE A PIXEL IS READ.

    A reference a caller can point anywhere is not a reference. Pointed at the
    render under test it would pass by construction, which is exactly the
    `oracle-identity-must-be-asserted` failure, and this gate's entire claim is
    that the bound came from upstream rather than from us. So the digests decide
    admission, and they are the ones committed in
    `tests/parity/goldens/ltx2_oracle/SHA256SUMS` by #1864 -- a file
    `tests/scripts/test_ltx2_oracle_goldens.py` already recomputes for the two
    artefacts that are in the tree.

    Two forms, and both are anchored by the same list:

      an .mp4    the committed `upstream-render.mp4`, decoded with ffmpeg. This
                 needs nothing outside the tree. That an H.264 file can carry the
                 very block artefact this gate looks for is a real objection, and
                 it is answered by MEASUREMENT rather than by argument: on this
                 render the decoded frames give `blockiness_grid8` bounds that
                 differ from the true PPM frames' by 2.66e-04 relative and
                 `blockiness_grid32` by 7.56e-04. Section 2 of the spec carries
                 the table. `clipped_fraction` does NOT survive the round trip,
                 which is one of the reasons it is not gated.

      a directory of `frame_*.ppm`, the exact form. Every frame's digest must
                 appear in the list. These are the frames #1864's job wrote to
                 the NAS at `/workspace/ltx2-oracle/out/upstream_frames`, and
                 SHA256SUMS' own preamble says their digests are recorded "so a
                 later copy of them is checkable against this run rather than
                 trusted". This is that check.

    A digest that is absent from the list is refused at EXIT_UNREADABLE and never
    at EXIT_FAIL: an unverifiable reference means NOTHING was compared, and a 1
    would say the render is worse than a reference that was never established.
    """
    sums = parse_sha256sums(sums_path)
    known = set(sums.values())
    if os.path.isdir(path):
        paths = frame_paths(path)
        checked = []
        for p in paths:
            digest = sha256_file(p)
            name = os.path.basename(p)
            if sums.get(name) != digest:
                raise UnreadableInput(
                    f"{p}: sha256 {digest} is not what {os.path.basename(sums_path)} "
                    f"records for {name} ({sums.get(name, 'no entry at all')}). This is "
                    f"not the #1864 reference render, and a bound taken from it would be "
                    f"a bound taken from an unknown file")
            checked.append(name)
        frames = [read_ppm(p) for p in paths]
        form, digest_count, source_digest = "frames", len(checked), None
    elif os.path.isfile(path):
        source_digest = sha256_file(path)
        if source_digest not in known:
            raise UnreadableInput(
                f"{path}: sha256 {source_digest} appears nowhere in "
                f"{os.path.basename(sums_path)}, so it is not the #1864 reference render")
        frames = decode_reference_video(path)
        form, digest_count = "mp4", 1
    else:
        raise UnreadableInput(f"{path}: not a directory of frames and not a file")
    if not frames:
        raise UnreadableInput(f"{path}: the reference decoded to zero frames")
    return {"source": os.path.abspath(path), "form": form,
            "sums": os.path.abspath(sums_path), "digests_verified": digest_count,
            "source_sha256": source_digest, "frames": len(frames)}, frames


def decode_reference_video(path: str) -> list[np.ndarray]:
    """ffmpeg to `rgb24` PPM, in a temporary directory that is always removed.

    Written out rather than piped because the PPM reader above is the one this
    file already trusts, and a second frame decoder inside the same tool would be
    a second definition of what a pixel is.
    """
    with tempfile.TemporaryDirectory(prefix="ltx25-ref-") as tmp:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", path, "-pix_fmt", "rgb24",
               os.path.join(tmp, "frame_%06d.ppm")]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as exc:
            raise UnreadableInput(
                f"{path}: cannot run ffmpeg to decode the reference ({exc}). Pass a "
                f"directory of frame_*.ppm instead, or install ffmpeg")
        if proc.returncode != 0:
            raise UnreadableInput(
                f"{path}: ffmpeg exited {proc.returncode} decoding the reference: "
                f"{proc.stderr.strip()[:400]}")
        return [read_ppm(p) for p in frame_paths(tmp)]


def reference_bounds(frames: list[np.ndarray]) -> dict:
    """The band each gated statistic must lie in, COMPUTED from the reference.

    Per frame, then reduced. `frame_min` and `frame_max` are the reference's own
    observed range; `mean` and `sd` are printed beside them so a reader can see
    how far our value sits from the reference in the reference's own units,
    rather than only whether it cleared a line.
    """
    per: dict[str, list[float]] = {name: [] for name in
                                   ("sharpness_mean", "blockiness_grid8",
                                    "blockiness_grid32", "clipped_fraction")}
    # THE INSTRUMENT'S OWN PRECONDITION, checked before its reading is used. A
    # reference whose bands collapsed has a ceiling of 0.0, which every render
    # would then fail; a reference that is itself degenerate is a broken
    # instrument and not a strict oracle. It has never happened to the #1864
    # render and it is checked anyway, because the cost of finding out inside a
    # GPU lease is a lease.
    collapsed = 0
    for a in frames:
        l = luma(a).astype(np.float64)
        per["sharpness_mean"].append(float(sharpness_map(l).mean()))
        r8 = blockiness_bands(l, grid=BLOCK_GRID)
        r32 = blockiness_bands(l, grid=ALT_GRID)
        collapsed += int((r8 == 0.0).sum()) + int((r32 == 0.0).sum())
        if r8.size:
            per["blockiness_grid8"].append(float(r8.mean()))
        if r32.size:
            per["blockiness_grid32"].append(float(r32.mean()))
        per["clipped_fraction"].append(float(((a == 0) | (a == 255)).sum()) / a.size)
    if collapsed:
        raise UnreadableInput(
            f"the reference has {collapsed} blockiness bands reading 0.0, so its own "
            f"off-grid denominator collapsed. A degenerate reference supplies a "
            f"degenerate bound, and nothing may be measured against it")
    out: dict[str, dict] = {}
    for name, vals in per.items():
        v = np.asarray(vals, dtype=np.float64)
        if v.size == 0:
            out[name] = {"n": 0, "mean": None, "sd": None,
                         "frame_min": None, "frame_max": None}
            continue
        out[name] = {
            "n": int(v.size),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "frame_min": float(v.min()),
            "frame_max": float(v.max()),
        }
    return out


def reference_checks(label: str, panel: dict, bounds: dict) -> list[tuple]:
    """One check per gated statistic, as `(name, pass, detail, judges)` tuples.

    THE BOUND IS THE REFERENCE'S OWN PER-FRAME RANGE, and the asymmetry in it is
    deliberate. Our MEAN is held against the reference's per-frame MAX rather
    than against its mean, because mean-against-mean has no margin at all and
    would fire on the difference in CONTENT between two renders of the same
    prompt by two different engines. It is not held against the reference's per
    frame max by our own per-frame max either: with 25 frames on each side and no
    real difference, the probability that our maximum exceeds theirs is about one
    half, and a gate that fires on a coin toss is not a gate. Our mean against
    their max fires when our render exceeds the reference by roughly two of the
    reference's own per-frame standard deviations, and that number is a
    consequence of the construction rather than a constant anyone picked.

    THE CEILING ALONE WOULD BE A MUTE SWITCH, and the guard beside it is a COUNT
    rather than a second edge. `blockiness_bands` divides the on-grid step by the
    off-grid step and returns 0.0 for a band whose denominator collapsed, which
    is what a fully flat block grid produces: the worst artefact this statistic
    can be shown, reading as the smallest possible value and clearing any
    ceiling. Measured on the reference's own frames, flattening them completely
    onto the 8x8 grid takes `blockiness_grid8` to exactly 0.0000.

    A TWO-SIDED BAND WAS THE FIRST DESIGN AND IT WAS WRONG. Holding the value
    inside the reference's per-frame range makes "much LESS blocky than the
    reference" a failure, and less blocky is not worse. A test caught it rather
    than a reading of the code: one render at 1.185808 against a deliberately
    blocky reference whose band was [1.892608, 2.161415] FAILED, on the side
    where it was better. So the quality claim is one-sided -- `v <= frame_max` --
    and the degeneracy it needed a floor for is checked directly, by requiring
    that NO band collapsed. That count is not a threshold: the ratio of two
    non-negative means is 0.0 only when one of them has collapsed, and neither
    collapses in a render with a picture in it.
    """
    out: list[tuple] = []
    for name in REFERENCE_GATED:
        b = bounds.get(name) or {}
        v = panel.get(name)
        hi = b.get("frame_max")
        if v is None or hi is None:
            out.append((f"absolute.{label}.{name}", False,
                        f"not computed (arm {v}, reference ceiling {hi})", "treatment"))
            continue
        out.append((
            f"absolute.{label}.{name}", v <= hi,
            f"{v:.6f} <= {hi:.6f}, the reference's per-frame maximum "
            f"(reference mean {b['mean']:.6f}, per-frame sd {b['sd']:.6f}, "
            f"n={b['n']}); margin {hi - v:+.6f}; "
            + ("worse than the oracle on this statistic" if v > hi
               else "no worse than the oracle on this statistic"),
            "treatment"))
        collapsed = panel.get(f"{name}_collapsed_bands")
        total = panel.get(f"{name}_bands")
        out.append((
            f"absolute.{label}.{name}_defined", collapsed == 0,
            f"{collapsed} of {total} bands read 0.0, which is the off-grid "
            f"denominator collapsing; a flat block grid reads as the SMALLEST "
            f"possible value and would clear the ceiling above",
            "treatment"))
    return out


# --- prompt adherence (#1854 sub-question 1, owned by #2295) ------------------
# THE HALF OF #1854 THAT WAS STILL OPEN. The block above asks whether this render
# has artefacts in it. It cannot ask whether the render depicts WHAT THE PROMPT
# ASKED FOR, and until this section landed the tool said so in its own output.
#
# WHAT SCORES IT, AND WHY THAT IS NOT A NEW ORACLE. The scorer is CLIP, and CLIP
# is an INSTRUMENT here rather than an oracle. That is the developer's answer to
# section 9 of `.agents/specs/ltx25-prompt-adherence.md`, on that spec's own
# recommendation, and the argument is that the scorer never answers a question on
# its own: every number it produces is consumed as a comparison against the #1864
# render, which is the pinned `ltx-2` oracle's output. Delete the reference and
# neither check below has a bound. So AGENTS.md's oracle table is unchanged, there
# is no `.agents/oracles/` file for the checkpoint, and the checkpoint is pinned by
# revision AND sha256 like any other artefact this project loads --
# `tests/parity/goldens/ltx25_adherence/scorer-pin.json`.
#
# IT IS UPSTREAM'S OWN CHOICE OF SCORER, NOT OURS. vLLM registers the family as a
# first-class runner (`vllm/model_executor/models/registry.py:251` at
# `5559679229`, `"CLIPModel": ("clip", "CLIPEmbeddingModel")`, the only occurrence
# of that literal in the file), and vLLM-Omni's own accuracy suite scores
# prompt-faithfulness for VIDEO with it (`tests/e2e/accuracy/helpers.py:496`,
# `class CLIPScorer`, default `openai/clip-vit-base-patch16`, at
# `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7`; that repository is UNPINNED, #633,
# so this is a source reading at a stated revision and not a pinned citation).
# `score()` at `:504` normalises both projected embeddings and returns their
# cosine times 100, and `_clip_matrix` below is that same arithmetic in a batch.
#
# WHAT WAS NOT INHERITED FROM UPSTREAM, AND WHY. vLLM-Omni asserts
# `clip >= CLIP_ABSOLUTE_FLOOR` with the floor at `20.0` (`:54`), env-overridable
# and derived nowhere. That is INADMISSIBLE here. #1854 was filed rather than
# closed with a proxy precisely to keep a chosen constant out of the centre of the
# verdict, and it names the one admissible shape: "worse than the oracle on this
# statistic", **because that is a comparison and not a convention**. So neither
# check below contains a threshold. S1's bound is recomputed from the reference's
# own frames on every run; S2's null is 1/(N+1) and is arithmetic.
#
# THE THREE CHECKS.
#
#   S0, the precondition. Before either number is published the scorer must rank
#       the TRUE prompt first on the REFERENCE's frames, by a margin strictly
#       greater than zero. If it cannot do that on upstream's own good render of
#       this prompt, the instrument is broken, and a broken instrument that keeps
#       running reports a CODE verdict. It exits EXIT_UNREADABLE and publishes
#       nothing, exactly as `reference_bounds` refuses a degenerate reference. An
#       instrument that has never failed is not known to be able to.
#
#   S1, a comparison. `ours_mean >= ref_frame_min`, both against the true prompt.
#       This is the exact mirror of the blockiness bound: there, higher is worse
#       and the form is `ours_mean <= ref_frame_max`; here higher is better, so
#       the inequality and the order statistic both flip. Every digit is measured
#       off the reference. Mean-against-mean was rejected because it has no margin
#       at all and fires on the difference in CONTENT between two renders of one
#       prompt by two engines; min-against-min was rejected because two single
#       order statistics over 25 frames a side make the verdict a coin toss.
#
#   S2, a SET assertion. The argmax over {true prompt, committed decoys} must be
#       the true prompt. This is the question #1854 actually poses, and it is
#       where CLIP is used for what it was trained to do. Its absolute cosine is
#       uncalibrated -- that is why `20.0` means nothing -- but contrastive
#       RANKING is the training objective. A discrete selection has bimodal error,
#       so a tolerance would bound nothing; the assertion is set equality and the
#       MARGIN to the best decoy is printed on every run, passing or failing,
#       because a gate that prints only its verdict cannot be seen degrading.
#
# S2 ALSO COVERS A BLIND SPOT THE LANDED GATE DECLARES. `ltx25-oracle-absolute.md`
# records that a pure-noise render PASSES blockiness and passes C0, with a test
# that says so. Noise ranks no prompt first, and S2 fails it.
#
# THE 77-POSITION BOUND, STATED WHEREVER THE GATE IS STATED. CLIP's text context
# is 77 positions (`config.json` at the pinned revision,
# `text_config.max_position_embeddings = 77`, measured rather than transcribed --
# its sha256 is in the pin). A prompt that does not fit is REFUSED and never
# truncated. The #1864 reference request fits at 13 words. #1854's own motivating
# prompt, the 70-word golden retriever at
# `scripts/ltx25-dit-attn-flash-pixel-ab.sh:697`, does NOT: CLIP's pre-tokenizer
# splits it into 81 chunks and each chunk is at least one BPE token, so it needs
# at least 83 positions. **This instrument answers the request the reference
# render was taken at and cannot answer the one #1854 quotes.** Its tower is also
# 224x224, so a 320x192 frame reaches it as a resized centre crop.
ADHERENCE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 os.pardir, "tests", "parity", "goldens", "ltx25_adherence"))
DEFAULT_ADHERENCE_PIN = os.path.join(ADHERENCE_DIR, "scorer-pin.json")
DEFAULT_ADHERENCE_DECOYS = os.path.join(ADHERENCE_DIR, "decoys.json")
DEFAULT_ORACLE_MANIFEST = os.path.join(GOLDEN_DIR, "ltx2_oracle_manifest.json")


def load_json_record(path: str, what: str) -> dict:
    """A committed record, or `UnreadableInput`. Never a default."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        raise UnreadableInput(f"{path}: cannot read {what} ({exc})")
    except ValueError as exc:
        raise UnreadableInput(f"{path}: {what} is not valid JSON ({exc})")


def true_prompt(manifest_path: str = DEFAULT_ORACLE_MANIFEST) -> str:
    """The prompt the #1864 reference render was ACTUALLY taken at.

    Read from the committed manifest rather than accepted from a flag, and that is
    the whole point: a `--prompt` a caller could set would let the prompt and the
    reference disagree, and the resulting number would score our render against a
    request nobody rendered. There is exactly one true prompt in this tree and it
    is `request.prompt` in the manifest #1864 committed.
    """
    rec = load_json_record(manifest_path, "the #1864 oracle manifest")
    p = (rec.get("request") or {}).get("prompt")
    if not isinstance(p, str) or not p.strip():
        raise UnreadableInput(
            f"{manifest_path}: no request.prompt, so the prompt the reference was "
            f"rendered at is unknown and nothing may be scored against it")
    return p


def load_decoys(path: str = DEFAULT_ADHERENCE_DECOYS) -> list[dict]:
    """The committed decoy set, with both required kinds present.

    NOT overridable from the command line, deliberately. A caller who could supply
    their own decoys could supply trivially far ones and clear S2 by construction,
    which is `gate-comparing-shared-helper-proves-consistency-not-correctness` with
    the caller holding both sides. The set is committed so it cannot be chosen
    after the scores are seen.
    """
    rec = load_json_record(path, "the committed decoy prompts")
    items = rec.get("decoys")
    if not isinstance(items, list) or len(items) < 2:
        raise UnreadableInput(
            f"{path}: fewer than two decoys, so S2's null 1/(N+1) would be at "
            f"least one half and the argmax would be close to a coin toss")
    out = []
    for i, it in enumerate(items):
        text, kind = (it or {}).get("text"), (it or {}).get("kind")
        if not isinstance(text, str) or not text.strip():
            raise UnreadableInput(f"{path}: decoy {i} has no text")
        if kind not in ("near", "far"):
            raise UnreadableInput(
                f"{path}: decoy {i} has kind {kind!r}, and the set must declare "
                f"each decoy near or far: a set of only far decoys measures "
                f"almost nothing")
        out.append({"text": text, "kind": kind})
    kinds = {d["kind"] for d in out}
    if kinds != {"near", "far"}:
        raise UnreadableInput(
            f"{path}: the decoy set has kinds {sorted(kinds)} and needs both. A "
            f"scorer that can only tell a snowy forest from a bowl of soup has "
            f"been shown very little")
    # DUPLICATES ARE REFUSED BECAUSE THEY CORRUPT THE NULL, not because they are
    # untidy. `adherence_null` divides by the COUNT, so a set that lists one
    # prompt twice reports 1/(N+1) while the argmax really ranges over fewer
    # distinct alternatives than that. The printed null would then understate
    # chance, which is the one number S2 has to be read against.
    texts = [d["text"] for d in out]
    if len(set(texts)) != len(texts):
        dupes = sorted({t for t in texts if texts.count(t) > 1})
        raise UnreadableInput(
            f"{path}: the decoy set repeats {len(dupes)} prompt(s), so its null "
            f"1/(N+1) counts an alternative the argmax does not actually have: "
            f"{dupes[0][:60]!r}")
    return out


def adherence_null(n_decoys: int) -> float:
    """S2's null: an uninformative scorer's chance of picking the true prompt.

    Arithmetic, not a convention. With `n` decoys the argmax ranges over `n + 1`
    prompts, so chance is `1 / (n + 1)`. It is printed beside every S2 verdict
    because a set assertion whose null nobody states is a verdict nobody can size.
    """
    if n_decoys < 1:
        raise UnreadableInput("S2 needs at least one decoy: with none, the argmax "
                              "over a single prompt is the true prompt by "
                              "construction and the check cannot fail")
    return 1.0 / float(n_decoys + 1)


def assert_scorer_identity(model_dir: str, pin: dict) -> dict:
    """The scorer's WEIGHTS are its identity, asserted before it reads a pixel.

    SSIM is a closed form; a neural scorer is a file. A swapped checkpoint moves
    every reading silently and nothing in the report would change, which is #1723
    exactly. So each file the scorer loads is hashed and compared against the
    committed pin, the way `load_reference` hashes the reference render.

    A `null` digest in the pin is UNMEASURED, and unmeasured REFUSES. It does not
    mean "skip this file": it means nobody has ever downloaded and hashed that
    file, so admitting it would be scoring with an unidentified model and quoting
    the number. AGENTS.md's gateability rule -- demonstrably builds and runs the
    model, and constructing a config proves nothing -- is written for an oracle
    and applies here for the same reason.
    """
    files = pin.get("files")
    if not isinstance(files, dict) or not files:
        raise UnreadableInput(
            f"{DEFAULT_ADHERENCE_PIN}: the scorer pin lists no files, so nothing "
            f"anchors the scorer's identity")
    if not os.path.isdir(model_dir):
        raise UnreadableInput(f"{model_dir}: not a directory of scorer weights")
    unmeasured = sorted(n for n, f in files.items() if (f or {}).get("sha256") is None)
    if unmeasured:
        raise UnreadableInput(
            f"the scorer checkpoint is UNMEASURED: {len(unmeasured)} of "
            f"{len(files)} pinned files carry sha256 null "
            f"({', '.join(unmeasured)}). Nothing in this tree has downloaded "
            f"{pin.get('repo')} at {pin.get('revision')}, hashed it, or produced "
            f"one embedding from it, so `gateable` is false in "
            f"{os.path.basename(DEFAULT_ADHERENCE_PIN)} and this run publishes no "
            f"adherence number. The download is "
            f"{pin.get('required_bytes_total')} bytes, the repository declares "
            f"NO licence, and the weights ship as a pickle rather than "
            f"safetensors; `.agents/developer-preferences.md` records no "
            f"authority for it. Owner: row LTX25-PROMPT-ADHERENCE, issue #2295")
    checked = {}
    for name in sorted(files):
        want = files[name]["sha256"]
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            raise UnreadableInput(
                f"{p}: the pin names this file and it is not there, so the "
                f"checkpoint in {model_dir} is not {pin.get('repo')} at "
                f"{pin.get('revision')}")
        got = sha256_file(p)
        if got != want:
            raise UnreadableInput(
                f"{p}: sha256 {got} is not the pinned {want}. This is not "
                f"{pin.get('repo')} at {pin.get('revision')}, and a score from an "
                f"unknown scorer is a number with no instrument behind it")
        checked[name] = got
    return {"repo": pin.get("repo"), "revision": pin.get("revision"),
            "model_dir": os.path.abspath(model_dir), "files_verified": len(checked),
            "digests": checked}


def refuse_overlong_prompts(prompts: list[str], count_tokens, limit: int) -> dict:
    """REFUSE a prompt that does not fit the scorer's context. Never truncate.

    Truncating is the failure mode that produces a plausible number for a question
    nobody asked: the tokens fit, the cosine comes back, and the report says the
    render depicts a prompt whose second half the scorer never saw. So the run
    stops at EXIT_UNREADABLE and names the prompt and its length.

    `count_tokens` is the SCORER'S OWN tokenizer, passed in rather than
    reimplemented here, because a second definition of "how long is this prompt"
    could drift from the one the model actually uses.
    """
    if not isinstance(limit, int) or limit < 1:
        raise UnreadableInput(
            f"the scorer pin records text_context_positions {limit!r}, which is "
            f"not a length, so no prompt can be checked against it")
    counts = {}
    for text in prompts:
        n = int(count_tokens(text))
        counts[text] = n
        if n > limit:
            raise UnreadableInput(
                f"a prompt needs {n} of the scorer's {limit} text positions and "
                f"is REFUSED rather than truncated: {text[:80]!r}... A truncated "
                f"prompt scores a question nobody asked. This is the bound #1854's "
                f"own 70-word example runs into; the #1864 reference request fits")
    return {"limit": limit, "counts": counts,
            "max": max(counts.values()) if counts else 0}


def prompt_score_stats(scores: np.ndarray) -> list[dict]:
    """Per-prompt reduction of a `(frames, prompts)` score matrix.

    `frame_min` and `frame_max` are the observed per-frame range, and `mean` and
    `sd` sit beside them so a reader can see how far a value is from the reference
    IN THE REFERENCE'S OWN UNITS rather than only whether it cleared a line. This
    is `reference_bounds`' reduction, on a different statistic.
    """
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2:
        raise UnreadableInput(
            f"a score matrix of shape {a.shape} cannot be reduced: it needs at "
            f"least one frame and at least a true prompt plus one decoy")
    out = []
    for j in range(a.shape[1]):
        v = a[:, j]
        out.append({"n": int(v.size), "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                    "frame_min": float(v.min()), "frame_max": float(v.max())})
    return out


def discrimination(scores: np.ndarray, labels: list[str],
                   true_index: int = 0) -> dict:
    """S2's ranking: which prompt this render scores highest, and by how much.

    The ranking statistic is the per-prompt MEAN over frames, because the question
    is about the render and not about one frame. The per-frame win fraction is
    carried beside it and reported, so a render that ranks correctly on average
    while losing most frames is visible rather than hidden behind one number.

    `margin` is the true prompt's mean minus the BEST decoy's mean. It is reported
    on every run, passing or failing. It is not a threshold and nothing is
    compared against it: the verdict is the set assertion `argmax == true`.
    """
    a = np.asarray(scores, dtype=np.float64)
    stats = prompt_score_stats(a)
    means = np.array([s["mean"] for s in stats])
    order = list(np.argsort(-means))
    best = int(order[0])
    decoys = [j for j in range(a.shape[1]) if j != true_index]
    best_decoy = int(max(decoys, key=lambda j: means[j]))
    per_frame_true_wins = int((a.argmax(axis=1) == true_index).sum())
    return {
        "labels": list(labels),
        "true_index": int(true_index),
        "argmax_index": best,
        "argmax_label": labels[best],
        "true_first": bool(best == true_index),
        "means": [float(m) for m in means],
        "ranking": [labels[j] for j in order],
        "best_decoy_index": best_decoy,
        "best_decoy_label": labels[best_decoy],
        "margin": float(means[true_index] - means[best_decoy]),
        "per_frame_true_wins": per_frame_true_wins,
        "frames": int(a.shape[0]),
        "per_frame_true_win_fraction": per_frame_true_wins / float(a.shape[0]),
        "null": adherence_null(len(decoys)),
        "stats": stats,
    }


def adherence_arm_valid(disc: dict) -> dict:
    """IS THIS ARM'S ADHERENCE NUMBER READABLE AT ALL? One predicate, every arm.

    `scorer_precondition` below asks this of the ORACLE's own render and RAISES,
    because a scorer that cannot rank upstream's own prompt first on upstream's
    own frames is broken and nothing it says afterwards is a measurement. The
    same two conditions decide whether a number is readable on ANY set of frames,
    and an ablation arm is a set of frames: blur a render far enough and the
    scorer stops ranking the true prompt first there too, at which point that
    arm's CLIP mean has stopped measuring adherence and has become a number.

    THIS EXISTS BECAUSE A ROW PUBLISHED SUCH A NUMBER AS ITS HEADLINE.
    `LTX25-ADHERENCE-DETAIL-LOSS` reported `+1.9131` from a blurred arm on which
    a decoy outranked the true prompt by 0.4036, while the check that would have
    caught it ran once, on the unblurred reference. A precondition asked of one
    arm is not a precondition of the run. So this returns a verdict rather than
    raising, every scored arm records it, and a caller that wants the raise calls
    `scorer_precondition`.

    A ZERO margin fails it, and that case is the one worth naming: a scorer that
    returns the same value for every prompt has an argmax, and numpy's is the
    first index, which is the true prompt. It would clear a bare `argmax == true`
    while measuring nothing. The margin must be strictly positive.
    """
    argmax, margin = disc["argmax_label"], float(disc["margin"])
    if not disc["true_first"]:
        return {"valid": False, "argmax_label": argmax, "margin": margin,
                "reason": (f"the scorer ranks {argmax!r} above the true prompt "
                           f"by {-margin:+.4f}, so this arm's CLIP mean is not a "
                           f"reading of prompt adherence")}
    if not (margin > 0.0):
        return {"valid": False, "argmax_label": argmax, "margin": margin,
                "reason": (f"the true prompt and the best decoy "
                           f"{disc['best_decoy_label']!r} score identically "
                           f"(margin {margin:+.6f}); a scorer with no separation "
                           f"has an argmax and no information")}
    return {"valid": True, "argmax_label": argmax, "margin": margin, "reason": ""}


def scorer_precondition(ref_disc: dict) -> dict:
    """S0. The instrument must prove it can say no, on the ORACLE's own frames.

    Upstream rendered this prompt and the result is committed. If the scorer
    cannot rank that prompt first over the decoys, on that render, then it is
    broken here -- wrong checkpoint, wrong preprocessing, wrong frames -- and
    anything it then says about OUR render is a code verdict wearing a
    measurement's clothes. So this raises, and the run exits EXIT_UNREADABLE with
    no adherence number published at all. A FAIL would say our render is worse
    than a reference that was never established.

    THE PREDICATE IS `adherence_arm_valid` ABOVE, and it is shared rather than
    restated. Two copies of a refusal rule drift, and the copy that drifts is the
    one nothing calls on the failing path -- which is how an ablation sweep in
    `ltx25-adherence-detail-loss.py` published four rows this rule forbids.
    """
    if not adherence_arm_valid(ref_disc)["valid"]:
        if not ref_disc["true_first"]:
            raise UnreadableInput(
                f"S0 FAILED: on the #1864 REFERENCE render the scorer ranks "
                f"{ref_disc['argmax_label']!r} above the prompt that render was "
                f"actually made from, by {-ref_disc['margin']:+.4f}. The "
                f"instrument cannot tell upstream's own good render of this "
                f"prompt from a decoy, so it measures nothing here and no "
                f"adherence number is published. This is a dead candidate and a "
                f"finding, not a thing to loosen")
        raise UnreadableInput(
            f"S0 FAILED: on the #1864 REFERENCE render the true prompt and the "
            f"best decoy {ref_disc['best_decoy_label']!r} score identically "
            f"(margin {ref_disc['margin']:+.6f}). A scorer with no separation has "
            f"an argmax and no information; it would clear a bare set assertion "
            f"while measuring nothing")
    return {"passed": True, "margin": ref_disc["margin"],
            "reference_ranking": ref_disc["ranking"],
            "null": ref_disc["null"]}


def adherence_checks(label: str, ours: dict, ref: dict) -> list[tuple]:
    """S1 and S2 for OUR render, as `(name, pass, detail, judges)` tuples.

    The reference's own S2 is not repeated here: it is S0, and S0 raises rather
    than failing, because a broken instrument is an unreadable input and not a bad
    render. Two checks in the table, and each of them can fail.
    """
    out: list[tuple] = []
    t = ours["true_index"]
    o, r = ours["stats"][t], ref["stats"][t]
    lo = r["frame_min"]
    v = o["mean"]
    # HOW LOOSE, MEASURED RATHER THAN ASSUMED. The reference's own per-frame
    # minimum sits some number of its own per-frame standard deviations below its
    # mean, and that number is a property of this reference render, not a constant.
    # It is printed so the bound's looseness is visible instead of argued.
    slack_sd = ((r["mean"] - lo) / r["sd"]) if r["sd"] > 0 else float("inf")
    out.append((
        f"absolute.{label}.adherence_clip", v >= lo,
        f"{v:.4f} >= {lo:.4f}, the reference's per-frame MINIMUM CLIP score "
        f"against its own prompt (reference mean {r['mean']:.4f}, per-frame sd "
        f"{r['sd']:.4f}, n={r['n']}); margin {v - lo:+.4f}; the bound sits "
        f"{slack_sd:.2f} of the reference's own per-frame sd below its mean; "
        + ("worse than the oracle on this statistic" if v < lo
           else "no worse than the oracle on this statistic"),
        "treatment"))
    out.append((
        f"absolute.{label}.adherence_argmax", ours["true_first"],
        f"argmax over {len(ours['labels'])} prompts is {ours['argmax_label']!r}; "
        f"margin to the best decoy {ours['best_decoy_label']!r} is "
        f"{ours['margin']:+.4f}; per-frame wins "
        f"{ours['per_frame_true_wins']}/{ours['frames']} "
        f"({ours['per_frame_true_win_fraction']:.3f}); null for an uninformative "
        f"scorer is 1/{len(ours['labels'])} = {ours['null']:.4f}; ranking "
        f"{' > '.join(ours['ranking'])}",
        "treatment"))
    return out


class ClipAdherenceScorer:
    """CLIP, from a LOCAL directory whose bytes are already pinned.

    EVERY CHOICE HERE IS UPSTREAM'S, AND EACH ONE IS CITED. vLLM is the primary
    reference and it implements this path, so the feature route is vLLM's:

      the projections     `vllm/model_executor/models/clip.py:808` (text) and
                          `:820` (vision) at `5559679229` -- `nn.Linear(..., bias=False)`
                          into `config.projection_dim`.
      what a feature IS   `clip.py:847`, `text_features = self.text_projection(pooled_output)`,
                          and `:867`, `image_features = self.visual_projection(pooled_output)`.
                          The model returns the PROJECTED features and does not
                          normalise them.
      the ENTRY POINTS    `tests/models/multimodal/pooling/test_clip.py:52-57`
                          calls HuggingFace `get_image_features(pixel_values=...)`
                          and `get_text_features(input_ids=..., attention_mask=...)`
                          as the reference vLLM's own `.embed()` is checked
                          against, through `check_embeddings_close` at `:67`.
                          So those two calls are what this scorer calls.
      dtype               `test_clip.py:75`, `@pytest.mark.parametrize("dtype", ["float"])`.
                          Upstream runs this pooling path in f32, so f32 is
                          inherited rather than chosen, and it is upstream's
                          polarity and not a widening of ours.
      the CONTEXT         `test_clip.py:41`, `max_model_len=77`.
      the SCORE itself    vLLM-Omni `tests/e2e/accuracy/helpers.py:507-511` at
                          `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7`: L2-normalise
                          both projected embeddings, take the cosine, multiply by
                          100. That repository is UNPINNED (#633), so it is a
                          source reading at a stated revision.

    Computed as a matrix rather than one pair at a time, and the value is the same
    number because each embedding is L2-normalised independently of the other
    side, so the batched dot product IS the per-pair cosine. The ported case in
    `tests/scripts/test_ltx25_prompt_adherence.py` asserts that against
    upstream's own tolerance rather than leaving it as an argument.

    `local_files_only=True` everywhere. A gate that can reach the network can
    silently score with whatever the hub is serving today, and the pin would be
    decoration.
    """

    def __init__(self, model_dir: str, pin: dict):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise UnreadableInput(
                f"the adherence scorer needs `transformers` and torch ({exc}). "
                f"They are the runtime vLLM's own CLIP pooling test and "
                f"vLLM-Omni's CLIPScorer use; install them or omit "
                f"--adherence-model")
        try:
            # f32: upstream's own dtype for this path, `test_clip.py:75`.
            self._model = CLIPModel.from_pretrained(
                model_dir, local_files_only=True, dtype=torch.float32)
            self._processor = CLIPProcessor.from_pretrained(model_dir,
                                                            local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - any load failure is unreadable
            raise UnreadableInput(
                f"{model_dir}: cannot load the pinned CLIP scorer ({exc})")
        self._model.eval()
        self.limit = int(pin.get("text_context_positions") or 0)

    def count_tokens(self, text: str) -> int:
        """The SCORER'S own tokenizer, with no truncation, so the count is real."""
        tok = getattr(self._processor, "tokenizer", None)
        if tok is None:
            raise UnreadableInput(
                "the pinned scorer exposes no tokenizer, so a prompt's length "
                "against its 77-position context cannot be established and no "
                "prompt may be admitted")
        return len(tok(text, truncation=False)["input_ids"])

    def features(self, frames: list[np.ndarray], prompts: list[str]):
        """The two projected feature blocks, by upstream's own entry points."""
        import torch
        from PIL import Image
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8), mode="RGB")
                  for f in frames]
        img_in = self._processor(images=images, return_tensors="pt")
        txt_in = self._processor(text=list(prompts), return_tensors="pt",
                                 padding=True)
        with torch.no_grad():
            img = self._model.get_image_features(pixel_values=img_in["pixel_values"])
            txt = self._model.get_text_features(
                input_ids=txt_in["input_ids"],
                attention_mask=txt_in.get("attention_mask"))
        # UPSTREAM'S OWN UNWRAP, not a local convenience. `test_clip.py:59-61`
        # reads `if not isinstance(pooled_output, torch.Tensor): pooled_output =
        # pooled_output.pooler_output`, because `transformers` returns a
        # `BaseModelOutputWithPooling` from these two calls and has already
        # REPLACED its `pooler_output` with the projected features -- exactly
        # `clip.py:867`'s `self.visual_projection(pooled_output)`. Taking `.
        # pooler_output` here is therefore the projected feature and not the
        # pre-projection pooled one, and porting the guard rather than assuming a
        # tensor is what makes that true across `transformers` versions.
        return self._unwrap(img), self._unwrap(txt)

    @staticmethod
    def _unwrap(v):
        import torch
        return v if isinstance(v, torch.Tensor) else v.pooler_output

    def score(self, frames: list[np.ndarray], prompts: list[str]) -> np.ndarray:
        """`(len(frames), len(prompts))` of cosine * 100."""
        try:
            img, txt = self.features(frames, prompts)
        except UnreadableInput:
            raise
        except ImportError as exc:
            raise UnreadableInput(
                f"the adherence scorer needs torch and Pillow ({exc})")
        img = img / img.norm(p=2, dim=-1, keepdim=True)
        txt = txt / txt.norm(p=2, dim=-1, keepdim=True)
        return (img @ txt.T).cpu().numpy().astype(np.float64) * 100.0


def adherence_report(render_dir: str, ref_frames: list[np.ndarray],
                     model_dir: str, pin_path: str = DEFAULT_ADHERENCE_PIN,
                     decoys_path: str = DEFAULT_ADHERENCE_DECOYS,
                     manifest_path: str = DEFAULT_ORACLE_MANIFEST) -> dict:
    """S0 then S1 and S2, in the one order that is safe.

    IDENTITY, THEN LENGTH, THEN S0, THEN NUMBERS. Each step refuses at
    EXIT_UNREADABLE rather than failing, because each of them is a statement that
    NOTHING was measured. An unidentified checkpoint, a truncated prompt and a
    scorer that cannot read the oracle's own render are all inputs that could not
    be read, and a 1 from any of them would say our render is worse than a
    reference that was never established.

    NO SCORER CAN BE INJECTED HERE, and the omission is deliberate rather than an
    oversight. A `scorer=` seam would let a caller hand in an object that never
    went through `assert_scorer_identity`, which is the same hole `load_decoys`
    refuses for the decoy set two functions above: a caller holding both sides of
    the comparison can clear it by construction. The scorer is built from the
    directory whose bytes were just verified, and from nothing else. The suite
    tests the arithmetic through the pure functions and the refusals through the
    command line, so nothing needs the seam.
    """
    pin = load_json_record(pin_path, "the pinned scorer record")
    identity = assert_scorer_identity(model_dir, pin)
    decoys = load_decoys(decoys_path)
    prompt = true_prompt(manifest_path)
    # THE TRUE PROMPT MAY NOT ALSO BE A DECOY. It would score identically against
    # itself, so the argmax would be a TIE that numpy breaks by index -- in favour
    # of the true prompt, at index 0. S2 would then pass by construction while
    # measuring nothing, which is exactly the trap `scorer_precondition` refuses a
    # zero margin for. Checked here rather than in `load_decoys`, which does not
    # know the prompt.
    if prompt in {d["text"] for d in decoys}:
        raise UnreadableInput(
            f"{decoys_path}: the true prompt is also listed as a decoy, so S2's "
            f"argmax is a tie broken in its favour by index and would pass "
            f"whatever the render depicts")
    prompts = [prompt] + [d["text"] for d in decoys]
    labels = ["true"] + [f"{d['kind']}:{i}" for i, d in enumerate(decoys)]
    scorer = ClipAdherenceScorer(model_dir, pin)
    lengths = refuse_overlong_prompts(prompts, scorer.count_tokens,
                                      int(pin.get("text_context_positions") or 0))
    ref_disc = discrimination(scorer.score(ref_frames, prompts), labels)
    s0 = scorer_precondition(ref_disc)
    ours_frames = [read_ppm(p) for p in frame_paths(render_dir)]
    ours_disc = discrimination(scorer.score(ours_frames, prompts), labels)
    return {
        "scorer": identity,
        "role": "instrument",
        "prompt": prompt,
        "prompt_source": os.path.abspath(manifest_path),
        "decoys": decoys,
        "decoys_source": os.path.abspath(decoys_path),
        "labels": labels,
        "text_context_positions": lengths["limit"],
        "prompt_token_counts": lengths["counts"],
        "s0": s0,
        "reference": ref_disc,
        "ours": ours_disc,
        "null": ours_disc["null"],
    }


def print_adherence_panel(ad: dict | None) -> None:
    """What was measured, or the declaration that nothing was.

    #1854 shipped the "not measured anywhere in this tree" line and the
    declaration was the point: a reader had to be able to see the gap. The same
    obligation runs the other way once a scorer is attached, so the heading
    changes with the fact, and the 77-position bound is stated in BOTH states
    because it limits what this gate can ever be asked.
    """
    if ad is None:
        print("PROMPT ADHERENCE IS NOT MEASURED IN THIS RUN (#1854). Pass "
              "--adherence-model to score it")
        print("against the committed prompt and decoys. The scorer is CLIP, an "
              "INSTRUMENT and not an oracle:")
        print("its checkpoint is pinned by revision and sha256 in "
              "tests/parity/goldens/ltx25_adherence/scorer-pin.json,")
        print("and its text context is 77 positions, so a prompt longer than that "
              "is REFUSED and never truncated.")
        return
    print("--- prompt adherence: CHECKED against the #1864 reference and the "
          "committed decoys (#1854) ---")
    print(f"scorer {ad['scorer']['repo']} at {ad['scorer']['revision']}, "
          f"{ad['scorer']['files_verified']} file digest(s) verified; role "
          f"{ad['role']}, not an oracle")
    print(f"prompt {ad['prompt']!r}")
    print(f"  read from {os.path.basename(ad['prompt_source'])}, the request the "
          f"reference render was taken at")
    print(f"  text context {ad['text_context_positions']} positions; the longest "
          f"prompt scored needs {max(ad['prompt_token_counts'].values())}. A "
          f"prompt over the limit is REFUSED, never truncated")
    print(f"  {len(ad['decoys'])} committed decoys, so S2's null is "
          f"{ad['null']:.4f} = 1/{len(ad['labels'])}")
    print(f"S0 PASSED on the reference: true prompt first by "
          f"{ad['s0']['margin']:+.4f}; ranking "
          f"{' > '.join(ad['reference']['ranking'])}")
    r = ad["reference"]["stats"][ad["reference"]["true_index"]]
    print(f"  reference CLIP against its own prompt: mean {r['mean']:.4f}, "
          f"per-frame [{r['frame_min']:.4f}, {r['frame_max']:.4f}], sd "
          f"{r['sd']:.4f}, n {r['n']}")
    print("  no absolute floor is applied: upstream's own `clip >= 20.0` is "
          "env-overridable and derived nowhere,")
    print("  and #1854 exists to keep a chosen constant out of the verdict. Both "
          "checks below are comparisons.")

def content_checks(content: dict, label: str, judges: str) -> list[tuple]:
    """C0 for ONE render, judged on its own content before anything is subtracted.

    Three checks, not four: "frames written" used to be a fourth and it could
    never be False, because `frame_paths` refuses an empty directory at
    EXIT_UNREADABLE long before this runs, and a row that cannot fail is a
    decoration in a table whose entire value is that every row can.

    Module level rather than a closure inside `_compare`, because the
    absolute-only path judges the same content by the same rule and two
    definitions of C0 could drift apart without anything noticing.
    """
    c = content
    return [
        (f"content.{label}.not_uniform", c["near_uniform_frames"] == 0,
         f"near-uniform frames {c['near_uniform_frames']} == 0 "
         f"(min per-frame variance {c['per_frame_var_min']:.3f})", judges),
        (f"content.{label}.distinct_frames",
         c["distinct_frame_hashes"] == c["frames"],
         f"{c['distinct_frame_hashes']} distinct of {c['frames']}", judges),
        (f"content.{label}.motion",
         c["zero_motion_pairs"] == 0 and c["adjacent_frame_mad_mean"] > 0.0,
         f"zero-motion pairs {c['zero_motion_pairs']}, "
         f"mean adjacent MAD {c['adjacent_frame_mad_mean']:.4f}", judges),
    ]


def print_absolute_panel(report: dict, gated: bool) -> None:
    """The panel, and a heading that says which of the two states it is in.

    #1854 shipped this block declaring itself unchecked, and the declaration was
    the point: a reader had to be able to see that the numbers decided nothing.
    The same obligation runs the other way now, so the heading changes with the
    fact rather than staying the reassuring one.
    """
    if not gated:
        print("--- absolute quality: REPORTED, and NOT CHECKED (#1854) ---")
        print("no threshold over these means anything without an oracle that renders "
              "LTX-2.5 or a pinned scoring model; pass --reference to gate the two "
              "blockiness ratios against the committed #1864 reference render")
    else:
        ref = report["reference"]
        print("--- absolute quality: the blockiness ratios are CHECKED against the "
              "#1864 reference (#1854) ---")
        print(f"reference {ref['source']}")
        print(f"  form {ref['form']}, {ref['frames']} frames, "
              f"{ref['digests_verified']} digest(s) verified against "
              f"{os.path.basename(ref['sums'])}")
        for name in REFERENCE_GATED:
            b = ref["bounds"][name]
            print(f"  {name:18s} reference mean {b['mean']:.6f}  per-frame "
                  f"[{b['frame_min']:.6f}, {b['frame_max']:.6f}]  sd {b['sd']:.6f}  "
                  f"n {b['n']}")
        print("  sharpness, the clipped fraction and audio RMS stay REPORTED: "
              "spec ltx25-oracle-absolute.md section 5")
    for lbl, q in report["absolute_quality"].items():
        print(f"{lbl:12s} sharpness={q['sharpness_mean']} "
              f"block8={q['blockiness_grid8']} block32={q['blockiness_grid32']} "
              f"clipped={q['clipped_fraction']} "
              f"audio_rms={q.get('audio_rms_mean')}")


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
    # OPTIONAL SINCE #1854. The absolute question is about ONE render, and the
    # tool used to be unable to ask it. `_absolute_only` records why neither
    # workaround -- passing arm A twice, or making the reference arm B -- is
    # admissible.
    ap.add_argument("--b", default=None,
                    help="arm B render directory (the change under test); omit it to "
                         "judge ONE render against --reference")
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
    # THE ABSOLUTE REFERENCE (#1854). Not a threshold: a render, whose identity is
    # asserted against the committed digests before a pixel of it is read, and
    # from whose own frames every bound is recomputed on each run.
    ap.add_argument("--reference", default=None,
                    help="the #1864 oracle render: the committed "
                         "tests/parity/goldens/ltx2_oracle/upstream-render.mp4, or a "
                         "directory of its frame_*.ppm. Every byte is checked against "
                         "SHA256SUMS before it is read")
    ap.add_argument("--reference-sums", default=DEFAULT_REFERENCE_SUMS,
                    help="the digest list the reference must appear in "
                         "(default: the committed one, resolved from this script)")
    # PROMPT ADHERENCE (#1854 sub-question 1, #2295). The scorer is an
    # INSTRUMENT and not an oracle: `ltx-2` still supplies the other side of
    # every comparison. Only the model DIRECTORY is a flag. The prompt comes
    # from the #1864 manifest, the decoys are committed, and the checkpoint's
    # every byte is hashed against the pin -- none of the three is overridable,
    # because a caller who could set them could clear S2 by construction.
    ap.add_argument("--adherence-model", default=None,
                    help="a LOCAL directory holding "
                         "openai/clip-vit-base-patch16 at the revision pinned in "
                         "tests/parity/goldens/ltx25_adherence/scorer-pin.json. "
                         "Every pinned file's sha256 is checked before a pixel is "
                         "read. Needs --reference: the bound is the reference "
                         "render's own per-frame minimum. CLIP's text context is "
                         "77 positions and a longer prompt is REFUSED, never "
                         "truncated")
    args = ap.parse_args()
    if args.adherence_model is not None and args.reference is None:
        ap.error("--adherence-model needs --reference: S1's bound IS the reference "
                 "render's own per-frame minimum CLIP score, and S0 refuses to "
                 "publish a number until the scorer has ranked the true prompt "
                 "first on the reference's frames. Without the reference there is "
                 "no bound and no precondition, only an absolute CLIP value, which "
                 "is uncalibrated and is exactly the convention #1854 refused")
    if args.b is None and args.reference is None:
        ap.error("--b or --reference is required: without either there is nothing "
                 "to compare this render against, and a tool that compared a render "
                 "with nothing would report a pass nobody may read")
    if args.b is None and args.control is not None:
        ap.error("--control repeats one of TWO arms and calibrates the delta between "
                 "them; with no --b there is no delta for it to calibrate")

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



def _absolute_only(args: argparse.Namespace) -> int:
    """ONE render, judged against the #1864 reference. No arm B anywhere.

    #1854 asks an ABSOLUTE question -- is this a good render of this prompt --
    and this tool could not ask it, because `--b` was required and the caller had
    to supply a comparison the absolute question does not use. Two ways around
    that were considered and both are worse than a second entry point:

      PASS THE RENDER AS BOTH ARMS. Every check then passes by construction:
      `bit_identical` short-circuits the identity block, `coherence` returns
      `k = 0.0` on a zero difference, and each alignment check matches a frame to
      itself. Landing that as the gate's invocation is
      `gate-comparing-shared-helper-proves-consistency-not-correctness` with both
      sides of the comparison the same directory.

      MAKE THE REFERENCE ARM B. #1743 relocated the identity bounds out of the
      verdict, but NOT `align.*` and `coherence.*`, which still decide it. Two
      renders of one prompt by two different engines are two different pictures,
      so those checks fail by construction and the run would exit 1 for a reason
      that is not a finding.

    What is judged here: C0 on the render's own content, then the reference
    checks. C0 first and for the reason it is first in `_compare` -- an arm that
    rendered nothing, rendered one colour or rendered one frame 25 times would
    otherwise produce a blockiness number and clear a band with it. A blank frame
    has no off-grid step either.

    The exit statuses keep their meanings exactly. 0 every check passed, 1 a
    check failed, 2 an input could not be read -- which is where an unverifiable
    reference lands, because a 1 would say this render is worse than a reference
    that was never established.
    """
    if not os.path.isdir(args.a):
        raise UnreadableInput(f"not a directory: {args.a}")

    meta, ref_frames = load_reference(args.reference, args.reference_sums)
    bounds = reference_bounds(ref_frames)
    meta["bounds"] = bounds
    meta["gated"] = list(REFERENCE_GATED)
    meta["reported"] = list(REFERENCE_REPORTED)

    report: dict = {
        "mode": "absolute_only",
        "thresholds": {},
        "inputs": {"a": os.path.abspath(args.a), "b": None, "control": None},
        "control_of": None,
        "reference": meta,
    }
    report["content"] = {args.label_a: arm_content(args.a)}
    report["absolute_quality"] = {
        args.label_a: absolute_quality(args.a, os.path.join(args.a, args.audio_name),
                                       gated=True)
    }

    # PROMPT ADHERENCE (#1854 sub-question 1). Runs only when a scorer is
    # supplied, and it refuses rather than fails at every step before the
    # numbers: an unidentified checkpoint, a prompt that does not fit CLIP's 77
    # positions, and a scorer that cannot rank the true prompt first on the
    # ORACLE's own frames are each a statement that NOTHING was measured, and
    # `UnreadableInput` is how this file says that.
    adherence = None
    if args.adherence_model:
        adherence = adherence_report(args.a, ref_frames, args.adherence_model)
        report["adherence"] = adherence

    checks: list[tuple[str, bool, str, str]] = []
    checks.extend(content_checks(report["content"][args.label_a], args.label_a,
                                 "treatment"))
    checks.extend(reference_checks(args.label_a, report["absolute_quality"][args.label_a],
                                   bounds))
    if adherence is not None:
        checks.extend(adherence_checks(args.label_a, adherence["ours"],
                                       adherence["reference"]))

    report["checks"] = [{"name": n, "pass": p, "detail": d, "judges": j}
                        for n, p, d, j in checks]
    treatment = [c for c in checks if c[3] == "treatment"]
    ok = all(c[1] for c in treatment)
    report["treatment_verdict"] = "PASS" if ok else "FAIL"
    # NO IDENTITY VERDICT AND NO CONTROL VERDICT, rather than a null one: both
    # are statements about a second render, and there is no second render. A
    # field carrying `IDENTICAL` here would answer a question nobody asked.
    report["identity_verdict"] = None
    report["identity_failed"] = []
    report["control_verdict"] = None

    c0_failed = [c[0] for c in treatment if c[0].startswith("content.") and not c[1]]
    abs_failed = [c[0] for c in treatment if c[0].startswith("absolute.") and not c[1]]
    if c0_failed:
        reading = "CONTENT_DEGENERATE"
    elif abs_failed:
        reading = "WORSE_THAN_ORACLE"
    elif adherence is not None:
        # THE READING NAMES BOTH HALVES ONCE BOTH WERE MEASURED. It still names
        # what was measured and nothing more: blockiness against the reference's
        # own band, and CLIP adherence against the reference's own per-frame
        # minimum plus the committed decoy ranking, on one request at one
        # geometry, with a scorer whose 77-position context cannot read #1854's
        # own 70-word example.
        reading = "NO_WORSE_THAN_ORACLE_ON_BLOCKINESS_AND_ADHERENCE"
    else:
        # NAMED FOR EXACTLY WHAT WAS MEASURED. Not "as good as the oracle" and
        # not "a good render": two of the four panel statistics sit inside the
        # reference's own per-frame band, on one request at one geometry, and
        # #1854's prompt-adherence half is unmeasured in THIS run.
        reading = "NO_WORSE_THAN_ORACLE_ON_BLOCKINESS"
    report["reading"] = reading
    report["verdict"] = "PASS" if ok else "FAIL"
    status = EXIT_PASS if ok else EXIT_FAIL

    print("=== ONE render, against the #1864 oracle reference (#1854) ===")
    print("this run makes NO arm-to-arm comparison: there is no arm B, so nothing "
          "here is about")
    print("identity, correspondence or coherence, and no check below is one of "
          "those.")
    for label, c in report["content"].items():
        print(f"{label:12s} frames={c['frames']} distinct={c['distinct_frame_hashes']} "
              f"mean={c['pixel_mean']:.3f} min_var={c['per_frame_var_min']:.1f} "
              f"near_uniform={c['near_uniform_frames']} "
              f"adj_mad={c['adjacent_frame_mad_mean']:.4f} "
              f"zero_motion_pairs={c['zero_motion_pairs']}")
    print_absolute_panel(report, gated=True)
    print_adherence_panel(adherence)
    print("--- checks ---")
    for n, p, d, j in treatment:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    print(f"READING {report['reading']}")
    print(f"VERDICT {report['verdict']} (exit {status})")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return status


def _compare(args: argparse.Namespace) -> int:
    # ONE RENDER OR TWO, decided here and nowhere else. The A/B body below is
    # unchanged by #1854 -- `--b` present runs exactly the code it ran before --
    # and the single-render question gets its own function rather than a hundred
    # guards threaded through this one.
    if args.b is None:
        return _absolute_only(args)
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
    # AND THE ABSOLUTE PANEL. Instrumentation without `--reference` (#1854 as
    # filed), and a gate on the two blockiness ratios with one.
    report["reference"] = None
    ref_bounds: dict = {}
    if args.reference:
        meta, ref_frames = load_reference(args.reference, args.reference_sums)
        ref_bounds = reference_bounds(ref_frames)
        meta["bounds"] = ref_bounds
        meta["gated"] = list(REFERENCE_GATED)
        meta["reported"] = list(REFERENCE_REPORTED)
        report["reference"] = meta
    report["absolute_quality"] = {
        lbl: absolute_quality(d, os.path.join(d, args.audio_name),
                              gated=bool(args.reference))
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
        """C0 for ONE render. ONE definition, at module scope, because the
        absolute-only path judges the same content by the same rule and two
        copies of a criterion drift without anything noticing."""
        checks.extend(content_checks(report["content"][label], label, judges))

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

    # THE ABSOLUTE CHECKS (#1854). Registered LAST among the treatment entries so
    # that a reader meets the two-render question first and the one-render
    # question second, which is the order the report has always been argued in.
    # They are absent, not vacuously true, when no reference was supplied.
    if args.reference:
        for lbl in (args.label_a, args.label_b):
            checks.extend(reference_checks(lbl, report["absolute_quality"][lbl],
                                           ref_bounds))

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
    abs_failed = [c[0] for c in treatment if c[0].startswith("absolute.") and not c[1]]
    other_failed = [c[0] for c in treatment if not c[1]
                    and c[0] not in c0_failed + align_failed + coh_failed + abs_failed]
    if c0_failed:
        reading = "CONTENT_DEGENERATE"
    elif abs_failed:
        # THE STRONGEST STATEMENT THIS TOOL CAN MAKE, so it outranks every
        # relative one. `align.*` and `coherence.*` compare the two arms with each
        # other; this compares an arm with upstream's own render of the same
        # request. An arm outside the reference's own band is worse than the
        # oracle whatever the other arm does, and burying that under MISALIGNED
        # would report the smaller finding.
        reading = "WORSE_THAN_ORACLE"
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
    print_absolute_panel(report, gated=report["reference"] is not None)
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
