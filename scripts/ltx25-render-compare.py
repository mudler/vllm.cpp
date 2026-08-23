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

    if v["bit_identical"]:
        checks.append(("video.bit_identical", True, "every frame file sha256-equal",
                       "treatment"))
    else:
        checks.append(
            ("video.mean_abs", v["mean_abs"] <= args.max_mean_abs,
             f"{v['mean_abs']:.6f} <= {args.max_mean_abs}", "treatment")
        )
        checks.append(
            ("video.psnr_min_db", v["psnr_min_db"] >= args.min_psnr_db,
             f"{v['psnr_min_db']:.3f} >= {args.min_psnr_db}", "treatment")
        )
        checks.append(
            ("video.ssim_min", v["ssim_min"] >= args.min_ssim,
             f"{v['ssim_min']:.6f} >= {args.min_ssim}", "treatment")
        )
        if v.get("temporal_ratio") is not None:
            checks.append(
                ("video.temporal_ratio", v["temporal_ratio"] <= args.max_temporal_ratio,
                 f"{v['temporal_ratio']:.6f} <= {args.max_temporal_ratio}", "treatment")
            )
        else:
            checks.append(("video.temporal_ratio", False, "no adjacent-frame denominator",
                           "treatment"))

    a = report["audio"]
    if not a.get("present"):
        checks.append(("audio.present", False, a.get("reason", "absent"), "treatment"))
    elif not a.get("comparable"):
        checks.append(("audio.comparable", False, "shape or sample rate differs",
                       "treatment"))
    elif a.get("bit_identical"):
        checks.append(("audio.bit_identical", True, "wav sha256-equal", "treatment"))
    else:
        checks.append(("audio.psnr_db", a["psnr_db"] >= args.min_audio_psnr_db,
                       f"{a['psnr_db']:.3f} >= {args.min_audio_psnr_db}", "treatment"))
        checks.append(("audio.pearson_r", (a["pearson_r"] or 0.0) >= args.min_audio_corr,
                       f"{a['pearson_r']} >= {args.min_audio_corr}", "treatment"))

    report["checks"] = [{"name": n, "pass": p, "detail": d, "judges": j}
                        for n, p, d, j in checks]
    treatment = [c for c in checks if c[3] == "treatment"]
    control_c = [c for c in checks if c[3] == "control"]
    ok = all(c[1] for c in treatment)
    report["treatment_verdict"] = "PASS" if ok else "FAIL"

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
    print("--- checks ---")
    print(f"  these decide the verdict: the {args.label_a} vs {args.label_b} comparison")
    for n, p, d, j in treatment:
        print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    if control_c:
        print(f"  these decide whether the control {args.label_control} is a noise floor "
              f"at all, and they do NOT decide the "
              f"{args.label_a} vs {args.label_b} verdict")
        for n, p, d, j in control_c:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}: {d}")
    if degenerate_reason:
        print(f"CONTROL DEGENERATE: {degenerate_reason}")
    print(f"VERDICT {report['verdict']} (exit {status})")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
