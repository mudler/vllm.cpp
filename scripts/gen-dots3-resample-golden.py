#!/usr/bin/env python3
"""Generate the scipy resample goldens for dots3-note W7c-2 (#2828).

The oracle is `scipy.signal.resample_poly` at its defaults, which is what
upstream's own `resample_audio_scipy` calls
(`vllm/multimodal/audio.py:232-250` @ `9035151d6`). Upstream's DEFAULT arm is
`pyav`/libswresample, which is not bit-identical to itself across CPU dispatch
and therefore cannot be gated at all; `.agents/specs/dots3-note.md` §4.17.1
records the measurement and the decision.

This script writes `tests/vllm/models/dots3_note_resample_golden.h`. It emits
BOTH sides — the input samples and scipy's answer — because the input must be
identical in C++ to the bit, and two libm `sin` implementations are not
required to agree. Nothing in the gate then depends on a transcendental
agreeing across platforms.

It also re-derives scipy's algorithm from its documented steps, in plain
Python, and prints how far that reimplementation is from scipy itself. That
number is what makes the C++ port a port rather than a guess: the same six
steps, written twice, agreeing.

Usage:  python3 scripts/gen-dots3-resample-golden.py [--check]
        --check re-generates and diffs instead of writing, so CI or a reviewer
        can see that the committed header is the script's own output.
"""

import argparse
import math
import os
import sys

import numpy as np
import scipy
from scipy.signal import resample_poly

# ── the fixture signals ─────────────────────────────────────────────────────
#
# BAND-LIMITED: three tones, all below the 8 kHz output Nyquist, so a correct
# resample preserves them and the case measures interpolation.
#
# ALIASING: a 15 kHz tone, representable at 44100 (Nyquist 22050) and ABOVE the
# 8 kHz output Nyquist. This is the only content that separates a real
# anti-alias filter from picking samples, which is mutation M3.


def signal_band_limited(n, sr):
    t = np.arange(n, dtype=np.float64) / float(sr)
    v = (0.6 * np.sin(2.0 * np.pi * 440.0 * t)
         + 0.3 * np.sin(2.0 * np.pi * 1234.0 * t)
         + 0.1 * np.sin(2.0 * np.pi * 3000.0 * t + 0.7))
    return v.astype(np.float32)


def signal_aliasing(n, sr):
    t = np.arange(n, dtype=np.float64) / float(sr)
    v = (0.5 * np.sin(2.0 * np.pi * 300.0 * t)
         + 0.45 * np.sin(2.0 * np.pi * 15000.0 * t))
    return v.astype(np.float32)


# name, orig_sr, target_sr, n_in, generator
CASES = [
    ("Wav44100", 44100, 16000, 441, signal_band_limited),
    ("Wav48000", 48000, 16000, 480, signal_band_limited),
    ("Wav22050", 22050, 16000, 220, signal_band_limited),
    ("Wav8000", 8000, 16000, 80, signal_band_limited),
    ("Alias44100", 44100, 16000, 441, signal_aliasing),
]


# ── scipy's algorithm, re-derived, to check the transcription ───────────────
#
# The six steps of `resample_poly` as `.agents/specs/dots3-note.md` §4.17.3
# states them, written from scipy's documented behaviour with nothing imported
# from scipy. `i0` is its own power series `sum_k (x^2/4)^k / (k!)^2`, which
# converges quickly for the beta = 5 the kaiser default uses.


def _i0(x):
    term = 1.0
    total = 1.0
    q = x * x / 4.0
    for k in range(1, 200):
        term *= q / (k * k)
        total += term
        if term < 1e-18 * total:
            break
    return total


def _sinc(x):
    if x == 0.0:
        return 1.0
    p = math.pi * x
    return math.sin(p) / p


def _design(up, down):
    max_rate = max(up, down)
    f_c = 1.0 / max_rate
    half_len = 10 * max_rate
    numtaps = 2 * half_len + 1
    alpha = 0.5 * (numtaps - 1)
    i0_beta = _i0(5.0)
    h = []
    for i in range(numtaps):
        m = i - alpha
        r = (i - alpha) / alpha
        win = _i0(5.0 * math.sqrt(max(0.0, 1.0 - r * r))) / i0_beta
        h.append(f_c * _sinc(f_c * m) * win)
    s = 0.0
    for v in h:
        s += v
    return [v / s for v in h], half_len


def rederived_resample(x, orig_sr, target_sr):
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    if up == down == 1:
        return np.asarray(x, dtype=np.float64)
    h, half_len = _design(up, down)
    h = [v * up for v in h]
    n_in = len(x)
    n_out = (n_in * up + down - 1) // down
    n_pre_pad = down - (half_len % down)
    n_pre_remove = (half_len + n_pre_pad) // down

    def out_len(len_h):
        return ((n_in - 1) * up + len_h - 1) // down + 1

    n_post_pad = 0
    while out_len(len(h) + n_pre_pad + n_post_pad) < n_out + n_pre_remove:
        n_post_pad += 1
    hh = [0.0] * n_pre_pad + h + [0.0] * n_post_pad
    len_h = len(hh)
    y = np.empty(n_out, dtype=np.float64)
    for oi in range(n_out):
        t = (oi + n_pre_remove) * down
        j_lo = max(0, (t - len_h + up) // up)
        j_hi = min(n_in - 1, t // up)
        acc = 0.0
        for j in range(j_lo, j_hi + 1):
            acc += hh[t - j * up] * x[j]
        y[oi] = acc
    return y


def scipy_resample(x, orig_sr, target_sr):
    g = math.gcd(orig_sr, target_sr)
    return resample_poly(np.asarray(x, dtype=np.float64),
                         target_sr // g, orig_sr // g)


def fmt(v):
    # `%.9g` alone emits "0" for a zero and "1" for a one, and `0f` is not a
    # C++ literal — the suffix needs a decimal point or an exponent in front of
    # it. Nine significant digits is what makes a float32 round-trip exactly.
    text = "%.9g" % float(np.float32(v))
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def emit(values, per_line=6, indent="    "):
    lines = []
    for i in range(0, len(values), per_line):
        lines.append(indent + " ".join(fmt(v) + "," for v in values[i:i + per_line]))
    return "\n".join(lines)


def build():
    out = []
    out.append("// GENERATED by scripts/gen-dots3-resample-golden.py. DO NOT EDIT BY HAND.")
    out.append("//")
    out.append("// The oracle is `scipy.signal.resample_poly` at its defaults, which is what")
    out.append("// upstream's `resample_audio_scipy` calls (`vllm/multimodal/audio.py:232-250`")
    out.append("// @ `9035151d6`). Upstream's DEFAULT arm is `pyav`/libswresample, which is not")
    out.append("// bit-identical to itself across CPU dispatch and cannot be gated by anyone;")
    out.append("// `.agents/specs/dots3-note.md` §4.17.1 records that measurement and the")
    out.append("// decision to implement the scipy arm instead. This header is a CONSISTENCY")
    out.append("// gate against a stated algorithm and NOT parity with upstream's default.")
    out.append("//")
    out.append("// Both sides are emitted. The INPUT is committed as float32 literals rather")
    out.append("// than recomputed in C++ because two libm `sin` implementations are not")
    out.append("// required to agree to the bit, and a gate that depends on that is a gate")
    out.append("// that reddens on a different machine for a reason no diff caused.")
    out.append("//")
    out.append("// scipy " + scipy.__version__ + ", numpy " + np.__version__ + ".")
    out.append("")
    out.append("#ifndef VLLM_TESTS_DOTS3_NOTE_RESAMPLE_GOLDEN_H_")
    out.append("#define VLLM_TESTS_DOTS3_NOTE_RESAMPLE_GOLDEN_H_")
    out.append("")
    out.append("#include <cstdint>")
    out.append("")
    out.append("namespace dots3_resample_golden {")
    out.append("")
    out.append("struct Case {")
    out.append("  const char* name;")
    out.append("  int orig_sr;")
    out.append("  int target_sr;")
    out.append("  const float* input;")
    out.append("  int n_in;")
    out.append("  const float* expected;")
    out.append("  int n_out;")
    out.append("};")
    out.append("")

    report = []
    for name, orig, target, n_in, gen in CASES:
        x32 = gen(n_in, orig)
        x = x32.astype(np.float64)
        ref = scipy_resample(x, orig, target)
        mine = rederived_resample(list(x), orig, target)
        assert len(ref) == len(mine)
        expect_len = -(-n_in * target // orig)
        assert len(ref) == expect_len, (name, len(ref), expect_len)
        worst = float(np.max(np.abs(ref - mine)))
        scale = float(np.max(np.abs(ref)))
        report.append((name, orig, target, n_in, len(ref), worst, scale,
                       float(np.max(np.abs(ref.astype(np.float32)
                                           - mine.astype(np.float32))))))
        out.append("// %s: %d -> %d Hz, %d in, %d out." % (name, orig, target, n_in, len(ref)))
        out.append("inline constexpr float k%sIn[] = {" % name)
        out.append(emit(list(x32)))
        out.append("};")
        out.append("inline constexpr float k%sOut[] = {" % name)
        out.append(emit(list(ref.astype(np.float32))))
        out.append("};")
        out.append("")

    out.append("inline constexpr Case kCases[] = {")
    for name, orig, target, n_in, gen in CASES:
        x32 = gen(n_in, orig)
        n_out = -(-n_in * target // orig)
        out.append('    {"%s", %d, %d, k%sIn, %d, k%sOut, %d},'
                   % (name, orig, target, name, n_in, name, n_out))
    out.append("};")
    out.append("inline constexpr int kNumCases = %d;" % len(CASES))
    out.append("")
    out.append("}  // namespace dots3_resample_golden")
    out.append("")
    out.append("#endif  // VLLM_TESTS_DOTS3_NOTE_RESAMPLE_GOLDEN_H_")
    return "\n".join(out) + "\n", report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "tests", "vllm", "models",
                        "dots3_note_resample_golden.h")
    text, report = build()

    print("scipy %s, numpy %s" % (scipy.__version__, np.__version__))
    print("%-12s %8s %8s %6s %6s %12s %12s %10s"
          % ("case", "orig", "target", "n_in", "n_out",
             "|redrv-scipy|", "scale", "as float32"))
    for row in report:
        print("%-12s %8d %8d %6d %6d %12.4e %12.6f %10.4e" % row)

    if args.check:
        with open(path) as f:
            have = f.read()
        if have != text:
            print("MISMATCH: %s is not this script's output" % path, file=sys.stderr)
            return 1
        print("OK: %s is this script's output" % path)
        return 0
    with open(path, "w") as f:
        f.write(text)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
