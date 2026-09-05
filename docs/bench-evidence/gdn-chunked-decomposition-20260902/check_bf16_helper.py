#!/usr/bin/env python3
"""Validation (a) of the replica: gdn_decomp.bf16() is BIT-IDENTICAL to
torch.bfloat16 round-to-nearest-even.

This is the one script in this directory that needs `torch`. It is separate
because the other four are numpy-only and must stay runnable without it.

The check is on bit patterns, not on a tolerance: a helper that rounds
differently in the last bit is a different instrument, and every number in
this study is a rounding-placement measurement.

Three populations, because a uniform random sample cannot reach the cases
that break a hand-written rounder:
  1. magnitude sweep -- 12 decades, positive and negative
  2. exact ties -- mantissas whose discarded low half is exactly 0x8000, the
     only inputs on which round-half-to-even differs from round-half-away
  3. structural corners -- zeros, subnormals, the bf16 min/max normals,
     infinities and NaN

Run:  python3 docs/bench-evidence/gdn-chunked-decomposition-20260902/check_bf16_helper.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_decomp import bf16  # noqa: E402

import torch  # noqa: E402


def torch_bf16(x):
    """The oracle: f32 -> bfloat16 -> f32, torch's own RTNE."""
    return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16).float().numpy()


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def compare(name, x):
    x = np.asarray(x, np.float32)
    got, want = bf16(x), torch_bf16(x)
    gb, wb = bits(got), bits(want)
    # NaN payloads are not preserved identically by every path; compare NaN by
    # NaN-ness and everything else by bit pattern.
    nan = np.isnan(want)
    same = np.array_equal(gb[~nan], wb[~nan]) and bool(np.all(np.isnan(got[nan])))
    n = int(x.size)
    bad = int((gb[~nan] != wb[~nan]).sum())
    print(f"  {name:34s} n={n:6d}  identical={n - bad:6d}/{n}  "
          f"{'OK' if same else 'MISMATCH'}")
    if not same:
        i = int(np.flatnonzero(gb[~nan] != wb[~nan])[0])
        print(f"    first mismatch: x=0x{bits(x[~nan])[i]:08x} "
              f"got=0x{gb[~nan][i]:08x} want=0x{wb[~nan][i]:08x}")
    return same, n, bad


def main():
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    rng = np.random.default_rng(20260902)
    ok = True
    total = 0
    bad_total = 0

    # 1. magnitude sweep: 12 decades, both signs.
    mags = 10.0 ** rng.uniform(-6.0, 6.0, size=10000)
    sweep = (mags * rng.choice([-1.0, 1.0], size=mags.shape)).astype(np.float32)
    o, n, b = compare("magnitude sweep, 12 decades", sweep)
    ok &= o; total += n; bad_total += b

    # 2. exact ties: force the discarded low 16 bits to exactly 0x8000, which
    #    is the ONLY input class where round-half-to-even and round-half-away
    #    disagree. A random sample essentially never lands here.
    hi = rng.integers(0, 1 << 16, size=5000, dtype=np.uint64).astype(np.uint32)
    # keep the exponent away from inf/nan so the tie is a real rounding tie
    hi = (hi & np.uint32(0x7FFF)) | (rng.integers(1, 0xFE, size=5000).astype(np.uint32) << np.uint32(23))
    hi = hi | (rng.integers(0, 2, size=5000, dtype=np.uint64).astype(np.uint32) << np.uint32(31))
    ties = ((hi & np.uint32(0xFFFF0000)) | np.uint32(0x8000)).view(np.float32)
    o, n, b = compare("exact ties (low half == 0x8000)", ties)
    ok &= o; total += n; bad_total += b

    # 3. structural corners.
    corners = np.array(
        [0.0, -0.0,
         np.float32(1e-45), np.float32(-1e-45),          # smallest subnormals
         np.float32(1.1754944e-38), np.float32(-1.1754944e-38),  # min normal f32
         np.float32(9.1835e-41),                        # subnormal, rounds to 0 or subnormal
         np.float32(1.1755e-38), np.float32(3.3895314e38),
         np.float32(-3.3895314e38),                     # bf16 max normal
         np.float32(3.4028235e38), np.float32(-3.4028235e38),  # f32 max -> overflows bf16
         np.inf, -np.inf, np.nan,
         np.float32(1.0), np.float32(-1.0), np.float32(2.0), np.float32(0.5)],
        dtype=np.float32)
    # every bf16 tie at unit scale: 1.0 + k*2^-9 for k odd
    unit_ties = (np.float32(1.0) + (np.arange(1, 512, 2).astype(np.float32) * np.float32(2.0 ** -9))).astype(np.float32)
    o, n, b = compare("structural corners", np.concatenate([corners, unit_ties, -unit_ties]))
    ok &= o; total += n; bad_total += b

    print(f"\nTOTAL: {total - bad_total}/{total} bit-identical to torch.bfloat16")
    if not ok:
        print("FAIL: gdn_decomp.bf16 is NOT torch.bfloat16")
        return 1
    print("PASS: gdn_decomp.bf16 IS torch.bfloat16, bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
