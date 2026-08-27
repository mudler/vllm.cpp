#!/usr/bin/env python3
"""safetensors -> GGUF converter for `glm5_next` (GLM-5.3-Flash).

W7a of `MODEL-MM-GLM53-FLASH` ([#1998](https://github.com/mudler/vllm.cpp/issues/1998),
wave issue in the pull request body). Spec: `.agents/specs/glm5-next-flash.md`.

**Why this file exists at all.** `zai-org/GLM-5.3-Flash` publishes FP8 (305.78
GiB), BF16 (598.53 GiB) and NVFP4 (181.32 GiB) artifacts, and the largest device
this project reaches is `dgx:gpu0` at ~119.63 GiB unified. Nothing published
fits. All four HuggingFace repositories named `*-GGUF` contain zero `.gguf`
files, and no upstream tool can emit this architecture: llama.cpp at our pin
`b10451` knows `glm4`, `glm4moe` and `glm-dsa` and has no `glm5_next` (verified
2026-08-26, `src/llama-arch.h` and a repository-wide search). So the converter is
ours to write, and until it exists there is no artifact of this model that any
gate on this fleet can load.

**What this file is NOT.** It has never been run against the real checkpoint.
W7a deliberately produces no artifact -- the checkpoint is 300-600 GiB and
downloading it needs authority and disk this wave has neither of. Everything
below is gated on synthetic tiny-shape fixtures, and artifact production is owed
(spec `## Owed`, O7). The k-quant encoders are the exception: they are gated
byte-for-byte against the pinned llama.cpp reference, so the one part that
cannot be checked by inspection is checked by an oracle.

**Where the bytes come from.** llama.cpp is never the mirror source -- AGENTS.md
settles behaviour against vLLM, and where vLLM implements nothing, against the
declared secondary oracle. What llama.cpp supplies here is *convention*: the
GGUF container layout, the metadata key spellings, the tensor-name spellings and
the k-quant block encodings. Every one of those is read at our pin `b10451` and
not at llama.cpp `master`:

  - metadata keys       `gguf-py/gguf/constants.py` @ b10451, including
                        `class KDA` (:262) with `{arch}.kda.head_dim` (:263) and
                        `{arch}.kda.gate_lower_bound` (:264). `KDA.SAFE_GATE`
                        exists only at `master` and is NOT used here, because
                        GLM-5.3-Flash's `linear_attn_config` declares no
                        `safe_gate` key. Nothing in this file requires a pin
                        advance.
  - tensor names        `src/llama-arch.cpp` @ b10451: the KDA set at :465-479
                        (`ssm_conv1d_q/k/v`, `ssm_f_a`, `ssm_f_b`, `ssm_g_a`,
                        `ssm_g_b`, `ssm_beta`, `ssm_a`, `ssm_dt`, `ssm_norm`),
                        the MLA set at :481-486 and the indexer set at :626-636.
  - HF module paths     `gguf-py/gguf/tensor_mapping.py` @ b10451 :896-933, whose
                        Kimi-Linear KDA paths (`self_attn.q_conv1d`,
                        `self_attn.f_a_proj`, ...) are GLM-5.3-Flash's paths
                        verbatim.
  - k-quant encoders    `ggml/src/ggml-quants.c` @ b10451, ported function by
                        function with the line anchors on each one.

The in-tree precedent for the per-layer spellings is
`scripts/check-dsv4-gguf-namemap.py`, which already carries `hc_attn_fn.weight`,
`ffn_gate_exps.weight` and the DeepSeek-V4 MLA names.

**The arch string `glm5next` is OURS.** llama.cpp has no `glm5_next` enumerator
at any revision and its PR #27752 is open, so no upstream spelling exists to
mirror. If llama.cpp lands a different string, this converter and the loader
reconcile onto it in the same change.

Self-contained on purpose: numpy is the only import beyond the standard library,
matching `scripts/gen-minimax-h3-gguf-manifest.py` and
`scripts/gen-gguf-nvfp4-goldens.py`. gguf-py is deliberately not used -- it has
no `glm5_next` and, more to the point, `gguf.quants.Q2_K` implements
`dequantize_blocks` and **no** `quantize_blocks`, so upstream Python cannot
produce a Q2_K artifact at all.

Usage:

    scripts/convert-glm5-next-gguf.py --src <checkpoint-dir> --dst out.gguf \\
        --arm q2_k [--keep-mtp] [--no-vision] [--dry-run]

`--dry-run` reads only the safetensors headers and prints the plan and the exact
output size, which is how an arm's arithmetic is checked before 100+ GiB of
writing starts.
"""

import argparse
import json
import os
import struct
import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover - reported, never guessed around
    sys.stderr.write(
        "convert-glm5-next-gguf: numpy is required and is not importable.\n")
    raise SystemExit(4)

f32 = np.float32

# ---------------------------------------------------------------------------
# ggml type ids. Only the ids this converter can EMIT are named. Every one of
# them has both traits and a dequant path in our own reader -- traits alone are
# not enough, and `gguf_dequant.cpp:89-160` is the list that decides:
# IQ4_XS (23) and IQ1_XXXS (21) have traits and NO dequant, which is why no arm
# below reaches for them.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q6_K = 14
GGML_BF16 = 30

TYPE_NAME = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q8_0: "Q8_0",
             GGML_Q2_K: "Q2_K", GGML_Q6_K: "Q6_K", GGML_BF16: "BF16"}
# (block elements, bytes per block)
TYPE_BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2),
              GGML_Q8_0: (32, 34), GGML_Q2_K: (256, 84), GGML_Q6_K: (256, 210)}

QK_K = 256

# GGUF metadata value type ids (gguf-py/gguf/constants.py `GGUFValueType`).
GV_UINT8, GV_INT8, GV_UINT16, GV_INT16, GV_UINT32, GV_INT32 = 0, 1, 2, 3, 4, 5
GV_FLOAT32, GV_BOOL, GV_STRING, GV_ARRAY, GV_UINT64 = 6, 7, 8, 9, 10
GV_INT64, GV_FLOAT64 = 11, 12

ARCH = "glm5next"
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32


# ---------------------------------------------------------------------------
# k-quant encoders, ported from llama.cpp `ggml/src/ggml-quants.c` @ b10451.
#
# These are ported rather than called because no upstream Python can produce
# them (see the module docstring) and because building llama.cpp is not a
# dependency this converter may take. They are gated byte-for-byte against the
# pinned C reference by `tests/scripts/test_convert_glm5_next_gguf.py` over a
# frozen golden captured from that reference; a divergence in the last ulp shows
# up as a byte difference, not as a tolerance argument.
#
# Two traps the port had to reproduce exactly, both of which changed bytes:
#   - `nearest_int` is NOT `round`. It is the 12582912.0 add-and-mask trick at
#     :621, which rounds half to EVEN.
#   - C `roundf` in `quantize_row_q8_0_ref` rounds half AWAY FROM ZERO, where
#     `np.rint` rounds half to even. Using `np.rint` mis-encodes every exact .5.
# Float32 accumulation order is mirrored lane by lane rather than handed to
# `ndarray.sum`, whose pairwise summation is a different order.

def _nearest_int(v):
    """`ggml-quants.c:621` @ b10451, vectorized."""
    v = np.asarray(v, dtype=f32) + f32(12582912.0)
    i = v.view(np.int32)
    return (i & 0x007FFFFF) - 0x00400000


def _fp16(x):
    return np.asarray(x, dtype=f32).astype(np.float16)


GROUP_MAX_EPS = f32(1e-15)


def _make_qkx2_quants(x, nmax=3, rmin=f32(-0.5), rdelta=f32(0.1), nstep=15):
    """`ggml-quants.c:799` @ b10451, vectorized over rows of `x` (M, n).

    `min` is state that UPDATES inside the search loop and the next step's
    `nearest_int(iscale*(x - min))` reads the updated value, so the 16 steps stay
    sequential; only the rows are vectorized.
    """
    M, n = x.shape
    w = np.abs(x).astype(f32)
    mn = x.min(axis=1).astype(f32)
    mx = x.max(axis=1).astype(f32)
    sum_w = w[:, 0].copy()
    sum_x = (sum_w * x[:, 0]).astype(f32)
    for i in range(1, n):
        sum_w = (sum_w + w[:, i]).astype(f32)
        sum_x = (sum_x + (w[:, i] * x[:, i]).astype(f32)).astype(f32)
    mn = np.where(mn > f32(0), f32(0), mn).astype(f32)
    degen = mx == mn
    span = (mx - mn).astype(f32)
    safe = np.where(degen, f32(1), span).astype(f32)
    iscale = (f32(nmax) / safe).astype(f32)
    scale = (f32(1) / iscale).astype(f32)
    L = np.clip(_nearest_int((iscale[:, None] * (x - mn[:, None]).astype(f32)).astype(f32)),
                0, nmax).astype(np.uint8)
    best = np.zeros(M, dtype=f32)
    for i in range(n):
        diff = np.abs((scale * L[:, i].astype(f32) + mn - x[:, i]).astype(f32)).astype(f32)
        best = (best + (w[:, i] * diff).astype(f32)).astype(f32)
    for istep in range(nstep + 1):
        denom = np.where(degen, f32(1), (mx - mn).astype(f32)).astype(f32)
        isc = ((rmin + rdelta * f32(istep) + f32(nmax)) / denom).astype(f32)
        isc = np.where(degen, f32(0), isc).astype(f32)
        Laux = np.clip(_nearest_int((isc[:, None] * (x - mn[:, None]).astype(f32)).astype(f32)),
                       0, nmax).astype(np.uint8)
        sum_l = np.zeros(M, dtype=f32)
        sum_l2 = np.zeros(M, dtype=f32)
        sum_xl = np.zeros(M, dtype=f32)
        for i in range(n):
            li = Laux[:, i].astype(f32)
            wl = (w[:, i] * li).astype(f32)
            sum_l = (sum_l + wl).astype(f32)
            sum_l2 = (sum_l2 + (wl * li).astype(f32)).astype(f32)
            sum_xl = (sum_xl + (wl * x[:, i]).astype(f32)).astype(f32)
        D = ((sum_w * sum_l2).astype(f32) - (sum_l * sum_l).astype(f32)).astype(f32)
        okD = D > f32(0)
        Dsafe = np.where(okD, D, f32(1)).astype(f32)
        this_scale = ((((sum_w * sum_xl).astype(f32) - (sum_x * sum_l).astype(f32)).astype(f32))
                      / Dsafe).astype(f32)
        this_min = ((((sum_l2 * sum_x).astype(f32) - (sum_l * sum_xl).astype(f32)).astype(f32))
                    / Dsafe).astype(f32)
        pos = this_min > f32(0)
        l2safe = np.where(sum_l2 == f32(0), f32(1), sum_l2).astype(f32)
        this_scale = np.where(pos, (sum_xl / l2safe).astype(f32), this_scale).astype(f32)
        this_min = np.where(pos, f32(0), this_min).astype(f32)
        cur = np.zeros(M, dtype=f32)
        for i in range(n):
            diff = np.abs(((this_scale * Laux[:, i].astype(f32)).astype(f32)
                           + this_min - x[:, i]).astype(f32)).astype(f32)
            cur = (cur + (w[:, i] * diff).astype(f32)).astype(f32)
        upd = okD & (cur < best)
        L = np.where(upd[:, None], Laux, L)
        best = np.where(upd, cur, best).astype(f32)
        scale = np.where(upd, this_scale, scale).astype(f32)
        mn = np.where(upd, this_min, mn).astype(f32)
    scale = np.where(degen, f32(0), scale).astype(f32)
    L = np.where(degen[:, None], np.uint8(0), L)
    return scale, (-mn).astype(f32), L


def quantize_q2_k(x):
    """(nb, 256) f32 -> (nb, 84) u8. `ggml-quants.c:891` @ b10451."""
    x = np.ascontiguousarray(x, dtype=f32)
    nb = x.shape[0]
    scales, mins, L = _make_qkx2_quants(x.reshape(nb * 16, 16))
    scales = scales.reshape(nb, 16)
    mins = mins.reshape(nb, 16)
    L = L.reshape(nb, 256)
    q4scale = f32(15.0)
    max_scale = np.maximum(scales.max(axis=1), f32(0)).astype(f32)
    max_min = np.maximum(mins.max(axis=1), f32(0)).astype(f32)
    pos_s = max_scale > f32(0)
    isc = np.where(pos_s, (q4scale / np.where(pos_s, max_scale, f32(1))).astype(f32),
                   f32(0)).astype(f32)
    out_scales = np.where(pos_s[:, None],
                          _nearest_int((isc[:, None] * scales).astype(f32)).astype(np.uint8),
                          np.uint8(0))
    d = np.where(pos_s, (max_scale / q4scale).astype(f32), f32(0)).astype(f32)
    pos_m = max_min > f32(0)
    iscm = np.where(pos_m, (q4scale / np.where(pos_m, max_min, f32(1))).astype(f32),
                    f32(0)).astype(f32)
    lm = _nearest_int((iscm[:, None] * mins).astype(f32)).astype(np.uint8)
    out_scales = np.where(pos_m[:, None], out_scales | (lm << 4), out_scales)
    dmin = np.where(pos_m, (max_min / q4scale).astype(f32), f32(0)).astype(f32)
    dh = _fp16(d)
    dminh = _fp16(dmin)
    dl = (dh.astype(f32)[:, None] * (out_scales & 0xF).astype(f32)).astype(f32)
    dm = (dminh.astype(f32)[:, None] * (out_scales >> 4).astype(f32)).astype(f32)
    dl_e = np.repeat(dl, 16, axis=1)
    dm_e = np.repeat(dm, 16, axis=1)
    nz = dl_e != f32(0)
    num = (x + dm_e).astype(f32)
    Lq = np.clip(_nearest_int(np.divide(num, np.where(nz, dl_e, f32(1))).astype(f32)),
                 0, 3).astype(np.uint8)
    L = np.where(nz, Lq, L)
    qs = np.zeros((nb, 64), dtype=np.uint8)
    for j in range(0, 256, 128):
        base = j // 4
        seg = L[:, j:j + 128]
        qs[:, base:base + 32] = (seg[:, 0:32] | (seg[:, 32:64] << 2)
                                 | (seg[:, 64:96] << 4) | (seg[:, 96:128] << 6))
    out = np.zeros((nb, 84), dtype=np.uint8)
    out[:, 0:16] = out_scales
    out[:, 16:80] = qs
    out[:, 80:82] = dh.view(np.uint8).reshape(nb, 2)
    out[:, 82:84] = dminh.view(np.uint8).reshape(nb, 2)
    return out


def _make_qx_quants(x, nmax=32):
    """`ggml-quants.c:628` @ b10451 with `rmse_type=1`, `qw=NULL`."""
    M, n = x.shape
    ax = np.abs(x).astype(f32)
    idx = np.argmax(ax, axis=1)
    rows = np.arange(M)
    amax = ax[rows, idx]
    mx = x[rows, idx].astype(f32)
    allzero = amax < GROUP_MAX_EPS
    mxs = np.where(allzero, f32(1), mx).astype(f32)
    w = (x * x).astype(f32)
    iscale = (f32(-nmax) / mxs).astype(f32)
    lq = np.clip(_nearest_int((iscale[:, None] * x).astype(f32)), -nmax, nmax - 1).astype(np.int32)
    L = (lq + nmax).astype(np.uint8)
    sumlx = np.zeros(M, dtype=f32)
    suml2 = np.zeros(M, dtype=f32)
    for i in range(n):
        li = lq[:, i].astype(f32)
        sumlx = (sumlx + ((w[:, i] * x[:, i]).astype(f32) * li).astype(f32)).astype(f32)
        suml2 = (suml2 + ((w[:, i] * li).astype(f32) * li).astype(f32)).astype(f32)
    l2safe = np.where(suml2 == f32(0), f32(1), suml2).astype(f32)
    scale = np.where(suml2 != f32(0), (sumlx / l2safe).astype(f32), f32(0)).astype(f32)
    best = (scale * sumlx).astype(f32)
    for istep in list(range(-9, 0)) + list(range(1, 10)):
        isc = ((-(f32(nmax) + (f32(0.1) * f32(istep)).astype(f32)).astype(f32)) / mxs).astype(f32)
        lq2 = np.clip(_nearest_int((isc[:, None] * x).astype(f32)),
                      -nmax, nmax - 1).astype(np.int32)
        sx = np.zeros(M, dtype=f32)
        s2 = np.zeros(M, dtype=f32)
        for i in range(n):
            li = lq2[:, i].astype(f32)
            sx = (sx + ((w[:, i] * x[:, i]).astype(f32) * li).astype(f32)).astype(f32)
            s2 = (s2 + ((w[:, i] * li).astype(f32) * li).astype(f32)).astype(f32)
        upd = (s2 > f32(0)) & ((sx * sx).astype(f32) > (best * s2).astype(f32))
        L = np.where(upd[:, None], (lq2 + nmax).astype(np.uint8), L)
        news = (sx / np.where(s2 == f32(0), f32(1), s2)).astype(f32)
        best = np.where(upd, (news * sx).astype(f32), best).astype(f32)
        scale = np.where(upd, news, scale).astype(f32)
    scale = np.where(allzero, f32(0), scale).astype(f32)
    L = np.where(allzero[:, None], np.uint8(0), L)
    return scale, L


def quantize_q6_k(x):
    """(nb, 256) f32 -> (nb, 210) u8. `ggml-quants.c:1869` @ b10451."""
    x = np.ascontiguousarray(x, dtype=f32)
    nb = x.shape[0]
    scales, L8 = _make_qx_quants(x.reshape(nb * 16, 16))
    scales = scales.reshape(nb, 16)
    L = L8.reshape(nb, 256).astype(np.int32)
    absc = np.abs(scales).astype(f32)
    j = np.argmax(absc, axis=1)
    rows = np.arange(nb)
    dead = absc[rows, j] < GROUP_MAX_EPS
    ms = np.where(dead, f32(1), scales[rows, j].astype(f32)).astype(f32)
    iscale = (f32(-128.0) / ms).astype(f32)
    dh = _fp16((f32(1) / iscale).astype(f32))
    sc = np.minimum(_nearest_int((iscale[:, None] * scales).astype(f32)), 127).astype(np.int8)
    dl = (dh.astype(f32)[:, None] * sc.astype(f32)).astype(f32)
    dl_e = np.repeat(dl, 16, axis=1)
    nz = dl_e != f32(0)
    Lq = np.clip(_nearest_int(np.divide(x, np.where(nz, dl_e, f32(1))).astype(f32)),
                 -32, 31) + 32
    L = np.where(nz, Lq.astype(np.int32), L)
    ql = np.zeros((nb, 128), dtype=np.uint8)
    qh = np.zeros((nb, 64), dtype=np.uint8)
    for jj in range(0, 256, 128):
        o_ql = (jj // 128) * 64
        o_qh = (jj // 128) * 32
        s = L[:, jj:jj + 128].astype(np.uint8)
        ql[:, o_ql:o_ql + 32] = (s[:, 0:32] & 0xF) | ((s[:, 64:96] & 0xF) << 4)
        ql[:, o_ql + 32:o_ql + 64] = (s[:, 32:64] & 0xF) | ((s[:, 96:128] & 0xF) << 4)
        qh[:, o_qh:o_qh + 32] = ((s[:, 0:32] >> 4) | ((s[:, 32:64] >> 4) << 2)
                                 | ((s[:, 64:96] >> 4) << 4) | ((s[:, 96:128] >> 4) << 6))
    out = np.zeros((nb, 210), dtype=np.uint8)
    out[:, 0:128] = ql
    out[:, 128:192] = qh
    out[:, 192:208] = sc.view(np.uint8)
    out[:, 208:210] = dh.view(np.uint8).reshape(nb, 2)
    if dead.any():
        out[dead] = 0
    return out


def quantize_q8_0(x):
    """(nb, 32) f32 -> (nb, 34) u8. `ggml-quants.c:276` @ b10451."""
    x = np.ascontiguousarray(x, dtype=f32)
    nb = x.shape[0]
    amax = np.abs(x).max(axis=1).astype(f32)
    d = (amax / f32(127.0)).astype(f32)
    idv = np.where(d != f32(0), (f32(1.0) / np.where(d == f32(0), f32(1), d)).astype(f32),
                   f32(0)).astype(f32)
    dh = _fp16(d)
    v = (x * idv[:, None]).astype(f32)
    # C `roundf` rounds half AWAY FROM ZERO; `np.rint` rounds half to EVEN.
    qs = np.where(v >= f32(0), np.floor((v + f32(0.5)).astype(f32)),
                  np.ceil((v - f32(0.5)).astype(f32))).astype(np.int8)
    out = np.zeros((nb, 34), dtype=np.uint8)
    out[:, 0:2] = dh.view(np.uint8).reshape(nb, 2)
    out[:, 2:34] = qs.view(np.uint8)
    return out


def encode(rows, ggml_type):
    """Encode a 2-D f32 array (rows x ne0) into the ggml block layout."""
    rows = np.ascontiguousarray(rows, dtype=f32)
    if ggml_type == GGML_F32:
        return rows.tobytes()
    if ggml_type == GGML_F16:
        return rows.astype(np.float16).tobytes()
    if ggml_type == GGML_BF16:
        u = rows.view(np.uint32)
        # round-to-nearest-even bf16, the same rounding the loader's reader
        # expects; a plain truncation biases every weight toward zero.
        rounded = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))
                   >> np.uint32(16)).astype(np.uint16)
        return rounded.tobytes()
    n_elem, _ = TYPE_BLOCK[ggml_type]
    flat = rows.reshape(-1, n_elem)
    if ggml_type == GGML_Q8_0:
        return quantize_q8_0(flat).tobytes()
    if ggml_type == GGML_Q2_K:
        return quantize_q2_k(flat).tobytes()
    if ggml_type == GGML_Q6_K:
        return quantize_q6_k(flat).tobytes()
    raise ValueError("no encoder for ggml type %d" % ggml_type)


def type_nbytes(shape, ggml_type):
    n = 1
    for d in shape:
        n *= d
    n_elem, blk_bytes = TYPE_BLOCK[ggml_type]
    if n % n_elem:
        raise ValueError("element count %d is not a multiple of the %s block %d"
                         % (n, TYPE_NAME[ggml_type], n_elem))
    return (n // n_elem) * blk_bytes


# ---------------------------------------------------------------------------
# safetensors reading. Header only for the plan pass; mmap slices for the data
# pass, so the peak resident set is one tensor rather than one shard.

_ST_DTYPE = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "BF16": np.uint16, "F8_E4M3": np.uint8, "F8_E5M2": np.uint8,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


class Safetensors:
    """Lazy multi-shard reader: headers up front, payload on demand."""

    def __init__(self, src):
        self.src = src
        index = os.path.join(src, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as fh:
                self.weight_map = json.load(fh)["weight_map"]
        else:
            single = os.path.join(src, "model.safetensors")
            if not os.path.exists(single):
                raise SystemExit(
                    "convert-glm5-next-gguf: %s has neither "
                    "model.safetensors.index.json nor model.safetensors" % src)
            self.weight_map = None
        self._headers = {}
        self._maps = {}
        if self.weight_map is None:
            self.weight_map = {k: "model.safetensors"
                               for k in self._header("model.safetensors")
                               if k != "__metadata__"}

    def _header(self, shard):
        if shard not in self._headers:
            path = os.path.join(self.src, shard)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._headers[shard] = json.loads(fh.read(n))
        return self._headers[shard]

    def info(self, name):
        shard = self.weight_map[name]
        return self._header(shard)[name]

    def has(self, name):
        return name in self.weight_map

    def dtype(self, name):
        return self.info(name)["dtype"]

    def shape(self, name):
        return list(self.info(name)["shape"])

    def _mmap(self, shard):
        if shard not in self._maps:
            path = os.path.join(self.src, shard)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
            self._maps[shard] = (np.memmap(path, dtype=np.uint8, mode="r"), 8 + n)
        return self._maps[shard]

    def raw(self, name):
        shard = self.weight_map[name]
        info = self._header(shard)[name]
        mm, base = self._mmap(shard)
        lo, hi = info["data_offsets"]
        buf = mm[base + lo:base + hi]
        dt = info["dtype"]
        if dt not in _ST_DTYPE:
            raise SystemExit("convert-glm5-next-gguf: unhandled safetensors dtype "
                             "%r on tensor %s" % (dt, name))
        return np.frombuffer(buf.tobytes(), dtype=_ST_DTYPE[dt]).reshape(info["shape"]), dt


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


_E4M3_TABLE = None


def _e4m3_table():
    """256-entry FP8 e4m3 (OCP, finite, no inf) decode table.

    e4m3 is 1 sign / 4 exponent / 3 mantissa, bias 7, with subnormals and with
    `S1111111` reserved for NaN -- `S1111110` is the finite max, 448. This is
    the `fmt: "e4m3"` the checkpoint's `quantization_config` declares.
    """
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        out = np.zeros(256, dtype=np.float32)
        for b in range(256):
            s = -1.0 if (b >> 7) else 1.0
            e = (b >> 3) & 0xF
            m = b & 0x7
            if e == 0xF and m == 0x7:
                out[b] = np.float32("nan")
            elif e == 0:
                out[b] = np.float32(s * (m / 8.0) * (2.0 ** -6))
            else:
                out[b] = np.float32(s * (1.0 + m / 8.0) * (2.0 ** (e - 7)))
        _E4M3_TABLE = out
    return _E4M3_TABLE


def dequant_fp8_block(q_u8, scale_inv, block):
    """FP8 e4m3 with a `weight_scale_inv` block grid -> f32.

    The checkpoint stores `[128, 128]` block scales and `w = q * scale_inv`,
    which is the DeepSeek/vLLM block-wise FP8 convention this repository already
    carries. Trailing partial blocks take the last block's scale, which is what
    the ceil-division grid in the shard headers encodes (a `[2048, 4096]` expert
    has a `[16, 32]` scale grid).
    """
    q = _e4m3_table()[q_u8]
    rows, cols = q.shape
    br, bc = block
    sr, sc = scale_inv.shape
    ri = np.minimum(np.arange(rows) // br, sr - 1)
    ci = np.minimum(np.arange(cols) // bc, sc - 1)
    return (q * scale_inv[np.ix_(ri, ci)].astype(f32)).astype(f32)


# ---------------------------------------------------------------------------
# GGUF writing.

def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


_SCALAR_FMT = {GV_UINT8: "<B", GV_INT8: "<b", GV_UINT16: "<H", GV_INT16: "<h",
               GV_UINT32: "<I", GV_INT32: "<i", GV_FLOAT32: "<f", GV_BOOL: "<?",
               GV_UINT64: "<Q", GV_INT64: "<q", GV_FLOAT64: "<d"}


def _pack_value(vtype, value):
    if vtype == GV_STRING:
        return _pack_str(value)
    if vtype == GV_ARRAY:
        etype, items = value
        out = struct.pack("<IQ", etype, len(items))
        for it in items:
            out += _pack_value(etype, it)
        return out
    return struct.pack(_SCALAR_FMT[vtype], value)


class GgufWriter:
    def __init__(self):
        self.kv = []
        self.tensors = []  # (name, dims_ggml, ggml_type, nbytes)

    def add(self, key, vtype, value):
        self.kv.append((key, vtype, value))

    def add_tensor(self, name, dims_ggml, ggml_type, nbytes):
        self.tensors.append((name, list(dims_ggml), ggml_type, nbytes))

    def header_bytes(self):
        out = bytearray()
        out += GGUF_MAGIC
        out += struct.pack("<I", GGUF_VERSION)
        out += struct.pack("<QQ", len(self.tensors), len(self.kv))
        for key, vtype, value in self.kv:
            out += _pack_str(key)
            out += struct.pack("<I", vtype)
            out += _pack_value(vtype, value)
        offset = 0
        for name, dims, ttype, nbytes in self.tensors:
            out += _pack_str(name)
            out += struct.pack("<I", len(dims))
            for d in dims:
                out += struct.pack("<Q", d)
            out += struct.pack("<I", ttype)
            out += struct.pack("<Q", offset)
            offset += nbytes
            pad = (-offset) % GGUF_ALIGNMENT
            offset += pad
        head = len(out)
        out += b"\x00" * ((-head) % GGUF_ALIGNMENT)
        return bytes(out), offset


# ---------------------------------------------------------------------------
# The arms.
#
# Bits per weight are our own reader's block traits (`gguf_reader.cpp:200`), so
# they are exact for the type rather than quoted: Q2_K 84B/256 = 2.625,
# Q6_K 210/256 = 6.5625, Q8_0 34/32 = 8.5.
#
# A mixed arm names the type the ROUTED AND SHARED EXPERTS get. Everything else
# rides at `other`. On GLM-5.3-Flash the experts are 97.0% of the parameters, so
# the arm's mixed bpw is the expert type's bpw to within a percent, and `other`
# buys quality almost for free.

# `llama_ftype` values, `include/llama.h:117-149` @ b10451.
FTYPE = {"f32": 0, "f16": 1, "q8_0": 7, "q2_k": 10, "q6_k": 18, "bf16": 32}

ARMS = {
    "f32":   {"experts": GGML_F32,  "other": GGML_F32},
    "f16":   {"experts": GGML_F16,  "other": GGML_F16},
    "bf16":  {"experts": GGML_BF16, "other": GGML_BF16},
    "q8_0":  {"experts": GGML_Q8_0, "other": GGML_Q8_0},
    "q6_k":  {"experts": GGML_Q6_K, "other": GGML_Q6_K},
    "q2_k":  {"experts": GGML_Q2_K, "other": GGML_Q6_K},
}

# Arms this converter REFUSES, each with the part that is missing. AGENTS.md
# requires an unimplemented arm to be refused by name rather than discovered
# later, and the i-quant refusal is a boundary rather than a to-do: an
# importance matrix requires a forward pass over the model, a forward pass
# requires 181 GiB, and the smallest published artifact is 181.32 GiB. The
# dependency is circular on this fleet (spec `## Risks` R4, `## Owed` O5).
REFUSED_ARMS = {
    "iq1_s": "IQ1_S", "iq2_xxs": "IQ2_XXS", "iq2_s": "IQ2_S",
    "iq3_xxs": "IQ3_XXS", "iq4_xs": "IQ4_XS", "iq1_xxxs": "IQ1_XXXS",
    "q3_k": "Q3_K", "q4_k": "Q4_K", "q5_k": "Q5_K",
}


def refuse_arm(arm):
    name = REFUSED_ARMS[arm]
    if name.startswith("IQ"):
        return (
            "convert-glm5-next-gguf: the %s arm is REFUSED. Missing part: an "
            "importance matrix. Every i-quant encoder needs one, an imatrix "
            "needs a forward pass over the model, and the smallest published "
            "GLM-5.3-Flash artifact is 181.32 GiB against ~119.63 GiB on the "
            "largest device this project reaches. The dependency is circular on "
            "this fleet, so no i-quant arm is producible here; k-quants need no "
            "imatrix. Owed as O5 in .agents/specs/glm5-next-flash.md. "
            "Producible arms: %s." % (name, ", ".join(sorted(ARMS))))
    return (
        "convert-glm5-next-gguf: the %s arm is REFUSED. Missing part: the %s "
        "encoder is not ported. Only Q8_0, Q6_K and Q2_K are ported from the "
        "pinned llama.cpp reference (ggml/src/ggml-quants.c @ b10451) and gated "
        "byte-for-byte against it; emitting %s would mean shipping an unchecked "
        "encoder. Owed as O8 in .agents/specs/glm5-next-flash.md. "
        "Producible arms: %s." % (name, name, name, ", ".join(sorted(ARMS))))


# ---------------------------------------------------------------------------
# Tensor name map: HuggingFace module path -> GGUF tensor name.
#
# Spellings are llama.cpp's at b10451 (`src/llama-arch.cpp`, anchors in the
# module docstring); the HF side is what the real checkpoint's
# `model.safetensors.index.json` carries, read live 2026-08-26 over 76,108
# tensors. `model.language_model.` is stripped because `Glm5NextForConditional
# Generation` nests the text model under the multimodal wrapper and the GGUF
# block namespace is flat.

KDA_MAP = {
    "self_attn.q_proj.weight":   "attn_q.weight",
    "self_attn.k_proj.weight":   "attn_k.weight",
    "self_attn.v_proj.weight":   "attn_v.weight",
    "self_attn.o_proj.weight":   "attn_output.weight",
    "self_attn.q_conv1d.weight": "ssm_conv1d_q.weight",
    "self_attn.k_conv1d.weight": "ssm_conv1d_k.weight",
    "self_attn.v_conv1d.weight": "ssm_conv1d_v.weight",
    "self_attn.f_a_proj.weight": "ssm_f_a.weight",
    "self_attn.f_b_proj.weight": "ssm_f_b.weight",
    "self_attn.g_a_proj.weight": "ssm_g_a.weight",
    "self_attn.g_b_proj.weight": "ssm_g_b.weight",
    "self_attn.b_proj.weight":   "ssm_beta.weight",
    "self_attn.A_log":           "ssm_a",
    "self_attn.dt_bias":         "ssm_dt",
    "self_attn.o_norm.weight":   "ssm_norm.weight",
}

DSA_MAP = {
    "self_attn.q_a_proj.weight":            "attn_q_a.weight",
    "self_attn.q_a_layernorm.weight":       "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight":            "attn_q_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight":  "attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight":      "attn_kv_a_norm.weight",
    "self_attn.kv_b_proj.weight":           "attn_kv_b.weight",
    "self_attn.o_proj.weight":              "attn_output.weight",
    "self_attn.indexer.wq_b.weight":        "indexer.attn_q_b.weight",
    "self_attn.indexer.wk.weight":          "indexer.attn_k.weight",
    "self_attn.indexer.k_norm.weight":      "indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias":        "indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "self_attn.indexer.index_kpool_compress_ape":  "indexer_compressor_ape.weight",
    "self_attn.indexer.index_kpool_compress_gate": "indexer_compressor_gate.weight",
}

COMMON_MAP = {
    "input_layernorm.weight":          "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # mHC lives FLAT on the layer in the checkpoint, not under `attn_hc.*`.
    # `hc_head.*` does not exist at any layer, which is what settles the
    # unweighted-mean head collapse; nothing here allocates one.
    "hc_attn_fn":    "hc_attn_fn.weight",
    "hc_attn_base":  "hc_attn_base.weight",
    "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn":     "hc_ffn_fn.weight",
    "hc_ffn_base":   "hc_ffn_base.weight",
    "hc_ffn_scale":  "hc_ffn_scale.weight",
}

DENSE_MLP_MAP = {
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight":   "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

SPARSE_MLP_MAP = {
    "mlp.gate.weight":                    "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias":   "exp_probs_b.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.up_proj.weight":   "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
}

# Per-expert tensors are STACKED: the checkpoint stores
# `mlp.experts.{0..287}.{gate,up,down}_proj.weight` and the GGUF carries one
# 3-D tensor per projection, which is llama.cpp's `ffn_*_exps` convention and
# our own reader's `kStackedExpertWeight` role.
EXPERT_STACK = {
    "gate_proj": "ffn_gate_exps.weight",
    "up_proj":   "ffn_up_exps.weight",
    "down_proj": "ffn_down_exps.weight",
}

VISION_MAP = {
    "patch_embed.proj.weight": "v.patch_embd.weight",
    "patch_embed.proj.bias":   "v.patch_embd.bias",
    "post_layernorm.weight":   "v.post_ln.weight",
    "downsample.weight":       "v.downsample.weight",
    "downsample.bias":         "v.downsample.bias",
    "merger.proj.weight":              "v.merger.proj.weight",
    "merger.gate_proj.weight":         "v.merger.gate.weight",
    "merger.up_proj.weight":           "v.merger.up.weight",
    "merger.down_proj.weight":         "v.merger.down.weight",
    "merger.post_projection_norm.weight": "v.merger.norm.weight",
    "merger.post_projection_norm.bias":   "v.merger.norm.bias",
}

VISION_BLOCK_MAP = {
    "norm1.weight":         "ln1.weight",
    "norm2.weight":         "ln2.weight",
    "attn.qkv.weight":      "attn_qkv.weight",
    "attn.qkv.bias":        "attn_qkv.bias",
    "attn.proj.weight":     "attn_out.weight",
    "attn.proj.bias":       "attn_out.bias",
    "attn.q_norm.weight":   "attn_q_norm.weight",
    "attn.k_norm.weight":   "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.gate_proj.bias":   "ffn_gate.bias",
    "mlp.up_proj.weight":   "ffn_up.weight",
    "mlp.up_proj.bias":     "ffn_up.bias",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.down_proj.bias":   "ffn_down.bias",
}


# ---------------------------------------------------------------------------

def pick_type(gguf_name, shape, arm):
    """Resolve the ggml type for one output tensor.

    Two rules, and both are refusals to guess rather than preferences:

      - A 1-D tensor (norm, bias, `A_log`, `dt_bias`, the mHC scalars) stays
        F32. It is a vector, not a GEMM operand; quantizing it saves nothing and
        costs accuracy on every layer.
      - A quantized type needs `ne0 % block == 0`. `ne0` is the ROW length, the
        contraction dimension. Where a k-quant's 256-element super-block does not
        divide the row -- the k-pool positional embedding at 128, the vision
        head norms -- the tensor steps DOWN to Q8_0, whose block is 32, and only
        falls all the way to F32 when even 32 does not divide (the depthwise conv
        kernels at 4) or the tensor is 4-D or 5-D (the patch-embed and downsample
        kernels). Stepping down rather than jumping to F32 is what keeps the arms
        ordered by size; an arm whose fallback is 32 bits can be LARGER than a
        coarser one, which is not a property a size table should have. Nothing is
        reshaped to make a row divide, and every fallback shows in the plan.
    """
    types = ARMS[arm]
    want = types["experts"] if gguf_name.endswith("_exps.weight") else types["other"]
    quantized = (GGML_Q8_0, GGML_Q2_K, GGML_Q6_K)
    if want not in quantized:
        return want
    if len(shape) < 2 or len(shape) > 3:
        return GGML_F32
    ne0 = shape[-1]
    for candidate in (want, GGML_Q8_0):
        n_elem, _ = TYPE_BLOCK[candidate]
        if ne0 % n_elem == 0:
            return candidate
    return GGML_F32


class Plan:
    """One output tensor: where its bytes come from and what they become."""

    def __init__(self, gguf_name, sources, shape, ggml_type, kind):
        self.gguf_name = gguf_name
        self.sources = sources        # list of HF tensor names, stacked in order
        self.shape = shape            # logical shape, numpy/torch order
        self.ggml_type = ggml_type
        self.kind = kind              # "plain" | "stacked"
        self.nbytes = type_nbytes(shape, ggml_type)

    @property
    def dims_ggml(self):
        return list(reversed(self.shape))


def _layer_types(text_cfg):
    lt = text_cfg.get("layer_types")
    if not lt:
        raise SystemExit(
            "convert-glm5-next-gguf: text_config carries no `layer_types`. That "
            "list is the AUTHORITATIVE per-layer schedule for this architecture "
            "-- the reference ignores `linear_attn_config.kda_layers` and "
            "`full_attn_layers` entirely -- so there is nothing to fall back to.")
    return lt


def build_plan(st, cfg, arm, keep_mtp, want_vision):
    text = cfg["text_config"]
    n_layers = int(text["num_hidden_layers"])
    layer_types = _layer_types(text)
    if len(layer_types) != n_layers:
        raise SystemExit("convert-glm5-next-gguf: layer_types has %d entries and "
                         "num_hidden_layers is %d" % (len(layer_types), n_layers))
    mlp_types = text.get("mlp_layer_types") or []
    n_experts = int(text["n_routed_experts"])
    prefix = "model.language_model."
    plans = []
    skipped = []

    def add(gguf_name, hf_names, shape, kind="plain"):
        t = pick_type(gguf_name, shape, arm)
        plans.append(Plan(gguf_name, hf_names, shape, t, kind))

    def logical(name):
        # The FP8 form stores one byte per element, so the safetensors shape IS
        # the logical shape; the `weight_scale_inv` companion is metadata and is
        # never emitted as a tensor of its own.
        #
        # ggml carries at most GGML_MAX_DIMS = 4 axes, and this checkpoint has
        # two tensors that exceed it or sit awkwardly inside it: the vision
        # patch embed is a Conv3d at `[1024, 3, 2, 14, 14]` (five axes, which
        # simply cannot be written) and the downsample is a Conv2d at
        # `[4096, 1024, 2, 2]`. Both are consumed as a matmul over flattened
        # patches, so both are flattened to `[out, prod(rest)]` here -- the
        # im2col form llama.cpp's own vision converters write. Flattening is
        # NOT free and is not applied blindly: it is confined to 4-D and 5-D
        # convolution kernels. The 3-D shapes are left alone, because they mean
        # something in ggml -- the depthwise KDA conv is `[ch, 1, k]` and the
        # stacked expert lane is `[experts, n, m]`, and reshaping either would
        # change how the loader indexes it.
        shape = st.shape(name)
        if len(shape) > 3:
            n = 1
            for d in shape[1:]:
                n *= d
            return [shape[0], n]
        return shape

    add("token_embd.weight", [prefix + "embed_tokens.weight"],
        logical(prefix + "embed_tokens.weight"))
    add("output_norm.weight", [prefix + "norm.weight"], logical(prefix + "norm.weight"))
    if st.has("lm_head.weight"):
        add("output.weight", ["lm_head.weight"], logical("lm_head.weight"))

    for L in range(n_layers):
        base = "%slayers.%d." % (prefix, L)
        table = dict(COMMON_MAP)
        table.update(KDA_MAP if layer_types[L] == "linear_attention" else DSA_MAP)
        for hf_suffix, gg_suffix in sorted(table.items()):
            hf = base + hf_suffix
            if not st.has(hf):
                continue
            add("blk.%d.%s" % (L, gg_suffix), [hf], logical(hf))
        sparse = (mlp_types[L] if L < len(mlp_types)
                  else ("sparse" if st.has(base + "mlp.gate.weight") else "dense"))
        if sparse == "dense":
            for hf_suffix, gg_suffix in sorted(DENSE_MLP_MAP.items()):
                hf = base + hf_suffix
                if st.has(hf):
                    add("blk.%d.%s" % (L, gg_suffix), [hf], logical(hf))
        else:
            for hf_suffix, gg_suffix in sorted(SPARSE_MLP_MAP.items()):
                hf = base + hf_suffix
                if st.has(hf):
                    add("blk.%d.%s" % (L, gg_suffix), [hf], logical(hf))
            for proj, gg_suffix in sorted(EXPERT_STACK.items()):
                names = ["%smlp.experts.%d.%s.weight" % (base, e, proj)
                         for e in range(n_experts)]
                missing = [n for n in names if not st.has(n)]
                if missing:
                    raise SystemExit(
                        "convert-glm5-next-gguf: layer %d is sparse but %d of its "
                        "%d `%s` expert tensors are absent (first: %s)"
                        % (L, len(missing), n_experts, proj, missing[0]))
                per = logical(names[0])
                add("blk.%d.%s" % (L, gg_suffix), names, [n_experts] + per, "stacked")

    # The MTP block. `num_hidden_layers` is 45 and the checkpoint carries 46
    # layer directories; `model.language_model.layers.45.*` is a DeepSeek-V3
    # style multi-token-prediction block that the transformers reference
    # DISCARDS (`_keys_to_ignore_on_load_unexpected` names `layers\.45\.` and
    # `shared_head\.`). We follow the reference and the in-tree
    # `glm4_moe_lite_registry.cpp:21-26` precedent and request only
    # `[0, num_hidden_layers)`. It is 888 tensors and 7.43B parameters -- 2.31%
    # of the model, ~2.3 GiB of a Q2_K arm.
    for name in st.weight_map:
        if name.startswith("%slayers.%d." % (prefix, n_layers)) or ".shared_head." in name:
            skipped.append(name)
    if keep_mtp and skipped:
        raise SystemExit(
            "convert-glm5-next-gguf: --keep-mtp is REFUSED. Missing part: the "
            "multi-token-prediction head is not implemented on the loader side "
            "either, so a GGUF carrying %d layer-%d tensors would be %d tensors "
            "nothing can read. Owed as O2 in .agents/specs/glm5-next-flash.md."
            % (len(skipped), n_layers, len(skipped)))

    vision_plans = []
    if want_vision:
        vcfg = cfg.get("vision_config") or {}
        depth = int(vcfg.get("depth", 0))
        for hf_suffix, gg_name in sorted(VISION_MAP.items()):
            hf = "model.visual." + hf_suffix
            if st.has(hf):
                t = pick_type(gg_name, logical(hf), arm)
                vision_plans.append(Plan(gg_name, [hf], logical(hf), t, "plain"))
        for b in range(depth):
            for hf_suffix, gg_suffix in sorted(VISION_BLOCK_MAP.items()):
                hf = "model.visual.blocks.%d.%s" % (b, hf_suffix)
                if st.has(hf):
                    gg = "v.blk.%d.%s" % (b, gg_suffix)
                    t = pick_type(gg, logical(hf), arm)
                    vision_plans.append(Plan(gg, [hf], logical(hf), t, "plain"))
    plans.extend(vision_plans)
    return plans, skipped


def write_metadata(w, cfg, arm, n_layers, name):
    text = cfg["text_config"]
    vis = cfg.get("vision_config") or {}
    lin = text.get("linear_attn_config") or {}
    layer_types = _layer_types(text)

    def u32(k, v):
        w.add(k, GV_UINT32, int(v))

    def f(k, v):
        w.add(k, GV_FLOAT32, float(v))

    def s(k, v):
        w.add(k, GV_STRING, str(v))

    def b(k, v):
        w.add(k, GV_BOOL, bool(v))

    s("general.architecture", ARCH)
    s("general.type", "model")
    s("general.name", name)
    s("general.basename", "GLM-5.3-Flash")
    s("general.size_label", "%dA%dB" % (
        round(int(text["n_routed_experts"]) * int(text["moe_intermediate_size"])
              * 3 * int(text["hidden_size"]) / 1e9),
        round(int(text["num_experts_per_tok"]) * int(text["moe_intermediate_size"])
              * 3 * int(text["hidden_size"]) / 1e9)))
    # `general.file_type` is a UINT32 enum upstream, not a string:
    # `llama_ftype` in `include/llama.h:117-149` @ b10451. A mixed arm
    # reports the type its BULK carries, which is what "MOSTLY" means
    # there, so the q2_k arm reports Q2_K even though the other 3% is
    # Q6_K. The arm name travels beside it as a string, because "q2_k"
    # names a policy of ours that no upstream enum has a value for.
    u32("general.file_type", FTYPE[arm])
    s("general.quantization_arm", arm)
    u32("general.quantization_version", 2)
    u32("general.alignment", GGUF_ALIGNMENT)

    k = ARCH
    u32("%s.vocab_size" % k, text["vocab_size"])
    u32("%s.context_length" % k, text["max_position_embeddings"])
    u32("%s.embedding_length" % k, text["hidden_size"])
    u32("%s.block_count" % k, n_layers)
    u32("%s.feed_forward_length" % k, text["intermediate_size"])
    u32("%s.expert_feed_forward_length" % k, text["moe_intermediate_size"])
    u32("%s.expert_shared_feed_forward_length" % k, text["moe_intermediate_size"])
    u32("%s.leading_dense_block_count" % k,
        sum(1 for t in (text.get("mlp_layer_types") or []) if t == "dense"))
    u32("%s.expert_count" % k, text["n_routed_experts"])
    u32("%s.expert_used_count" % k, text["num_experts_per_tok"])
    u32("%s.expert_shared_count" % k, text["n_shared_experts"])
    u32("%s.expert_group_count" % k, text["n_group"])
    u32("%s.expert_group_used_count" % k, text["topk_group"])
    f("%s.expert_weights_scale" % k, text["routed_scaling_factor"])
    b("%s.expert_weights_norm" % k, text["norm_topk_prob"])
    # `scoring_func: sigmoid` -> gating func 2, llama.cpp's
    # `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`.
    u32("%s.expert_gating_func" % k, 2 if text["scoring_func"] == "sigmoid" else 1)
    u32("%s.attention.head_count" % k, text["num_attention_heads"])
    u32("%s.attention.head_count_kv" % k, text["num_key_value_heads"])
    f("%s.attention.layer_norm_rms_epsilon" % k, text["rms_norm_eps"])
    u32("%s.attention.q_lora_rank" % k, text["q_lora_rank"])
    u32("%s.attention.kv_lora_rank" % k, text["kv_lora_rank"])
    u32("%s.attention.key_length_mla" % k, text["qk_head_dim"])
    u32("%s.attention.value_length_mla" % k, text["v_head_dim"])
    u32("%s.attention.key_length" % k, text["qk_nope_head_dim"])
    u32("%s.attention.value_length" % k, text["v_head_dim"])
    f("%s.swiglu_clamp_exp" % k, text["swiglu_limit"])
    f("%s.swiglu_clamp_shexp" % k, text["swiglu_limit"])
    u32("%s.nextn_predict_layers" % k, 0)  # the MTP tail is not carried; O2.

    # NO ROPE. `qk_rope_head_dim` is 0 and `mla_use_nope` is true; the reference
    # deletes the inherited rope parameters and passes `position_embeddings=None`
    # to every layer. `indexer_rope_interleave` is a VESTIGIAL config flag the
    # indexer override ignores, so no `{arch}.rope.*` key is written and a reader
    # that finds one here is reading a different file.
    u32("%s.rope.dimension_count" % k, 0)

    # DSA indexer geometry. Keys: llama.cpp b10451
    # `src/llama-arch.cpp:272-277`.
    u32("%s.attention.indexer.head_count" % k, text["index_n_heads"])
    u32("%s.attention.indexer.key_length" % k, text["index_head_dim"])
    u32("%s.attention.indexer.top_k" % k, text["index_topk"])
    w.add("%s.attention.indexer.types" % k, GV_ARRAY,
          (GV_STRING, list(text.get("indexer_types") or [])))
    # The k-pool compression stage has NO upstream key at any revision, because
    # no upstream implements this indexer. These three are OURS, namespaced under
    # the indexer they belong to. `index_kpool` is 4 HERE and the config class
    # default is 16, so a reader that defaults instead of reading is wrong.
    u32("%s.attention.indexer.kpool" % k, text["index_kpool"])
    b("%s.attention.indexer.kpool_compress" % k, text["index_kpool_compress"])
    b("%s.attention.indexer.kpool_always_select_tail" % k,
      text["index_kpool_always_select_tail"])

    # KDA. Keys: llama.cpp b10451 `gguf-py/gguf/constants.py:262-264` /
    # `src/llama-arch.cpp:320-321`. `gate_lower_bound` is the parameter this port
    # hinges on: it is -5.0 here and NOT None, which selects
    # `-bound * sigmoid(exp(A_log) * (g + dt_bias))`, a DIFFERENT function from
    # the `-exp(A_log) * softplus(g + dt_bias)` our Kimi-Linear KDA implements --
    # different formula, and the sign of `decay_rate` differs too. Writing the
    # value into the file is what lets the loader take the right branch instead
    # of inheriting Kimi's.
    u32("%s.kda.head_dim" % k, lin["head_dim"])
    f("%s.kda.gate_lower_bound" % k, lin["gate_lower_bound"])
    u32("%s.attention.linear_head_count" % k, lin["num_heads"])
    u32("%s.ssm.conv_kernel" % k, lin["short_conv_kernel_size"])

    # mHC. Keys: llama.cpp b10451 `HyperConnection` in constants.py.
    # `hc_eps` is a DIFFERENT constant from `rms_norm_eps` (1e-6 vs 1e-5) and is
    # added to every Sinkhorn denominator rather than used as a floor.
    u32("%s.hyper_connection.count" % k, text["hc_mult"])
    u32("%s.hyper_connection.sinkhorn_iterations" % k, text["hc_sinkhorn_iters"])
    f("%s.hyper_connection.epsilon" % k, text["hc_eps"])

    # The per-layer schedule. The top-level `layer_types` list is authoritative;
    # `linear_attn_config.kda_layers` and `full_attn_layers` are IGNORED by the
    # reference and are not carried here, so nothing downstream can read the
    # wrong one.
    w.add("%s.layer_types" % k, GV_ARRAY, (GV_STRING, list(layer_types)))
    if text.get("mlp_layer_types"):
        w.add("%s.mlp_layer_types" % k, GV_ARRAY,
              (GV_STRING, list(text["mlp_layer_types"])))

    # Multimodal. Image and video share ONE token id: the processor emits
    # `image_token_id` for video frames too and disambiguates by the
    # begin/end-of-video SPAN, so all six ids travel together or a reader
    # classifies every frame as an image.
    for key in ("image_token_id", "video_token_id", "image_start_token_id",
                "image_end_token_id", "video_start_token_id", "video_end_token_id"):
        if key in cfg:
            u32("%s.%s" % (k, key), cfg[key])
    if vis:
        u32("%s.vision.block_count" % k, vis["depth"])
        u32("%s.vision.embedding_length" % k, vis["hidden_size"])
        u32("%s.vision.feed_forward_length" % k, vis["intermediate_size"])
        u32("%s.vision.head_count" % k, vis["num_heads"])
        u32("%s.vision.patch_size" % k, vis["patch_size"])
        u32("%s.vision.image_size" % k, vis["image_size"])
        u32("%s.vision.spatial_merge_size" % k, vis["spatial_merge_size"])
        u32("%s.vision.temporal_patch_size" % k, vis["temporal_patch_size"])
        u32("%s.vision.out_embedding_length" % k, vis["out_hidden_size"])
        u32("%s.vision.projection_intermediate_size" % k,
            vis["projection_intermediate_size"])
        f("%s.vision.attention.layer_norm_rms_epsilon" % k, vis["rms_norm_eps"])
        f("%s.vision.swiglu_clamp" % k, vis["swiglu_limit"])


def materialize(st, plan, block):
    """Read a plan's sources and return its rows as f32 (rows x ne0)."""
    ne0 = plan.shape[-1]
    chunks = []
    for name in plan.sources:
        raw, dt = st.raw(name)
        if dt == "F8_E4M3":
            scale_name = name[:-len(".weight")] + ".weight_scale_inv" \
                if name.endswith(".weight") else name + ".weight_scale_inv"
            if not st.has(scale_name):
                raise SystemExit(
                    "convert-glm5-next-gguf: %s is F8_E4M3 and has no "
                    "`weight_scale_inv` companion. A block-quantized FP8 tensor "
                    "without its scales cannot be dequantized, and treating the "
                    "bytes as values would produce a loadable, wrong file."
                    % name)
            scale, _ = st.raw(scale_name)
            vals = dequant_fp8_block(raw, np.asarray(scale, dtype=f32), block)
        elif dt == "BF16":
            vals = _bf16_to_f32(raw)
        else:
            vals = np.asarray(raw, dtype=f32)
        chunks.append(np.ascontiguousarray(vals, dtype=f32).reshape(-1, ne0))
    return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a GLM-5.3-Flash (`glm5_next`) checkpoint to GGUF.")
    ap.add_argument("--src", required=True, help="checkpoint directory")
    ap.add_argument("--dst", help="output .gguf (omit with --dry-run)")
    ap.add_argument("--arm", default="q2_k",
                    help="quantization arm: " + ", ".join(sorted(ARMS))
                         + " (refused: " + ", ".join(sorted(REFUSED_ARMS)) + ")")
    ap.add_argument("--keep-mtp", action="store_true",
                    help="carry the layer-45 MTP block (refused; see O2)")
    ap.add_argument("--no-vision", action="store_true",
                    help="omit the vision tower")
    ap.add_argument("--dry-run", action="store_true",
                    help="read headers only; print the plan and the exact output size")
    ap.add_argument("--name", default="GLM-5.3-Flash", help="general.name")
    args = ap.parse_args(argv)

    arm = args.arm.lower()
    if arm in REFUSED_ARMS:
        sys.stderr.write(refuse_arm(arm) + "\n")
        return 3
    if arm not in ARMS:
        sys.stderr.write("convert-glm5-next-gguf: unknown arm %r. Producible: %s. "
                         "Refused by name: %s.\n"
                         % (args.arm, ", ".join(sorted(ARMS)),
                            ", ".join(sorted(REFUSED_ARMS))))
        return 2
    if not args.dry_run and not args.dst:
        sys.stderr.write("convert-glm5-next-gguf: --dst is required without --dry-run\n")
        return 2

    cfg_path = os.path.join(args.src, "config.json")
    if not os.path.exists(cfg_path):
        sys.stderr.write("convert-glm5-next-gguf: %s has no config.json\n" % args.src)
        return 2
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    arch_list = cfg.get("architectures") or []
    if cfg.get("model_type") != "glm5_next" or \
            "Glm5NextForConditionalGeneration" not in arch_list:
        sys.stderr.write(
            "convert-glm5-next-gguf: this converter handles `model_type: "
            "glm5_next` / `Glm5NextForConditionalGeneration` only. This "
            "checkpoint declares model_type=%r architectures=%r. Converting it "
            "anyway would emit a file whose metadata describes a model the "
            "weights are not.\n" % (cfg.get("model_type"), arch_list))
        return 2

    qcfg = cfg.get("quantization_config") or {}
    block = tuple(qcfg.get("weight_block_size") or (128, 128))
    if qcfg and qcfg.get("quant_method") not in (None, "fp8"):
        sys.stderr.write(
            "convert-glm5-next-gguf: quantization_config.quant_method=%r is not "
            "handled. Missing part: only the published `fp8` e4m3 block form and "
            "unquantized bf16 checkpoints have a dequant path here.\n"
            % qcfg.get("quant_method"))
        return 3
    if qcfg and qcfg.get("fmt") not in (None, "e4m3"):
        sys.stderr.write("convert-glm5-next-gguf: quantization_config.fmt=%r is "
                         "not handled; only e4m3.\n" % qcfg.get("fmt"))
        return 3

    st = Safetensors(args.src)
    text = cfg["text_config"]
    n_layers = int(text["num_hidden_layers"])
    plans, skipped = build_plan(st, cfg, arm, args.keep_mtp, not args.no_vision)

    w = GgufWriter()
    write_metadata(w, cfg, arm, n_layers, args.name)
    for p in plans:
        if len(p.dims_ggml) > 4:
            sys.stderr.write(
                "convert-glm5-next-gguf: %s has %d axes and ggml carries at most "
                "4 (GGML_MAX_DIMS). Writing it would produce a header no reader "
                "can index, so nothing is written.\n"
                % (p.gguf_name, len(p.dims_ggml)))
            return 3
        w.add_tensor(p.gguf_name, p.dims_ggml, p.ggml_type, p.nbytes)
    head, data_bytes = w.header_bytes()

    counts = {}
    for p in plans:
        counts[TYPE_NAME[p.ggml_type]] = counts.get(TYPE_NAME[p.ggml_type], 0) + 1
    total = len(head) + data_bytes
    sys.stderr.write(
        "convert-glm5-next-gguf: arch=%s arm=%s layers=%d tensors=%d "
        "skipped_mtp=%d header=%d data=%d total=%d (%.2f GiB)\n"
        % (ARCH, arm, n_layers, len(plans), len(skipped), len(head), data_bytes,
           total, total / float(1 << 30)))
    sys.stderr.write("convert-glm5-next-gguf: types " + " ".join(
        "%s=%d" % (t, c) for t, c in sorted(counts.items())) + "\n")
    if args.dry_run:
        for p in plans:
            sys.stdout.write("%s\t%s\t%s\t%d\n"
                             % (p.gguf_name, TYPE_NAME[p.ggml_type],
                                ",".join(str(d) for d in p.dims_ggml), p.nbytes))
        return 0

    with open(args.dst, "wb") as out:
        out.write(head)
        for p in plans:
            rows = materialize(st, p, block)
            blob = encode(rows, p.ggml_type)
            if len(blob) != p.nbytes:
                raise SystemExit(
                    "convert-glm5-next-gguf: %s encoded to %d bytes, the header "
                    "reserved %d. The file would be unreadable, so nothing is "
                    "written past this point." % (p.gguf_name, len(blob), p.nbytes))
            out.write(blob)
            pad = (-len(blob)) % GGUF_ALIGNMENT
            if pad:
                out.write(b"\x00" * pad)
    sys.stderr.write("convert-glm5-next-gguf: wrote %s\n" % args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
