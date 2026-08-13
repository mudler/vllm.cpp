#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_quant_goldens.inc — the LTX-2.5 phase-L6 quant oracle.

Two things need gating and they need DIFFERENT oracles, so they get different
sections and their provenance is recorded separately.

1. THE SCALE SWIZZLE (`kLtx2Blocked*`). torchao writes NVFP4 group scales in the
   cuBLAS "block scaling factors layout"
   (https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout).
   The producer is `to_blocked` in
   vllm/model_executor/layers/quantization/qutlass_utils.py:165-180, whose own
   header records that it was copied from
   https://github.com/pytorch/ao/tree/main/torchao/prototype/mx_formats — i.e.
   from the exact torchao module that quantized our checkpoint. vLLM writes the
   same permutation a second time as `swizzle_blockscale`
   (vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:44-49).

   This generator EXECUTES that function. `import vllm.*` dies in
   `vllm.distributed` on a missing `zmq`, but qutlass_utils.py itself imports
   only torch, `vllm.triton_utils` and `vllm.utils.math_utils`, all of which
   load, so `importlib.util.spec_from_file_location` runs the pinned FILE
   directly and `to_blocked(x, backend="torch")` is called for real. The goldens
   below therefore come from the producer upstream itself calls, not from a copy
   of it. `--vllm` pins the checkout, and `check_transcription` additionally
   diffs the body the C++ INVERSE was written against, so a change to the
   permutation is caught as a source change and not only as a golden diff.

   The `backend="triton"` arm is NOT exercised: it dispatches to a Triton kernel
   and this host has no active driver. That arm is a separate implementation of
   the same permutation upstream, and gating it needs a GPU host.

   `swizzle_blockscale` — vLLM's SECOND writing of the same permutation — still
   cannot be executed here, because it calls `.cuda()` unconditionally. It is
   pinned by source fragment only, and that is stated where it is checked.

2. THE REAL CHECKPOINT BYTES (`kLtx2Real*`). These come off the SHIPPED files on
   $CHECKPOINT_ROOT — a few hundred bytes read at their own offsets, never a
   payload download — and the expected values are decoded with TORCH, which is a
   genuinely independent implementation of fp8-e4m3: `torch.uint8 ->
   view(float8_e4m3fn) -> float()`. So the fp8 half of both dequant paths is
   gated against something that is not ours. The e2m1 nibble LUT is not decoded
   by torch (no fp4 dtype); it is the one already gated by the modelopt path in
   tests/vllm/test_nvfp4_dequant.cpp and is reused, not re-derived.

Usage:
    python3 scripts/gen-ltx2-quant-goldens.py \\
        --vllm ~/_git/vllm \\
        --checkpoint-root /mnt/nas_share/checkpoints \\
        --out tests/vllm/models/ltx2_quant_goldens.inc

Needs torch + numpy (CPU only).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Provenance. AGENTS.md wants the upstream revision anchor; L4's decoy
# experiment (spec section 7.0(b)) showed the anchor is worthless unless the
# tree it names is the tree that ran, so a DIRTY checkout is refused outright —
# `git rev-parse HEAD` reports the committed SHA whatever the worktree holds,
# which is precisely how a clean anchor gets stamped onto drifted goldens.
# ---------------------------------------------------------------------------


def pinned_revision(root: Path, label: str, paths: list[str]) -> tuple[str, bool]:
    """(sha, whole_tree_clean). REFUSES when any of `paths` is dirty.

    The anchor is scoped to what was actually read. A dirty file among `paths`
    is fatal — that is exactly the case where `rev-parse` stamps a clean SHA onto
    goldens the committed tree cannot reproduce. A dirty file ELSEWHERE is not
    fatal, but it is recorded in the emitted header, because an anchor that
    quietly implies more than it checked is the same defect one step removed.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001 - a tarball checkout has no git metadata
        raise SystemExit(f"{label}: cannot read a revision from {root}: {exc}") from exc
    dirty_paths = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--"] + paths, text=True
    ).strip()
    if dirty_paths:
        raise SystemExit(
            f"{label}: the sources this generator reads are DIRTY at {sha}.\n"
            "  Refusing: rev-parse would stamp a CLEAN anchor onto goldens produced "
            "by an edited tree, which is the exact failure .agents/specs/ltx-2-5.md "
            "section 7.0(b) records. Commit or stash first.\n"
            f"  {dirty_paths}"
        )
    whole = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    return sha, not whole


# ---------------------------------------------------------------------------
# The transcription, and the pin that keeps it honest
# ---------------------------------------------------------------------------

# The `backend="torch"` body of `to_blocked`, verbatim from
# vllm/model_executor/layers/quantization/qutlass_utils.py:167-180. Nothing here
# is EXECUTED — the goldens come from calling the real function — so this is not
# a second hand-typing that could drift from what runs. It is the text the C++
# inverse in Ltx2UnswizzleNvfp4BlockScale was written against, pinned so that a
# change to the permutation upstream is reported as a source change rather than
# only as a silent golden diff on the next regeneration.
_TO_BLOCKED_SOURCE_ANCHOR = """    rows, cols = input_matrix.shape
    n_row_blocks = cdiv(rows, 128)
    n_col_blocks = cdiv(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    assert (rows, cols) == (padded_rows, padded_cols)

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()"""


# The only vLLM files this generator reads. The revision anchor is scoped to
# exactly these, and they are the ones a dirty tree is refused over.
_PINNED_VLLM_PATHS = [
    "vllm/model_executor/layers/quantization/qutlass_utils.py",
    "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py",
    "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py",
]

# ── THE NIBBLE ORDER, PINNED ON BOTH SIDES ──────────────────────────────────
#
# .agents/specs/nvfp4-nibble-order.md. E2M1 packs two values per byte and the two
# producers we read DISAGREE about which logical element gets which nibble, so
# both conventions are anchored against a pinned, clean checkout. A silent flip
# on either side transposes every adjacent weight pair: finite, correctly shaped,
# and wrong.

# LOW-first, vLLM's reader — and therefore torchao's and ModelOpt's, which it
# reads. Matches torchao's own producer, `pack_uint4`
# (pytorch/ao torchao/prototype/mx_formats/kernels.py:160,
#  `uint8_data[::2] | uint8_data[1::2] << 4`), which is NOT pinnable here: torchao
# is not installed on this host and not vendored. vLLM's reader is the local
# witness, and it is an independent implementation of the same convention.
_VLLM_LOW_NIBBLE_ANCHORS = [
    "high = (a_flat & 0xF0) >> 4  # Upper nibbles",
    "low = a_flat & 0x0F  # Lower nibbles",
    "combined = torch.stack((low, high), dim=1).flatten()",
]

# HIGH-first, Lightricks' own runtime, which is what wrote the first-party
# LTX-2.5 NVFP4 DiT.
_PINNED_LTX2_PATHS = [
    "packages/ltx-kernels/docs/NVFP4.md",
    "packages/ltx-kernels/csrc/nvfp4/quantize.cu",
    "packages/ltx-core/src/ltx_core/quantization/nvfp4/linear.py",
]
_LTX2_HIGH_NIBBLE_ANCHORS = [
    ("packages/ltx-kernels/docs/NVFP4.md",
     "`hi_first=True` (default) puts element `2j` in the **high** nibble of byte `j`;"),
    ("packages/ltx-core/src/ltx_core/quantization/nvfp4/linear.py",
     "element ``2j`` in the high nibble)"),
    ("packages/ltx-core/src/ltx_core/quantization/nvfp4/linear.py",
     "``weight_scale`` — E4M3 block scales as ``uint8``, cuBLAS 128x4 tiled layout"),
    # The permutation itself, which our Ltx2UnswizzleNvfp4BlockScale inverts, and
    # the `padded_cols == roundup(K/16, 4)` framing that the DiT declares.
    ("packages/ltx-kernels/csrc/nvfp4/quantize.cu",
     "return static_cast<int64_t>(tile) * 512 + (r & 31) * 16 + (r >> 5) * 4 + (col & 3);"),
]


def check_transcription(vllm_root: Path) -> None:
    """Fail if the transcription above no longer matches the pinned checkout."""
    path = vllm_root / "vllm/model_executor/layers/quantization/qutlass_utils.py"
    if not path.is_file():
        raise SystemExit(f"not a vLLM checkout: {path} is missing")
    text = path.read_text(encoding="utf-8")
    if _TO_BLOCKED_SOURCE_ANCHOR not in text:
        raise SystemExit(
            f"{path}: the transcribed `to_blocked` body no longer appears verbatim.\n"
            "  The swizzle this port inverts has MOVED. Re-read it and re-transcribe; "
            "do not relax this check."
        )
    # The second writing of the same permutation, pinned so a divergence between
    # vLLM's two producers cannot pass unnoticed either.
    other = vllm_root / "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py"
    if not other.is_file():
        raise SystemExit(f"not a vLLM checkout: {other} is missing")
    otext = other.read_text(encoding="utf-8")
    for fragment in (
        "padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)",
        "swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()",
    ):
        if fragment not in otext:
            raise SystemExit(
                f"{other}: `swizzle_blockscale` no longer contains {fragment!r}; "
                "the two vLLM producers may have diverged. Re-read both."
            )
    # LOW-nibble-first, vLLM's own reader.
    emu = vllm_root / "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
    if not emu.is_file():
        raise SystemExit(f"not a vLLM checkout: {emu} is missing")
    etext = emu.read_text(encoding="utf-8")
    for fragment in _VLLM_LOW_NIBBLE_ANCHORS:
        if fragment not in etext:
            raise SystemExit(
                f"{emu}: `break_fp4_bytes` no longer contains {fragment!r}.\n"
                "  That is the LOW-nibble-first convention DequantNvfp4ToBf16 defaults "
                "to. If it moved, re-read it; do not relax this check."
            )


def check_ltx2_nibble_order(ltx2_root: Path) -> None:
    """Fail if Lightricks' HIGH-first convention is no longer stated where we read it."""
    for rel, fragment in _LTX2_HIGH_NIBBLE_ANCHORS:
        path = ltx2_root / rel
        if not path.is_file():
            raise SystemExit(f"not an LTX-2 checkout: {path} is missing")
        if fragment not in path.read_text(encoding="utf-8"):
            raise SystemExit(
                f"{path}: no longer contains {fragment!r}.\n"
                "  The first-party NVFP4 DiT is read HIGH-nibble-first on the strength of "
                "that statement (.agents/specs/nvfp4-nibble-order.md section 1). Re-read "
                "it; do not relax this check."
            )


def load_upstream_to_blocked(vllm_root: Path):
    """Return vLLM's OWN `to_blocked`, executed from the pinned checkout.

    `import vllm.model_executor...` pulls in `vllm.distributed`, which dies on a
    missing `zmq` on this host. qutlass_utils.py itself needs only torch,
    `vllm.triton_utils` and `vllm.utils.math_utils`, so loading the FILE through
    `spec_from_file_location` runs the real function without importing the
    package graph around it. That is what makes this a RUNNING oracle rather
    than a transcription of one.
    """
    path = vllm_root / "vllm/model_executor/layers/quantization/qutlass_utils.py"
    if not path.is_file():
        raise SystemExit(f"not a vLLM checkout: {path} is missing")
    # `vllm.triton_utils` / `vllm.utils.math_utils` are imported by name from
    # inside the file, so the checkout has to be importable as a package root.
    if str(vllm_root) not in sys.path:
        sys.path.insert(0, str(vllm_root))
    spec = importlib.util.spec_from_file_location("vllm_qutlass_utils_oracle", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    to_blocked = getattr(module, "to_blocked", None)
    if to_blocked is None:
        raise SystemExit(f"{path}: no `to_blocked` to execute; the producer has MOVED")
    return to_blocked


# ---------------------------------------------------------------------------
# Deterministic byte stream, mirrored bit-for-bit by the C++ suite
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def fnv1a64(data) -> int:
    h = 0xCBF29CE484222325
    if isinstance(data, str):
        data = data.encode("utf-8")
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def rand_bytes(name: str, count: int) -> np.ndarray:
    """`count` bytes reproducible from `name` alone. The C++ suite rebuilds these."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.uint8)
    for i in range(count):
        out[i] = (splitmix64((seed + i) & _MASK64) >> 24) & 0xFF
    return out


# ---------------------------------------------------------------------------
# Safetensors header reading — no payload beyond the named byte ranges
# ---------------------------------------------------------------------------


def open_header(path: Path):
    fh = path.open("rb")
    length = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(length))
    header.pop("__metadata__", None)
    return fh, header, 8 + length


def read_slice(fh, header, base: int, name: str, nbytes: int) -> bytes:
    info = header[name]
    begin, end = info["data_offsets"]
    take = min(nbytes, end - begin)
    fh.seek(base + begin)
    raw = fh.read(take)
    if len(raw) != take:
        raise SystemExit(f"{name}: short read ({len(raw)} of {take})")
    return raw


def f8e4m3_to_f32(raw: bytes) -> np.ndarray:
    """Decode fp8-e4m3fn with TORCH — an implementation that is not ours."""
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return t.float().numpy()


_E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_nibbles_to_f32(raw: bytes) -> np.ndarray:
    """Low-nibble-first E2M1 decode, the LUT nvfp4_dequant.h:37-38 already gates."""
    b = np.frombuffer(raw, dtype=np.uint8)
    nib = np.empty(b.size * 2, dtype=np.uint8)
    nib[0::2] = b & 0x0F
    nib[1::2] = b >> 4
    mag = _E2M1_LUT[nib & 0x7]
    return np.where((nib & 0x8) != 0, -mag, mag).astype(np.float32)


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def cxx_f32(v: float) -> str:
    # `%.9g` of an integral value emits `64`, and `64F` is not a float literal —
    # it is an integer with an unknown user-defined suffix, which is a hard
    # compile error rather than a silent narrowing. Force a decimal point.
    text = f"{float(v):.9g}"
    if not any(c in text for c in ".eE"):
        text += ".0"
    return text + "F"


def emit_bytes(out, name: str, data) -> None:
    values = ", ".join(str(int(b)) for b in data)
    out.write(f"inline constexpr uint8_t {name}[] = {{{values}}};\n")
    out.write(f"inline constexpr int64_t {name}Count = {len(data)};\n\n")


def emit_f32(out, name: str, data) -> None:
    values = ", ".join(cxx_f32(v) for v in data)
    out.write(f"inline constexpr float {name}[] = {{{values}}};\n")
    out.write(f"inline constexpr int64_t {name}Count = {len(data)};\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", required=True, type=Path, help="the pinned vLLM checkout")
    parser.add_argument("--ltx2", required=True, type=Path,
                        help="the Lightricks LTX-2 checkout (the HIGH-nibble authority)")
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    check_transcription(args.vllm)
    check_ltx2_nibble_order(args.ltx2)
    to_blocked = load_upstream_to_blocked(args.vllm)
    vllm_sha, vllm_clean = pinned_revision(args.vllm, "vllm", _PINNED_VLLM_PATHS)

    ltx2_sha, ltx2_clean = pinned_revision(args.ltx2, "ltx2", _PINNED_LTX2_PATHS)

    root = args.checkpoint_root / "ltx-2.5/vonkaiser-fp8-nvfp4"
    dit_path = root / "transformer/ltx-2.5-22b-distilled-fp8.safetensors"
    te_path = root / "text_encoders/gemma4-12b-with-proj-nvfp4-torchao.safetensors"
    nvfp4_dit_path = (
        args.checkpoint_root
        / "ltx-2.5/lightricks-ltx-2.5/diffusion_models"
        / "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
    )
    for p in (dit_path, te_path, nvfp4_dit_path):
        if not p.is_file():
            raise SystemExit(f"missing shipped checkpoint: {p}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open("w", encoding="utf-8")
    out.write(
        "// GENERATED by scripts/gen-ltx2-quant-goldens.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// LTX-2.5 phases L6 + L9a (.agents/specs/ltx-2-5.md, issue #435): the\n"
        "// torchao-NVFP4 scale swizzle, the NIBBLE ORDER\n"
        "// (.agents/specs/nvfp4-nibble-order.md), and the three shipped checkpoints'\n"
        "// own bytes.\n"
        "//\n"
        f"// FP8 DiT     {dit_path}\n"
        f"// NVFP4 TE    {te_path}\n"
        f"// NVFP4 DiT   {nvfp4_dit_path} ({nvfp4_dit_path.stat().st_size} bytes)\n"
        "//\n"
        f"// LTX-2 revision (HIGH-nibble-first authority): {ltx2_sha}"
        + ("" if ltx2_clean else "  [worktree not fully clean; the pinned files are]")
        + "\n"
        f"// vLLM revision (swizzle transcription pinned against it): {vllm_sha}\n"
        + (
            "// Both pinned vLLM sources are clean at that revision; the rest of that\n"
            "// checkout's worktree is NOT, so the anchor covers those two files only.\n"
            if not vllm_clean
            else "// That vLLM checkout was entirely clean at generation time.\n"
        )
        +
        "// The swizzle oracle RUNS: vLLM's own `to_blocked(x, backend=\"torch\")` is\n"
        "// loaded out of the pinned checkout and called. Its `backend=\"triton\"` arm and\n"
        "// `swizzle_blockscale` are not executed here (no driver / unconditional\n"
        "// .cuda()); both are pinned by source. See the script.\n"
        "//\n"
        "// Regenerate (one line; a trailing backslash in a // comment is a\n"
        "// -Werror=comment line continuation):\n"
        "//   python3 scripts/gen-ltx2-quant-goldens.py --vllm <vllm> --ltx2 <ltx2>"
        " --checkpoint-root <root> --out tests/vllm/models/ltx2_quant_goldens.inc\n"
        "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
    )
    out.write(f'inline constexpr const char* kLtx2QuantVllmRevision = "{vllm_sha}";\n\n')

    # --- section 1: the swizzle, over shapes that exercise every tile boundary -
    cases = [(128, 4), (128, 16), (256, 4), (384, 12), (128, 240)]
    out.write(f"inline constexpr int64_t kLtx2BlockedCaseCount = {len(cases)};\n")
    out.write("struct Ltx2BlockedCase { int64_t rows, cols; };\n")
    out.write("inline constexpr Ltx2BlockedCase kLtx2BlockedCases[] = {\n")
    for rows, cols in cases:
        out.write(f"    {{{rows}, {cols}}},\n")
    out.write("};\n\n")
    for idx, (rows, cols) in enumerate(cases):
        src = rand_bytes(f"blocked.{rows}x{cols}", rows * cols)
        t = torch.from_numpy(src.reshape(rows, cols).copy())
        blocked = to_blocked(t, backend="torch").numpy()
        assert blocked.size == rows * cols, (blocked.size, rows * cols)
        emit_bytes(out, f"kLtx2BlockedLinear{idx}", src)
        emit_bytes(out, f"kLtx2BlockedSwizzled{idx}", blocked)

    # --- section 2: the SHIPPED text encoder's own swizzled scale ---------------
    #
    # The first 512 bytes of a swizzled buffer are exactly the (rt=0, ct=0) tile,
    # i.e. logical rows 0..127 x cols 0..3 — self-contained, so a 512-byte read
    # pins the real layout without materializing a 7.4 GB file.
    te_fh, te_hdr, te_base = open_header(te_path)
    te_module = "text_embedding_projection.video_aggregate_embed"
    scale_name = te_module + ".weight_scale"
    scale_info = te_hdr[scale_name]
    real_tile = read_slice(te_fh, te_hdr, te_base, scale_name, 512)
    out.write(
        "// The (row-tile 0, col-tile 0) block of the SHIPPED text encoder's\n"
        f"// `{scale_name}` — logical rows 0..127, cols 0..3.\n"
        f"// Stored shape {scale_info['shape']} (= [out/4, (in/16)*4]).\n"
    )
    emit_bytes(out, "kLtx2RealTeScaleTileSwizzled", real_tile)
    # Unswizzle by inverting the oracle rather than by re-deriving it: push an
    # index map (int32, because a 512-entry arange does not fit in uint8) through
    # UPSTREAM's own `to_blocked` to get linear->swizzled, then scatter through
    # it. Writing the permute chain out a second time here would be the same
    # hand-typing the transcription used to be.
    idx = torch.arange(128 * 4, dtype=torch.int32).reshape(128, 4)
    fwd = to_blocked(idx, backend="torch").numpy()
    linear_tile = np.empty(128 * 4, dtype=np.uint8)
    linear_tile[fwd] = np.frombuffer(real_tile, dtype=np.uint8)
    emit_bytes(out, "kLtx2RealTeScaleTileLinear", linear_tile)
    out.write(
        "// Decoded with TORCH's own fp8-e4m3fn, so the byte->value half of this\n"
        "// gate is not our implementation checking itself.\n"
    )
    emit_f32(out, "kLtx2RealTeScaleTileLinearF32", f8e4m3_to_f32(bytes(linear_tile)))

    scale2_raw = read_slice(te_fh, te_hdr, te_base, te_module + ".weight_scale_2", 4)
    out.write(
        f"inline constexpr float kLtx2RealTeScale2 = "
        f"{cxx_f32(struct.unpack('<f', scale2_raw)[0])};\n\n"
    )
    packed = read_slice(te_fh, te_hdr, te_base, te_module + ".weight", 32)
    emit_bytes(out, "kLtx2RealTePackedHead", packed)
    # weight[0, 0:64] = e2m1(nibbles) * f8(scale[0, 0..3]) * scale2, group 16.
    nib = e2m1_nibbles_to_f32(bytes(packed))
    grp = f8e4m3_to_f32(bytes(linear_tile[:4]))
    scale2 = struct.unpack("<f", scale2_raw)[0]
    expect = np.array(
        [nib[i] * grp[i // 16] * scale2 for i in range(64)], dtype=np.float32
    )
    emit_f32(out, "kLtx2RealTeWeightHeadF32", expect)

    marker = te_hdr[te_module + ".torchao_nvfp4"]
    marker_raw = read_slice(
        te_fh, te_hdr, te_base, te_module + ".torchao_nvfp4",
        marker["data_offsets"][1] - marker["data_offsets"][0],
    )
    out.write(
        "// The shipped marker, verbatim — what makes this torchao and not\n"
        "// compressed-tensors, read rather than assumed.\n"
        f"inline constexpr const char* kLtx2RealTeMarkerJson =\n"
        f"    {json.dumps(marker_raw.decode('utf-8'))};\n\n"
    )
    te_fh.close()

    # --- section 3: the SHIPPED DiT's own FP8 bytes ----------------------------
    dit_fh, dit_hdr, dit_base = open_header(dit_path)
    dit_module = "model.diffusion_model.proj_out"
    w_raw = read_slice(dit_fh, dit_hdr, dit_base, dit_module + ".weight", 32)
    s_raw = read_slice(dit_fh, dit_hdr, dit_base, dit_module + ".weight_scale", 4)
    dit_scale = struct.unpack("<f", s_raw)[0]
    out.write(
        f"// The SHIPPED FP8 DiT's `{dit_module}` head: raw E4M3 bytes, its per-tensor\n"
        "// F32 scale, and the product TORCH computes for them.\n"
    )
    emit_bytes(out, "kLtx2RealDitFp8Head", w_raw)
    out.write(f"inline constexpr float kLtx2RealDitFp8Scale = {cxx_f32(dit_scale)};\n\n")
    emit_f32(out, "kLtx2RealDitFp8HeadF32", f8e4m3_to_f32(bytes(w_raw)) * dit_scale)

    # --- section 4: the FIRST-PARTY NVFP4 DiT, against the FP8 file as ORACLE ---
    #
    # Phase L9a / .agents/specs/nvfp4-nibble-order.md section 5.2. The two shipped
    # DiTs quantize the SAME base weights, so the FP8 file is an oracle that is not
    # ours for the NVFP4 read. What is emitted here lets the C++ suite run the
    # WHOLE pipeline -- unswizzle, then hi-nibble-first dequant -- on real bytes and
    # correlate the result against real FP8 weights, with a CONTROL arm.
    n4_fh, n4_hdr, n4_base = open_header(nvfp4_dit_path)
    n4_module = "model.diffusion_model.transformer_blocks.0.attn1.to_q"
    # The control: a DIFFERENT module, so the gate must prove it can tell the right
    # answer from a wrong one rather than passing on any two finite arrays.
    n4_control = "model.diffusion_model.transformer_blocks.0.attn1.to_k"

    n4_w = n4_hdr[n4_module + ".weight"]
    n4_out, n4_packed_in = n4_w["shape"]
    n4_in = n4_packed_in * 2
    n4_groups = n4_in // 16
    if any(name.endswith(".torchao_nvfp4") for name in n4_hdr):
        raise SystemExit(
            f"{nvfp4_dit_path}: a torchao_nvfp4 marker appeared. Its ABSENCE is half of "
            "how the loader discriminates the producer. Re-read the file."
        )
    # The cuBLAS-PADDED framing of the swizzled scale, which is what this file
    # declares, and which for these dims is numerically identical to the LINEAR
    # [N, K/16] shape -- the ambiguity spec section 4.1 records.
    padded_framing = [((n4_out + 127) // 128) * 128, ((n4_groups + 3) // 4) * 4]
    to_blocked_framing = [32 * ((n4_out + 127) // 128), 16 * ((n4_groups + 3) // 4)]
    if list(n4_hdr[n4_module + ".weight_scale"]["shape"]) != padded_framing:
        raise SystemExit(
            f"{n4_module}.weight_scale is "
            f"{n4_hdr[n4_module + '.weight_scale']['shape']}, not the cuBLAS-padded "
            f"{padded_framing} this section was written for."
        )

    # THE SCALE TILE, and why exactly 2048 bytes is a self-contained problem.
    # Swizzled offset is tile*512 + (r%32)*16 + ((r%128)//32)*4 + (c%4), with
    # tile = (r//128)*n_col_blocks + c//4. For r < 128 the row-tile index is 0, so
    # n_col_blocks drops out entirely and the first 2048 bytes of the REAL scale
    # are, byte for byte, a VALID STANDALONE swizzled [128, 16] grid. The C++ test
    # therefore runs the real unswizzle on real bytes without the 4096x256 buffer.
    n4_scale_tile = read_slice(n4_fh, n4_hdr, n4_base, n4_module + ".weight_scale", 2048)
    n4_scale2 = struct.unpack(
        "<f", read_slice(n4_fh, n4_hdr, n4_base, n4_module + ".weight_scale_2", 4)
    )[0]

    # Rows spanning all four 32-row QUARTERS of the swizzle tile. Rows 0..15 alone
    # would leave quarter=1,2,3 unexercised -- the section 7.0(c) trap, where the
    # fixture never enters the regime that discriminates.
    n4_rows = [1, 33, 65, 97, 2, 34, 66, 98]
    n4_elems = 256  # logical elements per row => 128 packed bytes, 16 groups
    out.write(
        "// ---------------------------------------------------------------------\n"
        "// Phase L9a: the FIRST-PARTY NVFP4 DiT vs the FP8 DiT as an ORACLE\n"
        f"// {nvfp4_dit_path.name}\n"
        f"// module  {n4_module}\n"
        f"// control {n4_control}\n"
        f"// logical [{n4_out}, {n4_in}]; weight_scale stored "
        f"{n4_hdr[n4_module + '.weight_scale']['shape']} = the cuBLAS-PADDED framing\n"
        f"// {padded_framing}. torchao's to_blocked framing would be "
        f"{to_blocked_framing};\n"
        "// both dress the SAME bytes. This file carries NO torchao_nvfp4 marker.\n"
        "//\n"
        "// The first 2048 scale bytes are a valid standalone SWIZZLED [128, 16] grid\n"
        "// (the row-tile index is 0 below row 128, so n_col_blocks drops out).\n"
        f"// Rows {n4_rows} span all four 32-row quarters of that tile.\n"
    )
    # The module's REAL geometry and the shape it REALLY declares, so the C++ gate
    # can run the producer resolver on the artifact's own numbers rather than on a
    # geometry invented in the test.
    out.write(f"inline constexpr int64_t kLtx2RealDitNvfp4OutFeatures = {n4_out};\n")
    out.write(f"inline constexpr int64_t kLtx2RealDitNvfp4InFeatures = {n4_in};\n")
    out.write(
        f"inline constexpr int64_t kLtx2RealDitNvfp4DeclaredScaleShape[] = "
        f"{{{n4_hdr[n4_module + '.weight_scale']['shape'][0]}, "
        f"{n4_hdr[n4_module + '.weight_scale']['shape'][1]}}};\n"
    )
    out.write(
        f'inline constexpr const char* kLtx2RealDitNvfp4Module =\n    "{n4_module}";\n'
    )
    out.write(f"inline constexpr int64_t kLtx2RealDitNvfp4ScaleTileRows = 128;\n")
    out.write(f"inline constexpr int64_t kLtx2RealDitNvfp4ScaleTileCols = 16;\n")
    out.write(f"inline constexpr int64_t kLtx2RealDitNvfp4RowElems = {n4_elems};\n")
    out.write(
        f"inline constexpr int64_t kLtx2RealDitNvfp4Rows[] = "
        f"{{{', '.join(str(r) for r in n4_rows)}}};\n"
        f"inline constexpr int64_t kLtx2RealDitNvfp4RowCount = {len(n4_rows)};\n"
    )
    out.write(f"inline constexpr float kLtx2RealDitNvfp4Scale2 = {cxx_f32(n4_scale2)};\n\n")
    emit_bytes(out, "kLtx2RealDitNvfp4ScaleTile", n4_scale_tile)

    # The packed NVFP4 weight rows, and the FP8 oracle's same rows.
    def packed_rows(fh, hdr, base, module, rows, nbytes, stride):
        begin = hdr[module + ".weight"]["data_offsets"][0]
        acc = bytearray()
        for r in rows:
            fh.seek(base + begin + r * stride)
            chunk = fh.read(nbytes)
            if len(chunk) != nbytes:
                raise SystemExit(f"{module} row {r}: short read")
            acc += chunk
        return bytes(acc)

    n4_packed = packed_rows(n4_fh, n4_hdr, n4_base, n4_module, n4_rows,
                            n4_elems // 2, n4_packed_in)
    emit_bytes(out, "kLtx2RealDitNvfp4Packed", n4_packed)
    n4_fh.close()

    # The ORACLE: the same rows of the same module out of the FP8 checkpoint,
    # decoded by TORCH. Emitted as f32 VALUES, per section 7.0's "emit values".
    dit_fh2, dit_hdr2, dit_base2 = open_header(dit_path)
    for label, module in (("Oracle", n4_module), ("Control", n4_control)):
        fw_shape = dit_hdr2[module + ".weight"]["shape"]
        fw_scale = struct.unpack(
            "<f", read_slice(dit_fh2, dit_hdr2, dit_base2, module + ".weight_scale", 4)
        )[0]
        raw = packed_rows(dit_fh2, dit_hdr2, dit_base2, module, n4_rows, n4_elems,
                          fw_shape[1])
        vals = f8e4m3_to_f32(raw) * fw_scale
        out.write(
            f"// FP8 `{module}` rows {n4_rows}, elements 0..{n4_elems - 1},\n"
            f"// per-tensor scale {fw_scale!r}, decoded by torch.\n"
        )
        emit_f32(out, f"kLtx2RealDitFp8{label}F32", vals)
    dit_fh2.close()
    dit_fh.close()

    out.write("}  // namespace vllm_test\n")
    out.close()
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
