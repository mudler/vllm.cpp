#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_tiling_goldens.inc — the LTX-2.5 tiled-decode oracle.

The tiling algebra (`ltx_core.tiling`) and the streaming decode
(`ConvVideoDecoder.tiled_decode`) are pure-Python and gateable exactly on any
CPU. This generator imports the upstream modules VERBATIM, runs them at REDUCED
dimensions with the same deterministic pseudo-random weights
`gen-ltx2-vae-goldens.py` uses, and emits the resulting intervals, masks and
tensors as C++ goldens. No weight byte is checked in.

Upstream sources (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/):
  tiling.py                             -> sections 1-3, 5
  model/video_vae/video_vae.py:520-592  -> section 4 (the axis mappers)
  model/video_vae/conv_video_decoder.py:359-557 -> sections 6-7 (tiled decode)
  ../ltx-pipelines/.../utils/helpers.py:59-88   -> section 3 (the AUTO layout)

Usage:
    python3 scripts/gen-ltx2-tiling-goldens.py \\
        --ltx2 ~/_git/LTX-2 \\
        --out tests/vllm/models/ltx2_tiling_goldens.inc

Needs torch + numpy + einops (CPU only).

UPSTREAM REVISION ANCHOR, for the reason gen-ltx2-vae-goldens.py states at
length: the numbers are only interpretable against the tree that produced them,
so the resolved SHA is emitted as `kLtx2TilingUpstreamRevision` and the C++ suite
asserts it against its own pin. Advancing the pin is a deliberate edit in BOTH
places, never a side effect of regenerating.

  Pinned revision: fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca

THE DETERMINISTIC STREAM IS NOT REDEFINED HERE. `ltx_rand`, `param_values`,
`fill_from_stream` and the emit helpers are imported from
`gen-ltx2-vae-goldens.py` by path, so the two generators cannot drift: a change
to the stream moves both sets of goldens or neither. That is the whole reason
this file does not carry its own copy.

WHY THE DECODE FIXTURES CARRY NO NOISE. Tiled decode calls `forward` once per
tile and untiled calls it once, so a decoder with `timestep_conditioning` or
`inject_noise` draws a DIFFERENT number of noise tensors on the two paths and the
equivalence comparison the whole gate rests on becomes meaningless. Section 5 of
the VAE goldens already pins the noise ORDER against upstream; these fixtures pin
the blend, and they set `timestep_conditioning=False` and use no `inject_noise`
block so the two paths differ only in the tiling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_PIN = "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca"


def _load_vae_generator():
    """Import gen-ltx2-vae-goldens.py by path (its name is not an identifier)."""
    path = _REPO / "scripts" / "gen-ltx2-vae-goldens.py"
    spec = importlib.util.spec_from_file_location("gen_ltx2_vae_goldens", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


G = _load_vae_generator()


# ---------------------------------------------------------------------------
# The reduced decoder the decode sections run.
#
# Its scale factors are what make a TINY latent tile at all: two `compress_*`
# blocks touching space and two touching time, with patch_size 2, give
# SpatioTemporalScaleFactors(time=4, height=8, width=8) (types.py:45-53). So a
# 16px / 12-frame tile is 2 latent cells and 3 latent frames, and a (5, 4, 4)
# latent splits into 2 temporal groups x 3 x 3 spatial tiles — 18 real tiles with
# real overlaps, at a size both paths can be run to completion.
# ---------------------------------------------------------------------------

TILING_BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_all", {"multiplier": 2, "residual": True}),
    ("res_x_y", {"num_layers": 1, "multiplier": 2}),
    ("compress_space", {"multiplier": 1}),
    ("compress_time", {"multiplier": 1}),
    ("res_x", {"num_layers": 1}),
]
TILING_DEC = dict(
    convolution_dimensions=3,
    in_channels=6,
    out_channels=3,
    patch_size=2,
    timestep_conditioning=False,
    base_channels=8,
)
TILING_LATENT = (1, 6, 5, 4, 4)

# Pixel/frame units, converted to the latent grid by `to_splitters`.
TILING_FRAMES = (12, 4)  # 3 latent frames, 1 latent overlap
TILING_SPATIAL = (16, 8)  # 2 latent cells, 1 latent overlap


def _tiling_config():
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig

    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=TILING_FRAMES[0], overlap=TILING_FRAMES[1]),
        height=DimensionSizeConfig(tile_size=TILING_SPATIAL[0], overlap=TILING_SPATIAL[1]),
        width=DimensionSizeConfig(tile_size=TILING_SPATIAL[0], overlap=TILING_SPATIAL[1]),
    )


# The ONE-TILE CONTROL, and it is the load-bearing equivalence statement.
#
# MEASURED, and it corrects the assumption this row was dispatched with: tiled
# decode is NOT an approximation of untiled decode for this VAE. Swept over tile
# sizes and both causality arms, upstream's own max|tiled - untiled| stays at
# 0.67-1.29 TIMES the full output range and does not converge as the tile grows.
# The reason is structural: the decoder's receptive field in LATENT units spans
# roughly the whole of the reduced fixture, and in production it is far wider than
# the 2 latent cells of overlap the AUTO layout uses (768px tile / 64px overlap on
# a 32x grid). Upstream accepts that seam and blends it; there is no bound to
# state, and inventing one would be fabricating a gate.
#
# What IS exact — max|diff| == 0, measured on upstream, on every arm — is the
# degenerate case: a tiling config whose splits all short-circuit produces ONE
# tile, and `tiled_decode` then reproduces `forward` bit for bit. That is the
# property that makes routing the pipeline through the streaming entry point safe
# at every size where the AUTO layout does not tile, which today includes every
# resolution this project has run.
CONTROL_FRAMES = (10_000, 0)
CONTROL_SPATIAL = (10_000, 0)


def _control_config():
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig

    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=CONTROL_FRAMES[0], overlap=CONTROL_FRAMES[1]),
        height=DimensionSizeConfig(tile_size=CONTROL_SPATIAL[0], overlap=CONTROL_SPATIAL[1]),
        width=DimensionSizeConfig(tile_size=CONTROL_SPATIAL[0], overlap=CONTROL_SPATIAL[1]),
    )


# THE UNTILED-AXIS CONTROL, which is a DIFFERENT arm from the one above and was
# ungated until it was asked for by name.
#
# `tile_size = 0` means "this axis is not tiled" (tiling.py:619-645), and it is a
# legal, documented value that `DimensionSizeConfig.__post_init__` blesses. It is
# NOT the same thing as `tile_size = 10_000`: a huge tile still declares the axis
# tiled, so `_prepare_tiles` installs the REAL mapper and the split merely
# short-circuits. A zero tile size installs `DEFAULT_MAPPING_OPERATION`
# (tiling.py:126-132) instead, which returns `slice(0, None)` and a length-1
# broadcast mask from `untiled_mask_1d` (tiling.py:121-123) — a completely
# separate code path that the 10_000 control never touches.
#
# Nothing in the shipped pipeline reaches it (`Ltx2AutoTileSizeConfig` always
# declares all three axes tiled), but it is on the public signature of
# `Ltx2ConvVideoDecodeTiled`, so it ships. A defect there multiplies the whole
# decoded volume by the mask — i.e. renders a black clip — and no assertion in
# this suite could see it until this arm existed.
#
# ─── AND THE TWO AXES ARE NOT SYMMETRIC, WHICH IS THE MEASURED PART ──────────
# Swept over all eight (frames, height, width) x (tiled, untiled) combinations on
# this fixture at the pinned SHA:
#
#   frames tiled,   any spatial combination  -> runs, max|diff| vs forward == 0.0
#   frames UNTILED, every spatial combination -> TypeError
#
# because `tiled_decode` computes `curr_temporal_slice.stop - .start`
# (conv_video_decoder.py:424) and `DEFAULT_MAPPING_OPERATION` hands it
# `slice(0, None)`, whose `.stop` is None. So an untiled FRAMES axis is legal to
# `create_tiles` and unusable in `tiled_decode` — upstream raises, and this port
# refuses with the reason named rather than inventing a concrete stop upstream
# never computes. The spatial half is gated end to end below.
UNTILED_SPATIAL = (0, 0)


def _untiled_spatial_config():
    """Frames still tiled (so `tiled_decode` runs), height and width NOT tiled."""
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig

    zero = DimensionSizeConfig(tile_size=UNTILED_SPATIAL[0], overlap=UNTILED_SPATIAL[1])
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=CONTROL_FRAMES[0], overlap=CONTROL_FRAMES[1]),
        height=zero,
        width=zero,
    )


def _untiled_axes_config():
    """All three axes untiled — legal to `create_tiles`, fatal to `tiled_decode`."""
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig

    zero = DimensionSizeConfig(tile_size=UNTILED_SPATIAL[0], overlap=UNTILED_SPATIAL[1])
    return TileSizeConfig(frames=zero, height=zero, width=zero)


# ---------------------------------------------------------------------------
# Emit helpers on top of the imported ones.
# ---------------------------------------------------------------------------


def emit_i64_array(out, name: str, values) -> None:
    values = list(values)
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    if not values:
        out.write("    0,  // empty; the length constant below is what is asserted\n")
    for i in range(0, len(values), 12):
        out.write("    " + ", ".join(str(int(v)) for v in values[i : i + 12]) + ",\n")
    out.write("};\n")


def emit_intervals(out, name: str, cases) -> None:
    """Each case is (label, [DimensionInterval, ...]) flattened to 4 ints per tile."""
    out.write(f"inline constexpr const char* {name}Labels[] = {{\n")
    for label, _ in cases:
        out.write(f'    "{label}",\n')
    out.write("};\n")
    emit_i64_array(out, f"{name}Counts", [len(ivs) for _, ivs in cases])
    flat = []
    for _, ivs in cases:
        for iv in ivs:
            flat += [iv.start, iv.end, iv.left_ramp, iv.right_ramp]
    emit_i64_array(out, f"{name}Flat", flat)
    out.write("\n")


# ---------------------------------------------------------------------------
# Section 1 — compute_trapezoidal_mask_1d (tiling.py:13-49)
# ---------------------------------------------------------------------------

MASK_CASES = [
    # (length, ramp_left, ramp_right, left_starts_from_0)
    (9, 0, 4, True),
    (13, 5, 0, True),
    (16, 8, 8, False),
    (16, 0, 8, False),
    (16, 8, 0, False),
    (5, 3, 3, False),  # the ramps OVERLAP: both multiply the same samples
    (5, 3, 3, True),
    (4, 9, 0, False),  # ramp clamped to the length (tiling.py:33-34)
    (1, 1, 1, True),
    (7, 1, 1, True),
    (7, 1, 1, False),
]


def section_masks(out) -> None:
    from ltx_core.tiling import compute_trapezoidal_mask_1d

    out.write("// --- section 1: compute_trapezoidal_mask_1d (tiling.py:13-49) ---\n")
    emit_i64_array(out, "kLtx2MaskCaseParams", [v for c in MASK_CASES for v in (c[0], c[1], c[2], int(c[3]))])
    G.emit_scalar(out, "kLtx2MaskCaseCount", len(MASK_CASES))
    values = []
    for length, left, right, from0 in MASK_CASES:
        values += compute_trapezoidal_mask_1d(length, left, right, from0).tolist()
    G.emit_f32(out, "kLtx2MaskGolden", np.asarray(values, dtype=np.float32))


# ---------------------------------------------------------------------------
# Section 2 — the split operations (tiling.py:174-361)
# ---------------------------------------------------------------------------


def section_splits(out) -> None:
    from ltx_core.tiling import (
        split_by_count,
        split_by_count_temporal_causal,
        split_by_size,
        split_temporal_causal,
    )

    cases = []
    # split_by_size, including the `dim <= size` short-circuit that makes
    # upstream's own default layout a NO-OP below one tile (tiling.py:199-200).
    for dim, size, overlap in [
        (4, 2, 1), (5, 3, 1), (10, 4, 2), (14, 24, 2), (8, 14, 2),
        (16, 24, 2), (28, 24, 2), (40, 24, 2), (60, 24, 2), (31, 10, 3),
        (7, 3, 0), (9, 4, 1), (2, 2, 1), (3, 2, 1),
    ]:
        cases.append((f"by_size:{dim}:{size}:{overlap}", split_by_size(size, overlap)(dim).intervals))
    # ...with min_tile_size, which grows a short last tile LEFT and widens the
    # penultimate right ramp (tiling.py:140-154), then validates.
    for dim, size, overlap, min_tile in [(9, 4, 1, 3), (10, 4, 1, 4), (11, 5, 2, 4), (2, 4, 1, 3)]:
        cases.append(
            (
                f"by_size_min:{dim}:{size}:{overlap}:{min_tile}",
                split_by_size(size, overlap, min_tile_size=min_tile)(dim).intervals,
            )
        )
    # split_temporal_causal: every tile after the first shifts back by 1 and
    # grows its left ramp by 1 (tiling.py:244-247).
    for dim, size, overlap in [(5, 3, 1), (4, 10, 3), (16, 10, 3), (31, 10, 3), (10, 4, 2)]:
        cases.append(
            (f"temporal_causal:{dim}:{size}:{overlap}", split_temporal_causal(size, overlap)(dim).intervals)
        )
    # split_by_count and its causal wrapper (tiling.py:275-361).
    for dim, n, overlap in [(10, 3, 1), (10, 1, 0), (12, 4, 2), (13, 3, 2), (17, 4, 1)]:
        cases.append((f"by_count:{dim}:{n}:{overlap}", split_by_count(n, overlap)(dim).intervals))
        cases.append(
            (
                f"by_count_causal:{dim}:{n}:{overlap}",
                split_by_count_temporal_causal(n, overlap)(dim).intervals,
            )
        )

    out.write("// --- section 2: split operations (tiling.py:174-361) ---\n")
    emit_intervals(out, "kLtx2Split", cases)


# ---------------------------------------------------------------------------
# Section 3 — TileSizeConfig.from_long_side and the CONV AUTO layout
# (tiling.py:754-795, ltx_pipelines/utils/helpers.py:59-88)
#
# This is what decides whether tiling happens AT ALL, and the measured answer is
# that it does not below 768px on the long side or 81 frames — which is why
# 448x256/25f resolves to ONE tile and ONE temporal group. The sweep records that
# as a golden rather than as a claim.
# ---------------------------------------------------------------------------

AUTO_CASES = [
    (128, 128, 9),
    (320, 192, 25),
    (448, 256, 25),
    (896, 512, 25),
    (1280, 704, 121),
    (1920, 1088, 241),
    (768, 768, 81),
    (1024, 576, 97),
]


def section_auto_layout(out) -> None:
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
    from ltx_core.types import SpatioTemporalScaleFactors

    factors = SpatioTemporalScaleFactors(time=8, height=32, width=32)
    long_side = DimensionSizeConfig(tile_size=768, overlap=64)  # helpers.py:62
    frames = DimensionSizeConfig(tile_size=80, overlap=24)  # helpers.py:63

    rows = []
    for width, height, num_frames in AUTO_CASES:
        cfg = TileSizeConfig.from_long_side(
            long_side=long_side, height=height, width=width, scale_factors=factors, frames=frames
        )
        latent_t = (num_frames - 1) // factors.time + 1
        latent_h = height // factors.height
        latent_w = width // factors.width
        t_split, h_split, w_split = cfg.to_splitters(factors)
        rows.append(
            [
                width, height, num_frames,
                cfg.height.tile_size, cfg.height.overlap,
                cfg.width.tile_size, cfg.width.overlap,
                cfg.frames.tile_size, cfg.frames.overlap,
                len(t_split(latent_t).intervals),
                len(h_split(latent_h).intervals),
                len(w_split(latent_w).intervals),
                cfg.video_chunks_number(num_frames),
            ]
        )

    out.write(
        "// --- section 3: the CONV AUTO layout (helpers.py:59-88 ->\n"
        "// TileSizeConfig.from_long_side, tiling.py:754-795). Each row is\n"
        "// {width, height, frames, h_tile, h_overlap, w_tile, w_overlap,\n"
        "//  f_tile, f_overlap, t_intervals, h_intervals, w_intervals, chunks}. ---\n"
    )
    G.emit_scalar(out, "kLtx2AutoCaseCount", len(rows))
    G.emit_scalar(out, "kLtx2AutoCaseStride", len(rows[0]))
    emit_i64_array(out, "kLtx2AutoCases", [v for row in rows for v in row])
    out.write("\n")


def section_untiled_mapping(out) -> None:
    """Section 3b — `DEFAULT_MAPPING_OPERATION` (tiling.py:126-132), executed.

    The AUTO layout never produces it, so it is emitted from an explicitly
    untiled config rather than from a resolution. `slice(0, None)` has no
    integer stop upstream; -1 is emitted for it and the C++ side asserts its own
    concrete stop equals the full axis extent, which is what `None` means.
    """
    tiles = _prepare_tiles(TILING_LATENT, _untiled_axes_config())

    starts, stops, mask_lens, mask_values = [], [], [], []
    for axis in (2, 3, 4):  # frames, height, width — batch/channel are never split
        out_slice = tiles[0].out_coords[axis]
        mask = tiles[0].masks_1d[axis]
        starts.append(0 if out_slice.start is None else int(out_slice.start))
        stops.append(-1 if out_slice.stop is None else int(out_slice.stop))
        mask_lens.append(int(mask.numel()))
        mask_values.append(float(mask.reshape(-1)[0]))

    out.write(
        "// --- section 3b: the UNTILED-AXIS mapping arm — `DEFAULT_MAPPING_OPERATION`\n"
        "// (tiling.py:126-132) with `untiled_mask_1d` (tiling.py:121-123). Reached by\n"
        "// `tile_size = 0`, which is legal (tiling.py:619-645) and is a DIFFERENT path\n"
        "// from a huge tile size: that one still installs the real mapper. Order is\n"
        "// {frames, height, width}; a stop of -1 is upstream's `slice(0, None)`. ---\n"
    )
    G.emit_scalar(out, "kLtx2UntiledMapTileCount", len(tiles))
    emit_i64_array(out, "kLtx2UntiledMapStart", starts)
    emit_i64_array(out, "kLtx2UntiledMapStop", stops)
    emit_i64_array(out, "kLtx2UntiledMapMaskLen", mask_lens)
    # `repr`, not `%g`: `f"{1.0:.9g}"` is "1", and "1f" is not a C++ float literal.
    out.write(
        "inline constexpr float kLtx2UntiledMapMaskValue[] = {"
        + ", ".join(f"{float(v)!r}f" for v in mask_values)
        + "};\n\n"
    )


# ---------------------------------------------------------------------------
# Section 4 — the axis mappers (video_vae.py:549-555, 586-592)
# ---------------------------------------------------------------------------

MAP_CASES = [
    # (begin, end, left_ramp, right_ramp, scale)
    (0, 3, 0, 1, 4),
    (1, 5, 2, 0, 4),
    (0, 2, 0, 1, 8),
    (1, 3, 1, 1, 8),
    (2, 4, 1, 0, 8),
    (0, 10, 0, 3, 8),
    (9, 19, 4, 0, 8),
    (0, 14, 0, 2, 32),
]


def section_mappers(out) -> None:
    from ltx_core.model.video_vae.video_vae import map_spatial_slice, map_temporal_slice

    out.write("// --- section 4: the axis mappers (video_vae.py:549-555, 586-592) ---\n")
    G.emit_scalar(out, "kLtx2MapCaseCount", len(MAP_CASES))
    emit_i64_array(out, "kLtx2MapCaseParams", [v for c in MAP_CASES for v in c])
    t_bounds, s_bounds, t_masks, s_masks = [], [], [], []
    for begin, end, left, right, scale in MAP_CASES:
        sl, mask = map_temporal_slice(begin, end, left, right, scale)
        t_bounds += [sl.start, sl.stop]
        t_masks += mask.tolist()
        sl, mask = map_spatial_slice(begin, end, left, right, scale)
        s_bounds += [sl.start, sl.stop]
        s_masks += mask.tolist()
    emit_i64_array(out, "kLtx2MapTemporalBounds", t_bounds)
    emit_i64_array(out, "kLtx2MapSpatialBounds", s_bounds)
    G.emit_f32(out, "kLtx2MapTemporalMasks", np.asarray(t_masks, dtype=np.float32))
    G.emit_f32(out, "kLtx2MapSpatialMasks", np.asarray(s_masks, dtype=np.float32))


# ---------------------------------------------------------------------------
# Section 5 — masks_are_complementary and compute_summed_weights
# (tiling.py:438-491)
#
# The complementary verdict decides whether a DENOMINATOR is allocated at all.
# The second case here is deliberately NOT complementary — two overlapping
# out-slices whose masks are both all-ones, so the overlap sums to 2 — which is
# the only way to gate the fallback arm. Without it, a port that always takes the
# complementary branch would be green and would emit a double-bright seam.
# ---------------------------------------------------------------------------


def section_complementary(out) -> None:
    from ltx_core.tiling import Tile, compute_summed_weights, masks_are_complementary
    import torch

    cfg = _tiling_config()
    tiles = _prepare_tiles(TILING_LATENT, cfg)
    full = _full_pixel_shape(TILING_LATENT)
    good = masks_are_complementary(tiles, full)

    # The non-complementary layout: length-4 and length-4 out-slices overlapping
    # by 2, both all-ones, on a length-6 axis; the other two axes untiled.
    ones = torch.ones(1)
    bad_tiles = [
        Tile(
            in_coords=(slice(0, 1), slice(0, 3), slice(0, 4), slice(0, 2), slice(0, 2)),
            out_coords=(slice(0, 1), slice(0, 3), slice(0, 4), slice(0, 2), slice(0, 2)),
            masks_1d=(ones, ones, torch.ones(4), torch.ones(2), torch.ones(2)),
        ),
        Tile(
            in_coords=(slice(0, 1), slice(0, 3), slice(2, 6), slice(0, 2), slice(0, 2)),
            out_coords=(slice(0, 1), slice(0, 3), slice(2, 6), slice(0, 2), slice(0, 2)),
            masks_1d=(ones, ones, torch.ones(4), torch.ones(2), torch.ones(2)),
        ),
    ]
    bad_shape = (1, 3, 6, 2, 2)
    bad = masks_are_complementary(bad_tiles, bad_shape)
    weights = compute_summed_weights(bad_tiles, bad_shape)

    out.write("// --- section 5: masks_are_complementary / compute_summed_weights ---\n")
    G.emit_scalar(out, "kLtx2ComplementaryShipped", int(good))
    G.emit_scalar(out, "kLtx2ComplementaryBad", int(bad))
    emit_i64_array(out, "kLtx2SummedWeightsShape", list(bad_shape))
    G.emit_f32(out, "kLtx2SummedWeights", weights.numpy())


# ---------------------------------------------------------------------------
# Sections 6/7 — tiled decode vs untiled forward (conv_video_decoder.py:359-557)
# ---------------------------------------------------------------------------


def _prepare_tiles(latent_shape, cfg):
    """`ConvVideoDecoder._prepare_tiles` (conv_video_decoder.py:359-381)."""
    import torch
    from ltx_core.model.video_vae.video_vae import (
        map_spatial_slice,
        map_temporal_slice,
        to_mapping_operation,
    )
    from ltx_core.tiling import (
        DEFAULT_MAPPING_OPERATION,
        DEFAULT_SPLIT_OPERATION,
        create_tiles,
    )
    from ltx_core.types import SpatioTemporalScaleFactors

    factors = SpatioTemporalScaleFactors.from_blocks(TILING_BLOCKS, TILING_DEC["patch_size"])
    splitters = [DEFAULT_SPLIT_OPERATION] * 5
    mappers = [DEFAULT_MAPPING_OPERATION] * 5
    t_split, h_split, w_split = cfg.to_splitters(factors)
    splitters[2], splitters[3], splitters[4] = t_split, h_split, w_split
    if cfg.height.is_tiled():
        mappers[3] = to_mapping_operation(map_spatial_slice, scale=factors.height)
    if cfg.width.is_tiled():
        mappers[4] = to_mapping_operation(map_spatial_slice, scale=factors.width)
    if cfg.frames.is_tiled():
        mappers[2] = to_mapping_operation(map_temporal_slice, scale=factors.time)
    return create_tiles(torch.Size(latent_shape), splitters, mappers)


def _full_pixel_shape(latent_shape):
    import torch
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    factors = SpatioTemporalScaleFactors.from_blocks(TILING_BLOCKS, TILING_DEC["patch_size"])
    return VideoLatentShape.from_torch_shape(torch.Size(latent_shape)).upscale(factors).to_torch_shape()


def _decode_arm(out, out_prefix: str, causal: bool, name_prefix: str) -> None:
    import torch
    from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
    from ltx_core.tiling import group_tiles_by_temporal_slice, masks_are_complementary

    decoder = ConvVideoDecoder(
        decoder_blocks=TILING_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        causal=causal,
        **TILING_DEC,
    ).eval()
    manifest = G.fill_from_stream(decoder, prefix=name_prefix)
    latent = G.make_input(name_prefix + "input", TILING_LATENT, 1.0)

    with torch.no_grad():
        untiled = decoder(latent)
        chunks = list(decoder.tiled_decode(latent, _tiling_config()))
        tiled = torch.cat(chunks, dim=2)
        control_chunks = list(decoder.tiled_decode(latent, _control_config()))
        control = torch.cat(control_chunks, dim=2)
        untiled_spatial_chunks = list(decoder.tiled_decode(latent, _untiled_spatial_config()))
        untiled_spatial = torch.cat(untiled_spatial_chunks, dim=2)
    # And the arm upstream CANNOT run, recorded as the exception type it raises
    # rather than as prose. `emit_scalar` of the boolean is what the C++ refusal
    # test asserts its own refusal against.
    #
    # THE TYPE ALONE IS NOT THE MECHANISM. A bare `except TypeError` keeps this
    # constant at 1 for ANY future `TypeError` from ANY line of `tiled_decode`,
    # so the cited cause — `conv_video_decoder.py:424` subtracting the `None`
    # stop `DEFAULT_MAPPING_OPERATION` handed it — could go stale silently while
    # the golden still read "upstream raises". So the FILE, LINE and message of
    # the innermost frame are pinned too, and the C++ suite asserts all three
    # against the mechanism its own refusal message names.
    untiled_frames_raise_file = ""
    untiled_frames_raise_line = 0
    untiled_frames_raise_message = ""
    try:
        with torch.no_grad():
            list(decoder.tiled_decode(latent, _untiled_axes_config()))
        untiled_frames_raises = 0
    except TypeError as exc:
        untiled_frames_raises = 1
        tb = exc.__traceback__
        while tb.tb_next is not None:  # the innermost frame is the one that raised
            tb = tb.tb_next
        untiled_frames_raise_file = Path(tb.tb_frame.f_code.co_filename).name
        untiled_frames_raise_line = tb.tb_lineno
        untiled_frames_raise_message = str(exc)
    assert untiled_frames_raises == 1, (
        "upstream's tiled_decode ran an UNTILED frames axis; the C++ refusal in "
        "Ltx2ConvVideoDecodeTiled is now wrong and this assertion is what says so"
    )
    assert untiled_frames_raise_file == "conv_video_decoder.py", (
        f"the untiled-frames TypeError now comes from {untiled_frames_raise_file}, not "
        "conv_video_decoder.py — the refusal cites a mechanism that has moved"
    )
    assert "unsupported operand type" in untiled_frames_raise_message, (
        f"the untiled-frames TypeError now reads {untiled_frames_raise_message!r}; it is no "
        "longer the None-stop arithmetic the refusal cites"
    )

    tiles = _prepare_tiles(TILING_LATENT, _tiling_config())
    groups = group_tiles_by_temporal_slice(tiles)
    complementary = masks_are_complementary(tiles, _full_pixel_shape(TILING_LATENT))

    G.emit_manifest(out, out_prefix + "Param", manifest)
    G.emit_scalar(out, out_prefix + "OutC", untiled.shape[1])
    G.emit_scalar(out, out_prefix + "OutT", untiled.shape[2])
    G.emit_scalar(out, out_prefix + "OutH", untiled.shape[3])
    G.emit_scalar(out, out_prefix + "OutW", untiled.shape[4])
    G.emit_scalar(out, out_prefix + "TileCount", len(tiles))
    G.emit_scalar(out, out_prefix + "GroupCount", len(groups))
    G.emit_scalar(out, out_prefix + "Complementary", int(complementary))
    emit_i64_array(out, out_prefix + "ChunkFrames", [c.shape[2] for c in chunks])
    G.emit_scalar(out, out_prefix + "ChunkCount", len(chunks))
    G.emit_f32(out, out_prefix + "Untiled", untiled.numpy())
    G.emit_f32(out, out_prefix + "Tiled", tiled.numpy())
    # The number the C++ band is DERIVED from, not a round guess. Emitted so the
    # test can assert its own band against upstream's own tiled-vs-untiled gap
    # instead of against a constant somebody chose.
    gap = float(np.max(np.abs(untiled.numpy() - tiled.numpy())))
    span = float(np.max(np.abs(untiled.numpy())))
    out.write(
        f"// upstream's own max|tiled - untiled| on this arm: {gap:.9g}\n"
        f"// (the output's own max|value| is {span:.9g} — the tiled decode is NOT an\n"
        "//  approximation of the untiled one, and no bound is claimed. See the\n"
        "//  ONE-TILE CONTROL below, whose bound IS zero and is measured.)\n"
    )
    out.write(
        f"inline constexpr double {out_prefix}UpstreamTiledVsUntiled = {gap:.9g};\n"
        f"inline constexpr double {out_prefix}OutputSpan = {span:.9g};\n"
    )
    # THE ONE-TILE CONTROL. Upstream's own value here is exactly 0.0 on every arm
    # swept; it is emitted rather than assumed so a regeneration that broke it
    # would show as a changed constant instead of a silently relaxed assertion.
    control_gap = float(np.max(np.abs(untiled.numpy() - control.numpy())))
    G.emit_scalar(out, out_prefix + "ControlChunkCount", len(control_chunks))
    out.write(
        f"inline constexpr double {out_prefix}UpstreamControlVsUntiled = {control_gap:.9g};\n"
    )
    # THE UNTILED-SPATIAL CONTROL. Same bound, different code path: `tile_size = 0`
    # routes height and width through `DEFAULT_MAPPING_OPERATION` and its length-1
    # broadcast mask, which the 10_000 control above never touches.
    untiled_spatial_gap = float(np.max(np.abs(untiled.numpy() - untiled_spatial.numpy())))
    G.emit_scalar(out, out_prefix + "UntiledSpatialChunkCount", len(untiled_spatial_chunks))
    out.write(
        f"inline constexpr double {out_prefix}UpstreamUntiledSpatialVsUntiled = "
        f"{untiled_spatial_gap:.9g};\n"
    )
    G.emit_scalar(out, out_prefix + "UpstreamUntiledFramesRaises", untiled_frames_raises)
    G.emit_scalar(out, out_prefix + "UpstreamUntiledFramesRaiseLine", untiled_frames_raise_line)
    # `json.dumps` and not an f-string quote: both of these are UPSTREAM text, so a
    # future `TypeError` message carrying a `"` or a `\` — or a checkout path that
    # does — would otherwise emit a C string literal that does not compile, and the
    # generator would look like it succeeded. JSON string escaping is a subset of
    # C's for these bytes, so on today's values the output is byte-identical.
    out.write(
        f"inline constexpr const char* {out_prefix}UpstreamUntiledFramesRaiseFile = "
        f"{json.dumps(untiled_frames_raise_file)};\n"
        f"inline constexpr const char* {out_prefix}UpstreamUntiledFramesRaiseMessage = "
        f"{json.dumps(untiled_frames_raise_message)};\n"
    )
    out.write("\n")


def section_decode(out) -> None:
    out.write(
        "// --- section 6: tiled decode vs untiled forward, CAUSAL\n"
        "// (conv_video_decoder.py:383-484, 508-557) ---\n"
    )
    _decode_arm(out, "kLtx2TileDecCausal", True, "ltx2.tiledec.causal.")
    out.write(
        "// --- section 7: the same, NON-CAUSAL — the polarity the shipped\n"
        "// ltx-2.5-video-vae-conv checkpoint actually sets (`causal_decoder: false`),\n"
        "// where a temporal tile's convolutions can see one frame past its own end\n"
        "// and the temporal seam is therefore NOT exact in the interior. ---\n"
    )
    _decode_arm(out, "kLtx2TileDecNonCausal", False, "ltx2.tiledec.noncausal.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path, help="Lightricks/LTX-2 checkout")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    src = G.load_upstream(args.ltx2)
    revision = G.upstream_revision(args.ltx2)
    if revision != _PIN:
        print(
            f"WARNING: --ltx2 resolves to {revision}, this generator documents {_PIN}. "
            f"The emitted kLtx2TilingUpstreamRevision will be {revision} and the C++ "
            f"pin will REFUSE it until both are advanced deliberately.",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-tiling-goldens.py — DO NOT EDIT.\n"
            "//\n"
            f"// Upstream: Lightricks/LTX-2 @ {revision}\n"
            f"//           {src}\n"
            "//\n"
            "// Every tensor here was produced by EXECUTING the upstream module; the C++\n"
            "// suite rebuilds the identical weights from the identical PRNG, so no weight\n"
            "// byte is checked in.\n"
            "#pragma once\n\n"
            "#include <cstdint>\n\n"
            "namespace vllm_test {\n\n"
        )
        out.write(f'inline constexpr const char* kLtx2TilingUpstreamRevision = "{revision}";\n\n')
        section_masks(out)
        out.write("\n")
        section_splits(out)
        section_auto_layout(out)
        section_untiled_mapping(out)
        section_mappers(out)
        out.write("\n")
        section_complementary(out)
        out.write("\n")
        section_decode(out)
        out.write("}  // namespace vllm_test\n")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
