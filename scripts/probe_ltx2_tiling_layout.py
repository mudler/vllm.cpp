"""Run UPSTREAM's own auto-tiling resolution for the sizes this campaign measured.

No model, no checkpoint: pure ltx_core tiling code at the pinned SHA
fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca.
"""
import sys

sys.path.insert(0, "/home/mudler/_git/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/home/mudler/_git/LTX-2/packages/ltx-pipelines/src")
import torch  # noqa: E402

import ltx_core  # noqa: E402

assert "/home/mudler/_git/LTX-2" in ltx_core.__file__, ltx_core.__file__
from ltx_core.model.video_vae.video_vae import (  # noqa: E402
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
)
from ltx_core.tiling import (  # noqa: E402
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionSizeConfig,
    TileSizeConfig,
    create_tiles,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape  # noqa: E402

SF = SpatioTemporalScaleFactors(time=8, height=32, width=32)
LONG = DimensionSizeConfig(tile_size=768, overlap=64)  # helpers.py:62
FR = DimensionSizeConfig(tile_size=80, overlap=24)  # helpers.py:63


def prepare(latent_shape, cfg):
    """conv_video_decoder.py:359-381 (`ConvVideoDecoder._prepare_tiles`)."""
    splitters = [DEFAULT_SPLIT_OPERATION] * 5
    mappers = [DEFAULT_MAPPING_OPERATION] * 5
    t, h, w = cfg.to_splitters(SF)
    splitters[2], splitters[3], splitters[4] = t, h, w
    if cfg.height.is_tiled():
        mappers[3] = to_mapping_operation(map_spatial_slice, scale=SF.height)
    if cfg.width.is_tiled():
        mappers[4] = to_mapping_operation(map_spatial_slice, scale=SF.width)
    if cfg.frames.is_tiled():
        mappers[2] = to_mapping_operation(map_temporal_slice, scale=SF.time)
    return create_tiles(torch.Size(latent_shape), splitters, mappers)


class _P:
    def __init__(self, f, h, w):
        self.batch, self.frames, self.height, self.width, self.fps = 1, f, h, w, 25.0


for (W, H, F) in [
    (128, 128, 9),
    (320, 192, 25),
    (448, 256, 25),
    (896, 512, 25),
    # The temporal binding point and the two frame counts around it. The first
    # version of this sweep went 25 -> 121 and skipped 81 entirely, which is how
    # "temporal chunking starts at 121 frames" got written down; see the temporal
    # walk below, which is the part that makes skipping it impossible.
    (768, 768, 73),
    (768, 768, 81),
    (1024, 576, 97),
    (1280, 704, 121),
    (1920, 1088, 241),
]:
    cfg = TileSizeConfig.from_long_side(
        long_side=LONG, height=H, width=W, scale_factors=SF, frames=FR
    )
    lat = VideoLatentShape.from_pixel_shape(_P(F, H, W), 128, SF)
    shape = (1, 128, lat.frames, lat.height, lat.width)
    tiles = prepare(shape, cfg)
    groups = group_tiles_by_temporal_slice(tiles)
    full = VideoLatentShape.from_torch_shape(torch.Size(shape)).upscale(SF)
    comp = masks_are_complementary(tiles, full.to_torch_shape())
    peak_frames = max(
        (g[0].out_coords[2].stop - g[0].out_coords[2].start) for g in groups
    )
    peak = peak_frames * 3 * full.height * full.width * 4
    print(
        f"{W}x{H}/{F}f latent={shape[2:]} "
        f"cfg(h={cfg.height.tile_size}/{cfg.height.overlap},"
        f"w={cfg.width.tile_size}/{cfg.width.overlap},"
        f"f={cfg.frames.tile_size}/{cfg.frames.overlap}) "
        f"-> tiles={len(tiles)} groups={len(groups)} complementary={comp} "
        f"chunkbuf={peak / 2**20:.1f} MiB "
        f"fullpix={3 * full.frames * full.height * full.width * 4 / 2**20:.1f} MiB"
    )


# ---------------------------------------------------------------------------
# THE TEMPORAL BINDING POINT, walked rather than sampled.
#
# The resolution sweep above cannot establish a threshold: it visits a handful of
# frame counts, and whichever of them happens to be the first that tiles reads as
# the answer. Its first version went 9, 25, 25, 25, 121, 241 and so recorded
# "temporal chunking starts at 121 frames" — the smallest number it sampled above
# 25. The real answer is 81, and the row's own goldens had it all along.
#
# So this walks the axis one latent frame at a time across the boundary and
# ASSERTS both where the transition is and that it saw both sides of it. A sweep
# that steps over its own binding point now fails instead of publishing a number.
#
# THE DENSITY IS ASSERTED, NOT COMMENTED. `range(1, 137, 8)` is one latent frame
# per step only because `latent_t = (F - 1) // 8 + 1`, and nothing below used to
# check that: changing the step 8 -> 80 left the walk visiting exactly TWO frame
# counts (1 and 81), skipping latent_t 2..10 entirely, and it still printed
# "first splits at 81 frames" and exited 0. `saw_untiled` and
# `first_tiled is not None` cannot see that, because a coarse step that happens to
# land on the boundary satisfies both. So the walk now asserts that consecutive
# steps move latent_t by exactly 1, that it started at latent_t 1, and that the
# step immediately BEFORE the transition was latent_t 10 — the short-circuit's own
# bound. A lucky-but-coarse sweep is now a failure.
# ---------------------------------------------------------------------------

WALK_W, WALK_H = 1024, 576
walk_cfg = TileSizeConfig.from_long_side(
    long_side=LONG, height=WALK_H, width=WALK_W, scale_factors=SF, frames=FR
)
t_split, _h_split, _w_split = walk_cfg.to_splitters(SF)

print()
print(f"temporal walk @ {WALK_W}x{WALK_H}, one latent frame per step:")
first_tiled = None
saw_untiled = False
visited = []  # (frames, latent_t, interval_count), in walk order
for F in range(1, 137, 8):  # (frames - 1) % 8 == 0 — exactly one latent frame per step
    lat_t = (F - 1) // SF.time + 1
    n = len(t_split(lat_t).intervals)
    visited.append((F, lat_t, n))
    if n == 1:
        saw_untiled = True
    elif first_tiled is None:
        first_tiled = F
    mark = "   <<< FIRST FRAME COUNT THAT SPLITS" if F == first_tiled else ""
    print(
        f"  frames={F:4d} latent_t={lat_t:3d} t_intervals={n} "
        f"chunks={walk_cfg.video_chunks_number(F)}{mark}"
    )

# THE WALK IS DENSE. Without this, a step that skips latent depths still satisfies
# every assertion below by landing on the boundary from the outside.
assert visited, "the walk visited no frame counts at all"
assert visited[0][1] == 1, (
    f"the walk starts at latent_t {visited[0][1]}, not 1; it cannot establish that the "
    "untiled side it saw is the whole untiled side"
)
gaps = [
    (prev, cur)
    for prev, cur in zip(visited, visited[1:])
    if cur[1] - prev[1] != 1
]
assert not gaps, (
    f"the walk is NOT dense: latent_t jumps {[(p[1], c[1]) for p, c in gaps]} between "
    f"frame counts {[(p[0], c[0]) for p, c in gaps]}. It visited "
    f"{len(visited)} frame counts covering latent_t "
    f"{[v[1] for v in visited]}. A walk that steps over latent depths can land on the "
    "boundary by luck and report a threshold it never bracketed — which is exactly the "
    "defect this block exists to prevent. Use a step of one latent frame (8 pixel frames). "
    "A jump of 0 is the OPPOSITE failure and is not upstream moving anything: a step FINER "
    "than 8 pixel frames repeats a latent depth, because latent_t = (frames - 1) // 8 + 1 "
    "is constant across 8 consecutive frame counts. Widen the step back to 8; do not "
    "relax this assertion."
)
assert saw_untiled, "the walk never saw an untiled frame count; it cannot locate a boundary"
assert first_tiled is not None, (
    "the walk never saw a tiled frame count; extend the range before trusting a threshold"
)
# ...and the step IMMEDIATELY before the transition is the short-circuit's own
# bound, `latent_t <= 10` (tiling.py:239-240). Asserting the neighbour, not just
# the frame count, is what ties the number to the mechanism that produces it.
_idx = [i for i, v in enumerate(visited) if v[0] == first_tiled][0]
assert _idx > 0, "the first tiled frame count is the walk's first step; it bracketed nothing"
assert visited[_idx - 1][1] == 10 and visited[_idx][1] == 11, (
    f"the transition is between latent_t {visited[_idx - 1][1]} and {visited[_idx][1]}, "
    "not 10 -> 11; `split_temporal_causal` short-circuits while latent_t <= 10 "
    "(tiling.py:239-240), so either upstream moved that bound or this walk is not "
    "measuring it"
)
assert first_tiled == 81, (
    f"the temporal axis first splits at {first_tiled} frames, not 81 — reconcile "
    "ltx2_tiling.h, ltx2_video.cpp, docs/USAGE.md, docs/FEATURES.md and "
    ".agents/specs/ltx25-tiled-decode.md before publishing a new number"
)
print(
    f"  -> the temporal axis first splits at {first_tiled} frames "
    f"(latent_t {visited[_idx - 1][1]} -> {visited[_idx][1]}, walked densely over "
    f"{len(visited)} frame counts)"
)
