#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_dfr_goldens.inc — the LTX-2.5 DFR LAYOUT oracle.

Row LTX25-DFR-PIPELINE, issue #986, spec .agents/specs/ltx25-dfr-pipeline.md.

Everything below is produced by IMPORTING AND EXECUTING upstream's own
`ltx_pipelines.dfr_layout` and the three pure-index helpers in
`ltx_pipelines.dfr_pipeline`. Nothing is restated from memory and nothing is
derived by reading the source: each number here is what the module RETURNED.

The distinction matters more for this module than for a numeric kernel. Every
value here is an INDEX, so a port that gets one wrong still produces a correctly
shaped, finite, plausible latent — there is no NaN, no shape failure and no
magnitude to notice. The exact vectors are the only instrument.

Upstream source (EXECUTED):
    packages/ltx-pipelines/src/ltx_pipelines/dfr_layout.py     (imported)
    packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py   (three helpers, see below)

WHY dfr_pipeline IS NOT IMPORTED, AND WHAT IS RUN INSTEAD. `import
ltx_pipelines.dfr_pipeline` pulls `iclora_utils` -> `utils.media_io` ->
`ltx_core.color.hlg`, which imports PyAV. That is a codec binding this project
deliberately does not vendor — the same dependency the `image_crf` round trip is
refused over — and installing it to read three functions of pure index
arithmetic would make the oracle depend on a media stack none of these values
touch.

So the three helpers and the three module constants are lifted from the real
file by AST and their OWN SOURCE TEXT is executed. This is upstream's code
running, not a transcription: the extraction is by name from the parsed module,
each name is asserted present, and the source slice is compiled verbatim. If
upstream renames or removes one, this generator raises rather than silently
falling back to a local definition. `dfr_layout` itself is imported normally,
because it only needs torch and `ltx_core.types`.

Regenerate with:
    python3 scripts/gen-ltx2-dfr-goldens.py --ltx2 <LTX-2 checkout> \
        --out tests/vllm/models/ltx2_dfr_goldens.inc
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys


def git_revision(root: pathlib.Path) -> str:
    """The checkout's SHA, and REFUSE a dirty tree.

    A SHA that does not describe the code that ran is worse than no SHA: it reads
    as a pin while the oracle is whatever was in the working tree.
    """
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"REFUSING: {root} is dirty, so {head} would not describe the executed code:\n{dirty}"
        )
    return head


def load_dfr_pipeline_pieces(path: pathlib.Path, namespace: dict) -> dict:
    """Execute the named helpers and constants out of `dfr_pipeline.py` itself.

    Parses the real file, slices the source of each requested top-level name, and
    compiles that slice verbatim into `namespace`. Every name is asserted
    present, so an upstream rename raises here rather than leaving this generator
    quietly running a local definition of upstream's function — which would make
    the "oracle" a copy of the thing under test.
    """
    wanted_funcs = ("_slot_initials_from_video", "_merge_carry_forward_keyframes")
    wanted_consts = ("_ANCHOR_KEYFRAME_STRENGTH", "_TEMPORAL_ANCESTRAL_ETA", "_MAX_CONDITIONING_FPS")

    source = path.read_text()
    tree = ast.parse(source)
    found: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_consts:
                    found[target.id] = node

    missing = [n for n in (*wanted_funcs, *wanted_consts) if n not in found]
    if missing:
        raise SystemExit(
            f"REFUSING: {path} no longer defines {missing} at module level. The generator will "
            "not substitute a local definition for an upstream one."
        )

    for name in (*wanted_consts, *wanted_funcs):
        node = found[name]
        slice_src = ast.get_source_segment(source, node)
        exec(compile(slice_src, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def cxx_i64_array(name: str, values) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return (
        f"inline constexpr int64_t {name}[] = {{{body}}};\n"
        f"inline constexpr int64_t {name}Count = {len(list(values))};\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ltx2", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    root = args.ltx2.resolve()
    revision = git_revision(root)

    sys.path.insert(0, str(root / "packages" / "ltx-pipelines" / "src"))
    sys.path.insert(0, str(root / "packages" / "ltx-core" / "src"))

    import torch  # noqa: PLC0415
    from ltx_core.types import VIDEO_SCALE_FACTORS  # noqa: PLC0415
    from ltx_pipelines import dfr_layout  # noqa: PLC0415

    pieces = load_dfr_pipeline_pieces(
        root / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "dfr_pipeline.py",
        {"torch": torch, "Sequence": __import__("collections.abc", fromlist=["Sequence"]).Sequence},
    )
    _ANCHOR_KEYFRAME_STRENGTH = pieces["_ANCHOR_KEYFRAME_STRENGTH"]
    _TEMPORAL_ANCESTRAL_ETA = pieces["_TEMPORAL_ANCESTRAL_ETA"]
    _MAX_CONDITIONING_FPS = pieces["_MAX_CONDITIONING_FPS"]
    _slot_initials_from_video = pieces["_slot_initials_from_video"]
    _merge_carry_forward_keyframes = pieces["_merge_carry_forward_keyframes"]

    out: list[str] = []
    add = out.append

    add("// GENERATED by scripts/gen-ltx2-dfr-goldens.py — DO NOT EDIT BY HAND.")
    add("//")
    add("// LTX-2.5 DFR CANVAS LAYOUT goldens (row LTX25-DFR-PIPELINE, issue #986),")
    add("// produced by IMPORTING AND EXECUTING upstream's `ltx_pipelines.dfr_layout`")
    add("// and the three pure-index helpers of `ltx_pipelines.dfr_pipeline`.")
    add("//")
    add("// Every value here is an INDEX. A port that gets one wrong still produces a")
    add("// correctly shaped, finite, plausible latent, so these exact vectors are the")
    add("// only instrument that can see the defect — which is why the suite compares")
    add("// them element-wise rather than checking a property.")
    add("//")
    add(f"// Upstream revision (Lightricks/LTX-2): {revision}")
    add("#pragma once")
    add("")
    add("#include <cstdint>")
    add("")
    add("namespace vllm_test {")
    add("")
    add("// The upstream tree these numbers came from. The suite asserts this equals the")
    add("// SHA it pins, so regenerating against a DIFFERENT checkout fails the gate")
    add("// instead of silently replacing the oracle. The generator REFUSES a dirty")
    add("// tree, so this SHA always describes the code that ran.")
    add(f'inline constexpr const char* kLtx2DfrUpstreamRevision = "{revision}";')
    add("")

    scale = int(VIDEO_SCALE_FACTORS.time)
    add("// VIDEO_SCALE_FACTORS.time, read off the executed module rather than restated.")
    add(f"inline constexpr int64_t kLtx2DfrTemporalScale = {scale};")
    add("")

    # ---------------------------------------------------------------- section 1
    add("// --------------------------------------------------------------------------")
    add("// 1. SEGMENT_CANDIDATES and choose_segment_length (dfr_layout.py:12, :40-57)")
    add("// --------------------------------------------------------------------------")
    add("//")
    add("// The candidates are read off the module's own tuple, so a candidate added")
    add("// upstream changes this gate rather than sliding past it.")
    add(cxx_i64_array("kLtx2DfrSegmentCandidatesGolden", dfr_layout.SEGMENT_CANDIDATES))
    add(f"inline constexpr int64_t kLtx2DfrTileLeadSegmentsGolden = {int(dfr_layout.TILE_LEAD_SEGMENTS)};")
    add("")

    # Content counts chosen so the set contains, deliberately:
    #   - a TIE that must resolve to the LARGER candidate,
    #   - an exact multiple of each candidate,
    #   - counts where each candidate wins outright.
    content_cases = [8, 24, 32, 48, 64, 72, 96, 120, 128, 144, 168, 192, 200, 240, 264, 288]
    segments = [dfr_layout.choose_segment_length(c) for c in content_cases]
    pads = [
        [(cand - c % cand) % cand for cand in dfr_layout.SEGMENT_CANDIDATES]
        for c in content_cases
    ]
    add("// `content_frames` inputs and the segment upstream chose for each. The set is")
    add("// built to contain a TIE (both pads equal), an exact multiple of each")
    add("// candidate, and counts where each candidate wins outright — so a port that")
    add("// resolved the tie the other way, or that seeded its running best with the")
    add("// largest candidate, is separated from a correct one by this table alone.")
    add(cxx_i64_array("kLtx2DfrSegmentContent", content_cases))
    add(cxx_i64_array("kLtx2DfrSegmentChosen", segments))
    add("// The pad each candidate would take, in SEGMENT_CANDIDATES order. Emitted so")
    add("// the suite can assert that a tie is PRESENT in the table rather than trusting")
    add("// that one was included — an intent that no longer holds is the failure this")
    add("// campaign has already paid for.")
    flat_pads = [p for row in pads for p in row]
    add(cxx_i64_array("kLtx2DfrSegmentPads", flat_pads))
    add(f"inline constexpr int64_t kLtx2DfrSegmentCandidateCount = {len(dfr_layout.SEGMENT_CANDIDATES)};")
    add("")

    # ---------------------------------------------------------------- section 2
    add("// --------------------------------------------------------------------------")
    add("// 2. resolve_canvas (dfr_layout.py:60-81)")
    add("// --------------------------------------------------------------------------")
    canvas_cases = [f * scale + 1 for f in (1, 3, 4, 6, 9, 12, 15, 16, 24, 25, 30, 33)]
    canvas_frames, canvas_segment, canvas_pos_flat, canvas_pos_counts = [], [], [], []
    for n in canvas_cases:
        padded, segment, positions = dfr_layout.resolve_canvas(n)
        canvas_frames.append(padded)
        canvas_segment.append(segment)
        canvas_pos_counts.append(len(positions))
        canvas_pos_flat.extend(positions)
    add("// Requested frame counts, all satisfying `(n - 1) % 8 == 0`, and what")
    add("// `resolve_canvas` returned for each: the PADDED canvas, the segment, and the")
    add("// keyframe positions flattened with a per-case count.")
    add(cxx_i64_array("kLtx2DfrCanvasRequest", canvas_cases))
    add(cxx_i64_array("kLtx2DfrCanvasFrames", canvas_frames))
    add(cxx_i64_array("kLtx2DfrCanvasSegment", canvas_segment))
    add(cxx_i64_array("kLtx2DfrCanvasPositionCount", canvas_pos_counts))
    add(cxx_i64_array("kLtx2DfrCanvasPositions", canvas_pos_flat))
    add("")

    # ---------------------------------------------------------------- section 3
    add("// --------------------------------------------------------------------------")
    add("// 3. tile_ranges (dfr_layout.py:137-182)")
    add("// --------------------------------------------------------------------------")
    add("//")
    add("// One flattened record per tile, in tile order, with a per-case tile count.")
    add("// The two keyframe bags are flattened separately with their own counts,")
    add("// because their lengths differ per tile and a single interleaved stream would")
    add("// make an off-by-one in one bag readable as a correct value in the other.")

    # Each case is (requested_frames, rounds) put through the SAME arithmetic the
    # pipeline uses, so the seam positions are the ones a real round would carry.
    tile_cases = []
    for requested in (97, 121, 193, 241):
        for rounds in (1, 2):
            padded, _segment, positions = dfr_layout.resolve_canvas(requested)
            # dfr_pipeline.py:408-411 — the round doubles the canvas and the
            # carried positions before it tiles.
            round_frames = 2 * (padded - 1) + 1
            seams = [2 * p for p in positions]
            tile_cases.append((requested, rounds, round_frames, seams, 2 ** rounds))

    case_request, case_rounds, case_frames, case_numtiles, case_tilecount = [], [], [], [], []
    seam_flat, seam_counts = [], []
    px_start, px_end, lat_start, lat_end, drop = [], [], [], [], []
    anchor_flat, anchor_counts, slot_flat, slot_counts = [], [], [], []

    for requested, rounds, frames, seams, num_tiles in tile_cases:
        tiles = dfr_layout.tile_ranges(seams, frames, num_tiles)
        case_request.append(requested)
        case_rounds.append(rounds)
        case_frames.append(frames)
        case_numtiles.append(num_tiles)
        case_tilecount.append(len(tiles))
        seam_counts.append(len(seams))
        seam_flat.extend(seams)
        for tile in tiles:
            px_start.append(tile.pixel_start)
            px_end.append(tile.pixel_end)
            lat_start.append(tile.latent_start)
            lat_end.append(tile.latent_end_exclusive)
            drop.append(tile.drop_latent_prefix)
            anchor_counts.append(len(tile.anchor_kf_global))
            anchor_flat.extend(tile.anchor_kf_global)
            slot_counts.append(len(tile.slot_kf_global))
            slot_flat.extend(tile.slot_kf_global)

    add(cxx_i64_array("kLtx2DfrTileCaseRequest", case_request))
    add(cxx_i64_array("kLtx2DfrTileCaseRounds", case_rounds))
    add(cxx_i64_array("kLtx2DfrTileCaseFrames", case_frames))
    add(cxx_i64_array("kLtx2DfrTileCaseNumTiles", case_numtiles))
    add(cxx_i64_array("kLtx2DfrTileCaseTileCount", case_tilecount))
    add(cxx_i64_array("kLtx2DfrTileCaseSeamCount", seam_counts))
    add(cxx_i64_array("kLtx2DfrTileCaseSeams", seam_flat))
    add(cxx_i64_array("kLtx2DfrTilePixelStart", px_start))
    add(cxx_i64_array("kLtx2DfrTilePixelEnd", px_end))
    add(cxx_i64_array("kLtx2DfrTileLatentStart", lat_start))
    add(cxx_i64_array("kLtx2DfrTileLatentEndExclusive", lat_end))
    add(cxx_i64_array("kLtx2DfrTileDropLatentPrefix", drop))
    add(cxx_i64_array("kLtx2DfrTileAnchorCount", anchor_counts))
    add(cxx_i64_array("kLtx2DfrTileAnchors", anchor_flat))
    add(cxx_i64_array("kLtx2DfrTileSlotCount", slot_counts))
    add(cxx_i64_array("kLtx2DfrTileSlots", slot_flat))
    add("")

    # ---------------------------------------------------------------- section 4
    add("// --------------------------------------------------------------------------")
    add("// 4. stitch_tile_latents (dfr_layout.py:185-208)")
    add("// --------------------------------------------------------------------------")
    add("//")
    add("// The stitch is gated on IDENTITY rather than on a value: each tile latent is")
    add("// filled with its own GLOBAL latent index, so the stitched result reads back")
    add("// as the exact latent frames the canvas should contain. A misplaced tile, a")
    add("// dropped prefix of the wrong size or a reversed concatenation all move a")
    add("// value that the suite can name, where a random fill would only move a digest.")

    stitch_request, stitch_rounds = 121, 1
    padded, _seg, positions = dfr_layout.resolve_canvas(stitch_request)
    stitch_frames = 2 * (padded - 1) + 1
    stitch_seams = [2 * p for p in positions]
    stitch_tiles = dfr_layout.tile_ranges(stitch_seams, stitch_frames, 2 ** stitch_rounds)
    channels, height, width = 2, 1, 2
    tile_tensors = []
    for tile in stitch_tiles:
        t = tile.latent_end_exclusive - tile.latent_start
        # value = global latent index * 1000 + channel * 10 + (h * width + w),
        # so every axis is separable in the readback.
        data = torch.zeros(1, channels, t, height, width, dtype=torch.float32)
        for local in range(t):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        data[0, c, local, h, w] = float(
                            (tile.latent_start + local) * 1000 + c * 10 + (h * width + w)
                        )
        tile_tensors.append(data)
    stitched = dfr_layout.stitch_tile_latents(tile_tensors, stitch_tiles)

    add(f"inline constexpr int64_t kLtx2DfrStitchRequest = {stitch_request};")
    add(f"inline constexpr int64_t kLtx2DfrStitchFrames = {stitch_frames};")
    add(f"inline constexpr int64_t kLtx2DfrStitchChannels = {channels};")
    add(f"inline constexpr int64_t kLtx2DfrStitchHeight = {height};")
    add(f"inline constexpr int64_t kLtx2DfrStitchWidth = {width};")
    add(f"inline constexpr int64_t kLtx2DfrStitchOutFrames = {int(stitched.shape[2])};")
    add("// The stitched volume, [B, C, T, H, W] row-major and flattened in that order.")
    add(cxx_i64_array("kLtx2DfrStitchExpected",
                      [int(v) for v in stitched.reshape(-1).tolist()]))
    add("")

    # ---------------------------------------------------------------- section 5
    add("// --------------------------------------------------------------------------")
    add("// 5. _slot_initials_from_video (dfr_pipeline.py:101-111)")
    add("// --------------------------------------------------------------------------")
    add("//")
    add("// Same identity fill: the seed volume reads back as the latent index each slot")
    add("// selected, so the assertion names the FRAME rather than a digest of it.")
    add("//")
    add("// The positions include 52, 60 and 68, whose quotients are 6.5, 7.5 and 8.5.")
    add("// Those three SEPARATE the three plausible ports: Python's `round` is")
    add("// half-to-EVEN and gives 6, 8, 8; C's `std::round` is half-away-from-zero and")
    add("// gives 7, 8, 9; a floor gives 6, 7, 8. No two agree on all three, so the")
    add("// golden names the rounding mode rather than merely being consistent with it.")
    add("// No canvas this engine can build reaches a half quotient — every slot position")
    add("// is an odd multiple of a segment length and both candidates divide by 8 — so")
    add("// this is the only place the distinction is observable at all.")
    seed_frames = 12
    seed_positions = [0, 8, 24, 40, 52, 60, 68, 95, 200]
    seed_video = torch.zeros(1, 1, seed_frames, 1, 1, dtype=torch.float32)
    for f in range(seed_frames):
        seed_video[0, 0, f, 0, 0] = float(f)
    seeds = _slot_initials_from_video(seed_video, seed_positions, scale)
    add(f"inline constexpr int64_t kLtx2DfrSeedFrames = {seed_frames};")
    add(cxx_i64_array("kLtx2DfrSeedPositions", seed_positions))
    add("// The latent frame index each position selected, INCLUDING the clamp: 200/8 is")
    add("// 25, past the 12-frame window, and upstream clamps to T-1 rather than raising.")
    add(cxx_i64_array("kLtx2DfrSeedSelected",
                      [int(v) for v in seeds.reshape(-1).tolist()]))
    add("")

    # ---------------------------------------------------------------- section 6
    add("// --------------------------------------------------------------------------")
    add("// 6. _merge_carry_forward_keyframes (dfr_pipeline.py:114-139)")
    add("// --------------------------------------------------------------------------")
    add("//")
    add("// The case DELIBERATELY overlaps a slot onto an anchor position, because that")
    add("// overlap is the only thing that distinguishes upstream's insertion order from")
    add("// its reverse. Every shape and every count is identical either way; only the")
    add("// CONTENT at the shared position moves, and it moves to a value that is still")
    add("// a valid keyframe latent.")
    anchor_positions = [24, 48, 72]
    slot_positions = [12, 48, 60]
    anchors = torch.zeros(1, 1, len(anchor_positions), 1, 1, dtype=torch.float32)
    slots = torch.zeros(1, 1, len(slot_positions), 1, 1, dtype=torch.float32)
    for i, p in enumerate(anchor_positions):
        anchors[0, 0, i, 0, 0] = float(1000 + p)   # anchors tagged 1000 + position
    for i, p in enumerate(slot_positions):
        slots[0, 0, i, 0, 0] = float(2000 + p)     # slots tagged 2000 + position
    merged_positions, merged = _merge_carry_forward_keyframes(
        anchor_positions, anchors, slot_positions, slots
    )
    add(cxx_i64_array("kLtx2DfrMergeAnchorPositions", anchor_positions))
    add(cxx_i64_array("kLtx2DfrMergeSlotPositions", slot_positions))
    add("// Anchors carry `1000 + position`, slots carry `2000 + position`, so the tag at")
    add("// the shared position 48 says which side won.")
    add(cxx_i64_array("kLtx2DfrMergePositions", merged_positions))
    add(cxx_i64_array("kLtx2DfrMergeValues",
                      [int(v) for v in merged.reshape(-1).tolist()]))
    add("")

    # ---------------------------------------------------------------- section 7
    add("// --------------------------------------------------------------------------")
    add("// 7. The pipeline constants (dfr_pipeline.py:72-78, :534)")
    add("// --------------------------------------------------------------------------")
    add("//")
    add("// Read off the executed module's own module-level names, not retyped. The")
    add("// ancestral eta in particular is NOT the distilled sampler's 1.0, and a port")
    add("// that reused that constant would inject twice the noise into a refinement")
    add("// that cannot remove it.")
    add(f"inline constexpr double kLtx2DfrAnchorStrengthGolden = {_ANCHOR_KEYFRAME_STRENGTH!r};")
    add(f"inline constexpr double kLtx2DfrTemporalEtaGolden = {_TEMPORAL_ANCESTRAL_ETA!r};")
    add(f"inline constexpr double kLtx2DfrMaxConditioningFpsGolden = {_MAX_CONDITIONING_FPS!r};")

    add("// `(requested - 1) * 2**rounds + 1` (dfr_pipeline.py:534), evaluated upstream's")
    add("// way for each (requested, rounds) pair.")
    trim_requested, trim_rounds, trim_target = [], [], []
    for requested in (97, 121, 193, 241):
        for rounds in (0, 1, 2):
            trim_requested.append(requested)
            trim_rounds.append(rounds)
            trim_target.append((requested - 1) * 2 ** rounds + 1)
    add(cxx_i64_array("kLtx2DfrTrimRequested", trim_requested))
    add(cxx_i64_array("kLtx2DfrTrimRounds", trim_rounds))
    add(cxx_i64_array("kLtx2DfrTrimTarget", trim_target))
    add("")

    add("}  // namespace vllm_test")
    add("")

    args.out.write_text("\n".join(out))
    print(f"wrote {args.out} ({len(out)} lines) from {root} @ {revision}")


if __name__ == "__main__":
    main()
