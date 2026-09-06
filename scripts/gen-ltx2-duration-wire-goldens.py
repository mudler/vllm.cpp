#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_duration_wire_goldens.inc — the LTX-2.5 duration
head's DRIVER, as opposed to its arithmetic.

Row LTX25-DURATION-HEAD-WIRE, issue #2900,
spec .agents/specs/ltx25-duration-head-wire.md.

`Ltx2DurationPredict` — the head's forward — is gated elsewhere and this script
does not touch it. What it pins is everything upstream wraps AROUND that
forward, which is where the frame count a render actually uses comes from:
`snap_frames_to_grid` and `seconds_to_clamped_num_frames`
(ltx_pipelines/utils/helpers.py:554,565) and `DurationPredictor.__call__`
(ltx_pipelines/utils/blocks.py:850).

EVERY NUMBER BELOW IS WHAT UPSTREAM'S OWN CODE RETURNED. The two helpers and
`DurationPredictor` are imported from a Lightricks/LTX-2 checkout and CALLED.
Nothing is transcribed from reading the source and nothing is recomputed by a
local reimplementation, because for this module the values that matter most are
the ones a plausible implementation gets wrong.

THE REJECTED HYPOTHESIS RIDES BESIDE THE ANSWER. A single-hypothesis golden
cannot see an error that lands equidistant from the correct rule and a wrong
one, so each case here also carries what the NEAREST WRONG RULE would have
returned, and the emitter REFUSES TO WRITE when the two agree. A probe at a
convenient scale where the rules coincide is a mute switch, not a test.

THE THREE RULES THIS EXISTS TO SEPARATE, each of which type-checks both ways:

1. THE CLAMP PRECEDES THE SNAP (helpers.py:579-580). Snapping to the grid first
   and clamping afterwards is the same two operations in the other order, and it
   disagrees for every prediction whose floored grid point leaves the window.
2. THE UNDERSHOOT REPAIR SNAPS UP, AND IS ITSELF CAPPED (helpers.py:581-584).
   `snap_frames_to_grid` FLOORS, so a `min_frames` that is not already a grid
   point produces a count BELOW the window the caller asked for. Upstream
   recovers with a ceiling division back onto the grid and then takes `min` with
   `max_frames`. An implementation that floors and stops returns 1 where
   upstream returns 9.
3. `round`, NOT TRUNCATION (helpers.py:578). `round(seconds * frame_rate)` and
   a C++ cast-to-int agree on about half of all inputs.
4. THAT `round` IS BANKER'S ROUNDING, NOT `std::llround`. Python rounds a half
   to EVEN and `llround` rounds it away from zero, and the difference is not a
   measure-zero curiosity: 0.34 s at 25 fps is exactly 8.5, where upstream takes
   8 and returns frame 1 while `llround` takes 9 and returns frame 9. The
   obvious C++ spelling is wrong by eight frames on that request.

DETERMINISM (#2855). The head runs under `torch.no_grad` on a single thread
pinned by `torch.set_num_threads(1)`, and the synthetic weights are a closed
form of the parameter index rather than a draw from any RNG. The file this
emits is therefore byte-identical at any OMP_NUM_THREADS; the spec's `## Gates`
regenerates at 1 and at 8 and compares.

Regenerate with:

    python3 scripts/gen-ltx2-duration-wire-goldens.py --ltx2 /path/to/LTX-2 \
        --out tests/vllm/models/ltx2_duration_wire_goldens.inc

and diff. The committed file is what this emits at `fd4ded7f`; a difference is
either an upstream change or a defect in one of the two.
"""

from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

PIN = "fd4ded7f"


def load_upstream(root: Path):
    """Import the pinned modules, stubbing the two media-io packages nothing numeric touches."""
    for pkg in ("ltx-core", "ltx-pipelines", "ltx-kernels"):
        sys.path.insert(0, str(root / "packages" / pkg / "src"))
    # `ltx_pipelines.utils.helpers` reaches PyAV and OpenImageIO through the color
    # and image paths. Neither is vendored here and no number crosses either; if
    # upstream ever routes one through them, this stub is what breaks LOUDLY
    # rather than a value silently changing.
    for name in ("av", "OpenImageIO"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import torch

    torch.set_num_threads(1)
    from ltx_core.duration_head import DurationHead, DurationHeadConfigurator
    from ltx_pipelines.utils.blocks import DurationPredictor, require_num_frames_source
    from ltx_pipelines.utils.helpers import seconds_to_clamped_num_frames, snap_frames_to_grid
    from ltx_pipelines.utils.types import AutoDuration

    return types.SimpleNamespace(
        torch=torch,
        DurationHead=DurationHead,
        DurationHeadConfigurator=DurationHeadConfigurator,
        DurationPredictor=DurationPredictor,
        require_num_frames_source=require_num_frames_source,
        seconds_to_clamped_num_frames=seconds_to_clamped_num_frames,
        snap_frames_to_grid=snap_frames_to_grid,
        AutoDuration=AutoDuration,
    )


# ── The synthetic head ───────────────────────────────────────────────────────
#
# A CLOSED FORM OF THE PARAMETER INDEX, not a draw. `torch.randn` would tie this
# file to a generator's stream across two torch versions; `sin` of the flat index
# is reproducible anywhere and is still dense enough that no projection collapses.
# The C++ side builds the same bag from the same expression, so the two agree on
# the WEIGHTS before they are asked to agree on the ANSWER.
#
# THE AMPLITUDE WAS CHOSEN BY SWEEPING IT, and it is the difference between a
# fixture and a mute switch. At 0.05 the head predicts 1.0839 s for EVERY case --
# the MLP output sits so close to zero that `exp` returns almost 1 whatever the
# tokens are, and four of the five cases then snap to the same frame count. At
# 0.5 the pre-exponential blows past 25 and one case returns 8.2e10 s, which the
# clamp flattens to `max_frames` and which would gate the clamp rather than the
# head. 0.2 spans 7.17 s to 9.34 s and 97 to 217 frames. `assert_discriminating`
# below re-checks that at generation time rather than trusting this paragraph.
def fill_synthetic(up, head) -> None:
    torch = up.torch
    with torch.no_grad():
        for slot, (_name, param) in enumerate(head.named_parameters()):
            n = param.numel()
            idx = torch.arange(n, dtype=torch.float64)
            values = torch.sin(idx * 0.7391 + float(slot) * 1.13) * 0.2
            param.copy_(values.reshape(param.shape).to(param.dtype))


def synthetic_tokens(up, count: int, width: int, phase: float):
    torch = up.torch
    idx = torch.arange(count * width, dtype=torch.float64)
    return (torch.sin(idx * 0.013 + phase) * 0.5).reshape(1, count, width).to(torch.float32)


# ── The rejected rules, written out so the emitter can DISAGREE with them ────
def rejected_snap_then_clamp(seconds, frame_rate, min_frames, max_frames, time_scale):
    """Rule 1's alternative: snap first, then clamp. Same two steps, other order."""
    raw = round(seconds * frame_rate)
    if raw < 1:
        raw = 1
    snapped = ((raw - 1) // time_scale) * time_scale + 1
    return max(min_frames, min(snapped, max_frames))


def rejected_no_undershoot_repair(seconds, frame_rate, min_frames, max_frames, time_scale):
    """Rule 2's alternative: floor to the grid and stop, leaving the window broken."""
    raw = round(seconds * frame_rate)
    raw = max(min_frames, min(raw, max_frames))
    return ((raw - 1) // time_scale) * time_scale + 1


def rejected_half_away(seconds, frame_rate, min_frames, max_frames, time_scale):
    """Rule 4's alternative: round half AWAY FROM ZERO, which is `std::llround`.

    Python's `round` is BANKER'S rounding -- half to EVEN -- and C++'s `llround`
    is half away from zero. They differ only when `seconds * frame_rate` lands
    exactly on a half-integer, which sounds like a measure-zero curiosity and is
    not: 0.34 s at 25 fps is exactly 8.5, where upstream floors to 8 and lands on
    frame 1, and `llround` takes 9 and lands on frame 9. A port that reached for
    the obvious C++ spelling would be wrong by eight frames on that request.
    """
    product = seconds * frame_rate
    raw = math.floor(product + 0.5) if product >= 0 else math.ceil(product - 0.5)
    raw = max(min_frames, min(int(raw), max_frames))
    frames = ((raw - 1) // time_scale) * time_scale + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


def rejected_truncate(seconds, frame_rate, min_frames, max_frames, time_scale):
    """Rule 3's alternative: truncate toward zero where upstream rounds."""
    raw = int(seconds * frame_rate)
    raw = max(min_frames, min(raw, max_frames))
    frames = ((raw - 1) // time_scale) * time_scale + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames


def assert_discriminating(raw_seconds: list[float], frames: list[int]) -> None:
    """Refuse a predict table that cannot see the chain it claims to gate.

    Two ways this fixture dies quietly. The head can go input-insensitive, so
    every case returns one duration and the table gates `exp` and nothing else.
    Or it can saturate, so every case leaves the clamp window and the table gates
    the clamp. Both look exactly like a passing test.
    """
    if len(set(frames)) < 3:
        raise SystemExit(
            f"REFUSING TO EMIT: the predict cases produce only {len(set(frames))} distinct "
            f"frame counts ({sorted(set(frames))}); this table gates the snap, not the head"
        )
    for value in raw_seconds:
        if not math.isfinite(value) or not (0.05 < value < 1e3):
            raise SystemExit(
                f"REFUSING TO EMIT: a raw prediction is {value}, outside the range where the "
                "clamp is exercised rather than saturated"
            )
    spread = max(raw_seconds) - min(raw_seconds)
    if spread < 0.5:
        raise SystemExit(
            f"REFUSING TO EMIT: the head's predictions span only {spread:.6g} s across every "
            "case, so the predict table cannot distinguish the forward from a constant. At a "
            "weight amplitude of 0.05 this spread is 4e-4 s and four cases still land on "
            "different frame counts purely through the frame RATE -- which is why the count "
            "check above is not enough on its own"
        )


TIME_SCALE = 8  # VIDEO_SCALE_FACTORS.time (ltx_core/types.py:70)


def cpp_lines(up) -> list[str]:
    out: list[str] = []
    add = out.append

    add("// GENERATED by scripts/gen-ltx2-duration-wire-goldens.py. DO NOT EDIT.")
    add(f"// Upstream: Lightricks/LTX-2 @ {PIN}, executed -- not transcribed.")
    add("// Row LTX25-DURATION-HEAD-WIRE, issue #2900.")
    add("")

    # ── T1: snap_frames_to_grid ──────────────────────────────────────────────
    add("// snap_frames_to_grid (utils/helpers.py:554-562), time_scale = 8.")
    add("// `rejected` is the CEILING to the same grid: it agrees with upstream on")
    add("// every input that is already a grid point, which is why the emitter")
    add("// refuses a case table where only such points appear.")
    add("struct Ltx2SnapCase { int64_t frames; int64_t expected; int64_t rejected_ceil; };")
    add("inline constexpr Ltx2SnapCase kLtx2SnapCases[] = {")
    snap_separating = 0
    for frames in (1, 2, 7, 8, 9, 10, 16, 17, 24, 25, 33, 64, 65, 121, 200, 201, 1024):
        got = up.snap_frames_to_grid(frames)
        ceil_rule = -(-(frames - 1) // TIME_SCALE) * TIME_SCALE + 1
        if got != ceil_rule:
            snap_separating += 1
        add(f"    {{{frames}, {got}, {ceil_rule}}},")
    add("};")
    add(f"inline constexpr int kLtx2SnapSeparatingCases = {snap_separating};")
    add("")
    if snap_separating == 0:
        raise SystemExit("REFUSING TO EMIT: no snap case separates the floor rule from the ceiling rule")

    # ── T3/T4: seconds_to_clamped_num_frames ─────────────────────────────────
    add("// seconds_to_clamped_num_frames (utils/helpers.py:565-585).")
    add("// Three rejected rules ride beside each answer: clamping AFTER the snap,")
    add("// omitting the undershoot repair, and truncating where upstream rounds.")
    add("// A case where all four agree proves nothing, so the counters below say")
    add("// how many cases actually separate each pair and the test asserts they")
    add("// are non-zero -- a fixture that stopped discriminating fails loudly")
    add("// instead of passing quietly.")
    add("struct Ltx2ClampCase {")
    add("    double seconds; double frame_rate; int64_t min_frames; int64_t max_frames;")
    add("    int64_t expected; int64_t rejected_snap_first; int64_t rejected_no_repair;")
    add("    int64_t rejected_truncate; int64_t rejected_half_away;")
    add("};")
    add("inline constexpr Ltx2ClampCase kLtx2ClampCases[] = {")

    cases = [
        # (seconds, frame_rate, min_frames, max_frames)
        (3.0, 24.0, 24, 480),      # ordinary
        (0.1, 24.0, 24, 480),      # below the window: clamp fires low
        (99.0, 24.0, 24, 480),     # above the window: clamp fires high
        (1.0, 24.0, 24, 480),      # exactly min_seconds
        (20.0, 24.0, 24, 480),     # exactly max_seconds
        (0.5, 30.0, 5, 480),       # min_frames OFF the grid -> undershoot repair
        (0.01, 25.0, 3, 9),        # repair capped by a tight max_frames
        (0.01, 25.0, 5, 5),        # degenerate window, min == max, both off grid
        (2.5, 24.0, 1, 1024),      # wide window, mid-grid
        (1.9583333, 24.0, 1, 1024),  # 46.999992 -> round vs truncate DISAGREE
        (0.9999, 25.0, 1, 1024),   # 24.9975 -> rounds to 25, truncates to 24
        (7.0, 30.0, 30, 240),      # 210 -> floors to 209
        (13.7, 24.0, 24, 240),     # 328.8 -> clamped to 240 -> 233
        (4.0, 8.0, 8, 64),         # frame_rate == time_scale
        # EXACT HALF-INTEGERS, where banker's rounding and `llround` part. These
        # are the cases a port that reached for the obvious C++ spelling fails.
        (0.34, 25.0, 1, 1024),     # 8.5 -> upstream 8 (even), llround 9
        (0.5, 49.0, 1, 1024),      # 24.5 -> upstream 24 (even), llround 25
        (0.5, 25.0, 1, 1024),      # 12.5 -> upstream 12 (even), llround 13
        (1.7, 25.0, 1, 1024),      # 42.5 -> upstream 42 (even), llround 43
    ]
    sep_snap_first = sep_no_repair = sep_truncate = sep_half_away = 0
    for seconds, fps, lo, hi in cases:
        got = up.seconds_to_clamped_num_frames(
            seconds, frame_rate=fps, min_frames=lo, max_frames=hi
        )
        r1 = rejected_snap_then_clamp(seconds, fps, lo, hi, TIME_SCALE)
        r2 = rejected_no_undershoot_repair(seconds, fps, lo, hi, TIME_SCALE)
        r3 = rejected_truncate(seconds, fps, lo, hi, TIME_SCALE)
        r4 = rejected_half_away(seconds, fps, lo, hi, TIME_SCALE)
        sep_snap_first += got != r1
        sep_no_repair += got != r2
        sep_truncate += got != r3
        sep_half_away += got != r4
        add(f"    {{{seconds!r}, {fps!r}, {lo}, {hi}, {got}, {r1}, {r2}, {r3}, {r4}}},")
    add("};")
    add(f"inline constexpr int kLtx2ClampSeparatingSnapFirst = {sep_snap_first};")
    add(f"inline constexpr int kLtx2ClampSeparatingNoRepair = {sep_no_repair};")
    add(f"inline constexpr int kLtx2ClampSeparatingTruncate = {sep_truncate};")
    add(f"inline constexpr int kLtx2ClampSeparatingHalfAway = {sep_half_away};")
    add("")
    for label, count in (
        ("clamp-before-snap", sep_snap_first),
        ("the undershoot repair", sep_no_repair),
        ("round-not-truncate", sep_truncate),
        ("banker's rounding from llround's half-away-from-zero", sep_half_away),
    ):
        if count == 0:
            raise SystemExit(f"REFUSING TO EMIT: no case separates {label} from upstream")

    # ── T5: the whole chain, DurationPredictor.__call__ ──────────────────────
    add("// DurationPredictor.__call__ (utils/blocks.py:850-889) end to end, on a")
    add("// head whose weights are a closed form of the parameter index. The C++")
    add("// side builds the SAME bag from the SAME expression, so a disagreement")
    add("// here is the driver and not the weights.")
    add("//")
    add("// `seconds` is the head's raw prediction, carried so a frame-count")
    add("// mismatch localizes to the forward or to the snapping rather than")
    add("// arriving as one wrong integer.")
    add("struct Ltx2PredictCase {")
    add("    int64_t video_tokens; int64_t audio_tokens; double frame_rate;")
    add("    double min_seconds; double max_seconds; float seconds; int64_t frames;")
    add("};")
    add("inline constexpr Ltx2PredictCase kLtx2PredictCases[] = {")

    head = up.DurationHeadConfigurator.from_metadata({})
    fill_synthetic(up, head)
    head.eval()
    predictor = up.DurationPredictor(head)

    predict_cases = [
        (7, 5, 24.0, 1.0, 20.0),
        (7, 0, 24.0, 1.0, 20.0),    # video only
        (0, 5, 24.0, 1.0, 20.0),    # audio only -- T2AOneStagePipeline's shape
        (13, 11, 30.0, 1.0, 20.0),
        (7, 5, 25.0, 2.0, 4.0),     # a window tight enough that the clamp fires
    ]
    raw_seconds: list[float] = []
    frame_counts: list[int] = []
    for v_count, a_count, fps, lo_s, hi_s in predict_cases:
        v = synthetic_tokens(up, v_count, 4096, 0.0) if v_count else None
        a = synthetic_tokens(up, a_count, 2048, 1.7) if a_count else None
        with up.torch.no_grad():
            raw = head(v, a).item()
            frames = predictor(v, a, frame_rate=fps, min_seconds=lo_s, max_seconds=hi_s)
        raw_seconds.append(raw)
        frame_counts.append(frames)
        add(
            f"    {{{v_count}, {a_count}, {fps!r}, {lo_s!r}, {hi_s!r}, "
            f"{raw:.9g}f, {frames}}},"
        )
    add("};")
    assert_discriminating(raw_seconds, frame_counts)
    add(f"inline constexpr int kLtx2PredictDistinctFrameCounts = {len(set(frame_counts))};")
    add("")

    # ── T6: the message require_num_frames_source raises ─────────────────────
    add("// require_num_frames_source (utils/blocks.py:894-905). The MESSAGE is")
    add("// pinned so the port's refusal says what upstream's says; the POSITION")
    add("// -- top of __call__, before any work -- is what the ordering case in")
    add("// test_ltx2_video gates, and it is the half a message check cannot see.")
    try:
        up.require_num_frames_source(up.AutoDuration(), None)
        raise SystemExit("REFUSING TO EMIT: require_num_frames_source did not raise")
    except ValueError as exc:
        text = str(exc).replace('"', '\\"')
    add(f'inline constexpr const char* kLtx2RequireNumFramesMessage =\n    "{text}";')
    add("")
    # Upstream returns None (does not raise) when a predictor IS available, and
    # when the count is explicit. Both polarities pinned, because a guard that
    # always raises passes the case above.
    up.require_num_frames_source(up.AutoDuration(), predictor)
    up.require_num_frames_source(48, None)
    add("// Upstream does NOT raise for (AutoDuration, predictor) or (48, None).")
    add("// Both were executed at generation time; a guard that always raised")
    add("// would have failed this script, not this test.")
    add("inline constexpr bool kLtx2RequireNumFramesAllowsExplicitWithoutHead = true;")
    add("inline constexpr bool kLtx2RequireNumFramesAllowsAutoWithHead = true;")
    add("")

    # ── AutoDuration's defaults ──────────────────────────────────────────────
    default = up.AutoDuration()
    add("// AutoDuration's defaults (utils/types.py:116-124), read off the class.")
    add(f"inline constexpr double kLtx2AutoDurationMinSeconds = {default.min_seconds!r};")
    add(f"inline constexpr double kLtx2AutoDurationMaxSeconds = {default.max_seconds!r};")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltx2", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    up = load_upstream(args.ltx2)
    body = cpp_lines(up)
    header = [
        "// clang-format off",
        "#pragma once",
        "#include <cstdint>",
        "",
        "namespace vllm_test {",
        "",
    ]
    args.out.write_text("\n".join(header + body + ["", "}  // namespace vllm_test"]) + "\n")
    print(f"wrote {args.out} ({len(body)} lines of goldens)")


if __name__ == "__main__":
    main()
