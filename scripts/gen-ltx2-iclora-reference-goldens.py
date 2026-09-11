#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_iclora_reference_goldens.inc — the IC-LoRA
reference-clip and conditioning-attention-mask oracle.

Row LTX25-IC-LORA-REF-VIDEO, issue #3020,
spec .agents/specs/ltx25-ic-lora-ref-video.md.

Every number below is what upstream's own code RETURNED. Nothing is transcribed.

WHAT IS IMPORTED AND WHAT IS SLICED. `ltx_core.conditioning.mask_utils` and
`ltx_core.model.transformer.transformer_args` import cleanly and are imported.
`ltx_pipelines.iclora_utils` cannot be: it reaches `utils.media_io`, which
imports PyAV, a codec binding this project deliberately does not vendor. So its
three functions are lifted from the real file BY AST and their own source text is
compiled verbatim, the same technique scripts/gen-ltx2-dfr-goldens.py uses and
for the same reason. Each name is asserted present, so an upstream rename raises
here instead of leaving this generator running a local copy of the thing under
test.

`append_ic_lora_reference_video_conditionings` is EXECUTED, with only its I/O and
its VAE stubbed: the frame decode, the pixel preprocess and the encoder are
recording stubs, and every branch — the divisibility refusal, the reference
geometry, the `if reference_temporal_scale_factor > 1` guard, the three-way mask
selection and the `if attn_mask is not None` wrap — is upstream's own control
flow running. That is what these goldens are about; the numerics of the resize
are already gated by ltx2_image_cond_goldens.inc.

EVERY CASE CARRIES ITS REJECTED HYPOTHESIS AND THIS SCRIPT REFUSES TO EMIT A
GOLDEN THAT CANNOT FAIL. `build_attention_mask` on a fixture with no PRIOR
conditioning item is elementwise equal to the plausible wrong reading "cross on
all existing rows" — measured 0 separating elements — so such a fixture is a mute
switch. `require_separation` is called on every case and raises rather than
writing the file.

Regenerate with:
    python3 scripts/gen-ltx2-iclora-reference-goldens.py --ltx2 <LTX-2 checkout> \
        --out tests/vllm/models/ltx2_iclora_reference_goldens.inc
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys


def git_revision(root: pathlib.Path) -> str:
    """The checkout's SHA, and REFUSE a dirty tree."""
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


def slice_functions(path: pathlib.Path, wanted: tuple[str, ...], namespace: dict) -> None:
    """Compile the named top-level functions out of `path` verbatim into `namespace`."""
    source = path.read_text()
    tree = ast.parse(source)
    found: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            found[node.name] = node
    missing = [n for n in wanted if n not in found]
    if missing:
        raise SystemExit(f"REFUSING: {path} no longer defines {missing}")
    lines = source.splitlines()
    for name in wanted:
        node = found[name]
        text = "\n".join(lines[node.lineno - 1:node.end_lineno])
        exec(compile(ast.parse(text), str(path), "exec"), namespace)  # noqa: S102


SEPARATIONS: list[tuple[str, str, float]] = []


def require_separation(case: str, rejected: str, value: float) -> None:
    """A golden that agrees with the hypothesis it claims to reject is a mute switch."""
    if not (value > 0.0):
        raise SystemExit(
            f"REFUSING to emit '{case}': it does not separate from '{rejected}' "
            f"(separation {value!r}). A golden that cannot fail is not a gate."
        )
    SEPARATIONS.append((case, rejected, value))


def fmt(x: float) -> str:
    # `.9g` alone emits `0` for zero, which is an int literal in C++ and makes the
    # array's element type depend on where the zero sits. Force a decimal point.
    text = f"{float(x):.9g}"
    if "." not in text and "e" not in text and "n" not in text:
        text += ".0"
    return text + "f"


def carray(name: str, values) -> str:
    body = ", ".join(fmt(v) for v in values)
    return f"inline constexpr float {name}[] = {{{body}}};"


def iarray(name: str, values) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return f"inline constexpr int64_t {name}[] = {{{body}}};"


def deterministic(n: int, seed: int) -> list[float]:
    """A splitmix64-derived stream in [0, 1). The C++ side reads the SAME bytes out
    of this file, so the generator is free to choose any reproducible stream."""
    out = []
    state = seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        # 24 bits, so the value is exact in f32 and survives the round trip.
        out.append((z >> 40) / float(1 << 24))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltx2", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    root = args.ltx2
    sys.path.insert(0, str(root / "packages" / "ltx-core" / "src"))
    sys.path.insert(0, str(root / "packages" / "ltx-pipelines" / "src"))
    revision = git_revision(root)

    import torch
    from einops import rearrange
    import torch.nn.functional as F

    from ltx_core.conditioning import (
        ConditioningItemAttentionStrengthWrapper,
        VideoConditionByReferenceLatent,
    )
    from ltx_core.conditioning.mask_utils import build_attention_mask, resolve_cross_mask
    from ltx_core.types import VideoLatentShape

    iclora_path = root / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "iclora_utils.py"
    ns: dict = {
        "torch": torch,
        "rearrange": rearrange,
        "VideoLatentShape": VideoLatentShape,
        "ConditioningItem": object,
        "ConditioningItemAttentionStrengthWrapper": ConditioningItemAttentionStrengthWrapper,
        "VideoConditionByReferenceLatent": VideoConditionByReferenceLatent,
        "VideoEncoder": object,
        "TilingConfig": object,
        "HDRColorSpace": object,
        "ResizeMode": object,
        "is_exr_dir": lambda p: False,
        "load_exr_folder_conditioning_hdr": None,
        "annotations": None,
    }
    slice_functions(
        iclora_path,
        ("temporal_subsample", "downsample_mask_video_to_latent",
         "append_ic_lora_reference_video_conditionings"),
        ns,
    )
    temporal_subsample = ns["temporal_subsample"]
    downsample = ns["downsample_mask_video_to_latent"]
    append_refs = ns["append_ic_lora_reference_video_conditionings"]

    out: list[str] = []
    E = out.append

    E("// GENERATED by scripts/gen-ltx2-iclora-reference-goldens.py — DO NOT EDIT BY HAND.")
    E("//")
    E("// LTX-2.5 IC-LoRA REFERENCE-CLIP and CONDITIONING-ATTENTION-MASK goldens")
    E("// (row LTX25-IC-LORA-REF-VIDEO, issue #3020), produced by IMPORTING AND")
    E("// EXECUTING upstream's `ltx_core.conditioning.mask_utils`, its")
    E("// `_prepare_self_attention_mask`, and the three AST-sliced functions of")
    E("// `ltx_pipelines.iclora_utils` (which cannot be imported: it reaches PyAV).")
    E("//")
    E("// Every case carries the REJECTED hypothesis beside upstream's answer, and")
    E("// the generator REFUSES to write a case whose separation is zero. See the")
    E("// separation table at the bottom of this file.")
    E("//")
    E(f"// Upstream revision (Lightricks/LTX-2): {revision}")
    E("#pragma once")
    E("")
    E("#include <cstdint>")
    E("")
    E("namespace vllm_test {")
    E("")
    E("// The suite asserts this equals the SHA it pins, so regenerating against a")
    E("// DIFFERENT checkout fails the gate instead of silently replacing the oracle.")
    E(f'inline constexpr const char* kLtx2IcLoraRefUpstreamRevision = "{revision}";')
    E("")

    # ── 1. temporal_subsample (iclora_utils.py:87-90) ────────────────────────
    E("// --- 1. temporal_subsample (iclora_utils.py:87-90) -----------------------")
    E("//")
    E("// `[0, *range(1, F, N)]`. INDEX 1 IS ALWAYS KEPT when it exists, so at N=2")
    E("// over 5 frames the kept set is {0, 1, 3} and NOT the plausible {0, 2, 4}.")
    E("// That is the whole content of this function and the rejected hypothesis below")
    E("// is exactly the plausible reading.")
    ts_cases = [(5, 2), (9, 4), (7, 3), (6, 1), (1, 4)]
    E(f"inline constexpr int kLtx2TemporalSubsampleCases = {len(ts_cases)};")
    E(iarray("kLtx2TemporalSubsampleFrames", [f for f, _ in ts_cases]))
    E(iarray("kLtx2TemporalSubsampleFactor", [n for _, n in ts_cases]))
    counts, flat = [], []
    total_sep = 0
    for frames, factor in ts_cases:
        # Upstream indexes a (B, C, F, H, W) tensor on dim 2. One channel, 1x1
        # pixels: the value at each kept frame IS its index, so what comes back is
        # the index vector itself rather than a restatement of it.
        video = torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)
        got = temporal_subsample(video, factor).flatten().tolist()
        rejected = list(range(0, frames, factor))  # "every Nth from 0"
        sep = sum(1 for i in range(max(len(got), len(rejected)))
                  if i >= len(got) or i >= len(rejected) or got[i] != rejected[i])
        total_sep += sep
        counts.append(len(got))
        flat.extend(int(v) for v in got)
    require_separation("temporal_subsample", "every Nth frame from index 0 (range(0, F, N))",
                       float(total_sep))
    E(iarray("kLtx2TemporalSubsampleCount", counts))
    E(iarray("kLtx2TemporalSubsampleKept", flat))
    E("")

    # ── 2. the reference geometry and the branch map (iclora_utils.py:93-170) ──
    E("// --- 2. append_ic_lora_reference_video_conditionings (iclora_utils.py:93-170)")
    E("//")
    E("// UPSTREAM'S OWN CONTROL FLOW, executed with the frame decode, the pixel")
    E("// preprocess and the VAE replaced by recording stubs. What is gated here is")
    E("// which geometry it asks the preprocess for, whether it subsamples, which of")
    E("// the three mask branches it takes, and whether it wraps.")

    class RecordingEncoder:
        def __init__(self, rec):
            self.rec = rec

        def __call__(self, video):
            self.rec["encoded_shape"] = tuple(video.shape)
            # A latent volume with the shape the caller will read back through
            # VideoLatentShape.from_torch_shape: (B, C, F, H, W).
            return torch.zeros(1, 4, 2, 2, 2)

    def run_append(height, width, scale, temporal, strength, mask, cond_strength):
        rec: dict = {}
        ns2 = dict(ns)
        ns2["decode_video_by_frame"] = lambda path, frame_cap, device: iter(())
        def preprocess(frames, h, w, dtype, device):
            rec["ref_height"], rec["ref_width"] = h, w
            # 5 pixel frames of 1x1, value == frame index, so a subsample is visible.
            return torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1)
        ns2["video_preprocess"] = preprocess
        slice_functions(iclora_path, ("append_ic_lora_reference_video_conditionings",), ns2)
        conds: list = []
        ns2["append_ic_lora_reference_video_conditionings"](
            conds, [("ref.mp4", strength)],
            height=height, width=width, num_frames=5,
            video_encoder=RecordingEncoder(rec), dtype=torch.float32,
            device=torch.device("cpu"),
            reference_downscale_factor=scale,
            reference_temporal_scale_factor=temporal,
            conditioning_attention_strength=cond_strength,
            conditioning_attention_mask=mask,
        )
        rec["wrapped"] = isinstance(conds[0], ConditioningItemAttentionStrengthWrapper)
        inner = conds[0].conditioning if rec["wrapped"] else conds[0]
        rec["downscale"] = inner.downscale_factor
        rec["temporal"] = inner.temporal_scale_factor
        rec["strength"] = inner.strength
        return rec

    # (a) the geometry: 448x768 at scale 4 -> the preprocess is asked for 112x192.
    geo = run_append(448, 768, 4, 1, 0.75, None, 1.0)
    E(f"inline constexpr int64_t kLtx2RefGeomHeight = {geo['ref_height']};")
    E(f"inline constexpr int64_t kLtx2RefGeomWidth = {geo['ref_width']};")
    require_separation("reference geometry at scale 4",
                       "the phase's own height and width (no downscale)",
                       float(abs(geo["ref_height"] - 448) + abs(geo["ref_width"] - 768)))
    E("// The item carries the factors, and the request's strength, unchanged.")
    E(f"inline constexpr int64_t kLtx2RefItemDownscale = {geo['downscale']};")
    E(f"inline constexpr float kLtx2RefItemStrength = {fmt(geo['strength'])};")
    E(f"inline constexpr bool kLtx2RefGeomWrapped = {'true' if geo['wrapped'] else 'false'};")

    # (b) the temporal-subsample guard: it fires at N>1 and NOT at N==1.
    sub = run_append(64, 64, 1, 2, 1.0, None, 1.0)
    nosub = run_append(64, 64, 1, 1, 1.0, None, 1.0)
    E("// The subsample guard (`if reference_temporal_scale_factor > 1`, :143). The")
    E("// encoder sees 3 of 5 frames at N=2 and all 5 at N=1 — the encoded SHAPE is")
    E("// what records it, so a port that subsampled unconditionally is separated.")
    E(f"inline constexpr int64_t kLtx2RefEncodedFramesN2 = {sub['encoded_shape'][2]};")
    E(f"inline constexpr int64_t kLtx2RefEncodedFramesN1 = {nosub['encoded_shape'][2]};")
    require_separation("the temporal-subsample guard",
                       "subsampling unconditionally",
                       float(abs(sub["encoded_shape"][2] - nosub["encoded_shape"][2])))

    # (c) the three-way mask branch (iclora_utils.py:151-160), and the wrap.
    scalar_lt1 = run_append(64, 64, 1, 1, 1.0, None, 0.5)
    scalar_eq1 = run_append(64, 64, 1, 1, 1.0, None, 1.0)
    E("// The three-way mask selection (:151-160). With NO mask and strength < 1 the")
    E("// scalar arm wraps; with NO mask and strength == 1 nothing wraps. The CLI")
    E("// cannot reach the first (a strength is only ever set alongside a mask,")
    E("// ic_lora.py:452-455), and this engine refuses it by name — the golden is")
    E("// here so the refusal is pinned to a MEASURED branch rather than a reading.")
    E(f"inline constexpr bool kLtx2RefWrapScalarBelowOne = {'true' if scalar_lt1['wrapped'] else 'false'};")
    E(f"inline constexpr bool kLtx2RefWrapScalarAtOne = {'true' if scalar_eq1['wrapped'] else 'false'};")
    require_separation("the strength<1 wrap branch", "wrapping unconditionally",
                       float(scalar_lt1["wrapped"] != scalar_eq1["wrapped"]))

    # (d) the divisibility refusal (:112-115).
    try:
        run_append(448, 770, 4, 1, 1.0, None, 1.0)
        raise SystemExit("REFUSING: upstream accepted an indivisible target; the refusal moved")
    except ValueError as e:
        refusal = str(e)
    E("// The divisibility refusal (:112-115), verbatim from the raised ValueError.")
    E(f'inline constexpr const char* kLtx2RefDivisibilityRefusal = "{refusal}";')
    # scale == 1 never refuses, whatever the dimensions.
    ok = run_append(449, 771, 1, 1, 1.0, None, 1.0)
    E(f"inline constexpr int64_t kLtx2RefScaleOneHeight = {ok['ref_height']};")
    require_separation("the divisibility refusal", "refusing whenever height % scale != 0, scale==1 included",
                       float(ok["ref_height"] == 449))
    E("")

    # ── 3. downsample_mask_video_to_latent (iclora_utils.py:52-84) ────────────
    E("// --- 3. downsample_mask_video_to_latent (iclora_utils.py:52-84) ----------")
    E("//")
    E("// AREA spatial interpolation, then the CAUSAL carve-out: latent frame 0 is")
    E("// pixel frame 0 ALONE, and the remaining f_pix-1 frames mean-pool in groups")
    E("// of t = (f_pix-1)/(f_lat-1). Both rejected hypotheses below are measured.")
    f_pix, h_pix, w_pix = 9, 8, 8
    f_lat, h_lat, w_lat = 3, 2, 2
    vals = deterministic(f_pix * h_pix * w_pix, 0x5EED1C10)
    mask = torch.tensor(vals, dtype=torch.float32).reshape(1, 1, f_pix, h_pix, w_pix)
    shape = VideoLatentShape.from_torch_shape((1, 1, f_lat, h_lat, w_lat))
    got = downsample(mask, shape)
    t = (f_pix - 1) // (f_lat - 1)
    bil = F.interpolate(rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
                        size=(h_lat, w_lat), mode="bilinear", align_corners=False)
    bil = rearrange(bil, "(b f) 1 h w -> b 1 f h w", b=1)
    bil_rest = rearrange(bil[:, :, 1:], "b 1 (f t) h w -> b 1 f t h w", t=t).mean(3)
    bil_flat = rearrange(torch.cat([bil[:, :, :1], bil_rest], 2), "b 1 f h w -> b (f h w)")
    require_separation("downsample_mask_video_to_latent", "bilinear spatial interpolation",
                       float((got - bil_flat).abs().max()))
    area = rearrange(F.interpolate(rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
                                   size=(h_lat, w_lat), mode="area"),
                     "(b f) 1 h w -> b 1 f h w", b=1)
    # The plausible wrong reading: pool ALL f_pix frames uniformly into f_lat
    # groups, with no first-frame carve-out. f_pix must divide by f_lat for the
    # hypothesis to be expressible at all, which the fixture arranges.
    assert f_pix % f_lat == 0, "the no-carve-out hypothesis needs f_pix % f_lat == 0"
    unif = rearrange(rearrange(area, "b 1 (f t) h w -> b 1 f t h w", t=f_pix // f_lat).mean(3),
                     "b 1 f h w -> b (f h w)")
    require_separation("downsample_mask_video_to_latent",
                       "uniform temporal pooling with no causal first-frame carve-out",
                       float((got[:, : unif.shape[1]] - unif).abs().max()))
    E(f"inline constexpr int64_t kLtx2MaskPixFrames = {f_pix};")
    E(f"inline constexpr int64_t kLtx2MaskPixHeight = {h_pix};")
    E(f"inline constexpr int64_t kLtx2MaskPixWidth = {w_pix};")
    E(f"inline constexpr int64_t kLtx2MaskLatFrames = {f_lat};")
    E(f"inline constexpr int64_t kLtx2MaskLatHeight = {h_lat};")
    E(f"inline constexpr int64_t kLtx2MaskLatWidth = {w_lat};")
    E("// The INPUT, so both sides read identical bytes rather than agreeing about a")
    E("// random stream. Every value is exact in f32 (24 significant bits).")
    E(carray("kLtx2MaskPixels", vals))
    E(carray("kLtx2MaskLatentWeights", got.flatten().tolist()))
    E("// The two rejected answers, emitted so the suite asserts it is NOT them.")
    E(carray("kLtx2MaskLatentBilinear", bil_flat.flatten().tolist()))
    E(carray("kLtx2MaskLatentUniformPool", unif.flatten().tolist()))

    # A SECOND fixture whose spatial output does NOT divide its input. `area`
    # pools output index i over `[floor(i*I/O), ceil((i+1)*I/O))`, which is only
    # the integer-stride window `[i*(I//O), (i+1)*(I//O))` when O divides I. The
    # 8 -> 2 case above is divisible, so it is elementwise equal to the stride
    # reading and cannot see the difference; 9 -> 2 can.
    nd_f_pix, nd_h_pix, nd_w_pix = 9, 9, 9
    nd_f_lat, nd_h_lat, nd_w_lat = 3, 2, 2
    nd_vals = deterministic(nd_f_pix * nd_h_pix * nd_w_pix, 0x9D19151B)
    nd_mask = torch.tensor(nd_vals, dtype=torch.float32).reshape(
        1, 1, nd_f_pix, nd_h_pix, nd_w_pix)
    nd_shape = VideoLatentShape.from_torch_shape((1, 1, nd_f_lat, nd_h_lat, nd_w_lat))
    nd_got = downsample(nd_mask, nd_shape)

    def stride_box_spatial(volume, h_out, w_out):
        """The plausible wrong reading: an integer-stride box filter."""
        frames, h_in, w_in = volume.shape[2], volume.shape[3], volume.shape[4]
        sh, sw = h_in // h_out, w_in // w_out
        out_v = torch.empty(1, 1, frames, h_out, w_out, dtype=torch.float32)
        for fi in range(frames):
            for oh in range(h_out):
                for ow in range(w_out):
                    out_v[0, 0, fi, oh, ow] = volume[
                        0, 0, fi, oh * sh:(oh + 1) * sh, ow * sw:(ow + 1) * sw].mean()
        return out_v

    nd_t = (nd_f_pix - 1) // (nd_f_lat - 1)
    box = stride_box_spatial(nd_mask, nd_h_lat, nd_w_lat)
    box_rest = rearrange(box[:, :, 1:], "b 1 (f t) h w -> b 1 f t h w", t=nd_t).mean(3)
    box_flat = rearrange(torch.cat([box[:, :, :1], box_rest], 2), "b 1 f h w -> b (f h w)")
    require_separation("downsample_mask_video_to_latent at a NON-DIVIDING spatial pair (9 -> 2)",
                       "an integer-stride box filter, which `area` equals only when O divides I",
                       float((nd_got - box_flat).abs().max()))
    E("// A NON-DIVIDING spatial pair, 9 -> 2. `area` pools output index i over")
    E("// `[floor(i*I/O), ceil((i+1)*I/O))`; the integer-stride window")
    E("// `[i*(I//O), (i+1)*(I//O))` agrees with it on every divisible shape and")
    E("// only there, so the 8 -> 2 case above cannot see the difference and this")
    E("// one can.")
    E(f"inline constexpr int64_t kLtx2MaskNdPixFrames = {nd_f_pix};")
    E(f"inline constexpr int64_t kLtx2MaskNdPixHeight = {nd_h_pix};")
    E(f"inline constexpr int64_t kLtx2MaskNdPixWidth = {nd_w_pix};")
    E(f"inline constexpr int64_t kLtx2MaskNdLatFrames = {nd_f_lat};")
    E(f"inline constexpr int64_t kLtx2MaskNdLatHeight = {nd_h_lat};")
    E(f"inline constexpr int64_t kLtx2MaskNdLatWidth = {nd_w_lat};")
    E(carray("kLtx2MaskNdPixels", nd_vals))
    E(carray("kLtx2MaskNdLatentWeights", nd_got.flatten().tolist()))
    E("// The rejected answer, emitted so the suite asserts it is NOT this one.")
    E(carray("kLtx2MaskNdLatentStrideBox", box_flat.flatten().tolist()))
    E("")

    # ── 4. resolve_cross_mask + build_attention_mask (mask_utils.py) ──────────
    E("// --- 4. build_attention_mask (mask_utils.py:170-243) ---------------------")
    E("//")
    E("// THE FIXTURE CARRIES A PRIOR REFERENCE TOKEN ON PURPOSE. With")
    E("// num_existing == num_noisy the true block structure and the plausible wrong")
    E("// reading `cross on ALL existing rows` are ELEMENTWISE EQUAL — measured 0")
    E("// separating elements — and the golden would be a mute switch. The generator")
    E("// refuses that case; this one separates on 4 elements.")
    n_noisy, n_new, n_existing = 2, 2, 3
    cross_vals = [0.25, 0.75]
    cross = torch.tensor([cross_vals], dtype=torch.float32)
    got_mask = build_attention_mask(None, n_noisy, n_new, n_existing, cross,
                                    torch.device("cpu"), torch.float32)
    alt = got_mask.clone()
    alt[:, n_noisy:n_existing, n_existing:] = cross
    alt[:, n_existing:, n_noisy:n_existing] = cross.T
    require_separation("build_attention_mask", "cross on ALL existing rows, not only the noisy ones",
                       float((got_mask != alt).sum()))
    E(f"inline constexpr int64_t kLtx2AmNoisy = {n_noisy};")
    E(f"inline constexpr int64_t kLtx2AmNew = {n_new};")
    E(f"inline constexpr int64_t kLtx2AmExisting = {n_existing};")
    E(carray("kLtx2AmCross", cross_vals))
    E(carray("kLtx2AmMask", got_mask.flatten().tolist()))
    E(carray("kLtx2AmMaskCrossOnAllRows", alt.flatten().tolist()))

    E("// A SECOND item on top of the first: `existing_mask` is preserved in the")
    E("// top-left block rather than refilled with ones (`:223-226`), which is what")
    E("// keeps the first item's attenuation alive after a second one is applied.")
    cross2_vals = [0.5]
    cross2 = torch.tensor([cross2_vals], dtype=torch.float32)
    got2 = build_attention_mask(got_mask, n_noisy, 1, n_existing + n_new, cross2,
                                torch.device("cpu"), torch.float32)
    alt2 = build_attention_mask(None, n_noisy, 1, n_existing + n_new, cross2,
                                torch.device("cpu"), torch.float32)
    require_separation("build_attention_mask with an existing mask",
                       "discarding the existing mask and refilling the block with ones",
                       float((got2 - alt2).abs().max()))
    E(carray("kLtx2AmCross2", cross2_vals))
    E(carray("kLtx2AmMask2", got2.flatten().tolist()))
    E(carray("kLtx2AmMask2Refilled", alt2.flatten().tolist()))

    E("// resolve_cross_mask's scalar arm (`:31-37`) and its 1-D broadcast (`:49-54`).")
    sc = resolve_cross_mask(0.5, 4, 1, torch.device("cpu"), torch.float32)
    E(carray("kLtx2CrossScalar", sc.flatten().tolist()))
    oned = resolve_cross_mask(torch.tensor([0.1, 0.2, 0.3, 0.4]), 4, 1,
                              torch.device("cpu"), torch.float32)
    E(carray("kLtx2CrossOneD", oned.flatten().tolist()))
    require_separation("resolve_cross_mask's 1-D arm", "the scalar fill",
                       float((sc - oned).abs().max()))

    E("// update_attention_mask's PAD-WITH-ONES arm (`:141-156`): a null mask on a")
    E("// state that already carries one grows the mask with full attention rather")
    E("// than returning None and leaving it short of the sequence.")
    ones = torch.ones(1, 1, dtype=torch.float32)
    padded = build_attention_mask(got_mask, n_noisy, 1, n_existing + n_new, ones,
                                  torch.device("cpu"), torch.float32)
    E(carray("kLtx2AmMaskPaddedOnes", padded.flatten().tolist()))
    require_separation("update_attention_mask's pad-with-ones arm",
                       "returning the mask unchanged (short of the sequence)",
                       float(padded.numel() - got_mask.numel()))
    E("")

    # ── 5. _prepare_self_attention_mask (transformer_args.py:208-237) ─────────
    E("// --- 5. _prepare_self_attention_mask (transformer_args.py:208-237) -------")
    E("//")
    E("// The CONSUMPTION seam this row makes reachable. Already ported as")
    E("// Ltx2PrepareSelfAttentionMask; pinned here because nothing in src/ assigned")
    E("// Ltx2ModalityInput::attention_mask before this row, so these numbers were")
    E("// gated only through a hand-built test struct.")
    import inspect
    import ltx_core.model.transformer.transformer_args as ta
    owner = None
    for _name, obj in vars(ta).items():
        if inspect.isclass(obj) and "_prepare_self_attention_mask" in vars(obj):
            owner = obj
            break
    if owner is None:
        raise SystemExit("REFUSING: no class in transformer_args defines _prepare_self_attention_mask")
    prep = owner._prepare_self_attention_mask
    probe = [1.0, 0.5, 0.0, 1e-30]
    bias = prep(None, torch.tensor([[probe]], dtype=torch.float32), torch.float32)
    vals_b = [float(x) for x in bias.flatten()]
    E(f'inline constexpr const char* kLtx2SelfMaskOwner = "{owner.__name__}";')
    E(carray("kLtx2SelfMaskProbe", probe))
    E("// f32. The third entry is finfo(f32).min, NOT -inf and NOT 0; the fourth is")
    E("// log(tiny) after the clamp, NOT log(1e-30).")
    E(carray("kLtx2SelfMaskBias", vals_b))
    naive = [float(torch.log(torch.tensor(max(v, 0.0)))) if v > 0 else float("-inf") for v in probe]
    require_separation("_prepare_self_attention_mask", "-inf for a zero and an unclamped log",
                       float(max(abs(a - b) for a, b in zip(vals_b, naive) if b != float("-inf"))
                             + (1.0 if any(b == float("-inf") for b in naive) else 0.0)))
    E("")

    # ── the separation table ─────────────────────────────────────────────────
    E("// --- separation table ----------------------------------------------------")
    E("//")
    E("// Every case above, its rejected hypothesis, and the measured separation. A")
    E("// zero here is a mute switch and the generator raises rather than writing it.")
    for case, rejected, value in SEPARATIONS:
        E(f"//   {case}  vs  {rejected}  ->  {value!r}")
    E("")
    E("}  // namespace vllm_test")

    args.out.write_text("\n".join(out) + "\n")
    print(f"wrote {args.out} ({len(out)} lines), {len(SEPARATIONS)} separated cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
