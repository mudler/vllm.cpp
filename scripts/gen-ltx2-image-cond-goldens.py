#!/usr/bin/env python3
"""Emit tests/vllm/multimodal/ltx2_image_cond_goldens.inc — the LTX-2.5 IMAGE
CONDITIONING parity oracle. Row LTX25-IMAGE-COND, issue #644.

Spec: .agents/specs/ltx25-image-conditioning.md §5.

WHAT THIS GATES, and why it is a separate generator from
`scripts/gen-ltx2-vae-goldens.py`. That script gates the VAE BRICKS. This one
gates the CHAIN a conditioning image travels: pixels -> aspect-fill resize ->
normalize -> VideoEncoder -> VideoConditionByLatentIndex -> GaussianNoiser. Every
link but the first two already had a golden; the chain did not, and a chain whose
links are each green can still be wired in the wrong ORDER — which for this
particular chain is exactly the defect that survives every shape and finiteness
check (§4 of the spec).

Upstream sources (Lightricks/LTX-2):
  ltx-pipelines/.../utils/media_io/resize.py:41-73     -> section 1, 2
  ltx-pipelines/.../utils/media_io/range_map.py:8-9    -> section 2
  ltx-pipelines/.../utils/media_io/decode.py:413-435   -> section 2 (the crf==0 branch)
  ltx-core/.../model/video_vae/video_vae.py:148-336    -> section 3
  ltx-core/.../conditioning/types/latent_cond.py:22-43 -> section 4
  ltx-core/.../components/noisers.py:30-37             -> section 5

Usage:
    python3 scripts/gen-ltx2-image-cond-goldens.py \\
        --ltx2 ~/_git/LTX-2 \\
        --out tests/vllm/multimodal/ltx2_image_cond_goldens.inc

Needs torch + numpy + einops (CPU only). NO checkpoint and no gated download.

UPSTREAM REVISION ANCHOR, and a DIRTY-TREE REFUSAL, exactly as
gen-ltx2-vae-goldens.py has them and for the identical reason: without a SHA
nobody can tell a PORT drift from an UPSTREAM one, and `git rev-parse HEAD` on a
dirty tree stamps a clean anchor onto numbers that commit cannot reproduce.

  Pinned revision: fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca

Advancing the pin is a deliberate edit in BOTH places (here and
`kLtx2ImgCondUpstreamRevisionPin` in tests/vllm/multimodal/test_ltx2_image_cond.cpp).

── THREE HARNESS ADAPTATIONS, all recorded because none changes the math ───────

1. `av` AND `OpenImageIO` ARE STUBBED, and the stub is a TRIPWIRE.
   `ltx_pipelines.utils.media_io.decode` imports both at module scope, and
   neither is installed here. The crf==0 path this row ports never reaches
   either — that is its whole point — so they are registered as modules whose
   every attribute is a stub object. This is NOT a way to avoid running upstream:
   `preprocess(image, 0)` is executed for real and asserted to return the
   IDENTICAL object it was given (`result is image`), which no stub can
   manufacture, and `preprocess(image, 18)` is executed too and asserted to RAISE
   inside the stubbed codec — which proves the branch is live rather than
   assumed. A source-text assertion about `if crf == 0` would have been
   self-confirming; this is not.

2. THE PACKAGE `__init__` FILES ARE NOT EXECUTED. `ltx_pipelines/__init__.py`
   and its parents pull in the same codec chain. The three package levels are
   registered as module objects carrying only `__path__`, so `import
   ltx_pipelines.utils.media_io.resize` loads the SUBMODULE from its real file
   without running any `__init__.py`. The submodules themselves are executed
   verbatim.

3. `GaussianNoiser._sample_noise` DRAWS FROM THE SHARED STREAM instead of
   `torch.randn`. Upstream keys its draw to a `torch.Generator`; the C++ side
   consumes the same deterministic stream in the same order. This mirrors the
   `torch.randn` patch gen-ltx2-vae-goldens.py already applies to the decoder.

ORACLE IDENTITY is asserted, not assumed: `ltx_core.__file__` and
`ltx_pipelines...resize.__file__` are both checked to live under `--ltx2` before
anything runs, because a `.pth`, an editable install or a namespace-package
layout would otherwise resolve a DIFFERENT source silently.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# THE GOLDEN BAND, READ from the C++ suite rather than repeated here — the same
# arrangement gen-ltx2-vae-goldens.py uses, and for the same reason: a literal
# here would be a second definition of one number in a second language, and a
# widened C++ band would leave this generator certifying arms nobody checks
# against. A parse that does not find EXACTLY ONE definition is fatal.
# ---------------------------------------------------------------------------

_GOLDEN_TOL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "vllm"
    / "multimodal"
    / "test_ltx2_image_cond.cpp"
)


def _read_golden_tol() -> float:
    text = _GOLDEN_TOL_SOURCE.read_text(encoding="utf-8")
    hits = re.findall(r"^constexpr double kLtx2ImgGoldenTol = ([0-9eE.+-]+);", text, re.M)
    if len(hits) != 1:
        raise SystemExit(
            f"expected EXACTLY ONE `constexpr double kLtx2ImgGoldenTol = ...;` in "
            f"{_GOLDEN_TOL_SOURCE}, found {len(hits)} — the generator cannot assert against a "
            f"band it cannot resolve"
        )
    return float(hits[0])


# ---------------------------------------------------------------------------
# The shared deterministic stream. Byte-for-byte the one
# scripts/gen-ltx2-vae-goldens.py uses and tests/vllm/models/test_ltx2_vae.cpp
# mirrors, so a tensor built by this script and one built by that one from the
# same NAME are the same tensor.
# ---------------------------------------------------------------------------


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def ltx_rand(name: str, count: int) -> np.ndarray:
    """`count` values uniform in [-1, 1), reproducible from `name` alone."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def ltx_bytes(name: str, count: int) -> np.ndarray:
    """`count` UINT8 codes — a conditioning image is uint8 out of the decoder
    (`np.array(image, dtype=np.uint8)`, decode.py:170), and quantizing a float
    stream afterwards would gate a different input than the one a real PPM
    carries."""
    seed = fnv1a64(name)
    return np.array(
        [splitmix64((seed + i) & _MASK64) % 256 for i in range(count)], dtype=np.uint8
    )


def param_values(name: str, shape) -> np.ndarray:
    """The per-parameter role rule, IDENTICAL to gen-ltx2-vae-goldens.py's. Only
    the roles a video ENCODER actually carries are reachable here; the audio and
    vocoder roles that script also handles have no counterpart in this chain."""
    count = int(np.prod(shape)) if len(shape) else 1
    rank = len(shape)
    if name.endswith("std-of-means"):
        return ltx_rand(name, count) * 0.1 + 1.0
    if name.endswith("mean-of-means"):
        return ltx_rand(name, count) * 0.1
    if name.endswith(".bias"):
        return ltx_rand(name, count) * 0.05
    if rank == 1 and name.endswith(".weight"):
        return ltx_rand(name, count) * 0.1 + 1.0
    return ltx_rand(name, count) * 0.1


def fill_from_stream(module, prefix: str = "") -> list[tuple[str, int]]:
    import torch

    state = module.state_dict()
    manifest: list[tuple[str, int]] = []
    filled = {}
    for name, tensor in state.items():
        values = param_values(prefix + name, tuple(tensor.shape))
        filled[name] = torch.from_numpy(values.astype(np.float32)).reshape(tensor.shape)
        manifest.append((prefix + name, int(values.size)))
    module.load_state_dict(filled, strict=True)
    return manifest


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------


def _cxx_float(value: float, digits: int) -> str:
    if not math.isfinite(value):
        raise ValueError(f"refusing to emit non-finite golden value: {value}")
    text = f"{value:.{digits}g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(_cxx_float(v, 9) + "f" for v in flat[i : i + 6])
        out.write("    " + chunk + ",\n")
    out.write("};\n\n")


def emit_scalar(out, name: str, value) -> None:
    out.write(f"inline constexpr int64_t {name} = {int(value)};\n")


def emit_double(out, name: str, value) -> None:
    out.write(f"inline constexpr double {name} = {_cxx_float(float(value), 17)};\n")


def emit_manifest(out, name: str, manifest: list[tuple[str, int]]) -> None:
    out.write(f"inline constexpr const char* {name}Names[] = {{\n")
    for key, _ in manifest:
        out.write(f'    "{key}",\n')
    out.write("};\n")
    out.write(f"inline constexpr int64_t {name}Counts[] = {{\n")
    for i in range(0, len(manifest), 10):
        out.write("    " + ", ".join(str(c) for _, c in manifest[i : i + 10]) + ",\n")
    out.write("};\n\n")


# ---------------------------------------------------------------------------
# Reduced-dimension geometry
# ---------------------------------------------------------------------------

# Section 1 — the resize cases. Chosen so that between them they cover: pure
# upscale, pure downscale, a source WIDER than the target aspect, a source
# TALLER than it, an exact identity, and — case 1 — a scale at which `ceil`
# disagrees with BOTH `round` and `int`: `32 * (16/24)` is 21.3333, so upstream
# resizes to 22 rows and crops 3 off the top, while a port that rounded would
# resize to 21 and crop 2. Every value in the output moves, and every shape
# check still passes. That case is asserted below rather than trusted.
#
# MEASURED, and recorded because it corrects what an earlier draft of this
# comment claimed: no source/target pair in this size range makes
# `src * scale` land just ABOVE an integer in IEEE double, which is the
# negative-crop hazard resize.py:61-62 names. The `ceil` is still load-bearing —
# for the ordinary reason above — but not for the reason upstream's comment
# gives, and asserting the reason upstream gives would have been asserting
# something false.
RESIZE_CASES = [
    # (src_h, src_w, dst_h, dst_w)
    (12, 20, 16, 16),
    (32, 24, 16, 16),
    (16, 16, 16, 16),
    (8, 8, 16, 24),
    (10, 7, 16, 16),
]

# Section 2-5 — the conditioning chain. The image is a 12x20 PPM (case 0's
# geometry, so the `ceil` branch is LIVE on the path that actually renders) and
# the target is 16x16, which the encoder's (2 patch x 2 x 2) spatial factor of 8
# turns into a 2x2 latent.
IMAGE_SRC_H, IMAGE_SRC_W = 12, 20
IMAGE_DST_H, IMAGE_DST_W = 16, 16

IMG_ENC_BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_space_res", {"multiplier": 2}),
    ("compress_all_res", {"multiplier": 1}),
]
IMG_ENC = dict(convolution_dimensions=3, in_channels=3, out_channels=4, patch_size=2)

# The target latent the conditioning is placed into: 3 latent frames of the
# encoder's own (channels, height, width), conditioned at index 0 — which is what
# `combined_image_conditionings` selects for `frame_idx == 0`
# (ltx-pipelines/utils/helpers.py:295-300).
COND_TARGET_FRAMES = 3
COND_PATCH = 1
COND_FPS = 8.0
COND_STRENGTH = 0.7
COND_LATENT_IDX = 0

# A NON-UNIT noise scale, deliberately. `noise_scale == 1` is the ONLY value at
# which ltx_core's composition and diffusers' agree (spec §3.3), so gating there
# would leave the divergence invisible.
NOISE_SCALE = 0.625


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_resize(out, resize_mod) -> None:
    import torch

    out.write(
        "// --- section 1: resize_and_center_crop (media_io/resize.py:41-73) ---\n"
        "// Aspect FILL, `ceil`, bilinear align_corners=False, then centre crop.\n"
    )
    emit_scalar(out, "kLtx2ImgResizeCases", len(RESIZE_CASES))
    out.write("\n")
    for index, (src_h, src_w, dst_h, dst_w) in enumerate(RESIZE_CASES):
        codes = ltx_bytes(f"ltx2.imgcond.resize{index}", src_h * src_w * 3)
        hwc = torch.from_numpy(codes.reshape(src_h, src_w, 3).astype(np.float32))
        y = resize_mod.resize_and_center_crop(hwc, dst_h, dst_w)
        # `1 c f h w` with f == 1 for a 3-D input (resize.py:73).
        assert tuple(y.shape) == (1, 3, 1, dst_h, dst_w), f"unexpected resize shape {y.shape}"
        out.write(f"// case {index}: {src_h}x{src_w} -> {dst_h}x{dst_w}\n")
        emit_scalar(out, f"kLtx2ImgResize{index}SrcH", src_h)
        emit_scalar(out, f"kLtx2ImgResize{index}SrcW", src_w)
        emit_scalar(out, f"kLtx2ImgResize{index}DstH", dst_h)
        emit_scalar(out, f"kLtx2ImgResize{index}DstW", dst_w)
        out.write("\n")
        emit_f32(out, f"kLtx2ImgResize{index}Golden", y.numpy())

    # The `ceil` is asserted rather than described. At least one case must have a
    # dimension where ceil disagrees with BOTH round and floor, or nothing in
    # this section can tell those three ports apart.
    separating = []
    for index, (src_h, src_w, dst_h, dst_w) in enumerate(RESIZE_CASES):
        scale = max(dst_h / src_h, dst_w / src_w)
        for axis, src in (("h", src_h), ("w", src_w)):
            exact = src * scale
            if math.ceil(exact) != round(exact) and math.ceil(exact) != math.floor(exact):
                separating.append((index, axis, exact))
    assert separating, (
        "no resize case has a dimension where `ceil` disagrees with both `round` and `floor`, "
        "so section 1 cannot tell those three ports apart. Add a shape pair that does rather "
        "than deleting this assertion"
    )


def section_preprocess(out, resize_mod, range_mod, decode_mod, tol) -> None:
    import torch

    codes = ltx_bytes("ltx2.imgcond.image", IMAGE_SRC_H * IMAGE_SRC_W * 3)
    image = codes.reshape(IMAGE_SRC_H, IMAGE_SRC_W, 3)

    # UPSTREAM'S OWN `preprocess`, EXECUTED. `result is image` is the assertion
    # that matters: the crf==0 branch RETURNS THE ARGUMENT (decode.py:425-426),
    # and no codec stub can produce object identity.
    passed = decode_mod.preprocess(image=image, crf=0)
    assert passed is image, (
        "preprocess(crf=0) did not return its argument — the short-circuit this whole row "
        "is built on (decode.py:425-426) is not where the spec says it is"
    )

    # ...and the OTHER branch is live. Executed too, so "a non-zero CRF needs a
    # codec" is measured rather than read.
    reached_codec = False
    try:
        decode_mod.preprocess(image=image, crf=18)
    except Exception:  # noqa: BLE001 - the stubbed codec is what raises
        reached_codec = True
    assert reached_codec, (
        "preprocess(crf=18) did NOT reach the stubbed codec, so either the round trip moved or "
        "the stub is answering for it — either way the CRF refusal this row ships would be "
        "guarding nothing"
    )

    # decode.py:76-78, in order: f32 in 0..255 -> resize -> normalize.
    as_float = torch.from_numpy(passed.astype(np.float32))
    resized = resize_mod.resize_and_center_crop(as_float, IMAGE_DST_H, IMAGE_DST_W)
    normalized = range_mod.normalize_images(resized, device="cpu", dtype=torch.float32)

    # THE ORDER IS A CLAIM, so it is MEASURED — and the measurement is a NEGATIVE
    # RESULT that is recorded rather than engineered away.
    #
    # `resize` is a convex combination and `normalize` is affine, so
    # resize(normalize(x)) and normalize(resize(x)) are EQUAL in exact
    # arithmetic. Their f32 difference is pure rounding and CANNOT be amplified
    # by choosing a different image: measured at 1.94e-07 here, against a golden
    # band of 5e-06 and against this port's own distance from torch, which is
    # larger still (see kLtx2ImgPixelTol in the suite: torch's bilinear
    # contracts to FMA, so no portable f32 port reproduces it bit for bit).
    #
    # So NO GOLDEN IN THIS FILE CAN SEE THE ORDER SWAP. The order is still
    # mirrored, for the only reason left: it is upstream's. That puts it in the
    # same class AGENTS.md names for a too-WIDE dtype — correct, invisible to
    # every gate we own, and therefore checked deliberately once and written
    # down rather than assumed to be covered.
    swapped = resize_mod.resize_and_center_crop(
        range_mod.normalize_images(as_float, device="cpu", dtype=torch.float32),
        IMAGE_DST_H,
        IMAGE_DST_W,
    )
    order_gap = float((normalized - swapped).abs().max())
    assert order_gap < tol, (
        f"resize-then-normalize and normalize-then-resize now differ by {order_gap:g}, ABOVE the "
        f"golden band {tol:g}. That would be new information — the two are algebraically equal, "
        f"so a gap this size means one of them stopped being the affine/convex pair this comment "
        f"assumes. Investigate before touching this assertion"
    )

    out.write(
        "// --- section 2: load_image_and_preprocess at crf=0 (decode.py:46-79) ---\n"
        "// decode -> preprocess(crf=0) -> f32 0..255 -> resize+crop -> /127.5 - 1.\n"
        "// `preprocess(crf=0) is image` was ASSERTED at generation time, and\n"
        "// `preprocess(crf=18)` was asserted to reach the codec and raise.\n"
    )
    emit_scalar(out, "kLtx2ImgPreSrcH", IMAGE_SRC_H)
    emit_scalar(out, "kLtx2ImgPreSrcW", IMAGE_SRC_W)
    emit_scalar(out, "kLtx2ImgPreDstH", IMAGE_DST_H)
    emit_scalar(out, "kLtx2ImgPreDstW", IMAGE_DST_W)
    out.write(
        "// max|resize-then-normalize - normalize-then-resize|, measured upstream. It is\n"
        "// BELOW the golden band, which is the recorded NEGATIVE RESULT: the two orders\n"
        "// are algebraically identical (a convex combination commutes with an affine\n"
        "// map), so their f32 gap is pure rounding and cannot be amplified. No golden\n"
        "// here can see the swap; the order is mirrored because it is upstream's, and\n"
        "// that is written down rather than assumed to be covered.\n"
    )
    emit_double(out, "kLtx2ImgPreOrderGap", order_gap)
    out.write("\n")
    emit_f32(out, "kLtx2ImgPreGolden", normalized.numpy())
    return normalized


def section_encode(out, image_5d) -> None:
    from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
    from ltx_core.model.video_vae.video_vae import VideoEncoder

    enc = VideoEncoder(
        encoder_blocks=IMG_ENC_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode=PaddingModeType.ZEROS,
        **IMG_ENC,
    ).eval()
    manifest = fill_from_stream(enc, prefix="ltx2.imgenc.")
    latent = enc(image_5d)

    out.write(
        "// --- section 3: VideoEncoder over the preprocessed image "
        "(video_vae.py:264-336) ---\n"
        "// This is `video_encoder(image)` at ltx-pipelines/utils/helpers.py:294.\n"
    )
    emit_scalar(out, "kLtx2ImgEncOutC", latent.shape[1])
    emit_scalar(out, "kLtx2ImgEncOutT", latent.shape[2])
    emit_scalar(out, "kLtx2ImgEncOutH", latent.shape[3])
    emit_scalar(out, "kLtx2ImgEncOutW", latent.shape[4])
    emit_scalar(out, "kLtx2ImgEncTemporalFactor", enc.video_scale_factors.time)
    emit_scalar(out, "kLtx2ImgEncSpatialFactor", enc.video_scale_factors.height)
    out.write("\n")
    emit_manifest(out, "kLtx2ImgEncParam", manifest)
    emit_f32(out, "kLtx2ImgEncGolden", latent.numpy())
    return latent


def section_condition_and_noise(out, latent, tol) -> None:
    import torch

    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    target = VideoLatentShape(
        batch=1,
        channels=int(latent.shape[1]),
        frames=COND_TARGET_FRAMES,
        height=int(latent.shape[3]),
        width=int(latent.shape[4]),
    )
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=COND_PATCH),
        target_shape=target,
        fps=COND_FPS,
        scale_factors=SpatioTemporalScaleFactors.default(),
        causal_fix=True,
    )
    base = tools.create_initial_state(device="cpu", dtype=torch.float32)

    item = VideoConditionByLatentIndex(
        latent=latent, strength=COND_STRENGTH, latent_idx=COND_LATENT_IDX
    )
    conditioned = item.apply_to(base, tools)

    assert not torch.equal(conditioned.clean_latent, base.clean_latent), (
        "the item must CHANGE the clean latent or section 4 gates nothing"
    )
    assert not torch.equal(conditioned.denoise_mask, base.denoise_mask), (
        "the item must CHANGE the denoise mask or section 4 gates nothing"
    )
    assert torch.equal(conditioned.latent, base.latent), (
        "upstream leaves the NOISY tensor untouched (latent_cond.py:38-39). diffusers does "
        "NOT (pipeline_ltx2_condition.py:1002-1004); if this ever flips, spec §3.3's choice "
        "has to be revisited rather than the assertion deleted"
    )

    out.write(
        "// --- section 4: VideoConditionByLatentIndex (latent_cond.py:22-43) ---\n"
        "// clean_latent[start:stop] = tokens; denoise_mask[start:stop] = 1 - strength.\n"
        "// The NOISY tensor is deliberately untouched — that was asserted here.\n"
    )
    emit_scalar(out, "kLtx2ImgCondTokens", conditioned.latent.shape[1])
    emit_scalar(out, "kLtx2ImgCondWidth", conditioned.latent.shape[2])
    emit_scalar(out, "kLtx2ImgCondTargetFrames", COND_TARGET_FRAMES)
    emit_scalar(out, "kLtx2ImgCondPatch", COND_PATCH)
    emit_scalar(out, "kLtx2ImgCondLatentIdx", COND_LATENT_IDX)
    emit_double(out, "kLtx2ImgCondStrength", COND_STRENGTH)
    emit_double(out, "kLtx2ImgCondFps", COND_FPS)
    out.write("\n")
    emit_f32(out, "kLtx2ImgCondClean", conditioned.clean_latent.numpy())
    emit_f32(out, "kLtx2ImgCondMask", conditioned.denoise_mask.numpy())

    # --- section 5: the noiser, at a NON-UNIT scale.
    noise = torch.from_numpy(
        ltx_rand("ltx2.imgcond.noise", int(conditioned.latent.numel())).astype(np.float32)
    ).reshape(conditioned.latent.shape)

    class DeterministicNoiser(GaussianNoiser):
        def _sample_noise(self, latent_state):  # noqa: ARG002 - shape comes from the closure
            return noise

    noised = DeterministicNoiser(generator=None)(conditioned, noise_scale=NOISE_SCALE)

    # THE DIFFUSERS FORM, computed here only so the DIVERGENCE is a measured
    # number in the record rather than a claim. It is NOT emitted as a golden:
    # this port follows ltx_core (spec §3.3).
    diffusers_latent = conditioned.latent.clone()
    diffusers_latent = torch.where(
        conditioned.denoise_mask.unsqueeze(-1) < 1.0,
        conditioned.clean_latent,
        diffusers_latent,
    )
    diffusers_out = torch.lerp(diffusers_latent.float(), noise.float(), NOISE_SCALE)
    divergence = float((noised.latent - diffusers_out).abs().max())
    assert divergence > tol, (
        f"the ltx_core and diffusers compositions differ by only {divergence:g} at "
        f"noise_scale={NOISE_SCALE}, inside the band {tol:g} — section 5 would then not be "
        f"gating the choice spec §3.3 makes. Pick a scale that separates them"
    )

    out.write(
        "// --- section 5: GaussianNoiser over the conditioned state "
        "(components/noisers.py:30-37) ---\n"
        "// latent = lerp(latent, noise, noise_scale); latent = lerp(clean, latent, mask).\n"
        "// The DOUBLE lerp, at a NON-UNIT scale — the only regime in which ltx_core and\n"
        "// diffusers disagree. `kLtx2ImgNoiseDivergence` is how far apart they are here,\n"
        "// measured; the suite asserts it is above the band, which is what makes this\n"
        "// section able to catch a silent switch to the diffusers form.\n"
    )
    emit_double(out, "kLtx2ImgNoiseScale", NOISE_SCALE)
    emit_double(out, "kLtx2ImgNoiseDivergence", divergence)
    out.write("\n")
    emit_f32(out, "kLtx2ImgNoisedGolden", noised.latent.numpy())


# ---------------------------------------------------------------------------
# Upstream loading
# ---------------------------------------------------------------------------


class _StubModule(types.ModuleType):
    """A module whose every attribute is another stub. Registered for `av` and
    `OpenImageIO`, which `decode.py` imports at module scope and the crf==0 path
    never reaches. See the module docstring: the stub is a tripwire, not a
    shortcut — the assertions in section 2 are what prove upstream ran."""

    def __getattr__(self, name):
        # DUNDERS ARE NOT FABRICATED. `inspect` walks `sys.modules` looking for
        # `__file__` on every module while resolving a frame, and a stub that
        # answers it makes `os.path.splitext` raise from inside torch's custom-op
        # registration — a failure with nothing to do with this script. Anything
        # a real module would not have must stay absent.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        stub = _StubModule(f"{self.__name__}.{name}")
        setattr(self, name, stub)
        return stub

    def __call__(self, *args, **kwargs):
        return _StubModule(f"{self.__name__}()")


def load_upstream(root: Path):
    core_src = root / "packages" / "ltx-core" / "src"
    pipe_src = root / "packages" / "ltx-pipelines" / "src"
    if not (core_src / "ltx_core" / "model" / "video_vae" / "video_vae.py").is_file():
        raise SystemExit(f"no ltx_core under {core_src}; point --ltx2 at a Lightricks/LTX-2 tree")
    if not (pipe_src / "ltx_pipelines" / "utils" / "media_io" / "resize.py").is_file():
        raise SystemExit(f"no ltx_pipelines under {pipe_src}")
    sys.path.insert(0, str(core_src))
    sys.path.insert(0, str(pipe_src))

    for dep in ("av", "OpenImageIO"):
        if dep not in sys.modules:
            sys.modules[dep] = _StubModule(dep)

    # Register the package levels WITHOUT executing their __init__.py.
    for name, rel in (
        ("ltx_pipelines", "ltx_pipelines"),
        ("ltx_pipelines.utils", "ltx_pipelines/utils"),
        ("ltx_pipelines.utils.media_io", "ltx_pipelines/utils/media_io"),
    ):
        if name in sys.modules:
            continue
        package = types.ModuleType(name)
        package.__path__ = [str(pipe_src / rel)]
        package.__package__ = name
        sys.modules[name] = package

    import ltx_core  # noqa: PLC0415
    import ltx_pipelines.utils.media_io.decode as decode_mod  # noqa: PLC0415
    import ltx_pipelines.utils.media_io.range_map as range_mod  # noqa: PLC0415
    import ltx_pipelines.utils.media_io.resize as resize_mod  # noqa: PLC0415

    # ORACLE IDENTITY, asserted rather than assumed, on BOTH packages.
    for module, expected in ((ltx_core, core_src), (resize_mod, pipe_src)):
        resolved = Path(module.__file__).resolve()
        if not resolved.is_relative_to(expected.resolve()):
            raise SystemExit(
                f"{module.__name__} resolved to {resolved}, which is NOT under {expected}. "
                "Refusing to generate goldens from an oracle this script did not choose."
            )
    return resize_mod, range_mod, decode_mod


def upstream_revision(root: Path) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001 - a tarball checkout carries no git metadata
        return "unknown"
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"the LTX-2 checkout at {root} is DIRTY:\n{dirty}\n"
            "Refusing to generate: `git rev-parse HEAD` would stamp a CLEAN revision anchor "
            "onto goldens produced by a tree that commit cannot reproduce."
        )
    return done.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path,
                        help="a checkout of Lightricks/LTX-2 (the repo root)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tol = _read_golden_tol()
    root = args.ltx2.expanduser().resolve()
    resize_mod, range_mod, decode_mod = load_upstream(root)
    revision = upstream_revision(root)

    import torch

    torch.set_grad_enabled(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-image-cond-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// LTX-2.5 IMAGE CONDITIONING goldens (row LTX25-IMAGE-COND, issue #644),\n"
            "// produced by EXECUTING upstream Lightricks/LTX-2 at reduced dimensions on\n"
            "// CPU. Weights and pixels come from the shared deterministic stream, so no\n"
            "// weight byte and no image byte is checked in. Regenerate with:\n"
            "//   python3 scripts/gen-ltx2-image-cond-goldens.py --ltx2 <LTX-2 checkout>\n"
            "//       --out tests/vllm/multimodal/ltx2_image_cond_goldens.inc\n"
            "//\n"
            f"// Upstream revision: {revision}\n"
            "//\n"
            "// See .agents/specs/ltx25-image-conditioning.md section 5.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
            "// The upstream tree these numbers came from. The suite asserts this equals\n"
            "// the SHA it pins, so regenerating against a DIFFERENT checkout fails the\n"
            "// gate instead of silently replacing the oracle.\n"
            f'inline constexpr const char* kLtx2ImgCondUpstreamRevision = "{revision}";\n\n'
        )
        section_resize(out, resize_mod)
        image_chw = section_preprocess(out, resize_mod, range_mod, decode_mod, tol)
        latent = section_encode(out, image_chw)
        section_condition_and_noise(out, latent, tol)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
