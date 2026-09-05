#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_vae_goldens.inc — the LTX-2.5 VAE parity oracle.

LTX-2.5's decoders are pure-Python `ltx_core` modules (Lightricks/LTX-2,
`packages/ltx-core/src/ltx_core/model/`). The full checkpoint is ~30 GB and its
DiT does not fit this project's CI, but the VAE MATH is gateable exactly on any
CPU: this generator imports the upstream modules VERBATIM, builds them at REDUCED
dimensions with deterministic pseudo-random weights, runs them, and emits the
resulting tensors as C++ goldens. The C++ suite regenerates the identical weights
and inputs from the identical PRNG and must reproduce these outputs, so NO WEIGHT
BYTE is checked in. This mirrors the method that made the MiniMax-H3 VAE bricks
trustworthy (scripts/gen-minimax-h3-audio-vae-goldens.py).

Upstream sources (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/):
  model/audio_vae/audio_vae.py          -> section 1 (audio decoder)
  model/audio_vae/vocoder.py            -> sections 2-4 (vocoder, legacy arm, BWE)
  model/video_vae/conv_video_decoder.py -> section 5 (Conv video decoder)
  model/video_vae/video_vae.py          -> section 6 (VideoEncoder)
  model/audio_vae/audio_vae.py          -> section 7 (AudioEncoder)
  model/audio_vae/ops.py                -> section 8 (AudioProcessor mel front-end,
                                           and 8d-8f its RESAMPLER)
  conditioning/types/*.py               -> section 9 (the conditioning items)

Usage:
    python3 scripts/gen-ltx2-vae-goldens.py \\
        --ltx2 ~/_git/LTX-2 \\
        --out tests/vllm/models/ltx2_vae_goldens.inc

Needs torch + numpy + einops (CPU only). `ltx_core` is imported with a single
sys.path insert; no checkpoint, venv, or gated download is involved.

UPSTREAM REVISION ANCHOR. The goldens are only interpretable against the exact
upstream tree that produced them: if Lightricks changes `video_vae/resnet.py` and
someone regenerates, the numbers move, and without a SHA nobody can tell whether
the PORT drifted or UPSTREAM did, nor bisect from anywhere. So this generator
resolves `git -C <--ltx2> rev-parse HEAD` at generation time and emits it into the
`.inc` twice: once as a header comment for a human, and once as
`kLtx2VaeUpstreamRevision`, which the C++ suite asserts against the SHA PINNED in
the test. That makes the anchor load-bearing rather than decorative — a
regeneration against a different checkout fails the gate instead of silently
replacing the oracle.

  Pinned revision: fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca

Advancing the pin is a deliberate, reviewable edit in BOTH places (here and
`kLtx2VaeUpstreamRevisionPin` in tests/vllm/models/test_ltx2_vae.cpp), never a
side effect of regenerating.

ORACLE IDENTITY, asserted rather than assumed. `sys.path.insert(0, ...)` normally
wins, but path precedence is not a guarantee this script should be staking the
oracle on: a `.pth` file, an editable install, a namespace-package layout or a
later refactor to `sys.path.append` all make a pip-installed `ltx_core` resolve
instead, and it would import silently and gate against the wrong source. So the
resolved `ltx_core.__file__` is checked to live under `--ltx2` before anything
runs. This mirrors scripts/gen-ltx2-goldens.py (phase L2).

Two harness adaptations, both recorded because they change nothing about the math:

  * `ConvVideoDecoder` injects Gaussian noise through `torch.randn` (decoder
    timestep conditioning, and `inject_noise` blocks). `torch.randn` is patched
    for the duration of the call to draw from the shared deterministic stream,
    keyed by CALL INDEX — exactly the ordering guarantee an upstream
    `torch.Generator` gives. The C++ side consumes an Ltx2NoiseStream in the same
    order.
  * Anti-aliasing filters (`*.filter`) are SKIPPED when filling weights: they are
    kaiser-sinc / hann-sinc windows COMPUTED at construction, never loaded, and
    both sides must build them independently. They are gated on their own
    (sections 2 and 4).
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# THE GOLDEN BAND, READ from the C++ suite rather than repeated here.
# `kLtx2GoldenTol` (tests/vllm/models/test_ltx2_vae.cpp) is the ONE authority on
# what "green" means for these goldens, and section 5d asserts against it before
# emitting — an arm that exists to make a constant reachable is worthless if it
# does not clear the band the C++ side actually applies. A literal `5e-6` here
# would be a second definition of one number in a second language, and the two
# would drift the moment either moved: a widened C++ band would leave this
# generator certifying arms against a band nobody uses, and a tightened one would
# let it emit arms the suite already rejects.
#
# So the value is PARSED from that file, and a parse that does not find EXACTLY
# ONE definition is fatal. An anchor that silently stops matching is how a gate
# goes quiet, which is the failure this whole section of the suite exists to
# prevent.
# ---------------------------------------------------------------------------

_GOLDEN_TOL_SOURCE = (
    Path(__file__).resolve().parents[1] / "tests" / "vllm" / "models" / "test_ltx2_vae.cpp"
)


def _read_golden_tol() -> float:
    text = _GOLDEN_TOL_SOURCE.read_text(encoding="utf-8")
    hits = re.findall(r"^constexpr double kLtx2GoldenTol = ([0-9eE.+-]+);", text, re.M)
    assert len(hits) == 1, (
        f"expected EXACTLY ONE `constexpr double kLtx2GoldenTol = ...;` in "
        f"{_GOLDEN_TOL_SOURCE}, found {len(hits)} — the generator cannot assert "
        f"against a band it cannot resolve"
    )
    return float(hits[0])


GOLDEN_TOL = _read_golden_tol()


# ---------------------------------------------------------------------------
# Deterministic weight/input stream, mirrored bit-for-bit by the C++ suite
# (tests/vllm/models/test_ltx2_vae.cpp :: Ltx2Rand). A per-tensor FNV-1a seed plus
# a splitmix64 counter makes every tensor independent of fill ORDER, so the two
# sides cannot silently drift by reordering their parameter construction. It is
# the same stream the MiniMax-H3 goldens use.
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


# ---------------------------------------------------------------------------
# The per-parameter role rule. Both sides implement it; the C++ side calls it
# with the same (name, rank), so a divergence here shows up as a golden mismatch
# rather than as a silently different tensor.
# ---------------------------------------------------------------------------


def param_values(name: str, shape) -> np.ndarray:
    count = int(np.prod(shape)) if len(shape) else 1
    rank = len(shape)
    if name == "timestep_scale_multiplier":
        # Upstream's own value (conv_video_decoder.py:257); not a random weight.
        return np.full(count, 1000.0)
    if name.endswith("mel_basis"):
        # A mel filterbank is NON-NEGATIVE. Signed values would push most bins
        # under log's 1e-5 clamp, and a saturated golden hides errors.
        basis = np.abs(ltx_rand(name, count)) * 0.2 + 0.05
        # ...EXCEPT for the one arm that exists precisely to saturate it. The
        # clamp `torch.clamp(mel, min=1e-5)` (vocoder.py:515) sets the floor of the
        # log-mel the bwe_generator consumes, and it is the member of the
        # invisible-constant class that BINDS IN PRODUCTION, because real silence
        # reaches it. The well-scaled basis above never can — measured: the raw
        # mel minimum is ~4.4e-3, and it stays there even for a zero input,
        # because the vocoder's conv biases keep the waveform off silence. Scaling
        # the basis by 1e-4 puts EVERY bin under the clamp, so on this arm the
        # constant alone decides the generator's input and a mutation of it moves
        # the golden.
        if ".bwequiet." in name:
            basis *= 1e-4
        return basis
    if name.endswith(".gamma"):
        return ltx_rand(name, count) * 0.1 + 1.0
    if name.endswith(".alpha") or name.endswith(".beta"):
        return ltx_rand(name, count) * 0.2
    if name.endswith("std-of-means"):
        return ltx_rand(name, count) * 0.1 + 1.0
    if name.endswith("mean-of-means"):
        return ltx_rand(name, count) * 0.1
    if name.endswith("scale_shift_table"):
        return ltx_rand(name, count) * 0.1
    if name.endswith("per_channel_scale1") or name.endswith("per_channel_scale2"):
        return ltx_rand(name, count) * 0.1
    if name.endswith(".bias"):
        return ltx_rand(name, count) * 0.05
    if rank == 1 and name.endswith(".weight"):
        # A 1-D `.weight` is an affine norm gain, initialized to ones upstream.
        return ltx_rand(name, count) * 0.1 + 1.0
    return ltx_rand(name, count) * 0.1


def fill_from_stream(module, prefix: str = "") -> list[tuple[str, int]]:
    """Overwrite every parameter/buffer from the shared stream.

    Returns the (name, count) manifest in state_dict order, which the C++ side
    asserts its own parameter bag matches EXACTLY — so a parameter one side
    builds and the other does not is a test failure, not a silent no-op.
    """
    import torch

    state = module.state_dict()
    manifest: list[tuple[str, int]] = []
    filled = {}
    for name, tensor in state.items():
        if name.endswith(".filter"):
            # kaiser-sinc / hann-sinc windows are COMPUTED, never loaded.
            filled[name] = tensor
            continue
        values = param_values(prefix + name, tuple(tensor.shape))
        filled[name] = torch.from_numpy(values.astype(np.float32)).reshape(tensor.shape)
        manifest.append((prefix + name, int(values.size)))
    module.load_state_dict(filled, strict=True)
    return manifest


def make_input(name: str, shape, scale: float):
    import torch

    count = int(np.prod(shape))
    values = ltx_rand(name, count) * scale
    return torch.from_numpy(values.astype(np.float32)).reshape(shape)


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
# Reduced-dimension architectures. Every structural ratio the port branches on is
# preserved; only the magnitudes shrink.
# ---------------------------------------------------------------------------

# Section 1 — audio decoder. attn_resolutions={8} makes the DEEPEST up level carry
# attention (curr_res = resolution // 2**(levels-1) = 8), so both the mid-block
# attention and the per-level attention list are exercised. causality_axis=height
# is the shipped default (model_configurator.py:134): dim 2 is TIME, so all its
# padding must land on the LEFT.
AUDIO_DEC = dict(
    ch=8,
    out_ch=2,
    ch_mult=(1, 2, 4),
    num_res_blocks=1,
    attn_resolutions={8},
    resolution=32,
    z_channels=4,
    dropout=0.0,
    mid_block_add_attention=True,
    sample_rate=16000,
    mel_hop_length=160,
    is_causal=True,
)
AUDIO_DEC_LATENT_T = 3
AUDIO_DEC_LATENT_F = 2  # z_channels * mel_bins(latent) must equal `ch` (patchified width)

# Sections 1d / 7d — the GROUP-NORM arms, which exist so `norm_eps` is REACHABLE.
#
# `build_normalization_layer` has two branches (common/normalization.py:56-59) and
# every arm above takes the PIXEL one, which is parameter-free and reads
# `pixel_norm_eps`. On those arms `norm_eps` is not merely inert, it is never
# LOADED: nothing in the audio VAE ever touches the GroupNorm branch, so mutating
# the constant 100x moved no golden at all.
#
# `norm_type=group` is legal upstream, but NOT on defaults alone: both
# `AudioEncoder.__init__` and `AudioDecoder.__init__` declare `norm_type = GROUP`
# (audio_vae.py:82, :294) and, on the very next line, `causality_axis = WIDTH`
# (audio_vae.py:83, :295) — a pair `ResnetBlock` refuses with
# `ValueError: Causal ResnetBlock with GroupNorm is not supported`
# (audio_vae/resnet.py:130-131). Constructing either class on pure defaults raises,
# verified by construction against the pinned tree. What a checkpoint may legally
# declare, and what these arms gate, is GROUP alongside `causality_axis=NONE`.
#
# TWO REDUCED DIMENSIONS MOVE, and both are forced, not chosen:
#  * `ch` becomes 32 because `build_normalization_layer` forwards its own
#    `num_groups` keyword, whose default is 32 (normalization.py:44, 56), and no
#    audio_vae call site passes one — so `torch.nn.GroupNorm` refuses any channel
#    count that 32 does not divide. Every level is then 32/64/128.
#  * `z_channels` becomes 16 because `PerChannelStatistics(latent_channels=ch)`
#    indexes the patchified `(c, f)` axis, so `z_channels * mel_bins(latent)` has
#    to equal the new `ch` (audio_vae.py:118, 318).
# No other DIMENSION changes; the rest of the diff against the pixel arms is the
# norm type and the causality axis it forces, which is what the arms are for.
AUDIO_GROUP_DEC = {**AUDIO_DEC, "ch": 32, "z_channels": 16}
AUDIO_GROUP_DEC_LATENT_T = 3
AUDIO_GROUP_DEC_LATENT_F = 2
# The mel-bin count the network itself produces (latent F doubled once per level
# transition), so `_adjust_output_shape` neither crops nor pads and the golden is
# the decoder's own output.
AUDIO_GROUP_DEC_MEL_BINS = 8

# Section 2 — BigVGAN v2 vocoder (resblock "AMP1", snakebeta). conv_pre's input is
# hardcoded to 128 upstream (2 stereo channels x 64 mel bins), so the reduced arm
# keeps 64 mel bins and shrinks everything else.
VOC = dict(
    resblock_kernel_sizes=[3, 7],
    upsample_rates=[2, 2],
    upsample_kernel_sizes=[4, 4],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    upsample_initial_channel=16,
    resblock="AMP1",
    output_sampling_rate=16000,
    activation="snakebeta",
    use_tanh_at_final=True,
    apply_final_activation=True,
    use_bias_at_final=True,
)
VOC_FRAMES = 5
VOC_MEL_BINS = 64
# Large enough that the convolution path, not conv_post's bias, dominates the
# golden: a bias-dominated golden is nearly constant and gates almost nothing.
VOC_INPUT_SCALE = 1.0

# Section 3 — the LEGACY resblock "1" arm (ResBlock1 + leaky ReLU, no anti-aliased
# activation). Pre-2.3 checkpoints select it (model_configurator.py:53).
VOC_LEGACY = dict(
    resblock_kernel_sizes=[3],
    upsample_rates=[2],
    upsample_kernel_sizes=[4],
    resblock_dilation_sizes=[[1, 3, 5]],
    upsample_initial_channel=8,
    resblock="1",
    output_sampling_rate=16000,
)

# Section 4 — VocoderWithBWE. hop_length 8 against a 20-sample vocoder output
# leaves a remainder, so the pad-to-a-multiple-of-hop branch is exercised, and the
# bwe upsample product (16) x mel frames (3) matches 2 x padded length (48).
BWE_GEN = dict(
    resblock_kernel_sizes=[3],
    upsample_rates=[4, 4],
    upsample_kernel_sizes=[8, 8],
    resblock_dilation_sizes=[[1, 3, 5]],
    upsample_initial_channel=16,
    resblock="AMP1",
    output_sampling_rate=32000,
    activation="snakebeta",
    apply_final_activation=False,
)
BWE = dict(filter_length=16, hop_length=8, win_length=16, n_mel_channels=64,
           input_sampling_rate=16000, output_sampling_rate=32000)

# Section 5 — Conv video decoder. The block list covers every block kind the
# decoder can build: res_x (with inject_noise), compress_all (residual),
# res_x_y (channel-halving, so norm3 + conv_shortcut are live), compress_space,
# attn, compress_time. `attn_res_x` is deliberately absent: upstream passes
# `attention_head_dim` to UNetMidBlock3D, which does not accept it, so the block
# cannot be constructed at this revision (see the C++ refusal).
VIDEO_BLOCKS = [
    ("res_x", {"num_layers": 1, "inject_noise": True}),
    ("compress_all", {"multiplier": 2, "residual": True}),
    ("res_x_y", {"num_layers": 1, "multiplier": 2}),
    ("compress_space", {"multiplier": 1}),
    ("attn", {"num_layers": 1}),
    ("compress_time", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
]
VIDEO_DEC = dict(
    convolution_dimensions=3,
    in_channels=6,
    out_channels=3,
    patch_size=2,
    causal=True,
    timestep_conditioning=True,
    base_channels=8,
)
VIDEO_LATENT = (1, 6, 3, 2, 2)

# Section 5d — the arm on which `norm_eps` is a FIRST-ORDER term.
#
# `norm_eps` is NOT unreachable on the video decoder, and the earlier record that
# called it "invisible" was wrong about the reason. `ResnetBlock3D.__init__`
# builds `norm3 = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps)`
# whenever `in_channels != out_channels` (resnet.py:93-97) — REGARDLESS of
# `norm_layer` — and `forward` applies it to the residual (resnet.py:178). So
# every `res_x_y` block reads the constant even on the PIXEL_NORM arms, and
# section 5 above has one.
#
# What was actually true is a sensitivity statement about the FIXTURE, not a
# reachability statement about the constant. norm3 divides by
# `sqrt(var + eps)` over ALL of (C, T, H, W), and on section 5's arm that
# variance is ~0.2 at a norm3 that sits five blocks deep, so 1e-6 -> 1e-4 moves
# the golden 1.8e-6 — under the 5e-6 band — while 1e-6 -> 1.0 moves it 1.6e-2.
# The band accepted a 100x error only because the denominator was large.
#
# This arm removes that accident instead of recording it. ONE `res_x_y` block, so
# norm3 sits directly behind conv_in, and a latent at a tenth of the usual scale,
# so the variance it competes with is ~5e-3 rather than ~0.2. Every mutation of
# the constant then moves the golden by 1e-5 or more, INCLUDING eps -> 0, which
# no other arm in this file can see. Nothing else about the arm is unusual: the
# weights come from the same stream, the padding mode and causality match section
# 5, and `timestep_conditioning` is off only because the epsilon is what is under
# test and the noise stream is not.
VIDEO_EPS_BLOCKS = [("res_x_y", {"num_layers": 1, "multiplier": 2})]
VIDEO_EPS_DEC = dict(
    convolution_dimensions=3,
    in_channels=6,
    out_channels=3,
    patch_size=2,
    causal=True,
    timestep_conditioning=False,
    base_channels=8,
)
VIDEO_EPS_LATENT = (1, 6, 3, 2, 2)
VIDEO_EPS_LATENT_SCALE = 0.1


def section_audio_decoder(out) -> None:
    from ltx_core.model.audio_vae.audio_vae import AudioDecoder
    from ltx_core.model.audio_vae.causality_axis import CausalityAxis
    from ltx_core.model.common.normalization import NormType

    mel_bins = AUDIO_DEC["ch"] // AUDIO_DEC["z_channels"] * 4  # 8 output mel bins
    decoder = AudioDecoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=mel_bins,
        **AUDIO_DEC,
    ).eval()
    manifest = fill_from_stream(decoder, prefix="ltx2.audiodec.")
    latent = make_input(
        "ltx2.audiodec.input",
        (1, AUDIO_DEC["z_channels"], AUDIO_DEC_LATENT_T, AUDIO_DEC_LATENT_F),
        1.0,
    )
    y = decoder(latent)

    out.write("// --- section 1: AudioDecoder (audio_vae.py:277-494) ---\n")
    emit_scalar(out, "kLtx2AudioDecLatentT", AUDIO_DEC_LATENT_T)
    emit_scalar(out, "kLtx2AudioDecLatentF", AUDIO_DEC_LATENT_F)
    emit_scalar(out, "kLtx2AudioDecOutFrames", y.shape[2])
    emit_scalar(out, "kLtx2AudioDecOutMelBins", y.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioDecParam", manifest)
    emit_f32(out, "kLtx2AudioDecGolden", y.numpy())

    # The same decoder asked for MORE mel bins than the network produces: upstream
    # zero-pads on the right of the frequency axis (audio_vae.py:458-467). A port
    # that silently returns the unpadded tensor passes every other assertion.
    decoder_pad = AudioDecoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=mel_bins + 3,
        **AUDIO_DEC,
    ).eval()
    fill_from_stream(decoder_pad, prefix="ltx2.audiodec.")
    y_pad = decoder_pad(latent)
    emit_scalar(out, "kLtx2AudioDecPadOutMelBins", y_pad.shape[3])
    out.write("\n")
    emit_f32(out, "kLtx2AudioDecPadGolden", y_pad.numpy())

    # --- section 1b: what "causal" actually reaches ---
    # Measured, not assumed. The decoder's AttnBlocks attend over the WHOLE
    # (time, mel) map (attention.py:31-55), so with the shipped attention on, a
    # change anywhere reaches every output frame and the causality claim is only
    # about the CONVOLUTIONS. Turning attention off isolates exactly the trap this
    # port must not fall into — a symmetric temporal pad instead of a one-sided
    # one — and upstream itself says which frames may move.
    causal = AudioDecoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        mel_bins=mel_bins,
        **{**AUDIO_DEC, "attn_resolutions": set(), "mid_block_add_attention": False},
    ).eval()
    causal_manifest = fill_from_stream(causal, prefix="ltx2.audiodeccausal.")
    import torch

    bumped = latent.clone()
    bumped[:, :, -1, :] += 3.0
    base_out = causal(latent)
    bumped_out = causal(bumped)
    moved = [
        t
        for t in range(base_out.shape[2])
        if not torch.equal(base_out[:, :, t], bumped_out[:, :, t])
    ]
    assert moved, "the causality probe must move SOMETHING or it gates nothing"
    out.write("// --- section 1b: convolution-only causal reach (attention off) ---\n")
    emit_scalar(out, "kLtx2AudioDecCausalFirstMoved", moved[0])
    emit_scalar(out, "kLtx2AudioDecCausalLastMoved", moved[-1])
    # With the SHIPPED attention on, upstream moves EVERY frame; the C++ side
    # asserts that too, so the port cannot claim a causality it does not have.
    out.write("\n")
    emit_manifest(out, "kLtx2AudioDecCausalParam", causal_manifest)

    # --- section 1c: the OTHER THREE causality axes ---
    # Everything above runs `causality_axis=height`, the shipped default. The
    # remaining three arms of `CausalConv2d`'s padding switch (causal_conv_2d.py,
    # via causality_axis.py:4-10) were never executed, so the port's pad split for
    # them was an untested claim. They are cheap to gate and one of them is subtle:
    # WIDTH_COMPATIBILITY pads the width one-sidedly like WIDTH, but the
    # UPSAMPLER treats it differently (upsample.py:44-48 does NOT drop the first
    # element for that axis, while WIDTH and HEIGHT do), so the two axes are not
    # interchangeable however similar their convolution padding looks.
    for label, axis in (
        ("None", CausalityAxis.NONE),
        ("Width", CausalityAxis.WIDTH),
        ("WidthCompat", CausalityAxis.WIDTH_COMPATIBILITY),
    ):
        arm = AudioDecoder(
            norm_type=NormType.PIXEL,
            causality_axis=axis,
            mel_bins=mel_bins,
            **AUDIO_DEC,
        ).eval()
        arm_manifest = fill_from_stream(arm, prefix="ltx2.audiodec.")
        y_axis = arm(latent)
        out.write(f"// --- section 1c: causality_axis = {axis.name} ---\n")
        emit_scalar(out, f"kLtx2AudioDec{label}OutFrames", y_axis.shape[2])
        emit_scalar(out, f"kLtx2AudioDec{label}OutMelBins", y_axis.shape[3])
        out.write("\n")
        emit_manifest(out, f"kLtx2AudioDec{label}Param", arm_manifest)
        emit_f32(out, f"kLtx2AudioDec{label}Golden", y_axis.numpy())

    # --- section 1d: norm_type = GROUP, the arm that READS `norm_eps` ---
    # See AUDIO_GROUP_DEC for why this arm exists and why its two dimensions move.
    # GroupNorm is the only consumer of `norm_eps` in the audio VAE, and it also
    # makes `num_groups` load-bearing: the norms carry `weight`/`bias` here, which
    # PixelNorm does not, so the parameter manifest differs too.
    group = AudioDecoder(
        norm_type=NormType.GROUP,
        causality_axis=CausalityAxis.NONE,
        mel_bins=AUDIO_GROUP_DEC_MEL_BINS,
        **AUDIO_GROUP_DEC,
    ).eval()
    group_manifest = fill_from_stream(group, prefix="ltx2.audiodecgroup.")
    group_latent = make_input(
        "ltx2.audiodecgroup.input",
        (1, AUDIO_GROUP_DEC["z_channels"], AUDIO_GROUP_DEC_LATENT_T, AUDIO_GROUP_DEC_LATENT_F),
        1.0,
    )
    y_group = group(group_latent)
    out.write("// --- section 1d: norm_type = GROUP (normalization.py:56-57) ---\n")
    emit_scalar(out, "kLtx2AudioDecGroupLatentC", AUDIO_GROUP_DEC["z_channels"])
    emit_scalar(out, "kLtx2AudioDecGroupLatentT", AUDIO_GROUP_DEC_LATENT_T)
    emit_scalar(out, "kLtx2AudioDecGroupLatentF", AUDIO_GROUP_DEC_LATENT_F)
    emit_scalar(out, "kLtx2AudioDecGroupOutFrames", y_group.shape[2])
    emit_scalar(out, "kLtx2AudioDecGroupOutMelBins", y_group.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioDecGroupParam", group_manifest)
    emit_f32(out, "kLtx2AudioDecGroupGolden", y_group.numpy())


def _vocoder(cfg):
    from ltx_core.model.audio_vae.vocoder import Vocoder

    return Vocoder(**cfg).eval()


def section_vocoder(out) -> None:
    from ltx_core.model.audio_vae.vocoder import kaiser_sinc_filter1d

    voc = _vocoder(VOC)
    manifest = fill_from_stream(voc, prefix="ltx2.voc.")
    mel = make_input("ltx2.voc.input", (1, 2, VOC_FRAMES, VOC_MEL_BINS), VOC_INPUT_SCALE)
    y = voc(mel)

    out.write("// --- section 2: Vocoder, BigVGAN v2 arm (vocoder.py:293-438) ---\n")
    emit_scalar(out, "kLtx2VocFrames", VOC_FRAMES)
    emit_scalar(out, "kLtx2VocMelBins", VOC_MEL_BINS)
    emit_scalar(out, "kLtx2VocOutSamples", y.shape[2])
    out.write("\n")
    # The anti-aliased activation's kaiser-sinc filter is COMPUTED, never loaded.
    # Gate it first: a wrong filter makes every SnakeBeta wrong and the decoder
    # mismatch impossible to localize.
    emit_f32(out, "kLtx2VocUpFilterGolden",
             kaiser_sinc_filter1d(cutoff=0.5 / 2, half_width=0.6 / 2, kernel_size=12).reshape(-1))
    emit_manifest(out, "kLtx2VocParam", manifest)
    emit_f32(out, "kLtx2VocGolden", y.numpy())

    # --- section 2b: resblock "AMP1" with activation "snake" ---
    # The arm that proves act_post is NOT governed by `activation`. Upstream builds
    # `self.act_post = Activation1d(SnakeBeta(final_channels))` (vocoder.py:388)
    # inside `if self.is_amp` and passes it no `activation=` argument, unlike the
    # resblocks one line earlier (vocoder.py:376). So on THIS arm every resblock
    # activation is plain Snake — which reuses ALPHA as its reciprocal scale
    # (vocoder.py:198) — while act_post still reads `.beta`. A port that keys
    # act_post off `activation` produces a plausible waveform from the wrong
    # scale, and no other arm can tell, because on the shipped snakebeta arm the
    # two agree.
    voc_snake = _vocoder({**VOC, "activation": "snake"})
    snake_manifest = fill_from_stream(voc_snake, prefix="ltx2.vocsnake.")
    y_snake = voc_snake(mel)
    out.write('// --- section 2b: AMP1 with activation "snake" (act_post stays SnakeBeta) ---\n')
    emit_scalar(out, "kLtx2VocSnakeOutSamples", y_snake.shape[2])
    out.write("\n")
    emit_manifest(out, "kLtx2VocSnakeParam", snake_manifest)
    emit_f32(out, "kLtx2VocSnakeGolden", y_snake.numpy())


def section_vocoder_legacy(out) -> None:
    voc = _vocoder(VOC_LEGACY)
    manifest = fill_from_stream(voc, prefix="ltx2.vocleg.")
    mel = make_input("ltx2.voc.input", (1, 2, VOC_FRAMES, VOC_MEL_BINS), VOC_INPUT_SCALE)
    y = voc(mel)

    out.write('// --- section 3: Vocoder, legacy resblock "1" arm (resnet.py:12-80) ---\n')
    emit_scalar(out, "kLtx2VocLegacyOutSamples", y.shape[2])
    out.write("\n")
    emit_manifest(out, "kLtx2VocLegacyParam", manifest)
    emit_f32(out, "kLtx2VocLegacyGolden", y.numpy())


def section_bwe(out) -> None:
    from ltx_core.model.audio_vae.vocoder import MelSTFT, UpSample1d, VocoderWithBWE

    voc = _vocoder(VOC)
    gen = _vocoder(BWE_GEN)
    mel_stft = MelSTFT(
        filter_length=BWE["filter_length"],
        hop_length=BWE["hop_length"],
        win_length=BWE["win_length"],
        n_mel_channels=BWE["n_mel_channels"],
    )
    bwe = VocoderWithBWE(
        vocoder=voc,
        bwe_generator=gen,
        mel_stft=mel_stft,
        input_sampling_rate=BWE["input_sampling_rate"],
        output_sampling_rate=BWE["output_sampling_rate"],
        hop_length=BWE["hop_length"],
    ).eval()
    manifest = fill_from_stream(bwe, prefix="ltx2.bwe.")
    mel = make_input("ltx2.voc.input", (1, 2, VOC_FRAMES, VOC_MEL_BINS), VOC_INPUT_SCALE)
    y = bwe(mel)

    out.write("// --- section 4: VocoderWithBWE (vocoder.py:519-630) ---\n")
    emit_scalar(out, "kLtx2BweOutSamples", y.shape[2])
    out.write("\n")
    # The hann-sinc resampler filter is persistent=False: it is COMPUTED, never in
    # the checkpoint, and it is NOT the kaiser filter the activations use.
    resampler = UpSample1d(ratio=2, persistent=False, window_type="hann")
    emit_scalar(out, "kLtx2BweResamplerKernel", resampler.kernel_size)
    out.write("\n")
    emit_f32(out, "kLtx2BweResamplerFilterGolden", resampler.filter.reshape(-1).numpy())
    emit_manifest(out, "kLtx2BweParam", manifest)
    emit_f32(out, "kLtx2BweGolden", y.numpy())

    # --- section 4b: the arm where the mel log CLAMP actually binds ---
    # Everything above leaves `torch.clamp(mel, min=1e-5)` (vocoder.py:515) inert:
    # the raw mel minimum is ~4.4e-3. So the clamp is an INVISIBLE CONSTANT there,
    # and mutation confirms 1e-5 -> 1e-8 changes nothing. This arm attenuates
    # mel_basis by 1e-4 (see param_values) so every bin lands under the clamp and
    # the constant alone decides what the bwe_generator sees — which is what real
    # silence does in production. The saturation COUNT is emitted too, so the C++
    # side asserts the probe is actually saturated rather than trusting it.
    quiet_voc = _vocoder(VOC)
    quiet_gen = _vocoder(BWE_GEN)
    quiet_stft = MelSTFT(
        filter_length=BWE["filter_length"],
        hop_length=BWE["hop_length"],
        win_length=BWE["win_length"],
        n_mel_channels=BWE["n_mel_channels"],
    )
    quiet = VocoderWithBWE(
        vocoder=quiet_voc,
        bwe_generator=quiet_gen,
        mel_stft=quiet_stft,
        input_sampling_rate=BWE["input_sampling_rate"],
        output_sampling_rate=BWE["output_sampling_rate"],
        hop_length=BWE["hop_length"],
    ).eval()
    quiet_manifest = fill_from_stream(quiet, prefix="ltx2.bwequiet.")
    y_quiet = quiet(mel)

    # Recompute the RAW (pre-clamp) mel exactly as VocoderWithBWE.forward does, to
    # count how many values the clamp is holding up.
    import torch

    low = quiet_voc(mel)
    pad = (-low.shape[-1]) % BWE["hop_length"]
    padded = torch.nn.functional.pad(low, (0, pad))
    magnitude, _ = quiet_stft.stft_fn(padded.reshape(-1, padded.shape[-1]))
    raw_mel = torch.matmul(quiet_stft.mel_basis.to(magnitude.dtype), magnitude)
    saturated = int((raw_mel < 1e-5).sum())
    assert saturated == raw_mel.numel(), (
        f"the saturating probe must saturate EVERY bin or it gates nothing: "
        f"{saturated}/{raw_mel.numel()}, min={float(raw_mel.min()):g}"
    )
    out.write("// --- section 4b: the BWE mel log clamp, SATURATED (vocoder.py:515) ---\n")
    emit_scalar(out, "kLtx2BweQuietSaturatedBins", saturated)
    emit_scalar(out, "kLtx2BweQuietOutSamples", y_quiet.shape[2])
    out.write("\n")
    emit_manifest(out, "kLtx2BweQuietParam", quiet_manifest)
    emit_f32(out, "kLtx2BweQuietGolden", y_quiet.numpy())


def section_conv_video_decoder(out) -> None:
    import torch

    from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType

    decoder = ConvVideoDecoder(
        decoder_blocks=VIDEO_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **VIDEO_DEC,
    ).eval()
    manifest = fill_from_stream(decoder, prefix="ltx2.videodec.")
    latent = make_input("ltx2.videodec.input", VIDEO_LATENT, 1.0)

    # Patch torch.randn to the shared stream, keyed by CALL INDEX. Upstream's own
    # determinism knob is a torch.Generator, which is likewise call-ordered; the
    # C++ Ltx2NoiseStream consumes the identical sequence.
    draws: list[int] = []
    real_randn = torch.randn

    def patched_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(draws)}", count)
        draws.append(count)
        return torch.from_numpy(values.astype(np.float32)).reshape(shape)

    torch.randn = patched_randn
    try:
        y = decoder(latent)
    finally:
        torch.randn = real_randn

    out.write("// --- section 5: ConvVideoDecoder (conv_video_decoder.py:146-357) ---\n")
    emit_scalar(out, "kLtx2VideoDecLatentC", VIDEO_LATENT[1])
    emit_scalar(out, "kLtx2VideoDecLatentT", VIDEO_LATENT[2])
    emit_scalar(out, "kLtx2VideoDecLatentH", VIDEO_LATENT[3])
    emit_scalar(out, "kLtx2VideoDecLatentW", VIDEO_LATENT[4])
    emit_scalar(out, "kLtx2VideoDecOutC", y.shape[1])
    emit_scalar(out, "kLtx2VideoDecOutT", y.shape[2])
    emit_scalar(out, "kLtx2VideoDecOutH", y.shape[3])
    emit_scalar(out, "kLtx2VideoDecOutW", y.shape[4])
    emit_scalar(out, "kLtx2VideoDecNoiseDraws", len(draws))
    out.write("inline constexpr int64_t kLtx2VideoDecNoiseCounts[] = {\n    "
              + ", ".join(str(c) for c in draws) + ",\n};\n\n")
    emit_manifest(out, "kLtx2VideoDecParam", manifest)
    emit_f32(out, "kLtx2VideoDecGolden", y.numpy())

    # --- section 5b: what "causal" actually reaches ---
    # Measured, not assumed, exactly like section 1b. `res_x_y`'s shortcut norm is
    # a GroupNorm with ONE group over (C, T, H, W) (resnet.py:93-97), whose
    # statistics span TIME, so the shipped block list is not end-to-end causal
    # however correct the padding is. Stripping it leaves a decoder whose reach is
    # decided by the causal padding alone — which is the trap worth gating.
    causal_blocks = [
        ("res_x", {"num_layers": 1}),
        ("compress_all", {"multiplier": 1, "residual": False}),
        ("res_x", {"num_layers": 1}),
    ]
    causal_dec = ConvVideoDecoder(
        decoder_blocks=causal_blocks,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **{**VIDEO_DEC, "timestep_conditioning": False},
    ).eval()
    causal_manifest = fill_from_stream(causal_dec, prefix="ltx2.videodeccausal.")
    bumped = latent.clone()
    bumped[:, :, -1] += 5.0
    base_out = causal_dec(latent)
    bumped_out = causal_dec(bumped)
    moved = [
        t
        for t in range(base_out.shape[2])
        if not torch.equal(base_out[:, :, t], bumped_out[:, :, t])
    ]
    assert moved, "the causality probe must move SOMETHING or it gates nothing"
    out.write("// --- section 5b: convolution-only causal reach (no res_x_y, no timestep) ---\n")
    emit_scalar(out, "kLtx2VideoDecCausalOutT", base_out.shape[2])
    emit_scalar(out, "kLtx2VideoDecCausalFirstMoved", moved[0])
    emit_scalar(out, "kLtx2VideoDecCausalLastMoved", moved[-1])
    out.write("\n")
    emit_manifest(out, "kLtx2VideoDecCausalParam", causal_manifest)

    # --- section 5c: causal=False, which is UPSTREAM'S OWN DEFAULT ---
    # `ConvVideoDecoder.__init__` declares `causal: bool = False`
    # (conv_video_decoder.py:184) and its docstring calls that the standard
    # decoder, yet every arm above runs causal=True. The non-causal branch is a
    # DIFFERENT padding rule, not a disabled one: CausalConv3d replicates the
    # first AND last frame (kernel-1)//2 times each instead of putting kernel-1
    # copies of frame 0 on the left (convolution.py:266-317), which is why the
    # frame count comes out the same either way and why getting it wrong shifts
    # the whole clip without changing a single shape.
    # It shares the causal arm's WEIGHTS, INPUT and NOISE stream deliberately, so
    # the only difference between the two goldens is the padding rule — which lets
    # the C++ side assert the two arms actually diverge.
    noncausal_draws: list[int] = []

    def noncausal_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(noncausal_draws)}", count)
        noncausal_draws.append(count)
        return torch.from_numpy(values.astype(np.float32)).reshape(shape)

    noncausal = ConvVideoDecoder(
        decoder_blocks=VIDEO_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **{**VIDEO_DEC, "causal": False},
    ).eval()
    noncausal_manifest = fill_from_stream(noncausal, prefix="ltx2.videodec.")
    torch.randn = noncausal_randn
    try:
        y_nc = noncausal(latent)
    finally:
        torch.randn = real_randn
    assert not torch.equal(y, y_nc), (
        "causal and non-causal must DIFFER on identical weights, or the arm gates nothing"
    )

    out.write("// --- section 5c: causal=False, upstream's default arm ---\n")
    emit_scalar(out, "kLtx2VideoDecNcOutT", y_nc.shape[2])
    emit_scalar(out, "kLtx2VideoDecNcOutH", y_nc.shape[3])
    emit_scalar(out, "kLtx2VideoDecNcOutW", y_nc.shape[4])
    emit_scalar(out, "kLtx2VideoDecNcNoiseDraws", len(noncausal_draws))
    out.write("inline constexpr int64_t kLtx2VideoDecNcNoiseCounts[] = {\n    "
              + ", ".join(str(c) for c in noncausal_draws) + ",\n};\n\n")
    emit_manifest(out, "kLtx2VideoDecNcParam", noncausal_manifest)
    emit_f32(out, "kLtx2VideoDecNcGolden", y_nc.numpy())

    # --- section 5d: the arm where `norm_eps` actually BINDS ---
    # See VIDEO_EPS_BLOCKS for why this arm exists. No torch.randn patch: with
    # `timestep_conditioning=False` and no `inject_noise` block, the decoder draws
    # nothing, which the C++ side asserts by requiring an EMPTY draw list.
    eps_dec = ConvVideoDecoder(
        decoder_blocks=VIDEO_EPS_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **VIDEO_EPS_DEC,
    ).eval()
    eps_manifest = fill_from_stream(eps_dec, prefix="ltx2.videodeceps.")
    eps_latent = make_input("ltx2.videodeceps.input", VIDEO_EPS_LATENT, VIDEO_EPS_LATENT_SCALE)

    norm3 = [m for m in eps_dec.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert len(norm3) == 1, (
        f"the eps arm must have EXACTLY ONE nn.GroupNorm (norm3), found {len(norm3)}"
    )
    assert norm3[0].eps == 1e-6, "norm3 must carry the constant this arm gates"
    y_eps = eps_dec(eps_latent)

    # PROVEN SENSITIVE, not assumed — the same discipline section 5b applies to its
    # causality probe and section 4b to the mel clamp. eps -> 0 is the mutation the
    # OTHER arms are blindest to (section 5's golden moves 5.4e-7 under it, a tenth
    # of the band), so it is the one this arm has to catch.
    norm3[0].eps = 0.0
    try:
        y_zero = eps_dec(eps_latent)
    finally:
        norm3[0].eps = 1e-6
    zero_move = float((y_eps - y_zero).abs().max())
    assert zero_move > 10 * GOLDEN_TOL, (
        f"the eps arm must move well past the {GOLDEN_TOL:g} band when the "
        f"constant is removed, moved only {zero_move:g}"
    )
    print(f"[eps arm] norm3 in {tuple(y_eps.shape)}; eps 1e-6 -> 0 moves {zero_move:g}",
          file=sys.stderr)

    out.write("// --- section 5d: the arm on which `norm_eps` BINDS (resnet.py:93-97) ---\n")
    emit_scalar(out, "kLtx2VideoDecEpsLatentC", VIDEO_EPS_LATENT[1])
    emit_scalar(out, "kLtx2VideoDecEpsLatentT", VIDEO_EPS_LATENT[2])
    emit_scalar(out, "kLtx2VideoDecEpsLatentH", VIDEO_EPS_LATENT[3])
    emit_scalar(out, "kLtx2VideoDecEpsLatentW", VIDEO_EPS_LATENT[4])
    out.write("inline constexpr double kLtx2VideoDecEpsLatentScale = "
              + _cxx_float(VIDEO_EPS_LATENT_SCALE, 17) + ";\n")
    emit_scalar(out, "kLtx2VideoDecEpsOutC", y_eps.shape[1])
    emit_scalar(out, "kLtx2VideoDecEpsOutT", y_eps.shape[2])
    emit_scalar(out, "kLtx2VideoDecEpsOutH", y_eps.shape[3])
    emit_scalar(out, "kLtx2VideoDecEpsOutW", y_eps.shape[4])
    # How far the golden travels when the constant is REMOVED, measured on the
    # oracle. The C++ arm requires it to clear the band by a wide margin, so the
    # sensitivity this arm exists for is gated rather than narrated.
    out.write("inline constexpr double kLtx2VideoDecEpsZeroMove = "
              + _cxx_float(zero_move, 9) + ";\n\n")
    emit_manifest(out, "kLtx2VideoDecEpsParam", eps_manifest)
    emit_f32(out, "kLtx2VideoDecEpsGolden", y_eps.numpy())


# ---------------------------------------------------------------------------
# Sections 5e/5f — THE BFLOAT16 ARM (A24 wave 3, row LTX25-A24-VIDEO-VAE-BF16,
# issue #2786).
#
# Upstream resolves ONE pipeline dtype and it is bfloat16
# (ltx-pipelines/.../distilled.py:109), handed to `VideoDecoder` at `:148`. Every
# section above runs the module in f32, which is the PARITY arm and stays; this
# one runs the SAME arm with the module and the latent at bfloat16, so the port's
# bf16 branch has an oracle at all. Without it the six rounding rules the port
# implements are ungated, and `ltx2_video_vae.cpp` already records why an f32
# oracle cannot supply one: "a dtype comparison against it is vacuous by
# construction".
#
# TWO HARNESS ADAPTATIONS, both recorded because both change something.
#
#   1. `fill_from_stream` still builds f32 parameters and the module is then
#      `.to(torch.bfloat16)`. That is not the cast the file warns about -- it is
#      exactly `module.to(dtype)` on an f32 state dict, which is what
#      `Ltx2LoadVaeWeights(..., kBF16)` does to a checkpoint. Both sides therefore
#      narrow the SAME f32 values once, and the manifest is unchanged.
#
#   2. The patched `torch.randn` HONOURS the `dtype=` keyword upstream passes
#      (conv_video_decoder.py:288-294, resnet.py:115) by narrowing the shared f32
#      stream, rather than drawing at bf16. `torch.randn(dtype=bfloat16)` at a
#      fixed seed is a DIFFERENT SEQUENCE from `torch.randn(dtype=float32)` -- not
#      the f32 stream rounded -- so mirroring it would change every render digest
#      this repository has captured. That decision is #2780 and is the
#      developer's; this generator mirrors the port's recorded divergence so the
#      two sides compare the same thing.
#
# WHY THE SHIPPED FIXTURE SEPARATES THE _RMSNorm2D ORDERING, which is the one
# rule a badly chosen width would hide. `_RMSNorm2D` is
# `F.normalize(x, dim=1) * (self.scale * self.gamma)` (attention.py:23), and at a
# channel count whose square root is a power of two every ordering of that
# product agrees -- measured 0 of 4800 for all three at C=64. This decoder's
# `attn` block sits at 32 channels, and `sqrt(32)` is not representable in
# bfloat16, so the ordering is a first-order term here. It is stated rather than
# assumed because a later fixture change that moved the attn block to 64 channels
# would silently mute it.
def section_conv_video_decoder_bf16(out) -> None:
    import torch
    import torch.nn as nn

    from ltx_core.model.common.normalization import PixelNorm
    from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType

    decoder = ConvVideoDecoder(
        decoder_blocks=VIDEO_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **VIDEO_DEC,
    ).eval()
    manifest = fill_from_stream(decoder, prefix="ltx2.videodec.")
    decoder = decoder.to(torch.bfloat16)
    latent = make_input("ltx2.videodec.input", VIDEO_LATENT, 1.0).to(torch.bfloat16)

    draws: list[int] = []
    real_randn = torch.randn

    def patched_randn(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(draws)}", count)
        draws.append(count)
        drawn = torch.from_numpy(values.astype(np.float32)).reshape(shape)
        # ADAPTATION 2. The f32 stream, NARROWED -- see this section's header.
        dtype = kwargs.get("dtype")
        return drawn.to(dtype) if dtype is not None else drawn

    # THE ATTENTION BACKEND IS PINNED, AND AT bf16 THAT IS NOT A FORMALITY.
    # `AttnBlock3D` resolves `AttentionFunction.PYTORCH` once in `__init__`
    # (video_vae/attention.py:50, :53) and calls SDPA. Its docstring argues that
    # the single-head `head_dim == in_channels` exceeds FlashAttention's limit so
    # the dispatcher falls back to an efficient or math kernel -- and that is NOT
    # what happens. Measured on this CPU with `sdpa_kernel`, at head_dim 64, 128
    # and 256, the bare call is BIT-IDENTICAL to FLASH (0 of 8192, 0 of 16384, 0
    # of 32768) and 37-38% of words away from MATH.
    #
    # At f32 the question does not arise: FlashAttention serves only fp16 and
    # bf16, so every f32 section above is already getting MATH, which is why the
    # port's own f32 softmax matches them. At bf16 the default arm is a kernel
    # this port does not reproduce, so the golden is taken under MATH and the
    # DISTANCE to the unpinned arm is emitted beside it rather than hidden. The
    # arm upstream actually runs is therefore NOT gated here, and the row's spec
    # says so under `## Risks`.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.randn = patched_randn
    try:
        with sdpa_kernel(SDPBackend.MATH):
            y = decoder(latent)
    finally:
        torch.randn = real_randn

    draws_bare: list[int] = []

    def patched_randn_bare(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(draws_bare)}", count)
        draws_bare.append(count)
        drawn = torch.from_numpy(values.astype(np.float32)).reshape(shape)
        dtype = kwargs.get("dtype")
        return drawn.to(dtype) if dtype is not None else drawn

    torch.randn = patched_randn_bare
    try:
        y_bare = decoder(latent)
    finally:
        torch.randn = real_randn
    backend_gap = float((y.float() - y_bare.float()).abs().max())
    print(f"[bf16 arm] MATH vs the module as constructed: max|diff| = {backend_gap:g}",
          file=sys.stderr)

    assert y.dtype == torch.bfloat16, f"the bf16 arm returned {y.dtype}"

    # THE ARM IS NOT THE f32 ARM, measured rather than asserted. If the two agreed
    # everywhere this whole section would be a second copy of section 5 and would
    # gate nothing new.
    f32_decoder = ConvVideoDecoder(
        decoder_blocks=VIDEO_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **VIDEO_DEC,
    ).eval()
    fill_from_stream(f32_decoder, prefix="ltx2.videodec.")
    draws32: list[int] = []

    def patched_randn32(*args, **kwargs):
        shape = tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)
        count = int(np.prod(shape)) if shape else 1
        values = ltx_rand(f"ltx2.videodec.noise.{len(draws32)}", count)
        draws32.append(count)
        return torch.from_numpy(values.astype(np.float32)).reshape(shape)

    torch.randn = patched_randn32
    try:
        y32 = f32_decoder(make_input("ltx2.videodec.input", VIDEO_LATENT, 1.0))
    finally:
        torch.randn = real_randn
    arm_gap = float((y.float() - y32).abs().max())
    assert arm_gap > 10 * GOLDEN_TOL, (
        f"the bf16 arm must be measurably different from the f32 arm or it gates "
        f"nothing new; max|diff| is only {arm_gap:g}"
    )
    print(f"[bf16 arm] conv video decoder: bf16 vs f32 max|diff| = {arm_gap:g}", file=sys.stderr)

    # ── WHY THIS SECTION EMITS NO VALUE BOUND, MEASURED RATHER THAN ARGUED ──
    #
    # The port cannot be bit-identical to this golden and the reason is upstream's
    # own convolution, not a rule the port gets wrong. `cpu_conv3d`'s contract is
    # an f32 accumulator seeded with the bias and one rounding on store, which is
    # what torch does -- but torch BLOCKS its reduction, and at this fixture's own
    # convolution shapes the two association orders disagree on 3 to 5 outputs of
    # 8192 to 24576, with max|diff| up to 0.015625. This chain then amplifies a
    # single last-bit difference: perturbing ONE `conv_in` weight by ONE bf16 ulp
    # moves the whole output by `kLtx2VideoDecBf16UlpSensitivity`.
    #
    # So a tolerance would have to be wider than that irreducible term -- and the
    # three defect distances below say what such a tolerance would then admit.
    # Each is a REAL rounding rule replaced by the hypothesis the row rejected,
    # measured end to end on this fixture. This is wave 2's finding in a second
    # place: a bound that admits the port admits real defects, so the C++ side
    # applies NO value bound here and the arithmetic is gated bit-exactly, one
    # rule per kernel, in section 5g.
    from ltx_core.model.video_vae import attention as _vae_attn
    from ltx_core.model.video_vae import ops as _vae_ops

    def _rerun(dec):
        local: list[int] = []

        def _patched(*args, **kwargs):
            shape = (tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int)
                     else tuple(args))
            count = int(np.prod(shape)) if shape else 1
            values = ltx_rand(f"ltx2.videodec.noise.{len(local)}", count)
            local.append(count)
            drawn = torch.from_numpy(values.astype(np.float32)).reshape(shape)
            dtype = kwargs.get("dtype")
            return drawn.to(dtype) if dtype is not None else drawn

        torch.randn = _patched
        try:
            with sdpa_kernel(SDPBackend.MATH):
                return dec(latent).float()
        finally:
            torch.randn = real_randn

    def _fresh():
        d = ConvVideoDecoder(
            decoder_blocks=VIDEO_BLOCKS,
            norm_layer=NormLayerType.PIXEL_NORM,
            decoder_spatial_padding_mode=PaddingModeType.REFLECT,
            **VIDEO_DEC,
        ).eval()
        fill_from_stream(d, prefix="ltx2.videodec.")
        return d

    base = y.float()
    # DEFECT A: `un_normalize` keeps its statistics in f32 (ops.py:76-79 applies
    # `.to(x)` to both registered buffers).
    _d = _fresh()
    _s32 = _d.per_channel_statistics.get_buffer("std-of-means").clone()
    _m32 = _d.per_channel_statistics.get_buffer("mean-of-means").clone()
    _orig_un = _vae_ops.PerChannelStatistics.un_normalize

    def _bad_un(self, x):
        return (
            (x.float() * _s32.view(1, -1, 1, 1, 1)).to(x.dtype).float()
            + _m32.view(1, -1, 1, 1, 1)
        ).to(x.dtype)

    _vae_ops.PerChannelStatistics.un_normalize = _bad_un
    try:
        defect_stats = float((base - _rerun(_d.to(torch.bfloat16))).abs().max())
    finally:
        _vae_ops.PerChannelStatistics.un_normalize = _orig_un

    # DEFECT B: `_RMSNorm2D` multiplies by sqrt(C) and by gamma SEPARATELY instead
    # of forming the gain first (attention.py:23).
    _orig_rms = _vae_attn._RMSNorm2D.forward

    def _bad_rms(self, x):
        return (torch.nn.functional.normalize(x, dim=1) * self.scale) * self.gamma

    _vae_attn._RMSNorm2D.forward = _bad_rms
    try:
        defect_rms = float((base - _rerun(_fresh().to(torch.bfloat16))).abs().max())
    finally:
        _vae_attn._RMSNorm2D.forward = _orig_rms

    # DEFECT C: `nn.GroupNorm`'s affine kept in f32. On this fixture it is reached
    # ONLY through `res_x_y`'s norm3, and it turns out not to reach the output at
    # all -- which is exactly why section 5g gates it on the kernel directly.
    _orig_gn = nn.GroupNorm.forward

    def _bad_gn(self, x):
        return torch.nn.functional.group_norm(
            x.float(), self.num_groups, self.weight.float(), self.bias.float(), self.eps
        ).to(x.dtype)

    nn.GroupNorm.forward = _bad_gn
    try:
        defect_gn = float((base - _rerun(_fresh().to(torch.bfloat16))).abs().max())
    finally:
        nn.GroupNorm.forward = _orig_gn

    # The chain's response to ONE last bit, which is the irreducible term.
    _d = _fresh().to(torch.bfloat16)
    with torch.no_grad():
        _flat = dict(_d.named_parameters())["conv_in.conv.weight"].reshape(-1)
        _w = _flat[0].view(torch.int16).item()
        _flat[0] = torch.tensor([_w + 1], dtype=torch.int16).view(torch.bfloat16)[0]
    ulp_sensitivity = float((base - _rerun(_d)).abs().max())

    print(f"[bf16 arm] defects: f32 statistics {defect_stats:g}, _RMSNorm2D order "
          f"{defect_rms:g}, f32 GroupNorm affine {defect_gn:g}; one-ulp sensitivity "
          f"{ulp_sensitivity:g}", file=sys.stderr)
    assert ulp_sensitivity > 0, (
        "the one-ulp sensitivity probe moved nothing, so it cannot support the "
        "claim that this arm has an irreducible term"
    )

    out.write("// --- section 5e: the BF16 arm of ConvVideoDecoder (A24 wave 3, #2786) ---\n")
    emit_scalar(out, "kLtx2VideoDecBf16OutC", y.shape[1])
    emit_scalar(out, "kLtx2VideoDecBf16OutT", y.shape[2])
    emit_scalar(out, "kLtx2VideoDecBf16OutH", y.shape[3])
    emit_scalar(out, "kLtx2VideoDecBf16OutW", y.shape[4])
    emit_scalar(out, "kLtx2VideoDecBf16NoiseDraws", len(draws))
    # How far the bf16 arm sits from the f32 one on the SAME fixture. The C++ side
    # requires the port's bf16 output to be closer to this golden than this
    # distance, so "the port just ran the f32 arm" is a red rather than a pass.
    out.write("inline constexpr double kLtx2VideoDecBf16ArmGap = "
              + _cxx_float(arm_gap, 9) + ";\n")
    # The distance from the golden to the arm upstream actually runs, printed so
    # that "held to MATH" is a stated limit with a number rather than a footnote.
    out.write("inline constexpr double kLtx2VideoDecBf16BackendGap = "
              + _cxx_float(backend_gap, 9) + ";\n")
    # The irreducible term and what a bound wide enough to admit it would let
    # through. The C++ side asserts the RELATION rather than any of the numbers,
    # so this is a measurement that justifies the absence of a bound and not a
    # bound in disguise.
    out.write("inline constexpr double kLtx2VideoDecBf16UlpSensitivity = "
              + _cxx_float(ulp_sensitivity, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecBf16DefectStats = "
              + _cxx_float(defect_stats, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecBf16DefectRmsOrder = "
              + _cxx_float(defect_rms, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecBf16DefectGroupNormAffine = "
              + _cxx_float(defect_gn, 9) + ";\n\n")
    emit_manifest(out, "kLtx2VideoDecBf16Param", manifest)
    emit_f32(out, "kLtx2VideoDecBf16Golden", y.float().numpy())

    # --- section 5f: the PixelNorm epsilon, at the scales where it separates ---
    #
    # The decoder golden above cannot gate this constant's WIDTH and saying so is
    # the point. `PixelNorm.forward` adds `self.eps` to a bf16 `mean_sq`
    # (normalization.py:37-40), so what reaches the arithmetic is
    # `bf16(1e-8) = 1.0011717677116394e-08` and not `1e-8` -- and holding the rest
    # of the chain fixed while varying ONLY that width separates on 0 of 8192
    # values at ordinary magnitude and on 842 of 8192 at a row scale of 2^-14.
    # Removing the epsilon entirely is a DIFFERENT question and separates from
    # 2^-10. A probe at the shipped fixture's scale is a mute switch for both, so
    # this emits three scales and the count each one separates on, and REFUSES to
    # emit an arm that separates nothing.
    # BATCH 1, DELIBERATELY. `PixelNorm` reduces over dim 1, and the port's kernel
    # takes a channel-major [C, spatial] volume at batch 1 -- torch's memory order
    # puts a batch axis OUTSIDE the channel axis, so a batch-2 tensor flattens to
    # something the kernel would read as a different volume entirely. The first
    # form of this probe had batch 2 and reported the port 0.44 away from
    # upstream, which was the layout and not the arithmetic.
    shape = (1, 8, 2, 3, 3)
    scales = [0, 10, 14]
    out.write("// --- section 5f: the PixelNorm epsilon, where its WIDTH binds ---\n")
    out.write("inline constexpr int64_t kLtx2PixelNormEpsScales[] = {\n    "
              + ", ".join(str(-e) for e in scales) + ",\n};\n")
    golden = []
    rejected_f32_eps = []
    rejected_no_eps = []
    sep_width = []
    sep_read = []
    for e in scales:
        x = torch.from_numpy(
            ltx_rand(f"ltx2.pixelnorm.bf16.{e}", int(np.prod(shape)))
            .astype(np.float32)
        ).reshape(shape) * (2.0 ** -e)
        xb = x.to(torch.bfloat16)
        up = PixelNorm(dim=1, eps=1e-8)(xb)
        ms = torch.mean(xb ** 2, dim=1, keepdim=True)
        # REJECTED 1: the epsilon added in f32 and the result rounded back. Same
        # chain, one width changed.
        wide = (ms.float() + 1e-8).to(torch.bfloat16)
        r_f32 = xb / torch.sqrt(wide)
        # REJECTED 2: no epsilon at all -- "is it read", not "at what width".
        r_none = xb / torch.sqrt(ms)
        n = up.numel()
        sw = int((r_f32.reshape(-1) != up.reshape(-1)).sum())
        sr = int((r_none.reshape(-1) != up.reshape(-1)).sum())
        sep_width.append(sw)
        sep_read.append(sr)
        golden.append(up.float().reshape(-1).numpy())
        rejected_f32_eps.append(r_f32.float().reshape(-1).numpy())
        rejected_no_eps.append(r_none.float().reshape(-1).numpy())
        print(f"[bf16 eps] scale 2^-{e}: f32-eps separates {sw}/{n}, "
              f"no-eps separates {sr}/{n}", file=sys.stderr)
    assert max(sep_width) > 0, (
        "the epsilon WIDTH probe separates nothing at any emitted scale, so it "
        "would gate the constant at zero -- rebuild it rather than emitting it"
    )
    assert max(sep_read) > 0, (
        "the epsilon READ probe separates nothing at any emitted scale"
    )
    emit_scalar(out, "kLtx2PixelNormEpsChannels", shape[1])
    emit_scalar(out, "kLtx2PixelNormEpsSpatial", shape[0] * shape[2] * shape[3] * shape[4])
    out.write("inline constexpr int64_t kLtx2PixelNormEpsWidthSeparating[] = {\n    "
              + ", ".join(str(v) for v in sep_width) + ",\n};\n")
    out.write("inline constexpr int64_t kLtx2PixelNormEpsReadSeparating[] = {\n    "
              + ", ".join(str(v) for v in sep_read) + ",\n};\n\n")
    emit_f32(out, "kLtx2PixelNormEpsInput",
             np.concatenate([
                 ltx_rand(f"ltx2.pixelnorm.bf16.{e}", int(np.prod(shape))).astype(np.float32)
                 * (2.0 ** -e)
                 for e in scales
             ]))
    emit_f32(out, "kLtx2PixelNormEpsGolden", np.concatenate(golden))
    emit_f32(out, "kLtx2PixelNormEpsRejectedF32Eps", np.concatenate(rejected_f32_eps))
    emit_f32(out, "kLtx2PixelNormEpsRejectedNoEps", np.concatenate(rejected_no_eps))


# ---------------------------------------------------------------------------
# Section 5g — THE PER-KERNEL BF16 RULES, each with the answer it rejects.
#
# Section 5e holds the whole chain and is the gate that matters; these hold ONE
# rounding rule each, so a red says WHICH rule moved instead of only that the
# clip did. Every arm is BATCH 1 and channel-major, because the port's kernels
# take a [C, spatial] volume and torch puts a batch axis outside the channel axis
# -- the first form of the PixelNorm probe had batch 2 and read 0.44 away from
# upstream, which was the layout and not the arithmetic.
#
# Each arm emits upstream's answer AND the hypothesis it rejects, with the count
# the two differ on, and REFUSES to emit an arm whose alternatives all agree.
def section_video_vae_bf16_kernels(out) -> None:
    import torch
    import torch.nn as nn

    BF = torch.bfloat16

    def stream(name, count, scale=1.0):
        return torch.from_numpy(
            (ltx_rand(name, count) * scale).astype(np.float32)
        )

    def sep(a, b):
        return int((a.reshape(-1) != b.reshape(-1)).sum())

    out.write("// --- section 5g: the per-kernel bf16 rules, with what they reject ---\n")

    # --- GroupNorm: the AFFINE narrows, the statistics do not ------------------
    C, G, T, H, W = 8, 2, 2, 3, 3
    n = C * T * H * W
    gn = nn.GroupNorm(G, C, eps=1e-6)
    with torch.no_grad():
        gn.weight.copy_(stream("ltx2.gnbf16.weight", C) * 0.1 + 1.0)
        gn.bias.copy_(stream("ltx2.gnbf16.bias", C) * 0.1)
    w32 = gn.weight.detach().clone()
    b32 = gn.bias.detach().clone()
    x = stream("ltx2.gnbf16.input", n).reshape(1, C, T, H, W).to(BF)
    up = gn.to(BF)(x)
    gn32 = nn.GroupNorm(G, C, eps=1e-6)
    with torch.no_grad():
        gn32.weight.copy_(w32)
        gn32.bias.copy_(b32)
    rejected = gn32(x.float()).to(BF)          # f32 affine, one store rounding
    gn_sep = sep(rejected, up)
    assert gn_sep > 0, "the GroupNorm affine-width probe separates nothing"
    print(f"[bf16 kernels] group_norm: f32 affine separates {gn_sep}/{up.numel()}",
          file=sys.stderr)
    emit_scalar(out, "kLtx2Bf16GnChannels", C)
    emit_scalar(out, "kLtx2Bf16GnGroups", G)
    emit_scalar(out, "kLtx2Bf16GnSpatial", T * H * W)
    emit_scalar(out, "kLtx2Bf16GnSeparating", gn_sep)
    emit_f32(out, "kLtx2Bf16GnInput", x.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16GnWeight", w32.numpy())
    emit_f32(out, "kLtx2Bf16GnBias", b32.numpy())
    emit_f32(out, "kLtx2Bf16GnGolden", up.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16GnRejectedF32Affine", rejected.float().reshape(-1).numpy())

    # --- AdaLN: THREE roundings ----------------------------------------------
    C, T, H, W = 8, 2, 2, 2
    n = C * T * H * W
    xa = stream("ltx2.adabf16.input", n).reshape(1, C, T, H, W).to(BF)
    table = stream("ltx2.adabf16.table", 4 * C).reshape(4, C).to(BF)
    embed = stream("ltx2.adabf16.embed", 4 * C).reshape(4, C).to(BF)
    shift = (table[0] + embed[0]).reshape(1, C, 1, 1, 1)
    scale = (table[1] + embed[1]).reshape(1, C, 1, 1, 1)
    up_ada = xa * (1 + scale) + shift
    # REJECTED: one fused f32 expression instead of three roundings.
    rej_ada = (xa.float() * (1.0 + scale.float()) + shift.float()).to(BF)
    ada_sep = sep(rej_ada, up_ada)
    assert ada_sep > 0, "the AdaLN rounding probe separates nothing"
    print(f"[bf16 kernels] ada_ln: one-rounding separates {ada_sep}/{up_ada.numel()}",
          file=sys.stderr)
    emit_scalar(out, "kLtx2Bf16AdaChannels", C)
    emit_scalar(out, "kLtx2Bf16AdaSpatial", T * H * W)
    emit_scalar(out, "kLtx2Bf16AdaSeparating", ada_sep)
    emit_f32(out, "kLtx2Bf16AdaInput", xa.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16AdaTable", table.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16AdaEmbed", embed.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16AdaGolden", up_ada.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16AdaRejectedOneRounding", rej_ada.float().reshape(-1).numpy())

    # --- spatial noise: the PRODUCT rounds, then the ADD rounds ---------------
    C, T, H, W = 8, 2, 3, 3
    n = C * T * H * W
    xs = stream("ltx2.noisebf16.input", n).reshape(1, C, T, H, W).to(BF)
    plane = stream("ltx2.noisebf16.plane", H * W).reshape(H, W).to(BF)
    pcs = (stream("ltx2.noisebf16.scale", C) * 0.1).reshape(C, 1, 1).to(BF)
    up_sn = xs + (plane[None] * pcs)[None, :, None, ...]
    rej_sn = (xs.float() + (plane[None].float() * pcs.float())[None, :, None, ...]).to(BF)
    sn_sep = sep(rej_sn, up_sn)
    assert sn_sep > 0, "the spatial-noise rounding probe separates nothing"
    print(f"[bf16 kernels] spatial_noise: fused separates {sn_sep}/{up_sn.numel()}",
          file=sys.stderr)
    emit_scalar(out, "kLtx2Bf16NoiseChannels", C)
    emit_scalar(out, "kLtx2Bf16NoiseT", T)
    emit_scalar(out, "kLtx2Bf16NoiseH", H)
    emit_scalar(out, "kLtx2Bf16NoiseW", W)
    emit_scalar(out, "kLtx2Bf16NoiseSeparating", sn_sep)
    emit_f32(out, "kLtx2Bf16NoiseInput", xs.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16NoisePlane", plane.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16NoiseScale", pcs.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16NoiseGolden", up_sn.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16NoiseRejectedFused", rej_sn.float().reshape(-1).numpy())

    # --- linear_cn: a 1x1x1 nn.Conv3d, bias INSIDE the accumulator ------------
    CIN, COUT, T, H, W = 6, 4, 2, 2, 2
    N = T * H * W
    xl = stream("ltx2.linbf16.input", CIN * N).reshape(1, CIN, T, H, W).to(BF)
    wl = (stream("ltx2.linbf16.weight", COUT * CIN) * 0.3).reshape(COUT, CIN, 1, 1, 1).to(BF)
    bl = (stream("ltx2.linbf16.bias", COUT) * 0.1).to(BF)
    up_lin = torch.nn.functional.conv3d(xl, wl, bl)
    # REJECTED: the bias added AFTER the store rounding.
    rej_lin = (
        torch.nn.functional.conv3d(xl, wl, None).to(BF).float()
        + bl.float().reshape(1, -1, 1, 1, 1)
    ).to(BF)
    lin_sep = sep(rej_lin, up_lin)
    assert lin_sep > 0, "the linear_cn bias-placement probe separates nothing"
    print(f"[bf16 kernels] linear_cn: bias-after separates {lin_sep}/{up_lin.numel()}",
          file=sys.stderr)
    emit_scalar(out, "kLtx2Bf16LinIn", CIN)
    emit_scalar(out, "kLtx2Bf16LinOut", COUT)
    emit_scalar(out, "kLtx2Bf16LinN", N)
    emit_scalar(out, "kLtx2Bf16LinSeparating", lin_sep)
    emit_f32(out, "kLtx2Bf16LinInput", xl.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16LinWeight", wl.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16LinBias", bl.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16LinGolden", up_lin.float().reshape(-1).numpy())
    emit_f32(out, "kLtx2Bf16LinRejectedBiasAfter", rej_lin.float().reshape(-1).numpy())


# ---------------------------------------------------------------------------
# Section 5h — THE SHALLOW BF16 ARM, which is where the rules section 5e cannot
# gate become gateable.
#
# Section 5e's whole-decode arm carries no value bound, because torch's blocked
# convolution reduction gives it an irreducible term and this chain amplifies one
# last bit into 0.0117 at the output. That is a property of DEPTH: the shipped
# fixture runs thirteen convolutions and the residues compound.
#
# This arm runs TWO -- `conv_in` and `conv_out` -- with an `attn` block between
# them and timestep conditioning on. It therefore reaches, and can hold
# BIT-EXACTLY, three rules the kernel table does not own and section 5e cannot
# see: `_RMSNorm2D`'s multiply order, `per_channel_statistics.un_normalize`'s
# narrowing of its two registered buffers, and the PixArt timestep embedding
# computed AT the activation dtype. Each is emitted with the answer it rejects.
#
# `sqrt(8)` IS NOT A POWER OF TWO, and that is why this width was chosen.
# `_RMSNorm2D` is `F.normalize(x, dim=1) * (self.scale * self.gamma)`
# (attention.py:23) and at C=64 every ordering of that product agrees on 4800 of
# 4800 values, so a probe built on a power-of-two channel count would gate
# nothing. The `attn` block here sits at `base_channels * 1 = 8`.
VIDEO_SHALLOW_BLOCKS = [("attn", {"num_layers": 1})]

# `ConvVideoDecoder.__init__` declares `decode_timestep: float = 0.05`
# (conv_video_decoder.py:236) and `forward` builds the default timestep tensor
# from it (`:304-305`); the port's `Ltx2ConvVideoDecoderConfig::decode_timestep`
# carries the same default. Named once here so section 5i's wide-product arm and
# the C++ probe cannot drift from it.
VIDEO_DEC_DECODE_TIMESTEP = 0.05


def section_conv_video_decoder_bf16_shallow(out) -> None:
    import torch

    from ltx_core.model.video_vae import attention as vae_attn
    from ltx_core.model.video_vae import ops as vae_ops
    from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder
    from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
    from torch.nn.attention import SDPBackend, sdpa_kernel

    real_randn = torch.randn

    def build():
        d = ConvVideoDecoder(
            decoder_blocks=VIDEO_SHALLOW_BLOCKS,
            norm_layer=NormLayerType.PIXEL_NORM,
            decoder_spatial_padding_mode=PaddingModeType.REFLECT,
            **VIDEO_DEC,
        ).eval()
        m = fill_from_stream(d, prefix="ltx2.videodecshallow.")
        return d, m

    latent = make_input("ltx2.videodecshallow.input", VIDEO_LATENT, 1.0).to(torch.bfloat16)

    def run(dec, timestep=None):
        local: list[int] = []

        def patched(*args, **kwargs):
            shape = (tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int)
                     else tuple(args))
            count = int(np.prod(shape)) if shape else 1
            values = ltx_rand(f"ltx2.videodecshallow.noise.{len(local)}", count)
            local.append(count)
            drawn = torch.from_numpy(values.astype(np.float32)).reshape(shape)
            dtype = kwargs.get("dtype")
            return drawn.to(dtype) if dtype is not None else drawn

        torch.randn = patched
        try:
            with sdpa_kernel(SDPBackend.MATH):
                return dec(latent, timestep=timestep), local
        finally:
            torch.randn = real_randn

    dec, manifest = build()
    y, draws = run(dec.to(torch.bfloat16))
    assert y.dtype == torch.bfloat16
    base = y.float()

    # REJECTED 1: `_RMSNorm2D` multiplies by sqrt(C) and by gamma separately.
    orig_rms = vae_attn._RMSNorm2D.forward

    def bad_rms(self, x):
        return (torch.nn.functional.normalize(x, dim=1) * self.scale) * self.gamma

    vae_attn._RMSNorm2D.forward = bad_rms
    try:
        d2, _ = build()
        y_rms, _ = run(d2.to(torch.bfloat16))
    finally:
        vae_attn._RMSNorm2D.forward = orig_rms

    # REJECTED 2: `un_normalize` fuses its multiply and its add in f32.
    orig_un = vae_ops.PerChannelStatistics.un_normalize

    def bad_un(self, x):
        s = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)
        m = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)
        return (x.float() * s.float() + m.float()).to(x.dtype)

    vae_ops.PerChannelStatistics.un_normalize = bad_un
    try:
        d3, _ = build()
        y_un, _ = run(d3.to(torch.bfloat16))
    finally:
        vae_ops.PerChannelStatistics.un_normalize = orig_un

    # REJECTED 3: the timestep embedding computed in f32 and rounded once.
    from ltx_core.model.transformer import timestep_embedding as te

    orig_te = te.TimestepEmbedding.forward

    def bad_te(self, sample, condition=None):  # noqa: ARG001
        was = sample.dtype
        h = self.linear_1(sample.float().to(self.linear_1.weight.dtype).float())
        return h.to(was)

    # A faithful f32 re-run of the module rather than a re-derivation: run the
    # WHOLE embedder in f32 and narrow once.
    def f32_te(self, sample, condition=None):  # noqa: ARG001
        was = sample.dtype
        w1 = self.linear_1.weight.float()
        b1 = self.linear_1.bias.float()
        w2 = self.linear_2.weight.float()
        b2 = self.linear_2.bias.float()
        h = torch.nn.functional.linear(sample.float(), w1, b1)
        h = torch.nn.functional.silu(h)
        return torch.nn.functional.linear(h, w2, b2).to(was)

    te.TimestepEmbedding.forward = f32_te
    try:
        d4, _ = build()
        y_te, _ = run(d4.to(torch.bfloat16))
    finally:
        te.TimestepEmbedding.forward = orig_te

    sep_rms = float((base - y_rms.float()).abs().max())
    sep_un = float((base - y_un.float()).abs().max())
    sep_te = float((base - y_te.float()).abs().max())
    print(f"[bf16 shallow] rejected distances: _RMSNorm2D order {sep_rms:g}, "
          f"un_normalize fused {sep_un:g}, f32 timestep embedding {sep_te:g}",
          file=sys.stderr)
    assert sep_rms > 0 and sep_un > 0 and sep_te > 0, (
        "a shallow-arm rejected hypothesis separates NOTHING, so this section "
        "would gate the rule it names at zero -- rebuild it rather than emitting it"
    )

    out.write("// --- section 5h: the SHALLOW bf16 arm (two convolutions, attn, timestep) ---\n")
    emit_scalar(out, "kLtx2VideoDecShallowOutC", y.shape[1])
    emit_scalar(out, "kLtx2VideoDecShallowOutT", y.shape[2])
    emit_scalar(out, "kLtx2VideoDecShallowOutH", y.shape[3])
    emit_scalar(out, "kLtx2VideoDecShallowOutW", y.shape[4])
    emit_scalar(out, "kLtx2VideoDecShallowNoiseDraws", len(draws))
    out.write("inline constexpr double kLtx2VideoDecShallowRejectRmsOrder = "
              + _cxx_float(sep_rms, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecShallowRejectUnNormalize = "
              + _cxx_float(sep_un, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecShallowRejectTimestepF32 = "
              + _cxx_float(sep_te, 9) + ";\n\n")
    emit_manifest(out, "kLtx2VideoDecShallowParam", manifest)
    emit_f32(out, "kLtx2VideoDecShallowGolden", base.numpy())
    # The rejected ANSWERS themselves, not only their distances. A golden that
    # carries upstream's answer alone proves agreement; carrying the alternatives
    # lets the C++ side say WHICH hypothesis a failing port landed on, which is
    # the difference between "something is wrong" and a diagnosis.
    emit_f32(out, "kLtx2VideoDecShallowRejectedRmsOrderGolden", y_rms.float().numpy())
    emit_f32(out, "kLtx2VideoDecShallowRejectedUnNormalizeGolden", y_un.float().numpy())
    emit_f32(out, "kLtx2VideoDecShallowRejectedTimestepF32Golden", y_te.float().numpy())

    # --- section 5i: the SCALED TIMESTEP's own rule, at a multiplier that can
    # SEPARATE it (A24 wave 3 fresh review, #2786) ---
    #
    # `scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)`
    # (conv_video_decoder.py:313) narrows BOTH operands and rounds the product,
    # because `timestep` is built at `sample.dtype` (`:304-305`) and `.to(sample)`
    # narrows the parameter. The port mirrors that. Section 5h above CANNOT gate
    # it, and the reason is arithmetic rather than an oversight:
    #
    #   * the fixture's multiplier is drawn from the shared stream and is
    #     0.0675802556968955, so `scaled_timestep` is ~3.4e-3;
    #   * the whole difference between the rule and the WIDE product is the
    #     narrowing of `0.05` itself, a RELATIVE 1.5e-7;
    #   * `get_timestep_embedding` widens the value back to f32 and takes
    #     sin/cos of `t * exp(...)`, whose largest argument is that same 3.4e-3,
    #     and only THEN narrows to bf16 (`timesteps_proj.to(hidden_dtype)`,
    #     timestep_embedding.py:142). A 3e-6 move in a 3.4e-3 angle moves every
    #     sin/cos by less than a quarter of a bf16 ulp, so all 256 projection
    #     entries round to the same bf16 word and the arm is bit-identical.
    #
    # Measured, not asserted: reverting the port to its pre-row f64 product left
    # section 5h fully green.
    #
    # THE MULTIPLIER WAS SWEPT, and the obvious candidates DO NOT WORK. The value
    # this row's spec §4.12 names as separating -- 7.3 -- separates the two
    # SCALARS (0.365234375 against 0.365625) and then separates NOTHING of the
    # arm's 144 outputs, because a 0.1% move in a 0.365-radian angle is still
    # under half a bf16 ulp everywhere the sinusoid lands. Upstream re-run at
    # each candidate, max|rule - wide product| over the whole output:
    #
    #     3.7 -> 0        7.3 -> 0         23.7 -> 0        41.3 -> 0
    #     499.7 -> 0      1000 -> 0        (the SHIPPED value separates nothing)
    #     251.3 -> 0.00195312 (16/144)     17.3 -> 0.00390625 (64/144)
    #     57.9 -> 0.00390625 (53/144)      733.1 -> 0.00390625 (100/144)
    #     911.3 -> 0.0078125 (98/144)      113.7 -> 0.0117188 (119/144)
    #
    # 113.7 is taken because it separates the most of them and because it is
    # still small enough to be SAFE: the port builds its frequency table in f64
    # where upstream builds it in f32 (an annotated exception, see
    # `TimestepEmbedding`), and at 500 that difference alone already flips one of
    # the 256 projection entries after narrowing. At 113.7 it flips none, so this
    # arm measures the product's rounding and not the table's width.
    #
    # NOTHING ELSE ABOUT THE ARM CHANGES. Same module, same stream, same weights,
    # same manifest as 5h; only the `timestep_scale_multiplier` PARAMETER is
    # overwritten, on both sides.
    TSSCALE_MULTIPLIER = 113.7
    d5, _ = build()
    with torch.no_grad():
        d5.timestep_scale_multiplier.fill_(TSSCALE_MULTIPLIER)
    y_ts, _ = run(d5.to(torch.bfloat16))
    assert y_ts.dtype == torch.bfloat16

    # THE REJECTED ANSWER IS UPSTREAM RE-RUN, not a re-derivation. Handing
    # `forward` a WIDE `timestep` tensor is exactly the pre-row hypothesis and
    # nothing else: `timestep * multiplier.to(sample)` then promotes to f64 and
    # the product is never rounded, while every other tensor in the module stays
    # bf16. `get_timestep_embedding` widens whatever it is given to f32 anyway
    # (timestep_embedding.py:91), so this arm changes the VALUE of the scaled
    # timestep and no dtype downstream of it.
    d6, _ = build()
    with torch.no_grad():
        d6.timestep_scale_multiplier.fill_(TSSCALE_MULTIPLIER)
    wide = torch.full((VIDEO_LATENT[0],), VIDEO_DEC_DECODE_TIMESTEP, dtype=torch.float64)
    y_ts_rej, _ = run(d6.to(torch.bfloat16), timestep=wide)

    sep_ts = float((y_ts.float() - y_ts_rej.float()).abs().max())
    # The two scalars themselves, so a reader can see WHY it separates here and
    # not at the stream's own multiplier.
    _mb = float(torch.tensor([TSSCALE_MULTIPLIER]).to(torch.bfloat16)[0])
    _tb = float(torch.tensor([VIDEO_DEC_DECODE_TIMESTEP]).to(torch.bfloat16)[0])
    ts_rule = float(torch.tensor([_tb * _mb]).to(torch.bfloat16)[0])
    ts_wide = VIDEO_DEC_DECODE_TIMESTEP * _mb
    print(f"[bf16 shallow] scaled timestep: rule {ts_rule!r} vs wide product "
          f"{ts_wide!r}; whole-arm separation {sep_ts:g}", file=sys.stderr)
    assert sep_ts > 0, (
        "the wide-product hypothesis separates NOTHING even at multiplier "
        f"{TSSCALE_MULTIPLIER}, so this section would gate the narrowing rule at "
        "zero exactly as the stream's own multiplier does -- find a separating "
        "value or record the rule as ungated, do not emit this"
    )

    out.write("// --- section 5i: the scaled-timestep rule at a SEPARATING multiplier ---\n")
    out.write("inline constexpr double kLtx2VideoDecTsScaleMultiplier = "
              + _cxx_float(TSSCALE_MULTIPLIER, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecTsScaleTimestep = "
              + _cxx_float(VIDEO_DEC_DECODE_TIMESTEP, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecTsScaleRuleValue = "
              + _cxx_float(ts_rule, 17) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecTsScaleWideValue = "
              + _cxx_float(ts_wide, 17) + ";\n")
    out.write("inline constexpr double kLtx2VideoDecTsScaleRejectWideProduct = "
              + _cxx_float(sep_ts, 9) + ";\n\n")
    emit_f32(out, "kLtx2VideoDecTsScaleGolden", y_ts.float().numpy())
    emit_f32(out, "kLtx2VideoDecTsScaleRejectedWideProductGolden", y_ts_rej.float().numpy())


# ---------------------------------------------------------------------------
# Sections 6-8 — the ENCODER halves (phase L11), which L4 recorded as owed.
# ---------------------------------------------------------------------------

# Section 6a — VideoEncoder, the `*_res` family. Covers SpaceToDepthDownsample on
# BOTH a spatial-only and a temporal stride, `res_x` (UNetMidBlock3D),
# `res_x_y` (channel growth, so norm3 + conv_shortcut are live) and `attn`.
# patch_size=2 keeps the patchify channel order load-bearing.
VIDEO_ENC_A_BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_space_res", {"multiplier": 2}),
    ("res_x_y", {"num_layers": 1, "multiplier": 2}),
    ("attn", {"num_layers": 1}),
    ("compress_time_res", {"multiplier": 1}),
]
VIDEO_ENC_A = dict(
    convolution_dimensions=3,
    in_channels=3,
    out_channels=4,
    patch_size=2,
)
# 5 frames satisfies 1 + 2k for this block list's single temporal step.
VIDEO_ENC_A_INPUT = (1, 3, 5, 8, 8)

# Section 6b — VideoEncoder, the PLAIN STRIDED CONVOLUTIONS (`compress_all`,
# `compress_space`, `compress_time`, `compress_all_x_y`), the `per_channel`
# log-variance mode, and `reflect` spatial padding — which is NOT the encoder's
# default and is what a flat LTX-2 checkpoint's `spatial_padding_mode` key
# actually selects for both halves.
VIDEO_ENC_B_BLOCKS = [
    ("compress_all", {}),
    ("compress_space", {}),
    ("compress_time", {}),
    ("compress_all_x_y", {"multiplier": 2}),
]
VIDEO_ENC_B = dict(
    convolution_dimensions=3,
    in_channels=3,
    out_channels=4,
    patch_size=1,
)
# 3 temporal steps -> 1 + 8k.
VIDEO_ENC_B_INPUT = (1, 3, 9, 16, 16)

# Section 6d — the convolution-only causality probe. No `res_x_y`, so no
# GroupNorm-with-one-group whose statistics span time, and no `attn`; PixelNorm is
# per-location. What is left is decided by the causal padding alone.
VIDEO_ENC_CAUSAL_BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_time_res", {"multiplier": 1}),
]

# Section 7 — AudioEncoder. attn_resolutions={8} puts attention on the DEEPEST
# level (curr_res = 32 -> 16 -> 8), so both the per-level list and the mid block
# are exercised. z_channels * (mel_bins // 4) must equal `ch`, because the
# patchifier flattens (c, f) into the axis the per-channel statistics index.
AUDIO_ENC = dict(
    ch=8,
    ch_mult=(1, 2, 4),
    num_res_blocks=1,
    attn_resolutions={8},
    resolution=32,
    z_channels=4,
    double_z=True,
    in_channels=2,
    dropout=0.0,
    resamp_with_conv=True,
    mid_block_add_attention=True,
    sample_rate=16000,
    mel_hop_length=160,
    n_fft=1024,
    is_causal=True,
    mel_bins=8,
)
AUDIO_ENC_FRAMES = 8
AUDIO_ENC_MEL = 8
# Section 7d — the encoder's GROUP-NORM arm. Same two forced dimension changes as
# AUDIO_GROUP_DEC, for the same two reasons: `build_normalization_layer`'s own
# `num_groups` default is 32 and nothing overrides it (normalization.py:44, 56),
# and `z_channels * mel_bins(latent)` must equal `ch`
# because `PerChannelStatistics` indexes the patchified `(c, f)` axis. The encoder
# reaches its deepest level at 8 -> 4 -> 2 mel bins, so 16 * 2 = 32.
AUDIO_GROUP_ENC = {**AUDIO_ENC, "ch": 32, "z_channels": 16}

# Section 8 — the mel front-end. n_fft is small so the direct DFT the C++ side
# uses stays cheap, but every parameter that decides a value is preserved.
MEL = dict(target_sample_rate=16000, mel_bins=8, mel_hop_length=16, n_fft=64)
MEL_SAMPLES = 200

# Section 8d — `AudioProcessor.resample_audio` (ops.py:36-42). Four rate pairs,
# each chosen for an ARM rather than for coverage arithmetic:
#
#   (Up)    16000 -> 48000, gcd 16000 => o = 1. The degenerate ratio the tree
#           already ports for the vocoder's BWE stage, so a port that only
#           handles this one still passes here and fails the next two.
#   (Down)  48000 -> 16000, o = 3, n = 1. `width` is 19 and the kernel has ONE
#           phase row, which is the transpose of the arm above.
#   (Wide)  44100 -> 16000, gcd 100 => o = 441, n = 160, `width` 17. The widest
#           kernel and the only pair here whose `t == 0` tap is not at an obvious
#           index. This is also the pair a real 44.1 kHz take hits.
#   (Same)  16000 -> 16000. `resample` returns the INPUT (functional.py:1473),
#           and `resample_audio` returns before it is even called (ops.py:38-39).
#           A port that filtered anyway would be wrong by the filter's own
#           passband ripple — small enough to pass a loose tolerance.
RESAMPLE_CASES = (
    ("Up", 16000, 48000, 64),
    ("Down", 48000, 16000, 192),
    ("Wide", 44100, 16000, 600),
    ("Same", 16000, 16000, 64),
)
RESAMPLE_CHANNELS = 2

# Section 8e — `waveform_to_mel` at a rate the processor does NOT target, which
# is the production shape: `resample_audio` runs first and the mel transform then
# sees the RESAMPLED length (ops.py:44-49). 600 samples at 44100 resample to 218
# at 16000, comfortably past the `n_fft // 2` reflect-pad floor.
MEL_SOURCE_RATE = 44100
MEL_SOURCE_SAMPLES = 600

# Section 8f — the TRUNCATION BOUNDARY, which no arm above can reach.
#
# `_apply_sinc_resample_kernel` ends on TWO lines, not one (functional.py:1427-1428):
#
#     target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
#     resampled = resampled[..., :target_length]
#
# and both of them decide the output length.
#
# The FIRST line is not an exact integer ceil. `torch.as_tensor` of a PYTHON
# FLOAT takes `torch.get_default_dtype()`, which is float32, so the f64 quotient
# is narrowed to f32 BEFORE the ceil, and the narrowing moves in BOTH directions:
#
#   * it rounds DOWN onto an integer the exact quotient sits just above, and
#     `target_length` comes out one BELOW the exact integer ceil;
#   * it rounds UP past that integer, and `target_length` comes out one ABOVE.
#
# The SECOND line is a Python slice, so it CLAMPS. `resampled` carries exactly
# `(length // o + 1) * n` columns, and the slack `n - ceil(n * (length % o) / o)`
# between that and the exact ceil has minimum `n // o` over the residues — ZERO
# for every downsampling ratio. So on any ratio with `n < o` there are lengths
# where the columns are exactly the exact ceil, and an upward narrowing then asks
# for one column more than the convolution produced. Upstream returns what it
# has; a port that trusts `target_length` alone emits a trailing sample upstream
# never computed.
#
# The four arms of 8d top out at 218 output samples, three orders of magnitude
# below where any of this happens, so they cannot see it: at 44100 -> 16000 the
# first downward-divergent length is 180697 (4.097 s) and 48102 of the first 60 s
# worth of lengths diverge. One output sample either way moves the last STFT
# windows and, where `samples % hop == 0`, the mel FRAME COUNT — that is, the
# conditioning shape.
#
# Four ratios, because no single one shows all three behaviours. The downward
# narrowing needs `orig_freq` large enough for the quotient to land within half
# an ulp below an integer; the upward one needs the quotient past 2^24, where an
# f32 ulp is at least 2. `CeilBelow` and `CeilAbove` bracket 180697 and are
# lengths where an exact ceil is RIGHT, so an arm that always subtracted one
# would fail them; `CeilOver` and `CeilClamp` are lengths where an exact ceil is
# one too SMALL, so an arm that always clamped to it would fail them too.
#
# Each arm declares `ceil_delta = target_length - exact_ceil` and whether the
# slice clamps, and the generator asserts both against upstream's own numbers.
RESAMPLE_CEIL_CASES = (
    # tag, orig_freq, new_freq, length, ceil_delta, clamps
    ("CeilBelow", 44100, 16000, 180696, 0, False),
    ("CeilAt", 44100, 16000, 180697, -1, False),
    ("CeilAbove", 44100, 16000, 180698, 0, False),
    ("CeilAlt", 22050, 16000, 90569, -1, False),
    # The narrowing rounds UP, and the columns happen to accommodate it: upstream
    # returns exact_ceil + 1 and the slice takes everything. 12.7 min of audio.
    ("CeilOver", 44100, 22050, 33554438, +1, False),
    # The narrowing rounds UP past the last column: `target_length` is 33554436
    # and the convolution produced 33554435, so the SLICE decides the answer.
    # 48000 -> 16000 is 8d's `Down` ratio at a length 8d cannot reach — 2097.2 s,
    # ~34.95 min. This is the only arm where the two lines disagree.
    ("CeilClamp", 48000, 16000, 100663303, +1, True),
)
# The goldens carry the LENGTH and the last few samples rather than 65 559 floats
# per arm. The length is the discriminator; the tail is there so a port that
# produced the right count from a shifted signal cannot pass on the count alone.
RESAMPLE_CEIL_TAIL = 8


def section_video_encoder(out) -> None:
    import torch

    from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
    from ltx_core.model.video_vae.video_vae import VideoEncoder

    # --- 6a: the *_res family, `uniform` log variance, `zeros` padding ---
    enc = VideoEncoder(
        encoder_blocks=VIDEO_ENC_A_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode=PaddingModeType.ZEROS,
        **VIDEO_ENC_A,
    ).eval()
    manifest = fill_from_stream(enc, prefix="ltx2.videoenc.")
    frames = make_input("ltx2.videoenc.input", VIDEO_ENC_A_INPUT, 1.0)
    y = enc(frames)

    out.write("// --- section 6a: VideoEncoder, the *_res family (video_vae.py:148-336) ---\n")
    emit_scalar(out, "kLtx2VideoEncInC", VIDEO_ENC_A_INPUT[1])
    emit_scalar(out, "kLtx2VideoEncInT", VIDEO_ENC_A_INPUT[2])
    emit_scalar(out, "kLtx2VideoEncInH", VIDEO_ENC_A_INPUT[3])
    emit_scalar(out, "kLtx2VideoEncInW", VIDEO_ENC_A_INPUT[4])
    emit_scalar(out, "kLtx2VideoEncOutC", y.shape[1])
    emit_scalar(out, "kLtx2VideoEncOutT", y.shape[2])
    emit_scalar(out, "kLtx2VideoEncOutH", y.shape[3])
    emit_scalar(out, "kLtx2VideoEncOutW", y.shape[4])
    emit_scalar(out, "kLtx2VideoEncTemporalFactor", enc.video_scale_factors.time)
    emit_scalar(out, "kLtx2VideoEncSpatialFactor", enc.video_scale_factors.height)
    out.write("\n")
    emit_manifest(out, "kLtx2VideoEncParam", manifest)
    emit_f32(out, "kLtx2VideoEncGolden", y.numpy())

    # --- 6a-constant: `constant` produces the IDENTICAL means, and that is a
    # FINDING, not an omission. Both modes size conv_out at out_channels + 1 and
    # both keep `sample[:, :-1]` as the means (video_vae.py:315 vs :327), so with
    # the same deterministic weights the returned latent is bit-identical. A
    # separate golden would therefore gate nothing; the EQUALITY is what carries
    # information, so it is asserted here and re-asserted in C++.
    enc_const = VideoEncoder(
        encoder_blocks=VIDEO_ENC_A_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.CONSTANT,
        encoder_spatial_padding_mode=PaddingModeType.ZEROS,
        **VIDEO_ENC_A,
    ).eval()
    const_manifest = fill_from_stream(enc_const, prefix="ltx2.videoenc.")
    assert const_manifest == manifest, (
        "the `constant` arm must build the SAME parameter set as `uniform`, or the equality "
        "below compares two different models"
    )
    y_const = enc_const(frames)
    assert torch.equal(y, y_const), (
        "uniform and constant must produce identical MEANS; if upstream ever changes that, this "
        "assertion is the thing that notices"
    )

    # --- 6a-none: upstream RAISES, and the port refuses by name instead of
    # inventing semantics. Measured rather than assumed.
    raised = ""
    try:
        VideoEncoder(
            encoder_blocks=VIDEO_ENC_A_BLOCKS,
            norm_layer=NormLayerType.PIXEL_NORM,
            latent_log_var=LogVarianceType.NONE,
            encoder_spatial_padding_mode=PaddingModeType.ZEROS,
            **VIDEO_ENC_A,
        ).eval()(frames)
    except Exception as exc:  # noqa: BLE001 - the exception IS the observation
        raised = type(exc).__name__
    assert raised, (
        "latent_log_var=none must FAIL upstream; if it started working, the C++ refusal is now "
        "wrong and this assertion is what says so"
    )
    out.write("// --- section 6a-none: latent_log_var=`none` raises upstream ---\n")
    out.write(f'inline constexpr const char* kLtx2VideoEncNoneRaises = "{raised}";\n\n')

    # --- 6b: the plain strided convolutions, `per_channel`, `reflect` padding ---
    enc_b = VideoEncoder(
        encoder_blocks=VIDEO_ENC_B_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.PER_CHANNEL,
        encoder_spatial_padding_mode=PaddingModeType.REFLECT,
        **VIDEO_ENC_B,
    ).eval()
    manifest_b = fill_from_stream(enc_b, prefix="ltx2.videoencb.")
    frames_b = make_input("ltx2.videoencb.input", VIDEO_ENC_B_INPUT, 1.0)
    y_b = enc_b(frames_b)

    out.write("// --- section 6b: VideoEncoder, strided convs + per_channel + reflect ---\n")
    emit_scalar(out, "kLtx2VideoEncBInC", VIDEO_ENC_B_INPUT[1])
    emit_scalar(out, "kLtx2VideoEncBInT", VIDEO_ENC_B_INPUT[2])
    emit_scalar(out, "kLtx2VideoEncBInH", VIDEO_ENC_B_INPUT[3])
    emit_scalar(out, "kLtx2VideoEncBInW", VIDEO_ENC_B_INPUT[4])
    emit_scalar(out, "kLtx2VideoEncBOutC", y_b.shape[1])
    emit_scalar(out, "kLtx2VideoEncBOutT", y_b.shape[2])
    emit_scalar(out, "kLtx2VideoEncBOutH", y_b.shape[3])
    emit_scalar(out, "kLtx2VideoEncBOutW", y_b.shape[4])
    emit_scalar(out, "kLtx2VideoEncBTemporalFactor", enc_b.video_scale_factors.time)
    emit_scalar(out, "kLtx2VideoEncBSpatialFactor", enc_b.video_scale_factors.height)
    out.write("\n")
    emit_manifest(out, "kLtx2VideoEncBParam", manifest_b)
    emit_f32(out, "kLtx2VideoEncBGolden", y_b.numpy())

    # --- 6c: the FRAME-COUNT CROP (video_vae.py:276-286). One extra frame, whose
    # first VIDEO_ENC_A_INPUT[2] frames are the 6a input, must give the 6a answer.
    # An implementation that skips the crop cannot even run: the temporal fold
    # needs an even frame count after the first-frame duplication.
    crop_input = torch.cat([frames, make_input("ltx2.videoenc.tail", (1, 3, 1, 8, 8), 1.0)], dim=2)
    enc_crop = VideoEncoder(
        encoder_blocks=VIDEO_ENC_A_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode=PaddingModeType.ZEROS,
        **VIDEO_ENC_A,
    ).eval()
    fill_from_stream(enc_crop, prefix="ltx2.videoenc.")
    y_crop = enc_crop(crop_input)
    assert torch.equal(y, y_crop), "the crop arm must reproduce the 6a golden exactly"
    out.write("// --- section 6c: the frame-count crop ---\n")
    emit_scalar(out, "kLtx2VideoEncCropInT", crop_input.shape[2])
    emit_scalar(out, "kLtx2VideoEncCropDropped", crop_input.shape[2] - VIDEO_ENC_A_INPUT[2])
    out.write("\n")

    # --- 6d: what the encoder's causality actually reaches, MEASURED ---
    enc_causal = VideoEncoder(
        encoder_blocks=VIDEO_ENC_CAUSAL_BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        latent_log_var=LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode=PaddingModeType.ZEROS,
        **VIDEO_ENC_A,
    ).eval()
    causal_manifest = fill_from_stream(enc_causal, prefix="ltx2.videoenccausal.")
    base = enc_causal(frames)
    bumped_in = frames.clone()
    bumped_in[:, :, -1] += 5.0
    bumped = enc_causal(bumped_in)
    moved = [t for t in range(base.shape[2]) if not torch.equal(base[:, :, t], bumped[:, :, t])]
    assert moved, "the causality probe must move SOMETHING or it gates nothing"
    assert len(moved) < base.shape[2], (
        "the causality probe must leave at least one EARLY latent frame untouched, or it is not "
        "testing causality at all"
    )
    out.write("// --- section 6d: convolution-only causal reach ---\n")
    emit_scalar(out, "kLtx2VideoEncCausalOutT", base.shape[2])
    emit_scalar(out, "kLtx2VideoEncCausalFirstMoved", moved[0])
    emit_scalar(out, "kLtx2VideoEncCausalLastMoved", moved[-1])
    out.write("\n")
    emit_manifest(out, "kLtx2VideoEncCausalParam", causal_manifest)
    emit_f32(out, "kLtx2VideoEncCausalGolden", base.numpy())


# Section 6-bf16 — VideoEncoder at the arm upstream actually runs (A24 wave 4,
# row LTX25-A24-LEAVES-BF16, issue #2850).
#
# THE BLOCK LIST IS CHOSEN SO ONE RULE IS NOT MUTED. `SpaceToDepthDownsample`'s
# skip is a group mean over `group_size = in_channels * prod(stride) //
# out_channels` (sampling.py:23, 50-51), and a TWO-element mean is exact in any
# order, so a fixture that only reaches `group_size == 2` gates that rule at zero.
# Section 6a's blocks both land on 2. `compress_all_res` with multiplier 2 gives
# `8 * 8 // 16 = 4`, which is where the rule is first-order.
#
# NO `attn` BLOCK, DELIBERATELY. At bf16 the VAE attention is served by FLASH and
# is 37-38% of words from MATH (section 5e's header), so an attention block here
# would drag that whole unresolved question into a case about three unrelated
# rules. `AttnBlock3d` is SHARED with the decoder and is already gated by
# section 5e; nothing about it is encoder-specific.
VIDEO_ENC_BF16_BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_all_res", {"multiplier": 2}),
]
VIDEO_ENC_BF16 = dict(
    convolution_dimensions=3,
    in_channels=3,
    out_channels=8,
    patch_size=2,
)
# 3 frames satisfies 1 + 2k for this block list's single temporal step.
VIDEO_ENC_BF16_INPUT = (1, 3, 3, 8, 8)


def section_video_encoder_bf16(out) -> None:
    import torch

    from ltx_core.model.video_vae import sampling as _vae_sampling
    from ltx_core.model.video_vae import ops as _vae_ops
    from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
    from ltx_core.model.video_vae.video_vae import VideoEncoder

    def _fresh():
        enc = VideoEncoder(
            encoder_blocks=VIDEO_ENC_BF16_BLOCKS,
            norm_layer=NormLayerType.PIXEL_NORM,
            latent_log_var=LogVarianceType.UNIFORM,
            encoder_spatial_padding_mode=PaddingModeType.ZEROS,
            **VIDEO_ENC_BF16,
        ).eval()
        fill_from_stream(enc, prefix="ltx2.videoencbf16.")
        return enc

    enc_f32 = _fresh()
    manifest = fill_from_stream(enc_f32, prefix="ltx2.videoencbf16.")
    frames_f32 = make_input("ltx2.videoencbf16.input", VIDEO_ENC_BF16_INPUT, 1.0)

    # THE GROUP SIZE IS ASSERTED, NOT ASSUMED. It is the whole reason this block
    # list exists, and a later edit that changed a multiplier would otherwise mute
    # section 4.1's rule in silence.
    group_sizes = [
        m.group_size for m in enc_f32.modules() if isinstance(m, _vae_sampling.SpaceToDepthDownsample)
    ]
    assert group_sizes and max(group_sizes) >= 4, (
        f"this fixture must reach a group_size of at least 4 or the group-mean rule is "
        f"gated at zero; it reaches {group_sizes}"
    )

    y_f32 = enc_f32(frames_f32)

    enc = _fresh().to(torch.bfloat16)
    frames = frames_f32.to(torch.bfloat16)
    y = enc(frames)
    assert y.dtype == torch.bfloat16, f"the bf16 arm returned {y.dtype}"

    # HOW FAR THE bf16 ARM SITS FROM THE f32 ONE on the same fixture. The C++ side
    # requires the port's bf16 output to be CLOSER to this golden than this
    # distance, so "the port just ran the f32 arm" is a red rather than a pass.
    arm_gap = float((y.float() - y_f32).abs().max())
    assert arm_gap > 0, (
        "the bf16 arm is bit-identical to the f32 arm on this fixture, so the golden "
        "gates nothing"
    )

    # ── DEFECT A: the per-channel statistics kept f32 through `normalize`.
    # This is what the port did before this row. ops.py:81-84 applies `.to(x)` to
    # both registered buffers.
    #
    # THE f32 BUFFERS ARE CAPTURED BEFORE ANY CAST, and the first form of this
    # probe was not. `.to(torch.bfloat16)` narrows a registered buffer IN PLACE,
    # so a `.float()` taken after it returns the ALREADY-NARROWED value, both
    # hypotheses become the same tensor and the probe reports a defect of exactly
    # zero. This generator's own assert caught it. It is the same trap wave 3's
    # implementer and BOTH its reviewers hit, recorded there under section 4.4.
    _mean_f32 = enc_f32.per_channel_statistics.get_buffer("mean-of-means").clone().float()
    _std_f32 = enc_f32.per_channel_statistics.get_buffer("std-of-means").clone().float()
    assert _mean_f32.dtype == torch.float32 and _std_f32.dtype == torch.float32
    # ...and the capture is only worth anything if the values are OFF the bf16
    # grid. `param_values` draws them from the shared stream, so they are, but a
    # later change to that rule could put them on it and mute this defect in
    # silence.
    _off_grid = int(((_mean_f32.bfloat16().float() != _mean_f32).sum()
                     + (_std_f32.bfloat16().float() != _std_f32).sum()))
    assert _off_grid > 0, (
        "every per-channel statistic on this fixture is exactly representable in "
        "bfloat16, so narrowing them is a no-op and this defect cannot separate"
    )
    _orig_norm = _vae_ops.PerChannelStatistics.normalize

    def _bad_norm(self, x):
        mean = _mean_f32.view(1, -1, 1, 1, 1)
        std = _std_f32.view(1, -1, 1, 1, 1)
        return ((x.float() - mean).to(x.dtype).float() / std).to(x.dtype)

    _vae_ops.PerChannelStatistics.normalize = _bad_norm
    try:
        defect_stats = float((y.float() - _fresh().to(torch.bfloat16)(frames).float()).abs().max())
    finally:
        _vae_ops.PerChannelStatistics.normalize = _orig_norm

    # ── DEFECT B: the group mean carried UNROUNDED across the skip add.
    # The add's own width separates nothing at any scale; the rounding point is
    # the STORE. This monkeypatch reproduces upstream's forward exactly except for
    # that one point, so what it measures is the rounding and not the arithmetic.
    _orig_s2d = _vae_sampling.SpaceToDepthDownsample.forward

    def _bad_s2d(self, x, causal: bool = True):
        from einops import rearrange

        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)
        x_in = rearrange(
            x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.float().mean(dim=2)  # <-- the defect: no rounding here
        x = self.conv(x, causal=causal)
        x = rearrange(
            x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
        )
        return (x.float() + x_in).to(x.dtype)

    _vae_sampling.SpaceToDepthDownsample.forward = _bad_s2d
    try:
        defect_mean = float((y.float() - _fresh().to(torch.bfloat16)(frames).float()).abs().max())
    finally:
        _vae_sampling.SpaceToDepthDownsample.forward = _orig_s2d

    # The chain's response to ONE last bit, which is the irreducible term and is
    # what justifies the ABSENCE of a value tolerance rather than standing in for
    # one.
    _d = _fresh().to(torch.bfloat16)
    with torch.no_grad():
        _flat = dict(_d.named_parameters())["conv_in.conv.weight"].reshape(-1)
        _w = _flat[0].view(torch.int16).item()
        _flat[0] = torch.tensor([_w + 1], dtype=torch.int16).view(torch.bfloat16)[0]
    ulp_sensitivity = float((y.float() - _d(frames).float()).abs().max())

    print(f"[bf16 encoder] arm gap {arm_gap:g}; defects: f32 statistics {defect_stats:g}, "
          f"unrounded group mean {defect_mean:g}; one-ulp sensitivity {ulp_sensitivity:g}",
          file=sys.stderr)
    assert defect_stats > 0, "the f32-statistics defect moved nothing, so the case gates it at zero"
    assert defect_mean > 0, "the unrounded-group-mean defect moved nothing, so the case gates it at zero"
    assert ulp_sensitivity > 0, (
        "the one-ulp sensitivity probe moved nothing, so it cannot support the claim "
        "that this arm has an irreducible term"
    )

    out.write("// --- section 6e: the BF16 arm of VideoEncoder (A24 wave 4, #2850) ---\n")
    emit_scalar(out, "kLtx2VideoEncBf16InC", VIDEO_ENC_BF16_INPUT[1])
    emit_scalar(out, "kLtx2VideoEncBf16InT", VIDEO_ENC_BF16_INPUT[2])
    emit_scalar(out, "kLtx2VideoEncBf16InH", VIDEO_ENC_BF16_INPUT[3])
    emit_scalar(out, "kLtx2VideoEncBf16InW", VIDEO_ENC_BF16_INPUT[4])
    emit_scalar(out, "kLtx2VideoEncBf16OutC", y.shape[1])
    emit_scalar(out, "kLtx2VideoEncBf16OutT", y.shape[2])
    emit_scalar(out, "kLtx2VideoEncBf16OutH", y.shape[3])
    emit_scalar(out, "kLtx2VideoEncBf16OutW", y.shape[4])
    emit_scalar(out, "kLtx2VideoEncBf16GroupSize", max(group_sizes))
    out.write("inline constexpr double kLtx2VideoEncBf16ArmGap = "
              + _cxx_float(arm_gap, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoEncBf16DefectStats = "
              + _cxx_float(defect_stats, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoEncBf16DefectGroupMean = "
              + _cxx_float(defect_mean, 9) + ";\n")
    out.write("inline constexpr double kLtx2VideoEncBf16UlpSensitivity = "
              + _cxx_float(ulp_sensitivity, 9) + ";\n\n")
    emit_manifest(out, "kLtx2VideoEncBf16Param", manifest)
    emit_f32(out, "kLtx2VideoEncBf16Golden", y.float().numpy())



def section_audio_encoder(out) -> None:
    from ltx_core.model.audio_vae.attention import AttentionType
    from ltx_core.model.audio_vae.audio_vae import AudioEncoder
    from ltx_core.model.audio_vae.causality_axis import CausalityAxis
    from ltx_core.model.common.normalization import NormType

    def build(prefix, **overrides):
        cfg = dict(AUDIO_ENC)
        cfg.update(overrides)
        enc = AudioEncoder(
            norm_type=NormType.PIXEL,
            causality_axis=CausalityAxis.HEIGHT,
            attn_type=AttentionType.VANILLA,
            **cfg,
        ).eval()
        return enc, fill_from_stream(enc, prefix=prefix)

    spectrogram = make_input(
        "ltx2.audioenc.input", (1, AUDIO_ENC["in_channels"], AUDIO_ENC_FRAMES, AUDIO_ENC_MEL), 1.0
    )

    enc, manifest = build("ltx2.audioenc.")
    y = enc(spectrogram)
    out.write("// --- section 7a: AudioEncoder with attention (audio_vae.py:60-246) ---\n")
    emit_scalar(out, "kLtx2AudioEncInC", AUDIO_ENC["in_channels"])
    emit_scalar(out, "kLtx2AudioEncInT", AUDIO_ENC_FRAMES)
    emit_scalar(out, "kLtx2AudioEncInF", AUDIO_ENC_MEL)
    emit_scalar(out, "kLtx2AudioEncOutC", y.shape[1])
    emit_scalar(out, "kLtx2AudioEncOutT", y.shape[2])
    emit_scalar(out, "kLtx2AudioEncOutF", y.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioEncParam", manifest)
    emit_f32(out, "kLtx2AudioEncGolden", y.numpy())

    # --- 7b: the SHIPPED configuration, which carries NO attention at all
    # (`attn_resolutions: []`, `mid_block_add_attention: false` in the 2.x audio
    # VAE metadata). Gating only the attention arm would gate a model nobody runs.
    enc_plain, manifest_plain = build(
        "ltx2.audioencplain.", attn_resolutions=set(), mid_block_add_attention=False
    )
    y_plain = enc_plain(spectrogram)
    out.write("// --- section 7b: AudioEncoder, the SHIPPED no-attention config ---\n")
    emit_scalar(out, "kLtx2AudioEncPlainOutC", y_plain.shape[1])
    emit_scalar(out, "kLtx2AudioEncPlainOutT", y_plain.shape[2])
    emit_scalar(out, "kLtx2AudioEncPlainOutF", y_plain.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioEncPlainParam", manifest_plain)
    emit_f32(out, "kLtx2AudioEncPlainGolden", y_plain.numpy())

    # --- 7c: the AVERAGE-POOL downsample arm (`resamp_with_conv=False`), which
    # upstream only allows with causality NONE (downsample.py:28-29). It is a
    # different sampling lattice, not a cheaper one.
    enc_pool = AudioEncoder(
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.NONE,
        attn_type=AttentionType.VANILLA,
        **{**AUDIO_ENC, "resamp_with_conv": False},
    ).eval()
    manifest_pool = fill_from_stream(enc_pool, prefix="ltx2.audioencpool.")
    y_pool = enc_pool(spectrogram)
    out.write("// --- section 7c: AudioEncoder, avg-pool downsample, causality NONE ---\n")
    emit_scalar(out, "kLtx2AudioEncPoolOutC", y_pool.shape[1])
    emit_scalar(out, "kLtx2AudioEncPoolOutT", y_pool.shape[2])
    emit_scalar(out, "kLtx2AudioEncPoolOutF", y_pool.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioEncPoolParam", manifest_pool)
    emit_f32(out, "kLtx2AudioEncPoolGolden", y_pool.numpy())

    # --- 7d: norm_type = GROUP, the arm that READS `norm_eps` on the encoder half.
    # See AUDIO_GROUP_ENC. `ResnetBlock` only permits GroupNorm at causality NONE
    # (resnet.py:130-131), which is a configuration a checkpoint may legally
    # declare and which the pixel arms above leave entirely unexecuted.
    enc_group = AudioEncoder(
        norm_type=NormType.GROUP,
        causality_axis=CausalityAxis.NONE,
        attn_type=AttentionType.VANILLA,
        **AUDIO_GROUP_ENC,
    ).eval()
    manifest_group = fill_from_stream(enc_group, prefix="ltx2.audioencgroup.")
    y_group = enc_group(spectrogram)
    out.write("// --- section 7d: AudioEncoder, norm_type = GROUP, causality NONE ---\n")
    emit_scalar(out, "kLtx2AudioEncGroupOutC", y_group.shape[1])
    emit_scalar(out, "kLtx2AudioEncGroupOutT", y_group.shape[2])
    emit_scalar(out, "kLtx2AudioEncGroupOutF", y_group.shape[3])
    out.write("\n")
    emit_manifest(out, "kLtx2AudioEncGroupParam", manifest_group)
    emit_f32(out, "kLtx2AudioEncGroupGolden", y_group.numpy())


def section_audio_mel(out) -> None:
    import torch
    import torchaudio

    from ltx_core.model.audio_vae.ops import AudioProcessor
    from ltx_core.types import Audio

    # --- 8a: the slaney filterbank on its own, at TWO sample rates. The second is
    # ODD, which is the only way to see `torch.linspace(0, sample_rate // 2, ...)`
    # halving with INTEGER division while `f_max` stays `sample_rate / 2.0`.
    banks = []
    for tag, sr, n_fft, n_mels in (("", 16000, 64, 8), ("Odd", 8001, 32, 6)):
        n_freqs = n_fft // 2 + 1
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=0.0,
            f_max=sr / 2.0,
            n_mels=n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
        assert float(fb.abs().max()) > 0.0, "an all-zero filterbank gates nothing"
        banks.append((tag, sr, n_freqs, n_mels, fb))

    out.write("// --- section 8a: the slaney mel filterbank (torchaudio melscale_fbanks) ---\n")
    for tag, sr, n_freqs, n_mels, fb in banks:
        emit_scalar(out, f"kLtx2Mel{tag}Rate", sr)
        emit_scalar(out, f"kLtx2Mel{tag}Freqs", n_freqs)
        emit_scalar(out, f"kLtx2Mel{tag}Bins", n_mels)
        # TRANSPOSED to [n_mels, n_freqs], the layout every consumer contracts in.
        emit_f32(out, f"kLtx2Mel{tag}BasisGolden", fb.transpose(0, 1).contiguous().numpy())

    # --- 8b: waveform_to_mel end to end ---
    processor = AudioProcessor(
        target_sample_rate=MEL["target_sample_rate"],
        mel_bins=MEL["mel_bins"],
        mel_hop_length=MEL["mel_hop_length"],
        n_fft=MEL["n_fft"],
    )
    wave = make_input("ltx2.mel.input", (1, 2, MEL_SAMPLES), 0.5)
    mel = processor.waveform_to_mel(Audio(waveform=wave, sampling_rate=MEL["target_sample_rate"]))
    out.write("// --- section 8b: AudioProcessor.waveform_to_mel (ops.py:44-55) ---\n")
    emit_scalar(out, "kLtx2MelWaveChannels", wave.shape[1])
    emit_scalar(out, "kLtx2MelWaveSamples", MEL_SAMPLES)
    emit_scalar(out, "kLtx2MelHop", MEL["mel_hop_length"])
    emit_scalar(out, "kLtx2MelNFft", MEL["n_fft"])
    emit_scalar(out, "kLtx2MelOutFrames", mel.shape[2])
    emit_scalar(out, "kLtx2MelOutBins", mel.shape[3])
    out.write("\n")
    emit_f32(out, "kLtx2MelGolden", mel.numpy())

    # --- 8c: SILENCE, the arm that actually enters the clamp's regime. The
    # `torch.clamp(mel, min=1e-5)` at ops.py:52 is invisible to 8b — a well-scaled
    # waveform never gets near it — but real silence saturates EVERY bin, and then
    # the constant alone decides the encoder's input. Asserted, not hoped for.
    silence = torch.zeros_like(wave)
    raw = processor.mel_transform(silence)
    assert float(raw.max()) < 1e-5, (
        "the silence probe must drive EVERY mel bin under the clamp or it gates nothing; "
        f"max was {float(raw.max())}"
    )
    quiet = processor.waveform_to_mel(
        Audio(waveform=silence, sampling_rate=MEL["target_sample_rate"])
    )
    out.write("// --- section 8c: silence, which SATURATES the 1e-5 log clamp ---\n")
    emit_scalar(out, "kLtx2MelQuietFrames", quiet.shape[2])
    out.write("\n")
    emit_f32(out, "kLtx2MelQuietGolden", quiet.numpy())

    # --- 8d: resample_audio on its own, at four ratios (ops.py:36-42) ---
    #
    # Run through `AudioProcessor.resample_audio` rather than through
    # `torchaudio.functional.resample` directly, so the goldens carry upstream's
    # OWN call — its argument order, its equal-rate early return, and its
    # `.to(dtype=waveform.dtype)` — and not this generator's reading of it.
    out.write("// --- section 8d: AudioProcessor.resample_audio (ops.py:36-42) ---\n")
    emit_scalar(out, "kLtx2ResampleChannels", RESAMPLE_CHANNELS)
    for tag, orig, target, length in RESAMPLE_CASES:
        wave = make_input(
            f"ltx2.resample.{tag}", (1, RESAMPLE_CHANNELS, length), 0.5
        )
        proc = AudioProcessor(
            target_sample_rate=target,
            mel_bins=MEL["mel_bins"],
            mel_hop_length=MEL["mel_hop_length"],
            n_fft=MEL["n_fft"],
        )
        got = proc.resample_audio(Audio(waveform=wave, sampling_rate=orig))
        assert got.sampling_rate == target
        assert got.waveform.dtype == torch.float32, (
            "upstream resamples in the WAVEFORM's dtype; a golden emitted from "
            "anything but float32 would gate the wrong arithmetic"
        )
        if orig == target:
            assert got.waveform.data_ptr() == wave.data_ptr(), (
                "ops.py:38-39 returns the SAME Audio when the rates match; a "
                "golden that went through the filter would not gate that branch"
            )
        assert float(got.waveform.abs().max()) > 0.0, "an all-zero golden gates nothing"
        emit_scalar(out, f"kLtx2Resample{tag}OrigRate", orig)
        emit_scalar(out, f"kLtx2Resample{tag}NewRate", target)
        emit_scalar(out, f"kLtx2Resample{tag}InSamples", length)
        emit_scalar(out, f"kLtx2Resample{tag}OutSamples", got.waveform.shape[-1])
        emit_f32(out, f"kLtx2Resample{tag}Golden", got.waveform.numpy())

    # --- 8e: waveform_to_mel THROUGH the resampler, the production shape ---
    #
    # 8d proves the filter; this proves the CALL. `waveform_to_mel` resamples
    # first (ops.py:49) and the mel transform then runs on the resampled length,
    # so a port that resampled AFTER the transform, or that padded before it,
    # matches 8d exactly and fails here.
    source = make_input("ltx2.mel.resampled.input", (1, 2, MEL_SOURCE_SAMPLES), 0.5)
    resampled_mel = processor.waveform_to_mel(
        Audio(waveform=source, sampling_rate=MEL_SOURCE_RATE)
    )
    out.write("// --- section 8e: waveform_to_mel at a NON-target rate (ops.py:44-55) ---\n")
    emit_scalar(out, "kLtx2MelSourceRate", MEL_SOURCE_RATE)
    emit_scalar(out, "kLtx2MelSourceSamples", MEL_SOURCE_SAMPLES)
    emit_scalar(out, "kLtx2MelResampledFrames", resampled_mel.shape[2])
    out.write("\n")
    emit_f32(out, "kLtx2MelResampledGolden", resampled_mel.numpy())

    # --- 8f: the truncation boundary (functional.py:1427-1428) ---
    #
    # Same call as 8d, at lengths that reach where upstream's f32-narrowed ceil
    # and an exact integer ceil disagree, in BOTH directions, and where the slice
    # on the next line clamps that ceil to the columns the convolution produced.
    # Emitted as a length plus a tail window, because the widest arm is 33 554 435
    # samples.
    out.write("// --- section 8f: the truncation boundary (functional.py:1427-1428) ---\n")
    emit_scalar(out, "kLtx2ResampleCeilTail", RESAMPLE_CEIL_TAIL)
    for tag, orig, target, length, ceil_delta, clamps in RESAMPLE_CEIL_CASES:
        wave = make_input(f"ltx2.resample.{tag}", (1, 1, length), 0.5)
        proc = AudioProcessor(
            target_sample_rate=target,
            mel_bins=MEL["mel_bins"],
            mel_hop_length=MEL["mel_hop_length"],
            n_fft=MEL["n_fft"],
        )
        got = proc.resample_audio(Audio(waveform=wave, sampling_rate=orig))
        assert got.sampling_rate == target
        assert got.waveform.dtype == torch.float32
        # `.shape[-1]` — what upstream RETURNS, which is the only thing a port has
        # to reproduce. The 276060-pair sweep that this row ran on its first pass
        # compared the port's expression against the FORMULA at :1427 and found
        # zero divergences, and it still missed the clamp at :1428, because the
        # clamp is not in the expression. So every number below is measured from
        # this tensor, and the two predictions are checked against it.
        produced = int(got.waveform.shape[-1])
        # The POSITIVE CONTROL on this arm, in both directions.
        #
        # `exact_ceil` is what a reader writes from the formula as printed;
        # `narrowed` is what :1427 actually computes; `columns` is what the
        # convolution at :1425-1426 leaves for :1428 to slice. Declaring the
        # relation between the three per arm is what stops an arm from quietly
        # ceasing to discriminate: a torchaudio that stopped narrowing, or a
        # length whose arithmetic moved, fails the GENERATOR here rather than
        # emitting a golden that gates nothing.
        gcd = math.gcd(orig, target)
        o, n = orig // gcd, target // gcd
        exact_ceil = -(-(n * length) // o)
        narrowed = int(torch.ceil(torch.as_tensor(n * length / o)).long())
        columns = (length // o + 1) * n
        assert narrowed - exact_ceil == ceil_delta, (
            f"{tag}: expected the f32 narrowing to land {ceil_delta:+d} from the "
            f"exact integer ceil, got narrowed={narrowed} exact={exact_ceil}"
        )
        assert (narrowed > columns) == clamps, (
            f"{tag}: expected clamps={clamps}, got narrowed={narrowed} "
            f"columns={columns}"
        )
        assert produced == min(narrowed, columns), (
            f"{tag}: upstream returned {produced}, but :1427-1428 predict "
            f"min({narrowed}, {columns}) = {min(narrowed, columns)}"
        )
        tail = got.waveform.reshape(-1)[-RESAMPLE_CEIL_TAIL:]
        assert float(tail.abs().max()) > 0.0, "an all-zero tail gates nothing"
        emit_scalar(out, f"kLtx2Resample{tag}OrigRate", orig)
        emit_scalar(out, f"kLtx2Resample{tag}NewRate", target)
        emit_scalar(out, f"kLtx2Resample{tag}InSamples", length)
        emit_scalar(out, f"kLtx2Resample{tag}OutSamples", produced)
        emit_f32(out, f"kLtx2Resample{tag}TailGolden", tail.numpy())


# ---------------------------------------------------------------------------
# Section 9 — the CONDITIONING ITEMS (phase L11). What the encoders' output is
# FOR: placing an encoded image / keyframe / reference video / reference audio
# into the token stream the DiT already accepts.
# ---------------------------------------------------------------------------

COND_VIDEO_SHAPE = dict(batch=1, channels=4, frames=3, height=2, width=2)
COND_VIDEO_PATCH = 1
COND_VIDEO_FPS = 8.0
COND_AUDIO_SHAPE = dict(batch=1, channels=2, frames=4, mel_bins=2)


def section_conditioning(out) -> None:
    import torch

    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
    from ltx_core.conditioning.types.latent_cond import VideoConditionByLatentIndex
    from ltx_core.conditioning.types.reference_audio_cond import AudioConditionByReferenceLatent
    from ltx_core.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape

    factors = SpatioTemporalScaleFactors.default()
    target = VideoLatentShape(**COND_VIDEO_SHAPE)
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=COND_VIDEO_PATCH),
        target_shape=target,
        fps=COND_VIDEO_FPS,
        scale_factors=factors,
        causal_fix=True,
    )
    base = tools.create_initial_state(device="cpu", dtype=torch.float32)

    def emit_state(out, prefix, state, tokens_before):
        emit_scalar(out, prefix + "Tokens", state.latent.shape[1])
        emit_scalar(out, prefix + "Width", state.latent.shape[2])
        emit_scalar(out, prefix + "PosDims", state.positions.shape[1])
        emit_scalar(out, prefix + "TokensBefore", tokens_before)
        out.write("\n")
        emit_f32(out, prefix + "Clean", state.clean_latent.numpy())
        emit_f32(out, prefix + "Latent", state.latent.numpy())
        emit_f32(out, prefix + "Mask", state.denoise_mask.numpy())
        emit_f32(out, prefix + "Positions", state.positions.numpy())

    out.write("// --- section 9a: the INITIAL video state (tools.py:139-186) ---\n")
    emit_state(out, "kLtx2CondVideoBase", base, base.latent.shape[1])
    emit_f32(out, "kLtx2CondVideoBaseKeyframesMask", base.keyframes_mask.numpy())

    # --- 9b: VideoConditionByLatentIndex — an image REPLACING one latent frame.
    cond_latent = make_input("ltx2.cond.image", (1, 4, 1, 2, 2), 1.0)
    by_index = VideoConditionByLatentIndex(latent=cond_latent, strength=0.7, latent_idx=1)
    state_index = by_index.apply_to(base, tools)
    assert not torch.equal(state_index.clean_latent, base.clean_latent), (
        "the latent-index item must CHANGE the clean latent or it gates nothing"
    )
    assert not torch.equal(state_index.denoise_mask, base.denoise_mask), (
        "the latent-index item must CHANGE the denoise mask or it gates nothing"
    )
    assert torch.equal(state_index.latent, base.latent), (
        "upstream leaves the NOISY tensor untouched here (latent_cond.py:40-41); if that ever "
        "changes, the port's noising composition changes with it"
    )
    out.write("// --- section 9b: VideoConditionByLatentIndex (latent_cond.py:9-43) ---\n")
    emit_scalar(out, "kLtx2CondIndexLatentIdx", 1)
    emit_state(out, "kLtx2CondIndex", state_index, base.latent.shape[1])

    # --- 9c: VideoConditionByKeyframeIndex — an image APPENDED at a pixel frame.
    keyframe = make_input("ltx2.cond.keyframe", (1, 4, 1, 2, 2), 1.0)
    by_keyframe = VideoConditionByKeyframeIndex(
        keyframes=keyframe, frame_idx=5, strength=0.6, num_pixel_frames=1
    )
    state_kf = by_keyframe.apply_to(base, tools)
    assert state_kf.latent.shape[1] > base.latent.shape[1], "the keyframe item must APPEND tokens"
    assert state_kf.attention_mask is None, (
        "with attention_mask=None and no existing mask, update_attention_mask returns None "
        "(mask_utils.py:110-140); the port relies on that"
    )
    out.write("// --- section 9c: VideoConditionByKeyframeIndex (keyframe_cond.py:36-90) ---\n")
    emit_scalar(out, "kLtx2CondKeyframeFrameIdx", 5)
    emit_state(out, "kLtx2CondKeyframe", state_kf, base.latent.shape[1])

    # --- 9d: VideoConditionByReferenceLatent — a reference video, IC-LoRA style.
    reference = make_input("ltx2.cond.reference", (1, 4, 2, 2, 2), 1.0)
    by_reference = VideoConditionByReferenceLatent(
        latent=reference, downscale_factor=2, temporal_scale_factor=2, strength=1.0
    )
    state_ref = by_reference.apply_to(base, tools)
    assert state_ref.latent.shape[1] > base.latent.shape[1], "the reference item must APPEND tokens"
    out.write("// --- section 9d: VideoConditionByReferenceLatent (reference_video_cond.py) ---\n")
    emit_scalar(out, "kLtx2CondRefDownscale", 2)
    emit_scalar(out, "kLtx2CondRefTemporalScale", 2)
    emit_state(out, "kLtx2CondRef", state_ref, base.latent.shape[1])

    # A downscale/temporal factor of 1 must take the OTHER branch — both guards
    # are `if factor != 1`, so an implementation that always applied them would
    # pass 9d and fail here.
    plain_ref = VideoConditionByReferenceLatent(
        latent=reference, downscale_factor=1, temporal_scale_factor=1, strength=1.0
    ).apply_to(base, tools)
    assert not torch.equal(plain_ref.positions, state_ref.positions), (
        "the scaled and unscaled reference positions must DIFFER, or 9d gates neither factor"
    )
    out.write("// --- section 9d': the unscaled reference branch ---\n")
    emit_f32(out, "kLtx2CondRefPlainPositions", plain_ref.positions.numpy())

    # --- 9e: AudioConditionByReferenceLatent ---
    audio_target = AudioLatentShape(**COND_AUDIO_SHAPE)
    audio_patchifier = AudioPatchifier(patch_size=1)
    audio_tools = AudioLatentTools(patchifier=audio_patchifier, target_shape=audio_target)
    audio_base = audio_tools.create_initial_state(device="cpu", dtype=torch.float32)
    out.write("// --- section 9e: the INITIAL audio state (tools.py:246-279) ---\n")
    emit_state(out, "kLtx2CondAudioBase", audio_base, audio_base.latent.shape[1])

    ref_audio_latent = make_input("ltx2.cond.refaudio", (1, 2, 3, 2), 1.0)
    ref_tokens = audio_patchifier.patchify(ref_audio_latent)
    ref_positions = audio_patchifier.get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=1, channels=2, frames=3, mel_bins=2)
    )
    audio_ref = AudioConditionByReferenceLatent(
        patchified=ref_tokens, positions=ref_positions, strength=1.0
    ).apply_to(audio_base, audio_tools)
    assert audio_ref.latent.shape[1] > audio_base.latent.shape[1], (
        "the audio reference item must APPEND tokens"
    )
    out.write("// --- section 9f: AudioConditionByReferenceLatent (reference_audio_cond.py) ---\n")
    emit_scalar(out, "kLtx2CondRefAudioFrames", 3)
    emit_state(out, "kLtx2CondRefAudio", audio_ref, audio_base.latent.shape[1])


def load_upstream(root: Path) -> Path:
    """Import `ltx_core` BY PATH from `root`, and prove that is what resolved."""
    src = root / "packages" / "ltx-core" / "src"
    if not (src / "ltx_core" / "model" / "audio_vae" / "audio_vae.py").is_file():
        raise SystemExit(f"no ltx_core under {src}; point --ltx2 at a Lightricks/LTX-2 checkout")
    sys.path.insert(0, str(src))
    import ltx_core  # noqa: PLC0415

    # ORACLE IDENTITY, asserted rather than assumed: a pip-installed or editable
    # `ltx_core` that wins path resolution would import silently and gate every
    # golden below against the wrong source. See the module docstring.
    resolved = Path(ltx_core.__file__).resolve()
    if not resolved.is_relative_to(src.resolve()):
        raise SystemExit(
            f"ltx_core resolved to {resolved}, which is NOT under the checkout at {src}. "
            "Refusing to generate goldens from an oracle this script did not choose."
        )
    return src


def upstream_revision(root: Path) -> str:
    """The exact upstream tree these goldens were produced from.

    AND REFUSE A DIRTY ONE. `git rev-parse HEAD` reports the COMMITTED sha
    whatever the working tree looks like, so a checkout with one edited line
    stamps a clean anchor onto goldens the pinned commit cannot reproduce — the
    anchor then actively misleads, which is worse than having none. `git status
    --porcelain` is empty exactly when the tree matches the commit, so it is the
    check that makes the recorded sha mean what it says.

    A checkout with no git metadata at all still returns "unknown", which the
    suite rejects on its own; only a REAL repository is held to cleanliness.
    """
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
            "Refusing to generate: `git rev-parse HEAD` would stamp a CLEAN revision anchor onto "
            "goldens produced by a tree that commit cannot reproduce. Commit or stash first."
        )
    return done.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path,
                        help="a checkout of Lightricks/LTX-2 (the repo root)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.ltx2.expanduser().resolve()
    load_upstream(root)
    revision = upstream_revision(root)

    import torch

    torch.set_grad_enabled(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-vae-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// LTX-2.5 VAE goldens, produced by executing the UPSTREAM ltx_core modules\n"
            "// (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/model/) at reduced\n"
            "// dimensions on CPU. Weights and inputs come from the shared deterministic\n"
            "// stream, so no weight byte is checked in. Regenerate with:\n"
            "//   python3 scripts/gen-ltx2-vae-goldens.py --ltx2 <LTX-2 checkout>\n"
            "//       --out tests/vllm/models/ltx2_vae_goldens.inc\n"
            "//\n"
            f"// Upstream revision: {revision}\n"
            "//\n"
            "// See .agents/specs/ltx-2-5.md section 7 for why this is the gate.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
            "// The upstream tree these numbers came from. The suite asserts this equals\n"
            "// the SHA it pins, so regenerating against a DIFFERENT checkout fails the\n"
            "// gate instead of silently replacing the oracle.\n"
            f'inline constexpr const char* kLtx2VaeUpstreamRevision = "{revision}";\n\n'
        )
        section_audio_decoder(out)
        section_vocoder(out)
        section_vocoder_legacy(out)
        section_bwe(out)
        section_conv_video_decoder(out)
        section_conv_video_decoder_bf16(out)
        section_video_vae_bf16_kernels(out)
        section_conv_video_decoder_bf16_shallow(out)
        section_video_encoder(out)
        section_video_encoder_bf16(out)
        section_audio_encoder(out)
        section_audio_mel(out)
        section_conditioning(out)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
