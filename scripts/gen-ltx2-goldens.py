#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_goldens.inc — the LTX-2.5 DiT parity oracle.

LTX-2.5 (`Lightricks/LTX-2.5`) is a 21.00B joint video+audio flow-matching
diffusion transformer. Its DiT alone is ~19 GB even in NVFP4, and there is no
vLLM-Omni native path to it (.agents/specs/ltx-2-5.md section 3), so the port
cannot be gated end to end here. What CAN be gated exactly, on any CPU, is the
MATH: this generator IMPORTS THE UPSTREAM MODULES BY PATH and runs them at
REDUCED dimensions with deterministic pseudo-random weights, then emits the
resulting tensors as C++ goldens. The C++ suite regenerates the identical weights
and inputs from the identical PRNG and must reproduce these outputs, so not one
weight byte is checked in.

This is MiniMax-H3's method (scripts/gen-minimax-h3-goldens.py), applied to an
upstream that is easier to reach: `ltx_core` needs only a sys.path entry pointing
at a LTX-2 checkout, no venv and no checkpoint. Nothing here is a restatement —
every tensor below comes out of upstream's own `LTXModel`, `Attention`,
`FeedForward`, `AdaLayerNormSingle` and `precompute_freqs_cis`.

Upstream sources (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/):
  model/transformer/model.py            -> the full DiT forward goldens
  model/transformer/transformer.py      -> exercised through that forward
  model/transformer/attention.py        -> the gated-attention brick
  model/transformer/feed_forward.py     -> the ff / audio_ff bias-asymmetry brick
  model/transformer/adaln.py            -> the AdaLN-single brick
  model/transformer/rope.py             -> the split / interleaved / float64 bricks

Usage:
    python3 scripts/gen-ltx2-goldens.py \
        --ltx2 ~/_git/LTX-2 \
        --out tests/vllm/models/ltx2_goldens.inc

Needs torch + numpy (CPU only).
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Deterministic weight/input stream, mirrored bit-for-bit by the C++ suite
# (tests/vllm/models/test_ltx2.cpp :: Ltx2Rand). A per-tensor FNV-1a seed plus a
# splitmix64 counter makes every tensor independent of fill ORDER, so the two
# sides cannot silently drift by reordering their parameter construction.
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


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


def ltx2_rand(name: str, count: int) -> np.ndarray:
    """`count` values uniform in [-1, 1), reproducible from `name` alone."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        # Top 53 bits -> [0, 1), then map to [-1, 1). Both sides use the same
        # 53-bit mantissa construction so the doubles are bit-identical.
        unit = (u >> 11) * (2.0**-53)
        out[i] = unit * 2.0 - 1.0
    return out


def make_param(name: str, shape, scale: float, offset: float = 0.0) -> torch.Tensor:
    count = int(np.prod(shape)) if len(shape) else 1
    values = ltx2_rand(name, count) * scale + offset
    return torch.from_numpy(values.astype(np.float32)).reshape(tuple(shape))


def param_spec(name: str) -> tuple[float, float]:
    """(scale, offset) for a parameter, keyed ONLY by its name.

    The C++ suite applies the identical rule, so a name either side invents that
    the other does not have produces different numbers and the gate fails — which
    is what makes the weight CONTRACT part of the gate rather than an assumption.
    RMSNorm weights sit around 1.0 (torch initializes them to ones) so the q/k
    norms do not crush the attention scores; everything else is small and centred.
    """
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        return 0.1, 1.0
    if name.endswith(".bias"):
        return 0.02, 0.0
    return 0.05, 0.0


def fill_parameters(module: torch.nn.Module) -> None:
    for name, param in module.named_parameters():
        scale, offset = param_spec(name)
        param.copy_(make_param(name, tuple(param.shape), scale, offset))


# ---------------------------------------------------------------------------
# Reduced-dimension arch. Every ratio the port branches on is preserved: two
# streams of DIFFERENT width, a head count shared between them (which
# model_configurator.py:44 asserts) but DIFFERENT head dims, distinct video and
# audio latent widths, a text context per stream, gated attention on, cross
# attention AdaLN on, the prompt AdaLN MLP off, and the ff / audio_ff bias
# asymmetry. Only the magnitudes shrink.
#
# Two constraints the shapes must respect, both from rope.py:
#   inner_dim // (2 * n_pos_dims) must be >= 1  (video: 32 // 6 = 5)
#   audio_cross_attention_dim == audio inner    (the cross RoPE is built at it)
# ---------------------------------------------------------------------------

ARCH = dict(
    num_attention_heads=4,
    attention_head_dim=8,  # video inner dim 32
    in_channels=8,
    out_channels=8,
    num_layers=2,
    cross_attention_dim=32,
    norm_eps=1e-6,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    audio_num_attention_heads=4,
    audio_attention_head_dim=4,  # audio inner dim 16
    audio_in_channels=6,
    audio_out_channels=6,
    audio_cross_attention_dim=16,
    audio_positional_embedding_max_pos=[20],
    av_ca_timestep_scale_multiplier=1,
    apply_gated_attention=True,
    cross_attention_adaln=True,
    use_prompt_adaln_single=False,
    ff_bias=False,
    audio_ff_bias=True,
)

BATCH = 2
VIDEO_GRID = (2, 2, 2)  # (t, h, w) -> 8 video tokens
VIDEO_TOKENS = VIDEO_GRID[0] * VIDEO_GRID[1] * VIDEO_GRID[2]
AUDIO_TOKENS = 5
VIDEO_CONTEXT = 3
AUDIO_CONTEXT = 4


def video_dim() -> int:
    return ARCH["num_attention_heads"] * ARCH["attention_head_dim"]


def audio_dim() -> int:
    return ARCH["audio_num_attention_heads"] * ARCH["audio_attention_head_dim"]


# ---------------------------------------------------------------------------
# Upstream import — by PATH, so no install and no environment leakage.
# ---------------------------------------------------------------------------


def load_upstream(root: Path):
    src = root / "packages" / "ltx-core" / "src"
    if not (src / "ltx_core").is_dir():
        raise SystemExit(f"not an LTX-2 checkout: {root} (expected {src}/ltx_core)")
    sys.path.insert(0, str(src))
    import ltx_core  # noqa: PLC0415

    # ORACLE IDENTITY, asserted rather than assumed: an ltx_core installed in
    # site-packages would import silently and gate against the wrong source.
    resolved = Path(ltx_core.__file__).resolve()
    if not str(resolved).startswith(str(src.resolve())):
        raise SystemExit(f"ltx_core resolved to {resolved}, not to the checkout at {src}")
    return src


def upstream_revision(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - a tarball checkout has no git metadata
        return "unknown"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def video_positions() -> torch.Tensor:
    """The middle-indices patch grid (patchifiers.py:95-138), int64 as upstream builds it."""
    t, h, w = VIDEO_GRID
    coords = []
    for ti in range(t):
        for hi in range(h):
            for wi in range(w):
                coords.append((ti, hi, wi))
    grid = torch.zeros(BATCH, 3, VIDEO_TOKENS, 2, dtype=torch.int64)
    for idx, (ti, hi, wi) in enumerate(coords):
        for axis, value in enumerate((ti, hi, wi)):
            grid[:, axis, idx, 0] = value
            grid[:, axis, idx, 1] = value + 1
    return grid


def audio_positions() -> torch.Tensor:
    grid = torch.zeros(BATCH, 1, AUDIO_TOKENS, 2, dtype=torch.int64)
    for idx in range(AUDIO_TOKENS):
        grid[:, 0, idx, 0] = idx
        grid[:, 0, idx, 1] = idx + 1
    return grid


def rand_input(name: str, shape, scale: float, offset: float = 0.0) -> torch.Tensor:
    return make_param(name, shape, scale, offset)


# The masked case's inputs, defined ONCE so the emitted copies and the copies the
# forward actually runs on cannot drift apart. A {0,1} prompt mask (the padding
# form _prepare_attention_mask converts) plus a key-only [0,1] self-attention
# STRENGTH mask.
def video_context_mask() -> torch.Tensor:
    return torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)


def audio_context_mask() -> torch.Tensor:
    return torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)


def video_self_mask() -> torch.Tensor:
    return make_param("input.video.attention_mask", (BATCH, 1, VIDEO_TOKENS), 0.5, 0.5)


def audio_self_mask() -> torch.Tensor:
    return make_param("input.audio.attention_mask", (BATCH, 1, AUDIO_TOKENS), 0.5, 0.5)


# The DENSE `(B, T, T)` self-attention mask — the form transformer_args.py:212-215
# documents as *the* dense one, where every QUERY carries its own row of key
# strengths. The key-only `(B, 1, T)` broadcast above cannot distinguish a kernel
# that reads each query's own bias row from one that reads row 0 for every query,
# because there is only one row; this shape is what separates them.
def video_self_mask_dense() -> torch.Tensor:
    return make_param(
        "input.video.attention_mask_dense", (BATCH, VIDEO_TOKENS, VIDEO_TOKENS), 0.5, 0.5
    )


def audio_self_mask_dense() -> torch.Tensor:
    return make_param(
        "input.audio.attention_mask_dense", (BATCH, AUDIO_TOKENS, AUDIO_TOKENS), 0.5, 0.5
    )


def build_modalities(masked: bool, audio_enabled: bool = True, dense_self_mask: bool = False):
    from ltx_core.model.transformer.modality import Modality  # noqa: PLC0415

    video_ctx_mask = None
    audio_ctx_mask = None
    video_attn_mask = None
    audio_attn_mask = None
    if masked:
        video_ctx_mask = video_context_mask()
        audio_ctx_mask = audio_context_mask()
        if dense_self_mask:
            video_attn_mask = video_self_mask_dense()
            audio_attn_mask = audio_self_mask_dense()
        else:
            video_attn_mask = video_self_mask()
            audio_attn_mask = audio_self_mask()

    video = Modality(
        latent=rand_input("input.video.latent", (BATCH, VIDEO_TOKENS, ARCH["in_channels"]), 0.5),
        sigma=rand_input("input.video.sigma", (BATCH,), 0.25, 0.5),
        timesteps=rand_input("input.video.timesteps", (BATCH, VIDEO_TOKENS), 0.25, 0.5),
        positions=video_positions(),
        context=rand_input(
            "input.video.context", (BATCH, VIDEO_CONTEXT, ARCH["cross_attention_dim"]), 0.5
        ),
        enabled=True,
        context_mask=video_ctx_mask,
        attention_mask=video_attn_mask,
    )
    audio = Modality(
        latent=rand_input(
            "input.audio.latent", (BATCH, AUDIO_TOKENS, ARCH["audio_in_channels"]), 0.5
        ),
        sigma=rand_input("input.audio.sigma", (BATCH,), 0.25, 0.5),
        timesteps=rand_input("input.audio.timesteps", (BATCH, AUDIO_TOKENS), 0.25, 0.5),
        positions=audio_positions(),
        context=rand_input(
            "input.audio.context", (BATCH, AUDIO_CONTEXT, ARCH["audio_cross_attention_dim"]), 0.5
        ),
        enabled=audio_enabled,
        context_mask=audio_ctx_mask,
        attention_mask=audio_attn_mask,
    )
    return video, audio


def build_model(rope_type_name: str, double_rope: bool):
    from ltx_core.model.transformer.model import LTXModel, LTXModelType  # noqa: PLC0415
    from ltx_core.model.transformer.rope import LTXRopeType  # noqa: PLC0415

    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=ARCH["num_attention_heads"],
        attention_head_dim=ARCH["attention_head_dim"],
        in_channels=ARCH["in_channels"],
        out_channels=ARCH["out_channels"],
        num_layers=ARCH["num_layers"],
        cross_attention_dim=ARCH["cross_attention_dim"],
        norm_eps=ARCH["norm_eps"],
        positional_embedding_theta=ARCH["positional_embedding_theta"],
        positional_embedding_max_pos=list(ARCH["positional_embedding_max_pos"]),
        timestep_scale_multiplier=ARCH["timestep_scale_multiplier"],
        use_middle_indices_grid=ARCH["use_middle_indices_grid"],
        audio_num_attention_heads=ARCH["audio_num_attention_heads"],
        audio_attention_head_dim=ARCH["audio_attention_head_dim"],
        audio_in_channels=ARCH["audio_in_channels"],
        audio_out_channels=ARCH["audio_out_channels"],
        audio_cross_attention_dim=ARCH["audio_cross_attention_dim"],
        audio_positional_embedding_max_pos=list(ARCH["audio_positional_embedding_max_pos"]),
        av_ca_timestep_scale_multiplier=ARCH["av_ca_timestep_scale_multiplier"],
        rope_type=LTXRopeType(rope_type_name),
        double_precision_rope=double_rope,
        apply_gated_attention=ARCH["apply_gated_attention"],
        cross_attention_adaln=ARCH["cross_attention_adaln"],
        use_prompt_adaln_single=ARCH["use_prompt_adaln_single"],
        ff_bias=ARCH["ff_bias"],
        audio_ff_bias=ARCH["audio_ff_bias"],
    )
    fill_parameters(model)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------


def emit_header(out, argv: str, revision: str) -> None:
    out.write(
        "// GENERATED by scripts/gen-ltx2-goldens.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// LTX-2.5 DiT parity goldens produced by IMPORTING and EXECUTING the upstream\n"
        "// Lightricks LTX-2 modules (packages/ltx-core/src/ltx_core/model/transformer/)\n"
        "// at reduced dimensions with the deterministic Ltx2Rand stream.\n"
        f"// Upstream revision: {revision}\n"
        "// Regenerate with:\n"
        f"//   {argv}\n"
        "//\n"
        "// See .agents/specs/ltx-2-5.md sections 0 and 7 for why this is the gate: the\n"
        "// shipped DiT is ~19 GB and vLLM-Omni carries no native 2.5 path, so the MATH is\n"
        "// gated exactly here and no weight byte is checked in.\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace vllm_test {\n\n"
    )


def _cxx_float(value: float, digits: int) -> str:
    """Format as a valid C++ floating literal (`0` alone is an integer literal)."""
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


def emit_f64(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1).tolist()
    out.write(f"inline constexpr double {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(_cxx_float(v, 17) for v in flat[i : i + 6])
        out.write("    " + chunk + ",\n")
    out.write("};\n\n")


def emit_i64(out, name: str, values) -> None:
    flat = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(str(v) for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_scalar(out, name: str, value) -> None:
    out.write(f"inline constexpr int64_t {name} = {int(value)};\n")


def tensor(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().numpy()


# ---------------------------------------------------------------------------
# Golden sections
# ---------------------------------------------------------------------------


def emit_arch(out) -> None:
    out.write("// --- section 0: the reduced architecture, mirrored by the C++ suite ---\n")
    for key in (
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "num_layers",
        "cross_attention_dim",
        "timestep_scale_multiplier",
        "audio_num_attention_heads",
        "audio_attention_head_dim",
        "audio_in_channels",
        "audio_out_channels",
        "audio_cross_attention_dim",
        "av_ca_timestep_scale_multiplier",
    ):
        emit_scalar(out, f"kLtx2Arch_{key}", ARCH[key])
    emit_scalar(out, "kLtx2Batch", BATCH)
    emit_scalar(out, "kLtx2VideoTokens", VIDEO_TOKENS)
    emit_scalar(out, "kLtx2AudioTokens", AUDIO_TOKENS)
    emit_scalar(out, "kLtx2VideoContext", VIDEO_CONTEXT)
    emit_scalar(out, "kLtx2AudioContext", AUDIO_CONTEXT)
    out.write("\n")
    emit_f64(out, "kLtx2VideoPositions", video_positions().to(torch.float64))
    emit_f64(out, "kLtx2AudioPositions", audio_positions().to(torch.float64))


def emit_manifest(out, model) -> None:
    """The upstream parameter LIST is the layout contract; gate it verbatim."""
    out.write("// --- section 1: upstream named_parameters() — the weight contract ---\n")
    names = []
    ranks = []
    dims = []
    for name, param in model.named_parameters():
        names.append(name)
        ranks.append(len(param.shape))
        dims.extend(int(d) for d in param.shape)
    out.write(f"inline constexpr const char* kLtx2ParamNames[] = {{\n")
    for name in names:
        out.write(f'    "{name}",\n')
    out.write("};\n\n")
    emit_i64(out, "kLtx2ParamRanks", ranks)
    emit_i64(out, "kLtx2ParamDims", dims)
    emit_scalar(out, "kLtx2ParamCount", len(names))
    out.write("\n")


def emit_rope(out, model) -> None:
    """precompute_freqs_cis, both flavours and both frequency precisions."""
    from ltx_core.model.transformer.rope import (  # noqa: PLC0415
        LTXRopeType,
        generate_freq_grid_np,
        generate_freq_grid_pytorch,
        precompute_freqs_cis,
    )

    out.write("// --- section 2: rope.py — the frequency ladder and both cos/sin layouts ---\n")
    # The ladder itself: theta ** linspace(0, 1, n) * pi/2, in f32 and in f64.
    emit_f32(out, "kLtx2FreqGridVideoF32", generate_freq_grid_pytorch(10000.0, 3, video_dim()))
    emit_f32(out, "kLtx2FreqGridVideoF64", generate_freq_grid_np(10000.0, 3, video_dim()))
    emit_f32(out, "kLtx2FreqGridAudioF32", generate_freq_grid_pytorch(10000.0, 1, audio_dim()))
    emit_f32(out, "kLtx2FreqGridAudioF64", generate_freq_grid_np(10000.0, 1, audio_dim()))

    vpos = video_positions()
    apos = audio_positions()
    cross_max = max(
        ARCH["positional_embedding_max_pos"][0], ARCH["audio_positional_embedding_max_pos"][0]
    )
    cases = (
        ("Split", LTXRopeType.SPLIT, False),
        ("Interleaved", LTXRopeType.INTERLEAVED, False),
        ("Double", LTXRopeType.SPLIT, True),
    )
    for tag, rope_type, double in cases:
        gen = generate_freq_grid_np if double else generate_freq_grid_pytorch
        vcos, vsin = precompute_freqs_cis(
            vpos,
            dim=video_dim(),
            out_dtype=torch.float32,
            theta=ARCH["positional_embedding_theta"],
            max_pos=list(ARCH["positional_embedding_max_pos"]),
            use_middle_indices_grid=True,
            num_attention_heads=ARCH["num_attention_heads"],
            rope_type=rope_type,
            freq_grid_generator=gen,
        )
        acos, asin = precompute_freqs_cis(
            apos,
            dim=audio_dim(),
            out_dtype=torch.float32,
            theta=ARCH["positional_embedding_theta"],
            max_pos=list(ARCH["audio_positional_embedding_max_pos"]),
            use_middle_indices_grid=True,
            num_attention_heads=ARCH["audio_num_attention_heads"],
            rope_type=rope_type,
            freq_grid_generator=gen,
        )
        # transformer_args.py:364-371 — the audio<->video cross RoPE: the TIME
        # axis only, built at audio_cross_attention_dim.
        ccos, csin = precompute_freqs_cis(
            vpos[:, 0:1, :],
            dim=ARCH["audio_cross_attention_dim"],
            out_dtype=torch.float32,
            theta=ARCH["positional_embedding_theta"],
            max_pos=[cross_max],
            use_middle_indices_grid=True,
            num_attention_heads=ARCH["num_attention_heads"],
            rope_type=rope_type,
            freq_grid_generator=gen,
        )
        emit_f32(out, f"kLtx2Rope{tag}VideoCos", tensor(vcos))
        emit_f32(out, f"kLtx2Rope{tag}VideoSin", tensor(vsin))
        emit_f32(out, f"kLtx2Rope{tag}AudioCos", tensor(acos))
        emit_f32(out, f"kLtx2Rope{tag}AudioSin", tensor(asin))
        emit_f32(out, f"kLtx2Rope{tag}CrossCos", tensor(ccos))
        emit_f32(out, f"kLtx2Rope{tag}CrossSin", tensor(csin))


def emit_bricks(out, model) -> None:
    """The leaf modules, each run standalone so a failure localizes."""
    out.write("// --- section 3: the leaf bricks (adaln, gated attention, ff) ---\n")
    dim = video_dim()
    adim = audio_dim()

    # AdaLayerNormSingle (adaln.py:39-45). The timesteps arrive ALREADY scaled by
    # timestep_scale_multiplier, exactly as _prepare_timestep passes them.
    ts = rand_input("brick.adaln.timesteps", (BATCH * VIDEO_TOKENS,), 0.25, 0.5) * float(
        ARCH["timestep_scale_multiplier"]
    )
    modulation, embedded = model.adaln_single(ts.flatten(), hidden_dtype=torch.float32)
    emit_f32(out, "kLtx2AdalnTimesteps", tensor(ts))
    emit_f32(out, "kLtx2AdalnModulation", tensor(modulation))
    emit_f32(out, "kLtx2AdalnEmbedded", tensor(embedded))

    # Attention with PER-HEAD GATING (attention.py:576-579). Self-attention shape,
    # no RoPE and no mask, so this brick isolates the gate and its ORDER.
    x = rand_input("brick.attn.x", (BATCH, VIDEO_TOKENS, dim), 0.5)
    attn_out = model.transformer_blocks[0].attn1(x)
    emit_f32(out, "kLtx2AttnGatedInput", tensor(x))
    emit_f32(out, "kLtx2AttnGatedOutput", tensor(attn_out))

    # The ASYMMETRIC cross-modal pair: Q from the video stream, K/V from the audio
    # stream, distinct token counts, both cross RoPE tables applied.
    from ltx_core.model.transformer.rope import LTXRopeType, precompute_freqs_cis  # noqa: PLC0415

    cross_max = max(
        ARCH["positional_embedding_max_pos"][0], ARCH["audio_positional_embedding_max_pos"][0]
    )
    vpe = precompute_freqs_cis(
        video_positions()[:, 0:1, :],
        dim=ARCH["audio_cross_attention_dim"],
        out_dtype=torch.float32,
        theta=ARCH["positional_embedding_theta"],
        max_pos=[cross_max],
        use_middle_indices_grid=True,
        num_attention_heads=ARCH["num_attention_heads"],
        rope_type=LTXRopeType.SPLIT,
    )
    ape = precompute_freqs_cis(
        audio_positions()[:, 0:1, :],
        dim=ARCH["audio_cross_attention_dim"],
        out_dtype=torch.float32,
        theta=ARCH["positional_embedding_theta"],
        max_pos=[cross_max],
        use_middle_indices_grid=True,
        num_attention_heads=ARCH["audio_num_attention_heads"],
        rope_type=LTXRopeType.SPLIT,
    )
    a2v_q = rand_input("brick.a2v.q", (BATCH, VIDEO_TOKENS, dim), 0.5)
    a2v_kv = rand_input("brick.a2v.kv", (BATCH, AUDIO_TOKENS, adim), 0.5)
    a2v_out = model.transformer_blocks[0].audio_to_video_attn(
        a2v_q, context=a2v_kv, pe=vpe, k_pe=ape
    )
    emit_f32(out, "kLtx2A2vQuery", tensor(a2v_q))
    emit_f32(out, "kLtx2A2vContext", tensor(a2v_kv))
    emit_f32(out, "kLtx2A2vOutput", tensor(a2v_out))

    # The FFN BIAS ASYMMETRY: `ff` has none, `audio_ff` has one.
    ffx = rand_input("brick.ff.x", (BATCH, VIDEO_TOKENS, dim), 0.5)
    emit_f32(out, "kLtx2FfInput", tensor(ffx))
    emit_f32(out, "kLtx2FfOutput", tensor(model.transformer_blocks[0].ff(ffx)))
    affx = rand_input("brick.audio_ff.x", (BATCH, AUDIO_TOKENS, adim), 0.5)
    emit_f32(out, "kLtx2AudioFfInput", tensor(affx))
    emit_f32(out, "kLtx2AudioFfOutput", tensor(model.transformer_blocks[0].audio_ff(affx)))


def emit_forward(
    out, tag: str, rope_type_name: str, double_rope: bool, masked: bool,
    audio_enabled: bool = True, dense_self_mask: bool = False,
) -> None:
    out.write(
        f"// --- forward case {tag}: rope={rope_type_name} float64_freqs={double_rope} "
        f"masked={masked} audio_enabled={audio_enabled} dense_self_mask={dense_self_mask} ---\n"
    )
    model = build_model(rope_type_name, double_rope)
    video, audio = build_modalities(masked, audio_enabled, dense_self_mask)
    with torch.no_grad():
        vx, ax = model(video=video, audio=audio, perturbations=None)
    emit_f32(out, f"kLtx2Forward{tag}Video", tensor(vx))
    emit_f32(out, f"kLtx2Forward{tag}Audio", tensor(ax))
    return model


def emit_masks(out) -> None:
    out.write("// --- section 5: the prompt and self-attention masks the masked case runs ---\n")
    emit_i64(out, "kLtx2VideoContextMask", video_context_mask())
    emit_i64(out, "kLtx2AudioContextMask", audio_context_mask())
    emit_f32(out, "kLtx2VideoSelfMask", tensor(video_self_mask()))
    emit_f32(out, "kLtx2AudioSelfMask", tensor(audio_self_mask()))
    emit_f32(out, "kLtx2VideoSelfMaskDense", tensor(video_self_mask_dense()))
    emit_f32(out, "kLtx2AudioSelfMaskDense", tensor(audio_self_mask_dense()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path, help="path to a Lightricks/LTX-2 checkout")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.ltx2.expanduser()
    load_upstream(root)
    revision = upstream_revision(root)

    torch.set_grad_enabled(False)
    argv = "python3 " + " ".join(
        [os.path.relpath(sys.argv[0]), f"--ltx2 {root}", f"--out {args.out}"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        emit_header(out, argv, revision)
        emit_arch(out)
        reference = build_model("split", False)
        emit_manifest(out, reference)
        emit_rope(out, reference)
        emit_bricks(out, reference)
        emit_masks(out)
        out.write("// --- section 4: the full DiT forward ---\n")
        emit_forward(out, "Split", "split", False, False)
        emit_forward(out, "Interleaved", "interleaved", False, False)
        emit_forward(out, "Double", "split", True, False)
        emit_forward(out, "Masked", "split", False, True)
        # Modality.enabled=False on the audio stream: the audio blocks are skipped
        # entirely (transformer.py:266) while the audio->video cross attention still
        # reads the audio state (:268), and the audio output head still runs over the
        # untouched patchified latent (model.py:527-536).
        emit_forward(out, "AudioOff", "split", False, False, audio_enabled=False)
        # The DENSE (B, T, T) self-attention mask (transformer_args.py:212-215).
        # Every query carries its OWN row of key strengths, so this case — and
        # only this case — separates a kernel that indexes the bias by query from
        # one that reads bias row 0 for every query.
        emit_forward(out, "DenseMask", "split", False, True, dense_self_mask=True)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
