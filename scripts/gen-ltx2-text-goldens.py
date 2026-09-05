#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_text_goldens.inc — the LTX-2.5 TEXT CONDITIONING oracle.

LTX-2.5 does NOT condition on a text encoder's last hidden state. It takes EVERY
Gemma-4 hidden state — the embedding output plus all 48 decoder outputs, 49 in
total — normalizes them, concatenates ACROSS THE LAYER AXIS, and feeds the result
to two caption projections (4096-wide video, 2048-wide audio). Measured on
`vonkaiser/LTX-2.5-FP8-NVFP4`'s `gemma4-12b-with-proj-nvfp4-torchao.safetensors`:
`text_embedding_projection.video_aggregate_embed.weight` is U8 [4096, 94080], and
NVFP4 packs TWO values per byte, so the real in_features is 188160 = 3840 x 49.

Getting the normalization variant, the mask handling, the reduction axes or the
LAYER ORDER wrong produces conditioning that is finite, correctly shaped and
WRONG — it renders a plausible video for the wrong prompt. So each of those is
gated on its own here, against upstream IMPORTED BY PATH and EXECUTED at reduced
dimensions on CPU. Both sides rebuild every weight and every input from one
deterministic FNV-1a + splitmix64 stream keyed by the parameter's own NAME, so no
weight byte is checked in and the weight CONTRACT is itself part of the gate.

This is MiniMax-H3's method (scripts/gen-minimax-h3-goldens.py) as applied by this
campaign's L2 (scripts/gen-ltx2-goldens.py) and L4 (scripts/gen-ltx2-vae-goldens.py);
the PRNG, `param_spec` and emission helpers are deliberately byte-identical to L2's
so the three generators stay diffable by eye.

Upstream sources (Lightricks/LTX-2, packages/ltx-core/src/ltx_core/):
  text_encoders/gemma/feature_extractor.py          -> both norm variants + both extractors
  text_encoders/gemma/encoders/encoder_configurator.py:163-209 -> the VARIANT SELECTION
  text_encoders/gemma/embeddings_processor.py:16-95 -> additive mask + right-pad ordering
  text_encoders/gemma/encoders/base_encoder.py:49-71 -> which hidden states, and their order
  text_encoders/gemma/gemma_assets.py:104-142       -> the EMBEDDED tokenizer/sidecar tensors
  text_encoders/gemma/feature_extractor.py:28,41,61 -> the EPSILONS, measured by probe
  text_encoders/gemma/feature_extractor.py:92,94,122 -> the two `.to(dtype)` casts that
      make the BF16 arm (section 7) two different dtypes rather than one

Usage:
    python3 scripts/gen-ltx2-text-goldens.py \
        --ltx2 ~/_git/LTX-2 \
        --out tests/vllm/models/ltx2_text_goldens.inc

Needs torch + numpy + einops (CPU only). No checkpoint and no gated download.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Deterministic weight/input stream, mirrored bit-for-bit by the C++ suite
# (tests/vllm/models/test_ltx2_text_encoder.cpp :: Ltx2Rand). Identical to
# scripts/gen-ltx2-goldens.py's: a per-tensor FNV-1a seed plus a splitmix64
# counter makes every tensor independent of fill ORDER, so the two sides cannot
# silently drift by reordering their parameter construction.
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
        unit = (u >> 11) * (2.0**-53)
        out[i] = unit * 2.0 - 1.0
    return out


def make_param(name: str, shape, scale: float, offset: float = 0.0) -> torch.Tensor:
    count = int(np.prod(shape)) if len(shape) else 1
    values = ltx2_rand(name, count) * scale + offset
    return torch.from_numpy(values.astype(np.float32)).reshape(tuple(shape))


def param_spec(name: str) -> tuple[float, float]:
    """(scale, offset) for a parameter, keyed ONLY by its name — L2's rule verbatim."""
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
# Reduced-dimension arch. Every ratio the port branches on is preserved:
#
#   * the video projection is WIDER than the Gemma hidden state (4096 > 3840) and
#     the audio one is NARROWER (2048 < 3840), so `_rescale_norm`'s sqrt factor is
#     > 1 on one arm and < 1 on the other — a sign error survives one arm alone;
#   * num_layers is hidden_layers + 1, the "+1 for the embedding layer" that makes
#     the flat width 49 x hidden rather than 48 x hidden;
#   * the two projections share ONE normalized input but have DIFFERENT widths and
#     both carry a bias (V2), while V1's single projection carries NONE.
#
# Only the magnitudes shrink.
# ---------------------------------------------------------------------------

GEMMA_HIDDEN = 6  # gemma_text_config.hidden_size            (real 3840)
GEMMA_HIDDEN_LAYERS = 3  # gemma_text_config.num_hidden_layers     (real 48)
NUM_LAYERS = GEMMA_HIDDEN_LAYERS + 1  # encoder_configurator.py:182 (real 49)
FLAT_DIM = GEMMA_HIDDEN * NUM_LAYERS  # (real 188160)

VIDEO_HEADS = 4
VIDEO_HEAD_DIM = 2
VIDEO_INNER = VIDEO_HEADS * VIDEO_HEAD_DIM  # (real 4096 = 32 x 128)
AUDIO_HEADS = 2
AUDIO_HEAD_DIM = 2
AUDIO_INNER = AUDIO_HEADS * AUDIO_HEAD_DIM  # (real 2048 = 32 x 64)

BATCH = 2
SEQ = 5

# encoder_configurator.py:163-168 — the EXACT V2 marker set. A V1 checkpoint has
# NONE of these keys; a partial or value-drifted set is NotImplementedError.
V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def v2_transformer_config() -> dict:
    return {
        **V2_EXPECTED_CONFIG,
        "num_attention_heads": VIDEO_HEADS,
        "attention_head_dim": VIDEO_HEAD_DIM,
        "audio_num_attention_heads": AUDIO_HEADS,
        "audio_attention_head_dim": AUDIO_HEAD_DIM,
    }


def v1_transformer_config() -> dict:
    """A pre-2.5 checkpoint: none of the V2 marker keys (encoder_configurator.py:185-188)."""
    return {"num_attention_heads": VIDEO_HEADS, "attention_head_dim": VIDEO_HEAD_DIM}


def gemma_text_config() -> object:
    """The two fields `_create_feature_extractor` reads (encoder_configurator.py:176-182)."""
    return types.SimpleNamespace(
        hidden_size=GEMMA_HIDDEN, num_hidden_layers=GEMMA_HIDDEN_LAYERS
    )


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


def hidden_state(layer: int) -> torch.Tensor:
    """One Gemma hidden state, [batch, seq, hidden]. Named by LAYER INDEX, so the
    C++ side reproducing them in a different order changes every golden."""
    return make_param(f"input.hidden.{layer}", (BATCH, SEQ, GEMMA_HIDDEN), 0.5)


def hidden_states() -> list[torch.Tensor]:
    return [hidden_state(i) for i in range(NUM_LAYERS)]


# Padding-side agnosticism is a documented upstream property
# (feature_extractor.py:18-19), so BOTH layouts are gated. The tokenizer LEFT-pads
# (base_encoder.py:235, PaddingSide.LEFT), the connector wants RIGHT-padding
# (embeddings_processor.py:82-84), and the extractor sits between them.
def mask_left() -> torch.Tensor:
    return torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.int64)


def mask_right() -> torch.Tensor:
    return torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.int64)


MASK_CASES = (("Left", mask_left), ("Right", mask_right))


# ---------------------------------------------------------------------------
# Emission helpers (identical to scripts/gen-ltx2-goldens.py's)
# ---------------------------------------------------------------------------


def emit_header(out, argv: str, revision: str) -> None:
    out.write(
        "// GENERATED by scripts/gen-ltx2-text-goldens.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// LTX-2.5 TEXT CONDITIONING goldens produced by IMPORTING and EXECUTING the\n"
        "// upstream Lightricks LTX-2 modules\n"
        "// (packages/ltx-core/src/ltx_core/text_encoders/gemma/) at reduced dimensions\n"
        "// with the deterministic Ltx2Rand stream.\n"
        f"// Upstream revision: {revision}\n"
        "// Regenerate with:\n"
        f"//   {argv}\n"
        "//\n"
        "// See .agents/specs/ltx-2-5.md sections 0 and 7: the shipped text tower is a\n"
        "// 12B Gemma-4 and there is no vLLM-Omni native 2.5 path, so the MATH is gated\n"
        "// exactly here and no weight byte is checked in.\n"
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


def emit_u16(out, name: str, values) -> None:
    """Raw BF16 BIT PATTERNS, not decimal values.

    A bf16 golden emitted as a decimal float and re-narrowed on the C++ side
    would gate the port against its own rounding, not against upstream's: the
    two narrowings could disagree and the comparison would still pass. Emitting
    the 16 bits upstream actually produced makes every bf16 case an EXACT
    integer comparison instead, so `torch.bfloat16` -> `uint16_t` is the only
    conversion in the loop and nothing rounds twice.
    """
    t = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    flat = t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).reshape(-1).tolist()
    out.write(f"inline constexpr uint16_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(f"0x{int(v):04x}" for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_i64(out, name: str, values) -> None:
    flat = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    for i in range(0, len(flat), 12):
        out.write("    " + ", ".join(str(v) for v in flat[i : i + 12]) + ",\n")
    out.write("};\n\n")


def emit_scalar(out, name: str, value) -> None:
    out.write(f"inline constexpr int64_t {name} = {int(value)};\n")


def emit_f64_scalar(out, name: str, value) -> None:
    out.write(f"inline constexpr double {name} = {_cxx_float(float(value), 17)};\n")


def tensor(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().numpy()


# ---------------------------------------------------------------------------
# Golden sections
# ---------------------------------------------------------------------------


def emit_arch(out) -> None:
    out.write("// --- section 0: the reduced architecture, mirrored by the C++ suite ---\n")
    emit_scalar(out, "kLtxTeGemmaHidden", GEMMA_HIDDEN)
    emit_scalar(out, "kLtxTeGemmaHiddenLayers", GEMMA_HIDDEN_LAYERS)
    emit_scalar(out, "kLtxTeNumLayers", NUM_LAYERS)
    emit_scalar(out, "kLtxTeFlatDim", FLAT_DIM)
    emit_scalar(out, "kLtxTeVideoHeads", VIDEO_HEADS)
    emit_scalar(out, "kLtxTeVideoHeadDim", VIDEO_HEAD_DIM)
    emit_scalar(out, "kLtxTeVideoInner", VIDEO_INNER)
    emit_scalar(out, "kLtxTeAudioHeads", AUDIO_HEADS)
    emit_scalar(out, "kLtxTeAudioHeadDim", AUDIO_HEAD_DIM)
    emit_scalar(out, "kLtxTeAudioInner", AUDIO_INNER)
    emit_scalar(out, "kLtxTeBatch", BATCH)
    emit_scalar(out, "kLtxTeSeq", SEQ)
    out.write("\n")


def emit_inputs(out) -> None:
    """The hidden states, the STACK the extractor actually consumes, and the masks.

    `torch.stack(hidden_states, dim=-1)` (feature_extractor.py:120) produces
    [B, T, D, L] — LAYER IS THE LAST AXIS — and the later `.reshape(B, T, D*L)`
    therefore interleaves as `d * L + l`, NOT as layer-major blocks of hidden.
    Emitting the stack itself makes that ordering a golden of its own, so a port
    that concatenates layer-major fails HERE rather than silently conditioning on
    a permuted feature vector.
    """
    out.write("// --- section 1: hidden states, their [B,T,D,L] stack, and the masks ---\n")
    states = hidden_states()
    for i, h in enumerate(states):
        emit_f32(out, f"kLtxTeHiddenLayer{i}", tensor(h))
    emit_f32(out, "kLtxTeStacked", tensor(torch.stack(states, dim=-1)))
    for tag, fn in MASK_CASES:
        emit_i64(out, f"kLtxTeMask{tag}", fn())


def emit_norms(out) -> None:
    """Both normalization variants, on both padding layouts."""
    from ltx_core.text_encoders.gemma.feature_extractor import (  # noqa: PLC0415
        _norm_and_concat_padded_batch,
        _rescale_norm,
        norm_and_concat_per_token_rms,
    )

    out.write("// --- section 2: the two normalization variants (feature_extractor.py:12-64) ---\n")
    stacked = torch.stack(hidden_states(), dim=-1)
    for tag, fn in MASK_CASES:
        mask = fn()
        emit_f32(out, f"kLtxTeNormV1{tag}", tensor(_norm_and_concat_padded_batch(stacked, mask)))
        emit_f32(out, f"kLtxTeNormV2{tag}", tensor(norm_and_concat_per_token_rms(stacked, mask)))

    # _rescale_norm (feature_extractor.py:67-69): x * sqrt(target_dim / source_dim),
    # applied SEPARATELY per projection with that projection's OWN out_features.
    out.write("// _rescale_norm factors: sqrt(out_features / gemma_hidden_size)\n")
    probe = torch.ones(1, dtype=torch.float32)
    emit_f64_scalar(
        out, "kLtxTeRescaleVideo", float(_rescale_norm(probe, VIDEO_INNER, GEMMA_HIDDEN)[0])
    )
    emit_f64_scalar(
        out, "kLtxTeRescaleAudio", float(_rescale_norm(probe, AUDIO_INNER, GEMMA_HIDDEN)[0])
    )
    out.write("\n")


def build_v1_extractor():
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (  # noqa: PLC0415
        _create_feature_extractor,
    )

    fx = _create_feature_extractor(v1_transformer_config(), gemma_text_config())
    fill_parameters(fx)
    fx.eval()
    return fx


def build_v2_extractor():
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (  # noqa: PLC0415
        _create_feature_extractor,
    )

    fx = _create_feature_extractor(v2_transformer_config(), gemma_text_config())
    fill_parameters(fx)
    fx.eval()
    return fx


def emit_manifest(out, tag: str, module: torch.nn.Module) -> None:
    """The upstream parameter LIST is the layout contract; gate it verbatim."""
    names, ranks, dims = [], [], []
    for name, param in module.named_parameters():
        names.append(name)
        ranks.append(len(param.shape))
        dims.extend(int(d) for d in param.shape)
    out.write(f"inline constexpr const char* kLtxTe{tag}ParamNames[] = {{\n")
    for name in names:
        out.write(f'    "{name}",\n')
    out.write("};\n\n")
    emit_i64(out, f"kLtxTe{tag}ParamRanks", ranks)
    emit_i64(out, f"kLtxTe{tag}ParamDims", dims)
    emit_scalar(out, f"kLtxTe{tag}ParamCount", len(names))
    out.write("\n")


def emit_selection(out, v1, v2) -> None:
    """The VARIANT SELECTION itself, resolved by upstream rather than by us.

    encoder_configurator.py:171-209 picks V1 when NONE of the four V2 marker keys
    are present and V2 when ALL FOUR are present with their exact expected values,
    and raises NotImplementedError otherwise. The class it returned and the
    projection shapes it built are the goldens; the C++ suite additionally asserts
    its own selector REFUSES the two drift cases.
    """
    from ltx_core.text_encoders.gemma.feature_extractor import (  # noqa: PLC0415
        FeatureExtractorV1,
        FeatureExtractorV2,
    )

    out.write("// --- section 3: variant selection + the weight contract ---\n")
    out.write(f"// upstream V1 -> {type(v1).__name__}, V2 -> {type(v2).__name__}\n")
    emit_scalar(out, "kLtxTeSelectedV1IsV2", int(isinstance(v1, FeatureExtractorV2)))
    emit_scalar(out, "kLtxTeSelectedV2IsV2", int(isinstance(v2, FeatureExtractorV2)))
    emit_scalar(out, "kLtxTeV1IsAv", int(v1.is_av))
    emit_scalar(out, "kLtxTeV1AggregateIn", v1.aggregate_embed.in_features)
    emit_scalar(out, "kLtxTeV1AggregateOut", v1.aggregate_embed.out_features)
    emit_scalar(out, "kLtxTeV1AggregateHasBias", int(v1.aggregate_embed.bias is not None))
    emit_scalar(out, "kLtxTeV2EmbeddingDim", v2.embedding_dim)
    emit_scalar(out, "kLtxTeV2VideoIn", v2.video_aggregate_embed.in_features)
    emit_scalar(out, "kLtxTeV2VideoOut", v2.video_aggregate_embed.out_features)
    emit_scalar(out, "kLtxTeV2AudioIn", v2.audio_aggregate_embed.in_features)
    emit_scalar(out, "kLtxTeV2AudioOut", v2.audio_aggregate_embed.out_features)
    emit_scalar(out, "kLtxTeV2VideoHasBias", int(v2.video_aggregate_embed.bias is not None))
    emit_scalar(out, "kLtxTeV2AudioHasBias", int(v2.audio_aggregate_embed.bias is not None))
    out.write("\n")
    emit_manifest(out, "V1", v1)
    emit_manifest(out, "V2", v2)


def emit_extractors(out, v1, v2) -> None:
    """The two extractors' full forwards, on both padding layouts."""
    out.write("// --- section 4: FeatureExtractorV1 / V2 forwards (feature_extractor.py:77-129) ---\n")
    states = hidden_states()
    for tag, fn in MASK_CASES:
        mask = fn()
        with torch.no_grad():
            v1_video, v1_audio = v1(states, mask)
            v2_video, v2_audio = v2(states, mask)
        emit_f32(out, f"kLtxTeV1Video{tag}", tensor(v1_video))
        # V1 is constructed with is_av=True (encoder_configurator.py:188), so its
        # audio arm is the SAME tensor as its video arm — not a second projection.
        emit_f32(out, f"kLtxTeV1Audio{tag}", tensor(v1_audio))
        emit_f32(out, f"kLtxTeV2Video{tag}", tensor(v2_video))
        emit_f32(out, f"kLtxTeV2Audio{tag}", tensor(v2_audio))


def emit_conditioning(out, v2) -> None:
    """The encoder -> conditioning path: additive mask, right-pad ordering, binary mask.

    embeddings_processor.py:16-95. `create_embeddings` normalizes the padding
    layout to RIGHT-padded before handing features to the connectors, using a sort
    index computed ONCE from the mask and reused for the audio arm. Everything up
    to the connector call is gated here; `Embeddings1DConnector` itself is built on
    the DiT's Attention/FeedForward/RoPE (embeddings_connector.py:4-11) and is
    recorded as OWED for after phase L2 lands.
    """
    from ltx_core.text_encoders.gemma.embeddings_processor import (  # noqa: PLC0415
        _apply_right_pad_order,
        _compute_right_pad_order,
        _to_binary_mask,
        convert_to_additive_mask,
    )

    out.write("// --- section 5: additive mask + right-pad ordering (embeddings_processor.py) ---\n")
    states = hidden_states()
    for tag, fn in MASK_CASES:
        mask = fn()
        with torch.no_grad():
            video, audio = v2(states, mask)
            additive = convert_to_additive_mask(mask, video.dtype)
            sort_idx, reordered_mask = _compute_right_pad_order(additive)
            video_r = _apply_right_pad_order(video, sort_idx)
            audio_r = _apply_right_pad_order(audio, sort_idx)
            binary = _to_binary_mask(reordered_mask[:, 0, 0, :], (BATCH, SEQ))
        emit_f32(out, f"kLtxTeAdditiveMask{tag}", tensor(additive))
        emit_i64(out, f"kLtxTeSortIdx{tag}", sort_idx)
        emit_f32(out, f"kLtxTeReorderedMask{tag}", tensor(reordered_mask))
        emit_f32(out, f"kLtxTeReorderedVideo{tag}", tensor(video_r))
        emit_f32(out, f"kLtxTeReorderedAudio{tag}", tensor(audio_r))
        # `_to_binary_mask` is applied to the CONNECTOR's returned mask, which is
        # `zeros_like(additive)` when learnable registers are on (2.5 has them,
        # embeddings_connector.py:152) and the additive mask itself when they are
        # off (:191). Its predicate is `< 1e-6`, and BOTH candidate inputs satisfy
        # it everywhere — 0.0 for a valid position and -finfo.max for a pad are
        # both below 1e-6 — so the mask `EmbeddingsProcessor` hands the DiT is
        # ALL ONES either way. That is upstream's behaviour, measured rather than
        # assumed, and it is gated on both inputs so a port cannot "fix" it.
        emit_i64(out, f"kLtxTeBinaryMask{tag}", binary.reshape(BATCH, SEQ))
        binary_zeros = _to_binary_mask(torch.zeros_like(reordered_mask)[:, 0, 0, :], (BATCH, SEQ))
        emit_i64(out, f"kLtxTeBinaryMaskFromRegisters{tag}", binary_zeros.reshape(BATCH, SEQ))


def emit_epsilons(out) -> None:
    """The two normalization EPSILONS, MEASURED out of upstream rather than restated.

    Both constants are invisible to the RANDOM goldens above, and that is a
    property of the algorithm, not of this fixture:

      * `range_ + eps` (feature_extractor.py:41) only matters when a whole
        (batch, layer) slice is CONSTANT over its valid positions, so `range_`
        collapses to 0. Random inputs never do that.
      * `denom + eps` (feature_extractor.py:34-35) only matters when
        `sequence_lengths == 0` — but its DTYPE matters far more widely, and is
        observable at the output. `sequence_lengths * d` is an int64 tensor, and
        `int64 + python float` promotes to the DEFAULT dtype, so upstream adds
        the epsilon in float32: `18 + 1e-6` is 18.000001907348633, not the
        18.000001 a float64 add gives. That is one f32 ULP in `mean`, and the
        RANGE epsilon amplifies it by 8/eps whenever `range_` collapses. On a
        constant 0.5 stack under `mask_right` it moves row 0 from 0.476837158 to
        0.238418579 — 23842x this suite's kTol. The claim this docstring used to
        make, that the epsilon is "unobservable at the output for any input", was
        false; it is unobservable only on an all-pad row.
      * `variance + 1e-6` (feature_extractor.py:61) only matters when a token's
        whole hidden slice is zero.

    So they are gated two ways. The VALUE is recovered from upstream numerically,
    by probes whose algebra inverts the epsilon exactly — no source parsing, no
    restating our own constant back to ourselves. And the degenerate inputs
    themselves are run through upstream and emitted as FULL OUTPUT TENSORS, so
    the C++ side compares against upstream's measured VALUES.

    Emitting values rather than a property is the whole point of this section. An
    earlier revision ran each degenerate input through upstream, had the output
    tensor in hand, and reduced it to one `isfinite` boolean. Both a float32 and a
    float64 mean denominator are finite, so that gate could not see the dtype
    defect above, and it sat green under it.

    Upstream has ONE `eps = 1e-6` (feature_extractor.py:28) used at BOTH :35 and
    :41; our port has one `kLtx2TextNormV1Eps` used at both. Measuring it through
    the range denominator therefore pins the same constant the mean denominator
    uses.
    """
    from ltx_core.text_encoders.gemma.feature_extractor import (  # noqa: PLC0415
        _norm_and_concat_padded_batch,
        norm_and_concat_per_token_rms,
    )

    out.write("// --- section 6: the epsilons, MEASURED from upstream by probe ---\n")

    # V1 range epsilon. [b=1, t=2, d=1, l=1] with two valid tokens whose values
    # differ by exactly r, so range_ == r. The two outputs are
    #   y_i = 8 * (x_i - mean) / (r + eps)
    # and their DIFFERENCE cancels `mean` exactly:
    #   y_0 - y_1 = 8 * r / (r + eps)   =>   eps = 8 * r / (y_0 - y_1) - r.
    # r is a power of two so it is exact in f32, and the difference removes the
    # mean, which is the only place the OTHER use of eps enters.
    r = 2.0**-10
    probe = torch.tensor([[[[r]], [[0.0]]]], dtype=torch.float32)
    ones = torch.ones(1, 2, dtype=torch.int64)
    y = _norm_and_concat_padded_batch(probe, ones).reshape(-1).tolist()
    v1_eps = 8.0 * r / (float(y[0]) - float(y[1])) - r

    # V2 epsilon. [1, 1, 1, 1] with value v: variance == v**2 and the output is
    #   y = v * rsqrt(v**2 + eps)   =>   eps = (v / y)**2 - v**2.
    v = 2.0**-10
    probe2 = torch.tensor([[[[v]]]], dtype=torch.float32)
    y2 = float(norm_and_concat_per_token_rms(probe2, torch.ones(1, 1, dtype=torch.int64))[0, 0, 0])
    v2_eps = (v / y2) ** 2 - v**2

    emit_f64_scalar(out, "kLtxTeNormV1EpsUpstream", v1_eps)
    emit_f64_scalar(out, "kLtxTeNormV2EpsUpstream", v2_eps)

    # The degenerate inputs, run through UPSTREAM and emitted as the FULL OUTPUT
    # TENSOR. `_cxx_float` refuses a non-finite literal, so each array carries the
    # old "still finite" property AND the values it used to discard; the finiteness
    # scalars are kept so nothing that was asserted before stops being asserted.
    #
    # 1. A CONSTANT stack: every (batch, layer) slice has range_ == 0, so `range_ +
    #    eps` is the entire denominator and it multiplies the mean's rounding by
    #    8/eps = 8e6. This is the input that falsifies a float64 mean denominator:
    #    row 0 has 3 valid tokens (denom 18) and reads 0.476837158 upstream against
    #    0.238418579 from a float64 add, while row 0 of the ONE-VALID-TOKEN case and
    #    every position of row 1 (denom 24) agree either way. A gate needs all
    #    three, because two of them cannot see the defect.
    const_stack = torch.full((BATCH, SEQ, GEMMA_HIDDEN, NUM_LAYERS), 0.5, dtype=torch.float32)
    const_out = _norm_and_concat_padded_batch(const_stack, mask_right())
    emit_scalar(out, "kLtxTeNormV1ConstantSliceFinite", int(bool(torch.isfinite(const_out).all())))
    emit_f32(out, "kLtxTeNormV1ConstantSlice", tensor(const_out))

    # 2. The same constant stack with exactly ONE valid token in row 0, so
    #    sequence_lengths == 1 and denom == d. Emitted with its mask so the two
    #    sides cannot drift on which positions are valid.
    one_token_mask = torch.tensor(
        [[1] + [0] * (SEQ - 1), [1] * (SEQ - 1) + [0]], dtype=torch.int64
    )
    one_token_out = _norm_and_concat_padded_batch(const_stack, one_token_mask)
    emit_i64(out, "kLtxTeNormV1OneValidTokenMask", one_token_mask)
    emit_f32(out, "kLtxTeNormV1OneValidToken", tensor(one_token_out))

    # 3. A range of exactly one f32 ULP. `nextafter(0.5)` in one position makes
    #    range_ == 2**-24, which is the same order as eps itself, so the two
    #    epsilons interact instead of one dominating.
    near_stack = const_stack.clone()
    near_stack[0, 0, 0, 0] = torch.nextafter(torch.tensor(0.5), torch.tensor(1.0))
    near_out = _norm_and_concat_padded_batch(near_stack, mask_right())
    emit_f32(out, "kLtxTeNormV1NearConstant", tensor(near_out))

    # 4. A batch row with NO valid token at all: sequence_lengths == 0, so denom == 0
    #    and range_ == -inf. This is the ONE case in which `denom + eps` really is
    #    unobservable, because :44-45 zeroes every position of that row. The array
    #    is what proves it, rather than the two booleans asserting it.
    zero_len_mask = torch.tensor([[0] * SEQ, [1] * SEQ], dtype=torch.int64)
    zero_len_out = _norm_and_concat_padded_batch(torch.stack(hidden_states(), dim=-1), zero_len_mask)
    emit_scalar(out, "kLtxTeNormV1ZeroLenFinite", int(bool(torch.isfinite(zero_len_out).all())))
    emit_scalar(out, "kLtxTeNormV1ZeroLenRowIsZero", int(bool((zero_len_out[0] == 0.0).all())))
    emit_f32(out, "kLtxTeNormV1ZeroLen", tensor(zero_len_out))

    # 5. A token whose whole hidden slice is zero: variance == 0.
    zero_var = torch.stack(hidden_states(), dim=-1).clone()
    zero_var[0, 0, :, :] = 0.0
    zero_var_out = norm_and_concat_per_token_rms(zero_var, mask_right())
    emit_scalar(out, "kLtxTeNormV2ZeroVarianceFinite", int(bool(torch.isfinite(zero_var_out).all())))
    emit_f32(out, "kLtxTeNormV2ZeroVariance", tensor(zero_var_out))
    out.write("\n")


# ---------------------------------------------------------------------------
# The BF16 arm — A24 wave 1 (LTX25-A24-TEXT-TOWER-BF16)
# ---------------------------------------------------------------------------


class _StableRsqrt:
    """Run upstream with `torch.rsqrt` replaced by the CORRECTLY ROUNDED one.

    `torch.rsqrt` on bf16 is not a function of its input — the vectorized body
    and the scalar tail round differently, so the same value gives 0x4065 in a
    length-1 tensor and 0x4066 in a length-1000 one (measured; 0x4066 is the
    correctly rounded answer). A port cannot be bit-equal to a kernel that is not
    bit-equal to itself, and chasing one would fit this machine's SIMD width.

    So each bf16 golden below is emitted TWICE. The module's own output is the
    oracle and is compared loosely. This variant — upstream's SAME module, with
    exactly one non-deterministic kernel replaced by the CORRECTLY ROUNDED value,
    `bf16(1.0 / sqrt(f32(x)))` — is what the port is held to BIT-EXACTLY, so
    nothing about our arithmetic is left on a tolerance. That is deliberately not
    "the value the vectorized path computes": the two agree on all but 6 of the
    32639 finite positive bf16 values, so the phrasings are not interchangeable.

    Nothing else is patched, and `torch.rsqrt` is restored on the way out.
    """

    def __enter__(self):
        self._orig = torch.rsqrt

        def correctly_rounded(x, *args, **kwargs):
            if x.dtype in (torch.bfloat16, torch.float16):
                return (1.0 / torch.sqrt(x.to(torch.float32))).to(x.dtype)
            return self._orig(x, *args, **kwargs)

        torch.rsqrt = correctly_rounded
        return self

    def __exit__(self, *exc):
        torch.rsqrt = self._orig
        return False


def build_bf16(builder):
    """The SAME module the f32 section builds, narrowed to upstream's own dtype.

    `distilled.py:109` resolves one pipeline dtype, `torch.bfloat16`, and hands
    it to `PromptEncoder` at `:111-113`; `base_encoder.py:41` carries the same
    default. So the bf16 arm is not a second architecture, it is this one at the
    width upstream constructs it in — which is why the module is built by the
    f32 builder and cast, rather than rebuilt.
    """
    fx = builder()
    return fx.to(torch.bfloat16)


def _bf16_eps_separating_bits() -> list[int]:
    """The bf16 inputs at which the V2 norm CAN see the epsilon's width.

    A constant slice of `GEMMA_HIDDEN` copies of one bf16 value `x` has variance
    `bf16(x*x)`, so the whole norm collapses to a function of `x` alone and the
    domain is sweepable exhaustively. This returns the finite non-negative bf16
    values at which `x * rsqrt(var + bf16(1e-6))` and `x * rsqrt(var + f32(1e-6))`
    disagree in bits — the only inputs on which a port that fails to narrow the
    epsilon is observably wrong. `rsqrt` here is the correctly rounded one, for
    the reason `_StableRsqrt` gives.
    """
    bits = torch.arange(0, 0x7F80, dtype=torch.int32).to(torch.uint16)
    x = bits.view(torch.bfloat16)
    var = torch.mean((x * x).unsqueeze(1).expand(-1, GEMMA_HIDDEN), dim=1)
    def normed(ve: torch.Tensor) -> torch.Tensor:
        inv = (1.0 / torch.sqrt(ve.to(torch.float32))).to(torch.bfloat16)
        return (x * inv).view(torch.uint16)
    narrowed = normed(var + 1e-6)
    f32_scalar = normed((var.to(torch.float32) + 1e-6).to(torch.bfloat16))
    return [int(b) for b in bits[narrowed != f32_scalar].tolist()]


def emit_bf16_arm(out) -> None:
    """Upstream executed in bfloat16, and the four facts that are NOT guessable.

    Each of these was measured out of the pinned oracle rather than read off it,
    and each is emitted so a port cannot restate its own assumption:

      1. `_norm_and_concat_padded_batch` returns FLOAT32 on a bf16 input. Its
         mean denominator is `(sequence_lengths * d) + eps`, an int64 tensor plus
         a Python float, which promotes to the default dtype; the f32 mean then
         carries through every later term. That is why `FeatureExtractorV1.forward`
         writes `self.aggregate_embed(normed.to(dtype))` at :94 — the cast is the
         only narrowing V1 does, and a port that runs V1's norm in bf16 is wrong
         everywhere rather than in the last ulp.
      2. `norm_and_concat_per_token_rms` IS bf16 throughout, and `encoded_text**2`
         materializes a bf16 tensor, so each square is rounded to 8 mantissa bits
         BEFORE the f32-accumulated mean sees it. Squaring in f32 is bit-wrong.
      3. A Python float paired with a bf16 tensor is narrowed FIRST, so the
         epsilon that actually reaches this arm is `bf16(1e-6)`, and the rescale
         factor is `bf16(f32(sqrt(target/source)))`.
      4. `convert_to_additive_mask` takes the FEATURES' dtype
         (embeddings_processor.py:117), so the pad value here is
         `-finfo(bfloat16).max`, not `-finfo(float32).max`.

    Every array is emitted as raw bf16 BIT PATTERNS (`emit_u16`) except the V1
    norm, which is emitted as f32 because fact 1 says it IS f32.
    """
    from ltx_core.text_encoders.gemma.embeddings_processor import (  # noqa: PLC0415
        _apply_right_pad_order,
        _compute_right_pad_order,
        convert_to_additive_mask,
    )
    from ltx_core.text_encoders.gemma.feature_extractor import (  # noqa: PLC0415
        _norm_and_concat_padded_batch,
        _rescale_norm,
        norm_and_concat_per_token_rms,
    )

    out.write("// --- section 7: the BF16 arm, upstream's own resolved dtype (A24) ---\n")

    states_bf16 = [h.to(torch.bfloat16) for h in hidden_states()]
    stacked_bf16 = torch.stack(states_bf16, dim=-1)
    emit_u16(out, "kLtxTeBf16Stacked", stacked_bf16)

    # FACT 1, as a gateable scalar and as the f32 array it implies.
    v1_left = _norm_and_concat_padded_batch(stacked_bf16, mask_left())
    emit_scalar(out, "kLtxTeBf16NormV1IsF32", int(v1_left.dtype == torch.float32))
    emit_scalar(
        out,
        "kLtxTeBf16NormV2IsBf16",
        int(norm_and_concat_per_token_rms(stacked_bf16, mask_left()).dtype == torch.bfloat16),
    )

    for tag, fn in MASK_CASES:
        mask = fn()
        emit_f32(out, f"kLtxTeBf16NormV1{tag}", tensor(_norm_and_concat_padded_batch(stacked_bf16, mask)))
        emit_u16(out, f"kLtxTeBf16NormV2{tag}", norm_and_concat_per_token_rms(stacked_bf16, mask))
        with _StableRsqrt():
            emit_u16(
                out,
                f"kLtxTeBf16NormV2{tag}Stable",
                norm_and_concat_per_token_rms(stacked_bf16, mask),
            )

    # FACT 3, and the reason it is gated by BITS and by a degenerate input rather
    # than by the algebraic probe the f32 section uses.
    #
    # MEASURED: that probe returns 0.0 at bf16. It inverts
    # `y = v * rsqrt(v**2 + eps)` for eps, and bf16 carries 8 mantissa bits, so
    # for any v whose square is large against 1e-6 the epsilon does not survive
    # `variance + eps` at all — `bf16(f32(0.015625) + f32(1e-6))` IS 0.015625.
    # Emitting the probe's answer would therefore have published `eps == 0`, which
    # is true of the arithmetic and false of the port's obligation.
    #
    # So the epsilon is held two ways that DO discriminate at this width. Its
    # BITS, which is the value upstream actually adds and the value the port has
    # to add. And the inputs on which it is the only thing between the port and a
    # division by zero, run through upstream and emitted as full output tensors.
    emit_u16(out, "kLtxTeBf16NormEpsBits", torch.tensor([1e-6], dtype=torch.bfloat16))

    # A token whose whole hidden slice is zero: variance == 0, and the epsilon is
    # the entire denominator. The f32 section's case 5, at upstream's own dtype.
    zero_var = stacked_bf16.clone()
    zero_var[0, 0, :, :] = 0.0
    zero_var_out = norm_and_concat_per_token_rms(zero_var, mask_right())
    emit_scalar(
        out, "kLtxTeBf16NormV2ZeroVarianceFinite", int(bool(torch.isfinite(zero_var_out).all()))
    )
    emit_u16(out, "kLtxTeBf16NormV2ZeroVariance", zero_var_out)
    with _StableRsqrt():
        emit_u16(
            out,
            "kLtxTeBf16NormV2ZeroVarianceStable",
            norm_and_concat_per_token_rms(zero_var, mask_right()),
        )

    # A variance of the SAME ORDER as the epsilon, which is the only regime in
    # which a wrong-but-nonzero epsilon is visible at bf16. 2**-10 squares to
    # 9.5e-07 against the epsilon's 1e-06.
    tiny = torch.full_like(stacked_bf16, 2.0**-10)
    tiny_out = norm_and_concat_per_token_rms(tiny, mask_right())
    emit_u16(out, "kLtxTeBf16NormV2TinyVariance", tiny_out)
    # What the SAME input reads with the epsilon dropped and with it 100x too
    # large. Emitted so the C++ side can assert this input DISCRIMINATES rather
    # than assume it does — if these ever stop differing from the golden above,
    # the case that uses them reds here instead of passing a wrong port.
    tiny_f = tiny.to(torch.float32)
    var_t = torch.mean(tiny**2, dim=2, keepdim=True)
    for label, eps in (("NoEps", 0.0), ("Eps1e4", 1e-4)):
        alt = tiny * torch.rsqrt(var_t + eps)
        alt = alt.reshape(BATCH, SEQ, GEMMA_HIDDEN * NUM_LAYERS)
        alt = torch.where(mask_right().bool().unsqueeze(-1), alt, torch.zeros_like(alt))
        emit_u16(out, f"kLtxTeBf16NormV2TinyVariance{label}", alt)
    del tiny_f

    # AND THE PROBE THAT SEPARATES THE TWO EPSILON WIDTHS, which the tiny-variance
    # input above does NOT. `2**-10` discriminates against a DROPPED epsilon and
    # against a 100x one, and it reads the same under `bf16(1e-6)` and under the
    # un-narrowed `1e-6` — so the arm's own constant could be widened to the f32
    # literal and every case above would stay green. The port's obligation is to
    # add the value upstream adds, and this is the input on which that is visible.
    #
    # MEASURED at the pin over the whole bf16 domain: exactly 140 of the 32640
    # finite non-negative bf16 values, all inside [0x384a, 0x3b25], make
    # `x * rsqrt(var + bf16(1e-6))` differ from `x * rsqrt(var + f32(1e-6))` for a
    # constant-D slice. Forty of them go in below, one per (b, t, layer) slice, so
    # every slice's variance is a separating one and the two hypotheses part on
    # every unmasked output rather than on a lucky few.
    #
    # Each slice is CONSTANT over d, so upstream's two rsqrt paths see one value
    # per slice and this case is held BIT-EXACT, exactly as the tiny-variance and
    # zero-variance cases are.
    eps_probe_bits = _bf16_eps_separating_bits()[: BATCH * SEQ * NUM_LAYERS]
    if len(eps_probe_bits) < BATCH * SEQ * NUM_LAYERS:
        raise SystemExit(
            "fewer than B*T*L bf16 values now separate bf16(1e-6) from f32(1e-6) "
            "in the V2 norm; the epsilon's width is no longer gateable by this "
            "probe, so pick a new one rather than emitting a golden that cannot fail"
        )
    eps_probe = (
        torch.tensor(eps_probe_bits, dtype=torch.uint16)
        .view(torch.bfloat16)
        .reshape(BATCH, SEQ, 1, NUM_LAYERS)
        .expand(BATCH, SEQ, GEMMA_HIDDEN, NUM_LAYERS)
        .contiguous()
    )
    eps_probe_out = norm_and_concat_per_token_rms(eps_probe, mask_right())
    emit_u16(out, "kLtxTeBf16NormV2EpsProbeIn", eps_probe)
    emit_u16(out, "kLtxTeBf16NormV2EpsProbe", eps_probe_out)
    # THE REJECTED HYPOTHESIS on that same input: upstream's own body with the one
    # change of adding the epsilon at f32 instead of letting torch narrow it.
    # Emitted so the C++ side asserts the probe discriminates instead of assuming
    # it does, and so a port that widens the constant reds against a MEASURED
    # alternative rather than against an argument.
    eps_var = torch.mean(eps_probe**2, dim=2, keepdim=True)
    eps_rejected = eps_probe * torch.rsqrt(
        (eps_var.to(torch.float32) + 1e-6).to(torch.bfloat16)
    )
    eps_rejected = eps_rejected.reshape(BATCH, SEQ, GEMMA_HIDDEN * NUM_LAYERS)
    eps_rejected = torch.where(
        mask_right().bool().unsqueeze(-1), eps_rejected, torch.zeros_like(eps_rejected)
    )
    if torch.equal(eps_probe_out.view(torch.uint16), eps_rejected.view(torch.uint16)):
        raise SystemExit(
            "the V2 epsilon probe no longer separates the bf16-narrowed epsilon "
            "from the f32 one; pick new probe values rather than emitting a "
            "golden that cannot fail"
        )
    emit_u16(out, "kLtxTeBf16NormV2EpsProbeF32Scalar", eps_rejected)

    # THE RESCALE, AND WHY IT IS PROBED ON A NON-TRIVIAL VECTOR.
    #
    # "A Python float paired with a bf16 tensor is narrowed first" holds for `add`
    # and NOT for `mul`. MEASURED exhaustively over the bf16 domain at the pin:
    # `t + 1e-6` equals the bf16-narrowed-scalar form at ALL 32640 points and the
    # f32-scalar form at all but 387, while `t * sqrt(8/6)` equals the F32-scalar
    # form at all 32639 and the bf16-narrowed one at all but 7881. So the epsilon
    # really is narrowed and the rescale factor is NOT: two scalar ops, two
    # answers, and no single rule covers both.
    #
    # `_rescale_norm(ones, ...)` CANNOT SEE THAT, and an earlier revision of this
    # section emitted exactly that probe: `1.0 * f` narrows to `bf16(f)` under
    # both hypotheses, so its golden agreed with a port that was wrong on nearly a
    # quarter of every other value. The vector below carries products that
    # separate them.
    # The last four are values at which the two hypotheses are KNOWN to differ,
    # found by sweeping the bf16 exponent range against each factor — two for the
    # video factor sqrt(8/6) and two for the audio factor sqrt(4/6), because a
    # value that separates one does not necessarily separate the other. The
    # generator asserts below that both arms really are discriminated; a probe
    # that silently stopped separating them would make this whole section a
    # tautology.
    probe = torch.tensor(
        [
            0.3, -1.7, 0.04, 2.5,
            0.008056640625, -0.0084228515625,   # video factor
            0.00872802734375, -0.011474609375,  # audio factor
        ],
        dtype=torch.bfloat16,
    )
    emit_u16(out, "kLtxTeBf16RescaleProbeIn", probe)
    emit_u16(out, "kLtxTeBf16RescaleVideoOut", _rescale_norm(probe, VIDEO_INNER, GEMMA_HIDDEN))
    emit_u16(out, "kLtxTeBf16RescaleAudioOut", _rescale_norm(probe, AUDIO_INNER, GEMMA_HIDDEN))
    # The REJECTED hypothesis on the same input, emitted so the C++ side asserts
    # this probe discriminates instead of assuming it does.
    for label, dim in (("Video", VIDEO_INNER), ("Audio", AUDIO_INNER)):
        narrowed = torch.tensor(math.sqrt(dim / GEMMA_HIDDEN), dtype=torch.bfloat16)
        rejected = (probe.to(torch.float32) * narrowed.to(torch.float32)).to(torch.bfloat16)
        accepted = _rescale_norm(probe, dim, GEMMA_HIDDEN)
        if torch.equal(accepted.view(torch.uint16), rejected.view(torch.uint16)):
            raise SystemExit(
                f"the {label} rescale probe no longer separates the f32-scalar "
                "hypothesis from the bf16-narrowed one; pick new probe values "
                "rather than emitting a golden that cannot fail"
            )
        emit_u16(out, f"kLtxTeBf16Rescale{label}OutNarrowedScalar", rejected)

    # FACT 5, WHICH IS WHY THE V2 LANE IS GATED AT ONE BF16 ULP AND NOT BIT-EXACT.
    #
    # `torch.rsqrt` ON BF16 IS NOT A FUNCTION OF ITS INPUT. The same value gives a
    # different bf16 answer depending on the tensor's LENGTH, because the
    # vectorized body and the scalar tail round differently. MEASURED at the pin:
    # `torch.rsqrt(t)` for t = 0.07763671875 is 0x4065 in a length-1 tensor and
    # 0x4066 in a length-1000 one. 0x4066 is the correctly rounded answer
    # (`bf16(1/sqrt(f32))`); 0x4065 is one bf16 ulp below it.
    #
    # So there is no implementation a port can choose that is bit-equal to
    # upstream everywhere, and chasing one would be fitting to this machine's SIMD
    # width. The port takes the correctly rounded single rounding, and the V2 lane
    # is held to ONE ULP of upstream's own module output — the format's own
    # resolution, and exactly the distance upstream disagrees with itself by.
    #
    # The count below is what makes that allowance a measurement rather than
    # slack: it is the number of variance entries, out of B*T*L, at which torch's
    # two kernel paths disagree on this fixture. The C++ side asserts it is
    # non-zero, so if a future torch makes rsqrt path-independent the bound stops
    # being justified HERE rather than silently staying loose.
    ve = torch.mean(stacked_bf16**2, dim=2, keepdim=True) + 1e-6
    rsqrt_module = torch.rsqrt(ve)
    rsqrt_correct = (1.0 / torch.sqrt(ve.to(torch.float32))).to(torch.bfloat16)
    disagree = int((rsqrt_module.view(torch.uint16) != rsqrt_correct.view(torch.uint16)).sum())
    emit_scalar(out, "kLtxTeBf16RsqrtSelfDisagree", disagree)
    emit_scalar(out, "kLtxTeBf16RsqrtEntries", int(ve.numel()))
    emit_u16(out, "kLtxTeBf16RsqrtModule", rsqrt_module)
    emit_u16(out, "kLtxTeBf16RsqrtCorrectlyRounded", rsqrt_correct)

    # FACT 2, as a bit-level discriminator: the same variance computed with and
    # without the intermediate bf16 rounding of each square. If a future torch
    # stops materializing the square, these two stop differing and the C++ case
    # that asserts they differ reds HERE rather than passing a wrong port.
    var_upstream = torch.mean(stacked_bf16**2, dim=2, keepdim=True)
    var_f32_squares = torch.mean(stacked_bf16.to(torch.float32) ** 2, dim=2, keepdim=True).to(
        torch.bfloat16
    )
    emit_u16(out, "kLtxTeBf16VarianceBf16Squares", var_upstream)
    emit_u16(out, "kLtxTeBf16VarianceF32Squares", var_f32_squares)
    emit_scalar(
        out,
        "kLtxTeBf16SquaresRoundingIsObservable",
        int(not torch.equal(var_upstream, var_f32_squares)),
    )

    # The two extractors, and the conditioning hand-off, at bf16.
    v1 = build_bf16(build_v1_extractor)
    v2 = build_bf16(build_v2_extractor)
    with torch.no_grad():
        for tag, fn in MASK_CASES:
            mask = fn()
            v1_video, v1_audio = v1(states_bf16, mask)
            v2_video, v2_audio = v2(states_bf16, mask)
            emit_u16(out, f"kLtxTeBf16V1Video{tag}", v1_video)
            emit_u16(out, f"kLtxTeBf16V1Audio{tag}", v1_audio)
            emit_u16(out, f"kLtxTeBf16V2Video{tag}", v2_video)
            emit_u16(out, f"kLtxTeBf16V2Audio{tag}", v2_audio)
            with _StableRsqrt():
                sv, sa = v2(states_bf16, mask)
            emit_u16(out, f"kLtxTeBf16V2Video{tag}Stable", sv)
            emit_u16(out, f"kLtxTeBf16V2Audio{tag}Stable", sa)

            # FACT 4: the additive mask inherits the FEATURES' dtype.
            additive = convert_to_additive_mask(mask, v2_video.dtype)
            sort_idx, reordered_mask = _compute_right_pad_order(additive)
            emit_u16(out, f"kLtxTeBf16AdditiveMask{tag}", additive)
            # `_compute_right_pad_order` REBUILDS the mask with
            # `finfo(additive_mask.dtype).max` (:37), so the reordered one is bf16
            # too. THIS is what `Ltx2TextConditioning.additive_mask` holds; the
            # unreordered array above is what `Ltx2ConvertToAdditiveMask` returns,
            # and comparing the conditioning against the wrong one of the two was
            # a real defect in this suite's first draft.
            emit_u16(out, f"kLtxTeBf16ReorderedMask{tag}", reordered_mask)
            emit_i64(out, f"kLtxTeBf16SortIdx{tag}", sort_idx)
            emit_u16(
                out, f"kLtxTeBf16ReorderedVideo{tag}", _apply_right_pad_order(v2_video, sort_idx)
            )
            emit_u16(
                out, f"kLtxTeBf16ReorderedAudio{tag}", _apply_right_pad_order(v2_audio, sort_idx)
            )
            emit_u16(
                out, f"kLtxTeBf16ReorderedVideo{tag}Stable", _apply_right_pad_order(sv, sort_idx)
            )
            emit_u16(
                out, f"kLtxTeBf16ReorderedAudio{tag}Stable", _apply_right_pad_order(sa, sort_idx)
            )
    emit_scalar(
        out,
        "kLtxTeBf16AdditiveMaskDtypeIsBf16",
        int(convert_to_additive_mask(mask_left(), torch.bfloat16).dtype == torch.bfloat16),
    )
    out.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ltx2", required=True, type=Path, help="path to a Lightricks/LTX-2 checkout"
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.ltx2.expanduser()
    load_upstream(root)
    revision = upstream_revision(root)

    torch.set_grad_enabled(False)
    argv = "python3 " + " ".join(
        [os.path.relpath(sys.argv[0]), f"--ltx2 {root}", f"--out {args.out}"]
    )

    v1 = build_v1_extractor()
    v2 = build_v2_extractor()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        emit_header(out, argv, revision)
        emit_arch(out)
        emit_inputs(out)
        emit_norms(out)
        emit_selection(out, v1, v2)
        emit_extractors(out, v1, v2)
        emit_conditioning(out, v2)
        emit_epsilons(out)
        emit_bf16_arm(out)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
