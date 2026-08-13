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
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
