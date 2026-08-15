#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_music3_ar_goldens.inc.

The AUTOREGRESSIVE half of MiniMax-Music3 (spec .agents/specs/minimax-music3.md
phases W2 + W3, issue #672): the prompt the checkpoint contract fixes, the
semantic stage's classifier-free-guidance logit pipeline, the learned 8-layer
condition mix, and the 4-layer RVQ depth decoder.

WHY THIS GENERATOR EXISTS. The committed full-scale goldens under
tests/parity/goldens/minimax_music3_oracle/ need the 28.5 GB checkpoint and are
BF16, so CI can neither run them nor separate an algebra defect from bf16
rounding. This generator runs upstream's OWN classes at REDUCED dimensions in
FLOAT32 with a name-seeded weight stream, which is the H3 pattern
(gen-minimax-h3-goldens.py) and what spec section 5 asks for: "the exact
correctness gate runs upstream at reduced dimensions on CPU". Nothing but shapes
and float values crosses into the .inc; no weight byte of the real checkpoint is
checked in.

The oracle is the pinned diffusers PR head:
  huggingface/diffusers#14456 @ c6da9936e4bda83107943a16eb8682e9a37d8527
installed per tools/oracle/README.md. Run it with that venv's interpreter:

    ~/venvs/music3-oracle/bin/python scripts/gen-minimax-music3-ar-goldens.py \\
        --out tests/vllm/models/minimax_music3_ar_goldens.inc

WHAT IS AND IS NOT AN ORACLE HERE. The condition encoder and the depth decoder
are IMPORTED and EXECUTED (`MiniMaxMusic3ConditionEncoder`,
`MiniMaxMusic3RVQDepthDecoder`), as are the prompt helpers (`_clean_caption`,
`_normalize_lyrics`, the template constants) and `_sample_top_k`. The semantic
stage's guidance block is NOT a function upstream exposes -- it is inline in
`MiniMaxMusic3SemanticGenerationStep.__call__` (encoders.py:318-334) -- so it is
reproduced here with torch ops line for line against that anchor, and the anchor
is cited in the emitted header so a reviewer can diff it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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


def music3_rand(name: str, count: int) -> np.ndarray:
    """Identical to gen-minimax-h3-goldens.py :: h3_rand and the C++ Music3Rand."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def param(name: str, shape, scale: float = 0.5, offset: float = 0.0) -> torch.Tensor:
    count = int(np.prod(shape)) if shape else 1
    raw = music3_rand(name, count) * scale + offset
    return torch.from_numpy(raw.astype(np.float32).reshape(shape))


# ---------------------------------------------------------------------------
# Reduced dimensions. Small enough to print, large enough that every axis the
# real config exercises is >1 and no two of them are equal (a transposed or
# swapped axis has to be visible).
# ---------------------------------------------------------------------------

COND = dict(
    condition_hidden_dim=6,
    num_condition_layers=3,
    out_dim=4,
    input_sampling_rate=24000,
    input_hop_length=960,
    output_sampling_rate=44100,
    output_hop_length=512,
)
COND_FRAMES = 5  # -> latent_length 17: an UPSAMPLING nearest interpolation.

# The same module with the rates reversed, so the nearest interpolation is a
# DOWNSAMPLE (7 frames -> 2). Upsample-only coverage cannot see a scale computed
# as output/input instead of input/output.
COND_DOWN = dict(COND, input_sampling_rate=44100, output_sampling_rate=24000,
                 input_hop_length=512, output_hop_length=960)
COND_DOWN_FRAMES = 7

DEPTH = dict(
    hidden_size=8,
    num_layers=2,
    num_attention_heads=2,
    intermediate_size=12,
    audio_vocab_size=5,
    num_codebooks=4,
    max_position_embeddings=6,
)

# The prompt fixtures. The first pair is the one the committed full-scale oracle
# capture used (tests/parity/goldens/minimax_music3_oracle/manifest.json), so the
# assembled string is checkable against a real run. The rest exercise every
# rewrite `_clean_caption` / `_normalize_lyrics` performs.
PROMPT_CASES = [
    (
        "oracle_capture",
        "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate. Vocals: "
        "soft female lead, close and breathy. Arrangement: fingerpicked guitar "
        "and soft piano.",
        "[verse]\nMorning light filtering through the pine\n",
    ),
    (
        "markdown_and_tags",
        "## Genre\n- **dream pop**\n  * *hazy*\n<|mood dark and warm|><|solo|>\n"
        "---\nbullet• x\n\n\nend    stop",
        "[Verse] dropped words\n[Chorus][Bridge]\nkeep this line\n"
        "tail [outro] ^ after caret",
    ),
]


def emit_floats(out, name: str, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    out.append(f"inline constexpr float {name}[] = {{")
    row: list[str] = []
    for value in flat:
        # `%g` spells these "inf"/"nan", which is not C++. They are load-bearing
        # here: a masked logit IS -inf and `_sample_top_k`'s nan_to_num exists
        # for the NaN, so they are emitted rather than substituted away.
        if np.isnan(value):
            row.append("NAN")
        elif np.isposinf(value):
            row.append("INFINITY")
        elif np.isneginf(value):
            row.append("-INFINITY")
        else:
            text = f"{float(value):.9g}"
            # "0" and "3" are integer literals; the `f` suffix needs a float.
            if "." not in text and "e" not in text and "E" not in text:
                text += ".0"
            row.append(text + "f")
        if len(row) == 6:
            out.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        out.append("    " + ", ".join(row) + ",")
    out.append("};")
    out.append("")


def cpp_string(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        from diffusers.models.condition_embedders.condition_embedder_minimax_music3 import (
            MiniMaxMusic3ConditionEncoder,
        )
        from diffusers.models.transformers.minimax_music3_rvq_depth_decoder import (
            MiniMaxMusic3RVQDepthDecoder,
        )
        from diffusers.modular_pipelines.minimax_music3 import encoders as up
    except ImportError as exc:  # pragma: no cover - environment guard
        print(
            "This generator needs the pinned diffusers PR head "
            "(c6da9936e4bda83107943a16eb8682e9a37d8527); see tools/oracle/README.md.\n"
            f"import failed: {exc}",
            file=sys.stderr,
        )
        return 2

    torch.set_grad_enabled(False)
    out: list[str] = []
    out.append(
        "// GENERATED by scripts/gen-minimax-music3-ar-goldens.py --- DO NOT EDIT BY HAND."
    )
    out.append("//")
    out.append(
        "// MiniMax-Music3 AUTOREGRESSIVE-half goldens (#672, spec phases W2 + W3),"
    )
    out.append(
        "// produced by EXECUTING upstream's own classes at reduced dimensions in"
    )
    out.append(
        "// float32. Oracle pin: huggingface/diffusers#14456 @ c6da9936e4bda83107943a"
    )
    out.append("// 16eb8682e9a37d8527. Weights come from the name-seeded Music3Rand")
    out.append("// stream, so no weight byte of the 28.5 GB checkpoint is checked in.")
    out.append("//")
    out.append("// Upstream anchors:")
    out.append("//   condition mix   condition_embedder_minimax_music3.py:48-76")
    out.append("//   depth decoder   minimax_music3_rvq_depth_decoder.py:127-142, :75-88")
    out.append("//   depth sequence  encoders.py:118-142  (_generate_depth_codes)")
    out.append("//   frame feedback  encoders.py:106-115  (_embed_audio_frame)")
    out.append("//   prompt          encoders.py:54-91, :207-218")
    out.append("//   semantic CFG    encoders.py:318-334  (inline in __call__)")
    out.append("//   top-k filter    encoders.py:94-103   (_sample_top_k)")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cmath>  // INFINITY / NAN appear as golden VALUES below")
    out.append("#include <cstdint>")
    out.append("")
    out.append("namespace vllm_test {")
    out.append("")

    # -- upstream constants, re-emitted from the module so a rename is caught ---
    out.append("// Re-emitted FROM the upstream module, not transcribed.")
    for const in (
        "_AUDIO_END_TOKEN_ID",
        "_AUDIO_CFG_TOKEN_ID",
        "_AUDIO_CODE_OFFSET",
        "_SEMANTIC_VOCAB_SIZE",
        "_MAX_PROMPT_TOKENS",
        "_MAX_AUDIO_FRAMES",
        "_AR_CFG_TOP_K",
        "_AR_SAMPLING_TOP_K",
    ):
        cpp = "kMusic3" + "".join(p.capitalize() for p in const.strip("_").split("_"))
        out.append(f"inline constexpr int64_t {cpp} = {getattr(up, const)};")
    out.append(f"inline constexpr double kMusic3ArCfgScale = {up._AR_CFG_SCALE!r};")
    out.append("")

    # ---------------- prompt assembly ----------------------------------------
    out.append("struct Music3PromptGolden {")
    out.append("  const char* name;")
    out.append("  const char* prompt;")
    out.append("  const char* lyrics;")
    out.append("  const char* clean_caption;")
    out.append("  const char* normalized_lyrics;")
    out.append("  const char* assembled;")
    out.append("};")
    out.append("")
    out.append("inline constexpr Music3PromptGolden kMusic3PromptGoldens[] = {")
    for name, prompt, lyrics in PROMPT_CASES:
        clean = up._clean_caption(prompt)
        norm = up._normalize_lyrics(lyrics)
        assembled = (
            f"{up._IM_START}{up._CAPTION_START}{clean}{up._CAPTION_END}"
            f"{up._LYRICS_START}{norm}{up._LYRICS_END}{up._IM_END}{up._AUDIO_START}"
        )
        out.append("    {")
        out.append(f"        {cpp_string(name)},")
        out.append(f"        {cpp_string(prompt)},")
        out.append(f"        {cpp_string(lyrics)},")
        out.append(f"        {cpp_string(clean)},")
        out.append(f"        {cpp_string(norm)},")
        out.append(f"        {cpp_string(assembled)},")
        out.append("    },")
    out.append("};")
    out.append(
        f"inline constexpr int64_t kMusic3PromptGoldenCount = {len(PROMPT_CASES)};"
    )
    out.append("")

    # ---------------- unconditional id rewrite (encoders.py:216-217) ----------
    ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17]], dtype=torch.int64)
    uncond = ids.clone()
    uncond[:, 1:-2] = up._AUDIO_CFG_TOKEN_ID
    out.append("// encoders.py:216-217 - every token but the first and the two")
    out.append("// trailing structure tokens becomes the audio-CFG token.")
    out.append(
        "inline constexpr int32_t kMusic3UncondIdsIn[] = {"
        + ", ".join(str(int(v)) for v in ids[0])
        + "};"
    )
    out.append(
        "inline constexpr int32_t kMusic3UncondIdsOut[] = {"
        + ", ".join(str(int(v)) for v in uncond[0])
        + "};"
    )
    out.append(
        f"inline constexpr int64_t kMusic3UncondIdsCount = {ids.shape[1]};"
    )
    out.append("")

    # ---------------- semantic CFG + top-k pipeline ---------------------------
    # encoders.py:318-334, reproduced op for op. VOCAB is tiny but the masked
    # region straddles the audio-code window so the mask itself is exercised.
    vocab = 40
    offset, semantic = 8, 6  # stand-ins for _AUDIO_CODE_OFFSET/_SEMANTIC_VOCAB_SIZE
    end_id = 30
    cfg_top_k = 4
    vocab_mask = torch.ones(vocab, dtype=torch.bool)
    vocab_mask[offset : offset + semantic] = False
    vocab_mask[end_id] = False
    logits = param("music3.semantic.logits", (2, vocab), scale=4.0).double().float()
    masked = logits.masked_fill(vocab_mask, -float("inf"))
    conditional, unconditional = masked[0:1], masked[1:2]
    guided = unconditional + (conditional - unconditional) * up._AR_CFG_SCALE
    threshold = torch.topk(conditional, cfg_top_k, dim=-1).values[..., -1, None]
    guided = guided.masked_fill(conditional < threshold, -float("inf"))
    guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
    out.append(
        f"inline constexpr int64_t kMusic3SemanticVocab = {vocab};\n"
        f"inline constexpr int64_t kMusic3SemanticOffset = {offset};\n"
        f"inline constexpr int64_t kMusic3SemanticWindow = {semantic};\n"
        f"inline constexpr int64_t kMusic3SemanticEndId = {end_id};\n"
        f"inline constexpr int64_t kMusic3SemanticCfgTopK = {cfg_top_k};"
    )
    emit_floats(out, "kMusic3SemanticLogitsIn", logits.numpy())
    emit_floats(out, "kMusic3SemanticGuided", guided.numpy())
    out.append("// true == BLOCKED (encoders.py:318-320).")
    out.append(
        "inline constexpr bool kMusic3SemanticVocabMask[] = {"
        + ", ".join("true" if bool(v) else "false" for v in vocab_mask)
        + "};"
    )
    out.append("")

    # ---------------- _sample_top_k's deterministic half ----------------------
    # encoders.py:94-100: everything up to (not including) torch.multinomial.
    probe = param("music3.topk.logits", (1, 12), scale=3.0)
    probe[0, 3] = float("nan")
    probe[0, 7] = float("-inf")
    top_k = 5
    values = torch.nan_to_num(probe.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    thresh = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < thresh, -float("inf"))
    probs = torch.nan_to_num(torch.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    out.append(f"inline constexpr int64_t kMusic3TopKProbeN = {probe.shape[1]};")
    out.append(f"inline constexpr int64_t kMusic3TopKProbeK = {top_k};")
    out.append("// index 3 is NaN and index 7 is -inf on the way in.")
    emit_floats(out, "kMusic3TopKProbeIn", probe.numpy())
    emit_floats(out, "kMusic3TopKProbeProbs", probs.numpy())
    out.append("")

    # ---------------- condition mix ------------------------------------------
    for tag, cfg, frames in (
        ("", COND, COND_FRAMES),
        ("Down", COND_DOWN, COND_DOWN_FRAMES),
    ):
        module = MiniMaxMusic3ConditionEncoder(**cfg)
        state = {
            "layer_weight_logits": param(
                f"music3.cond{tag}.layer_weight_logits", (cfg["num_condition_layers"],)
            ),
            "layer_scale": param(f"music3.cond{tag}.layer_scale", (1,), 0.5, 1.0),
            "proj.weight": param(
                f"music3.cond{tag}.proj.weight",
                (cfg["out_dim"], cfg["condition_hidden_dim"], 3),
                0.3,
            ),
            "proj.bias": param(f"music3.cond{tag}.proj.bias", (cfg["out_dim"],), 0.2),
        }
        module.load_state_dict(state)
        module.eval()
        width = cfg["num_condition_layers"] * cfg["condition_hidden_dim"]
        hidden = param(f"music3.cond{tag}.hidden", (1, frames, width), 1.5)
        result = module(hidden)
        prefix = f"kMusic3Cond{tag}"
        out.append(f"inline constexpr int64_t {prefix}Layers = {cfg['num_condition_layers']};")
        out.append(f"inline constexpr int64_t {prefix}Hidden = {cfg['condition_hidden_dim']};")
        out.append(f"inline constexpr int64_t {prefix}OutDim = {cfg['out_dim']};")
        out.append(f"inline constexpr int64_t {prefix}Frames = {frames};")
        out.append(f"inline constexpr int64_t {prefix}LatentLength = {result.shape[1]};")
        out.append(
            f"inline constexpr int64_t {prefix}InputSamplingRate = {cfg['input_sampling_rate']};"
        )
        out.append(
            f"inline constexpr int64_t {prefix}InputHopLength = {cfg['input_hop_length']};"
        )
        out.append(
            f"inline constexpr int64_t {prefix}OutputSamplingRate = {cfg['output_sampling_rate']};"
        )
        out.append(
            f"inline constexpr int64_t {prefix}OutputHopLength = {cfg['output_hop_length']};"
        )
        emit_floats(out, f"{prefix}LayerWeightLogits", state["layer_weight_logits"].numpy())
        emit_floats(out, f"{prefix}LayerScale", state["layer_scale"].numpy())
        emit_floats(out, f"{prefix}ProjWeight", state["proj.weight"].numpy())
        emit_floats(out, f"{prefix}ProjBias", state["proj.bias"].numpy())
        emit_floats(out, f"{prefix}HiddenIn", hidden.numpy())
        emit_floats(out, f"{prefix}Out", result.numpy())

    # ---------------- depth decoder ------------------------------------------
    module = MiniMaxMusic3RVQDepthDecoder(**DEPTH)
    hidden_size = DEPTH["hidden_size"]
    inter = DEPTH["intermediate_size"]
    state = {
        "audio_embeddings.weight": param(
            "music3.depth.audio_embeddings",
            (DEPTH["audio_vocab_size"] * (DEPTH["num_codebooks"] - 1), hidden_size),
            0.7,
        ),
        "projection.weight": param(
            "music3.depth.projection", (hidden_size, hidden_size), 0.4
        ),
        "pos_embedding.weight": param(
            "music3.depth.pos_embedding",
            (DEPTH["max_position_embeddings"], hidden_size),
            0.3,
        ),
        "norm.weight": param("music3.depth.norm", (hidden_size,), 0.2, 1.0),
    }
    for layer in range(DEPTH["num_layers"]):
        base = f"layers.{layer}"
        state[f"{base}.input_layernorm.weight"] = param(
            f"music3.depth.{layer}.input_layernorm", (hidden_size,), 0.2, 1.0
        )
        state[f"{base}.post_attention_layernorm.weight"] = param(
            f"music3.depth.{layer}.post_attention_layernorm", (hidden_size,), 0.2, 1.0
        )
        for proj in ("to_q", "to_k", "to_v", "to_out"):
            state[f"{base}.attn.{proj}.weight"] = param(
                f"music3.depth.{layer}.attn.{proj}", (hidden_size, hidden_size), 0.4
            )
        state[f"{base}.gate_proj.weight"] = param(
            f"music3.depth.{layer}.gate_proj", (inter, hidden_size), 0.4
        )
        state[f"{base}.up_proj.weight"] = param(
            f"music3.depth.{layer}.up_proj", (inter, hidden_size), 0.4
        )
        state[f"{base}.down_proj.weight"] = param(
            f"music3.depth.{layer}.down_proj", (hidden_size, inter), 0.4
        )
    for head in range(DEPTH["num_codebooks"] - 1):
        state[f"audio_heads.{head}.weight"] = param(
            f"music3.depth.audio_heads.{head}",
            (DEPTH["audio_vocab_size"], hidden_size),
            0.4,
        )
    module.load_state_dict(state)
    module.eval()

    for key, tensor in state.items():
        cpp = "kMusic3Depth" + "".join(
            part.capitalize() for part in key.replace(".", "_").split("_")
        )
        emit_floats(out, cpp, tensor.numpy())
    out.append(f"inline constexpr int64_t kMusic3DepthHidden = {hidden_size};")
    out.append(f"inline constexpr int64_t kMusic3DepthLayers = {DEPTH['num_layers']};")
    out.append(f"inline constexpr int64_t kMusic3DepthHeads = {DEPTH['num_attention_heads']};")
    out.append(f"inline constexpr int64_t kMusic3DepthIntermediate = {inter};")
    out.append(
        f"inline constexpr int64_t kMusic3DepthAudioVocab = {DEPTH['audio_vocab_size']};"
    )
    out.append(
        f"inline constexpr int64_t kMusic3DepthCodebooks = {DEPTH['num_codebooks']};"
    )
    out.append(
        "inline constexpr int64_t kMusic3DepthMaxPositions = "
        f"{DEPTH['max_position_embeddings']};"
    )
    out.append("")

    # The depth sequence exactly as _generate_depth_codes assembles it
    # (encoders.py:125-141): projection(last_hidden), projection(semantic embed),
    # then projection(audio_embeddings(code + (index-1) * audio_vocab_size)).
    last_hidden = param("music3.depth.last_hidden", (1, hidden_size), 1.2)
    semantic_embed = param("music3.depth.semantic_embed", (1, hidden_size), 1.1)
    residual_codes = [2, 4]  # c1, c2 -> a 4-position sequence at num_codebooks=4
    sequence = [
        module.projection(last_hidden).unsqueeze(1),
        module.projection(semantic_embed).unsqueeze(1),
    ]
    for index, code in enumerate(residual_codes, start=1):
        embed = module.audio_embeddings(
            torch.tensor([code + (index - 1) * DEPTH["audio_vocab_size"]])
        )
        sequence.append(module.projection(embed).unsqueeze(1))
    inputs_embeds = torch.cat(sequence, dim=1)
    hidden_states = module(inputs_embeds)
    head_logits = torch.stack(
        [module.audio_heads[j](hidden_states[:, j + 1]) for j in range(len(residual_codes) + 1)],
        dim=1,
    )
    out.append(f"inline constexpr int64_t kMusic3DepthSeqLen = {inputs_embeds.shape[1]};")
    out.append(
        "inline constexpr int32_t kMusic3DepthResidualCodes[] = {"
        + ", ".join(str(c) for c in residual_codes)
        + "};"
    )
    emit_floats(out, "kMusic3DepthLastHidden", last_hidden.numpy())
    emit_floats(out, "kMusic3DepthSemanticEmbed", semantic_embed.numpy())
    emit_floats(out, "kMusic3DepthInputsEmbeds", inputs_embeds.numpy())
    emit_floats(out, "kMusic3DepthOut", hidden_states.numpy())
    for j in range(DEPTH["num_codebooks"] - 1):
        emit_floats(
            out,
            f"kMusic3DepthAudioHead{j}",
            module.audio_heads[j].weight.detach().numpy(),
        )
    emit_floats(out, "kMusic3DepthHeadLogits", head_logits.numpy())

    # The 16-position BOUNDARY (spec W3 "the depth decoder's 16-position window
    # exercised at its boundary"): a sequence exactly max_position_embeddings long.
    boundary = param(
        "music3.depth.boundary", (1, DEPTH["max_position_embeddings"], hidden_size), 0.9
    )
    boundary_out = module(boundary)
    emit_floats(out, "kMusic3DepthBoundaryIn", boundary.numpy())
    emit_floats(out, "kMusic3DepthBoundaryOut", boundary_out.numpy())
    out.append("")

    # ---------------- frame feedback embedding -------------------------------
    # encoders.py:106-115. `embed_tokens` belongs to the LANGUAGE MODEL, so the
    # reduced-dimension stand-in is a plain table of the same hidden width.
    frame_codes = [3, 1, 4, 2]  # semantic + c1..c3 at num_codebooks=4
    lm_row = param("music3.feedback.lm_row", (1, hidden_size), 1.3)
    offsets = torch.arange(DEPTH["num_codebooks"] - 1) * DEPTH["audio_vocab_size"]
    extra = module.audio_embeddings(
        torch.tensor([frame_codes[1:]]) + offsets.unsqueeze(0)
    ).sum(dim=1, keepdim=True)
    feedback = (lm_row.unsqueeze(1) + extra) * DEPTH["num_codebooks"] ** -0.5
    out.append(
        "inline constexpr int32_t kMusic3FeedbackCodes[] = {"
        + ", ".join(str(c) for c in frame_codes)
        + "};"
    )
    emit_floats(out, "kMusic3FeedbackLmRow", lm_row.numpy())
    emit_floats(out, "kMusic3FeedbackOut", feedback.numpy())
    out.append("")

    out.append("}  // namespace vllm_test")
    out.append("")
    args.out.write_text("\n".join(out))
    print(f"wrote {args.out} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
