#!/usr/bin/env python3
"""Emit tests/vllm/models/minimax_music3_acoustic_goldens.inc.

The ACOUSTIC half of MiniMax-Music3 (spec .agents/specs/minimax-music3.md phases
W4 + W5, issue #672): the flow-matching DiT, its `FlowMatchEulerDiscreteScheduler`
with `invert_sigmas`, the classifier-free-guidance mix, the window bookkeeping the
denoise loop performs, and the DAC-style Flow-VAE vocoder.

WHY THIS GENERATOR EXISTS. The committed full-scale goldens under
tests/parity/goldens/minimax_music3_oracle/ need the 28.5 GB checkpoint, so CI
can neither run them nor separate an algebra defect from rounding. This generator
runs upstream's OWN classes at REDUCED dimensions in FLOAT32 with a name-seeded
weight stream, which is the pattern gen-minimax-music3-ar-goldens.py established
for the autoregressive half and what spec section 5 asks for: "the exact
correctness gate runs upstream at reduced dimensions on CPU". Nothing but shapes
and float values crosses into the .inc; no weight byte of the real checkpoint is
checked in.

The oracle is the pinned diffusers PR head:
  huggingface/diffusers#14456 @ c6da9936e4bda83107943a16eb8682e9a37d8527
installed per tools/oracle/README.md. Run it with that venv's interpreter:

    ~/venvs/music3-oracle/bin/python \\
        scripts/gen-minimax-music3-acoustic-goldens.py \\
        --out tests/vllm/models/minimax_music3_acoustic_goldens.inc

WHAT IS AND IS NOT AN ORACLE HERE. `MiniMaxMusic3Transformer1DModel`,
`MiniMaxMusic3Vocoder` and `FlowMatchEulerDiscreteScheduler` are IMPORTED and
EXECUTED, as are their submodules (the Fourier embedding, the partial rotary, the
snake activation, one vocoder block, one residual unit). `ClassifierFreeGuidance`
is imported and EXECUTED through its own `forward`. The denoise loop's window
bookkeeping is NOT a function upstream exposes -- it is inline in
`MiniMaxMusic3ChunkDenoiseInner.__call__` (denoise.py:207-234),
`MiniMaxMusic3ChunkUpdateStep.__call__` (:249-260) and
`MiniMaxMusic3VocoderDecodeStep.__call__` (decoders.py:83-89) -- so it is
reproduced here with torch ops line for line against those anchors, and the
anchors are cited in the emitted header so a reviewer can diff them.
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
    """Identical to gen-minimax-music3-ar-goldens.py :: music3_rand."""
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
# real config exercises is > 1 and no two of them are equal, so a transposed or
# swapped axis has to be visible.
# ---------------------------------------------------------------------------

# rotary_dim 4 of attention_head_dim 6 is the PARTIAL rotary the real config has
# (32 of 64): two head dims stay unrotated, and `inv_freq` has two distinct
# entries so a wrong `theta` cannot hide behind a constant 1.0.
DIT = dict(
    in_channels=5,
    condition_dim=10,
    num_layers=2,
    num_attention_heads=3,
    attention_head_dim=6,
    ff_inner_dim=7,
    rotary_dim=4,
    fourier_embedding_dim=8,
)
DIT_LENGTH = 4  # -> 5 transformer tokens once the timestep token is prepended.

# Two even upsampling ratios, DISTINCT, because `padding=ceil(stride/2)` only
# yields the exact `stride x` length for an even stride and every shipped ratio
# is even (8, 8, 4, 2). hop = 2 * 4 = 8.
VOCODER = dict(
    latent_channels=8,
    decoder_input_dim=6,
    decoder_hidden_dim=16,
    upsampling_ratios=(2, 4),
    sampling_rate=44100,
)
VOCODER_LENGTH = 3

# The shipped scheduler config (scheduler/scheduler_config.json, and the oracle
# manifest's `spec_facts.pipeline.scheduler_config`), plus three variants that
# each move ONE lever. A schedule gate that only ever sees the shipped values
# cannot tell `invert_sigmas` from a no-op at shift 1.0.
SCHEDULER_CASES = [
    ("shipped", dict(num_train_timesteps=1, shift=1.0, invert_sigmas=True), 4),
    ("shipped_30", dict(num_train_timesteps=1, shift=1.0, invert_sigmas=True), 30),
    ("no_invert", dict(num_train_timesteps=1, shift=1.0, invert_sigmas=False), 4),
    ("shift3", dict(num_train_timesteps=1, shift=3.0, invert_sigmas=True), 4),
    ("train1000", dict(num_train_timesteps=1000, shift=1.0, invert_sigmas=True), 4),
]

# before_denoise.py:67-70. `_CHUNK_FRAMES` 200, `_CHUNK_HOP` 100.
CHUNK_CASES = [1, 25, 199, 200, 201, 300, 301, 500, 901]


def emit_floats(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    out.append(f"inline constexpr float {name}[] = {{")
    row: list[str] = []
    for value in flat:
        if np.isnan(value):
            row.append("NAN")
        elif np.isposinf(value):
            row.append("INFINITY")
        elif np.isneginf(value):
            row.append("-INFINITY")
        else:
            text = f"{float(value):.9g}"
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


def emit_ints(out, name: str, values, ctype: str = "int64_t") -> None:
    flat = [int(v) for v in np.asarray(values).reshape(-1)]
    out.append(
        f"inline constexpr {ctype} {name}[] = {{" + ", ".join(str(v) for v in flat) + "};"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    try:
        from diffusers.guiders import ClassifierFreeGuidance
        from diffusers.models.autoencoders.minimax_music3_vocoder import (
            MiniMaxMusic3Snake1d,
            MiniMaxMusic3Vocoder,
            MiniMaxMusic3VocoderBlock,
            MiniMaxMusic3VocoderResidualUnit,
        )
        from diffusers.models.transformers.transformer_minimax_music3 import (
            MiniMaxMusic3FourierEmbedding,
            MiniMaxMusic3RotaryEmbedding,
            MiniMaxMusic3Transformer1DModel,
            _apply_partial_rotary_emb,
        )
        from diffusers.modular_pipelines.minimax_music3 import before_denoise as bd
        from diffusers.modular_pipelines.minimax_music3 import decoders as dec
        from diffusers.modular_pipelines.minimax_music3 import denoise as dn
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
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
        "// GENERATED by scripts/gen-minimax-music3-acoustic-goldens.py --- DO NOT EDIT BY HAND."
    )
    out.append("//")
    out.append(
        "// MiniMax-Music3 ACOUSTIC-half goldens (#672, spec phases W4 + W5),"
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
    out.append("//   DiT forward       transformer_minimax_music3.py:196-242")
    out.append("//   DiT block         transformer_minimax_music3.py:140-144")
    out.append("//   partial rotary    transformer_minimax_music3.py:42-71")
    out.append("//   fourier time      transformer_minimax_music3.py:30-39")
    out.append("//   scheduler         scheduling_flow_match_euler_discrete.py:283-381, :423-522")
    out.append("//   sigma ramp        denoise.py:154-156")
    out.append("//   CFG mix           guiders/classifier_free_guidance.py:114-127")
    out.append("//   overlap blend     denoise.py:207-212")
    out.append("//   window carry      denoise.py:249-260")
    out.append("//   chunk starts      before_denoise.py:63-73")
    out.append("//   vocoder forward   minimax_music3_vocoder.py:100-115")
    out.append("//   vocoder block     minimax_music3_vocoder.py:51-68")
    out.append("//   residual unit     minimax_music3_vocoder.py:37-48")
    out.append("//   snake             minimax_music3_vocoder.py:25-34")
    out.append("//   waveform crop     decoders.py:83-89")
    out.append("#pragma once")
    out.append("")
    out.append("#include <cmath>")
    out.append("#include <cstdint>")
    out.append("")
    out.append("namespace vllm_test {")
    out.append("")

    # -- upstream constants, re-emitted FROM the modules so a rename is caught --
    out.append("// Re-emitted FROM the upstream modules, not transcribed.")
    out.append(f"inline constexpr int64_t kMusic3ChunkFrames = {bd._CHUNK_FRAMES};")
    out.append(f"inline constexpr int64_t kMusic3ChunkHop = {bd._CHUNK_HOP};")
    out.append(
        f"inline constexpr int64_t kMusic3OverlapLatentLength = {dn._OVERLAP_LATENT_LENGTH};"
    )
    out.append(f"inline constexpr int64_t kMusic3CropLeftLatent = {dec._CROP_LEFT_LATENT};")
    out.append(f"inline constexpr int64_t kMusic3CropRightLatent = {dec._CROP_RIGHT_LATENT};")
    # The guidance scale is a ComponentSpec default rather than a module constant
    # (denoise.py:177-182), so it is read out of that spec rather than typed.
    guider_spec = next(
        spec
        for spec in dn.MiniMaxMusic3ChunkDenoiseInner().expected_components
        if spec.name == "guider"
    )
    guidance_scale = float(dict(guider_spec.config)["guidance_scale"])
    out.append(f"inline constexpr double kMusic3GuidanceScale = {guidance_scale!r};")
    # denoise.py:210 -- the blend's `1 - (1 - 1e-6) * t` coefficient.
    out.append("inline constexpr double kMusic3BlendEpsilon = 1e-6;")
    out.append("")

    # ---------------- scheduler ----------------------------------------------
    out.append("// denoise.py:154 -- np.linspace(1.0, 1.0/n, n), BEFORE any shift.")
    out.append("struct Music3ScheduleGolden {")
    out.append("  const char* name;")
    out.append("  int64_t num_train_timesteps;")
    out.append("  double shift;")
    out.append("  bool invert_sigmas;")
    out.append("  int64_t num_inference_steps;")
    out.append("  const float* ramp;      // [num_inference_steps]")
    out.append("  const float* timesteps; // [num_inference_steps]")
    out.append("  const float* sigmas;    // [num_inference_steps + 1]")
    out.append("};")
    out.append("")
    schedule_names = []
    for name, cfg, steps in SCHEDULER_CASES:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=cfg["num_train_timesteps"],
            shift=cfg["shift"],
            invert_sigmas=cfg["invert_sigmas"],
            use_dynamic_shifting=False,
            time_shift_type="exponential",
        )
        ramp = np.linspace(1.0, 1.0 / steps, steps)
        scheduler.set_timesteps(sigmas=ramp)
        tag = "".join(p.capitalize() for p in name.split("_"))
        emit_floats(out, f"kMusic3Sched{tag}Ramp", ramp.astype(np.float32))
        emit_floats(out, f"kMusic3Sched{tag}Timesteps", scheduler.timesteps.numpy())
        emit_floats(out, f"kMusic3Sched{tag}Sigmas", scheduler.sigmas.numpy())
        schedule_names.append((name, cfg, steps, tag))
    out.append("inline constexpr Music3ScheduleGolden kMusic3ScheduleGoldens[] = {")
    for name, cfg, steps, tag in schedule_names:
        out.append(
            f'    {{"{name}", {cfg["num_train_timesteps"]}, {cfg["shift"]!r}, '
            f'{"true" if cfg["invert_sigmas"] else "false"}, {steps}, '
            f"kMusic3Sched{tag}Ramp, kMusic3Sched{tag}Timesteps, kMusic3Sched{tag}Sigmas}},"
        )
    out.append("};")
    out.append(
        f"inline constexpr int64_t kMusic3ScheduleGoldenCount = {len(SCHEDULER_CASES)};"
    )
    out.append("")

    # One Euler step, executed through the scheduler's own `step`, at each index
    # of the shipped 4-step schedule -- so the dt the C++ picks is checked at the
    # boundary as well as in the middle.
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1, shift=1.0, invert_sigmas=True,
        use_dynamic_shifting=False, time_shift_type="exponential",
    )
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.25, 4))
    step_sample = param("music3.sched.sample", (1, 3, 4), 1.5)
    step_velocity = param("music3.sched.velocity", (1, 3, 4), 2.0)
    step_outs = []
    for index in range(4):
        scheduler._step_index = index
        step_outs.append(
            scheduler.step(step_velocity, scheduler.timesteps[index], step_sample,
                           return_dict=False)[0]
        )
    out.append("inline constexpr int64_t kMusic3StepChannels = 3;")
    out.append("inline constexpr int64_t kMusic3StepLength = 4;")
    emit_floats(out, "kMusic3StepSample", step_sample.numpy())
    emit_floats(out, "kMusic3StepVelocity", step_velocity.numpy())
    emit_floats(out, "kMusic3StepOut", torch.cat(step_outs, dim=0).numpy())
    out.append("")

    # ---------------- classifier-free guidance --------------------------------
    guider = ClassifierFreeGuidance(guidance_scale=guidance_scale)
    guider.set_state(step=0, num_inference_steps=4, timestep=torch.tensor(0.0))
    cfg_cond = param("music3.cfg.cond", (1, 3, 4), 2.0)
    cfg_uncond = param("music3.cfg.uncond", (1, 3, 4), 2.0)
    cfg_out = guider.forward(pred_cond=cfg_cond, pred_uncond=cfg_uncond).pred
    out.append(f"inline constexpr int64_t kMusic3CfgCount = {cfg_cond.numel()};")
    emit_floats(out, "kMusic3CfgCond", cfg_cond.numpy())
    emit_floats(out, "kMusic3CfgUncond", cfg_uncond.numpy())
    emit_floats(out, "kMusic3CfgOut", cfg_out.numpy())
    out.append(
        f"inline constexpr int64_t kMusic3CfgNumConditions = {guider.num_conditions};"
    )
    out.append("")

    # ---------------- window bookkeeping --------------------------------------
    out.append("struct Music3ChunkGolden {")
    out.append("  int64_t num_frames;")
    out.append("  int64_t count;")
    out.append("  const int64_t* starts;")
    out.append("};")
    out.append("")
    for frames in CHUNK_CASES:
        state = type("S", (), {})()
        state.frame_hiddens = torch.zeros(1, frames, 1)
        num_frames = state.frame_hiddens.shape[1]
        starts = (
            [0]
            if num_frames <= bd._CHUNK_FRAMES
            else list(range(0, num_frames - bd._CHUNK_HOP, bd._CHUNK_HOP))
        )
        emit_ints(out, f"kMusic3ChunkStarts{frames}", starts)
    out.append("inline constexpr Music3ChunkGolden kMusic3ChunkGoldens[] = {")
    for frames in CHUNK_CASES:
        starts = (
            [0]
            if frames <= bd._CHUNK_FRAMES
            else list(range(0, frames - bd._CHUNK_HOP, bd._CHUNK_HOP))
        )
        out.append(f"    {{{frames}, {len(starts)}, kMusic3ChunkStarts{frames}}},")
    out.append("};")
    out.append(f"inline constexpr int64_t kMusic3ChunkGoldenCount = {len(CHUNK_CASES)};")
    out.append("")

    # The overlap blend, denoise.py:210-212, at three flow times including both
    # ends. A blend that dropped the `(1 - 1e-6)` is invisible at t = 0 and t = 1.
    blend_ch, blend_len, blend_overlap = 2, 7, 3
    blend_latents = param("music3.blend.latents", (blend_ch, blend_len), 1.5)
    blend_noise = param("music3.blend.noise", (blend_ch, blend_overlap), 1.5)
    blend_prev = param("music3.blend.prev", (blend_ch, blend_overlap), 1.5)
    blend_times = [0.0, 0.25, 1.0]
    blended = []
    for t in blend_times:
        work = blend_latents.clone()
        time_value = torch.tensor(t, dtype=work.dtype)
        work[..., :blend_overlap] = (1.0 - (1.0 - 1e-6) * time_value) * blend_noise + (
            time_value * blend_prev
        )
        blended.append(work)
    out.append(f"inline constexpr int64_t kMusic3BlendChannels = {blend_ch};")
    out.append(f"inline constexpr int64_t kMusic3BlendLength = {blend_len};")
    out.append(f"inline constexpr int64_t kMusic3BlendOverlap = {blend_overlap};")
    out.append(f"inline constexpr int64_t kMusic3BlendTimeCount = {len(blend_times)};")
    emit_floats(out, "kMusic3BlendTimes", np.array(blend_times, dtype=np.float32))
    emit_floats(out, "kMusic3BlendLatents", blend_latents.numpy())
    emit_floats(out, "kMusic3BlendNoise", blend_noise.numpy())
    emit_floats(out, "kMusic3BlendPrev", blend_prev.numpy())
    emit_floats(out, "kMusic3BlendOut", torch.cat(blended, dim=0).numpy())
    out.append("")

    # The window carry, denoise.py:254-256, including the SHORT window where both
    # `max(0, ...)` clamps fire and the carry is empty.
    out.append("struct Music3CarryGolden {")
    out.append("  int64_t latent_length;")
    out.append("  int64_t overlap_start;")
    out.append("  int64_t overlap_end;")
    out.append("};")
    out.append("inline constexpr Music3CarryGolden kMusic3CarryGoldens[] = {")
    carry_lengths = [86, 172, 344, 345, 688, 1000]
    for length in carry_lengths:
        start = max(0, length - 2 * dn._OVERLAP_LATENT_LENGTH)
        end = max(start, length - dn._OVERLAP_LATENT_LENGTH)
        out.append(f"    {{{length}, {start}, {end}}},")
    out.append("};")
    out.append(
        f"inline constexpr int64_t kMusic3CarryGoldenCount = {len(carry_lengths)};"
    )
    out.append("")

    # The waveform crop, decoders.py:85-87.
    out.append("struct Music3CropGolden {")
    out.append("  int64_t chunk_index;")
    out.append("  int64_t num_chunks;")
    out.append("  int64_t waveform_length;")
    out.append("  int64_t left;")
    out.append("  int64_t right_exclusive;")
    out.append("};")
    out.append("inline constexpr Music3CropGolden kMusic3CropGoldens[] = {")
    hop = 512
    # 44032 is the committed capture's single window (86 latents). 352768 is a
    # FULL 200-frame window (689 latents x 512), which is the only size for
    # which the 258-latent right crop is even defined -- a shorter one would be
    # emptied by it, and upstream never produces one in a multi-window song.
    crop_cases = [(0, 1, 44032), (0, 3, 352768), (1, 3, 352768), (2, 3, 352768)]
    for chunk_index, num_chunks, wave_len in crop_cases:
        left = 0 if chunk_index == 0 else dec._CROP_LEFT_LATENT * hop
        right = 0 if chunk_index == num_chunks - 1 else dec._CROP_RIGHT_LATENT * hop
        out.append(
            f"    {{{chunk_index}, {num_chunks}, {wave_len}, {left}, {wave_len - right}}},"
        )
    out.append("};")
    out.append(f"inline constexpr int64_t kMusic3CropGoldenCount = {len(crop_cases)};")
    out.append(f"inline constexpr int64_t kMusic3CropHopLength = {hop};")
    out.append("")

    # ---------------- DiT sub-pieces ------------------------------------------
    out.append("// --- DiT reduced-dimension config ---")
    for key, value in DIT.items():
        cpp = "kMusic3Dit" + "".join(p.capitalize() for p in key.split("_"))
        out.append(f"inline constexpr int64_t {cpp} = {value};")
    inner_dim = DIT["num_attention_heads"] * DIT["attention_head_dim"]
    concat = 2 * DIT["in_channels"] + DIT["condition_dim"]
    out.append(f"inline constexpr int64_t kMusic3DitInnerDim = {inner_dim};")
    out.append(f"inline constexpr int64_t kMusic3DitConcatChannels = {concat};")
    out.append(f"inline constexpr int64_t kMusic3DitLength = {DIT_LENGTH};")
    out.append(f"inline constexpr int64_t kMusic3DitSeqLen = {DIT_LENGTH + 1};")
    out.append("")

    # The Fourier time embedding, on its own, at three flow times.
    fourier = MiniMaxMusic3FourierEmbedding(DIT["fourier_embedding_dim"])
    fourier_weight = param(
        "music3.dit.time_proj.weight", (DIT["fourier_embedding_dim"] // 2, 1), 1.5
    )
    fourier.weight.copy_(fourier_weight)
    fourier_times = torch.tensor([0.0, 0.25, 0.75], dtype=torch.float32)
    fourier_out = fourier(fourier_times)
    out.append(f"inline constexpr int64_t kMusic3FourierTimeCount = {fourier_times.numel()};")
    emit_floats(out, "kMusic3FourierTimes", fourier_times.numpy())
    emit_floats(out, "kMusic3FourierWeight", fourier_weight.numpy())
    emit_floats(out, "kMusic3FourierOut", fourier_out.numpy())
    out.append("")

    # The rotary tables and the PARTIAL application, on their own.
    rotary = MiniMaxMusic3RotaryEmbedding(DIT["rotary_dim"])
    cos, sin = rotary(DIT_LENGTH + 1, torch.device("cpu"))
    rot_in = param(
        "music3.dit.rotary.in",
        (1, DIT_LENGTH + 1, DIT["num_attention_heads"], DIT["attention_head_dim"]),
        1.5,
    )
    rot_out = _apply_partial_rotary_emb(rot_in, (cos, sin))
    out.append(f"inline constexpr double kMusic3RotaryTheta = {rotary.theta!r};")
    emit_floats(out, "kMusic3RotaryCos", cos.numpy())
    emit_floats(out, "kMusic3RotarySin", sin.numpy())
    emit_floats(out, "kMusic3RotaryIn", rot_in.numpy())
    emit_floats(out, "kMusic3RotaryOut", rot_out.numpy())
    out.append("")

    # The whole DiT, weights and all, plus its output.
    dit = MiniMaxMusic3Transformer1DModel(**DIT)
    state = {}
    for name, tensor in dit.state_dict().items():
        scale = 0.3 if tensor.dim() > 1 else 0.2
        state[name] = param(f"music3.dit.{name}", tuple(tensor.shape), scale)
    dit.load_state_dict(state)
    dit.eval()
    dit_latents = param("music3.dit.latents", (1, DIT["in_channels"], DIT_LENGTH), 1.5)
    dit_condition = param(
        "music3.dit.condition", (1, DIT_LENGTH, DIT["condition_dim"]), 1.5
    )
    dit_timestep = torch.tensor([0.25], dtype=torch.float32)
    dit_out = dit(
        hidden_states=dit_latents,
        timestep=dit_timestep,
        encoder_hidden_states=dit_condition,
        return_dict=False,
    )[0]
    # The UNCONDITIONAL branch: zeros for the condition, same latents and time
    # (denoise.py:204). It is emitted so the C++ gate exercises the exact pair the
    # guider mixes rather than a second conditional pass.
    dit_out_uncond = dit(
        hidden_states=dit_latents,
        timestep=dit_timestep,
        encoder_hidden_states=torch.zeros_like(dit_condition),
        return_dict=False,
    )[0]
    out.append(f"inline constexpr double kMusic3DitTimestep = {float(dit_timestep):.9g};")
    for name in sorted(state):
        cpp = "kMusic3DitW_" + name.replace(".", "_")
        emit_floats(out, cpp, state[name].numpy())
    emit_floats(out, "kMusic3DitLatents", dit_latents.numpy())
    emit_floats(out, "kMusic3DitCondition", dit_condition.numpy())
    emit_floats(out, "kMusic3DitOut", dit_out.numpy())
    emit_floats(out, "kMusic3DitOutUncond", dit_out_uncond.numpy())
    out.append("")

    # ---------------- vocoder sub-pieces --------------------------------------
    out.append("// --- vocoder reduced-dimension config ---")
    out.append(f"inline constexpr int64_t kMusic3VocLatentChannels = {VOCODER['latent_channels']};")
    out.append(f"inline constexpr int64_t kMusic3VocInputDim = {VOCODER['decoder_input_dim']};")
    out.append(f"inline constexpr int64_t kMusic3VocHiddenDim = {VOCODER['decoder_hidden_dim']};")
    out.append(
        f"inline constexpr int64_t kMusic3VocRatioCount = {len(VOCODER['upsampling_ratios'])};"
    )
    emit_ints(out, "kMusic3VocRatios", list(VOCODER["upsampling_ratios"]))
    out.append(f"inline constexpr int64_t kMusic3VocLength = {VOCODER_LENGTH};")
    hop_reduced = int(np.prod(VOCODER["upsampling_ratios"]))
    out.append(f"inline constexpr int64_t kMusic3VocHop = {hop_reduced};")
    out.append(f"inline constexpr int64_t kMusic3VocSamplingRate = {VOCODER['sampling_rate']};")
    out.append("")

    # The snake, on its own, at a channel count and length that are not equal.
    snake = MiniMaxMusic3Snake1d(3)
    snake_alpha = param("music3.voc.snake.alpha", (1, 3, 1), 0.6, 1.0)
    snake.alpha.copy_(snake_alpha)
    snake_in = param("music3.voc.snake.in", (1, 3, 5), 2.0)
    snake_out_t = snake(snake_in)
    out.append("inline constexpr int64_t kMusic3SnakeChannels = 3;")
    out.append("inline constexpr int64_t kMusic3SnakeLength = 5;")
    emit_floats(out, "kMusic3SnakeAlpha", snake_alpha.numpy())
    emit_floats(out, "kMusic3SnakeIn", snake_in.numpy())
    emit_floats(out, "kMusic3SnakeOut", snake_out_t.numpy())
    out.append("")

    # One residual unit at the LARGEST dilation, whose padding arithmetic is the
    # piece a 1-off breaks silently (pad = (7 - 1) * dilation // 2).
    unit = MiniMaxMusic3VocoderResidualUnit(4, dilation=9)
    unit_state = {}
    for name, tensor in unit.state_dict().items():
        offset = 1.0 if name.endswith("alpha") else 0.0
        unit_state[name] = param(f"music3.voc.unit.{name}", tuple(tensor.shape), 0.4, offset)
    unit.load_state_dict(unit_state)
    unit.eval()
    unit_in = param("music3.voc.unit.in", (1, 4, 6), 1.2)
    unit_out = unit(unit_in)
    out.append("inline constexpr int64_t kMusic3UnitChannels = 4;")
    out.append("inline constexpr int64_t kMusic3UnitLength = 6;")
    out.append("inline constexpr int64_t kMusic3UnitDilation = 9;")
    # FOLDED, exactly as W1's loader folds the checkpoint: the live module's
    # `.weight` is `g * v / ||v||`, so the .inc carries what the C++ consumes.
    unit_emitted = {
        name: tensor
        for name, tensor in unit.state_dict().items()
        if not name.endswith(".weight_g") and not name.endswith(".weight_v")
    }
    unit_emitted["conv1.weight"] = unit.conv1.weight.detach()
    unit_emitted["conv2.weight"] = unit.conv2.weight.detach()
    for name in sorted(unit_emitted):
        cpp = "kMusic3UnitW_" + name.replace(".", "_")
        emit_floats(out, cpp, unit_emitted[name].numpy())
    emit_floats(out, "kMusic3UnitIn", unit_in.numpy())
    emit_floats(out, "kMusic3UnitOut", unit_out.numpy())
    out.append("")

    # The whole vocoder. `weight_norm` is folded here the way W1's loader folds
    # it, so the .inc carries ONE weight per convolution and the C++ consumes the
    # same thing it will consume from the checkpoint.
    vocoder = MiniMaxMusic3Vocoder(**VOCODER)
    voc_state = {}
    for name, tensor in vocoder.state_dict().items():
        offset = 1.0 if name.endswith("alpha") else 0.0
        voc_state[name] = param(f"music3.voc.{name}", tuple(tensor.shape), 0.35, offset)
    vocoder.load_state_dict(voc_state)
    vocoder.eval()
    voc_in = param(
        "music3.voc.latents", (1, VOCODER["latent_channels"], VOCODER_LENGTH), 1.5
    )
    voc_out = vocoder(voc_in)
    # The FOLDED weights, taken from the live module after weight_norm resolved
    # them -- the same `w = g * v / ||v||` W1 computes at load.
    folded = dict(vocoder.named_parameters())
    folded_names = []
    for module_name, module in vocoder.named_modules():
        if not hasattr(module, "weight_g"):
            continue
        folded_names.append(module_name)
    emitted = {}
    for name, tensor in vocoder.state_dict().items():
        if name.endswith(".weight_g") or name.endswith(".weight_v"):
            continue
        emitted[name] = tensor
    for module_name in folded_names:
        module = dict(vocoder.named_modules())[module_name]
        emitted[module_name + ".weight"] = module.weight.detach()
    for name in sorted(emitted):
        cpp = "kMusic3VocW_" + name.replace(".", "_")
        emit_floats(out, cpp, emitted[name].numpy())
    out.append(
        "// The weight-normed modules, in the order W1's "
        "MiniMaxMusic3WeightNormedModules walks them."
    )
    out.append(
        "inline constexpr const char* kMusic3VocWeightNormedModules[] = {"
        + ", ".join(f'"{name}"' for name in folded_names)
        + "};"
    )
    out.append(
        f"inline constexpr int64_t kMusic3VocWeightNormedCount = {len(folded_names)};"
    )
    emit_floats(out, "kMusic3VocLatents", voc_in.numpy())
    emit_floats(out, "kMusic3VocOut", voc_out.numpy())
    out.append(f"inline constexpr int64_t kMusic3VocOutSamples = {voc_out.shape[-1]};")
    out.append("")

    out.append("}  // namespace vllm_test")
    out.append("")

    args.out.write_text("\n".join(out))
    print(f"wrote {args.out} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
