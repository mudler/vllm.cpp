#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_pipeline_goldens.inc — the LTX-2.5 PIPELINE oracle.

Phase L5 of .agents/specs/ltx-2-5.md (issue #435): the flow-matching schedule,
the noiser, the diffusion steps, guidance and the guiders, the patchifiers, the
distilled two-stage recipe, the latent spatial x2 upsampler, the duration head
and the `Embeddings1DConnector`.

Everything below is produced by EXECUTING the upstream modules at REDUCED
dimensions on CPU, with both sides rebuilding weights and inputs from one
deterministic FNV-1a + splitmix64 stream keyed by parameter NAME, so no weight
byte is checked in. This is the method that made L2 and L4 trustworthy
(scripts/gen-ltx2-goldens.py, scripts/gen-ltx2-vae-goldens.py).

Upstream sources:

  A. Lightricks/LTX-2, packages/ltx-core/src/ltx_core/  (EXECUTED)
       components/schedulers.py          -> section 1
       components/noisers.py             -> section 2
       components/diffusion_steps.py     -> section 3
       components/guiders.py             -> section 4
       guidance/perturbations.py         -> section 5
       components/patchifiers.py         -> section 6
       model/upsampler/*.py              -> section 8
       duration_head/duration_head.py    -> section 9
       text_encoders/gemma/embeddings_connector.py -> section 10

  B. Lightricks/LTX-2, packages/ltx-pipelines/src/ltx_pipelines/  (READ)
       utils/constants.py  -> section 7, EXECUTED by path (its package __init__
                              pulls in `av`, which is not a dependency of the math)
       distilled.py        -> section 7, read with `ast` (its imports reach the
                              whole pipeline stack; the constants are literals)

  C. vLLM-Omni, vllm_omni/diffusion/models/ltx2/ltx2_recipes.py  (READ with `ast`)
       -> section 7. This is the BINDING oracle's serving model, and it carries
          NO 2.5 row — see the spec section 3. Its table is emitted so the C++
          refusal is gated against the real key set rather than against a
          remembered one, and so a 2.5 row appearing upstream shows up as a
          golden change instead of going unnoticed.

Usage:
    python3 scripts/gen-ltx2-pipeline-goldens.py \\
        --ltx2 ~/_git/LTX-2 \\
        --vllm-omni ~/_git/vllm-omni \\
        --out tests/vllm/models/ltx2_pipeline_goldens.inc

Needs torch + numpy + einops (CPU only). No checkpoint, venv, or gated download.

WHY THIS SCRIPT REFUSES THREE THINGS. Each refusal is a finding this campaign
paid for, not a defensive habit:

 1. ORACLE IDENTITY. The resolved `ltx_core.__file__` must live under `--ltx2`.
    L4 proved a decoy `ltx_core` on `sys.path` produces goldens with an IDENTICAL
    md5 (spec section 7.0(b)), so `sys.path.insert` winning is not something to
    assume.
 2. PROVENANCE. `git -C <root> rev-parse HEAD` is emitted for BOTH upstreams and
    asserted against a SHA the C++ suite pins.
 3. A DIRTY TREE IS REFUSED. A revision anchor read from a tree with uncommitted
    edits stamps a clean SHA on goldens the SHA does not describe — the anchor
    then actively misleads a bisect. `git status --porcelain` must be empty for
    both upstreams.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Deterministic weight/input stream, mirrored bit-for-bit by the C++ suite
# (tests/vllm/models/test_ltx2_pipeline.cpp :: Ltx2Rand). A per-tensor FNV-1a seed
# plus a splitmix64 counter makes every tensor independent of fill ORDER, so the
# two sides cannot silently drift by reordering their parameter construction.
# Identical to the L2 and L4 generators'.
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


def make(name: str, count: int, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    return (ltx_rand(name, count) * scale + offset).astype(np.float32)


# ---------------------------------------------------------------------------
# The per-parameter role rule. The C++ side implements the SAME rule keyed on the
# same names, so a divergence shows up as a golden mismatch rather than as a
# silently different tensor.
# ---------------------------------------------------------------------------


def param_values(name: str, shape) -> np.ndarray:
    count = int(np.prod(shape)) if len(shape) else 1
    rank = len(shape)
    if name.endswith(".bias") or name.endswith("in_proj_bias"):
        return make(name, count, 0.02)
    if rank == 1 and name.endswith(".weight"):
        # A 1-D `.weight` is an affine norm gain (GroupNorm, RMSNorm), whose
        # trained value sits around 1.0. Centring it on 0.0 would make every
        # normalized activation ~0 and hide a scale error.
        return make(name, count, 0.1, 1.0)
    if ".dur." in name:
        # The DURATION HEAD needs a wider fixture than everything else, and this
        # is a gate property rather than a taste. Its output is
        # `exp(mlp_out(...))` through a chain that ATTENUATES: at the shared 0.05
        # scale every projection shrinks its input, `log_duration` lands within
        # ~0.007 of `mlp_out.bias`, and the both / video-only / audio-only arms
        # collapse to within 3e-6 of each other -- BELOW this suite's round-off
        # bound. A gate that cannot separate its three arms would accept an
        # implementation that ignored one of the two streams entirely. MEASURED
        # spreads at this fixture: 0.05 -> 2.98e-06, 0.2 -> 2.3e-03,
        # 0.35 -> 4.9e-02, which is ~10^4 x the bound while keeping the predicted
        # duration in a sane 0.9-1.0 second range.
        return make(name, count, 0.35)
    return make(name, count, 0.05)


def fill_module(module, prefix: str) -> list[tuple[str, int]]:
    """Fill every parameter from the shared stream; return the state_dict manifest.

    Iterates `named_parameters()`, which is the ORDER the C++ side asserts. The
    fill is by NAME, so the order only decides the manifest, never the values.
    """
    import torch  # noqa: PLC0415

    manifest: list[tuple[str, int]] = []
    for name, param in module.named_parameters():
        values = param_values(prefix + name, tuple(param.shape))
        # `.copy_` rather than assignment: it ROUNDS into the parameter's own
        # dtype. Embeddings1DConnector.learnable_registers is bfloat16 by
        # construction (embeddings_connector.py:135-137), and a port that keeps
        # those values in f32 is WIDER than upstream — the exact polarity
        # AGENTS.md warns a value gate cannot catch, so it is baked into the
        # oracle here and mirrored on the C++ side.
        param.data.copy_(torch.from_numpy(values).reshape(param.shape))
        manifest.append((prefix + name, int(values.size)))
    return manifest


# ---------------------------------------------------------------------------
# Emission
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


def emit_f64(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1).tolist()
    out.write(f"inline constexpr double {name}[] = {{\n")
    for i in range(0, len(flat), 4):
        chunk = ", ".join(_cxx_float(v, 17) for v in flat[i : i + 4])
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


def emit_double(out, name: str, value: float) -> None:
    out.write(f"inline constexpr double {name} = {_cxx_float(float(value), 17)};\n")


def emit_bool(out, name: str, value: bool) -> None:
    out.write(f"inline constexpr bool {name} = {'true' if value else 'false'};\n")


def emit_string(out, name: str, value: str) -> None:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    out.write(f'inline constexpr const char* {name} = "{escaped}";\n')


def emit_manifest(out, name: str, manifest: list[tuple[str, int]]) -> None:
    out.write(f"inline constexpr const char* {name}Names[] = {{\n")
    for key, _ in manifest:
        out.write(f'    "{key}",\n')
    out.write("};\n")
    out.write(f"inline constexpr int64_t {name}Counts[] = {{\n")
    for i in range(0, len(manifest), 10):
        out.write("    " + ", ".join(str(c) for _, c in manifest[i : i + 10]) + ",\n")
    out.write("};\n\n")


def section(out, title: str) -> None:
    out.write(f"// {'-' * 74}\n// {title}\n// {'-' * 74}\n\n")


# ---------------------------------------------------------------------------
# Section 1 — schedulers (components/schedulers.py)
# ---------------------------------------------------------------------------

# Each case is (tag, steps, tokens-or-None, max_shift, base_shift, stretch, terminal).
# `tokens` None exercises `default_number_of_tokens = MAX_SHIFT_ANCHOR`; a value
# exercises the `math.prod(latent.shape[2:])` path, which is what makes the shift
# resolution-dependent (schedulers.py:32).
_SCHED_CASES = (
    ("Default", 8, None, 2.05, 0.95, True, 0.1),
    ("NoStretch", 8, None, 2.05, 0.95, False, 0.1),
    # tokens BELOW the base anchor: sigma_shift goes negative, which is the arm a
    # small render actually takes and the one where the linear fit extrapolates.
    ("FewTokens", 6, 256, 2.05, 0.95, True, 0.1),
    ("ManyTokens", 6, 16384, 2.05, 0.95, True, 0.1),
    ("Terminal0", 5, 1024, 2.05, 0.95, True, 0.0),
    ("OneStep", 1, 4096, 2.05, 0.95, True, 0.1),
    # STEP COUNTS WHERE A NAIVE FORWARD linspace WALK MISSES EXACT 0. torch walks
    # the second half BACKWARDS from `end` (aten RangeFactories.cpp), which is the
    # only reason the terminal sigma is exactly 0. `start + step * i` instead leaves
    # 5.96e-08 there, which survives the `sigmas != 0` guard at schedulers.py:42,
    # takes the shift transform, and then becomes `one_minus_z[-1]` in the stretch
    # branch (:52) -- so the WHOLE schedule moves and the denoise loop never reaches
    # zero noise. `--steps 41` is a plain user-reachable render, not a corner.
    # 24 of the first 198 counts are affected; these are the first two above 1.
    ("Steps41", 41, None, 2.05, 0.95, True, 0.1),
    ("Steps47", 47, None, 2.05, 0.95, True, 0.1),
)

# (tag, steps, threshold_noise, linear_steps-or-None)
_LINQUAD_CASES = (
    ("Default", 6, 0.025, None),
    ("OneStep", 1, 0.025, None),
    ("Explicit", 7, 0.05, 2),
    # linear_steps == steps leaves quadratic_steps == 0, the branch that skips the
    # quadratic tail entirely (schedulers.py:79).
    ("AllLinear", 4, 0.025, 4),
)


def section_schedulers(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.components.schedulers import (  # noqa: PLC0415
        BASE_SHIFT_ANCHOR,
        MAX_SHIFT_ANCHOR,
        LinearQuadraticScheduler,
        LTX2Scheduler,
    )

    section(out, "Section 1 - sigma schedules (components/schedulers.py)")
    # The two anchors are module constants (schedulers.py:10-11), not arguments:
    # they set the token axis the shift is fitted on, so they belong to the gate.
    emit_scalar(out, "kLtx2SchedBaseShiftAnchor", BASE_SHIFT_ANCHOR)
    emit_scalar(out, "kLtx2SchedMaxShiftAnchor", MAX_SHIFT_ANCHOR)
    out.write("\n")

    scheduler = LTX2Scheduler()
    for tag, steps, tokens, max_shift, base_shift, stretch, terminal in _SCHED_CASES:
        latent = None
        if tokens is not None:
            # `math.prod(latent.shape[2:])` — the only thing read off the latent.
            latent = torch.zeros(1, 1, tokens)
        sigmas = scheduler.execute(
            steps=steps,
            latent=latent,
            max_shift=max_shift,
            base_shift=base_shift,
            stretch=stretch,
            terminal=terminal,
        )
        assert sigmas.dtype == torch.float32
        emit_scalar(out, f"kLtx2Sched{tag}Steps", steps)
        emit_scalar(out, f"kLtx2Sched{tag}Tokens", tokens if tokens is not None else 0)
        emit_double(out, f"kLtx2Sched{tag}MaxShift", max_shift)
        emit_double(out, f"kLtx2Sched{tag}BaseShift", base_shift)
        emit_bool(out, f"kLtx2Sched{tag}Stretch", stretch)
        emit_double(out, f"kLtx2Sched{tag}Terminal", terminal)
        emit_f32(out, f"kLtx2Sched{tag}Golden", sigmas.numpy())

    linear = LinearQuadraticScheduler()
    for tag, steps, threshold, linear_steps in _LINQUAD_CASES:
        kwargs = {"steps": steps, "threshold_noise": threshold}
        if linear_steps is not None:
            kwargs["linear_steps"] = linear_steps
        sigmas = linear.execute(**kwargs)
        emit_scalar(out, f"kLtx2LinQuad{tag}Steps", steps)
        emit_double(out, f"kLtx2LinQuad{tag}Threshold", threshold)
        emit_scalar(out, f"kLtx2LinQuad{tag}LinearSteps", -1 if linear_steps is None else linear_steps)
        emit_f32(out, f"kLtx2LinQuad{tag}Golden", sigmas.numpy())


# ---------------------------------------------------------------------------
# Section 2 — the noiser (components/noisers.py)
# ---------------------------------------------------------------------------

_NOISE_COUNT = 24


def section_noiser(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.components.noisers import GaussianNoiser  # noqa: PLC0415
    from ltx_core.types import LatentState  # noqa: PLC0415

    section(out, "Section 2 - GaussianNoiser (components/noisers.py:30-37)")

    latent = torch.from_numpy(make("ltx2.noiser.latent", _NOISE_COUNT, 1.0))
    clean = torch.from_numpy(make("ltx2.noiser.clean", _NOISE_COUNT, 1.0))
    noise = torch.from_numpy(make("ltx2.noiser.noise", _NOISE_COUNT, 1.0))
    # The denoise mask is upstream's per-element blend weight between the CLEAN
    # latent (conditioning) and the noised one. 0 and 1 both appear in a real
    # request (a conditioned frame is fully clean), so both endpoints are here.
    mask_raw = make("ltx2.noiser.mask", _NOISE_COUNT, 0.5, 0.5)
    mask_raw[0] = 0.0
    mask_raw[1] = 1.0
    mask = torch.from_numpy(mask_raw)

    emit_scalar(out, "kLtx2NoiserCount", _NOISE_COUNT)
    emit_f32(out, "kLtx2NoiserMask", mask.numpy())

    class _FixedNoiser(GaussianNoiser):
        """The one harness adaptation, and it changes no arithmetic.

        Upstream draws `torch.randn(..., generator=self.generator)`; the C++ side
        consumes an Ltx2NoiseStream. Substituting a FIXED draw keeps the lerp
        chain — which is the whole of the module's math — byte-comparable without
        also porting torch's Philox.
        """

        def _sample_noise(self, latent_state):  # noqa: ANN001, ANN201
            return noise

    noiser = _FixedNoiser(generator=None)
    for tag, scale in (("Full", 1.0), ("Half", 0.5), ("Zero", 0.0)):
        state = LatentState(
            latent=latent.clone(),
            clean_latent=clean.clone(),
            denoise_mask=mask,
            # `positions` is required by the dataclass and never read by the
            # noiser (noisers.py:30-37 touches latent / clean_latent / mask only).
            positions=torch.zeros(_NOISE_COUNT),
        )
        result = noiser(state, noise_scale=scale)
        emit_double(out, f"kLtx2Noiser{tag}Scale", scale)
        emit_f32(out, f"kLtx2Noiser{tag}Golden", result.latent.numpy())


# ---------------------------------------------------------------------------
# Section 3 — diffusion steps (components/diffusion_steps.py)
# ---------------------------------------------------------------------------

_STEP_COUNT = 20
# A schedule with a mid-range sigma, a near-terminal one and the terminal 0, so
# the `sigma_next == 0` early-outs are exercised rather than described.
_STEP_SIGMAS = (1.0, 0.725, 0.421875, 0.0)


def section_diffusion_steps(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.components.diffusion_steps import (  # noqa: PLC0415
        EulerAncestralDiffusionStep,
        EulerCfgPpDiffusionStep,
        EulerDiffusionStep,
        Res2sDiffusionStep,
        _get_ancestral_step,
    )

    section(out, "Section 3 - diffusion steps (components/diffusion_steps.py)")

    sample = torch.from_numpy(make("ltx2.step.sample", _STEP_COUNT, 1.0))
    denoised = torch.from_numpy(make("ltx2.step.denoised", _STEP_COUNT, 0.8))
    uncond = torch.from_numpy(make("ltx2.step.uncond", _STEP_COUNT, 0.8))
    noise = torch.from_numpy(make("ltx2.step.noise", _STEP_COUNT, 1.0))
    sigmas = torch.tensor(_STEP_SIGMAS, dtype=torch.float32)

    emit_scalar(out, "kLtx2StepCount", _STEP_COUNT)
    emit_f32(out, "kLtx2StepSigmas", sigmas.numpy())
    emit_scalar(out, "kLtx2StepSigmaCount", len(_STEP_SIGMAS))
    out.write("\n")

    euler = EulerDiffusionStep()
    for index in (0, 1):
        emit_f32(
            out,
            f"kLtx2StepEuler{index}Golden",
            euler.step(sample, denoised, sigmas, index).numpy(),
        )

    for tag, eta, s_noise in (("Eta1", 1.0, 1.0), ("Eta0", 0.0, 1.0), ("EtaHalf", 0.5, 0.75)):
        stepper = EulerAncestralDiffusionStep(eta=eta, s_noise=s_noise)
        emit_double(out, f"kLtx2StepAncestral{tag}Eta", eta)
        emit_double(out, f"kLtx2StepAncestral{tag}SNoise", s_noise)
        for index in (0, 1):
            emit_f32(
                out,
                f"kLtx2StepAncestral{tag}Step{index}Golden",
                stepper.step(sample, denoised, sigmas, index, noise=noise).numpy(),
            )
        # step_index 2 has sigma_next == 0: upstream returns the denoised
        # prediction outright (diffusion_steps.py:85-86). A port that instead
        # divided by sigma_next would produce inf, so the branch is gated.
        emit_f32(
            out,
            f"kLtx2StepAncestral{tag}TerminalGolden",
            stepper.step(sample, denoised, sigmas, 2, noise=noise).numpy(),
        )

    res2s = Res2sDiffusionStep()
    for tag, eta in (("EtaHalf", 0.5), ("Eta1", 1.0)):
        emit_double(out, f"kLtx2StepRes2s{tag}Eta", eta)
        for index in (0, 1):
            emit_f32(
                out,
                f"kLtx2StepRes2s{tag}Step{index}Golden",
                res2s.step(sample, denoised, sigmas, index, noise, eta=eta).numpy(),
            )
        emit_f32(
            out,
            f"kLtx2StepRes2s{tag}TerminalGolden",
            res2s.step(sample, denoised, sigmas, 2, noise, eta=eta).numpy(),
        )

    for tag, eta, s_noise in (("Eta1", 1.0, 1.0), ("Eta0", 0.0, 1.0)):
        cfgpp = EulerCfgPpDiffusionStep(eta=eta, s_noise=s_noise)
        emit_double(out, f"kLtx2StepCfgPp{tag}Eta", eta)
        emit_double(out, f"kLtx2StepCfgPp{tag}SNoise", s_noise)
        for index in (0, 1, 2):
            # index 0 has sigma_s == 1.0 EXACTLY, so `alpha_s = (1 - sigma_s)`
            # is 0 and the finfo(float32).eps clamp (diffusion_steps.py:233-235)
            # is what keeps `d = (x - alpha_s * uncond) / sigma_s` finite. That
            # makes it the ONE arm in this file where a member of the
            # invisible-constant class actually decides the numbers.
            emit_f32(
                out,
                f"kLtx2StepCfgPp{tag}Step{index}Golden",
                cfgpp.step(sample, denoised, sigmas, index, uncond_denoised=uncond,
                           noise=noise).numpy(),
            )

    # _get_ancestral_step (diffusion_steps.py:7-22), the DDIM helper CFG++ uses.
    pairs = ((1.0, 0.725), (0.725, 0.421875), (0.421875, 0.0))
    downs, ups = [], []
    for eta in (1.0, 0.5, 0.0):
        for sigma_from, sigma_to in pairs:
            down, up = _get_ancestral_step(
                torch.tensor(sigma_from), torch.tensor(sigma_to), eta=eta
            )
            downs.append(float(down))
            ups.append(float(up))
    emit_f64(out, "kLtx2AncestralHelperEtas", [1.0, 0.5, 0.0])
    emit_f64(out, "kLtx2AncestralHelperPairs", [v for pair in pairs for v in pair])
    emit_f32(out, "kLtx2AncestralHelperDown", downs)
    emit_f32(out, "kLtx2AncestralHelperUp", ups)


# ---------------------------------------------------------------------------
# Section 4 — guiders (components/guiders.py)
# ---------------------------------------------------------------------------

# rank-4 on purpose: LtxAPGGuider reduces over dim=[-1,-2,-3] and projection_coef
# flattens per BATCH row, so a rank-2 fixture would make the two agree by
# accident (guiders.py:114, 364-369).
_GUIDE_SHAPE = (2, 3, 4, 5)


def section_guiders(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.components.guiders import (  # noqa: PLC0415
        CFGGuider,
        CFGStarRescalingGuider,
        LegacyStatefulAPGGuider,
        LtxAPGGuider,
        MultiModalGuider,
        MultiModalGuiderFactory,
        MultiModalGuiderParams,
        STGGuider,
        projection_coef,
    )

    section(out, "Section 4 - guiders (components/guiders.py)")

    count = int(np.prod(_GUIDE_SHAPE))
    cond = torch.from_numpy(make("ltx2.guide.cond", count, 1.0)).reshape(_GUIDE_SHAPE)
    uncond = torch.from_numpy(make("ltx2.guide.uncond", count, 0.9)).reshape(_GUIDE_SHAPE)
    perturbed = torch.from_numpy(make("ltx2.guide.perturbed", count, 0.7)).reshape(_GUIDE_SHAPE)
    modality = torch.from_numpy(make("ltx2.guide.modality", count, 0.6)).reshape(_GUIDE_SHAPE)

    emit_i64(out, "kLtx2GuideShape", _GUIDE_SHAPE)
    emit_scalar(out, "kLtx2GuideCount", count)
    out.write("\n")

    emit_f32(out, "kLtx2GuideProjCoefGolden", projection_coef(cond, uncond).numpy())
    # The 1e-8 in `squared_norm` (guiders.py:368) is a member of the
    # invisible-constant class: with an O(1) denominator it is ~1e-8 relative and
    # no golden can see it. This arm drives `project_onto` to EXACTLY zero, where
    # the constant alone decides the result (0 / 1e-8 = 0 instead of 0/0 = NaN),
    # so a mutation of it moves the number rather than being absorbed.
    zeros = torch.zeros(_GUIDE_SHAPE)
    emit_f32(out, "kLtx2GuideProjCoefZeroGolden", projection_coef(cond, zeros).numpy())
    # ...and one where the denominator is small but NOT zero, so the epsilon is an
    # additive term against a comparable magnitude.
    tiny = torch.full(_GUIDE_SHAPE, 1e-5)
    emit_f32(out, "kLtx2GuideProjCoefTinyGolden", projection_coef(cond, tiny).numpy())
    emit_double(out, "kLtx2GuideProjCoefTinyValue", 1e-5)
    out.write("\n")

    for tag, scale in (("Scale3", 3.0), ("Scale1", 1.0)):
        cfg = CFGGuider(scale=scale)
        emit_double(out, f"kLtx2GuideCfg{tag}Scale", scale)
        emit_bool(out, f"kLtx2GuideCfg{tag}Enabled", cfg.enabled())
        emit_f32(out, f"kLtx2GuideCfg{tag}Golden", cfg.delta(cond, uncond).numpy())

    for tag, scale in (("Scale1", 1.0), ("Scale0", 0.0)):
        stg = STGGuider(scale=scale)
        emit_double(out, f"kLtx2GuideStg{tag}Scale", scale)
        emit_bool(out, f"kLtx2GuideStg{tag}Enabled", stg.enabled())
        emit_f32(out, f"kLtx2GuideStg{tag}Golden", stg.delta(cond, perturbed).numpy())
    out.write("\n")

    # --- The three PROJECTION guiders, and why the port refuses them ----------
    # THE REFUSAL RESTS ON REACHABILITY, NOT ON SHAPES. An earlier revision of
    # this file claimed the shape expression "raises at every rectangular rank
    # >= 3, i.e. at every real (B, C, F, H, W) video latent". That premise is
    # FALSE and the matrix below is what disproves it, so it is recorded here
    # rather than quietly dropped (spec §7.0(b): a wrong finding frozen as a
    # golden is worse than no golden).
    #
    # `projection_coef` returns a rank-2 `(B, 1)` tensor (guiders.py:363-369) and
    # the three guiders multiply it straight into a latent (`proj_coeff * cond`,
    # guiders.py:48, 118, 184). torch right-aligns, so `(B, 1)` lands on the
    # latent's LAST TWO axes. The real predicate is therefore
    #
    #     raises  <=>  B > 1 and shape[-2] not in {1, B}
    #
    # which is NOT "every video latent": at B = 1 — the ordinary single-request
    # render — `(1, 1)` broadcasts as a plain scalar and the result is also
    # numerically CORRECT, and `(2, 128, 8, 2, 16)` composes as well because its
    # `shape[-2]` happens to equal B. Where it composes with B > 1 it is silently
    # WRONG, applying the per-batch coefficient along axis -2 instead of the batch
    # axis. The `norm(dim=[-1,-2,-3])` in the two threshold arms is a SEPARATE
    # constraint that additionally needs rank >= 3.
    #
    # What actually justifies the refusal is that NOTHING UPSTREAM CONSTRUCTS
    # THEM. `CFGStarRescalingGuider`, `LtxAPGGuider` and `LegacyStatefulAPGGuider`
    # appear in the whole LTX-2 tree only at their own `class` statements in
    # guiders.py (:31, :78, :129); every pipeline builds `MultiModalGuider` from
    # `MultiModalGuiderParams` (utils/constants.py:49-68). So the arm is UNPORTED
    # and refused by name and recorded as owed, per AGENTS.md. The matrix stays a
    # golden so that upstream wiring one of them up shows up as a golden change.
    probe_shapes = ((2, 6), (2, 3, 4), (2, 3, 4, 5), (4, 4, 4, 4), (1, 3, 4),
                    (1, 128, 8, 16, 16), (2, 128, 8, 2, 16))
    probe_guiders = (
        ("CFGGuider", lambda: CFGGuider(scale=3.0)),
        ("STGGuider", lambda: STGGuider(scale=1.0)),
        ("CFGStarRescalingGuider", lambda: CFGStarRescalingGuider(scale=3.0)),
        ("LtxAPGGuiderNoThreshold", lambda: LtxAPGGuider(scale=3.0, eta=1.0, norm_threshold=0.0)),
        ("LtxAPGGuiderThreshold", lambda: LtxAPGGuider(scale=3.0, eta=1.0, norm_threshold=1.0)),
        ("LegacyStatefulAPGGuider",
         lambda: LegacyStatefulAPGGuider(scale=2.0, eta=0.75, norm_threshold=5.0, momentum=0.5)),
    )
    composes: list[int] = []
    for _, factory in probe_guiders:
        for shape in probe_shapes:
            left = torch.from_numpy(make("ltx2.guide.probe.a", int(np.prod(shape)))).reshape(shape)
            right = torch.from_numpy(make("ltx2.guide.probe.b", int(np.prod(shape)))).reshape(shape)
            try:
                factory().delta(left, right)
                composes.append(1)
            except (RuntimeError, IndexError):
                composes.append(0)
    out.write("inline constexpr const char* kLtx2GuideProbeNames[] = {\n")
    for name, _ in probe_guiders:
        out.write(f'    "{name}",\n')
    out.write("};\n")
    emit_scalar(out, "kLtx2GuideProbeGuiderCount", len(probe_guiders))
    emit_scalar(out, "kLtx2GuideProbeShapeCount", len(probe_shapes))
    emit_i64(out, "kLtx2GuideProbeRanks", [len(s) for s in probe_shapes])
    # The two axes the real predicate is written in: the batch, and the axis the
    # `(B, 1)` coefficient actually lands on. `square` is deliberately NOT emitted
    # any more — it was a mis-generalization of these two.
    emit_i64(out, "kLtx2GuideProbeBatch", [s[0] for s in probe_shapes])
    emit_i64(out, "kLtx2GuideProbeSecondLast", [s[-2] for s in probe_shapes])
    emit_i64(out, "kLtx2GuideProbeComposes", composes)
    out.write("\n")

    # MultiModalGuider — the ONLY guider any ltx-pipelines entry point constructs
    # (utils/constants.py:49-68 builds MultiModalGuiderParams for both streams).
    mm_cases = (
        ("Official", dict(cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
                          modality_scale=3.0, skip_step=0)),
        ("Audio", dict(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                       modality_scale=3.0, skip_step=0)),
        ("NoRescale", dict(cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.0,
                           modality_scale=3.0, skip_step=0)),
        ("PositiveOnly", dict(cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
                              modality_scale=1.0, skip_step=0)),
        ("Skip2", dict(cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
                       modality_scale=3.0, skip_step=2)),
    )
    for tag, kwargs in mm_cases:
        params = MultiModalGuiderParams(stg_blocks=[29], **kwargs)
        guider = MultiModalGuider(params=params)
        emit_double(out, f"kLtx2GuideMm{tag}CfgScale", kwargs["cfg_scale"])
        emit_double(out, f"kLtx2GuideMm{tag}StgScale", kwargs["stg_scale"])
        emit_double(out, f"kLtx2GuideMm{tag}RescaleScale", kwargs["rescale_scale"])
        emit_double(out, f"kLtx2GuideMm{tag}ModalityScale", kwargs["modality_scale"])
        emit_scalar(out, f"kLtx2GuideMm{tag}SkipStep", kwargs["skip_step"])
        emit_bool(out, f"kLtx2GuideMm{tag}DoUncond", guider.do_unconditional_generation())
        emit_bool(out, f"kLtx2GuideMm{tag}DoPerturbed", guider.do_perturbed_generation())
        emit_bool(out, f"kLtx2GuideMm{tag}DoModality", guider.do_isolated_modality_generation())
        emit_i64(
            out,
            f"kLtx2GuideMm{tag}SkipStepMask",
            [1 if guider.should_skip_step(step) else 0 for step in range(8)],
        )
        emit_f32(
            out,
            f"kLtx2GuideMm{tag}Golden",
            guider.calculate(cond, uncond, perturbed, modality).numpy(),
        )

    # The sigma-binned factory (guiders.py:214-230, 317-335). Keys are bin UPPER
    # bounds sorted descending; the rule picks the SMALLEST key >= sigma, and a
    # sigma above every key falls back to the largest. Both edges are queried.
    bins = {
        1.0: MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0, stg_blocks=[29]),
        0.5: MultiModalGuiderParams(cfg_scale=5.0, stg_scale=0.5, stg_blocks=[29]),
        0.25: MultiModalGuiderParams(cfg_scale=7.0, stg_scale=0.0, stg_blocks=[29]),
    }
    factory = MultiModalGuiderFactory.from_dict(bins)
    queries = [2.0, 1.0, 0.75, 0.5, 0.4, 0.25, 0.1, 0.0]
    emit_f64(out, "kLtx2GuideBinKeys", sorted(bins.keys(), reverse=True))
    emit_f64(out, "kLtx2GuideBinCfgScales", [bins[k].cfg_scale for k in sorted(bins, reverse=True)])
    emit_f64(out, "kLtx2GuideBinQueries", queries)
    emit_f64(out, "kLtx2GuideBinResolvedCfg", [factory.params(q).cfg_scale for q in queries])
    constant = MultiModalGuiderFactory.constant(
        MultiModalGuiderParams(cfg_scale=4.0, stg_blocks=[29])
    )
    emit_f64(out, "kLtx2GuideConstantResolvedCfg",
             [constant.params(q).cfg_scale for q in queries])


# ---------------------------------------------------------------------------
# Section 5 — perturbations (guidance/perturbations.py)
# ---------------------------------------------------------------------------

_PERTURB_BLOCKS = 4


def section_perturbations(out) -> None:
    from ltx_core.guidance.perturbations import (  # noqa: PLC0415
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )

    section(out, "Section 5 - perturbations (guidance/perturbations.py)")

    emit_scalar(out, "kLtx2PerturbTypeCount", len(PerturbationType))
    emit_scalar(out, "kLtx2PerturbNumBlocks", _PERTURB_BLOCKS)
    emit_scalar(out, "kLtx2PerturbSkipVideoSelfAttn", int(PerturbationType.SKIP_VIDEO_SELF_ATTN))
    emit_scalar(out, "kLtx2PerturbSkipAudioSelfAttn", int(PerturbationType.SKIP_AUDIO_SELF_ATTN))
    emit_scalar(out, "kLtx2PerturbSkipA2vCrossAttn", int(PerturbationType.SKIP_A2V_CROSS_ATTN))
    emit_scalar(out, "kLtx2PerturbSkipV2aCrossAttn", int(PerturbationType.SKIP_V2A_CROSS_ATTN))
    out.write("\n")

    # The EMPTY configuration is the one LTX-2.5 actually runs (spec section 2 puts
    # STG behind the guider, and L2 ships `perturbations=None`), so its all-ones
    # mask is the shipped default rather than a corner case.
    empty = BatchedPerturbationConfig.empty(batch_size=3, num_blocks=_PERTURB_BLOCKS)
    emit_i64(out, "kLtx2PerturbEmptyMask", empty.block_masks.to(int).numpy())

    # A mixed batch: sample 0 skips video self-attn in blocks {1,2}, sample 1
    # skips A2V cross-attn in EVERY block (blocks=None), sample 2 is unperturbed.
    configs = [
        PerturbationConfig([Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, [1, 2])]),
        PerturbationConfig([Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, None)]),
        PerturbationConfig.empty(),
    ]
    mixed = BatchedPerturbationConfig(configs, num_blocks=_PERTURB_BLOCKS)
    emit_i64(out, "kLtx2PerturbMixedMask", mixed.block_masks.to(int).numpy())
    emit_i64(
        out,
        "kLtx2PerturbMixedAny",
        [
            1 if mixed.any_in_batch(PerturbationType(direction), block) else 0
            for direction in range(len(PerturbationType))
            for block in range(_PERTURB_BLOCKS)
        ],
    )
    emit_i64(
        out,
        "kLtx2PerturbMixedAll",
        [
            1 if mixed.all_in_batch(PerturbationType(direction), block) else 0
            for direction in range(len(PerturbationType))
            for block in range(_PERTURB_BLOCKS)
        ],
    )
    sliced = mixed.batch_slice(1, 3)
    emit_i64(out, "kLtx2PerturbSlicedMask", sliced.block_masks.to(int).numpy())


# ---------------------------------------------------------------------------
# Section 6 — patchifiers (components/patchifiers.py)
# ---------------------------------------------------------------------------

_PATCH_SIZE = 2
_PATCH_C, _PATCH_F, _PATCH_H, _PATCH_W = 3, 2, 4, 6


def section_patchifiers(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.components.patchifiers import (  # noqa: PLC0415
        AudioPatchifier,
        VideoLatentPatchifier,
        get_pixel_coords,
    )
    from ltx_core.types import (  # noqa: PLC0415
        AudioLatentShape,
        SpatioTemporalScaleFactors,
        VideoLatentShape,
    )

    section(out, "Section 6 - patchifiers (components/patchifiers.py)")

    video = VideoLatentPatchifier(patch_size=_PATCH_SIZE)
    count = _PATCH_C * _PATCH_F * _PATCH_H * _PATCH_W
    latent = torch.from_numpy(make("ltx2.patch.video", count, 1.0)).reshape(
        1, _PATCH_C, _PATCH_F, _PATCH_H, _PATCH_W
    )
    tokens = video.patchify(latent)
    shape = VideoLatentShape(batch=1, channels=_PATCH_C, frames=_PATCH_F,
                             height=_PATCH_H, width=_PATCH_W)
    restored = video.unpatchify(tokens, shape)

    emit_scalar(out, "kLtx2PatchSize", _PATCH_SIZE)
    emit_scalar(out, "kLtx2PatchChannels", _PATCH_C)
    emit_scalar(out, "kLtx2PatchFrames", _PATCH_F)
    emit_scalar(out, "kLtx2PatchHeight", _PATCH_H)
    emit_scalar(out, "kLtx2PatchWidth", _PATCH_W)
    emit_scalar(out, "kLtx2PatchTokenCount", video.get_token_count(shape))
    emit_scalar(out, "kLtx2PatchTokens", tokens.shape[1])
    emit_scalar(out, "kLtx2PatchTokenDim", tokens.shape[2])
    emit_f32(out, "kLtx2PatchVideoLatent", latent.numpy())
    emit_f32(out, "kLtx2PatchVideoTokens", tokens.numpy())
    emit_f32(out, "kLtx2PatchVideoRestored", restored.numpy())

    bounds = video.get_patch_grid_bounds(shape)
    emit_i64(out, "kLtx2PatchBoundsShape", list(bounds.shape))
    emit_i64(out, "kLtx2PatchBoundsGolden", bounds.numpy())

    scale = SpatioTemporalScaleFactors.default()
    emit_i64(out, "kLtx2PatchScaleFactors", [scale.time, scale.height, scale.width])
    for tag, causal in (("Plain", False), ("CausalFix", True)):
        pixels = get_pixel_coords(bounds.clone(), scale, causal_fix=causal)
        emit_i64(out, f"kLtx2PatchPixelCoords{tag}Golden", pixels.numpy())

    audio_frames, audio_channels, audio_mels = 5, 2, 3
    audio_count = audio_channels * audio_frames * audio_mels
    audio_latent = torch.from_numpy(make("ltx2.patch.audio", audio_count, 1.0)).reshape(
        1, audio_channels, audio_frames, audio_mels
    )
    audio_shape = AudioLatentShape(batch=1, channels=audio_channels, frames=audio_frames,
                                   mel_bins=audio_mels)
    emit_scalar(out, "kLtx2PatchAudioFrames", audio_frames)
    emit_scalar(out, "kLtx2PatchAudioChannels", audio_channels)
    emit_scalar(out, "kLtx2PatchAudioMelBins", audio_mels)
    emit_f32(out, "kLtx2PatchAudioLatent", audio_latent.numpy())
    for tag, causal, shift in (("Causal", True, 0), ("NonCausal", False, 0), ("Shift2", True, 2)):
        patchifier = AudioPatchifier(patch_size=1, is_causal=causal, shift=shift)
        audio_tokens = patchifier.patchify(audio_latent)
        emit_scalar(out, f"kLtx2PatchAudio{tag}Shift", shift)
        emit_bool(out, f"kLtx2PatchAudio{tag}IsCausal", causal)
        emit_scalar(out, f"kLtx2PatchAudio{tag}TokenCount", patchifier.get_token_count(audio_shape))
        emit_f32(out, f"kLtx2PatchAudio{tag}Tokens", audio_tokens.numpy())
        emit_f32(
            out,
            f"kLtx2PatchAudio{tag}Restored",
            patchifier.unpatchify(audio_tokens, audio_shape).numpy(),
        )
        timings = patchifier.get_patch_grid_bounds(audio_shape)
        emit_i64(out, f"kLtx2PatchAudio{tag}TimingShape", list(timings.shape))
        emit_f32(out, f"kLtx2PatchAudio{tag}TimingGolden", timings.numpy())
    # The two rates the timings are built from (patchifiers.py:177-180). They set
    # the seconds-per-latent-frame the DiT's audio RoPE is indexed by, so a wrong
    # one is a silently mistimed soundtrack rather than an error.
    default_audio = AudioPatchifier(patch_size=1)
    emit_scalar(out, "kLtx2PatchAudioSampleRate", default_audio.sample_rate)
    emit_scalar(out, "kLtx2PatchAudioHopLength", default_audio.hop_length)
    emit_scalar(out, "kLtx2PatchAudioDownsample", default_audio.audio_latent_downsample_factor)


# ---------------------------------------------------------------------------
# Section 7 — the pipeline recipes
# ---------------------------------------------------------------------------


def _module_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_literals(path: Path, wanted: tuple[str, ...]) -> dict:
    """Module-level literal assignments, read with `ast` and NOT executed.

    `ltx_pipelines/distilled.py` and vLLM-Omni's `ltx2_recipes.py` both import
    their whole stacks, so neither can be executed here; their constants are
    plain literals, and reading them from the source is exact. A name that
    disappears upstream RAISES rather than falling back to a remembered value.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict = {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        for target in targets:
            if target in wanted:
                found[target] = node.value
    missing = [name for name in wanted if name not in found]
    if missing:
        raise SystemExit(f"{path}: expected module-level constants not found: {missing}")
    return found


def section_recipes(out, ltx2_root: Path, omni_root: Path) -> None:
    section(out, "Section 7 - pipeline recipes")

    pipelines = ltx2_root / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines"
    constants = _module_by_path("_ltx2_pipeline_constants", pipelines / "utils" / "constants.py")

    emit_f32(out, "kLtx2DistilledSigmas", constants.DISTILLED_SIGMA_VALUES)
    emit_scalar(out, "kLtx2DistilledSigmaCount", len(constants.DISTILLED_SIGMA_VALUES))
    emit_f32(out, "kLtx2Stage2DistilledSigmas", constants.STAGE_2_DISTILLED_SIGMA_VALUES)
    emit_scalar(out, "kLtx2Stage2DistilledSigmaCount",
                len(constants.STAGE_2_DISTILLED_SIGMA_VALUES))
    emit_f32(out, "kLtx2TdpDistilledSigmas", constants.TDP_DISTILLED_SIGMAS.numpy())
    emit_scalar(out, "kLtx2TdpDistilledSigmaCount", int(constants.TDP_DISTILLED_SIGMAS.numel()))
    emit_string(out, "kLtx2LightricksNegativePrompt", constants.DEFAULT_NEGATIVE_PROMPT)
    emit_scalar(out, "kLtx2DefaultImageCrf", constants.DEFAULT_IMAGE_CRF)
    emit_scalar(out, "kLtx2Ltx24ImageCrf", constants.LTX_2_4_IMAGE_CRF)
    emit_scalar(out, "kLtx2VideoLatentChannels", constants.VIDEO_LATENT_CHANNELS)
    out.write("\n")

    def emit_params(tag: str, params) -> None:
        emit_scalar(out, f"kLtx2Params{tag}Seed", params.seed)
        emit_scalar(out, f"kLtx2Params{tag}Stage1Height", params.stage_1_height)
        emit_scalar(out, f"kLtx2Params{tag}Stage1Width", params.stage_1_width)
        emit_scalar(out, f"kLtx2Params{tag}Stage2Height", params.stage_2_height)
        emit_scalar(out, f"kLtx2Params{tag}Stage2Width", params.stage_2_width)
        emit_scalar(out, f"kLtx2Params{tag}NumFrames", params.num_frames)
        emit_double(out, f"kLtx2Params{tag}FrameRate", params.frame_rate)
        emit_scalar(out, f"kLtx2Params{tag}NumInferenceSteps", params.num_inference_steps)
        emit_scalar(out, f"kLtx2Params{tag}ImageCrf", params.default_image_crf)
        for stream in ("video", "audio"):
            guider = getattr(params, f"{stream}_guider_params")
            name = f"kLtx2Params{tag}{stream.capitalize()}"
            emit_double(out, f"{name}CfgScale", guider.cfg_scale)
            emit_double(out, f"{name}StgScale", guider.stg_scale)
            emit_double(out, f"{name}RescaleScale", guider.rescale_scale)
            emit_double(out, f"{name}ModalityScale", guider.modality_scale)
            emit_scalar(out, f"{name}SkipStep", guider.skip_step)
            # An EMPTY `stg_blocks` is a real configuration (LTX_2_3_HQ_PARAMS
            # turns STG off, constants.py:99-114) and C++ has no zero-length
            # array, so the count travels separately and the array is padded.
            emit_scalar(out, f"{name}StgBlockCount", len(guider.stg_blocks))
            emit_i64(out, f"{name}StgBlocks", guider.stg_blocks or [-1])
        out.write("\n")

    emit_params("Ltx2", constants.LTX_2_PARAMS)
    emit_params("Ltx23", constants.LTX_2_3_PARAMS)
    emit_params("Ltx24", constants.LTX_2_4_PARAMS)
    emit_params("Ltx23Hq", constants.LTX_2_3_HQ_PARAMS)

    # `_PARAMS_SINCE_VERSION` (constants.py:130-133) is the newest-generation-at-
    # or-below rule that gives LTX-2.5 its parameters: (2,5) >= (2,4), so 2.5
    # inherits LTX_2_4_PARAMS. That inheritance is the whole reason a 2.5 recipe
    # can be written at all, so the rule is emitted, not assumed.
    since = constants._PARAMS_SINCE_VERSION  # noqa: SLF001 - upstream's own name
    emit_scalar(out, "kLtx2ParamsSinceCount", len(since))
    emit_i64(out, "kLtx2ParamsSinceMajor", [v[0] for v, _ in since])
    emit_i64(out, "kLtx2ParamsSinceMinor", [v[1] for v, _ in since])
    emit_i64(out, "kLtx2ParamsSinceSteps", [p.num_inference_steps for _, p in since])
    emit_i64(out, "kLtx2ParamsSinceCrf", [p.default_image_crf for _, p in since])

    # parse_model_version + detect_params, applied to the versions this port keys
    # on. Executing the real function is what makes the C++ mirror gateable.
    from ltx_core.loader.helpers import parse_model_version  # noqa: PLC0415

    versions = ("", "2", "2.3", "2.4", "2.5", "2.5.1", "2.4-rc2", "2.3.rc1", "3", "banana")
    parsed = [parse_model_version(v.replace("-", ".")) for v in versions]
    out.write("inline constexpr const char* kLtx2VersionStrings[] = {\n")
    for v in versions:
        out.write(f'    "{v}",\n')
    out.write("};\n")
    emit_scalar(out, "kLtx2VersionCount", len(versions))
    emit_i64(out, "kLtx2VersionParsedLen", [len(p) for p in parsed])
    emit_i64(out, "kLtx2VersionParsedMajor", [p[0] if len(p) > 0 else -1 for p in parsed])
    emit_i64(out, "kLtx2VersionParsedMinor", [p[1] if len(p) > 1 else -1 for p in parsed])
    emit_i64(
        out,
        "kLtx2VersionResolvedSteps",
        [_detect_params_for(constants, p).num_inference_steps for p in parsed],
    )
    emit_i64(
        out,
        "kLtx2VersionResolvedCrf",
        [_detect_params_for(constants, p).default_image_crf for p in parsed],
    )
    out.write("\n")

    # distilled.py's ancestral-sampler rule. THIS is the one thing that separates
    # a 2.5 distilled run from a 2.0 one: stage 1 samples with the ancestral (SDE)
    # Euler step instead of the deterministic one (distilled.py:60-84, 170-185).
    literals = _module_literals(
        pipelines / "distilled.py",
        ("ANCESTRAL_SAMPLER_SINCE_VERSION", "ANCESTRAL_ETA", "ANCESTRAL_S_NOISE",
         "ANCESTRAL_NOISE_SEED_OFFSET"),
    )
    since_version = ast.literal_eval(literals["ANCESTRAL_SAMPLER_SINCE_VERSION"])
    emit_i64(out, "kLtx2AncestralSinceVersion", list(since_version))
    emit_double(out, "kLtx2AncestralEta",
                ast.literal_eval(literals["ANCESTRAL_ETA"]))
    emit_double(out, "kLtx2AncestralSNoise",
                ast.literal_eval(literals["ANCESTRAL_S_NOISE"]))
    emit_scalar(out, "kLtx2AncestralNoiseSeedOffset",
                ast.literal_eval(literals["ANCESTRAL_NOISE_SEED_OFFSET"]))
    emit_i64(
        out,
        "kLtx2VersionUsesAncestral",
        [1 if tuple(p) >= since_version else 0 for p in parsed],
    )
    out.write("\n")

    # --- vLLM-Omni, the BINDING oracle's serving model (spec section 3) --------
    omni = omni_root / "vllm_omni" / "diffusion" / "models" / "ltx2" / "ltx2_recipes.py"
    omni_literals = _module_literals(
        omni,
        ("LTX_DISTILLED_SIGMAS", "LTX_STAGE_2_DISTILLED_SIGMAS",
         "LTX_DEFAULT_NEGATIVE_PROMPT", "_PIPELINE_RECIPES"),
    )
    omni_distilled = list(ast.literal_eval(omni_literals["LTX_DISTILLED_SIGMAS"]))
    omni_stage2 = list(ast.literal_eval(omni_literals["LTX_STAGE_2_DISTILLED_SIGMAS"]))
    emit_f32(out, "kLtx2OmniDistilledSigmas", omni_distilled)
    emit_f32(out, "kLtx2OmniStage2DistilledSigmas", omni_stage2)
    emit_string(out, "kLtx2OmniNegativePrompt",
                ast.literal_eval(omni_literals["LTX_DEFAULT_NEGATIVE_PROMPT"]))
    # The two references DISAGREE on the default negative prompt: Lightricks'
    # carries five leading tags ("has_subtitles, has_blurbox, transition from
    # black, transition to black, speech_ending_short, ") that vLLM-Omni's lacks.
    # Recorded as a value, per spec section 3 ("where they disagree, the
    # disagreement is the finding"), instead of one being quietly preferred.
    emit_bool(out, "kLtx2NegativePromptsAgree",
              constants.DEFAULT_NEGATIVE_PROMPT
              == ast.literal_eval(omni_literals["LTX_DEFAULT_NEGATIVE_PROMPT"]))

    keys = [ast.literal_eval(k) for k in omni_literals["_PIPELINE_RECIPES"].keys]
    emit_scalar(out, "kLtx2OmniRecipeKeyCount", len(keys))
    out.write("inline constexpr const char* kLtx2OmniRecipeKinds[] = {\n")
    for kind, _ in keys:
        out.write(f'    "{kind}",\n')
    out.write("};\n")
    out.write("inline constexpr const char* kLtx2OmniRecipeVersions[] = {\n")
    for _, version in keys:
        out.write(f'    "{version}",\n')
    out.write("};\n\n")


def _detect_params_for(constants, parsed: tuple):
    """`detect_params` (constants.py:166-179) applied to an ALREADY-parsed version."""
    for since, params in constants._PARAMS_SINCE_VERSION:  # noqa: SLF001
        if tuple(parsed) >= since:
            return params
    return constants.LTX_2_PARAMS


# ---------------------------------------------------------------------------
# Section 8 — the latent spatial upsampler (model/upsampler/)
# ---------------------------------------------------------------------------

# GroupNorm(32, mid_channels) fixes mid_channels to a multiple of 32, so 32 is the
# smallest fixture that keeps the norm's group structure intact (one channel per
# group) rather than changing it.
_UPS_IN, _UPS_MID, _UPS_BLOCKS = 6, 32, 1
_UPS_F, _UPS_H, _UPS_W = 2, 4, 6
# The TEMPORAL arm gets its own frame count. 3 is the smallest that makes the
# frame axis carry information the other two axes cannot supply: it differs from
# H (4) and W (6), so an axis mix-up is a SHAPE failure rather than a value one,
# and `2F - 1 = 5` differs from `2F = 6`, so the dropped first frame
# (model.py:113) is visible in the shape as well as in the values.
_UPS_TEMPORAL_F = 3


def section_upsampler(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.model.upsampler.model import LatentUpsampler  # noqa: PLC0415
    from ltx_core.model.upsampler.spatial_rational_resampler import (  # noqa: PLC0415
        _rational_for_scale,
    )

    section(out, "Section 8 - latent spatial upsampler (model/upsampler/)")

    emit_scalar(out, "kLtx2UpsInChannels", _UPS_IN)
    emit_scalar(out, "kLtx2UpsMidChannels", _UPS_MID)
    emit_scalar(out, "kLtx2UpsBlocksPerStage", _UPS_BLOCKS)
    emit_scalar(out, "kLtx2UpsFrames", _UPS_F)
    emit_scalar(out, "kLtx2UpsHeight", _UPS_H)
    emit_scalar(out, "kLtx2UpsWidth", _UPS_W)
    # ResBlock and LatentUpsampler both hardcode GroupNorm(32, ...)
    # (res_block.py:24,26; model.py:50), so the group count is a pinned constant
    # and not a config key a checkpoint could move.
    emit_scalar(out, "kLtx2UpsNormGroups", 32)
    # BlurDownsample's kernel_size default, READ OFF upstream's own signature
    # (blur_downsample.py:14). `SpatialRationalResampler` never passes one
    # (spatial_rational_resampler.py:38), so this default IS the shipped kernel
    # width, and it silently changes the whole binomial kernel if it moves.
    import inspect  # noqa: PLC0415

    from ltx_core.model.upsampler.blur_downsample import BlurDownsample  # noqa: PLC0415

    emit_scalar(
        out,
        "kLtx2UpsBlurKernelSize",
        inspect.signature(BlurDownsample.__init__).parameters["kernel_size"].default,
    )

    count = _UPS_IN * _UPS_F * _UPS_H * _UPS_W
    latent = torch.from_numpy(make("ltx2.ups.latent", count, 1.0)).reshape(
        1, _UPS_IN, _UPS_F, _UPS_H, _UPS_W
    )
    emit_f32(out, "kLtx2UpsLatent", latent.numpy())

    # `_rational_for_scale`'s supported map (spatial_rational_resampler.py:11-14)
    # decides which scales exist at all; an unsupported one RAISES upstream, which
    # is the refusal the port mirrors.
    supported = (0.75, 1.5, 2.0, 4.0)
    emit_f64(out, "kLtx2UpsRationalScales", supported)
    emit_i64(out, "kLtx2UpsRationalNum", [_rational_for_scale(s)[0] for s in supported])
    emit_i64(out, "kLtx2UpsRationalDen", [_rational_for_scale(s)[1] for s in supported])
    out.write("\n")

    arms = (
        # (tag, rational_resampler, spatial_scale)
        ("PixelShuffle", False, 2.0),
        ("Rational2", True, 2.0),
        # den > 1, so BlurDownsample actually runs its binomial kernel rather than
        # short-circuiting at stride 1 (blur_downsample.py:36-37). H and W must
        # stay divisible by 2 after the 3x upshuffle for the stride-2 conv to land
        # on the same grid upstream lands on.
        ("Rational1p5", True, 1.5),
    )
    for tag, rational, scale in arms:
        module = LatentUpsampler(
            in_channels=_UPS_IN,
            mid_channels=_UPS_MID,
            num_blocks_per_stage=_UPS_BLOCKS,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=scale,
            rational_resampler=rational,
        )
        module.eval()
        manifest = fill_module(module, f"ltx2.ups.{tag}.")
        result = module(latent)
        emit_bool(out, f"kLtx2Ups{tag}Rational", rational)
        emit_double(out, f"kLtx2Ups{tag}Scale", scale)
        emit_i64(out, f"kLtx2Ups{tag}OutShape", list(result.shape))
        emit_manifest(out, f"kLtx2Ups{tag}Param", manifest)
        emit_f32(out, f"kLtx2Ups{tag}Golden", result.numpy())

    # The binomial anti-alias kernel is COMPUTED at construction, never loaded
    # (blur_downsample.py:29-33), so both sides must build it independently — the
    # same rule the VAE's kaiser-sinc filters follow.
    from ltx_core.model.upsampler.blur_downsample import BlurDownsample  # noqa: PLC0415

    for size in (3, 5, 7):
        blur = BlurDownsample(dims=2, stride=2, kernel_size=size)
        emit_f32(out, f"kLtx2UpsBlurKernel{size}", blur.kernel.numpy())

    # ---- the TEMPORAL x2 arm (model.py:68-71, 109-113) --------------------
    #
    # `spatial_upsample=False, temporal_upsample=True` selects
    # `Conv3d(mid, 2*mid, k=3, p=1)` + `PixelShuffleND(1)`, and then `forward`
    # DROPS THE FIRST FRAME. Everything else in the class is the same modules the
    # three spatial arms above already gate.
    from ltx_core.model.upsampler.pixel_shuffle import PixelShuffleND  # noqa: PLC0415

    emit_scalar(out, "kLtx2UpsTemporalFrames", _UPS_TEMPORAL_F)
    # `PixelShuffleND.__init__`'s `upscale_factors` default (pixel_shuffle.py:25),
    # READ OFF upstream's own signature. No construction site passes one, so
    # element 0 IS the shipped temporal factor and a change to it silently
    # changes how many frames come out.
    emit_scalar(
        out,
        "kLtx2UpsTemporalFactor",
        inspect.signature(PixelShuffleND.__init__).parameters["upscale_factors"].default[0],
    )
    tcount = _UPS_IN * _UPS_TEMPORAL_F * _UPS_H * _UPS_W
    tlatent = torch.from_numpy(make("ltx2.ups.temporal.latent", tcount, 1.0)).reshape(
        1, _UPS_IN, _UPS_TEMPORAL_F, _UPS_H, _UPS_W
    )
    emit_f32(out, "kLtx2UpsTemporalLatent", tlatent.numpy())

    temporal = LatentUpsampler(
        in_channels=_UPS_IN,
        mid_channels=_UPS_MID,
        num_blocks_per_stage=_UPS_BLOCKS,
        dims=3,
        spatial_upsample=False,
        temporal_upsample=True,
        spatial_scale=2.0,
        rational_resampler=False,
    )
    temporal.eval()
    tmanifest = fill_module(temporal, "ltx2.ups.Temporal.")
    tresult = temporal(tlatent)
    emit_i64(out, "kLtx2UpsTemporalOutShape", list(tresult.shape))
    emit_manifest(out, "kLtx2UpsTemporalParam", tmanifest)
    emit_f32(out, "kLtx2UpsTemporalGolden", tresult.numpy())

    # The spatial+temporal arm stays REFUSED (model.py:55-59: `8 * mid_channels`
    # and `PixelShuffleND(3)`, a different operator). Its parameter shape is
    # emitted anyway so the C++ refusal is gated against what upstream would
    # actually build, not against a remembered description of it.
    spatiotemporal = LatentUpsampler(
        in_channels=_UPS_IN,
        mid_channels=_UPS_MID,
        num_blocks_per_stage=_UPS_BLOCKS,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=True,
    )
    emit_i64(
        out,
        "kLtx2UpsSpatiotemporalUpsamplerShape",
        list(spatiotemporal.upsampler[0].weight.shape),
    )


# ---------------------------------------------------------------------------
# Section 9 — the duration head (duration_head/duration_head.py)
# ---------------------------------------------------------------------------

_DUR_VIDEO_DIM, _DUR_AUDIO_DIM = 16, 8
_DUR_HIDDEN, _DUR_QUERIES, _DUR_HEADS, _DUR_MLP = 12, 2, 3, 10
_DUR_VIDEO_TOKENS, _DUR_AUDIO_TOKENS = 5, 3


def section_duration_head(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.duration_head.duration_head import DurationHead  # noqa: PLC0415

    section(out, "Section 9 - duration head (duration_head/duration_head.py)")

    head = DurationHead(
        video_cross_attention_dim=_DUR_VIDEO_DIM,
        audio_cross_attention_dim=_DUR_AUDIO_DIM,
        pooler_hidden_dim=_DUR_HIDDEN,
        num_queries=_DUR_QUERIES,
        num_pooler_heads=_DUR_HEADS,
        mlp_hidden=_DUR_MLP,
    )
    head.eval()
    manifest = fill_module(head, "ltx2.dur.")

    video = torch.from_numpy(
        make("ltx2.dur.video_tokens", _DUR_VIDEO_TOKENS * _DUR_VIDEO_DIM, 1.0)
    ).reshape(1, _DUR_VIDEO_TOKENS, _DUR_VIDEO_DIM)
    audio = torch.from_numpy(
        make("ltx2.dur.audio_tokens", _DUR_AUDIO_TOKENS * _DUR_AUDIO_DIM, 1.0)
    ).reshape(1, _DUR_AUDIO_TOKENS, _DUR_AUDIO_DIM)

    emit_scalar(out, "kLtx2DurVideoDim", _DUR_VIDEO_DIM)
    emit_scalar(out, "kLtx2DurAudioDim", _DUR_AUDIO_DIM)
    emit_scalar(out, "kLtx2DurHidden", _DUR_HIDDEN)
    emit_scalar(out, "kLtx2DurQueries", _DUR_QUERIES)
    emit_scalar(out, "kLtx2DurHeads", _DUR_HEADS)
    emit_scalar(out, "kLtx2DurMlpHidden", _DUR_MLP)
    emit_scalar(out, "kLtx2DurVideoTokens", _DUR_VIDEO_TOKENS)
    emit_scalar(out, "kLtx2DurAudioTokens", _DUR_AUDIO_TOKENS)
    emit_f32(out, "kLtx2DurVideoInput", video.numpy())
    emit_f32(out, "kLtx2DurAudioInput", audio.numpy())
    emit_manifest(out, "kLtx2DurParam", manifest)

    # Both streams, then each alone. The concat is along the TOKEN axis
    # (duration_head.py:113), so a port that concatenated along the feature axis
    # would still produce a finite duration for the both-streams arm alone.
    emit_f32(out, "kLtx2DurBothGolden", head(video, audio).numpy())
    emit_f32(out, "kLtx2DurVideoOnlyGolden", head(video_tokens=video).numpy())
    emit_f32(out, "kLtx2DurAudioOnlyGolden", head(audio_tokens=audio).numpy())
    # The pooled attention output, before the MLP, so a pooler defect localizes
    # instead of arriving as one wrong scalar.
    projected = torch.cat(
        [
            head.video_input_proj(video) + head.video_modality_emb,
            head.audio_input_proj(audio) + head.audio_modality_emb,
        ],
        dim=1,
    )
    emit_f32(out, "kLtx2DurProjectedGolden", projected.numpy())
    emit_f32(out, "kLtx2DurPooledGolden", head.attention_pooler(projected).numpy())

    # AN INVARIANCE OF UPSTREAM'S OWN MODULE, measured rather than assumed, and
    # emitted so it is gated rather than left as a silent hole.
    #
    # A mutation that REVERSED the token-axis concat left every golden in this
    # file green. The reason is not a weak fixture: AttentionPooler is
    # cross-attention with no mask and no positional encoding over the token axis
    # (duration_head.py:45-49), so it is PERMUTATION INVARIANT and upstream cannot
    # distinguish the two orders either. Measured on upstream: a reversed concat
    # and a random permutation both move the pooled output by 2.98e-08, i.e. f32
    # reduction-order noise, while giving the audio stream the VIDEO modality
    # embedding moves it by 4.80e-03.
    #
    # So what tags the two streams is the modality EMBEDDING, not the order, and
    # that is what the gate must hold. Both goldens are emitted: the permuted pool
    # (which must MATCH) and the mis-tagged one (which must NOT).
    reversed_tokens = torch.cat(
        [
            head.audio_input_proj(audio) + head.audio_modality_emb,
            head.video_input_proj(video) + head.video_modality_emb,
        ],
        dim=1,
    )
    emit_f32(out, "kLtx2DurPooledReversedGolden",
             head.attention_pooler(reversed_tokens).numpy())
    mistagged = torch.cat(
        [
            head.video_input_proj(video) + head.video_modality_emb,
            head.audio_input_proj(audio) + head.video_modality_emb,
        ],
        dim=1,
    )
    emit_f32(out, "kLtx2DurPooledMistaggedGolden", head.attention_pooler(mistagged).numpy())


# ---------------------------------------------------------------------------
# Section 10 — Embeddings1DConnector (text_encoders/gemma/embeddings_connector.py)
# ---------------------------------------------------------------------------

_CONN_HEADS, _CONN_HEAD_DIM, _CONN_LAYERS = 3, 8, 2
_CONN_REGISTERS, _CONN_SEQ, _CONN_BATCH = 4, 8, 2


def section_connector(out) -> None:
    import torch  # noqa: PLC0415
    from ltx_core.model.transformer.rope import LTXRopeType  # noqa: PLC0415
    from ltx_core.text_encoders.gemma.embeddings_connector import (  # noqa: PLC0415
        Embeddings1DConnector,
    )

    section(out, "Section 10 - Embeddings1DConnector (text_encoders/gemma/)")

    inner = _CONN_HEADS * _CONN_HEAD_DIM
    emit_scalar(out, "kLtx2ConnHeads", _CONN_HEADS)
    emit_scalar(out, "kLtx2ConnHeadDim", _CONN_HEAD_DIM)
    emit_scalar(out, "kLtx2ConnLayers", _CONN_LAYERS)
    emit_scalar(out, "kLtx2ConnInnerDim", inner)
    emit_scalar(out, "kLtx2ConnRegisters", _CONN_REGISTERS)
    emit_scalar(out, "kLtx2ConnSeq", _CONN_SEQ)
    emit_scalar(out, "kLtx2ConnBatch", _CONN_BATCH)
    emit_double(out, "kLtx2ConnTheta", 10000.0)

    # The rms_norm eps, READ OFF upstream's own signature (utils.py:7) rather than
    # retyped. Spec §7.0(a): this is the invisible-constant class — the fixture's
    # rows are never near-zero, so the value comparison alone accepts a 100x
    # change. The C++ header says it is "pinned here"; this is what makes that
    # true, and it moves if upstream moves.
    import inspect  # noqa: PLC0415

    from ltx_core.utils import rms_norm  # noqa: PLC0415

    emit_double(
        out, "kLtx2ConnRmsNormEps", inspect.signature(rms_norm).parameters["eps"].default
    )

    count = _CONN_BATCH * _CONN_SEQ * inner
    hidden = torch.from_numpy(make("ltx2.conn.hidden", count, 1.0)).reshape(
        _CONN_BATCH, _CONN_SEQ, inner
    )
    emit_f32(out, "kLtx2ConnHidden", hidden.numpy())

    # The additive mask upstream's own preprocessor produces: 0 for a kept token,
    # -finfo(f32).max for a padded one (transformer_args.py:199-206). Row 0 keeps
    # 5 of 8, row 1 keeps all 8, so the register substitution is exercised on one
    # row and inert on the other.
    keep = np.ones((_CONN_BATCH, _CONN_SEQ), dtype=np.int32)
    keep[0, 5:] = 0
    mask = torch.zeros(_CONN_BATCH, 1, 1, _CONN_SEQ, dtype=torch.float32)
    mask[0, 0, 0, 5:] = -torch.finfo(torch.float32).max
    emit_i64(out, "kLtx2ConnKeep", keep)

    for tag, rope_type, double_precision, registers, gated, ff_bias in (
        ("Split", LTXRopeType.SPLIT, False, _CONN_REGISTERS, False, True),
        ("Interleaved", LTXRopeType.INTERLEAVED, False, _CONN_REGISTERS, False, True),
        ("Float64", LTXRopeType.SPLIT, True, _CONN_REGISTERS, False, True),
        # num_learnable_registers=None disables the substitution entirely and the
        # mask survives into every attention (embeddings_connector.py:167-170).
        ("NoRegisters", LTXRopeType.SPLIT, False, None, False, True),
        # The two connector config keys a 2.5 checkpoint can flip
        # (embeddings_connector.py:216-217).
        ("GatedNoBias", LTXRopeType.SPLIT, False, _CONN_REGISTERS, True, False),
    ):
        module = Embeddings1DConnector(
            attention_head_dim=_CONN_HEAD_DIM,
            num_attention_heads=_CONN_HEADS,
            num_layers=_CONN_LAYERS,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[1],
            num_learnable_registers=registers,
            rope_type=rope_type,
            double_precision_rope=double_precision,
            apply_gated_attention=gated,
            ff_bias=ff_bias,
        )
        module.eval()
        manifest = fill_module(module, f"ltx2.conn.{tag}.")
        hidden_states, out_mask = module(hidden.clone(), mask.clone())

        emit_bool(out, f"kLtx2Conn{tag}Interleaved", rope_type == LTXRopeType.INTERLEAVED)
        emit_bool(out, f"kLtx2Conn{tag}DoublePrecision", double_precision)
        emit_scalar(out, f"kLtx2Conn{tag}Registers", registers if registers else 0)
        emit_bool(out, f"kLtx2Conn{tag}Gated", gated)
        emit_bool(out, f"kLtx2Conn{tag}FfBias", ff_bias)
        emit_manifest(out, f"kLtx2Conn{tag}Param", manifest)
        emit_f32(out, f"kLtx2Conn{tag}Golden", hidden_states.numpy())
        emit_f32(out, f"kLtx2Conn{tag}MaskGolden", out_mask.numpy())

        if registers:
            # The register substitution on its own, so a defect there localizes
            # rather than arriving as a wrong final tensor. `learnable_registers`
            # is a BFLOAT16 parameter (embeddings_connector.py:135-137); this
            # golden is what proves the C++ side rounds to bf16 too instead of
            # carrying a wider value the final comparison would absorb.
            replaced, zeroed = module._replace_padded_with_learnable_registers(  # noqa: SLF001
                hidden.clone(), mask.clone()
            )
            emit_f32(out, f"kLtx2Conn{tag}RegistersGolden",
                     module.learnable_registers.float().numpy())
            emit_f32(out, f"kLtx2Conn{tag}ReplacedGolden", replaced.numpy())
            emit_f32(out, f"kLtx2Conn{tag}ZeroedMaskGolden", zeroed.numpy())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_upstream(root: Path) -> Path:
    """Import `ltx_core` BY PATH from `root`, and prove that is what resolved."""
    src = root / "packages" / "ltx-core" / "src"
    if not (src / "ltx_core" / "components" / "schedulers.py").is_file():
        raise SystemExit(f"no ltx_core under {src}; point --ltx2 at a Lightricks/LTX-2 checkout")
    sys.path.insert(0, str(src))
    import ltx_core  # noqa: PLC0415

    resolved = Path(ltx_core.__file__).resolve()
    if not resolved.is_relative_to(src.resolve()):
        raise SystemExit(
            f"ltx_core resolved to {resolved}, which is NOT under the checkout at {src}. "
            "Refusing to generate goldens from an oracle this script did not choose."
        )
    return src


def upstream_revision(root: Path, label: str) -> str:
    """The exact upstream tree these goldens were produced from.

    A DIRTY tree is refused outright. A revision anchor is only worth having if it
    describes the code that ran, and `rev-parse` on a tree with uncommitted edits
    reports a clean SHA for goldens that SHA does not produce — which is worse
    than no anchor, because it survives a bisect and misdirects it.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"cannot read the {label} revision at {root}: {exc}. These goldens are only "
            "interpretable against a known upstream tree; refusing to emit an unanchored one."
        ) from exc
    if status:
        raise SystemExit(
            f"{label} checkout at {root} is DIRTY:\n{status}\n"
            "Refusing to stamp a clean revision on goldens that tree does not describe."
        )
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path,
                        help="a checkout of Lightricks/LTX-2 (the repo root)")
    parser.add_argument("--vllm-omni", required=True, type=Path,
                        help="a checkout of vllm-project/vllm-omni (the repo root)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    root = args.ltx2.expanduser().resolve()
    omni_root = args.vllm_omni.expanduser().resolve()
    load_upstream(root)
    revision = upstream_revision(root, "LTX-2")
    omni_revision = upstream_revision(omni_root, "vllm-omni")

    import torch

    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            "// GENERATED by scripts/gen-ltx2-pipeline-goldens.py — DO NOT EDIT BY HAND.\n"
            "//\n"
            "// LTX-2.5 PIPELINE goldens (phase L5), produced by executing the UPSTREAM\n"
            "// ltx_core modules at reduced dimensions on CPU and by reading the recipe\n"
            "// constants out of ltx-pipelines and vLLM-Omni. Weights and inputs come from\n"
            "// the shared deterministic stream, so no weight byte is checked in.\n"
            "// Regenerate with:\n"
            "//   python3 scripts/gen-ltx2-pipeline-goldens.py --ltx2 <LTX-2 checkout>\n"
            "//       --vllm-omni <vllm-omni checkout>\n"
            "//       --out tests/vllm/models/ltx2_pipeline_goldens.inc\n"
            "//\n"
            f"// Upstream revision (Lightricks/LTX-2): {revision}\n"
            f"// Upstream revision (vllm-omni):        {omni_revision}\n"
            "//\n"
            "// See .agents/specs/ltx-2-5.md section 7 for why this is the gate, and\n"
            "// section 7.0 for why BOTH the identity assertion and these SHAs are here.\n"
            "#pragma once\n\n#include <cstdint>\n\nnamespace vllm_test {\n\n"
            "// The upstream trees these numbers came from. The suite asserts both equal\n"
            "// the SHAs it pins, so regenerating against a DIFFERENT checkout fails the\n"
            "// gate instead of silently replacing the oracle. The generator additionally\n"
            "// REFUSES a dirty tree, so a SHA here always describes the code that ran.\n"
            f'inline constexpr const char* kLtx2PipelineUpstreamRevision = "{revision}";\n'
            f'inline constexpr const char* kLtx2OmniUpstreamRevision = "{omni_revision}";\n\n'
        )
        section_schedulers(out)
        section_noiser(out)
        section_diffusion_steps(out)
        section_guiders(out)
        section_perturbations(out)
        section_patchifiers(out)
        section_recipes(out, root, omni_root)
        section_upsampler(out)
        section_duration_head(out)
        section_connector(out)
        out.write("}  // namespace vllm_test\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
