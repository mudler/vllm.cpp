#!/usr/bin/env python3
"""Emit tests/vllm/models/ltx2_res2s_goldens.inc — the LTX-2.5 res_2s oracle.

Row LTX25-RES2S-LOOP, issue #921, spec .agents/specs/ltx25-res2s-loop.md.

Every number this writes is what UPSTREAM'S OWN CODE RETURNED. `phi`,
`get_res2s_coefficients`, `Res2sDiffusionStep`, `post_process_latent`,
`_get_new_noise`'s two normalization steps and
`res2s_audio_video_denoising_loop` are imported from a Lightricks/LTX-2 checkout
and run. Nothing here is transcribed from reading the source, and nothing is
recomputed by a local reimplementation — which for this module is the whole
point, because the values it is most important to pin are the ones a CORRECT
implementation gets wrong. `phi(2, -1e-10)` is 0.0 upstream, not 0.5, because
the formula cancels just outside its own 1e-10 guard; a Taylor series near zero
is numerically better and diverges from the model's runtime.

THE FOUR SUBSTITUTIONS, each one a thing this port reproduces exactly:

1. THE DENOISER. Upstream's loop takes a `Denoiser` callable and never reaches
   for a model (samplers.py:214), so a fixed quadratic stands in for the 21B
   transformer. It is quadratic and not affine on purpose: a build that
   evaluated once per step and reused the result cannot land on the same
   trajectory by luck.
2. THE NOISE DRAW. Upstream draws `torch.randn` on a seeded `torch.Generator`;
   this port has `SplitMixGaussian` and does not have that stream. The draw is
   replaced by the same deterministic pattern the C++ fixture uses, so the loop
   ARITHMETIC around the injection is gated even though the stream is not. The
   NORMALIZATION upstream applies after its draw is gated separately and
   against upstream's own code (`kLtx2Res2sNoise*` below).
3. `model_dtype`. Upstream's loop declares `torch.bfloat16` (samplers.py:221);
   this passes `torch.float32`, which is this port's model dtype on every
   LTX-2.5 host path. Recorded as a divergence in ltx2_samplers.h rather than
   hidden here.
4. TWO MEDIA-IO MODULES. `import ltx_pipelines.utils.samplers` pulls
   `ltx_core.color.hlg`, which imports PyAV, and the image path imports
   OpenImageIO. Neither is vendored here and nothing numeric touches either, so
   both are stubbed as empty modules BEFORE the import. If upstream ever routes
   a number through them this stub is what breaks, loudly, rather than a value
   silently changing.

Regenerate with:

    python3 scripts/gen-ltx2-res2s-goldens.py --ltx2 /path/to/LTX-2 \\
        --out tests/vllm/models/ltx2_res2s_goldens.inc

and diff. The committed file is what this script emits at `fd4ded7f`; a
difference is either an upstream change or a defect in one of the two.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import struct
import subprocess
import sys
import types

PIN = "fd4ded7f"

# ─── the fixture, which is INPUT and therefore stated here rather than read ───
#
# Six elements, because the loop's arithmetic is elementwise and six is enough
# to carry a mask that is not all ones. The mask and the clean latent matter:
# with an all-ones mask `post_process_latent` is the identity and a build that
# dropped the blend passes. Three of the six positions are pinned to the clean
# latent, so a dropped blend fails at three positions rather than nowhere.
LATENT_COUNT = 6
VIDEO_0 = [i / 6.0 for i in range(LATENT_COUNT)]
AUDIO_0 = [0.5 - i / 12.0 for i in range(LATENT_COUNT)]
MASK = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
CLEAN = [-0.3, 0.2, 0.7, -0.1, 0.4, 0.05]

# The raw vector `_get_new_noise`'s normalization is measured on. An INPUT, not
# an upstream output: what upstream produces from it is the golden below it. The
# expression is written out rather than the six doubles being pasted in, because
# a normalization golden is only meaningful beside the exact bits it was taken
# over, and `-1/3` and `((0*13+5)%17)/3 - 2` are the same number to fifteen
# digits and not to seventeen.
NOISE_RAW = [((i * 13 + 5) % 17) / 3.0 - 2.0 for i in range(6)]

PHI_Z = [
    0.0, -1e-12, -1e-11, -1e-10, -1e-09, -1e-08, -1e-06,
    -0.001, -0.125, -0.25, -0.5, -1.0, -2.0, -5.0,
]
COEFF_H = [1e-12, 1e-10, 1e-08, 1e-06, 0.01, 0.125, 0.25, 0.5, 1.0, 3.0, 7.0]

# name -> (sigmas, eta). Each forces one branch of the bong guard
# `bongmath and h < 0.5 and sigma > 0.03` (samplers.py:357) and says which:
#   BongOn          every h < 0.5 AND every sigma > 0.03  -> refines
#   BongOffByH      every h >= 0.5, every sigma > 0.03    -> off by h alone
#   BongOffBySigma  every h < 0.5, every sigma <= 0.03    -> off by sigma alone
#   TerminalZero    a schedule ending at 0                -> the injected 0.0011
#   Eta1            BongOn's schedule at eta 1.0          -> separates the loop's
#                                                            eta from the substep's
#                                                            pinned 0.5 (:273-274)
FIXTURES = [
    ("BongOn", [0.9, 0.8, 0.7, 0.62], 0.5),
    ("BongOffByH", [0.9, 0.5, 0.25, 0.12], 0.5),
    ("BongOffBySigma", [0.03, 0.028, 0.026, 0.025], 0.5),
    ("TerminalZero", [1.0, 0.75, 0.5, 0.25, 0.0], 0.5),
    ("Eta1", [0.9, 0.8, 0.7, 0.62], 1.0),
]


def git_revision(root: pathlib.Path) -> str:
    """The checkout's SHA, and REFUSE a dirty tree.

    A SHA that does not describe the code that ran is worse than no SHA: it
    reads as a pin while the oracle is whatever was in the working tree.
    """
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"{root} has uncommitted changes; the goldens would carry a SHA that "
            f"does not describe the code that produced them:\n{dirty}"
        )
    return head


def load_upstream(root: pathlib.Path):
    """Import upstream's own modules, with the two media-IO stubs."""
    for pkg in ("ltx-pipelines", "ltx-core"):
        src = root / "packages" / pkg / "src"
        if not src.is_dir():
            raise SystemExit(f"{src} is not a directory; is --ltx2 an LTX-2 checkout?")
        sys.path.insert(0, str(src))
    for name in ("av", "av.video", "av.audio", "OpenImageIO"):
        sys.modules.setdefault(name, types.ModuleType(name))

    import torch  # noqa: PLC0415
    from ltx_core.components.diffusion_steps import Res2sDiffusionStep  # noqa: PLC0415
    from ltx_core.types import LatentState  # noqa: PLC0415
    from ltx_pipelines.utils import samplers  # noqa: PLC0415
    from ltx_pipelines.utils.res2s import get_res2s_coefficients, phi  # noqa: PLC0415
    from ltx_pipelines.utils.types import DenoisedLatentResult  # noqa: PLC0415

    return types.SimpleNamespace(
        torch=torch,
        Res2sDiffusionStep=Res2sDiffusionStep,
        LatentState=LatentState,
        samplers=samplers,
        phi=phi,
        get_res2s_coefficients=get_res2s_coefficients,
        DenoisedLatentResult=DenoisedLatentResult,
    )


def make_state(up, values, *, reversed_conditioning: bool):
    """One modality's `LatentState`.

    THE AUDIO SIDE REVERSES THE MASK AND THE CLEAN LATENT. Not decoration: with
    both modalities carrying the same conditioning, a build that fed one
    stream's mask or clean latent to the other produces the identical result and
    nothing says so. Reversed, the two disagree at four of six positions.
    """
    torch = up.torch
    mask = list(reversed(MASK)) if reversed_conditioning else MASK
    clean = list(reversed(CLEAN)) if reversed_conditioning else CLEAN
    return up.LatentState(
        latent=torch.tensor([values], dtype=torch.float32),
        denoise_mask=torch.tensor([mask], dtype=torch.float32),
        positions=torch.zeros(1, LATENT_COUNT, dtype=torch.float32),
        clean_latent=torch.tensor([clean], dtype=torch.float32),
    )


class QuadraticDenoiser:
    """The stand-in for the 21B transformer, in upstream's `Denoiser` shape.

    `0.5x + 0.25 - 0.125x^2` on video and `-0.25x + 0.1 - 0.0625x^2` on audio,
    in the model dtype and in that operation order, mirrored exactly by the C++
    fixture in tests/vllm/models/test_ltx2_pipeline.cpp. The two modalities get
    DIFFERENT functions so a build that fed one stream's state to the other is
    visible rather than symmetric.
    """

    def __init__(self, up):
        self.up = up
        self.eval_sigmas: list[float] = []
        self.eval_step_indices: list[int] = []

    def __call__(self, transformer, video_state=None, audio_state=None, sigmas=None, step_index=0):
        # `sigmas[step_index]` is what every upstream denoiser reduces the pair
        # to (denoisers.py:237), so the record below is the sigma the model saw.
        self.eval_sigmas.append(float(sigmas[step_index].item()))
        self.eval_step_indices.append(int(step_index))
        result_v = None
        result_a = None
        if video_state is not None:
            v = video_state.latent
            result_v = self.up.DenoisedLatentResult(denoised=0.5 * v + 0.25 - 0.125 * (v * v))
        if audio_state is not None:
            a = audio_state.latent
            result_a = self.up.DenoisedLatentResult(denoised=-0.25 * a + 0.1 - 0.0625 * (a * a))
        return result_v, result_a


class PatternNoise:
    """The stand-in for `torch.randn`, mirroring the C++ fixture's hook.

    STATEFUL PER GENERATOR, because upstream's generators advance. Within one
    step the video injection and the audio injection are two draws from the SAME
    generator, so they receive DIFFERENT tensors and the ORDER of the two calls
    decides which modality gets which. A stateless stand-in makes swapping the
    two injections invisible, which is exactly the mutation that survived until
    this was made stateful.

    The two generators are told apart by their seed rather than by identity,
    because the loop constructs them itself (samplers.py:267-268): the step
    generator is seeded `noise_seed` and the substep generator
    `noise_seed + 10000`.
    """

    def __init__(self, up, noise_seed: int):
        self.up = up
        self.step_seed = up.torch.Generator().manual_seed(noise_seed).initial_seed()
        self.draws = {False: 0, True: 0}

    def __call__(self, x, generator):
        substep = generator.initial_seed() != self.step_seed
        draw = self.draws[substep]
        count = x.numel()
        values = [
            ((i * 7 + 3 + (1 if substep else 0) + 13 * draw) % 11) / 5.0 - 1.0
            for i in range(count)
        ]
        self.draws[substep] = draw + 1
        return self.up.torch.tensor(values, dtype=self.up.torch.float64).reshape(x.shape)


def run_loop(up, sigmas, eta, bongmath):
    """One `res2s_audio_video_denoising_loop` call, at this port's model dtype."""
    torch = up.torch
    denoiser = QuadraticDenoiser(up)
    video_out, audio_out = up.samplers.res2s_audio_video_denoising_loop(
        sigmas=torch.tensor(sigmas, dtype=torch.float32),
        video_state=make_state(up, VIDEO_0, reversed_conditioning=False),
        audio_state=make_state(up, AUDIO_0, reversed_conditioning=True),
        stepper=up.Res2sDiffusionStep(),
        transformer=None,
        denoiser=denoiser,
        eta=eta,
        bongmath=bongmath,
        new_noise_fn=PatternNoise(up, noise_seed=-1),
        model_dtype=torch.float32,
    )
    return {
        "video": video_out.latent.reshape(-1).tolist(),
        "audio": audio_out.latent.reshape(-1).tolist(),
        "eval_sigmas": denoiser.eval_sigmas,
        "eval_step_indices": denoiser.eval_step_indices,
        "evaluations": len(denoiser.eval_sigmas),
    }


# ─── emission ─────────────────────────────────────────────────────────────────


def f32(value: float) -> str:
    """A float32 literal in the shortest form that round-trips at 9 digits.

    ROUNDED THROUGH float32 FIRST. A Python float is a double, so emitting
    `0.9` where the C++ array holds `0.899999976f` would put a value in the
    header that the compiler then rounds to something else — a golden that
    describes the double the generator held rather than the float the test
    compares.
    """
    value = struct.unpack("f", struct.pack("f", value))[0]
    text = f"{value:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text + "f"


def f64(value: float) -> str:
    return repr(float(value))


def array(kind: str, name: str, values, fmt) -> str:
    body = [fmt(v) for v in values]
    lines = []
    for i in range(0, len(body), 3):
        lines.append("    " + ", ".join(body[i:i + 3]))
    return f"inline constexpr {kind} {name}[] = {{\n" + ",\n".join(lines) + "};\n"


def emit(up, revision: str) -> str:
    torch = up.torch
    out = []
    out.append(
        f"// GENERATED from Lightricks/LTX-2 @ {revision[:8]} by\n"
        "// scripts/gen-ltx2-res2s-goldens.py. Do not hand-edit.\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace vllm_test {\n\n"
    )

    out.append(
        "// res2s.py:4-22. `phi(j, z)` at j = 1 and j = 2, INCLUDING the small-z\n"
        "// cliff: the guard is `abs(z) < 1e-10` and outside it the formula\n"
        "// cancels catastrophically, so upstream's own phi2(-1e-10) is 0.0 and\n"
        "// phi2(-1e-8) is 1.1102230246251563. These are upstream's values, not a\n"
        "// series expansion's, and a 'better' port fails here.\n"
    )
    out.append(array("double", "kLtx2PhiZ", PHI_Z, f64))
    out.append(array("double", "kLtx2Phi1", [up.phi(1, z) for z in PHI_Z], f64))
    out.append(array("double", "kLtx2Phi2", [up.phi(2, z) for z in PHI_Z], f64))
    out.append(f"inline constexpr int64_t kLtx2PhiCount = {len(PHI_Z)};\n\n")

    coeffs = [up.get_res2s_coefficients(h, {}, 0.5) for h in COEFF_H]
    out.append("// res2s.py:25-62, c2 = 0.5 (samplers.py:288).\n")
    out.append(array("double", "kLtx2Res2sCoeffH", COEFF_H, f64))
    out.append(array("double", "kLtx2Res2sCoeffA21", [c[0] for c in coeffs], f64))
    out.append(array("double", "kLtx2Res2sCoeffB1", [c[1] for c in coeffs], f64))
    out.append(array("double", "kLtx2Res2sCoeffB2", [c[2] for c in coeffs], f64))
    out.append(f"inline constexpr int64_t kLtx2Res2sCoeffCount = {len(COEFF_H)};\n\n")

    for name, sigmas, eta in FIXTURES:
        bong = run_loop(up, sigmas, eta, bongmath=True)
        nobong = run_loop(up, sigmas, eta, bongmath=False)
        hs = [
            -math.log(sigmas[i + 1] / sigmas[i])
            for i in range(len(sigmas) - 1)
            if sigmas[i + 1] != 0.0
        ]
        moved = bong["video"] != nobong["video"]
        out.append(
            f"// {name}: sigmas {sigmas}, eta {eta}, "
            f"h [{', '.join(f'{h:.6f}' for h in hs)}], "
            f"bong changed the result: {moved}\n"
        )
        out.append(f"inline constexpr double kLtx2Res2s{name}Eta = {f64(eta)};\n")
        out.append(array("float", f"kLtx2Res2s{name}Sigmas", sigmas, f32))
        out.append(
            f"inline constexpr int64_t kLtx2Res2s{name}SigmaCount = {len(sigmas)};\n"
            f"inline constexpr int64_t kLtx2Res2s{name}Evaluations = {bong['evaluations']};\n"
        )
        out.append(array("double", f"kLtx2Res2s{name}EvalSigmas", bong["eval_sigmas"], f64))
        # The `step_index` each call was handed, in call order. NOT the position
        # in this vector: upstream passes `step_idx`, then a literal 0 for the
        # substep, then `n_full_steps` for the terminal evaluation
        # (samplers.py:301, :385, :437). The denoiser reads it through
        # `should_skip_step` (guiders.py:287-291), so it decides which
        # evaluations run a forward at all on a request with `skip_step != 0`.
        out.append(
            array("int64_t", f"kLtx2Res2s{name}EvalStepIndices", bong["eval_step_indices"], str)
        )
        out.append(array("float", f"kLtx2Res2s{name}Video", bong["video"], f32))
        out.append(array("float", f"kLtx2Res2s{name}Audio", bong["audio"], f32))
        out.append(
            f"inline constexpr bool kLtx2Res2s{name}BongMoved = "
            f"{'true' if moved else 'false'};\n"
        )
        out.append(array("float", f"kLtx2Res2s{name}NoBongVideo", nobong["video"], f32))
        out.append("\n")

    out.append(f"inline constexpr int64_t kLtx2Res2sLatentCount = {LATENT_COUNT};\n")
    out.append(array("float", "kLtx2Res2sVideo0", VIDEO_0, f32))
    out.append(array("float", "kLtx2Res2sAudio0", AUDIO_0, f32))
    out.append(array("float", "kLtx2Res2sMask", MASK, f32))
    out.append(array("float", "kLtx2Res2sClean", CLEAN, f32))
    out.append("\n")

    raw = torch.tensor([NOISE_RAW], dtype=torch.float64)
    normalized = up.samplers._channelwise_normalize(  # noqa: SLF001
        (raw - raw.mean()) / raw.std()
    )
    out.append(
        "// samplers.py:160-170. `_get_new_noise` normalizes globally and then\n"
        "// channelwise; the DRAW itself is torch.randn, whose stream this port\n"
        "// does not have, so only the normalization is gated.\n"
    )
    out.append(array("double", "kLtx2Res2sNoiseRaw", NOISE_RAW, f64))
    out.append(
        array("double", "kLtx2Res2sNoiseNormalized", normalized.reshape(-1).tolist(), f64)
    )
    out.append("\n}  // namespace vllm_test\n")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltx2", required=True, type=pathlib.Path,
                    help="a Lightricks/LTX-2 checkout at the pinned revision")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    revision = git_revision(args.ltx2)
    if not revision.startswith(PIN):
        raise SystemExit(
            f"{args.ltx2} is at {revision}, not the pinned {PIN}. Advancing the pin "
            "reconciles the row's spec and every gate that reads these goldens; it is "
            "not something this generator may do silently."
        )
    up = load_upstream(args.ltx2)
    args.out.write_text(emit(up, revision))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
