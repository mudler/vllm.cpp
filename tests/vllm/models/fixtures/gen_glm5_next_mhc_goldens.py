#!/usr/bin/env python3
"""Regenerate `glm5_next_mhc_goldens.inc` by RUNNING the reference oracle.

Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation W4, issue #2098,
`.agents/specs/glm5-next-flash.md` §W4.

WHAT THE ORACLE IS. `transformers` **v5.16.1**, the lane revision this row cites
(W0, #2096, owns writing it into `.agents/oracles/transformers.md`). vLLM
registers no `glm5_next` at our parity pin `555967922` nor at its `main`, and
neither do vllm-omni, SGLang or llama.cpp, so under AGENTS.md "When vLLM has no
implementation" transformers is the reference for this surface.

Every golden below is the return value of an UNMODIFIED reference module called
through its own `__call__`:

  * `Glm5NextTextHyperConnection.forward` -> `(post, comb, collapsed)`
    (`modular_glm5_next.py:364-365`, a `pass` over `DeepseekV4HyperConnection`;
    the flattened body is `modeling_glm5_next.py:267-295`).
  * `Glm5NextTextHyperHead.forward` -> the stream collapse
    (`modular_glm5_next.py:368-372`; flattened `modeling_glm5_next.py:298-302`).

WHAT IS TRANSCRIBED, AND WHY IT IS NOT THE GATE. One thing: the decoder layer's
two-line residual update

    hidden = post.to(dtype).unsqueeze(-1) * hidden.unsqueeze(-2)
             + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)

(`modeling_glm5_next.py:1316-1318`, and identically at `:1325-1327`), which is
run here on the oracle's OWN `post` and `comb` rather than on recomputed ones.
It is two tensor ops with no learned state, and the values it consumes come out
of the oracle's execution. The `mhc_post` golden exists so that the C++
`MhcPost` reuse is gated on the axis it is easiest to get wrong: `comb` is
consumed TRANSPOSED, so `out[j] = sum_i comb[i, j] * residual[i]`, and the
Sinkhorn result is doubly stochastic but ASYMMETRIC, so transposing it wrongly
degrades quality silently instead of crashing.

WHY float32 AND NOT bfloat16. The reference computes the whole mHC mapping in
fp32 and casts `post` / `comb` to the activation dtype only at the mix
(`.to(dtype)` above); the C++ side here is a host f32 reference, exactly as
`deepseek_v4_mhc.cpp` is. Capturing at fp32 makes those casts no-ops, which is
honest for what this file gates and leaves the bf16 storage rounding between
steps as the named device seam it already is (`deepseek_v4_mhc.h`, "W7 seam").

WHY hc_mult 4 AND hidden 8. `hc_mult` is the published checkpoint's value and
the wave's gate names it. `hidden_size` 8 keeps `fn` at [24, 32] so the emitted
golden is readable, and no code path branches on the hidden width.

Usage:
    python3 tests/vllm/models/fixtures/gen_glm5_next_mhc_goldens.py [--out PATH]

Requires `transformers==5.16.1` and torch. The generator REFUSES to emit under
any other transformers version, because a golden captured off an unpinned
oracle is not reproducible.
"""

import argparse
import pathlib
import sys

LANE_PIN = "5.16.1"

HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
RMS_NORM_EPS = 1e-5
HIDDEN = 8
SEQ = 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).with_name("glm5_next_mhc_goldens.inc")),
    )
    args = ap.parse_args()

    import torch
    import transformers

    if transformers.__version__ != LANE_PIN:
        sys.stderr.write(
            f"REFUSING: transformers {transformers.__version__} is not the lane "
            f"revision {LANE_PIN}. A golden captured off an unpinned oracle is "
            f"not reproducible (AGENTS.md 'Pin every oracle').\n"
        )
        return 2

    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextHyperConnection,
        Glm5NextTextHyperHead,
    )

    torch.manual_seed(2098)

    cfg = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=HC_SINKHORN_ITERS,
        hc_eps=HC_EPS,
        rms_norm_eps=RMS_NORM_EPS,
    )
    assert (cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps, cfg.rms_norm_eps) == (
        HC_MULT,
        HC_SINKHORN_ITERS,
        HC_EPS,
        RMS_NORM_EPS,
    ), "the reference config did not keep this row's mHC constants"

    site = Glm5NextTextHyperConnection(cfg)
    # `fn`, `base` and `scale` are `torch.empty` in the reference constructor and
    # are filled by `_init_weights` at `from_pretrained` time, which is not run
    # here. Fill them explicitly: an uninitialised buffer makes the golden depend
    # on whatever was in that allocation.
    with torch.no_grad():
        site.fn.normal_(0.0, 0.5)
        site.base.normal_(0.0, 0.5)
        # The three scales are the per-output learned gains. Keep them distinct
        # and non-unit, or a port that drops one, or applies scale[0] to all
        # three, still passes.
        site.scale.copy_(torch.tensor([0.75, -1.25, 1.5]))
    site.eval()

    head = Glm5NextTextHyperHead()

    # [B, S, hc_mult, hidden] -- the manifold the text model threads through
    # every layer (`modeling_glm5_next.py:1477`, the `unsqueeze(2).expand` of the
    # embeddings).
    streams = torch.randn(1, SEQ, HC_MULT, HIDDEN)
    # The sublayer (attn / MLP) output the decoder layer folds back in.
    sublayer_out = torch.randn(1, SEQ, HIDDEN)

    with torch.no_grad():
        post, comb, collapsed = site(streams)
        # The decoder layer's residual update, on the oracle's own post/comb.
        mixed = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2), streams
        )
        head_out = head(streams)

    assert tuple(post.shape) == (1, SEQ, HC_MULT), post.shape
    assert tuple(comb.shape) == (1, SEQ, HC_MULT, HC_MULT), comb.shape
    assert tuple(collapsed.shape) == (1, SEQ, HIDDEN), collapsed.shape
    assert tuple(mixed.shape) == (1, SEQ, HC_MULT, HIDDEN), mixed.shape
    assert tuple(head_out.shape) == (1, SEQ, HIDDEN), head_out.shape

    out = pathlib.Path(args.out)
    out.write_text(
        _emit(
            tf_ver=transformers.__version__,
            torch_ver=torch.__version__,
            fn=site.fn,
            base=site.base,
            scale=site.scale,
            streams=streams[0],
            sublayer_out=sublayer_out[0],
            post=post[0],
            comb=comb[0],
            collapsed=collapsed[0],
            mixed=mixed[0],
            head_out=head_out[0],
        )
    )
    sys.stderr.write(f"wrote {out}\n")
    return 0


def _wrap(items, indent="    ") -> str:
    """One value per column-bounded line. `repr` on a float always emits a `.` or
    an exponent, so `1.0f` never degenerates into the invalid literal `1f`."""
    out, line = [], indent
    for it in items:
        if len(line) + len(it) + 2 > 96:
            out.append(line.rstrip())
            line = indent
        line += it + ", "
    if line.strip():
        out.append(line.rstrip().rstrip(","))
    else:
        out[-1] = out[-1].rstrip(",")
    return "\n" + "\n".join(out) + "\n"


def _floats(t) -> str:
    import torch

    flat = t.detach().to(torch.float32).reshape(-1).tolist()
    return _wrap([f"{float(v)!r}f" for v in flat])


def _emit(*, tf_ver, torch_ver, fn, base, scale, streams, sublayer_out, post,
          comb, collapsed, mixed, head_out) -> str:
    lines = [
        "// GENERATED by tests/vllm/models/fixtures/gen_glm5_next_mhc_goldens.py",
        "// -- DO NOT EDIT BY HAND.",
        "//",
        "// GLM-5.3-Flash manifold hyper-connection (mHC) goldens, row",
        "// MODEL-MM-glm5-next-glm5-next-for-conditional-generation W4, issue #2098.",
        "// Produced by RUNNING the reference oracle:",
        f"//   transformers {tf_ver}   (this row's lane revision; W0/#2096 records it)",
        f"//   torch        {torch_ver}",
        "//",
        "// `kPost`, `kComb`, `kCollapsed` are the return value of the unmodified",
        "// `Glm5NextTextHyperConnection.forward`; `kHeadOut` is the return value of",
        "// the unmodified `Glm5NextTextHyperHead.forward`, which is an UNWEIGHTED",
        "// MEAN over the stream axis and NOT DeepSeek-V4's gated collapse. `kMixed`",
        "// is the decoder layer's residual update run on the oracle's own",
        "// `post`/`comb` (modeling_glm5_next.py:1316-1318).",
        "//",
        "// Layout, row-major: kFn [(2+hc)*hc, hc*hidden], kBase [(2+hc)*hc],",
        "// kScale [3], kStreams [seq, hc, hidden], kSublayerOut [seq, hidden],",
        "// kPost [seq, hc], kComb [seq, hc, hc], kCollapsed [seq, hidden],",
        "// kMixed [seq, hc, hidden], kHeadOut [seq, hidden].",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace glm5_next_mhc_goldens {",
        "",
        f'inline constexpr const char* kOracle = "transformers {tf_ver}";',
        f"inline constexpr int64_t kHcMult = {HC_MULT};",
        f"inline constexpr int64_t kHcSinkhornIters = {HC_SINKHORN_ITERS};",
        f"inline constexpr float kHcEps = {HC_EPS!r}f;",
        f"inline constexpr float kRmsNormEps = {RMS_NORM_EPS!r}f;",
        f"inline constexpr int64_t kHidden = {HIDDEN};",
        f"inline constexpr int64_t kSeq = {SEQ};",
        "",
        f"inline constexpr float kFn[] = {{{_floats(fn)}}};",
        f"inline constexpr float kBase[] = {{{_floats(base)}}};",
        f"inline constexpr float kScale[] = {{{_floats(scale)}}};",
        f"inline constexpr float kStreams[] = {{{_floats(streams)}}};",
        f"inline constexpr float kSublayerOut[] = {{{_floats(sublayer_out)}}};",
        f"inline constexpr float kPost[] = {{{_floats(post)}}};",
        f"inline constexpr float kComb[] = {{{_floats(comb)}}};",
        f"inline constexpr float kCollapsed[] = {{{_floats(collapsed)}}};",
        f"inline constexpr float kMixed[] = {{{_floats(mixed)}}};",
        f"inline constexpr float kHeadOut[] = {{{_floats(head_out)}}};",
        "",
        "}  // namespace glm5_next_mhc_goldens",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
