#!/usr/bin/env python3
"""Regenerate `glm5_next_moe_goldens.inc` by RUNNING the reference oracle.

Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation W5, issue #2223,
`.agents/specs/glm5-next-flash.md` §W5.

WHAT THE ORACLE IS. `transformers` **v5.16.1**, the lane revision this row cites
(W0, #2096, recorded it in `.agents/oracles/transformers.md`). vLLM registers no
`glm5_next` at our parity pin `555967922` nor at its `main`, and neither do
vllm-omni, SGLang or llama.cpp, so under AGENTS.md "When vLLM has no
implementation" transformers is the reference for this surface.

Every golden below is the return value of an UNMODIFIED reference module called
through its own `__call__`:

  * `Glm5NextTextTopkRouter.forward` -> `(router_logits, topk_weights,
    topk_indices)` (`modeling_glm5_next.py:158-184`). Captured at the PUBLISHED
    288 experts / top-8 / `n_group` 1 / `topk_group` 1 / `routed_scaling_factor`
    2.5 / `norm_topk_prob` True, twice: once with the `e_score_correction_bias`
    buffer left at its constructor zeros and once with it filled, so the gate can
    prove the bias SELECTS while the unbiased score WEIGHTS.
  * `Glm5NextTextExperts._apply_gate` -> the clamped SwiGLU epilogue
    (`:137-143`), on an input that reaches PAST the limit on both sides.
  * `Glm5NextTextMLP.forward` -> the dense (non-MoE) feed-forward with its own
    clamp (`:98-105`).
  * `Glm5NextTextMoE.forward` -> the composed routed + shared block
    (`:200-207`).

WHY THE ROUTER IS AT 288 AND THE COMPOSITION IS AT 8. The router's inputs and
outputs are `[T, E]` and `[T, k]`, so 288 experts cost 288 floats a token and the
golden stays readable. The composed block's weights are `[E, 2I, H]` and
`[E, H, I]`, which at 288 experts is 41472 floats even at these toy widths -- a
half-megabyte fixture that gates nothing the 8-expert one does not. The
SELECTION is where expert count matters and that is captured at 288.

WHY THE WEIGHTS ARE FILLED EXPLICITLY. `Glm5NextTextExperts` declares
`gate_up_proj` and `down_proj` with `torch.empty` and `Glm5NextTextTopkRouter`
declares `weight` with `torch.zeros`; `_init_weights` runs at `from_pretrained`
time and not here. An unfilled buffer makes the golden depend on whatever was in
that allocation, and an all-zero router makes every expert score exactly 0.5, so
the top-k degenerates into a tie-break and the gate stops measuring routing.

Usage:
    python3 tests/vllm/models/fixtures/gen_glm5_next_moe_goldens.py [--out PATH]

Requires `transformers==5.16.1` and torch. The generator REFUSES to emit under
any other transformers version, because a golden captured off an unpinned oracle
is not reproducible.
"""

import argparse
import pathlib
import sys

LANE_PIN = "5.16.1"

# --- the published router geometry, every value from the checkpoint -----------
N_ROUTED_EXPERTS = 288
NUM_EXPERTS_PER_TOK = 8
N_GROUP = 1
TOPK_GROUP = 1
ROUTED_SCALING_FACTOR = 2.5
NORM_TOPK_PROB = True
SWIGLU_LIMIT = 10.0

# --- the toy widths the composition is captured at ---------------------------
HIDDEN = 8
INTERMEDIATE = 12  # the DENSE MLP width
MOE_INTERMEDIATE = 6
N_SHARED_EXPERTS = 1
SMALL_EXPERTS = 8
SMALL_TOP_K = 3
SEQ = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).with_name("glm5_next_moe_goldens.inc")),
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
        Glm5NextTextExperts,
        Glm5NextTextMLP,
        Glm5NextTextMoE,
        Glm5NextTextTopkRouter,
    )

    torch.manual_seed(2223)

    # ── (1) the router, at the PUBLISHED 288 + 8 ─────────────────────────────
    big = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=N_ROUTED_EXPERTS,
        n_shared_experts=N_SHARED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        norm_topk_prob=NORM_TOPK_PROB,
        swiglu_limit=SWIGLU_LIMIT,
        num_hidden_layers=2,
    )
    assert big.num_local_experts == N_ROUTED_EXPERTS, big.num_local_experts
    assert big.routed_scaling_factor == ROUTED_SCALING_FACTOR
    assert big.norm_topk_prob is NORM_TOPK_PROB

    router = Glm5NextTextTopkRouter(big)
    with torch.no_grad():
        router.weight.normal_(0.0, 1.0)
    router.eval()

    hidden = torch.randn(SEQ, HIDDEN)
    with torch.no_grad():
        logits_nb, w_nb, i_nb = router(hidden)
    # `e_score_correction_bias` is a BUFFER the constructor zeros. Fill it with a
    # spread wide enough to move the selection: a bias smaller than the gap
    # between the 8th and the 9th sigmoid score would leave both sets equal and
    # the case would prove nothing.
    with torch.no_grad():
        bias = torch.empty(N_ROUTED_EXPERTS).uniform_(-0.30, 0.30)
        router.e_score_correction_bias.copy_(bias)
        logits_b, w_b, i_b = router(hidden)

    assert torch.equal(logits_nb, logits_b), "the bias must not move the logits"
    moved = sum(
        1
        for t in range(SEQ)
        if set(i_nb[t].tolist()) != set(i_b[t].tolist())
    )
    assert moved > 0, (
        "the biased and unbiased selections are identical, so this fixture "
        "cannot show that the bias SELECTS"
    )

    # The separation margin, per token: the biased score of the LAST selected
    # expert minus the biased score of the BEST rejected one. A discrete
    # selection gate has bimodal error, so the gate prints this rather than
    # relying on a tolerance.
    scores_b = logits_b.sigmoid() + bias
    margins = []
    for t in range(SEQ):
        sel = set(i_b[t].tolist())
        lo = min(float(scores_b[t, j]) for j in sel)
        hi = max(
            float(scores_b[t, j]) for j in range(N_ROUTED_EXPERTS) if j not in sel
        )
        margins.append(lo - hi)
    assert all(m > 0 for m in margins), margins

    # ── (2) the clamped SwiGLU epilogue, WITH a case that clamps ─────────────
    experts = Glm5NextTextExperts(big)
    assert experts.swiglu_limit == SWIGLU_LIMIT
    # Row 0 stays inside the limit. Row 1 reaches past it on BOTH sides in the
    # gate half and BOTH sides in the up half, which is the asymmetry the epilogue
    # is built around: the gate is clamped MAX-ONLY, the up on both sides. A
    # fixture that never leaves [-limit, limit] passes a port with no clamp at
    # all, and one that only exceeds +limit passes a port that clamps the gate
    # symmetrically.
    gate_up = torch.stack(
        [
            torch.linspace(-3.0, 3.0, 2 * MOE_INTERMEDIATE),
            torch.tensor(
                [-14.0, -10.5, -2.0, 0.5, 11.0, 25.0]        # gate half
                + [-31.0, -10.25, -1.5, 0.25, 10.75, 40.0]   # up half
            ),
        ]
    )
    assert gate_up.shape == (2, 2 * MOE_INTERMEDIATE), gate_up.shape
    assert (gate_up[1, :MOE_INTERMEDIATE] > SWIGLU_LIMIT).any()
    assert (gate_up[1, :MOE_INTERMEDIATE] < -SWIGLU_LIMIT).any()
    assert (gate_up[1, MOE_INTERMEDIATE:] > SWIGLU_LIMIT).any()
    assert (gate_up[1, MOE_INTERMEDIATE:] < -SWIGLU_LIMIT).any()
    with torch.no_grad():
        gate_out = experts._apply_gate(gate_up)

    # ── (3) the DENSE feed-forward ───────────────────────────────────────────
    dense = Glm5NextTextMLP(big)
    assert dense.intermediate_size == INTERMEDIATE
    assert dense.swiglu_limit == SWIGLU_LIMIT
    with torch.no_grad():
        # Scaled so the pre-activation actually reaches past +-10 on some
        # channels; at unit scale the clamp is never exercised and a port that
        # drops it passes.
        dense.gate_proj.weight.normal_(0.0, 3.0)
        dense.up_proj.weight.normal_(0.0, 3.0)
        dense.down_proj.weight.normal_(0.0, 0.5)
        dense_in = torch.randn(SEQ, HIDDEN) * 2.0
        pre_gate = dense.gate_proj(dense_in)
        pre_up = dense.up_proj(dense_in)
        dense_out = dense(dense_in)
    assert (pre_gate > SWIGLU_LIMIT).any(), "the dense gate clamp is not exercised"
    assert (pre_up.abs() > SWIGLU_LIMIT).any(), "the dense up clamp is not exercised"

    # ── (4) the composed MoE block, at the SMALL expert count ────────────────
    small = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_routed_experts=SMALL_EXPERTS,
        n_shared_experts=N_SHARED_EXPERTS,
        num_experts_per_tok=SMALL_TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        norm_topk_prob=NORM_TOPK_PROB,
        swiglu_limit=SWIGLU_LIMIT,
        num_hidden_layers=2,
    )
    moe = Glm5NextTextMoE(small)
    with torch.no_grad():
        moe.gate.weight.normal_(0.0, 1.0)
        moe.gate.e_score_correction_bias.uniform_(-0.30, 0.30)
        moe.experts.gate_up_proj.normal_(0.0, 0.5)
        moe.experts.down_proj.normal_(0.0, 0.5)
        moe.shared_experts.gate_proj.weight.normal_(0.0, 0.5)
        moe.shared_experts.up_proj.weight.normal_(0.0, 0.5)
        moe.shared_experts.down_proj.weight.normal_(0.0, 0.5)
    moe.eval()
    assert moe.shared_experts.intermediate_size == MOE_INTERMEDIATE * N_SHARED_EXPERTS

    moe_in = torch.randn(1, SEQ, HIDDEN)
    with torch.no_grad():
        _, small_w, small_i = moe.gate(moe_in)
        moe_out = moe(moe_in)
        shared_out = moe.shared_experts(moe_in)
        routed_out = moe_out - shared_out
    assert tuple(moe_out.shape) == (1, SEQ, HIDDEN), moe_out.shape

    out = pathlib.Path(args.out)
    out.write_text(
        _emit(
            tf_ver=transformers.__version__,
            torch_ver=torch.__version__,
            hidden=hidden,
            router_weight=router.weight,
            router_logits=logits_b,
            bias=bias,
            w_nb=w_nb,
            i_nb=i_nb,
            w_b=w_b,
            i_b=i_b,
            margins=margins,
            gate_up=gate_up,
            gate_out=gate_out,
            dense_gate=dense.gate_proj.weight,
            dense_up=dense.up_proj.weight,
            dense_down=dense.down_proj.weight,
            dense_in=dense_in,
            dense_out=dense_out,
            small_router_weight=moe.gate.weight,
            small_bias=moe.gate.e_score_correction_bias,
            small_gate_up=moe.experts.gate_up_proj,
            small_down=moe.experts.down_proj,
            small_sh_gate=moe.shared_experts.gate_proj.weight,
            small_sh_up=moe.shared_experts.up_proj.weight,
            small_sh_down=moe.shared_experts.down_proj.weight,
            moe_in=moe_in[0],
            moe_out=moe_out[0],
            shared_out=shared_out[0],
            routed_out=routed_out[0],
            small_w=small_w,
            small_i=small_i,
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

    if isinstance(t, list):
        flat = [float(v) for v in t]
    else:
        flat = t.detach().to(torch.float32).reshape(-1).tolist()
    return _wrap([f"{float(v)!r}f" for v in flat])


def _ints(t) -> str:
    if isinstance(t, list):
        flat = [int(v) for v in t]
    else:
        flat = t.detach().reshape(-1).tolist()
    return _wrap([str(int(v)) for v in flat])


def _emit(**g) -> str:
    lines = [
        "// GENERATED by tests/vllm/models/fixtures/gen_glm5_next_moe_goldens.py",
        "// -- DO NOT EDIT BY HAND.",
        "//",
        "// GLM-5.3-Flash MoE goldens, row",
        "// MODEL-MM-glm5-next-glm5-next-for-conditional-generation W5, issue #2223.",
        "// Produced by RUNNING the reference oracle:",
        f"//   transformers {g['tf_ver']}   (this row's lane revision; W0/#2096 records it)",
        f"//   torch        {g['torch_ver']}",
        "//",
        "// The router block is captured at the PUBLISHED 288 experts / top-8, TWICE:",
        "// `kTopk*NoBias` with `e_score_correction_bias` at its constructor zeros and",
        "// `kTopk*Bias` with it filled. The two SELECTIONS differ, which is what makes",
        "// `bias selects, unbiased score weights` gateable rather than asserted.",
        "//",
        "// `kSelectionMargin[t]` is the biased score of the LAST selected expert minus",
        "// the biased score of the BEST rejected one, per token. A discrete selection",
        "// has BIMODAL error -- it is right or it is a different set -- so the gate",
        "// asserts SET equality and prints this margin, rather than putting a tolerance",
        "// on values that can be close while the set is wrong.",
        "//",
        "// `kGateUp` row 1 reaches past +-limit in BOTH halves, so a port that drops the",
        "// clamp, or clamps the gate on both sides instead of max-only, reds.",
        "//",
        "// Layout, row-major: kHidden [seq, hidden], kRouterWeight [E, hidden],",
        "// kRouterLogits [seq, E], kBias [E], kTopkWeights* [seq, top_k],",
        "// kTopkIndices* [seq, top_k], kSelectionMargin [seq], kGateUp [2, 2*I_moe],",
        "// kGateOut [2, I_moe], kDense{Gate,Up} [I_dense, hidden], kDenseDown",
        "// [hidden, I_dense], kDenseIn/kDenseOut [seq, hidden], kSmallGateUp",
        "// [E_small, 2*I_moe, hidden], kSmallDown [E_small, hidden, I_moe],",
        "// kSmallSh{Gate,Up} [I_moe, hidden], kSmallShDown [hidden, I_moe],",
        "// kMoeIn/kMoeOut/kSharedOut/kRoutedOut [seq, hidden].",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace glm5_next_moe_goldens {",
        "",
        f'inline constexpr const char* kOracle = "transformers {g["tf_ver"]}";',
        f"inline constexpr int64_t kNumRoutedExperts = {N_ROUTED_EXPERTS};",
        f"inline constexpr int64_t kNumExpertsPerTok = {NUM_EXPERTS_PER_TOK};",
        f"inline constexpr int64_t kNGroup = {N_GROUP};",
        f"inline constexpr int64_t kTopkGroup = {TOPK_GROUP};",
        f"inline constexpr float kRoutedScalingFactor = {ROUTED_SCALING_FACTOR!r}f;",
        f"inline constexpr bool kNormTopkProb = {'true' if NORM_TOPK_PROB else 'false'};",
        f"inline constexpr float kSwigluLimit = {SWIGLU_LIMIT!r}f;",
        f"inline constexpr int64_t kHiddenSize = {HIDDEN};",
        f"inline constexpr int64_t kIntermediate = {INTERMEDIATE};",
        f"inline constexpr int64_t kMoeIntermediate = {MOE_INTERMEDIATE};",
        f"inline constexpr int64_t kNSharedExperts = {N_SHARED_EXPERTS};",
        f"inline constexpr int64_t kSmallExperts = {SMALL_EXPERTS};",
        f"inline constexpr int64_t kSmallTopK = {SMALL_TOP_K};",
        f"inline constexpr int64_t kSeq = {SEQ};",
        "",
        f"inline constexpr float kHidden[] = {{{_floats(g['hidden'])}}};",
        f"inline constexpr float kRouterWeight[] = {{{_floats(g['router_weight'])}}};",
        f"inline constexpr float kRouterLogits[] = {{{_floats(g['router_logits'])}}};",
        f"inline constexpr float kBias[] = {{{_floats(g['bias'])}}};",
        f"inline constexpr float kTopkWeightsNoBias[] = {{{_floats(g['w_nb'])}}};",
        f"inline constexpr int32_t kTopkIndicesNoBias[] = {{{_ints(g['i_nb'])}}};",
        f"inline constexpr float kTopkWeightsBias[] = {{{_floats(g['w_b'])}}};",
        f"inline constexpr int32_t kTopkIndicesBias[] = {{{_ints(g['i_b'])}}};",
        f"inline constexpr float kSelectionMargin[] = {{{_floats(g['margins'])}}};",
        f"inline constexpr float kGateUp[] = {{{_floats(g['gate_up'])}}};",
        f"inline constexpr float kGateOut[] = {{{_floats(g['gate_out'])}}};",
        f"inline constexpr float kDenseGate[] = {{{_floats(g['dense_gate'])}}};",
        f"inline constexpr float kDenseUp[] = {{{_floats(g['dense_up'])}}};",
        f"inline constexpr float kDenseDown[] = {{{_floats(g['dense_down'])}}};",
        f"inline constexpr float kDenseIn[] = {{{_floats(g['dense_in'])}}};",
        f"inline constexpr float kDenseOut[] = {{{_floats(g['dense_out'])}}};",
        f"inline constexpr float kSmallRouterWeight[] = {{{_floats(g['small_router_weight'])}}};",
        f"inline constexpr float kSmallBias[] = {{{_floats(g['small_bias'])}}};",
        f"inline constexpr float kSmallGateUp[] = {{{_floats(g['small_gate_up'])}}};",
        f"inline constexpr float kSmallDown[] = {{{_floats(g['small_down'])}}};",
        f"inline constexpr float kSmallShGate[] = {{{_floats(g['small_sh_gate'])}}};",
        f"inline constexpr float kSmallShUp[] = {{{_floats(g['small_sh_up'])}}};",
        f"inline constexpr float kSmallShDown[] = {{{_floats(g['small_sh_down'])}}};",
        f"inline constexpr float kMoeIn[] = {{{_floats(g['moe_in'])}}};",
        f"inline constexpr float kMoeOut[] = {{{_floats(g['moe_out'])}}};",
        f"inline constexpr float kSharedOut[] = {{{_floats(g['shared_out'])}}};",
        f"inline constexpr float kRoutedOut[] = {{{_floats(g['routed_out'])}}};",
        f"inline constexpr float kSmallTopkWeights[] = {{{_floats(g['small_w'])}}};",
        f"inline constexpr int32_t kSmallTopkIndices[] = {{{_ints(g['small_i'])}}};",
        "",
        "}  // namespace glm5_next_moe_goldens",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
