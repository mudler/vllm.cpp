#!/usr/bin/env python3
"""Regenerate `qwen4_exp_qsa_block_goldens.inc` by RUNNING the algorithm oracle.

Row MODEL-MM-QWEN4-EXP W5b-5, issue #2211 (wave #2031),
`.agents/specs/qwen4-exp-flash-next.md`.

WHAT THIS CAPTURES THAT `gen_qwen4_exp_qsa_goldens.py` DOES NOT. That generator
gates the four `vt::` ops of W4/W5b-4 from the INDEXER's own inputs: it starts at
`q_raw`/`k_raw`, i.e. AFTER `index_qk_proj`, and its consumer golden is a
transcribed dense eager attention over the oracle's mask. This one gates the
WHOLE `Qwen4ExpTextAttention.forward` (modeling_qwen4_exp.py:785-841) end to end
-- the projections, the per-head q/k norms, the RoPE, the indexer, the gather
consumer, the sigmoid OUTPUT GATE and `o_proj` -- because W5b-5 lands the
production composition and the composition is what nothing yet measures.

THE VALUE GATE ON THE LOGITS, AND WHY IT IS THE POINT OF THIS FILE. The spec's
`## Owed` records that two of the four settings the composed indexer depends on
(`n_head_scale == 1`, `softmax_scale == index_head_dim ** -0.5`) are INVISIBLE to
any selection-based gate BY CONSTRUCTION: top-k is invariant under a positive
rescale of every score, so a wrong value there cannot move a selected set.
`DsaIndexerLogitsArgs` says the same thing in its own comment. So this generator
captures the oracle's own `scores` tensor -- the one it feeds to `.topk` -- by
intercepting `torch.Tensor.topk` for the duration of the call and recording
`self`. That is a capture of the oracle's execution, not a transcription of its
two scoring lines: nothing here recomputes `relu(q . k).sum(-1) / sqrt(D)`.

WHY BFLOAT16, AND WHAT THAT COSTS. The oracle runs in bf16, as the sibling
generator does, and for the same two reasons plus a third. The pooled key's
`.to(raw_keys.dtype)` round-trip is a no-op at f32 and load-bearing at bf16. The
production side cache is bf16, which is what makes it 64 B/token/layer. And the
whole `vt::` output-gate family — `SigmoidGateBf16`, `SharedExpertGate` — STORES
bf16 by construction on every backend, because vLLM resolves one model dtype and
this tree inherits that polarity (AGENTS.md, "Inherit vLLM defaults"); an f32
block would have had to widen a shared dispatcher across five backends this host
cannot gate, which is a worse trade than a wider tolerance.

The cost is paid where it is cheapest. The BLOCK OUTPUT is compared at a
RELATIVE bound sized to the bf16 quantum, which is strong enough for every
structural property (the RoPE position of a pooled key, a dropped gate, the GQA
head map, a dense-masked consumer) and too weak for an epsilon. The INDEXER
LOGITS are compared TIGHTLY, because they are f32 on both sides — the oracle
computes its scores through an explicit `.float()` (modeling_qwen4_exp.py:691)
and `vt::DsaIndexerLogits` writes f32 — and the logits are exactly where the two
invisible settings live. That split is the point: the tight gate sits on the
quantity a selection cannot see.

WHY EAGER AND NOT SDPA. Upstream permits both and says so ("we only allow eager
and sdpa"). Eager is chosen because its mask is a FLOAT additive mask and its
attention is the explicit `softmax(q k^T * scale + mask) v`, so the golden is a
number this port can be held to rather than a kernel selection. Under sdpa the
same call would route into a fused kernel whose reduction order is not ours.

Usage:
    python3 tests/vllm/models/fixtures/gen_qwen4_exp_qsa_block_goldens.py [--out PATH]

Requires `transformers==5.16.0` and torch. The generator REFUSES to emit under any
other transformers version, because a golden captured off an unpinned oracle is
not reproducible.
"""

import argparse
import pathlib
import sys

LANE_PIN = "5.16.0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).with_name("qwen4_exp_qsa_block_goldens.inc")),
    )
    args = ap.parse_args()

    import torch
    import transformers

    if transformers.__version__ != LANE_PIN:
        sys.stderr.write(
            f"REFUSING: transformers {transformers.__version__} is not the lane pin "
            f"{LANE_PIN}. A golden captured off an unpinned oracle is not "
            f"reproducible (AGENTS.md 'Pin vLLM' / 'Pin every oracle').\n"
        )
        return 2

    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextAttention,
        Qwen4ExpTextRotaryEmbedding,
    )

    torch.manual_seed(2031)
    dt = torch.bfloat16

    # The same tiny QSA geometry the W4 goldens use, so the two fixtures describe
    # ONE model rather than two. Every invariant `validate_architecture` enforces
    # still holds: indexer_kv_heads == 1, budget % compress_ratio == 0, and
    # rotary_dim (head_dim * partial_rotary_factor = 8) <= indexer_head_dim.
    HID, NH, NKV, HD = 64, 4, 2, 32
    IDX_H, IDX_KV, IDX_D, BUDGET, CR = 4, 1, 16, 8, 4
    ROT = HD // 4  # partial_rotary_factor 0.25
    cfg = Qwen4ExpTextConfig(
        vocab_size=128,
        hidden_size=HID,
        num_hidden_layers=4,
        num_attention_heads=NH,
        num_key_value_heads=NKV,
        head_dim=HD,
        rms_norm_eps=1e-6,
        indexer_n_heads=IDX_H,
        indexer_kv_heads=IDX_KV,
        indexer_head_dim=IDX_D,
        indexer_budget=BUDGET,
        indexer_compress_ratio=CR,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
    )
    cfg._attn_implementation = "eager"
    width = BUDGET + CR - 1

    rotary = Qwen4ExpTextRotaryEmbedding(cfg).to(dt)

    # ONE layer, two contexts. The weight set is emitted once and shared by both
    # cases: they are the SAME layer at two sequence lengths, which is both the
    # cheaper fixture and the sharper one -- a defect that only appears when the
    # top-k starts discarding blocks is then a difference in the context and
    # nothing else.
    attn = Qwen4ExpTextAttention(cfg, layer_idx=0).to(dt)
    # `Qwen4ExpTextRMSNorm.weight` is ZERO-initialised and the forward is
    # `out * (1.0 + weight)`, so a zeroed gamma is an identity scale and a
    # port that dropped the `(1.0 + w)` term would pass. Randomise all four.
    with torch.no_grad():
        attn.q_norm.weight.normal_(0.0, 0.5)
        attn.k_norm.weight.normal_(0.0, 0.5)
        attn.indexer.q_layernorm.weight.normal_(0.0, 0.5)
        attn.indexer.k_layernorm.weight.normal_(0.0, 0.5)

    cases = []
    for name, seq in (("sub_budget", width), ("over_budget", 23)):
        hidden = torch.randn(1, seq, HID, dtype=dt)
        pos = torch.arange(seq).view(1, seq)
        cos, sin = rotary(hidden, pos)  # [1, seq, ROT] each

        # The EAGER additive mask, exactly the shape the oracle documents it
        # takes ("always 4D with either bool (sdpa) or float (eager)"). `finfo.min`
        # rather than -inf: the oracle ADDS this to the logits and a row that was
        # all -inf would softmax to NaN. Every causal row has at least one 0.
        neg = torch.finfo(dt).min
        causal = torch.full((seq, seq), neg, dtype=dt).triu(1).view(1, 1, seq, seq)

        # --- the ORACLE call, unmodified -------------------------------------
        # `torch.Tensor.topk` is INTERCEPTED so the indexer's own `scores`
        # tensor -- the argument of `scores.topk(...)` at
        # modeling_qwen4_exp.py:694 -- is recorded as the oracle computed it.
        # This is the VALUE gate the spec's `## Owed` demands, and it is the only
        # observable that can convict a wrong `softmax_scale` or `n_head_scale`:
        # both are positive global rescales and cannot move a selection.
        #
        # `apply_rotary_pos_emb` is intercepted in the same breath, and for a
        # different reason: it yields the oracle's OWN roped indexer query. The
        # value gate on the logits has to isolate the FOUR SETTINGS, and feeding
        # our composition an input we computed ourselves would fold this port's
        # bf16 GEMM and bf16 RoPE rounding into a comparison that is meant to be
        # about a multiplicative constant. Handed the oracle's own query and the
        # oracle's own raw keys, the two sides differ only by the reassociation
        # the spec names, and the bound can be tight enough to matter.
        import transformers.models.qwen4_exp.modeling_qwen4_exp as mod

        scores_by_query = []
        rope_calls = []
        original_topk = torch.Tensor.topk
        original_rope = mod.apply_rotary_pos_emb

        def _recording_topk(self, *a, **kw):
            scores_by_query.append(self.detach().clone())
            return original_topk(self, *a, **kw)

        def _recording_rope(*a, **kw):
            out = original_rope(*a, **kw)
            rope_calls.append(out)
            return out

        torch.Tensor.topk = _recording_topk
        mod.apply_rotary_pos_emb = _recording_rope
        try:
            with torch.no_grad():
                block_out, _ = attn(hidden, (cos, sin), causal, None)
        finally:
            torch.Tensor.topk = original_topk
            mod.apply_rotary_pos_emb = original_rope

        # Call 0 is the INDEXER's query rope: the indexer runs on the attention
        # forward's first line, before any q/k/v projection. `apply_rotary_pos_emb`
        # returns a tuple when it is given a `k` and a bare tensor when it is not,
        # and the indexer's call passes only `q` (modeling_qwen4_exp.py:652).
        idx_q_post = rope_calls[0]
        assert not isinstance(idx_q_post, tuple), type(idx_q_post)
        idx_q_post = idx_q_post[0]
        assert idx_q_post.shape == (seq, IDX_H, IDX_D), idx_q_post.shape

        # The RAW indexer keys the side cache stores -- un-normed and un-roped,
        # which is what `Cache.update_indexer` holds and what the compressor
        # expects. SHAPE ONLY: this is the `torch.split` of `index_qk_proj`'s
        # output, and nothing here reproduces the projection.
        with torch.no_grad():
            qk = attn.indexer.index_qk_proj(hidden)
            _, token_k = torch.split(qk, [IDX_H * IDX_D, IDX_KV * IDX_D], dim=-1)
            idx_k_raw = token_k.reshape(1, seq, IDX_D)[0]

        # One entry per query token that saw at least one complete block. The
        # indexer skips `.topk` entirely when `num_complete_blocks == 0`, so the
        # first `CR - 1` queries contribute nothing and the offsets below say so.
        assert len(scores_by_query) == max(0, seq - (CR - 1)), (
            name,
            len(scores_by_query),
            seq,
        )
        score_offsets = [0]
        score_values = []
        for s in scores_by_query:
            assert s.ndim == 1, s.shape
            score_values.extend(s.tolist())
            score_offsets.append(len(score_values))

        # The selected token sets, read back out of the oracle's own mask so the
        # C++ selection can be compared against them directly. Recomputed here
        # from a SECOND indexer call would be a transcription; instead the mask
        # the block just consumed is rebuilt by calling the indexer submodule,
        # which is the same object with the same weights and no randomness.
        with torch.no_grad():
            sel_mask = attn.indexer(hidden, (cos, sin), causal, None)
        sel = torch.full((seq, width), -1, dtype=torch.int32)
        for qi in range(seq):
            keep = (sel_mask[0, 0, qi] == 0) if sel_mask.is_floating_point() else sel_mask[0, 0, qi]
            idx = torch.nonzero(keep, as_tuple=False).flatten()
            idx = idx[idx <= qi]  # the indexer's mask is not itself causal
            assert idx.numel() <= width, (name, qi, idx.numel())
            sel[qi, : idx.numel()] = idx.to(torch.int32)

        cases.append(
            dict(
                name=name,
                seq=seq,
                hidden=hidden[0],
                cos=cos[0],
                sin=sin[0],
                scores=torch.tensor(score_values, dtype=torch.float32),
                score_offsets=torch.tensor(score_offsets, dtype=torch.int32),
                sel=sel,
                idx_q_post=idx_q_post,
                idx_k_raw=idx_k_raw,
                out=block_out[0],
            )
        )

    out = pathlib.Path(args.out)
    out.write_text(
        _emit(cases, attn, cfg, transformers.__version__, torch.__version__, HID,
              NH, NKV, HD, IDX_H, IDX_KV, IDX_D, BUDGET, CR, ROT)
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


def _f32_literal(v: float) -> str:
    """The SHORTEST decimal that round-trips through float32, plus the `f` suffix.

    `repr` on a Python float prints the shortest decimal that round-trips through
    float64, which for a float32 value is up to 17 digits of noise below the
    float32 ulp. Nine significant digits always suffice for float32, so this
    walks up from one and stops at the first that reads back bit-identical --
    which keeps a fixture of this size four times smaller with no loss at all.
    The round-trip is checked, not assumed."""
    import struct

    bits = struct.pack("<f", v)
    for prec in range(1, 10):
        s = f"{v:.{prec}g}"
        if struct.pack("<f", float(s)) == bits:
            break
    else:  # pragma: no cover -- 9 digits always round-trips a float32
        s = repr(v)
    # `1e+05` and `1` are both valid C++ float literals with an `f` suffix; a
    # bare `1f` is NOT, so a value that printed without `.` or `e` gets a `.`.
    if "." not in s and "e" not in s and "n" not in s:
        s += "."
    return s + "f"


def _floats(t) -> str:
    import torch

    flat = t.detach().to(torch.float32).reshape(-1).tolist()
    import struct as _struct

    return _wrap([_f32_literal(_struct.unpack("<f", _struct.pack("<f", float(v)))[0])
                  for v in flat])


def _ints(t) -> str:
    flat = t.detach().reshape(-1).tolist()
    return _wrap([str(int(v)) for v in flat])


def _emit(cases, attn, cfg, tf_ver, torch_ver, HID, NH, NKV, HD,
          IDX_H, IDX_KV, IDX_D, BUDGET, CR, ROT) -> str:
    lines = [
        "// GENERATED by tests/vllm/models/fixtures/gen_qwen4_exp_qsa_block_goldens.py",
        "// -- DO NOT EDIT BY HAND.",
        "//",
        "// Qwen4-Exp Qwen Sparse Attention BLOCK goldens, row MODEL-MM-QWEN4-EXP",
        "// W5b-5, issue #2031. Produced by RUNNING the algorithm oracle:",
        f"//   transformers {tf_ver}   (the lane pin; see .agents/oracles/transformers.md)",
        f"//   torch        {torch_ver}",
        "// `kOut` is the return value of the UNMODIFIED",
        "// `Qwen4ExpTextAttention.forward`, and `kScores` is the indexer's own",
        "// pre-top-k score tensor, captured by intercepting `torch.Tensor.topk`.",
        "// Neither is a transcription of the code it gates.",
        "//",
        "// The oracle runs in BFLOAT16 and `_attn_implementation` is EAGER. Both are",
        "// arguments, not defaults; the generator's docstring gives them.",
        "//",
        "// `kScores` is f32 on BOTH sides -- the oracle scores through an explicit",
        "// `.float()` and `vt::DsaIndexerLogits` writes f32 -- so it carries the",
        "// TIGHT bound, but only when it is fed `kIdxQPost` and `kIdxKRaw`, the",
        "// oracle's OWN roped indexer query and raw indexer keys. Fed this port's",
        "// own bf16 projections instead, the same comparison inherits a bf16 ulp",
        "// and lands three orders looser. `kOut` is bf16 and carries a relative one.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace qwen4_exp_qsa_block_goldens {",
        "",
        f"inline constexpr const char* kOracle = \"transformers {tf_ver}\";",
        f"inline constexpr int64_t kHiddenSize = {HID};",
        f"inline constexpr int64_t kNumAttentionHeads = {NH};",
        f"inline constexpr int64_t kNumKeyValueHeads = {NKV};",
        f"inline constexpr int64_t kHeadDim = {HD};",
        f"inline constexpr int64_t kIndexNHeads = {IDX_H};",
        f"inline constexpr int64_t kIndexKvHeads = {IDX_KV};",
        f"inline constexpr int64_t kIndexHeadDim = {IDX_D};",
        f"inline constexpr int64_t kTokenBudget = {BUDGET};",
        f"inline constexpr int64_t kCompressRatio = {CR};",
        f"inline constexpr int64_t kRotaryDim = {ROT};",
        f"inline constexpr float kRmsNormEps = {cfg.rms_norm_eps:.9g}f;",
        f"inline constexpr int64_t kIndexWidth = {BUDGET + CR - 1};",
        "",
        "// ---- the ONE layer's weights, shared by every case below. Every tensor is",
        "// in PyTorch `nn.Linear(bias=False)` (out_features, in_features) row-major",
        "// order, i.e. the raw [N, K] a `vt::MatmulBT` consumes. The four gammas are",
        "// the RAW HuggingFace values: `Qwen4ExpTextRMSNorm.forward` applies",
        "// `out * (1.0 + weight)`, so a consumer whose norm op is `out * w` owes the",
        "// `+1` and the C++ side is where that is done.",
        f"inline constexpr float kQProj[] = {{{_floats(attn.q_proj.weight)}}};",
        f"inline constexpr float kKProj[] = {{{_floats(attn.k_proj.weight)}}};",
        f"inline constexpr float kVProj[] = {{{_floats(attn.v_proj.weight)}}};",
        f"inline constexpr float kOProj[] = {{{_floats(attn.o_proj.weight)}}};",
        f"inline constexpr float kQNormW[] = {{{_floats(attn.q_norm.weight)}}};",
        f"inline constexpr float kKNormW[] = {{{_floats(attn.k_norm.weight)}}};",
        f"inline constexpr float kIdxQkProj[] = {{{_floats(attn.indexer.index_qk_proj.weight)}}};",
        f"inline constexpr float kIdxQNormW[] = {{{_floats(attn.indexer.q_layernorm.weight)}}};",
        f"inline constexpr float kIdxKNormW[] = {{{_floats(attn.indexer.k_layernorm.weight)}}};",
        "",
    ]
    for c in cases:
        p = c["name"].title().replace("_", "")
        lines += [
            f"// ---- case '{c['name']}': kv_len = {c['seq']} "
            f"({'every candidate selected' if c['seq'] <= BUDGET + CR - 1 else 'top-k discards blocks'})",
            f"inline constexpr int64_t k{p}Seq = {c['seq']};",
            f"inline constexpr float k{p}Hidden[] = {{{_floats(c['hidden'])}}};",
            f"inline constexpr float k{p}Cos[] = {{{_floats(c['cos'])}}};",
            f"inline constexpr float k{p}Sin[] = {{{_floats(c['sin'])}}};",
            f"inline constexpr float k{p}Scores[] = {{{_floats(c['scores'])}}};",
            f"inline constexpr int32_t k{p}ScoreOffsets[] = {{{_ints(c['score_offsets'])}}};",
            f"inline constexpr int32_t k{p}Selected[] = {{{_ints(c['sel'])}}};",
            f"inline constexpr float k{p}IdxQPost[] = {{{_floats(c['idx_q_post'])}}};",
            f"inline constexpr float k{p}IdxKRaw[] = {{{_floats(c['idx_k_raw'])}}};",
            f"inline constexpr float k{p}Out[] = {{{_floats(c['out'])}}};",
            "",
        ]
    lines += ["}  // namespace qwen4_exp_qsa_block_goldens", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
