#!/usr/bin/env python3
"""Regenerate `qwen4_exp_qsa_goldens.inc` by RUNNING the algorithm oracle.

Row MODEL-MM-QWEN4-EXP W4, issue #1991, `.agents/specs/qwen4-exp-flash-next.md`.

WHAT THE ORACLE IS. `transformers` **5.16.0**, the lane pin recorded in
`.agents/oracles/transformers.md` and accepted by the developer on 2026-08-26 as
this row's ALGORITHM oracle (vLLM registers no `qwen4_exp` at `6a5e8f5979`, so
there is nothing to mirror on the primary). The selected-token sets below are the
output of the UNMODIFIED `Qwen4ExpTextQSAIndexer.forward`
(`transformers/models/qwen4_exp/modeling_qwen4_exp.py`), called through its own
`__call__`. Nothing in the selection path is transcribed here: a transcription
cannot gate the function it transcribes.

WHAT IS TRANSCRIBED, AND WHY THAT IS NOT THE GATE. Three things:
  * the `torch.split` of `index_qk_proj`'s output into `(q, token_k)`
    (modeling_qwen4_exp.py, `Qwen4ExpTextQSAIndexer.forward`), so the C++ side can
    start from `q_raw` / `k_raw` instead of re-running a GEMM. The split is a
    shape, not arithmetic, and the C++ side never reproduces the projection.
  * the dense eager attention used for the CONSUMER golden -- plain
    `softmax(q k^T * scale + mask) v` in float32. What is under test in the
    consumer is WHICH keys are attended, and that comes from the oracle's own
    mask.
  * `repeat_kv` for GQA, five lines of indexing.

WHY THE MODULE RUNS IN BFLOAT16. `pooled_keys.float().mean(1).to(raw_keys.dtype)`
in the oracle rounds the pooled key back to the cache dtype BEFORE `k_layernorm`.
At float32 that cast is a no-op and the gate would never see it; the production
side cache is bf16 (that is what makes it 64 B/token/layer), so the goldens are
captured at bf16 and the C++ reference performs the same round-trip.

WHY TWO CASES. `sub_budget` has kv_len == indexer_budget + compress_ratio - 1, the
largest context at which EVERY candidate is still selected: llama.cpp #27742
measures a max logit delta of 0.0 there, so QSA must be bit-identical to dense
attention. `over_budget` runs past `indexer_budget` so that top-k actually
discards blocks -- below that threshold a wrong implementation attending pooled
keys instead of real tokens is indistinguishable and still emits plausible
tokens (spec `## Gates` G2).

Usage:
    python3 tests/vllm/models/fixtures/gen_qwen4_exp_qsa_goldens.py [--out PATH]

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
        default=str(pathlib.Path(__file__).with_name("qwen4_exp_qsa_goldens.inc")),
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
        Qwen4ExpTextQSAIndexer,
        Qwen4ExpTextRotaryEmbedding,
    )

    torch.manual_seed(1991)
    dt = torch.bfloat16

    # A deliberately tiny QSA config. Every invariant `validate_architecture`
    # enforces still holds: indexer_kv_heads == 1, budget % compress_ratio == 0,
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
    width = BUDGET + CR - 1

    rotary = Qwen4ExpTextRotaryEmbedding(cfg).to(dt)
    cases = []

    for name, seq in (("sub_budget", width), ("over_budget", 23)):
        indexer = Qwen4ExpTextQSAIndexer(cfg, layer_idx=0).to(dt)
        # `weight` is zero-initialised on Qwen4ExpTextRMSNorm and the forward is
        # `out * (1.0 + weight)`, so a zeroed norm is an identity scale. Randomise
        # it, or the gate cannot see a port that drops the (1.0 + w) term.
        with torch.no_grad():
            indexer.q_layernorm.weight.normal_(0.0, 0.5)
            indexer.k_layernorm.weight.normal_(0.0, 0.5)

        hidden = torch.randn(1, seq, HID, dtype=dt)
        pos = torch.arange(seq).view(1, seq)
        cos, sin = rotary(hidden, pos)  # [1, seq, ROT] each, bf16

        # Causal bool mask, 4D, exactly the shape the oracle documents it takes
        # ("always 4D with either bool (sdpa) or float (eager)").
        causal = torch.ones(seq, seq, dtype=torch.bool).tril().view(1, 1, seq, seq)

        # --- the ORACLE call, unmodified -------------------------------------
        # `apply_rotary_pos_emb` is INTERCEPTED, not reimplemented: the wrapper
        # records the arguments the oracle chose and delegates to the original.
        # That is how `q_post` and `block_keys` below come out of the oracle's own
        # execution rather than out of a second implementation of it. In
        # particular the RoPE position of a pooled key -- the block's FIRST token,
        # not its last -- is captured rather than asserted.
        import transformers.models.qwen4_exp.modeling_qwen4_exp as mod

        original_rope = mod.apply_rotary_pos_emb
        rope_calls = []

        def _recording_rope(*a, **kw):
            out = original_rope(*a, **kw)
            rope_calls.append((kw.get("cos"), out))
            return out

        mod.apply_rotary_pos_emb = _recording_rope
        try:
            with torch.no_grad():
                sel_mask = indexer(hidden, (cos, sin), causal, None)  # [1,1,seq,seq]
        finally:
            mod.apply_rotary_pos_emb = original_rope

        # Call 0 is the query rope (cos is [1, seq, ROT]); every later call is one
        # query's pooled-key rope (cos is [num_blocks, ROT]). The LAST one belongs
        # to the last query, which sees the most complete blocks.
        q_post = rope_calls[0][1][0]                       # [seq, IDX_H, IDX_D]
        # The recorder sees the key rope BEFORE the oracle's `.squeeze(1)`.
        block_keys = rope_calls[-1][1].squeeze(1)          # [num_blocks, IDX_D]
        assert q_post.shape == (seq, IDX_H, IDX_D), q_post.shape
        assert block_keys.shape == (seq // CR, IDX_D), block_keys.shape

        # The C++ entry point starts from the projection's two halves. Transcribed
        # SHAPE only (see module docstring).
        with torch.no_grad():
            qk = indexer.index_qk_proj(hidden)
            q_raw, k_raw = torch.split(qk, [IDX_H * IDX_D, IDX_KV * IDX_D], dim=-1)
            q_raw = q_raw.reshape(1, seq, IDX_H, IDX_D)
            k_raw = k_raw.reshape(1, seq, IDX_D)

        # Selected token indices per query, ASCENDING, -1 padded to `width`. The
        # oracle emits a scatter mask, so ascending order is the only order the
        # mask can express; the C++ gather is defined to produce the same order so
        # that a sub-budget gather reduces over keys in dense positional order.
        sel = torch.full((seq, width), -1, dtype=torch.int32)
        for qi in range(seq):
            idx = torch.nonzero(sel_mask[0, 0, qi], as_tuple=False).flatten()
            assert idx.numel() <= width, (name, qi, idx.numel())
            sel[qi, : idx.numel()] = idx.to(torch.int32)

        # --- the CONSUMER golden ---------------------------------------------
        attn_q = torch.randn(1, seq, NH, HD, dtype=dt)
        attn_k = torch.randn(1, seq, NKV, HD, dtype=dt)
        attn_v = torch.randn(1, seq, NKV, HD, dtype=dt)
        combined = causal & sel_mask  # oracle line: `attention_mask & selected_token_mask`
        groups = NH // NKV
        qf = attn_q[0].float().transpose(0, 1)                       # [NH, seq, HD]
        kf = attn_k[0].float().transpose(0, 1).repeat_interleave(groups, 0)
        vf = attn_v[0].float().transpose(0, 1).repeat_interleave(groups, 0)
        logits = torch.matmul(qf, kf.transpose(-1, -2)) * (HD**-0.5)
        logits = logits.masked_fill(~combined[0, 0].view(1, seq, seq), float("-inf"))
        attn_out = torch.matmul(torch.softmax(logits, dim=-1), vf)   # [NH, seq, HD]
        attn_out = attn_out.transpose(0, 1).contiguous()             # [seq, NH, HD]

        cases.append(
            dict(
                name=name,
                seq=seq,
                q_raw=q_raw[0],
                k_raw=k_raw[0],
                cos=cos[0],
                sin=sin[0],
                q_norm_w=indexer.q_layernorm.weight,
                k_norm_w=indexer.k_layernorm.weight,
                q_post=q_post,
                block_keys=block_keys,
                sel=sel,
                attn_q=attn_q[0],
                attn_k=attn_k[0],
                attn_v=attn_v[0],
                attn_out=attn_out,
            )
        )

    out = pathlib.Path(args.out)
    out.write_text(_emit(cases, cfg, transformers.__version__, torch.__version__,
                         HID, NH, NKV, HD, IDX_H, IDX_KV, IDX_D, BUDGET, CR, ROT))
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


def _ints(t) -> str:
    flat = t.detach().reshape(-1).tolist()
    return _wrap([str(int(v)) for v in flat])


def _emit(cases, cfg, tf_ver, torch_ver, HID, NH, NKV, HD,
          IDX_H, IDX_KV, IDX_D, BUDGET, CR, ROT) -> str:
    lines = [
        "// GENERATED by tests/vllm/models/fixtures/gen_qwen4_exp_qsa_goldens.py",
        "// -- DO NOT EDIT BY HAND.",
        "//",
        "// Qwen4-Exp Qwen Sparse Attention (QSA) goldens, row MODEL-MM-QWEN4-EXP W4,",
        "// issue #1991. Produced by RUNNING the algorithm oracle:",
        f"//   transformers {tf_ver}   (the lane pin; see .agents/oracles/transformers.md)",
        f"//   torch        {torch_ver}",
        "// The selected-token sets are the output of the UNMODIFIED",
        "// `Qwen4ExpTextQSAIndexer.forward`, not a transcription of it.",
        "//",
        "// Every float below is exactly bf16-representable: the oracle module runs in",
        "// bfloat16 so that the `.to(raw_keys.dtype)` round-trip after the mean pool is",
        "// inside the gate rather than optimised away by a float32 capture.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace qwen4_exp_qsa_goldens {",
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
    ]
    for c in cases:
        p = c["name"].title().replace("_", "")
        lines += [
            f"// ---- case '{c['name']}': kv_len = {c['seq']} "
            f"({'every candidate selected' if c['seq'] <= BUDGET + CR - 1 else 'top-k discards blocks'})",
            f"inline constexpr int64_t k{p}Seq = {c['seq']};",
            f"inline constexpr float k{p}QRaw[] = {{{_floats(c['q_raw'])}}};",
            f"inline constexpr float k{p}KRaw[] = {{{_floats(c['k_raw'])}}};",
            f"inline constexpr float k{p}Cos[] = {{{_floats(c['cos'])}}};",
            f"inline constexpr float k{p}Sin[] = {{{_floats(c['sin'])}}};",
            f"inline constexpr float k{p}QNormW[] = {{{_floats(c['q_norm_w'])}}};",
            f"inline constexpr float k{p}KNormW[] = {{{_floats(c['k_norm_w'])}}};",
            f"inline constexpr float k{p}QPost[] = {{{_floats(c['q_post'])}}};",
            f"inline constexpr float k{p}BlockKeys[] = {{{_floats(c['block_keys'])}}};",
            f"inline constexpr int32_t k{p}Selected[] = {{{_ints(c['sel'])}}};",
            f"inline constexpr float k{p}AttnQ[] = {{{_floats(c['attn_q'])}}};",
            f"inline constexpr float k{p}AttnK[] = {{{_floats(c['attn_k'])}}};",
            f"inline constexpr float k{p}AttnV[] = {{{_floats(c['attn_v'])}}};",
            f"inline constexpr float k{p}AttnOut[] = {{{_floats(c['attn_out'])}}};",
            "",
        ]
    lines += ["}  // namespace qwen4_exp_qsa_goldens", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
