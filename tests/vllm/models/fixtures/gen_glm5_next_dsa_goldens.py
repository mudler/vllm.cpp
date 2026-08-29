#!/usr/bin/env python3
"""Regenerate `glm5_next_dsa_goldens.inc` by RUNNING the reference oracle.

Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation W3, issue #2213,
`.agents/specs/glm5-next-flash.md` section W3.

WHAT THE ORACLE IS. `transformers` **v5.16.1**, the lane revision this row cites
(W0, #2096, owns recording it in `.agents/oracles/transformers.md`). vLLM
registers no `glm5_next` at our parity pin `555967922` nor at its `main`, and
neither do vllm-omni, SGLang or llama.cpp, so under AGENTS.md "When vLLM has no
implementation" transformers is the reference for this surface.

Every golden below is a value produced by an UNMODIFIED reference module,
`Glm5NextTextIndexer` (`modular_glm5_next.py:749-1022`, flattened
`modeling_glm5_next.py`), called through its own `__call__`. Nothing is
transcribed: the intermediate goldens are captured by monkeypatching NOTHING and
instead calling the module's OWN public sub-methods (`get_visible_tokens`,
`get_pooled_states`, `append_visible_tail`) on the same inputs, then calling
`forward` for the end-to-end selection.

WHY THIS SHAPE. The trap this wave exists to close is that a top-k over pooled
candidates and a top-k over raw tokens BOTH produce plausible indices. The
fixture is therefore chosen so the two DISAGREE:

  * `seq_len` 21 is STRICTLY GREATER than `index_topk` 8. At or below `index_topk`
    a raw top-k selects everything, the selection is the identity, and the
    pooling is unobservable (spec section "The k-pool indexer is a compression
    stage DeepSeek-V4 does not have").
  * `index_kpool` 4 is the PUBLISHED checkpoint's value, not the config class
    default 16. `keep = pool_valid.any(0)` drops the pool no row can use, so P is
    5 and not ceil(21 / 4) = 6, and `select_k = min(index_topk // index_kpool, P)
    = min(2, 5) = 2`: two of five pools are chosen and three are rejected.
  * Row 1 is LEFT-PADDED by three tokens, so its pool grid starts at token 3 and
    not at slot 0 (`get_pooled_states`, `:938-945`). A port that pools from slot
    0 passes row 0 and fails row 1.
  * `index_kpool_always_select_tail` widens the output to
    `index_topk + index_kpool - 1 = 11`, not 8.

WHY float32. The reference computes the scores in fp32 (`scores = matmul(q.float(),
...)`) and the C++ side here is a host f32 reference, exactly as
`glm5_next_mhc.cpp` is. Capturing at fp32 makes the module's own `.to(dtype)`
casts no-ops, which is honest for what this file gates.

Usage:  python3 gen_glm5_next_dsa_goldens.py > glm5_next_dsa_goldens.inc
"""

import sys

import torch

import transformers
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

EXPECTED_VERSION = "5.16.1"

# ── the fixture geometry ────────────────────────────────────────────────────
B, S = 2, 21
HIDDEN = 16
Q_LORA = 12
N_HEADS = 8
HEAD_DIM = 8
INDEX_TOPK = 8
INDEX_KPOOL = 4          # the CHECKPOINT value; the config class defaults to 16
PAD_ROW1 = 3             # row 1 is left-padded by three tokens
SEED = 20260828

# ── the SHORT fixture: sequences with NO complete pool ───────────────────────
# `number_of_pools` is `ceil(kv_len / index_kpool)`, and a pool is valid only
# when EVERY one of its `index_kpool` members exists. Below `index_kpool` tokens
# no pool is complete, `keep = pool_valid.any(0)` is empty (`:967-970`), P is 0,
# `select_k = min(index_topk // index_kpool, 0)` is 0 and the pooled selection is
# EMPTY. Upstream does not refuse this: `append_visible_tail` still returns the
# raw visible tail, so the row is served with a tail-only selection. Each entry
# is (seq_len, left_pad); the last one is a row with FEWER valid tokens than
# `index_kpool` behind a left pad, which is the same state a real prefill of a
# short padded prompt reaches.
SHORT_CASES = [(1, 0), (2, 0), (3, 0), (3, 1)]


def config():
    return Glm5NextTextConfig(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA,
        kv_lora_rank=8,
        qk_rope_head_dim=0,          # the NoPE condition validate_architecture requires
        qk_nope_head_dim=8,
        v_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=4,
        index_topk=INDEX_TOPK,
        index_head_dim=HEAD_DIM,
        index_n_heads=N_HEADS,
        index_kpool=INDEX_KPOOL,
        index_kpool_always_select_tail=True,
    )


def main():
    if transformers.__version__ != EXPECTED_VERSION:
        raise SystemExit(
            f"oracle identity: expected transformers {EXPECTED_VERSION}, "
            f"got {transformers.__version__}"
        )
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float32)

    cfg = config()
    idx = Glm5NextTextIndexer(cfg, layer_idx=0).eval()

    # Real, non-degenerate parameters. `zeros` is the module's own init and would
    # make every pool softmax uniform, which hides the learned pooling.
    with torch.no_grad():
        for p in idx.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5))
        idx.k_norm.weight.copy_(torch.empty_like(idx.k_norm.weight).uniform_(0.5, 1.5))

    hidden = torch.empty(B, S, HIDDEN).uniform_(-1.0, 1.0)
    q_resid = torch.empty(B, S, Q_LORA).uniform_(-1.0, 1.0)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, :PAD_ROW1] = False
    # A left-padded row's padded positions carry garbage upstream too; make them
    # non-zero so a port that forgets the mask cannot accidentally agree.
    hidden[1, :PAD_ROW1] = 7.5
    q_resid[1, :PAD_ROW1] = -7.5

    with torch.no_grad():
        # The exact packed state the forward builds (`:795-801`).
        hidden_shape = (B, S, -1, HEAD_DIM)
        q = idx.wq_b(q_resid).view(hidden_shape)
        k = idx.k_norm(idx.wk(hidden)).view(hidden_shape).squeeze(2)
        gate_scores = torch.nn.functional.linear(hidden, idx.index_kpool_compress_gate)
        valid_channel = mask.to(k.dtype)[..., None]
        packed = torch.cat([k, gate_scores, valid_channel], dim=-1)

        valid_keys = packed[..., -1].bool()
        visible = idx.get_visible_tokens(valid_keys=valid_keys, q_length=S, current_length=S)
        pool_keys, pool_indices, pool_valid = idx.get_pooled_states(packed_states=packed)

        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = torch.nn.functional.relu(scores * idx.softmax_scale)
        weights = idx.weights_proj(hidden.to(idx.weights_proj.weight.dtype)).float() * (
            idx.n_heads**-0.5
        )
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        topk = idx(hidden_states=hidden, q_resid=q_resid, attention_mask=mask,
                   past_key_values=None)

    n_pools = pool_keys.shape[1]
    out_w = topk.shape[-1]

    def flat(t):
        return t.reshape(-1).tolist()

    def lit(v):
        # `%.9g` of 0.0 is "0", and `0f` is not a C++ float literal. Every value
        # therefore carries a decimal point before the suffix.
        s = f"{v:.9g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"

    def emit_f(name, t):
        vals = flat(t.float())
        print(f"// {name}: {list(t.shape)}")
        print(f"inline constexpr float {name}[] = {{")
        for i in range(0, len(vals), 6):
            print("    " + ", ".join(lit(v) for v in vals[i:i + 6]) + ",")
        print("};")

    def emit_i(name, t):
        vals = [int(v) for v in flat(t.long())]
        print(f"// {name}: {list(t.shape)}")
        print(f"inline constexpr int32_t {name}[] = {{")
        for i in range(0, len(vals), 12):
            print("    " + ", ".join(str(v) for v in vals[i:i + 12]) + ",")
        print("};")

    print("// GENERATED by tests/vllm/models/fixtures/gen_glm5_next_dsa_goldens.py.")
    print("// DO NOT EDIT BY HAND. Oracle: transformers "
          f"{transformers.__version__}, torch {torch.__version__}.")
    print("// `Glm5NextTextIndexer` @ modular_glm5_next.py:749-1022.")
    print("#pragma once")
    print("#include <cstdint>")
    print()
    print("namespace glm5_next_dsa_goldens {")
    print()
    print(f"inline constexpr int64_t kBatch = {B};")
    print(f"inline constexpr int64_t kSeqLen = {S};")
    print(f"inline constexpr int64_t kHidden = {HIDDEN};")
    print(f"inline constexpr int64_t kQLora = {Q_LORA};")
    print(f"inline constexpr int64_t kNHeads = {N_HEADS};")
    print(f"inline constexpr int64_t kHeadDim = {HEAD_DIM};")
    print(f"inline constexpr int64_t kIndexTopk = {INDEX_TOPK};")
    print(f"inline constexpr int64_t kIndexKpool = {INDEX_KPOOL};")
    print(f"inline constexpr int64_t kPadRow1 = {PAD_ROW1};")
    print(f"inline constexpr int64_t kNumPools = {n_pools};")
    print(f"inline constexpr int64_t kOutputWidth = {out_w};")
    print()
    emit_f("kHiddenStates", hidden)
    emit_f("kQResid", q_resid)
    emit_i("kMask", mask)
    print()
    emit_f("kWqB", idx.wq_b.weight)
    emit_f("kWk", idx.wk.weight)
    emit_f("kKNormWeight", idx.k_norm.weight)
    emit_f("kKNormBias", idx.k_norm.bias)
    emit_f("kWeightsProj", idx.weights_proj.weight)
    emit_f("kKpoolApe", idx.index_kpool_compress_ape)
    emit_f("kKpoolGate", idx.index_kpool_compress_gate)
    print()
    emit_i("kVisible", visible)
    emit_f("kPoolKeys", pool_keys)
    emit_i("kPoolIndices", pool_indices)
    emit_i("kPoolValid", pool_valid)
    emit_f("kIndexScores", index_scores)
    emit_i("kTopkIndices", topk)

    # ── the SHORT cases ─────────────────────────────────────────────────────
    # Drawn AFTER every tensor above, so the main fixture's bytes are unchanged.
    # Batch is 1 throughout and the arrays are CONCATENATED over the cases; the
    # C++ side walks them with a running offset built from `kShortSeqLen`.
    short_hidden, short_qresid, short_mask, short_topk, short_pools = [], [], [], [], []
    for s_len, pad in SHORT_CASES:
        sh = torch.empty(1, s_len, HIDDEN).uniform_(-1.0, 1.0)
        sq = torch.empty(1, s_len, Q_LORA).uniform_(-1.0, 1.0)
        sm = torch.ones(1, s_len, dtype=torch.bool)
        sm[0, :pad] = False
        sh[0, :pad] = 7.5
        sq[0, :pad] = -7.5
        with torch.no_grad():
            sk = idx.k_norm(idx.wk(sh)).view(1, s_len, -1, HEAD_DIM).squeeze(2)
            sg = torch.nn.functional.linear(sh, idx.index_kpool_compress_gate)
            spacked = torch.cat([sk, sg, sm.to(sk.dtype)[..., None]], dim=-1)
            spk, _, _ = idx.get_pooled_states(packed_states=spacked)
            st = idx(hidden_states=sh, q_resid=sq, attention_mask=sm, past_key_values=None)
        short_pools.append(int(spk.shape[1]))
        short_hidden.append(sh.reshape(-1))
        short_qresid.append(sq.reshape(-1))
        short_mask.append(sm.reshape(-1))
        short_topk.append(st.reshape(-1))

    print()
    print("// The SHORT cases: sequences with NO complete k-pool. Upstream serves")
    print("// them with a tail-only selection; it does not refuse them.")
    print(f"inline constexpr int64_t kShortCases = {len(SHORT_CASES)};")
    print("inline constexpr int64_t kShortSeqLen[] = {"
          + ", ".join(str(s_len) for s_len, _ in SHORT_CASES) + "};")
    print("inline constexpr int64_t kShortPad[] = {"
          + ", ".join(str(pad) for _, pad in SHORT_CASES) + "};")
    print("inline constexpr int64_t kShortNumPools[] = {"
          + ", ".join(str(n) for n in short_pools) + "};")
    emit_f("kShortHidden", torch.cat(short_hidden))
    emit_f("kShortQResid", torch.cat(short_qresid))
    emit_i("kShortMask", torch.cat(short_mask))
    emit_i("kShortTopk", torch.cat(short_topk))

    print()
    print("}  // namespace glm5_next_dsa_goldens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
