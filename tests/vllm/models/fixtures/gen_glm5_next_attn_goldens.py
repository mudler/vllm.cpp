#!/usr/bin/env python3
"""Regenerate `glm5_next_attn_goldens.inc` by RUNNING the reference oracle.

Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation W5b-1, issue #2241,
`.agents/specs/glm5-next-flash.md` section W5b.

WHAT THE ORACLE IS. `transformers` **v5.16.1**, this row's lane pin (W0, #2096,
`.agents/oracles/transformers.md`). vLLM registers no `glm5_next` at our parity
pin `555967922` nor at its `main`, and neither do vllm-omni, SGLang or
llama.cpp, so under AGENTS.md "When vLLM has no implementation" transformers is
the reference for this surface. The identity is ASSERTED below against the
sha256 W3 and W5c both recorded, not assumed from the version string: a version
string is a label on a package and the hash is the file that ran.

Every golden here is a value produced by an UNMODIFIED `Glm5NextTextAttention`
(`modeling_glm5_next.py:1064-1257`), reached through its own `forward` and its
own public sub-methods `expand_kv` (`:1136-1153`) and
`build_attention_mask_from_topk` (`:1218-1256`). Nothing is transcribed.

WHY THIS SHAPE. Four properties, each present because dropping it makes the
file a tautology a wrong port passes:

  1. `kv_lora_rank` 8, `qk_nope_head_dim` 4 and `v_head_dim` 6 are ALL DIFFERENT.
     The checkpoint stores `kv_b_proj` SPLIT with the k half TRANSPOSED
     (llama.cpp #27752 inheriting `conversion/deepseek.py`'s
     `modify_tensors`), so `k_b` is `[H, kv_lora, qk_nope]` and `v_b` is
     `[H, v_head, kv_lora]`. Three distinct widths mean a port that reads either
     half at the other's orientation is a shape error rather than a silent
     value error.
  2. ...and because a shape error is the EASY failure, a SQUARE case follows
     with `kv_lora == qk_nope == v_head == 6`, where the untransposed reading of
     `k_b` is perfectly shape-valid and merely WRONG. That case carries the
     wrong-reading value beside the right one so the gate can assert they differ.
  3. `seq_len` 25 is STRICTLY GREATER than `index_topk` 8, so the DSA selection
     is not the identity and the top-k mask actually masks something. At or
     below `index_topk` every key is visible and `build_attention_mask_from_topk`
     degenerates into the causal mask.
  4. Row 1 is LEFT-PADDED by three tokens. Its padded query rows reach a state
     where EVERY key is masked, and upstream fills that row with
     `torch.finfo(dtype).min` rather than `-inf`, so its softmax is UNIFORM and
     its output is FINITE. A port that writes `-inf` produces NaN there. The
     golden carries that row.

WHY float32. The C++ side is a host f32 reference exactly as
`glm5_next_dsa.cpp`, `glm5_next_mhc.cpp` and `glm5_next_moe.cpp` are, and the
reference's own softmax is `dtype=torch.float32` (`:1056`). Capturing at fp32
makes the module's `.to(dtype)` casts no-ops, which is honest for what this
file gates. The device arm is owed, not implied.

Usage:  python3 gen_glm5_next_attn_goldens.py [--out PATH]
"""

import argparse
import hashlib
import pathlib
import sys

import torch

import transformers
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

EXPECTED_VERSION = "5.16.1"
# `modeling_glm5_next.py` at `refs/tags/v5.16.1`. W3 (#2213) and W5c (#2242)
# both recorded this value; it is re-asserted here rather than trusted.
EXPECTED_SHA256 = "2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b"

# ── the fixture geometry ────────────────────────────────────────────────────
B, S = 2, 25
HIDDEN = 16
Q_LORA = 12
KV_LORA = 8
QK_NOPE = 4          # NOT equal to KV_LORA and NOT equal to V_HEAD
QK_ROPE = 0          # the NoPE condition `validate_architecture` requires
V_HEAD = 6
N_HEADS = 3
INDEX_TOPK = 8
INDEX_KPOOL = 4
IDX_N_HEADS = 8
IDX_HEAD_DIM = 8
PAD_ROW1 = 3
SEED = 20260829

# The cross-layer schedule. Layer 1 is `shared`: it runs NO indexer of its own
# and reuses layer 0's selection (`:1130-1134`, `:1181-1191`). Layer 0 therefore
# has `next_skip_topk` TRUE and propagates its selection upward; layers 2 and 3
# are full with no sharer above them.
INDEXER_TYPES = ["full", "shared", "full", "full"]


def config(kv_lora=KV_LORA, qk_nope=QK_NOPE, qk_rope=QK_ROPE, v_head=V_HEAD):
    return Glm5NextTextConfig(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=qk_rope,
        qk_nope_head_dim=qk_nope,
        v_head_dim=v_head,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        num_hidden_layers=len(INDEXER_TYPES),
        indexer_types=list(INDEXER_TYPES),
        index_topk=INDEX_TOPK,
        index_head_dim=IDX_HEAD_DIM,
        index_n_heads=IDX_N_HEADS,
        index_kpool=INDEX_KPOOL,
        index_kpool_always_select_tail=True,
        attention_bias=False,
        attention_dropout=0.0,
    )


def randomize(module, gen):
    """Real, non-degenerate parameters.

    `zeros`/`ones` is the module's own init for several of these and would make
    the pool softmax uniform and every RMSNorm the identity, which hides
    precisely the learned parts this file exists to gate.
    """
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "layernorm" in name or "k_norm.weight" in name:
                p.copy_(torch.empty_like(p).uniform_(0.5, 1.5, generator=gen))
            else:
                p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=gen))


def split_kv_b(attn, cfg):
    """The CONVERTER's own transform, run here rather than described.

    `conversion/deepseek.py` `DeepseekV2Model.modify_tensors`, inherited by
    `conversion/glm5next.py`'s `Glm5NextModel(GlmMoeDsaModel)` at llama.cpp
    #27752 head `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc`:

        kv_b = W.view(n_head_kv, v_head_dim + qk_nope_head_dim, -1)
        k_b, v_b = split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
        k_b = k_b.transpose(1, 2)

    Returns (k_b, v_b) in the shapes the file carries and `Glm5NextMlaWeights`
    declares: `[H, kv_lora, qk_nope]` and `[H, v_head, kv_lora]`.
    """
    w = attn.kv_b_proj.weight  # [H * (qk_nope + v_head), kv_lora]
    kv_b = w.view(cfg.num_attention_heads, cfg.v_head_dim + cfg.qk_nope_head_dim, -1)
    k_b, v_b = torch.split(kv_b, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=1)
    return k_b.transpose(1, 2).contiguous(), v_b.contiguous()


def intermediates(attn, cfg, hidden):
    """The forward's own intermediates (`:1163-1175`), recomputed by the module."""
    batch, seq = hidden.shape[:-1]
    q_resid = attn.q_a_layernorm(attn.q_a_proj(hidden))
    compressed_kv = attn.kv_a_proj_with_mqa(hidden)
    kv_pass, k_rot = torch.split(
        compressed_kv, [cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1
    )
    k_pass = attn.kv_a_layernorm(kv_pass).view(batch, 1, seq, cfg.kv_lora_rank)
    k_rot = k_rot.view(batch, 1, seq, cfg.qk_rope_head_dim)
    return q_resid, k_pass, k_rot


# ── emission ────────────────────────────────────────────────────────────────
OUT = []


def w(line=""):
    OUT.append(line)


def lit(v):
    # `%.9g` of 0.0 is "0", and `0f` is not a C++ float literal. Every value
    # therefore carries a decimal point before the suffix.
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def emit_f(name, t):
    vals = t.detach().reshape(-1).float().tolist()
    w(f"// {name}: {list(t.shape)}")
    w(f"inline constexpr float {name}[] = {{")
    for i in range(0, len(vals), 6):
        w("    " + ", ".join(lit(v) for v in vals[i:i + 6]) + ",")
    w("};")


def emit_i(name, t):
    vals = [int(v) for v in t.detach().reshape(-1).long().tolist()]
    w(f"// {name}: {list(t.shape)}")
    w(f"inline constexpr int32_t {name}[] = {{")
    for i in range(0, len(vals), 12):
        w("    " + ", ".join(str(v) for v in vals[i:i + 12]) + ",")
    w("};")


def emit_const(name, value):
    w(f"inline constexpr int64_t {name} = {value};")


def emit_mla(prefix, attn, cfg):
    """One layer's MLA weights, in the shapes `Glm5NextMlaWeights` declares."""
    emit_f(prefix + "QAProj", attn.q_a_proj.weight)
    emit_f(prefix + "QANorm", attn.q_a_layernorm.weight)
    emit_f(prefix + "QBProj", attn.q_b_proj.weight)
    emit_f(prefix + "KvAProj", attn.kv_a_proj_with_mqa.weight)
    emit_f(prefix + "KvANorm", attn.kv_a_layernorm.weight)
    k_b, v_b = split_kv_b(attn, cfg)
    emit_f(prefix + "KB", k_b)
    emit_f(prefix + "VB", v_b)
    emit_f(prefix + "OProj", attn.o_proj.weight)


def emit_indexer(prefix, idx):
    emit_f(prefix + "WqB", idx.wq_b.weight)
    emit_f(prefix + "Wk", idx.wk.weight)
    emit_f(prefix + "KNormWeight", idx.k_norm.weight)
    emit_f(prefix + "KNormBias", idx.k_norm.bias)
    emit_f(prefix + "WeightsProj", idx.weights_proj.weight)
    emit_f(prefix + "KpoolApe", idx.index_kpool_compress_ape)
    emit_f(prefix + "KpoolGate", idx.index_kpool_compress_gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).with_name("glm5_next_attn_goldens.inc")),
    )
    args = ap.parse_args()

    if transformers.__version__ != EXPECTED_VERSION:
        raise SystemExit(
            f"oracle identity: expected transformers {EXPECTED_VERSION}, "
            f"got {transformers.__version__}"
        )
    import transformers.models.glm5_next.modeling_glm5_next as mod
    got = hashlib.sha256(pathlib.Path(mod.__file__).read_bytes()).hexdigest()
    if got != EXPECTED_SHA256:
        raise SystemExit(
            f"oracle identity: modeling_glm5_next.py sha256 {got}, "
            f"expected {EXPECTED_SHA256}"
        )

    torch.set_default_dtype(torch.float32)
    gen = torch.Generator().manual_seed(SEED)

    cfg = config()
    # EAGER, deliberately. `build_attention_mask_from_topk` returns the ADDITIVE
    # `finfo.min` mask on this arm and the boolean one on `sdpa` (`:1249-1256`),
    # and the additive arm is the one the module's own forward consumes. The
    # boolean mask is captured separately below, off the same call.
    cfg._attn_implementation = "eager"

    layers = [Glm5NextTextAttention(cfg, layer_idx=i).eval() for i in range(4)]
    for a in layers:
        randomize(a, gen)

    hidden = torch.empty(B, S, HIDDEN).uniform_(-1.0, 1.0, generator=gen)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, :PAD_ROW1] = False
    # A left-padded row carries garbage upstream too; make it non-zero so a port
    # that forgets the mask cannot accidentally agree.
    hidden[1, :PAD_ROW1] = 7.5

    l0, l1 = layers[0], layers[1]
    assert l0.indexer is not None and not l0.skip_topk and l0.next_skip_topk
    assert l1.indexer is None and l1.skip_topk and not l1.next_skip_topk

    with torch.no_grad():
        q_resid0, k_pass0, k_rot0 = intermediates(l0, cfg, hidden)
        key0, value0 = l0.expand_kv(k_pass0, k_rot0)
        out0, _, prop0 = l0(hidden_states=hidden, attention_mask=mask)
        assert prop0 is not None, "layer 0 must propagate its selection upward"
        topk0 = prop0
        # The BOOLEAN mask, off the module's own method on the `sdpa` arm.
        cfg._attn_implementation = "sdpa"
        bool_mask = l0.build_attention_mask_from_topk(
            topk_indices=topk0, query_states=hidden, kv_length=S
        )
        cfg._attn_implementation = "eager"

        # ── the cross-layer share ───────────────────────────────────────────
        # Layer 1 runs with layer 0's selection. This is the value a correct
        # port produces.
        out1, _, prop1 = l1(
            hidden_states=hidden, attention_mask=mask, prev_topk_indices=topk0
        )
        assert prop1 is None, "a shared layer propagates nothing (`:1132-1133`, `:1216`)"

        # ...and this is the value a port that RECOMPUTES produces. Layer 1 has
        # no indexer of its own, so the decoy is layer 2's — a full layer whose
        # weights are real and different. A recomputing port would have to reach
        # for exactly such a set.
        decoy = layers[2].indexer
        topk_recomputed = decoy(
            hidden_states=hidden, q_resid=intermediates(l1, cfg, hidden)[0],
            attention_mask=mask, past_key_values=None,
        )
        saved, l1.indexer, l1.skip_topk = l1.indexer, decoy, False
        out1_recomputed, _, _ = l1(hidden_states=hidden, attention_mask=mask)
        l1.indexer, l1.skip_topk = saved, True

        # ── the SQUARE case: kv_lora == qk_nope == v_head ───────────────────
        sq_cfg = config(kv_lora=6, qk_nope=6, v_head=6)
        sq_cfg._attn_implementation = "eager"
        sq = Glm5NextTextAttention(sq_cfg, layer_idx=0).eval()
        randomize(sq, gen)
        sq_k_b, sq_v_b = split_kv_b(sq, sq_cfg)
        _, sq_k_pass, sq_k_rot = intermediates(sq, sq_cfg, hidden)
        sq_key, sq_value = sq.expand_kv(sq_k_pass, sq_k_rot)
        # What the UNTRANSPOSED reading of `k_b` yields. Shape-valid at this
        # geometry and wrong; the gate asserts ours is the first and not this.
        sq_k_nope_wrong = torch.einsum("btr,hdr->bhtd", sq_k_pass[:, 0], sq_k_b)

        # ── the ROPE half has NO WIDTH, and upstream is what says so ────────
        # `Glm5NextTextConfig` REFUSES any positive `qk_rope_head_dim`:
        # `validate_architecture` (`configuration_glm5_next.py:225-228`) raises
        # "Expecting NoPE for the DSA attention layers, but got {n} as RoPE
        # dim." — MEASURED here rather than described, by constructing one and
        # catching it. So `expand_kv`'s concat at `:1150-1152` copies a
        # zero-width `k_rot` and `key_states` IS `k_nope`, and this port
        # implements no rope branch because upstream can reach none. The C++
        # side mirrors the refusal instead.
        rope_refusal = None
        try:
            config(qk_rope=2)
        except Exception as e:  # noqa: BLE001 -- the message is the golden
            rope_refusal = str(e)
        assert rope_refusal is not None and "NoPE" in rope_refusal, rope_refusal
        assert key0.shape[-1] == QK_NOPE, (
            "with qk_rope 0 the key width IS qk_nope; a port that grew it has a "
            "rope half upstream cannot reach"
        )

    # The refusal's first line only: huggingface_hub wraps it in a
    # StrictDataclassClassValidationError whose repr carries a traceback-shaped
    # preamble, and the sentence upstream authored is the part that gates.
    core = [ln for ln in rope_refusal.splitlines() if "NoPE" in ln][0].strip()
    # huggingface_hub re-raises inside a StrictDataclassClassValidationError and
    # prefixes the line with the wrapped exception's CLASS. The sentence
    # upstream authored starts after it; the class name is the wrapper's, not
    # the reference's, and gating on it would gate huggingface_hub.
    if ": " in core and core.split(": ", 1)[0].endswith("Error"):
        core = core.split(": ", 1)[1]
    rope_refusal_lit = '"' + core.replace('\\', '\\\\').replace('"', '\\"') + '"'

    w("// GENERATED by tests/vllm/models/fixtures/gen_glm5_next_attn_goldens.py.")
    w("// DO NOT EDIT BY HAND. Oracle: transformers "
      f"{transformers.__version__}, torch {torch.__version__}.")
    w("// `Glm5NextTextAttention` @ modeling_glm5_next.py:1064-1257, sha256")
    w(f"// {EXPECTED_SHA256}.")
    w("#pragma once")
    w("#include <cstdint>")
    w()
    w("namespace glm5_next_attn_goldens {")
    w()
    emit_const("kBatch", B)
    emit_const("kSeqLen", S)
    emit_const("kHidden", HIDDEN)
    emit_const("kQLora", Q_LORA)
    emit_const("kKvLora", KV_LORA)
    emit_const("kQkNope", QK_NOPE)
    emit_const("kQkRope", QK_ROPE)
    emit_const("kVHead", V_HEAD)
    emit_const("kNumHeads", N_HEADS)
    emit_const("kIndexTopk", INDEX_TOPK)
    emit_const("kIndexKpool", INDEX_KPOOL)
    emit_const("kIdxNHeads", IDX_N_HEADS)
    emit_const("kIdxHeadDim", IDX_HEAD_DIM)
    emit_const("kPadRow1", PAD_ROW1)
    emit_const("kTopkWidth", topk0.shape[-1])
    w()
    w("// The cross-layer schedule this fixture runs, `indexer_types`. Index 1 is")
    w("// `shared`: no indexer of its own, reuses layer 0's selection.")
    w(f"inline constexpr int64_t kNumLayers = {len(INDEXER_TYPES)};")
    w("inline constexpr bool kIndexerShared[] = {"
      + ", ".join("true" if t == "shared" else "false" for t in INDEXER_TYPES) + "};")
    w()
    emit_f("kHiddenStates", hidden)
    emit_i("kMask", mask)
    w()
    w("// --- layer 0: a FULL indexer layer -------------------------------------")
    emit_mla("kL0", l0, cfg)
    emit_indexer("kL0Idx", l0.indexer)
    w()
    emit_f("kL0QResid", q_resid0)
    emit_f("kL0KPass", k_pass0)
    emit_f("kL0KeyStates", key0)
    emit_f("kL0ValueStates", value0)
    emit_i("kL0Topk", topk0)
    emit_i("kL0BoolMask", bool_mask)
    emit_f("kL0AttnOut", out0)
    w()
    w("// --- layer 1: a SHARED layer -------------------------------------------")
    emit_mla("kL1", l1, cfg)
    w("// The DECOY indexer. Layer 1 has none of its own; these are layer 2's, and")
    w("// they are what a port that RECOMPUTES would have to reach for.")
    emit_indexer("kL1DecoyIdx", layers[2].indexer)
    emit_i("kL1DecoyTopk", topk_recomputed)
    w("// The CORRECT output: layer 1 driven by layer 0's selection.")
    emit_f("kL1SharedAttnOut", out1)
    w("// The output a RECOMPUTING port produces. The gate asserts ours matches")
    w("// the line above and NOT this one, and prints the separation.")
    emit_f("kL1RecomputedAttnOut", out1_recomputed)
    w()
    w("// --- the SQUARE case: kv_lora == qk_nope == v_head == 6 -----------------")
    emit_const("kSqDim", 6)
    w("// Only the four tensors the square case CONSUMES are emitted. An")
    w("// unread golden is how this row already lost two scale defects to 1602")
    w("// passing assertions (spec `## Owed`, W3), so the fixture carries none.")
    emit_f("kSqKvAProj", sq.kv_a_proj_with_mqa.weight)
    emit_f("kSqKvANorm", sq.kv_a_layernorm.weight)
    emit_f("kSqKB", sq_k_b)
    emit_f("kSqVB", sq_v_b)
    emit_f("kSqKPass", sq_k_pass)
    emit_f("kSqKeyStates", sq_key)
    emit_f("kSqValueStates", sq_value)
    w("// `k_nope` as an UNTRANSPOSED reading of `k_b` yields it. Shape-valid at")
    w("// this geometry, and wrong.")
    emit_f("kSqKNopeUntransposed", sq_k_nope_wrong)
    w()
    w("// --- the rope half is UNREPRESENTABLE, and this is upstream saying so ---")
    w("// `Glm5NextTextConfig(qk_rope_head_dim=2)` raises at construction. The")
    w("// message is the reference's own, caught by the generator, so the C++")
    w("// refusal is gated against it rather than against a transcription.")
    w("inline constexpr char kRopeRefusal[] =")
    w(f"    {rope_refusal_lit};")
    w()
    w("}  // namespace glm5_next_attn_goldens")

    pathlib.Path(args.out).write_text("\n".join(OUT) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
