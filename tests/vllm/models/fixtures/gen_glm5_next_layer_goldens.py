#!/usr/bin/env python3
"""Regenerate `glm5_next_layer_goldens.inc` by RUNNING the reference oracle.

Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation W5b-2, issue #2241,
`.agents/specs/glm5-next-flash.md` section W5b-2.

WHAT THE ORACLE IS. `transformers` **v5.16.1**, this row's lane pin (W0, #2096,
`.agents/oracles/transformers.md`). vLLM registers no `glm5_next` at our parity
pin `555967922` nor at its `main`, and neither do vllm-omni, SGLang or
llama.cpp, so under AGENTS.md "When vLLM has no implementation" transformers is
the reference for this surface. The identity is ASSERTED below against the
sha256 W3, W5b-1 and W5c all recorded, not assumed from the version string: a
version string is a label on a package and the hash is the file that ran.

HOW TO GET THAT ORACLE, because 5.16.1 is not what `pip install transformers`
resolves today:

    git -C <transformers clone> worktree add /tmp/tf-5.16.1 v5.16.1
    python3 -m venv --system-site-packages /tmp/tf-venv
    /tmp/tf-venv/bin/pip install 'tokenizers>=0.23.1,<0.24.0' 'safetensors>=0.8.0'
    PYTHONPATH=/tmp/tf-5.16.1/src /tmp/tf-venv/bin/python gen_glm5_next_layer_goldens.py

Every golden here is a value produced by an UNMODIFIED `Glm5NextTextModel`
(`modeling_glm5_next.py:1409-1494`) and its UNMODIFIED
`Glm5NextTextDecoderLayer` (`:1259-1329`), reached through their own `forward`.
Nothing is transcribed.

WHY THIS SHAPE. Six properties, each present because dropping it makes the file
a tautology a wrong port passes:

  1. FIVE layers with a MIXED schedule, so every one of the four control-flow
     combinations `:1261-1272` can select is exercised AND the fifth case — a
     `shared` DSA layer — is reached:
         0  linear_attention + dense       (KDA arm, `Glm5NextTextMLP`)
         1  deepseek_sparse_attention + dense   (`full` indexer, propagates)
         2  deepseek_sparse_attention + sparse  (`shared` indexer)
         3  linear_attention + sparse      (KDA arm, `Glm5NextTextMoE`)
         4  deepseek_sparse_attention + sparse  (`full` indexer)
     A schedule of all-one-kind gates the arm that ran and says nothing about
     the selection between them.
  2. `hc_mult` is 4, the PUBLISHED value, and the per-layer `[B, S, 4, H]`
     hidden streams are captured for EVERY layer. A port that threads
     `[B, S, H]` and collapses early runs, is finite, and emits fluent tokens;
     the only thing that separates it from the reference is these values.
  3. ...and because "the values differ" is worth nothing if the four streams
     happen to agree, `kStreamSeparation` carries the MINIMUM pairwise distance
     between streams over every token of the golden. A port that broadcasts one
     stream into four produces zero there.
  4. ...and because a decoy is stronger than a margin, `kEarlyCollapseFinal` is
     what the SAME oracle modules produce when the manifold is collapsed to its
     mean and re-broadcast after every layer — the exact fluent-wrong-port
     shape. The gate asserts ours matches the real final state and DIFFERS from
     this one.
  5. `seq_len` 12 is STRICTLY GREATER than `index_topk` 8, so the DSA selection
     is not the identity, and row 1 is LEFT-PADDED by three tokens, so the KDA
     arm's `apply_mask_to_padding_states` (`:636`) and the indexer's validity
     channel both do something.
  6. The CACHED continuation is captured from the reference's own `DynamicCache`:
     an 8-token prefill followed by a 4-token continuation, against the 12-token
     one-shot. A port whose cache is write-only passes every single-step gate.

WHY float32. The C++ side is a host f32 reference exactly as every other
glm5_next primitive on this row is, and the reference computes the mHC mapping
(`:278`) and both RMSNorms (`:77`) in fp32 already. Capturing at fp32 makes the
modules' `.to(dtype)` casts no-ops, which is honest for what this file gates.
The device arm is owed, not implied.

Usage:  python3 gen_glm5_next_layer_goldens.py [--out PATH]
"""

import argparse
import hashlib
import pathlib
import sys

import torch

import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

EXPECTED_VERSION = "5.16.1"
# `modeling_glm5_next.py` at `refs/tags/v5.16.1`. W3 (#2213), W5b-1 (#2324) and
# W5c (#2242) all recorded this value; it is re-asserted here rather than trusted.
EXPECTED_SHA256 = "2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b"

# ── the fixture geometry ────────────────────────────────────────────────────
B, S = 2, 12
PAD_ROW1 = 3          # row 1 is left-padded; its first three query rows see nothing
PREFILL, CONT = 8, 4  # the cached split, PREFILL + CONT == S

HIDDEN = 16
INTERMEDIATE = 20     # the DENSE MLP width; NOT moe_intermediate_size
VOCAB = 32

# MLA — three DISTINCT widths, so a port that reads either `kv_b_proj` half at
# the other's orientation is a shape error rather than a silent value error.
Q_LORA = 12
KV_LORA = 8
QK_NOPE = 4
QK_ROPE = 0           # the NoPE condition `validate_architecture` requires
V_HEAD = 6
N_HEADS = 3

# the DSA indexer
INDEX_TOPK = 8
INDEX_KPOOL = 4
IDX_N_HEADS = 8
IDX_HEAD_DIM = 8

# KDA. `gate_lower_bound` PRESENT selects the sigmoid forget gate, which is the
# branch the published checkpoint takes and the one `kimi_kda.cpp` does not have.
LIN_HEADS = 2
LIN_HEAD_DIM = 6
LIN_CONV_K = 4
GATE_LOWER_BOUND = -5.0

# mHC — `hc_mult` is the PUBLISHED 4. The manifold is the point of this file.
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6

# MoE
N_ROUTED = 8
N_SHARED = 1
TOP_K = 2
MOE_INTERMEDIATE = 12   # a multiple of 4 floats: `torch.grouped_mm`, which the
                        # reference MoE integration selects, refuses a stride
                        # that is not 16-byte aligned.
ROUTED_SCALING = 2.5
SWIGLU_LIMIT = 10.0

RMS_EPS = 1e-5
SEED = 20260830

# The schedule. See docstring property 1.
LAYER_TYPES = [
    "linear_attention",
    "deepseek_sparse_attention",
    "deepseek_sparse_attention",
    "linear_attention",
    "deepseek_sparse_attention",
]
MLP_TYPES = ["dense", "dense", "sparse", "sparse", "sparse"]
# Layer 2 is `shared`, so layer 1 has `next_skip_topk` TRUE and propagates.
INDEXER_TYPES = ["full", "full", "shared", "full", "full"]
NUM_LAYERS = len(LAYER_TYPES)


def config():
    return Glm5NextTextConfig(
        vocab_size=VOCAB,
        # The class default is the published checkpoint's 154820, which is
        # outside this fixture's 32-entry vocabulary and makes `nn.Embedding`
        # refuse. Nothing here reads the pad id -- the model is driven from
        # `inputs_embeds` -- so it is set inside the fixture vocabulary rather
        # than the fixture vocabulary being grown to 154880 to accommodate it.
        pad_token_id=0,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=NUM_LAYERS,
        layer_types=list(LAYER_TYPES),
        mlp_layer_types=list(MLP_TYPES),
        indexer_types=list(INDEXER_TYPES),
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        index_topk=INDEX_TOPK,
        index_head_dim=IDX_HEAD_DIM,
        index_n_heads=IDX_N_HEADS,
        index_kpool=INDEX_KPOOL,
        index_kpool_always_select_tail=True,
        linear_attn_config={
            "num_heads": LIN_HEADS,
            "head_dim": LIN_HEAD_DIM,
            "short_conv_kernel_size": LIN_CONV_K,
            "gate_lower_bound": GATE_LOWER_BOUND,
        },
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=HC_SINKHORN_ITERS,
        hc_eps=HC_EPS,
        n_routed_experts=N_ROUTED,
        n_shared_experts=N_SHARED,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING,
        norm_topk_prob=True,
        swiglu_limit=SWIGLU_LIMIT,
        rms_norm_eps=RMS_EPS,
        attention_bias=False,
        attention_dropout=0.0,
    )


def randomize(module, gen):
    """Real, non-degenerate parameters.

    `zeros`/`ones` is the module's own init for several of these — the mHC
    `base`, the indexer's k-pool APE, every RMSNorm gain — and would make the
    Sinkhorn projection uniform, the pool softmax uniform and every norm the
    identity, which hides precisely the learned parts this file exists to gate.
    """
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("A_log"):
                # `A_log` is exponentiated in the forget gate; a large positive
                # draw saturates the sigmoid and flattens the decay to a
                # constant, which would make the recurrence unobservable.
                p.copy_(torch.empty_like(p).uniform_(-1.0, 0.5, generator=gen))
            elif "norm" in name.lower() and name.endswith("weight"):
                p.copy_(torch.empty_like(p).uniform_(0.5, 1.5, generator=gen))
            elif name.endswith("hc.scale") or name.endswith("_hc.scale"):
                # The three mHC gains. Kept near 1 so `pre`, `post` and the
                # Sinkhorn logits all stay in a range where the projection has
                # something to do.
                p.copy_(torch.empty_like(p).uniform_(0.5, 1.5, generator=gen))
            else:
                p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5, generator=gen))


def split_kv_b(attn, cfg):
    """The CONVERTER's own transform, run here rather than described.

    `conversion/deepseek.py` `DeepseekV2Model.modify_tensors`, inherited by
    `conversion/glm5next.py`'s `Glm5NextModel(GlmMoeDsaModel)` at llama.cpp
    #27752 head `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc`. Returns (k_b, v_b)
    in the shapes the file carries and `MlaWeights` declares:
    `[H, kv_lora, qk_nope]` and `[H, v_head, kv_lora]`.
    """
    w_ = attn.kv_b_proj.weight
    kv_b = w_.view(cfg.num_attention_heads, cfg.v_head_dim + cfg.qk_nope_head_dim, -1)
    k_b, v_b = torch.split(kv_b, [cfg.qk_nope_head_dim, cfg.v_head_dim], dim=1)
    return k_b.transpose(1, 2).contiguous(), v_b.contiguous()


def run_layers(model, cfg, embeds, mask, collapse_every_layer=False):
    """The `:1477-1493` loop, run over the model's OWN layer modules.

    `collapse_every_layer` is the DECOY of docstring property 4: it collapses the
    manifold to its mean and re-broadcasts it after every layer, which is what a
    port that threads `[B, S, H]` computes. Every value stays finite and the
    shape never changes, which is exactly why no shape check catches it.
    """
    hidden = embeds.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    per_layer = []
    topk_per_layer = []
    topk = None
    with torch.no_grad():
        for i, layer in enumerate(model.layers[: cfg.num_hidden_layers]):
            hidden, topk = layer(
                hidden,
                attention_mask=mask,
                position_ids=None,
                position_embeddings=None,
                past_key_values=None,
                prev_topk_indices=topk,
            )
            if collapse_every_layer:
                hidden = hidden.mean(dim=2, keepdim=True).expand(
                    -1, -1, cfg.hc_mult, -1
                ).contiguous()
            per_layer.append(hidden.clone())
            topk_per_layer.append(None if topk is None else topk.clone())
        final = model.norm(model.hc_head(hidden))
    return per_layer, topk_per_layer, final


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


def emit_u8(name, t):
    vals = [int(v) for v in t.detach().reshape(-1).long().tolist()]
    w(f"// {name}: {list(t.shape)}")
    w(f"inline constexpr uint8_t {name}[] = {{")
    for i in range(0, len(vals), 24):
        w("    " + ", ".join(str(v) for v in vals[i:i + 24]) + ",")
    w("};")


def emit_const(name, value):
    w(f"inline constexpr int64_t {name} = {value};")


def emit_hc(prefix, hc):
    emit_f(prefix + "Fn", hc.fn)
    emit_f(prefix + "Base", hc.base)
    emit_f(prefix + "Scale", hc.scale)


def emit_mla(prefix, attn, cfg):
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


def emit_kda(prefix, sa):
    """One KDA layer, in the shapes `Glm5NextKdaLayerWeights` declares.

    The THREE separate `{q,k,v}_conv1d` the checkpoint stores are what the C++
    side takes; the reference declares ONE grouped conv over the concatenated
    `[q; k; v]` channel axis, so the grouped kernel is SPLIT here in q, k, v
    order — the inverse of `Glm5NextMixedQkvConvWeight`. Any other order is a
    silent permutation of channels, so doing the split here (from the module the
    oracle actually ran) rather than in the test is what makes the order gated.
    """
    emit_f(prefix + "QProj", sa.q_proj.weight)
    emit_f(prefix + "KProj", sa.k_proj.weight)
    emit_f(prefix + "VProj", sa.v_proj.weight)
    conv = sa.conv1d.weight.squeeze(1)  # [3 * qkv_dim, K]
    qkv = conv.shape[0] // 3
    emit_f(prefix + "QConv", conv[:qkv])
    emit_f(prefix + "KConv", conv[qkv:2 * qkv])
    emit_f(prefix + "VConv", conv[2 * qkv:])
    # The forget gate is its own module upstream (`Glm5NextTextForgetGate`);
    # `Glm5NextKdaLayerWeights` keeps its four tensors flat on the layer, which
    # is also how the checkpoint stores them.
    emit_f(prefix + "FAProj", sa.forget_gate.f_a_proj.weight)
    emit_f(prefix + "FBProj", sa.forget_gate.f_b_proj.weight)
    emit_f(prefix + "DtBias", sa.forget_gate.dt_bias)
    emit_f(prefix + "ALog", sa.forget_gate.A_log)
    emit_f(prefix + "BProj", sa.b_proj.weight)
    emit_f(prefix + "GAProj", sa.g_a_proj.weight)
    emit_f(prefix + "GBProj", sa.g_b_proj.weight)
    emit_f(prefix + "ONorm", sa.o_norm.weight)
    emit_f(prefix + "OProj", sa.o_proj.weight)


def emit_dense_mlp(prefix, mlp):
    emit_f(prefix + "Gate", mlp.gate_proj.weight)
    emit_f(prefix + "Up", mlp.up_proj.weight)
    emit_f(prefix + "Down", mlp.down_proj.weight)


def emit_moe(prefix, moe):
    emit_f(prefix + "Router", moe.gate.weight)
    emit_f(prefix + "RouterBias", moe.gate.e_score_correction_bias)
    emit_f(prefix + "ExpertGateUp", moe.experts.gate_up_proj)
    emit_f(prefix + "ExpertDown", moe.experts.down_proj)
    emit_dense_mlp(prefix + "Shared", moe.shared_experts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(pathlib.Path(__file__).with_name("glm5_next_layer_goldens.inc")),
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
            f"oracle identity: modeling_glm5_next.py sha256 {got} != "
            f"{EXPECTED_SHA256}"
        )

    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    cfg = config()

    # The schedule the CONFIG resolved, not the one this file asked for.
    # `__post_init__` rewrites `full_attention` and can substitute defaults, so
    # asserting the resolved value is what makes the C++ schedule a port of
    # upstream's rather than of this generator's constants.
    assert list(cfg.layer_types) == LAYER_TYPES, cfg.layer_types
    assert list(cfg.mlp_layer_types) == MLP_TYPES, cfg.mlp_layer_types
    assert list(cfg.indexer_types) == INDEXER_TYPES, cfg.indexer_types
    assert cfg.hc_mult == HC_MULT
    assert cfg.linear_lower_bound == GATE_LOWER_BOUND, cfg.linear_lower_bound
    assert cfg.linear_num_heads == LIN_HEADS and cfg.linear_head_dim == LIN_HEAD_DIM
    assert cfg.linear_conv_kernel_dim == LIN_CONV_K

    # EAGER, and this is not a style choice. `Glm5NextPreTrainedModel` sets
    # `_supports_sdpa = True`, so the DEFAULT `_attn_implementation` this config
    # resolves is `sdpa`, and `build_attention_mask_from_topk` returns a BOOLEAN
    # mask on that arm (`:1249-1250`) instead of the additive `finfo.min` one the
    # eager arm builds (`:1252-1256`). The two disagree on a query row where
    # EVERY key is masked -- a left-padded row -- because torch's SDPA emits
    # ZERO there while eager's uniform softmax emits the mean of the values. Our
    # port inlines `eager_attention_forward` (`:1039-1061`), which is what
    # W5b-1 gated and what `:1227-1228` names as the only interface this model's
    # 3-D per-(query, key) mask can reach, so the oracle is pinned to the same
    # arm rather than the goldens being captured from the other one. Measured:
    # leaving this at the default put the reference's padded rows at 0.0 against
    # our 0.509 and reddened four of ten cases.
    cfg._attn_implementation = "eager"
    model = Glm5NextTextModel(cfg).to(torch.float32)
    randomize(model, gen)
    model.eval()

    embeds = torch.empty(B, S, HIDDEN).uniform_(-1.0, 1.0, generator=gen)
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, :PAD_ROW1] = False

    per_layer, topk_per_layer, final = run_layers(model, cfg, embeds, mask)
    _, _, early = run_layers(model, cfg, embeds, mask, collapse_every_layer=True)

    # The minimum pairwise stream distance in the golden — docstring property 3.
    # A port that broadcasts one stream into `hc_mult` produces ZERO here, so a
    # non-zero value is what makes the per-stream assertions discriminating.
    last = per_layer[-1]  # [B, S, hc, H]
    seps = []
    for a in range(HC_MULT):
        for b_ in range(a + 1, HC_MULT):
            seps.append((last[:, :, a, :] - last[:, :, b_, :]).abs().max().item())
    stream_sep = min(seps)
    assert stream_sep > 1e-3, f"the fixture's streams barely differ: {stream_sep}"

    # The CACHED continuation — docstring property 6. The reference's own
    # `DynamicCache` carries the state, so this is upstream's cache semantics and
    # not this port's opinion of them.
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        pre_out = model(
            inputs_embeds=embeds[:, :PREFILL],
            attention_mask=mask[:, :PREFILL],
            past_key_values=cache,
            use_cache=True,
        )
        cont_out = model(
            inputs_embeds=embeds[:, PREFILL:],
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
        )
    with torch.no_grad():
        one_shot = model(
            inputs_embeds=embeds, attention_mask=mask, use_cache=False
        ).last_hidden_state
    cached_tail = cont_out.last_hidden_state
    tail_gap = (cached_tail - one_shot[:, PREFILL:]).abs().max().item()

    w("// GENERATED by tests/vllm/models/fixtures/gen_glm5_next_layer_goldens.py.")
    w("// DO NOT EDIT. Regenerate with the recipe in that file's docstring.")
    w("//")
    w(f"// Oracle: transformers {EXPECTED_VERSION},")
    w(f"//   models/glm5_next/modeling_glm5_next.py sha256 {EXPECTED_SHA256}")
    w("// Captured by RUNNING `Glm5NextTextModel` and `Glm5NextTextDecoderLayer`.")
    w("#pragma once")
    w("#include <cstdint>")
    w("")
    w("namespace glm5_next_layer_goldens {")
    w("")
    emit_const("kBatch", B)
    emit_const("kSeqLen", S)
    emit_const("kPadRow1", PAD_ROW1)
    emit_const("kPrefill", PREFILL)
    emit_const("kCont", CONT)
    emit_const("kHidden", HIDDEN)
    emit_const("kIntermediate", INTERMEDIATE)
    emit_const("kNumLayers", NUM_LAYERS)
    emit_const("kHcMult", HC_MULT)
    emit_const("kHcSinkhornIters", HC_SINKHORN_ITERS)
    emit_const("kNHeads", N_HEADS)
    emit_const("kQLora", Q_LORA)
    emit_const("kKvLora", KV_LORA)
    emit_const("kQkNope", QK_NOPE)
    emit_const("kVHead", V_HEAD)
    emit_const("kIndexTopk", INDEX_TOPK)
    emit_const("kIndexKpool", INDEX_KPOOL)
    emit_const("kIdxNHeads", IDX_N_HEADS)
    emit_const("kIdxHeadDim", IDX_HEAD_DIM)
    emit_const("kLinHeads", LIN_HEADS)
    emit_const("kLinHeadDim", LIN_HEAD_DIM)
    emit_const("kLinConvK", LIN_CONV_K)
    emit_const("kNRouted", N_ROUTED)
    emit_const("kNShared", N_SHARED)
    emit_const("kTopK", TOP_K)
    emit_const("kMoeIntermediate", MOE_INTERMEDIATE)
    emit_const("kVocab", VOCAB)
    w(f"inline constexpr double kRmsNormEps = {RMS_EPS!r};")
    w(f"inline constexpr double kHcEps = {HC_EPS!r};")
    w(f"inline constexpr double kGateLowerBound = {GATE_LOWER_BOUND!r};")
    w(f"inline constexpr float kRoutedScaling = {lit(ROUTED_SCALING)};")
    w(f"inline constexpr float kSwigluLimit = {lit(SWIGLU_LIMIT)};")
    w("")
    w("// The RESOLVED schedule, read off the config object after `__post_init__`")
    w("// rather than off this generator's constants: the C++ side is a port of")
    w("// what upstream resolved, and `full_attention` is rewritten in there.")
    w("inline constexpr const char* kLayerTypes[] = {")
    for t in cfg.layer_types:
        w(f'    "{t}",')
    w("};")
    w("inline constexpr const char* kMlpTypes[] = {")
    for t in cfg.mlp_layer_types:
        w(f'    "{t}",')
    w("};")
    w("inline constexpr const char* kIndexerTypes[] = {")
    for t in cfg.indexer_types:
        w(f'    "{t}",')
    w("};")
    w("")

    emit_f("kInputsEmbeds", embeds)
    emit_u8("kMask", mask.to(torch.uint8))
    w("")
    w("// --- the weight tower, layer by layer ---------------------------------")
    for i in range(NUM_LAYERS):
        layer = model.layers[i]
        p = f"kL{i}"
        w(f"// layer {i}: {cfg.layer_types[i]} + {cfg.mlp_layer_types[i]}"
          f" (indexer {cfg.indexer_types[i]})")
        emit_f(p + "InputNorm", layer.input_layernorm.weight)
        emit_f(p + "PostAttnNorm", layer.post_attention_layernorm.weight)
        emit_hc(p + "AttnHc", layer.attn_hc)
        emit_hc(p + "FfnHc", layer.ffn_hc)
        if cfg.layer_types[i] == "linear_attention":
            emit_kda(p + "Kda", layer.self_attn)
        else:
            emit_mla(p + "Mla", layer.self_attn, cfg)
            if layer.self_attn.indexer is not None:
                emit_indexer(p + "Idx", layer.self_attn.indexer)
        if cfg.mlp_layer_types[i] == "dense":
            emit_dense_mlp(p + "Mlp", layer.mlp)
        else:
            emit_moe(p + "Moe", layer.mlp)
        w("")
    emit_f("kFinalNorm", model.norm.weight)
    w("")

    w("// --- the manifold, layer by layer -------------------------------------")
    w("// [batch, seq_len, hc_mult, hidden]. THE assertion of this wave: a port")
    w("// that threads [batch, seq_len, hidden] cannot produce these.")
    for i, h in enumerate(per_layer):
        emit_f(f"kL{i}Streams", h)
    w("")
    w("// The MINIMUM pairwise distance between the four streams of the last")
    w("// layer's manifold. A port that broadcasts one stream into four scores")
    w("// zero here, so this is what makes the per-stream goldens discriminating")
    w("// rather than four copies of one assertion.")
    w(f"inline constexpr float kStreamSeparation = {lit(stream_sep)};")
    w("")
    w("// `self.norm(self.hc_head(hidden_states))` (`:1493`).")
    emit_f("kFinalHidden", final)
    w("")
    w("// THE DECOY. The SAME oracle modules, with the manifold collapsed to its")
    w("// mean and re-broadcast after every layer -- what a port that threads")
    w("// [batch, seq_len, hidden] computes. Finite, right-shaped, fluent, wrong.")
    emit_f("kEarlyCollapseFinal", early)
    early_gap = (final - early).abs().max().item()
    w(f"inline constexpr float kEarlyCollapseGap = {lit(early_gap)};")
    w("")

    w("// --- the `prev_topk_indices` thread ------------------------------------")
    w("// What each layer RETURNS at `:1329`. A KDA layer returns nothing and")
    w("// WIPES the thread; a `full` DSA layer returns its selection only when")
    w("// `next_skip_topk`; a `shared` layer never propagates.")
    for i, t in enumerate(topk_per_layer):
        if t is None:
            w(f"inline constexpr bool kL{i}Propagates = false;")
        else:
            w(f"inline constexpr bool kL{i}Propagates = true;")
            emit_i(f"kL{i}Topk", t)
            emit_const(f"kL{i}TopkWidth", t.shape[-1])
    w("")

    w("// --- the CACHED continuation ------------------------------------------")
    w("// An 8-token prefill then a 4-token continuation through the reference's")
    w("// own `DynamicCache`, against the 12-token one-shot. A port whose cache")
    w("// is write-only passes every single-step gate and fails this one.")
    emit_f("kCachedPrefillHidden", pre_out.last_hidden_state)
    emit_f("kCachedContHidden", cached_tail)
    w("// The reference's own agreement between the cached tail and the one-shot")
    w("// tail, so the C++ tolerance is set against a MEASURED number rather than")
    w("// against a hope.")
    w(f"inline constexpr float kCachedVsOneShotGap = {lit(tail_gap)};")
    w("")
    w("}  // namespace glm5_next_layer_goldens")

    pathlib.Path(args.out).write_text("\n".join(OUT) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"stream separation {stream_sep:.6g}, early-collapse gap {early_gap:.6g}, "
          f"cached-vs-one-shot gap {tail_gap:.6g}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
