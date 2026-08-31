#!/usr/bin/env python3
"""Regenerate tests/vllm/models/qwen4_exp_forward_goldens.inc from the ORACLE.

Row MODEL-MM-QWEN4-EXP W5f, issues #2031 and #2336, spec
`.agents/specs/qwen4-exp-flash-next.md`.

WHAT THIS GATES. `Qwen4ExpTextModel::Forward` — the LAYER LOOP. Every block it
composes already has its own golden; what none of them can see is the ORDER the
loop puts them in, the hyper-connection arithmetic between them, where the PLE
layer sits in the stack, and the terminal `use_combine=False` mixer. Those are
loop properties and only an end-to-end observation of `Qwen4ExpTextModel`
answers them.

ORACLE. huggingface/transformers **5.16.0**, this row's ACCEPTED lane pin
(`.agents/oracles/transformers.md`), at
`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`, sha256
`77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`. vLLM
registers no `qwen4_exp` at the pinned revision, so there is no primary oracle
to mirror instead.

NOTHING HERE RE-IMPLEMENTS UPSTREAM, and the method differs from
`gen-qwen4-exp-ple-goldens.py` deliberately. That script `exec`s upstream LINE
RANGES because it gates two components. This one gates the WHOLE MODEL, so it
imports and instantiates `Qwen4ExpTextModel` itself and calls its `forward`: the
observation is the installed package's own output, and the sha256 above is
asserted against the file that produced it. Only the CONFIG and the parameter
VALUES are local.

THE PARAMETER VALUES ARE bf16-EXACT ON PURPOSE. Every weight is a multiple of
1/128 in [-0.5, 0.5), which needs at most seven significant bits and therefore
survives a round trip through bf16 unchanged. The oracle and the C++ side
consume the SAME NUMBER rather than two roundings of one decimal literal, so the
residual this gate measures is the forward's arithmetic and not the fixture's
transcription. The script ASSERTS the round trip per tensor rather than assuming
it.

Needs `torch` and `transformers==5.16.0`. Usage:

    python3 scripts/gen-qwen4-exp-forward-goldens.py
"""

import hashlib
import inspect
import pathlib
import sys

import torch

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

PIN_SHA256 = "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"

OUT_PATH = (pathlib.Path(__file__).resolve().parent.parent
            / "tests" / "vllm" / "models" / "qwen4_exp_forward_goldens.inc")

# ─── THE TINY GEOMETRY, AND WHY EACH NUMBER IS THE ONE IT IS ──────────────────
#
# Every dimension is the smallest that keeps a STRUCTURE the released config has
# and that a smaller value would ERASE. A fixture that erases a structure cannot
# gate it, which is the failure this row has produced repeatedly.
#
#   LAYERS = 4 with the released `full_attention_interval` 4 is the 3-linear-
#     then-1-sparse pattern at its shortest: 0,1,2 are Gated DeltaNet and 3 is
#     Qwen Sparse Attention, so BOTH arms of the per-layer branch run and a loop
#     that ran the wrong arm on any layer moves the output.
#   PLE_LAYER_IDS = [2] is ONE-BASED, so the PLE layer is 0-based 1 — an
#     INTERIOR layer and a linear_attention one (upstream refuses PLE on a
#     sparse layer). Interior matters: at index 0 a mis-placed PLE would still
#     see the same input, and at the last index it would still land on the same
#     output. Only an interior position separates "before layer 1" from every
#     other position in the stack.
#   HC_COUNT = 2 is the minimum upstream's `__post_init__` accepts (it rejects
#     <= 1), and the hyper-connection collapse is a MEAN over hc: at hc == 1 the
#     mean, the widen and the injection are all identities.
#   LINEAR_NUM_KEY_HEADS = 2 against LINEAR_NUM_VALUE_HEADS = 6 gives R = 3, the
#     released model's own 16:48 ratio and the smallest pair where the GGUF
#     V-head permutation is NOT its own inverse.
#   NUM_EXPERTS = 4 at top-2, with the router SCALED (see ROUTER_SCALE): four
#     experts choose-two makes the SELECTION non-trivial and leaves the top-k
#     RENORMALIZATION live, which two experts at top-1 cannot — there the routing
#     weight is exactly 1.0 whatever the logits are, so `norm_topk_prob` is
#     unobservable.
#   INDEXER_COMPRESS_RATIO = 2 with INDEXER_BUDGET = 8 over T = 6 leaves the
#     selection SUB-BUDGET, which is the arm that must agree with dense
#     attention exactly.
#   MOE_INTERMEDIATE_SIZE = 8 (not 6) because torch's `grouped_mm` refuses a
#     stride that is not a multiple of 16 bytes; at f32 that is 4 elements.
HIDDEN = 8
LAYERS = 4
VOCAB = 16
SEQ_LEN = 6

# The prompt carries an EOS (id 3) in the INTERIOR at t == 3. EOS is a segment
# boundary in the hashed n-gram construction (`_shift_right_ignore_eos`,
# modeling_qwen4_exp.py:1053-1067), so a loop that fed the PLE layer the wrong
# ids — or fed it activations instead of ids — diverges on the tokens after it.
INPUT_IDS = [5, 9, 13, 3, 7, 2]
EOS_TOKEN_ID = 3

# ─── WHY THE ROUTER IS SCALED, AND WHY THE FIXTURE CARRIES A SALT ────────────
#
# A TOP-K SELECTION HAS BIMODAL ERROR, NOT A TOLERANCE. The oracle runs f32 and
# this tree runs the model path at bf16 (AGENTS.md, "Inherit vLLM defaults"), so
# the MoE router's input differs by a few units in the last place. If two experts'
# logits sit closer together than that, the two sides select DIFFERENT experts and
# the block output moves by O(1) — a fixture artefact that looks exactly like a
# broken layer loop and cannot be absorbed by any tolerance.
#
# Measured on the first draft of this fixture, which had none of this: two
# experts at top-1, unscaled, minimum router margin 0.0164 in logit space against
# a hidden-state residual of 0.0152. Layer 0 flipped an expert and the end-to-end
# residual read 0.466 — thirteen times the bf16 residual of every other stage.
# The loop was correct; the fixture was on the boundary.
#
# TWO THINGS FIX IT, and both are fixture choices rather than tolerances:
#
#   * ROUTER_SCALE multiplies `mlp.gate.weight` and NOTHING else. A uniform
#     random router over an 8-wide hidden separates its experts by ~0.05; a
#     TRAINED router does not, which is the property this fixture was missing.
#     Scaling the router changes only the LOGITS' spread — the routed
#     arithmetic, the expert towers and every other stage are untouched.
#   * SALT perturbs the parameter stream so the search below could pick a draw
#     whose worst margin is large. Both were chosen by MEASURING the minimum
#     margin across all four layers and all six tokens, not by trying values
#     until the case passed.
#
# The chosen pair gives a worst-case margin of 0.5346 against a hidden residual
# of order 0.015 — about a 5x separation once the router's own gain is taken into
# account. The generator EMITS that margin and the suite REQUIRES it stays above
# a floor, so a future regeneration that drifts back onto the boundary fails
# loudly instead of quietly reporting a large residual.
ROUTER_SCALE = 8.0
SALT = 4


def make_config():
    return Qwen4ExpTextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        linear_conv_kernel_dim=4, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=6,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=4, num_experts_per_tok=2,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        hc_count=2, hc_lowrank=4,
        ple_layer_ids=[2], ple_embed_dim=8, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=1, ngram_vocab_size_base=20,
        make_ngram_vocab_size_divisible_by=8, seed=1234,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4,
        indexer_budget=8, indexer_compress_ratio=2,
        eos_token_id=EOS_TOKEN_ID, dtype=torch.float32,
        max_position_embeddings=64,
        # SIGMOID, not the `hidden_act` fallback. The released config.json says
        # `output_gate_type: "sigmoid"` and this tree's shared reader defaults to
        # "silu"; the GDN output gate is the only consumer, no shape check can
        # see the difference, and #489 is that defect in another model. Setting
        # it here makes the loop's HfConfig projection gateable: a projection
        # that rebuilt the config from `Qwen4ExpParams` alone would lose it,
        # because `ParseQwen4ExpParams` VALIDATES the key and stores no field.
        output_gate_type="sigmoid",
    )


def name_tag(name):
    """FNV-1a over the parameter name: a per-tensor stream that cannot collide
    by position, so two tensors of the same shape never receive the same
    values and a transposed or swapped assignment is visible."""
    h = 1469598103934665603
    for ch in name.encode():
        h = ((h ^ ch) * 1099511628211) & ((1 << 64) - 1)
    return h


def lcg_stream(tag, n):
    """`n` values on the bf16-exact grid k/128, k in [-64, 64)."""
    s = (tag * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    out = []
    for _ in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out.append((((s >> 33) & 127) - 64) / 128.0)
    return out


def assert_pin():
    path = inspect.getsourcefile(Qwen4ExpTextModel)
    got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if got != PIN_SHA256:
        raise SystemExit(
            f"ORACLE MISMATCH: {path}\n  expected {PIN_SHA256}\n  got      {got}\n"
            "The lane pin is transformers 5.16.0; this observation would not be "
            "reproducible against it.")
    return path


def fmt(v):
    # `%.9g` of an integral value emits `0` / `-1`, and `0f` is an INTEGER
    # literal with a float suffix, which C++ rejects ("unable to find numeric
    # literal operator"). A decimal point is added when the mantissa has none.
    s = f"{float(v):.9g}"
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s + "f"


def emit_floats(w, name, values):
    w.append(f"static const float {name}[{len(values)}] = {{")
    for i in range(0, len(values), 8):
        w.append("    " + ", ".join(fmt(v) for v in values[i:i + 8]) + ",")
    w.append("};")
    w.append("")


def main():
    src = assert_pin()
    cfg = make_config()
    torch.manual_seed(0)
    model = Qwen4ExpTextModel(cfg).eval()

    with torch.no_grad():
        for name, p in model.named_parameters():
            t = torch.tensor(lcg_stream(name_tag(name) ^ SALT, p.numel()),
                             dtype=torch.float32).view(p.shape)
            # The ONE tensor that is scaled, and only its spread changes. 8.0 is a
            # power of two, so the bf16-exactness of the grid survives it exactly
            # — the assertion below proves that rather than assuming it.
            if name.endswith("mlp.gate.weight"):
                t = t * ROUTER_SCALE
            p.copy_(t)
            if not torch.equal(p.to(torch.bfloat16).to(torch.float32), p):
                raise SystemExit(f"{name}: value is not bf16-exact")

    # THE CONDITIONING MEASUREMENT, taken on the same run that produces the
    # golden rather than on a separate one, so the number cannot describe a
    # different draw than the one below it.
    router_logits = {}
    for i, layer in enumerate(model.layers):
        layer.mlp.gate.register_forward_hook(
            (lambda i: (lambda mod, a, o: router_logits.__setitem__(
                i, o[0].detach().clone())))(i))

    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)

    k = cfg.num_experts_per_tok
    min_margin = float("inf")
    for i in range(LAYERS):
        srt = router_logits[i].sort(dim=-1, descending=True).values
        min_margin = min(min_margin, float((srt[:, k - 1] - srt[:, k]).min()))
    hidden = out.last_hidden_state[0]           # [T, hidden_size]
    if not bool(torch.isfinite(hidden).all()):
        raise SystemExit("the oracle produced a non-finite last_hidden_state")

    params = dict(model.named_parameters())
    ple = model.layers[1].ple.ple_embedding

    w = []
    w.append("// GENERATED by scripts/gen-qwen4-exp-forward-goldens.py "
             "-- do not edit.")
    w.append("// Oracle: huggingface/transformers v5.16.0,")
    w.append("//   src/transformers/models/qwen4_exp/modeling_qwen4_exp.py")
    w.append(f"//   sha256 {PIN_SHA256}")
    w.append("// Observed by instantiating `Qwen4ExpTextModel` and calling its")
    w.append("// `forward`; only the config and the weight VALUES are local, and")
    w.append("// every value is bf16-exact so the two sides consume one number.")
    w.append("//")
    w.append("// WHAT ONLY THIS GOLDEN CAN SEE: the LAYER ORDER (3 Gated DeltaNet")
    w.append("// then 1 Qwen Sparse Attention), the hyper-connection arithmetic")
    w.append("// between the blocks, the PLE layer's POSITION in the stack, the")
    w.append("// `repeat(1,1,hc_count)` widen and the terminal `use_combine=False`")
    w.append("// mixer. Every block below it already has its own golden; none of")
    w.append("// them can see the composition.")
    w.append("")

    consts = [
        ("kFwdHiddenSize", HIDDEN), ("kFwdLayers", LAYERS),
        ("kFwdVocabSize", VOCAB), ("kFwdSeqLen", SEQ_LEN),
        ("kFwdHcCount", cfg.hc_count), ("kFwdHcLowrank", cfg.hc_lowrank),
        ("kFwdNumAttentionHeads", cfg.num_attention_heads),
        ("kFwdNumKeyValueHeads", cfg.num_key_value_heads),
        ("kFwdHeadDim", cfg.head_dim),
        ("kFwdLinearNumKeyHeads", cfg.linear_num_key_heads),
        ("kFwdLinearNumValueHeads", cfg.linear_num_value_heads),
        ("kFwdLinearKeyHeadDim", cfg.linear_key_head_dim),
        ("kFwdLinearValueHeadDim", cfg.linear_value_head_dim),
        ("kFwdLinearConvKernelDim", cfg.linear_conv_kernel_dim),
        ("kFwdNumExperts", cfg.num_experts),
        ("kFwdNumExpertsPerTok", cfg.num_experts_per_tok),
        ("kFwdMoeIntermediateSize", cfg.moe_intermediate_size),
        ("kFwdSharedExpertIntermediateSize", cfg.shared_expert_intermediate_size),
        ("kFwdIndexerNHeads", cfg.indexer_n_heads),
        ("kFwdIndexerKvHeads", cfg.indexer_kv_heads),
        ("kFwdIndexerHeadDim", cfg.indexer_head_dim),
        ("kFwdIndexerBudget", cfg.indexer_budget),
        ("kFwdIndexerCompressRatio", cfg.indexer_compress_ratio),
        ("kFwdPleLayerZeroBased", 1),
        ("kFwdPleEmbedDim", cfg.ple_embed_dim),
        ("kFwdPleConvKernelSize", cfg.ple_conv_kernel_size),
        ("kFwdNgramSize", cfg.ngram_size),
        ("kFwdHeadsPerNgram", cfg.heads_per_ngram),
        ("kFwdNgramVocabBase", cfg.ngram_vocab_size_base),
        ("kFwdNgramVocabDivisor", cfg.make_ngram_vocab_size_divisible_by),
        ("kFwdSeed", cfg.seed),
        ("kFwdEosTokenId", EOS_TOKEN_ID),
        ("kFwdNgramPaddedVocab", int(ple.ngram_embedding.weight.shape[0])),
        ("kFwdNgramHeadDim", int(ple.ngram_embedding.weight.shape[1])),
        ("kFwdRotaryDim", int(cfg.head_dim * getattr(cfg, "partial_rotary_factor", 1.0))),
    ]
    for n, v in consts:
        w.append(f"static const int64_t {n} = {int(v)}LL;")
    w.append("")
    w.append(f'static const char kFwdOutputGateType[] = "{cfg.output_gate_type}";')
    w.append(f"static const double kFwdRmsNormEps = {float(cfg.rms_norm_eps):.17g};")
    w.append(f"static const double kFwdRopeTheta = "
             f"{float(cfg.rope_parameters['rope_theta']):.17g};")
    w.append("")
    w.append("// The WORST router margin over every layer and every token, in")
    w.append("// LOGIT space: min over (layer, token) of logit[rank k-1] -")
    w.append("// logit[rank k]. A top-k selection has BIMODAL error, so this is the")
    w.append("// number that says whether the fixture can be gated by a tolerance")
    w.append("// at all; see the generator's header. The suite REQUIRES it stays")
    w.append("// above a floor, which is what stops a regeneration from drifting")
    w.append("// back onto the boundary and reporting the flip as a loop defect.")
    w.append(f"static const double kFwdMinRouterMargin = {min_margin:.9g};")
    w.append(f"static const double kFwdRouterScale = {ROUTER_SCALE:.9g};")
    w.append("")
    w.append(f"static const int64_t kFwdInputIds[{SEQ_LEN}] = {{"
             + ", ".join(f"{i}LL" for i in INPUT_IDS) + "};")
    w.append("")
    w.append("// The n-gram table LAYOUT upstream derived for this config: the")
    w.append("// per-head vocabulary sizes, their exclusive-prefix offsets and the")
    w.append("// three splitmix64 layer multipliers. Emitted so the C++ side can")
    w.append("// CHECK its own derivation rather than only agree with itself.")
    w.append("static const int64_t kFwdNgramHeadVocabSizes["
             + str(ple.ngram_heads_vocab_sizes.numel()) + "] = {"
             + ", ".join(f"{int(v)}LL" for v in ple.ngram_heads_vocab_sizes.tolist())
             + "};")
    w.append("static const int64_t kFwdNgramHeadOffsets["
             + str(ple.ngram_heads_offsets.numel()) + "] = {"
             + ", ".join(f"{int(v)}LL" for v in ple.ngram_heads_offsets.tolist())
             + "};")
    w.append("static const int64_t kFwdNgramLayerMultipliers["
             + str(ple.layer_multipliers.numel()) + "] = {"
             + ", ".join(f"{int(v)}LL" for v in ple.layer_multipliers.tolist())
             + "};")
    w.append("")

    # ─── the weights, in the ORACLE's own orientation ────────────────────────
    # Each array below is `[out_features, in_features]` exactly as PyTorch's
    # bias-free `nn.Linear` stores it, and each gamma is the RAW HuggingFace
    # value (`Qwen4ExpTextRMSNorm.weight` is zero-init and its forward is
    # `output * (1.0 + weight)`), which is the SAME parameterization
    # `Qwen4ExpWeights` holds after the loader inverts the converter's `+1`
    # fold (#2218). No transposition and no fold happens here: a golden that
    # pre-chewed either would be gating the generator.
    def emit(name, tensor):
        emit_floats(w, name, tensor.detach().flatten().tolist())

    emit("kFwdEmbedTokens", params["embed_tokens.weight"])

    for i in range(LAYERS):
        pre = f"layers.{i}."
        tag = f"kFwdL{i}"
        for site, sfx in (("attn_hyper_connection", "AttnHc"),
                          ("mlp_hyper_connection", "MlpHc")):
            emit(f"{tag}{sfx}Norm", params[f"{pre}{site}.hc_norm.weight"])
            emit(f"{tag}{sfx}Down", params[f"{pre}{site}.input_mix_weight_down.weight"])
            emit(f"{tag}{sfx}Up", params[f"{pre}{site}.input_mix_weight_up.weight"])
            emit(f"{tag}{sfx}Inject", params[f"{pre}{site}.block_inject_weight.weight"])
        if cfg.layer_types[i] == "linear_attention":
            emit(f"{tag}GdnInProjQkv", params[f"{pre}linear_attn.in_proj_qkv.weight"])
            emit(f"{tag}GdnInProjZ", params[f"{pre}linear_attn.in_proj_z.weight"])
            emit(f"{tag}GdnInProjB", params[f"{pre}linear_attn.in_proj_b.weight"])
            emit(f"{tag}GdnInProjA", params[f"{pre}linear_attn.in_proj_a.weight"])
            # `nn.Conv1d` stores [conv_dim, 1, K]; the middle axis is the
            # depthwise group of one and carries no information, so it is
            # flattened to the [conv_dim, K] both loaders produce.
            emit(f"{tag}GdnConv1d", params[f"{pre}linear_attn.conv1d.weight"])
            emit(f"{tag}GdnALog", params[f"{pre}linear_attn.A_log"])
            emit(f"{tag}GdnDtBias", params[f"{pre}linear_attn.dt_bias"])
            # `linear_attn.norm.weight` is the architecture's ONE gamma with no
            # `+1` anywhere: `Qwen4ExpTextRMSNormGated.weight` is ONES-init and
            # its forward multiplies by `weight` directly, which is why
            # ggml-org/llama.cpp#27742 excludes exactly this tensor from the
            # fold and why the loader does not unshift it.
            emit(f"{tag}GdnNorm", params[f"{pre}linear_attn.norm.weight"])
            emit(f"{tag}GdnOutProj", params[f"{pre}linear_attn.out_proj.weight"])
        else:
            emit(f"{tag}QsaQProj", params[f"{pre}self_attn.q_proj.weight"])
            emit(f"{tag}QsaKProj", params[f"{pre}self_attn.k_proj.weight"])
            emit(f"{tag}QsaVProj", params[f"{pre}self_attn.v_proj.weight"])
            emit(f"{tag}QsaOProj", params[f"{pre}self_attn.o_proj.weight"])
            emit(f"{tag}QsaQNorm", params[f"{pre}self_attn.q_norm.weight"])
            emit(f"{tag}QsaKNorm", params[f"{pre}self_attn.k_norm.weight"])
            # ONE `index_qk_proj` upstream; the converter SPLITS it at
            # `indexer_n_heads * indexer_head_dim` and `Qwen4ExpQsaWeights`
            # holds the two halves. Split here, so the C++ side receives what
            # the loader would have produced rather than re-deriving the point.
            iq = cfg.indexer_n_heads * cfg.indexer_head_dim
            qk = params[f"{pre}self_attn.indexer.index_qk_proj.weight"]
            emit(f"{tag}QsaIdxQProj", qk[:iq, :])
            emit(f"{tag}QsaIdxKProj", qk[iq:, :])
            emit(f"{tag}QsaIdxQNorm", params[f"{pre}self_attn.indexer.q_layernorm.weight"])
            emit(f"{tag}QsaIdxKNorm", params[f"{pre}self_attn.indexer.k_layernorm.weight"])
        emit(f"{tag}MoeRouter", params[f"{pre}mlp.gate.weight"])
        # `gate_up_proj` is [E, 2I, H] and upstream chunks the LINEAR OUTPUT in
        # two (`linear(x, gate_up_proj[e]).chunk(2, dim=-1)`), so the first I
        # ROWS are the gate tower and the second I are the up tower. Feeding the
        # halves in the other order is mutation M2 of W5d-4 and is red.
        gu = params[f"{pre}mlp.experts.gate_up_proj"]
        I = cfg.moe_intermediate_size
        emit(f"{tag}MoeGateExps", gu[:, :I, :])
        emit(f"{tag}MoeUpExps", gu[:, I:, :])
        emit(f"{tag}MoeDownExps", params[f"{pre}mlp.experts.down_proj"])
        emit(f"{tag}MoeSharedGate", params[f"{pre}mlp.shared_expert_gate.weight"])
        emit(f"{tag}MoeSharedGateProj", params[f"{pre}mlp.shared_expert.gate_proj.weight"])
        emit(f"{tag}MoeSharedUpProj", params[f"{pre}mlp.shared_expert.up_proj.weight"])
        emit(f"{tag}MoeSharedDownProj", params[f"{pre}mlp.shared_expert.down_proj.weight"])
        if model.layers[i].ple is not None:
            emit(f"{tag}PleNgramTable",
                 params[f"{pre}ple.ple_embedding.ngram_embedding.weight"])
            emit(f"{tag}PleKeyProj", params[f"{pre}ple.key_proj.weight"])
            emit(f"{tag}PleValueProj", params[f"{pre}ple.value_proj.weight"])
            emit(f"{tag}PleNormKey", params[f"{pre}ple.norm_key.weight"])
            emit(f"{tag}PleNormQuery", params[f"{pre}ple.norm_query.weight"])
            emit(f"{tag}PleNormConv", params[f"{pre}ple.norm_conv.weight"])
            emit(f"{tag}PleConv1d", params[f"{pre}ple.conv1d.weight"])

    # The model-level `use_combine=False` mixer. It has NO `block_inject_weight`,
    # and `Qwen4ExpTextModel` has NO final RMSNorm after it — its `hc_norm` is
    # the last normalization in the model.
    emit("kFwdMixerNorm", params["hyper_connection_mixer.hc_norm.weight"])
    emit("kFwdMixerDown", params["hyper_connection_mixer.input_mix_weight_down.weight"])
    emit("kFwdMixerUp", params["hyper_connection_mixer.input_mix_weight_up.weight"])
    if "hyper_connection_mixer.block_inject_weight.weight" in params:
        raise SystemExit("the mixer grew a block_inject_weight; use_combine=False "
                         "is what makes it the terminal collapse")

    w.append("// `Qwen4ExpTextModel.forward`'s `last_hidden_state`, [T, hidden_size].")
    w.append("// THE MIXER'S OUTPUT, not the logits: `Qwen4ExpTextModel` carries no")
    w.append("// `lm_head` (that is `Qwen4ExpForCausalLM`), so this is exactly what")
    w.append("// the layer loop must return.")
    emit_floats(w, "kFwdExpectedHidden", hidden.detach().flatten().tolist())

    OUT_PATH.write_text("\n".join(w) + "\n")
    print(f"wrote {OUT_PATH} from {src}")
    print(f"  {len(w)} lines, |hidden|max = {float(hidden.abs().max()):.6g}")


if __name__ == "__main__":
    main()
