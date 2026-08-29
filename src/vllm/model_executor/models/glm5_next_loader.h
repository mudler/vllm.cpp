// GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) W5c — the loaded weight
// set and the `glm5next` GGUF loader that fills it.
//
// Issue [#2242](https://github.com/mudler/vllm.cpp/issues/2242), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md` §W5c.
//
// Model-private, deliberately not under `include/`: nothing outside this model
// needs these types, and `include/vllm.h` is the ABI seam a SHIPPED capability
// is exposed through. This wave ships a LOAD, not a capability — the FORWARD
// still refuses by name, and W5b
// ([#2241](https://github.com/mudler/vllm.cpp/issues/2241)) owns the forward
// this tower feeds. The KV-CACHE SPEC is no longer part of that sentence: W5
// ([#2223](https://github.com/mudler/vllm.cpp/issues/2223)) wired
// `MakeGlm5NextKVCache` into `kGlm5NextFactory` and it publishes three real
// groups, so this comment names only what is still owed.
//
// ─── WHY THIS FILE IS NOT `glm5_next_weights.h` ──────────────────────────────
// That name is already taken, by the PUBLIC header the `general.architecture`
// dispatch table includes: it holds `Glm5NextHfConfigFromGguf` and the HF ->
// GGUF name map. The sibling row spells the same split
// `qwen4_exp_gguf_weights.h` (config builder) beside `qwen4_exp_weights.h`
// (loader); renaming a public header that `entrypoints/model_loader.cpp`
// includes is churn this wave does not owe, so the loader takes a new name and
// says so here.
//
// ─── WHY GGUF AND ONLY GGUF ──────────────────────────────────────────────────
// Every safetensors artifact of this model is larger than any device this
// project owns: FP8 305.78 GiB and BF16 598.53 GiB from `zai-org/GLM-5.3-Flash`,
// and 181.32 GiB for `LibertAIDAI/...-NVFP4`, against ~119.63 GiB usable on
// GB10. `unsloth/GLM-5.3-Flash-GGUF UD-Q2_K_XL` is 101.2535 GiB in four shards
// and is the ONE published artifact that fits. The safetensors arm is therefore
// not "later because it is easy"; it is deferred because no host we own could
// read it, and the spec records it as owed.
//
// ─── THE TWO ORACLES THIS FILE ANSWERS TO ────────────────────────────────────
// vLLM registers no `glm5_next` at any revision, so under AGENTS.md "When vLLM
// has no implementation" this row runs a SPLIT oracle:
//
//   * the ALGORITHM — what each tensor IS, and what shape it has — is
//     `transformers` **v5.16.1**, this row's lane pin (W0, #2096). Every module
//     and `nn.Parameter` cited below is at
//     `transformers/models/glm5_next/modeling_glm5_next.py` at that tag, whose
//     sha256 is
//     `2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b`
//     — asserted, not assumed, and re-measured for this wave against
//     `raw.githubusercontent.com` at the tag.
//   * the CONTAINER — what the converter DID to those tensors on the way into
//     the file, and therefore what this loader must undo — is llama.cpp pull
//     request [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) at head
//     `8a8d0bcc4d5fdf024c457526245bec4bc3a12adc`, the pin
//     `.agents/oracles/llama-cpp-glm5next.md` records, whose
//     `conversion/glm5next.py` (4714 bytes, sha256
//     `bfacba27746096e7bb3ca4a2549c9026d3475e226c7f3edf230c37ffadc7b6b3`)
//     declares `class Glm5NextModel(GlmMoeDsaModel)`. Both files were READ AT
//     SOURCE for this wave rather than relayed.
//
// The container oracle is an OPEN pull request, and that is a live risk rather
// than a footnote: if #27752 renames a key or a tensor before it merges, this
// file changes and the committed manifest is regrown. There is no compatibility
// shim for a spelling that never shipped.
//
// ─── THE FOUR CONVERT-TIME TRANSFORMS THIS FILE INVERTS ──────────────────────
//
// 1. `ssm_a = -exp(A_log)`, so the loader recovers `A_log = log(-x)`
//    (`conversion/glm5next.py`: `if name.endswith(".A_log"): data_torch =
//    -torch.exp(data_torch.float())`). Identical to what
//    `qwen4_exp_weights.cpp` and `qwen3_5_gguf_weights.cpp` already do.
//    THE SIGN IS A GATE, not a formality: `log(-x)` on a positive `x` is NaN,
//    so a file written without the transform poisons every decay in the layer
//    rather than shifting it, and the loader refuses by name instead.
//
// 2. `dt_bias` IS RENAMED, not reshaped. `conversion/glm5next.py`:
//    `if name.endswith(".dt_bias"): name = name.rpartition(".dt_bias")[0] +
//    ".dt_proj.bias"`, which maps through `MODEL_TENSOR.SSM_DT`
//    (`gguf-py/gguf/tensor_mapping.py`, `model.layers.{bid}.self_attn.dt_proj`
//    — the `# kimi` row) to **`blk.N.ssm_dt.bias`**. The converter's own
//    comment says why: "the time-step bias to be named like a bias so it is
//    not loaded as a MUL_MAT weight."
//
// 3. `kv_b_proj` IS SPLIT AND HALF-TRANSPOSED, inherited from
//    `DeepseekV2Model.modify_tensors` (`conversion/deepseek.py`, "MLA with the
//    absorption optimization, needs these two split and k_b_proj transposed"):
//
//        kv_b = W.view(n_head_kv, v_head_dim + qk_nope_head_dim, -1)
//        k_b, v_b = split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
//        k_b = k_b.transpose(1, 2)
//
//    so the file carries `attn_k_b.weight` `[H, kv_lora, qk_nope]` and
//    `attn_v_b.weight` `[H, v_head_dim, kv_lora]` and NO `attn_kv_b.weight`.
//    Both halves are kept in the file's own shape here rather than re-fused:
//    the absorbed form is what an MLA forward wants, and re-fusing at load
//    would undo the transform only to have W5b redo it.
//
// 4. THE EXPERTS ARE STACKED, `torch.stack(datas, dim=0)` over the 288 routed
//    experts, in the order `down_proj, gate_proj, up_proj`. Nothing to invert —
//    stated so the 3-D shape is not read as a defect.
//
// AND ONE NON-TRANSFORM, stated because the sibling row HAS it and copying that
// file is the natural move: **there is no `+1` norm fold here.** The Qwen3-Next
// converter adds 1.0 to every `norm.weight`; the DeepSeek/GLM chain this
// architecture converts through does not, at any level
// (`ModelBase.modify_tensors` is the identity for a norm, and neither
// `DeepseekV2Model` nor `GlmMoeDsaModel` touches one). A loader that inherited
// the fold would subtract 1.0 from every gamma in the model, which is a uniform
// ~1.0-sized error on every normalization and a fluent wrong model.
//
// ─── WHAT IS NOT HERE ────────────────────────────────────────────────────────
// The forward, the KV-cache spec, the vision tower, the MTP head, and the
// safetensors arm. All five refuse by name and all five are listed under
// `## Owed` in the spec with the wave that owns them. This header is the load
// and nothing else.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_LOADER_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_LOADER_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/model_registry.h"   // LoadedModel
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// One `Glm5NextTextHyperConnection` (`modeling_glm5_next.py:219-296`). Two per
// decoder layer, one before the attention sublayer and one before the MLP.
//
// The checkpoint stores these FLAT on the layer — `hc_attn_fn`, not
// `attn_hc.fn` — and carries NO `hc_head.*` at any layer, which is what
// independently settles the unweighted-mean head collapse W4 ported: there is
// nothing to weight with.
struct Glm5NextMhcWeights {
  // `self.fn`, `[(2 + hc_mult) * hc_mult, hc_mult * hidden_size]` = [24, 16384]
  // on the published checkpoint. A GEMM operand, so it takes the residency the
  // policy gives it.
  OwnedTensor fn;
  // `self.base`, `[(2 + hc_mult) * hc_mult]` = [24], and `self.scale`, `[3]`.
  //
  // **f32, and this is one of the annotated exceptions.** Upstream computes the
  // whole mHC block in fp32 — `F.linear(flat, self.fn.float())` at :278 on a
  // `.float()`-ed input at :277 — and every Sinkhorn denominator adds
  // `hc_eps = 1e-6` (:283-290). The bf16 quantum near 1.0 is 2^-8 = 3.9e-3,
  // which is 3900x that eps, so a bf16 store would make the eps arithmetically
  // invisible and change what the Sinkhorn projection converges to. The whole
  // model carries 45 * 27 = 1215 of these scalars, so the cost of the exception
  // is 4.86 kB.
  OwnedTensor base;   // f32 [(2 + hc_mult) * hc_mult]
  OwnedTensor scale;  // f32 [3] — the pre, post and comb gains, in that order
};

// One KDA layer: `Glm5NextTextLinearAttention` (`:584-746`) plus its
// `Glm5NextTextForgetGate` (`:305-337`). Field names and shapes mirror
// `Glm5NextKdaLayerWeights` (`include/vllm/model_executor/models/glm5_next_kda.h`)
// one for one, so W5b's bridge from these buffers to that host reference is
// mechanical rather than a second name map.
struct Glm5NextKdaWeights {
  OwnedTensor q_proj;    // [qkv_dim, hidden]
  OwnedTensor k_proj;    // [qkv_dim, hidden]
  OwnedTensor v_proj;    // [qkv_dim, hidden]
  // THREE separate depthwise convs, each [qkv_dim, conv_kernel_size]. The
  // reference declares ONE `nn.Conv1d` over the concatenated `[q; k; v]`
  // channel axis (`conv_dim = 3 * qkv_dim`, `:608-616`), and the shipped
  // checkpoint stores it pre-split as `self_attn.{q,k,v}_conv1d`
  // (`tensor_mapping.py`, `MODEL_TENSOR.SSM_CONV1D_{Q,K,V}`). The file's
  // tensor is 3-D `[qkv_dim, 1, K]` — `nn.Conv1d` stores `[C, 1, K]` — and the
  // middle axis is dropped here so the shape matches what
  // `Glm5NextMixedQkvConvWeight` concatenates.
  OwnedTensor q_conv1d;  // [qkv_dim, conv_kernel_size]
  OwnedTensor k_conv1d;  // [qkv_dim, conv_kernel_size]
  OwnedTensor v_conv1d;  // [qkv_dim, conv_kernel_size]
  OwnedTensor f_a_proj;  // [head_dim, hidden]
  OwnedTensor f_b_proj;  // [qkv_dim, head_dim]
  OwnedTensor g_a_proj;  // [head_dim, hidden]
  OwnedTensor g_b_proj;  // [qkv_dim, head_dim]
  OwnedTensor b_proj;    // [num_heads, hidden]  (`ssm_beta`)
  // **f32, annotated.** `A_log` is RECOVERED here as `log(-ssm_a)` (transform
  // 1) and is then exponentiated in the forget gate, so it is a value this
  // loader computes rather than one it copies; rounding a recomputed log to
  // bf16 costs precision the file never lost. 64 floats per KDA layer.
  OwnedTensor a_log;     // f32 [num_heads]
  // **f32, annotated**, for the same reason `qwen4_exp_weights.h` gives: the
  // time-step bias is added INSIDE the forget gate's `exp`, where a bf16
  // quantum is a multiplicative error on the decay rather than an additive one
  // on a logit. 8192 floats per KDA layer.
  OwnedTensor dt_bias;   // f32 [qkv_dim]
  OwnedTensor o_norm;    // [head_dim]   `RMSNormGated` gamma
  OwnedTensor o_proj;    // [hidden, qkv_dim]
};

// `Glm5NextTextIndexer` (`:736-1062`) — the DSA lightning indexer plus this
// model's net-new k-pool compression stage. Field names mirror
// `vllm::glm5_next_dsa::IndexerWeights`.
struct Glm5NextIndexerWeights {
  OwnedTensor wq_b;           // [n_heads * head_dim, q_lora_rank]
  OwnedTensor wk;             // [head_dim, hidden]
  // `nn.LayerNorm(head_dim, eps=1e-6)` — a LayerNorm WITH BIAS, not an
  // RMSNorm, which the presence of `indexer.k_norm.bias` in the file settles.
  OwnedTensor k_norm_weight;  // [head_dim]
  OwnedTensor k_norm_bias;    // [head_dim]
  OwnedTensor weights_proj;   // [n_heads, hidden]
  // `index_kpool_compress_ape`, a LEARNED intra-pool absolute-position
  // embedding [index_kpool, head_dim], and `index_kpool_compress_gate`,
  // [head_dim, hidden]. Unconditional in the reference: they are declared at
  // `:770-771` with no predicate, and the config's `index_kpool_compress` is
  // an inert kwarg the config class does not declare.
  OwnedTensor kpool_ape;      // [index_kpool, head_dim]
  OwnedTensor kpool_gate;     // [head_dim, hidden]
};

// One DSA layer's NoPE MLA: `Glm5NextTextAttention` (`:1064-1257`) on the
// `q_lora_rank is not None` branch, which is the branch this checkpoint takes.
struct Glm5NextMlaWeights {
  OwnedTensor q_a_proj;            // [q_lora_rank, hidden]
  OwnedTensor q_a_layernorm;       // [q_lora_rank]
  OwnedTensor q_b_proj;            // [num_heads * qk_head_dim, q_lora_rank]
  OwnedTensor kv_a_proj_with_mqa;  // [kv_lora_rank + qk_rope_head_dim, hidden]
  OwnedTensor kv_a_layernorm;      // [kv_lora_rank]
  // Transform 3's two halves, in the file's own absorbed shapes. `k_b_proj` is
  // ALREADY TRANSPOSED by the converter and is left that way.
  OwnedTensor k_b_proj;            // [num_heads, kv_lora_rank, qk_nope_head_dim]
  OwnedTensor v_b_proj;            // [num_heads, v_head_dim, kv_lora_rank]
  OwnedTensor o_proj;              // [hidden, num_heads * v_head_dim]
  Glm5NextIndexerWeights indexer;
};

// A gated (SwiGLU) MLP — the dense layer's `Glm5NextTextMLP` (`:86-105`) and
// the shared expert alike. The clamp is a forward-time constant
// (`swiglu_limit`), not a weight, so nothing here carries it.
struct Glm5NextMlpWeights {
  OwnedTensor gate_proj;  // [intermediate, hidden]
  OwnedTensor up_proj;    // [intermediate, hidden]
  OwnedTensor down_proj;  // [hidden, intermediate]
};

// `Glm5NextTextMoE` (`:186-208`): 288 routed experts at top-8 through a
// sigmoid `noaux_tc` router, plus ONE shared expert.
struct Glm5NextMoeWeights {
  // `Glm5NextTextTopkRouter.weight` [num_experts, hidden].
  //
  // **f32, and upstream is the one who says so**, not this port: the router
  // GEMM is `F.linear(hidden_states.type(torch.float32),
  // self.weight.type(torch.float32))` (`:158`). The file agrees — every
  // `ffn_gate_inp.weight` in the published artifact is F32 — so keeping it f32
  // is mirroring rather than widening.
  OwnedTensor router;
  // `e_score_correction_bias` [num_experts]. **f32, annotated**: it is added to
  // the fp32 sigmoid scores to pick the top-8 (`:160`) and is NOT part of the
  // weight the chosen experts are scaled by (`:174` gathers from `scores`, not
  // from `scores_for_choice`). Expert selection is discrete, so a rounding
  // error here does not scale an output — it swaps an expert, and no tolerance
  // gate can see that.
  OwnedTensor e_score_correction_bias;
  // The 288 routed experts, stacked (transform 4). `[E, moe_I, hidden]` for
  // gate and up, `[E, hidden, moe_I]` for down. These are the tensors the whole
  // 101.14 GiB residency result turns on: 82 of the artifact's tensors are
  // IQ2_XS and 3 are IQ4_XS, and both encodings keep their blocks here only
  // because #2247 landed their `vec_dot`.
  OwnedTensor gate_exps;
  OwnedTensor up_exps;
  OwnedTensor down_exps;
  // `shared_experts`, sized `moe_intermediate_size * n_shared_experts`
  // (`:196-198`) = 2048 on the published checkpoint.
  Glm5NextMlpWeights shared;
};

// `Glm5NextTextDecoderLayer` (`:1259-1331`).
struct Glm5NextLayerWeights {
  bool is_linear_attention = false;  // KDA when true, NoPE MLA + DSA when false
  bool is_dense_mlp = false;         // `Glm5NextTextMLP` when true, MoE when false
  OwnedTensor input_layernorm;           // `attn_norm.weight`
  OwnedTensor post_attention_layernorm;  // `ffn_norm.weight`
  Glm5NextMhcWeights attn_hc;
  Glm5NextMhcWeights mlp_hc;
  Glm5NextKdaWeights kda;   // iff is_linear_attention
  Glm5NextMlaWeights mla;   // iff !is_linear_attention
  Glm5NextMlpWeights dense_mlp;  // iff is_dense_mlp
  Glm5NextMoeWeights moe;        // iff !is_dense_mlp
};

struct Glm5NextWeights {
  Glm5NextParams params;

  // [vocab, hidden]. A GATHER, so it expands to bf16 like every other token
  // table in this tree.
  OwnedTensor embed_tokens;
  // `Glm5NextTextModel.norm`, the final RMSNorm (`:1493`). It exists, unlike
  // the sibling `qwen4exp` where the last normalization is inside the mixer;
  // the file's `output_norm.weight` is what says so.
  OwnedTensor norm;
  // [vocab, hidden] in the file's own order, or EMPTY when the file ties the
  // head to the embedding table. Tie is read off the FILE (is `output.weight`
  // there?) rather than off a config key, because llama.cpp's writer decides it
  // that way and a config that disagreed with the file would be the config's
  // error. The published artifact carries both, so this model is untied.
  OwnedTensor lm_head;
  bool tied_word_embeddings = false;

  std::vector<Glm5NextLayerWeights> layers;

  // STRUCTURAL accounting, the same shape every other loader in this tree
  // reports: how many tensors the name map ENUMERATES for this config, and how
  // many of them the file actually carries. The load below reads through the
  // SAME names, so the two can never disagree.
  int64_t enumerated_tensors = 0;
  int64_t accounted_tensors = 0;

  // THE MTP BLOCKS THIS LOAD DELIBERATELY DROPPED, counted rather than assumed.
  //
  // `blk.45` on the published artifact is a DeepSeek-V3-style multi-token-
  // prediction block, and the reference DISCARDS it —
  // `_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", ...]`. Loading it
  // as a 46th decoder layer builds a fluent wrong model that no token gate on
  // this fleet could detect, because no oracle for this model runs on any
  // device this project reaches.
  //
  // So the exclusion is stated POSITIVELY. `layers.size()` alone cannot tell
  // "45 layers were built from blocks 0..44" from "45 were built from 1..45",
  // and an assertion that no `blk.45.*` name is enumerated cannot tell a
  // deliberate exclusion from a file that never had one. This counts the
  // TENSORS the FILE carries at a block index >= `num_hidden_layers` — 29 of
  // them on the published artifact — so a test can assert that the loader saw
  // the MTP block and did not build it. It counts tensors and not blocks
  // because that is what makes it add up: `enumerated_tensors` plus this is the
  // file's whole table, and a count of blocks would leave that identity
  // unstated and the field unfalsifiable.
  int64_t mtp_block_tensors_dropped = 0;

  // The layer index of each kind, for the same reason: a test that asserts
  // `layers[3].is_linear_attention == false` proves the schedule was READ
  // rather than synthesized only if it also knows the file said so.
  int64_t num_kda_layers() const;
  int64_t num_dsa_layers() const;
};

// Every GGUF tensor name this architecture expects at `params`, in a stable
// order, INCLUDING the model-level three. The load reads through exactly these
// names. Delegates to `Glm5NextExpectedGgufTensors` (the public name map in
// `glm5_next_weights.h`) so there is ONE enumeration and not two.
std::vector<std::string> EnumerateGlm5NextGgufTensors(
    const Glm5NextParams& params);

// Load the text tower. Refuses BY NAME on a missing tensor, a shape
// disagreement, or an encoding this build cannot decode. `policy` is borrowed
// and may be null, in which case `GgufLoadPolicy::FromEnv()` decides residency —
// which is what a production load gets.
Glm5NextWeights LoadGlm5NextFromGguf(const GgufFile& gguf, const HfConfig& config,
                                     const GgufLoadPolicy* policy = nullptr);

// The concrete model the registry's `load_weights` hook produces. It exists so
// the type-erased `LoadedModel` the registry hands around has something real
// behind it, and so `ModelAs<>` has a type to open — never a `static_cast`,
// which is undefined behaviour on an object that is not really this type
// (#775, #730).
//
// DECLARED HERE RATHER THAN IN THE REGISTRY TU'S ANONYMOUS NAMESPACE, and the
// sibling row paid for the lesson: an anonymous type cannot be `dynamic_cast`
// to from another translation unit, so a reachability case could assert
// `REQUIRE_NOTHROW` and `model != nullptr` and nothing more — and both hold for
// a hook that returns a default-constructed `Glm5NextWeights{}`. Deleting the
// `LoadGlm5NextFromGguf` call site would then leave that case GREEN, measuring
// that something was registered and never that anything was loaded. The fix is
// the visibility, not another assertion.
class Glm5NextLoadedModel final : public LoadedModel {
 public:
  Glm5NextLoadedModel(const ModelRegistration& registration,
                      Glm5NextWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Glm5NextWeights& weights() const { return weights_; }

 private:
  Glm5NextWeights weights_;
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_LOADER_H_
