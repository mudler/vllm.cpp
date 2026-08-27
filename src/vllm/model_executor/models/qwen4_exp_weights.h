// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) W5a — the loaded weight set and the
// `qwen4exp` GGUF loader that fills it.
//
// Issue [#2031](https://github.com/mudler/vllm.cpp/issues/2031), campaign issue
// [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// Model-private, deliberately not under `include/`: nothing outside this model
// needs these types, and `include/vllm.h` is the ABI seam a SHIPPED capability
// is exposed through. This wave ships a load, not a capability — the forward and
// the KV-cache spec still refuse by name.
//
// ─── WHY GGUF AND ONLY GGUF ──────────────────────────────────────────────────
// Every safetensors artifact of this model is larger than any device this
// project owns: bf16 ~360 GB, official FP8 ~180 GB, NVFP4 ~128 GB, against
// ~119.6 GiB usable on GB10. `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` is
// 67.56 GiB in three shards and is the ONE published artifact that fits. The
// safetensors arm is therefore not "later because it is easy"; it is deferred
// because no host we own could ever read it, and the spec records it as owed.
//
// ─── THE TWO ORACLES THIS FILE ANSWERS TO ────────────────────────────────────
// vLLM registers no `qwen4_exp` at any revision, so under AGENTS.md "When vLLM
// has no implementation" this row runs a SPLIT oracle:
//
//   * the ALGORITHM — what each tensor IS, and what shape it has — is
//     transformers **5.16.0**, this row's accepted lane pin. Every module and
//     `nn.Linear` cited below is at
//     `transformers/models/qwen4_exp/modeling_qwen4_exp.py` at that tag.
//   * the CONTAINER — what the converter DID to those tensors on the way into
//     the file, and therefore what this loader must undo — is llama.cpp pull
//     request [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) at head
//     `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, whose `conversion/qwen4exp.py`
//     declares `class Qwen4ExpTextModel(_Qwen35MRopeMixin,
//     _LinearAttentionVReorderBase)`. Both files were READ AT SOURCE for this
//     wave rather than relayed, because the narrow version of any of these
//     sentences causes the exact defect it warns about.
//
// The container oracle is an OPEN pull request, and that is a live risk rather
// than a footnote: if #27742 renames a key or a tensor before it merges, this
// file changes and the committed manifest is regrown. There is no compatibility
// shim for a spelling that never shipped.
//
// ─── THE FOUR CONVERT-TIME TRANSFORMS THIS FILE INVERTS ──────────────────────
//
// 1. `+1` ON EVERY NORM BUT ONE. The inherited Qwen3-Next rule is
//    `elif name.endswith("norm.weight") and not
//     name.endswith("linear_attn.norm.weight"): data_torch = data_torch + 1`
//    (`conversion/qwen.py`, `Qwen3NextModel.modify_tensors`), and #27742 adds an
//    EARLY-RETURNING branch that folds the five gammas that rule would otherwise
//    reach twice or not at all:
//    `if name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight",
//     ".ple.norm_conv.weight", ".indexer.q_layernorm.weight",
//     ".indexer.k_layernorm.weight")): return [(…, data_torch + 1)]`.
//    So EVERY norm in the file carries the fold and `ssm_norm` is the ONLY
//    exception. A loader that skips the fold for one tensor double-folds every
//    other one — the same silent ~2x defect, moved one tensor to the left.
//    Corroborated independently on published artifacts during the fresh review
//    of #1988: every `*hc_norm.weight` is HF + 1.0 exactly, elementwise, while
//    `ssm_norm` is unfolded and sits in [0.875, 1.023].
//
// 2. `ssm_a = -exp(A_log)`, so the loader recovers `A_log = log(-x)`
//    (`conversion/qwen.py`: `if name.endswith(".A_log"): data_torch =
//    -torch.exp(data_torch)`). Identical to what `qwen3_5_gguf_weights.cpp`
//    already does for `qwen3next`.
//
// 3. THE V-HEAD REORDER, and it is active on this model.
//    `_LinearAttentionVReorderBase` fires whenever
//    `num_k_heads != num_v_heads`, which for the released config is 16 vs 48.
//    It rewrites every Gated DeltaNet tensor from HF's GROUPED V order
//    (`[G0_v0..v{R-1}, G1_v0..v{R-1}, …]`) into ggml's TILED order
//    (`[G0_v0, G1_v0, …, G0_v1, G1_v1, …]`) so `ggml_repeat` can broadcast. Our
//    kernels consume HF order, so every one of them is un-reordered here. The
//    permutation is the IDENTITY when `num_k_heads == 1`, which is why the gate's
//    fixture uses two.
//
// 4. `conv1d` IS SQUEEZED. `nn.Conv1d` stores `[C, 1, K]`; the converter emits
//    `[C, K]` for both `ssm_conv1d` and `ple_conv1d`. Nothing to invert — stated
//    so the missing middle axis is not read as a defect.
//
// ─── WHAT IS NOT HERE ────────────────────────────────────────────────────────
// The forward, the KV-cache spec, the vision tower, MTP, and the safetensors
// arm. All four refuse by name and all four are listed under `## Owed` in the
// spec with the wave that owns them. This header is the load and nothing else.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_WEIGHTS_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_WEIGHTS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/model_registry.h"  // LoadedModel
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// One `Qwen4ExpTextGatedResidual` (modeling_qwen4_exp.py:941-969).
//
// `use_combine=false` on the model-level mixer, which is the ONLY instance
// without `block_inject_weight`; `has_inject` is that switch, and it is a stored
// bool rather than an emptiness test because an empty tensor is also what a
// half-finished load leaves behind.
struct Qwen4ExpGatedResidualWeights {
  OwnedTensor hc_norm;  // [stream]  grouped RMSNorm gamma, fold INVERTED
  OwnedTensor down;     // [lowrank, stream]   input_mix_weight_down
  OwnedTensor up;       // [stream, lowrank]   input_mix_weight_up
  OwnedTensor inject;   // [hc_count, stream]  block_inject_weight, or empty
  bool has_inject = false;
};

// `Qwen4ExpTextGatedDeltaNet` (modeling_qwen4_exp.py:403-446). Every tensor here
// carries transform 3 except `in_proj_qkv`'s leading Q and K rows and the norm.
struct Qwen4ExpGdnWeights {
  OwnedTensor in_proj_qkv;  // [2*key_dim + value_dim, H]  V rows un-reordered
  OwnedTensor in_proj_z;    // [value_dim, H]              rows un-reordered
  OwnedTensor in_proj_b;    // [num_v, H]                  rows un-reordered
  OwnedTensor in_proj_a;    // [num_v, H]                  rows un-reordered
  OwnedTensor conv1d;       // [conv_dim, K]        V channels un-reordered
  OwnedTensor a_log;        // f32 [num_v]  = log(-ssm_a), un-reordered
  OwnedTensor dt_bias;      // f32 [num_v]                 un-reordered
  // `RMSNormGated` over `head_v_dim`. THE ONE NORM WITH NO `+1` TO INVERT.
  OwnedTensor norm_weight;  // [value_head_dim]
  OwnedTensor out_proj;     // [H, value_dim]      V columns un-reordered
};

// `Qwen4ExpTextAttention` + its `Qwen4ExpTextQSAIndexer`
// (modeling_qwen4_exp.py:757-782 and :611-629).
struct Qwen4ExpQsaWeights {
  // `num_attention_heads * head_dim * 2` — query and OUTPUT GATE in one tensor,
  // split by `torch.chunk(..., 2, dim=-1)` per head at :810. Not two heads'
  // worth of queries.
  OwnedTensor q_proj;      // [q_heads * head_dim * 2, H]
  OwnedTensor k_proj;      // [kv_heads * head_dim, H]
  OwnedTensor v_proj;      // [kv_heads * head_dim, H]
  OwnedTensor o_proj;      // [H, q_heads * head_dim]
  OwnedTensor q_norm;      // [head_dim]      fold INVERTED
  OwnedTensor k_norm;      // [head_dim]      fold INVERTED
  // The indexer's single `index_qk_proj` is SPLIT by the converter into two
  // tensors ("one projection feeds indexer q and k; split it, as minimax-m3
  // does"). The split point is `indexer_n_heads * indexer_head_dim`.
  OwnedTensor idx_q_proj;  // [idx_heads * idx_head_dim, H]
  OwnedTensor idx_k_proj;  // [idx_kv_heads * idx_head_dim, H]
  OwnedTensor idx_q_norm;  // [idx_head_dim]  fold INVERTED
  OwnedTensor idx_k_norm;  // [idx_head_dim]  fold INVERTED
};

// `Qwen4ExpTextSparseMoeBlock` (modeling_qwen4_exp.py:919-938): 512 routed
// experts at top-10 plus ONE shared expert with its own scalar gate.
struct Qwen4ExpMoeWeights {
  OwnedTensor router;        // [num_experts, H]   TopKRouter
  // `Linear(H, 1)` squeezed to a vector by the converter: the shared expert's
  // sigmoid gate. One-dimensional in the file and one-dimensional here.
  OwnedTensor shared_gate;   // [H]
  OwnedTensor gate_exps;     // [E, moe_I, H]
  OwnedTensor up_exps;       // [E, moe_I, H]
  OwnedTensor down_exps;     // [E, H, moe_I]
  OwnedTensor shared_gate_proj;  // [shared_I, H]
  OwnedTensor shared_up_proj;    // [shared_I, H]
  OwnedTensor shared_down_proj;  // [H, shared_I]
};

// `Qwen4ExpTextPLELayer` (modeling_qwen4_exp.py:1117-1147). Present on exactly
// the layers `ple_layer_ids` names; on the released checkpoint that is one
// layer, 0-based 1.
struct Qwen4ExpPleWeights {
  OwnedTensor key_proj;    // [stream, ple_embed_dim]
  OwnedTensor value_proj;  // [H, ple_embed_dim]
  OwnedTensor norm_key;    // [stream]   fold INVERTED
  OwnedTensor norm_query;  // [stream]   fold INVERTED
  OwnedTensor norm_conv;   // [stream]   fold INVERTED
  OwnedTensor conv1d;      // [stream, ple_conv_kernel_size]
};

struct Qwen4ExpLayerWeights {
  bool is_linear_attention = false;
  Qwen4ExpGdnWeights gdn;  // iff is_linear_attention
  Qwen4ExpQsaWeights qsa;  // iff !is_linear_attention
  Qwen4ExpMoeWeights moe;  // every layer
  Qwen4ExpGatedResidualWeights attn_hc;
  Qwen4ExpGatedResidualWeights mlp_hc;
  bool has_ple = false;
  Qwen4ExpPleWeights ple;
};

struct Qwen4ExpWeights {
  Qwen4ExpParams params;

  // [vocab, H]. A GATHER, so it expands to bf16 like every other token table in
  // this tree.
  OwnedTensor embed_tokens;
  // [padded_ngram_vocab, ple_embed_dim / ngram_heads]. THE 51.2 G-PARAMETER
  // TABLE, and the one gather in this repository that KEEPS ITS BLOCKS: W6a
  // (#1989) made `GgufTensorRole::kEmbeddingTable` keep-quant eligible and
  // taught `vt::Embedding` to decode one row per gathered token, which is what
  // turns 102.4 GB of expanded bf16 into 28.8 GB of resident Q4_K. Empty when
  // the config names no PLE layer.
  OwnedTensor ngram_table;
  // [vocab, H] in the file's own order, or EMPTY when the file ties the head to
  // the embedding table. Tie is read off the FILE (is `output.weight` there?)
  // rather than off a config key, because llama.cpp's writer decides it that way
  // and a config that disagreed with the file would be the config's error.
  OwnedTensor lm_head;
  bool tied_word_embeddings = false;

  std::vector<Qwen4ExpLayerWeights> layers;
  // `hyper_connection_mixer`, the `use_combine=false` instance that collapses
  // the 10240-wide stream to 2560 at the very end.
  //
  // ITS `hc_norm` IS THE LAST NORMALIZATION IN THE MODEL. `Qwen4ExpTextModel`
  // has NO final RMSNorm (there is no `output_norm.weight` in the file, and the
  // manifest confirms the absence). A port that copies our DeepSeek-V4 tail will
  // insert one that does not exist, and that tail is the natural thing to copy.
  Qwen4ExpGatedResidualWeights mixer;

  // STRUCTURAL accounting, the same shape every other loader in this tree
  // reports: how many tensors the name map ENUMERATES for this config, and how
  // many of them the file actually carries. The load below reads through the
  // SAME names, so the two can never disagree.
  int64_t enumerated_tensors = 0;
  int64_t accounted_tensors = 0;
};

// Every GGUF tensor name this architecture expects at `params`, in a stable
// order. The load reads through exactly these names.
std::vector<std::string> EnumerateQwen4ExpGgufTensors(
    const Qwen4ExpParams& params);

// Load the text tower. Refuses BY NAME on a missing tensor, a shape
// disagreement, or an encoding this build cannot decode. `policy` is borrowed
// and may be null, in which case `GgufLoadPolicy::FromEnv()` decides residency —
// which is what a production load gets.
//
// `device` IS THE DEVICE THE FORWARD WILL RUN ON, and it has NO DEFAULT on
// purpose. The load refuses ahead of any tensor I/O when that device cannot
// gather from a block table, because on this architecture that refusal is the
// difference between a named error and 95.4 GiB of anonymous host memory
// (#2083; the reason is in the function's own body). A default would let a new
// caller disable the guard by saying nothing, which is the failure mode the
// guard exists for.
Qwen4ExpWeights LoadQwen4ExpFromGguf(const GgufFile& gguf, const HfConfig& config,
                                     vt::DeviceType device,
                                     const GgufLoadPolicy* policy = nullptr);

// The concrete model the registry's `load_weights` hook produces. It exists so
// the type-erased `LoadedModel` the registry hands around has something real
// behind it, and so `ModelAs<>` has a type to open — never a `static_cast`,
// which is undefined behaviour on an object that is not really this type
// (#775, #730).
//
// DECLARED HERE RATHER THAN IN THE REGISTRY TU'S ANONYMOUS NAMESPACE, and the
// reason is a mutation this gate failed. Every sibling model keeps its
// `LoadedModel` file-local, and while `load_weights` is the only production
// entry a row has, that choice makes the load's RESULT unobservable from
// outside: the reachability case could assert `REQUIRE_NOTHROW` and
// `model != nullptr` and nothing more, and both hold for a hook that returns a
// default-constructed `Qwen4ExpWeights{}`. Deleting the `LoadQwen4ExpFromGguf`
// call site (mutation M1) therefore left that case GREEN, so it measured that
// something was registered and never that anything was loaded. An anonymous
// type cannot be `dynamic_cast` to from another translation unit, so the fix is
// the visibility and not another assertion.
//
// W5b opens the same handle from `ForwardQwen4ExpForConditionalGeneration`
// once there is a forward to open it for.
class Qwen4ExpLoadedModel final : public LoadedModel {
 public:
  Qwen4ExpLoadedModel(const ModelRegistration& registration,
                      Qwen4ExpWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const Qwen4ExpWeights& weights() const { return weights_; }

 private:
  Qwen4ExpWeights weights_;
};

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_WEIGHTS_H_
