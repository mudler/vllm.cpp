// DeepSeek-V2 (`DeepseekV2ForCausalLM`) — the MLA campaign's W7 model TU: the
// first model in this tree whose attention is MLA. It COMPOSES everything
// W1-W6 built (spec-driven MLA KV allocation, the MLA backend selection, the
// MLA cache write, the MLA decode/prefill kernels, the MLA attention block +
// load-time weight absorption) into a registry-visible model with a config
// parse, a weight loader and a forward.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin e24d1b24) ───────
//   OURS                             <-  UPSTREAM
//   DeepseekV2Params                 <-  vllm/model_executor/models/deepseek_v2.py
//                                        :1189-1244 (DecoderLayer config reads)
//                                        + :286-393 (DeepseekV2MoE config reads)
//                                        + the shipped V2-Lite config.json
//   DeepseekV2MlaWeights             <-  deepseek_v2.py:1003-1049
//                                        `DeepseekV2MLAAttention.__init__`
//   DeepseekV2DenseMlp               <-  deepseek_v2.py:229-274 `DeepseekV2MLP`
//                                        (MergedColumnParallelLinear gate_up +
//                                         RowParallelLinear down + SiluAndMul)
//   DeepseekV2MoeWeights             <-  deepseek_v2.py:276-393 `DeepseekV2MoE`
//                                        (gate, n_routed_experts FusedMoE,
//                                         n_shared_experts shared MLP)
//   DeepseekV2LayerWeights           <-  deepseek_v2.py:1172-1262
//                                        `DeepseekV2DecoderLayer` (the
//                                        `first_k_dense_replace` / `moe_layer_freq`
//                                        dense-vs-MoE selection at :1214-1218)
//   LoadDeepseekV2ForCausalLMWeights <-  deepseek_v2.py:1780-1906
//                                        `DeepseekV2ForCausalLM` +
//                                        vllm/model_executor/layers/attention/
//                                        mla_attention.py:875-962
//                                        `process_weights_after_loading`
//                                        (the LOAD-TIME kv_b_proj absorption)
//   DeepseekV2Model::Forward         <-  deepseek_v2.py:1347-1520 `DeepseekV2Model`
//                                        + :1262-1345 the decoder-layer forward
//   BuildMlaBatchSplit               <-  mla_attention.py:1640-1649
//                                        (`split_decodes_and_prefills`,
//                                         reorder_batch_threshold == 1 at :1420)
//                                        + :1806-1810
//                                        (`prefill_tokens_with_context`)
//
// ─── WHAT IS DELIBERATELY NOT REGISTERED ────────────────────────────────────
// Upstream routes FOUR architecture strings at `registry.py:90-93` into this one
// module. This TU registers EXACTLY ONE — `DeepseekV2ForCausalLM` — because the
// other three are not things we can genuinely serve today:
//   * `DeepseekForCausalLM` is **NOT an MLA model at all**. deepseek_v2.py
//     :1201-1211 computes `use_mha = config.model_type == "deepseek" or all(dim
//     == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))`, and `use_mha`
//     selects `DeepseekAttention` (`:133`) — plain MHA. Registering it here
//     would silently claim MLA support for a model that needs none of this code.
//   * `DeepseekV3ForCausalLM` resolves to the same Python class, but every
//     shipped V3 checkpoint is fp8 block-quantized and 671B — neither our bf16
//     loader nor GB10's 119 GiB can take it (spike §5). The MLA CODE PATHS it
//     would use (the `q_lora_rank` query branch, the sigmoid/`noaux_tc` grouped
//     router) are implemented and unit-gated at V3 dimensions, but a registry
//     entry is a support claim, not a code-coverage claim.
//   * `DeepseekV32ForCausalLM` additionally needs the DSA sparse indexer
//     (deepseek_v2.py:613-642), which does not exist in this tree.
// Those rows stay `SPIKE`/`HW-BLOCKED` in the model matrix; W10 owns the honest
// blocked-row pass.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/mla_attention.h"      // MlaBlockDims/Weights
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"            // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"    // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"                     // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"
#include "vt/ops.h"  // vt::MoeScoringFunc

namespace vllm {

class SafetensorsFile;

// Every DeepSeek-V2 config field the loader/forward consume, resolved ONCE from
// the HfConfig (DeepSeek uses `n_routed_experts`, not the `num_experts` key
// HfConfig types, so nothing here can be read off the typed struct).
//
// DeepSeek-V2-Lite (the W7/W8 vehicle, from the shipped config.json):
//   hidden 2048, 27 layers, 16 heads, qk_nope 128, qk_rope 64, v 128,
//   kv_lora 512, q_lora NULL (the DIRECT q_proj branch), vocab 102400,
//   intermediate 10944, moe_intermediate 1408, 64 routed + 2 shared experts,
//   top-6, first_k_dense_replace 1, n_group 1, topk_group 1,
//   scoring_func "softmax", topk_method "greedy", norm_topk_prob false,
//   routed_scaling_factor 1.0, YaRN factor 40 / mscale 0.707.
struct DeepseekV2Params {
  // --- shared model geometry ---
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  int64_t intermediate_size = 0;  // the DENSE (first_k_dense_replace) MLP width
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  int64_t max_position_embeddings = 0;

  // --- MLA (deepseek_v2.py:960-1002) + the YaRN rope + mscale^2 scale ---
  mla::MlaBlockDims mla{};
  mla::DeepseekYarnRopeParams rope{};

  // --- MoE (deepseek_v2.py:286-393) ---
  int64_t n_routed_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t n_shared_experts = 0;
  // `first_k_dense_replace` / `moe_layer_freq` (:1214-1218): layer L is a MoE
  // layer iff `L >= first_k_dense_replace && L % moe_layer_freq == 0`.
  int64_t first_k_dense_replace = 0;
  int64_t moe_layer_freq = 1;
  int64_t n_group = 1;
  int64_t topk_group = 1;
  bool norm_topk_prob = false;
  // "softmax" (V2) vs "sigmoid" (V3/R1) — grouped_topk_router.py:110-117.
  vt::MoeScoringFunc scoring_func = vt::MoeScoringFunc::kSoftmax;
  // `topk_method == "noaux_tc"` is the ONLY setting that creates the learned
  // `e_score_correction_bias` gate parameter (:313-318). V2-Lite is "greedy",
  // so no bias tensor exists in its checkpoint.
  bool has_e_score_correction_bias = false;
  float routed_scaling_factor = 1.0f;

  bool is_moe_layer(int64_t layer) const {
    return n_routed_experts > 0 && layer >= first_k_dense_replace &&
           (moe_layer_freq <= 1 || layer % moe_layer_freq == 0);
  }
  // The shared-expert MLP width (:346): `moe_intermediate_size * n_shared_experts`.
  int64_t shared_intermediate_size() const {
    return moe_intermediate_size * n_shared_experts;
  }
};

// Resolve DeepseekV2Params from a HfConfig. Throws with a precise message on any
// field this port cannot serve. Pure/host — unit-testable without a checkpoint.
DeepseekV2Params ParseDeepseekV2Params(const HfConfig& config);

// `DeepseekV2MLAAttention` weights (deepseek_v2.py:1003-1049), host-resident.
// EXACTLY ONE query branch is populated (`q_lora_rank is not None` at :1003);
// the other's tensors stay Empty(). All 2-D projections are RAW-NK
// ([out_features, in_features], `nk == true`) because every consumer in the MLA
// block is `vt::MatmulBT` with output-row slicing.
struct DeepseekV2MlaWeights {
  // (a) `q_lora_rank is not None` (:1003-1026). There is no standalone `q_a_proj`
  //     module upstream — :1812-1820 packs `fused_qkv_a_proj <- [q_a_proj,
  //     kv_a_proj_with_mqa]`, so the loader FUSES the two checkpoint tensors,
  //     row blocks in order [q_lora_rank | kv_lora_rank | qk_rope_head_dim].
  OwnedTensor fused_qkv_a_proj;  // [q_lora + kv_lora + qk_rope, H] raw-NK
  OwnedTensor q_a_layernorm;     // [q_lora]                        (:1019)
  OwnedTensor q_b_proj;          // [N*qk_head_dim, q_lora] raw-NK  (:1020-1026)
  // (b) `q_lora_rank is None` (:1010-1016, :1028-1034) — the V2-Lite branch.
  OwnedTensor kv_a_proj_with_mqa;  // [kv_lora + qk_rope, H] raw-NK (:511)
  OwnedTensor q_proj;              // [N*qk_head_dim, H] raw-NK
  // shared
  OwnedTensor kv_a_layernorm;  // [kv_lora] — the rope part is NOT normed (:516)
  OwnedTensor kv_b_proj;       // [N*(qk_nope+v), kv_lora] raw-NK  (:518-519)
  // The LOAD-TIME absorption split (mla_attention.py:892-900, :959-962), done
  // by mla::AbsorbKvBProjBf16 while the checkpoint bytes are in hand.
  OwnedTensor w_uk_t;  // bf16 [N, qk_nope, kv_lora]  (:962)
  OwnedTensor w_uv;    // bf16 [N, kv_lora, v]        (:960)
  OwnedTensor o_proj;  // [H, N*v] raw-NK             (:526)
};

// `DeepseekV2MLP` (deepseek_v2.py:229-274) — the dense layers' MLP AND the MoE
// layers' shared-expert MLP. gate/up are merged into ONE raw-NK owner in exactly
// upstream's MergedColumnParallelLinear order (gate rows then up rows), so the
// forward is `MatmulBT -> vt::SiluAndMul -> MatmulBT`, byte-for-byte the shape
// the Qwen3-dense MlpBlock already runs.
struct DeepseekV2DenseMlp {
  OwnedTensor gate_up_proj;  // [2*I, H] raw-NK
  OwnedTensor down_proj;     // [H, I]   raw-NK
  bool Empty() const { return gate_up_proj.Empty(); }
};

// `DeepseekV2MoE` (deepseek_v2.py:276-393). Routed experts are bf16 Matmul-B
// ([K,N], `nk == false`) — the layout the grouped bf16 MoE GEMM
// (`vt::MoeGroupedGemmBf16`) reads. UNLIKE Qwen3-Coder, DeepSeek-V2 HAS shared
// experts, and unlike Qwen3.6's they carry NO sigmoid gate: `shared` is a plain
// `DeepseekV2MLP` whose output is ADDED to the routed sum (:344-357 + the
// FusedMoE `shared_experts=` wiring at :360).
struct DeepseekV2MoeWeights {
  OwnedTensor router_gate;  // [H, E] Matmul-B (from the [E,H] `mlp.gate.weight`)
  // `e_score_correction_bias` [E] f32 — present ONLY for topk_method
  // "noaux_tc" (:313-318). Empty for V2-Lite.
  OwnedTensor e_score_correction_bias;
  std::vector<OwnedTensor> expert_gate;  // E x [H, I] Matmul-B
  std::vector<OwnedTensor> expert_up;    // E x [H, I] Matmul-B
  std::vector<OwnedTensor> expert_down;  // E x [I, H] Matmul-B
  DeepseekV2DenseMlp shared;             // n_shared_experts > 0
};

// One `DeepseekV2DecoderLayer` (deepseek_v2.py:1172-1262).
struct DeepseekV2LayerWeights {
  OwnedTensor input_layernorm;           // [H]
  OwnedTensor post_attention_layernorm;  // [H]
  DeepseekV2MlaWeights attn;
  bool is_moe = false;
  DeepseekV2DenseMlp dense;  // populated iff !is_moe
  DeepseekV2MoeWeights moe;  // populated iff  is_moe
};

// Whole DeepSeek-V2 text-model weights + the resolved params. DeepSeek-V2-Lite is
// UNTIED (`tie_word_embeddings: false`), so `lm_head` is a real tensor.
struct DeepseekV2Weights {
  DeepseekV2Params params{};
  OwnedTensor embed_tokens;  // bf16 [vocab, H] (embed lookup; NOT transposed)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B; EMPTY only when tied
  // The YaRN [cos|sin] rope cache, built ONCE at load time from the resolved
  // rope params (mla::BuildDeepseekRopeCosSinCache) and shared by every layer.
  OwnedTensor rope_cos_sin_cache;  // bf16 [rows, qk_rope_head_dim]
  std::vector<DeepseekV2LayerWeights> layers;
};

// ─── the batch split MLA requires (mla_attention.py:1640-1649, :1806-1810) ───
//
// MLA's metadata builder assumes the scheduler already ran
// `reorder_batch_to_split_decodes_and_prefills` with `decode_threshold == 1`
// (`reorder_batch_threshold: int = 1`, mla_attention.py:1420), so:
//   1. every DECODE request (query_len <= 1) precedes every PREFILL request, and
//      `num_decode_tokens` names a batch PREFIX (the MLA block slices
//      `q[:num_mqa_tokens]` / `q[num_mqa_tokens:]`, mla_attention.py:700-737);
//   2. inside the prefill tail, every request WITH context precedes every request
//      without, because `prefill_tokens_with_context =
//      prefill_query_start_loc_cpu[num_prefills_with_context]` (:1806-1810) is a
//      PREFIX LENGTH — it is only a correct answer if the with-context requests
//      lead.
// W6's gate found what happens when (2) is violated: a batch whose with-context
// prefill was NOT first produced 0.86 relative error. So this is checked, loudly,
// rather than assumed: BuildMlaBatchSplit throws naming the offending request.
struct MlaBatchSplit {
  int num_decodes = 0;
  int num_decode_tokens = 0;   // upstream `num_decode_tokens` / `num_mqa_tokens`
  int num_prefills = 0;
  int num_prefill_tokens = 0;
  int num_prefills_with_context = 0;
  // `prefill_query_start_loc - prefill_query_start_loc[0]` (:1670-1675):
  // [num_prefills + 1], RELATIVE to the prefill sub-batch.
  std::vector<int32_t> prefill_cu_seqlens_q;
  // [num_prefills] `seq_lens - query_lens` (:1663-1665).
  std::vector<int32_t> prefill_context_lens;
  int32_t prefill_max_query_len = 0;
  int32_t decode_max_seq_len = 0;
};

// Compute (and VALIDATE) the split from a CommonAttentionMetadata. Pure host —
// no device, no weights — so the batch-ordering invariant is directly gateable.
MlaBatchSplit BuildMlaBatchSplit(const v1::CommonAttentionMetadata& meta);

// ─── MLA campaign W8: PROOF the split ran, and WHAT SHAPES the engine produced ─
//
// A passing correctness gate does not prove the interesting path executed — the
// W1 lesson (`GPUModelRunner::fa_page_size_bytes()` was added for exactly this
// reason: "proof the new path is EXERCISED, not merely compiled"). W8 wires a
// real PAGED ENGINE over this model, so the question it must answer is not only
// "are the tokens right" but "did the scheduler/runner actually hand the MLA
// block MIXED batches, in the legal order, with a with-context prefill in them".
//
// These counters are accumulated inside the per-step split build (one call per
// forward). They are DIAGNOSTIC ONLY — nothing in the forward reads them — and
// they are written from the single forward thread the runner drives, so they
// carry no synchronization. The paged-engine gate resets them, runs the battery,
// and ASSERTS on the shapes observed (see
// tests/vllm/models/test_deepseek_v2_paged_engine.cpp).
struct MlaBatchSplitStats {
  int64_t steps = 0;                   // BuildMlaStep invocations (= forwards)
  int64_t decode_only_steps = 0;       // num_prefills == 0 && num_decodes > 0
  int64_t prefill_only_steps = 0;      // num_decodes == 0 && num_prefills > 0
  int64_t mixed_steps = 0;             // BOTH halves non-empty in one step
  int64_t with_context_prefill_steps = 0;  // num_prefills_with_context > 0
  int64_t max_num_decodes = 0;
  int64_t max_num_prefills = 0;
  int64_t max_num_reqs = 0;
  int64_t total_decode_tokens = 0;
  int64_t total_prefill_tokens = 0;
};
const MlaBatchSplitStats& GetMlaBatchSplitStats();
void ResetMlaBatchSplitStats();

// The DeepSeek-V2 forward: embed -> N decoder layers (std add+RMSNorm ->
// mla::ForwardMlaAttentionBlock -> std add+RMSNorm -> dense MLP or MoE block) ->
// final RMSNorm -> untied lm_head. `attn_kv` carries ONE MLA cache per layer,
// viewed [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim] (num_kv_heads
// == 1, NO separate V — the MLAAttentionSpec allocation W1 landed).
class DeepseekV2Model {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const DeepseekV2Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident logits variant (the sampler-on-device hot path).
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const DeepseekV2Weights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});
};

// ─── MLA campaign W9: the decode CUDA-graph sibling ─────────────────────────
//
// The MLA member of the decode-graph driver family (Qwen3_5DecodeGraph,
// Qwen3_5DenseDecodeGraph, Qwen3MoeDecodeGraph). Owned by the LoadedModel so it
// outlives a single forward, it routes PURE-DECODE CUDA steps through a captured
// graph per padded batch size, mirroring vLLM's full-decode-cudagraph regime
// (gpu_model_runner.py capture/replay + compilation/cuda_graph.py's
// pad-to-nearest dispatch @ e24d1b24). Prefill, mixed steps, batches above
// max_num_seqs and CPU fall back to the eager forward INTERNALLY, so the caller
// has exactly one dispatch point. Real-row output is bit-identical to eager.
//
// Rollback for a same-binary A/B: `VT_DEEPSEEK_CUDAGRAPH=0` (model-local) or
// `VLLM_CPP_CUDAGRAPH=0` (framework-wide).
class DeepseekV2DecodeGraph {
 public:
  DeepseekV2DecodeGraph(const DeepseekV2Weights& weights, vt::Queue queue,
                        int64_t max_num_reqs);
  ~DeepseekV2DecodeGraph();
  DeepseekV2DecodeGraph(const DeepseekV2DecodeGraph&) = delete;
  DeepseekV2DecodeGraph& operator=(const DeepseekV2DecodeGraph&) = delete;

  // One pure-decode step. Falls back to the eager forward internally.
  ForwardLogits Step(const std::vector<int32_t>& token_ids,
                     const std::vector<int32_t>& positions,
                     const v1::CommonAttentionMetadata& attn_meta,
                     const std::vector<PagedKvCache>& attn_kv);

  bool captured() const;         // diagnostics: at least one live graph
  int64_t replay_count() const;  // diagnostics: total replays

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Load `DeepseekV2ForCausalLM` safetensors into DeepseekV2Weights, INCLUDING the
// `kv_b_proj -> W_UK / W_UV` absorption split (mla_attention.py:875-962 is a
// `process_weights_after_loading` hook upstream; doing it at load time is the
// same transform at the same point in the lifecycle) and the YaRN rope cache.
DeepseekV2Weights LoadDeepseekV2ForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// Per-family config hook (the registry `parse_config`): resolves + validates
// DeepseekV2Params and throws on anything unsupported.
void ParseDeepseekV2Config(const HfConfig& config);

// KV-cache spec builder: exactly ONE **MLA** attention group, no Mamba group.
// `MLAAttentionSpec(block_size, head_size = kv_lora_rank + qk_rope_head_dim,
// dtype, num_kv_heads = 1)` — the factor-2 K+V page cost does not apply
// (kv_cache_interface.h `MLAAttentionSpec::real_page_size_bytes`).
v1::KVCacheConfig MakeDeepseekV2KVCache(const HfConfig& config, int block_size,
                                        int num_blocks);

}  // namespace vllm
