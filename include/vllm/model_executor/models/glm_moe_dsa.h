// GLM-5.3 (`GlmMoeDsaForCausalLM`, HF `model_type: glm_moe_dsa`) — the CONFIG
// surface, the `glm-dsa` GGUF arm and the refuse-by-name forward. W2 of
// `.agents/specs/glm-dsa-latest-deepseek.md` §3.7, issue
// [#2214](https://github.com/mudler/vllm.cpp/issues/2214).
//
// ─── WHY THIS MODEL HAS ITS OWN PARAMS STRUCT ────────────────────────────────
// Upstream at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98` is a
// literal zero-override subclass: `deepseek_v2.py:1930` is
// `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM): pass`, and the only
// behavioural special case is the fp32 router dtype forced on
// `model_type == "glm_moe_dsa"` at `deepseek_v2.py:127`. Sharing
// `DeepseekV2Params` would therefore be the obvious move, and it is the wrong
// one: `ParseDeepseekV2Params` REFUSES any checkpoint carrying `index_topk`
// (`deepseek_v2_weights.cpp:358-364`) and any `quantization_config` (`:365-369`),
// and GLM-5.3 carries both. Relaxing either refusal to let this model through
// turns a wall into a choice for DeepSeek-V2 as well — that tripwire is what
// stops a V3.2 checkpoint silently loading onto a dense-attention forward.
// So GLM-5.3 gets its own struct and DeepSeek-V2's refusals are not touched.
// Recorded in the spec §3.7 W2 Exclusions.
//
// ─── WHAT THIS TRANSLATION UNIT DOES NOT DO ──────────────────────────────────
// No forward math and no weight materialization. `GlmMoeDsaModel::Forward`
// refuses by name and lists every primitive that is missing, the wave that owes
// it and the issue that tracks it, so a reader meets the gap at the call rather
// than discovering it. The GGUF loader is W7's; the safetensors arms are refused
// permanently (spec D1: the published bf16/fp8 checkpoint is 703.74 GiB with no
// streaming loader and no MoE block-fp8 rung, so the quantized arm is the only
// one that can be fed on this fleet).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_keep_quant.h"  // GgufLoadPolicy
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/mla_attention.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

// llama.cpp's `general.architecture` for this family, `b10451`
// `src/llama-arch.cpp:85` (`{ LLM_ARCH_GLM_DSA, "glm-dsa" }`). It is the FAMILY
// key, not the HF class name, which is why the config builder below sets
// `architectures` explicitly instead of deriving it from this string.
inline constexpr const char* kGlmMoeDsaGgufArch = "glm-dsa";

// Whether a decoder layer runs its own lightning indexer, or reuses the
// selection of the preceding `kFull` layer. Upstream never stores this as a
// list: it evaluates the rule per layer inside `DeepseekV2MLAAttention.__init__`
// (`deepseek_v2.py:1092-1103`) and builds an indexer at `:1115` when
// `not _skip_topk or is_mtp_layer`. `kShared` is upstream's `_skip_topk == True`.
enum class GlmMoeDsaIndexerKind { kFull, kShared };

// Whether a decoder layer's MLP is the dense one or the 256-expert MoE block.
// Upstream derives this from `first_k_dense_replace` / `moe_layer_freq`
// (`deepseek_v2.py:1214-1218`) and never reads the checkpoint's own
// `mlp_layer_types` — `grep -c mlp_layer_types` over `deepseek_v2.py` at the pin
// is 0. See `ParseGlmMoeDsaParams` for why we read it anyway.
enum class GlmMoeDsaMlpKind { kDense, kSparse };

// The resolved GLM-5.3 configuration. Field-for-field an HF read; no derived
// geometry that a forward would own.
struct GlmMoeDsaParams {
  // --- backbone ---
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;  // BACKBONE depth, excluding the MTP block
  int64_t vocab_size = 0;
  int64_t num_attention_heads = 0;
  int64_t intermediate_size = 0;  // the dense MLP width of the leading layers
  double rms_norm_eps = 0.0;
  int64_t max_position_embeddings = 0;
  double rope_theta = 10000.0;

  // --- MLA geometry (deepseek_v2.py:1040-1074) ---
  int64_t q_lora_rank = 0;
  int64_t kv_lora_rank = 0;
  int64_t qk_nope_head_dim = 0;
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;
  // INTERLEAVED (GPT-J) rope, ALWAYS. Upstream passes `is_neox_style=False`
  // unconditionally at `deepseek_v2.py:1073` and reads no top-level
  // `rope_interleave` for it, so no checkpoint key can move this. It is a field
  // rather than an implicit default because `mla::MlaBlockDims::is_neox_style`
  // ALSO defaults false (`mla_attention.h:136`) — the value is right today by
  // coincidence of two defaults agreeing, and a default that is never read is a
  // value nothing can gate. `ParseGlmMoeDsaParams` writes it; a test asserts it.
  bool is_neox_style = false;
  // The INDEXER's rope, which is a separate decision and IS config-driven:
  // `is_neox_style=not getattr(config, "indexer_rope_interleave", False)`
  // (`deepseek_v2.py:1120`). GLM-5.3 ships `indexer_rope_interleave: true`, so
  // this is false on the published checkpoint and true on a checkpoint that
  // omits the key. The two rope styles are NOT the same field.
  bool indexer_rope_is_neox_style = true;

  // --- MoE (deepseek_v2.py:286-393) ---
  int64_t n_routed_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t n_shared_experts = 0;
  int64_t first_k_dense_replace = 0;
  int64_t moe_layer_freq = 1;
  int64_t n_group = 1;
  int64_t topk_group = 1;
  bool norm_topk_prob = false;
  bool has_e_score_correction_bias = false;  // `topk_method == "noaux_tc"`
  double routed_scaling_factor = 1.0;
  // The router GEMM runs in f32. THIS IS THE ANNOTATED EXCEPTION the dtype
  // polarity requires: `_get_moe_router_dtype` (`deepseek_v2.py:123-133`) forces
  // `torch.float32` on `model_type == "glm_moe_dsa"` at `:127`, BEFORE the
  // generic `moe_router_dtype == "float32"` branch at `:131`. GLM-5.3 also
  // declares `moe_router_dtype: float32`, so the special case is redundant on
  // this checkpoint and still fires first. Everything else on the model path
  // inherits the checkpoint's bf16.
  bool router_dtype_is_f32 = true;

  // --- the DSA lightning indexer (deepseek_v2.py:803-842) ---
  int64_t index_topk = 0;
  int64_t index_n_heads = 0;
  int64_t index_head_dim = 0;
  int64_t index_topk_freq = 1;
  int64_t index_skip_topk_offset = 2;

  // --- MTP, skipped rather than implemented (spec O5) ---
  int64_t num_nextn_predict_layers = 0;
  bool index_share_for_mtp_iteration = false;

  // --- the per-layer schedules, both `num_hidden_layers` long ---
  std::vector<GlmMoeDsaIndexerKind> indexer_types;
  std::vector<GlmMoeDsaMlpKind> mlp_layer_types;

  // Upstream's own rule, `deepseek_v2.py:1214-1218`, mirrored exactly as
  // `DeepseekV2Params::is_moe_layer` (`deepseek_v2.h:125-128`) mirrors it.
  bool is_moe_layer(int64_t layer) const {
    return n_routed_experts > 0 && layer >= first_k_dense_replace &&
           (moe_layer_freq <= 1 || layer % moe_layer_freq == 0);
  }
  // The one latent row the paged cache stores per token.
  int64_t mla_kv_head_size() const { return kv_lora_rank + qk_rope_head_dim; }
};

// Upstream's DERIVED indexer schedule, `deepseek_v2.py:1097-1101`, evaluated for
// every backbone layer instead of once per constructed layer:
//
//     _skip_topk = max(layer_id - offset + 1, 0) % freq != 0
//
// This is the ONLY source of the schedule this port synthesizes. llama.cpp
// survives a file that states nothing by falling back to a hardcoded 78-entry
// table (`b10451:src/models/glm-dsa.cpp:6-27`, `GLM_5_2_DEFAULT_INDEXER_TYPES`)
// that is bit-identical to GLM-5.3's own list; we deliberately do not copy it,
// because a constant that happens to be right is the shape that silently becomes
// wrong on the next checkpoint (spec D3).
std::vector<GlmMoeDsaIndexerKind> DeriveGlmMoeDsaIndexerSchedule(
    int64_t num_hidden_layers, int64_t index_topk_freq,
    int64_t index_skip_topk_offset);

// Upstream's dense/MoE layout for the same layers, `deepseek_v2.py:1214-1218`.
std::vector<GlmMoeDsaMlpKind> DeriveGlmMoeDsaMlpSchedule(
    int64_t num_hidden_layers, int64_t first_k_dense_replace,
    int64_t moe_layer_freq, int64_t n_routed_experts);

// Resolve `GlmMoeDsaParams` from an `HfConfig`. Throws `std::runtime_error` with
// a precise message on every field this port cannot serve. Pure/host — testable
// without a checkpoint.
GlmMoeDsaParams ParseGlmMoeDsaParams(const HfConfig& config);

// ─── W4: the heterogeneous per-layer MLA schedule ────────────────────────────
// The `mla::MlaBlockDims` for ONE backbone layer, with the indexer geometry
// present on a `kFull` layer and `skip_topk` set on a `kShared` one. This is the
// only place the two-way split becomes block geometry, and it reads
// `p.indexer_types` — the schedule `ParseGlmMoeDsaParams` resolved from the
// checkpoint — rather than re-deriving the rule. On GLM-5.3 it puts an indexer
// on 21 of the 78 backbone layers; the 22nd is the MTP block, which upstream
// forces full at `deepseek_v2.py:1110-1115` and which this row skips (spec O5).
//
// Throws `std::out_of_range` when `layer` is outside `[0, num_hidden_layers)`,
// and whatever `mla::MlaBlockDims::Validate` throws when the resolved geometry
// is one the MLA block refuses.
mla::MlaBlockDims GlmMoeDsaMlaBlockDims(const GlmMoeDsaParams& p, int64_t layer);

// Every backbone layer's dims, in layer order, each already `Validate()`d. The
// caller allocates ONE `mla::MlaSharedSelection` for the model and hands it to
// every layer in this order, which is what makes a `kShared` layer read the
// selection its owning `kFull` layer wrote (`mla.py:180`).
std::vector<mla::MlaBlockDims> GlmMoeDsaMlaSchedule(const GlmMoeDsaParams& p);

// How many of `p.indexer_types` are `kFull`. Named because the split is the
// wave's headline number and a reader should not have to count a vector.
int64_t GlmMoeDsaFullIndexerLayerCount(const GlmMoeDsaParams& p);

// The registry's config hook. The resolve IS the validation.
void ParseGlmMoeDsaConfig(const HfConfig& config);

// True when `gguf` carries `general.architecture == kGlmMoeDsaGgufArch`.
bool IsGlmMoeDsaGguf(const GgufFile& gguf);

// Synthesize an HF-shaped config from a `glm-dsa` GGUF header and hand it to the
// SAME `ParseGlmMoeDsaParams` a `config.json` descends through, so both sources
// meet one validator (the `Glm5NextHfConfigFromGguf` arrangement).
HfConfig GlmMoeDsaHfConfigFromGguf(const GgufFile& gguf);

// One MLA group, `kv_lora_rank + qk_rope_head_dim` wide. The indexer's own
// 132 B/token side cache is a SECOND group upstream
// (`DeepseekV32IndexerCache`, `deepseek_v2.py:696-701`) and does not exist here
// yet — spec O4, owed by W5 / `KV-DSV4-MULTICACHE`
// ([#1925](https://github.com/mudler/vllm.cpp/issues/1925)).
v1::KVCacheConfig MakeGlmMoeDsaKVCache(const HfConfig& config, int block_size,
                                       int num_blocks);

// ─── W7: the weights, and the two classes the artifact splits into ───────────
//
// The split is the GGUF tensor NAME, and it is the streamer's own admission
// rule: `_exps.weight` (`model_loader.cpp:2486`, `kStreamedExpertSuffix`). On
// `unsloth/GLM-5.3-GGUF UD-IQ1_S` that draws the line exactly where §3.3
// predicts — 228 stacked `[256, out, in]` towers at 187.312 GiB against 1581
// per-layer tensors at 14.511 GiB, both reproduced from the six shard headers
// by `tests/vllm/models/test_glm_moe_dsa_gguf_census.cpp`.

// The DSA lightning indexer of ONE `kFull` layer. Absent on a `kShared` layer,
// which runs no indexer at all and attends through the preceding full layer's
// selection (`vllm/model_executor/layers/mla.py:180`).
//
// THE PUBLISHED FILE SHIPS THESE ON ALL 79 BLOCKS AND ONLY 22 ARE REAL. The
// conversion broadcast the shared layers' weights (spec D3), so the loader
// reads the schedule and drops the surplus rather than believing the file —
// which is upstream's own posture at `deepseek_v2.py:1566-1582`.
struct GlmMoeDsaIndexerWeights {
  OwnedTensor wq_b;           // [index_n_heads * index_head_dim, q_lora_rank]
  OwnedTensor wk;             // [index_head_dim, hidden_size]
  OwnedTensor k_norm_weight;  // [index_head_dim]
  // The BIAS is what makes this a LayerNorm rather than an RMSNorm
  // (`deepseek_v2.py:803-842`; ours `mla_attention.cpp:663-664`, eps 1e-6), so
  // it is required rather than optional: a file without it describes a
  // different operator.
  OwnedTensor k_norm_bias;  // [index_head_dim]
  // `nn.Linear(hidden_size, n_heads)` — one row per INDEXER head (32), not per
  // MLA head (64). The published file stores it F32 and it stays F32.
  OwnedTensor weights_proj;  // [index_n_heads, hidden_size]
  bool Empty() const { return wq_b.Empty(); }
};

// One layer's MLA operands, in the file's own orientation.
//
// `attn_k_b` / `attn_v_b` ARRIVE ALREADY ABSORBED. llama.cpp's DeepSeek
// converter splits `kv_b_proj` into the two halves and transposes `k_b`, so
// there is no `AbsorbKvBProjBf16` at load on this arm — unlike the safetensors
// DeepSeek-V2 path (`deepseek_v2_weights.cpp:138-153`). The two are NOT the
// same shape even though `qk_nope_head_dim` and `v_head_dim` are both 256 on
// this checkpoint: `[64, 512, 192]` against `[64, 256, 512]`.
struct GlmMoeDsaMlaWeights {
  OwnedTensor q_a_proj;            // [q_lora_rank, hidden]
  OwnedTensor q_a_layernorm;       // [q_lora_rank]
  OwnedTensor q_b_proj;            // [heads * (qk_nope + qk_rope), q_lora_rank]
  OwnedTensor kv_a_proj_with_mqa;  // [kv_lora_rank + qk_rope, hidden]
  OwnedTensor kv_a_layernorm;      // [kv_lora_rank]
  OwnedTensor k_b_proj;            // [heads, kv_lora_rank, qk_nope]
  OwnedTensor v_b_proj;            // [heads, v_head_dim, kv_lora_rank]
  OwnedTensor o_proj;              // [hidden, heads * v_head_dim]
  GlmMoeDsaIndexerWeights indexer;  // populated on a `kFull` layer only

  // ─── W9: the POST-LOAD ABSORPTION (#2214) ─────────────────────────────────
  // Upstream's own stage, under upstream's own name:
  // `MLAAttention.process_weights_after_loading` (`mla_attention.py:875-962`).
  // It runs in the LOADER, which is where `deepseek_v2_weights.cpp:138-153`
  // already calls the same shared absorber for the safetensors arm — the stage
  // needs the dequantizer and the open `GgufFile`, and both live there.
  //
  // WHY IT EXISTS HERE AT ALL, measured rather than assumed. `attn_k_b` and
  // `attn_v_b` above arrive in llama.cpp's orientation, where a row runs along
  // the CONTRACTION axis (`ggml_mul_mat` contracts `ne[0]`), and they arrive
  // Q8_0. The shared MLA decode arm needs `w_uk_t` `[heads, qk_nope, kv_lora]`
  // and `w_uv` `[heads, kv_lora, v_head]` for `vt::BatchedMatmul`, whose
  // contract is "a/b share f32 or bf16" and "only the innermost dimension must
  // be unit-stride" — so neither a transposed view nor the quantized bytes are
  // admissible, and a per-head TRANSPOSE INTO BF16 is the only form that is.
  //
  // That is exactly upstream's situation and exactly upstream's answer:
  // `W_UK_T` and `W_UV` are plain bf16 copies, and `mla_attention.py:876-878`
  // gives the reason — "we currently do not have quantized bmm's which are
  // needed for W_UV and W_UK_T ... the extra memory overhead of this is fairly
  // low".
  //
  // THE COST IS STATED RATHER THAN DISCOVERED: 58.8 MB per layer over 78 layers
  // = 4.48 GiB on top of §3.3's 14.511 GiB resident class, so the class this
  // model actually holds is ~18.99 GiB. Spec §3.7 W9 F4, and O9 is restated
  // against that number rather than quietly failed.
  //
  // `kv_b_proj` is the CHECKPOINT-layout `[heads*(qk_nope+v_head), kv_lora]`
  // linear the PREFILL (materialized-MHA) arm needs. The GGUF does not ship it
  // — llama.cpp's converter already split it — so it is rebuilt from the two
  // halves: rows `[0, qk_nope)` of each head are `attn_k_b` transposed, and rows
  // `[qk_nope, qk_nope+v_head)` are `attn_v_b` VERBATIM, which is already in
  // that orientation. `w_uk_t` / `w_uv` are then produced from it by the SHARED
  // `mla::AbsorbKvBProjBf16` rather than by a second absorber written here.
  OwnedTensor kv_b_proj;  // bf16 [heads*(qk_nope+v_head), kv_lora]
  OwnedTensor w_uk_t;     // bf16 [heads, qk_nope, kv_lora]
  OwnedTensor w_uv;       // bf16 [heads, kv_lora, v_head]
};

// A SwiGLU MLP: the three leading dense layers, and every MoE layer's shared
// expert. The shared expert is `moe_intermediate_size * n_shared_experts`
// (2048), NOT `intermediate_size` (12288, what blocks 0-2 use).
struct GlmMoeDsaMlpWeights {
  OwnedTensor gate_proj;  // [inter, hidden]
  OwnedTensor up_proj;    // [inter, hidden]
  OwnedTensor down_proj;  // [hidden, inter]
  bool Empty() const { return gate_proj.Empty(); }
};

// One MoE layer.
//
// THE THREE TOWERS ARE STACKED AND STAY STACKED, and that is the whole reason
// this model is loadable at all. DeepSeek-V2 holds `std::vector<OwnedTensor>`
// per expert (`deepseek_v2.h:250-259`), which at 256 experts x 75 layers x 3
// is 57,600 host tensors and no streaming source. Only the stacked keep-quant
// form reaches `expert_stream::ExpertSlice`, because a slice of it is a pure
// byte offset over whole rows of the same K (`gguf_expert_span.h:11-16`).
struct GlmMoeDsaMoeWeights {
  OwnedTensor router;                   // f32 [n_routed_experts, hidden]
  OwnedTensor e_score_correction_bias;  // f32 [n_routed_experts]
  OwnedTensor gate_exps;  // [E, moe_inter, hidden], held flat as [E*out, K]
  OwnedTensor up_exps;    // [E, moe_inter, hidden]
  OwnedTensor down_exps;  // [E, hidden, moe_inter]
  GlmMoeDsaMlpWeights shared;
  bool Empty() const { return router.Empty(); }
};

struct GlmMoeDsaLayerWeights {
  OwnedTensor input_layernorm;           // [hidden]
  OwnedTensor post_attention_layernorm;  // [hidden]
  GlmMoeDsaMlaWeights attn;
  bool is_moe = false;
  GlmMoeDsaMlpWeights dense;  // populated iff !is_moe
  GlmMoeDsaMoeWeights moe;    // populated iff  is_moe
};

// The loaded model.
struct GlmMoeDsaWeights {
  GlmMoeDsaParams params;
  OwnedTensor embed_tokens;  // [vocab, hidden]
  OwnedTensor final_norm;    // [hidden]
  // NOT tied on this checkpoint: the file ships `output.weight` beside
  // `token_embd.weight`, both Q4_K. The tie is read OFF THE FILE rather than
  // out of the config, because the config's `tie_word_embeddings` describes the
  // source checkpoint and a converter is free to materialize either shape.
  OwnedTensor lm_head;             // [vocab, hidden]; empty only when tied
  OwnedTensor rope_cos_sin_cache;  // bf16 [rows, qk_rope_head_dim]
  std::vector<GlmMoeDsaLayerWeights> layers;  // num_hidden_layers, MTP excluded

  // Structural accounting, so a load that silently skipped a tensor is visible
  // rather than inferred from a green suite. `accounted + dropped` must equal
  // the file's own tensor count; the load refuses when it does not.
  int64_t file_tensors = 0;
  int64_t accounted_tensors = 0;
  // Block 78 is the multi-token-prediction block. It is READ, COUNTED and
  // DROPPED, which is what `allow_mtp_tail` means here (spec O5); there is no
  // MTP drafter in this tree.
  int64_t mtp_block_tensors_dropped = 0;
  // The `indexer.*` tensors the conversion broadcast onto `kShared` blocks.
  // Counted as dropped rather than ignored, because the count is what makes
  // spec D3 executable: 57 shared backbone layers x 5 tensors = 285.
  int64_t broadcast_indexer_tensors_dropped = 0;

  // W9: whether the post-load absorption ran (`GlmMoeDsaMlaWeights`'s
  // `kv_b_proj` / `w_uk_t` / `w_uv`). The forward REFUSES a model without it BY
  // NAME rather than reading the empty tensors, because
  // `mla::MlaBlockWeights`'s own `RequireWeight` would name `w_uk_t` without
  // saying that a whole stage did not run. Only `LoadGlmMoeDsaFromGguf` sets
  // it; a hand-constructed `GlmMoeDsaWeights` is therefore refused, which is
  // the polarity a test that builds the struct by hand needs.
  bool absorbed = false;
};

// Load the `glm-dsa` GGUF arm. `policy` may be null, in which case the process
// policy is read from the environment. Throws `std::runtime_error` by name on
// every tensor this port cannot serve.
GlmMoeDsaWeights LoadGlmMoeDsaFromGguf(const GgufFile& gguf,
                                       const HfConfig& config,
                                       const GgufLoadPolicy* policy);

// Every tensor name this port CLAIMS for a given config. Exposed so the census
// gate can assert the claim set against the real shard headers without loading
// 201.83 GiB, which is the only way that assertion runs in CI.
std::vector<std::string> EnumerateGlmMoeDsaGgufTensors(
    const GlmMoeDsaParams& params);

// ─── W9: what the post-load absorption does, and the narrowing it verifies ───
//
// `LoadGlmMoeDsaFromGguf` runs it and sets `GlmMoeDsaWeights::absorbed`. It does
// three things, and each is a MEMORY-FORMAT decision a token gate cannot see, so
// each is named here:
//
//  1. builds `kv_b_proj`, `w_uk_t` and `w_uv` (see `GlmMoeDsaMlaWeights`) —
//     `w_uk_t` and `w_uv` through the SHARED `mla::AbsorbKvBProjBf16`, not a
//     second absorber;
//  2. NARROWS every `kFull` layer's `indexer.weights_proj` from the file's F32
//     to bf16, because the shared MLA block computes the indexer in the block
//     dtype and `vt::MatmulBT` needs matching float operands. Upstream's
//     `wk_weights_proj` carries no `params_dtype` (`deepseek_v2.py:700-707`) and
//     is therefore the model dtype, so bf16 IS the mirror — the file's F32 is
//     llama.cpp's lossless upcast of a bf16 tensor. That claim is EXECUTABLE
//     rather than assumed: every value must round-trip through bf16 exactly, and
//     a file whose indexer projection is genuinely f32 is REFUSED BY NAME rather
//     than silently rounded, because rounding it would move a discrete top-k
//     that no tolerance bounds;
//  3. leaves the ROUTER gate at F32, and this is the one place this port is
//     deliberately WIDER than vLLM. Upstream's tier-5 path is bf16 x bf16 ->
//     f32 (`fused_moe/router/gate_linear.py`); the forward here runs f32 x f32
//     -> f32, on the smallest GEMM in the model (`[T,6144] x [256,6144]`), in
//     the direction vLLM's OWN `force_fp32_compute` arm takes when no
//     specialized kernel is available — and it is what llama.cpp does on this
//     identical artifact, which §3.6 makes the only oracle this row can compare
//     against. The cost is one `vt::CastF32` of the hidden state per MoE layer.
//
// The forward. Composes only shared seams: `mla::ForwardMlaAttentionBlock` over
// `GlmMoeDsaMlaSchedule`'s per-layer dims and ONE `mla::MlaSharedSelection`,
// `expert_stream::ExpertSlice` for the 256-expert towers,
// `layers::MlpGateUpMethodBase` for every dense MLP, `vt::FusedChain` for the
// residual add + RMSNorm, and `vt::MoeRouterTopK` / `vt::MoeCombine` for the
// routed-expert composition.
//
// WHAT IT STILL REFUSES, BY NAME. A step in which any request RESUMES while the
// selection PRUNES: the indexer's `k` for a token comes from that token's own
// hidden state, so a resumed request needs the indexer's own 128-wide side
// cache, which is `KV-DSV4-MULTICACHE`'s (spec O4,
// [#1925](https://github.com/mudler/vllm.cpp/issues/1925),
// [#2323](https://github.com/mudler/vllm.cpp/issues/2323)). A FIRST token on a
// fresh prompt is reachable and a SECOND is not.
class GlmMoeDsaModel {
 public:
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const v1::CommonAttentionMetadata& attn_meta,
                                    const std::vector<PagedKvCache>& attn_kv,
                                    const GlmMoeDsaWeights& weights,
                                    vt::Queue& queue,
                                    const std::vector<int32_t>& logits_indices);

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const GlmMoeDsaWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices);
};

// The refusal text both entry points raise, exposed so a test can assert the
// message NAMES each missing primitive rather than re-spelling it.
const char* GlmMoeDsaForwardRefusal();

// The refusal the safetensors arm raises (spec D1).
const char* GlmMoeDsaSafetensorsRefusal();

}  // namespace vllm
