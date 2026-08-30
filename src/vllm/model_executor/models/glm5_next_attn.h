// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W5b-1: `Glm5NextTextAttention`, the
// NoPE MLA block the DSA layers run, and its CROSS-LAYER top-k sharing.
//
// Issue [#2241](https://github.com/mudler/vllm.cpp/issues/2241), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md` §W5b.
//
// Model-private header, deliberately not under `include/`: nothing outside this
// model needs these types yet, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. Same arrangement as `glm5_next.h` (W1),
// `glm5_next_mhc.h` (W4), `glm5_next_dsa.h` (W3) and `glm5_next_moe.h` (W5).
//
// ORACLE. vLLM registers no `glm5_next` at our parity pin `555967922` nor at its
// `main`, and neither do vllm-omni, SGLang or llama.cpp. Under AGENTS.md "When
// vLLM has no implementation" the reference for this surface is `transformers`
// **v5.16.1**, this row's lane pin (`.agents/oracles/transformers.md`), whose
// `models/glm5_next/modeling_glm5_next.py` sha256 is
// `2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b` — asserted
// by the golden generator against the INSTALLED module, not assumed from a
// version string.
//
// ─── PORT ANCHORS (file:line on BOTH sides) ──────────────────────────────────
//   OURS                                 <-  transformers v5.16.1, models/glm5_next/
//   glm5_next::MlaDims                   <-  modeling_glm5_next.py:1079-1088
//                                            plus :1128 (`scaling`), in `__init__`
//   glm5_next::IndexerRoleFor            <-  :1130 (`skip_topk`) and
//                                            :1132-1134 (`next_skip_topk`)
//   glm5_next::QResid                    <-  :1167 (`q_a_layernorm(q_a_proj(x))`)
//   glm5_next::CompressKv                <-  :1170-1172
//                                            (`kv_a_proj_with_mqa`, the split,
//                                             `kv_a_layernorm`)
//   glm5_next::ExpandKv                  <-  :1136-1153 (`expand_kv`)
//   glm5_next::BuildAttentionMaskFromTopk<-  :1218-1256
//   glm5_next::Attention                 <-  :1155-1216 (`forward`) plus
//                                            :1039-1061 (`eager_attention_forward`)
//
// ─── THREE THINGS A PORT GETS SILENTLY WRONG HERE ────────────────────────────
//
// 1. **The `kv_b_proj` halves are SPLIT and only the K half is TRANSPOSED.**
//    Upstream declares ONE `nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim
//    + v_head_dim))` and `expand_kv` applies it whole. The published GGUF does
//    not carry it: llama.cpp #27752 inherits `conversion/deepseek.py`'s
//    `DeepseekV2Model.modify_tensors` ("MLA with the absorption optimization,
//    needs these two split and k_b_proj transposed"), so the file has
//    `attn_k_b.weight` at `[H, kv_lora, qk_nope]` and `attn_v_b.weight` at
//    `[H, v_head, kv_lora]` and NO `attn_kv_b.weight`. `Glm5NextMlaWeights`
//    (`glm5_next_loader.h`) keeps both in the file's own shapes, so THIS file is
//    where the asymmetry has to be honoured: K contracts over its FIRST inner
//    axis and V over its SECOND. At the published geometry the two orientations
//    have different widths (kv_lora 512, qk_nope 256) and a swap is a shape
//    error — but at any geometry where they coincide it is a SILENT value
//    error, so the gate carries a square case for exactly that.
//
// 2. **Cross-layer top-k sharing.** `config.indexer_types[layer_idx] ==
//    "shared"` means the layer builds NO indexer and REUSES the previous full
//    layer's selection (`:1130-1134`, `:1181-1191`). A layer that builds its own
//    indexer where upstream shares is a fluent wrong model: it runs, it selects
//    a plausible key set, and it emits plausible tokens. Nothing about the
//    output's shape, finiteness or scale says otherwise. `IndexerRoleFor` is
//    the whole decision, isolated so it can be gated on its own.
//
// 3. **The all-masked row is `finfo.min`, NOT `-inf`.** `:1253-1256` fills a
//    non-visible position with `torch.finfo(dtype).min`. A left-padded query row
//    reaches a state where EVERY key is masked; with `finfo.min` its softmax is
//    UNIFORM and its output is finite, and with `-inf` every term is NaN and the
//    NaN propagates through `o_proj` into the residual stream for the rest of
//    the stack. The fixture carries such a row.
//
// ─── THE ROPE HALF HAS NO WIDTH, AND UPSTREAM IS WHAT SAYS SO ────────────────
//
// `expand_kv` concatenates `k_nope` with `k_rot` (`:1150-1152`) and `forward`
// splits `k_rot` off `compressed_kv` (`:1171`). Both halves are ZERO-WIDTH for
// this architecture: `Glm5NextTextConfig.validate_architecture`
// (`configuration_glm5_next.py:225-228`) RAISES "Expecting NoPE for the DSA
// attention layers, but got {n} as RoPE dim." for any positive
// `qk_rope_head_dim`, and the golden generator constructs one to MEASURE that
// rather than describe it. So this port implements no rope branch — a branch no
// released config can select is the "unselected branch" shape
// (`.agents/reachability.md`) — and `MlaDims::Validate` mirrors the refusal
// instead. `key_states` IS `k_nope`, at width `qk_head_dim() == qk_nope_head_dim`.
//
// ─── `repeat_kv` IS THE IDENTITY HERE ────────────────────────────────────────
//
// `num_key_value_groups = num_attention_heads // num_key_value_heads`
// (`:1088`), and `validate_architecture` requires the two counts EQUAL for this
// model (mirrored in `ParseGlm5NextParams`), so `n_rep` is 1 and `repeat_kv`
// returns its input unchanged (`:1033-1034`). MLA expands the latent to
// `num_heads` keys in `expand_kv` already; there is no grouped-query stage to
// port.
//
// ─── HOST REFERENCE, f32 ─────────────────────────────────────────────────────
//
// This file is a host f32 reference, exactly as `glm5_next_dsa.cpp`,
// `glm5_next_mhc.cpp` and `glm5_next_moe.cpp` are. The reference's own softmax
// is `dtype=torch.float32` (`:1056`), so f32 there is upstream's arithmetic and
// not a widening; the projections upstream runs in the model dtype are widened
// here and that IS a deviation, recorded as such and shared with every other
// host reference on this row. The device arm is owed, not implied — see the
// spec's `## Owed`.
//
// ─── WHAT THIS FILE DOES NOT DO ──────────────────────────────────────────────
//
// No KV cache: upstream's `past_key_values.update` (`:1177-1179`) is a Cache
// object this reference has no equivalent of, and `MakeGlm5NextKVCache` (W5) is
// the production spec it will bind to. No decoder layer, no mHC threading, no
// `Glm5NextTextModel::Forward`, and NOTHING here is reached from a production
// entry point. W5b-2 owns all four; the spec's `## Owed` names it.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_ATTN_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_ATTN_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_dsa.h"

namespace vllm::glm5_next {

// The NoPE MLA geometry, resolved. Built from a parsed config by `MlaDimsFrom`
// below — never by hand in production code, for the reason `IndexerDims` gives:
// every field here has a class default that differs from the published
// checkpoint's value.
struct MlaDims {
  int64_t hidden_size = 0;       // `config.hidden_size`        — 4096
  int64_t num_heads = 0;         // `config.num_attention_heads`— 64
  int64_t q_lora_rank = 0;       // `config.q_lora_rank`        — 1536
  int64_t kv_lora_rank = 0;      // `config.kv_lora_rank`       — 512
  int64_t qk_nope_head_dim = 0;  // `config.qk_nope_head_dim`   — 256
  // ZERO, and `Validate()` refuses anything else. See the header comment.
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;        // `config.v_head_dim`         — 256
  // `config.rms_norm_eps` — 1e-5 here, NOT the 1e-6 the `GlmMoeDsa` parent uses
  // and NOT the indexer's `kIndexerKNormEps`.
  double rms_norm_eps = 0.0;

  // `self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim`
  // (`:1087`). 256 here, because the rope half has no width.
  int64_t qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }

  // `self.scaling = self.qk_head_dim ** (-0.5)` (`:1128`). NOTE this is the MLA
  // head dim, not the indexer's, and not `v_head_dim`.
  float scaling() const;

  // Refuses a partial or incoherent group BY NAME rather than serving a wrong
  // attention: every field positive, and `qk_rope_head_dim` EXACTLY zero, which
  // is upstream's own `validate_architecture` clause and the reason this port
  // has no rope branch.
  void Validate() const;
};

MlaDims MlaDimsFrom(const Glm5NextParams& p);

// One DSA layer's MLA projections, host f32, row-major, torch `[out, in]`
// layout for every plain linear — EXCEPT the two `kv_b_proj` halves, which are
// in the CHECKPOINT's own absorbed shapes. Every projection is bias-free
// (`attention_bias` is `false` on the published checkpoint and the loader
// carries no bias tensor for any of them).
struct MlaWeights {
  std::vector<float> q_a_proj;            // [q_lora_rank, hidden_size]
  std::vector<float> q_a_layernorm;       // [q_lora_rank]
  std::vector<float> q_b_proj;            // [num_heads * qk_head_dim, q_lora_rank]
  // [kv_lora_rank + qk_rope_head_dim, hidden_size] — [kv_lora_rank, hidden_size]
  // at this model's only admissible geometry.
  std::vector<float> kv_a_proj_with_mqa;
  std::vector<float> kv_a_layernorm;      // [kv_lora_rank]
  // TRANSPOSED by the converter and left that way: `k_b_proj[h][r][d]` is the
  // weight of latent channel `r` on nope channel `d`, so K contracts over the
  // FIRST inner axis.
  std::vector<float> k_b_proj;            // [num_heads, kv_lora_rank, qk_nope_head_dim]
  // NOT transposed: `v_b_proj[h][d][r]`, so V contracts over the SECOND.
  std::vector<float> v_b_proj;            // [num_heads, v_head_dim, kv_lora_rank]
  std::vector<float> o_proj;              // [hidden_size, num_heads * v_head_dim]
};

// The cross-layer sharing decision, and nothing else. Isolated because it is
// the whole of trap 2 and because a gate on it can then be a set of equalities
// over a schedule rather than a forward it has to run.
struct IndexerRole {
  // `self.skip_topk = config.indexer_types[layer_idx] == "shared"` (`:1130`).
  // True means this layer builds NO indexer and REQUIRES `prev_topk_indices`.
  bool skip_topk = false;
  // `self.next_skip_topk = not self.skip_topk and
  //  config.indexer_types[min(layer_idx + 1, len - 1)] == "shared"` (`:1132-1134`).
  // True means `Attention` returns its selection for the NEXT layer to reuse.
  // NOTE the `min` CLAMP: the last layer looks at ITSELF, so a final `full`
  // layer never propagates and a final `shared` layer would make its own
  // predecessor propagate — which is upstream's arithmetic, not a guard.
  bool next_skip_topk = false;
};

// Throws by name for a `layer_idx` outside `[0, indexer_types.size())` rather
// than reading past the end: the clamp above is upstream's and applies to
// `layer_idx + 1` only.
IndexerRole IndexerRoleFor(const Glm5NextParams& p, int64_t layer_idx);

// `q_a_layernorm(q_a_proj(hidden_states))` (`:1167`). This value is used TWICE
// upstream — as the input to `q_b_proj` and as the indexer's `q_resid`
// (`:1184`) — so it is returned rather than recomputed.
//
//   hidden : [batch, seq_len, hidden_size]  row-major
//   returns: [batch, seq_len, q_lora_rank]  row-major
std::vector<float> QResid(const MlaDims& d, const MlaWeights& w,
                          const std::vector<float>& hidden, int64_t batch,
                          int64_t seq_len);

// `kv_a_layernorm(split(kv_a_proj_with_mqa(hidden))[0])` (`:1170-1172`).
//
// Returns `k_pass` ONLY. Upstream's `k_rot` is the second half of the split and
// has ZERO WIDTH at this architecture's only admissible geometry, so there is
// nothing to return; see the header comment for the measurement.
//
//   returns: [batch, seq_len, kv_lora_rank]  row-major
std::vector<float> CompressKv(const MlaDims& d, const MlaWeights& w,
                              const std::vector<float>& hidden, int64_t batch,
                              int64_t seq_len);

// `expand_kv` (`:1136-1153`) over the SPLIT, half-transposed halves.
struct ExpandedKv {
  // [batch, num_heads, seq_len, qk_head_dim] — head-major, as upstream's
  // `.view(...).transpose(1, 2)` produces.
  std::vector<float> key_states;
  // [batch, num_heads, seq_len, v_head_dim]
  std::vector<float> value_states;
};

//   k_pass : [batch, seq_len, kv_lora_rank]  row-major, from `CompressKv`
ExpandedKv ExpandKv(const MlaDims& d, const MlaWeights& w,
                    const std::vector<float>& k_pass, int64_t batch,
                    int64_t seq_len);

// `build_attention_mask_from_topk` (`:1218-1256`), returning the BOOLEAN
// visibility upstream's `sdpa` arm returns (`:1249-1250`). 1 == visible.
//
// The eager arm's `torch.where(mask, 0.0, finfo.min)` (`:1253-1256`) is a pure
// re-encoding of this same boolean and is applied inside `Attention`, which is
// the only consumer; materializing an additive float mask here would double the
// buffer for no information.
//
// Duplicates and the `-1` sentinel are absorbed exactly as upstream does:
// out-of-range entries contribute a ZERO to the scatter-add and an index that
// appears twice still yields one visible key (`selected_counts.ne(0)`).
//
//   topk   : [batch, q_length, width] int32, -1 is the invalid sentinel
//   returns: [batch, q_length, kv_length] uint8 — upstream's `unsqueeze(1)`
//            head axis is 1 and broadcasts, so it is not materialized.
std::vector<uint8_t> BuildAttentionMaskFromTopk(const std::vector<int32_t>& topk,
                                                int64_t batch, int64_t q_length,
                                                int64_t width, int64_t kv_length);

// What `Glm5NextTextAttention.forward` returns (`:1216`), made explicit.
struct AttentionResult {
  std::vector<float> attn_output;      // [batch, seq_len, hidden_size]
  // The selection this layer USED — its own when `full`, the caller's when
  // `shared`. Returned unconditionally because a gate that cannot see it cannot
  // tell a reused selection from a recomputed one.
  std::vector<int32_t> topk_indices;   // [batch, seq_len, topk_width]
  int64_t topk_width = 0;
  // `topk_indices if self.next_skip_topk else None` (`:1216`). The caller
  // propagates `topk_indices` to the next layer IFF this is true; upstream
  // returns `None` otherwise and a layer that propagates anyway lets a `full`
  // layer be silently overridden.
  bool propagates_topk = false;
};

// The whole block (`:1155-1216`), with `eager_attention_forward` (`:1039-1061`)
// inlined because it is the only interface this model's 3-D top-k mask can
// reach — upstream says so at `:1227-1228`, and the reason is that the mask
// selects per (query, key) pair and no FlashAttention kernel takes one.
//
// `indexer` is the layer's own indexer weights and MUST be null exactly when
// `role.skip_topk` is true, which is upstream's `self.indexer = None if
// self.skip_topk else Glm5NextTextIndexer(...)` (`:1131`). Both mismatches
// throw by name rather than silently choosing an arm.
//
// `prev_topk_indices` is REQUIRED when `role.skip_topk` is true and IGNORED
// otherwise. Upstream raises `ValueError("Shared DSA layers require top-k
// indices from a previous full indexer layer.")` (`:1189-1190`) and so does
// this; the message is mirrored so a log line means the same thing on both
// sides.
//
//   hidden : [batch, seq_len, hidden_size]  row-major
//   mask   : [batch, seq_len] uint8, 0 for a padding slot — the indexer's
//            `attention_mask`, NOT an attention bias.
AttentionResult Attention(const MlaDims& d, const MlaWeights& w,
                          const IndexerDims& id, const IndexerWeights* indexer,
                          const IndexerRole& role,
                          const std::vector<float>& hidden,
                          const std::vector<uint8_t>& mask,
                          const std::vector<int32_t>* prev_topk_indices,
                          int64_t prev_topk_width, int64_t batch,
                          int64_t seq_len);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_ATTN_H_
