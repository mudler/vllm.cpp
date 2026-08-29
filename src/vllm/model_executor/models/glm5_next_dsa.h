// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W3: the DSA indexer's k-pool
// compression stage and its always-kept ragged tail.
//
// Model-private header, deliberately not under `include/`: nothing outside this
// model needs these types yet, and `include/vllm.h` is the ABI seam a shipped
// capability is exposed through. Same arrangement as `glm5_next.h` (W1),
// `glm5_next_mhc.h` (W4) and `qwen4_exp.h`.
//
// ORACLE. vLLM registers no `glm5_next` at our parity pin `555967922` nor at
// its `main`, and neither do vllm-omni, SGLang or llama.cpp. Under AGENTS.md
// "When vLLM has no implementation" the reference for this surface is
// `transformers` **v5.16.1**, the commit `refs/tags/v5.16.1` resolves to,
// `93c8b7b485963a10800c91f55304db6be211c2bd`. W0 (#2096) owns recording that
// lane revision in `.agents/oracles/transformers.md`; this file cites it.
//
// ─── WHAT IS NET-NEW, AND WHY DEEPSEEK-V4'S INDEXER IS THE WRONG REUSE ───────
//
// `Glm5NextTextIndexer` INHERITS `GlmMoeDsaIndexer`, the DeepSeek-V3.2 lightning
// indexer whose numerics `deepseek_v4_dsa.cpp` already ports, and then changes
// WHAT THE TOP-K RUNS OVER. The parent scores raw tokens. This one scores
// **pooled keys**: `index_kpool` consecutive valid tokens are compressed into
// one candidate by a LEARNED, per-channel, 4-way softmax, the top
// `index_topk / index_kpool` POOLS are selected, and the selected pools are then
// expanded back into raw token indices. The ragged tail that does not fill a
// pool is appended raw and UNSCORED.
//
// `deepseek_v4_dsa.cpp` has NO pooling stage at all, so reaching for
// `DsaTopkSelect` here selects the wrong candidate set. It is the documented
// wrong reuse (`.agents/specs/glm5-next-flash.md`, "The k-pool indexer is a
// compression stage DeepSeek-V4 does not have"), and it fails in the quiet
// direction: feeding raw-token candidates into the top-k, or pooled candidates
// into a consumer expecting raw ones, yields plausible indices either way.
// `DeepseekV4HCACompressor`'s softmax-over-window weighted pool is the closest
// SHAPE in this tree, and it is still not this: it carries a `compress_rate`, a
// post-pool RMSNorm and a RoPE, none of which appear here.
//
// **A SHORT-PROMPT GATE CANNOT SEE ANY OF THIS.** With `index_topk = 2048`, any
// context at or below 2048 candidate positions selects everything, the selection
// is the identity, and the pooling is unobservable. Every gate on this file runs
// past that threshold; see `tests/vllm/models/test_glm5_next_dsa.cpp`.
//
// ─── PORT ANCHORS (file:line on BOTH sides) ──────────────────────────────────
//   OURS                                <-  transformers v5.16.1, models/glm5_next/
//   glm5_next::PackIndexerStates        <-  modular_glm5_next.py:795-801
//                                           (`Glm5NextTextIndexer.forward`, the
//                                           packed [k | gate_scores | valid] row)
//   glm5_next::GetVisibleTokens         <-  modular_glm5_next.py:877-895
//                                           (`Glm5NextTextIndexer.get_visible_tokens`)
//   glm5_next::GetPooledStates          <-  modular_glm5_next.py:897-970
//                                           (`Glm5NextTextIndexer.get_pooled_states`)
//   glm5_next::AppendVisibleTail        <-  modular_glm5_next.py:972-1022
//                                           (`Glm5NextTextIndexer.append_visible_tail`)
//   glm5_next::SelectIndexerTopk        <-  modular_glm5_next.py:771-875
//                                           (`Glm5NextTextIndexer.forward`)
//
// ─── THE PACKED CACHE IS WIDER THAN DEEPSEEK-V4'S, AND THAT IS STRUCTURAL ────
//
// The parent caches `k` alone, `index_head_dim` floats per token per layer. This
// one caches `concat[k(128), gate_scores(128), valid(1)]` = **257** floats per
// token per layer (`:798-808`), because the pools are re-formed over the WHOLE
// history on every call and the gate score of a token that has already left the
// current window still decides how that token is weighted inside its pool. A
// port that recomputes the gate from the current step's hidden states alone is
// wrong for every token but the last. `PackIndexerStates` produces exactly that
// row so the future cache carries it unchanged; W5 owns the allocator.
//
// ─── HOST REFERENCE, f32 ─────────────────────────────────────────────────────
//
// This file is a host f32 reference, exactly as `glm5_next_mhc.cpp` and
// `glm5_next_kda.cpp` are. The reference itself computes the scores in fp32
// (`scores = torch.matmul(q.float(), pool_keys...float())`, `:823`) and the pool
// softmax in fp32 (`:960-964`), so f32 here is upstream's own arithmetic and not
// a widening. The device arm is W5's, and it is named as owed rather than
// implied.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DSA_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DSA_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"

namespace vllm::glm5_next {

// The indexer geometry, resolved. Built from a parsed config by
// `IndexerDimsFrom` below — never by hand in production code, because
// `index_kpool` is 4 on the published checkpoint and 16 in the upstream config
// class, and a hand-built value is exactly the defaulting this model punishes.
struct IndexerDims {
  int64_t hidden_size = 0;   // `config.hidden_size`      — 4096
  int64_t q_lora_rank = 0;   // `config.q_lora_rank`      — 1536
  int64_t n_heads = 0;       // `config.index_n_heads`    — 32
  int64_t head_dim = 0;      // `config.index_head_dim`   — 128
  int64_t index_topk = 0;    // `config.index_topk`       — 2048
  int64_t index_kpool = 0;   // `config.index_kpool`      — 4 (class default 16)
  bool always_select_tail = true;  // `config.index_kpool_always_select_tail`

  // `self.softmax_scale = self.head_dim**-0.5` (`modeling_glm5_next.py:765`).
  // NOTE this is the INDEXER head dim, not the MLA one.
  float softmax_scale() const;

  // `select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])`
  // (`:845`). `validate_architecture` (`configuration_glm5_next.py:219-220`)
  // enforces `index_topk % index_kpool == 0`, so the floor division is exact.
  int64_t SelectK(int64_t num_pools) const;

  // `output_width = self.index_topk (+ self.index_kpool - 1 with the tail)`
  // (`:864-867`) — **2051** on the published checkpoint, not 2048. Getting this
  // to 2048 truncates the tail the model always keeps.
  int64_t OutputWidth() const;

  // Refuses a partial or incoherent group BY NAME rather than serving a wrong
  // selection: every field > 0, and `index_topk % index_kpool == 0`, which is
  // upstream's own `validate_architecture` clause.
  void Validate() const;
};

// Reads the RESOLVED config. `p.indexer.kpool` is the checkpoint's 4; nothing
// here re-derives a default, which is the whole point of routing through the
// parsed params instead of taking an `index_kpool` argument.
IndexerDims IndexerDimsFrom(const Glm5NextParams& p);

// The indexer's own projections. Row-major, host f32, torch `[out, in]` layout
// for every linear (`nn.Linear.weight`).
struct IndexerWeights {
  const float* wq_b = nullptr;           // [n_heads * head_dim, q_lora_rank]
  const float* wk = nullptr;             // [head_dim, hidden_size]
  // `self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)` — a LayerNorm WITH
  // BIAS, not an RMSNorm. The checkpoint carries `indexer.k_norm.bias`, which
  // settles it (spec, "Constants and layouts a port gets silently wrong").
  const float* k_norm_weight = nullptr;  // [head_dim]
  const float* k_norm_bias = nullptr;    // [head_dim]
  const float* weights_proj = nullptr;   // [n_heads, hidden_size]
  // `index_kpool_compress_ape` — a LEARNED intra-pool absolute-position
  // embedding, [index_kpool, head_dim]. It is added to the gate scores inside
  // the pool softmax, so it is what makes the pool weighting position-aware.
  const float* kpool_ape = nullptr;      // [index_kpool, head_dim]
  // `index_kpool_compress_gate` — applied as `F.linear(x, W)`, so [head_dim,
  // hidden_size] and the output is head_dim wide: **128 INDEPENDENT 4-way
  // softmaxes**, one per channel, not one softmax over the pool.
  const float* kpool_gate = nullptr;     // [head_dim, hidden_size]
};

// `nn.LayerNorm` eps for `k_norm` (`modeling_glm5_next.py:763`). Named because
// it is NOT `rms_norm_eps` and is not the config's `hc_eps` either.
inline constexpr float kIndexerKNormEps = 1e-6f;

// The packed per-token indexer state, `[B, S, 2 * head_dim + 1]`:
// `concat[k(head_dim), gate_scores(head_dim), valid(1)]` (`:798-801`).
// `mask` is the local boolean padding mask `[B, S]`, 0 for a pad slot.
std::vector<float> PackIndexerStates(const IndexerDims& d, const IndexerWeights& w,
                                     const std::vector<float>& hidden,
                                     const std::vector<uint8_t>& mask, int64_t batch,
                                     int64_t seq_len);

// `[B, q_length, kv_len]`, 1 where the key is causally reachable AND not padding
// (`:877-895`). `current_length` is the cache's sequence length, which equals
// `q_length` only on a fresh prefill.
std::vector<uint8_t> GetVisibleTokens(const std::vector<uint8_t>& valid_keys, int64_t batch,
                                      int64_t kv_len, int64_t q_length,
                                      int64_t current_length);

// The compressed candidate set (`:897-970`).
struct PooledStates {
  int64_t num_pools = 0;              // P AFTER the `keep` compaction (`:968-970`)
  std::vector<float> pool_keys;       // [B, P, head_dim]
  std::vector<int32_t> pool_indices;  // [B, P, index_kpool], -1 for an invalid member
  std::vector<uint8_t> pool_valid;    // [B, P], 1 iff ALL members are valid
};
// Takes the WEIGHTS because upstream's method reads the module's own
// `self.index_kpool_compress_ape` (`:960`); passing the pooled keys in from
// outside would move the learned part of the pooling out of the mirror.
PooledStates GetPooledStates(const IndexerDims& d, const IndexerWeights& w,
                             const std::vector<float>& packed, int64_t batch,
                             int64_t kv_len);

// Appends the current INCOMPLETE pool as raw, unscored token indices (`:972-1022`),
// widening each row by `index_kpool - 1`. `topk` is `[B, q_length, in_width]`.
std::vector<int32_t> AppendVisibleTail(const IndexerDims& d, const std::vector<int32_t>& topk,
                                       int64_t in_width,
                                       const std::vector<uint8_t>& visible,
                                       const std::vector<uint8_t>& valid_keys, int64_t batch,
                                       int64_t q_length, int64_t kv_len);

// The whole selection (`:771-875`).
struct IndexerSelection {
  // `[B, q_length, OutputWidth()]` int32 token indices, **-1 is the invalid
  // sentinel throughout** and duplicates are possible — upstream absorbs them
  // downstream with `scatter_add_` + `ne(0)` (`:1119-1129`).
  std::vector<int32_t> topk_indices;
  // `[B, q_length, P]`, the per-pool scores BEFORE the validity mask (`:828`).
  // Returned rather than discarded because it is the only way a gate can show
  // the selection is a strict separation and not a coin flip: top-k error is
  // BIMODAL, so a tolerance on the selected values passes a wrong set whose
  // values happen to be close. The gate asserts SET equality and prints the
  // margin these scores give.
  std::vector<float> index_scores;
  PooledStates pooled;
};

IndexerSelection SelectIndexerTopk(const IndexerDims& d, const IndexerWeights& w,
                                   const std::vector<float>& hidden,
                                   const std::vector<float>& q_resid,
                                   const std::vector<uint8_t>& mask, int64_t batch,
                                   int64_t seq_len);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_DSA_H_
