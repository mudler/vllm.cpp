// The MLA (Multi-head Latent Attention) block — MLA campaign W6.
//
// This is the model-side analog of dense_attn_block.h: it composes the pieces
// W3/W4/W5 landed (vt::ConcatAndCacheMla, vt::MlaDecodeAttention,
// vt::MlaPrefillAttention + the chunked-context loop) into ONE attention layer,
// and it adds the two things only a MODEL can own — the projections and WEIGHT
// ABSORPTION.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin e24d1b24) ───────
//   OURS                              <-  UPSTREAM
//   MlaBlockDims                      <-  vllm/model_executor/models/deepseek_v2.py
//                                         :952-1075 `DeepseekV2MLAAttention.__init__`
//                                         (qk_head_dim :969, scaling :981-996,
//                                          mscale^2 :1067-1075)
//   MlaBlockWeights                   <-  deepseek_v2.py:1003-1049 (the module
//                                         set) + vllm/model_executor/layers/mla.py
//                                         :14-30 `MLAModules`
//   AbsorbKvBProjBf16                 <-  vllm/model_executor/layers/attention/
//                                         mla_attention.py:875-962
//                                         `MLAAttention.process_weights_after_loading`
//                                         (:892-900 split, :959-962 the two
//                                          permutes)
//   YarnGetMscale                     <-  vllm/model_executor/layers/
//                                         rotary_embedding/deepseek_scaling_rope.py
//                                         :20-23 `yarn_get_mscale`
//   BuildDeepseekRopeCosSinCache      <-  deepseek_scaling_rope.py:76-118
//                                         (`_compute_inv_freq` + the YaRN ramp,
//                                          `_compute_cos_sin_cache`) over
//                                         vllm/model_executor/layers/
//                                         rotary_embedding/common.py
//                                         (`yarn_find_correction_range`,
//                                          `yarn_linear_ramp_mask`)
//   MlaAttentionScale                 <-  deepseek_v2.py:981-996 + :1067-1075
//   ForwardMlaAttentionBlock          <-  mla.py:119-181
//                                         `MultiHeadLatentAttentionWrapper.forward`
//                                         + mla_attention.py:553-620
//                                         `MLAAttention.forward` (the kv-cache
//                                         update ORDER) + :624-874
//                                         `MLAAttention.forward_impl` (the
//                                         prefill/decode dispatch and the
//                                         absorbed decode path)
//
// ─── WEIGHT ABSORPTION, precisely ───────────────────────────────────────────
// The single most important structural fact, established by the spike (§2.2)
// and re-read here at the pin: absorption is a LOAD-TIME WEIGHT TRANSFORM PLUS
// TWO BATCHED GEMMs. It is not a fused kernel, which is why a portable C++ port
// needs no new attention kernel for it — only vt::BatchedMatmul (`torch.bmm`).
//
//   load time (mla_attention.py:892-900, :959-962):
//     kv_b_proj.weight [N*(P+V), L]  --.T-->  [L, N*(P+V)]
//                                    --view-> [L, N, P+V]
//                                    --split-> W_UK [L,N,P] , W_UV [L,N,V]
//     W_UK.permute(1,2,0) -> W_UK_T [N, P, L]        (:962)
//     W_UV.transpose(0,1) -> W_UV   [N, L, V]        (:960)
//
//   decode (mla_attention.py:775-789, 1024-1034):
//     ql_nope = bmm(q_nope^T [N,B,P], W_UK_T [N,P,L]) -> [N,B,L]   (:789)
//     mqa_q   = concat(ql_nope^T [B,N,L], q_pe [B,N,R]) -> [B,N,576]
//     attn    = MQA(mqa_q, latent)                     -> [B,N,L]
//     out     = bmm(attn^T [N,B,L], W_UV [N,L,V])      -> [N,B,V]  (:1034)
//
// The identity that makes this legal, and that the W6 gate PROVES numerically
// rather than argues:
//     q_nope . (W_UK^T kv_c)  ==  (q_nope W_UK) . kv_c            (scores)
//     sum_s p_s (W_UV^T kv_c_s)  ==  W_UV^T (sum_s p_s kv_c_s)    (output)
// i.e. the ABSORBED decode and the UNABSORBED materialized-MHA reference
// compute the same function — the first in latent space (QK 576 / V 512, ONE
// KV head, K/V never materialized), the second in per-head space (QK 192 /
// V 128, N KV heads). `tests/vllm/model_executor/layers/attention/
// test_mla_attention_block.cpp` asserts that equivalence directly.
//
// ─── PREFILL/DECODE DISPATCH ────────────────────────────────────────────────
// Exactly upstream's rule (mla_attention.py:700-709): the split is purely the
// SCHEDULER's prefill/decode label — `num_mqa_tokens = num_decode_tokens`, the
// rest are MHA — and DECODE TOKENS ARE PACKED FIRST in the batch, so the MHA
// call takes the tail `q[num_mqa_tokens:]` (:722-737) and the MQA call the head
// `q[:num_mqa_tokens]` (:739+). Upstream's own comment at `:19-22` notes this is
// a heuristic they may tune; we mirror it and say so rather than treating it as
// a hard invariant.
//
// ─── DECOUPLED RoPE ─────────────────────────────────────────────────────────
// Only the qk_rope_head_dim (64) slice rotates. The rotary is constructed over
// `qk_rope_head_dim` ONLY and with `is_neox_style=False`
// (deepseek_v2.py:1053-1064), i.e. the adjacent-pair GPT-J rotation, and it is
// applied to `q[..., qk_nope_head_dim:]` and to the SINGLE shared `k_pe` head
// (mla.py:158-167). The attention softmax scale carries the YaRN **mscale
// SQUARED** correction (deepseek_v2.py:1067-1075) — getting that squared factor
// wrong is a silent accuracy bug, not a crash, which is why MlaAttentionScale
// exists as its own gated function.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/layers/attention/mla_chunked_context.h"
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev / DBuf / MakeTensor
#include "vllm/v1/attention/backend.h"                     // MLACommonMetadata
#include "vt/ops.h"

namespace vllm {
namespace mla {

// `DeepseekV2MLAAttention.__init__` geometry (deepseek_v2.py:960-1002). The
// DeepSeek-V2-Lite values (W0-confirmed against the shipped config.json) are
// hidden_size 2048, num_heads 16, qk_nope 128, qk_rope 64, v_head_dim 128,
// kv_lora_rank 512, q_lora_rank NULL; V3 differs only in hidden_size 7168,
// num_heads 128 and q_lora_rank 1536 — the MLA geometry itself is IDENTICAL.
struct MlaBlockDims {
  int64_t hidden_size = 0;
  int64_t num_heads = 0;
  int64_t qk_nope_head_dim = 0;
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;
  int64_t kv_lora_rank = 0;
  // `q_lora_rank` — 0 means upstream's `None`, i.e. the DIRECT `q_proj` branch
  // (deepseek_v2.py:1028-1034). DeepSeek-V2-Lite is this branch.
  int64_t q_lora_rank = 0;
  float rms_norm_eps = 1e-6f;
  // The softmax scale INCLUDING the YaRN mscale^2 correction — build it with
  // MlaAttentionScale(), never by hand.
  float scale = 0.0f;

  // `self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` (:969) — 192.
  int64_t qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // The MLA cache head_size `kv_lora_rank + qk_rope_head_dim`
  // (mla_attention.py:387) — 576 for every DeepSeek variant.
  int64_t head_size() const { return kv_lora_rank + qk_rope_head_dim; }
  // `q_lora_rank is not None` (deepseek_v2.py:1003).
  bool has_q_lora() const { return q_lora_rank > 0; }
  void Validate() const;
};

// ─── load-time weight absorption ────────────────────────────────────────────
// `MLAAttention.process_weights_after_loading` (mla_attention.py:875-962),
// bf16 in / bf16 out. Upstream keeps these as plain fp16/bf16 copies rather than
// quantizing them, and says why at `:876-878` ("we currently do not have
// quantized bmm's which are needed for W_UV and W_UK_T ... the extra memory
// overhead of this is fairly low"); we mirror that exactly.
//
// `kv_b_proj_weight` is the CHECKPOINT-layout row-major
// [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank] linear weight
// (deepseek_v2.py:518-519, i.e. torch's [out_features, in_features]); upstream
// transposes it at `:880` before the view/split, which this function folds into
// the index arithmetic.
struct AbsorbedKvBProj {
  std::vector<uint16_t> w_uk_t;  // bf16 [num_heads, qk_nope_head_dim, kv_lora_rank]
  std::vector<uint16_t> w_uv;    // bf16 [num_heads, kv_lora_rank, v_head_dim]
};
AbsorbedKvBProj AbsorbKvBProjBf16(const uint16_t* kv_b_proj_weight, const MlaBlockDims& dims);

// ─── decoupled YaRN RoPE ────────────────────────────────────────────────────
// `yarn_get_mscale` (deepseek_scaling_rope.py:20-23):
//     scale <= 1 -> 1.0 ; else 0.1 * mscale * log(scale) + 1.0
double YarnGetMscale(double scale, double mscale);

// The `rope_scaling` block plus the base rope parameters, as
// `DeepseekScalingRotaryEmbedding.__init__` consumes them
// (deepseek_scaling_rope.py:31-74). DeepSeek-V2-Lite (W0-confirmed):
// base 10000, factor 40, mscale 0.707, mscale_all_dim 0.707, beta_fast 32,
// beta_slow 1, original_max_position_embeddings 4096, rotary_dim 64.
struct DeepseekYarnRopeParams {
  double base = 10000.0;
  // `rope_scaling["factor"]`; 1.0 (or `yarn == false`) selects plain RoPE.
  double scaling_factor = 1.0;
  double extrapolation_factor = 1.0;
  double attn_factor = 1.0;
  double beta_fast = 32.0;
  double beta_slow = 1.0;
  double mscale = 1.0;
  double mscale_all_dim = 0.0;
  int64_t rotary_dim = 0;  // == qk_rope_head_dim
  // `original_max_position_embeddings` — the PRE-scaling context length the
  // correction range is computed against (deepseek_scaling_rope.py:95-101).
  int64_t original_max_position_embeddings = 4096;
  // `rope_parameters["rope_type"] != "default"` (deepseek_v2.py:1053-1058).
  bool yarn = true;
};

// `_compute_cos_sin_cache` (deepseek_scaling_rope.py:105-118): the cache rows
// are `[cos(rotary_dim/2) | sin(rotary_dim/2)]`, each scaled by the ROTATION
// mscale `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor,
// mscale_all_dim) * attn_factor` (`:55-59`) — note this is a DIFFERENT mscale
// from the softmax-scale correction below (that one is the SQUARE of
// `yarn_get_mscale(factor, mscale_all_dim)`; conflating them is the classic
// DeepSeek RoPE bug). Returns `rows * rotary_dim` f32 values in the [cos|sin]
// layout vt::RopeFromCache expects. All intermediate math is double, mirroring
// torch's float64-free but f32-accumulated construction closely enough that the
// f32 result is bit-stable; see the gate for the independent-oracle check.
std::vector<float> BuildDeepseekRopeCosSinCache(const DeepseekYarnRopeParams& p, int64_t rows);

// `self.scaling = qk_head_dim ** -0.5` (deepseek_v2.py:995) then
// `self.scaling = self.scaling * mscale * mscale` where
// `mscale = yarn_get_mscale(factor, mscale_all_dim)` (:1067-1075). For
// DeepSeek-V2-Lite: 192**-0.5 * (0.1*0.707*ln(40)+1)^2 = 192**-0.5 * 1.5897.
float MlaAttentionScale(const MlaBlockDims& dims, const DeepseekYarnRopeParams& p);

// ─── the block's device-resident weights ────────────────────────────────────
// Every tensor is bf16 and device-resident. EXACTLY ONE query branch is
// populated (see `MlaBlockDims::has_q_lora`); the other's tensors stay empty.
struct MlaBlockWeights {
  // (a) `q_lora_rank is not None` — deepseek_v2.py:1003-1026. NOTE that there is
  //     no standalone `q_a_proj` module upstream: q_a is FUSED with kv_a into
  //     `fused_qkv_a_proj` (:905-950 DeepSeekV2FusedQkvAProjLinear), and
  //     :1812-1820 registers `packed_modules_mapping["fused_qkv_a_proj"] =
  //     ["q_a_proj", "kv_a_proj_with_mqa"]`, so a loader must FUSE the two
  //     checkpoint tensors into this one weight. Row blocks, in order:
  //       [0, q_lora_rank)                                   q_a_proj
  //       [q_lora_rank, q_lora_rank + kv_lora_rank)          kv_a (latent)
  //       [.. + kv_lora_rank, .. + kv_lora_rank + qk_rope)   kv_a (rope part)
  vt::Tensor fused_qkv_a_proj;  // [q_lora_rank + kv_lora_rank + qk_rope, hidden]
  vt::Tensor q_a_layernorm;     // [q_lora_rank]                 (:1019)
  vt::Tensor q_b_proj;          // [num_heads*qk_head_dim, q_lora_rank]  (:1020-1026)
  // (b) `q_lora_rank is None` — deepseek_v2.py:1010-1016, :1028-1034.
  vt::Tensor kv_a_proj_with_mqa;  // [kv_lora_rank + qk_rope, hidden]
  vt::Tensor q_proj;              // [num_heads*qk_head_dim, hidden]
  // shared
  vt::Tensor kv_a_layernorm;  // [kv_lora_rank]  — the ROPE PART IS NOT NORMED (:516)
  // The UNABSORBED up-projection, used by the PREFILL (materialized-MHA) path
  // (mla_attention.py:2371-2375) and by the chunked-context callback W5 left
  // open. [num_heads*(qk_nope_head_dim + v_head_dim), kv_lora_rank]  (:518-519)
  vt::Tensor kv_b_proj;
  // The ABSORBED forms, used by the DECODE (MQA) path. Produced by
  // AbsorbKvBProjBf16 from the same `kv_b_proj` weight.
  vt::Tensor w_uk_t;  // [num_heads, qk_nope_head_dim, kv_lora_rank]  (:962)
  vt::Tensor w_uv;    // [num_heads, kv_lora_rank, v_head_dim]        (:960)
  vt::Tensor o_proj;  // [hidden, num_heads*v_head_dim]               (:526)
  // BuildDeepseekRopeCosSinCache uploaded in the BLOCK's dtype (bf16 for a bf16
  // forward), [rows, qk_rope_head_dim].
  vt::Tensor rope_cos_sin_cache;
};

// Per-step metadata. `num_decode_tokens` is upstream's `num_mqa_tokens`
// (mla_attention.py:707) and the tokens it names are PACKED FIRST.
struct MlaBlockMetadata {
  int64_t num_decode_tokens = 0;
  // The decode half's device tensors — `attn_metadata.decode.block_table` /
  // `.seq_lens` plus the host `max_seq_len` that sizes the split heuristic
  // (triton_mla.py:214-216, 245-246).
  v1::MLACommonMetadata decode{};

  // The prefill half. `cu_seqlens_q` offsets are RELATIVE to the prefill
  // sub-batch (i.e. token `num_decode_tokens` is offset 0), mirroring the fact
  // that upstream slices `q[num_mqa_tokens:]` before calling forward_mha.
  vt::Tensor prefill_cu_seqlens_q;  // [num_prefills+1] i32, device
  vt::Tensor prefill_block_table;   // [num_prefills, max_blocks] i32, device
  // Empty => no prefill request has context, so the causal new-tokens result IS
  // the answer (mla_attention.py:2421-2425).
  std::vector<MlaChunkDeviceMetadata> chunks;
  int32_t max_query_len = 0;
  int32_t prefill_tokens_with_context = 0;  // (:1806-1810)
  // Rows of chunked-prefill workspace to allocate; must be >= the largest
  // chunk's `total_tokens`. DetermineChunkedPrefillWorkspaceSize sizes it.
  int64_t chunk_workspace_tokens = 0;
};

// The whole layer: projections -> two RMSNorms -> decoupled RoPE -> the MLA
// cache write -> the prefill-MHA / decode-MQA dispatch -> o_proj.
//
//   hidden        [T, hidden_size]  bf16 — the INPUT-NORMED hidden state
//                 (upstream's decoder layer applies input_layernorm before
//                  calling the attention module)
//   positions     [T] i32
//   kv_cache      [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
//   slot_mapping  [T] i64
//   out           [T, hidden_size] bf16 — the o_proj output
//
// `impl` is the decode backend (TritonMLAImpl on GB10); its `queue` is set to
// `d.q` for the call, which is the wiring W4's recorded deviation (i) deferred.
void ForwardMlaAttentionBlock(dense_attn::Dev d, const MlaBlockDims& dims,
                              const MlaBlockWeights& w, const vt::Tensor& hidden,
                              const vt::Tensor& positions, vt::Tensor& kv_cache,
                              const vt::Tensor& slot_mapping, const MlaBlockMetadata& meta,
                              v1::TritonMLAImpl& impl, vt::Tensor& out);

// The `kv_b_proj` up-projection callback W5 left OPEN (`MlaUpProjectFn`,
// mla_chunked_context.h:228). Binds the model's `kv_b_proj` weight and the block
// dims into the closure `_compute_prefill_context` calls per chunk
// (mla_attention.py:2141-2170): workspace rows -> per-head K/V. Exposed so the
// chunked loop can be driven directly in tests and by W7.
//
// `scratch` must outlive every call: the returned MlaContextChunkKv holds views
// into it.
struct MlaUpProjectScratch {
  std::vector<dense_attn::DBuf> bufs;
};
MlaUpProjectFn MakeMlaUpProjectFn(dense_attn::Dev d, const MlaBlockDims& dims,
                                  const MlaBlockWeights& w, MlaUpProjectScratch& scratch);

}  // namespace mla
}  // namespace vllm
