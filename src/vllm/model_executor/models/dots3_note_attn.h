// dots3-note (`Dots3NoteForCausalLM`) — W3: the FULL-attention layer.
// Issue #699, spec `.agents/specs/dots3-note.md` (§2.2, §4 and §4.4 are the
// load-bearing parts). Indexer-slice obligation: #1846.
//
// ─── WHAT THIS IS, AND WHAT IT IS NOT ────────────────────────────────────────
// This is a PORTABLE HOST (CPU) reference forward of `_forward_note_mla`'s
// full-attention arm, computed in DOUBLE internally. It is NOT the device
// forward: `Dots3NoteModel::ForwardDevice` still refuses by name, because the
// full layers need the PADDED SPARSE MLA backend (`nvidia/attention.py`
// ::Dots3NotePaddedSparseImpl) over a heterogeneous KV cache, and that backend
// is W4's. The device integration is a NAMED follow-on, not an omission.
//
// The precedent for that shape is in this tree and is deliberate:
// `deepseek_v4_dsa.{h,cpp}` landed the DSA primitives the same way — portable
// host reference, unit-gated against a from-first-principles double-precision
// reference, device kernel a later brick — and recorded why. Two AGENTS.md
// rules bear on it and both are answered here rather than silently:
//
//   * "Route decode through ModelRegistry::Forward / dense_attn::AttnBlock."
//     This file is not on the decode path AT ALL; nothing routes around a seam
//     because nothing routes. `Dots3NoteFullAttnDimsFrom` takes the params the
//     REAL registry loader produces, so the geometry this reference runs is the
//     released `config.json`'s and not a hand-typed struct.
//   * "If a shared seam cannot represent the upstream behavior, extend it."
//     `mla::ForwardMlaAttentionBlock` (mla_attention.h:278) cannot represent
//     three of the four dots3 deltas — the two LoRA rescales and
//     `k_rope_only_layernorm` sit BETWEEN its projections and its RoPE, and the
//     headwise gate sits between its attention and its `o_proj`. Extending it
//     is the right move when the DEVICE layer lands (it is two `double`s, one
//     optional norm weight and one optional gate weight on `MlaBlockDims` /
//     `MlaBlockWeights`), and doing it here — where no device forward would
//     exercise it — would put untested optional branches into the SACRED
//     DeepSeek-V2 path for no gate in return. W4 owes that extension.
//
// The indexer's SELECTION math is NOT re-implemented here: it is
// `deepseek_v4::DsaIndexerWeightFold` / `DsaIndexerLogits` / `DsaTopkSelect`
// (deepseek_v4_dsa.h), which are ports of the SAME upstream files dots3-note
// reaches — `layers/sparse_attn_indexer.py` and
// `v1/attention/ops/triton_fp8_mqa_logits.py`, through the shared
// `deepseek_v2.py::Indexer`. dots3-note's indexer delta is not the math, it is
// the RoPE GEOMETRY (§4 trap 2 and #1846), which is exactly what this file adds.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// BEYOND-PIN. `dots3_note` does NOT exist at our parity pin `555967922`
// (0.26.0.dev0). Every anchor below was read at upstream `origin/main` =
// `06ecec7a84` (2026-08-25). W2 read the same files at `185cada36b`;
// `git log 185cada36b..origin/main -- vllm/models/dots3_note/` is EMPTY, so the
// two revisions carry byte-identical dots3 sources and the anchors below
// supersede §2.2's W0-era line numbers rather than contradicting them (spec R2).
//
//   OURS                          <-  UPSTREAM
//   ForwardFullAttention          <-  `vllm/models/dots3_note/nvidia/model.py`
//                                     ::_forward_note_mla (:135-201), on the
//                                     `Dots3NoteFullAttention` arm (:219-326)
//   ..lora rescales               <-  model.py:155 (`q_c`), :159 (`kv_c_normed`);
//                                     the scalars at :305-307
//   ..k_rope_only_layernorm       <-  model.py:160 (applied), :299-301 (built,
//                                     RMSNorm over qk_rope_head_dim, rms_norm_eps)
//   ..headwise gate               <-  model.py:190-197 — `g_proj`, sigmoid in
//                                     FP32 then cast back, per-head multiply
//   ..indexer runs only if sparse <-  model.py:171 (`attention.is_sparse`);
//                                     the sliding class sets it False (:430-432)
//   RmsNorm                       <-  `vllm/ir/ops/layernorm.py`:10-21
//                                     (fp32 variance, cast back at the end)
//   LayerNorm (indexer k_norm)    <-  torch.nn.LayerNorm, built at
//                                     `deepseek_v2.py`::Indexer.__init__ :708
//                                     (`LayerNorm(head_dim, eps=1e-6)`)
//   RopeCosSinCache               <-  `layers/rotary_embedding/base.py`:80-103
//                                     (`_compute_inv_freq`/`_compute_cos_sin_cache`)
//   ApplyRopeInPlace              <-  `layers/rotary_embedding/common.py`:169-181
//                                     (`ApplyRotaryEmb.forward_static`) driven by
//                                     `base.py`:178-201 (`forward_static`)
//   MLA core (materialized MHA)   <-  `deepseek_v2.py`:1027 (`scaling =
//                                     qk_head_dim ** -0.5`; dots3 forces
//                                     `rope_type="default"` at model.py:230-238
//                                     so NO YaRN mscale applies) +
//                                     `layers/attention/mla_attention.py`
//                                     (the unabsorbed kv_b_proj up-projection)
//   IndexerRopeOffset             <-  `deepseek_v2.py`::Indexer.forward
//                                     :804-805 / :813-814 — `torch.split(q,
//                                     [rope_dim, head_dim - rope_dim])`, i.e. the
//                                     rotated half is the LEADING 64 of the
//                                     128-wide index head. #1846.
//
// ─── THE FOUR SILENT DELTAS, AND WHY EACH IS ITS OWN NAMED STEP ──────────────
// This row has NO oracle on any hardware we own (spec §6.2/§6.4 option B), so
// there is no token gate downstream and every one of these is numerically
// silent when wrong: the model still emits plausible text. Each therefore gets
// its own exported entry point and its own RED-first assertion in
// `tests/vllm/models/test_dots3_note_attn.cpp`, against a reference transcribed
// from the python rather than from this file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dots3_note.h"

namespace vllm::dots3_note {

// ── primitives ───────────────────────────────────────────────────────────────

// `vllm/ir/ops/layernorm.py`:10-21. Row-wise RMS norm over the LAST dim:
//   variance = mean(x^2); x * rsqrt(variance + eps) * weight
// Upstream computes the variance in fp32 and casts back at the end; this
// reference is double throughout, which is strictly wider, and the memory
// format the DEVICE layer owes is bf16 in / bf16 out (porting.md).
// `x` is [rows, cols] row-major; `weight` is [cols].
std::vector<double> RmsNorm(const std::vector<double>& x,
                            const std::vector<double>& weight, int64_t rows,
                            int64_t cols, double eps);

// torch.nn.LayerNorm — the DSA indexer's `k_norm`
// (`deepseek_v2.py`::Indexer.__init__ :708, `LayerNorm(head_dim, eps=1e-6)`).
// This is NOT an RMSNorm: it subtracts the mean and adds a bias, and the
// released checkpoint ships BOTH `indexer.k_norm.weight` AND
// `indexer.k_norm.bias` ([128] each), which is how we know.
//   y = (x - mean) / sqrt(var + eps) * weight + bias, var the BIASED variance.
std::vector<double> LayerNorm(const std::vector<double>& x,
                              const std::vector<double>& weight,
                              const std::vector<double>& bias, int64_t rows,
                              int64_t cols, double eps);

// `base.py`:80-103. Returns `rows * rotary_dim` doubles laid out as
// [cos(rotary_dim/2) | sin(rotary_dim/2)] per row, with
// inv_freq[j] = base^(-2j/rotary_dim).
std::vector<double> RopeCosSinCache(double base, int64_t rotary_dim,
                                    int64_t rows);

// `common.py`:169-181 driven by `base.py`:178-201. Rotates, IN PLACE, the
// `rotary_dim` lanes starting at `lane_offset` of every head of every token:
//
//   GPT-J  (is_neox_style == false): x1 = x[0::2], x2 = x[1::2],
//                                    out interleaved (o1,o2,o1,o2,...)
//   NeoX   (is_neox_style == true) : x1 = first half, x2 = second half,
//                                    out concatenated
//   o1 = x1*cos - x2*sin ; o2 = x2*cos + x1*sin
//
// `x` is [num_tokens, num_heads, head_stride] row-major. `lane_offset` is the
// WHOLE point of #1846 and §4 trap 6 being two different questions: the MLA
// rope rotates the TRAILING `qk_rope_head_dim` of a 192-wide head
// (`lane_offset = qk_nope_head_dim`, model.py:167-169) while the INDEXER
// rotates the LEADING 64 of a 128-wide head (`lane_offset = 0`).
void ApplyRopeInPlace(std::vector<double>& x,
                      const std::vector<int32_t>& positions,
                      const std::vector<double>& cos_sin_cache,
                      int64_t num_tokens, int64_t num_heads,
                      int64_t head_stride, int64_t lane_offset,
                      int64_t rotary_dim, bool is_neox_style);

// ── geometry ─────────────────────────────────────────────────────────────────

// One dots3-note FULL-attention layer's resolved geometry. Built from the
// params the REAL registry loader returns, never typed by hand.
struct FullAttnDims {
  int64_t hidden_size = 0;
  int64_t num_heads = 0;
  int64_t qk_nope_head_dim = 0;
  int64_t qk_rope_head_dim = 0;
  int64_t v_head_dim = 0;
  int64_t q_lora_rank = 0;
  int64_t kv_lora_rank = 0;
  double rms_norm_eps = 1e-5;
  // The MLA rope. dots3 rebuilds `rope_parameters` as
  // `{"rope_type": "default", "rope_theta": config.rope_parameters["rope_theta"]}`
  // (model.py:230-238), so the full layers get PLAIN RoPE at theta 8e7 with NO
  // YaRN and therefore NO mscale correction on the softmax scale.
  double rope_theta = 0.0;
  // FALSE on both dots3 geometries — §4 item 6, corrected at W1 (#1804).
  bool rope_is_neox_style = false;
  // §4 trap 5: sqrt(hidden_size / rank), applied AFTER the respective layernorm.
  double q_lora_scale = 1.0;
  double kv_lora_scale = 1.0;
  // "headwise" on the released checkpoint. Upstream also has a non-headwise
  // arm (model.py:198-200) that multiplies the whole [num_heads*v_head_dim] row
  // lane-by-lane; `ForwardFullAttention` REFUSES it by name rather than porting
  // it, and the reason is a rule and not laziness — W1's config parse already
  // refuses every non-`headwise` value (dots3_note.cpp:376-381), so a ported
  // arm would be production code no input can reach. That is exactly the shape
  // W2's M12 finding deleted rather than kept (spec §4.4).
  std::string attention_gate_type;

  // DSA lightning indexer (full layers only).
  int64_t index_n_heads = 0;
  int64_t index_head_dim = 0;
  int64_t index_topk = 0;
  // §4 trap 2: `is_neox_style = not indexer_rope_interleave`
  // (`deepseek_v2.py`:1159). dots3-note sets `indexer_rope_interleave = True`,
  // so this is FALSE and the indexer rope agrees with the MLA rope. At
  // DeepSeek-V3.2's absent-key default it would be TRUE and the indexer would
  // rotate split-half beside a GPT-J MLA rope.
  bool indexer_rope_is_neox_style = false;
  // torch.nn.LayerNorm's own default, hard-coded at `deepseek_v2.py`:708.
  double indexer_k_norm_eps = 1e-6;

  int64_t qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // `deepseek_v2.py`:1027. No YaRN => no mscale^2 factor.
  double softmax_scale() const;
  void Validate() const;
};

// Resolve the FULL-attention geometry out of the params the registry produced.
// REFUSES BY NAME when the schedule contains no `full_attention` layer at all,
// so a caller cannot quietly obtain a plausible struct for a model whose full
// arm is never used; and it REFUSES a params object whose full arm carries no
// indexer, because `is_sparse` is what makes this the full arm (model.py:171).
FullAttnDims Dots3NoteFullAttnDimsFrom(const Dots3NoteParams& params);

// #1846. The lane at which the indexer's 64 rotated lanes START inside the
// 128-wide index head. The released `model.safetensors.index.json` metadata
// declares `indexer_rope_layout: "leading"` / `indexer_rope_converted_from:
// "tail"`, and NO upstream code reads either key — it is the publisher stating
// how `wq_b` and `wk` are laid out. Upstream's code agrees anyway:
// `deepseek_v2.py`::Indexer.forward splits `[..., :rope_dim]` off the FRONT
// (:804-805, :813-814). So the answer is 0, and W3 ASSERTS it rather than
// inheriting it: a `tail` layout would be `index_head_dim - qk_rope_head_dim`
// = 64, is numerically silent, and this row has no oracle that could catch it.
int64_t IndexerRopeOffset(const FullAttnDims& dims);

// ── weights ──────────────────────────────────────────────────────────────────

// One full-attention layer's host weights, by the names the checkpoint SHIPS
// (`q_a_proj` and `kv_a_proj_with_mqa` SEPARATE — upstream fuses them into
// `fused_qkv_a_proj` at load time, so the module view hides the two real
// tensors). Every matrix is row-major [out_features, in_features], which is
// torch's own `nn.Linear.weight` layout and therefore the checkpoint's; the
// shapes are the released `dots-studio/dots3-note-prev` ones, read off the
// committed shard-index fixture rather than inferred.
struct FullAttnWeights {
  std::vector<double> q_a_proj;               // [q_lora_rank, hidden]
  std::vector<double> kv_a_proj_with_mqa;     // [kv_lora_rank + qk_rope, hidden]
  std::vector<double> q_a_layernorm;          // [q_lora_rank]
  std::vector<double> kv_a_layernorm;         // [kv_lora_rank]
  std::vector<double> k_rope_only_layernorm;  // [qk_rope]      <- dots3 ONLY
  std::vector<double> q_b_proj;               // [heads*qk_head_dim, q_lora_rank]
  std::vector<double> kv_b_proj;              // [heads*(qk_nope+v), kv_lora_rank]
  std::vector<double> o_proj;                 // [hidden, heads*v]
  std::vector<double> g_proj;                 // [heads, hidden] <- dots3 ONLY
  // DSA indexer, all five tensors the checkpoint ships.
  std::vector<double> indexer_wq_b;           // [index_n_heads*index_head_dim, q_lora_rank]
  std::vector<double> indexer_wk;             // [index_head_dim, hidden]
  std::vector<double> indexer_weights_proj;   // [index_n_heads, hidden]
  std::vector<double> indexer_k_norm_weight;  // [index_head_dim]
  std::vector<double> indexer_k_norm_bias;    // [index_head_dim]

  // REFUSES BY NAME on the first tensor whose element count disagrees with
  // `dims`. A wrong-sized weight read as a silently truncated matrix is the
  // failure mode this row cannot afford.
  void Validate(const FullAttnDims& dims) const;
};

// ── the layer ────────────────────────────────────────────────────────────────

// The intermediates a gate needs to attribute a mismatch to ONE mechanism.
// Without these a single output difference says "something is wrong" and
// nothing about which of the four deltas produced it.
struct FullAttnTrace {
  std::vector<double> q_c;          // [T, q_lora_rank]  normed AND rescaled
  std::vector<double> kv_c_normed;  // [T, kv_lora_rank] normed AND rescaled
  std::vector<double> k_pe;         // [T, qk_rope] k_rope_only_layernorm'd THEN rotated
  std::vector<double> q;            // [T, heads, qk_head_dim] after RoPE
  std::vector<double> indexer_q;    // [T, index_n_heads, index_head_dim] after RoPE
  std::vector<double> indexer_k;    // [T, index_head_dim] k_norm'd then rotated
  std::vector<double> indexer_logits;  // [T, T]
  std::vector<int64_t> topk;        // [T, index_topk], -1 padded
  std::vector<double> attn_out;     // [T, heads*v]  BEFORE the gate
  std::vector<double> gate;         // [T, heads]    AFTER the sigmoid
  std::vector<double> gated;        // [T, heads*v]  AFTER the gate, before o_proj
};

// `_forward_note_mla` on the FULL-attention arm (model.py:135-201 with
// `attention.is_sparse == True`), as a whole-sequence PREFILL over T tokens
// with no KV cache: key `s` is token `s`, and the causal mask is `s <= t`.
//
//   hidden     [T, hidden_size]   the INPUT-LAYERNORM'd hidden state, which is
//                                 what the decoder layer hands the attention
//                                 module and what `g_proj` and the indexer's
//                                 `wk`/`weights_proj` also read (model.py:190,
//                                 :172 -> Indexer.forward's `hidden_states`)
//   positions  [T]                i32, as upstream's `positions`
//   out        [T, hidden_size]   the o_proj output
//
// `trace` may be null.
std::vector<double> ForwardFullAttention(const FullAttnDims& dims,
                                         const FullAttnWeights& w,
                                         const std::vector<double>& hidden,
                                         const std::vector<int32_t>& positions,
                                         int64_t num_tokens,
                                         FullAttnTrace* trace);

// The headwise gate on its own (model.py:191-197), exported because it is the
// one delta whose DTYPE is part of the specification: upstream computes the
// sigmoid in FP32 and casts back to the activation dtype, and mirroring that is
// a `porting.md` memory-format obligation rather than an accident.
//   attn_out [T, heads*v_head_dim], gate_logits [T, heads] -> [T, heads*v]
std::vector<double> ApplyHeadwiseGate(const std::vector<double>& attn_out,
                                      const std::vector<double>& gate_logits,
                                      int64_t num_tokens, int64_t num_heads,
                                      int64_t v_head_dim);

}  // namespace vllm::dots3_note
