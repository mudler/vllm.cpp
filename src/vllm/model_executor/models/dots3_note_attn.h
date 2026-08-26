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

// ═════════════════════════════════════════════════════════════════════════════
// W4b-1 — THE SLIDING-WINDOW ARM, AND THE PADDED KV ROW BOTH CLASSES SHARE
// ═════════════════════════════════════════════════════════════════════════════
//
// ─── WHY THIS IS W4b-1 AND NOT W4b ───────────────────────────────────────────
// The spec's W4b is "the §2.3 stack: windowed metadata, the KV gather, the
// score mask, and the padded/heterogeneous KV spec" PLUS three device refusals
// W4a handed on. That is two bricks, split on exactly the line W3/W4a already
// used on this row:
//
//   W4b-1 (here)  the SEMANTICS, as portable host code gated against an
//                 independent double-precision reference. Nothing runs on the
//                 decode path; `Dots3NoteModel::ForwardDevice` still refuses a
//                 sliding layer and a padded physical row BY NAME.
//   W4b-2 (owed)  those semantics ON the decode path: a `vt` MLA cache whose
//                 PHYSICAL row is wider than the row a layer reads, and a
//                 windowed decode/prefill through the shared MLA seam.
//
// TWO DIFFERENT REASONS HOLD THE TWO HALVES APART, and an earlier draft of this
// comment gave one WRONG reason for both. It said `vt::ConcatAndCacheMla`,
// `vt::MlaDecodeAttention` and the MLA prefill gather address the cache as
// CONTIGUOUS, so a padded row would be a kernel change on both backends. **That
// is false, and it was refuted by execution rather than by argument** (the
// fresh review of #1949). All three source `stride[0]` and `stride[1]` FROM THE
// TENSOR — `cpu_cache.cpp:99-100`, `cpu_mla_attn.cpp:99`,
// `cpu_mla_prefill.cpp:180` — and `Tensor::Slice(2, 0, logical)` shrinks
// `shape[2]` while KEEPING both strides (`tensor.cpp:80-84`), which is exactly
// upstream's `kv_cache[..., : self.head_size]`. The tree already gates it:
// `tests/vt/test_ops_mla_cache.cpp:259` is
// `TEST_CASE("concat_and_cache_mla is STRIDE-driven (cache view + split
// sources)")`, with CUDA-vs-CPU strided parity at `:403`. A scratch probe at
// physical row 7 / logical row 5 wrote, gathered and DECODED through the
// narrowed view with the pad lanes asserted untouched: compiler exit 0, binary
// exit 0, 30/30 assertions, and ZERO changes to any `vt` op. Leaving the false
// claim here would have told a W4b-2 implementer that the shared seam cannot
// express a padded row, which licenses either editing three ops on two backends
// or hand-rolling the parallel path AGENTS.md forbids.
//
//   * THE PADDED ROW IS DEFERRED BY SCOPE CHOICE, NOT BY CONSTRAINT. It is
//     expressible today through `Tensor::Slice` and it is CPU-gateable. The one
//     contiguous construction is a single model-level line,
//     `dots3_note_device.cpp:470`. It is deferred because a padded row with no
//     windowed attention to read it is half a capability, not a shipped one.
//   * THE WINDOW IS A REAL CONSTRAINT, and it is what actually holds W4b-2
//     apart. `vt::MlaDecodeAttention` attends over the WHOLE sequence —
//     `for (int64_t j = 0; j < seq_len; ++j)` at `cpu_mla_attn.cpp:94` — with no
//     window bound and no per-slot `valid`, and neither
//     `MlaDecodeAttentionArgs` nor `MlaPrefillAttentionArgs` carries a window
//     field at all. A windowed decode and prefill is therefore a NEW KERNEL on
//     both backends, including a CUDA half no CPU-only box can verify, and it
//     owes the seam byte-identity W4a produced for the four callers of
//     `mla::ForwardMlaAttentionBlock`.
//
// W4b-2 owns both; see `## Owed`.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// BEYOND-PIN, exactly as W3/W4a. Every anchor below was RE-DERIVED at upstream
// `origin/main` = `d9fbe526c0` (2026-08-25). `git diff 06ecec7a84 origin/main
// -- vllm/models/dots3_note/` is EMPTY, so these files are byte-identical to
// what W3 and W4a read and the numbers below supersede §2.3's W0-era ones
// rather than contradicting them. §2.3's own anchors are ALL stale by one line
// (it cited the `@triton.jit` / `@dataclass` DECORATOR, not the `def` / `class`
// it decorates) except the `Dots3NotePaddedSparseImpl` family, which is stale
// by EIGHT — `:689` is the `Dots3NotePaddedSparseBackend` docstring and `:692`
// is `supported_kv_cache_layouts`, both plausible code a reader would accept.
// The spec's §2.3 list is corrected in place (spec R2, and `#1139`: the anchor
// checker's own docstring says it does not verify line citations).
//
//   OURS                        <-  UPSTREAM (`vllm/models/dots3_note/nvidia/`)
//   SlidingAttnDims             <-  model.py::Dots3NoteSlidingAttention.__init__
//                                   (:329-460): the geometry at :340-346, the
//                                   rope at :401-409 (theta `swa_rope_theta`
//                                   :406, `is_neox_style=False` :408), the two
//                                   LoRA scalars at :438-443, the softmax scale
//                                   `qk_head_dim**-0.5` at :446 and the window
//                                   `sliding_window_size` at :457
//   ..no indexer                <-  model.py:432-434 (`self.indexer = None`,
//                                   `self.is_sparse = False`) — the ONE
//                                   structural difference from the full arm
//   SwaGatherLen                <-  attention.py:484 (and :321, the workspace
//                                   reservation, which must agree with it)
//   GatherSwaKv                 <-  attention.py::_gather_swa_kv_kernel
//                                   (:49-114, decorated at :48)
//   ApplySwaScoreMask           <-  attention.py::_apply_swa_score_mask_kernel
//                                   (:119-163, decorated at :118)
//   BuildSlidingWindowMetadata  <-  attention.py::_build_sliding_window_metadata
//                                   (:192-254) + `_SlidingWindowChunk` (:172)
//   PaddedMlaCacheSpec          <-  model.py::Dots3NotePaddedMLAAttention
//                                   (:204-216) — `get_kv_cache_spec` reports
//                                   `physical_head_size` at :216, and
//                                   `Dots3NoteFullAttention` passes
//                                   `swa_kv_lora_rank + swa_qk_rope_head_dim`
//                                   at :283
//   NarrowLogicalCacheRow       <-  attention.py::Dots3NotePaddedSparseImpl
//                                   ::_logical_cache (:700-702)
//   ForwardSlidingAttention     <-  model.py::_forward_note_mla (:135-201) on
//                                   the `Dots3NoteSlidingAttention` arm
//                                   (:462-475), with the MLA attention itself
//                                   being attention.py::Dots3NoteTritonMLAImpl
//                                   ::_forward_swa_mqa (:470-563)
//
// ─── WHAT IS DELIBERATELY NOT PORTED HERE ────────────────────────────────────
//   * The Triton kernels' LAUNCH geometry (BLOCK_T, BLOCK_D, num_warps,
//     `triton.next_power_of_2`). Those are tiling choices, not semantics; the
//     answer they compute is what this file carries.
//   * `Dots3NoteFlashAttnPrefillBackend.run_sliding_window` (:279-306). Its
//     window is FlashAttention's `window_size=(sliding_window - 1, 0)` over a
//     causal varlen batch, which is the SAME predicate `ApplySwaScoreMask`
//     applies — one of the two mechanisms this file gates, reached the other
//     way. The FA-3 call itself is a kernel binding, and W4b-2 owns it.
//   * `Dots3NoteMLAMetadataBuilder.build` (:353-417) beyond the chunk plan:
//     the rest is vLLM engine plumbing (`dcp_world_size`, workspace managers,
//     `prepare_metadata`) with no counterpart in this tree yet.
//   * The fp8 KV-cache rescale inside the gather (:98-99). Our MLA cache write
//     already refuses `fp8_ds_mla`; the branch would be unreachable code, which
//     is the shape spec §4.4's M12 finding deleted rather than kept.

// ── the sliding geometry ─────────────────────────────────────────────────────

// One dots3-note SLIDING-attention layer's resolved geometry. It is NOT a
// parameterisation of `FullAttnDims`: the head count, BOTH LoRA ranks, the NoPE
// width, the rope theta and the presence of an indexer all differ, so the two
// are separate structs on purpose. Built from the params the REAL registry
// loader produces, never typed by hand.
struct SlidingAttnDims {
  int64_t hidden_size = 0;
  int64_t num_heads = 0;         // `swa_num_attention_heads`   — 64  (128 full)
  int64_t qk_nope_head_dim = 0;  // `swa_qk_nope_head_dim`      — 192 (128 full)
  int64_t qk_rope_head_dim = 0;  // `swa_qk_rope_head_dim`      — 64  (same)
  int64_t v_head_dim = 0;        // `swa_v_head_dim`            — 128 (same)
  int64_t q_lora_rank = 0;       // `swa_q_lora_rank`           — 1024 (same)
  int64_t kv_lora_rank = 0;      // `swa_kv_lora_rank`          — 1024 (512 full)
  double rms_norm_eps = 1e-5;
  // `swa_rope_theta` (model.py:406) — 5e4, against the full layers' 8e7. Three
  // orders of magnitude apart, on 33 of the 46 layers, and numerically silent.
  double rope_theta = 0.0;
  // FALSE, passed LITERALLY at model.py:408 rather than inherited. §4 item 6.
  bool rope_is_neox_style = false;
  // §4 trap 5 at the SLIDING ranks: `sqrt(hidden/swa_q_lora_rank)` and
  // `sqrt(hidden/swa_kv_lora_rank)` (model.py:438-443). On the released model
  // both ranks are 1024, so the two scales are EQUAL there and DIFFER on the
  // full arm — which is why a fixture must not make them equal by accident.
  double q_lora_scale = 1.0;
  double kv_lora_scale = 1.0;
  std::string attention_gate_type;  // "headwise"
  // `sliding_window_size` (model.py:457) — 513. INCLUSIVE of the query's own
  // position: the mask keeps `kv_pos >= query_pos - WINDOW_SIZE + 1`
  // (attention.py:152), so a window of W admits exactly W keys.
  int64_t sliding_window = 0;
  // The PHYSICAL MLA cache row every layer of the model allocates, full and
  // sliding alike (model.py:283 -> :216). Equal to this arm's own `latent_row()`
  // by construction — the padding exists for the FULL layers, and it is carried
  // here so one struct can state the whole cache contract.
  int64_t physical_latent_row = 0;

  int64_t qk_head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
  // The LOGICAL MLA cache row this arm reads: `kv_lora_rank + qk_rope_head_dim`
  // = 1088. `Dots3NoteTritonMLABackend.get_supported_head_sizes` returns exactly
  // `[1088]` (attention.py:422-424), which is upstream pinning the same number.
  int64_t latent_row() const { return kv_lora_rank + qk_rope_head_dim; }
  // `scale=qk_head_dim**-0.5` (model.py:446). 256**-0.5 == 1/16 on the released
  // model, against the full arm's 192**-0.5 — and NO YaRN mscale on either,
  // because both rebuild `rope_parameters` with `rope_type="default"`.
  double softmax_scale() const;
  void Validate() const;
};

// Resolve the SLIDING geometry out of the params the registry produced.
// REFUSES BY NAME when the schedule contains no `sliding_attention` layer, and
// REFUSES a params object whose sliding arm reports an indexer — `is_sparse ==
// False` is what makes this the sliding arm (model.py:432-434), and a sliding
// layer that ran the DSA indexer would prune a window that is already 513 wide.
SlidingAttnDims Dots3NoteSlidingAttnDimsFrom(const Dots3NoteParams& params);

// ── §2.3, mechanism by mechanism ─────────────────────────────────────────────

// `gather_len = (sliding_window + query_len - 1 + 7) // 8 * 8`
// (attention.py:484, and :321 where the workspace is reserved at the same
// formula with `max_query_len`). The two MUST agree or the decode gathers into
// a workspace sized for a different window; they are one function here so they
// cannot drift.
//
// The `+ query_len - 1` is the reason this is not just the window: a batch
// whose requests each carry `query_len` queries needs the union of `query_len`
// windows, which spans `sliding_window + query_len - 1` positions. The round up
// to 8 is the kernel's `BLOCK_T`, and it is part of the CONTRACT rather than a
// tiling detail, because the mask kernel is handed the SAME `GATHER_LEN` and
// masks the tail itself.
int64_t SwaGatherLen(int64_t sliding_window, int64_t query_len);

// One chunk of `_build_sliding_window_metadata`'s plan (`_SlidingWindowChunk`,
// attention.py:172-184). Requests are packed into chunks whose gathered KV
// tokens fit one MLA prefill workspace; a request that cannot fit ALONE is a
// hard error rather than a silent truncation (attention.py:218-221).
struct SlidingWindowChunk {
  int64_t req_start = 0;    // inclusive, into the prefill sub-batch
  int64_t req_end = 0;      // exclusive
  int64_t query_start = 0;  // inclusive, into the prefill token stream
  int64_t query_end = 0;    // exclusive
  // `torch.cumsum` of the per-request query / KV lengths, both [n_reqs + 1]
  // with a leading 0 (attention.py:228-233).
  std::vector<int32_t> cu_seq_lens_q;
  std::vector<int32_t> cu_seq_lens_k;
  // `starts = seq_lens - kv_lens` (attention.py:207) — the FIRST cached
  // position each request's gather reads.
  std::vector<int32_t> starts;
  // `torch.repeat_interleave(arange(n_reqs), kv_lens)` (attention.py:234-236):
  // for each gathered KV token, which request it belongs to.
  std::vector<int32_t> token_to_seq;
  int64_t num_kv_tokens = 0;
  int64_t max_seq_len_q = 0;
  int64_t max_seq_len_k = 0;
};

// `_build_sliding_window_metadata` (attention.py:192-254), host-side and exact.
//   seq_lens         [n_reqs]     the TOTAL cached length of each request
//   query_start_loc  [n_reqs + 1] cumulative query lengths, leading 0
// `workspace_size` is the MLA chunked-prefill workspace in TOKENS.
//
// The load-bearing line is `kv_lens = min(seq_lens, query_lens + window - 1)`
// (:206): a request never gathers more than its own window union, however long
// its context is. That is what makes a 524288-position model affordable on a
// 513-wide layer, and getting it wrong is silent — too many rows still produce
// the same answer once the mask runs, at a cost nothing asserts.
std::vector<SlidingWindowChunk> BuildSlidingWindowMetadata(
    const std::vector<int32_t>& seq_lens,
    const std::vector<int32_t>& query_start_loc, int64_t sliding_window,
    int64_t workspace_size);

// `_gather_swa_kv_kernel` (attention.py:49-114). Gathers, per request, the LAST
// `gather_len` cached latent rows out of the PAGED cache into a dense
// [n_reqs, gather_len, kv_dim] workspace, and reports which of those slots are
// real.
//
//   cache        [num_blocks, page_size, physical_row]  row-major
//   block_table  [n_reqs, blocks_per_req] i32; a NEGATIVE entry is an unmapped
//                page and yields `valid == 0` (`physical_pages >= 0`, :86)
//   seq_lens     [n_reqs]
//
// `kv_dim` is the LOGICAL row this layer reads (`KV_DIM`, :527) and may be
// narrower than `physical_row`, which is the padding contract: the gather reads
// the head of each physical row and the stride stays physical. Reading at the
// LOGICAL stride instead is the exact defect `_logical_cache` exists to prevent
// and it is silent — every value is a real cached number, just the wrong one.
struct SwaGatherResult {
  std::vector<double> kv;   // [n_reqs, gather_len, kv_dim], zero where invalid
  std::vector<char> valid;  // [n_reqs, gather_len]
};
SwaGatherResult GatherSwaKv(const std::vector<double>& cache,
                            const std::vector<int32_t>& block_table,
                            const std::vector<int32_t>& seq_lens, int64_t n_reqs,
                            int64_t blocks_per_req, int64_t num_blocks,
                            int64_t page_size, int64_t physical_row,
                            int64_t kv_dim, int64_t gather_len);

// The value `_apply_swa_score_mask_kernel` stores into a masked slot
// (attention.py:161). It is `-FLT_MAX`, NOT `-inf`: upstream writes the f32
// literal, and `exp(-FLT_MAX - max)` underflows to exactly 0 while
// `exp(-inf - -inf)` is NaN. Mirroring the literal is what keeps an all-masked
// row's failure mode identical to upstream's.
extern const double kSwaMaskedScore;

// `_apply_swa_score_mask_kernel` (attention.py:119-163). Masks, IN PLACE, the
// scores of every gathered slot that is not in the query's window.
//
//   scores  [n_reqs, num_heads, query_len, gather_len]  — upstream's
//           `scores_by_head`, i.e. the [reqs, qlen, heads, gather] tensor
//           TRANSPOSED on dims 1/2 (attention.py:538-540)
//   valid   [n_reqs, gather_len] from the gather
//
// The predicate, transcribed from :148-154 and stated in ABSOLUTE positions:
//     gather_start   = max(seq_len - gather_len, 0)
//     kv_pos         = gather_start + slot
//     query_pos      = seq_len - query_len + query_idx
//     keep  <=>  slot < gather_len  AND  valid[req, slot]
//                AND kv_pos <= query_pos                       (causal)
//                AND kv_pos >= query_pos - window + 1          (the window)
//                AND query_pos >= 0
// Note what is NOT in it: the mask does not re-derive causality from the token
// ORDER, it derives it from the POSITIONS, which is what lets a decode batch
// carry `query_len` queries whose positions are the tail of a much longer
// sequence.
void ApplySwaScoreMask(std::vector<double>& scores,
                       const std::vector<int32_t>& seq_lens,
                       const std::vector<char>& valid, int64_t n_reqs,
                       int64_t num_heads, int64_t query_len, int64_t gather_len,
                       int64_t window_size);

// ── the padded / heterogeneous KV spec ───────────────────────────────────────

// `Dots3NotePaddedMLAAttention.get_kv_cache_spec` (model.py:213-216): ONE block
// shape for the whole model, whose row is the SLIDING arm's 1088, while a full
// layer reads only the leading 576 of it.
struct PaddedMlaCacheSpec {
  int64_t num_blocks = 0;
  int64_t page_size = 0;
  int64_t physical_row = 0;  // `physical_head_size` (model.py:283) — 1088
  int64_t logical_row = 0;   // this layer's own `kv_lora_rank + qk_rope` — 576/1088
  void Validate() const;
  int64_t slots() const { return num_blocks * page_size; }
};

// `MLAAttention.do_kv_cache_update` through
// `Dots3NotePaddedSparseImpl.do_kv_cache_update` (attention.py:704-720): the
// latent and the roped k_pe are written CONCATENATED into `[0, logical_row)` of
// the addressed physical row. Everything past `logical_row` is left untouched,
// which is what makes one cache serve two logical widths.
void WritePaddedMlaCache(std::vector<double>& cache,
                         const PaddedMlaCacheSpec& spec,
                         const std::vector<double>& kv_c_normed,
                         const std::vector<double>& k_pe, int64_t kv_lora_rank,
                         int64_t qk_rope_head_dim,
                         const std::vector<int64_t>& slot_mapping,
                         int64_t num_tokens);

// `Dots3NotePaddedSparseImpl._logical_cache` (attention.py:700-702) —
// `kv_cache[..., : self.head_size]` with the PHYSICAL row stride preserved.
// Returns the [slots, logical_row] dense narrowing.
//
// The defect this exists to prevent has no shape signature: a reader that keeps
// the LOGICAL stride still finds finite, plausible, previously-cached numbers,
// just from the wrong tokens. The gate's property is pad invariance — the same
// layer over a padded and an unpadded cache must agree EXACTLY.
std::vector<double> NarrowLogicalCacheRows(const std::vector<double>& cache,
                                           const PaddedMlaCacheSpec& spec);

// ── the sliding layer ────────────────────────────────────────────────────────

// One sliding layer's host weights, by the names the checkpoint SHIPS. The
// difference from `FullAttnWeights` is exactly the indexer's five tensors,
// which the sliding layers do not carry (spec §4.4, measured off the released
// shard index: `q_b_proj` is [16384, 1024] here and [24576, 1024] on a full
// layer, and no `self_attn.indexer.*` name exists at all).
struct SlidingAttnWeights {
  std::vector<double> q_a_proj;               // [q_lora_rank, hidden]
  std::vector<double> kv_a_proj_with_mqa;     // [kv_lora_rank + qk_rope, hidden]
  std::vector<double> q_a_layernorm;          // [q_lora_rank]
  std::vector<double> kv_a_layernorm;         // [kv_lora_rank]
  std::vector<double> k_rope_only_layernorm;  // [qk_rope]
  std::vector<double> q_b_proj;               // [heads*qk_head_dim, q_lora_rank]
  std::vector<double> kv_b_proj;              // [heads*(qk_nope+v), kv_lora_rank]
  std::vector<double> o_proj;                 // [hidden, heads*v]
  std::vector<double> g_proj;                 // [heads, hidden]
  void Validate(const SlidingAttnDims& dims) const;
};

// How the layer's ONE request is paged. The block table is explicit so a gate
// can SHUFFLE it: a contiguous 0,1,2,... table makes a paged read and a flat
// read the same answer, which would leave the block lookup unproven.
struct SlidingPaging {
  int64_t page_size = 0;
  std::vector<int32_t> block_table;  // [blocks_per_req], physical page ids
};

struct SlidingAttnTrace {
  std::vector<double> q_c;            // [T, q_lora_rank]  normed AND rescaled
  std::vector<double> kv_c_normed;    // [T, kv_lora_rank] normed AND rescaled
  std::vector<double> k_pe;           // [T, qk_rope] normed THEN rotated
  std::vector<double> q;              // [T, heads, qk_head_dim] after RoPE
  std::vector<double> q_absorbed;     // [T, heads, latent_row] after the W_UK fold
  std::vector<double> cache;          // [num_blocks*page_size, physical_row]
  std::vector<double> gathered;       // [1, gather_len, latent_row]
  std::vector<char> gather_valid;     // [1, gather_len]
  std::vector<double> masked_scores;  // [1, heads, T, gather_len]
  std::vector<double> attn_out;       // [T, heads*v]  BEFORE the gate
  std::vector<double> gate;           // [T, heads]    AFTER the sigmoid
  std::vector<double> gated;          // [T, heads*v]
  int64_t gather_len = 0;
  int64_t rows_pruned_by_the_window = 0;  // queries that really lose a key
};

// `_forward_note_mla` on the SLIDING arm (model.py:462-475 -> :135-201 with
// `attention.is_sparse == False`), over ONE request of `num_tokens` queries at
// positions [0, T) — i.e. `seq_len == query_len == T`, which is the shape
// `_forward_swa_mqa` reduces to at `num_reqs == 1`.
//
// THE ATTENTION IS COMPUTED THE WAY UPSTREAM COMPUTES IT, which is the point of
// this function existing rather than a windowed copy of `ForwardFullAttention`:
// the ABSORBED MQA form of `_forward_swa_mqa` (attention.py:470-563) over a
// PAGED, PADDED latent cache, gathered by `GatherSwaKv` and masked by
// `ApplySwaScoreMask`. So all four §2.3 mechanisms are REACHED by the layer's
// own gate rather than only by their unit cases, and the independent reference
// they are compared against takes the other route entirely — materialized MHA
// with a direct positional window predicate and no cache at all.
//
//   hidden     [T, hidden_size]  the INPUT-LAYERNORM'd hidden state
//   positions  [T]               i32
//   out        [T, hidden_size]  the o_proj output
//
// `trace` may be null.
std::vector<double> ForwardSlidingAttention(const SlidingAttnDims& dims,
                                            const SlidingAttnWeights& w,
                                            const std::vector<double>& hidden,
                                            const std::vector<int32_t>& positions,
                                            int64_t num_tokens,
                                            const SlidingPaging& paging,
                                            SlidingAttnTrace* trace);

}  // namespace vllm::dots3_note
