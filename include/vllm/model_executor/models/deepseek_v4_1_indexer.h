// DeepSeek-V4.1-Flash — the indexer arm (W3c) and the `kv_source_layer`-keyed
// KV topology (W3d), as portable HOST references.
//
// SCOPE. W3c and W3d of `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`
// (ISSUE-LOCAL-01M2TMSCJJH7X8WZAM58PB3PMB). They land as ONE change because
// they are ONE mechanism: the topology decides which layers own an indexer, an
// indexer-K cache and a compressed-KV cache, and the indexer op is what those
// owners run. Splitting them would give each half a gate that cannot see the
// other half's defect.
//
// NOTHING REACHES THIS YET, AND THAT IS DISCLOSED RATHER THAN IMPLIED.
// AGENTS.md §"Nothing lands dead" asks for three things: what is not reached,
// the ROW that owns the wiring, and the issue that tracks it.
//   * NOT REACHED: every function declared below. `DeepseekV41ResolveLayerTopology`,
//     `DeepseekV41ResolveModelTopology`, `DeepseekV41ResolveLayerRope`,
//     `DeepseekV41IndexerKCacheRowBytes`, `DeepseekV41IndexerBlockSizes`,
//     `DeepseekV41FusedQKvRmsNorm` and `DeepseekV41IndexerKNormRopeStore` have
//     no caller outside `tests/vllm/models/test_deepseek_v4_1_indexer.cpp`.
//     W1 registered `deepseek_v41`, but its forward and its loader refuse by
//     name, so no production entry point can arrive here.
//   * THE ROW: `MODEL-MM-deepseek-v4-1-deepseek-v41-for-causal-lm`. Inside it,
//     WAVE W4 owns the host FORWARD assembly that composes W3a-W3f and is what
//     will call these. W4 is a wave, not a row, and the rule asks for the row.
//   * THE ISSUE: ISSUE-LOCAL-01M2TMSCJJH7X8WZAM58PB3PMB, named at the top of
//     this file and in the landing commit body.
// The landing commit body and `.agents/specs/deepseek-v4-1-flash.md` `## Owed`
// carry the same three facts. A reviewer who deletes a production call site
// here will find none, because there is none.
//
// WHY A NEW FILE RATHER THAN A WIDENING OF `deepseek_v4_dsa.{h,cpp}`. Upstream
// FORKED `vllm/models/deepseek_v4_1/` from `vllm/models/deepseek_v4/` instead of
// parameterizing it, and the V4 indexer in this tree is not extensible to V4.1
// even in principle: `IndexerCompressedKeys` opens with
// `VT_CHECK(compress_ratio == 4, ...)` and fixes `coff = 2`
// (`deepseek_v4_dsa.cpp:485-488`). V4.1 has no ratio 4 anywhere — its ratios are
// drawn from {0,1,2} — and it DELETES the indexer-local compressor that function
// implements. Widening it would put V4's landed gates at risk to express a
// function V4.1 does not have.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream revision: vLLM `e77daef89e` (vllm#56214, 2026-09-11). This is AHEAD
// OF our parity pin `e126687a9a`, where `vllm/models/deepseek_v4_1/` does not
// exist at all, so nothing here claims a run gate against the pinned oracle.
// W2 owns the pin advance. The gate these functions carry is the one V4's
// W3-W7 carried: a host reference unit-gated at small shapes against an
// independently written reference and hand-derived expectations, which a later
// device port is then gated against. That reference computes in fp32 and NOT in
// double, because upstream's does (`fused_qk_rmsnorm.py:71-73`,
// `indexer_k_store.py:156`) and because the device port it will gate does; see
// `DeepseekV41FusedQKvRmsNorm`.
//
//   OURS                                  <-  UPSTREAM (at e77daef89e)
//   DeepseekV41IndexerKNormRopeStore      <-  common/ops/indexer_k_store.py
//                                             :29 (the entry point and its
//                                             asserts) and :120 (the kernel)
//   DeepseekV41IndexerKCacheRowBytes      <-  attention.py:82-90
//                                             (_indexer_k_cache_head_dim)
//   DeepseekV41IndexerBlockSizes          <-  v1/attention/backends/mla/
//                                             indexer.py:252-259
//   DeepseekV41FusedQKvRmsNorm            <-  models/common/ops/
//                                             fused_qk_rmsnorm.py:194
//   DeepseekV41ResolveLayerTopology       <-  attention.py:244-292
//   DeepseekV41ResolveModelTopology       <-  attention.py:369-395 (which
//                                             layer ALLOCATES an index-K cache)
//   DeepseekV41LayerRope                  <-  common/rope.py (the V4.1 copy;
//                                             its delta against V4's is below)
//
// ─── THE FOUR UPSTREAM FACTS A WRONG PORT PASSES EVERY SHAPE CHECK ON ────────
//
// 1. `fused_indexer_q_rope_quant` AND `fused_inv_rope_fp8_quant` ARE NOT
//    REPLACED. `common/ops/__init__.py:12-22` re-exports both from
//    `vllm.models.deepseek_v4.common.ops` byte-identically, and its docstring
//    says the re-exports are "the only place this package reaches into
//    vllm.models.deepseek_v4". Verified by reading that file, not inferred.
//    The net-new op is `indexer_k_norm_rope_store` and only that one. Anyone
//    porting the Q-side fused ops here is porting V4 a second time.
//
// 2. RATIO 1 IS NOT "SLIDING WINDOW". V4 reads `compress_ratio <= 1` as
//    sliding-window-only; in V4.1, `compress_ratios` is `[0,0, 2 x18, 1 x20,
//    0,0,0]` and the twenty `1`s are layers 20-39 running a FULL-LENGTH
//    compressed cache (`attention.py:246-248`: "0 = pure sliding window,
//    1 = full-length compressed cache, 2 = ratio-2 compressed"). Inheriting
//    V4's `> 1` polarity silently kills the compressed arm of half the stack
//    while every shape still matches. The same polarity change is visible in
//    the RoPE selection, and that is the ONLY textual delta between V4's
//    `common/rope.py` and V4.1's: `compress_ratio > 1` becomes
//    `compress_ratio > 0` at V4.1's `:23`, `:29` and `:31`, and the one-line
//    comment above `:31` is rewritten into the two lines `:32-33`. `diff` of
//    the two files at `e77daef89e` is exactly that: four lines removed, five
//    added. It is THREE characters plus a comment, never one. See
//    `DeepseekV41LayerRope`.
//
// 3. SOURCE RESOLUTION IS `max(s <= layer_id)`, NOT "the last source" AND NOT
//    "the previous layer" (`attention.py:284-286`, index twin `:287-289`).
//    With `kv_source_layer_ids = [2,8,14,20]` every layer 20-39 resolves to 20,
//    which IS the release card's "decoder KV projected from the final encoder
//    hidden states" — layer 20's compressor consumes layer 19's output. There
//    is no encoder-decoder split to port; the topology is config.
//    `kv_source_layers[-1]` is INDISTINGUISHABLE from `max(s <= L)` for every
//    layer at or above the last source (here, 20-39), and differs only BELOW
//    it. A gate that only exercises layers >= 20 cannot see that mutation.
//
// 4. EIGHT INDEXERS, FOUR INDEX-K CACHES. `index_source_layer_ids` has EIGHT
//    entries `[2,8,14,20,24,28,32,36]` against `kv_source_layer_ids`' four, so
//    eight layers carry an indexer — but an index-K cache is ALLOCATED only
//    where `is_kv_source` also holds, because only a kv source has the
//    `wk`/`k_norm` pair that produces keys. The other four index sources
//    (24/28/32/36) READ the cache of `kv_source_layer_id`, which is 20 for all
//    of them (`attention.py:371-395`). So the counts are 8 indexers, 4 index-K
//    caches, 4 compressed-KV caches, and a port that allocates eight caches is
//    wrong in a way no shape check sees. `.agents/specs/deepseek-v4-1-flash.md`
//    W3d says "FOUR indexer-K caches", which is right about the CACHES and is
//    silent about the eight indexers; this header is the place that separates
//    them.
#pragma once

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_1.h"

namespace vllm {

// ─────────────────────────────────────────────────────────────────────────────
// W3d — the `kv_source_layer`-keyed topology
// ─────────────────────────────────────────────────────────────────────────────

// Everything `attention.py:244-292` resolves for ONE layer before it builds a
// single module. Every field is a function of the config and the layer id, so
// this is the whole of "the topology" and there is nothing stateful in it.
struct DeepseekV41LayerTopology {
  // `attention.py:251-256` with its bounds guard: a layer past the end of
  // `compress_ratios` is 0 ("MTP layers past the configured list are pure
  // sliding-window"), never out of range.
  int64_t compress_ratio = 0;
  // `attention.py:273-275`. `is_backbone` gates BOTH: an MTP layer whose index
  // happened to appear in a source list is still not a source.
  bool is_backbone = false;
  bool is_kv_source = false;
  bool is_index_source = false;
  // `attention.py:284-289`. -1 when `compress_ratio == 0`, mirroring upstream's
  // `None`: a sliding-window layer resolves NO source, and a caller that reads
  // one is asking a question the layer does not have an answer to.
  int64_t kv_source_layer_id = -1;
  int64_t index_source_layer_id = -1;
  // `attention.py:371-395`. Which layer's paged index-K cache this layer's
  // indexer WRITES (when it owns one) or READS (when it does not). It is keyed
  // by `kv_source_layer_id` and NOT by `index_source_layer_id`, because the
  // producing pair `wk`/`k_norm` lives on the kv source. -1 when this layer has
  // no indexer at all.
  int64_t index_k_cache_owner_layer_id = -1;
  // True only on the layer that also OWNS the cache it uses (`is_kv_source`).
  // A non-owning index source reads a cache it never writes.
  bool owns_index_k_cache = false;
  // `attention.py:398-399`. The candidate pre-filter: one layer PUBLISHES the
  // top `candidate_topk_blocks` blocks, and every indexer strictly above it
  // masks its scores to those blocks before its own top-k.
  bool is_candidate_source = false;
  bool uses_candidates = false;
};

// Resolve one layer. Throws with a message naming the layer and the value when
// `compress_ratios[layer_id]` is outside {0,1,2} (`attention.py:257-262`) or
// when a compressed layer has empty source lists (`attention.py:275-283`).
//
// `layer_id` may exceed `num_hidden_layers`: the MTP/DSpark tail is addressed
// by the same function and upstream's guards are what make that safe.
DeepseekV41LayerTopology DeepseekV41ResolveLayerTopology(
    const DeepseekV41Params& params, int64_t layer_id);

// The whole-model roll-up, which is what a KV-cache-spec builder needs and what
// makes fact 4 above executable. Every vector is sorted ascending and holds
// DISTINCT layer ids.
struct DeepseekV41ModelTopology {
  // Layers that allocate a compressed-KV cache. Exactly `kv_source_layer_ids`
  // restricted to the backbone.
  std::vector<int64_t> compressed_kv_cache_layers;
  // Layers that CARRY an indexer: `index_source_layer_ids` on the backbone.
  // EIGHT on the published config.
  std::vector<int64_t> indexer_layers;
  // Layers that ALLOCATE an index-K cache: `indexer_layers` intersected with
  // `compressed_kv_cache_layers`. FOUR on the published config. The gap
  // between this and `indexer_layers` IS fact 4.
  std::vector<int64_t> index_k_cache_layers;
};

DeepseekV41ModelTopology DeepseekV41ResolveModelTopology(
    const DeepseekV41Params& params);

// The per-layer RoPE parameters (`common/rope.py`, the V4.1 copy).
//
// THE PORT'S WHOLE RISK IS THE THRESHOLD. V4's file keys on
// `compress_ratio > 1` and V4.1's on `compress_ratio > 0`, at V4.1's `:23`,
// `:29` and `:31`; a `diff` of the two at `e77daef89e` is exactly those three
// lines plus the comment above `:31`, which becomes the two lines `:32-33`. So
// under V4's rule the twenty ratio-1 layers would take `rope_theta = 10000`
// with no scaling, and under V4.1's they take `compress_rope_theta = 160000`
// with YaRN at the config's factor. Half the stack, no shape change.
//
// `use_yarn` is the BEHAVIOUR, not upstream's literal `rope_type` string.
// Upstream sets `rope_type = "deepseek_yarn"` on BOTH arms; on the
// sliding-window arm it then pins `factor = 1.0` and
// `original_max_position_embeddings = max_position_embeddings`, which is YaRN
// evaluating to the identity, and its own comment calls that "plain RoPE
// (theta=rope_theta, no YaRN)". Carrying the string would record a difference
// that is not one; carrying the factor records the one that is.
struct DeepseekV41LayerRope {
  double rope_theta = 0.0;
  bool use_yarn = false;   // false == factor pinned to 1.0 (the identity)
  double factor = 1.0;     // 16 on the compressed arm, 1.0 on the SWA arm
  int64_t original_max_position_embeddings = 0;
  double beta_fast = 0.0;
  double beta_slow = 0.0;
  // `rope_parameters["mscale"] = 0` and `["mscale_all_dim"] = 0` — upstream
  // DISABLES mscale on both arms unconditionally (`rope.py:44-45`). Carried as
  // a named constant rather than omitted, because "absent" and "disabled" are
  // different instructions to a device port.
  static constexpr double kMscale = 0.0;
  static constexpr double kMscaleAllDim = 0.0;
};

DeepseekV41LayerRope DeepseekV41ResolveLayerRope(const DeepseekV41Params& params,
                                                 int64_t layer_id);

// ─────────────────────────────────────────────────────────────────────────────
// W3c — the indexer arm
// ─────────────────────────────────────────────────────────────────────────────

// `dsa_indexer_uses_fp4(vllm_config)` resolved to a value. The cache row width,
// the quantizer and the legal layouts all key on this.
enum class DeepseekV41IndexerKFormat {
  kFp8,   // 128 fp8-e4m3 bytes + one f32 scale = 132 bytes at head_dim 128
  kMxfp4  // 64 packed nibble-pairs + 4 UE8M0 bytes = 68 bytes at head_dim 128
};

// The ROCm readers take the values back 16x16-tiled; CUDA takes them row-major.
// `indexer_k_store.py:76-90` selects `shuffle = is_rocm() and block_size > 1`,
// and its own comment says writing row-major on ROCm "would hand both readers
// permuted key bytes". This is a WRITE layout, so it is a parameter here rather
// than a platform query: a host reference has no platform.
enum class DeepseekV41IndexerKLayout { kRowMajor, kRocmTiled16x16 };

// `attention.py:82-90` (`_indexer_k_cache_head_dim`). The per-token BYTE width
// of one paged indexer-K row, values and scales together.
//   fp8 : index_head_dim + index_head_dim/128 * 4   -> 132 at 128
//   fp4 : index_head_dim/2 + index_head_dim/32      ->  68 at 128
int64_t DeepseekV41IndexerKCacheRowBytes(int64_t index_head_dim,
                                         DeepseekV41IndexerKFormat format);

// `v1/attention/backends/mla/indexer.py:252-259`. The V4.1 backend's ENTIRE
// delta against V4's is two methods: the name, and this.
//
// It is ARCH-CONDITIONAL and not a flat change. V4 returns `[256]`
// unconditionally; V4.1 returns `[64]` on a Hopper-family device (capability
// family 90) and `[128]` everywhere else. Reading it as "V4.1 uses 128" is
// wrong on exactly one vendor generation, and reading it as "V4.1 uses 64" is
// wrong on every other. The predicate is passed in because a host reference
// does not have a device to ask.
std::vector<int64_t> DeepseekV41IndexerBlockSizes(bool is_hopper_family);

// `models/common/ops/fused_qk_rmsnorm.py:194` (`fused_q_kv_rmsnorm`) — the
// UNFUSED q/kv RMSNorm pair, which is the correct host port of this seam.
//
// WHY THIS AND NOT `query_quant.py`. `fused_q_kv_rmsnorm_quant` fuses the
// MXFP8 activation quantization into the same kernel, and
// `can_fuse_query_quant` (`query_quant.py:127-129`) returns False immediately
// unless `current_platform.is_cuda()`, then further requires both consumer
// linears to be FlashInfer CUTLASS/CuTeDSL MXFP8 kernels — a Blackwell path.
// `attention.py:602` therefore falls through to `fused_q_kv_rmsnorm` on every
// platform this reference targets, so implementing the plain norm is not a gap
// to be closed later; it is what the unfused branch DOES. The fusion is owed
// PERF, recorded under the spec's `## Owed`, and W5 owns it.
//
// The norm itself is plain: accumulate the sum of squares in f32, scale by
// `rsqrt(mean + eps)`, multiply by the per-column weight. `q` and `kv` have
// independent widths and independent weights and share only the token count.
//
//   q      [num_tokens, q_size]     row-major
//   kv     [num_tokens, kv_size]    row-major
//   q_out  [num_tokens, q_size]     caller-owned, PACKED row-major
//   kv_out [num_tokens, kv_size]    caller-owned, PACKED row-major
//
// `q_out`/`kv_out` are packed by construction, which is the invariant
// upstream's `test_fused_q_kv_rmsnorm_outputs_are_packed` exists to hold:
// there, `torch.empty_like` on a size-1 column slice inherited the WIDE row
// stride of the fused projection buffer and downstream dispatchers silently
// took a slower GEMM. A `std::vector<float>` cannot express that defect, so
// that upstream case has no counterpart here and this comment is its record.
void DeepseekV41FusedQKvRmsNorm(const float* q, const float* kv,
                                const float* q_weight, const float* kv_weight,
                                int64_t num_tokens, int64_t q_size,
                                int64_t kv_size, double eps, float* q_out,
                                float* kv_out);

// `common/ops/indexer_k_store.py:29` / `:120` — THE net-new V4.1 op.
//
// WHAT IT REPLACES. V4 derived index keys from the hidden state through an
// indexer-LOCAL `DeepseekCompressor` carrying its own rolling state
// (`IndexerCompressedKeys` here, `deepseek_v4_dsa.cpp:478`). V4.1 deletes that:
// the index key is `k_norm(wk(latent))` over the MAIN compressor's PRE-RoPE
// latent, so there is no hidden-state GEMM and no second compressor state.
// `indexer_k_store.py`'s docstring states it: "there is no per-token pooling
// from a compressor state cache: the latent already stands for a whole group".
// The function is PURE given its inputs, which is the direct consequence.
//
// THE FOUR STEPS, each with the upstream line that fixes it:
//
//   a. SKIP. `slot < 0` returns (`:146-148`). Then `(position + 1) %
//      compress_ratio != 0` returns (`:151-152`): only the LAST token of a
//      group publishes that group's key. At ratio 1 every token qualifies, so
//      the guard is not dead code at ratio 1 — it is the identity there, and a
//      port that hoists it out breaks ratio 2 only.
//   b. RMSNORM in f32 with a bf16 ROUND-TRIP (`:156-161`). `k` is normalized
//      in f32, multiplied by the weight, rounded to bf16, then widened back to
//      f32. That round-trip is not decoration: it is the rounding boundary the
//      reference materializes, and dropping it changes the fp8 bytes.
//   c. ROPE at the GROUP's first position (`:177`), `(position /
//      compress_ratio) * compress_ratio` — NOT at `position`. GPT-J/interleaved
//      layout over the TRAILING `rope_head_dim` lanes only: with head_dim 128
//      and rope_head_dim 64, lanes 0-63 pass through untouched and lanes
//      64-127 rotate as 32 (even, odd) pairs. The `cos_sin_cache` row is
//      `[cos x32 | sin x32]`. Both rotated halves take a second bf16
//      round-trip (`:186-187`).
//   d. QUANTIZE AND STORE into a SEGREGATED page (`:189-206`): all
//      `block_size` value rows first, then all `block_size` scale rows. The
//      value region and the scale region are not interleaved per token, so a
//      port that writes `[value | scale]` per row passes every length check
//      and hands the reader garbage.
//        fp8 : the 128 lanes are re-interleaved to (even0,odd0,even1,...),
//              rounded to bf16 once more, then scaled by
//              `2^-ceil(log2(max(absmax,1e-4)/448))`, clamped to +/-448 and
//              rounded to fp8-e4m3. The stored scale is the f32 `2^exponent`,
//              four bytes, little-endian.
//        fp4 : the even and odd halves are reshaped to 4 blocks of 16 pairs
//              (so one MXFP4 block == 32 consecutive lanes), one UE8M0 scale
//              per block is `ceil(log2(max(amax, 6*2^-126)/6))` stored as
//              `exponent + 127`, and each byte packs LOW nibble = even lane,
//              HIGH nibble = odd lane, round-to-nearest-EVEN and saturating
//              at +/-6 (`cvt.rn.satfinite.e2m1x2.f32`).
//
// `k_pre` rows at skipped tokens ARE READ BY NOBODY and upstream's own tests
// fill them with NaN to prove it. This port must have the same property: a NaN
// at a skipped row that reaches any output is the defect those tests are for.
//
//   k_pre            [num_rows, index_head_dim]  f32, the `wk(latent)` output
//   positions        [num_rows] int64
//   cos_sin_cache    [max_pos, rope_head_dim] f32, GPT-J: cos half then sin
//   rms_norm_weight  [index_head_dim]
//   k_cache          [num_blocks, block_size, row_bytes] uint8, MUTATED
//   slot_mapping     [num_tokens] int64, -1 = skip; num_tokens <= num_rows
//
// Bytes outside the written value/scale slots are LEFT UNTOUCHED, which is what
// makes "preserve skipped slots and page padding" gateable.
//
// ONE DELIBERATE NARROWING, recorded rather than left silent: `k_cache` is
// assumed CONTIGUOUS, so the page stride is `block_size * row_bytes`. Upstream
// reads `k_cache.stride(0)` instead, which lets a caller hand it a strided
// view; its own test uses `as_strided` with a wider row precisely to catch an
// over-write. Nothing in this tree produces a strided paged cache, and a
// `std::vector<uint8_t>` cannot express one, so the parameter is not carried.
// The over-write it guards against is caught here a different way: the gate
// compares the WHOLE buffer byte for byte, trailing untouched block included.
void DeepseekV41IndexerKNormRopeStore(
    const float* k_pre, const int64_t* positions, const float* cos_sin_cache,
    const float* rms_norm_weight, double eps, int64_t num_tokens,
    int64_t num_rows, int64_t index_head_dim, int64_t rope_head_dim,
    int64_t compress_ratio, DeepseekV41IndexerKFormat format,
    DeepseekV41IndexerKLayout layout, const int64_t* slot_mapping,
    int64_t num_blocks, int64_t block_size, uint8_t* k_cache);

}  // namespace vllm
