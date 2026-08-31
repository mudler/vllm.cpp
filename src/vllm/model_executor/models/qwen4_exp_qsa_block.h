// Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) W5b-5 — `Qwen4ExpTextAttention` as ONE
// production block, and the first place under `src/` that COMPOSES the Qwen
// Sparse Attention indexer.
//
// Issue [#2211](https://github.com/mudler/vllm.cpp/issues/2211), wave issue
// [#2031](https://github.com/mudler/vllm.cpp/issues/2031), campaign issue
// [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHY THIS FILE EXISTS, IN THE SPEC'S OWN WORDS ───────────────────────────
// `## Owed`: "**W5b OWES THE INDEXER COMPOSITION IN PRODUCTION CODE, AND FOUR
// SETTINGS WITH IT.** … That composition exists in exactly one place: the
// `RunIndexer` helper in `tests/vllm/models/test_qwen4_exp_qsa_device.cpp`.
// Nothing under `src/` composes it, so nothing outside that helper enforces any
// of the four settings the collapse depends on".
//
// The four are stated once, here, and asserted once, in `Qwen4ExpQsaIndex`:
//
//   1. `weights` is all ones `[T, index_n_heads]`. This is what collapses
//      `vt::DsaIndexerLogits`'s per-head fold to a single constant, which is
//      what makes that op QSA's block score at all. QSA has no `weights_proj`.
//   2. `n_head_scale == 1.0f`, NOT DeepSeek-V4's `n_head ** -0.5`. QSA has no
//      tensor for it and upstream's scoring line has no such factor.
//   3. `softmax_scale == index_head_dim ** -0.5`, QSA's own scale
//      (`modeling_qwen4_exp.py:693`, `/ math.sqrt(self.index_head_dim)`).
//   4. `win_end == kv_len / compress_ratio` per query token — the COMPLETE
//      VISIBLE blocks, not the whole cache.
//
// **TWO OF THE FOUR ARE INVISIBLE TO ANY SELECTION-BASED GATE, BY CONSTRUCTION,
// AND THAT IS WHY THIS HEADER NAMES THEM.** Top-k is invariant under a positive
// rescale of every score, so a wrong `n_head_scale` or `softmax_scale` cannot
// move a selected set — spec mutation M26 measures exactly that survival, and
// `DsaIndexerLogitsArgs` says the same thing in its own comment. `Qwen4ExpQsaIndex`
// therefore takes an OPTIONAL `logits` out-parameter and the gate compares it BY
// VALUE against the oracle's own pre-top-k score tensor. A selection gate here
// would be an instrument nobody wired up.
//
// ─── THE OTHER TRAP THIS BLOCK CARRIES ───────────────────────────────────────
// A mask-shaped consumer passes a token gate and forfeits the whole point of the
// row: `exp(-inf - m)` is exactly +0, so a sparse mask over a dense cache agrees
// with a gather VALUE FOR VALUE, and under CUDA flash attention it costs the
// full dense prefix (llama.cpp #27739). This block routes the consumer through
// `vt::Qwen4ExpQsaGatherAttention` and forwards that op's `keys_visited`
// instrument, so the block's own gate can run the NaN-poison and unmapped-tail
// probes W5b-4 established rather than trusting the shape of the call.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// vLLM registers `qwen4_exp` at NO revision, so under AGENTS.md "When vLLM has
// no implementation" the ALGORITHM oracle is transformers **5.16.0**, this row's
// accepted lane pin, at `models/qwen4_exp/modeling_qwen4_exp.py`:
// `Qwen4ExpTextAttention.forward` (:785-841) and `Qwen4ExpTextQSAIndexer.forward`
// (:631-716). Every op this block calls is a vLLM-mirrored primitive; the block
// is the composition, and the composition is what upstream has and vLLM does not.
//
// ─── SCOPE, AND WHAT IS NOT HERE ─────────────────────────────────────────────
// This block is REACHED ONLY BY ITS TEST at its merge commit, and the spec's
// `## Owed` records that with the row and the issue that own the wiring:
// `Qwen4ExpTextModel::Forward` does not exist yet, so `ModelRegistry::Forward`
// still refuses `Qwen4ExpForConditionalGeneration` by name. Also not here: the
// cos/sin table build (taken as an operand, so the interleaved-mRoPE section
// layout stays owed by the wave that builds it), and the CUDA arm.
//
// ─── W5d-3 (#2249 item 2): THE PAGED K/V ARM ─────────────────────────────────
// W5b-5 shipped ONE cache shape, contiguous, and said so. The engine allocates
// another: `MakeQwen4ExpKVCache` publishes the QSA layers' K/V as a PAGED
// `FullAttentionSpec` group, so nothing could serve from the cache a runner
// actually hands a forward. `RunQwen4ExpQsaBlockPaged` below is that consumer.
//
// ONE BODY, TWO CACHE ARMS. The two entry points share `QsaBlockCore`; what
// forks is where the new K/V rows are STORED (contiguous rows in place, or
// `dense_attn::WriteKvCache` at a slot mapping) and how a key row is ADDRESSED
// by the consumer (flat, or through a page table). Everything between —
// projections, the per-head norms, the RoPE, the indexer, the output gate,
// `o_proj` — is one copy.
//
// ─── W5i (#2249 item 3): THE INDEXER SIDE CACHE IS PAGED TOO ─────────────────
// W5d-3 left one cache shape behind: the QSA indexer side cache, KV group 2, an
// `MLAAttentionSpec` whose block table W5c-2 gathers. The MAP reached a forward
// while the STORE and the READ still demanded a contiguous `[max_kv, D]` buffer
// indexed by absolute position, so even with W5h's geometry fix the forward could
// not address what the engine allocates. W5i closes that with NO NEW OP: the
// store scatters through `vt::IndexCopy` and the read gathers through
// `vt::IndexSelect`, both at physical slots resolved from group 2's own block
// table. `Qwen4ExpQsaIndex`'s contract is UNCHANGED — it still takes a
// contiguous `[rows, D]` — because the gather is what hands it one.
//
// WHAT IS STILL NOT CLOSED, AND WHO OWNS IT. The engine's group-2 buffer does not
// REACH this block yet: `ModelRegistry::Forward` refuses `multi_kv` by name
// (#2353), so the registry hook hands a per-call scratch in the same paged shape
// and with the same translation. Lifting that refusal is W5j's, and #2353 records
// why it must not be lifted here. The spec's `## Owed` carries both.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_QSA_BLOCK_H_
#define VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_QSA_BLOCK_H_

#include <cstdint>
#include <memory>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev
#include "vllm/model_executor/models/qwen3_5.h"             // PagedKvCache
#include "vllm/model_executor/models/qwen4_exp.h"          // Qwen4ExpParams
#include "vllm/model_executor/models/qwen4_exp_weights.h"  // Qwen4ExpQsaWeights
#include "vt/tensor.h"

namespace vllm {

// The per-sequence caches one QSA layer reads and writes. CONTIGUOUS, not paged,
// and that is a scope statement rather than a design preference: both `vt::` ops
// this block drives address their caches as flat `[rows, …]` arrays and never
// read `stride[0]` (see `Qwen4ExpQsaGatherAttention`'s own contract). This arm is
// what a caller that owns its own flat buffers uses; the arm that serves what the
// ENGINE allocates is `Qwen4ExpQsaPagedCaches` below, and since W5i that includes
// the indexer side cache.
//
// `key`/`value` hold the RAW model K/V — what upstream's `past_key_values.update`
// returns — and `index_key` holds the RAW, UN-normed and UN-roped indexer keys,
// which is what `Cache.update_indexer` stores and what
// `vt::Qwen4ExpQsaCompress` expects. Storing them normed or roped would be a
// double application, because the compressor does both.
struct Qwen4ExpQsaCaches {
  vt::Tensor key;        // [max_kv, num_kv_heads, head_dim]     READ-WRITE
  vt::Tensor value;      // [max_kv, num_kv_heads, head_dim]     READ-WRITE
  vt::Tensor index_key;  // [max_kv, indexer_head_dim]           READ-WRITE
};

// The per-layer caches one QSA layer reads and writes when the K/V lives where
// the ENGINE puts it: a paged `FullAttentionSpec` group.
// (Row MODEL-MM-QWEN4-EXP W5d-3, [#2249](https://github.com/mudler/vllm.cpp/issues/2249)
// item 2.)
//
// `kv` is the runner's own per-layer handle, unchanged and un-narrowed. Taking
// `PagedKvCache` rather than a QSA-private copy of its five numbers is the point:
// `dense_attn::KvSlice` builds the two rank-4 unbind views from it, and
// `dense_attn::WriteKvCache` / `dense_attn::IsFp8KvCache` read the `fp8_kind` and
// the scales out of the SAME struct, so the store and the read cannot disagree
// about how wide a KV element is. An fp8 cache is REFUSED BY NAME here rather
// than read as floats: `vt::Qwen4ExpQsaGatherAttention` has no dequantising read,
// and a silently-wrong one is wrong tokens instead of a crash.
//
// `slot_mapping` is i64 [T], the runner's own per-token destination slot
// (`block * block_size + offset`), and it is what makes the store paged. The
// contiguous arm's "project straight into the cache rows" trick has no paged
// equivalent, because a step's tokens can cross a page boundary.
//
// `block_table` is i32 [1, max_pages]: ONE sequence per call, as the contiguous
// arm takes one. A ragged multi-request batch needs `query_start_loc` plumbing
// this block does not carry; the spec's `## Owed` records it.
//
// `index_key` is the QSA INDEXER side cache, and W5i makes it PAGED too
// (#2249 item 3). It is KV GROUP 2 — its own `MLAAttentionSpec`, its own
// physical page pool, and therefore its OWN block table, which is why
// `index_block_table` is a second field rather than a reuse of `block_table`
// above. Group 0 and group 2 are allocated from separate pools; a body that read
// one group's cache through the other group's map would return another
// sequence's keys with no shape error.
//
// THE SHAPE IS THE RUNNER'S OWN. An `MLAAttentionSpec` group is allocated as the
// FUSED 3-dim page `[num_pages, block_size, head_size]` — there is no K/V pair
// and no factor 2 (`runner.cpp`, "the fused MLA 3-dim (num_blocks, block_size,
// head_size) for an MLA group"; `MLAAttentionSpec::real_page_size_bytes`). Taking
// that shape rather than `{ptr, num_pages, block_size}` is the same argument
// `PagedKvCache` makes for itself: a geometry carried in the tensor cannot
// disagree with the buffer it describes.
//
// Flattened, those pages are a CONTIGUOUS `[num_pages * block_size,
// indexer_head_dim]` array of rows addressed by PHYSICAL SLOT, so the map from a
// sequence's logical position `i` is the ordinary paged one,
// `index_block_table[i / block_size] * block_size + i % block_size`. That is what
// makes this addressable with `vt::IndexSelect` / `vt::IndexCopy` and no new op:
// see `IndexerRows` in the .cpp for the translation and for what it costs.
//
// ONE ROW PER TOKEN, not one per `indexer_compress_ratio`. W5h established that
// against upstream — `Cache.update_indexer` concatenates one raw key per token
// (`cache_utils.py:350-351`, returning `[batch, total_len, index_head_dim]`) and
// the ratio is the SELECTION algorithm's (`modeling_qwen4_exp.py:622`,
// `block_topk = token_budget // compress_ratio`), never the page geometry — and
// `MakeQwen4ExpKVCache` publishes the group at `compress_ratio = 1` because of
// it. The page size therefore needs NO relationship to the compress ratio: the
// gather below linearises the visible prefix before `vt::Qwen4ExpQsaCompress`
// sees it, so a compress block that straddles two pages costs nothing and is not
// refused. (The K/V group's `block_size % CR == 0` requirement above is a
// different one, and it stays: that consumer resolves a selected block through
// the page table itself.)
struct Qwen4ExpQsaPagedCaches {
  PagedKvCache kv;         // the runner's paged K+V for THIS layer  READ-WRITE
  vt::Tensor block_table;  // i32 [1, max_pages]  group 0: logical -> physical page
  vt::Tensor slot_mapping; // i64 [T]             this step's destination K/V slots
  // GROUP 2, the indexer side cache. [num_pages, block_size, indexer_head_dim].
  vt::Tensor index_key;           // READ-WRITE
  vt::Tensor index_block_table;   // i32 [1, max_pages]  group 2's OWN page map
};

// Owning device-resident output of one QSA block: a [T, hidden_size] view plus
// the shared_ptr that returns its pool block to the DevicePool when the last
// reference drops. Mirrors `GdnBlockOutput` / `MoeBlockOutput` exactly, so the
// composing forward keeps one release idiom across all three block kinds.
struct Qwen4ExpQsaBlockOutput {
  vt::Tensor tensor;              // [T, H] at the block's dtype
  std::shared_ptr<void> storage;  // owns the pool block (Pool().Put on release)
};

// The composed indexer's result for one step: the selected BLOCK ids and how
// many of them each query token got.
//
// The ids are what `vt::DsaTopkSelect` emits — ASCENDING, `-1`-terminated,
// all-select below `block_topk` — which is exactly what
// `vt::Qwen4ExpQsaGatherAttention` consumes. Nothing between the two reorders
// them, and nothing may: the ascending order is what makes a sub-budget
// selection reduce over the same positions in the same order dense attention
// would, and therefore bit-identical to it.
struct Qwen4ExpQsaSelection {
  vt::Tensor block_ids;  // [T, block_topk] i32, ascending, -1 = no block
  vt::Tensor counts;     // [T] i32
  std::shared_ptr<void> storage;  // owns both pool blocks
};

// `Qwen4ExpTextQSAIndexer.forward` (:631-716) as a COMPOSITION of three ops —
// `vt::Qwen4ExpQsaCompress`, `vt::DsaIndexerLogits`, `vt::DsaTopkSelect` — with
// the four settings this header names applied where no test helper is watching.
//
//   q_index    [T, index_n_heads, index_head_dim]  the indexer query, ALREADY
//              q-layernormed and roped by the caller (the block below does it)
//   index_key  a CONTIGUOUS [rows, indexer_head_dim] with the sequence's raw
//              indexer keys at logical positions [0, rows), rows >= kv_len:
//              UN-normed and UN-roped. THE TENSOR, not the cache struct, and it
//              stays contiguous after W5i made the side cache paged — the paged
//              arm GATHERS the visible prefix into one before calling this, so
//              this function never resolves a page and the two cache arms hand it
//              the same thing. Taking the struct would mean handing it one with
//              three dead fields
//   k_norm_w   [index_head_dim]  the RAW HuggingFace gamma; the compressor
//              applies `(1.0 + w)` itself, mirroring `Qwen4ExpTextRMSNorm`
//   cos/sin    [>= kv_len, rotary_dim] f32 FULL-position tables; the compressor
//              reads row `compress_ratio * b`, the block's FIRST token
//   kv_lens    [T] i32, the causal visible length of each query token
//   logits     OPTIONAL [T, kv_len / compress_ratio] f32 OUT — the VALUE-gate
//              surface. Two of the four settings cannot be seen any other way;
//              see the header comment. `nullptr` on the production path.
//
// The POOLED-block-key scratch is allocated per call and dropped, which mirrors
// upstream exactly: `Qwen4ExpTextQSAIndexer.forward` rebuilds its `pooled_keys`
// from `raw_keys` on every step and caches none of them
// (`modeling_qwen4_exp.py:679-682`). It is the RAW key that is cached, and that
// cache is the side cache this function reads.
Qwen4ExpQsaSelection Qwen4ExpQsaIndex(dense_attn::Dev d, const Qwen4ExpQsaParams& qsa,
                                      float rms_norm_eps, const vt::Tensor& q_index,
                                      const vt::Tensor& index_key, const vt::Tensor& k_norm_w,
                                      const vt::Tensor& cos, const vt::Tensor& sin,
                                      const vt::Tensor& kv_lens, int64_t kv_len,
                                      bool round_intermediates_to_bf16,
                                      vt::Tensor* logits = nullptr);

// One `Qwen4ExpTextAttention` block, end to end.
//
//   hidden     [T, hidden_size]  the block input — on this architecture that is
//              the gated residual's COLLAPSED 2560-wide output, never the
//              10240-wide stream, which only the hyper-connection ops see
//   positions  [T] i32, the position of each new token; used for the q/k RoPE
//   cos_sin    [P, rotary_dim] the packed cos|sin cache `vt::RopeFromCache`
//              reads: columns [0, rot/2) are cos and [rot/2, rot) are sin
//   cos/sin    [P, rotary_dim] the SEPARATE full tables the compressor reads.
//              Two layouts for one set of angles, because the two ops were
//              ported from two upstreams that spell it differently. The caller
//              must build both from ONE table, and the block CROSS-CHECKS that
//              rather than trusting it: equal heights, then a BOUNDED SAMPLE of
//              rows compared value for value. The sample is what a table-wide
//              construction difference shows in and is not a per-row guarantee;
//              `CheckRopeLayoutsAgree` in the .cpp states exactly what it can
//              and cannot see, and why a full comparison is not paid per call.
//   past_len   how many tokens the caches already hold; the new tokens land at
//              rows [past_len, past_len + T)
//   keys_visited  OPTIONAL, forwarded to `vt::Qwen4ExpQsaGatherAttention`. It is
//              counted AT THE KEY-ROW READ and is the discriminator between a
//              gather and a mask; see that op's contract for what it cannot see
//              and which two probes close the gap.
Qwen4ExpQsaBlockOutput RunQwen4ExpQsaBlock(dense_attn::Dev d, const Qwen4ExpQsaWeights& w,
                                           const Qwen4ExpParams& params, const vt::Tensor& hidden,
                                           const vt::Tensor& positions, const vt::Tensor& cos_sin,
                                           const vt::Tensor& cos, const vt::Tensor& sin,
                                           const Qwen4ExpQsaCaches& caches, int64_t past_len,
                                           int64_t* keys_visited = nullptr);

// The SAME block over the PAGED K/V the engine allocates (W5d-3, #2249 item 2).
//
// Every operand it shares with `RunQwen4ExpQsaBlock` means exactly what it means
// there, and the two run one body. `past_len` still counts the LOGICAL tokens the
// sequence already holds — the page table is what turns a logical position into a
// physical row, so nothing about the causal arithmetic moves when the pages are
// permuted, which is the property this arm's gate asserts.
//
// The caller owns `caches.slot_mapping` and must have sized it to this step's T;
// the block does not build it, exactly as `dense_attn::AttnBlock` does not
// (`StepInputs` carries the runner's own).
Qwen4ExpQsaBlockOutput RunQwen4ExpQsaBlockPaged(dense_attn::Dev d, const Qwen4ExpQsaWeights& w,
                                                const Qwen4ExpParams& params,
                                                const vt::Tensor& hidden,
                                                const vt::Tensor& positions,
                                                const vt::Tensor& cos_sin, const vt::Tensor& cos,
                                                const vt::Tensor& sin,
                                                const Qwen4ExpQsaPagedCaches& caches,
                                                int64_t past_len,
                                                int64_t* keys_visited = nullptr);

}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_QWEN4_EXP_QSA_BLOCK_H_
