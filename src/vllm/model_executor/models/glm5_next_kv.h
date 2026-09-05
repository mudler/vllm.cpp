// GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) — W5b-2c: THE ENGINE'S KV TOPOLOGY,
// consumed. What makes `ModelRegistry::Forward` stop refusing a cache set keyed
// by layer name, and what makes a second decode step read history instead of
// inventing one.
//
// Issue [#2348](https://github.com/mudler/vllm.cpp/issues/2348), campaign issue
// [#1998](https://github.com/mudler/vllm.cpp/issues/1998), spec
// `.agents/specs/glm5-next-flash.md` §W5b-2c. Follows W5b-2b
// ([#2337](https://github.com/mudler/vllm.cpp/issues/2337)), which made
// `ModelRegistry::Forward` reach this model, and O28, which measured the guard
// that stands above it.
//
// Model-private under `src/`, the same arrangement every other file on this row
// uses.
//
// ─── WHAT THE ENGINE ACTUALLY HANDS OVER, MEASURED AND NOT ASSUMED ───────────
//
// The refusal O28 recorded says it exactly: "22 KV cache(s) from 2 published
// group(s) reached this forward, first 'model.layers.3.self_attn.attn', with
// block tables gathered for 3 of 3 published group(s)". Every number in that
// sentence is a different denominator, and reading them as one is the first way
// this mapping goes wrong:
//
//   * `MakeGlm5NextKVCache` publishes THREE groups — 0 the 11 DSA layers' MLA
//     latent (`MLAAttentionSpec`, head 512), 1 the 34 KDA layers' recurrent
//     state (`MambaSpec`, two states), 2 the 11 DSA layers' indexer side cache
//     (`MLAAttentionSpec`, head 257).
//   * TWENTY-TWO caches reach `ModelForwardInput::attn_kv`, not 45 and not 3:
//     `attn_kv` carries one entry per PUBLISHED NAME of every ATTENTION group,
//     which is 11 + 11. The recurrent group contributes NOTHING to `attn_kv` —
//     its 34 layers land in `ModelForwardInput::gdn_state` instead
//     (`runner.cpp`, the `alloc_recurrent_layer_states` arm of the multi-cache
//     path). This line used to add "which is why `num_groups()` answers TWO
//     while `num_published_groups()` answers three", and W5b-2d measured that
//     and it is FALSE: `num_groups()` counts distinct ids over the whole
//     `group_ids` vector, which `9e7621efc` widened to cover the recurrent
//     group, so it answers THREE. The accessor's own comment in
//     `model_registry.h` still describes the pre-widening behaviour and #2459
//     owns it. Nothing on this row reads `num_groups()` for a decision — only
//     `ChannelSummary` prints it — but the next reader would have inherited the
//     stale number, which is exactly how this wave's defect travelled.
//   * The 22 arrive in PUBLICATION order — group 0's eleven, then group 2's
//     eleven — which is why the first published name is layer 3's MLA latent
//     and not layer 0's anything. Layer 3 is this checkpoint's first DSA layer.
//   * BLOCK TABLES are gathered for all three groups, indexed BY GROUP ID and
//     not parallel to `attn_kv` (`MultiKvCacheIndex::BlockTableForGroup`).
//
// **NOTHING HERE IS RESOLVED BY POSITION.** Every cache — paged and recurrent
// alike — is found by the NAME `MakeGlm5NextKVCache` published it under, and its
// group id is READ off the channel rather than assumed to be 0 and 2. A port
// that indexed `attn_kv` by DSA-layer ordinal would be right today and wrong the
// moment a group is added, reordered or renamed — and it would be wrong
// SILENTLY, because every one of the 22 entries is a plausible float buffer.
//
// ─── THE FLAT INDEX IS NOT THE PAGED SLOT, and this row shipped that error ───
//
// W5b-2d ([#2445](https://github.com/mudler/vllm.cpp/issues/2445)). There are
// TWO indices and `MultiKvCacheIndex` answers both, which is the distinction the
// W5b-2c mapping did not carry:
//
//   * `Find(name)` returns the FLAT index — the cache's place among EVERY
//     published cache, paged and recurrent together, in publication order. On
//     this model that space is 56 wide: group 0's 11 latents, then group 1's 34
//     recurrent states, then group 2's 11 indexer caches. `layer_names`,
//     `group_ids`, `layer_indices`, `payload_kinds` and `payload_slots` are all
//     parallel to THAT.
//   * `PayloadAt(flat, &kind, &slot)` returns the slot in the container the
//     cache actually lives in — `attn_kv` when `kind` is `kPaged`, `gdn_state`
//     when it is `kRecurrent`. That space is 22 wide and 34 wide respectively.
//
// Layer 3's indexer side cache is flat 45 and paged slot 11. Feeding 45 to
// `attn_kv` is what `--device cuda` on the real 101.25 GiB artifact died of, and
// it died on `--device cpu` identically: it is not a device defect. It threw
// only because 45 happens to be out of range for a 22-entry vector; a topology
// whose recurrent group is published LAST would have made every flat index
// in-range and every one of them a plausible float buffer for the wrong layer.
// So the refusal that caught it is an accident of ordering and the payload
// locator is the check.
//
// The group id is still read at the FLAT index, because `group_ids` is parallel
// to the flat list. Two indices, two vectors, and each is used against its own.
//
// ─── GROUP 0 IS AN MLA LATENT AND NOT A K+V PAIR ─────────────────────────────
//
// This is the single highest-risk error on this wave, and it does not crash.
// `Glm5NextTextAttention` caches the compressed `kv_a_proj_with_mqa` output and
// reconstructs K and V from it through `kv_b_proj`
// (`modeling_glm5_next.py:1136-1153`), so the published page is
// `MLAAttentionSpec::real_page_size_bytes` = `block_size * 1 * head_size *
// dtype_size` — ONE vector per token, NO factor 2, NO separate V
// (`kv_cache_interface.h:27-29`). The buffer is therefore
// `[num_blocks, block_size, head_size]` and a row is at
// `(block_id * block_size + block_offset) * head_size`.
//
// A reader that assumed the ordinary `[num_blocks, 2, block_size, num_kv_heads,
// head_size]` pair layout would index at twice the stride, read the second half
// of a page that has no second half, and hand the layer numbers that are
// finite, correctly shaped and wrong. The model would generate fluent text. So
// `ResolveKvBinding` refuses by NAME on `num_kv_heads != 1` and on a
// `head_size` that is not the published latent row, and `PagedRowOffset` below
// is the ONE place in this row that turns a logical position into an element
// offset — there is deliberately no second copy of that arithmetic.
//
// **AND THE ARITHMETIC IS CHECKED AGAINST THE ENGINE'S OWN.** The runner
// computes `attn_meta.slot_mapping` for the target attention group with its own
// block-table walk. `ResolveKvBinding` recomputes those slots from the group
// block table and refuses by name if the two disagree, so a misread block table
// is a refusal on the first step rather than a wrong token on every step.
//
// ─── THE RECURRENT GROUP IS KEYED BY NAME TOO, AND IT WAS NOT ───────────────
//
// This section used to read "THE RECURRENT GROUP IS NOT KEYED BY NAME", because
// when W5b-2c landed `MultiKvCacheIndex` described `attn_kv` only: the KDA
// states arrived on `ModelForwardInput::gdn_state` in ascending layer order with
// no name, so the correspondence was POSITIONAL and a count against
// `p.num_kda_layers()` was the strongest available check. ENG-MULTIKV-BYNAME
// (`9e7621efc`) falsified that sentence 65 minutes later by widening the channel
// over every published cache, and W5b-2d consumed it: each KDA layer resolves
// `model.layers.<l>.linear_attn` through the same `PayloadAt` locator the paged
// caches use, and reads its `gdn_state` slot off `payload_slots`.
//
// The positional ordinal SURVIVES as a check rather than as the answer. The
// runner still allocates one state set per recurrent layer in layer order
// (`alloc_recurrent_layer_states` inside
// `for (l : layers) if (gdn_layer_mask[l])`), so the channel's slot and the
// ordinal agree on every topology this tree builds, and `ResolveKvBinding`
// refuses by name when they do not. The count survives too, and it now catches
// what a per-layer lookup structurally cannot: `gdn_state` sets that no declared
// layer claimed.
//
// The state SLOT within each of those buffers is the engine's, not ours:
// `GDNAttentionMetadata::non_spec_state_indices_tensor` carries the compact
// per-sequence slot the runner remapped block-table column 0 to
// (`runner.cpp`, `remap_gdn_state_slots`). Reading the raw block id instead
// would index a `[gdn_state_slots_, ...]` buffer with an attention block id.
//
// ─── WHAT A FRESH SEQUENCE MUST NOT READ ─────────────────────────────────────
//
// A recurrent slot is REUSED across sequences and is never pre-zeroed; upstream
// gathers the rows and zeros the fresh ones in the layer forward, keyed by
// `has_initial_state = num_computed_tokens > 0`
// (`qwen_gdn_linear_attn.py:1512-1513`, mirrored in `gdn_attn.h`'s caller
// obligation). This file honours that by NOT hydrating at all when
// `cached_len == 0`: `Glm5NextKdaCache`'s empty vectors are its own
// fresh-sequence signal and `Glm5NextKdaLayerForward` zero-fills them
// (`glm5_next_kda.cpp:344-345`, `:383-384`). Same polarity for the paged
// attention rows, which are simply not read when there are none.
#ifndef VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_KV_H_
#define VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_KV_H_

#include <cstdint>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_layer.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, GdnStateCache

namespace vllm::glm5_next {

// One decoder layer's binding onto the caches the engine published. Exactly one
// arm is populated, selected by `kind` — the same union shape
// `DecoderLayerWeights` uses, and for the same reason: a layer that read the
// other arm would read a default-constructed index.
struct LayerKvBinding {
  Glm5NextLayerKind kind = Glm5NextLayerKind::kLinearAttention;

  // kDeepseekSparseAttention. PAGED SLOTS in `ModelForwardInput::attn_kv` — the
  // channel's `payload_slots` answer and NOT `Find`'s flat index, see the
  // header's "THE FLAT INDEX IS NOT THE PAGED SLOT" — and the group ids they
  // were PUBLISHED under, read off the channel at the flat index.
  int64_t latent = -1;
  int64_t indexer = -1;
  int32_t latent_group = -1;
  int32_t indexer_group = -1;
  // `!IndexerRoleFor(p, l).skip_topk`. A `shared` layer builds no indexer, so
  // `Attention` never appends to its `indexer_packed` and never validates it
  // (`glm5_next_attn.cpp:353-366` vs `:372-387`). Its side cache therefore
  // stays EMPTY across steps while its `cached_len` advances, and hydrating it
  // to `cached_len` rows would describe history the layer never wrote.
  bool has_own_indexer = false;

  // kLinearAttention. The slot in `ModelForwardInput::gdn_state` the channel
  // published this layer's state at — read off `payload_slots`, not counted.
  int64_t recurrent = -1;
};

// The whole step's binding: one entry per decoder layer plus the per-step facts
// the hydration needs. Built once per forward and thrown away with it.
struct KvBinding {
  std::vector<LayerKvBinding> layers;  // exactly `num_hidden_layers`

  // Request 0's block ids for each PUBLISHED group, indexed by group id, in
  // logical page order. An entry is empty for a group whose table was not
  // gathered (the recurrent group's, which nothing here reads).
  std::vector<std::vector<int32_t>> group_blocks;

  int64_t block_size = 0;
  // Tokens already stored for request 0 before this step
  // (`attn_meta.num_computed_tokens_cpu[0]`). ZERO on a fresh prefill.
  int64_t cached_len = 0;
  // The step's scheduled tokens for request 0.
  int64_t new_tokens = 0;
  // The recurrent state row, from the GDN metadata's remapped slot.
  int64_t state_slot = -1;
};

// The element offset of logical position `pos` inside a FUSED per-token page
// set — the one arithmetic this row uses to address `MLAAttentionSpec` storage.
// See the header comment for why there is no pair-strided sibling.
int64_t PagedRowOffset(const std::vector<int32_t>& blocks, int64_t block_size,
                       int64_t head_size, int64_t pos);

// Resolve the engine's channel into a binding, or refuse BY NAME. Every failure
// mode names what was looked for, what the channel carried, and the issue.
KvBinding ResolveKvBinding(const Glm5NextParams& p,
                           const ModelForwardInput& input);

// Fill `*out` with `num_hidden_layers` layer states hydrated from the engine's
// pages. `out` is resized. A fresh sequence (`cached_len == 0`) leaves every
// state empty, which is each consumer's own fresh-sequence signal.
void LoadCaches(const Glm5NextParams& p, const KvBinding& b,
                const ModelForwardInput& input, std::vector<LayerCache>* out);

// Write this step's NEW rows back into the engine's pages. Only rows
// `[cached_len, cached_len + new_tokens)` move for the paged caches; the
// recurrent states are whole-slot writes because a recurrence has no rows.
void StoreCaches(const Glm5NextParams& p, const KvBinding& b,
                 const std::vector<LayerCache>& caches,
                 const ModelForwardInput& input);

}  // namespace vllm::glm5_next

#endif  // VLLM_MODEL_EXECUTOR_MODELS_GLM5_NEXT_KV_H_
