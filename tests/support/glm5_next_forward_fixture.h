// Shared production forward topology for the GLM-5.3-Flash registry tests.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "support/glm5_next_gguf_fixture.h"
#include "vllm/model_executor/models/glm5_next_kv.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"

namespace glm5_next_forward_fixture {
using namespace glm5_next_fixture;
namespace gn = vllm::glm5_next;

struct Topology {
  static constexpr int64_t kBlockSize = 4;
  static constexpr int64_t kNumBlocks = 8;
  // Deliberately not the identity, and page 0 is not block 0.
  static constexpr int32_t kBlockPerm[kNumBlocks] = {5, 2, 7, 1, 6, 0, 3, 4};

  // The fixture's one DSA layer, and its two published names.
  static constexpr int64_t kDsaLayer = 2;
  static int64_t LatentRow() { return kKvLora; }   // qk_rope_head_dim is 0
  static int64_t IndexerRow() { return 2 * kIdxHeadDim + 1; }
  static int64_t ConvElems() { return 3 * kKdaHeads * kKdaHeadDim * kConvKernel; }
  static int64_t RecElems() { return kKdaHeads * kKdaHeadDim * kKdaHeadDim; }

  // ─── W5b-2d (#2445): THE FLAT CHANNEL, and it is NOT `attn_kv`'s order ────
  //
  // This fixture used to publish TWO names, both paged, with no payload
  // locators — which made `MultiKvCacheIndex::Find` answer an `attn_kv` index
  // and every case below pass against a channel shape the runner STOPPED
  // producing at `9e7621efc`. The runner emits one entry per published cache in
  // PUBLICATION order over ALL groups (`runner.cpp`, the by-name index pass),
  // so the recurrent group lands BETWEEN the two attention groups and the flat
  // index of the indexer side cache is not 1.
  //
  // At this miniature: flat 0 is the MLA latent (paged slot 0), flat 1..3 are
  // the three KDA layers' recurrent states (gdn_state slots 0..2), and flat 4
  // is the indexer side cache (paged slot 1). `Find(indexer) == 4` against
  // `attn_kv.size() == 2` is the published checkpoint's `45` against `22`,
  // scaled down.
  static constexpr size_t kLatentFlat = 0;
  static constexpr size_t kIndexerFlat = 1 + static_cast<size_t>(kLayers - 1);

  vt::DType dtype = vt::DType::kF32;
  std::vector<std::vector<uint8_t>> attn_bytes;
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<std::string> names;
  std::vector<int32_t> group_ids;
  std::vector<int32_t> layer_indices;
  std::vector<uint8_t> payload_kinds;
  std::vector<int32_t> payload_slots;
  std::vector<std::vector<int32_t>> group_bt;
  std::vector<int32_t> group_cols;
  std::vector<std::vector<uint8_t>> conv_bytes;
  std::vector<std::vector<uint8_t>> ssm_bytes;
  std::vector<vllm::GdnStateCache> gdn;
  vllm::MultiKvCacheIndex mk;

  explicit Topology(vt::DType dt = vt::DType::kF32) : dtype(dt) {
    const int64_t elt = static_cast<int64_t>(vt::SizeOf(dtype));
    // group 0 — the MLA latent, and group 2 — the indexer side cache. ONE
    // vector per token each (`MLAAttentionSpec`), so the page is
    // block_size * 1 * head_size and NOT twice that.
    const int64_t rows[2] = {LatentRow(), IndexerRow()};
    const char* suffix[2] = {".self_attn.attn", ".self_attn.indexer.k_cache"};
    for (int i = 0; i < 2; ++i) {
      attn_bytes.emplace_back(
          static_cast<size_t>(kNumBlocks * kBlockSize * rows[i] * elt), 0);
    }
    // THE FLAT CHANNEL, in PUBLICATION order: group 0, then group 1, then group
    // 2 — one pass over the groups, exactly as `runner.cpp` builds it. The
    // paged slot is a RUNNING COUNTER over the paged entries only, which is the
    // whole distinction this fixture exists to carry.
    const auto emit = [&](const std::string& name, int32_t gid, int32_t layer,
                          vllm::KvCachePayload kind, int32_t slot) {
      names.push_back(name);
      group_ids.push_back(gid);
      layer_indices.push_back(layer);
      payload_kinds.push_back(static_cast<uint8_t>(kind));
      payload_slots.push_back(slot);
    };
    emit("model.layers." + std::to_string(kDsaLayer) + suffix[0], 0,
         static_cast<int32_t>(kDsaLayer), vllm::KvCachePayload::kPaged, 0);
    {
      int32_t rslot = 0;
      for (int64_t l = 0; l < kLayers; ++l) {
        if (l == kDsaLayer) continue;
        emit("model.layers." + std::to_string(l) + ".linear_attn", 1,
             static_cast<int32_t>(l), vllm::KvCachePayload::kRecurrent, rslot++);
      }
    }
    emit("model.layers." + std::to_string(kDsaLayer) + suffix[1], 2,
         static_cast<int32_t>(kDsaLayer), vllm::KvCachePayload::kPaged, 1);
    for (int i = 0; i < 2; ++i) {
      vllm::PagedKvCache kv;
      kv.data = attn_bytes[static_cast<size_t>(i)].data();
      kv.dtype = dtype;
      kv.num_blocks = kNumBlocks;
      kv.block_size = kBlockSize;
      kv.num_kv_heads = 1;  // MLA: one vector per token, never a K+V pair
      kv.head_size = rows[i];
      attn_kv.push_back(kv);
    }
    // Three published groups, so three gathered tables — `gather_group_block_tables`
    // walks EVERY published group, the recurrent one included, so an empty entry
    // here would be a shape the runner does not produce.
    //
    // W5b-2d (#2445): group 1's table is the RECURRENT group's and it is
    // deliberately NOT a copy of the attention groups'. On the real model that
    // table is one unified page per sequence, not `kNumBlocks` of them, and the
    // difference is what makes reading `group_ids` at the wrong index fatal
    // instead of invisible: a binding that took the indexer's group id from the
    // PAGED slot rather than the FLAT index lands on group 1 and finds a table
    // one column wide.
    group_bt.assign(3, std::vector<int32_t>(kBlockPerm, kBlockPerm + kNumBlocks));
    group_cols.assign(3, static_cast<int32_t>(kNumBlocks));
    group_bt[1] = std::vector<int32_t>{0};
    group_cols[1] = 1;

    // The recurrent group: one state set per KDA layer, in ASCENDING LAYER
    // ORDER, exactly as `alloc_recurrent_layer_states` pushes them. ONE slot,
    // because the runner reduced `max_num_seqs` to 1 on this model.
    const int64_t kda = kLayers - 1;  // the fixture has one DSA layer
    for (int64_t j = 0; j < kda; ++j) {
      conv_bytes.emplace_back(static_cast<size_t>(ConvElems() * elt), 0);
      ssm_bytes.emplace_back(static_cast<size_t>(RecElems() * 4), 0);
    }
    for (int64_t j = 0; j < kda; ++j) {
      vllm::GdnStateCache gs;
      gs.conv_state = vt::Tensor::Contiguous(
          conv_bytes[static_cast<size_t>(j)].data(), dtype,
          vt::Device{vt::DeviceType::kCPU, 0},
          {1, 3 * kKdaHeads * kKdaHeadDim, kConvKernel});
      // The recurrent half is f32 UNCONDITIONALLY, which is what
      // `MakeGlm5NextKVCache` publishes and what upstream's `kda_state_dtype`
      // returns: the state is a running sum and a bf16 store loses it.
      gs.ssm_state = vt::Tensor::Contiguous(
          ssm_bytes[static_cast<size_t>(j)].data(), vt::DType::kF32,
          vt::Device{vt::DeviceType::kCPU, 0},
          {1, kKdaHeads, kKdaHeadDim, kKdaHeadDim});
      gs.states = {gs.conv_state, gs.ssm_state};
      gdn.push_back(gs);
    }
    Publish();
  }

  // Re-point the channel at the (possibly edited) vectors. Called by every
  // mutator so a case can corrupt one field without rebuilding the object.
  void Publish() {
    for (size_t i = 0; i < attn_kv.size() && i < attn_bytes.size(); ++i)
      attn_kv[i].data = attn_bytes[i].data();
    mk.layer_names = &names;
    mk.group_ids = &group_ids;
    mk.layer_indices = &layer_indices;
    mk.payload_kinds = &payload_kinds;
    mk.payload_slots = &payload_slots;
    mk.group_block_tables = &group_bt;
    mk.group_block_table_cols = &group_cols;
  }

  // The flat KV slot the engine's own walk assigns to logical position `pos`.
  static int64_t Slot(int64_t pos) {
    return static_cast<int64_t>(kBlockPerm[pos / kBlockSize]) * kBlockSize +
           pos % kBlockSize;
  }

  void ZeroPages() {
    for (std::vector<uint8_t>& b : attn_bytes) std::fill(b.begin(), b.end(), 0);
  }
};


// A `ModelForwardInput` over one sequence, built the way the runner builds one.
struct Step {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
  std::vector<int32_t> logits_indices;
  vllm::v1::CommonAttentionMetadata attn_meta{};
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vllm::HfConfig config{};
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  int num_reqs = 1;
  // Non-owning: the caller's topology outlives the step (see `Bind`).
  const vllm::MultiKvCacheIndex* multi_kv = nullptr;
  // A step that is not handed one owns a fresh topology, so a single-step case
  // reads as it did before W5b-2c while still entering through the real
  // channel. A case that wants HISTORY binds the same topology to two steps.
  std::unique_ptr<Topology> own;

  explicit Step(std::vector<int32_t> ids, std::vector<int32_t> want = {},
                int64_t computed = 0)
      : token_ids(std::move(ids)), logits_indices(std::move(want)) {
    const int64_t T = static_cast<int64_t>(token_ids.size());
    positions.resize(token_ids.size());
    for (size_t i = 0; i < positions.size(); ++i)
      positions[i] = static_cast<int32_t>(computed + static_cast<int64_t>(i));
    // What the runner fills for one request: the history it already has, the
    // sequence length after this step, and the flat slot of every new token.
    attn_meta.num_reqs = 1;
    attn_meta.num_actual_tokens = static_cast<int>(T);
    attn_meta.num_computed_tokens_cpu = {static_cast<int32_t>(computed)};
    attn_meta.seq_lens_cpu = {static_cast<int32_t>(computed + T)};
    attn_meta.seq_lens = attn_meta.seq_lens_cpu;
    attn_meta.query_start_loc = {0, static_cast<int32_t>(T)};
    attn_meta.query_start_loc_cpu = attn_meta.query_start_loc;
    for (int64_t t = 0; t < T; ++t)
      attn_meta.slot_mapping.push_back(Topology::Slot(computed + t));
    // The COMPACT per-sequence recurrent slot the runner remapped block-table
    // column 0 to — never the raw block id.
    gdn_meta.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    own = std::make_unique<Topology>();
    Bind(*own);
  }

  // Point the step at a topology. Kept separate from the constructor so a case
  // can hand the SAME topology to two consecutive steps, which is the whole
  // point of a cache.
  Step& Bind(Topology& t) {
    attn_kv = t.attn_kv;
    gdn_state = t.gdn;
    multi_kv = &t.mk;
    return *this;
  }

  vllm::ModelForwardInput Get() {
    vllm::ModelForwardInput in{.token_ids = token_ids,
                               .positions = positions,
                               .attn_meta = attn_meta,
                               .gdn_meta = gdn_meta,
                               .attn_kv = attn_kv,
                               .gdn_state = gdn_state,
                               .config = config,
                               .queue = queue,
                               .logits_indices = logits_indices,
                               .num_reqs = num_reqs};
    in.multi_kv = multi_kv;
    return in;
  }
};

}  // namespace glm5_next_forward_fixture
