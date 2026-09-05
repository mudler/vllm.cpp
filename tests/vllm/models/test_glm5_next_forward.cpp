// GLM-5.3-Flash W5b-2b gate — THE ENGINE BINDING, and the reachability proof
// this whole row has been owing since W2.
//
// Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation, issue #2241,
// `.agents/specs/glm5-next-flash.md` section W5b-2b and `## Owed` O26.
//
// ─── WHY THIS FILE IS THE DELIVERABLE ───────────────────────────────────────
//
// O15, O16, O17, O23, O25 and O26 all say the same thing about six different
// files: the KDA arm, the mHC bricks, the DSA indexer, the MoE block, the
// attention plus bridge, and the decoder layer are each gated and NONE of them
// is reached from a production entry point. `.agents/reachability.md` is
// explicit that "an intermediate hop that is itself unreached does not carry",
// so W5b-2a assembling five dead ends into one changed the SHAPE of the debt
// and not its existence.
//
// This suite is what changes it. **Every case here enters through
// `ModelRegistry::Forward`** — the production entry point, reached the way a
// user reaches it: the GGUF architecture dispatch builds the config, the
// registry resolves the architecture, the registration's own `load_weights`
// hook loads the tower, and the registration's own `forward` hook runs. Nothing
// here calls `Glm5NextHostForward` or `TextModelForward` to produce the value
// under test.
//
// THE REACHABILITY MUTATION (`.agents/reachability.md`): deleting the
// `glm5_next::Glm5NextHostForward(...)` call in
// `ForwardGlm5NextForConditionalGeneration` and returning an empty
// `ForwardLogits{}` REDS this suite. Recorded in the spec with the measured
// counts.
//
// ─── WHAT EACH GROUP PROVES, AND WHY A CHEAPER ONE WOULD NOT ────────────────
//
//  (1) THE HOOK RUNS AND RETURNS REAL LOGITS. `REQUIRE_NOTHROW` plus
//      `rows != 0` would pass a hook that returned a zero-filled carrier of the
//      right shape, so the values are read: finite, not constant, and
//      responsive to the token ids.
//  (2) THE LOGITS ARE THE RESIDENT TOWER'S. An independently assembled
//      `TextModelWeights` — every layer bridged by hand, the expert banks
//      RESIDENT rather than sourced — is run through `TextModelForward` and the
//      production output must equal it EXACTLY. This is what catches a layer
//      source that returned the wrong layer, a bridge that mixed the KDA and
//      DSA arms, an embed gather that read row 0, and a head that assembled its
//      columns in the wrong order. None of those throws; all of them produce
//      finite, fluent, wrong logits.
//  (3) THE `lm_head` CHUNKING IS EXERCISED. At any geometry small enough to run
//      in a test the default chunk size gives ONE chunk, so the loop would
//      never be entered twice. The chunk size is a parameter for exactly that
//      reason and the case drives it down to one row per chunk.
//  (4) THE NARROW REFUSALS ARE LIVE. A non-CPU queue and a multi-request step
//      are refused BY NAME, because both would otherwise be a crash or a
//      cross-request attention that no gate on this fleet could detect.
//
// The substrate is the synthetic `glm5next` GGUF miniature W5c gates its loader
// against and W5b-1 gates its bridge against. The published artifact is 101.25
// GiB in four shards and its forward is not runnable in CI; the spec's §Gates
// records that no end-to-end token gate for this model exists or can exist on
// this fleet, and this file makes no token claim.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>  // W9c-3b: the shadow backend's memcpy/memset
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "support/glm5_next_gguf_fixture.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_forward.h"
#include "vllm/model_executor/models/glm5_next_kv.h"  // W9c-3b (#2480)
#include "vllm/model_executor/models/glm5_next_layer.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"      // ForwardLogits, *KvCache
#include "vllm/v1/attention/backend.h"               // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"     // GDNAttentionMetadata
#include "vt/backend.h"  // W9c-3a: vt::TryGetBackend
#include "vt/op_provider.h"  // W9c-3a: vt::OpRegistered
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

namespace {

using gguf_test::TempFile;
using namespace glm5_next_fixture;  // NOLINT(build/namespaces) — the fixture IS this suite's vocabulary

namespace gn = vllm::glm5_next;

// ─── the ENGINE's published KV topology, built the way the runner builds it ──
//
// W5b-2c (#2348). Every case in this file now enters the forward with the same
// three-group topology `MakeGlm5NextKVCache` publishes and the runner
// allocates, because that is what the production path always carries: a model
// that publishes three groups gets `ModelForwardInput::multi_kv` set on EVERY
// step (`runner.cpp`, `if (multi_cache_topology_) forward_input.multi_kv = ...`).
// A `Step` without one would test a shape the engine never produces.
//
// THE BLOCK IDS ARE A PERMUTATION ON PURPOSE. An implementation that ignored
// the gathered block table and addressed page `p` at block `p` is correct only
// for the identity table, and the identity table is what a hand-built fixture
// reaches for. `kBlockPerm` makes that shortcut wrong at the FIRST page.
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

const vllm::Glm5NextWeights& Weights(const std::unique_ptr<vllm::LoadedModel>& m) {
  return vllm::ModelAs<vllm::Glm5NextLoadedModel>(
             *m, "Glm5NextForConditionalGeneration")
      .weights();
}

// ─── the INDEPENDENT reference ──────────────────────────────────────────────
//
// A fully RESIDENT tower: every layer bridged by hand, every expert bank
// decoded whole and re-fused into the seam's `[E, 2I, H]` order. At the
// published geometry this is 426.72 GiB and cannot exist, which is precisely
// why production streams; at the fixture's geometry it is a few hundred
// kilobytes and it is the only oracle available for the streamed path's
// composition.
//
// It is NOT built through `Glm5NextGgufLayerSource`, so the layer INDEXING, the
// arm SELECTION and the per-expert source — the three things W5b-2b adds — are
// not shared between the two sides.
gn::TextModelWeights ResidentTower(const vllm::Glm5NextWeights& w) {
  const vllm::Glm5NextParams& p = w.params;
  gn::TextModelWeights out;
  out.params = p;
  out.norm = gn::DecodeOwnedTensorToF32(w.norm, "output_norm.weight");
  const gn::MoeDims md = gn::MoeDimsFrom(p);
  for (size_t i = 0; i < w.layers.size(); ++i) {
    const vllm::Glm5NextLayerWeights& src = w.layers[i];
    const std::string what = "ref.blk." + std::to_string(i);
    gn::DecoderLayerWeights d;
    d.attn_kind = src.is_linear_attention
                      ? vllm::Glm5NextLayerKind::kLinearAttention
                      : vllm::Glm5NextLayerKind::kDeepseekSparseAttention;
    d.mlp_kind = src.is_dense_mlp ? vllm::Glm5NextMlpKind::kDense
                                  : vllm::Glm5NextMlpKind::kSparse;
    d.input_layernorm =
        gn::DecodeOwnedTensorToF32(src.input_layernorm, what + ".attn_norm");
    d.post_attention_layernorm =
        gn::DecodeOwnedTensorToF32(src.post_attention_layernorm, what + ".ffn_norm");
    d.attn_hc = gn::BridgeMhcSite(src.attn_hc, p.mhc, p.hidden_size, what + ".hc_attn");
    d.ffn_hc = gn::BridgeMhcSite(src.mlp_hc, p.mhc, p.hidden_size, what + ".hc_ffn");
    if (d.attn_kind == vllm::Glm5NextLayerKind::kLinearAttention) {
      d.kda = gn::BridgeKdaLayer(src.kda, gn::KdaDimsFrom(p));
    } else {
      d.dsa = gn::BridgeDsaLayer(src.mla, gn::MlaDimsFrom(p), gn::IndexerDimsFrom(p));
    }
    if (d.mlp_kind == vllm::Glm5NextMlpKind::kDense) {
      d.dense_mlp = gn::BridgeMlp(src.dense_mlp, p.hidden_size, p.intermediate_size,
                                  what + ".ffn");
    } else {
      d.moe = gn::BridgeMoeLayer(src.moe, md, what + ".ffn");
      const std::vector<float> ge =
          gn::DecodeOwnedTensorToF32(src.moe.gate_exps, what + ".gate_exps");
      const std::vector<float> ue =
          gn::DecodeOwnedTensorToF32(src.moe.up_exps, what + ".up_exps");
      d.moe.expert_down =
          gn::DecodeOwnedTensorToF32(src.moe.down_exps, what + ".down_exps");
      const int64_t E = md.n_routed_experts, I = md.moe_intermediate_size,
                    H = md.hidden_size;
      d.moe.expert_gate_up.resize(static_cast<size_t>(E * 2 * I * H));
      for (int64_t e = 0; e < E; ++e) {
        std::copy_n(ge.begin() + e * I * H, I * H,
                    d.moe.expert_gate_up.begin() + e * 2 * I * H);
        std::copy_n(ue.begin() + e * I * H, I * H,
                    d.moe.expert_gate_up.begin() + e * 2 * I * H + I * H);
      }
      d.moe.expert_source = nullptr;  // RESIDENT, deliberately
    }
    out.layers.push_back(std::move(d));
  }
  return out;
}

// The reference logits for one step, assembled from the resident tower.
std::vector<float> ReferenceLogits(const vllm::Glm5NextWeights& w,
                                   const std::vector<int32_t>& ids,
                                   const std::vector<int32_t>& want_rows) {
  const vllm::Glm5NextParams& p = w.params;
  const int64_t H = p.hidden_size, V = p.vocab_size;
  const int64_t T = static_cast<int64_t>(ids.size());
  std::vector<float> embeds(static_cast<size_t>(T * H));
  const std::vector<float> table =
      gn::DecodeOwnedTensorToF32(w.embed_tokens, "ref.token_embd");
  for (int64_t t = 0; t < T; ++t) {
    std::copy_n(table.begin() + static_cast<int64_t>(ids[static_cast<size_t>(t)]) * H,
                H, embeds.begin() + t * H);
  }
  const std::vector<uint8_t> mask(static_cast<size_t>(T), 1);
  gn::TextModelWeights tower = ResidentTower(w);
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  const std::vector<float> hidden =
      gn::TextModelForward(tower, embeds, mask, /*batch=*/1, /*seq_len=*/T,
                           /*caches=*/nullptr, q);
  const std::vector<float> head = gn::DecodeOwnedTensorToF32(
      w.tied_word_embeddings ? w.embed_tokens : w.lm_head, "ref.output");
  std::vector<int64_t> rows;
  if (want_rows.empty()) {
    for (int64_t t = 0; t < T; ++t) rows.push_back(t);
  } else {
    for (int32_t r : want_rows) rows.push_back(r);
  }
  std::vector<float> logits(rows.size() * static_cast<size_t>(V), 0.0F);
  for (size_t r = 0; r < rows.size(); ++r) {
    const float* hr = &hidden[static_cast<size_t>(rows[r] * H)];
    for (int64_t o = 0; o < V; ++o) {
      double acc = 0.0;
      const float* wo = &head[static_cast<size_t>(o * H)];
      for (int64_t i = 0; i < H; ++i) acc += static_cast<double>(wo[i]) * hr[i];
      logits[r * static_cast<size_t>(V) + static_cast<size_t>(o)] =
          static_cast<float>(acc);
    }
  }
  return logits;
}

// The largest |a - b| over two runs, with BOTH sides guarded on `isfinite` and
// the non-finite count reported separately.
//
// **This guard is not defensive style; it is the finding W5b-2a recorded.** An
// all-NaN forward makes `NaN > max` FALSE for every max, so a running maximum
// never moves off zero and an entirely broken forward reads as a PERFECT match.
// That mutation survived 1647 of 1647 assertions on the first pass. A
// non-finite value is therefore an INFINITE gap here, and the count is returned
// so a failure distinguishes "wrong number" from "not a number".
struct Gap {
  double max_abs = 0.0;
  int nonfinite = 0;
};

Gap MaxGap(const std::vector<float>& a, const std::vector<float>& b) {
  Gap g;
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
      ++g.nonfinite;
      g.max_abs = std::numeric_limits<double>::infinity();
      continue;
    }
    const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > g.max_abs) g.max_abs = d;
  }
  return g;
}

}  // namespace

// ═══ (1) the hook runs, through the production entry point ══════════════════

TEST_CASE("glm5_next forward: ModelRegistry::Forward REACHES the model") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  REQUIRE(model != nullptr);
  const vllm::Glm5NextWeights& w = Weights(model);
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));

  // Four tokens, LAST position only — the decode shape.
  Step step({3, 11, 7, 20}, {3});
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());

  // Shape, from the carrier's own derivation (`HostLogits` divides
  // `host.size()` by `vocab`), so a hook that returned the wrong number of
  // floats reports the wrong `rows` rather than passing.
  CHECK(out.vocab == kVocab);
  CHECK(out.rows == 1);
  CHECK_FALSE(out.on_device());
  REQUIRE(out.host.size() == static_cast<size_t>(kVocab));

  // The VALUES are read. A carrier of zeros has the right shape.
  int nonfinite = 0;
  for (float v : out.host) if (!std::isfinite(v)) ++nonfinite;
  CHECK(nonfinite == 0);
  const auto mm = std::minmax_element(out.host.begin(), out.host.end());
  CHECK(*mm.first != *mm.second);
  MESSAGE("logits span [" << *mm.first << ", " << *mm.second << "] over "
          << out.host.size() << " vocab entries");

  // And the forward DEPENDS on the tokens: a hook that ignored `token_ids`
  // would return the same vector for a different prompt.
  Step other({20, 7, 11, 3}, {3});
  const vllm::ForwardLogits out2 = vllm::ModelRegistry::Forward(*model, other.Get());
  REQUIRE(out2.host.size() == out.host.size());
  CHECK(MaxGap(out.host, out2.host).max_abs > 0.0);
}

TEST_CASE("glm5_next forward: an EMPTY logits_indices means every row") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Step step({1, 2, 3});
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());
  CHECK(out.rows == 3);
  CHECK(out.host.size() == static_cast<size_t>(3 * kVocab));

  // ...and an out-of-range index is refused BY NAME rather than read.
  Step bad({1, 2, 3}, {7});
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, bad.Get()),
                       doctest::Contains("logits index"), std::runtime_error);
  Step bad_tok({1, 2, kVocab + 4});
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, bad_tok.Get()),
                       doctest::Contains("token id"), std::runtime_error);
}

// ═══ (2) the values are the RESIDENT tower's, exactly ═══════════════════════

TEST_CASE("glm5_next forward: the STREAMED forward equals the RESIDENT tower") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Weights(model);

  const std::vector<int32_t> ids{5, 9, 2, 14};
  Step step(ids);
  const vllm::ForwardLogits got = vllm::ModelRegistry::Forward(*model, step.Get());
  const std::vector<float> want = ReferenceLogits(w, ids, {});

  REQUIRE(got.host.size() == want.size());
  const Gap gap = MaxGap(got.host, want);
  CHECK(gap.nonfinite == 0);
  // EXACT. Both sides run the same host arithmetic on the same floats — the
  // grouped expert visit writes each `[t, j]` slot independently, and the
  // streaming source hands the block the same values the bank holds — so a
  // tolerance here would hide a defect rather than absorb noise.
  CHECK(gap.max_abs == 0.0);
  // The comparison is discriminating: the reference is not a constant.
  const auto mm = std::minmax_element(want.begin(), want.end());
  CHECK(*mm.first != *mm.second);
  MESSAGE("max |streamed - resident| = " << gap.max_abs << " over "
          << want.size() << " logits; reference span [" << *mm.first << ", "
          << *mm.second << "]");
}

TEST_CASE("glm5_next forward: the layer source holds ONE layer, not a tower") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Weights(model);

  gn::Glm5NextGgufLayerSource src(w);
  CHECK(src.size() == static_cast<int64_t>(kLayers));
  CHECK(src.bridged() == 0);

  // Each layer's `attn_norm` is a DISTINCT sequence in the fixture (`NormTag`),
  // so "the source returned layer i" is falsifiable: a source that always
  // handed back its first slot would fail on the second layer.
  std::vector<std::vector<float>> norms;
  for (int64_t i = 0; i < src.size(); ++i) {
    const gn::DecoderLayerWeights& d = src.Layer(i);
    CHECK(src.bridged() == i + 1);
    // The ARM matches the file's own schedule, which the fixture puts at index
    // 2 precisely where an `idx % 4 == 3` stride would not.
    const bool want_kda = IsKda(i);
    CHECK((d.attn_kind == vllm::Glm5NextLayerKind::kLinearAttention) == want_kda);
    CHECK((d.mlp_kind == vllm::Glm5NextMlpKind::kDense) == IsDense(i));
    // The unused arm is EMPTY, which is what says one layer was bridged and not
    // both arms of one.
    if (want_kda) {
      CHECK_FALSE(d.kda.q_proj.empty());
      CHECK(d.dsa.mla.q_a_proj.empty());
    } else {
      CHECK_FALSE(d.dsa.mla.q_a_proj.empty());
      CHECK(d.kda.q_proj.empty());
    }
    // The expert BANKS are never bridged; a sparse layer carries a source.
    if (d.mlp_kind == vllm::Glm5NextMlpKind::kSparse) {
      CHECK(d.moe.expert_gate_up.empty());
      CHECK(d.moe.expert_down.empty());
      CHECK(d.moe.expert_source != nullptr);
    } else {
      CHECK(d.moe.expert_source == nullptr);
    }
    norms.push_back(d.input_layernorm);
    CHECK(src.slot_f32_bytes() > 0);

    // THE TWO mHC SITES ARE MAPPED DIRECTLY, and this assertion exists because
    // the end-to-end comparison CANNOT see the mapping. Swapping `attn_hc` and
    // `mlp_hc` inside the source (mutation M12) leaves every logit in this
    // suite BIT-IDENTICAL, measured, not assumed. The reason is the fixture,
    // not the port: its mHC `fn` payloads are ramps in the hundreds and
    // thousands, so `F.linear(normed, fn) + base` saturates every sigmoid gate
    // and the Sinkhorn projection converges to the same matrix from either
    // site, and the swap becomes arithmetically invisible downstream.
    //
    // A gate that could only see the swap through the logits would therefore be
    // a mute switch here. So the mapping is asserted STRUCTURALLY, against the
    // loader's own two tensors, and the two are asserted to DIFFER so the
    // equality above is a fact and not a tautology.
    const gn::HcSite want_attn = gn::BridgeMhcSite(
        w.layers[static_cast<size_t>(i)].attn_hc, w.params.mhc,
        w.params.hidden_size, "want.attn");
    const gn::HcSite want_ffn = gn::BridgeMhcSite(
        w.layers[static_cast<size_t>(i)].mlp_hc, w.params.mhc,
        w.params.hidden_size, "want.ffn");
    CHECK(want_attn.fn != want_ffn.fn);
    CHECK(want_attn.base != want_ffn.base);
    CHECK(d.attn_hc.fn == want_attn.fn);
    CHECK(d.attn_hc.base == want_attn.base);
    CHECK(d.attn_hc.scale == want_attn.scale);
    CHECK(d.ffn_hc.fn == want_ffn.fn);
    CHECK(d.ffn_hc.base == want_ffn.base);
    CHECK(d.ffn_hc.scale == want_ffn.scale);
  }
  // Every layer's norm differs from every other's, so the loop above compared
  // things that can differ.
  for (size_t a = 0; a + 1 < norms.size(); ++a) {
    for (size_t b = a + 1; b < norms.size(); ++b) {
      INFO("layers ", a, " and ", b);
      CHECK(norms[a] != norms[b]);
    }
  }

  // RE-ASKING for the layer already in the slot does NOT re-bridge...
  const int64_t before = src.bridged();
  src.Layer(src.size() - 1);
  CHECK(src.bridged() == before);
  // ...and asking for another one does, which together say the source holds one
  // slot rather than a map that grew.
  src.Layer(0);
  CHECK(src.bridged() == before + 1);

  // The peak is ONE layer. Asserted as a bound rather than a value, because the
  // fixture's geometry is not the published one; the published numbers are
  // pinned in `test_glm5_next_bridge.cpp`.
  CHECK(src.slot_f32_bytes() > 0);
  MESSAGE("one fixture layer costs " << src.slot_f32_bytes() << " f32 bytes");
}

// ═══ (3) the lm_head chunking ══════════════════════════════════════════════

TEST_CASE("glm5_next forward: the lm_head CHUNK boundary changes nothing") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Weights(model);
  const std::vector<int32_t> ids{4, 17};
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // The DEFAULT gives ONE chunk at this geometry — 64 MiB against a 128-byte
  // row — so a test that could only take the default would never enter the
  // loop twice and a chunk-offset defect would be invisible.
  const int64_t row_bytes = kH * 4;
  CHECK(gn::kLmHeadChunkBytes / row_bytes > kVocab);

  const std::vector<float> one = gn::Glm5NextHostForward(w, ids, {}, q, /*caches=*/nullptr);
  // ONE ROW per chunk: 32 chunks over the fixture's vocab.
  const std::vector<float> many =
      gn::Glm5NextHostForward(w, ids, {}, q, /*caches=*/nullptr, row_bytes);
  const Gap gap = MaxGap(one, many);
  CHECK(gap.nonfinite == 0);
  CHECK(gap.max_abs == 0.0);
  // A chunked head that wrote its columns at the wrong offsets would still be
  // finite and the right size, so the values are checked against the
  // independent reference too, not only against each other.
  const Gap ref = MaxGap(many, ReferenceLogits(w, ids, {}));
  CHECK(ref.nonfinite == 0);
  CHECK(ref.max_abs == 0.0);

  CHECK_THROWS_AS(gn::Glm5NextHostForward(w, ids, {}, q, /*caches=*/nullptr, 0), std::runtime_error);
}

TEST_CASE("glm5_next forward: a TIED head reads the embedding table") {
  // `Glm5NextWeights::tied_word_embeddings` is read off the FILE — llama.cpp's
  // writer states a tie by OMITTING `output.weight` — so the tied arm is
  // reachable from a real container and is gated here rather than described.
  FixtureOpts o;
  o.tie_lm_head = true;
  TempFile f(BuildFixture(o));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextWeights& w = Weights(model);
  REQUIRE(w.tied_word_embeddings);
  REQUIRE(w.lm_head.bytes.empty());

  const std::vector<int32_t> ids{6, 13};
  Step step(ids);
  const vllm::ForwardLogits got = vllm::ModelRegistry::Forward(*model, step.Get());
  const Gap gap = MaxGap(got.host, ReferenceLogits(w, ids, {}));
  CHECK(gap.nonfinite == 0);
  CHECK(gap.max_abs == 0.0);

  // ...and the tied result DIFFERS from the untied one, so "tied" is not a
  // label the forward ignored.
  TempFile f2(BuildFixture());
  const vllm::GgufFile g2 = vllm::GgufFile::Open(f2.path());
  std::unique_ptr<vllm::LoadedModel> m2 = LoadThroughRegistry(g2);
  const vllm::ForwardLogits untied = vllm::ModelRegistry::Forward(*m2, step.Get());
  REQUIRE(untied.host.size() == got.host.size());
  CHECK(MaxGap(got.host, untied.host).max_abs > 0.0);
}

// ═══ (4) the narrow refusals ═══════════════════════════════════════════════

TEST_CASE("glm5_next forward W9c-3a: the queue is SPLIT, and a device that is "
          "neither CPU nor CUDA is refused BY NAME") {
  // W9c-3a replaced the blanket non-CPU refusal. The premise it stood on is
  // unchanged and is now asserted one level down (`MoeExpertsKeepQuant`'s host
  // arm refuses a non-CPU queue): nearly every buffer on this path is a host
  // `std::vector<float>`, and the ops dispatch on the queue's device. What
  // changed is that ONE arm can put its operands on a device, so the forward
  // interposes a CPU queue for the rest instead of refusing the step.
  //
  // Two things must still be refused, and this case is both of them.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);

  SUBCASE("a device with no provider for the two grouped ops") {
    // kMETAL registers neither `kMoeGateUpSwiGLUGrouped` nor
    // `kMatmulBTQuantGrouped`. Refusing here names this model, both ops and
    // which of them is missing; letting it through would throw from inside an
    // op about a device type, several frames from anything a reader could act
    // on.
    //
    // The refusal asks the OP TABLE and names no device, which is what
    // `scripts/check-device-leakage.py` requires of the device-agnostic layer
    // and is why this case asserts on the provider words rather than on
    // "--device cuda".
    Step step({1, 2});
    step.queue = vt::Queue{vt::Device{vt::DeviceType::kMETAL, 0}, nullptr};
    REQUIRE_FALSE(vt::OpRegistered(vt::OpId::kMoeGateUpSwiGLUGrouped,
                                   vt::DeviceType::kMETAL));
    CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                         doctest::Contains("routed-expert keep-quant GEMM"),
                         std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                         doctest::Contains("provider: NO"), std::runtime_error);
    CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                         doctest::Contains("#2464"), std::runtime_error);
  }

  SUBCASE("the device arm is OPT-IN, so the DEFAULT refuses a device queue") {
    // W9c-3a's device split defaults OFF after both `--device cuda` legs on the
    // real artifact died with SIGSEGV (spec O46). The default must therefore be
    // a REFUSAL, byte-for-byte the behaviour of the tree before this wave --
    // turning a clean error into a segfault is strictly worse for a user.
    //
    // This case only has something to assert where the op table admits the
    // device, because the op-table refusal above is ordered first and wins.
    // On a CPU-only build nothing registers the pair, so the guard under test
    // is unreachable and the case says so rather than asserting a message it
    // would get for the wrong reason.
    const vt::DeviceType dev = vt::DeviceType::kCUDA;
    const bool pair_here = vt::OpRegistered(vt::OpId::kMoeGateUpSwiGLUGrouped, dev) &&
                           vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, dev) &&
                           vt::TryGetBackend(vt::Device{dev, 0}) != nullptr;
    if (!pair_here) {
      MESSAGE("no CUDA provider for the grouped pair: the OPT-IN guard is "
              "unreachable on this build and is gated on dgx:gpu0 instead");
    } else if (std::getenv("VT_GLM5_NEXT_DEVICE_EXPERTS") != nullptr) {
      MESSAGE("VT_GLM5_NEXT_DEVICE_EXPERTS is set in this environment: the "
              "default-off guard cannot be observed here");
    } else {
      Step step({1, 2});
      step.queue = vt::Queue{vt::Device{dev, 0}, nullptr};
      CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                           doctest::Contains("OPT-IN"), std::runtime_error);
      CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                           doctest::Contains("SIGSEGV"), std::runtime_error);
      CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                           doctest::Contains("--device cpu"), std::runtime_error);
    }
  }

  SUBCASE("the probe follows the OP TABLE and not a device name") {
    // The discriminating half of the previous case. A refusal that hardcoded a
    // device name would answer identically for kMETAL and differently for a
    // device that DOES register the pair -- so the assertion is that the
    // predicate and the op table agree, whatever this build registered.
    //
    // On a CPU-only build no non-CPU device registers the pair and every such
    // queue is refused. On a CUDA build kCUDA registers both and the step
    // proceeds, which is what the `dgx:gpu0` end-to-end leg measures and what
    // this host lane cannot.
    const vt::DeviceType dev = vt::DeviceType::kCUDA;
    const bool pair_here = vt::OpRegistered(vt::OpId::kMoeGateUpSwiGLUGrouped, dev) &&
                           vt::OpRegistered(vt::OpId::kMatmulBTQuantGrouped, dev) &&
                           vt::TryGetBackend(vt::Device{dev, 0}) != nullptr;
    Step step({1, 2});
    step.queue = vt::Queue{vt::Device{dev, 0}, nullptr};
    if (!pair_here) {
      CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                           doctest::Contains("routed-expert keep-quant GEMM"),
                           std::runtime_error);
    } else {
      // The op-table refusal is correctly NOT taken here. The step is still
      // refused, by the OPT-IN guard the subcase above covers, so this branch
      // asserts only that whatever fires is NOT the op-table one -- caught by
      // hand, because doctest::Contains has no negation.
      Step s2({1, 2});
      s2.queue = vt::Queue{vt::Device{dev, 0}, nullptr};
      std::string what;
      try {
        vllm::ModelRegistry::Forward(*model, s2.Get());
      } catch (const std::exception& e) {
        what = e.what();
      }
      CHECK(what.find("cannot run the one primitive") == std::string::npos);
    }
  }

  SUBCASE("a CPU queue is NOT refused, so the split is not a blanket") {
    Step step({1, 2});
    CHECK_NOTHROW(vllm::ModelRegistry::Forward(*model, step.Get()));
  }
}

TEST_CASE("glm5_next forward: a MULTI-REQUEST step is refused BY NAME") {
  // The house pattern (`nemotron_h_registry.cpp`, `kimi_linear_forward.cpp`)
  // takes `token_ids` as one sequence whatever `num_reqs` says. For this model
  // that silently attends ACROSS the request boundary and emits fluent wrong
  // text no gate on this fleet could detect, so the divergence is deliberate
  // and it is in the safe direction. Ragged batching is owed.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Step step({1, 2, 3, 4});
  step.num_reqs = 2;
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                       doctest::Contains("SINGLE-SEQUENCE"), std::runtime_error);
  // ...and ONE request is not refused, so the guard is not a blanket.
  step.num_reqs = 1;
  CHECK_NOTHROW(vllm::ModelRegistry::Forward(*model, step.Get()));
}

TEST_CASE("glm5_next forward: a FOREIGN handle is refused by the DOWNCAST") {
  // `ModelAs<...>` comes FIRST now that there is a forward to open the handle
  // FOR, which is exactly the condition the old blanket refusal named. A bare
  // `static_cast` down the hierarchy is undefined behaviour on an object that
  // is not really this type (#775, #730), so the handle is checked and the
  // refusal names THIS architecture rather than reporting something generic.
  const std::vector<std::string> archs{"Glm5NextForConditionalGeneration"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(archs);
  struct Foreign final : vllm::LoadedModel {
    explicit Foreign(const vllm::ModelRegistration& r) : LoadedModel(r) {}
  };
  Foreign foreign(reg);
  Step step({1});
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(foreign, step.Get()),
                       doctest::Contains("Glm5NextForConditionalGeneration"),
                       std::runtime_error);
}

// ═══ (5) W5b-2c — the engine's KV topology is CONSUMED (#2348) ══════════════
//
// O28 measured the guard that stood above this row's hook and named what it
// cost: `ModelRegistry::Forward` refused every step because no forward consumed
// a cache set keyed by layer name. These cases are what makes it stop refusing,
// and what makes that safe.
//
// THE ASSERTION WORTH HAVING is the first one below. A cached forward that
// silently ignored its history would still return finite, correctly shaped,
// fluent logits — it would simply be attending to nothing — so the gate is that
// prefill(5) + continue(3) agrees with a one-shot forward over the same eight
// tokens, on the CONTINUATION's rows. Everything else here defends that.

namespace {
// The tail-row gap between two runs' logits over the same final positions.
Gap TailGap(const vllm::ForwardLogits& cont, const vllm::ForwardLogits& whole) {
  REQUIRE(cont.rows == whole.rows);
  REQUIRE(cont.vocab == whole.vocab);
  return MaxGap(cont.host, whole.host);
}
}  // namespace

TEST_CASE("glm5_next W5b-2c: prefill + continue AGREES with the one-shot tail") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const std::vector<int32_t> ids{3, 11, 7, 20, 2, 15, 9, 4};
  const int64_t kPrefill = 5, kCont = 3;

  // ── the CACHED path, two steps over ONE topology ────────────────────────
  Topology topo;
  Step s1(std::vector<int32_t>(ids.begin(), ids.begin() + kPrefill), {},
          /*computed=*/0);
  s1.Bind(topo);
  const vllm::ForwardLogits p1 = vllm::ModelRegistry::Forward(*model, s1.Get());
  CHECK(p1.rows == kPrefill);

  Step s2(std::vector<int32_t>(ids.begin() + kPrefill, ids.end()), {},
          /*computed=*/kPrefill);
  s2.Bind(topo);
  const vllm::ForwardLogits cont = vllm::ModelRegistry::Forward(*model, s2.Get());
  REQUIRE(cont.rows == kCont);

  // ── the ONE-SHOT path, a fresh topology, the LAST kCont rows ───────────
  Step whole_step(ids, {5, 6, 7}, /*computed=*/0);
  const vllm::ForwardLogits whole =
      vllm::ModelRegistry::Forward(*model, whole_step.Get());
  REQUIRE(whole.rows == kCont);

  const Gap gap = TailGap(cont, whole);
  CHECK(gap.nonfinite == 0);
  MESSAGE("cached-vs-one-shot tail gap: " << gap.max_abs);

  // EXACT, and that is the strongest assertion this comparison can carry rather
  // than a tolerance chosen to fit. Both sides run the same dot products over
  // the same values in the same order: `ExpandKv` is token-wise under NoPE, so
  // `ExpandKv(a ++ b) == ExpandKv(a) ++ ExpandKv(b)` holds exactly (asserted on
  // real values by `test_glm5_next_layer`); the KDA conv and delta recurrence
  // step per token; the indexer's causal window is `current_length - q_length +
  // s`, which is the same absolute position either way; and the MoE router, the
  // mHC folds and the head are per token. W5b-2a gated its version of this at a
  // tolerance because it compared against the ORACLE, whose own cached and
  // uncached runs differ by 3.01e-06. This one compares our cached path against
  // our own one-shot, and there is nothing left to be tolerant about. Measured:
  // 0.0, against 0.015625 for the M1 mutant that passes `caches = nullptr`.
  CHECK(gap.max_abs == 0.0);

  // AND THE COMPARISON IS NOT VACUOUS: a different prefix moves the same tail.
  std::vector<int32_t> other = ids;
  other[0] = 21;
  other[1] = 1;
  other[2] = 30;
  Step other_step(other, {5, 6, 7}, /*computed=*/0);
  const vllm::ForwardLogits other_whole =
      vllm::ModelRegistry::Forward(*model, other_step.Get());
  const Gap control = TailGap(cont, other_whole);
  CHECK(control.nonfinite == 0);
  MESSAGE("control (different prefix) tail gap: " << control.max_abs);
  REQUIRE(control.max_abs > 0.0);
}

TEST_CASE("glm5_next W5b-2c: the continuation READS the engine's pages") {
  // The strongest statement this suite can make about hydration: corrupt the
  // pages between the two steps and the continuation must change. A forward
  // that hydrated nothing — or that wrote the pages and then ignored them —
  // returns the same logits either way, finite and fluent.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const std::vector<int32_t> ids{3, 11, 7, 20, 2, 15, 9, 4};
  const int64_t kPrefill = 5;

  const auto run = [&](bool zero_between) {
    Topology topo;
    Step s1(std::vector<int32_t>(ids.begin(), ids.begin() + kPrefill), {}, 0);
    s1.Bind(topo);
    vllm::ModelRegistry::Forward(*model, s1.Get());
    if (zero_between) topo.ZeroPages();
    Step s2(std::vector<int32_t>(ids.begin() + kPrefill, ids.end()), {},
            kPrefill);
    s2.Bind(topo);
    return vllm::ModelRegistry::Forward(*model, s2.Get());
  };
  const vllm::ForwardLogits kept = run(false);
  const vllm::ForwardLogits wiped = run(true);
  REQUIRE(kept.host.size() == wiped.host.size());
  const Gap g2 = MaxGap(kept.host, wiped.host);
  CHECK(g2.nonfinite == 0);
  MESSAGE("zeroing the pages between steps moved the logits by " << g2.max_abs);
  CHECK(g2.max_abs > 0.0);

  // ...and the pages were WRITTEN in the first place, at the slots the engine's
  // own `slot_mapping` names and not at `position * head_size`.
  Topology t2;
  Step s(std::vector<int32_t>(ids.begin(), ids.begin() + kPrefill), {}, 0);
  s.Bind(t2);
  vllm::ModelRegistry::Forward(*model, s.Get());
  const float* latent = reinterpret_cast<const float*>(t2.attn_bytes[0].data());
  int64_t written = 0;
  for (int64_t t = 0; t < kPrefill; ++t) {
    const int64_t off = Topology::Slot(t) * Topology::LatentRow();
    for (int64_t i = 0; i < Topology::LatentRow(); ++i)
      if (latent[off + i] != 0.0F) ++written;
  }
  MESSAGE("nonzero latent elements at the mapped slots: " << written << " of "
          << kPrefill * Topology::LatentRow());
  CHECK(written > 0);
  // Page 0 is block 5, so the bytes at block 0 must be untouched. This is the
  // whole difference between reading the gathered block table and indexing by
  // logical position.
  int64_t at_block_zero = 0;
  for (int64_t i = 0; i < Topology::kBlockSize * Topology::LatentRow(); ++i)
    if (latent[i] != 0.0F) ++at_block_zero;
  CHECK(at_block_zero == 0);
}

TEST_CASE("glm5_next W5b-2c: the bf16 page dtype is the PRODUCTION default") {
  // `MakeGlm5NextKVCache` sizes both MLA groups at `v1::ResolveKvCacheDType()`,
  // which is bf16 unless `VT_KV_CACHE_F32` says otherwise, so the round trip
  // through the pages is LOSSY on the default configuration. That is upstream's
  // own polarity — it caches K and V in the model dtype — and it is recorded
  // here as a measured gap rather than left for a reader to assume the f32 case
  // above is what production runs.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const std::vector<int32_t> ids{3, 11, 7, 20, 2, 15, 9, 4};
  const int64_t kPrefill = 5;

  Topology topo(vt::DType::kBF16);
  Step s1(std::vector<int32_t>(ids.begin(), ids.begin() + kPrefill), {}, 0);
  s1.Bind(topo);
  vllm::ModelRegistry::Forward(*model, s1.Get());
  Step s2(std::vector<int32_t>(ids.begin() + kPrefill, ids.end()), {}, kPrefill);
  s2.Bind(topo);
  const vllm::ForwardLogits cont = vllm::ModelRegistry::Forward(*model, s2.Get());

  Step whole_step(ids, {5, 6, 7}, 0);
  const vllm::ForwardLogits whole =
      vllm::ModelRegistry::Forward(*model, whole_step.Get());
  const Gap gap = TailGap(cont, whole);
  CHECK(gap.nonfinite == 0);

  // THE BOUND IS ONE ULP OF THE LOGIT ITSELF, and it is measured from the data
  // rather than chosen. The fixture's logits sit near 2.65e5, where an f32 ULP
  // is 0.03125; a bf16 round trip of the cached latent costs at most that.
  float mag = 0.0F;
  for (float v : whole.host) mag = std::max(mag, std::abs(v));
  const double ulp = static_cast<double>(std::nextafterf(mag, HUGE_VALF) - mag);
  MESSAGE("bf16-cache tail gap: " << gap.max_abs << ", one ULP at |logit| max "
          << mag << " is " << ulp);
  CHECK(gap.max_abs <= ulp);

  // WHAT THIS CASE CANNOT DO, said rather than papered over. A control against
  // a different prefix is what makes the f32 case above discriminating, and at
  // THIS fixture it cannot serve here: changing three of the eight tokens moves
  // the tail logits by 0.03125 — two ULP — because the miniature's ramp weights
  // dominate its attention output, so the bf16 gap (one ULP) and a wrong prompt
  // (two ULP) are both at the floor and a ratio between them would be a
  // coincidence. The f32 case is the one that separates a read cache from an
  // ignored one, and it does it EXACTLY. This case says only what it can: the
  // production dtype still round-trips within the logits' own resolution.
  std::vector<int32_t> other = ids;
  other[0] = 21;
  other[1] = 1;
  other[2] = 30;
  Step other_step(other, {5, 6, 7}, 0);
  const Gap control =
      TailGap(cont, vllm::ModelRegistry::Forward(*model, other_step.Get()));
  REQUIRE(control.max_abs > 0.0);
  MESSAGE("bf16 control gap: " << control.max_abs);
}

// ═══ (6) W5b-2c — a WRONG mapping refuses rather than answering ═════════════

TEST_CASE("glm5_next W5b-2c: the PUBLICATION ORDER does not matter, the NAME does") {
  // WITHOUT THIS CASE THE SUITE CANNOT SEE THE DIFFERENCE. The miniature has
  // exactly ONE DSA layer, so its MLA latent is entry 0 of `attn_kv` and a
  // resolver that indexed by position instead of asking
  // `MultiKvCacheIndex::Find` would be right at this fixture and wrong on the
  // published checkpoint, whose 22 entries arrive as eleven latents followed by
  // eleven indexer caches. Permuting the channel — the names, the group ids,
  // the layer indices and the caches together, exactly as the runner would if
  // it published the groups in the other order — makes position wrong and name
  // right, and the two runs must be BIT-IDENTICAL.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const std::vector<int32_t> ids{3, 11, 7, 20};

  Step plain(ids);
  const vllm::ForwardLogits a = vllm::ModelRegistry::Forward(*model, plain.Get());

  Topology flipped;
  // The runner publishing group 2 BEFORE group 0: the two attention entries
  // trade flat positions, the recurrent group stays where it is, and the paged
  // SLOTS stay 0 and 1 because the runner's counter is positional — so
  // `attn_kv` is allocated in the new group order too and moves with them.
  std::swap(flipped.names[Topology::kLatentFlat],
            flipped.names[Topology::kIndexerFlat]);
  std::swap(flipped.group_ids[Topology::kLatentFlat],
            flipped.group_ids[Topology::kIndexerFlat]);
  std::swap(flipped.layer_indices[Topology::kLatentFlat],
            flipped.layer_indices[Topology::kIndexerFlat]);
  std::swap(flipped.attn_kv[0], flipped.attn_kv[1]);
  std::swap(flipped.attn_bytes[0], flipped.attn_bytes[1]);
  flipped.Publish();
  Step permuted(ids);
  permuted.Bind(flipped);
  const vllm::ForwardLogits b =
      vllm::ModelRegistry::Forward(*model, permuted.Get());

  REQUIRE(a.host.size() == b.host.size());
  const Gap gap = MaxGap(a.host, b.host);
  CHECK(gap.nonfinite == 0);
  MESSAGE("permuted-channel gap: " << gap.max_abs);
  CHECK(gap.max_abs == 0.0);
}

TEST_CASE("glm5_next W5b-2c: an MLA latent read as a K+V PAIR is REFUSED") {
  // The single highest-risk error on this wave, and it does not crash. The MLA
  // latent is ONE vector per token (`MLAAttentionSpec` fixes num_kv_heads at 1
  // and stores no separate V); a reader that took the ordinary
  // [num_blocks, 2, block_size, num_kv_heads, head_size] pair layout would
  // index at twice the stride and hand the layer finite, correctly shaped,
  // WRONG numbers. The model would generate fluent text and no gate on this
  // fleet could see it. So the geometry is checked and the refusal names it.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);

  Topology topo;
  topo.attn_kv[0].num_kv_heads = kHeads;  // what a K+V pair would publish
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("K+V pair"), std::runtime_error);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("#2348"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: the two DSA caches cannot be SWAPPED") {
  // Group 0 is 512 wide on the published checkpoint and group 2 is 257; the
  // fixture's are 32 and 33. Swapping the two names points the latent reader at
  // the indexer's page and back, which is shape-valid at every step and wrong
  // at every value.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  std::swap(topo.names[Topology::kLatentFlat],
            topo.names[Topology::kIndexerFlat]);
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("head_size"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: a MISSING published name is refused BY NAME") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.names[Topology::kIndexerFlat] =
      "model.layers.2.self_attn.indexer.WRONG";
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(
      vllm::ModelRegistry::Forward(*model, s.Get()),
      doctest::Contains("published no cache named"), std::runtime_error);
  // The diagnostic says what the channel DID carry, not only what was absent.
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("names begin"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: ONE group cannot hold both DSA caches") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.group_ids[Topology::kIndexerFlat] =
      topo.group_ids[Topology::kLatentFlat];
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("SAME group id"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2d: a SHORT gdn_state is refused, by name and by count") {
  // This case was titled "the RECURRENT set count is the only check there is",
  // and W5b-2d made that sentence false: the channel carries every published
  // recurrent cache with a payload locator, so each KDA layer now resolves its
  // own state BY NAME. The count survives beside the lookup because it catches
  // the other direction — sets no declared layer claimed — and a short
  // `gdn_state` must still refuse rather than read slot 2 of a two-entry vector.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.gdn.pop_back();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("recurrent state set"),
                       std::runtime_error);
}

// ═══ (6b) W5b-2d — the FLAT index against the PAYLOAD SLOT (#2445) ══════════

TEST_CASE("glm5_next W5b-2d: the FLAT index is NOT the paged slot") {
  // THE DEFECT THIS WAVE REPAIRED, pinned as an assertion rather than as a
  // shape. `MultiKvCacheIndex::Find` answers a cache's place among EVERY
  // published cache; `PayloadAt` answers its slot in the container it lives in.
  // Before `9e7621efc` those were the same number and `glm5_next_kv.cpp` used
  // `Find`'s directly; after it they are not, and on the real artifact the
  // indexer side cache of the first DSA layer is flat 45 against an `attn_kv`
  // of 22. A future edit that reverts to `Find` fails HERE, on a stated
  // inequality, instead of on an out-of-range access that another topology
  // would not even produce.
  Topology topo;
  const vllm::MultiKvCacheIndex& mk = topo.mk;
  const std::string latent = "model.layers.2.self_attn.attn";
  const std::string indexer = "model.layers.2.self_attn.indexer.k_cache";

  // The channel covers every published cache, not only the paged ones.
  CHECK(mk.size() == static_cast<size_t>(kLayers + 1));
  CHECK(mk.num_paged() == 2);
  CHECK(mk.num_recurrent() == static_cast<int>(kLayers - 1));
  // THREE, and the header of `MultiKvCacheIndex` still says TWO. `num_groups()`
  // is documented as "how many DISTINCT published groups THEY came from" about
  // the caches in `attn_kv`, and it is implemented as a distinct-count over the
  // whole `group_ids` vector — which `9e7621efc` widened to cover the recurrent
  // group as well. So the accessor answers 3 here and `glm5_next_kv.h` used to
  // repeat the stale 2. The value is asserted rather than the prose, and #2459
  // owns the shared header. `num_published_groups()` is unaffected: it counts
  // the block-table vector, which was always per published group.
  CHECK(mk.num_groups() == 3);
  CHECK(mk.num_published_groups() == 3);
  CHECK(topo.attn_kv.size() == 2);

  CHECK(mk.Find(latent) == static_cast<int64_t>(Topology::kLatentFlat));
  CHECK(mk.Find(indexer) == static_cast<int64_t>(Topology::kIndexerFlat));
  // 4 against 2 here IS 45 against 22 on the published checkpoint.
  CHECK(mk.Find(indexer) >= static_cast<int64_t>(topo.attn_kv.size()));

  vllm::KvCachePayload kind = vllm::KvCachePayload::kRecurrent;
  int32_t slot = -1;
  REQUIRE(mk.Resolve(indexer, &kind, &slot));
  CHECK(kind == vllm::KvCachePayload::kPaged);
  CHECK(slot == 1);
  // The inequality is the whole point, stated so it cannot silently collapse.
  CHECK(static_cast<int64_t>(slot) != mk.Find(indexer));

  REQUIRE(mk.Resolve(latent, &kind, &slot));
  CHECK(kind == vllm::KvCachePayload::kPaged);
  CHECK(slot == 0);

  // The recurrent half, addressable by name since `9e7621efc` and consumed by
  // this wave. Layer 3 is this miniature's LAST KDA layer and its state is
  // gdn_state slot 2 — not 3, which is what its layer index would have said.
  REQUIRE(mk.Resolve("model.layers.3.linear_attn", &kind, &slot));
  CHECK(kind == vllm::KvCachePayload::kRecurrent);
  CHECK(slot == static_cast<int32_t>(kLayers - 2));
  CHECK(mk.Find("model.layers.3.linear_attn") == static_cast<int64_t>(kLayers - 1));
  CHECK(static_cast<int64_t>(slot) != mk.Find("model.layers.3.linear_attn"));

  // And the group id is read at the FLAT index, which is the other half of the
  // repair: `group_ids` is parallel to the flat list and not to `attn_kv`.
  REQUIRE(mk.group_ids != nullptr);
  CHECK((*mk.group_ids)[Topology::kLatentFlat] == 0);
  CHECK((*mk.group_ids)[Topology::kIndexerFlat] == 2);
  CHECK((*mk.group_ids)[1] == 1);
}

TEST_CASE("glm5_next W5b-2d: the recurrent slot is READ, not counted") {
  // WITHOUT THIS CASE THE BY-NAME RECURRENT RESOLUTION IS NOT GATED. The runner
  // assigns gdn_state slots in ascending layer order, which is exactly what a
  // KDA-ordinal counter would produce, so on every topology this tree builds the
  // two agree and a test that only ran the forward could not tell them apart.
  // Permuting the published slots makes them disagree — and the binding must
  // then REFUSE, because it read the channel. A counter would sail through.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  // Flat 1 is layer 0's state (slot 0) and flat 3 is layer 3's (slot 2).
  REQUIRE(topo.payload_slots[1] == 0);
  REQUIRE(topo.payload_slots[3] == static_cast<int32_t>(kLayers - 2));
  std::swap(topo.payload_slots[1], topo.payload_slots[3]);
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("gdn_state slot"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2d: an attention name published as RECURRENT is refused") {
  // `attn_kv` and `gdn_state` are two containers, so a slot read against the
  // wrong one is an unrelated buffer with NO shape error — the same
  // wrong-answer-not-a-crash shape the MLA-latent refusal exists for.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.payload_kinds[Topology::kIndexerFlat] =
      static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent);
  topo.payload_slots[Topology::kIndexerFlat] = 0;  // in range for gdn_state
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("RECURRENT cache"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2d: a recurrent name published as PAGED is refused") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.payload_kinds[1] = static_cast<uint8_t>(vllm::KvCachePayload::kPaged);
  topo.payload_slots[1] = 0;  // in range for attn_kv
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("PAGED cache"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2d: a channel with NO payload locator is refused") {
  // The pre-`9e7621efc` channel shape, which is what this fixture published
  // until this wave. It is not silently tolerated: without the locator there is
  // no way to turn a name into a slot, and the flat index is not one.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.Publish();
  topo.mk.payload_kinds = nullptr;
  topo.mk.payload_slots = nullptr;
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("no payload locator"),
                       std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: a block table SHORTER than the step is refused") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  topo.group_cols[2] = 1;                  // one page = 4 tokens
  topo.group_bt[2].resize(1);
  topo.Publish();
  Step s({1, 2, 3, 4, 5});                 // five tokens needs two
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("block table"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: a block table the ENGINE disagrees with is refused") {
  // The runner computes `slot_mapping` for the target attention group with its
  // own walk of the same table, so it is an independent second opinion on the
  // map this forward builds. A disagreement is a refusal on the first step
  // rather than a wrong token on every step after it.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Topology topo;
  std::swap(topo.group_bt[0][0], topo.group_bt[0][1]);
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("slot_mapping"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: a NULL multi-KV channel is refused BY NAME") {
  // This model publishes three groups, so the runner sets the channel on every
  // step. A null one means the topology was classified as uniform, and the
  // positional `attn_kv` convention cannot say which of a DSA layer's two
  // caches an entry is. Running anyway would attend an empty prefix from the
  // second step on — the fluent-wrong-text failure this row exists to refuse.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Step s({1, 2, 3});
  s.multi_kv = nullptr;
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("no multi-KV channel"),
                       std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: a HISTORY the engine does not believe in is refused") {
  // `num_computed_tokens_cpu` and `seq_lens_cpu` must sum to the step's token
  // count, or this forward would write history rows the engine's block manager
  // never allocated.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Step s({1, 2, 3}, {}, /*computed=*/2);
  s.attn_meta.seq_lens_cpu = {9};
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("seq_len"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: ModelRegistry::Forward NARROWS its refusal, not drops it") {
  // The W3 guard (#2068) refused ANY keyed cache set. This row's forward
  // declares that it consumes one; a model that has NOT wired its forward
  // still inherits `consumes_multi_kv == false` and is still refused with the
  // same message, which is what keeps the narrowing from being a removal.
  const std::vector<std::string> archs{"Glm5NextForConditionalGeneration"};
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(archs);
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->consumes_multi_kv);

  // DeepSeek-V4 was the negative case when this test was written, and it is
  // NOT one any more: `KV-DSV4-MULTICACHE` W5 (#2323) landed its consuming
  // forward, so it now declares consumption for the same reason this row does.
  // Asserting it still refuses would gate the absence of that work.
  const std::vector<std::string> dsv4_archs{"DeepseekV4ForCausalLM"};
  const vllm::ModelRegistration& dsv4 = vllm::ModelRegistry::Resolve(dsv4_archs);
  REQUIRE(dsv4.factory != nullptr);
  CHECK(dsv4.factory->consumes_multi_kv);

  // The negative case has to be a model that genuinely has not wired one.
  // Kimi-Linear publishes a single attention group, so it never reaches the
  // guard, and it carries the default.
  const std::vector<std::string> kimi_archs{"KimiLinearForCausalLM"};
  const vllm::ModelRegistration& kimi = vllm::ModelRegistry::Resolve(kimi_archs);
  REQUIRE(kimi.factory != nullptr);
  CHECK_FALSE(kimi.factory->consumes_multi_kv);
}

// ─── W9c-3b (#2480): THE ENGINE'S PAGES ARE NOT ALWAYS HOST MEMORY ───────────
//
// WHAT WENT WRONG AND WHY NOTHING SAW IT. Every case above hands this model a
// CPU queue, and on a CPU queue `GPUModelRunner::CacheBuffer` keeps its pages in
// a `std::vector<uint8_t>` (`v1/worker/gpu/runner.cpp:575-593`). On any other
// queue `kv_cache_backend_resident_` is true (`:1119-1122`) and every paged
// cache and every recurrent state is a `vt::Alloc` allocation -- `cudaMalloc` on
// CUDA, which is NOT host-dereferenceable even on GB10
// (`src/vt/cuda/cuda_backend.cu:354-391`, held by a `static_assert` that names
// #844 and #1435 as the same fault twice already).
//
// `glm5_next_kv.cpp` read and wrote those pages with plain host loops, so a
// `--device cuda` step was a host store into device memory. `LoadCaches` returns
// before touching a page on a fresh sequence, so the first such access on step 1
// is inside `StoreCaches`, AFTER the whole forward has returned -- which is why
// the three legs in spec O46 died with SIGSEGV having emitted no token, and why
// the last two lines on their stderr were the MoE arm's two once-flags. O49
// carries the bisect.
//
// WHAT THIS CASE MEASURES, AND WHY IT IS NOT A CRASH TEST. A test binary cannot
// hold a `cudaMalloc` pointer, and a SIGSEGV is not an assertion. `ShadowBackend`
// is the next-strongest thing and is deterministic on every platform: the
// pointer `Alloc` hands back is a DECOY filled with a poison pattern, and the
// real storage lives in a side block only `Copy` can reach. Code that
// dereferences the pointer therefore reads poison and writes where nothing will
// ever look, and code that goes through the backend is correct -- which is
// exactly the distinction the defect is, with the fault turned into a value.
namespace {

class ShadowBackend final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    const size_t n = bytes == 0 ? 1 : bytes;
    auto block = std::make_unique<Block>();
    block->decoy.assign(n, kPoison);
    block->shadow.assign(n, 0);
    void* p = block->decoy.data();
    blocks_.push_back(std::move(block));
    ++allocs;
    return p;
  }
  void Free(void*) override {}
  void Memset(vt::Queue&, void* p, int v, size_t bytes) override {
    uint8_t* dst = Translate(p, bytes);
    std::memset(dst, v, bytes);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t bytes) override {
    ++copies;
    uint8_t* d = Translate(dst, bytes);
    const uint8_t* s = Translate(const_cast<void*>(src), bytes);
    std::memcpy(d, s, bytes);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  }
  void DestroyQueue(vt::Queue&) override {}
  // The GB10 CUDA backend's own two answers (`cuda_backend.cu:113` and the
  // inherited default): one physical RAM, and still not host-dereferenceable.
  bool UnifiedMemory() const override { return true; }
  bool DeviceMemoryIsHostAddressable() const override { return false; }

  // Is `p` a byte a HOST loop would have had to fault on? Used by the case to
  // read the shadow without going through `Copy` twice.
  const uint8_t* ShadowOf(const void* p, size_t bytes) const {
    for (const std::unique_ptr<Block>& b : blocks_) {
      const uint8_t* base = b->decoy.data();
      const auto* q = static_cast<const uint8_t*>(p);
      if (q >= base && q + bytes <= base + b->decoy.size())
        return b->shadow.data() + (q - base);
    }
    return nullptr;
  }

  static constexpr uint8_t kPoison = 0xDD;
  int allocs = 0;
  int copies = 0;

 private:
  struct Block {
    std::vector<uint8_t> decoy;
    std::vector<uint8_t> shadow;
  };
  // A pointer into one of our decoys resolves to the SAME offset in its shadow;
  // anything else is an ordinary host buffer and is used as it is.
  uint8_t* Translate(void* p, size_t bytes) {
    for (const std::unique_ptr<Block>& b : blocks_) {
      uint8_t* base = b->decoy.data();
      auto* q = static_cast<uint8_t*>(p);
      if (q >= base && q + bytes <= base + b->decoy.size())
        return b->shadow.data() + (q - base);
    }
    return static_cast<uint8_t*>(p);
  }
  std::vector<std::unique_ptr<Block>> blocks_;
};

ShadowBackend& Shadow() {
  static ShadowBackend b;
  return b;
}

struct ShadowRegistrar {
  ShadowRegistrar() {
    vt::RegisterBackend(vt::Device{vt::DeviceType::kXPU, 0}, &Shadow());
  }
};
const ShadowRegistrar kShadowRegistrar;

// A `Topology` whose every page and every recurrent state has been re-homed
// onto the shadow backend, with the sizes and the block permutation unchanged.
// The `vt::Tensor` device tags move with them, because a state that says kCPU
// while its bytes are on a device is the lie this whole case is about.
struct ShadowTopology {
  Topology t;
  ShadowTopology() {
    for (size_t i = 0; i < t.attn_kv.size(); ++i) {
      const size_t n = t.attn_bytes[i].size();
      t.attn_kv[i].data = Shadow().Alloc(n);
    }
    for (size_t j = 0; j < t.gdn.size(); ++j) {
      t.gdn[j].conv_state.data = Shadow().Alloc(t.conv_bytes[j].size());
      t.gdn[j].conv_state.device = vt::Device{vt::DeviceType::kXPU, 0};
      t.gdn[j].ssm_state.data = Shadow().Alloc(t.ssm_bytes[j].size());
      t.gdn[j].ssm_state.device = vt::Device{vt::DeviceType::kXPU, 0};
      t.gdn[j].states = {t.gdn[j].conv_state, t.gdn[j].ssm_state};
    }
    // NOT `Publish()`: that re-points `attn_kv[i].data` back at the host
    // vectors, which would silently undo this whole fixture.
    t.mk.layer_names = &t.names;
    t.mk.group_ids = &t.group_ids;
    t.mk.layer_indices = &t.layer_indices;
    t.mk.payload_kinds = &t.payload_kinds;
    t.mk.payload_slots = &t.payload_slots;
    t.mk.group_block_tables = &t.group_bt;
    t.mk.group_block_table_cols = &t.group_cols;
  }
};

// Read `n` f32 values out of the SHADOW at element offset `first`. Deliberately
// not `Copy`: a case that read the storage the same way the code under test does
// would pass whenever the two agreed, including when both were the decoy.
std::vector<float> ShadowFloats(const void* base, int64_t first, int64_t n) {
  const auto* p = static_cast<const uint8_t*>(base) +
                  static_cast<size_t>(first) * sizeof(float);
  const uint8_t* s = Shadow().ShadowOf(p, static_cast<size_t>(n) * sizeof(float));
  REQUIRE(s != nullptr);
  std::vector<float> out(static_cast<size_t>(n));
  std::memcpy(out.data(), s, out.size() * sizeof(float));
  return out;
}

// A deterministic, non-constant pattern. Constant fill would pass against a
// zeroed shadow for the zero value and against poison for nothing, so the values
// are spread and none of them is 0.
float Pattern(int64_t tag, int64_t i) {
  return 1.0F + static_cast<float>(tag) * 0.125F + static_cast<float>(i) * 0.03125F;
}

}  // namespace

TEST_CASE("glm5_next W9c-3b: the KV binding COPIES the engine's pages instead "
          "of dereferencing them") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  REQUIRE(model != nullptr);
  const vllm::Glm5NextParams& p = Weights(model).params;
  REQUIRE(p.num_hidden_layers == kLayers);

  ShadowTopology pages;
  const int64_t latent_row = Topology::LatentRow();
  const int64_t indexer_row = Topology::IndexerRow();
  const int64_t conv_elems = Topology::ConvElems();
  const int64_t rec_elems = Topology::RecElems();
  constexpr int64_t kNewTokens = 2;

  Step s1({1, 2});
  s1.queue = vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  s1.Bind(pages.t);
  const vllm::ModelForwardInput in1 = s1.Get();
  const gn::KvBinding b1 = gn::ResolveKvBinding(p, in1);
  REQUIRE(b1.cached_len == 0);
  REQUIRE(b1.new_tokens == kNewTokens);

  // A fresh sequence reads NOTHING, so this call must not touch a page at all.
  std::vector<gn::LayerCache> caches;
  gn::LoadCaches(p, b1, in1, &caches);
  REQUIRE(caches.size() == static_cast<size_t>(kLayers));

  // Fill the states the forward would have produced. The VALUES are the point:
  // they have to arrive in the shadow, at the offsets the block permutation
  // puts them at, or the write went to the decoy.
  for (int64_t l = 0; l < kLayers; ++l) {
    gn::LayerCache& c = caches[static_cast<size_t>(l)];
    if (l == Topology::kDsaLayer) {
      c.dsa.cached_len = kNewTokens;
      c.dsa.k_pass.resize(static_cast<size_t>(kNewTokens * latent_row));
      for (size_t i = 0; i < c.dsa.k_pass.size(); ++i)
        c.dsa.k_pass[i] = Pattern(l, static_cast<int64_t>(i));
      c.dsa.indexer_packed.resize(static_cast<size_t>(kNewTokens * indexer_row));
      for (size_t i = 0; i < c.dsa.indexer_packed.size(); ++i)
        c.dsa.indexer_packed[i] = Pattern(l + 64, static_cast<int64_t>(i));
      continue;
    }
    c.kda.assign(1, vllm::glm5_next_kda::Glm5NextKdaCache{});
    c.kda[0].conv_state.resize(static_cast<size_t>(conv_elems));
    for (size_t i = 0; i < c.kda[0].conv_state.size(); ++i)
      c.kda[0].conv_state[i] = Pattern(l + 128, static_cast<int64_t>(i));
    c.kda[0].recurrent_state.resize(static_cast<size_t>(rec_elems));
    for (size_t i = 0; i < c.kda[0].recurrent_state.size(); ++i)
      c.kda[0].recurrent_state[i] = Pattern(l + 192, static_cast<int64_t>(i));
  }

  const int copies_before = Shadow().copies;
  gn::StoreCaches(p, b1, caches, in1);
  // The write went through the backend at all. Necessary, never sufficient --
  // the value checks below are what say it went to the right place.
  CHECK(Shadow().copies > copies_before);

  // (1) THE PAGED ROWS. Read out of the SHADOW, at the flat slot the gathered
  // block table maps each logical position to, so an implementation that
  // addressed page `p` at block `p` lands on the wrong row rather than passing.
  const gn::LayerKvBinding& lb =
      b1.layers[static_cast<size_t>(Topology::kDsaLayer)];
  const void* lat = in1.attn_kv[static_cast<size_t>(lb.latent)].data;
  const void* ix = in1.attn_kv[static_cast<size_t>(lb.indexer)].data;
  for (int64_t t = 0; t < kNewTokens; ++t) {
    const std::vector<float> got =
        ShadowFloats(lat, Topology::Slot(t) * latent_row, latent_row);
    for (int64_t i = 0; i < latent_row; ++i) {
      CHECK(got[static_cast<size_t>(i)] ==
            doctest::Approx(Pattern(Topology::kDsaLayer, t * latent_row + i)));
    }
    const std::vector<float> gix =
        ShadowFloats(ix, Topology::Slot(t) * indexer_row, indexer_row);
    for (int64_t i = 0; i < indexer_row; ++i) {
      CHECK(gix[static_cast<size_t>(i)] ==
            doctest::Approx(Pattern(Topology::kDsaLayer + 64, t * indexer_row + i)));
    }
  }

  // (2) THE RECURRENT STATES, which are the FIRST thing `StoreCaches` writes on
  // this model and therefore the byte the three `dgx:gpu0` legs died on.
  for (int64_t l = 0; l < kLayers; ++l) {
    if (l == Topology::kDsaLayer) continue;
    const gn::LayerKvBinding& rb = b1.layers[static_cast<size_t>(l)];
    const vllm::GdnStateCache& gs =
        in1.gdn_state[static_cast<size_t>(rb.recurrent)];
    const std::vector<float> conv = ShadowFloats(gs.conv_state.data, 0, conv_elems);
    for (int64_t i = 0; i < conv_elems; ++i) {
      CHECK(conv[static_cast<size_t>(i)] == doctest::Approx(Pattern(l + 128, i)));
    }
    const std::vector<float> rec = ShadowFloats(gs.ssm_state.data, 0, rec_elems);
    for (int64_t i = 0; i < rec_elems; ++i) {
      CHECK(rec[static_cast<size_t>(i)] == doctest::Approx(Pattern(l + 192, i)));
    }
  }

  // (3) THE READ DIRECTION, on a second step that has history. The decoy still
  // holds nothing but poison, so a `LoadCaches` that dereferenced the page would
  // hydrate every state from 0xDDDDDDDD instead of from what step 1 stored.
  Step s2({3}, {}, kNewTokens);
  s2.queue = vt::Queue{vt::Device{vt::DeviceType::kXPU, 0}, nullptr};
  s2.Bind(pages.t);
  const vllm::ModelForwardInput in2 = s2.Get();
  const gn::KvBinding b2 = gn::ResolveKvBinding(p, in2);
  REQUIRE(b2.cached_len == kNewTokens);
  std::vector<gn::LayerCache> back;
  gn::LoadCaches(p, b2, in2, &back);
  REQUIRE(back.size() == static_cast<size_t>(kLayers));

  const gn::LayerCache& dsa = back[static_cast<size_t>(Topology::kDsaLayer)];
  REQUIRE(dsa.dsa.k_pass.size() ==
          static_cast<size_t>(kNewTokens * latent_row));
  for (size_t i = 0; i < dsa.dsa.k_pass.size(); ++i) {
    CHECK(dsa.dsa.k_pass[i] ==
          doctest::Approx(Pattern(Topology::kDsaLayer, static_cast<int64_t>(i))));
  }
  REQUIRE(dsa.dsa.indexer_packed.size() ==
          static_cast<size_t>(kNewTokens * indexer_row));
  for (size_t i = 0; i < dsa.dsa.indexer_packed.size(); ++i) {
    CHECK(dsa.dsa.indexer_packed[i] ==
          doctest::Approx(Pattern(Topology::kDsaLayer + 64,
                                  static_cast<int64_t>(i))));
  }
  for (int64_t l = 0; l < kLayers; ++l) {
    if (l == Topology::kDsaLayer) continue;
    const gn::LayerCache& c = back[static_cast<size_t>(l)];
    REQUIRE(c.kda.size() == 1);
    REQUIRE(c.kda[0].conv_state.size() == static_cast<size_t>(conv_elems));
    for (size_t i = 0; i < c.kda[0].conv_state.size(); ++i) {
      CHECK(c.kda[0].conv_state[i] ==
            doctest::Approx(Pattern(l + 128, static_cast<int64_t>(i))));
    }
    REQUIRE(c.kda[0].recurrent_state.size() == static_cast<size_t>(rec_elems));
    for (size_t i = 0; i < c.kda[0].recurrent_state.size(); ++i) {
      CHECK(c.kda[0].recurrent_state[i] ==
            doctest::Approx(Pattern(l + 192, static_cast<int64_t>(i))));
    }
  }
}

TEST_CASE("glm5_next W9c-3b: a CPU queue writes the pages IN PLACE, and the "
          "shadow backend is not involved") {
  // The `--device cpu` arm, which is the one this wave must not move. The bytes
  // land in the topology's OWN host vectors, at the slot the block permutation
  // maps each position to, and no other backend is touched on the way.
  //
  // WHAT THIS CASE CANNOT SEE, stated because a mutation proved it. Deleting
  // `PageIo`'s `if (q.device.type == kCPU) return;` -- so that a CPU queue
  // resolves `vt::GetBackend(kCPU)` and bounces through it as well -- SURVIVES
  // this case and the whole suite. It has to: the CPU backend's `Copy` IS
  // `std::memcpy`, so the bytes are identical either way, and `Shadow().copies`
  // below counts the SHADOW backend, which a CPU queue never reaches under
  // either version. The early return is therefore not load-bearing for
  // correctness. It is there so that a build with no CPU backend registered
  // does not `Fail` on the host path, and so the `--device cpu` instruction
  // stream is the one that was already measured. Spec `## Owed` O49 records the
  // survival rather than leaving it for the next reader to rediscover.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  const vllm::Glm5NextParams& p = Weights(model).params;

  Topology host;
  Step s({1, 2});
  s.Bind(host);
  const vllm::ModelForwardInput in = s.Get();
  const gn::KvBinding b = gn::ResolveKvBinding(p, in);
  std::vector<gn::LayerCache> caches;
  gn::LoadCaches(p, b, in, &caches);
  const int64_t latent_row = Topology::LatentRow();
  gn::LayerCache& c = caches[static_cast<size_t>(Topology::kDsaLayer)];
  c.dsa.cached_len = 2;
  c.dsa.k_pass.assign(static_cast<size_t>(2 * latent_row), 0.0F);
  for (size_t i = 0; i < c.dsa.k_pass.size(); ++i)
    c.dsa.k_pass[i] = Pattern(7, static_cast<int64_t>(i));
  c.dsa.indexer_packed.assign(
      static_cast<size_t>(2 * Topology::IndexerRow()), 0.5F);
  for (int64_t l = 0; l < kLayers; ++l) {
    if (l == Topology::kDsaLayer) continue;
    gn::LayerCache& k = caches[static_cast<size_t>(l)];
    k.kda.assign(1, vllm::glm5_next_kda::Glm5NextKdaCache{});
    k.kda[0].conv_state.assign(static_cast<size_t>(Topology::ConvElems()), 0.25F);
    k.kda[0].recurrent_state.assign(static_cast<size_t>(Topology::RecElems()), 0.75F);
  }
  const int copies_before = Shadow().copies;
  gn::StoreCaches(p, b, caches, in);
  CHECK(Shadow().copies == copies_before);

  // And the bytes landed in the topology's OWN host vectors, at the permuted
  // slot, so "no backend" did not become "no write".
  const gn::LayerKvBinding& lb =
      b.layers[static_cast<size_t>(Topology::kDsaLayer)];
  const auto* lat = static_cast<const float*>(
      in.attn_kv[static_cast<size_t>(lb.latent)].data);
  for (int64_t t = 0; t < 2; ++t) {
    for (int64_t i = 0; i < latent_row; ++i) {
      CHECK(lat[Topology::Slot(t) * latent_row + i] ==
            doctest::Approx(Pattern(7, t * latent_row + i)));
    }
  }
}

// ═══ (5) THE ASYNCHRONOUS DEVICE-IDENTIFIER GATE ════════════════════════════
//
// Row `ENG-ASYNC-DEVICE-IDS-2544`, spec
// `.agents/specs/eng-async-device-ids-2544.md`, issue
// [#2544](https://github.com/mudler/vllm.cpp/issues/2544); the contract is
// [#1305](https://github.com/mudler/vllm.cpp/issues/1305)'s.
//
// WHAT IT MEASURES. On the asynchronous serving path the runner's combine
// splices each decode row's sampled token into the DEVICE identifiers on the
// main queue and leaves the host `token_ids` deliberately stale for decode rows
// (`src/vllm/v1/worker/gpu/runner.cpp`, the mirror arm, which is the DEFAULT on
// CUDA -- integrated parts included). A forward that embeds the host vector
// therefore generates every step after the first from the previous step's
// identifiers, and because `token_ids_cpu` is zero-initialised that means from
// TOKEN ID 0.
//
// This was MEASURED for this model, not inferred. On `dgx:gpu0` over the real
// 101.25 GiB UD-Q2_K_XL artifact, one binary, one variable flipped:
//
//     default                     ->  ' Paris Paris'
//     VT_ASYNC_DEVICE_MIRROR=0    ->  ' Paris.'
//     --max-tokens 1 (prefill)    ->  ' Paris'
//
// The prefill agrees, so the first token looked right and everything after it
// came from id 0 -- silently, at rc=0.
//
// THE DEFECT IS SILENTLY WRONG TOKENS AND NOT A FAULT, so this asserts the
// identifiers themselves rather than that the call returns. Three runs, because
// two of them cannot separate the cases:
//
//     A  right host ids, no mirror              -> the reference
//     B  ZERO host ids, no mirror               -> must DIFFER from A
//     C  ZERO host ids, mirror carries A's      -> must EQUAL A, bit for bit
//
// B is the control. Without it a forward that ignored its identifiers entirely
// would satisfy C, and so would a gate whose two runs happened to share a
// buffer. ZERO is the wrong-host value on purpose: it is the value the real
// defect feeds.
//
// EVERY IDENTIFIER IN A IS NON-ZERO, so C cannot agree with A for the wrong
// reason.
//
// WHAT A CPU HARNESS CANNOT SHOW, stated the right way round: `vt::Backend::Alloc`
// returns host-addressable memory on this backend, so the mirror's buffer and the
// host vector are the same kind of pointer. That the copy reads DEVICE memory and
// that it is ordered on the main queue after the combine are the two halves this
// cannot see; the spec's `## Gates` carries the hardware legs that can.
namespace {

// The mirror's buffer is a REAL backend allocation, never the host vector's
// address. On CPU the two are the same kind of pointer, so this buys nothing
// today; it is what makes the case correct by construction on a device, where
// `device_token_ids` is a pointer the runner's combine wrote.
struct MirrorIds {
  vt::Backend* b = nullptr;
  int32_t* p = nullptr;
  MirrorIds(vt::Queue& q, const std::vector<int32_t>& ids) {
    b = &vt::GetBackend(q.device.type);
    const size_t bytes = ids.size() * sizeof(int32_t);
    p = static_cast<int32_t*>(b->Alloc(bytes));
    b->Copy(q, p, ids.data(), bytes);
    b->Synchronize(q);
  }
  ~MirrorIds() { if (b != nullptr && p != nullptr) b->Free(p); }
  MirrorIds(const MirrorIds&) = delete;
  MirrorIds& operator=(const MirrorIds&) = delete;
};

size_t DifferingFloats(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  size_t n = 0;
  // memcmp rather than `!=`, because `NaN != NaN` is TRUE and would read as a
  // difference on a forward that produced no numbers at all.
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) ++n;
  return n;
}

}  // namespace

TEST_CASE("glm5_next: the forward embeds the async mirror's DEVICE ids, not the stale host vector") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  REQUIRE(model != nullptr);

  // NON-ZERO throughout, and inside the fixture's vocabulary so run B is a
  // wrong ANSWER rather than a refusal.
  const std::vector<int32_t> truth{5, 11, 23, 17};
  const std::vector<int32_t> stale(truth.size(), 0);

  // ── RUN A — the reference ───────────────────────────────────────────────
  Step a(truth);
  const vllm::ForwardLogits ref = vllm::ModelRegistry::Forward(*model, a.Get());
  REQUIRE(ref.host.size() == static_cast<size_t>(truth.size() * kVocab));
  // FINITENESS BEFORE ANY COMPARISON. Against a NaN both `==` and `!=` are
  // false, so an equality gate and a difference gate BOTH report success on a
  // forward that produced no numbers -- which is exactly how this row's sibling
  // read an all-NaN forward as a perfect match.
  {
    int nonfinite = 0;
    for (float v : ref.host) if (!std::isfinite(v)) ++nonfinite;
    REQUIRE(nonfinite == 0);
  }

  // ── RUN B — THE CONTROL. Stale host ids, no mirror: the logits must MOVE. ─
  Step b(stale);
  const vllm::ForwardLogits zeroed = vllm::ModelRegistry::Forward(*model, b.Get());
  REQUIRE(zeroed.host.size() == ref.host.size());
  {
    int nonfinite = 0;
    for (float v : zeroed.host) if (!std::isfinite(v)) ++nonfinite;
    REQUIRE(nonfinite == 0);
  }
  const size_t moved = DifferingFloats(ref.host, zeroed.host);
  CHECK_MESSAGE(moved > 0,
                "zeroing the host ids changed nothing, so this fixture cannot "
                "see an identifier at all and run C would prove nothing");

  // ── RUN C — THE GATE. The same stale host vector, the true identifiers
  //    reaching the model ONLY through `device_token_ids`. ─────────────────
  Step c(stale);
  vllm::ModelForwardInput in = c.Get();
  MirrorIds mirror(in.queue, truth);
  // Set after aggregate construction, exactly as `runner.cpp` sets it.
  in.device_token_ids = mirror.p;
  const vllm::ForwardLogits via_device = vllm::ModelRegistry::Forward(*model, in);
  REQUIRE(via_device.host.size() == ref.host.size());
  const size_t differing = DifferingFloats(ref.host, via_device.host);
  CHECK_MESSAGE(differing == 0,
                "registry forward, mirror vs host reference: " << differing
                    << " of " << ref.host.size() << " logits differ, so this "
                    "forward embedded the STALE host ids and the model generates "
                    "from token id 0 on every decode step after the first "
                    "(#2544, #1305)");
  MESSAGE("glm5_next device-ids: control moved " << moved << " floats; mirror "
          "vs reference differ in " << differing << " of " << ref.host.size());
}

TEST_CASE("glm5_next: a published device buffer over an EMPTY step is refused by name") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);

  // A runner that published identifiers for a step whose host vector is empty
  // and a model that shortened the step both reach here, and the two disagree
  // about the step's shape. Refused rather than silently reduced to a no-op,
  // because a no-op here IS the defect.
  Step e(std::vector<int32_t>{});
  vllm::ModelForwardInput in = e.Get();
  const std::vector<int32_t> one{5};
  MirrorIds mirror(in.queue, one);
  in.device_token_ids = mirror.p;
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, in),
                       doctest::Contains("disagree about the step's shape"),
                       std::runtime_error);
}
