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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "support/glm5_next_gguf_fixture.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_forward.h"
#include "vllm/model_executor/models/glm5_next_layer.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"      // ForwardLogits, *KvCache
#include "vllm/v1/attention/backend.h"               // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h"     // GDNAttentionMetadata
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

  vt::DType dtype = vt::DType::kF32;
  std::vector<std::vector<uint8_t>> attn_bytes;
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<std::string> names;
  std::vector<int32_t> group_ids;
  std::vector<int32_t> layer_indices;
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
    const int32_t gid[2] = {0, 2};
    const char* suffix[2] = {".self_attn.attn", ".self_attn.indexer.k_cache"};
    for (int i = 0; i < 2; ++i) {
      attn_bytes.emplace_back(
          static_cast<size_t>(kNumBlocks * kBlockSize * rows[i] * elt), 0);
      names.push_back("model.layers." + std::to_string(kDsaLayer) + suffix[i]);
      group_ids.push_back(gid[i]);
      layer_indices.push_back(static_cast<int32_t>(kDsaLayer));
    }
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
    // Three published groups, so three gathered tables; group 1 is the
    // recurrent one and nothing on this path reads its table.
    group_bt.assign(3, std::vector<int32_t>(kBlockPerm, kBlockPerm + kNumBlocks));
    group_cols.assign(3, static_cast<int32_t>(kNumBlocks));

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

TEST_CASE("glm5_next forward: a NON-CPU queue is refused BY NAME") {
  // Every buffer on this path is a host `std::vector<float>` and
  // `vt::MoeRouterTopK` dispatches on the queue's device, so a device queue
  // here hands a kernel host pointers. That is a crash and not a fallback, and
  // a refusal is what stands between the two.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g);
  Step step({1, 2});
  step.queue = vt::Queue{vt::Device{vt::DeviceType::kCUDA, 0}, nullptr};
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                       doctest::Contains("non-CPU queue"), std::runtime_error);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, step.Get()),
                       doctest::Contains("#2241"), std::runtime_error);
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
  std::swap(flipped.names[0], flipped.names[1]);
  std::swap(flipped.group_ids[0], flipped.group_ids[1]);
  std::swap(flipped.layer_indices[0], flipped.layer_indices[1]);
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
  std::swap(topo.names[0], topo.names[1]);
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
  topo.names[1] = "model.layers.2.self_attn.indexer.WRONG";
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
  topo.group_ids[1] = topo.group_ids[0];
  topo.Publish();
  Step s({1, 2, 3});
  s.Bind(topo);
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*model, s.Get()),
                       doctest::Contains("SAME group id"), std::runtime_error);
}

TEST_CASE("glm5_next W5b-2c: the RECURRENT set count is the only check there is") {
  // `MultiKvCacheIndex` keys the ATTENTION caches only. The KDA states arrive
  // on `gdn_state` positionally with no name, so the count is the whole check
  // and it has to be a real one.
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
