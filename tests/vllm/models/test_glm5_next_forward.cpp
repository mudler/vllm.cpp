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

namespace {

using gguf_test::TempFile;
using namespace glm5_next_fixture;  // NOLINT(build/namespaces) — the fixture IS this suite's vocabulary

namespace gn = vllm::glm5_next;

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

  explicit Step(std::vector<int32_t> ids, std::vector<int32_t> want = {})
      : token_ids(std::move(ids)), logits_indices(std::move(want)) {
    positions.resize(token_ids.size());
    for (size_t i = 0; i < positions.size(); ++i)
      positions[i] = static_cast<int32_t>(i);
  }

  vllm::ModelForwardInput Get() {
    return vllm::ModelForwardInput{.token_ids = token_ids,
                                   .positions = positions,
                                   .attn_meta = attn_meta,
                                   .gdn_meta = gdn_meta,
                                   .attn_kv = attn_kv,
                                   .gdn_state = gdn_state,
                                   .config = config,
                                   .queue = queue,
                                   .logits_indices = logits_indices,
                                   .num_reqs = num_reqs};
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

  const std::vector<float> one = gn::Glm5NextHostForward(w, ids, {}, q);
  // ONE ROW per chunk: 32 chunks over the fixture's vocab.
  const std::vector<float> many = gn::Glm5NextHostForward(w, ids, {}, q, row_bytes);
  const Gap gap = MaxGap(one, many);
  CHECK(gap.nonfinite == 0);
  CHECK(gap.max_abs == 0.0);
  // A chunked head that wrote its columns at the wrong offsets would still be
  // finite and the right size, so the values are checked against the
  // independent reference too, not only against each other.
  const Gap ref = MaxGap(many, ReferenceLogits(w, ids, {}));
  CHECK(ref.nonfinite == 0);
  CHECK(ref.max_abs == 0.0);

  CHECK_THROWS_AS(gn::Glm5NextHostForward(w, ids, {}, q, 0), std::runtime_error);
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
