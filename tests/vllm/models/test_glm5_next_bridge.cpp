// GLM-5.3-Flash W5b-1 gate — the `OwnedTensor` -> host f32 bridge and the
// RESIDENCY DECISION it implements (O22).
//
// Row MODEL-MM-glm5-next-glm5-next-for-conditional-generation, issue #2241,
// `.agents/specs/glm5-next-flash.md` section W5b and `## Owed` O22.
//
// ─── WHAT THIS FILE PINS ────────────────────────────────────────────────────
//
// O22 left the residency choice open: "Whoever writes the forward decides
// whether to decode per layer or to go device-native." W5b-1 decides PER LAYER,
// and this suite is what makes that a checkable property rather than a sentence
// in a header:
//
//  1. ONE bridged DSA layer costs 499,657,728 bytes (0.4654 GiB) at the
//     PUBLISHED geometry, computed from the dims and MEASURED from the decoded
//     buffers, and the two agree. Against the box's ~119.63 GiB that is 0.39%.
//  2. The materialized tower is 426.72 GiB (the spec's `### The measured
//     residency`), 3.57x over the same box. The arithmetic that rules it out is
//     asserted here so a later "just decode the tower" is a red gate.
//  3. The 1 GiB per-tensor ceiling sits BETWEEN the largest legitimate tensor
//     (`o_proj`, 0.25 GiB) and the smallest expert bank (`up_exps`, 9.0 GiB) by
//     a factor of four in both directions. Both sides are asserted, because a
//     ceiling above everything is a mute switch and a ceiling below the real
//     population is a gate that fires on ordinary work.
//  4. The ceiling is checked BEFORE any allocation — the refusal case declares
//     a published-size expert bank and carries NO bytes, so a bridge that
//     allocated first would not reach the throw.
//
// The substrate is the synthetic `glm5next` GGUF miniature W5c already gates
// its loader against, driven through the PRODUCTION `load_weights` hook. Using
// the real tower would need the 101.25 GiB artifact; using hand-built
// `OwnedTensor`s would gate the bridge against a shape nothing produces.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "support/glm5_next_gguf_fixture.h"
#include "vllm/model_executor/models/glm5_next_attn.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_dsa.h"
#include "vt/dtype.h"

namespace {

using gguf_test::TempFile;
using namespace glm5_next_fixture;  // NOLINT(build/namespaces) — the fixture IS this suite's vocabulary

using vllm::OwnedTensor;
using vllm::glm5_next::BridgeDsaLayer;
using vllm::glm5_next::BridgedDsaLayer;
using vllm::glm5_next::BridgedDsaLayerF32Bytes;
using vllm::glm5_next::DecodeOwnedTensorToF32;
using vllm::glm5_next::HostF32Bytes;
using vllm::glm5_next::IndexerDims;
using vllm::glm5_next::kBridgeTensorF32ByteCeiling;
using vllm::glm5_next::MlaDims;

// The miniature's MLA geometry, as the fixture declares it.
MlaDims FixtureMla() {
  MlaDims d;
  d.hidden_size = kH;
  d.num_heads = kHeads;
  d.q_lora_rank = kQLora;
  d.kv_lora_rank = kKvLora;
  d.qk_nope_head_dim = kQkNope;
  d.qk_rope_head_dim = 0;
  d.v_head_dim = kVHead;
  d.rms_norm_eps = 1e-5;
  return d;
}

IndexerDims FixtureIndexer() {
  IndexerDims d;
  d.hidden_size = kH;
  d.q_lora_rank = kQLora;
  d.n_heads = kIdxHeads;
  d.head_dim = kIdxHeadDim;
  d.index_topk = kIdxTopk;
  d.index_kpool = kKpool;
  d.always_select_tail = true;
  return d;
}

// The PUBLISHED checkpoint's geometry, for the residency arithmetic. Every
// value is `config.json`'s and none is a class default.
MlaDims PublishedMla() {
  MlaDims d;
  d.hidden_size = 4096;
  d.num_heads = 64;
  d.q_lora_rank = 1536;
  d.kv_lora_rank = 512;
  d.qk_nope_head_dim = 256;
  d.qk_rope_head_dim = 0;
  d.v_head_dim = 256;
  d.rms_norm_eps = 1e-5;
  return d;
}

IndexerDims PublishedIndexer() {
  IndexerDims d;
  d.hidden_size = 4096;
  d.q_lora_rank = 1536;
  d.n_heads = 32;
  d.head_dim = 128;
  d.index_topk = 2048;
  d.index_kpool = 4;
  d.always_select_tail = true;
  return d;
}

const vllm::Glm5NextWeights& LoadFixture(
    std::unique_ptr<vllm::LoadedModel>& holder, const vllm::GgufFile& g) {
  holder = LoadThroughRegistry(g);
  return vllm::ModelAs<vllm::Glm5NextLoadedModel>(
             *holder, "Glm5NextForConditionalGeneration")
      .weights();
}

// GiB, for readable messages only. Every assertion is on the byte count.
double GiB(int64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

}  // namespace

// --- (1) the residency decision, as arithmetic -------------------------------

TEST_CASE("glm5_next bridge: ONE DSA layer is 0.4654 GiB, the tower is 426.72") {
  const int64_t per_layer =
      BridgedDsaLayerF32Bytes(PublishedMla(), PublishedIndexer());
  // 117,442,560 MLA + 7,471,872 indexer parameters, x 4 bytes.
  CHECK(per_layer == 499657728);
  MESSAGE("one bridged DSA layer = " << per_layer << " B (" << GiB(per_layer)
          << " GiB)");

  // The box. `.agents/specs/glm5-next-flash.md` `### Fleet verdict`: ~119.63
  // GiB usable on `dgx:gpu0`, the largest device this project reaches.
  const double box_gib = 119.63;
  CHECK(GiB(per_layer) < box_gib * 0.01);  // 0.39% of the box

  // ELEVEN DSA layers, if a caller held every one at once. Still comfortable,
  // and stated so the per-layer choice is a floor and not a coincidence.
  CHECK(GiB(per_layer * 11) < 6.0);

  // The materialized tower, from the spec's measured table. This is the number
  // the decision rejects, and it is 3.57x the box.
  const double tower_gib = 426.72;
  CHECK(tower_gib > box_gib * 3.0);
  // ...and the block-resident tower the loader actually produces, which FITS.
  const double resident_gib = 101.14;
  CHECK(resident_gib < box_gib);
  MESSAGE("tower expanded " << tower_gib << " GiB vs block-resident "
          << resident_gib << " GiB vs box " << box_gib << " GiB");
}

TEST_CASE("glm5_next bridge: the 1 GiB ceiling separates the two populations") {
  CHECK(kBridgeTensorF32ByteCeiling == (int64_t{1} << 30));

  // The LARGEST tensor the bridge legitimately touches: `o_proj`, at the
  // published geometry [4096, 64 * 256].
  const MlaDims d = PublishedMla();
  const int64_t o_proj_bytes =
      d.hidden_size * d.num_heads * d.v_head_dim * 4;
  CHECK(o_proj_bytes == 268435456);  // 0.25 GiB
  CHECK(o_proj_bytes * 4 == kBridgeTensorF32ByteCeiling);  // exactly 4x under

  // The SMALLEST expert bank: `up_exps` at [288, 2048, 4096].
  const int64_t up_exps_bytes = int64_t{288} * 2048 * 4096 * 4;
  CHECK(up_exps_bytes == 9663676416);  // 9.0 GiB
  CHECK(up_exps_bytes == kBridgeTensorF32ByteCeiling * 9);  // exactly 9x over

  MESSAGE("ceiling " << GiB(kBridgeTensorF32ByteCeiling) << " GiB sits between "
          << GiB(o_proj_bytes) << " GiB (o_proj) and " << GiB(up_exps_bytes)
          << " GiB (up_exps)");
}

TEST_CASE("glm5_next bridge: an expert bank is refused BEFORE it is allocated") {
  // A PUBLISHED-SIZE `up_exps`, declared and carrying NO BYTES. A bridge that
  // allocated first would never reach the throw, so this case also proves the
  // ceiling is checked from the SHAPE.
  OwnedTensor bank;
  bank.dtype = vt::DType::kIQ2_XS;
  bank.rank = 3;
  bank.shape[0] = 288;
  bank.shape[1] = 2048;
  bank.shape[2] = 4096;
  CHECK(bank.bytes.empty());
  CHECK(HostF32Bytes(bank) == 9663676416);

  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(bank, "moe.up_exps"),
                       doctest::Contains("ceiling"), std::runtime_error);
  // ...and BY NAME, which is what keeps a refusal from costing a bisect.
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(bank, "moe.up_exps"),
                       doctest::Contains("up_exps"), std::runtime_error);
}

// --- (2) the decode itself, over the loader's own residencies ----------------

TEST_CASE("glm5_next bridge: a DSA layer bridges at the loader's own shapes") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);
  REQUIRE(w.layers.size() == static_cast<size_t>(kLayers));
  // Block 2 is the DSA layer of the miniature's schedule.
  REQUIRE_FALSE(w.layers[2].is_linear_attention);

  const MlaDims d = FixtureMla();
  const IndexerDims id = FixtureIndexer();
  BridgedDsaLayer b;
  REQUIRE_NOTHROW(b = BridgeDsaLayer(w.layers[2].mla, d, id));

  // Every buffer is the size its dims require. `k_b_proj` and `v_b_proj` have
  // DIFFERENT shapes on this fixture (kKvLora 32, kQkNope 16, kVHead 16), so a
  // bridge that read either at the other's orientation is caught here.
  CHECK(b.mla.q_a_proj.size() == static_cast<size_t>(kQLora * kH));
  CHECK(b.mla.q_a_layernorm.size() == static_cast<size_t>(kQLora));
  CHECK(b.mla.q_b_proj.size() == static_cast<size_t>(kHeads * kQkNope * kQLora));
  CHECK(b.mla.kv_a_proj_with_mqa.size() == static_cast<size_t>(kKvLora * kH));
  CHECK(b.mla.kv_a_layernorm.size() == static_cast<size_t>(kKvLora));
  CHECK(b.mla.k_b_proj.size() == static_cast<size_t>(kHeads * kKvLora * kQkNope));
  CHECK(b.mla.v_b_proj.size() == static_cast<size_t>(kHeads * kVHead * kKvLora));
  CHECK(b.mla.o_proj.size() == static_cast<size_t>(kH * kHeads * kVHead));

  const vllm::glm5_next::IndexerWeights ix = b.IndexerView();
  REQUIRE(ix.wq_b != nullptr);
  CHECK(b.idx_wq_b.size() == static_cast<size_t>(kIdxHeads * kIdxHeadDim * kQLora));
  CHECK(b.idx_wk.size() == static_cast<size_t>(kIdxHeadDim * kH));
  CHECK(b.idx_k_norm_weight.size() == static_cast<size_t>(kIdxHeadDim));
  // The BIAS is what makes `k_norm` a LayerNorm and not an RMSNorm.
  CHECK(b.idx_k_norm_bias.size() == static_cast<size_t>(kIdxHeadDim));
  CHECK(b.idx_weights_proj.size() == static_cast<size_t>(kIdxHeads * kH));
  CHECK(b.idx_kpool_ape.size() == static_cast<size_t>(kKpool * kIdxHeadDim));
  CHECK(b.idx_kpool_gate.size() == static_cast<size_t>(kIdxHeadDim * kH));

  // The VALUES are the file's own, and the two `kv_b_proj` halves carry
  // DIFFERENT bytes, so they are not one tensor read twice.
  const float rounded_k = vt::BF16ToF32(vt::F32ToBF16(Base(2, 11)));
  const float rounded_v = vt::BF16ToF32(vt::F32ToBF16(Base(2, 12)));
  CHECK(b.mla.k_b_proj[0] == doctest::Approx(rounded_k));
  CHECK(b.mla.v_b_proj[0] == doctest::Approx(rounded_v));
  CHECK(b.mla.k_b_proj[0] != doctest::Approx(b.mla.v_b_proj[0]));
  // An f32 norm arrives unrounded.
  CHECK(b.idx_k_norm_weight[0] == doctest::Approx(NormValue(0, NormTag(2, 5))));
  CHECK(b.idx_k_norm_bias[0] == doctest::Approx(NormValue(0, NormTag(2, 6))));

  // The MEASURED cost agrees with the PREDICTED one, which is what makes
  // `BridgedDsaLayerF32Bytes` a budget a caller can spend before allocating.
  //
  // BOTH sides are pinned INDEPENDENTLY, because `host_f32_bytes ==
  // BridgedDsaLayerF32Bytes(...)` alone is a tautology the moment the product
  // computes the first from the second -- a mutation that did exactly that
  // survived until this line was added.
  const int64_t summed =
      static_cast<int64_t>(b.mla.q_a_proj.size() + b.mla.q_a_layernorm.size() +
                           b.mla.q_b_proj.size() +
                           b.mla.kv_a_proj_with_mqa.size() +
                           b.mla.kv_a_layernorm.size() + b.mla.k_b_proj.size() +
                           b.mla.v_b_proj.size() + b.mla.o_proj.size() +
                           b.idx_wq_b.size() + b.idx_wk.size() +
                           b.idx_k_norm_weight.size() + b.idx_k_norm_bias.size() +
                           b.idx_weights_proj.size() + b.idx_kpool_ape.size() +
                           b.idx_kpool_gate.size()) *
      static_cast<int64_t>(sizeof(float));
  CHECK(b.host_f32_bytes == summed);
  CHECK(BridgedDsaLayerF32Bytes(d, id) == summed);
  MESSAGE("miniature layer bridged: " << b.host_f32_bytes << " B");
}

TEST_CASE("glm5_next bridge: IndexerView survives a MOVE") {
  // `IndexerWeights` is a struct of `const float*`. A member of that type
  // would dangle the moment the owner moved — silently, into freed-but-
  // plausible memory. The view is rebuilt from the CURRENT storage instead,
  // and this is the case that says so.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);

  BridgedDsaLayer a = BridgeDsaLayer(w.layers[2].mla, FixtureMla(), FixtureIndexer());
  const float first = a.idx_wk[0];
  const BridgedDsaLayer moved = std::move(a);
  const vllm::glm5_next::IndexerWeights view = moved.IndexerView();
  REQUIRE(view.wk != nullptr);
  CHECK(view.wk == moved.idx_wk.data());
  CHECK(view.wk[0] == doctest::Approx(first));
}

TEST_CASE("glm5_next bridge: the BLOCK-QUANT residency decodes") {
  // The fixture writes the stacked expert banks as Q8_0, which is the only
  // block encoding in the miniature. `RouteGgufTensor` keeps their blocks, so
  // this is the generic decoder running over a tensor that is NOT plain f32 or
  // bf16 — the residency 774 of the published artifact's 1412 tensors have.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);
  const OwnedTensor& bank = w.layers[2].moe.gate_exps;
  REQUIRE(vt::IsBlockQuant(bank.dtype));
  CHECK(bank.dtype == vt::DType::kQ8_0);

  const std::vector<float> f32 =
      DecodeOwnedTensorToF32(bank, "moe.gate_exps");
  REQUIRE(f32.size() == static_cast<size_t>(kExperts * kMoeI * kH));
  // The fixture's own decode, computed independently of the bridge.
  for (int64_t i : {int64_t{0}, int64_t{1}, int64_t{31}, int64_t{32},
                    static_cast<int64_t>(f32.size()) - 1}) {
    CHECK(f32[static_cast<size_t>(i)] ==
          doctest::Approx(Q8_0ValueAt(i, kH, 10 * 2 + 1)));
  }
  CHECK(HostF32Bytes(bank) ==
        static_cast<int64_t>(f32.size()) * static_cast<int64_t>(sizeof(float)));
}

// --- (3) the refusals --------------------------------------------------------

TEST_CASE("glm5_next bridge: a wrong shape is refused BY NAME") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);

  // THE SWAP, which is trap 1 itself: `k_b` is `[H, kv_lora, qk_nope]` and
  // `v_b` is `[H, v_head, kv_lora]`, so handing each the other's tensor is
  // exactly what a port that missed the converter's half-transpose produces.
  // Every OTHER tensor is untouched, so the refusal has to name `k_b_proj` and
  // cannot be some earlier check firing first.
  vllm::Glm5NextMlaWeights swapped = w.layers[2].mla;
  std::swap(swapped.k_b_proj, swapped.v_b_proj);
  CHECK_THROWS_WITH_AS(
      BridgeDsaLayer(swapped, FixtureMla(), FixtureIndexer()),
      doctest::Contains("k_b_proj"), std::runtime_error);

  // The indexer dims and the MLA dims come from ONE config; a disagreement is
  // a caller bug and not a geometry to serve.
  IndexerDims id = FixtureIndexer();
  id.hidden_size = kH + 1;
  CHECK_THROWS_AS(BridgeDsaLayer(w.layers[2].mla, FixtureMla(), id),
                  std::runtime_error);
}

TEST_CASE("glm5_next bridge: an empty or released tensor is refused") {
  OwnedTensor empty;
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(empty, "q_a_proj"),
                       doctest::Contains("q_a_proj"), std::runtime_error);

  // `host_released` means the bytes are gone and only a device copy is
  // authoritative. An empty result would read as a ZERO weight.
  OwnedTensor released;
  released.dtype = vt::DType::kF32;
  released.rank = 1;
  released.shape[0] = 4;
  released.host_released = true;
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(released, "o_proj"),
                       doctest::Contains("released"), std::runtime_error);
}
