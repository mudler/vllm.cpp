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
#include <utility>
#include <vector>

#include "support/glm5_next_gguf_fixture.h"
#include "vllm/model_executor/models/glm5_next_attn.h"
#include "vllm/model_executor/models/glm5_next_bridge.h"
#include "vllm/model_executor/models/glm5_next_dsa.h"
#include "vllm/model_executor/models/glm5_next_moe.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"

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

// --- (4) the FOUR refusals `glm5_next_bridge.h` advertises BY NAME -----------
//
// The header's `REFUSES BY NAME` list has four entries. Before this section two
// of them were pinned (the ceiling, and `host_released`) and the review found
// the rest carrying NO coverage at all: every one could be deleted and this
// suite stayed 8/8 and 56/56. Two of the uncovered ones are not cosmetic —
// without the byte-span checks `std::memcpy(out.data(), src, need)` reads past
// a short buffer and SERVES THE HEAP AS WEIGHT VALUES, and without the `default:`
// arm an encoding the bridge cannot widen returns a buffer of ZEROS, which is
// the exact failure the `host_released` refusal already exists to prevent.
//
// A refusal with no test is a comment. These make each one a gate.

TEST_CASE("glm5_next bridge: a block count that is not whole BLOCKS is refused") {
  // Q8_0 has 32 elements per block. 33 is one element into a second block, so
  // there is no byte layout that could serve it and `to_float` would read a
  // block that was never written.
  OwnedTensor t;
  t.dtype = vt::DType::kQ8_0;
  t.rank = 1;
  t.shape[0] = 33;
  REQUIRE(vt::IsBlockQuant(t.dtype));
  REQUIRE(vt::BlockElems(t.dtype) == 32);
  // Non-empty, so the earlier "carries no bytes" refusal cannot be what fires.
  t.bytes.assign(static_cast<size_t>(vt::RowSizeBytes(t.dtype, 64)), 0U);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "k_b_proj"),
                       doctest::Contains("whole number"), std::runtime_error);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "k_b_proj"),
                       doctest::Contains("k_b_proj"), std::runtime_error);
}

TEST_CASE("glm5_next bridge: a SHORT block byte span is refused, not decoded") {
  // A whole number of blocks, and one byte less than those blocks occupy. The
  // element count agrees, so only the byte-span check separates this from a
  // decode that walks off the end of `t.bytes`.
  OwnedTensor t;
  t.dtype = vt::DType::kQ8_0;
  t.rank = 1;
  t.shape[0] = 32;
  const size_t need = vt::RowSizeBytes(t.dtype, 32);
  REQUIRE(need == 34U);  // one f16 scale + 32 int8 quants
  t.bytes.assign(need - 1U, 0U);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "moe.gate_exps"),
                       doctest::Contains("need"), std::runtime_error);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "moe.gate_exps"),
                       doctest::Contains("moe.gate_exps"), std::runtime_error);

  // A LONGER span is refused too. It cannot over-read, but it means the caller
  // and the bridge disagree about the geometry, and serving the prefix would
  // hide that disagreement behind plausible numbers.
  t.bytes.assign(need + 34U, 0U);
  CHECK_THROWS_AS(DecodeOwnedTensorToF32(t, "moe.gate_exps"),
                  std::runtime_error);

  // ...and the exact span decodes, so the check is a boundary and not a veto.
  t.bytes.assign(need, 0U);
  std::vector<float> ok;
  REQUIRE_NOTHROW(ok = DecodeOwnedTensorToF32(t, "moe.gate_exps"));
  CHECK(ok.size() == 32U);
}

TEST_CASE("glm5_next bridge: EVERY block dtype this build knows has a decoder") {
  // The `to_float == nullptr` refusal cannot be reached from the enum as it
  // stands, and this case is why rather than an assertion that it fires:
  // `vt::IsBlockQuant` is true exactly for the dtypes with block geometry, and
  // `vt::cpu::BlockToFloat` answers non-null for every one of them. So the
  // refusal is the "unselected branch" shape (`.agents/reachability.md`) — a
  // guard for the encoding that lands NEXT without a CPU decoder, which is the
  // state IQ2_XS and IQ4_XS were in before #2245.
  //
  // What is gated here is therefore the PREMISE, not the branch: the moment a
  // new block encoding is added to `vt::DType` without a `BlockToFloat` arm,
  // this case reds and the refusal in `DecodeOwnedTensorToF32` becomes live.
  // Deleting the refusal on its own does NOT red anything, and that is stated
  // rather than dressed up.
  //
  // MEASURED, not argued. Rewriting `BlockToFloat`'s `kQ8_0` arm to return
  // nullptr (`cpu_quant_dequant.cpp`, BUILD rc=0) reds THIS case at
  // `CHECK(vt::cpu::BlockToFloat(d) != nullptr)` and, in the same run, makes
  // the refusal fire by name in two others: "`moe.gate_exps` is q8_0, which
  // this build has no `BlockToFloat` decoder for". So the branch is live under
  // the one condition that can reach it, and the guard is what stands between
  // that condition and a buffer of zeros served as a weight.
  int block_dtypes = 0;
  for (int raw = 0; raw <= 255; ++raw) {
    const auto d = static_cast<vt::DType>(raw);
    if (raw > static_cast<int>(vt::DType::kIQ4_XS)) break;
    if (!vt::IsBlockQuant(d)) continue;
    ++block_dtypes;
    INFO("dtype ", vt::Name(d));
    CHECK(vt::cpu::BlockToFloat(d) != nullptr);
  }
  // A loop that found nothing would pass vacuously, which is the mute-switch
  // shape this repository names. The build carries 18 block encodings.
  CHECK(block_dtypes == 18);
  MESSAGE("block dtypes with a CPU decoder: " << block_dtypes);
}

TEST_CASE("glm5_next bridge: a SHORT elementwise byte span is refused") {
  // THE HEAP OVER-READ. `kF32` decodes with `std::memcpy(out.data(), src,
  // need)`, where `need` comes from the SHAPE. With a shorter `t.bytes` and no
  // check, that memcpy reads whatever follows the vector's allocation and the
  // bridge returns it AS WEIGHT VALUES — finite, plausible, and wrong, which no
  // token gate can see.
  OwnedTensor t;
  t.dtype = vt::DType::kF32;
  t.rank = 1;
  t.shape[0] = 4;
  const size_t need = 4U * vt::SizeOf(t.dtype);
  REQUIRE(need == 16U);
  t.bytes.assign(need - 4U, 0U);  // one element short
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "q_a_layernorm"),
                       doctest::Contains("need"), std::runtime_error);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "q_a_layernorm"),
                       doctest::Contains("q_a_layernorm"), std::runtime_error);

  // The same check covers the WIDENING arms, whose loop reads `numel`
  // half-words rather than memcpying, so it over-reads by a different route.
  OwnedTensor h;
  h.dtype = vt::DType::kBF16;
  h.rank = 2;
  h.shape[0] = 2;
  h.shape[1] = 3;
  h.bytes.assign(6U * sizeof(uint16_t) - 2U, 0U);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(h, "kv_a_layernorm"),
                       doctest::Contains("kv_a_layernorm"), std::runtime_error);

  // ...and the exact span decodes, so neither check is a blanket refusal.
  t.bytes.assign(need, 0U);
  CHECK(DecodeOwnedTensorToF32(t, "q_a_layernorm").size() == 4U);
  h.bytes.assign(6U * sizeof(uint16_t), 0U);
  CHECK(DecodeOwnedTensorToF32(h, "kv_a_layernorm").size() == 6U);
}

TEST_CASE("glm5_next bridge: a NON-FLOAT encoding is refused, not zero-filled") {
  // `kI32` is elementwise, so it passes the byte-span check with a correctly
  // sized span and reaches the `switch`. Nothing widens it, and WITHOUT the
  // `default:` arm the function falls off the end returning the zero-filled
  // `out` it allocated — a whole weight of zeros, served silently. That is the
  // same failure the `host_released` refusal exists to stop, reached by a
  // different door.
  OwnedTensor t;
  t.dtype = vt::DType::kI32;
  t.rank = 1;
  t.shape[0] = 4;
  REQUIRE_FALSE(vt::IsBlockQuant(t.dtype));
  t.bytes.assign(4U * vt::SizeOf(t.dtype), 0U);  // the RIGHT span, on purpose
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "o_proj"),
                       doctest::Contains("widen"), std::runtime_error);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "o_proj"),
                       doctest::Contains("o_proj"), std::runtime_error);
  // The dtype is NAMED, so the refusal says which encoding arrived.
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(t, "o_proj"),
                       doctest::Contains(vt::Name(t.dtype)), std::runtime_error);

  OwnedTensor i8;
  i8.dtype = vt::DType::kI8;
  i8.rank = 1;
  i8.shape[0] = 8;
  i8.bytes.assign(8U, 0U);
  CHECK_THROWS_WITH_AS(DecodeOwnedTensorToF32(i8, "idx_wk"),
                       doctest::Contains("widen"), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════════════
//  W5b-2b — THE PER-EXPERT SOURCE, AND THE ROW RANGE IT IS BUILT ON
// ═══════════════════════════════════════════════════════════════════════════
//
// The four arms `BridgeDsaLayer` does not cover are three mechanical decodes
// and one design problem. These cases pin the design problem and the primitive
// under it. The arithmetic they assert is the whole argument:
//
//   * ONE expert bank is 9.0 GiB in f32 and the 1 GiB ceiling refuses it BY
//     NAME. That is the gate working, not an obstacle to route around.
//   * ONE ROW of that bank is 32 MiB, 32x UNDER the same ceiling.
//   * `num_experts_per_tok` is 8 of 288, so a token needs at most 8 rows.
//
// So the per-expert source decodes rows and never a bank, and the cases below
// assert both halves: the SET of experts decoded is the SET the router
// selected, and the on-demand result is EXACTLY the resident result.

TEST_CASE("glm5_next bridge: an expert ROW is 32 MiB where the BANK is 9.0 GiB") {
  // Every number here is the published geometry's: 288 routed experts,
  // `moe_intermediate_size` 2048, `hidden_size` 4096.
  const int64_t E = 288, I = 2048, H = 4096;

  OwnedTensor bank;  // `ffn_up_exps.weight`, [E, I, H]
  bank.dtype = vt::DType::kIQ2_XS;
  bank.rank = 3;
  bank.shape[0] = E;
  bank.shape[1] = I;
  bank.shape[2] = H;

  // The BANK, and the ceiling refuses it.
  CHECK(HostF32Bytes(bank) == 9663676416);                 // 9.0 GiB
  CHECK(HostF32Bytes(bank) == kBridgeTensorF32ByteCeiling * 9);
  // ONE ROW of the same tensor, from the shape alone.
  CHECK(vllm::glm5_next::HostF32RowBytes(bank) == 33554432);  // 32 MiB
  CHECK(vllm::glm5_next::HostF32RowBytes(bank) * 32 == kBridgeTensorF32ByteCeiling);

  // One EXPERT is the fused `[2I, H]` plus `[H, I]`, which is three rows.
  const int64_t one_expert = 3 * vllm::glm5_next::HostF32RowBytes(bank);
  CHECK(one_expert == 100663296);  // 0.09375 GiB
  // One sparse LAYER's three banks, and the 42 sparse layers together. The
  // second number is what makes a resident float bank impossible rather than
  // expensive: 1,134 GiB against ~119.63 usable on the largest box we reach.
  CHECK(3 * HostF32Bytes(bank) == 28991029248);            // 27.0 GiB
  CHECK(42 * 3 * HostF32Bytes(bank) == 1217623228416);     // 1,134 GiB
  CHECK(GiB(42 * 3 * HostF32Bytes(bank)) > 9.0 * 119.63);  // 9.5x over the box

  MESSAGE("one expert " << GiB(one_expert) << " GiB, one layer's banks "
          << GiB(3 * HostF32Bytes(bank)) << " GiB, all sparse layers "
          << GiB(42 * 3 * HostF32Bytes(bank)) << " GiB, box ~119.63 GiB");
}

TEST_CASE("glm5_next bridge: the ROW API cannot decode a bank through the back door") {
  // The row range is checked against the SAME ceiling, so asking for every row
  // is refused by exactly the arithmetic that refuses the whole tensor. A row
  // API with a per-row ceiling would be a mute switch wearing a gate's name.
  OwnedTensor bank;
  bank.dtype = vt::DType::kIQ2_XS;
  bank.rank = 3;
  bank.shape[0] = 288;
  bank.shape[1] = 2048;
  bank.shape[2] = 4096;
  CHECK(bank.bytes.empty());

  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 0, 288),
      doctest::Contains("ceiling"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 0, 288),
      doctest::Contains("up_exps"), std::runtime_error);
  // 32 rows is exactly the ceiling and passes it; 33 does not. The two sides
  // are asserted because a bound that only ever refuses proves nothing about
  // where it sits.
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 0, 33),
      doctest::Contains("ceiling"), std::runtime_error);
  // 32 rows gets PAST the ceiling and stops on the empty payload instead —
  // a DIFFERENT refusal, which is what proves the ceiling is not what stopped
  // the single-row case.
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 0, 32),
      doctest::Contains("carries no bytes"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 0, 1),
      doctest::Contains("carries no bytes"), std::runtime_error);

  // A range outside the leading axis is refused by name rather than read.
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 288, 1),
      doctest::Contains("rows"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", -1, 1),
      doctest::Contains("rows"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(bank, "moe.up_exps", 0, 0),
      doctest::Contains("rows"), std::runtime_error);
}

TEST_CASE("glm5_next bridge: a row slice reads ITS OWN row, not row 0") {
  // THE OFFSET IS THE WHOLE POINT. A row decoder that ignored `first_row`
  // returns a correctly sized, finite buffer of the WRONG expert's weights,
  // which is the silent failure this row's headers keep naming. Every row is
  // compared elementwise against the same row of the whole-tensor decode, and
  // the rows are asserted to DIFFER from one another so the comparison is
  // discriminating rather than four copies of one value.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);
  REQUIRE(w.layers.size() >= 2U);
  const vllm::Glm5NextLayerWeights& sparse = w.layers[1];
  REQUIRE_FALSE(sparse.is_dense_mlp);

  for (const auto& pair : std::vector<std::pair<const vllm::OwnedTensor*, const char*>>{
           {&sparse.moe.gate_exps, "moe.gate_exps"},
           {&sparse.moe.down_exps, "moe.down_exps"},
           {&w.embed_tokens, "token_embd.weight"}}) {
    const vllm::OwnedTensor& t = *pair.first;
    INFO("tensor ", pair.second);
    REQUIRE(t.rank >= 2);
    const std::vector<float> whole = DecodeOwnedTensorToF32(t, pair.second);
    const int64_t rows = t.shape[0];
    const int64_t row_elems = static_cast<int64_t>(whole.size()) / rows;
    REQUIRE(rows >= 2);
    int distinct_rows = 0;
    for (int64_t r = 0; r < rows; ++r) {
      const std::vector<float> got =
          vllm::glm5_next::DecodeOwnedTensorRowsToF32(t, pair.second, r, 1);
      REQUIRE(static_cast<int64_t>(got.size()) == row_elems);
      bool same_as_row0 = true;
      for (int64_t i = 0; i < row_elems; ++i) {
        const float want = whole[static_cast<size_t>(r * row_elems + i)];
        const float have = got[static_cast<size_t>(i)];
        // isfinite on BOTH sides: an all-NaN decode compares equal to nothing
        // and `NaN != NaN` would read as a difference, while `!(a > b)` on a
        // NaN reads as a match. Both directions are stated.
        REQUIRE(std::isfinite(want));
        REQUIRE(std::isfinite(have));
        CHECK(have == want);
        if (want != whole[static_cast<size_t>(i)]) same_as_row0 = false;
      }
      if (!same_as_row0) ++distinct_rows;
    }
    // At least one row differs from row 0, so "reads its own row" is a claim
    // the fixture can actually falsify.
    CHECK(distinct_rows >= 1);

    // A CONTIGUOUS RANGE is the concatenation of its rows, which is what the
    // streamed `lm_head` relies on.
    const std::vector<float> two =
        vllm::glm5_next::DecodeOwnedTensorRowsToF32(t, pair.second, 1, 2);
    REQUIRE(static_cast<int64_t>(two.size()) == 2 * row_elems);
    for (int64_t i = 0; i < 2 * row_elems; ++i) {
      CHECK(two[static_cast<size_t>(i)] ==
            whole[static_cast<size_t>(row_elems + i)]);
    }
  }
}

TEST_CASE("glm5_next bridge: a row that is not whole BLOCKS is refused") {
  // A block row narrower than one block would make every slice after the first
  // start MID-BLOCK, and a decoder handed a misaligned base does not fail: it
  // reads the next block's scale and returns plausible values from the wrong
  // quantization. `DecodeOwnedTensorToF32` cannot see this, because the WHOLE
  // tensor is a whole number of blocks; only the row API can.
  OwnedTensor t;
  t.dtype = vt::DType::kQ8_0;
  t.rank = 2;
  t.shape[0] = 2;
  t.shape[1] = 16;  // 16 elements per row, and Q8_0 blocks are 32
  REQUIRE(vt::BlockElems(t.dtype) == 32);
  t.bytes.assign(vt::RowSizeBytes(t.dtype, 32), 0U);
  // The whole tensor IS a whole number of blocks, so the old entry point is
  // happy with exactly this tensor — which is what makes the new check load
  // bearing rather than a restatement.
  CHECK(DecodeOwnedTensorToF32(t, "moe.gate_exps").size() == 32U);
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(t, "moe.gate_exps", 1, 1),
      doctest::Contains("mid-block"), std::runtime_error);
  CHECK_THROWS_WITH_AS(
      vllm::glm5_next::DecodeOwnedTensorRowsToF32(t, "moe.gate_exps", 1, 1),
      doctest::Contains("gate_exps"), std::runtime_error);
}

TEST_CASE("glm5_next bridge: ONLY the SELECTED experts are decoded") {
  // The claim in one line: with `num_experts_per_tok` of `n_routed_experts`
  // selected, the source is asked for the SET the router chose and for nothing
  // else, and the result is EXACTLY what the resident banks produce.
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);
  const vllm::Glm5NextParams& p = w.params;
  REQUIRE(w.layers.size() >= 2U);
  const vllm::Glm5NextLayerWeights& src = w.layers[1];
  REQUIRE_FALSE(src.is_dense_mlp);

  const vllm::glm5_next::MoeDims d = vllm::glm5_next::MoeDimsFrom(p);
  REQUIRE(d.n_routed_experts == kExperts);
  REQUIRE(d.num_experts_per_tok == kExpertsPerTok);
  // A STRICT subset is only meaningful when one is possible: 2 of 4 per token.
  REQUIRE(d.num_experts_per_tok < d.n_routed_experts);

  // ── the ON-DEMAND arm ──────────────────────────────────────────────────────
  vllm::glm5_next::MoeLayerWeights on_demand =
      vllm::glm5_next::BridgeMoeLayer(src.moe, d, "blk.1.ffn");
  // The banks are EMPTY by construction. Nothing bridged 27 GiB.
  CHECK(on_demand.expert_gate_up.empty());
  CHECK(on_demand.expert_down.empty());
  vllm::glm5_next::GgufExpertSource source(src.moe, d, "blk.1.ffn");
  on_demand.expert_source = &source;

  // ── the RESIDENT arm, built by hand from the same tensors ─────────────────
  // The fixture's banks are 16 KiB, so decoding them whole here is legal; at
  // the published geometry it is the 9.0 GiB the ceiling refuses, which is why
  // production never does it. This arm exists as the ORACLE for the other one.
  vllm::glm5_next::MoeLayerWeights resident = on_demand;
  resident.expert_source = nullptr;
  {
    const std::vector<float> ge = DecodeOwnedTensorToF32(src.moe.gate_exps, "gate_exps");
    const std::vector<float> ue = DecodeOwnedTensorToF32(src.moe.up_exps, "up_exps");
    resident.expert_down = DecodeOwnedTensorToF32(src.moe.down_exps, "down_exps");
    const int64_t I = d.moe_intermediate_size, H = d.hidden_size;
    resident.expert_gate_up.resize(static_cast<size_t>(d.n_routed_experts * 2 * I * H));
    for (int64_t e = 0; e < d.n_routed_experts; ++e) {
      std::copy_n(ge.begin() + e * I * H, I * H,
                  resident.expert_gate_up.begin() + e * 2 * I * H);
      std::copy_n(ue.begin() + e * I * H, I * H,
                  resident.expert_gate_up.begin() + e * 2 * I * H + I * H);
    }
  }

  // ONE token, so at most `num_experts_per_tok` distinct experts can be hit.
  std::vector<float> hidden(static_cast<size_t>(d.hidden_size));
  for (int64_t i = 0; i < d.hidden_size; ++i) {
    hidden[static_cast<size_t>(i)] =
        0.03F * static_cast<float>((i * 7 + 5) % 23) - 0.3F;
  }
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  const vllm::glm5_next::MoeRouting r =
      vllm::glm5_next::RouteTopk(d, resident, hidden, /*num_tokens=*/1, q);
  std::vector<int64_t> want(r.topk_ids.begin(), r.topk_ids.end());
  std::sort(want.begin(), want.end());
  want.erase(std::unique(want.begin(), want.end()), want.end());
  REQUIRE(static_cast<int64_t>(want.size()) <= d.num_experts_per_tok);
  // The selection is a STRICT subset on this fixture, so "only the selected"
  // has something to exclude. Printed, because a bound nobody can see is a
  // bound nobody can check.
  CHECK(static_cast<int64_t>(want.size()) < d.n_routed_experts);
  MESSAGE("router selected " << want.size() << " of " << d.n_routed_experts
          << " experts for 1 token");

  const std::vector<float> from_source =
      vllm::glm5_next::MoeForward(d, on_demand, hidden, /*num_tokens=*/1, q);
  const std::vector<float> from_banks =
      vllm::glm5_next::MoeForward(d, resident, hidden, /*num_tokens=*/1, q);

  // (a) the SET decoded is the SET selected, and each expert exactly ONCE.
  std::vector<int64_t> got = source.decoded();
  CHECK(got.size() == want.size());
  std::vector<int64_t> uniq = got;
  std::sort(uniq.begin(), uniq.end());
  CHECK(std::adjacent_find(uniq.begin(), uniq.end()) == uniq.end());
  CHECK(uniq == want);

  // ── THE SAME CLAIM AT MORE THAN ONE TOKEN, WHICH IS WHERE "ONCE" BITES ────
  //
  // At one token every selected expert is hit exactly once whatever the code
  // does, so the `adjacent_find` above cannot see a missing grouping. Measured:
  // removing the grouping from `MoeForward` (mutation M13) leaves the
  // single-token case fully GREEN. With several tokens the same expert is hit
  // by more than one of them, and then a per-SLOT visit decodes it twice —
  // which is a second 96 MiB decode at the published geometry and is exactly
  // what the residency argument rules out.
  const int64_t T = 6;
  std::vector<float> many(static_cast<size_t>(T * d.hidden_size));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t i = 0; i < d.hidden_size; ++i) {
      many[static_cast<size_t>(t * d.hidden_size + i)] =
          0.05F * static_cast<float>((i * 3 + 11 * t) % 19) - 0.4F;
    }
  }
  vllm::glm5_next::GgufExpertSource multi(src.moe, d, "blk.1.ffn");
  vllm::glm5_next::MoeLayerWeights on_demand_multi = on_demand;
  on_demand_multi.expert_source = &multi;
  const std::vector<float> multi_src =
      vllm::glm5_next::MoeForward(d, on_demand_multi, many, T, q);
  const std::vector<float> multi_res =
      vllm::glm5_next::MoeForward(d, resident, many, T, q);

  const vllm::glm5_next::MoeRouting rm =
      vllm::glm5_next::RouteTopk(d, resident, many, T, q);
  std::vector<int64_t> want_multi(rm.topk_ids.begin(), rm.topk_ids.end());
  const int64_t slots = static_cast<int64_t>(want_multi.size());
  std::sort(want_multi.begin(), want_multi.end());
  want_multi.erase(std::unique(want_multi.begin(), want_multi.end()),
                   want_multi.end());
  // The routing must actually REPEAT an expert across the tokens, or the
  // "exactly once" claim has nothing to exclude here either. Printed, so a
  // fixture that stopped repeating is visible rather than silently vacuous.
  REQUIRE(static_cast<int64_t>(want_multi.size()) < slots);
  MESSAGE("over " << T << " tokens the router filled " << slots
          << " slots from " << want_multi.size() << " distinct experts");

  std::vector<int64_t> got_multi = multi.decoded();
  CHECK(got_multi.size() == want_multi.size());
  std::sort(got_multi.begin(), got_multi.end());
  CHECK(std::adjacent_find(got_multi.begin(), got_multi.end()) == got_multi.end());
  CHECK(got_multi == want_multi);

  // ...and the values still match the resident banks exactly.
  REQUIRE(multi_src.size() == multi_res.size());
  int multi_nonfinite = 0;
  for (size_t i = 0; i < multi_src.size(); ++i) {
    if (!std::isfinite(multi_src[i]) || !std::isfinite(multi_res[i])) {
      ++multi_nonfinite;
      continue;
    }
    CHECK(multi_src[i] == multi_res[i]);
  }
  CHECK(multi_nonfinite == 0);

  // (b) the values are EXACTLY the resident ones. Not a tolerance: the two
  // paths run the same arithmetic on the same floats, so anything but equality
  // is a defect and a tolerance would hide it.
  REQUIRE(from_source.size() == from_banks.size());
  int nonfinite = 0;
  for (size_t i = 0; i < from_source.size(); ++i) {
    // isfinite guarded on BOTH sides: an all-NaN forward makes every
    // inequality false and reads as a perfect match (W5b-2a's 15th mutation).
    if (!std::isfinite(from_source[i]) || !std::isfinite(from_banks[i])) {
      ++nonfinite;
      continue;
    }
    CHECK(from_source[i] == from_banks[i]);
  }
  CHECK(nonfinite == 0);
  // And the output is not trivially zero, which would make (b) vacuous.
  CHECK(std::any_of(from_banks.begin(), from_banks.end(),
                    [](float v) { return v != 0.0F; }));
}

TEST_CASE("glm5_next bridge: MoeForward refuses NEITHER and BOTH residencies") {
  TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  std::unique_ptr<vllm::LoadedModel> holder;
  const vllm::Glm5NextWeights& w = LoadFixture(holder, g);
  const vllm::glm5_next::MoeDims d = vllm::glm5_next::MoeDimsFrom(w.params);
  const vllm::Glm5NextLayerWeights& src = w.layers[1];

  vllm::glm5_next::MoeLayerWeights bare =
      vllm::glm5_next::BridgeMoeLayer(src.moe, d, "blk.1.ffn");
  std::vector<float> hidden(static_cast<size_t>(d.hidden_size), 0.1F);
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // NEITHER. Without this refusal the router would select and every expert
  // would read as a ZERO weight: a finite, fluent, wrong block.
  CHECK_THROWS_WITH_AS(vllm::glm5_next::MoeForward(d, bare, hidden, 1, q),
                       doctest::Contains("NEITHER"), std::runtime_error);

  // BOTH. There is no rule for which one wins, so the ambiguity is refused
  // rather than resolved silently in favour of whichever the code reads first.
  vllm::glm5_next::GgufExpertSource source(src.moe, d, "blk.1.ffn");
  vllm::glm5_next::MoeLayerWeights both = bare;
  both.expert_source = &source;
  both.expert_gate_up.assign(
      static_cast<size_t>(d.n_routed_experts * 2 * d.moe_intermediate_size *
                          d.hidden_size),
      0.0F);
  both.expert_down.assign(
      static_cast<size_t>(d.n_routed_experts * d.hidden_size *
                          d.moe_intermediate_size),
      0.0F);
  CHECK_THROWS_WITH_AS(vllm::glm5_next::MoeForward(d, both, hidden, 1, q),
                       doctest::Contains("BOTH"), std::runtime_error);

  // A HALF-FILLED resident layer is refused too, by the size check rather than
  // read as a bank of the wrong length.
  vllm::glm5_next::MoeLayerWeights half = bare;
  half.expert_gate_up = both.expert_gate_up;
  CHECK_THROWS_AS(vllm::glm5_next::MoeForward(d, half, hidden, 1, q),
                  std::runtime_error);

  // An out-of-range expert is refused BY NAME by the source itself.
  std::vector<float> gu, dn;
  CHECK_THROWS_WITH_AS(source.Expert(d.n_routed_experts, gu, dn),
                       doctest::Contains("outside"), std::runtime_error);
}
