// MODEL-MM-deepseek-v4 W3B (#2411) — `exp_probs_b_vl`, the SECOND MoE routing
// bias DeepSeek-V4-Flash-Vision carries for image tokens.
//
// WHAT THE ARTIFACT HOLDS. The first shard of the pinned
// `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF` `UD-IQ1_S` build holds 43 tensors
// and nothing else: one `blk.N.exp_probs_b_vl.bias`, F32 `[256]`, for every one
// of the 43 language layers. Read from the shipped header on 2026-09-05 by an
// HTTP range request over the first 14 MB (no download), beside
// `deepseek4.hash_layer_count = 3` and `split.tensors.count = 1371`. The
// safetensors checkpoint spells the same tensor `layers.N.ffn.gate.bias_vl`.
//
// WHY IT IS ON EVERY LAYER, HASH LAYERS INCLUDED. Text tokens on a hash layer
// are routed by the `tid2eid` table and take NO bias, which is why llama.cpp's
// converter drops `ffn.gate.bias` there (`conversion/deepseek.py`, PR #28154 at
// `llama-cpp-dsv4vision`). An image token has no meaningful token id to hash, so
// on those layers `exp_probs_b_vl` replaces the hash routing itself. llama.cpp
// therefore creates `ffn_exp_probs_b_vl` OUTSIDE its hash branch, for every
// layer, with `TENSOR_NOT_REQUIRED` (`src/models/deepseek4.cpp`, same PR).
//
// WHAT THIS WAVE DOES, AND WHAT IT DOES NOT. W3B ACCOUNTS FOR AND LOADS the
// tensor in both weight arms and presents it on the layer weight structs beside
// the text bias. Nothing SELECTS it yet: per-token selection between the two
// biases, the hash-layer replacement at forward time and the non-causal
// image-span window are W4's, are named in the commit body, and are listed under
// `## Owed` in `.agents/specs/deepseek-v4-flash-vision.md`.
//
// THE INERTNESS CLAIM THIS SUITE ALSO PINS. A DeepSeek-V4 TEXT checkpoint
// carries none of these tensors. Their absence must be accepted, and a text
// checkpoint's loaded tower must be unchanged. Making the tensor required makes
// the text cases here red, which is the mutation W3B ran.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <fstream>

#include <nlohmann/json.hpp>

#include "deepseek_v4_lang_gguf_fixture.h"
#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
#include "vllm/model_executor/models/deepseek_v4_mm.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

#include "dsv4_exl3_fixture.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::I32ArrayKv;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

using dsv4_lang_test::BiasWidths;
using dsv4_lang_test::BuildDeepseek4Gguf;
using dsv4_lang_test::TextBiasFill;
using dsv4_lang_test::VisionBiasFill;
using dsv4_lang_test::kExperts;
using dsv4_lang_test::kH;
using dsv4_lang_test::kHashLayers;
using dsv4_lang_test::kLayers;
using dsv4_lang_test::kUsed;
using dsv4_lang_test::kVocab;

vllm::GgufLoadPolicy KeepPolicy() {
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  return pol;
}

// ─── the safetensors arm's fixture, composed WITHOUT editing the shared one ──
// `dsv4_exl3_fixture.h` is read by three other suites, and a fixture option is a
// shared surface. The vision tensors are appended to the carried entries here
// instead, which is the same thing the checkpoint does.
using dsv4_exl3_fixture::CarriedEntries;
using dsv4_exl3_fixture::Fixture;
using dsv4_exl3_fixture::FixtureConfig;
using dsv4_exl3_fixture::FixtureOptions;
using dsv4_exl3_fixture::RankEntries;
using dsv4_exl3_fixture::StEntry;
using dsv4_exl3_fixture::ThrowMessage;
using dsv4_exl3_fixture::WriteSafetensors;

std::unique_ptr<Fixture> BuildStFixture(const FixtureOptions& opt, bool vision) {
  auto f = std::make_unique<Fixture>();
  f->config = FixtureConfig(opt);
  std::vector<StEntry> carried = CarriedEntries(opt);
  if (vision) {
    for (int l = 0; l < opt.layers; ++l) {
      // Distinguishable from `gate.bias`, which `CarriedEntries` writes at
      // scale 0.3 / center 0.0 — see the two GGUF fills above for why.
      carried.push_back(dsv4_exl3_fixture::F32Entry(
          "layers." + std::to_string(l) + ".ffn.gate.bias_vl",
          {dsv4_exl3_fixture::kExperts}, 0.9f, -1.0f));
    }
  }
  f->shards.push_back(vllm::SafetensorsFile::Open(
      WriteSafetensors(f->dir.path() / "carried-001.safetensors", carried)));
  const int rank_shards = opt.dense_routed_experts ? 0 : opt.ranks_written;
  for (int r = 0; r < rank_shards; ++r) {
    f->shards.push_back(vllm::SafetensorsFile::Open(WriteSafetensors(
        f->dir.path() / ("exl3-layer-000-tp4-rank" + std::to_string(r) + ".safetensors"),
        RankEntries(r, opt))));
  }
  return f;
}

// A two-layer model with one hash layer, so both routing shapes are present.
FixtureOptions TwoLayerHashOptions() {
  FixtureOptions opt;
  opt.layers = 2;
  opt.num_hash_layers = 1;
  opt.compress_ratios = {0, 0};
  return opt;
}

}  // namespace

// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("dsv4 vision GGUF: exp_probs_b_vl is accounted for and loaded on EVERY layer") {
  TempFile file(BuildDeepseek4Gguf(/*vision=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol);

  // Totality: the vision bias is routed, so nothing is left over and the
  // accounted count still equals the file's own tensor count.
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  REQUIRE(w.gguf.layers.size() == static_cast<size_t>(kLayers));
  REQUIRE(w.host.layers.size() == static_cast<size_t>(kLayers));

  for (int64_t l = 0; l < kLayers; ++l) {
    CAPTURE(l);
    const vllm::DeepseekV4GgufLayerWeights& lw = w.gguf.layers[static_cast<size_t>(l)];
    const vllm::DeepseekV4LayerHostWeights& hl = w.host.layers[static_cast<size_t>(l)];

    // Present on the keep-quant tower as an f32 `[E]` vector...
    REQUIRE_FALSE(lw.e_score_bias_vl.Empty());
    CHECK(lw.e_score_bias_vl.dtype == vt::DType::kF32);
    REQUIRE(lw.e_score_bias_vl.rank == 1);
    CHECK(lw.e_score_bias_vl.shape[0] == kExperts);

    // ...and on the host tower, holding the VISION values, not the text ones.
    REQUIRE(hl.gate_bias_vl.size() == static_cast<size_t>(kExperts));
    for (int64_t i = 0; i < kExperts; ++i)
      CHECK(hl.gate_bias_vl[static_cast<size_t>(i)] ==
            doctest::Approx(VisionBiasFill(l, i)));
  }
}

TEST_CASE("dsv4 vision GGUF: a HASH layer carries the vision bias and NO text bias") {
  TempFile file(BuildDeepseek4Gguf(/*vision=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const vllm::DeepseekV4Weights w =
      vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol);

  // Layer 0 is the hash layer. Text tokens there route through `tid2eid` and
  // take no bias at all, which is why the converter emits no `exp_probs_b` for
  // it; the vision bias is what an image token routes on instead. Losing this
  // distinction is the defect a per-layer count could not see.
  const vllm::DeepseekV4GgufLayerWeights& hash = w.gguf.layers[0];
  CHECK(hash.is_hash);
  CHECK_FALSE(hash.tid2eid.Empty());
  CHECK(hash.e_score_bias.Empty());
  CHECK_FALSE(hash.e_score_bias_vl.Empty());
  CHECK_FALSE(w.host.layers[0].tid2eid.empty());
  CHECK(w.host.layers[0].gate_bias.empty());
  CHECK_FALSE(w.host.layers[0].gate_bias_vl.empty());

  // A gated layer carries BOTH, and they hold different values.
  const vllm::DeepseekV4GgufLayerWeights& gated = w.gguf.layers[1];
  CHECK_FALSE(gated.is_hash);
  CHECK_FALSE(gated.e_score_bias.Empty());
  CHECK_FALSE(gated.e_score_bias_vl.Empty());
  const std::vector<float>& text = w.host.layers[1].gate_bias;
  const std::vector<float>& vl = w.host.layers[1].gate_bias_vl;
  REQUIRE(text.size() == static_cast<size_t>(kExperts));
  REQUIRE(vl.size() == static_cast<size_t>(kExperts));
  for (size_t i = 0; i < text.size(); ++i) CHECK(text[i] != doctest::Approx(vl[i]));
}

TEST_CASE("dsv4 TEXT GGUF: the absent vision bias is accepted and changes nothing") {
  TempFile file(BuildDeepseek4Gguf(/*vision=*/false));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  vllm::DeepseekV4Weights w;
  // Captured rather than bare: a vision bias made REQUIRED throws here, and an
  // uncaught throw is a failed CASE whose summary line still reads
  // `assertions: N | N passed`.
  const std::string msg = ThrowMessage(
      [&] { w = vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol); });
  CAPTURE(msg);
  REQUIRE(msg.empty());

  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  REQUIRE(w.gguf.layers.size() == static_cast<size_t>(kLayers));
  for (int64_t l = 0; l < kLayers; ++l) {
    CAPTURE(l);
    CHECK(w.gguf.layers[static_cast<size_t>(l)].e_score_bias_vl.Empty());
    CHECK(w.host.layers[static_cast<size_t>(l)].gate_bias_vl.empty());
  }
  // The text bias is untouched on the gated layers.
  for (int64_t l = kHashLayers; l < kLayers; ++l) {
    CAPTURE(l);
    const std::vector<float>& text = w.host.layers[static_cast<size_t>(l)].gate_bias;
    REQUIRE(text.size() == static_cast<size_t>(kExperts));
    for (int64_t i = 0; i < kExperts; ++i)
      CHECK(text[static_cast<size_t>(i)] == doctest::Approx(TextBiasFill(l, i)));
  }
}

// A `[E]` assertion on the LOADED vector is right only by construction while the
// fixture is the only thing that decides the width. These two cases move the
// decision to the loader: the file declares `expert_count` in its KV and writes a
// NARROWER tensor, which is what an artifact re-quantized in place under an
// unchanged name can ship. `Vec` checks residency and role and no geometry at
// all, so before the `Vec1D` repair both of these loaded in silence and only the
// suite's own read-back noticed — which measures the fixture, not the loader.
TEST_CASE("dsv4 vision GGUF: a NARROW exp_probs_b_vl is REFUSED, not read past") {
  BiasWidths narrow;
  narrow.vision = kExperts - 1;
  TempFile file(BuildDeepseek4Gguf(/*vision=*/true, narrow));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const std::string msg = ThrowMessage(
      [&] { (void)vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol); });
  CAPTURE(msg);
  // Named, so the refusal says WHICH tensor and WHAT width it owed — a bare
  // "shape mismatch" would leave the operator to find that out themselves.
  CHECK(msg.find("exp_probs_b_vl.bias") != std::string::npos);
  CHECK(msg.find("[" + std::to_string(kExperts) + "]") != std::string::npos);
}

TEST_CASE("dsv4 TEXT GGUF: a NARROW exp_probs_b is REFUSED too") {
  BiasWidths narrow;
  narrow.text = kExperts - 1;
  TempFile file(BuildDeepseek4Gguf(/*vision=*/false, narrow));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  const std::string msg = ThrowMessage(
      [&] { (void)vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol); });
  CAPTURE(msg);
  CHECK(msg.find("exp_probs_b.bias") != std::string::npos);
  CHECK(msg.find("[" + std::to_string(kExperts) + "]") != std::string::npos);
}

// The DECISION about per-layer optionality, made executable. llama.cpp declares
// `ffn_exp_probs_b_vl` with `TENSOR_NOT_REQUIRED` for every layer independently
// (`src/models/deepseek4.cpp`, PR #28154 at `llama-cpp-dsv4vision`), so a file
// converted with the tensor on only some layers LOADS there. This arm mirrors
// that rather than enforcing the all-or-nothing dichotomy W3B's own prose
// describes, because refusing a file the oracle accepts is a divergence, and
// because the empty slot is a state the consumer must already handle: a text
// checkpoint presents it on EVERY layer. Pinned here so that switching to a
// refusal is a red test somebody has to argue with, not a silent change.
TEST_CASE("dsv4 PARTIAL GGUF: the vision bias is optional PER LAYER, as in llama.cpp") {
  TempFile file(BuildDeepseek4Gguf(/*vision=*/true, BiasWidths{},
                                   /*vision_from=*/kLayers - 1));
  const vllm::GgufFile g = vllm::GgufFile::Open(file.path());
  const vllm::GgufLoadPolicy pol = KeepPolicy();
  vllm::DeepseekV4Weights w;
  const std::string msg = ThrowMessage(
      [&] { w = vllm::LoadDeepseekV4FromGguf(g, vllm::HfConfig{}, &pol); });
  CAPTURE(msg);
  REQUIRE(msg.empty());
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  REQUIRE(w.host.layers.size() == static_cast<size_t>(kLayers));
  for (int64_t l = 0; l < kLayers - 1; ++l) {
    CAPTURE(l);
    CHECK(w.gguf.layers[static_cast<size_t>(l)].e_score_bias_vl.Empty());
    CHECK(w.host.layers[static_cast<size_t>(l)].gate_bias_vl.empty());
  }
  const int64_t last = kLayers - 1;
  const std::vector<float>& vl = w.host.layers[static_cast<size_t>(last)].gate_bias_vl;
  REQUIRE(vl.size() == static_cast<size_t>(kExperts));
  for (int64_t i = 0; i < kExperts; ++i)
    CHECK(vl[static_cast<size_t>(i)] == doctest::Approx(VisionBiasFill(last, i)));
}

// ───────────────────────────────────────────────────────────────────────────
TEST_CASE("dsv4 vision safetensors: the EXL3 carried arm routes and loads gate.bias_vl") {
  const FixtureOptions opt = TwoLayerHashOptions();
  auto f = BuildStFixture(opt, /*vision=*/true);
  vllm::DeepseekV4Weights w;
  // Before W3B this arm REFUSED the checkpoint outright: its totality pass
  // rejects any tensor no arm routes, by name.
  const std::string msg = ThrowMessage(
      [&] { w = vllm::LoadDeepseekV4ForCausalLMWeights(f->shards, f->config); });
  CAPTURE(msg);
  REQUIRE(msg.empty());

  REQUIRE(w.host.layers.size() == static_cast<size_t>(opt.layers));
  // The BYTES, on every layer, against the generator the fixture wrote them
  // with. A width check and a `text != vl` inequality both survive the two
  // defects that matter here: a slot filled with zeros is still `[E]` wide, and
  // a SWAP of the two biases still leaves them unequal. Only the fixture's own
  // value for THIS name can say that THIS tensor reached THIS slot. The
  // scale/center pair repeats `BuildStFixture`'s `F32Entry` call above.
  for (int l = 0; l < opt.layers; ++l) {
    CAPTURE(l);
    const std::string name = "layers." + std::to_string(l) + ".ffn.gate.bias_vl";
    const std::vector<float>& vl = w.host.layers[static_cast<size_t>(l)].gate_bias_vl;
    REQUIRE(vl.size() == static_cast<size_t>(dsv4_exl3_fixture::kExperts));
    for (int64_t i = 0; i < dsv4_exl3_fixture::kExperts; ++i) {
      CAPTURE(i);
      CHECK(vl[static_cast<size_t>(i)] ==
            doctest::Approx(dsv4_exl3_fixture::CarriedValue(name, i, 0.9f, -1.0f)));
    }
  }
  // The hash layer still carries its table and no text bias; the gated layer
  // carries a text bias that differs from the vision one.
  CHECK_FALSE(w.host.layers[0].tid2eid.empty());
  CHECK(w.host.layers[0].gate_bias.empty());
  CHECK_FALSE(w.host.layers[0].gate_bias_vl.empty());
  const std::vector<float>& text = w.host.layers[1].gate_bias;
  const std::vector<float>& vl = w.host.layers[1].gate_bias_vl;
  REQUIRE(text.size() == vl.size());
  for (size_t i = 0; i < text.size(); ++i) CHECK(text[i] != doctest::Approx(vl[i]));
  // The OTHER half of the swap. `gate_bias_vl` asserted alone catches a
  // transposition on the layer that carries both, but this states the text slot
  // independently, so a one-way misroute cannot hide behind the vision check.
  for (int64_t i = 0; i < dsv4_exl3_fixture::kExperts; ++i) {
    CAPTURE(i);
    CHECK(text[static_cast<size_t>(i)] ==
          doctest::Approx(dsv4_exl3_fixture::CarriedValue("layers.1.ffn.gate.bias", i,
                                                          0.3f, 0.0f)));
  }
}

TEST_CASE("dsv4 TEXT safetensors: the EXL3 carried arm is byte-identical without it") {
  const FixtureOptions opt = TwoLayerHashOptions();
  auto plain = BuildStFixture(opt, /*vision=*/false);
  vllm::DeepseekV4Weights w;
  const std::string msg = ThrowMessage([&] {
    w = vllm::LoadDeepseekV4ForCausalLMWeights(plain->shards, plain->config);
  });
  CAPTURE(msg);
  REQUIRE(msg.empty());
  REQUIRE(w.host.layers.size() == static_cast<size_t>(opt.layers));
  for (int l = 0; l < opt.layers; ++l) {
    CAPTURE(l);
    CHECK(w.host.layers[static_cast<size_t>(l)].gate_bias_vl.empty());
  }

  // And the vision checkpoint accounts for EXACTLY `layers` more tensors than
  // the text one — no more, no fewer, so a bias counted twice or a layer skipped
  // is visible as a number rather than as a slot that happens to be filled.
  auto vision = BuildStFixture(opt, /*vision=*/true);
  const vllm::DeepseekV4Weights wv =
      vllm::LoadDeepseekV4ForCausalLMWeights(vision->shards, vision->config);
  CHECK(wv.accounted_tensors == w.accounted_tensors + opt.layers);
}

TEST_CASE("dsv4 vision safetensors: the OFFICIAL dense arm accounts for gate.bias_vl") {
  FixtureOptions opt = TwoLayerHashOptions();
  // The `deepseek_v4_fp8` vehicle: dense NVFP4 routed experts and no EXL3
  // rank shards, which is the arm the released vision safetensors takes.
  opt.quant_method = "fp8";
  opt.dense_routed_experts = true;

  auto plain = BuildStFixture(opt, /*vision=*/false);
  const vllm::DeepseekV4Weights text =
      vllm::LoadDeepseekV4ForCausalLMWeights(plain->shards, plain->config);
  CHECK_FALSE(text.has_exl3_weights);

  auto vision = BuildStFixture(opt, /*vision=*/true);
  vllm::DeepseekV4Weights w;
  const std::string msg = ThrowMessage([&] {
    w = vllm::LoadDeepseekV4ForCausalLMWeights(vision->shards, vision->config);
  });
  CAPTURE(msg);
  REQUIRE(msg.empty());
  CHECK(w.accounted_tensors == text.accounted_tensors + opt.layers);
}

// ═══════════════════════════════════════════════════════════════════════════
// THE OFFICIAL SAFETENSORS VISION ARM (#2411).
//
// Until `deepseek_v4_vision_weights.cpp` landed, the released checkpoint's 267
// vision tensors were read by NOTHING: `LoadDeepseekV4ForCausalLM`'s
// safetensors branch built no tower, the dense name-map pass did not count them
// (and has no leftover refusal, so they were not even noticed), and an image
// request on that arm refused inside `encode_mm`. Only the two-file GGUF
// vehicle could carry a tower.
//
// EVERY CASE BELOW IS SYNTHETIC. The real artifact is 156.287 GiB across 48
// shards and is staged on no gate device here, so its PAYLOAD HAS NEVER BEEN
// READ. What pins these cases to the real thing is the pair of committed
// manifests at the bottom of this file, which were taken from the pinned
// revision's `config.json` and shard-1 safetensors HEADER.
// ═══════════════════════════════════════════════════════════════════════════
namespace {

// Deliberately distinct in every axis, so a transposed or mis-strided read
// cannot pass by symmetry: `inter` is neither `dim` nor `2 * dim`, the patch
// and downsample sizes differ, and the aligner's input width (dim * ratio^2)
// equals no other width in the fixture.
constexpr int64_t kVisDim = 8;
constexpr int64_t kVisHeads = 2;  // head_dim 4, which the 2-D RoPE needs
constexpr int64_t kVisDepth = 2;
constexpr int64_t kVisInter = 6;
constexpr int64_t kVisPatch = 2;
constexpr int64_t kVisRatio = 3;
// The fill the vision entries are written with. Distinct from the 0.3/0.0 and
// 0.9/-1.0 pairs the carried text tensors use above, so a vision row that
// actually came from a text tensor is visible in its VALUE.
constexpr float kVisScale = 0.7F;
constexpr float kVisCenter = 0.1F;

// The vision keys the released `config.json` carries, added to the shared
// fixture's config HERE rather than as a `FixtureOptions` field, for the reason
// the text half of this file already gives: a fixture option is a shared
// surface read by three other suites, and this is a property of ONE checkpoint.
vllm::HfConfig VisionFixtureConfig(const FixtureOptions& opt) {
  vllm::HfConfig cfg = FixtureConfig(opt);
  cfg.raw["vision_n_layers"] = kVisDepth;
  cfg.raw["vision_dim"] = kVisDim;
  cfg.raw["vision_n_heads"] = kVisHeads;
  cfg.raw["vision_inter_dim"] = kVisInter;
  cfg.raw["vision_patch_size"] = kVisPatch;
  cfg.raw["vision_downsample_ratio"] = kVisRatio;
  cfg.raw["vision_rope_theta"] = 10000.0;
  cfg.raw["vision_max_n_token"] = 12;
  cfg.raw["vision_min_pixels"] = 48;
  cfg.raw["vision_max_wh_ratio"] = 8;
  return cfg;
}

// The released shapes, written out INDEPENDENTLY of the loader's own map. This
// is the description the loader is held to; if the two ever disagree, one of
// them is wrong and these cases say which name.
std::vector<int64_t> OfficialVisionShape(const std::string& name, int64_t hidden) {
  if (name == "vision.patch_embed.proj.weight")
    return {kVisDim, 3 * kVisPatch * kVisPatch};
  if (name == "vision.patch_embed.proj.bias" || name == "vision.norm.weight" ||
      name.ends_with("norm1.weight") || name.ends_with("norm2.weight") ||
      name.ends_with("attn.wo.bias"))
    return {kVisDim};
  if (name.ends_with("attn.wqkv.weight")) return {3 * kVisDim, kVisDim};
  if (name.ends_with("attn.wqkv.bias")) return {3 * kVisDim};
  if (name.ends_with("attn.wo.weight")) return {kVisDim, kVisDim};
  if (name.ends_with("mlp.w1.weight")) return {2 * kVisInter, kVisDim};
  if (name.ends_with("mlp.w2.weight")) return {kVisDim, kVisInter};
  if (name == "aligner.w1.weight") return {hidden, kVisDim * kVisRatio * kVisRatio};
  if (name == "aligner.w2.weight") return {hidden, hidden};
  if (name.starts_with("aligner.") || name.starts_with("image_")) return {hidden};
  throw std::runtime_error("test: no vision shape rule for " + name);
}

// The 27 names of this fixture's group: 2 + 8 per block + 1 + 4 + 4. At the
// released depth 32 the same rule gives 267, which is the pinned header's count
// and what `## the pinned manifests` below asserts.
std::vector<std::string> OfficialVisionNames() {
  std::vector<std::string> out{"vision.patch_embed.proj.weight",
                               "vision.patch_embed.proj.bias"};
  for (int64_t l = 0; l < kVisDepth; ++l) {
    const std::string p = "vision.blocks." + std::to_string(l) + ".";
    for (const char* stem : {"norm1.weight", "attn.wqkv.weight",
                             "attn.wqkv.bias", "attn.wo.weight",
                             "attn.wo.bias", "norm2.weight", "mlp.w1.weight",
                             "mlp.w2.weight"})
      out.push_back(p + stem);
  }
  out.emplace_back("vision.norm.weight");
  for (const char* n : {"aligner.w1.weight", "aligner.w1.bias",
                        "aligner.w2.weight", "aligner.w2.bias"})
    out.emplace_back(n);
  for (const char* n : {"image_start", "image_end", "image_newline", "image_pad"})
    out.emplace_back(n);
  return out;
}

std::vector<StEntry> OfficialVisionEntries(int64_t hidden) {
  std::vector<StEntry> out;
  for (const std::string& name : OfficialVisionNames()) {
    out.push_back(dsv4_exl3_fixture::Bf16Entry(
        name, OfficialVisionShape(name, hidden), kVisScale, kVisCenter));
  }
  return out;
}

// The bf16 WORD the fixture wrote at flat index `i` of `name`. Comparing words
// rather than floats is what makes "this tensor reached this slot" checkable:
// two different tensors have different words at the same index.
uint16_t VisionWord(const std::string& name, int64_t i) {
  return vt::F32ToBF16(
      dsv4_exl3_fixture::CarriedValue(name, i, kVisScale, kVisCenter));
}

// A vision checkpoint: the carried text tensors plus the official vision group.
// `drop` and `retype` and `reshape` inject exactly one defect, so each refusal
// case differs from the loading case in one tensor and nothing else.
struct VisionFixtureEdit {
  std::string drop;
  std::string retype;      // rewrite this name's dtype to F32
  std::string reshape;     // transpose this name's first two dimensions
  std::string duplicate;   // write this name into a SECOND shard as well
};

std::unique_ptr<Fixture> BuildVisionFixture(const FixtureOptions& opt,
                                            const VisionFixtureEdit& edit = {}) {
  auto f = std::make_unique<Fixture>();
  f->config = VisionFixtureConfig(opt);
  const int64_t hidden = f->config.hidden_size;
  std::vector<StEntry> carried = CarriedEntries(opt);
  for (int l = 0; l < opt.layers; ++l) {
    carried.push_back(dsv4_exl3_fixture::F32Entry(
        "layers." + std::to_string(l) + ".ffn.gate.bias_vl",
        {dsv4_exl3_fixture::kExperts}, 0.9f, -1.0f));
  }
  std::vector<StEntry> second;
  for (StEntry& entry : OfficialVisionEntries(hidden)) {
    if (entry.name == edit.drop) continue;
    // A REAL f32 tensor, payload and all. Rewriting only the header dtype would
    // leave a file whose data_offsets no longer match its dtype, and
    // `SafetensorsFile::Open` refuses THAT before this loader is ever called --
    // which would make this case gate the container reader rather than the
    // storage-variant refusal it is here to gate.
    if (entry.name == edit.retype)
      entry = dsv4_exl3_fixture::F32Entry(entry.name, entry.shape, kVisScale,
                                          kVisCenter);
    if (entry.name == edit.reshape && entry.shape.size() >= 2)
      std::swap(entry.shape[0], entry.shape[1]);
    if (entry.name == edit.duplicate) second.push_back(entry);
    carried.push_back(std::move(entry));
  }
  f->shards.push_back(vllm::SafetensorsFile::Open(
      WriteSafetensors(f->dir.path() / "carried-001.safetensors", carried)));
  if (!second.empty()) {
    f->shards.push_back(vllm::SafetensorsFile::Open(
        WriteSafetensors(f->dir.path() / "carried-002.safetensors", second)));
  }
  const int rank_shards = opt.dense_routed_experts ? 0 : opt.ranks_written;
  for (int r = 0; r < rank_shards; ++r) {
    f->shards.push_back(vllm::SafetensorsFile::Open(WriteSafetensors(
        f->dir.path() / ("exl3-layer-000-tp4-rank" + std::to_string(r) + ".safetensors"),
        RankEntries(r, opt))));
  }
  return f;
}

// The released vehicle's shape: dense NVFP4 routed experts, no EXL3 rank shards.
FixtureOptions OfficialVisionOptions() {
  FixtureOptions opt = TwoLayerHashOptions();
  opt.quant_method = "fp8";
  opt.dense_routed_experts = true;
  return opt;
}

// The OTHER vehicle's shape: `quant_method` stays "exl3" and
// `dense_routed_experts` stays false, so `BuildVisionFixture` writes the four
// EXL3 rank shards and `IsExl3Checkpoint` routes the load into
// `LoadDeepseekV4Exl3` instead of the dense name-map arm.
//
// THIS IS WHY THE EXL3 ARM'S VISION ACCOUNTING WAS UNGATED. Every vision case in
// this file used `OfficialVisionOptions`, which is the RELEASED vehicle's dense
// shape, so no case ever reached the EXL3 arm's own copy of the block. Deleting
// that block left BOTH loader suites fully green -- 16 of 16 here and 22 of 22
// in `test_deepseek_v4_exl3_loader` -- while its dense twin reds exactly one
// case (fresh review, 2026-09-12).
FixtureOptions Exl3VisionOptions() { return TwoLayerHashOptions(); }

nlohmann::json ReadJsonFixture(const std::string& path) {
  std::ifstream in(path);
  REQUIRE_MESSAGE(in.good(), "cannot open fixture ", path);
  return nlohmann::json::parse(in);
}

uint64_t Fnv1aLines(const std::vector<std::string>& lines) {
  uint64_t hash = 1469598103934665603ull;
  for (const std::string& line : lines) {
    for (unsigned char c : line) {
      hash ^= static_cast<uint64_t>(c);
      hash *= 1099511628211ull;
    }
    hash ^= static_cast<uint64_t>('\n');
    hash *= 1099511628211ull;
  }
  return hash;
}

// The same three-way partition `scripts/check-deepseek-v4-vision-manifests.py`
// applies. Two descriptions of one rule, held to ONE committed manifest, so a
// change to either that the other does not make turns this suite red.
std::string IndexClass(const std::string& name) {
  if (name.starts_with("vision.") || name.starts_with("aligner.") ||
      name == "image_start" || name == "image_end" ||
      name == "image_newline" || name == "image_pad")
    return "vision";
  if (name.starts_with("mtp.")) return "mtp";
  return "language";
}

}  // namespace

TEST_CASE("dsv4 vision safetensors: the EXL3 arm ROUTES and ACCOUNTS FOR the official vision group") {
  const FixtureOptions opt = Exl3VisionOptions();

  // The same checkpoint through the same arm WITHOUT the vision group, so the
  // group is counted as a DIFFERENCE rather than as an absolute. A change to the
  // carried half then cannot absorb a miscounted vision tensor.
  auto text = BuildStFixture(opt, /*vision=*/true);
  vllm::DeepseekV4Weights wt;
  const std::string text_msg = ThrowMessage([&] {
    wt = vllm::LoadDeepseekV4ForCausalLMWeights(text->shards, text->config);
  });
  CAPTURE(text_msg);
  REQUIRE(text_msg.empty());
  REQUIRE(wt.has_exl3_weights);

  auto vision = BuildVisionFixture(opt);
  vllm::DeepseekV4Weights wv;
  const std::string msg = ThrowMessage([&] {
    wv = vllm::LoadDeepseekV4ForCausalLMWeights(vision->shards, vision->config);
  });
  // THE RED THIS CASE EXISTS FOR. This arm's totality pass REFUSES BY NAME any
  // checkpoint tensor no arm routes, so with the vision accounting block gone
  // the load throws on `vision.patch_embed.proj.weight` and this line fails
  // carrying that refusal as its message.
  CAPTURE(msg);
  REQUIRE(msg.empty());

  // It took the EXL3 arm. Without this, the case would pass against the DENSE
  // twin's vision block, which the `OfficialVisionOptions` cases already gate.
  REQUIRE(wv.has_exl3_weights);

  // Counted EXACTLY once each. The difference alone would be a tautology,
  // because the loader walks this same list; the independent `27` pins the
  // list's SIZE to this fixture's geometry -- 2 + 8 per block + 1 + 4 + 4 at
  // depth 2 -- which is the rule that gives the released 267 at depth 32.
  const std::vector<std::string> group =
      vllm::DeepSeekV4OfficialVisionExpectedTensors(
          vllm::DeepSeekV4OfficialVisionConfig(vision->config));
  CHECK(group.size() == 27);
  CHECK(wv.accounted_tensors ==
        wt.accounted_tensors + static_cast<int64_t>(group.size()));
}

TEST_CASE("official vision safetensors fill every W2 field and outlive the shards") {
  const FixtureOptions opt = OfficialVisionOptions();
  vllm::DeepSeekV4ClipMmproj tower;
  vllm::multimodal::DeepSeekV4VisionConfig cfg;
  int64_t hidden = 0;
  {
    auto f = BuildVisionFixture(opt);
    hidden = f->config.hidden_size;
    cfg = vllm::DeepSeekV4OfficialVisionConfig(f->config);
    CHECK(cfg.hidden_size == kVisDim);
    CHECK(cfg.num_heads == kVisHeads);
    CHECK(cfg.depth == kVisDepth);
    CHECK(cfg.intermediate_size == kVisInter);
    CHECK(cfg.patch_size == kVisPatch);
    CHECK(cfg.downsample_ratio == kVisRatio);
    // The aligner lands in the TEXT hidden space, so the output width is the
    // language model's rather than a vision key.
    CHECK(cfg.output_size == hidden);
    CHECK(cfg.compute_dtype == vt::DType::kBF16);

    const std::string msg = ThrowMessage([&] {
      tower = vllm::LoadDeepSeekV4VisionFromSafetensors(f->shards, cfg);
    });
    CAPTURE(msg);
    REQUIRE(msg.empty());
  }
  // THE SHARDS ARE CLOSED HERE. Every view below therefore points into storage
  // the result owns; a loader that borrowed the mmap instead reads freed pages.
  const vllm::multimodal::DeepSeekV4VisionWeights& w = tower.weights;
  REQUIRE(w.blocks.size() == static_cast<size_t>(kVisDepth));

  const auto check_bf16 = [&](const std::string& name, const vt::Tensor& t,
                              std::vector<int64_t> shape) {
    CAPTURE(name);
    REQUIRE(t.data != nullptr);
    CHECK(t.dtype == vt::DType::kBF16);
    CHECK(t.IsContiguous());
    REQUIRE(t.rank == static_cast<int>(shape.size()));
    for (size_t i = 0; i < shape.size(); ++i) CHECK(t.shape[i] == shape[i]);
    // FIRST and LAST word, so a tensor that reached the right slot at the wrong
    // length or stride is visible too.
    const int64_t n = dsv4_exl3_fixture::Numel(shape);
    CHECK(t.Ptr<uint16_t>()[0] == VisionWord(name, 0));
    CHECK(t.Ptr<uint16_t>()[n - 1] == VisionWord(name, n - 1));
  };
  // An RMSNorm weight is the ONE dtype this arm changes on the way in: stored
  // bf16, widened once to f32 because the pinned module applies the affine in
  // f32. `ValidateWeights` requires f32 for exactly these three names.
  const auto check_norm = [&](const std::string& name, const vt::Tensor& t) {
    CAPTURE(name);
    REQUIRE(t.data != nullptr);
    CHECK(t.dtype == vt::DType::kF32);
    REQUIRE(t.rank == 1);
    CHECK(t.shape[0] == kVisDim);
    CHECK(t.Ptr<float>()[0] == doctest::Approx(vt::BF16ToF32(VisionWord(name, 0))));
    CHECK(t.Ptr<float>()[kVisDim - 1] ==
          doctest::Approx(vt::BF16ToF32(VisionWord(name, kVisDim - 1))));
  };

  check_bf16("vision.patch_embed.proj.weight", w.patch_weight,
             {kVisDim, 3 * kVisPatch * kVisPatch});
  check_bf16("vision.patch_embed.proj.bias", w.patch_bias, {kVisDim});
  for (int64_t l = 0; l < kVisDepth; ++l) {
    const std::string p = "vision.blocks." + std::to_string(l) + ".";
    const auto& b = w.blocks[static_cast<size_t>(l)];
    check_norm(p + "norm1.weight", b.norm1_weight);
    // FUSED ON DISK in q, k, v row order — the order the tower slices them back
    // out at. A reader that split or permuted them stays fluent and is wrong.
    check_bf16(p + "attn.wqkv.weight", b.qkv_weight, {3 * kVisDim, kVisDim});
    check_bf16(p + "attn.wqkv.bias", b.qkv_bias, {3 * kVisDim});
    check_bf16(p + "attn.wo.weight", b.out_weight, {kVisDim, kVisDim});
    check_bf16(p + "attn.wo.bias", b.out_bias, {kVisDim});
    check_norm(p + "norm2.weight", b.norm2_weight);
    // ALSO FUSED, GATE FIRST: `vt::SiluAndMul` reads the gate at column j and
    // the up at column d + j, so the first `inter` rows must be the gate.
    check_bf16(p + "mlp.w1.weight", b.mlp_w1_weight, {2 * kVisInter, kVisDim});
    check_bf16(p + "mlp.w2.weight", b.mlp_w2_weight, {kVisDim, kVisInter});
  }
  check_norm("vision.norm.weight", w.final_norm_weight);
  check_bf16("aligner.w1.weight", w.aligner_w1_weight,
             {hidden, kVisDim * kVisRatio * kVisRatio});
  check_bf16("aligner.w1.bias", w.aligner_w1_bias, {hidden});
  check_bf16("aligner.w2.weight", w.aligner_w2_weight, {hidden, hidden});
  check_bf16("aligner.w2.bias", w.aligner_w2_bias, {hidden});

  // The four learned sentinels, WIDENED to f32 — the dtype the merge in
  // `deepseek_v4_mm.cpp` reads them at, and the dtype the mmproj vehicle stores
  // them at, so both arms hand the merge the same thing.
  const auto check_sentinel = [&](const std::string& name,
                                  const std::vector<float>& v) {
    CAPTURE(name);
    REQUIRE(v.size() == static_cast<size_t>(hidden));
    CHECK(v[0] == doctest::Approx(vt::BF16ToF32(VisionWord(name, 0))));
    CHECK(v[static_cast<size_t>(hidden - 1)] ==
          doctest::Approx(vt::BF16ToF32(VisionWord(name, hidden - 1))));
  };
  check_sentinel("image_start", tower.image_start);
  check_sentinel("image_end", tower.image_end);
  check_sentinel("image_newline", tower.image_newline);
  check_sentinel("image_pad", tower.image_pad);

  // The four are DISTINCT. A reader that filled all of them from one name would
  // satisfy every shape and dtype check above.
  CHECK(tower.image_start[0] != doctest::Approx(tower.image_end[0]));
  CHECK(tower.image_newline[0] != doctest::Approx(tower.image_pad[0]));
}

TEST_CASE("official vision safetensors are accounted for, and a text checkpoint is not") {
  const FixtureOptions opt = OfficialVisionOptions();
  auto text = BuildStFixture(opt, /*vision=*/true);
  const vllm::DeepseekV4Weights without =
      vllm::LoadDeepseekV4ForCausalLMWeights(text->shards, text->config);

  auto vision = BuildVisionFixture(opt);
  vllm::DeepseekV4Weights with;
  const std::string msg = ThrowMessage([&] {
    with = vllm::LoadDeepseekV4ForCausalLMWeights(vision->shards, vision->config);
  });
  CAPTURE(msg);
  REQUIRE(msg.empty());
  // EXACTLY the 27 names of this fixture's group — no more, so a name counted
  // twice is a number rather than a slot that happens to be filled, and no
  // fewer, so the accounting cannot quietly skip one.
  CHECK(with.accounted_tensors ==
        without.accounted_tensors +
            static_cast<int64_t>(OfficialVisionNames().size()));
}

TEST_CASE("official vision safetensors refuse a missing, mistyped, misshaped or duplicated tensor") {
  const FixtureOptions opt = OfficialVisionOptions();
  const auto rejects = [&](const VisionFixtureEdit& edit,
                           const std::string& needle) {
    CAPTURE(needle);
    auto f = BuildVisionFixture(opt, edit);
    const vllm::multimodal::DeepSeekV4VisionConfig cfg =
        vllm::DeepSeekV4OfficialVisionConfig(f->config);
    const std::string msg = ThrowMessage(
        [&] { (void)vllm::LoadDeepSeekV4VisionFromSafetensors(f->shards, cfg); });
    CAPTURE(msg);
    CHECK(msg.find(needle) != std::string::npos);
  };
  // Each names the TENSOR, so an operator is told which one rather than being
  // handed a bare shape mismatch.
  VisionFixtureEdit edit;
  edit = {};
  edit.drop = "vision.blocks.1.attn.wo.bias";
  rejects(edit, "vision.blocks.1.attn.wo.bias");
  edit = {};
  edit.retype = "vision.patch_embed.proj.weight";
  rejects(edit, "vision.patch_embed.proj.weight");
  edit = {};
  edit.reshape = "vision.blocks.0.mlp.w1.weight";
  rejects(edit, "vision.blocks.0.mlp.w1.weight");
  edit = {};
  edit.duplicate = "image_pad";
  rejects(edit, "duplicate");
}

TEST_CASE("the official vision geometry refuses an absent or absurd config value by key") {
  const FixtureOptions opt = OfficialVisionOptions();
  struct Failure {
    const char* name;
    const char* key;
    int64_t value;
    const char* needle;
  };
  const std::vector<Failure> failures = {
      {"zero depth", "vision_n_layers", 0, "vision_n_layers"},
      {"zero dimension", "vision_dim", 0, "vision_dim"},
      {"zero heads", "vision_n_heads", 0, "vision_n_heads"},
      {"non-dividing heads", "vision_n_heads", 3, "must divide"},
      {"head dimension not divisible by four", "vision_n_heads", 4,
       "divisible by four"},
      {"zero intermediate width", "vision_inter_dim", 0, "vision_inter_dim"},
      {"zero patch", "vision_patch_size", 0, "vision_patch_size"},
      {"zero downsample ratio", "vision_downsample_ratio", 0,
       "vision_downsample_ratio"},
  };
  for (const Failure& failure : failures) {
    CAPTURE(failure.name);
    vllm::HfConfig cfg = VisionFixtureConfig(opt);
    cfg.raw[failure.key] = failure.value;
    const std::string msg =
        ThrowMessage([&] { (void)vllm::DeepSeekV4OfficialVisionConfig(cfg); });
    CAPTURE(msg);
    CHECK(msg.find(failure.needle) != std::string::npos);
  }
}

// ─── REACHABILITY. The production call site, not the class. ─────────────────
//
// `ModelRegistry::Load` is what `model_loader.cpp` calls. Deleting the
// `LoadDeepseekV4VisionRuntime` call in `deepseek_v4_registry.cpp`'s
// safetensors branch — the production wiring this wave added — must make this
// case red; every other case in this file stays green without it, because they
// enter through the reader directly.
TEST_CASE("official vision safetensors reach ModelRegistry::Load, and a text checkpoint stays tower-free") {
  const FixtureOptions opt = OfficialVisionOptions();

  auto vision = BuildVisionFixture(opt);
  const vllm::ModelSource vision_source =
      vllm::ModelSource::FromSafetensors(vision->shards);
  std::unique_ptr<vllm::LoadedModel> vision_model;
  const std::string vision_msg = ThrowMessage([&] {
    vision_model = vllm::ModelRegistry::Load(vision->config, vision_source);
  });
  CAPTURE(vision_msg);
  REQUIRE(vision_msg.empty());
  REQUIRE(vision_model != nullptr);
  const auto& loaded = vllm::ModelAs<vllm::DeepseekV4LoadedModel>(
      *vision_model, "DeepseekV4ForCausalLM");
  REQUIRE(loaded.has_vision());
  CHECK(loaded.vision().config.depth == kVisDepth);
  CHECK(loaded.vision().config.output_size == vision->config.hidden_size);
  // The tower is built on FIRST USE, so the load itself leaves it null while
  // the weights it will be built from are already resident.
  CHECK(loaded.vision().tower == nullptr);
  CHECK(loaded.vision().projector.image_start.size() ==
        static_cast<size_t>(vision->config.hidden_size));

  // INERTNESS. A DeepSeek-V4 TEXT checkpoint carries none of the group and must
  // still load, tower-free. Making the vision names REQUIRED makes this red.
  auto text = BuildStFixture(opt, /*vision=*/false);
  const vllm::ModelSource text_source =
      vllm::ModelSource::FromSafetensors(text->shards);
  std::unique_ptr<vllm::LoadedModel> text_model;
  const std::string text_msg = ThrowMessage(
      [&] { text_model = vllm::ModelRegistry::Load(text->config, text_source); });
  CAPTURE(text_msg);
  REQUIRE(text_msg.empty());
  REQUIRE(text_model != nullptr);
  CHECK_FALSE(vllm::ModelAs<vllm::DeepseekV4LoadedModel>(
                  *text_model, "DeepseekV4ForCausalLM")
                  .has_vision());
}

// ─── The pinned manifests: what ties all of the above to a 156 GiB artifact ──
//
// These two cases are ported from the same parallel line as the loader
// (`3f3860851`). They read the committed manifests that
// `scripts/check-deepseek-v4-vision-manifests.py` builds from the pinned
// revision, and hold this tree's derived name map and shape rules to them. The
// checker recomputes the same quantities in Python; these recompute them in
// C++. Neither reads a weight byte.
TEST_CASE("the pinned index manifest classifies the released tensor map exactly") {
  const nlohmann::json manifest = ReadJsonFixture(DEEPSEEK_V4_VISION_INDEX_MANIFEST);
  CHECK(manifest.at("repo") == "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp");
  CHECK(manifest.at("revision") == "86f746b36186f0e567729a5c06a8c918caba82a9");
  CHECK(manifest.at("shard_count") == 48);
  CHECK(manifest.at("total_size") == 167811372792ull);
  CHECK(manifest.at("tensor_count") == 72633);
  // 267 = 2 + 8 * 32 + 1 + 4 + 4, the same rule this fixture's 27 follows.
  CHECK(manifest.at("classifications").at("vision").at("count") == 267);
  const nlohmann::json config = ReadJsonFixture(DEEPSEEK_V4_VISION_CONFIG);
  CHECK(config.at("vision_n_layers") == 32);
  CHECK(config.at("vision_dim") == 1024);
  CHECK(config.at("vision_n_heads") == 16);
  CHECK(config.at("vision_inter_dim") == 2816);
  CHECK(config.at("vision_patch_size") == 14);
  CHECK(config.at("vision_downsample_ratio") == 3);
  CHECK(2 + 8 * config.at("vision_n_layers").get<int64_t>() + 1 + 4 + 4 ==
        manifest.at("classifications").at("vision").at("count").get<int64_t>());
}

TEST_CASE("the pinned shard-1 header gives every official vision tensor a BF16 shape this loader accepts") {
  const nlohmann::json config = ReadJsonFixture(DEEPSEEK_V4_VISION_CONFIG);
  const nlohmann::json manifest = ReadJsonFixture(DEEPSEEK_V4_VISION_HEADER_MANIFEST);
  const nlohmann::json& tensors = manifest.at("tensors");
  CHECK(manifest.at("shard") == "model-00001-of-00048.safetensors");
  CHECK(manifest.at("vision_tensor_count") == 267);
  CHECK(manifest.at("vision_payload_bytes") == 932786176ull);
  REQUIRE(manifest.at("header_tensor_count").get<size_t>() == tensors.size());

  const int64_t hidden = config.at("hidden_size").get<int64_t>();
  const int64_t dim = config.at("vision_dim").get<int64_t>();
  const int64_t inter = config.at("vision_inter_dim").get<int64_t>();
  const int64_t patch = config.at("vision_patch_size").get<int64_t>();
  const int64_t ratio = config.at("vision_downsample_ratio").get<int64_t>();
  // The RELEASED shapes, derived here from the released config by the same
  // rules the loader applies at this fixture's reduced geometry.
  const auto released_shape =
      [&](const std::string& name) -> std::vector<int64_t> {
    if (name == "vision.patch_embed.proj.weight") return {dim, 3 * patch * patch};
    if (name == "vision.patch_embed.proj.bias" || name == "vision.norm.weight" ||
        name.ends_with("norm1.weight") || name.ends_with("norm2.weight") ||
        name.ends_with("attn.wo.bias"))
      return {dim};
    if (name.ends_with("attn.wqkv.weight")) return {3 * dim, dim};
    if (name.ends_with("attn.wqkv.bias")) return {3 * dim};
    if (name.ends_with("attn.wo.weight")) return {dim, dim};
    if (name.ends_with("mlp.w1.weight")) return {2 * inter, dim};
    if (name.ends_with("mlp.w2.weight")) return {dim, inter};
    if (name == "aligner.w1.weight") return {hidden, dim * ratio * ratio};
    if (name == "aligner.w2.weight") return {hidden, hidden};
    if (name.starts_with("aligner.") || name.starts_with("image_")) return {hidden};
    return {};
  };

  size_t vision_names = 0;
  size_t language_names = 0;
  int64_t payload = 0;
  std::vector<std::string> records;
  for (const auto& [name, tensor] : tensors.items()) {
    CAPTURE(name);
    // EVERY tensor of the released group is BF16 on disk, which is what the
    // loader refuses anything else against.
    CHECK(tensor.at("dtype") == "BF16");
    const std::vector<int64_t> shape =
        tensor.at("shape").get<std::vector<int64_t>>();
    if (IndexClass(name) != "vision") {
      // Shard 1 carries exactly one language tensor beside the group.
      ++language_names;
      CHECK(name == "embed.weight");
      CHECK(shape == std::vector<int64_t>{config.at("vocab_size").get<int64_t>(),
                                          hidden});
      continue;
    }
    ++vision_names;
    const std::vector<int64_t> wanted = released_shape(name);
    REQUIRE_MESSAGE(!wanted.empty(), name);
    CHECK(shape == wanted);
    int64_t bytes = 2;
    for (int64_t d : shape) bytes *= d;
    payload += bytes;
    std::string record = name + "\tBF16\t";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != 0) record += ",";
      record += std::to_string(shape[i]);
    }
    records.push_back(record);
  }
  CHECK(language_names == 1);
  CHECK(vision_names == 267);
  CHECK(payload == manifest.at("vision_payload_bytes").get<int64_t>());
  // The records are sorted by name, which is the order the checker hashed them
  // in; this is the one assertion that would catch a shape changing while every
  // count above stayed the same.
  std::sort(records.begin(), records.end());
  CHECK(std::to_string(Fnv1aLines(records)) ==
        manifest.at("vision_records_fnv1a64").get<std::string>());
}
