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

#include "deepseek_v4_lang_gguf_fixture.h"
#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/deepseek_v4.h"
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
