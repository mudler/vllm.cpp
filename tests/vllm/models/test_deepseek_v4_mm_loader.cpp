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

// ─── the GGUF arm's own tiny `deepseek4` file ───────────────────────────────
// Deliberately SMALLER than the W2b suite's fixture: no compressor and no
// indexer on any layer (`compress_ratios` all zero), because the DSA families
// have nothing to do with the router bias and every tensor they add is a tensor
// this suite would have to account for again. One hash layer and two gated
// layers is the whole topology the vision bias interacts with.
constexpr int64_t kH = 32, kVocab = 16;
constexpr int64_t kHeads = 2, kHeadDim = 32, kRope = 8;
constexpr int64_t kQLora = 32, kOLora = 32, kOGroups = 2;
constexpr int64_t kExperts = 4, kUsed = 2, kInter = 32;
constexpr int64_t kHc = 2, kSinkhorn = 3;
constexpr int64_t kLayers = 3, kHashLayers = 1;

int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::vector<uint64_t> GgmlDims(const std::vector<int64_t>& torch_shape) {
  std::vector<uint64_t> d;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    d.push_back(static_cast<uint64_t>(*it));
  return d;
}

template <typename F>
std::string F32Data(int64_t n, F fill) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = fill(i);
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
  }
  return s;
}

// torch [out,in] (in % 32 == 0) -> Q8_0 blocks (`{ f16 d; int8 qs[32] }`).
template <typename F>
std::string Q8Data(int64_t out, int64_t in, F fill) {
  std::string s;
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t b = 0; b < in / 32; ++b) {
      float amax = 0.0f;
      float x[32];
      for (int j = 0; j < 32; ++j) {
        x[j] = fill(o * in + b * 32 + j);
        amax = std::max(amax, std::fabs(x[j]));
      }
      const float d = amax / 127.0f;
      const uint16_t dh = vt::F32ToF16(d);
      s.push_back(static_cast<char>(dh & 0xff));
      s.push_back(static_cast<char>((dh >> 8) & 0xff));
      for (int j = 0; j < 32; ++j) {
        int q = d > 0.0f ? static_cast<int>(std::lround(x[j] / d)) : 0;
        q = std::max(-127, std::min(127, q));
        s.push_back(static_cast<char>(static_cast<int8_t>(q)));
      }
    }
  }
  return s;
}

float WFill(int64_t i) { return 0.05f * static_cast<float>((i % 13) - 6); }

// The TWO biases are filled from DIFFERENT functions on purpose. A loader that
// routed `exp_probs_b_vl` into the text slot (or the reverse) would still put a
// plausible `[E]` vector in every slot, so only distinguishable CONTENTS can
// tell the two apart.
float TextBiasFill(int64_t l, int64_t i) {
  return 0.25f + 0.5f * static_cast<float>(l) + 0.125f * static_cast<float>(i);
}
float VisionBiasFill(int64_t l, int64_t i) {
  return -0.75f - 0.5f * static_cast<float>(l) - 0.0625f * static_cast<float>(i);
}

std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// `vision` writes `blk.N.exp_probs_b_vl.bias` on EVERY layer, which is what the
// pinned vision artifact carries; false is the text checkpoint.
std::string BuildDeepseek4Gguf(bool vision) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "deepseek4"));
  const std::string p = "deepseek4.";
  b.AddKv(U32Kv(p + "embedding_length", kH));
  b.AddKv(U32Kv(p + "block_count", kLayers));
  b.AddKv(U32Kv(p + "attention.head_count", kHeads));
  b.AddKv(U32Kv(p + "attention.head_count_kv", 1));
  b.AddKv(U32Kv(p + "attention.key_length", kHeadDim));
  b.AddKv(U32Kv(p + "rope.dimension_count", kRope));
  b.AddKv(U32Kv(p + "attention.q_lora_rank", kQLora));
  b.AddKv(U32Kv(p + "attention.output_lora_rank", kOLora));
  b.AddKv(U32Kv(p + "attention.output_group_count", kOGroups));
  b.AddKv(F32Kv(p + "rope.freq_base", 10000.0f));
  b.AddKv(F32Kv(p + "attention.compress_rope_freq_base", 160000.0f));
  b.AddKv(F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6f));
  b.AddKv(U32Kv(p + "expert_count", kExperts));
  b.AddKv(U32Kv(p + "expert_used_count", kUsed));
  b.AddKv(U32Kv(p + "expert_shared_count", 1));
  b.AddKv(U32Kv(p + "expert_feed_forward_length", kInter));
  b.AddKv(U32Kv(p + "hash_layer_count", kHashLayers));
  b.AddKv(F32Kv(p + "swiglu_clamp", 10.0f));
  b.AddKv(U32Kv(p + "hyper_connection.count", kHc));
  b.AddKv(U32Kv(p + "hyper_connection.sinkhorn_iterations", kSinkhorn));
  b.AddKv(F32Kv(p + "hyper_connection.epsilon", 1e-6f));
  b.AddKv(I32ArrayKv(p + "attention.compress_ratios",
                     std::vector<int32_t>(static_cast<size_t>(kLayers), 0)));

  const auto f32 = [&](const std::string& name, const std::vector<int64_t>& shape) {
    b.AddTensor(name, GgmlDims(shape), /*F32=*/0, F32Data(Prod(shape), WFill));
  };
  const auto q8 = [&](const std::string& name, const std::vector<int64_t>& shape) {
    const int64_t out =
        shape.size() == 3 ? shape[0] * shape[1] : shape[0];
    b.AddTensor(name, GgmlDims(shape), /*Q8_0=*/8, Q8Data(out, shape.back(), WFill));
  };

  const int64_t hcf = (2 + kHc) * kHc;
  f32("token_embd.weight", {kVocab, kH});
  q8("output.weight", {kVocab, kH});
  f32("output_norm.weight", {kH});
  f32("output_hc_base.weight", {kHc});
  f32("output_hc_fn.weight", {kHc, kHc * kH});
  f32("output_hc_scale.weight", {1});

  for (int64_t l = 0; l < kLayers; ++l) {
    q8(Blk(l, "attn_q_a.weight"), {kQLora, kH});
    q8(Blk(l, "attn_q_b.weight"), {kHeads * kHeadDim, kQLora});
    q8(Blk(l, "attn_kv.weight"), {kHeadDim, kH});
    q8(Blk(l, "attn_output_a.weight"),
       {kOGroups * kOLora, kHeads * kHeadDim / kOGroups});
    q8(Blk(l, "attn_output_b.weight"), {kH, kOGroups * kOLora});
    f32(Blk(l, "attn_norm.weight"), {kH});
    f32(Blk(l, "attn_q_a_norm.weight"), {kQLora});
    f32(Blk(l, "attn_kv_a_norm.weight"), {kHeadDim});
    f32(Blk(l, "attn_sinks.weight"), {kHeads});
    f32(Blk(l, "ffn_norm.weight"), {kH});
    f32(Blk(l, "hc_attn_base.weight"), {hcf});
    f32(Blk(l, "hc_attn_fn.weight"), {hcf, kHc * kH});
    f32(Blk(l, "hc_attn_scale.weight"), {3});
    f32(Blk(l, "hc_ffn_base.weight"), {hcf});
    f32(Blk(l, "hc_ffn_fn.weight"), {hcf, kHc * kH});
    f32(Blk(l, "hc_ffn_scale.weight"), {3});
    q8(Blk(l, "ffn_gate_inp.weight"), {kExperts, kH});
    q8(Blk(l, "ffn_gate_exps.weight"), {kExperts, kInter, kH});
    q8(Blk(l, "ffn_up_exps.weight"), {kExperts, kInter, kH});
    q8(Blk(l, "ffn_down_exps.weight"), {kExperts, kH, kInter});
    q8(Blk(l, "ffn_gate_shexp.weight"), {kInter, kH});
    q8(Blk(l, "ffn_up_shexp.weight"), {kInter, kH});
    q8(Blk(l, "ffn_down_shexp.weight"), {kH, kInter});
    if (l < kHashLayers) {
      b.AddTensor(Blk(l, "ffn_gate_tid2eid.weight"), GgmlDims({kVocab, kUsed}),
                  /*F32=*/0, F32Data(kVocab * kUsed, [](int64_t i) {
                    return static_cast<float>(i % kExperts);
                  }));
    } else {
      b.AddTensor(Blk(l, "exp_probs_b.bias"), GgmlDims({kExperts}), /*F32=*/0,
                  F32Data(kExperts, [l](int64_t i) { return TextBiasFill(l, i); }));
    }
    if (vision) {
      b.AddTensor(Blk(l, "exp_probs_b_vl.bias"), GgmlDims({kExperts}), /*F32=*/0,
                  F32Data(kExperts, [l](int64_t i) { return VisionBiasFill(l, i); }));
    }
  }
  return b.Build();
}

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
  for (int l = 0; l < opt.layers; ++l) {
    CAPTURE(l);
    CHECK(w.host.layers[static_cast<size_t>(l)].gate_bias_vl.size() ==
          static_cast<size_t>(dsv4_exl3_fixture::kExperts));
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
