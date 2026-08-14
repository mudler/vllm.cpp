// w2v-bert-2.0 Conformer parity gate — IndexTTS-2.5's semantic front end (#634).
//
// Gated against HuggingFace `transformers` executed DIRECTLY: the class
// `infer_v2_5.py:174` instantiates, so there is no restatement. Weights are
// rebuilt from the same FNV-1a -> splitmix64 stream the generator used, keyed by
// the upstream parameter NAME.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/w2vbert.h"
#include "w2vbert_goldens.inc"

namespace {
using namespace w2vbert_goldens;

std::vector<float> Rand(const std::string& name, int64_t n, double scale) {
  uint64_t seed = 0xCBF29CE484222325ULL;
  for (char c : name) { seed ^= static_cast<unsigned char>(c); seed *= 0x100000001B3ULL; }
  std::vector<float> o(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    uint64_t x = seed + static_cast<uint64_t>(i);
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    o[static_cast<size_t>(i)] =
        static_cast<float>(((static_cast<double>(z >> 11) * 0x1.0p-53) * 2.0 - 1.0) * scale);
  }
  return o;
}

std::map<std::string, std::vector<float>> Weights() {
  std::map<std::string, std::vector<float>> w;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    w[e.name] = Rand(e.name, n, 0.3);
  }
  return w;
}

double Worst(const std::vector<float>& got, const float* want, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(got[i]) - want[i]));
  return m;
}
}  // namespace

TEST_CASE("w2vbert FeedForward matches upstream (swish, not gelu)") {
  const auto w = Weights();
  const std::vector<float> x = Rand("hidden", kFrames * kHidden, 1.0);
  // ffn1 runs on the LAYER-NORMED input, mirroring the encoder layer.
  const std::vector<float> normed = vllm::models::w2vbert::LayerNorm(
      x, kFrames, kHidden, w.at("ffn1_layer_norm.weight"), w.at("ffn1_layer_norm.bias"),
      kLayerNormEps);
  const std::vector<float> got = vllm::models::w2vbert::FeedForward(
      normed, kFrames, kHidden, kIntermediate, w.at("ffn1.intermediate_dense.weight"),
      w.at("ffn1.intermediate_dense.bias"), w.at("ffn1.output_dense.weight"),
      w.at("ffn1.output_dense.bias"));
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  CHECK(Worst(got, kFfn1Out, got.size()) < 1e-5);
}

TEST_CASE("w2vbert ConvModule matches upstream, including the CAUSAL left pad") {
  // The pad is (kernel-1, 0). A symmetric pad yields the same shape while
  // letting every frame see the future -- no length or shape check can see it,
  // which is why this is gated on values against the real module.
  const auto w = Weights();
  vllm::models::w2vbert::ConvModuleWeights c;
  c.ln_gamma = w.at("conv_module.layer_norm.weight");
  c.ln_beta = w.at("conv_module.layer_norm.bias");
  c.pointwise1 = w.at("conv_module.pointwise_conv1.weight");
  c.depthwise = w.at("conv_module.depthwise_conv.weight");
  c.dw_ln_gamma = w.at("conv_module.depthwise_layer_norm.weight");
  c.dw_ln_beta = w.at("conv_module.depthwise_layer_norm.bias");
  c.pointwise2 = w.at("conv_module.pointwise_conv2.weight");

  const std::vector<float> x(std::begin(kConvIn), std::end(kConvIn));
  const std::vector<float> got =
      vllm::models::w2vbert::ConvModule(x, kFrames, kHidden, kConvKernel, c, kLayerNormEps);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  CHECK(Worst(got, kConvOut, got.size()) < 1e-5);
}

TEST_CASE("w2vbert conv module cannot see the future") {
  // Direct causality probe, independent of the goldens: perturbing the LAST
  // frame must leave every earlier output bit-identical.
  const auto w = Weights();
  vllm::models::w2vbert::ConvModuleWeights c;
  c.ln_gamma = w.at("conv_module.layer_norm.weight");
  c.ln_beta = w.at("conv_module.layer_norm.bias");
  c.pointwise1 = w.at("conv_module.pointwise_conv1.weight");
  c.depthwise = w.at("conv_module.depthwise_conv.weight");
  c.dw_ln_gamma = w.at("conv_module.depthwise_layer_norm.weight");
  c.dw_ln_beta = w.at("conv_module.depthwise_layer_norm.bias");
  c.pointwise2 = w.at("conv_module.pointwise_conv2.weight");

  std::vector<float> x(std::begin(kConvIn), std::end(kConvIn));
  const std::vector<float> base =
      vllm::models::w2vbert::ConvModule(x, kFrames, kHidden, kConvKernel, c, kLayerNormEps);
  x[static_cast<size_t>((kFrames - 1) * kHidden)] += 3.0F;
  const std::vector<float> moved =
      vllm::models::w2vbert::ConvModule(x, kFrames, kHidden, kConvKernel, c, kLayerNormEps);

  // NOTE: the module's leading layer_norm is per-frame, so a change to the last
  // frame cannot leak backwards through it either.
  for (size_t i = 0; i < static_cast<size_t>((kFrames - 1) * kHidden); ++i) {
    CHECK(base[i] == moved[i]);
  }
  double delta = 0.0;
  for (size_t i = static_cast<size_t>((kFrames - 1) * kHidden); i < base.size(); ++i) {
    delta = std::max(delta, std::fabs(static_cast<double>(base[i] - moved[i])));
  }
  CHECK(delta > 1e-6);  // the last frame MUST move, or the probe proves nothing
}

TEST_CASE("w2vbert relative-key self-attention matches upstream") {
  const auto w = Weights();
  vllm::models::w2vbert::SelfAttentionWeights a;
  a.q_w = w.at("self_attn.linear_q.weight"); a.q_b = w.at("self_attn.linear_q.bias");
  a.k_w = w.at("self_attn.linear_k.weight"); a.k_b = w.at("self_attn.linear_k.bias");
  a.v_w = w.at("self_attn.linear_v.weight"); a.v_b = w.at("self_attn.linear_v.bias");
  a.out_w = w.at("self_attn.linear_out.weight"); a.out_b = w.at("self_attn.linear_out.bias");
  a.distance_embedding = w.at("self_attn.distance_embedding.weight");

  const std::vector<float> x(std::begin(kAttnIn), std::end(kAttnIn));
  const std::vector<float> got = vllm::models::w2vbert::SelfAttentionRelativeKey(
      x, kFrames, kHidden, kHeads, kLeftMax, kRightMax, a);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  const double worst = Worst(got, kAttnOut, got.size());
  INFO("max abs diff vs upstream self-attention: ", worst);
  CHECK(worst < 1e-5);
}

TEST_CASE("w2vbert attention is NOT causal: every frame sees the whole sequence") {
  // The Conformer encoder is bidirectional. Perturbing the LAST frame must move
  // the FIRST output -- the opposite of the conv module's guarantee, and worth
  // pinning so a stray causal mask cannot be introduced unnoticed.
  const auto w = Weights();
  vllm::models::w2vbert::SelfAttentionWeights a;
  a.q_w = w.at("self_attn.linear_q.weight"); a.q_b = w.at("self_attn.linear_q.bias");
  a.k_w = w.at("self_attn.linear_k.weight"); a.k_b = w.at("self_attn.linear_k.bias");
  a.v_w = w.at("self_attn.linear_v.weight"); a.v_b = w.at("self_attn.linear_v.bias");
  a.out_w = w.at("self_attn.linear_out.weight"); a.out_b = w.at("self_attn.linear_out.bias");
  a.distance_embedding = w.at("self_attn.distance_embedding.weight");

  std::vector<float> x(std::begin(kAttnIn), std::end(kAttnIn));
  const std::vector<float> base = vllm::models::w2vbert::SelfAttentionRelativeKey(
      x, kFrames, kHidden, kHeads, kLeftMax, kRightMax, a);
  x[static_cast<size_t>((kFrames - 1) * kHidden)] += 2.0F;
  const std::vector<float> moved = vllm::models::w2vbert::SelfAttentionRelativeKey(
      x, kFrames, kHidden, kHeads, kLeftMax, kRightMax, a);
  double first = 0.0;
  for (int64_t d = 0; d < kHidden; ++d) {
    first = std::max(first, std::fabs(static_cast<double>(base[static_cast<size_t>(d)] -
                                                          moved[static_cast<size_t>(d)])));
  }
  CHECK(first > 1e-6);
}

TEST_CASE("w2vbert EncoderLayer reproduces the whole upstream Conformer block") {
  // The macaron 0.5 factors live HERE, not in FeedForward, so only a whole-layer
  // comparison can see them: every piecewise case above passes with or without.
  const auto m = Weights();
  vllm::models::w2vbert::EncoderLayerWeights w;
  w.ffn1_ln_gamma = m.at("ffn1_layer_norm.weight");
  w.ffn1_ln_beta = m.at("ffn1_layer_norm.bias");
  w.ffn1_in_w = m.at("ffn1.intermediate_dense.weight");
  w.ffn1_in_b = m.at("ffn1.intermediate_dense.bias");
  w.ffn1_out_w = m.at("ffn1.output_dense.weight");
  w.ffn1_out_b = m.at("ffn1.output_dense.bias");
  w.attn_ln_gamma = m.at("self_attn_layer_norm.weight");
  w.attn_ln_beta = m.at("self_attn_layer_norm.bias");
  w.attn.q_w = m.at("self_attn.linear_q.weight"); w.attn.q_b = m.at("self_attn.linear_q.bias");
  w.attn.k_w = m.at("self_attn.linear_k.weight"); w.attn.k_b = m.at("self_attn.linear_k.bias");
  w.attn.v_w = m.at("self_attn.linear_v.weight"); w.attn.v_b = m.at("self_attn.linear_v.bias");
  w.attn.out_w = m.at("self_attn.linear_out.weight");
  w.attn.out_b = m.at("self_attn.linear_out.bias");
  w.attn.distance_embedding = m.at("self_attn.distance_embedding.weight");
  w.conv.ln_gamma = m.at("conv_module.layer_norm.weight");
  w.conv.ln_beta = m.at("conv_module.layer_norm.bias");
  w.conv.pointwise1 = m.at("conv_module.pointwise_conv1.weight");
  w.conv.depthwise = m.at("conv_module.depthwise_conv.weight");
  w.conv.dw_ln_gamma = m.at("conv_module.depthwise_layer_norm.weight");
  w.conv.dw_ln_beta = m.at("conv_module.depthwise_layer_norm.bias");
  w.conv.pointwise2 = m.at("conv_module.pointwise_conv2.weight");
  w.ffn2_ln_gamma = m.at("ffn2_layer_norm.weight");
  w.ffn2_ln_beta = m.at("ffn2_layer_norm.bias");
  w.ffn2_in_w = m.at("ffn2.intermediate_dense.weight");
  w.ffn2_in_b = m.at("ffn2.intermediate_dense.bias");
  w.ffn2_out_w = m.at("ffn2.output_dense.weight");
  w.ffn2_out_b = m.at("ffn2.output_dense.bias");
  w.final_ln_gamma = m.at("final_layer_norm.weight");
  w.final_ln_beta = m.at("final_layer_norm.bias");

  const std::vector<float> x = Rand("hidden", kFrames * kHidden, 1.0);
  const std::vector<float> got = vllm::models::w2vbert::EncoderLayer(
      x, kFrames, kHidden, kHeads, kIntermediate, kConvKernel, kLeftMax, kRightMax, w,
      kLayerNormEps);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  const double worst = Worst(got, kLayerOut, got.size());
  INFO("max abs diff vs upstream Wav2Vec2BertEncoderLayer: ", worst);
  CHECK(worst < 2e-5);
}

namespace {
std::map<std::string, std::vector<float>> ModelWeights() {
  std::map<std::string, std::vector<float>> w;
  for (int64_t i = 0; i < kModelManifestSize; ++i) {
    const auto& e = kModelManifest[i];
    const int64_t d[3] = {e.d0, e.d1, e.d2};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    w[e.name] = Rand(e.name, n, 0.3);
  }
  return w;
}

vllm::models::w2vbert::EncoderLayerWeights LayerFrom(
    const std::map<std::string, std::vector<float>>& m, const std::string& p) {
  vllm::models::w2vbert::EncoderLayerWeights w;
  w.ffn1_ln_gamma = m.at(p + "ffn1_layer_norm.weight");
  w.ffn1_ln_beta = m.at(p + "ffn1_layer_norm.bias");
  w.ffn1_in_w = m.at(p + "ffn1.intermediate_dense.weight");
  w.ffn1_in_b = m.at(p + "ffn1.intermediate_dense.bias");
  w.ffn1_out_w = m.at(p + "ffn1.output_dense.weight");
  w.ffn1_out_b = m.at(p + "ffn1.output_dense.bias");
  w.attn_ln_gamma = m.at(p + "self_attn_layer_norm.weight");
  w.attn_ln_beta = m.at(p + "self_attn_layer_norm.bias");
  w.attn.q_w = m.at(p + "self_attn.linear_q.weight"); w.attn.q_b = m.at(p + "self_attn.linear_q.bias");
  w.attn.k_w = m.at(p + "self_attn.linear_k.weight"); w.attn.k_b = m.at(p + "self_attn.linear_k.bias");
  w.attn.v_w = m.at(p + "self_attn.linear_v.weight"); w.attn.v_b = m.at(p + "self_attn.linear_v.bias");
  w.attn.out_w = m.at(p + "self_attn.linear_out.weight");
  w.attn.out_b = m.at(p + "self_attn.linear_out.bias");
  w.attn.distance_embedding = m.at(p + "self_attn.distance_embedding.weight");
  w.conv.ln_gamma = m.at(p + "conv_module.layer_norm.weight");
  w.conv.ln_beta = m.at(p + "conv_module.layer_norm.bias");
  w.conv.pointwise1 = m.at(p + "conv_module.pointwise_conv1.weight");
  w.conv.depthwise = m.at(p + "conv_module.depthwise_conv.weight");
  w.conv.dw_ln_gamma = m.at(p + "conv_module.depthwise_layer_norm.weight");
  w.conv.dw_ln_beta = m.at(p + "conv_module.depthwise_layer_norm.bias");
  w.conv.pointwise2 = m.at(p + "conv_module.pointwise_conv2.weight");
  w.ffn2_ln_gamma = m.at(p + "ffn2_layer_norm.weight");
  w.ffn2_ln_beta = m.at(p + "ffn2_layer_norm.bias");
  w.ffn2_in_w = m.at(p + "ffn2.intermediate_dense.weight");
  w.ffn2_in_b = m.at(p + "ffn2.intermediate_dense.bias");
  w.ffn2_out_w = m.at(p + "ffn2.output_dense.weight");
  w.ffn2_out_b = m.at(p + "ffn2.output_dense.bias");
  w.final_ln_gamma = m.at(p + "final_layer_norm.weight");
  w.final_ln_beta = m.at(p + "final_layer_norm.bias");
  return w;
}
}  // namespace

TEST_CASE("w2vbert FeatureProjection normalizes BEFORE projecting") {
  const auto m = ModelWeights();
  const std::vector<float> x(std::begin(kFeatsIn), std::end(kFeatsIn));
  const std::vector<float> got = vllm::models::w2vbert::FeatureProjection(
      x, kFrames, kInDim, kHidden, m.at("fp.layer_norm.weight"), m.at("fp.layer_norm.bias"),
      m.at("fp.projection.weight"), m.at("fp.projection.bias"), kLayerNormEps);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  CHECK(Worst(got, kProjected, got.size()) < 1e-5);
}

TEST_CASE("w2vbert EncoderStack applies NO final norm after the layers") {
  // Most encoders end with a layer norm; this one does not. Adding one still
  // produces a well-formed tensor of the right shape, so only a value
  // comparison against the real Wav2Vec2BertEncoder can tell.
  const auto m = ModelWeights();
  std::vector<vllm::models::w2vbert::EncoderLayerWeights> layers;
  for (int64_t i = 0; i < kNumLayers; ++i) {
    layers.push_back(LayerFrom(m, "enc.layers." + std::to_string(i) + "."));
  }
  const std::vector<float> x(std::begin(kProjected), std::end(kProjected));
  const std::vector<float> got = vllm::models::w2vbert::EncoderStack(
      x, kFrames, kHidden, kHeads, kIntermediate, kConvKernel, kLeftMax, kRightMax, layers,
      kLayerNormEps);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  const double worst = Worst(got, kStacked, got.size());
  INFO("max abs diff vs upstream Wav2Vec2BertEncoder: ", worst);
  CHECK(worst < 5e-5);
}
