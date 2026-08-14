// EnhancedCodec.quantize against upstream goldens. See codec_encoder.h.
#include <cstdint>
#include <string>
#include <vector>

#include "codec_encoder_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/codec_encoder.h"

namespace {

std::vector<float> Rnd(const std::string& name, size_t n, double scale) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char ch : name) {
    h = (h ^ ch) * 0x100000001B3ULL;
  }
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    h += 0x9E3779B97F4A7C15ULL;
    uint64_t z = h;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    const double u = static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    out[i] = static_cast<float>((u * 2.0 - 1.0) * scale);
  }
  return out;
}

using namespace codec_encoder_goldens;

vllm::models::codec_encoder::Config Cfg() {
  vllm::models::codec_encoder::Config c;
  c.hidden = kHidden;
  c.vocos_dim = kVocosDim;
  c.vocos_intermediate = kVocosIntermediate;
  c.codebook_size = kCodebookSize;
  c.codebook_dim = kCodebookDim;
  c.downsample = true;
  c.eps = 1e-6;
  return c;
}

vllm::models::codec_encoder::Weights W() {
  vllm::models::codec_encoder::Weights w;
  const size_t H = static_cast<size_t>(kHidden);
  const size_t D = static_cast<size_t>(kVocosDim);
  const size_t I = static_cast<size_t>(kVocosIntermediate);

  // The generator ZEROES every bias on the down/encoder path -- see its header
  // for why -- so this side must too, or the two fixtures are different models.
  w.down_w = Rnd("codec.down.weight", H * H * 3, 0.4);
  w.down_b.assign(H, 0.0F);

  const std::string e = "codec.encoder.0.";
  w.backbone.embed_w = Rnd(e + "embed.weight", D * H * 7, 0.4);
  w.backbone.embed_b.assign(D, 0.0F);
  w.backbone.norm_gamma = Rnd(e + "norm.weight", D, 0.4);
  w.backbone.norm_beta.assign(D, 0.0F);
  for (int64_t i = 0; i < kVocosLayers; ++i) {
    const std::string p = e + "convnext." + std::to_string(i) + ".";
    vllm::models::vocos::BlockWeights b;
    b.dw_weight = Rnd(p + "dwconv.weight", D * 7, 0.4);
    b.dw_bias.assign(D, 0.0F);
    b.ln_gamma = Rnd(p + "norm.weight", D, 0.4);
    b.ln_beta.assign(D, 0.0F);
    b.pw1_w = Rnd(p + "pwconv1.weight", I * D, 0.4);
    b.pw1_b.assign(I, 0.0F);
    b.pw2_w = Rnd(p + "pwconv2.weight", D * I, 0.4);
    b.pw2_b.assign(D, 0.0F);
    b.gamma = Rnd(p + "gamma", D, 0.4);
    w.backbone.blocks.push_back(std::move(b));
  }
  w.backbone.final_gamma = Rnd(e + "final_layer_norm.weight", D, 0.4);
  w.backbone.final_beta.assign(D, 0.0F);

  w.proj_w = Rnd("codec.encoder.1.weight", H * D, 0.4);
  w.proj_b.assign(H, 0.0F);

  const std::string q = "codec.quantizer.quantizers.0.";
  w.quantizer.in_g = Rnd(q + "in_project.weight_g", static_cast<size_t>(kCodebookDim), 0.4);
  w.quantizer.in_v = Rnd(q + "in_project.weight_v",
                         static_cast<size_t>(kCodebookDim) * H, 0.4);
  w.quantizer.in_bias = Rnd(q + "in_project.bias", static_cast<size_t>(kCodebookDim), 0.4);
  w.quantizer.out_g = Rnd(q + "out_project.weight_g", H, 0.4);
  w.quantizer.out_v = Rnd(q + "out_project.weight_v",
                          H * static_cast<size_t>(kCodebookDim), 0.4);
  w.quantizer.out_bias = Rnd(q + "out_project.bias", H, 0.4);
  w.quantizer.codebook = Rnd(q + "codebook.weight",
                             static_cast<size_t>(kCodebookSize * kCodebookDim), 0.4);
  return w;
}

}  // namespace

TEST_CASE("the DOWNSAMPLE and GELU match upstream") {
  const std::vector<float> x = Rnd("codec.x", static_cast<size_t>(kFrames * kHidden), 2.5);
  const auto got = vllm::models::codec_encoder::Encode(Cfg(), W(), x, kFrames);
  REQUIRE(got.after_down.size() == static_cast<size_t>(kHidden * kOutFrames));
  for (size_t i = 0; i < got.after_down.size(); ++i) {
    CHECK(got.after_down[i] == doctest::Approx(kAfterDown[i]).epsilon(2e-5));
  }
}

TEST_CASE("the encoder LATENT matches upstream") {
  // This is the primary gate: down -> gelu -> backbone -> linear is what this
  // file computes. The quantizer downstream is `fvq`, gated by test_fvq.
  const std::vector<float> x = Rnd("codec.x", static_cast<size_t>(kFrames * kHidden), 2.5);
  const auto got = vllm::models::codec_encoder::Encode(Cfg(), W(), x, kFrames);
  REQUIRE(got.out_frames == kOutFrames);
  REQUIRE(got.latent.size() == static_cast<size_t>(kHidden * kOutFrames));
  for (size_t i = 0; i < got.latent.size(); ++i) {
    CHECK(got.latent[i] == doctest::Approx(kLatent[i]).epsilon(2e-5));
  }
}

TEST_CASE("every input frame reaches the LATENT") {
  // Stated on the latent rather than the codes deliberately: see below.
  const std::vector<float> base_x =
      Rnd("codec.x", static_cast<size_t>(kFrames * kHidden), 2.5);
  const auto base = vllm::models::codec_encoder::Encode(Cfg(), W(), base_x, kFrames);
  for (int64_t f = 0; f < kFrames; ++f) {
    std::vector<float> x = base_x;
    for (int64_t c = 0; c < kHidden; ++c) {
      x[static_cast<size_t>(f * kHidden + c)] += 3.0F;
    }
    const auto moved = vllm::models::codec_encoder::Encode(Cfg(), W(), x, kFrames);
    CHECK(moved.latent != base.latent);
  }
}

TEST_CASE("the codes agree with upstream, though this fixture cannot say much") {
  // HONEST LABEL: at these dimensions, with weights drawn from a stream rather
  // than trained, every frame's latent points the same way and the quantizer
  // returns ONE index repeated. Matching that is a consistency check, not
  // evidence the encoder is right -- which is why the latent case above exists
  // and why the sensitivity case is stated on the latent. Kept because a change
  // that moved the codes off this value would still be worth seeing.
  const std::vector<float> x = Rnd("codec.x", static_cast<size_t>(kFrames * kHidden), 2.5);
  const auto got = vllm::models::codec_encoder::Encode(Cfg(), W(), x, kFrames);
  REQUIRE(got.indices.size() == static_cast<size_t>(kOutFrames));
  for (size_t i = 0; i < got.indices.size(); ++i) {
    CHECK(got.indices[i] == kIndices[i]);
  }
}

TEST_CASE("the quantized embedding matches upstream") {
  // Upstream RETURNS it frame-major (quantize() transposes); ours is
  // channel-major like the rest of this port, so the comparison transposes
  // explicitly rather than either side silently reorienting.
  const std::vector<float> x = Rnd("codec.x", static_cast<size_t>(kFrames * kHidden), 2.5);
  const auto got = vllm::models::codec_encoder::Encode(Cfg(), W(), x, kFrames);
  REQUIRE(got.quantized.size() == static_cast<size_t>(kHidden * kOutFrames));
  for (int64_t c = 0; c < kHidden; ++c) {
    for (int64_t t = 0; t < kOutFrames; ++t) {
      const float ours = got.quantized[static_cast<size_t>(c * kOutFrames + t)];
      const float theirs = kQuantized[static_cast<size_t>(t * kHidden + c)];
      CHECK(ours == doctest::Approx(theirs).epsilon(2e-5));
    }
  }
}

TEST_CASE("the downsample HALVES the frame rate") {
  CHECK(kOutFrames == kFrames / 2);
  const std::vector<float> x = Rnd("codec.x", static_cast<size_t>(kFrames * kHidden), 2.5);
  auto cfg = Cfg();
  cfg.downsample = false;
  const auto without = vllm::models::codec_encoder::Encode(cfg, W(), x, kFrames);
  CHECK(without.out_frames == kFrames);
  CHECK(without.indices.size() == static_cast<size_t>(kFrames));
}
