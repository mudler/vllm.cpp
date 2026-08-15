// BigVGAN generator against upstream goldens. See bigvgan.h.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "bigvgan_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/bigvgan.h"

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

using namespace bigvgan_goldens;

// The dilations the generator used, matching resblock_dilation_sizes.
const std::vector<int64_t> kDil{1, 3};

vllm::models::bigvgan::Config Cfg() {
  vllm::models::bigvgan::Config c;
  c.mels = kMels;
  c.init_channels = kInitCh;
  c.up_rates.assign(kUpRates, kUpRates + kNumUpsamples);
  c.up_kernels.assign(kUpKernels, kUpKernels + kNumUpsamples);
  c.num_kernels = kNumKernels;
  c.snake_logscale = true;
  c.tanh_at_final = false;
  return c;
}

vllm::models::bigvgan::Weights W() {
  vllm::models::bigvgan::Weights w;
  w.conv_pre.weight = Rnd("bvg.conv_pre.weight",
                          static_cast<size_t>(kInitCh * kMels * 7), 0.3);
  w.conv_pre.bias = Rnd("bvg.conv_pre.bias", static_cast<size_t>(kInitCh), 0.3);

  int64_t ch = kInitCh;
  for (int64_t i = 0; i < kNumUpsamples; ++i) {
    const std::string p = "bvg.ups." + std::to_string(i) + ".0.";
    const int64_t out_ch = ch / 2;
    vllm::models::bigvgan::ConvSpec up;
    up.weight = Rnd(p + "weight", static_cast<size_t>(ch * out_ch * kUpKernels[i]), 0.3);
    up.bias = Rnd(p + "bias", static_cast<size_t>(out_ch), 0.3);
    w.ups.push_back(std::move(up));
    ch = out_ch;
  }

  ch = kInitCh;
  for (int64_t i = 0; i < kNumUpsamples; ++i) {
    ch /= 2;
    for (int64_t j = 0; j < kNumKernels; ++j) {
      const int64_t idx = i * kNumKernels + j;
      const std::string p = "bvg.resblocks." + std::to_string(idx) + ".";
      vllm::models::bigvgan::AmpBlock rb;
      rb.kernel = kRbKernels[j];
      rb.dilations = kDil;
      for (size_t d = 0; d < kDil.size(); ++d) {
        const std::string ds = std::to_string(d);
        vllm::models::bigvgan::ConvSpec c1, c2;
        c1.weight = Rnd(p + "convs1." + ds + ".weight",
                        static_cast<size_t>(ch * ch * rb.kernel), 0.3);
        c1.bias = Rnd(p + "convs1." + ds + ".bias", static_cast<size_t>(ch), 0.3);
        c2.weight = Rnd(p + "convs2." + ds + ".weight",
                        static_cast<size_t>(ch * ch * rb.kernel), 0.3);
        c2.bias = Rnd(p + "convs2." + ds + ".bias", static_cast<size_t>(ch), 0.3);
        rb.convs1.push_back(std::move(c1));
        rb.convs2.push_back(std::move(c2));
      }
      for (size_t a = 0; a < 2 * kDil.size(); ++a) {
        const std::string as = std::to_string(a);
        rb.alpha.push_back(Rnd(p + "activations." + as + ".act.alpha",
                               static_cast<size_t>(ch), 0.3));
        rb.beta.push_back(Rnd(p + "activations." + as + ".act.beta",
                              static_cast<size_t>(ch), 0.3));
      }
      w.resblocks.push_back(std::move(rb));
    }
  }

  w.post_alpha = Rnd("bvg.activation_post.act.alpha", static_cast<size_t>(ch), 0.3);
  w.post_beta = Rnd("bvg.activation_post.act.beta", static_cast<size_t>(ch), 0.3);
  w.conv_post.weight = Rnd("bvg.conv_post.weight", static_cast<size_t>(1 * ch * 7), 0.3);
  // use_bias_at_final is FALSE: conv_post has no bias.
  return w;
}

}  // namespace

TEST_CASE("the BigVGAN generator matches upstream, mel to waveform") {
  const std::vector<float> x = Rnd("bvg.x", static_cast<size_t>(kMels * kFrames), 1.0);
  const std::vector<float> got = vllm::models::bigvgan::Forward(Cfg(), W(), x, kFrames);
  REQUIRE(got.size() == static_cast<size_t>(kSamples));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kWave[i]).epsilon(2e-4));
  }
}

TEST_CASE("the upsample ratio is the product of the stage rates") {
  int64_t prod = 1;
  for (int64_t i = 0; i < kNumUpsamples; ++i) {
    prod *= kUpRates[i];
  }
  CHECK(kSamples == kFrames * prod);
}

TEST_CASE("the output is BOUNDED to [-1, 1]") {
  const std::vector<float> x = Rnd("bvg.x", static_cast<size_t>(kMels * kFrames), 1.0);
  const std::vector<float> got = vllm::models::bigvgan::Forward(Cfg(), W(), x, kFrames);
  for (const float v : got) {
    CHECK(v >= -1.0F);
    CHECK(v <= 1.0F);
  }
}

TEST_CASE("every AMP kernel contributes") {
  // The stage output is the MEAN over num_kernels. Perturbing any one block must
  // move the waveform; a port that used only the first would still make sound.
  const std::vector<float> x = Rnd("bvg.x", static_cast<size_t>(kMels * kFrames), 1.0);
  const std::vector<float> base = vllm::models::bigvgan::Forward(Cfg(), W(), x, kFrames);
  for (int64_t j = 0; j < kNumKernels; ++j) {
    auto w = W();
    w.resblocks[static_cast<size_t>(j)].convs1[0].bias[0] += 0.5F;
    CHECK(vllm::models::bigvgan::Forward(Cfg(), w, x, kFrames) != base);
  }
}

// ---------------------------------------------------------------------------
// Loading the SHIPPED BigVGAN checkpoint (#634).
//
// Runs when VLLM_CPP_INDEXTTS2_BIGVGAN points at the converted
// bigvgan.safetensors, and skips LOUDLY otherwise.
// ---------------------------------------------------------------------------
#include <cstdlib>

#include "vllm/model_executor/models/bigvgan_loader.h"

TEST_CASE("the SHIPPED BigVGAN loads with the geometry its config declares") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_BIGVGAN");
  if (env == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_BIGVGAN to the converted "
            "bigvgan.safetensors to check the real checkpoint");
    return;
  }
  const auto g = vllm::models::bigvgan::Load(std::string(env));
  CHECK(g.config.mels == 80);
  CHECK(g.config.init_channels == 1536);
  CHECK(g.config.up_rates.size() == 6);
  CHECK(g.config.num_kernels == 3);
  CHECK(g.weights.resblocks.size() == 18);  // 6 stages x 3 kernels
  CHECK(g.config.tanh_at_final == false);   // use_tanh_at_final is false
  CHECK(g.weights.conv_post.bias.empty());  // use_bias_at_final is false

  // The upsample product must be the HOP LENGTH: 4*4*2*2*2*2 = 256, which is
  // `kHopLength`. If they disagreed the audio would come out at the wrong rate.
  int64_t product = 1;
  for (const int64_t r : g.config.up_rates) {
    product *= r;
  }
  CHECK(product == 256);
}

TEST_CASE("the SHIPPED BigVGAN turns a mel into a bounded WAVEFORM") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_BIGVGAN");
  if (env == nullptr) {
    MESSAGE("SKIPPED: no VLLM_CPP_INDEXTTS2_BIGVGAN, so no audio was rendered");
    return;
  }
  const auto g = vllm::models::bigvgan::Load(std::string(env));
  const int64_t frames = 6;
  std::vector<float> mel(static_cast<size_t>(g.config.mels * frames));
  for (size_t i = 0; i < mel.size(); ++i) {
    mel[i] = -4.0F + 2.0F * std::sin(0.05F * static_cast<float>(i));
  }
  const auto wave = vllm::models::bigvgan::Forward(g.config, g.weights, mel, frames);

  REQUIRE(wave.size() == static_cast<size_t>(frames * 256));
  float lo = wave[0];
  float hi = wave[0];
  double energy = 0.0;
  for (const float v : wave) {
    REQUIRE(std::isfinite(v));
    CHECK(v >= -1.0F);
    CHECK(v <= 1.0F);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    energy += static_cast<double>(v) * static_cast<double>(v);
  }
  // Not silence, and not a rail: a real vocoder on a real mel.
  const double rms = std::sqrt(energy / static_cast<double>(wave.size()));
  CHECK(rms > 1e-3);
  CHECK(hi > lo);
  MESSAGE("rendered " << wave.size() << " samples, range [" << lo << ", " << hi
          << "], rms " << rms);
}
