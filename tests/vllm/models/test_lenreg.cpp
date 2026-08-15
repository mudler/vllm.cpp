// S2Mel length-regulator primitives (#634), gated against torch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "lenreg_goldens.inc"
#include "vllm/model_executor/models/lenreg.h"

namespace {
using namespace lenreg_goldens;

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
double Worst(const std::vector<float>& g, const float* w, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(static_cast<double>(g[i]) - w[i]));
  return m;
}
}  // namespace

TEST_CASE("lenreg nearest interpolate matches torch at a NON-INTEGER ratio") {
  // 7 -> 17 is the case that separates floor(i*in/out) from a rounded or
  // half-offset rule; both alternatives agree at exact multiples.
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got =
      vllm::models::lenreg::InterpolateNearest(x, kChannels, kInFrames, 17);
  REQUIRE(got.size() == static_cast<size_t>(kChannels * 17));
  // Exact: interpolation COPIES samples, it never blends them.
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == kInterp17[i]);
}

TEST_CASE("lenreg nearest interpolate matches torch at an exact multiple") {
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got =
      vllm::models::lenreg::InterpolateNearest(x, kChannels, kInFrames, 14);
  REQUIRE(got.size() == static_cast<size_t>(kChannels * 14));
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == kInterp14[i]);
}

TEST_CASE("lenreg nearest interpolate DOWNsamples the same way") {
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got =
      vllm::models::lenreg::InterpolateNearest(x, kChannels, kInFrames, 3);
  REQUIRE(got.size() == static_cast<size_t>(kChannels * 3));
  for (size_t i = 0; i < got.size(); ++i) CHECK(got[i] == kInterp3[i]);
}

TEST_CASE("lenreg GroupNorm shares statistics across a GROUP, not a channel") {
  // Per-channel statistics would also normalize, and would be a different
  // model; only the values distinguish them.
  const std::vector<float> x(std::begin(kX), std::end(kX));
  std::vector<float> gamma = Rand("gn.weight", kChannels, 0.5);
  for (float& g : gamma) g += 1.0F;
  const std::vector<float> beta = Rand("gn.bias", kChannels, 0.3);
  const std::vector<float> got = vllm::models::lenreg::GroupNorm(
      x, kChannels, kInFrames, kGroups, gamma, beta, 1e-5);
  REQUIRE(got.size() == x.size());
  CHECK(Worst(got, kGroupNorm, got.size()) < 1e-5);
}

TEST_CASE("lenreg Mish is x * tanh(softplus(x)), not SiLU") {
  const std::vector<float> in(std::begin(kMishIn), std::end(kMishIn));
  std::vector<float> got(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    got[i] = static_cast<float>(vllm::models::lenreg::Mish(static_cast<double>(in[i])));
  }
  CHECK(Worst(got, kMishOut, got.size()) < 1e-6);
}

// ---------------------------------------------------------------------------
// The regulator on the SHIPPED checkpoint, and the ACOUSTIC CHAIN it feeds.
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cstdlib>

#include "vllm/model_executor/models/bigvgan_loader.h"
#include "vllm/model_executor/models/cfm.h"
#include "vllm/model_executor/models/indextts2_s2mel_loader.h"

TEST_CASE("the SHIPPED length regulator loads and resamples to the mel rate") {
  const char* env = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  if (env == nullptr) {
    MESSAGE("SKIPPED: no VLLM_CPP_INDEXTTS2_S2MEL");
    return;
  }
  const auto file = vllm::SafetensorsFile::Open(std::string(env));
  vllm::models::lenreg::RegulatorConfig cfg;
  const auto w = vllm::models::lenreg::LoadRegulator(file, &cfg);
  CHECK(cfg.channels == 512);
  CHECK(cfg.in_channels == 1024);
  CHECK(w.conv_w.size() == 4);   // four (conv, norm) pairs
  CHECK(w.norm_w.size() == 4);
  CHECK(!w.out_conv_w.empty());

  const int64_t in_frames = 5;
  const int64_t out_frames = 20;
  std::vector<float> content(static_cast<size_t>(in_frames * cfg.in_channels));
  for (size_t i = 0; i < content.size(); ++i) {
    content[i] = 0.05F * std::sin(0.02F * static_cast<float>(i));
  }
  const auto out =
      vllm::models::lenreg::RegulateHost(cfg, w, content, in_frames, out_frames);
  REQUIRE(out.size() == static_cast<size_t>(out_frames * cfg.channels));
  for (const float v : out) {
    REQUIRE(std::isfinite(v));
  }
}

TEST_CASE("the ACOUSTIC CHAIN runs on real weights: codes to a waveform") {
  const char* s2 = std::getenv("VLLM_CPP_INDEXTTS2_S2MEL");
  const char* bv = std::getenv("VLLM_CPP_INDEXTTS2_BIGVGAN");
  if (s2 == nullptr || bv == nullptr) {
    MESSAGE("SKIPPED: set VLLM_CPP_INDEXTTS2_S2MEL and VLLM_CPP_INDEXTTS2_BIGVGAN "
            "to run the chain; without them NO audio was produced");
    return;
  }
  const auto file = vllm::SafetensorsFile::Open(std::string(s2));
  auto est = vllm::models::indextts2::LoadS2MelEstimator(file, 8);
  vllm::models::lenreg::RegulatorConfig rc;
  const auto rw = vllm::models::lenreg::LoadRegulator(file, &rc);
  const auto g = vllm::models::bigvgan::Load(std::string(bv));

  const int64_t code_frames = 4;
  const int64_t mel_frames = 16;
  std::vector<float> content(static_cast<size_t>(code_frames * rc.in_channels));
  for (size_t i = 0; i < content.size(); ++i) {
    content[i] = 0.05F * std::sin(0.02F * static_cast<float>(i));
  }
  const auto cond =
      vllm::models::lenreg::RegulateHost(rc, rw, content, code_frames, mel_frames);

  const int64_t D = est.stack_config.dim;
  const int64_t C = est.front_config.in_channels;
  est.front_config.frames = mel_frames;
  est.stack_config.frames = mel_frames;
  est.tail_config.frames = mel_frames;
  const std::vector<float> freqs(
      static_cast<size_t>(mel_frames * est.stack_config.head_dim), 0.0F);
  const std::vector<float> style(static_cast<size_t>(est.front_config.style), 0.02F);
  const std::vector<float> prompt(static_cast<size_t>(C * mel_frames), 0.0F);

  std::vector<float> x(static_cast<size_t>(C * mel_frames));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.01F * std::sin(0.7F * static_cast<float>(i));
  }
  const int steps = 4;  // enough to prove the loop integrates; not a quality claim
  for (int s = 0; s < steps; ++s) {
    const float t = static_cast<float>(s) / steps;
    const std::vector<float> t1(static_cast<size_t>(D), 0.01F * t);
    const auto in_c = vllm::models::dit_front::BuildXIn(est.front_config, est.front, x,
                                                        prompt, cond, style, false);
    const auto r_c = vllm::models::dit_stack::Forward(est.stack_config, est.stack, in_c,
                                                      t1, freqs);
    const auto v_c = vllm::models::dit_tail::Forward(est.tail_config, est.tail, r_c, x,
                                                     t, t1, {});
    const auto in_u = vllm::models::dit_front::BuildXIn(est.front_config, est.front, x,
                                                        prompt, cond, style, true);
    const auto r_u = vllm::models::dit_stack::Forward(est.stack_config, est.stack, in_u,
                                                      t1, freqs);
    const auto v_u = vllm::models::dit_tail::Forward(est.tail_config, est.tail, r_u, x,
                                                     t, t1, {});
    // The CONDITIONAL and UNCONDITIONAL velocities must differ, or CFG is inert
    // and the guidance rate does nothing.
    CHECK(v_c != v_u);
    x = vllm::models::cfm::EulerStepCfg(x, v_c, v_u, C, mel_frames, 1.0 / steps, 0.7, 0);
  }
  REQUIRE(x.size() == static_cast<size_t>(C * mel_frames));

  const auto wave = vllm::models::bigvgan::Forward(g.config, g.weights, x, mel_frames);
  REQUIRE(wave.size() == static_cast<size_t>(mel_frames * 256));
  double energy = 0.0;
  float lo = wave[0];
  float hi = wave[0];
  for (const float v : wave) {
    REQUIRE(std::isfinite(v));
    CHECK(v >= -1.0F);
    CHECK(v <= 1.0F);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    energy += static_cast<double>(v) * static_cast<double>(v);
  }
  const double rms = std::sqrt(energy / static_cast<double>(wave.size()));
  CHECK(rms > 1e-3);   // not silence
  CHECK(hi > lo);      // not a rail
  MESSAGE("acoustic chain: " << code_frames << " code frames -> " << mel_frames
          << " mel frames -> " << wave.size() << " samples ("
          << static_cast<double>(wave.size()) / 22050.0 << " s), rms " << rms);
}
