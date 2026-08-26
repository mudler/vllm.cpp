// BigVGAN generator against upstream goldens. See bigvgan.h.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "bigvgan_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/bigvgan.h"
// VT-CONV1D-MODEL-BLOCK (#1684): the time-block case reads the geometry it claims
// rather than assuming it. Same reach as test_vocoder1d and test_host_parallel.
#include "vt/cpu/cpu_conv1d_block.h"

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

TEST_CASE("the BigVGAN generator is exact ACROSS a time block boundary") {
  // WHY THIS CASE EXISTS (#1684). The `vt::Conv1d` CPU provider cuts its work
  // into (time block, output row) pairs (#1664, src/vt/cpu/cpu_conv1d_block.h).
  // Until this case existed THIS suite reached that provider at SINGLE-BLOCK
  // shapes only -- the golden fixture above is far too short to fill one work
  // unit's 512 KiB activation budget -- so a defect confined to the second axis
  // reddened the op's own suite and nothing else: a sign flip applied only where
  // `blocks > 1` left eight of the ten consumer suites green. This is BigVGAN's
  // own arm of that gate, and it enters through `bigvgan::Forward`, the
  // generator's only public entry point.
  //
  // WHY THE EXPECTATION IS TWO SHORTER DECODES AND NOT A GOLDEN. The generator
  // ends every stage in an anti-aliased Snake, so no closed form survives the
  // chain and a golden would need upstream re-run at a 512 KiB activation. It
  // does not need one. Every stage is a LOCAL, shift-equivariant operator --
  // zero-padded convolutions, a strided transpose, a pointwise activation, a
  // residual add and a mean -- so decoding a WINDOW of the mel reproduces the
  // long decode sample for sample, except within the window's own edge. Two
  // windows whose interiors OVERLAP therefore cover the whole long waveform, and
  // each of them is short enough that its convolutions take ONE block. That is
  // the contrast: the same arithmetic, blocked on one side and not on the other,
  // compared BIT FOR BIT -- a cell's reduction is `seed`, then `ic`, then `k`,
  // and none of that mentions the block, so anything less than equality is a
  // defect.
  //
  // AND THE SECOND WINDOW IS THE ONE THAT MATTERS. The block boundary always
  // falls at the block length, which is the longest a single-block reference can
  // be, so a prefix window alone can never reach it. The second window starts
  // late enough that the boundary lands in its interior, which is what makes
  // this case see a defect confined to the LAST block as well as one applied to
  // every block.
  constexpr int64_t kBlockMels = 256;
  constexpr int64_t kBlockInitCh = 8;
  constexpr int64_t kRefFrames = 480;   // == the conv_pre block length, asserted below
  constexpr int64_t kLongFrames = 704;  // > kRefFrames, so the long decode blocks
  constexpr int64_t kUpsample = 2;
  // The edge of a window contaminates inwards by the chain's receptive field,
  // about 44 output samples here (conv_pre 3 frames, the transpose 2, the three
  // dilated resblock pairs 12, seven 12-tap 2x resamplers 21, conv_post 3). 192
  // is that with a 4x margin, and the coverage assertion below proves the two
  // windows still meet.
  constexpr int64_t kEdge = 192;

  vllm::models::bigvgan::Config cfg;
  cfg.mels = kBlockMels;
  cfg.init_channels = kBlockInitCh;
  cfg.up_rates = {kUpsample};
  cfg.up_kernels = {4};
  cfg.num_kernels = 1;
  cfg.snake_logscale = true;
  // tanh rather than the shipped clamp: two clamped waveforms agree on every
  // saturated sample for the wrong reason.
  cfg.tanh_at_final = true;

  // conv_pre is the convolution that blocks: 256 input channels over 710 padded
  // positions is 710 KiB against the 512 KiB `kConv1dSliceBytes` budget.
  // Everything after the upsample is 4 channels over 1408 positions, 22 KiB, so
  // exactly ONE convolution in the chain crosses a boundary and this says which.
  const int64_t block = vt::cpu::Conv1dTimeBlock(kBlockMels, /*kernel=*/7, /*stride=*/1,
                                                 /*dilation=*/1, kLongFrames);
  INFO("conv_pre block=" << block << " long=" << kLongFrames << " ref=" << kRefFrames);
  REQUIRE(block < kLongFrames);   // TEETH: the long decode really blocks
  REQUIRE(block == kRefFrames);   // TEETH: the references really do not
  REQUIRE(block % vt::cpu::kConv1dPosTile == 0);

  const int64_t out_ch = kBlockInitCh / 2;
  vllm::models::bigvgan::Weights w;
  // Small scales on purpose: the comparison asserts the samples are NOT
  // saturated, because a flattened tanh compares equal without proving anything
  // about the convolution underneath it.
  w.conv_pre.weight =
      Rnd("bvgblk.conv_pre.weight", static_cast<size_t>(kBlockInitCh * kBlockMels * 7), 0.05);
  w.conv_pre.bias = Rnd("bvgblk.conv_pre.bias", static_cast<size_t>(kBlockInitCh), 0.05);
  vllm::models::bigvgan::ConvSpec up;
  up.weight = Rnd("bvgblk.ups.0.weight", static_cast<size_t>(kBlockInitCh * out_ch * 4), 0.2);
  up.bias = Rnd("bvgblk.ups.0.bias", static_cast<size_t>(out_ch), 0.05);
  w.ups.push_back(up);
  vllm::models::bigvgan::AmpBlock rb;
  rb.kernel = 3;
  rb.dilations = {1, 3, 5};
  for (size_t d = 0; d < rb.dilations.size(); ++d) {
    const std::string tag = std::to_string(d);
    vllm::models::bigvgan::ConvSpec c1;
    c1.weight = Rnd("bvgblk.rb.convs1." + tag, static_cast<size_t>(out_ch * out_ch * 3), 0.1);
    c1.bias = Rnd("bvgblk.rb.convs1.bias." + tag, static_cast<size_t>(out_ch), 0.05);
    vllm::models::bigvgan::ConvSpec c2;
    c2.weight = Rnd("bvgblk.rb.convs2." + tag, static_cast<size_t>(out_ch * out_ch * 3), 0.1);
    c2.bias = Rnd("bvgblk.rb.convs2.bias." + tag, static_cast<size_t>(out_ch), 0.05);
    rb.convs1.push_back(c1);
    rb.convs2.push_back(c2);
  }
  for (size_t i = 0; i < 2 * rb.dilations.size(); ++i) {
    rb.alpha.push_back(
        Rnd("bvgblk.rb.alpha." + std::to_string(i), static_cast<size_t>(out_ch), 0.2));
    rb.beta.push_back(
        Rnd("bvgblk.rb.beta." + std::to_string(i), static_cast<size_t>(out_ch), 0.2));
  }
  w.resblocks.push_back(rb);
  w.post_alpha = Rnd("bvgblk.post_alpha", static_cast<size_t>(out_ch), 0.2);
  w.post_beta = Rnd("bvgblk.post_beta", static_cast<size_t>(out_ch), 0.2);
  w.conv_post.weight = Rnd("bvgblk.conv_post.weight", static_cast<size_t>(out_ch * 7), 0.3);

  const std::vector<float> mel_long =
      Rnd("bvgblk.mel", static_cast<size_t>(kBlockMels * kLongFrames), 1.0);
  const std::vector<float> wave_long =
      vllm::models::bigvgan::Forward(cfg, w, mel_long, kLongFrames);
  REQUIRE(wave_long.size() == static_cast<size_t>(kUpsample * kLongFrames));

  // A window of the mel, decoded on its own, must reproduce the long decode over
  // every output sample that its own edges cannot reach.
  int64_t compared = 0;
  int64_t wrong = 0;
  int64_t first_wrong = -1;
  double worst = 0.0;
  double peak = 0.0;
  float lo = wave_long[0];
  float hi = wave_long[0];
  auto window = [&](int64_t start, int64_t frames, bool trim_left, bool trim_right) {
    REQUIRE(start % kUpsample == 0);  // the transpose is only equivariant on its own grid
    REQUIRE(frames <= block);         // TEETH: a reference that blocked would prove nothing
    std::vector<float> mel(static_cast<size_t>(kBlockMels * frames));
    for (int64_t c = 0; c < kBlockMels; ++c) {
      for (int64_t t = 0; t < frames; ++t) {
        mel[static_cast<size_t>(c * frames + t)] =
            mel_long[static_cast<size_t>(c * kLongFrames + start + t)];
      }
    }
    const std::vector<float> wave = vllm::models::bigvgan::Forward(cfg, w, mel, frames);
    REQUIRE(wave.size() == static_cast<size_t>(kUpsample * frames));
    const int64_t base = kUpsample * start;
    const int64_t from = trim_left ? kEdge : 0;
    const int64_t to = kUpsample * frames - (trim_right ? kEdge : 0);
    REQUIRE(to > from);
    for (int64_t i = from; i < to; ++i) {
      const float a = wave_long[static_cast<size_t>(base + i)];
      const float b = wave[static_cast<size_t>(i)];
      ++compared;
      if (a != b) {
        if (first_wrong < 0) first_wrong = base + i;
        ++wrong;
        worst = std::max(worst, std::abs(static_cast<double>(a) - static_cast<double>(b)));
      }
      peak = std::max(peak, std::abs(static_cast<double>(a)));
      lo = std::min(lo, a);
      hi = std::max(hi, a);
    }
    return std::pair<int64_t, int64_t>{base + from, base + to};
  };

  // Window A is the prefix, so it shares the long decode's own left edge and
  // only its right edge is trimmed. Window B ends where the long decode does.
  const auto span_a = window(0, kRefFrames, /*trim_left=*/false, /*trim_right=*/true);
  const auto span_b =
      window(kLongFrames - kRefFrames, kRefFrames, /*trim_left=*/true, /*trim_right=*/false);
  // COVERAGE, asserted rather than assumed: the two windows must MEET, and the
  // block boundary must land inside one of them.
  CHECK(span_a.first == 0);
  CHECK(span_b.second == kUpsample * kLongFrames);
  CHECK(span_b.first <= span_a.second);
  const int64_t boundary = kUpsample * block;
  INFO("spans [" << span_a.first << "," << span_a.second << ") and [" << span_b.first << ","
                 << span_b.second << "), boundary at sample " << boundary);
  CHECK(boundary > span_b.first);
  CHECK(boundary < span_b.second);

  INFO("samples compared=" << compared << " differing=" << wrong << " first at " << first_wrong
                           << " worst|diff|=" << worst);
  CHECK(wrong == 0);
  // A flattened tanh, or a constant waveform, would make the comparison vacuous.
  CHECK(peak < 0.99);
  CHECK(hi - lo > 0.1F);
  MESSAGE("bigvgan across a block boundary: " << compared << " samples compared bit for bit, peak "
                                              << peak << ", span " << (hi - lo));
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
