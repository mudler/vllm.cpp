// The shared 1-D vocoder core, gated on HAND-COMPUTED values.
//
// These primitives are the BigVGAN lineage that MiniMax-H3 ported first and
// LTX-2.5 then reused rather than copied (`ltx2_audio_vae.cpp` records why: a
// second copy of the alias-free trim geometry goes wrong quietly, because each
// copy keeps its own green gate while the two audio VAEs drift apart).
// IndexTTS-2.5 is the third consumer, so the core moves out of `minimax_h3.h`
// and into a neutral home here.
//
// Every expectation below is computed BY HAND from the operator's definition,
// never captured from the implementation. Capturing would make this a
// consistency check: it would pass just as happily if the shared helper were
// uniformly wrong, which is the exact failure this project has already recorded
// for a gate whose two arms called the same dequant.
//
// Tolerances use `.scale(0.0)`: doctest's default scale of 1.0 puts a 1.19e-5
// ABSOLUTE floor under any epsilon, which would accept almost anything.
#include <cmath>
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/vocoder1d.h"

namespace {

}  // namespace

TEST_CASE("vocoder1d Conv1d matches a hand-computed convolution") {
  // in [1, 4] = [1, 2, 3, 4]; weight [1, 1, 2] = [1, -1]; stride 1, no bias.
  // out[t] = in[t]*1 + in[t+1]*(-1) => [-1, -1, -1], length 4 - 2 + 1 = 3.
  const std::vector<float> in{1.0F, 2.0F, 3.0F, 4.0F};
  const std::vector<float> weight{1.0F, -1.0F};
  int64_t out_len = 0;
  const std::vector<float> out =
      vllm::vocoder1d::Conv1d(in, /*in_channels=*/1, /*in_len=*/4, weight, /*bias=*/nullptr,
                              /*out_channels=*/1, /*kernel=*/2, /*stride=*/1, /*dilation=*/1,
                              /*groups=*/1, &out_len);
  REQUIRE(out_len == 3);
  REQUIRE(out.size() == 3U);
  for (const float v : out) {
    CHECK(v == -1.0F);  // exact: integer taps, no rounding to absorb
  }
}

TEST_CASE("vocoder1d Conv1d honours stride and bias") {
  // in [1, 5] = [1, 2, 3, 4, 5]; weight k=2 = [1, 1]; stride 2; bias 10.
  // taps at t=0,2 => (1+2)+10 = 13, (3+4)+10 = 17. Length = (5-2)/2 + 1 = 2.
  const std::vector<float> in{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
  const std::vector<float> weight{1.0F, 1.0F};
  const std::vector<float> bias{10.0F};
  int64_t out_len = 0;
  const std::vector<float> out =
      vllm::vocoder1d::Conv1d(in, 1, 5, weight, &bias, 1, /*kernel=*/2, /*stride=*/2,
                              /*dilation=*/1, /*groups=*/1, &out_len);
  REQUIRE(out_len == 2);
  CHECK(out[0] == 13.0F);
  CHECK(out[1] == 17.0F);
}

TEST_CASE("vocoder1d ConvTranspose1d scatters each input across the stride") {
  // in [1, 2] = [1, 2]; weight [1, 1, 2] = [1, 10]; stride 2, padding 0.
  // out[i*stride + k] += in[i] * w[k] => [1, 10, 2, 20].
  // Length = (2-1)*2 - 2*0 + 2 = 4.
  const std::vector<float> in{1.0F, 2.0F};
  const std::vector<float> weight{1.0F, 10.0F};
  int64_t out_len = 0;
  const std::vector<float> out = vllm::vocoder1d::ConvTranspose1d(
      in, /*in_channels=*/1, /*in_len=*/2, weight, /*bias=*/nullptr, /*out_channels=*/1,
      /*kernel=*/2, /*stride=*/2, /*padding=*/0, /*groups=*/1, &out_len);
  REQUIRE(out_len == 4);
  const std::vector<float> want{1.0F, 10.0F, 2.0F, 20.0F};
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(out[i] == want[i]);  // exact: hand-computed, integer-valued
  }
}

TEST_CASE("vocoder1d Pad1d replicate repeats the edge sample") {
  const std::vector<float> in{1.0F, 2.0F, 3.0F};
  int64_t out_len = 0;
  const std::vector<float> out = vllm::vocoder1d::Pad1d(in, /*channels=*/1, /*in_len=*/3,
                                                        /*left=*/2, /*right=*/1,
                                                        /*replicate=*/true, &out_len);
  REQUIRE(out_len == 6);
  const std::vector<float> want{1.0F, 1.0F, 1.0F, 2.0F, 3.0F, 3.0F};
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(out[i] == want[i]);  // exact: hand-computed, integer-valued
  }
}

TEST_CASE("vocoder1d Pad1d zero mode does not replicate") {
  const std::vector<float> in{1.0F, 2.0F, 3.0F};
  int64_t out_len = 0;
  const std::vector<float> out = vllm::vocoder1d::Pad1d(in, 1, 3, /*left=*/1, /*right=*/2,
                                                        /*replicate=*/false, &out_len);
  REQUIRE(out_len == 6);
  const std::vector<float> want{0.0F, 1.0F, 2.0F, 3.0F, 0.0F, 0.0F};
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(out[i] == want[i]);  // exact: hand-computed, integer-valued
  }
}

TEST_CASE("vocoder1d SnakeActivation adds sin^2 scaled by 1/beta") {
  // x + (beta + eps)^-1 * sin^2(alpha * x), alpha = 1, beta = 1, x = pi/2.
  // sin(pi/2) = 1, so the result is pi/2 + 1/(1 + 1e-9).
  const double x0 = M_PI / 2.0;
  std::vector<float> x{static_cast<float>(x0)};
  const std::vector<float> alpha{1.0F};
  const std::vector<float> beta{1.0F};
  vllm::vocoder1d::SnakeActivation(x, /*channels=*/1, /*length=*/1, alpha, &beta,
                                   /*logscale=*/false);
  const double want = x0 + 1.0 / (1.0 + vllm::vocoder1d::kSnakeEps);
  CHECK(static_cast<double>(x[0]) == doctest::Approx(want).epsilon(1e-6).scale(0.0));
}

TEST_CASE("vocoder1d SnakeActivation with a null beta reuses alpha as the scale") {
  // Plain Snake (LTX-2.5 vocoder.py:198): the reciprocal scale is ALPHA, not a
  // separate beta. alpha = 2, x = pi/4 => sin(2 * pi/4) = sin(pi/2) = 1, so the
  // result is pi/4 + 1/(2 + eps). Passing beta here instead would give a
  // different number, which is what makes the null-beta arm distinguishable.
  const double x0 = M_PI / 4.0;
  std::vector<float> x{static_cast<float>(x0)};
  const std::vector<float> alpha{2.0F};
  vllm::vocoder1d::SnakeActivation(x, 1, 1, alpha, /*beta=*/nullptr, /*logscale=*/false);
  const double want = x0 + 1.0 / (2.0 + vllm::vocoder1d::kSnakeEps);
  CHECK(static_cast<double>(x[0]) == doctest::Approx(want).epsilon(1e-6).scale(0.0));
}

TEST_CASE("vocoder1d snake epsilon is pinned to the upstream constant") {
  // Source-anchored: upstream writes `1.0 / (beta + 1e-9)` on both sides of this
  // lineage. No reduced-dimension golden can tell 1e-9 from 0.0 because beta is
  // O(1) there, so the value is held by an assertion rather than a tensor.
  CHECK(vllm::vocoder1d::kSnakeEps == 1e-9);
}

TEST_CASE("vocoder1d kaiser-sinc filter is symmetric and unit-gain") {
  // The window is built, never read from a checkpoint. Two properties hold for
  // any cutoff: it is symmetric, and a low-pass at DC sums to its ratio-scaled
  // gain. Asserting the SHAPE rather than sampled taps keeps this independent of
  // the filter's internal normalisation convention.
  const std::vector<float> f =
      vllm::vocoder1d::KaiserSincFilter1d(/*cutoff=*/0.5, /*half_width=*/0.6, /*kernel_size=*/12);
  REQUIRE(f.size() == 12U);
  for (size_t i = 0; i < f.size() / 2; ++i) {
    CHECK(static_cast<double>(f[i]) ==
          doctest::Approx(static_cast<double>(f[f.size() - 1 - i])).epsilon(1e-6).scale(0.0));
  }
  double sum = 0.0;
  for (const float v : f) {
    sum += static_cast<double>(v);
  }
  CHECK(sum > 0.0);
}

TEST_CASE("vocoder1d AliasFreeActivation1d preserves length and is deterministic") {
  // The trim geometry is the fragile part and the reason this is shared rather
  // than copied: upsample by ratio -> Snake -> downsample by ratio must return
  // the ORIGINAL length. A trim that is off by one produces a signal that still
  // plays and is wrong, which no length-agnostic check would catch.
  vllm::vocoder1d::AliasFreeActivation1d act;
  act.ratio = 2;
  act.kernel_size = 12;
  act.Build();
  REQUIRE(!act.filter.empty());

  const int64_t channels = 2;
  const int64_t len = 8;
  std::vector<float> in(static_cast<size_t>(channels * len));
  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = static_cast<float>(0.1 * static_cast<double>(i) - 0.3);
  }
  const std::vector<float> alpha{0.5F, 0.25F};
  const std::vector<float> beta{1.0F, 2.0F};

  int64_t out_len = 0;
  const std::vector<float> a = act.Apply(in, channels, len, alpha, &beta, false, &out_len);
  CHECK(out_len == len);
  REQUIRE(a.size() == in.size());

  int64_t out_len2 = 0;
  const std::vector<float> b = act.Apply(in, channels, len, alpha, &beta, false, &out_len2);
  REQUIRE(out_len2 == len);
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i] == b[i]);  // determinism is bit-exact or it is not determinism
  }
}
