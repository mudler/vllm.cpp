// DSV4-DSPARK-DRAFTER W-2 — the drafter's ENTRY, gated against HAND-COMPUTED
// values rather than against the helpers the code calls.
//
// Spec: `.agents/specs/dsv4-dspark-drafter.md`. Every expectation below is worked
// out from the definitions: the tap is a stream MEAN
// (`exllamav3/modules/transformer.py:198-203`), the concatenation is on the last
// axis, and `main_norm` applies to `main_proj`'s output.
#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "vllm/model_executor/models/deepseek_v4_dspark.h"

namespace {

// A kPlain f32 view, so the gate exercises the real signature without needing a
// quantized payload -- the dequant path has its own gate.
vllm::DeepseekV4MtpTensorView PlainView(const std::vector<float>& data, int64_t out_dim,
                                        int64_t in_dim) {
  vllm::DeepseekV4MtpTensorView v;
  v.format = vllm::DeepseekV4MtpFormat::kPlain;
  v.dtype = "F32";
  v.shape = {out_dim, in_dim};
  v.data = reinterpret_cast<const uint8_t*>(data.data());
  v.out_dim = out_dim;
  v.in_dim = in_dim;
  return v;
}

}  // namespace

TEST_CASE("W-2: the tap is the STREAM MEAN, hand-computed") {
  // 2 tokens, hc_mult 4, hidden 3. Token 0 dim 0 holds {1,2,3,4} across streams,
  // whose mean is 2.5 -- a value no other reduction produces (sum 10, first 1,
  // last 4, max 4).
  const int64_t T = 2, hc = 4, H = 3;
  std::vector<float> x(static_cast<size_t>(T * hc * H), 0.0f);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t s = 0; s < hc; ++s)
      for (int64_t d = 0; d < H; ++d)
        x[static_cast<size_t>((t * hc + s) * H + d)] =
            static_cast<float>((s + 1) + 10 * d + 100 * t);

  const std::vector<float> got = vllm::dspark::StreamMeanTap(x, T, hc, H);
  REQUIRE(got.size() == static_cast<size_t>(T * H));
  CHECK(got[0] == doctest::Approx(2.5f));    // (1+2+3+4)/4
  CHECK(got[1] == doctest::Approx(12.5f));   // +10
  CHECK(got[2] == doctest::Approx(22.5f));   // +20
  CHECK(got[3] == doctest::Approx(102.5f));  // token 1
}

TEST_CASE("W-2: the stream mean is NOT the sum, the first stream, or the last") {
  // Guards the reductions a reader might substitute. Each would be a drafter that
  // runs and emits correct tokens while drafting badly.
  const int64_t T = 1, hc = 4, H = 1;
  const std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> got = vllm::dspark::StreamMeanTap(x, T, hc, H);
  REQUIRE(got.size() == 1u);
  CHECK(got[0] == doctest::Approx(2.5f));
  CHECK(got[0] != doctest::Approx(10.0f));  // sum
  CHECK(got[0] != doctest::Approx(1.0f));   // first stream
  CHECK(got[0] != doctest::Approx(4.0f));   // last stream
}

TEST_CASE("W-2: ProjectTaps concatenates on the LAST axis, then norms") {
  // hidden 2, n_taps 3 => main_proj is [2, 6]. Identity-ish rows chosen so the
  // pre-norm product is computable by inspection.
  const int64_t T = 1, H = 2, N = 3;
  const std::vector<std::vector<float>> taps{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
  // row 0 selects tap0[0], tap1[0], tap2[0]; row 1 selects tap0[1] only.
  const std::vector<float> w{
      1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,   // -> 1 + 3 + 5 = 9
      0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // -> 2
  };
  const auto view = PlainView(w, H, N * H);
  const std::vector<float> gamma{1.0f, 1.0f};
  const float eps = 0.0f;

  const std::vector<float> got =
      vllm::dspark::ProjectTaps(taps, view, gamma, eps, T, H);
  REQUIRE(got.size() == static_cast<size_t>(T * H));

  // Pre-norm [9, 2]; rms = sqrt((81+4)/2) = sqrt(42.5).
  const double rms = std::sqrt((81.0 + 4.0) / 2.0);
  CHECK(got[0] == doctest::Approx(static_cast<float>(9.0 / rms)));
  CHECK(got[1] == doctest::Approx(static_cast<float>(2.0 / rms)));
}

TEST_CASE("W-2: main_norm's gamma is applied per channel") {
  const int64_t T = 1, H = 2, N = 1;
  const std::vector<std::vector<float>> taps{{3.0f, 4.0f}};
  const std::vector<float> w{1.0f, 0.0f, 0.0f, 1.0f};  // identity [2,2]
  const auto view = PlainView(w, H, N * H);
  const std::vector<float> gamma{2.0f, 10.0f};
  const std::vector<float> got =
      vllm::dspark::ProjectTaps(taps, view, gamma, 0.0f, T, H);
  const double rms = std::sqrt((9.0 + 16.0) / 2.0);
  CHECK(got[0] == doctest::Approx(static_cast<float>(3.0 / rms * 2.0)));
  CHECK(got[1] == doctest::Approx(static_cast<float>(4.0 / rms * 10.0)));
}

TEST_CASE("W-2: a tap count that disagrees with main_proj REFUSES") {
  // `dspark_target_layer_ids` has 3 entries and `main_proj` is [H, 3H]. If they
  // ever disagree the projection would read past the taps, so it refuses rather
  // than producing a number.
  const int64_t T = 1, H = 2;
  const std::vector<std::vector<float>> two_taps{{1.0f, 2.0f}, {3.0f, 4.0f}};
  const std::vector<float> w(static_cast<size_t>(H * 3 * H), 0.0f);
  const auto view = PlainView(w, H, 3 * H);  // built for THREE taps
  const std::vector<float> gamma{1.0f, 1.0f};
  CHECK_THROWS(vllm::dspark::ProjectTaps(two_taps, view, gamma, 0.0f, T, H));
}

TEST_CASE("W-2: a tap that is still a STREAM STACK refuses") {
  // The most likely wiring error: handing over `[T, hc, H]` instead of its mean.
  // It has the wrong element count, so it is catchable here -- unlike a wrong
  // REDUCTION, which is not, and which is why the tap has its own gate above.
  const int64_t T = 1, H = 2;
  const std::vector<std::vector<float>> stacked{
      std::vector<float>(static_cast<size_t>(4 * H), 1.0f)};
  const std::vector<float> w(static_cast<size_t>(H * H), 0.0f);
  const auto view = PlainView(w, H, H);
  const std::vector<float> gamma{1.0f, 1.0f};
  CHECK_THROWS(vllm::dspark::ProjectTaps(stacked, view, gamma, 0.0f, T, H));
}
