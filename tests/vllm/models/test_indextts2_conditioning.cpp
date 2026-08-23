// The reference clip's two pure conditioning steps.
//
// These exist as a gate because the engine path that uses them needs a 5 GB
// checkpoint, so the only thing that ever exercised them was a manual probe.
// Both are hand-computable at this size, so the expected values below are
// arithmetic, not a recording of what the code happened to produce.

#include "vllm/model_executor/models/indextts2_conditioning.h"

#include <doctest/doctest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using vllm::indextts2::MeanCentreColumns;
using vllm::indextts2::ProjectSpeaker;

TEST_CASE("mean-centring removes each COLUMN's mean, not the global one") {
  // 3 frames x 2 bins, frame-major. Column 0: 1,3,5 -> mean 3. Column 1:
  // 10,20,30 -> mean 20. A global-mean implementation would subtract 11.5 from
  // everything and leave column 0 at -10.5, so the two are far apart here.
  std::vector<float> mel{1.0F, 10.0F, 3.0F, 20.0F, 5.0F, 30.0F};
  MeanCentreColumns(mel, /*bins=*/2, /*frames=*/3);

  CHECK(mel[0] == doctest::Approx(-2.0F));
  CHECK(mel[1] == doctest::Approx(-10.0F));
  CHECK(mel[2] == doctest::Approx(0.0F));
  CHECK(mel[3] == doctest::Approx(0.0F));
  CHECK(mel[4] == doctest::Approx(2.0F));
  CHECK(mel[5] == doctest::Approx(10.0F));

  // And every column now sums to zero, which is the property CAMPPlus assumes.
  for (int64_t c = 0; c < 2; ++c) {
    double s = 0.0;
    for (int64_t f = 0; f < 3; ++f) s += mel[static_cast<size_t>(f * 2 + c)];
    CHECK(std::abs(s) < 1e-5);
  }
}

TEST_CASE("mean-centring REFUSES a shape that does not describe the buffer") {
  std::vector<float> mel{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  CHECK_THROWS_AS(MeanCentreColumns(mel, 4, 3), std::exception);  // 12 != 6
  CHECK_THROWS_AS(MeanCentreColumns(mel, 0, 3), std::exception);
  CHECK_THROWS_AS(MeanCentreColumns(mel, 2, 0), std::exception);
}

TEST_CASE("the speaker projection is w * style + b, ROW-major over style") {
  // out_dim 2, style_dim 3. Row-major [out, style]:
  //   out0 = 1*1 + 2*2 + 3*3 = 14, + b0 0.5 -> 14.5
  //   out1 = 4*1 + 5*2 + 6*3 = 32, + b1 -1.0 -> 31.0
  // A column-major reading would pick w[0],w[2],w[4] = 1,3,5 -> 1+6+15 = 22,
  // so this distinguishes the two layouts.
  const std::vector<float> style{1.0F, 2.0F, 3.0F};
  const std::vector<float> w{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> b{0.5F, -1.0F};

  const std::vector<float> out = ProjectSpeaker(style, w, b, /*out_dim=*/2);
  REQUIRE(out.size() == 2u);
  CHECK(out[0] == doctest::Approx(14.5F));
  CHECK(out[1] == doctest::Approx(31.0F));
}

TEST_CASE("an EMPTY bias is legal and contributes nothing") {
  const std::vector<float> style{1.0F, 2.0F, 3.0F};
  const std::vector<float> w{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> out = ProjectSpeaker(style, w, {}, 2);
  REQUIRE(out.size() == 2u);
  CHECK(out[0] == doctest::Approx(14.0F));
  CHECK(out[1] == doctest::Approx(32.0F));
}

TEST_CASE("the projection DEPENDS on the style vector") {
  // The defect this whole change fixes is a conditioning that ignores the
  // reference clip, so a gate that never varies the style would not see it.
  const std::vector<float> w{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> a = ProjectSpeaker({1.0F, 2.0F, 3.0F}, w, {}, 2);
  const std::vector<float> c = ProjectSpeaker({3.0F, 2.0F, 1.0F}, w, {}, 2);
  CHECK(a[0] != doctest::Approx(c[0]));
}

TEST_CASE("a weight that does not match the widths is REFUSED") {
  const std::vector<float> style{1.0F, 2.0F, 3.0F};
  CHECK_THROWS_AS(ProjectSpeaker(style, {1.0F, 2.0F, 3.0F, 4.0F}, {}, 2), std::exception);
  CHECK_THROWS_AS(ProjectSpeaker(style, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}, {1.0F}, 2),
                  std::exception);
  CHECK_THROWS_AS(ProjectSpeaker({}, {}, {}, 2), std::exception);
}
