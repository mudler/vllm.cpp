// The supplied emotion vector. See emovec.h.
//
// Hand-computed throughout: the operation is a cosine argmax, a row selection
// and a weighted sum, so values chosen to be distinguishable prove it exactly.
#include <cstdint>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/emovec.h"

namespace ev = vllm::models::emovec;

namespace {

// Two emotions, two rows each, style_dim 2, out_dim 2.
//   bank 0 speakers: row0 = (1, 0), row1 = (0, 1)
//   bank 0 emotions: row0 = (10, 20), row1 = (30, 40)
//   bank 1 speakers: row0 = (0, 1), row1 = (1, 0)   <- REVERSED
//   bank 1 emotions: row0 = (1, 2), row1 = (3, 4)
std::vector<ev::EmotionBank> Banks() {
  ev::EmotionBank a;
  a.rows = 2;
  a.speakers = {1.0F, 0.0F, 0.0F, 1.0F};
  a.emotions = {10.0F, 20.0F, 30.0F, 40.0F};
  ev::EmotionBank b;
  b.rows = 2;
  b.speakers = {0.0F, 1.0F, 1.0F, 0.0F};
  b.emotions = {1.0F, 2.0F, 3.0F, 4.0F};
  return {a, b};
}

}  // namespace

TEST_CASE("each emotion searches ITS OWN speaker matrix") {
  // style = (1, 0) matches bank 0's row 0 and bank 1's row 1. One shared index
  // would pick the same row in both, which is the simplification this catches.
  std::vector<int64_t> chosen;
  const std::vector<float> out =
      ev::Select({1.0F, 0.0F}, 2, Banks(), {1.0F, 1.0F}, 2, &chosen);
  REQUIRE(chosen.size() == 2);
  CHECK(chosen[0] == 0);
  CHECK(chosen[1] == 1);
  // 1*(10,20) + 1*(3,4) = (13, 24)
  REQUIRE(out.size() == 2);
  CHECK(out[0] == 13.0F);
  CHECK(out[1] == 24.0F);
}

TEST_CASE("the weights scale each emotion's contribution") {
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, Banks(), {2.0F, 0.5F}, 2);
  // 2*(10,20) + 0.5*(3,4) = (21.5, 42)
  CHECK(out[0] == 21.5F);
  CHECK(out[1] == 42.0F);
}

TEST_CASE("a zero weight removes that emotion entirely") {
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, Banks(), {0.0F, 1.0F}, 2);
  CHECK(out[0] == 3.0F);
  CHECK(out[1] == 4.0F);
}

TEST_CASE("the match is COSINE, not Euclidean distance") {
  // A style of (5, 0) is far from (1, 0) in distance -- and further still from
  // (0, 1) -- but cosine sees only direction, so row 0 still wins. Scaling the
  // style must not change the selection at all.
  std::vector<int64_t> a;
  std::vector<int64_t> b;
  ev::Select({1.0F, 0.0F}, 2, Banks(), {1.0F, 1.0F}, 2, &a);
  ev::Select({5.0F, 0.0F}, 2, Banks(), {1.0F, 1.0F}, 2, &b);
  CHECK(a == b);

  // And a bank whose rows differ only in MAGNITUDE must be undecidable by
  // cosine, so the tie keeps the lower index.
  ev::EmotionBank same;
  same.rows = 2;
  same.speakers = {1.0F, 0.0F, 7.0F, 0.0F};  // same direction, different norms
  same.emotions = {1.0F, 1.0F, 9.0F, 9.0F};
  std::vector<int64_t> c;
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, {same}, {1.0F}, 2, &c);
  REQUIRE(c.size() == 1);
  CHECK(c[0] == 0);
  CHECK(out[0] == 1.0F);
}

TEST_CASE("a zero speaker row scores zero rather than NaN") {
  ev::EmotionBank z;
  z.rows = 2;
  z.speakers = {0.0F, 0.0F, 1.0F, 0.0F};
  z.emotions = {5.0F, 5.0F, 6.0F, 6.0F};
  std::vector<int64_t> chosen;
  const std::vector<float> out = ev::Select({1.0F, 0.0F}, 2, {z}, {1.0F}, 2, &chosen);
  CHECK(chosen[0] == 1);       // the real row wins over the degenerate one
  CHECK(out[0] == 6.0F);
}

TEST_CASE("mismatched shapes are refused") {
  auto banks = Banks();
  banks[0].emotions.pop_back();
  CHECK_THROWS(ev::Select({1.0F, 0.0F}, 2, banks, {1.0F, 1.0F}, 2));
  CHECK_THROWS(ev::Select({1.0F, 0.0F}, 2, Banks(), {1.0F}, 2));      // too few weights
  CHECK_THROWS(ev::Select({1.0F}, 2, Banks(), {1.0F, 1.0F}, 2));      // short style
}
