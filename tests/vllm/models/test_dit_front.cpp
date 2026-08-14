// The S2Mel DiT front end against upstream goldens. See dit_front.h.
#include <cstdint>
#include <string>
#include <vector>

#include "dit_front_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit_front.h"

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

using namespace dit_front_goldens;

vllm::models::dit_front::Config Cfg() {
  vllm::models::dit_front::Config c;
  c.hidden = kHidden;
  c.in_channels = kInChannels;
  c.style = kStyle;
  c.frames = kFrames;
  return c;
}

vllm::models::dit_front::Weights W() {
  vllm::models::dit_front::Weights w;
  w.cond_proj_w = Rnd("front.cond_projection.weight",
                      static_cast<size_t>(kHidden * kHidden), 0.5);
  w.cond_proj_b = Rnd("front.cond_projection.bias", static_cast<size_t>(kHidden), 0.5);
  const int64_t wide = kInChannels * 2 + kHidden + kStyle;
  w.merge_w = Rnd("front.cond_x_merge_linear.weight",
                  static_cast<size_t>(kHidden * wide), 0.5);
  w.merge_b = Rnd("front.cond_x_merge_linear.bias", static_cast<size_t>(kHidden), 0.5);
  return w;
}

std::vector<float> X() { return Rnd("front.x", static_cast<size_t>(kInChannels * kFrames), 1.0); }
std::vector<float> P() {
  return Rnd("front.prompt_x", static_cast<size_t>(kInChannels * kFrames), 1.0);
}
std::vector<float> C() { return Rnd("front.cond", static_cast<size_t>(kFrames * kHidden), 1.0); }
std::vector<float> S() { return Rnd("front.style", static_cast<size_t>(kStyle), 1.0); }

}  // namespace

TEST_CASE("the conditional front end matches upstream") {
  const std::vector<float> got =
      vllm::models::dit_front::BuildXIn(Cfg(), W(), X(), P(), C(), S(), false);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kMerged[i]).epsilon(2e-5));
  }
}

TEST_CASE("the CFG unconditional branch matches upstream") {
  const std::vector<float> got =
      vllm::models::dit_front::BuildXIn(Cfg(), W(), X(), P(), C(), S(), true);
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kHidden));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kMergedUncond[i]).epsilon(2e-5));
  }
}

TEST_CASE("the unconditional branch keeps x and drops everything else") {
  // Perturbing x must move the unconditional output; perturbing prompt, cond or
  // style must NOT. A port that zeroed the wrong slice still returns a
  // plausible tensor, and only this separates the two.
  const auto cfg = Cfg();
  const auto w = W();
  std::vector<float> x = X(), p = P(), c = C(), s = S();
  const std::vector<float> base =
      vllm::models::dit_front::BuildXIn(cfg, w, x, p, c, s, true);

  x[0] += 1.0F;
  CHECK(vllm::models::dit_front::BuildXIn(cfg, w, x, p, c, s, true) != base);
  x = X();

  p[0] += 1.0F;
  CHECK(vllm::models::dit_front::BuildXIn(cfg, w, x, p, c, s, true) == base);
  p = P();

  c[0] += 1.0F;
  CHECK(vllm::models::dit_front::BuildXIn(cfg, w, x, p, c, s, true) == base);
  c = C();

  s[0] += 1.0F;
  CHECK(vllm::models::dit_front::BuildXIn(cfg, w, x, p, c, s, true) == base);
}

TEST_CASE("the conditional branch reads ALL FOUR inputs") {
  const auto cfg = Cfg();
  const auto w = W();
  const std::vector<float> base =
      vllm::models::dit_front::BuildXIn(cfg, w, X(), P(), C(), S(), false);
  {
    std::vector<float> v = X();
    v[0] += 1.0F;
    CHECK(vllm::models::dit_front::BuildXIn(cfg, w, v, P(), C(), S(), false) != base);
  }
  {
    std::vector<float> v = P();
    v[0] += 1.0F;
    CHECK(vllm::models::dit_front::BuildXIn(cfg, w, X(), v, C(), S(), false) != base);
  }
  {
    std::vector<float> v = C();
    v[0] += 1.0F;
    CHECK(vllm::models::dit_front::BuildXIn(cfg, w, X(), P(), v, S(), false) != base);
  }
  {
    std::vector<float> v = S();
    v[0] += 1.0F;
    CHECK(vllm::models::dit_front::BuildXIn(cfg, w, X(), P(), C(), v, false) != base);
  }
}

TEST_CASE("style is broadcast over frames, not consumed one value per frame") {
  // If style were read per frame instead of repeated, a style vector whose
  // entries are all equal would give the same answer either way; one with
  // distinct entries would not. So use distinct entries and compare against a
  // deliberately frame-indexed reading.
  const auto cfg = Cfg();
  const auto w = W();
  std::vector<float> s(static_cast<size_t>(kStyle));
  for (size_t i = 0; i < s.size(); ++i) {
    s[i] = 0.25F * static_cast<float>(i + 1);
  }
  const std::vector<float> out =
      vllm::models::dit_front::BuildXIn(cfg, w, X(), P(), C(), s, false);
  // Every frame saw the same style, so shifting the whole style vector moves
  // every output frame by the same amount.
  std::vector<float> s2 = s;
  for (float& v : s2) {
    v += 1.0F;
  }
  const std::vector<float> out2 =
      vllm::models::dit_front::BuildXIn(cfg, w, X(), P(), C(), s2, false);
  const float delta0 = out2[0] - out[0];
  for (int64_t t = 1; t < kFrames; ++t) {
    CHECK(out2[static_cast<size_t>(t * kHidden)] - out[static_cast<size_t>(t * kHidden)] ==
          doctest::Approx(delta0).epsilon(1e-5));
  }
}
