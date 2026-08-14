// S2Mel DiT block primitives (#634), gated against gpt-fast executed directly.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "dit_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit.h"

namespace {
using namespace dit_goldens;

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

TEST_CASE("dit RMSNorm does not subtract the mean") {
  std::vector<float> w = Rand("rms.weight", kDim, 0.5);
  for (float& v : w) v += 1.0F;
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> got = vllm::models::dit::RmsNorm(x, kFrames, kDim, w, 1e-5);
  REQUIRE(got.size() == x.size());
  CHECK(Worst(got, kRmsOut, got.size()) < 1e-5);
}

TEST_CASE("dit RMSNorm on a constant row keeps the sign and magnitude ratio") {
  // Structural, independent of the goldens: for a row of identical values v > 0,
  // rms(x) = v, so x / rms = 1 for every component. LayerNorm would give 0
  // because it subtracts the mean first -- the exact difference being pinned.
  const int64_t D = 4;
  const std::vector<float> x(static_cast<size_t>(D), 3.0F);
  const std::vector<float> w(static_cast<size_t>(D), 1.0F);
  const std::vector<float> got = vllm::models::dit::RmsNorm(x, 1, D, w, 0.0);
  for (const float v : got) CHECK(v == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("dit AdaptiveLayerNorm is weight * norm + bias, with NO implicit 1") {
  const std::vector<float> x(std::begin(kX), std::end(kX));
  const std::vector<float> emb(std::begin(kEmb), std::end(kEmb));
  std::vector<float> nw = Rand("aln.norm.weight", kDim, 0.5);
  for (float& v : nw) v += 1.0F;
  const std::vector<float> got = vllm::models::dit::AdaptiveLayerNorm(
      x, kFrames, kDim, emb, Rand("aln.project_layer.weight", 2 * kDim * kDim, 0.3),
      Rand("aln.project_layer.bias", 2 * kDim, 0.3), nw, 1e-5);
  REQUIRE(got.size() == x.size());
  const double worst = Worst(got, kAlnOut, got.size());
  INFO("max abs diff vs upstream AdaptiveLayerNorm: ", worst);
  CHECK(worst < 1e-5);
}

TEST_CASE("dit rotary pairs ADJACENT components, not halves") {
  const std::vector<float> q(std::begin(kQ), std::end(kQ));
  const std::vector<float> freqs(std::begin(kFreqs), std::end(kFreqs));
  const std::vector<float> got =
      vllm::models::dit::ApplyRotary(q, kFrames, kHeads, kHeadDim, freqs);
  REQUIRE(got.size() == q.size());
  // bf16 freqs upstream, so the table itself carries ~3 decimal digits; the
  // bound reflects that rather than pretending to f32 precision.
  const double worst = Worst(got, kRotOut, got.size());
  INFO("max abs diff vs upstream apply_rotary_emb: ", worst);
  CHECK(worst < 1e-5);
}

TEST_CASE("dit rotary preserves the norm of each ADJACENT pair") {
  // A rotation cannot change a pair's magnitude. Under the half-split
  // convention the same check would hold for the WRONG pairs, so this runs
  // alongside the value comparison rather than instead of it.
  const std::vector<float> q(std::begin(kQ), std::end(kQ));
  const std::vector<float> freqs(std::begin(kFreqs), std::end(kFreqs));
  const std::vector<float> got =
      vllm::models::dit::ApplyRotary(q, kFrames, kHeads, kHeadDim, freqs);
  for (int64_t t = 0; t < kFrames; ++t) {
    for (int64_t h = 0; h < kHeads; ++h) {
      for (int64_t p = 0; p < kHeadDim / 2; ++p) {
        const size_t lo = static_cast<size_t>((t * kHeads + h) * kHeadDim + 2 * p);
        const double before = std::hypot(q[lo], q[lo + 1]);
        const double after = std::hypot(got[lo], got[lo + 1]);
        CHECK(after == doctest::Approx(before).epsilon(1e-3));
      }
    }
  }
}

TEST_CASE("dit SwiGLU gates on W1, not W3") {
  // w2(silu(w1 x) * w3 x). Swapping w1 and w3 gives an identically shaped
  // network that still trains; the checkpoint stores them separately, so only
  // the values distinguish them.
  const std::vector<float> x(std::begin(kFfIn), std::end(kFfIn));
  const std::vector<float> got = vllm::models::dit::SwiGlu(
      x, kFrames, kDim, kInter, Rand("ff.w1.weight", kInter * kDim, 0.3),
      Rand("ff.w3.weight", kInter * kDim, 0.3), Rand("ff.w2.weight", kDim * kInter, 0.3));
  REQUIRE(got.size() == static_cast<size_t>(kFrames * kDim));
  CHECK(Worst(got, kFfOut, got.size()) < 1e-5);
}

TEST_CASE("dit TransformerBlock reproduces the whole upstream block") {
  std::map<std::string, std::vector<float>> m;
  for (int64_t i = 0; i < kBlockManifestSize; ++i) {
    const auto& e = kBlockManifest[i];
    const int64_t d[2] = {e.d0, e.d1};
    int64_t n = 1;
    for (int64_t k = 0; k < e.rank; ++k) n *= d[k];
    m[e.name] = Rand(std::string("blk.") + e.name, n, 0.3);
  }
  vllm::models::dit::BlockWeights w;
  w.wqkv = m.at("attention.wqkv.weight");
  w.wo = m.at("attention.wo.weight");
  w.w1 = m.at("feed_forward.w1.weight");
  w.w3 = m.at("feed_forward.w3.weight");
  w.w2 = m.at("feed_forward.w2.weight");
  w.attn_proj_w = m.at("attention_norm.project_layer.weight");
  w.attn_proj_b = m.at("attention_norm.project_layer.bias");
  w.attn_norm_w = m.at("attention_norm.norm.weight");
  w.ffn_proj_w = m.at("ffn_norm.project_layer.weight");
  w.ffn_proj_b = m.at("ffn_norm.project_layer.bias");
  w.ffn_norm_w = m.at("ffn_norm.norm.weight");

  const std::vector<float> x(std::begin(kBx), std::end(kBx));
  const std::vector<float> c(std::begin(kBc), std::end(kBc));
  const std::vector<float> freqs(std::begin(kFreqs), std::end(kFreqs));
  const std::vector<float> got = vllm::models::dit::Block(
      x, c, kFrames, kDim, kHeads, kHeadDim, kInter, freqs, w, 1e-5);
  REQUIRE(got.size() == x.size());
  const double worst = Worst(got, kBlockOut, got.size());
  INFO("max abs diff vs upstream TransformerBlock: ", worst);
  CHECK(worst < 5e-5);
}
