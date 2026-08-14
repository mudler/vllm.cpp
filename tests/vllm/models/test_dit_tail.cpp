// The S2Mel DiT tail against upstream goldens. See dit_tail.h.
#include <cstdint>
#include <string>
#include <vector>

#include "dit_tail_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/dit_tail.h"

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

using namespace dit_tail_goldens;

// TimestepEmbedder's default frequency_embedding_size upstream.
constexpr int64_t kFreqSize = 256;

vllm::models::dit_tail::Linear Lin(const std::string& prefix, int64_t out_dim,
                                   int64_t in_dim) {
  vllm::models::dit_tail::Linear l;
  l.weight = Rnd(prefix + ".weight", static_cast<size_t>(out_dim * in_dim), 0.5);
  l.bias = Rnd(prefix + ".bias", static_cast<size_t>(out_dim), 0.5);
  return l;
}

vllm::models::wavenet::ConvWeights Conv(const std::string& prefix, int64_t out_ch,
                                        int64_t in_ch, int64_t kernel) {
  vllm::models::wavenet::ConvWeights c;
  c.g = Rnd(prefix + ".weight_g", static_cast<size_t>(out_ch), 0.5);
  c.v = Rnd(prefix + ".weight_v", static_cast<size_t>(out_ch * in_ch * kernel), 0.5);
  c.bias = Rnd(prefix + ".bias", static_cast<size_t>(out_ch), 0.5);
  return c;
}

vllm::models::dit_tail::Config Cfg() {
  vllm::models::dit_tail::Config c;
  c.hidden = kHidden;
  c.wn_hidden = kWnHidden;
  c.in_channels = kInChannels;
  c.frames = kFrames;
  c.freq_size = kFreqSize;
  c.wn.hidden = kWnHidden;
  c.wn.kernel = kWnKernel;
  c.wn.dilation_rate = kWnDilation;
  c.wn.layers = kWnLayers;
  c.wn.gin = kWnHidden;  // the wavenet is conditioned by t2, at its own width
  return c;
}

vllm::models::dit_tail::Weights BuildWeights() {
  vllm::models::dit_tail::Weights w;
  w.skip_linear = Lin("tail.skip_linear", kHidden, kHidden + kInChannels);
  w.conv1 = Lin("tail.conv1", kWnHidden, kHidden);
  w.res_projection = Lin("tail.res_projection", kWnHidden, kHidden);
  // conv2 is a Conv1d with kernel 1, so torch stores [out, in, 1].
  w.conv2 = Lin("tail.conv2", kInChannels, kWnHidden);
  w.t_embedder2_mlp0 = Lin("tail.t_embedder2.mlp.0", kWnHidden, kFreqSize);
  w.t_embedder2_mlp2 = Lin("tail.t_embedder2.mlp.2", kWnHidden, kWnHidden);

  w.wn.cond = Conv("tail.wavenet.cond_layer.conv.conv", 2 * kWnHidden * kWnLayers,
                   kWnHidden, 1);
  for (int64_t i = 0; i < kWnLayers; ++i) {
    const std::string idx = std::to_string(i);
    w.wn.in_layers.push_back(Conv("tail.wavenet.in_layers." + idx + ".conv.conv",
                                  2 * kWnHidden, kWnHidden, kWnKernel));
    const int64_t rs = (i < kWnLayers - 1) ? 2 * kWnHidden : kWnHidden;
    w.wn.res_skip_layers.push_back(
        Conv("tail.wavenet.res_skip_layers." + idx + ".conv.conv", rs, kWnHidden, 1));
  }

  w.final_layer.ada_w =
      Rnd("tail.final_layer.adaLN_modulation.1.weight",
          static_cast<size_t>(2 * kWnHidden * kWnHidden), 0.5);
  w.final_layer.ada_b = Rnd("tail.final_layer.adaLN_modulation.1.bias",
                            static_cast<size_t>(2 * kWnHidden), 0.5);
  w.final_layer.linear_g =
      Rnd("tail.final_layer.linear.weight_g", static_cast<size_t>(kWnHidden), 0.5);
  w.final_layer.linear_v = Rnd("tail.final_layer.linear.weight_v",
                               static_cast<size_t>(kWnHidden * kWnHidden), 0.5);
  w.final_layer.linear_bias =
      Rnd("tail.final_layer.linear.bias", static_cast<size_t>(kWnHidden), 0.5);
  return w;
}

}  // namespace

TEST_CASE("the DiT tail matches upstream end to end") {
  const std::vector<float> x_res =
      Rnd("tail.x_res", static_cast<size_t>(kFrames * kHidden), 1.0);
  const std::vector<float> x =
      Rnd("tail.x", static_cast<size_t>(kFrames * kInChannels), 1.0);
  const std::vector<float> t1 = Rnd("tail.t1", static_cast<size_t>(kHidden), 1.0);

  const std::vector<float> got = vllm::models::dit_tail::Forward(
      Cfg(), BuildWeights(), x_res, x, kT, t1, {});

  REQUIRE(got.size() == static_cast<size_t>(kInChannels * kFrames));
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(kOut[i]).epsilon(2e-5));
  }
}

TEST_CASE("the long skip actually reads x, not just the transformer output") {
  // Perturbing ONLY `x` must move the output. Without the concat the tail is a
  // function of x_res alone, which still produces a mel of the right shape.
  const std::vector<float> x_res =
      Rnd("tail.x_res", static_cast<size_t>(kFrames * kHidden), 1.0);
  std::vector<float> x = Rnd("tail.x", static_cast<size_t>(kFrames * kInChannels), 1.0);
  const std::vector<float> t1 = Rnd("tail.t1", static_cast<size_t>(kHidden), 1.0);
  const auto w = BuildWeights();

  const std::vector<float> base =
      vllm::models::dit_tail::Forward(Cfg(), w, x_res, x, kT, t1, {});
  x[0] += 1.0F;
  const std::vector<float> moved =
      vllm::models::dit_tail::Forward(Cfg(), w, x_res, x, kT, t1, {});

  bool differs = false;
  for (size_t i = 0; i < base.size(); ++i) {
    if (base[i] != moved[i]) {
      differs = true;
    }
  }
  CHECK(differs);
}

TEST_CASE("t1 and t2 are DIFFERENT conditioning vectors") {
  // `t1` conditions final_layer and `t2` conditions the wavenet. If the port
  // fed one to both, changing t1 alone would still move the output, so the
  // discriminating check is that t1 moves it while the raw t is held fixed.
  const std::vector<float> x_res =
      Rnd("tail.x_res", static_cast<size_t>(kFrames * kHidden), 1.0);
  const std::vector<float> x =
      Rnd("tail.x", static_cast<size_t>(kFrames * kInChannels), 1.0);
  std::vector<float> t1 = Rnd("tail.t1", static_cast<size_t>(kHidden), 1.0);
  const auto w = BuildWeights();

  const std::vector<float> base =
      vllm::models::dit_tail::Forward(Cfg(), w, x_res, x, kT, t1, {});
  for (float& v : t1) {
    v += 0.25F;
  }
  const std::vector<float> moved =
      vllm::models::dit_tail::Forward(Cfg(), w, x_res, x, kT, t1, {});

  bool differs = false;
  for (size_t i = 0; i < base.size(); ++i) {
    if (base[i] != moved[i]) {
      differs = true;
    }
  }
  CHECK(differs);

  // And the raw timestep must matter on its own, through t_embedder2.
  const std::vector<float> other_t = vllm::models::dit_tail::Forward(
      Cfg(), w, x_res, x, kT + 0.1F, Rnd("tail.t1", static_cast<size_t>(kHidden), 1.0), {});
  bool t_matters = false;
  for (size_t i = 0; i < base.size(); ++i) {
    if (base[i] != other_t[i]) {
      t_matters = true;
    }
  }
  CHECK(t_matters);
}

TEST_CASE("unequal DiT and wavenet widths are REFUSED, not silently composed") {
  auto cfg = Cfg();
  cfg.wn_hidden = kWnHidden + 1;
  const std::vector<float> x_res =
      Rnd("tail.x_res", static_cast<size_t>(kFrames * kHidden), 1.0);
  const std::vector<float> x =
      Rnd("tail.x", static_cast<size_t>(kFrames * kInChannels), 1.0);
  const std::vector<float> t1 = Rnd("tail.t1", static_cast<size_t>(kHidden), 1.0);
  CHECK_THROWS(vllm::models::dit_tail::Forward(cfg, BuildWeights(), x_res, x, kT, t1, {}));
}
