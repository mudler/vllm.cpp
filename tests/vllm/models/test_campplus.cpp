// CAMPPlus primitives — W3 of #634, the speaker-style encoder on the MANDATORY
// reference-audio path (the talker is built with spk_cond_mode="campplus", so
// this feeds it).
//
// Gated against `indextts/s2mel/modules/campplus/layers.py` executed DIRECTLY:
// it has no vllm dependency, so the goldens come from the real classes rather
// than a restatement. Weights are rebuilt both sides from one FNV-1a ->
// splitmix64 stream.
//
// THREE THINGS HERE FAIL SILENTLY, so each has its own case:
//   1. std is UNBIASED (N-1). The biased form differs by ~0.2% at T=250 — small
//      enough to look like noise, large enough to move a style vector.
//   2. BatchNorm1d in EVAL uses RUNNING statistics. Using batch statistics still
//      normalizes, and is a different model.
//   3. seg_pooling uses ceil_mode, expands each segment back over seg_len frames
//      and TRUNCATES to the input length. T=250 gives 3 segments, the last one
//      partial, which is where an off-by-one lives.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "campplus_goldens.inc"
#include "doctest/doctest.h"
#include "vllm/model_executor/models/campplus.h"

namespace {
using namespace campplus_goldens;

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
double Worst(const std::vector<float>& got, const float* want, size_t n) {
  double w = 0.0;
  for (size_t i = 0; i < n; ++i) w = std::max(w, std::fabs(static_cast<double>(got[i]) - want[i]));
  return w;
}
}  // namespace

TEST_CASE("campplus StatsPool concatenates mean and UNBIASED std") {
  const std::vector<float> x = Rand("x", kChannels * kFrames, 1.0);
  const std::vector<float> got = vllm::models::campplus::StatsPool(x, kChannels, kFrames);
  REQUIRE(got.size() == static_cast<size_t>(2 * kChannels));
  const double w = Worst(got, kStats, got.size());
  INFO("max abs diff vs upstream StatsPool: ", w);
  CHECK(w < 1e-5);
}

TEST_CASE("campplus BatchNorm1d in eval uses RUNNING statistics") {
  const std::vector<float> x = Rand("x", kChannels * kFrames, 1.0);
  std::vector<float> gamma = Rand("bn.w", kChannels, 0.5);
  for (float& g : gamma) g += 1.0F;
  const std::vector<float> beta = Rand("bn.b", kChannels, 0.3);
  const std::vector<float> mean = Rand("bn.rm", kChannels, 0.4);
  std::vector<float> var = Rand("bn.rv", kChannels, 0.2);
  for (float& v : var) v = std::fabs(v) + 0.5F;

  const std::vector<float> got =
      vllm::models::campplus::BatchNorm1dEval(x, kChannels, kFrames, gamma, beta, mean, var, 1e-5);
  REQUIRE(got.size() == x.size());
  const double w = Worst(got, kBatchNormEval, got.size());
  INFO("max abs diff vs upstream BatchNorm1d(eval): ", w);
  CHECK(w < 1e-5);
}

TEST_CASE("campplus seg_pooling expands ceil-mode segments and truncates") {
  const std::vector<float> x = Rand("xb", kBnChannels * kFrames, 1.0);
  const std::vector<float> got =
      vllm::models::campplus::SegPooling(x, kBnChannels, kFrames, kSegLen);
  // The expansion must return the INPUT length, not a multiple of seg_len.
  REQUIRE(got.size() == x.size());
  const double w = Worst(got, kSegPooling, got.size());
  INFO("max abs diff vs upstream seg_pooling: ", w);
  CHECK(w < 1e-5);
}

TEST_CASE("campplus CAMLayer gates the local branch by the pooled context") {
  const std::vector<float> x = Rand("xb", kBnChannels * kFrames, 1.0);
  vllm::models::campplus::CamLayerWeights w;
  w.linear_local = Rand("cam.ll.w", kOutChannels * kBnChannels * kKernel, 0.3);
  w.linear1_weight = Rand("cam.l1.w", (kBnChannels / 2) * kBnChannels, 0.3);
  w.linear1_bias = Rand("cam.l1.b", kBnChannels / 2, 0.2);
  w.linear2_weight = Rand("cam.l2.w", kOutChannels * (kBnChannels / 2), 0.3);
  w.linear2_bias = Rand("cam.l2.b", kOutChannels, 0.2);

  const std::vector<float> got = vllm::models::campplus::CamLayer(
      x, kBnChannels, kFrames, kOutChannels, kKernel, kDilation, kSegLen, w);
  REQUIRE(got.size() == static_cast<size_t>(kOutChannels * kFrames));
  const double d = Worst(got, kCamOut, got.size());
  INFO("max abs diff vs upstream CAMLayer: ", d);
  CHECK(d < 1e-5);
}

namespace {
// Fill a layer's parameters from the generator's stream. Names mirror torch's
// `named_parameters()` / `named_buffers()` exactly, so a mismatch here shows up
// as a tensor difference rather than as silently different weights.
vllm::models::campplus::DenseTdnnLayerWeights LayerW(const std::string& p, int64_t in_ch,
                                                     int64_t bn, int64_t growth, int64_t k) {
  vllm::models::campplus::DenseTdnnLayerWeights w;
  w.bn1_gamma = Rand(p + ".nonlinear1.batchnorm.weight", in_ch, 0.3);
  w.bn1_beta = Rand(p + ".nonlinear1.batchnorm.bias", in_ch, 0.3);
  w.bn1_mean = Rand(p + ".nonlinear1.batchnorm.running_mean", in_ch, 0.4);
  w.bn1_var = Rand(p + ".nonlinear1.batchnorm.running_var", in_ch, 0.2);
  for (float& v : w.bn1_var) v = std::fabs(v) + 0.5F;
  w.linear1 = Rand(p + ".linear1.weight", bn * in_ch, 0.3);
  w.bn2_gamma = Rand(p + ".nonlinear2.batchnorm.weight", bn, 0.3);
  w.bn2_beta = Rand(p + ".nonlinear2.batchnorm.bias", bn, 0.3);
  w.bn2_mean = Rand(p + ".nonlinear2.batchnorm.running_mean", bn, 0.4);
  w.bn2_var = Rand(p + ".nonlinear2.batchnorm.running_var", bn, 0.2);
  for (float& v : w.bn2_var) v = std::fabs(v) + 0.5F;
  w.cam.linear_local = Rand(p + ".cam_layer.linear_local.weight", growth * bn * k, 0.3);
  w.cam.linear1_weight = Rand(p + ".cam_layer.linear1.weight", (bn / 2) * bn, 0.3);
  w.cam.linear1_bias = Rand(p + ".cam_layer.linear1.bias", bn / 2, 0.3);
  w.cam.linear2_weight = Rand(p + ".cam_layer.linear2.weight", growth * (bn / 2), 0.3);
  w.cam.linear2_bias = Rand(p + ".cam_layer.linear2.bias", growth, 0.3);
  return w;
}
}  // namespace

TEST_CASE("campplus TransitLayer applies the nonlinear BEFORE the projection") {
  const std::vector<float> x = Rand("xt", kIn * kFrames, 1.0);
  std::vector<float> var = Rand("transit.nonlinear.batchnorm.running_var", kIn, 0.2);
  for (float& v : var) v = std::fabs(v) + 0.5F;
  const std::vector<float> got = vllm::models::campplus::TransitLayer(
      x, kIn, kFrames, kIn / 2, Rand("transit.nonlinear.batchnorm.weight", kIn, 0.3),
      Rand("transit.nonlinear.batchnorm.bias", kIn, 0.3),
      Rand("transit.nonlinear.batchnorm.running_mean", kIn, 0.4), var,
      Rand("transit.linear.weight", (kIn / 2) * kIn, 0.3),
      Rand("transit.linear.bias", kIn / 2, 0.3), 1e-5);
  REQUIRE(got.size() == static_cast<size_t>((kIn / 2) * kFrames));
  CHECK(Worst(got, kTransit, got.size()) < 1e-5);
}

TEST_CASE("campplus DenseLayer projects a POOLED stats vector as T=1") {
  // The final dense consumes StatsPool's [2C] output, which upstream reaches by
  // unsqueeze -> conv -> squeeze. Treating it as anything other than T=1 changes
  // the shape rather than the numbers, so the length is asserted too.
  const std::vector<float> stats = vllm::models::campplus::StatsPool(
      Rand("x", kChannels * kFrames, 1.0), kChannels, kFrames);
  std::vector<float> var = Rand("dense2d.nonlinear.batchnorm.running_var", 12, 0.2);
  for (float& v : var) v = std::fabs(v) + 0.5F;
  const std::vector<float> got = vllm::models::campplus::DenseLayer(
      stats, 2 * kChannels, /*frames=*/1, 12, Rand("dense2d.linear.weight", 12 * 2 * kChannels, 0.3),
      {}, {}, {}, Rand("dense2d.nonlinear.batchnorm.running_mean", 12, 0.4), var, 1e-5,
      /*apply_relu=*/false);  // config_str="batchnorm_" => batchnorm only
  REQUIRE(got.size() == 12U);
  CHECK(Worst(got, kDense2d, got.size()) < 1e-5);
}

TEST_CASE("campplus CAMDenseTDNNLayer chains the two nonlinears around linear1") {
  const std::vector<float> x = Rand("xt", kIn * kFrames, 1.0);
  const std::vector<float> got = vllm::models::campplus::DenseTdnnLayer(
      x, kIn, kFrames, kBnChannels, kGrowth, kKernel, kDilation, kSegLen,
      LayerW("dl", kIn, kBnChannels, kGrowth, kKernel), 1e-5);
  REQUIRE(got.size() == static_cast<size_t>(kGrowth * kFrames));
  CHECK(Worst(got, kDenseTdnnLayer, got.size()) < 1e-5);
}

TEST_CASE("campplus CAMDenseTDNNBlock GROWS its channel count by concatenation") {
  // x = cat([x, layer(x)], dim=1): each layer sees every earlier output, and the
  // width grows by `growth` per layer. Appending in the wrong order still yields
  // a correctly shaped tensor, so the values are what catch it.
  const std::vector<float> x = Rand("xt", kIn * kFrames, 1.0);
  std::vector<vllm::models::campplus::DenseTdnnLayerWeights> layers;
  for (int64_t i = 0; i < kNumLayers; ++i) {
    layers.push_back(LayerW("blk.tdnnd" + std::to_string(i + 1), kIn + i * kGrowth, kBnChannels,
                            kGrowth, kKernel));
  }
  const std::vector<float> got = vllm::models::campplus::DenseTdnnBlock(
      x, kIn, kFrames, kBnChannels, kGrowth, kKernel, kDilation, kSegLen, layers, 1e-5);
  REQUIRE(got.size() == static_cast<size_t>((kIn + kNumLayers * kGrowth) * kFrames));
  CHECK(Worst(got, kDenseTdnnBlock, got.size()) < 1e-5);
}

TEST_CASE("campplus BasicResBlock strides FREQUENCY only, never time") {
  // stride=(stride, 1) in layers.py:224. Striding both axes still yields a
  // well-formed tensor -- at half the frame rate -- which every later layer
  // accepts, so the WIDTH is asserted unchanged alongside the values.
  const std::vector<float> x = Rand("x4", kFcmIn * kFcmH * kFcmW, 1.0);
  vllm::models::campplus::ResBlock2dWeights w;
  auto var = [](std::vector<float> v) { for (float& f : v) f = std::fabs(f) + 0.5F; return v; };
  w.conv1 = Rand("rb.conv1.weight", kFcmPlanes * kFcmIn * 9, 0.3);
  w.bn1_gamma = Rand("rb.bn1.weight", kFcmPlanes, 0.3);
  w.bn1_beta = Rand("rb.bn1.bias", kFcmPlanes, 0.3);
  w.bn1_mean = Rand("rb.bn1.running_mean", kFcmPlanes, 0.4);
  w.bn1_var = var(Rand("rb.bn1.running_var", kFcmPlanes, 0.2));
  w.conv2 = Rand("rb.conv2.weight", kFcmPlanes * kFcmPlanes * 9, 0.3);
  w.bn2_gamma = Rand("rb.bn2.weight", kFcmPlanes, 0.3);
  w.bn2_beta = Rand("rb.bn2.bias", kFcmPlanes, 0.3);
  w.bn2_mean = Rand("rb.bn2.running_mean", kFcmPlanes, 0.4);
  w.bn2_var = var(Rand("rb.bn2.running_var", kFcmPlanes, 0.2));
  w.has_shortcut = true;  // stride != 1 and in_planes != planes
  w.short_conv = Rand("rb.shortcut.0.weight", kFcmPlanes * kFcmIn, 0.3);
  w.short_gamma = Rand("rb.shortcut.1.weight", kFcmPlanes, 0.3);
  w.short_beta = Rand("rb.shortcut.1.bias", kFcmPlanes, 0.3);
  w.short_mean = Rand("rb.shortcut.1.running_mean", kFcmPlanes, 0.4);
  w.short_var = var(Rand("rb.shortcut.1.running_var", kFcmPlanes, 0.2));

  int64_t out_h = 0;
  const std::vector<float> got = vllm::models::campplus::ResBlock2d(
      x, kFcmIn, kFcmH, kFcmW, kFcmPlanes, /*stride=*/2, w, 1e-5, &out_h);
  CHECK(out_h == kFcmOutH);                       // 16 -> 8: frequency halved
  REQUIRE(got.size() == static_cast<size_t>(kFcmPlanes * kFcmOutH * kFcmOutW));
  CHECK(Worst(got, kResBlock, got.size()) < 1e-5);
}

TEST_CASE("campplus forward reproduces the WHOLE upstream encoder") {
  // The manifest is the contract: every tensor is rebuilt from its upstream
  // state_dict NAME by the same rule the generator used, so a tensor one side
  // builds and the other does not is a lookup failure by name, not a silent zero.
  vllm::models::campplus::CampplusWeights w;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    int64_t n = 1;
    const int64_t dims[4] = {e.d0, e.d1, e.d2, e.d3};
    for (int64_t d = 0; d < e.rank; ++d) n *= dims[d];
    if (e.rank == 0) n = 1;
    std::vector<float> v = Rand(e.name, n, 0.3);
    const std::string name(e.name);
    if (name.size() > 11 && name.compare(name.size() - 11, 11, "running_var") == 0) {
      for (float& f : v) f = std::fabs(f) + 0.5F;
    }
    w.t[name] = std::move(v);
  }
  REQUIRE(static_cast<int64_t>(w.t.size()) == kManifestSize);

  vllm::models::campplus::CampplusParams p;
  p.feat_dim = kFeatDim;
  p.embedding_size = kEmbedding;
  p.growth_rate = kGrowth2;
  p.bn_size = kBnSize;
  p.init_channels = kInitChannels;

  const std::vector<float> feats = Rand("feats", kFullFrames * kFeatDim, 1.0);
  const std::vector<float> got =
      vllm::models::campplus::Forward(p, w, feats, kFullFrames);
  REQUIRE(got.size() == static_cast<size_t>(kEmbedding));
  const double worst = Worst(got, kFullEmbedding, got.size());
  INFO("max abs diff vs upstream CAMPPlus.forward: ", worst);
  CHECK(worst < 5e-5);
}

TEST_CASE("campplus forward throws BY NAME on a missing tensor") {
  vllm::models::campplus::CampplusWeights w;
  vllm::models::campplus::CampplusParams p;
  p.feat_dim = kFeatDim;  // else the shape check fires before any name lookup
  const std::vector<float> feats = Rand("feats", kFullFrames * kFeatDim, 1.0);
  CHECK_THROWS_WITH_AS(vllm::models::campplus::Forward(p, w, feats, kFullFrames),
                       doctest::Contains("head.conv1.weight"), std::runtime_error);
}

TEST_CASE("campplus TDNN head geometry is gated where pooling cannot hide it") {
  // A wrong stride/dilation/padding in xvector.tdnn changes the FRAME COUNT.
  // StatsPool averages over time, so the final embedding absorbs it and still
  // matches within tolerance -- discovered by mutation, which is why this case
  // exists. The intermediate is gated directly.
  vllm::models::campplus::CampplusWeights w;
  for (int64_t i = 0; i < kManifestSize; ++i) {
    const auto& e = kManifest[i];
    int64_t n = 1;
    const int64_t dims[4] = {e.d0, e.d1, e.d2, e.d3};
    for (int64_t d = 0; d < e.rank; ++d) n *= dims[d];
    std::vector<float> v = Rand(e.name, n, 0.3);
    const std::string name(e.name);
    if (name.size() > 11 && name.compare(name.size() - 11, 11, "running_var") == 0) {
      for (float& f : v) f = std::fabs(f) + 0.5F;
    }
    w.t[name] = std::move(v);
  }
  vllm::models::campplus::CampplusParams p;
  p.feat_dim = kFeatDim; p.embedding_size = kEmbedding; p.growth_rate = kGrowth2;
  p.bn_size = kBnSize; p.init_channels = kInitChannels;

  vllm::models::campplus::ForwardTrace trace;
  const std::vector<float> feats = Rand("feats", kFullFrames * kFeatDim, 1.0);
  vllm::models::campplus::Forward(p, w, feats, kFullFrames, &trace);

  CHECK(trace.tdnn_channels == kTdnnChannels);
  CHECK(trace.tdnn_frames == kTdnnFrames);   // 20, not 19: padding is (5-1)//2
  REQUIRE(trace.tdnn.size() == static_cast<size_t>(kTdnnChannels * kTdnnFrames));
  CHECK(Worst(trace.tdnn, kTdnnOut, trace.tdnn.size()) < 1e-5);
}
