// LTX-2.5 VAE parity gate — the audio decoder, its vocoder (both resblock arms
// plus the BWE chain), and the Conv video decoder, each compared against the
// UPSTREAM `ltx_core` module executed at reduced dimensions on CPU by
// scripts/gen-ltx2-vae-goldens.py.
//
// Both sides rebuild every weight and input from ONE deterministic stream, so no
// weight byte is checked in. Beyond the tensor comparison each brick also asserts
// its PARAMETER MANIFEST — name and element count, in state_dict order — against
// the generator's, so a parameter one side builds and the other does not is a
// failure rather than a silent no-op.
//
// Tolerances use `.scale(0.0)` wherever doctest::Approx appears: Approx's default
// scale of 1.0 puts a 1.19e-5 ABSOLUTE floor under any epsilon, which would make
// a tight relative tolerance silently accept anything.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "doctest/doctest.h"
#include "support/max_abs_diff.h"
// LTX25-DECODE-THREADS (issue #1009): Threadpool::SwapForTesting and the pool's
// work-stealing cursor, reached via -I src the way every other threading A/B in
// this tree does (tests/vt/test_ops_conv2d.cpp:25).
#include "vt/cpu/cpu_threadpool.h"
#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_audio_vae_encoder.h"
#include "vllm/model_executor/models/ltx2_conditioning.h"
// The accumulator-width case enters the decode through the PRODUCTION streaming
// entry point rather than through Ltx2ConvVideoDecode (issue #1008).
#include "vllm/model_executor/models/ltx2_tiling.h"
#include "vllm/model_executor/models/ltx2_video_vae.h"
#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"
// vocoder1d::kSnakeEps: the Snake/SnakeBeta stabilizer is SHARED with MiniMax-H3's
// BigVGAN, so the constant this suite pins lives in that header.
#include "vllm/model_executor/models/minimax_h3.h"
#include "vllm/model_executor/models/vocoder1d.h"

#include "ltx2_vae_goldens.inc"

namespace {

// ---------------------------------------------------------------------------
// Ltx2Rand — the exact mirror of the generator's stream
// (scripts/gen-ltx2-vae-goldens.py :: ltx_rand): a per-tensor FNV-1a seed plus a
// splitmix64 counter, so both sides build identical tensors from a NAME alone and
// cannot drift by reordering their parameter construction.
// ---------------------------------------------------------------------------

uint64_t Fnv1a64(const std::string& name) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char byte : name) {
    h ^= static_cast<uint64_t>(byte);
    h *= 0x100000001B3ULL;
  }
  return h;
}

uint64_t Splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::vector<double> Ltx2Rand(const std::string& name, int64_t count) {
  const uint64_t seed = Fnv1a64(name);
  std::vector<double> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    const uint64_t u = Splitmix64(seed + static_cast<uint64_t>(i));
    out[static_cast<size_t>(i)] = (static_cast<double>(u >> 11) * 0x1p-53) * 2.0 - 1.0;
  }
  return out;
}

// A raw input tensor: ltx_rand * scale (the generator's `make_input`).
std::vector<float> Ltx2Input(const std::string& name, int64_t count, double scale) {
  const std::vector<double> raw = Ltx2Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<float>(raw[static_cast<size_t>(i)] * scale);
  }
  return out;
}

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The generator's `param_values` rule, mirrored EXACTLY. `rank` is the upstream
// tensor's rank, which is why the bag builder passes shapes and not counts.
std::vector<float> Ltx2Param(const std::string& name, const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t dim : shape) count *= dim;
  const size_t rank = shape.size();

  double scale = 0.1;
  double offset = 0.0;
  double attenuate = 1.0;
  bool absolute = false;
  if (name == "timestep_scale_multiplier") {
    return std::vector<float>(static_cast<size_t>(count), 1000.0f);
  } else if (EndsWith(name, "mel_basis")) {
    scale = 0.2;
    offset = 0.05;
    absolute = true;  // a mel filterbank is non-negative
    // ...except on the arm that exists to SATURATE the mel log clamp. Attenuating
    // the basis by 1e-4 puts every bin under `kLtx2BweMelLogClamp`, which is the
    // only way the deterministic stream can reach a constant that otherwise only
    // real silence binds. Mirrors the generator's `param_values` exactly.
    if (name.find(".bwequiet.") != std::string::npos) attenuate = 1e-4;
  } else if (EndsWith(name, ".gamma")) {
    offset = 1.0;
  } else if (EndsWith(name, ".alpha") || EndsWith(name, ".beta")) {
    scale = 0.2;
  } else if (EndsWith(name, "std-of-means")) {
    offset = 1.0;
  } else if (EndsWith(name, "mean-of-means") || EndsWith(name, "scale_shift_table") ||
             EndsWith(name, "per_channel_scale1") || EndsWith(name, "per_channel_scale2")) {
    // scale 0.1, offset 0
  } else if (EndsWith(name, ".bias")) {
    scale = 0.05;
  } else if (rank == 1 && EndsWith(name, ".weight")) {
    offset = 1.0;  // a 1-D `.weight` is an affine norm gain, initialized to ones
  }

  const std::vector<double> raw = Ltx2Rand(name, count);
  std::vector<float> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    double value = raw[static_cast<size_t>(i)];
    if (absolute) value = std::abs(value);
    out[static_cast<size_t>(i)] = static_cast<float>((value * scale + offset) * attenuate);
  }
  return out;
}

// A weight bag that also records the (name, count) manifest in build order, so a
// test can prove its parameter set matches the generator's state_dict exactly.
struct ParamBag {
  vllm::Ltx2VaeWeights weights;
  std::vector<std::string> names;
  std::vector<int64_t> counts;

  void Put(const std::string& name, const std::vector<int64_t>& shape) {
    std::vector<float> values = Ltx2Param(name, shape);
    counts.push_back(static_cast<int64_t>(values.size()));
    names.push_back(name);
    weights.tensors[name] = std::move(values);
  }
};

void CheckManifest(const ParamBag& bag, const char* const* want_names, const int64_t* want_counts,
                   size_t want_size) {
  REQUIRE(bag.names.size() == want_size);
  REQUIRE(bag.counts.size() == want_size);
  for (size_t i = 0; i < want_size; ++i) {
    CHECK(bag.names[i] == std::string(want_names[i]));
    CHECK(bag.counts[i] == want_counts[i]);
  }
}

// ---------------------------------------------------------------------------
// Tolerances, derived from the measurement rather than picked.
//
// The tolerances started 6-14x above what the port actually produces, which makes
// them decoration: a band that can never bind reports nothing. These are set from
// the WORST arm in this file plus a stated margin, so a real drift moves them.
//
//   worst tensor arm   1.81794e-06  (the non-causal Conv video decoder)
//   worst filter arm   2.98023e-08  (the kaiser-sinc window)
//
// The margin is for libm, not for us: our side accumulates every reduction in
// double against a FIXED golden constant, so reduction-order variation across
// platforms cannot reach it — but `sin`, `exp`, `tanh` and `sqrt` differ by ~1 ulp
// between libm implementations, and the vocoder composes them through a deep
// sequential chain. ~2.7x on the tensors and ~3.4x on the filters covers that
// without leaving room for a structural porting error, which would move these by
// orders of magnitude rather than ulps.
//
// If a platform ever exceeds these, that is a finding to investigate and record —
// not a number to raise. AGENTS.md forbids widening a band to go green.
constexpr double kLtx2GoldenTol = 5e-6;
constexpr double kLtx2FilterTol = 1e-7;

// The shared, NaN-hardened reduction. The local copy this replaces used
// `std::max(worst, ...)`, which is `a < b ? b : a`; `a < NaN` is false, so an
// all-NaN result against a correct golden reduced to 0.0 (issue #449).
using vllm_test::MaxAbsDiff;

// The reduced audio decoder the generator built (AUDIO_DEC).
vllm::Ltx2AudioDecoderConfig ReducedAudioDecoderConfig(int64_t mel_bins) {
  vllm::Ltx2AudioDecoderConfig cfg;
  cfg.ch = 8;
  cfg.out_ch = 2;
  cfg.ch_mult = {1, 2, 4};
  cfg.num_res_blocks = 1;
  cfg.attn_resolutions = {8};
  cfg.resolution = 32;
  cfg.z_channels = 4;
  cfg.norm_type = vllm::Ltx2NormType::kPixel;
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  cfg.mid_block_add_attention = true;
  cfg.mel_bins = mel_bins;
  cfg.prefix = "ltx2.audiodec.";
  return cfg;
}

// The reduced GROUP-NORM audio decoder the generator built (AUDIO_GROUP_DEC).
// `ch` is 32 because `build_normalization_layer` forwards its own `num_groups`
// keyword, whose default is 32 and which no audio_vae call site passes
// (normalization.py:44, 56), and torch's GroupNorm refuses a channel count 32
// does not divide; `z_channels` is 16 because `PerChannelStatistics` indexes the
// patchified `(c, f)` axis, so z_channels x latent mel bins must equal `ch`.
vllm::Ltx2AudioDecoderConfig ReducedAudioDecoderGroupConfig() {
  vllm::Ltx2AudioDecoderConfig cfg = ReducedAudioDecoderConfig(
      vllm_test::kLtx2AudioDecGroupOutMelBins);
  cfg.ch = 32;
  cfg.z_channels = vllm_test::kLtx2AudioDecGroupLatentC;
  cfg.norm_type = vllm::Ltx2NormType::kGroup;
  // ResnetBlock REFUSES GroupNorm on any causal axis (resnet.py:130-131), so this
  // is the only causality a group-norm checkpoint can legally declare.
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kNone;
  cfg.prefix = "ltx2.audiodecgroup.";
  return cfg;
}

// Build the audio decoder's parameters in upstream state_dict ORDER:
// per_channel_statistics, conv_in, mid, up (block / attn / upsample per level),
// norm_out, conv_out. PixelNorm carries no parameters, which is why no norm
// tensor appears on the pixel arms; GroupNorm is affine, so on `kGroup` every
// `norm1` / `norm2` / `attn.norm` / `norm_out` contributes a weight and a bias
// AHEAD of the convolution it precedes (resnet.py:136-146, attention.py:25-29).
ParamBag BuildAudioDecoderParams(const vllm::Ltx2AudioDecoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;
  const int64_t levels = static_cast<int64_t>(cfg.ch_mult.size());
  const int64_t base = cfg.ch * cfg.ch_mult[static_cast<size_t>(levels - 1)];
  const bool group = cfg.norm_type == vllm::Ltx2NormType::kGroup;

  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.ch});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.ch});
  bag.Put(p + "conv_in.conv.weight", {base, cfg.z_channels, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {base});

  auto put_norm = [&](const std::string& prefix, int64_t channels) {
    if (!group) return;
    bag.Put(prefix + ".weight", {channels});
    bag.Put(prefix + ".bias", {channels});
  };
  auto put_resnet = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch) {
    put_norm(prefix + ".norm1", in_ch);
    bag.Put(prefix + ".conv1.conv.weight", {out_ch, in_ch, 3, 3});
    bag.Put(prefix + ".conv1.conv.bias", {out_ch});
    put_norm(prefix + ".norm2", out_ch);
    bag.Put(prefix + ".conv2.conv.weight", {out_ch, out_ch, 3, 3});
    bag.Put(prefix + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      bag.Put(prefix + ".nin_shortcut.conv.weight", {out_ch, in_ch, 1, 1});
      bag.Put(prefix + ".nin_shortcut.conv.bias", {out_ch});
    }
  };
  auto put_attn = [&](const std::string& prefix, int64_t channels) {
    put_norm(prefix + ".norm", channels);
    for (const char* leaf : {"q", "k", "v", "proj_out"}) {
      bag.Put(prefix + "." + leaf + ".weight", {channels, channels, 1, 1});
      bag.Put(prefix + "." + leaf + ".bias", {channels});
    }
  };

  put_resnet(p + "mid.block_1", base, base);
  if (cfg.mid_block_add_attention) put_attn(p + "mid.attn_1", base);
  put_resnet(p + "mid.block_2", base, base);

  // build_upsampling_path (upsample.py:58-106) walks levels in REVERSE and
  // inserts each stage at the front, so `up.<level>` is indexed by level.
  int64_t curr_res = cfg.resolution / (int64_t{1} << (levels - 1));
  int64_t block_in = base;
  struct Stage {
    std::vector<std::pair<int64_t, int64_t>> blocks;  // (in, out)
    std::vector<int64_t> attn;                        // channels
    int64_t upsample = 0;                             // channels, 0 = none
  };
  std::vector<Stage> stages(static_cast<size_t>(levels));
  for (int64_t level = levels - 1; level >= 0; --level) {
    Stage& stage = stages[static_cast<size_t>(level)];
    const int64_t block_out = cfg.ch * cfg.ch_mult[static_cast<size_t>(level)];
    for (int64_t i = 0; i < cfg.num_res_blocks + 1; ++i) {
      stage.blocks.emplace_back(block_in, block_out);
      block_in = block_out;
      if (std::find(cfg.attn_resolutions.begin(), cfg.attn_resolutions.end(), curr_res) !=
          cfg.attn_resolutions.end()) {
        stage.attn.push_back(block_in);
      }
    }
    if (level != 0) {
      stage.upsample = block_in;
      curr_res *= 2;
    }
  }
  for (int64_t level = 0; level < levels; ++level) {
    const Stage& stage = stages[static_cast<size_t>(level)];
    const std::string sp = p + "up." + std::to_string(level);
    for (size_t i = 0; i < stage.blocks.size(); ++i) {
      put_resnet(sp + ".block." + std::to_string(i), stage.blocks[i].first,
                 stage.blocks[i].second);
    }
    for (size_t i = 0; i < stage.attn.size(); ++i) {
      put_attn(sp + ".attn." + std::to_string(i), stage.attn[i]);
    }
    if (stage.upsample != 0) {
      bag.Put(sp + ".upsample.conv.conv.weight", {stage.upsample, stage.upsample, 3, 3});
      bag.Put(sp + ".upsample.conv.conv.bias", {stage.upsample});
    }
  }

  put_norm(p + "norm_out", block_in);
  bag.Put(p + "conv_out.conv.weight", {cfg.out_ch, block_in, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {cfg.out_ch});
  return bag;
}

// The reduced BigVGAN v2 vocoder the generator built (VOC).
vllm::Ltx2VocoderConfig ReducedVocoderConfig() {
  vllm::Ltx2VocoderConfig cfg;
  cfg.resblock_kernel_sizes = {3, 7};
  cfg.upsample_rates = {2, 2};
  cfg.upsample_kernel_sizes = {4, 4};
  cfg.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}};
  cfg.upsample_initial_channel = 16;
  cfg.amp = true;
  cfg.snakebeta = true;
  cfg.use_tanh_at_final = true;
  cfg.apply_final_activation = true;
  cfg.use_bias_at_final = true;
  cfg.output_sampling_rate = 16000;
  cfg.prefix = "ltx2.voc.";
  return cfg;
}

// Vocoder.__init__ registration order: conv_pre, ups, resblocks, act_post,
// conv_post. `*.filter` buffers are absent on purpose — the kaiser-sinc windows
// are COMPUTED at construction, never loaded.
void PutVocoderParams(ParamBag& bag, const vllm::Ltx2VocoderConfig& cfg) {
  const std::string p = cfg.prefix;
  const int64_t initial = cfg.upsample_initial_channel;
  const int64_t num_kernels = static_cast<int64_t>(cfg.resblock_kernel_sizes.size());

  bag.Put(p + "conv_pre.weight", {initial, 128, 7});
  bag.Put(p + "conv_pre.bias", {initial});
  for (size_t i = 0; i < cfg.upsample_rates.size(); ++i) {
    const int64_t in_ch = initial / (int64_t{1} << i);
    const int64_t out_ch = initial / (int64_t{1} << (i + 1));
    // ConvTranspose1d weight is [in, out, k].
    bag.Put(p + "ups." + std::to_string(i) + ".weight",
            {in_ch, out_ch, cfg.upsample_kernel_sizes[i]});
    bag.Put(p + "ups." + std::to_string(i) + ".bias", {out_ch});
  }
  for (size_t i = 0; i < cfg.upsample_rates.size(); ++i) {
    const int64_t ch = initial / (int64_t{1} << (i + 1));
    for (int64_t j = 0; j < num_kernels; ++j) {
      const std::string block =
          p + "resblocks." + std::to_string(static_cast<int64_t>(i) * num_kernels + j);
      const int64_t kernel = cfg.resblock_kernel_sizes[static_cast<size_t>(j)];
      for (int64_t d = 0; d < 3; ++d) {
        bag.Put(block + ".convs1." + std::to_string(d) + ".weight", {ch, ch, kernel});
        bag.Put(block + ".convs1." + std::to_string(d) + ".bias", {ch});
      }
      for (int64_t d = 0; d < 3; ++d) {
        bag.Put(block + ".convs2." + std::to_string(d) + ".weight", {ch, ch, kernel});
        bag.Put(block + ".convs2." + std::to_string(d) + ".bias", {ch});
      }
      if (cfg.amp) {
        for (const char* group : {"acts1", "acts2"}) {
          for (int64_t d = 0; d < 3; ++d) {
            const std::string act = block + "." + group + "." + std::to_string(d) + ".act.";
            bag.Put(act + "alpha", {ch});
            if (cfg.snakebeta) bag.Put(act + "beta", {ch});
          }
        }
      }
    }
  }
  const int64_t final_channels = initial / (int64_t{1} << cfg.upsample_rates.size());
  if (cfg.amp) {
    bag.Put(p + "act_post.act.alpha", {final_channels});
    // act_post is UNCONDITIONALLY SnakeBeta, so its `.beta` is present whenever
    // `amp` is — even on the `activation="snake"` arm, where every resblock
    // activation above has ONLY `.alpha`. Upstream builds it as
    // `Activation1d(SnakeBeta(final_channels))` with no `activation=` argument
    // (vocoder.py:388), unlike the resblocks one line earlier (vocoder.py:376).
    // The generator's own state_dict manifest for the snake arm proves it, and
    // gating this on `cfg.snakebeta` is what made the port read the wrong scale.
    bag.Put(p + "act_post.act.beta", {final_channels});
  }
  bag.Put(p + "conv_post.weight", {2, final_channels, 7});
  if (cfg.use_bias_at_final) bag.Put(p + "conv_post.bias", {2});
}

}  // namespace

TEST_CASE("ltx2 vae: the audio decoder matches upstream ltx_core") {
  const vllm::Ltx2AudioDecoderConfig cfg =
      ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecOutMelBins);
  ParamBag bag = BuildAudioDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioDecParamNames, vllm_test::kLtx2AudioDecParamCounts,
                std::size(vllm_test::kLtx2AudioDecParamNames));

  const int64_t latent_t = vllm_test::kLtx2AudioDecLatentT;
  const int64_t latent_f = vllm_test::kLtx2AudioDecLatentF;
  const std::vector<float> latent =
      Ltx2Input("ltx2.audiodec.input", cfg.z_channels * latent_t * latent_f, 1.0);

  const vllm::Ltx2AudioSpectrogram out =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, latent, cfg.z_channels, latent_t, latent_f);
  CHECK(out.channels == cfg.out_ch);
  CHECK(out.frames == vllm_test::kLtx2AudioDecOutFrames);
  CHECK(out.mel_bins == vllm_test::kLtx2AudioDecOutMelBins);
  const double err =
      MaxAbsDiff(out.data, vllm_test::kLtx2AudioDecGolden, std::size(vllm_test::kLtx2AudioDecGolden));
  INFO("audio decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // The SHIPPED configuration is NOT end-to-end causal, and this asserts that
  // rather than papering over it: the AttnBlocks attend over the whole (time,
  // mel) map (attention.py:31-55), so a change anywhere reaches every output
  // frame. Upstream agrees — measured in the generator, section 1b. The
  // causality claim is about the CONVOLUTIONS, and it is gated below.
  std::vector<float> bumped = latent;
  for (int64_t f = 0; f < latent_f; ++f) {
    for (int64_t c = 0; c < cfg.z_channels; ++c) {
      bumped[static_cast<size_t>((c * latent_t + (latent_t - 1)) * latent_f + f)] += 3.0f;
    }
  }
  const vllm::Ltx2AudioSpectrogram perturbed =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, bumped, cfg.z_channels, latent_t, latent_f);
  REQUIRE(perturbed.data.size() == out.data.size());
  for (int64_t t = 0; t < perturbed.frames; ++t) {
    bool moved = false;
    for (int64_t c = 0; c < perturbed.channels && !moved; ++c) {
      for (int64_t m = 0; m < perturbed.mel_bins; ++m) {
        const size_t i = static_cast<size_t>((c * perturbed.frames + t) * perturbed.mel_bins + m);
        if (perturbed.data[i] != out.data[i]) {
          moved = true;
          break;
        }
      }
    }
    INFO("output frame " << t << " under global attention");
    CHECK(moved);
  }
}

TEST_CASE("ltx2 vae: the GROUP-NORM audio decoder matches upstream ltx_core") {
  // THE ARM THAT MAKES `Ltx2AudioDecoderConfig::norm_eps` REACHABLE. Every other
  // audio arm in this file runs `norm_type = kPixel`, so `ApplyNorm` never enters
  // the GroupNorm branch and `norm_eps` is not merely inert but never READ:
  // mutating it 1e-6 -> 1e-4, a 100x change, left all 33 cases green.
  //
  // `norm_type = group` is not hypothetical, and it is not free either.
  // `AudioDecoder.__init__` declares `norm_type = GROUP` (audio_vae.py:294) and
  // on the next line `causality_axis = WIDTH` (audio_vae.py:295) — a pair
  // `ResnetBlock` REFUSES, with `ValueError: Causal ResnetBlock with GroupNorm is
  // not supported` (resnet.py:130-131), so constructing the upstream decoder on
  // pure defaults raises rather than group-normalizing. What is legal, and what
  // this arm runs, is a checkpoint that declares `causality_axis: none` alongside
  // it — the other half of `build_normalization_layer` (normalization.py:56-57).
  // Without this arm such a checkpoint would run a 100x-wrong stabilizer and
  // still produce a spectrogram.
  const vllm::Ltx2AudioDecoderConfig cfg = ReducedAudioDecoderGroupConfig();
  ParamBag bag = BuildAudioDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioDecGroupParamNames,
                vllm_test::kLtx2AudioDecGroupParamCounts,
                std::size(vllm_test::kLtx2AudioDecGroupParamNames));

  const int64_t c = vllm_test::kLtx2AudioDecGroupLatentC;
  const int64_t t = vllm_test::kLtx2AudioDecGroupLatentT;
  const int64_t f = vllm_test::kLtx2AudioDecGroupLatentF;
  const std::vector<float> latent = Ltx2Input("ltx2.audiodecgroup.input", c * t * f, 1.0);

  const vllm::Ltx2AudioSpectrogram mel =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, latent, c, t, f);
  CHECK(mel.channels == cfg.out_ch);
  CHECK(mel.frames == vllm_test::kLtx2AudioDecGroupOutFrames);
  CHECK(mel.mel_bins == vllm_test::kLtx2AudioDecGroupOutMelBins);

  const double err = MaxAbsDiff(mel.data, vllm_test::kLtx2AudioDecGroupGolden,
                                std::size(vllm_test::kLtx2AudioDecGroupGolden));
  INFO("group-norm audio decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // The combination upstream REFUSES must be refused here too, not silently
  // group-normalized on a causal axis.
  vllm::Ltx2AudioDecoderConfig bad = cfg;
  bad.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2AudioDecoderForward(bad, bag.weights, latent, c, t, f);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  CHECK(message.find("GroupNorm") != std::string::npos);
}

TEST_CASE("ltx2 vae: the audio decoder's CONVOLUTIONS are one-sided in time") {
  // The trap: padding the time axis symmetrically instead of on the LEFT still
  // produces a plausible spectrogram that merely peeks into the future. Isolate
  // the convolutions by turning attention off — the reach that leaves is a
  // property of the padding alone, and section 1b records what upstream's own
  // reach is, so this is not a claim the port makes about itself.
  vllm::Ltx2AudioDecoderConfig cfg =
      ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecOutMelBins);
  cfg.attn_resolutions.clear();
  cfg.mid_block_add_attention = false;
  cfg.prefix = "ltx2.audiodeccausal.";
  ParamBag bag = BuildAudioDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioDecCausalParamNames,
                vllm_test::kLtx2AudioDecCausalParamCounts,
                std::size(vllm_test::kLtx2AudioDecCausalParamNames));

  const int64_t latent_t = vllm_test::kLtx2AudioDecLatentT;
  const int64_t latent_f = vllm_test::kLtx2AudioDecLatentF;
  const std::vector<float> latent =
      Ltx2Input("ltx2.audiodec.input", cfg.z_channels * latent_t * latent_f, 1.0);
  std::vector<float> bumped = latent;
  for (int64_t f = 0; f < latent_f; ++f) {
    for (int64_t c = 0; c < cfg.z_channels; ++c) {
      bumped[static_cast<size_t>((c * latent_t + (latent_t - 1)) * latent_f + f)] += 3.0f;
    }
  }
  const vllm::Ltx2AudioSpectrogram base =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, latent, cfg.z_channels, latent_t, latent_f);
  const vllm::Ltx2AudioSpectrogram moved =
      vllm::Ltx2AudioDecoderForward(cfg, bag.weights, bumped, cfg.z_channels, latent_t, latent_f);
  REQUIRE(moved.data.size() == base.data.size());

  int64_t first_moved = base.frames;
  int64_t last_moved = -1;
  for (int64_t t = 0; t < base.frames; ++t) {
    bool differs = false;
    for (int64_t c = 0; c < base.channels && !differs; ++c) {
      for (int64_t m = 0; m < base.mel_bins; ++m) {
        const size_t i = static_cast<size_t>((c * base.frames + t) * base.mel_bins + m);
        if (base.data[i] != moved.data[i]) {
          differs = true;
          break;
        }
      }
    }
    if (differs) {
      if (first_moved == base.frames) first_moved = t;
      last_moved = t;
    }
  }
  INFO("convolution-only reach of a last-latent-frame bump: [" << first_moved << ", " << last_moved
                                                               << "]");
  CHECK(first_moved == vllm_test::kLtx2AudioDecCausalFirstMoved);
  CHECK(last_moved == vllm_test::kLtx2AudioDecCausalLastMoved);
  CHECK(first_moved > 0);  // the past cannot see the future
}

TEST_CASE("ltx2 vae: the other three causality axes match upstream ltx_core") {
  // Every arm elsewhere in this file runs `causality_axis=height`, the shipped
  // default (audio_vae/model_configurator.py:134). The remaining three branches of
  // CausalConv2d's padding switch (causality_axis.py:4-10) were never executed, so
  // the port's pad split for them was an untested claim rather than a gated one.
  struct Arm {
    vllm::Ltx2CausalityAxis axis;
    const char* name;
    const float* golden;
    size_t golden_size;
    int64_t frames;
    int64_t mel_bins;
  };
  const Arm arms[] = {
      {vllm::Ltx2CausalityAxis::kNone, "NONE", vllm_test::kLtx2AudioDecNoneGolden,
       std::size(vllm_test::kLtx2AudioDecNoneGolden), vllm_test::kLtx2AudioDecNoneOutFrames,
       vllm_test::kLtx2AudioDecNoneOutMelBins},
      {vllm::Ltx2CausalityAxis::kWidth, "WIDTH", vllm_test::kLtx2AudioDecWidthGolden,
       std::size(vllm_test::kLtx2AudioDecWidthGolden), vllm_test::kLtx2AudioDecWidthOutFrames,
       vllm_test::kLtx2AudioDecWidthOutMelBins},
      {vllm::Ltx2CausalityAxis::kWidthCompatibility, "WIDTH_COMPATIBILITY",
       vllm_test::kLtx2AudioDecWidthCompatGolden,
       std::size(vllm_test::kLtx2AudioDecWidthCompatGolden),
       vllm_test::kLtx2AudioDecWidthCompatOutFrames,
       vllm_test::kLtx2AudioDecWidthCompatOutMelBins},
  };

  std::vector<std::vector<float>> outputs;
  for (const Arm& arm : arms) {
    vllm::Ltx2AudioDecoderConfig cfg =
        ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecOutMelBins);
    cfg.causality_axis = arm.axis;
    ParamBag bag = BuildAudioDecoderParams(cfg);
    const std::vector<float> latent = Ltx2Input(
        "ltx2.audiodec.input",
        cfg.z_channels * vllm_test::kLtx2AudioDecLatentT * vllm_test::kLtx2AudioDecLatentF, 1.0);
    const vllm::Ltx2AudioSpectrogram out = vllm::Ltx2AudioDecoderForward(
        cfg, bag.weights, latent, cfg.z_channels, vllm_test::kLtx2AudioDecLatentT,
        vllm_test::kLtx2AudioDecLatentF);
    INFO("causality_axis = " << arm.name);
    CHECK(out.frames == arm.frames);
    CHECK(out.mel_bins == arm.mel_bins);
    const double err = MaxAbsDiff(out.data, arm.golden, arm.golden_size);
    INFO("max|diff| = " << err);
    CHECK(err <= kLtx2GoldenTol);
    outputs.push_back(out.data);
  }

  // WIDTH and WIDTH_COMPATIBILITY pad the width axis IDENTICALLY in the
  // convolutions, so a port could plausibly collapse them into one branch. It must
  // not: the UPSAMPLER treats them differently — upsample.py:44-48 does NOT drop
  // the first interpolated element for WIDTH_COMPATIBILITY, while every other axis
  // does. That is the whole difference, it is invisible in the padding code, and
  // this asserts the two arms actually diverge.
  REQUIRE(outputs[1].size() == outputs[2].size());
  CHECK(outputs[1] != outputs[2]);
}

TEST_CASE("ltx2 vae: the audio decoder pads the frequency axis to the target mel bins") {
  // audio_vae.py:458-467 zero-pads on the RIGHT of the frequency axis when the
  // configured mel_bins exceeds what the network produced. A port that returns
  // the unpadded tensor passes every other assertion in this file.
  const vllm::Ltx2AudioDecoderConfig cfg =
      ReducedAudioDecoderConfig(vllm_test::kLtx2AudioDecPadOutMelBins);
  ParamBag bag = BuildAudioDecoderParams(cfg);
  const std::vector<float> latent =
      Ltx2Input("ltx2.audiodec.input",
                cfg.z_channels * vllm_test::kLtx2AudioDecLatentT * vllm_test::kLtx2AudioDecLatentF,
                1.0);
  const vllm::Ltx2AudioSpectrogram out = vllm::Ltx2AudioDecoderForward(
      cfg, bag.weights, latent, cfg.z_channels, vllm_test::kLtx2AudioDecLatentT,
      vllm_test::kLtx2AudioDecLatentF);
  CHECK(out.mel_bins == vllm_test::kLtx2AudioDecPadOutMelBins);
  const double err = MaxAbsDiff(out.data, vllm_test::kLtx2AudioDecPadGolden,
                                std::size(vllm_test::kLtx2AudioDecPadGolden));
  INFO("padded audio decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the BigVGAN v2 vocoder matches upstream ltx_core") {
  // The anti-aliasing window is COMPUTED, never loaded. Gate it FIRST: a wrong
  // filter makes every SnakeBeta wrong and the decoder mismatch unlocalizable.
  const std::vector<float> filter = vllm::Ltx2KaiserSincFilter1d(0.5 / 2, 0.6 / 2, 12);
  REQUIRE(filter.size() == std::size(vllm_test::kLtx2VocUpFilterGolden));
  double filter_err = 0.0;
  double filter_sum = 0.0;
  for (size_t i = 0; i < filter.size(); ++i) {
    filter_err = std::max(filter_err, std::abs(static_cast<double>(filter[i]) -
                                               vllm_test::kLtx2VocUpFilterGolden[i]));
    filter_sum += filter[i];
  }
  INFO("kaiser-sinc filter max|diff| = " << filter_err);
  CHECK(filter_err <= kLtx2FilterTol);
  CHECK(filter_sum == doctest::Approx(1.0).epsilon(1e-6).scale(0.0));

  const vllm::Ltx2VocoderConfig cfg = ReducedVocoderConfig();
  ParamBag bag;
  PutVocoderParams(bag, cfg);
  CheckManifest(bag, vllm_test::kLtx2VocParamNames, vllm_test::kLtx2VocParamCounts,
                std::size(vllm_test::kLtx2VocParamNames));

  const int64_t frames = vllm_test::kLtx2VocFrames;
  const int64_t mel_bins = vllm_test::kLtx2VocMelBins;
  const std::vector<float> mel = Ltx2Input("ltx2.voc.input", 2 * frames * mel_bins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderForward(cfg, bag.weights, mel, 2, frames, mel_bins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2VocOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2VocGolden, std::size(vllm_test::kLtx2VocGolden));
  INFO("vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
  // The golden is deliberately UNSATURATED: a tanh-saturated golden would hide
  // errors instead of catching them.
  for (float value : wave) {
    CHECK(value > -0.999f);
    CHECK(value < 0.999f);
  }
}

TEST_CASE("ltx2 vae: act_post stays SnakeBeta on the plain-snake vocoder arm") {
  // `resblock="AMP1"` with `activation="snake"`. Upstream builds act_post as
  // `Activation1d(SnakeBeta(final_channels))` inside `if self.is_amp`, taking NO
  // `activation=` argument (vocoder.py:388) — unlike the resblocks one line
  // earlier (vocoder.py:376). So on this arm every resblock activation is plain
  // Snake, which reuses ALPHA as its reciprocal scale (vocoder.py:198), while
  // act_post alone still reads `.beta`.
  //
  // Keying act_post off `activation` therefore reads the wrong scale for one
  // activation and still produces a plausible waveform. No other arm can catch
  // it: on the shipped snakebeta arm the two spellings agree, and on the legacy
  // arm there is no act_post at all. The manifest below is the second half of the
  // proof — upstream's own state_dict has `act_post.act.beta` present while every
  // `acts1/acts2` entry has only `.alpha`.
  vllm::Ltx2VocoderConfig cfg = ReducedVocoderConfig();
  cfg.snakebeta = false;  // activation == "snake"
  cfg.prefix = "ltx2.vocsnake.";

  ParamBag bag;
  PutVocoderParams(bag, cfg);
  CheckManifest(bag, vllm_test::kLtx2VocSnakeParamNames, vllm_test::kLtx2VocSnakeParamCounts,
                std::size(vllm_test::kLtx2VocSnakeParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                               vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2VocSnakeOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2VocSnakeGolden, std::size(vllm_test::kLtx2VocSnakeGolden));
  INFO("snake-arm vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the legacy resblock-1 vocoder arm matches upstream ltx_core") {
  vllm::Ltx2VocoderConfig cfg;
  cfg.resblock_kernel_sizes = {3};
  cfg.upsample_rates = {2};
  cfg.upsample_kernel_sizes = {4};
  cfg.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.upsample_initial_channel = 8;
  cfg.amp = false;  // resblock "1": ResBlock1 + plain leaky ReLU, no anti-aliasing
  cfg.output_sampling_rate = 16000;
  cfg.prefix = "ltx2.vocleg.";

  ParamBag bag;
  PutVocoderParams(bag, cfg);
  CheckManifest(bag, vllm_test::kLtx2VocLegacyParamNames, vllm_test::kLtx2VocLegacyParamCounts,
                std::size(vllm_test::kLtx2VocLegacyParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                               vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2VocLegacyOutSamples);
  const double err = MaxAbsDiff(wave, vllm_test::kLtx2VocLegacyGolden,
                                std::size(vllm_test::kLtx2VocLegacyGolden));
  INFO("legacy vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the BWE vocoder chain matches upstream ltx_core") {
  // The hann-sinc resampler window is a DIFFERENT filter from the kaiser one the
  // activations use, and is likewise computed rather than loaded (persistent=False).
  int64_t kernel_size = 0, pad = 0, pad_left = 0, pad_right = 0;
  const std::vector<float> resample_filter =
      vllm::Ltx2HannSincResampleFilter1d(2, &kernel_size, &pad, &pad_left, &pad_right);
  CHECK(kernel_size == vllm_test::kLtx2BweResamplerKernel);
  CHECK(pad == 7);
  CHECK(pad_left == 28);
  CHECK(pad_right == 27);
  const double filter_err = MaxAbsDiff(resample_filter, vllm_test::kLtx2BweResamplerFilterGolden,
                                       std::size(vllm_test::kLtx2BweResamplerFilterGolden));
  INFO("hann-sinc resampler filter max|diff| = " << filter_err);
  CHECK(filter_err <= kLtx2FilterTol);

  vllm::Ltx2VocoderBweConfig cfg;
  cfg.vocoder = ReducedVocoderConfig();
  cfg.vocoder.prefix = "ltx2.bwe.vocoder.";
  cfg.bwe_generator = ReducedVocoderConfig();
  cfg.bwe_generator.prefix = "ltx2.bwe.bwe_generator.";
  cfg.bwe_generator.resblock_kernel_sizes = {3};
  cfg.bwe_generator.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.bwe_generator.upsample_rates = {4, 4};
  cfg.bwe_generator.upsample_kernel_sizes = {8, 8};
  cfg.bwe_generator.apply_final_activation = false;
  cfg.bwe_generator.output_sampling_rate = 32000;
  cfg.filter_length = 16;
  cfg.hop_length = 8;
  cfg.win_length = 16;
  cfg.n_mel_channels = 64;
  cfg.input_sampling_rate = 16000;
  cfg.output_sampling_rate = 32000;
  cfg.prefix = "ltx2.bwe.";

  ParamBag bag;
  PutVocoderParams(bag, cfg.vocoder);
  PutVocoderParams(bag, cfg.bwe_generator);
  bag.Put(cfg.prefix + "mel_stft.mel_basis", {cfg.n_mel_channels, cfg.filter_length / 2 + 1});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.forward_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.inverse_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  CheckManifest(bag, vllm_test::kLtx2BweParamNames, vllm_test::kLtx2BweParamCounts,
                std::size(vllm_test::kLtx2BweParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderWithBweForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                                      vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2BweOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2BweGolden, std::size(vllm_test::kLtx2BweGolden));
  INFO("BWE vocoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

namespace {

// The reduced Conv video decoder the generator built (VIDEO_BLOCKS / VIDEO_DEC).
vllm::Ltx2ConvVideoDecoderConfig ReducedVideoDecoderConfig() {
  vllm::Ltx2ConvVideoDecoderConfig cfg;
  cfg.in_channels = 6;
  cfg.out_channels = 3;
  cfg.patch_size = 2;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.causal = true;
  cfg.timestep_conditioning = true;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReflect;
  cfg.base_channels = 8;
  cfg.prefix = "ltx2.videodec.";
  cfg.decoder_blocks = {
      {"res_x", 1, 0, /*inject_noise=*/true, false},
      {"compress_all", 1, 2, false, /*residual=*/true},
      {"res_x_y", 1, 2, false, false},
      {"compress_space", 1, 1, false, false},
      {"attn", 1, 0, false, false},
      {"compress_time", 1, 1, false, false},
      {"res_x", 2, 0, false, false},
  };
  return cfg;
}

// ConvVideoDecoder state_dict order: the module's own PARAMETERS first
// (timestep_scale_multiplier, last_scale_shift_table), then submodules in
// registration order (per_channel_statistics, conv_in, up_blocks, conv_out,
// last_time_embedder). conv_norm_out is a PixelNorm and carries nothing.
ParamBag BuildVideoDecoderParams(const vllm::Ltx2ConvVideoDecoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;

  int64_t multiplier = 1;
  for (const vllm::Ltx2VideoDecoderBlock& block : cfg.decoder_blocks) {
    if (block.name == "compress_time" || block.name == "compress_space" ||
        block.name == "compress_all") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 1;
    } else if (block.name == "res_x_y") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 2;
    }
  }
  int64_t channels = cfg.base_channels * multiplier;

  // The bottleneck width is what conv_in widens to; the final width is what the
  // reversed block walk ends at, which is where last_scale_shift_table lives.
  int64_t final_channels = channels;
  for (auto it = cfg.decoder_blocks.rbegin(); it != cfg.decoder_blocks.rend(); ++it) {
    if (it->name == "res_x_y") {
      final_channels /= (it->multiplier != 0 ? it->multiplier : 2);
    } else if (it->name == "compress_time" || it->name == "compress_space" ||
               it->name == "compress_all") {
      final_channels /= (it->multiplier != 0 ? it->multiplier : 1);
    }
  }

  // Both module-level parameters exist only under timestep conditioning
  // (conv_video_decoder.py:256-261), and torch emits _parameters before _modules.
  if (cfg.timestep_conditioning) {
    bag.Put(p + "timestep_scale_multiplier", {});
    bag.Put(p + "last_scale_shift_table", {2, final_channels});
  }
  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.in_channels});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.in_channels});
  bag.Put(p + "conv_in.conv.weight", {channels, cfg.in_channels, 3, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {channels});

  auto put_resnet3d = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch,
                          bool inject_noise, bool timestep) {
    if (inject_noise) {
      bag.Put(prefix + ".per_channel_scale1", {in_ch, 1, 1});
      bag.Put(prefix + ".per_channel_scale2", {in_ch, 1, 1});
    }
    if (timestep) bag.Put(prefix + ".scale_shift_table", {4, in_ch});
    bag.Put(prefix + ".conv1.conv.weight", {out_ch, in_ch, 3, 3, 3});
    bag.Put(prefix + ".conv1.conv.bias", {out_ch});
    bag.Put(prefix + ".conv2.conv.weight", {out_ch, out_ch, 3, 3, 3});
    bag.Put(prefix + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      bag.Put(prefix + ".conv_shortcut.weight", {out_ch, in_ch, 1, 1, 1});
      bag.Put(prefix + ".conv_shortcut.bias", {out_ch});
      bag.Put(prefix + ".norm3.weight", {in_ch});
      bag.Put(prefix + ".norm3.bias", {in_ch});
    }
  };

  int64_t index = 0;
  for (auto it = cfg.decoder_blocks.rbegin(); it != cfg.decoder_blocks.rend(); ++it, ++index) {
    const vllm::Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      if (cfg.timestep_conditioning) {
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_1.weight", {channels * 4, 256});
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_1.bias", {channels * 4});
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_2.weight",
                {channels * 4, channels * 4});
        bag.Put(bp + ".time_embedder.timestep_embedder.linear_2.bias", {channels * 4});
      }
      for (int64_t i = 0; i < block.num_layers; ++i) {
        put_resnet3d(bp + ".res_blocks." + std::to_string(i), channels, channels,
                     block.inject_noise, cfg.timestep_conditioning);
      }
    } else if (block.name == "res_x_y") {
      const int64_t out_ch = channels / (block.multiplier != 0 ? block.multiplier : 2);
      put_resnet3d(bp, channels, out_ch, block.inject_noise, /*timestep=*/false);
      channels = out_ch;
    } else if (block.name == "attn") {
      bag.Put(bp + ".norm.gamma", {channels, 1, 1});
      bag.Put(bp + ".to_qkv.weight", {channels * 3, channels, 1, 1});
      bag.Put(bp + ".to_qkv.bias", {channels * 3});
      bag.Put(bp + ".proj.weight", {channels, channels, 1, 1});
      bag.Put(bp + ".proj.bias", {channels});
    } else {
      int64_t stride_product = 2;
      if (block.name == "compress_space") stride_product = 4;
      if (block.name == "compress_all") stride_product = 8;
      const int64_t reduction = block.multiplier != 0 ? block.multiplier : 1;
      const int64_t conv_out = stride_product * channels / reduction;
      bag.Put(bp + ".conv.conv.weight", {conv_out, channels, 3, 3, 3});
      bag.Put(bp + ".conv.conv.bias", {conv_out});
      channels /= reduction;
    }
  }

  bag.Put(p + "conv_out.conv.weight",
          {cfg.out_channels * cfg.patch_size * cfg.patch_size, channels, 3, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {cfg.out_channels * cfg.patch_size * cfg.patch_size});
  if (cfg.timestep_conditioning) {
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_1.weight", {channels * 2, 256});
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_1.bias", {channels * 2});
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_2.weight",
            {channels * 2, channels * 2});
    bag.Put(p + "last_time_embedder.timestep_embedder.linear_2.bias", {channels * 2});
  }
  return bag;
}

// The generator's patched torch.randn: one deterministic draw per call, keyed by
// CALL INDEX. `counts` records the sequence so the test can prove the port
// consumes noise in the same order and the same sizes upstream did.
class GoldenNoise : public vllm::Ltx2NoiseStream {
 public:
  // `prefix` selects the arm's noise stream; the generator keys its patched
  // torch.randn by the same name and CALL INDEX.
  explicit GoldenNoise(std::string prefix = "ltx2.videodec.") : prefix_(std::move(prefix)) {}

  std::vector<float> Draw(int64_t count) override {
    std::vector<float> values =
        Ltx2Input(prefix_ + "noise." + std::to_string(counts_.size()), count, 1.0);
    counts_.push_back(count);
    return values;
  }
  const std::vector<int64_t>& counts() const { return counts_; }

 private:
  std::string prefix_;
  std::vector<int64_t> counts_;
};

}  // namespace

TEST_CASE("ltx2 vae: the Conv video decoder matches upstream ltx_core") {
  const vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecParamNames, vllm_test::kLtx2VideoDecParamCounts,
                std::size(vllm_test::kLtx2VideoDecParamNames));

  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);

  GoldenNoise noise;
  const vllm::Ltx2VideoFrames frames =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise);
  CHECK(frames.channels == vllm_test::kLtx2VideoDecOutC);
  CHECK(frames.frames == vllm_test::kLtx2VideoDecOutT);
  CHECK(frames.height == vllm_test::kLtx2VideoDecOutH);
  CHECK(frames.width == vllm_test::kLtx2VideoDecOutW);

  // Noise is consumed in upstream's own call order and sizes. A port that drew a
  // full [C,T,H,W] block where upstream draws only [H,W] would still produce a
  // finite, plausible clip.
  REQUIRE(static_cast<int64_t>(noise.counts().size()) == vllm_test::kLtx2VideoDecNoiseDraws);
  for (size_t i = 0; i < noise.counts().size(); ++i) {
    CHECK(noise.counts()[i] == vllm_test::kLtx2VideoDecNoiseCounts[i]);
  }

  const double err = MaxAbsDiff(frames.data, vllm_test::kLtx2VideoDecGolden,
                                std::size(vllm_test::kLtx2VideoDecGolden));
  INFO("Conv video decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the NON-causal Conv video decoder matches upstream ltx_core") {
  // `causal=False` is UPSTREAM'S OWN DEFAULT (`causal: bool = False`,
  // conv_video_decoder.py:184, and the class docstring calls it the standard
  // decoder), yet every other video arm in this file runs causal=True — so the
  // default configuration was the one nothing executed.
  //
  // It is a DIFFERENT padding rule, not a disabled one: CausalConv3d replicates
  // the FIRST and LAST frame (kernel-1)/2 times each instead of putting kernel-1
  // copies of frame 0 on the left (convolution.py:266-317). The frame count comes
  // out identical either way, which is exactly why getting it wrong shifts the
  // whole clip while every shape assertion still passes.
  // Shares the causal arm's WEIGHTS, INPUT and NOISE stream deliberately, so the
  // padding rule is the ONLY difference between the two goldens.
  vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  cfg.causal = false;
  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecNcParamNames, vllm_test::kLtx2VideoDecNcParamCounts,
                std::size(vllm_test::kLtx2VideoDecNcParamNames));

  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  // The INPUT is the shared one — only the decoder's causality differs, so any
  // divergence is attributable to the padding rule and nothing else.
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);

  GoldenNoise noise;
  const vllm::Ltx2VideoFrames frames =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise);
  CHECK(frames.channels == vllm_test::kLtx2VideoDecOutC);
  CHECK(frames.frames == vllm_test::kLtx2VideoDecNcOutT);
  CHECK(frames.height == vllm_test::kLtx2VideoDecNcOutH);
  CHECK(frames.width == vllm_test::kLtx2VideoDecNcOutW);
  // Same frame count as the causal arm — the shapes cannot tell them apart.
  CHECK(vllm_test::kLtx2VideoDecNcOutT == vllm_test::kLtx2VideoDecOutT);

  REQUIRE(static_cast<int64_t>(noise.counts().size()) == vllm_test::kLtx2VideoDecNcNoiseDraws);
  for (size_t i = 0; i < noise.counts().size(); ++i) {
    CHECK(noise.counts()[i] == vllm_test::kLtx2VideoDecNcNoiseCounts[i]);
  }

  const double err = MaxAbsDiff(frames.data, vllm_test::kLtx2VideoDecNcGolden,
                                std::size(vllm_test::kLtx2VideoDecNcGolden));
  INFO("non-causal Conv video decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // Same weights, same latent, same noise — so if the two arms agreed, `causal`
  // would be doing nothing and this whole arm would gate nothing. Upstream asserts
  // the same thing in the generator.
  GoldenNoise causal_noise;
  vllm::Ltx2ConvVideoDecoderConfig causal_cfg = ReducedVideoDecoderConfig();
  ParamBag causal_bag = BuildVideoDecoderParams(causal_cfg);
  const vllm::Ltx2VideoFrames causal_frames = vllm::Ltx2ConvVideoDecode(
      causal_cfg, causal_bag.weights, latent, lc, lt, lh, lw, &causal_noise);
  REQUIRE(causal_frames.data.size() == frames.data.size());
  CHECK(causal_frames.data != frames.data);
}

TEST_CASE("ltx2 vae: the decode's convolution accumulates in f32, the width torch uses") {
  // THE ONE DEFECT THE GOLDENS STRUCTURALLY CANNOT REPORT, so it gets its own
  // instrument (issue #1008, .agents/specs/ltx25-decode-dtype.md).
  //
  // Every golden in this file is vacuous on accumulator WIDTH by construction:
  // scripts/gen-ltx2-vae-goldens.py:223 casts every upstream parameter with
  // `values.astype(np.float32)`, so the oracle ran f32 end to end and a dtype
  // comparison against it compares nothing. A decode that accumulated in f64 —
  // as this one did until #1008 — stays numerically plausible, keeps every
  // golden green, and moves twice the bytes. AGENTS.md says a token gate cannot
  // detect a dtype that is too wide; this case is what detects it here.
  //
  // THE REDUCTION IS ENGINEERED SO THE TWO WIDTHS ARE SEPARABLE, and the
  // expected value is UPSTREAM'S, not a recording of what this port emits.
  // MEASURED with torch 2.11.0 on exactly the taps below:
  //   F.conv3d, f32 tensors  -> 0.0
  //   F.conv3d, bf16 tensors -> 0.0   (upstream's own dtype, distilled.py:109)
  //   F.conv3d, f64 tensors  -> 2.500000014901161
  // Why 0.0 holds for ANY f32 summation order: half an ulp of 1e8 is 4.0, and
  // the 25 small taps sum to 2.5, so every partial sum rounds back to exactly
  // 1e8 and the closing -1e8 cancels it exactly. An f64 accumulator keeps the
  // 2.5. There is no order in which an f32 accumulator does.
  constexpr float kBig = 1e8f;
  constexpr float kSmall = 0.1f;

  vllm::Ltx2ConvVideoDecoderConfig cfg;
  cfg.prefix = "ltx2.videodec.accwidth.";
  cfg.in_channels = 1;
  cfg.out_channels = 1;
  cfg.patch_size = 1;
  cfg.base_channels = 2;
  cfg.causal = false;
  cfg.timestep_conditioning = false;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  // REPLICATE, not the fixtures' reflect: with a uniform input every conv tap
  // then reads exactly 1.0 at every output voxel INCLUDING the borders, so one
  // reduction is repeated everywhere and no voxel is a special case. `kZeros`
  // would zero the border taps and break the derivation below.
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReplicate;
  // No blocks at all. The decode is then conv_in -> PixelNorm -> SiLU ->
  // conv_out, which is short enough to carry an ANALYTIC expectation end to end
  // rather than a checked-in vector that only records today's behaviour.
  cfg.decoder_blocks = {};

  const std::string p = cfg.prefix;
  vllm::Ltx2VaeWeights weights;
  weights.tensors[p + "per_channel_statistics.std-of-means"] = {1.0f};
  weights.tensors[p + "per_channel_statistics.mean-of-means"] = {0.0f};

  // conv_in is [out=2, in=1, 3, 3, 3]. Output channel 0 carries the separable
  // reduction in the exact order CausalConv3d walks it (ic, then a, b, d, so
  // flat 0..26); output channel 1 is all-zero with a bias of 1, purely to give
  // PixelNorm a second channel with a known norm.
  std::vector<float> conv_in(2 * 1 * 27, 0.0f);
  conv_in[0] = kBig;
  for (size_t i = 1; i < 26; ++i) conv_in[i] = kSmall;
  conv_in[26] = -kBig;
  weights.tensors[p + "conv_in.conv.weight"] = conv_in;
  weights.tensors[p + "conv_in.conv.bias"] = {0.0f, 1.0f};

  // conv_out is [out=1, in=2, 3, 3, 3]; it selects channel 0's centre tap and
  // nothing else, so it forwards the value under test without adding a second
  // reduction that could mask it.
  //
  // THE BIAS IS 7, AND THAT IS NOT COSMETIC. With a bias of 0 the f32 answer is
  // zero, and a decode that never ran — a deleted call site handing back a
  // zero-filled buffer — produces zero as well, so the case would pass while
  // measuring nothing. This was not hypothetical: the reachability mutation was
  // run, the stub returned zeros, and an earlier draft of this case PASSED.
  // Offsetting the expectation off zero is what separates "accumulated in f32"
  // from "nothing reached this at all".
  constexpr float kOutBias = 7.0f;
  std::vector<float> conv_out(1 * 2 * 27, 0.0f);
  conv_out[13] = 1.0f;  // ic=0, a=b=d=1
  weights.tensors[p + "conv_out.conv.weight"] = conv_out;
  weights.tensors[p + "conv_out.conv.bias"] = {kOutBias};

  const int64_t lt = 1, lh = 2, lw = 2;
  const std::vector<float> latent(static_cast<size_t>(lt * lh * lw), 1.0f);

  vllm::Ltx2TileSizeConfig tiling;
  tiling.frames = vllm::Ltx2DimensionSizeConfig{10000, 0};
  tiling.height = vllm::Ltx2DimensionSizeConfig{10000, 0};
  tiling.width = vllm::Ltx2DimensionSizeConfig{10000, 0};

  // ENTERS THROUGH THE PRODUCTION ENTRY POINT. `Ltx2VideoDecodeStreaming` is
  // what the render path calls (src/vllm/multimodal/ltx2_video.cpp:3258), and it
  // reaches Ltx2ConvVideoDecode through ltx2_video_vae_tiled.cpp:113. Deleting
  // that call site turns this case RED, which is the reachability proof; a case
  // that called Ltx2ConvVideoDecode directly would prove only that the function
  // works. A null noise stream is deliberate: this configuration must not draw,
  // and the decode's own VT_CHECK fails loudly if that ever changes.
  int64_t chunks = 0;
  vllm::Ltx2VideoFrames got;
  vllm::Ltx2VideoDecodeStreaming(
      vllm::Ltx2VideoDecoderKind::kConv, cfg, weights, latent, cfg.in_channels, lt, lh, lw,
      /*noise=*/nullptr, tiling, [&](const vllm::Ltx2VideoChunk& chunk) {
        ++chunks;
        got = chunk.frames;
      });
  REQUIRE(chunks == 1);
  REQUIRE(got.data.size() == 4u);

  // THE DERIVATION, so a reader can check the number rather than trust it.
  //   conv_in ch0 = 0 (f32) or 2.5 (f64);  conv_in ch1 = bias = 1
  //   PixelNorm over 2 channels: f32 mean_sq = 0.5, and 0 * anything = 0
  //   SiLU(0) = 0;  conv_out forwards ch0 and adds its bias
  // so an f32 accumulator makes EVERY output element exactly the bias. An f64
  // one gives 2.5 -> 2.5/sqrt(3.625) = 1.31306 -> SiLU -> ~1.03473 on top of it,
  // a gap of 1.03, which is 200000x the golden tolerance. Nothing between the
  // two arms is a matter of tolerance.
  const std::vector<float> want(4, kOutBias);
  const double err = MaxAbsDiff(got.data, want.data(), got.data.size());
  INFO("conv accumulator width probe max|out - bias| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // The f64 answer stated as a value, so this case fails LOUDLY rather than
  // drifting if the accumulator is ever widened again. 1.03473 is what this
  // decode returned before #1008 narrowed it.
  CHECK(err < 1.0);
}

// ---------------------------------------------------------------------------
// LTX25-DECODE-THREADS (issue #1009): the decode had 20 cores and used one.
// .agents/specs/ltx25-decode-threads.md
// ---------------------------------------------------------------------------

// The fixture both threading cases decode. Deliberately shared, so the case that
// proves the dispatch happens and the case that proves the result does not
// depend on it are looking at the SAME work.
//
// It is the accumulator-width fixture's derivation at a size where the
// convolution has many output lines: `decoder_blocks` empty, patch_size 1,
// replicate padding and an all-ones latent, so every conv tap reads exactly 1.0
// at every output voxel including the borders and one reduction is repeated
// everywhere. conv_in channel 0 carries the f32/f64 separable reduction, channel
// 1 is all-zero weights with bias 1, conv_out selects channel 0's centre tap and
// adds a bias of 7.
struct Ltx2ThreadFixture {
  vllm::Ltx2ConvVideoDecoderConfig cfg;
  vllm::Ltx2VaeWeights weights;
  std::vector<float> latent;
  int64_t lt = 0, lh = 0, lw = 0;
};

Ltx2ThreadFixture MakeLtx2ThreadFixture() {
  Ltx2ThreadFixture f;
  f.cfg.prefix = "ltx2.videodec.threads.";
  f.cfg.in_channels = 1;
  f.cfg.out_channels = 1;
  f.cfg.patch_size = 1;
  // 24 output channels, so conv_in's parallel row count is 24 * out.t * out.h and
  // no thread count under test degenerates to one chunk.
  f.cfg.base_channels = 24;
  f.cfg.causal = false;
  f.cfg.timestep_conditioning = false;
  f.cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  f.cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReplicate;
  f.cfg.decoder_blocks = {};

  const std::string p = f.cfg.prefix;
  f.weights.tensors[p + "per_channel_statistics.std-of-means"] = {1.0f};
  f.weights.tensors[p + "per_channel_statistics.mean-of-means"] = {0.0f};

  // conv_in is [out=24, in=1, 3, 3, 3]. Channel 0 carries the separable
  // reduction in the exact order CausalConv3d walks it; channels 1..23 are
  // all-zero weights with bias 1, which gives PixelNorm 23 unit channels to
  // divide by and keeps channel 0 at zero on an f32 accumulator.
  constexpr float kBig = 1e8f;
  constexpr float kSmall = 0.1f;
  std::vector<float> conv_in(static_cast<size_t>(24 * 1 * 27), 0.0f);
  conv_in[0] = kBig;
  for (size_t i = 1; i < 26; ++i) conv_in[i] = kSmall;
  conv_in[26] = -kBig;
  f.weights.tensors[p + "conv_in.conv.weight"] = conv_in;
  std::vector<float> conv_in_bias(24, 1.0f);
  conv_in_bias[0] = 0.0f;
  f.weights.tensors[p + "conv_in.conv.bias"] = conv_in_bias;

  // conv_out is [out=1, in=24, 3, 3, 3]; it selects channel 0's centre tap and
  // nothing else. THE BIAS IS 7 for the reason #1008 recorded: with a bias of 0
  // the expected value is zero, and a decode that never ran hands back a
  // zero-filled buffer, so the case would pass while measuring nothing.
  std::vector<float> conv_out(static_cast<size_t>(1 * 24 * 27), 0.0f);
  conv_out[13] = 1.0f;  // ic = 0, a = b = d = 1
  f.weights.tensors[p + "conv_out.conv.weight"] = conv_out;
  f.weights.tensors[p + "conv_out.conv.bias"] = {7.0f};

  f.lt = 3;
  f.lh = 5;
  f.lw = 4;
  f.latent.assign(static_cast<size_t>(f.lt * f.lh * f.lw), 1.0f);
  return f;
}

// Untiled, so the ONE decode call the streaming entry point makes is the whole
// measurement.
vllm::Ltx2TileSizeConfig Ltx2ThreadUntiled() {
  vllm::Ltx2TileSizeConfig tiling;
  tiling.frames = vllm::Ltx2DimensionSizeConfig{10000, 0};
  tiling.height = vllm::Ltx2DimensionSizeConfig{10000, 0};
  tiling.width = vllm::Ltx2DimensionSizeConfig{10000, 0};
  return tiling;
}

vllm::Ltx2VideoFrames Ltx2ThreadDecode(const Ltx2ThreadFixture& f, int64_t* chunks) {
  // ENTERS THROUGH THE PRODUCTION ENTRY POINT. `Ltx2VideoDecodeStreaming` is what
  // the render path calls (src/vllm/multimodal/ltx2_video.cpp:3258) and it reaches
  // Ltx2ConvVideoDecode at ltx2_video_vae_tiled.cpp:113. A case that called
  // Ltx2ConvVideoDecode directly would prove the function works, never that the
  // shipping path reaches the threaded one.
  vllm::Ltx2VideoFrames got;
  *chunks = 0;
  vllm::Ltx2VideoDecodeStreaming(
      vllm::Ltx2VideoDecoderKind::kConv, f.cfg, f.weights, f.latent, f.cfg.in_channels, f.lt, f.lh,
      f.lw, /*noise=*/nullptr, Ltx2ThreadUntiled(), [&](const vllm::Ltx2VideoChunk& chunk) {
        ++*chunks;
        got = chunk.frames;
      });
  return got;
}

TEST_CASE("ltx2 vae: the decode DISPATCHES its convolutions to the CPU threadpool") {
  // THE CASE THAT IS RED BEFORE #1009, and the reason the determinism case below
  // is not enough on its own: two runs of a SERIAL decode are also bit-identical,
  // so a thread-count A/B is green on an implementation that never threads
  // anything. This case observes the dispatch itself.
  //
  // THE INSTRUMENT. `ParallelForRows` (src/vt/cpu/cpu_threadpool.cpp:413-458)
  // partitions its rows through the pool's shared work-stealing cursor: worker 0
  // seeds it with `ChunkSet(nth)` and every steal advances it with `ChunkAdd(1)`.
  // `ChunkAdd(0)` is a public non-mutating read of that cursor — `fetch_add(0)`
  // returns the current value. A fresh pool reads 0; a pool that has run at least
  // one multi-chunk partitioned dispatch reads at least `nth`. So the read is a
  // direct observation of "partitioned work ran on THIS pool", and the assertion
  // before the decode is its own positive control: the instrument demonstrably
  // reads zero when nothing has dispatched, which is exactly the state this row
  // removes.
  const Ltx2ThreadFixture f = MakeLtx2ThreadFixture();

  vt::cpu::Threadpool tp(4);
  REQUIRE(tp.NThreads() == 4);
  // Positive control: the cursor reads zero on a pool nothing has dispatched to.
  REQUIRE(tp.ChunkAdd(0) == 0);

  vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
  int64_t chunks = 0;
  vllm::Ltx2VideoFrames got;
  try {
    got = Ltx2ThreadDecode(f, &chunks);
  } catch (...) {
    vt::cpu::Threadpool::SwapForTesting(prev);
    throw;
  }
  const int cursor = tp.ChunkAdd(0);
  vt::cpu::Threadpool::SwapForTesting(prev);

  REQUIRE(chunks == 1);
  REQUIRE(got.data.size() == static_cast<size_t>(f.lt * f.lh * f.lw));

  // The decode ran, and it ran through the pool.
  CHECK(cursor > 0);

  // AND IT PRODUCED THE RIGHT PIXELS. The derivation, so a reader can check the
  // number rather than trust it: conv_in channel 0 accumulates to 0 in f32,
  // channels 1..23 to their bias of 1; PixelNorm leaves channel 0 at 0; SiLU(0)
  // is 0; conv_out forwards channel 0 and adds its bias. Every output element is
  // therefore exactly 7. A stubbed or deleted decode returns zeros and fails
  // here; an f64 accumulator keeps channel 0's 2.5 and fails here too.
  const std::vector<float> want(got.data.size(), 7.0f);
  const double err = MaxAbsDiff(got.data, want.data(), got.data.size());
  INFO("threaded decode max|out - 7| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the decode is BIT-IDENTICAL across thread counts") {
  // The determinism half of #1009. `ParallelForRows` steals work through an
  // atomic cursor, so which worker takes which output line is genuinely
  // non-deterministic run to run; the partition is over OUTPUT lines only and the
  // whole `ci * kernel^3` reduction stays inside one output element's body, so
  // every element is produced by the same instruction sequence on the same values
  // whatever the worker count is. That is the contract cpu_threadpool.h:39-43
  // states for the whole CPU backend, and this case holds the decode to it.
  //
  // A decode that returns different pixels at 1 thread and at 8 is a defect even
  // with every golden green, and no golden here would see it: the goldens run at
  // one thread count, the global pool's.
  //
  // Worker count 1 short-circuits ParallelForRows to `body(0, nr)` on the caller
  // (cpu_threadpool.cpp:423-426), so the 1-thread arm IS the pre-#1009 serial code
  // path byte for byte, and every other arm is compared against it.
  //
  // WHY 3 AND 5, STATED CORRECTLY. It is NOT that they fail to divide the row
  // counts — they divide both of this fixture's conv row counts exactly. conv_in
  // partitions 24*3*5 = 360 output lines and conv_out 1*3*5 = 15, and 3 and 5
  // divide each of those. The mechanism is that a chunk boundary is a function of
  // `nth * 4`, not of `nth`: ParallelForRows takes four chunks per thread
  // (cpu_threadpool.cpp:428-431), so `dr = ceil(nr / nchunk)` (:443) with
  // `nchunk = ceil(nr / ceil(nr / (nth*4)))`. At nr = 360 that is a stride of 45
  // at 2 workers, 30 at 3, 18 at 5 and 12 at 8 — four DIFFERENT partitions of the
  // same output, which is what the memcmp needs. Do NOT "fix" the fixture's row
  // counts to make them indivisible by 3 and 5: that would change the shape for a
  // reason that was never true, and 4 distinct strides is the property that
  // matters.
  const Ltx2ThreadFixture f = MakeLtx2ThreadFixture();

  std::vector<float> base;
  for (int nth : {1, 2, 3, 5, 8}) {
    vt::cpu::Threadpool tp(nth);
    vt::cpu::Threadpool* prev = vt::cpu::Threadpool::SwapForTesting(&tp);
    int64_t chunks = 0;
    vllm::Ltx2VideoFrames got;
    try {
      got = Ltx2ThreadDecode(f, &chunks);
    } catch (...) {
      vt::cpu::Threadpool::SwapForTesting(prev);
      throw;
    }
    vt::cpu::Threadpool::SwapForTesting(prev);

    REQUIRE(chunks == 1);
    REQUIRE(got.data.size() == static_cast<size_t>(f.lt * f.lh * f.lw));
    if (base.empty()) {
      // NOT DEGENERATE, so a stubbed decode cannot satisfy the comparison below.
      // An all-zero buffer is bit-identical to another all-zero buffer, so
      // "every arm agrees" is a vacuous statement about a decode that never ran.
      // This fixture's answer is 7 everywhere, which no absent computation
      // produces.
      const std::vector<float> want(got.data.size(), 7.0f);
      REQUIRE(MaxAbsDiff(got.data, want.data(), got.data.size()) <= kLtx2GoldenTol);
      base = got.data;
    } else {
      INFO("worker count " << nth);
      CHECK(std::memcmp(base.data(), got.data.data(), base.size() * sizeof(float)) == 0);
    }
  }
}

TEST_CASE("ltx2 vae: the video decoder's norm_eps is gated where it BINDS") {
  // THE ARM THAT MAKES `Ltx2ConvVideoDecoderConfig::norm_eps` NUMERICALLY
  // REACHABLE, and the correction of a record that said it was not.
  //
  // The earlier claim — that this constant is invisible because upstream discards
  // it on the PixelNorm arm — is FALSE. `ResnetBlock3D.__init__` builds
  // `norm3 = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=eps)`
  // whenever `in_channels != out_channels` (resnet.py:93-97), REGARDLESS of
  // `norm_layer`, and `forward` applies it to the residual (resnet.py:178). Every
  // `res_x_y` block therefore reads it, and the shipped section-5 arm above has
  // one, at `up_blocks.4.norm3`.
  //
  // What was true is a statement about the FIXTURE, not the constant. norm3
  // divides by `sqrt(var + eps)` over all of (C, T, H, W); five blocks deep that
  // variance is ~0.2, so on section 5's golden 1e-6 -> 1e-4 moves 1.8e-6 — under
  // the 5e-6 band — while 1e-6 -> 1.0 moves 1.6e-2. A 100x error passed because
  // the denominator was large, which is an accident of scale and not a property
  // worth recording as coverage.
  //
  // So this arm removes the accident. ONE `res_x_y` block puts norm3 directly
  // behind conv_in, and a latent at a tenth of the usual scale leaves it a
  // variance of ~5e-3 to compete with. `kLtx2VideoDecEpsZeroMove` is what the
  // ORACLE measured for the mutation the other arms are blindest to — removing
  // the epsilon entirely, which moves section 5's golden by 5.4e-7, a tenth of
  // the band — so the sensitivity is gated here rather than narrated.
  REQUIRE(vllm_test::kLtx2VideoDecEpsZeroMove > 10.0 * kLtx2GoldenTol);

  vllm::Ltx2ConvVideoDecoderConfig cfg;
  cfg.in_channels = 6;
  cfg.out_channels = 3;
  cfg.patch_size = 2;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.causal = true;
  cfg.timestep_conditioning = false;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReflect;
  cfg.base_channels = 8;
  cfg.prefix = "ltx2.videodeceps.";
  cfg.decoder_blocks = {{"res_x_y", 1, 2, false, false}};
  // The constant under test is the FIELD DEFAULT, never an override — an arm that
  // set it explicitly would gate the plumbing and not the value.
  CHECK(cfg.norm_eps == doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));

  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecEpsParamNames,
                vllm_test::kLtx2VideoDecEpsParamCounts,
                std::size(vllm_test::kLtx2VideoDecEpsParamNames));
  // norm3 exists only because the block changes channel count; without these two
  // parameters the manifest would match a decoder that never reads the epsilon.
  CHECK(bag.weights.Has("ltx2.videodeceps.up_blocks.0.norm3.weight"));
  CHECK(bag.weights.Has("ltx2.videodeceps.up_blocks.0.norm3.bias"));

  const int64_t lc = vllm_test::kLtx2VideoDecEpsLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecEpsLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecEpsLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecEpsLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodeceps.input", lc * lt * lh * lw,
                                              vllm_test::kLtx2VideoDecEpsLatentScale);

  GoldenNoise noise("ltx2.videodeceps.");
  const vllm::Ltx2VideoFrames frames =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise);
  CHECK(frames.channels == vllm_test::kLtx2VideoDecEpsOutC);
  CHECK(frames.frames == vllm_test::kLtx2VideoDecEpsOutT);
  CHECK(frames.height == vllm_test::kLtx2VideoDecEpsOutH);
  CHECK(frames.width == vllm_test::kLtx2VideoDecEpsOutW);
  // Neither timestep conditioning nor an inject_noise block, so upstream calls
  // torch.randn zero times and this port must draw nothing either.
  CHECK(noise.counts().empty());

  const double err = MaxAbsDiff(frames.data, vllm_test::kLtx2VideoDecEpsGolden,
                                std::size(vllm_test::kLtx2VideoDecEpsGolden));
  INFO("norm_eps-binding video decoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: video temporal causality is one-sided, proven by perturbation") {
  // The trap this catches: putting temporal padding on BOTH sides of a causal
  // Conv3d — or zero-padding it the way MiniMax-H3's Conv3d does instead of
  // replicating frame 0 — still yields a finite, plausible clip.
  //
  // The probe runs on a block list WITHOUT `res_x_y` and without timestep
  // conditioning, because `res_x_y`'s shortcut norm is a one-group GroupNorm over
  // (C, T, H, W) whose statistics span TIME (resnet.py:93-97): with it in the
  // stack nothing is causal however right the padding is. Section 5b records
  // upstream's OWN reach for this stripped config, so the expected window is the
  // oracle's, not a claim the port makes about itself.
  vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  cfg.timestep_conditioning = false;
  cfg.prefix = "ltx2.videodeccausal.";
  cfg.decoder_blocks = {
      {"res_x", 1, 0, false, false},
      {"compress_all", 1, 1, false, /*residual=*/false},
      {"res_x", 1, 0, false, false},
  };
  ParamBag bag = BuildVideoDecoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoDecCausalParamNames,
                vllm_test::kLtx2VideoDecCausalParamCounts,
                std::size(vllm_test::kLtx2VideoDecCausalParamNames));
  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);

  GoldenNoise noise_a;
  const vllm::Ltx2VideoFrames base =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, latent, lc, lt, lh, lw, &noise_a);

  std::vector<float> bumped = latent;
  for (int64_t c = 0; c < lc; ++c) {
    for (int64_t h = 0; h < lh; ++h) {
      for (int64_t w = 0; w < lw; ++w) {
        bumped[static_cast<size_t>(((c * lt + (lt - 1)) * lh + h) * lw + w)] += 5.0f;
      }
    }
  }
  GoldenNoise noise_b;
  const vllm::Ltx2VideoFrames moved =
      vllm::Ltx2ConvVideoDecode(cfg, bag.weights, bumped, lc, lt, lh, lw, &noise_b);
  REQUIRE(moved.data.size() == base.data.size());
  CHECK(base.frames == vllm_test::kLtx2VideoDecCausalOutT);
  // With timestep conditioning off and no inject_noise block, nothing may draw.
  CHECK(noise_a.counts().empty());
  CHECK(noise_b.counts().empty());

  const int64_t plane = base.height * base.width;
  int64_t first_moved = base.frames;
  int64_t last_moved = -1;
  for (int64_t t = 0; t < base.frames; ++t) {
    bool differs = false;
    for (int64_t c = 0; c < base.channels && !differs; ++c) {
      for (int64_t i = 0; i < plane; ++i) {
        const size_t idx = static_cast<size_t>((c * base.frames + t) * plane + i);
        if (base.data[idx] != moved.data[idx]) {
          differs = true;
          break;
        }
      }
    }
    if (differs) {
      if (first_moved == base.frames) first_moved = t;
      last_moved = t;
    }
  }
  INFO("frames moved by a last-latent-frame bump: [" << first_moved << ", " << last_moved << "]");
  CHECK(first_moved == vllm_test::kLtx2VideoDecCausalFirstMoved);
  CHECK(last_moved == vllm_test::kLtx2VideoDecCausalLastMoved);
  CHECK(first_moved > 0);   // the past cannot see the future
  CHECK(last_moved >= 0);   // and the bump must reach SOMETHING, or the probe is vacuous
}

TEST_CASE("ltx2 vae: the goldens carry the upstream revision they came from") {
  // AGENTS.md §"vLLM is the reference" requires a ported test to preserve the
  // upstream REVISION ANCHOR. Without one, a Lightricks change to (say)
  // video_vae/resnet.py plus a regeneration moves every number here with nothing
  // to say whether the PORT drifted or UPSTREAM did, and nothing to bisect from.
  //
  // The generator resolves `git -C <--ltx2> rev-parse HEAD` and emits it; this
  // asserts it equals the SHA the suite PINS, so regenerating against a different
  // checkout fails the gate instead of silently replacing the oracle. Advancing
  // the pin is a deliberate edit here and in the generator docstring — never a
  // side effect of regenerating.
  constexpr const char* kLtx2VaeUpstreamRevisionPin =
      "fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca";

  const std::string revision = vllm_test::kLtx2VaeUpstreamRevision;
  INFO("goldens were generated from Lightricks/LTX-2 @ " << revision);
  // "unknown" is what a tarball checkout with no git metadata yields, and it is
  // NOT an acceptable anchor: provenance you cannot bisect is not provenance.
  CHECK(revision.size() == 40);
  for (char c : revision) {
    CHECK(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
  }
  CHECK(revision == kLtx2VaeUpstreamRevisionPin);
}

TEST_CASE("ltx2 vae: every stabilizing epsilon is pinned to its upstream line") {
  // THE INVISIBLE-CONSTANT CLASS, and the PIN LIST that holds it. An epsilon that
  // exists to stabilize a division CAN be invisible to a reduced-dimension parity
  // gate: the deterministic stream produces O(1) activations, the guarded term
  // never binds, and the tensor comparison accepts ANY value — including 0.0 and
  // including one 100x off.
  //
  // Membership is a MEASURED, PER-ENTRY property, and it is NOT a property of
  // this list. Some entries below were mutated with every golden staying green.
  // Others are gated numerically by an arm that reaches them, and are pinned
  // anyway, because a pin catches the edit a golden cannot: a regeneration that
  // moves the constant and the goldens TOGETHER. Each entry says which it is, and
  // says it because the mutation was RUN, on this tree, with the numbers recorded
  // beside it.
  //
  // Adding a new constant without adding it here reopens the hole. Recording one
  // as invisible without mutating it reopens a worse one — this case has now
  // carried a wrong reachability verdict twice, which is the whole reason the
  // claim is per-entry and quantified rather than a sentence at the top.

  // ResnetBlock3D's `eps: float = 1e-6` (video_vae/resnet.py:31), handed to every
  // nn.GroupNorm it builds (resnet.py:44, 65, 94) and carried by UNetMidBlock3D as
  // `resnet_eps` (resnet.py:216). This is the norm `res_x_y`'s shortcut uses.
  //
  // CORRECTED. This one is NOT a member of the invisible class, and the record
  // that put it here said so for a reason that does not hold: `norm3` is built at
  // resnet.py:93-97 whenever `in_channels != out_channels`, REGARDLESS of
  // `norm_layer`, and applied at resnet.py:178 — so a PixelNorm decoder reads it
  // too, at every `res_x_y`. The 1e-6 -> 1e-4 mutation stayed green because
  // section 5's norm3 sits five blocks deep behind a variance of ~0.2, not
  // because nothing read the value. "ltx2 vae: the video decoder's norm_eps is
  // gated where it BINDS" is the arm that removes that accident of scale; the pin
  // stays because it still catches the edit a golden cannot, a regeneration that
  // moves the constant and the goldens together.
  CHECK(vllm::Ltx2ConvVideoDecoderConfig{}.norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));

  // `torch.clamp(mel, min=1e-5)` before the log (audio_vae/vocoder.py:515). The
  // member of the class that BINDS IN PRODUCTION, because real silence reaches it.
  // Mutation: 1e-5 -> 1e-8 left every golden green — until the saturating arm
  // below, which is the one place the reduced stream can be pushed into reaching
  // it.
  CHECK(vllm::kLtx2BweMelLogClamp == doctest::Approx(1e-5).epsilon(1e-12).scale(0.0));

  // Snake/SnakeBeta's `self.eps = 1e-9` (audio_vae/vocoder.py:198 and :221), the
  // stabilizer in `1 / (beta + eps)`. Shared with MiniMax-H3's BigVGAN, which is
  // why it lives in minimax_h3.h. Mutation: 1e-9 -> 0.0 left every golden green,
  // because beta is O(1) here and never approaches zero.
  CHECK(vllm::vocoder1d::kSnakeEps == doctest::Approx(1e-9).epsilon(1e-12).scale(0.0));

  // `_RMSNorm2D` is `F.normalize(x, dim=1) * (sqrt(C) * gamma)`
  // (video_vae/attention.py:11-30), so the floor is torch's `F.normalize` DEFAULT
  // eps of 1e-12 — an L2 normalize, not a mean-square RMS. Mutation: 1e-12 -> 0.0
  // left every golden green; it decides whether an all-zero channel vector
  // divides or produces NaN.
  CHECK(vllm::kLtx2RmsNorm2dEps == doctest::Approx(1e-12).epsilon(1e-12).scale(0.0));

  // The AUDIO VAE's GroupNorm eps, on BOTH halves. `build_normalization_layer`
  // passes `eps=1e-6` literally to `torch.nn.GroupNorm` (normalization.py:56),
  // the same line that gives PixelNorm its 1e-6 — so the two fields agree here
  // and, unlike the video VAE's pair below, are NOT deliberately different.
  //
  // These two were the fourth recurrence of this class, and worse than inert:
  // every audio arm ran `norm_type = kPixel`, so the GroupNorm branch was never
  // entered and the constant was never READ. 1e-6 -> 1e-4 on both left all 33
  // cases green. They are now reachable — the two group-norm arms above execute
  // the branch numerically — and pinned here as well, because a pin catches the
  // edit a golden cannot: replacing 1e-6 with the video VAE's 1e-8 while
  // regenerating would move the goldens and the arms would follow it.
  CHECK(vllm::Ltx2AudioDecoderConfig{}.norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));
  CHECK(vllm::Ltx2AudioEncoderConfig{}.norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));

  // The VIDEO VAE's GroupNorm eps on the ENCODER half, which phase L11 added and
  // this list did not grow to match. `_make_encoder_block` passes `resnet_eps=1e-6`
  // / `eps=1e-6` literally (video_vae.py:56, 66) and `conv_norm_out` takes
  // `eps=1e-6` (video_vae.py:240); there is no checkpoint key that moves it.
  //
  // NOT INVISIBLE — this entry arrived carrying the same wrong verdict the decoder
  // entry above did, for the same reason, and is corrected the same way. Encoder
  // arm A has a `res_x_y` block, so ResnetBlock3d builds norm3 (resnet.py:93-97,
  // applied at :178) exactly as the decoder does, and our port reads it at ONE
  // line for both halves: ltx2_video_vae.cpp:1051,1056 reach :405, which the
  // decoder reaches from :693,700. Measured on this tree: the field default
  // 1e-6 -> 1e-4 REDS two goldens at max|diff| = 4.38839e-05 against the 5e-6
  // band — "the video ENCODER (*_res family)" and "the video encoder CROPS a
  // frame count that is not 1 + k*factor". Forcing :405 to 1.0 reds those same
  // two at 0.150858, which is what IDENTIFIES norm3 as the reader: `norm_layer`
  // is kPixelNorm on both encoder arms, so neither ApplyNorm nor conv_norm_out
  // (ltx2_video_vae.cpp:1081-1087) ever enters a GroupNorm branch. Arm B holds no
  // `res_x_y` and therefore no norm3, and stays green throughout — the coverage
  // is real but PARTIAL, which is exactly what the pin is still for.
  CHECK(vllm::Ltx2ConvVideoEncoderConfig{}.norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));

  // And its PixelNorm eps, which is the video VAE's bare `PixelNorm()` DEFAULT of
  // 1e-8 (normalization.py:22) — NOT the audio VAE's 1e-6. The decoder-side pair
  // has its own case below; the encoder was missing from both.
  //
  // Also NOT INVISIBLE, and the more clearly so: PixelNorm IS the encoder's norm
  // on every arm, so this epsilon is a first-order term the whole way down rather
  // than a guard that never binds. Measured: 1e-8 -> 1e-6 REDS four encoder
  // goldens — 1.02744e-05 on the `*_res` family and on the frame-count crop,
  // 8.10623e-06 on encoder temporal causality, and 0.000175595 on "the video
  // ENCODER (strided convs, per_channel, reflect)".
  CHECK(vllm::Ltx2ConvVideoEncoderConfig{}.pixel_norm_eps ==
        doctest::Approx(1e-8).epsilon(1e-12).scale(0.0));
}

TEST_CASE("ltx2 vae: the BWE mel log clamp is gated where it actually binds") {
  // The ordinary BWE arm leaves `torch.clamp(mel, min=1e-5)` (vocoder.py:515)
  // inert — its raw mel minimum is ~4.4e-3, and it stays there even for a zero
  // input, because the vocoder's conv biases keep the waveform off silence. So
  // that arm can never move under a mutation of the constant.
  //
  // This arm attenuates mel_basis by 1e-4 (see the generator's `param_values`) so
  // EVERY bin lands under the clamp and the constant alone decides what the
  // bwe_generator consumes — which is what real silence does in production. The
  // saturation count comes from the generator, so the probe is proven saturated
  // rather than assumed to be.
  REQUIRE(vllm_test::kLtx2BweQuietSaturatedBins > 0);

  vllm::Ltx2VocoderBweConfig cfg;
  cfg.vocoder = ReducedVocoderConfig();
  cfg.vocoder.prefix = "ltx2.bwequiet.vocoder.";
  cfg.bwe_generator = ReducedVocoderConfig();
  cfg.bwe_generator.prefix = "ltx2.bwequiet.bwe_generator.";
  cfg.bwe_generator.resblock_kernel_sizes = {3};
  cfg.bwe_generator.resblock_dilation_sizes = {{1, 3, 5}};
  cfg.bwe_generator.upsample_rates = {4, 4};
  cfg.bwe_generator.upsample_kernel_sizes = {8, 8};
  cfg.bwe_generator.apply_final_activation = false;
  cfg.bwe_generator.output_sampling_rate = 32000;
  cfg.filter_length = 16;
  cfg.hop_length = 8;
  cfg.win_length = 16;
  cfg.n_mel_channels = 64;
  cfg.input_sampling_rate = 16000;
  cfg.output_sampling_rate = 32000;
  cfg.prefix = "ltx2.bwequiet.";

  ParamBag bag;
  PutVocoderParams(bag, cfg.vocoder);
  PutVocoderParams(bag, cfg.bwe_generator);
  bag.Put(cfg.prefix + "mel_stft.mel_basis", {cfg.n_mel_channels, cfg.filter_length / 2 + 1});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.forward_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  bag.Put(cfg.prefix + "mel_stft.stft_fn.inverse_basis",
          {(cfg.filter_length / 2 + 1) * 2, 1, cfg.filter_length});
  CheckManifest(bag, vllm_test::kLtx2BweQuietParamNames, vllm_test::kLtx2BweQuietParamCounts,
                std::size(vllm_test::kLtx2BweQuietParamNames));

  const std::vector<float> mel =
      Ltx2Input("ltx2.voc.input", 2 * vllm_test::kLtx2VocFrames * vllm_test::kLtx2VocMelBins, 1.0);
  int64_t out_samples = 0;
  const std::vector<float> wave =
      vllm::Ltx2VocoderWithBweForward(cfg, bag.weights, mel, 2, vllm_test::kLtx2VocFrames,
                                      vllm_test::kLtx2VocMelBins, &out_samples);
  CHECK(out_samples == vllm_test::kLtx2BweQuietOutSamples);
  const double err =
      MaxAbsDiff(wave, vllm_test::kLtx2BweQuietGolden, std::size(vllm_test::kLtx2BweQuietGolden));
  INFO("saturated-clamp BWE max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the two PixelNorm epsilons stay different") {
  // This is a SOURCE-ANCHORED CONSTANT guard, and it is NOT the only thing holding
  // either epsilon: both are also reached numerically, MEASURED on this tree by
  // mutating the field defaults every arm runs.
  //
  //   Ltx2ConvVideoDecoderConfig::pixel_norm_eps  1e-8 -> 1e-6 (100x)
  //     REDS "ltx2 vae: the video decoder's norm_eps is gated where it BINDS"
  //     at max|diff| = 0.000169305 against the 5e-6 band.
  //   Ltx2AudioDecoderConfig::pixel_norm_eps      1e-6 -> 1e-4 (100x)
  //     REDS 5 goldens across three arms — "the audio decoder matches upstream
  //     ltx_core" (0.0120053), "the other three causality axes" (0.00461239,
  //     0.00302449, 0.00245912) and "pads the frequency axis" (0.0120053).
  //
  // An earlier revision of this comment said the video half "leaves every golden
  // green". That was true when it was written and false when the low-scale
  // norm_eps arm landed: at a tenth of the usual latent scale this epsilon is a
  // first-order term too, which ltx2_video_vae.h already records. The pin stays
  // for the reason a pin always earns — a regeneration that moves the constant
  // and the goldens together, which no value comparison can see.
  //
  // The values are not interchangeable upstream. The audio VAE reaches PixelNorm
  // through build_normalization_layer, which passes eps=1e-6
  // (common/normalization.py:58); the video VAE constructs `PixelNorm()` bare and
  // gets its 1e-8 default (common/normalization.py:22, from video_vae/resnet.py:46
  // and conv_video_decoder.py:243). "Unifying" them is the mistake this catches.
  CHECK(vllm::Ltx2AudioDecoderConfig{}.pixel_norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));
  CHECK(vllm::Ltx2ConvVideoDecoderConfig{}.pixel_norm_eps ==
        doctest::Approx(1e-8).epsilon(1e-12).scale(0.0));
  CHECK(vllm::Ltx2AudioDecoderConfig{}.pixel_norm_eps !=
        vllm::Ltx2ConvVideoDecoderConfig{}.pixel_norm_eps);

  // The two ENCODER halves phase L11 added carry the SAME split for the SAME
  // reason, and this case only ever held the decoder pair. All FOUR are reachable
  // by their own goldens — the encoder pair per d45bcb5fb, the decoder pair per
  // the numbers above — so this is not closing a silent hole. It keeps the case's
  // claim ("the two PixelNorm epsilons stay different") true of every config that
  // has the field, rather than of the two it happened to be written for.
  CHECK(vllm::Ltx2AudioEncoderConfig{}.pixel_norm_eps ==
        doctest::Approx(1e-6).epsilon(1e-12).scale(0.0));
  CHECK(vllm::Ltx2AudioEncoderConfig{}.pixel_norm_eps !=
        vllm::Ltx2ConvVideoEncoderConfig{}.pixel_norm_eps);
}

TEST_CASE("ltx2 vae: the diffusion video decoder is refused by name, never downgraded") {
  // .agents/specs/ltx-2-5.md section 0 item 2. A silent fall-back to the Conv
  // decoder would return a lower-quality render as if it were the requested one,
  // and no gate this project owns could tell.
  CHECK(vllm::Ltx2ParseVideoDecoderKind("CausalVideoAutoencoder") ==
        vllm::Ltx2VideoDecoderKind::kConv);
  CHECK(vllm::Ltx2ParseVideoDecoderKind("") == vllm::Ltx2VideoDecoderKind::kConv);
  CHECK(vllm::Ltx2ParseVideoDecoderKind("CausalDiffusionVAE") ==
        vllm::Ltx2VideoDecoderKind::kDiffusion);

  const vllm::Ltx2ConvVideoDecoderConfig cfg = ReducedVideoDecoderConfig();
  ParamBag bag = BuildVideoDecoderParams(cfg);
  const int64_t lc = vllm_test::kLtx2VideoDecLatentC;
  const int64_t lt = vllm_test::kLtx2VideoDecLatentT;
  const int64_t lh = vllm_test::kLtx2VideoDecLatentH;
  const int64_t lw = vllm_test::kLtx2VideoDecLatentW;
  const std::vector<float> latent = Ltx2Input("ltx2.videodec.input", lc * lt * lh * lw, 1.0);
  GoldenNoise noise;

  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2VideoDecode(vllm::Ltx2VideoDecoderKind::kDiffusion, cfg, bag.weights, latent, lc, lt,
                          lh, lw, &noise);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  // The message must NAME the missing piece, not just say "unsupported".
  CHECK(message.find("NADiffusionDecoder") != std::string::npos);
  CHECK(message.find("neighborhood") != std::string::npos);
  // And nothing may have been decoded on the way to the refusal.
  CHECK(noise.counts().empty());
}

// ===========================================================================
// PHASE L11 — the ENCODER halves, gated against the same upstream `ltx_core`
// executed at reduced dimensions by scripts/gen-ltx2-vae-goldens.py sections
// 6-8.
// ===========================================================================

namespace {

// The reduced VideoEncoder arm A the generator built (VIDEO_ENC_A_BLOCKS).
vllm::Ltx2ConvVideoEncoderConfig ReducedVideoEncoderConfigA() {
  vllm::Ltx2ConvVideoEncoderConfig cfg;
  cfg.in_channels = 3;
  cfg.out_channels = 4;
  cfg.patch_size = 2;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.latent_log_var = vllm::Ltx2LogVarianceType::kUniform;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kZeros;
  cfg.encoder_blocks = {
      {"res_x", 1, 0},
      {"compress_space_res", 1, 2},
      {"res_x_y", 1, 2},
      {"attn", 1, 0},
      {"compress_time_res", 1, 1},
  };
  cfg.prefix = "ltx2.videoenc.";
  return cfg;
}

// The reduced VideoEncoder arm B (VIDEO_ENC_B_BLOCKS) — the plain strided
// convolutions, `per_channel` log variance, and REFLECT spatial padding.
vllm::Ltx2ConvVideoEncoderConfig ReducedVideoEncoderConfigB() {
  vllm::Ltx2ConvVideoEncoderConfig cfg;
  cfg.in_channels = 3;
  cfg.out_channels = 4;
  cfg.patch_size = 1;
  cfg.norm_layer = vllm::Ltx2NormLayer::kPixelNorm;
  cfg.latent_log_var = vllm::Ltx2LogVarianceType::kPerChannel;
  cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kReflect;
  cfg.encoder_blocks = {
      {"compress_all", 1, 0},
      {"compress_space", 1, 0},
      {"compress_time", 1, 0},
      {"compress_all_x_y", 1, 2},
  };
  cfg.prefix = "ltx2.videoencb.";
  return cfg;
}

// Build the VideoEncoder's parameters in upstream state_dict ORDER
// (video_vae.py:194-262): per_channel_statistics, conv_in, the FORWARD block
// walk, conv_norm_out (GroupNorm arm only), conv_out. PixelNorm carries no
// parameters, which is why no norm tensor appears on the pixel arm.
ParamBag BuildVideoEncoderParams(const vllm::Ltx2ConvVideoEncoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;
  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.out_channels});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.out_channels});
  const int64_t patched_in = cfg.in_channels * cfg.patch_size * cfg.patch_size;
  bag.Put(p + "conv_in.conv.weight", {cfg.out_channels, patched_in, 3, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {cfg.out_channels});

  auto put_resnet = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch) {
    bag.Put(prefix + ".conv1.conv.weight", {out_ch, in_ch, 3, 3, 3});
    bag.Put(prefix + ".conv1.conv.bias", {out_ch});
    bag.Put(prefix + ".conv2.conv.weight", {out_ch, out_ch, 3, 3, 3});
    bag.Put(prefix + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      // make_linear_nd -> a 1x1x1 Conv3d, and norm3 -> GroupNorm(1) over the
      // INPUT channels (resnet.py:85-97). Both exist only when the widths differ.
      bag.Put(prefix + ".conv_shortcut.weight", {out_ch, in_ch, 1, 1, 1});
      bag.Put(prefix + ".conv_shortcut.bias", {out_ch});
      bag.Put(prefix + ".norm3.weight", {in_ch});
      bag.Put(prefix + ".norm3.bias", {in_ch});
    }
  };

  int64_t feature = cfg.out_channels;
  for (size_t i = 0; i < cfg.encoder_blocks.size(); ++i) {
    const vllm::Ltx2VideoEncoderBlock& block = cfg.encoder_blocks[i];
    const std::string bp = p + "down_blocks." + std::to_string(i);
    const int64_t multiplier = block.multiplier != 0 ? block.multiplier : 2;
    if (block.name == "res_x") {
      for (int64_t j = 0; j < block.num_layers; ++j) {
        put_resnet(bp + ".res_blocks." + std::to_string(j), feature, feature);
      }
    } else if (block.name == "res_x_y") {
      put_resnet(bp, feature, feature * multiplier);
      feature *= multiplier;
    } else if (block.name == "attn") {
      bag.Put(bp + ".norm.gamma", {feature});
      bag.Put(bp + ".to_qkv.weight", {3 * feature, feature, 1, 1});
      bag.Put(bp + ".to_qkv.bias", {3 * feature});
      bag.Put(bp + ".proj.weight", {feature, feature, 1, 1});
      bag.Put(bp + ".proj.bias", {feature});
    } else if (block.name == "compress_time" || block.name == "compress_space" ||
               block.name == "compress_all" || block.name == "compress_all_x_y") {
      const int64_t out = block.name == "compress_all_x_y" ? feature * multiplier : feature;
      bag.Put(bp + ".conv.weight", {out, feature, 3, 3, 3});
      bag.Put(bp + ".conv.bias", {out});
      feature = out;
    } else {
      // The *_res family: SpaceToDepthDownsample's conv emits
      // out_channels / prod(stride), which the fold multiplies back up.
      const int64_t st = block.name == "compress_space_res" ? 1 : 2;
      const int64_t ss = block.name == "compress_time_res" ? 1 : 2;
      const int64_t out = feature * multiplier;
      const int64_t conv_out = out / (st * ss * ss);
      bag.Put(bp + ".conv.conv.weight", {conv_out, feature, 3, 3, 3});
      bag.Put(bp + ".conv.conv.bias", {conv_out});
      feature = out;
    }
  }

  if (cfg.norm_layer == vllm::Ltx2NormLayer::kGroupNorm) {
    bag.Put(p + "conv_norm_out.weight", {feature});
    bag.Put(p + "conv_norm_out.bias", {feature});
  }
  int64_t conv_out_channels = cfg.out_channels;
  if (cfg.latent_log_var == vllm::Ltx2LogVarianceType::kPerChannel) {
    conv_out_channels *= 2;
  } else if (cfg.latent_log_var == vllm::Ltx2LogVarianceType::kUniform ||
             cfg.latent_log_var == vllm::Ltx2LogVarianceType::kConstant) {
    conv_out_channels += 1;
  }
  bag.Put(p + "conv_out.conv.weight", {conv_out_channels, feature, 3, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {conv_out_channels});
  return bag;
}

// The reduced AudioEncoder the generator built (AUDIO_ENC).
vllm::Ltx2AudioEncoderConfig ReducedAudioEncoderConfig() {
  vllm::Ltx2AudioEncoderConfig cfg;
  cfg.ch = 8;
  cfg.in_channels = 2;
  cfg.ch_mult = {1, 2, 4};
  cfg.num_res_blocks = 1;
  cfg.attn_resolutions = {8};
  cfg.resolution = 32;
  cfg.z_channels = 4;
  cfg.double_z = true;
  cfg.resamp_with_conv = true;
  cfg.mid_block_add_attention = true;
  cfg.norm_type = vllm::Ltx2NormType::kPixel;
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  cfg.prefix = "ltx2.audioenc.";
  return cfg;
}

// The reduced GROUP-NORM audio encoder the generator built (AUDIO_GROUP_ENC).
// Same two forced dimensions as the decoder's group arm, for the same reasons.
vllm::Ltx2AudioEncoderConfig ReducedAudioEncoderGroupConfig() {
  vllm::Ltx2AudioEncoderConfig cfg = ReducedAudioEncoderConfig();
  cfg.ch = 32;
  cfg.z_channels = vllm_test::kLtx2AudioEncGroupOutC;
  cfg.norm_type = vllm::Ltx2NormType::kGroup;
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kNone;
  cfg.prefix = "ltx2.audioencgroup.";
  return cfg;
}

// Build the AudioEncoder's parameters in upstream state_dict ORDER
// (audio_vae.py:118-188): per_channel_statistics, conv_in, then per level the
// `block` ModuleList, then that level's `attn` ModuleList, then `downsample`;
// then mid, then conv_out.
ParamBag BuildAudioEncoderParams(const vllm::Ltx2AudioEncoderConfig& cfg) {
  ParamBag bag;
  const std::string p = cfg.prefix;
  const int64_t levels = cfg.num_resolutions();
  const bool group = cfg.norm_type == vllm::Ltx2NormType::kGroup;

  bag.Put(p + "per_channel_statistics.std-of-means", {cfg.ch});
  bag.Put(p + "per_channel_statistics.mean-of-means", {cfg.ch});
  bag.Put(p + "conv_in.conv.weight", {cfg.ch, cfg.in_channels, 3, 3});
  bag.Put(p + "conv_in.conv.bias", {cfg.ch});

  // GroupNorm is affine and PixelNorm is parameter-free, so the `kGroup` arm
  // carries a weight/bias pair AHEAD of each convolution the norm precedes.
  auto put_norm = [&](const std::string& prefix, int64_t channels) {
    if (!group) return;
    bag.Put(prefix + ".weight", {channels});
    bag.Put(prefix + ".bias", {channels});
  };
  auto put_resnet = [&](const std::string& prefix, int64_t in_ch, int64_t out_ch) {
    put_norm(prefix + ".norm1", in_ch);
    bag.Put(prefix + ".conv1.conv.weight", {out_ch, in_ch, 3, 3});
    bag.Put(prefix + ".conv1.conv.bias", {out_ch});
    put_norm(prefix + ".norm2", out_ch);
    bag.Put(prefix + ".conv2.conv.weight", {out_ch, out_ch, 3, 3});
    bag.Put(prefix + ".conv2.conv.bias", {out_ch});
    if (in_ch != out_ch) {
      bag.Put(prefix + ".nin_shortcut.conv.weight", {out_ch, in_ch, 1, 1});
      bag.Put(prefix + ".nin_shortcut.conv.bias", {out_ch});
    }
  };
  auto put_attn = [&](const std::string& prefix, int64_t channels) {
    put_norm(prefix + ".norm", channels);
    for (const char* leaf : {"q", "k", "v", "proj_out"}) {
      bag.Put(prefix + "." + leaf + ".weight", {channels, channels, 1, 1});
      bag.Put(prefix + "." + leaf + ".bias", {channels});
    }
  };

  // `in_ch_mult = (1, *ch_mult)` (downsample.py:78-85): level i READS
  // ch * ch_mult[i-1] and WRITES ch * ch_mult[i].
  int64_t curr_res = cfg.resolution;
  int64_t block_in = cfg.ch;
  for (int64_t level = 0; level < levels; ++level) {
    const std::string sp = p + "down." + std::to_string(level);
    const int64_t block_out = cfg.ch * cfg.ch_mult[static_cast<size_t>(level)];
    const bool has_attn = std::find(cfg.attn_resolutions.begin(), cfg.attn_resolutions.end(),
                                    curr_res) != cfg.attn_resolutions.end();
    for (int64_t i = 0; i < cfg.num_res_blocks; ++i) {
      put_resnet(sp + ".block." + std::to_string(i), block_in, block_out);
      block_in = block_out;
    }
    if (has_attn) {
      for (int64_t i = 0; i < cfg.num_res_blocks; ++i) {
        put_attn(sp + ".attn." + std::to_string(i), block_out);
      }
    }
    if (level != levels - 1) {
      if (cfg.resamp_with_conv) {
        // Downsample's conv is a BARE nn.Conv2d, so its key is `.downsample.conv`
        // and NOT the `.conv.conv` a CausalConv2d produces.
        bag.Put(sp + ".downsample.conv.weight", {block_in, block_in, 3, 3});
        bag.Put(sp + ".downsample.conv.bias", {block_in});
      }
      curr_res /= 2;
    }
  }

  put_resnet(p + "mid.block_1", block_in, block_in);
  if (cfg.mid_block_add_attention) put_attn(p + "mid.attn_1", block_in);
  put_resnet(p + "mid.block_2", block_in, block_in);

  put_norm(p + "norm_out", block_in);
  const int64_t conv_out_channels = cfg.double_z ? 2 * cfg.z_channels : cfg.z_channels;
  bag.Put(p + "conv_out.conv.weight", {conv_out_channels, block_in, 3, 3});
  bag.Put(p + "conv_out.conv.bias", {conv_out_channels});
  return bag;
}

}  // namespace

TEST_CASE("ltx2 vae: the video ENCODER (*_res family) matches upstream ltx_core") {
  const vllm::Ltx2ConvVideoEncoderConfig cfg = ReducedVideoEncoderConfigA();
  ParamBag bag = BuildVideoEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoEncParamNames, vllm_test::kLtx2VideoEncParamCounts,
                std::size(vllm_test::kLtx2VideoEncParamNames));

  // The scale factors are derived from the block list, not hardcoded, and they
  // are what decides the frame-count crop below.
  CHECK(vllm::Ltx2VideoTemporalScaleFactor(cfg.encoder_blocks) ==
        vllm_test::kLtx2VideoEncTemporalFactor);
  CHECK(vllm::Ltx2VideoSpatialScaleFactor(cfg.encoder_blocks, cfg.patch_size) ==
        vllm_test::kLtx2VideoEncSpatialFactor);

  const int64_t c = vllm_test::kLtx2VideoEncInC;
  const int64_t t = vllm_test::kLtx2VideoEncInT;
  const int64_t h = vllm_test::kLtx2VideoEncInH;
  const int64_t w = vllm_test::kLtx2VideoEncInW;
  const std::vector<float> frames = Ltx2Input("ltx2.videoenc.input", c * t * h * w, 1.0);

  int64_t cropped = -1;
  const vllm::Ltx2LatentVolume latent =
      vllm::Ltx2ConvVideoEncode(cfg, bag.weights, frames, c, t, h, w, &cropped);
  CHECK(cropped == 0);
  CHECK(latent.batch == 1);
  CHECK(latent.channels == vllm_test::kLtx2VideoEncOutC);
  CHECK(latent.frames == vllm_test::kLtx2VideoEncOutT);
  CHECK(latent.height == vllm_test::kLtx2VideoEncOutH);
  CHECK(latent.width == vllm_test::kLtx2VideoEncOutW);

  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2VideoEncGolden,
                                std::size(vllm_test::kLtx2VideoEncGolden));
  INFO("video encoder (*_res) max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // `constant` keeps `sample[:, :-1]` as the means exactly as `uniform` does
  // (video_vae.py:315 vs :327), so with the same weights the two are IDENTICAL.
  // The generator asserts this against upstream; asserting it here too is what
  // makes the C++ arm's shared code path a claim rather than an assumption.
  vllm::Ltx2ConvVideoEncoderConfig const_cfg = cfg;
  const_cfg.latent_log_var = vllm::Ltx2LogVarianceType::kConstant;
  const vllm::Ltx2LatentVolume const_latent =
      vllm::Ltx2ConvVideoEncode(const_cfg, bag.weights, frames, c, t, h, w, nullptr);
  CHECK(const_latent.data == latent.data);
}

TEST_CASE("ltx2 vae: the video ENCODER (strided convs, per_channel, reflect) matches upstream") {
  // A DIFFERENT block vocabulary, a DIFFERENT log-variance mode and a DIFFERENT
  // spatial padding mode from arm A. The padding matters on its own: the encoder
  // DEFAULTS to `zeros` while the decoder defaults to `reflect`
  // (model_configurator.py:63-67 vs :90), so an implementation that copied the
  // decoder's default would pass arm A only by luck of the border.
  const vllm::Ltx2ConvVideoEncoderConfig cfg = ReducedVideoEncoderConfigB();
  ParamBag bag = BuildVideoEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoEncBParamNames, vllm_test::kLtx2VideoEncBParamCounts,
                std::size(vllm_test::kLtx2VideoEncBParamNames));

  CHECK(vllm::Ltx2VideoTemporalScaleFactor(cfg.encoder_blocks) ==
        vllm_test::kLtx2VideoEncBTemporalFactor);
  CHECK(vllm::Ltx2VideoSpatialScaleFactor(cfg.encoder_blocks, cfg.patch_size) ==
        vllm_test::kLtx2VideoEncBSpatialFactor);

  const int64_t c = vllm_test::kLtx2VideoEncBInC;
  const int64_t t = vllm_test::kLtx2VideoEncBInT;
  const int64_t h = vllm_test::kLtx2VideoEncBInH;
  const int64_t w = vllm_test::kLtx2VideoEncBInW;
  const std::vector<float> frames = Ltx2Input("ltx2.videoencb.input", c * t * h * w, 1.0);

  const vllm::Ltx2LatentVolume latent =
      vllm::Ltx2ConvVideoEncode(cfg, bag.weights, frames, c, t, h, w, nullptr);
  CHECK(latent.channels == vllm_test::kLtx2VideoEncBOutC);
  CHECK(latent.frames == vllm_test::kLtx2VideoEncBOutT);
  CHECK(latent.height == vllm_test::kLtx2VideoEncBOutH);
  CHECK(latent.width == vllm_test::kLtx2VideoEncBOutW);

  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2VideoEncBGolden,
                                std::size(vllm_test::kLtx2VideoEncBGolden));
  INFO("video encoder (strided convs) max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // The padding mode is load-bearing, so prove the two modes DIFFER on identical
  // weights and input. Without this the arm would gate the block vocabulary only.
  vllm::Ltx2ConvVideoEncoderConfig zeros_cfg = cfg;
  zeros_cfg.spatial_padding_mode = vllm::Ltx2PaddingMode::kZeros;
  const vllm::Ltx2LatentVolume zeros_latent =
      vllm::Ltx2ConvVideoEncode(zeros_cfg, bag.weights, frames, c, t, h, w, nullptr);
  REQUIRE(zeros_latent.data.size() == latent.data.size());
  CHECK(zeros_latent.data != latent.data);
}

TEST_CASE("ltx2 vae: the video encoder CROPS a frame count that is not 1 + k*factor") {
  // Upstream WARNS and crops rather than failing (video_vae.py:276-286), so a
  // caller that hands 6 frames to an 8-frame-factor encoder gets a shorter clip
  // and no error. The crop must therefore be reproduced exactly, and the count
  // reported so the caller can surface it.
  const vllm::Ltx2ConvVideoEncoderConfig cfg = ReducedVideoEncoderConfigA();
  ParamBag bag = BuildVideoEncoderParams(cfg);

  const int64_t c = vllm_test::kLtx2VideoEncInC;
  const int64_t kept = vllm_test::kLtx2VideoEncInT;
  const int64_t t = vllm_test::kLtx2VideoEncCropInT;
  const int64_t h = vllm_test::kLtx2VideoEncInH;
  const int64_t w = vllm_test::kLtx2VideoEncInW;
  REQUIRE(t == kept + vllm_test::kLtx2VideoEncCropDropped);

  // The generator built the 6-frame input by CONCATENATING one extra frame onto
  // arm A's 5-frame input, so the same concatenation here must reproduce arm A's
  // golden exactly once the tail is cropped.
  const std::vector<float> head = Ltx2Input("ltx2.videoenc.input", c * kept * h * w, 1.0);
  const std::vector<float> tail = Ltx2Input("ltx2.videoenc.tail", c * 1 * h * w, 1.0);
  std::vector<float> frames(static_cast<size_t>(c * t * h * w));
  for (int64_t ch = 0; ch < c; ++ch) {
    const size_t plane = static_cast<size_t>(h * w);
    std::copy(head.begin() + static_cast<ptrdiff_t>(ch * kept * h * w),
              head.begin() + static_cast<ptrdiff_t>((ch + 1) * kept * h * w),
              frames.begin() + static_cast<ptrdiff_t>(ch * t * h * w));
    std::copy(tail.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(ch) * plane),
              tail.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(ch + 1) * plane),
              frames.begin() + static_cast<ptrdiff_t>(ch * t * h * w + kept * h * w));
  }

  int64_t cropped = -1;
  const vllm::Ltx2LatentVolume latent =
      vllm::Ltx2ConvVideoEncode(cfg, bag.weights, frames, c, t, h, w, &cropped);
  CHECK(cropped == vllm_test::kLtx2VideoEncCropDropped);
  CHECK(latent.frames == vllm_test::kLtx2VideoEncOutT);

  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2VideoEncGolden,
                                std::size(vllm_test::kLtx2VideoEncGolden));
  INFO("cropped video encoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: video ENCODER temporal causality is one-sided, proven by perturbation") {
  // MEASURED, not assumed. The block list deliberately excludes `res_x_y` (whose
  // shortcut norm3 is a one-group GroupNorm whose statistics span TIME) and `attn`,
  // so what is left is decided by the causal padding alone. Upstream itself
  // supplies the expected window, and the generator asserts the probe both moves
  // something AND leaves an early frame untouched — a probe that moved everything
  // would be indistinguishable from a non-causal encoder.
  vllm::Ltx2ConvVideoEncoderConfig cfg = ReducedVideoEncoderConfigA();
  cfg.encoder_blocks = {{"res_x", 1, 0}, {"compress_time_res", 1, 1}};
  cfg.prefix = "ltx2.videoenccausal.";
  ParamBag bag = BuildVideoEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2VideoEncCausalParamNames,
                vllm_test::kLtx2VideoEncCausalParamCounts,
                std::size(vllm_test::kLtx2VideoEncCausalParamNames));

  const int64_t c = vllm_test::kLtx2VideoEncInC;
  const int64_t t = vllm_test::kLtx2VideoEncInT;
  const int64_t h = vllm_test::kLtx2VideoEncInH;
  const int64_t w = vllm_test::kLtx2VideoEncInW;
  const std::vector<float> frames = Ltx2Input("ltx2.videoenc.input", c * t * h * w, 1.0);

  const vllm::Ltx2LatentVolume base =
      vllm::Ltx2ConvVideoEncode(cfg, bag.weights, frames, c, t, h, w, nullptr);
  CHECK(base.frames == vllm_test::kLtx2VideoEncCausalOutT);
  const double err = MaxAbsDiff(base.data, vllm_test::kLtx2VideoEncCausalGolden,
                                std::size(vllm_test::kLtx2VideoEncCausalGolden));
  INFO("causal-arm video encoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // Perturb the LAST pixel frame and see which LATENT frames move.
  std::vector<float> bumped = frames;
  for (int64_t ch = 0; ch < c; ++ch) {
    for (int64_t i = 0; i < h * w; ++i) {
      bumped[static_cast<size_t>((ch * t + (t - 1)) * h * w + i)] += 5.0f;
    }
  }
  const vllm::Ltx2LatentVolume moved_out =
      vllm::Ltx2ConvVideoEncode(cfg, bag.weights, bumped, c, t, h, w, nullptr);
  REQUIRE(moved_out.frames == base.frames);

  int64_t first_moved = -1;
  int64_t last_moved = -1;
  const int64_t plane = base.height * base.width;
  for (int64_t f = 0; f < base.frames; ++f) {
    bool differs = false;
    for (int64_t ch = 0; ch < base.channels && !differs; ++ch) {
      for (int64_t i = 0; i < plane; ++i) {
        const size_t index = static_cast<size_t>((ch * base.frames + f) * plane + i);
        if (base.data[index] != moved_out.data[index]) {
          differs = true;
          break;
        }
      }
    }
    if (differs) {
      if (first_moved < 0) first_moved = f;
      last_moved = f;
    }
  }
  INFO("moved latent frames: [" << first_moved << ", " << last_moved << "]");
  CHECK(first_moved == vllm_test::kLtx2VideoEncCausalFirstMoved);
  CHECK(last_moved == vllm_test::kLtx2VideoEncCausalLastMoved);
  // The claim only means something if EARLIER frames genuinely did not move.
  CHECK(first_moved > 0);
}

TEST_CASE("ltx2 vae: latent_log_var=`none` is refused by name, never approximated") {
  // Upstream RAISES here — the generator executes it and records the exception
  // type — because `none` sizes conv_out at out_channels and then still chunks in
  // two (video_vae.py:335), leaving half as many mean channels as the per-channel
  // statistics carry. Inventing a semantics for it would be a silent divergence.
  CHECK(std::string(vllm_test::kLtx2VideoEncNoneRaises) == "RuntimeError");

  vllm::Ltx2ConvVideoEncoderConfig cfg = ReducedVideoEncoderConfigA();
  cfg.latent_log_var = vllm::Ltx2LogVarianceType::kNone;
  ParamBag bag = BuildVideoEncoderParams(cfg);
  const int64_t c = vllm_test::kLtx2VideoEncInC;
  const int64_t t = vllm_test::kLtx2VideoEncInT;
  const int64_t h = vllm_test::kLtx2VideoEncInH;
  const int64_t w = vllm_test::kLtx2VideoEncInW;
  const std::vector<float> frames = Ltx2Input("ltx2.videoenc.input", c * t * h * w, 1.0);

  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2ConvVideoEncode(cfg, bag.weights, frames, c, t, h, w, nullptr);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  CHECK(message.find("latent_log_var") != std::string::npos);
  CHECK(message.find("video_vae.py:335") != std::string::npos);
}

TEST_CASE("ltx2 vae: the AUDIO encoder matches upstream ltx_core") {
  const vllm::Ltx2AudioEncoderConfig cfg = ReducedAudioEncoderConfig();
  ParamBag bag = BuildAudioEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioEncParamNames, vllm_test::kLtx2AudioEncParamCounts,
                std::size(vllm_test::kLtx2AudioEncParamNames));

  const int64_t c = vllm_test::kLtx2AudioEncInC;
  const int64_t t = vllm_test::kLtx2AudioEncInT;
  const int64_t f = vllm_test::kLtx2AudioEncInF;
  const std::vector<float> spec = Ltx2Input("ltx2.audioenc.input", c * t * f, 1.0);

  const vllm::Ltx2AudioSpectrogram latent =
      vllm::Ltx2AudioEncoderForward(cfg, bag.weights, spec, c, t, f);
  CHECK(latent.channels == vllm_test::kLtx2AudioEncOutC);
  CHECK(latent.frames == vllm_test::kLtx2AudioEncOutT);
  CHECK(latent.mel_bins == vllm_test::kLtx2AudioEncOutF);

  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2AudioEncGolden,
                                std::size(vllm_test::kLtx2AudioEncGolden));
  INFO("audio encoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the AUDIO encoder in the SHIPPED no-attention configuration") {
  // The 2.x audio VAE metadata carries `attn_resolutions: []` and
  // `mid_block_add_attention: false`, so the attention arm above is NOT the model
  // anybody runs. Gating only that arm would gate a configuration this project
  // never loads.
  vllm::Ltx2AudioEncoderConfig cfg = ReducedAudioEncoderConfig();
  cfg.attn_resolutions.clear();
  cfg.mid_block_add_attention = false;
  cfg.prefix = "ltx2.audioencplain.";
  ParamBag bag = BuildAudioEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioEncPlainParamNames,
                vllm_test::kLtx2AudioEncPlainParamCounts,
                std::size(vllm_test::kLtx2AudioEncPlainParamNames));

  const int64_t c = vllm_test::kLtx2AudioEncInC;
  const int64_t t = vllm_test::kLtx2AudioEncInT;
  const int64_t f = vllm_test::kLtx2AudioEncInF;
  const std::vector<float> spec = Ltx2Input("ltx2.audioenc.input", c * t * f, 1.0);

  const vllm::Ltx2AudioSpectrogram latent =
      vllm::Ltx2AudioEncoderForward(cfg, bag.weights, spec, c, t, f);
  CHECK(latent.channels == vllm_test::kLtx2AudioEncPlainOutC);
  CHECK(latent.frames == vllm_test::kLtx2AudioEncPlainOutT);
  CHECK(latent.mel_bins == vllm_test::kLtx2AudioEncPlainOutF);

  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2AudioEncPlainGolden,
                                std::size(vllm_test::kLtx2AudioEncPlainGolden));
  INFO("audio encoder (shipped, no attention) max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 vae: the AUDIO encoder's average-pool downsample arm") {
  // `resamp_with_conv=False` replaces the strided convolution with avg_pool2d, and
  // upstream only permits it with causality NONE (downsample.py:28-29). It is a
  // different sampling lattice, so it gets its own golden AND its own refusal.
  vllm::Ltx2AudioEncoderConfig cfg = ReducedAudioEncoderConfig();
  cfg.resamp_with_conv = false;
  cfg.causality_axis = vllm::Ltx2CausalityAxis::kNone;
  cfg.prefix = "ltx2.audioencpool.";
  ParamBag bag = BuildAudioEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioEncPoolParamNames,
                vllm_test::kLtx2AudioEncPoolParamCounts,
                std::size(vllm_test::kLtx2AudioEncPoolParamNames));

  const int64_t c = vllm_test::kLtx2AudioEncInC;
  const int64_t t = vllm_test::kLtx2AudioEncInT;
  const int64_t f = vllm_test::kLtx2AudioEncInF;
  const std::vector<float> spec = Ltx2Input("ltx2.audioenc.input", c * t * f, 1.0);

  const vllm::Ltx2AudioSpectrogram latent =
      vllm::Ltx2AudioEncoderForward(cfg, bag.weights, spec, c, t, f);
  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2AudioEncPoolGolden,
                                std::size(vllm_test::kLtx2AudioEncPoolGolden));
  INFO("audio encoder (avg-pool) max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // The combination upstream forbids must be refused, not silently pooled.
  vllm::Ltx2AudioEncoderConfig bad = cfg;
  bad.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2AudioEncoderForward(bad, bag.weights, spec, c, t, f);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  CHECK(message.find("with_conv") != std::string::npos);
}

TEST_CASE("ltx2 vae: the GROUP-NORM audio encoder matches upstream ltx_core") {
  // The encoder half of the same hole: `Ltx2AudioEncoderConfig::norm_eps` was
  // added by phase L11 and, like the decoder's, could not be read by any arm,
  // because 7a/7b/7c all run `norm_type = kPixel`. Mutating it 1e-6 -> 1e-4 left
  // all 33 cases green. This arm executes the GroupNorm branch of
  // `build_normalization_layer` (normalization.py:56-57) at the only causality
  // `ResnetBlock` permits it on (resnet.py:130-131).
  const vllm::Ltx2AudioEncoderConfig cfg = ReducedAudioEncoderGroupConfig();
  ParamBag bag = BuildAudioEncoderParams(cfg);
  CheckManifest(bag, vllm_test::kLtx2AudioEncGroupParamNames,
                vllm_test::kLtx2AudioEncGroupParamCounts,
                std::size(vllm_test::kLtx2AudioEncGroupParamNames));

  const int64_t c = vllm_test::kLtx2AudioEncInC;
  const int64_t t = vllm_test::kLtx2AudioEncInT;
  const int64_t f = vllm_test::kLtx2AudioEncInF;
  const std::vector<float> spec = Ltx2Input("ltx2.audioenc.input", c * t * f, 1.0);

  const vllm::Ltx2AudioSpectrogram latent =
      vllm::Ltx2AudioEncoderForward(cfg, bag.weights, spec, c, t, f);
  CHECK(latent.channels == vllm_test::kLtx2AudioEncGroupOutC);
  CHECK(latent.frames == vllm_test::kLtx2AudioEncGroupOutT);
  CHECK(latent.mel_bins == vllm_test::kLtx2AudioEncGroupOutF);

  const double err = MaxAbsDiff(latent.data, vllm_test::kLtx2AudioEncGroupGolden,
                                std::size(vllm_test::kLtx2AudioEncGroupGolden));
  INFO("group-norm audio encoder max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  vllm::Ltx2AudioEncoderConfig bad = cfg;
  bad.causality_axis = vllm::Ltx2CausalityAxis::kHeight;
  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2AudioEncoderForward(bad, bag.weights, spec, c, t, f);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  CHECK(message.find("GroupNorm") != std::string::npos);
}

TEST_CASE("ltx2 vae: the slaney mel filterbank matches torchaudio") {
  // The filterbank is gated on its own because a wrong one makes every mel bin
  // wrong at once, and the resulting mismatch is impossible to localize from the
  // spectrogram alone. It is also where the slaney constants (200/3, 1000 Hz,
  // log(6.4)/27) become load-bearing: unlike an epsilon, they move THIS golden.
  {
    const int64_t n_freqs = vllm_test::kLtx2MelFreqs;
    const int64_t n_mels = vllm_test::kLtx2MelBins;
    const int64_t rate = vllm_test::kLtx2MelRate;
    const std::vector<float> fb = vllm::Ltx2SlaneyMelFilterbank(
        n_freqs, 0.0, static_cast<double>(rate) / 2.0, n_mels, rate);
    REQUIRE(static_cast<int64_t>(fb.size()) ==
            static_cast<int64_t>(std::size(vllm_test::kLtx2MelBasisGolden)));
    const double err =
        MaxAbsDiff(fb, vllm_test::kLtx2MelBasisGolden, std::size(vllm_test::kLtx2MelBasisGolden));
    INFO("mel filterbank max|diff| = " << err);
    CHECK(err <= kLtx2FilterTol);
  }
  {
    // An ODD sample rate is the only arm that can see torchaudio's
    // `torch.linspace(0, sample_rate // 2, n_freqs)` halving with INTEGER
    // division while `f_max` stays `sample_rate / 2.0`. With an even rate the two
    // coincide and the distinction is invisible.
    const int64_t n_freqs = vllm_test::kLtx2MelOddFreqs;
    const int64_t n_mels = vllm_test::kLtx2MelOddBins;
    const int64_t rate = vllm_test::kLtx2MelOddRate;
    CHECK(rate % 2 == 1);
    const std::vector<float> fb = vllm::Ltx2SlaneyMelFilterbank(
        n_freqs, 0.0, static_cast<double>(rate) / 2.0, n_mels, rate);
    REQUIRE(static_cast<int64_t>(fb.size()) ==
            static_cast<int64_t>(std::size(vllm_test::kLtx2MelOddBasisGolden)));
    const double err = MaxAbsDiff(fb, vllm_test::kLtx2MelOddBasisGolden,
                                  std::size(vllm_test::kLtx2MelOddBasisGolden));
    INFO("odd-rate mel filterbank max|diff| = " << err);
    CHECK(err <= kLtx2FilterTol);
  }
}

TEST_CASE("ltx2 vae: waveform_to_mel matches upstream AudioProcessor") {
  vllm::Ltx2AudioProcessorConfig cfg;
  cfg.target_sample_rate = vllm_test::kLtx2MelRate;
  cfg.mel_bins = vllm_test::kLtx2MelOutBins;
  cfg.mel_hop_length = vllm_test::kLtx2MelHop;
  cfg.n_fft = vllm_test::kLtx2MelNFft;

  const int64_t channels = vllm_test::kLtx2MelWaveChannels;
  const int64_t samples = vllm_test::kLtx2MelWaveSamples;
  const std::vector<float> wave = Ltx2Input("ltx2.mel.input", channels * samples, 0.5);

  int64_t frames = 0;
  const std::vector<float> mel =
      vllm::Ltx2WaveformToLogMel(cfg, wave, channels, samples, cfg.target_sample_rate, &frames);
  CHECK(frames == vllm_test::kLtx2MelOutFrames);
  REQUIRE(static_cast<int64_t>(mel.size()) ==
          static_cast<int64_t>(std::size(vllm_test::kLtx2MelGolden)));
  const double err = MaxAbsDiff(mel, vllm_test::kLtx2MelGolden, std::size(vllm_test::kLtx2MelGolden));
  INFO("waveform_to_mel max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // A rate that does not match is REFUSED, because upstream would RESAMPLE and
  // this project does not carry that resampler. Treating the samples as if they
  // were already at the target rate conditions on audio that is pitched wrong.
  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2WaveformToLogMel(cfg, wave, channels, samples, 44100, nullptr);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  CHECK(message.find("resample") != std::string::npos);
  CHECK(message.find("44100") != std::string::npos);
}

TEST_CASE("ltx2 vae: SILENCE saturates the mel log clamp, and the clamp is pinned") {
  // The invisible-constant class, with the one arm that ESCAPES it. A well-scaled
  // waveform never reaches `torch.clamp(mel, min=1e-5)` (audio_vae/ops.py:52), so
  // the arm above cannot see the constant at all — but real silence saturates
  // every bin, and then the constant ALONE decides what the encoder is handed.
  // The generator asserts upstream's raw mel maximum is genuinely below the clamp.
  CHECK(vllm::kLtx2AudioMelLogClamp == doctest::Approx(1e-5).epsilon(1e-12).scale(0.0));
  // And the video encoder's constant log-variance, which no MEANS-only golden can
  // ever reach (video_vae.py:328).
  CHECK(vllm::kLtx2EncoderApproxLnZero == doctest::Approx(-30.0).epsilon(1e-12).scale(0.0));

  vllm::Ltx2AudioProcessorConfig cfg;
  cfg.target_sample_rate = vllm_test::kLtx2MelRate;
  cfg.mel_bins = vllm_test::kLtx2MelOutBins;
  cfg.mel_hop_length = vllm_test::kLtx2MelHop;
  cfg.n_fft = vllm_test::kLtx2MelNFft;

  const int64_t channels = vllm_test::kLtx2MelWaveChannels;
  const int64_t samples = vllm_test::kLtx2MelWaveSamples;
  const std::vector<float> silence(static_cast<size_t>(channels * samples), 0.0f);

  int64_t frames = 0;
  const std::vector<float> mel =
      vllm::Ltx2WaveformToLogMel(cfg, silence, channels, samples, cfg.target_sample_rate, &frames);
  CHECK(frames == vllm_test::kLtx2MelQuietFrames);
  const double err =
      MaxAbsDiff(mel, vllm_test::kLtx2MelQuietGolden, std::size(vllm_test::kLtx2MelQuietGolden));
  INFO("silence mel max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);

  // Every value must BE the clamp's logarithm — that is what makes this arm a
  // gate on the constant rather than on the spectrogram.
  for (float value : mel) {
    CHECK(static_cast<double>(value) ==
          doctest::Approx(std::log(vllm::kLtx2AudioMelLogClamp)).epsilon(1e-6).scale(0.0));
  }
}

// ===========================================================================
// PHASE L11 — the CONDITIONING ITEMS: what the encoders' output is FOR.
// Gated against the upstream `ConditioningItem` classes executed through the
// real `VideoLatentTools` / `AudioLatentTools` (generator section 9).
// ===========================================================================

namespace {

vllm::Ltx2VideoLatentShape CondVideoTarget() {
  vllm::Ltx2VideoLatentShape shape;
  shape.batch = 1;
  shape.channels = 4;
  shape.frames = 3;
  shape.height = 2;
  shape.width = 2;
  return shape;
}

constexpr int64_t kCondPatch = 1;
constexpr double kCondFps = 8.0;

vllm::Ltx2AudioLatentShape CondAudioTarget() {
  vllm::Ltx2AudioLatentShape shape;
  shape.batch = 1;
  shape.channels = 2;
  shape.frames = 4;
  shape.mel_bins = 2;
  return shape;
}

vllm::Ltx2LatentVolume CondVolume(const std::string& name, int64_t channels, int64_t frames,
                                  int64_t height, int64_t width) {
  vllm::Ltx2LatentVolume volume;
  volume.batch = 1;
  volume.channels = channels;
  volume.frames = frames;
  volume.height = height;
  volume.width = width;
  volume.data = Ltx2Input(name, channels * frames * height * width, 1.0);
  return volume;
}

// Every conditioning arm checks the SAME four fields, and the noisy tensor is
// checked too — leaving it out is exactly how the diffusers-style "write the
// clean tokens into the noisy tensor as well" divergence would slip through.
void CheckState(const vllm::Ltx2LatentState& state, int64_t want_tokens, int64_t want_width,
                int64_t want_pos_dims, const float* clean, size_t clean_size, const float* latent,
                size_t latent_size, const float* mask, size_t mask_size, const float* positions,
                size_t positions_size, const char* label) {
  CHECK(state.tokens == want_tokens);
  CHECK(state.width == want_width);
  CHECK(state.pos_dims == want_pos_dims);
  REQUIRE(state.clean.size() == clean_size);
  REQUIRE(state.latent.size() == latent_size);
  REQUIRE(state.mask.size() == mask_size);
  REQUIRE(state.positions.size() == positions_size);
  const double clean_err = MaxAbsDiff(state.clean, clean, clean_size);
  const double latent_err = MaxAbsDiff(state.latent, latent, latent_size);
  const double mask_err = MaxAbsDiff(state.mask, mask, mask_size);
  const double pos_err = MaxAbsDiff(state.positions, positions, positions_size);
  INFO(label << " clean=" << clean_err << " latent=" << latent_err << " mask=" << mask_err
             << " positions=" << pos_err);
  CHECK(clean_err <= kLtx2GoldenTol);
  CHECK(latent_err <= kLtx2GoldenTol);
  CHECK(mask_err <= kLtx2GoldenTol);
  CHECK(pos_err <= kLtx2GoldenTol);
}

}  // namespace

TEST_CASE("ltx2 conditioning: the initial video latent state matches upstream VideoLatentTools") {
  std::vector<float> keyframes_mask;
  const vllm::Ltx2ScaleFactors factors;
  const vllm::Ltx2LatentState state = vllm::Ltx2CreateVideoLatentState(
      CondVideoTarget(), kCondPatch, factors, kCondFps, /*causal_fix=*/true,
      /*initial_latent=*/nullptr, &keyframes_mask);

  CheckState(state, vllm_test::kLtx2CondVideoBaseTokens, vllm_test::kLtx2CondVideoBaseWidth,
             vllm_test::kLtx2CondVideoBasePosDims, vllm_test::kLtx2CondVideoBaseClean,
             std::size(vllm_test::kLtx2CondVideoBaseClean), vllm_test::kLtx2CondVideoBaseLatent,
             std::size(vllm_test::kLtx2CondVideoBaseLatent), vllm_test::kLtx2CondVideoBaseMask,
             std::size(vllm_test::kLtx2CondVideoBaseMask), vllm_test::kLtx2CondVideoBasePositions,
             std::size(vllm_test::kLtx2CondVideoBasePositions), "video base state");

  // _first_frame_keyframes_mask marks the target's FIRST latent frame
  // unconditionally, because the causal encoder makes it span one pixel frame.
  const double err = MaxAbsDiff(keyframes_mask, vllm_test::kLtx2CondVideoBaseKeyframesMask,
                                std::size(vllm_test::kLtx2CondVideoBaseKeyframesMask));
  INFO("keyframes mask max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
}

TEST_CASE("ltx2 conditioning: an image REPLACES one latent frame, and leaves the noise alone") {
  const vllm::Ltx2VideoLatentShape target = CondVideoTarget();
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);
  const std::vector<float> before_latent = state.latent;

  const vllm::Ltx2LatentVolume image = CondVolume("ltx2.cond.image", 4, 1, 2, 2);
  vllm::Ltx2ConditionVideoByLatentIndex(&state, target, kCondPatch, image, /*strength=*/0.7,
                                        vllm_test::kLtx2CondIndexLatentIdx);

  CheckState(state, vllm_test::kLtx2CondIndexTokens, vllm_test::kLtx2CondIndexWidth,
             vllm_test::kLtx2CondIndexPosDims, vllm_test::kLtx2CondIndexClean,
             std::size(vllm_test::kLtx2CondIndexClean), vllm_test::kLtx2CondIndexLatent,
             std::size(vllm_test::kLtx2CondIndexLatent), vllm_test::kLtx2CondIndexMask,
             std::size(vllm_test::kLtx2CondIndexMask), vllm_test::kLtx2CondIndexPositions,
             std::size(vllm_test::kLtx2CondIndexPositions), "video by latent index");

  // The NOISY tensor is untouched (latent_cond.py:38-39). diffusers writes the
  // clean tokens into it as well and only agrees at noise_scale == 1
  // (pipeline_ltx2_condition.py:1002 vs :1229-1232); copying that here would be a
  // silent divergence at every other noise scale, which no shape can catch.
  CHECK(state.latent == before_latent);
  // The token count does NOT grow — this item replaces, it does not append.
  CHECK(state.tokens == vllm_test::kLtx2CondVideoBaseTokens);

  // A conditioning whose spatial shape does not match the target is REFUSED, as
  // upstream's ConditioningError is.
  const vllm::Ltx2LatentVolume wrong = CondVolume("ltx2.cond.wrong", 4, 1, 4, 2);
  bool threw = false;
  std::string message;
  try {
    vllm::Ltx2ConditionVideoByLatentIndex(&state, target, kCondPatch, wrong, 0.7, 0);
  } catch (const std::exception& error) {
    threw = true;
    message = error.what();
  }
  REQUIRE(threw);
  INFO("refusal message: " << message);
  CHECK(message.find("same spatial shape") != std::string::npos);
}

TEST_CASE("ltx2 conditioning: a KEYFRAME appends tokens at its own pixel frame") {
  const vllm::Ltx2VideoLatentShape target = CondVideoTarget();
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);

  const vllm::Ltx2LatentVolume keyframe = CondVolume("ltx2.cond.keyframe", 4, 1, 2, 2);
  vllm::Ltx2ConditionVideoByKeyframe(&state, keyframe, kCondPatch, factors, kCondFps,
                                     vllm_test::kLtx2CondKeyframeFrameIdx, /*strength=*/0.6,
                                     /*num_pixel_frames=*/1, /*causal_fix=*/true);

  CHECK(state.tokens > vllm_test::kLtx2CondKeyframeTokensBefore);
  CheckState(state, vllm_test::kLtx2CondKeyframeTokens, vllm_test::kLtx2CondKeyframeWidth,
             vllm_test::kLtx2CondKeyframePosDims, vllm_test::kLtx2CondKeyframeClean,
             std::size(vllm_test::kLtx2CondKeyframeClean), vllm_test::kLtx2CondKeyframeLatent,
             std::size(vllm_test::kLtx2CondKeyframeLatent), vllm_test::kLtx2CondKeyframeMask,
             std::size(vllm_test::kLtx2CondKeyframeMask), vllm_test::kLtx2CondKeyframePositions,
             std::size(vllm_test::kLtx2CondKeyframePositions), "video by keyframe");

  // The causal fix is DISABLED for a keyframe that is not at pixel frame 0
  // (keyframe_cond.py:45-50), so passing it must change NOTHING at frame_idx 5.
  // Without this the flag could be wired backwards and every golden would agree.
  vllm::Ltx2LatentState no_fix =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);
  vllm::Ltx2ConditionVideoByKeyframe(&no_fix, keyframe, kCondPatch, factors, kCondFps,
                                     vllm_test::kLtx2CondKeyframeFrameIdx, 0.6, 1,
                                     /*causal_fix=*/false);
  CHECK(no_fix.positions == state.positions);
}

TEST_CASE("ltx2 conditioning: a REFERENCE VIDEO is translated into the target's frame") {
  const vllm::Ltx2VideoLatentShape target = CondVideoTarget();
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);

  const vllm::Ltx2LatentVolume reference = CondVolume("ltx2.cond.reference", 4, 2, 2, 2);
  vllm::Ltx2ConditionVideoByReference(&state, reference, kCondPatch, factors, kCondFps,
                                      vllm_test::kLtx2CondRefDownscale,
                                      vllm_test::kLtx2CondRefTemporalScale, /*strength=*/1.0,
                                      /*causal_fix=*/true);

  CheckState(state, vllm_test::kLtx2CondRefTokens, vllm_test::kLtx2CondRefWidth,
             vllm_test::kLtx2CondRefPosDims, vllm_test::kLtx2CondRefClean,
             std::size(vllm_test::kLtx2CondRefClean), vllm_test::kLtx2CondRefLatent,
             std::size(vllm_test::kLtx2CondRefLatent), vllm_test::kLtx2CondRefMask,
             std::size(vllm_test::kLtx2CondRefMask), vllm_test::kLtx2CondRefPositions,
             std::size(vllm_test::kLtx2CondRefPositions), "video by reference");

  // Both scale factors guard on `!= 1` (reference_video_cond.py:74, 80), so the
  // unscaled branch is a DIFFERENT computation, not the same one times one. An
  // implementation that always took the scaled path would pass the arm above and
  // fail here.
  vllm::Ltx2LatentState plain =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);
  vllm::Ltx2ConditionVideoByReference(&plain, reference, kCondPatch, factors, kCondFps,
                                      /*downscale_factor=*/1, /*temporal_scale_factor=*/1, 1.0,
                                      true);
  const double err = MaxAbsDiff(plain.positions, vllm_test::kLtx2CondRefPlainPositions,
                                std::size(vllm_test::kLtx2CondRefPlainPositions));
  INFO("unscaled reference positions max|diff| = " << err);
  CHECK(err <= kLtx2GoldenTol);
  CHECK(plain.positions != state.positions);
}

// ─── the token-APPEND seam (row LTX25-TOKEN-APPEND, issue #930) ─────────────
//
// UPSTREAM SHIPS NO TESTS. `find /home/mudler/_git/LTX-2 -name 'test_*.py'`
// returns 0 across the whole repository at pin `fd4ded7f`, so there is no suite
// to port and each case below is written against an upstream ANCHOR instead:
// every assertion names the `file:line` that justifies the behaviour it checks.
//
// These are the two halves of an append the conditioning items could not do for
// themselves. The items concatenate; nothing extended the per-token marker
// alongside them, and nothing trimmed the sequence back.

TEST_CASE("ltx2 conditioning: an APPEND extends the per-token keyframes marker") {
  const vllm::Ltx2VideoLatentShape target = CondVideoTarget();
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);

  // `create_initial_state` returns the marker ON the state (tools.py:184), not
  // beside it. A port that only filled the out-parameter would leave an
  // appending item with nothing to extend.
  REQUIRE(static_cast<int64_t>(state.keyframes_mask.size()) == state.tokens);
  const std::vector<float> before = state.keyframes_mask;

  const vllm::Ltx2LatentVolume keyframe = CondVolume("ltx2.cond.keyframe", 4, 1, 2, 2);
  vllm::Ltx2ConditionVideoByKeyframe(&state, keyframe, kCondPatch, factors, kCondFps,
                                     vllm_test::kLtx2CondKeyframeFrameIdx, /*strength=*/0.6,
                                     /*num_pixel_frames=*/1, /*causal_fix=*/true);

  // ONE VALUE PER TOKEN, still. This is the invariant `extend_keyframes_mask`
  // exists for, in upstream's own words: "otherwise the per-token marker goes
  // out of sync with the token sequence" (mask_utils.py:83-85). Out of sync is
  // invisible to every shape check downstream — the render stays the right size
  // and applies a trained term to the wrong tokens.
  CHECK(static_cast<int64_t>(state.keyframes_mask.size()) == state.tokens);
  REQUIRE(state.tokens == vllm_test::kLtx2CondKeyframeTokens);

  // The ORIGINAL values are untouched...
  for (size_t i = 0; i < before.size(); ++i) {
    INFO("target token " << i);
    CHECK(state.keyframes_mask[i] == before[i]);
  }
  // ...and every appended token is UNMARKED. `marked=False` for given keyframe
  // content (keyframe_cond.py:85-86, whose comment says given keyframe content
  // carries no keyframe marker). Marking them would add a trained bias to
  // tokens upstream leaves alone, and the render would still be finite and the
  // right shape.
  for (size_t i = before.size(); i < state.keyframes_mask.size(); ++i) {
    INFO("appended token " << i);
    CHECK(state.keyframes_mask[i] == 0.0F);
  }
}

TEST_CASE("ltx2 conditioning: extend_keyframes_mask mirrors BOTH of upstream's None branches") {
  // The branches are not symmetric and a port that treats them as one gets the
  // generated-slot arm silently wrong (mask_utils.py:96-101).
  SUBCASE("no existing mask and UNMARKED stays None") {
    // `if existing is None and not marked: return None`
    // (conditioning/mask_utils.py:98-99). An audio state carries no marker:
    // `AudioLatentTools.create_initial_state` (tools.py:246-280) returns
    // `self.patchify(LatentState(...))` with no `keyframes_mask` argument at
    // all, where the video tools' own `create_initial_state` sets one on the
    // line that builds the state (tools.py:184). Appending reference audio must
    // therefore not materialise a zero mask, because a zero mask IS a mask and
    // the DiT would read it as one.
    const vllm::Ltx2AudioLatentShape target = CondAudioTarget();
    const vllm::Ltx2AudioPatchifierParams params;
    vllm::Ltx2LatentState state = vllm::Ltx2CreateAudioLatentState(target, params);
    REQUIRE(state.keyframes_mask.empty());

    vllm::Ltx2ExtendKeyframesMask(&state, /*num_new_tokens=*/3, /*marked=*/false);
    CHECK(state.keyframes_mask.empty());
  }
  SUBCASE("no existing mask and MARKED zero-fills first, then marks the new tokens") {
    // `existing = torch.zeros_like(latent_state.denoise_mask)`
    // (mask_utils.py:100-101 — named in full because the nearest file above is
    // `tools.py`, whose :100-101 is a real but unrelated statement) sized
    // by the state BEFORE the append, then ones for the new tokens. The one
    // upstream caller that passes true is `VideoGeneratedKeyframeSlots`
    // (keyframe_slots.py:121).
    const vllm::Ltx2AudioLatentShape target = CondAudioTarget();
    const vllm::Ltx2AudioPatchifierParams params;
    vllm::Ltx2LatentState state = vllm::Ltx2CreateAudioLatentState(target, params);
    const int64_t before = state.tokens;
    REQUIRE(before > 0);

    vllm::Ltx2ExtendKeyframesMask(&state, /*num_new_tokens=*/3, /*marked=*/true);
    REQUIRE(static_cast<int64_t>(state.keyframes_mask.size()) == before + 3);
    for (int64_t i = 0; i < before; ++i) {
      INFO("pre-existing token " << i);
      CHECK(state.keyframes_mask[static_cast<size_t>(i)] == 0.0F);
    }
    for (int64_t i = before; i < before + 3; ++i) {
      INFO("new token " << i);
      CHECK(state.keyframes_mask[static_cast<size_t>(i)] == 1.0F);
    }
  }
}

TEST_CASE("ltx2 conditioning: clear_conditioning TRIMS an append back to the target grid") {
  const vllm::Ltx2VideoLatentShape target = CondVideoTarget();
  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(target, kCondPatch, factors, kCondFps, true);

  const int64_t target_tokens = state.tokens;
  REQUIRE(target_tokens == vllm_test::kLtx2CondVideoBaseTokens);
  const std::vector<float> target_positions = state.positions;

  // A CONDITIONED MASK VALUE INSIDE THE TARGET RANGE, and without it the
  // all-ones assertion below gates NOTHING. `Ltx2CreateVideoLatentState` already
  // fills every target token's mask with 1.0, and the keyframe append writes its
  // `1 - strength` only at the TAIL, which a slice to `target_tokens` drops
  // anyway — so "restore all ones" and "slice" produce identical bytes over the
  // range the loop walks, and a slicing build passes. MEASURED: mutation M6
  // sliced instead of restoring and this case stayed GREEN.
  //
  // `Ltx2ConditionVideoByLatentIndex` is the fix because it writes `1 - strength`
  // at `start .. start + count` INSIDE the target (latent_cond.py:41; ours at
  // ltx2_conditioning.cpp, the `state->mask[i] = 1.0 - strength` loop). At
  // `latent_idx = 0` that is `start = 0`, `count = 1*2*2 = 4` of the 12 target
  // tokens, so a slice leaves 0.4 at token 0 and the loop below REDs.
  const vllm::Ltx2LatentVolume first = CondVolume("ltx2.cond.first", 4, 1, 2, 2);
  vllm::Ltx2ConditionVideoByLatentIndex(&state, target, kCondPatch, first, /*strength=*/0.6,
                                        /*latent_idx=*/0);
  // THE INSTRUMENT IS ARMED, asserted rather than assumed. If this ever came
  // back 1.0 the all-ones loop below would silently return to gating nothing,
  // which is the exact state this case was repaired out of.
  REQUIRE(state.mask.front() == doctest::Approx(0.4F));

  const vllm::Ltx2LatentVolume keyframe = CondVolume("ltx2.cond.keyframe", 4, 1, 2, 2);
  vllm::Ltx2ConditionVideoByKeyframe(&state, keyframe, kCondPatch, factors, kCondFps,
                                     vllm_test::kLtx2CondKeyframeFrameIdx, /*strength=*/0.6,
                                     /*num_pixel_frames=*/1, /*causal_fix=*/true);
  REQUIRE(state.tokens > target_tokens);
  // The APPENDED tokens carry `1 - strength` = 0.4 too. This one is a check on
  // the append, NOT on the clear: `.back()` sits past `target_tokens`, so the
  // trim drops it under either implementation and it can say nothing about
  // all-ones-versus-slice. The value that separates those is `mask.front()`
  // above.
  REQUIRE(state.mask.back() == doctest::Approx(0.4F));

  vllm::Ltx2ClearConditioning(&state, target_tokens);

  // `latent`, `clean_latent` and `positions` truncated to
  // `patchifier.get_token_count(target_shape)` (tools.py:101-105).
  CHECK(state.tokens == target_tokens);
  CHECK(static_cast<int64_t>(state.latent.size()) == target_tokens * state.width);
  CHECK(static_cast<int64_t>(state.clean.size()) == target_tokens * state.width);

  // THE MASK COMES BACK ALL ONES, not the conditioned mask sliced
  // (tools.py:104 — `torch.ones_like(latent_state.denoise_mask)[:, :num_tokens]`).
  // Slicing instead would leave 0.4 on nothing here, but on the two-stage recipe
  // it would carry a conditioned mask into the next phase's initial latent.
  REQUIRE(static_cast<int64_t>(state.mask.size()) == target_tokens);
  for (int64_t i = 0; i < target_tokens; ++i) {
    INFO("mask token " << i);
    CHECK(state.mask[static_cast<size_t>(i)] == 1.0F);
  }

  // POSITIONS ARE TRIMMED PER DIMENSION. They are [pos_dims, tokens, 2], so a
  // plain resize keeps the first dimension's APPENDED tokens and drops the last
  // dimension's real ones — and the result still has the right length. Held to
  // the target's own positions byte for byte, which is the only statement that
  // can see the difference.
  REQUIRE(state.positions.size() == target_positions.size());
  CHECK(state.positions == target_positions);

  // `keyframes_mask=None` (tools.py:113).
  CHECK(state.keyframes_mask.empty());
}

TEST_CASE("ltx2 conditioning: the audio state and its REFERENCE AUDIO append") {
  const vllm::Ltx2AudioLatentShape target = CondAudioTarget();
  const vllm::Ltx2AudioPatchifierParams params;
  vllm::Ltx2LatentState state = vllm::Ltx2CreateAudioLatentState(target, params);

  CheckState(state, vllm_test::kLtx2CondAudioBaseTokens, vllm_test::kLtx2CondAudioBaseWidth,
             vllm_test::kLtx2CondAudioBasePosDims, vllm_test::kLtx2CondAudioBaseClean,
             std::size(vllm_test::kLtx2CondAudioBaseClean), vllm_test::kLtx2CondAudioBaseLatent,
             std::size(vllm_test::kLtx2CondAudioBaseLatent), vllm_test::kLtx2CondAudioBaseMask,
             std::size(vllm_test::kLtx2CondAudioBaseMask), vllm_test::kLtx2CondAudioBasePositions,
             std::size(vllm_test::kLtx2CondAudioBasePositions), "audio base state");

  // The reference is patchified through the SAME gated patchifier the pipeline
  // uses, so this arm also proves the encoder's [C, T, F] output is the layout
  // `Ltx2AudioPatchify` expects.
  vllm::Ltx2AudioLatentShape ref_shape = target;
  ref_shape.frames = vllm_test::kLtx2CondRefAudioFrames;
  const std::vector<float> ref_latent = Ltx2Input(
      "ltx2.cond.refaudio", ref_shape.channels * ref_shape.frames * ref_shape.mel_bins, 1.0);
  const std::vector<float> ref_tokens = vllm::Ltx2AudioPatchify(ref_latent.data(), ref_shape);
  const std::vector<float> ref_positions = vllm::Ltx2AudioPatchTimings(ref_shape, params);

  vllm::Ltx2ConditionAudioByReference(&state, ref_tokens, ref_shape.frames,
                                      ref_shape.channels * ref_shape.mel_bins, ref_positions,
                                      /*strength=*/1.0);

  CheckState(state, vllm_test::kLtx2CondRefAudioTokens, vllm_test::kLtx2CondRefAudioWidth,
             vllm_test::kLtx2CondRefAudioPosDims, vllm_test::kLtx2CondRefAudioClean,
             std::size(vllm_test::kLtx2CondRefAudioClean), vllm_test::kLtx2CondRefAudioLatent,
             std::size(vllm_test::kLtx2CondRefAudioLatent), vllm_test::kLtx2CondRefAudioMask,
             std::size(vllm_test::kLtx2CondRefAudioMask), vllm_test::kLtx2CondRefAudioPositions,
             std::size(vllm_test::kLtx2CondRefAudioPositions), "audio by reference");
}

TEST_CASE("ltx2 conditioning: an ENCODED image reaches the pipeline's token stream end to end") {
  // The point of phase L11, asserted as one chain rather than as parts: pixels ->
  // `Ltx2ConvVideoEncode` -> `Ltx2ConditionVideoByLatentIndex` -> the token state
  // the denoise loop already consumes. Nothing here re-derives a shape by hand.
  const vllm::Ltx2ConvVideoEncoderConfig enc = ReducedVideoEncoderConfigA();
  ParamBag bag = BuildVideoEncoderParams(enc);
  const int64_t c = vllm_test::kLtx2VideoEncInC;
  const int64_t t = vllm_test::kLtx2VideoEncInT;
  const int64_t h = vllm_test::kLtx2VideoEncInH;
  const int64_t w = vllm_test::kLtx2VideoEncInW;
  const std::vector<float> pixels = Ltx2Input("ltx2.videoenc.input", c * t * h * w, 1.0);

  const vllm::Ltx2LatentVolume encoded =
      vllm::Ltx2ConvVideoEncode(enc, bag.weights, pixels, c, t, h, w, nullptr);

  // A single ENCODED image is one latent frame wide; take the encoder's first.
  vllm::Ltx2LatentVolume first_frame;
  first_frame.batch = 1;
  first_frame.channels = encoded.channels;
  first_frame.frames = 1;
  first_frame.height = encoded.height;
  first_frame.width = encoded.width;
  const int64_t plane = encoded.height * encoded.width;
  first_frame.data.resize(static_cast<size_t>(encoded.channels * plane));
  for (int64_t ch = 0; ch < encoded.channels; ++ch) {
    for (int64_t i = 0; i < plane; ++i) {
      first_frame.data[static_cast<size_t>(ch * plane + i)] =
          encoded.data[static_cast<size_t>((ch * encoded.frames) * plane + i)];
    }
  }

  vllm::Ltx2VideoLatentShape target;
  target.batch = 1;
  target.channels = encoded.channels;
  target.frames = encoded.frames;
  target.height = encoded.height;
  target.width = encoded.width;

  const vllm::Ltx2ScaleFactors factors;
  vllm::Ltx2LatentState state =
      vllm::Ltx2CreateVideoLatentState(target, /*patch_size=*/1, factors, kCondFps, true);
  const std::vector<float> before = state.clean;
  vllm::Ltx2ConditionVideoByLatentIndex(&state, target, /*patch_size=*/1, first_frame,
                                        /*strength=*/1.0, /*latent_idx=*/0);

  // Strength 1 means the first latent frame's tokens are FULLY conditioned: mask
  // 0 there, still 1 everywhere else.
  const int64_t per_frame = state.tokens / target.frames;
  for (int64_t i = 0; i < state.tokens; ++i) {
    CHECK(state.mask[static_cast<size_t>(i)] == (i < per_frame ? 0.0f : 1.0f));
  }
  CHECK(state.clean != before);
  // And the conditioned tokens ARE the encoder's own first latent frame,
  // channel-for-channel, which is the join this phase exists to make.
  for (int64_t token = 0; token < per_frame; ++token) {
    for (int64_t ch = 0; ch < target.channels; ++ch) {
      CHECK(state.clean[static_cast<size_t>(token * state.width + ch)] ==
            first_frame.data[static_cast<size_t>(ch * plane + token)]);
    }
  }
}
