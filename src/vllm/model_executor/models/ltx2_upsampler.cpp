// LTX-2.5 LATENT SPATIAL UPSAMPLER — see
// include/vllm/model_executor/models/ltx2_upsampler.h for the upstream mapping
// and the refusals.
//
// WHY THE CONVOLUTION IS LOCAL AND NOT THE VAE'S. `LatentUpsampler` builds plain
// `torch.nn.Conv3d`/`Conv2d` with `padding=1`, i.e. ZERO padding on every axis
// including time. `ltx2_video_vae.cpp`'s `CausalConv3d` prepends replicated
// copies of frame 0 instead (convolution.py:306-307). Reusing that kernel here
// would shift the whole clip while still producing a correctly shaped, finite,
// plausible latent — so the two stay separate, deliberately.
//
// ARITHMETIC WIDTH, stated once and referenced per site below. Upstream is f32
// everywhere in this file; every `double` here is an ESCAPE and each one is
// annotated at its site. They come in two kinds and only the first is justified
// by the suite's convention:
//
//   REDUCTIONS (conv accumulators, GroupNorm mean/var, the blur accumulator).
//     Upstream reduces in f32 but in a blocked/vectorized order no straight loop
//     reproduces, so accumulating exactly and rounding ONCE is the closest
//     single-rounding approximation to any order. This is the same escape L3
//     took and documented at ltx2_text_encoder.cpp:259-269.
//
//   POINTWISE (`Silu` here, `GeluTanh` in ltx2_duration_head.cpp). These are NOT
//     reductions, so the convention above does not cover them: upstream rounds to
//     f32 at each step of the expression and computing wider is numerically FINER
//     than the mirror, not equal to it — the polarity AGENTS.md warns about, where
//     a too-wide dtype still passes a value gate. Left as-is here rather than
//     narrowed in a review-repair branch, because narrowing moves the upsampler
//     and duration-head goldens and so owes its own red-first change. Recorded so
//     it is visible debt rather than an unremarked default.
#include "vllm/model_executor/models/ltx2_upsampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

// The temporal arm's refusal is shared with every other L5 out-of-scope feature,
// so it is raised through the one seam that names them all.
#include "vllm/model_executor/models/ltx2_pipeline.h"

namespace vllm {
namespace {

[[noreturn]] void Refuse(const std::string& message) { throw std::runtime_error(message); }

void Require(bool condition, const std::string& message) {
  if (!condition) Refuse(message);
}

// A [channels, frames, height, width] volume at batch 1 — the shape every stage
// below operates on. Batch is carried by the caller loop.
struct Volume {
  int64_t channels = 0, frames = 0, height = 0, width = 0;
  std::vector<float> data;

  int64_t elems() const { return channels * frames * height * width; }
  float& at(int64_t c, int64_t f, int64_t y, int64_t x) {
    return data[static_cast<size_t>(((c * frames + f) * height + y) * width + x)];
  }
  float at(int64_t c, int64_t f, int64_t y, int64_t x) const {
    return data[static_cast<size_t>(((c * frames + f) * height + y) * width + x)];
  }
};

// `torch.nn.Conv3d(in, out, kernel_size=3, padding=1)` — zero padding on ALL
// three axes, unlike the VAE's causal replication.
Volume Conv3dPad1(const Volume& in, int64_t out_channels, const std::vector<float>& weight,
                  const std::vector<float>& bias) {
  constexpr int64_t k = 3;
  constexpr int64_t pad = 1;
  Volume out;
  out.channels = out_channels;
  out.frames = in.frames;
  out.height = in.height;
  out.width = in.width;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0f);
  Require(weight.size() ==
              static_cast<size_t>(out_channels * in.channels * k * k * k),
          "ltx2 upsampler: conv3d weight has the wrong element count");
  Require(bias.size() == static_cast<size_t>(out_channels),
          "ltx2 upsampler: conv3d bias has the wrong element count");

  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t f = 0; f < out.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          // f64 REDUCTION escape -- see the width note in the file header.
          double acc = static_cast<double>(bias[static_cast<size_t>(oc)]);
          for (int64_t ic = 0; ic < in.channels; ++ic) {
            for (int64_t kf = 0; kf < k; ++kf) {
              const int64_t sf = f + kf - pad;
              if (sf < 0 || sf >= in.frames) continue;
              for (int64_t ky = 0; ky < k; ++ky) {
                const int64_t sy = y + ky - pad;
                if (sy < 0 || sy >= in.height) continue;
                for (int64_t kx = 0; kx < k; ++kx) {
                  const int64_t sx = x + kx - pad;
                  if (sx < 0 || sx >= in.width) continue;
                  const size_t widx = static_cast<size_t>(
                      (((oc * in.channels + ic) * k + kf) * k + ky) * k + kx);
                  acc += static_cast<double>(weight[widx]) *
                         static_cast<double>(in.at(ic, sf, sy, sx));
                }
              }
            }
          }
          out.at(oc, f, y, x) = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

// `torch.nn.Conv2d(in, out, kernel_size=3, padding=1)` applied PER FRAME — what
// upstream reaches by `rearrange(x, "b c f h w -> (b f) c h w")` (model.py:117,
// spatial_rational_resampler.py:42).
Volume Conv2dPad1PerFrame(const Volume& in, int64_t out_channels,
                          const std::vector<float>& weight, const std::vector<float>& bias) {
  constexpr int64_t k = 3;
  constexpr int64_t pad = 1;
  Volume out;
  out.channels = out_channels;
  out.frames = in.frames;
  out.height = in.height;
  out.width = in.width;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0f);
  Require(weight.size() == static_cast<size_t>(out_channels * in.channels * k * k),
          "ltx2 upsampler: conv2d weight has the wrong element count");

  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t f = 0; f < out.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          // f64 REDUCTION escape -- see the width note in the file header.
          double acc = static_cast<double>(bias[static_cast<size_t>(oc)]);
          for (int64_t ic = 0; ic < in.channels; ++ic) {
            for (int64_t ky = 0; ky < k; ++ky) {
              const int64_t sy = y + ky - pad;
              if (sy < 0 || sy >= in.height) continue;
              for (int64_t kx = 0; kx < k; ++kx) {
                const int64_t sx = x + kx - pad;
                if (sx < 0 || sx >= in.width) continue;
                const size_t widx =
                    static_cast<size_t>(((oc * in.channels + ic) * k + ky) * k + kx);
                acc += static_cast<double>(weight[widx]) *
                       static_cast<double>(in.at(ic, f, sy, sx));
              }
            }
          }
          out.at(oc, f, y, x) = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

// `torch.nn.GroupNorm(32, channels)`: statistics over (channels_per_group, F, H, W)
// per group, per sample. The group COUNT is a literal upstream, not a config key.
void GroupNorm(Volume& x, const std::vector<float>& weight, const std::vector<float>& bias) {
  const int64_t groups = kLtx2UpsamplerNormGroups;
  Require(x.channels % groups == 0,
          "ltx2 upsampler: GroupNorm(32) requires channels divisible by 32, got " +
              std::to_string(x.channels));
  const int64_t per_group = x.channels / groups;
  const int64_t spatial = x.frames * x.height * x.width;
  const int64_t elems = per_group * spatial;

  for (int64_t g = 0; g < groups; ++g) {
    // f64 REDUCTION escape (mean and var) -- see the width note in the header.
    double mean = 0.0;
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        mean += static_cast<double>(x.data[static_cast<size_t>(c * spatial + i)]);
      }
    }
    mean /= static_cast<double>(elems);
    double var = 0.0;
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      for (int64_t i = 0; i < spatial; ++i) {
        const double d = static_cast<double>(x.data[static_cast<size_t>(c * spatial + i)]) - mean;
        var += d * d;
      }
    }
    // torch's GroupNorm uses the BIASED variance (divide by N), unlike `std`.
    var /= static_cast<double>(elems);
    const double inv = 1.0 / std::sqrt(var + kLtx2UpsamplerNormEps);
    for (int64_t c = g * per_group; c < (g + 1) * per_group; ++c) {
      const double gain = static_cast<double>(weight[static_cast<size_t>(c)]);
      const double shift = static_cast<double>(bias[static_cast<size_t>(c)]);
      for (int64_t i = 0; i < spatial; ++i) {
        float& value = x.data[static_cast<size_t>(c * spatial + i)];
        value = static_cast<float>((static_cast<double>(value) - mean) * inv * gain + shift);
      }
    }
  }
}

void Silu(std::vector<float>& x) {
  for (float& value : x) {
    // POINTWISE f64, WIDER than upstream's f32 -- see the width note in the
    // file header. Not covered by the reduction convention.
    const double v = static_cast<double>(value);
    value = static_cast<float>(v / (1.0 + std::exp(-v)));
  }
}

// ResBlock.forward (res_block.py:29-37). The residual is added BEFORE the
// activation — `activation(x + residual)`, not `activation(x) + residual`.
Volume ResBlockForward(const Ltx2VaeWeights& weights, const std::string& prefix,
                       const Volume& in) {
  Volume x = Conv3dPad1(in, in.channels, weights.Get(prefix + "conv1.weight"),
                        weights.Get(prefix + "conv1.bias"));
  GroupNorm(x, weights.Get(prefix + "norm1.weight"), weights.Get(prefix + "norm1.bias"));
  Silu(x.data);
  Volume y = Conv3dPad1(x, in.channels, weights.Get(prefix + "conv2.weight"),
                        weights.Get(prefix + "conv2.bias"));
  GroupNorm(y, weights.Get(prefix + "norm2.weight"), weights.Get(prefix + "norm2.bias"));
  for (size_t i = 0; i < y.data.size(); ++i) y.data[i] += in.data[i];
  Silu(y.data);
  return y;
}

// PixelShuffleND(2) — `b (c p1 p2) h w -> b c (h p1) (w p2)` (pixel_shuffle.py:41-47).
// p1 takes HEIGHT and p2 takes WIDTH; swapping them transposes every block.
Volume PixelShuffle2d(const Volume& in, int64_t up_h, int64_t up_w) {
  Require(in.channels % (up_h * up_w) == 0,
          "ltx2 upsampler: pixel shuffle requires channels divisible by the upscale product");
  Volume out;
  out.channels = in.channels / (up_h * up_w);
  out.frames = in.frames;
  out.height = in.height * up_h;
  out.width = in.width * up_w;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0f);
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t p1 = 0; p1 < up_h; ++p1) {
      for (int64_t p2 = 0; p2 < up_w; ++p2) {
        const int64_t src_c = (c * up_h + p1) * up_w + p2;
        for (int64_t f = 0; f < in.frames; ++f) {
          for (int64_t y = 0; y < in.height; ++y) {
            for (int64_t x = 0; x < in.width; ++x) {
              out.at(c, f, y * up_h + p1, x * up_w + p2) = in.at(src_c, f, y, x);
            }
          }
        }
      }
    }
  }
  return out;
}

// PixelShuffleND(1) — `b (c p1) f h w -> b c (f p1) h w` (pixel_shuffle.py:47-52).
// Both groupings put `p1` FASTEST: the source channel is `c * p1 + j` because the
// pattern is `(c p1)`, and the destination frame is `f * p1 + j` because it is
// `(f p1)`. Reversing either factor order yields a correctly shaped, finite,
// plausible latent, which is why the golden — not a shape check — is what holds
// this down.
Volume PixelShuffle1d(const Volume& in, int64_t up_f) {
  Require(in.channels % up_f == 0,
          "ltx2 upsampler: temporal pixel shuffle requires channels divisible by " +
              std::to_string(up_f));
  Volume out;
  out.channels = in.channels / up_f;
  out.frames = in.frames * up_f;
  out.height = in.height;
  out.width = in.width;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0f);
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t j = 0; j < up_f; ++j) {
      const int64_t src_c = c * up_f + j;
      for (int64_t f = 0; f < in.frames; ++f) {
        for (int64_t y = 0; y < in.height; ++y) {
          for (int64_t x = 0; x < in.width; ++x) {
            out.at(c, f * up_f + j, y, x) = in.at(src_c, f, y, x);
          }
        }
      }
    }
  }
  return out;
}

// `x = x[:, :, 1:, :, :]` (model.py:111-113): "remove the first frame after
// upsampling. This is done because the first frame encodes one pixel frame."
// Frames go 2F -> 2F - 1, which is exactly the count the only upstream consumer
// keeps for itself (`num_frames = 2 * (num_frames - 1) + 1`, dfr_pipeline.py:408).
Volume DropFirstFrame(const Volume& in) {
  Require(in.frames >= 2,
          "ltx2 upsampler: the temporal arm drops the first frame after upsampling "
          "(model/upsampler/model.py:109-113), so it needs at least 2 frames out of the "
          "shuffle");
  Volume out;
  out.channels = in.channels;
  out.frames = in.frames - 1;
  out.height = in.height;
  out.width = in.width;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0f);
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t f = 0; f < out.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          out.at(c, f, y, x) = in.at(c, f + 1, y, x);
        }
      }
    }
  }
  return out;
}

// BlurDownsample._apply_2d (blur_downsample.py:49-53): a DEPTHWISE conv2d with
// the fixed binomial kernel, stride `den`, padding `kernel_size // 2`, per frame.
Volume BlurDownsample(const Volume& in, int64_t stride, int64_t kernel_size) {
  if (stride == 1) return in;  // :36-37, the short circuit
  const std::vector<float> kernel = Ltx2BlurKernel(kernel_size);
  const int64_t pad = kernel_size / 2;
  Volume out;
  out.channels = in.channels;
  out.frames = in.frames;
  out.height = (in.height + 2 * pad - kernel_size) / stride + 1;
  out.width = (in.width + 2 * pad - kernel_size) / stride + 1;
  out.data.assign(static_cast<size_t>(out.elems()), 0.0f);

  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t f = 0; f < in.frames; ++f) {
      for (int64_t y = 0; y < out.height; ++y) {
        for (int64_t x = 0; x < out.width; ++x) {
          // f64 REDUCTION escape -- see the width note in the file header.
          double acc = 0.0;
          for (int64_t ky = 0; ky < kernel_size; ++ky) {
            const int64_t sy = y * stride + ky - pad;
            if (sy < 0 || sy >= in.height) continue;
            for (int64_t kx = 0; kx < kernel_size; ++kx) {
              const int64_t sx = x * stride + kx - pad;
              if (sx < 0 || sx >= in.width) continue;
              acc += static_cast<double>(kernel[static_cast<size_t>(ky * kernel_size + kx)]) *
                     static_cast<double>(in.at(c, f, sy, sx));
            }
          }
          out.at(c, f, y, x) = static_cast<float>(acc);
        }
      }
    }
  }
  return out;
}

}  // namespace

Ltx2RationalScale Ltx2RationalForScale(double scale) {
  // spatial_rational_resampler.py:11-14, exactly this map and no nearest match.
  if (scale == 0.75) return {3, 4};
  if (scale == 1.5) return {3, 2};
  if (scale == 2.0) return {2, 1};
  if (scale == 4.0) return {4, 1};
  Refuse("ltx2 upsampler: Unsupported scale " + std::to_string(scale) +
         ". Choose from [0.75, 1.5, 2.0, 4.0] (spatial_rational_resampler.py:11-14).");
}

std::vector<float> Ltx2BlurKernel(int64_t kernel_size) {
  Require(kernel_size >= 3 && kernel_size % 2 == 1,
          "ltx2 upsampler: BlurDownsample kernel_size must be odd and >= 3, got " +
              std::to_string(kernel_size));
  // blur_downsample.py:29-33: Pascal's row `kernel_size - 1`, outer-producted and
  // normalized to sum 1. Built in integer arithmetic then normalized ONCE, which
  // is what upstream's `k2d / k2d.sum()` on an integer tensor does.
  std::vector<double> row(static_cast<size_t>(kernel_size));
  double value = 1.0;
  for (int64_t i = 0; i < kernel_size; ++i) {
    row[static_cast<size_t>(i)] = value;
    value = value * static_cast<double>(kernel_size - 1 - i) / static_cast<double>(i + 1);
  }
  double total = 0.0;
  for (int64_t y = 0; y < kernel_size; ++y) {
    for (int64_t x = 0; x < kernel_size; ++x) {
      total += row[static_cast<size_t>(y)] * row[static_cast<size_t>(x)];
    }
  }
  std::vector<float> kernel(static_cast<size_t>(kernel_size * kernel_size));
  for (int64_t y = 0; y < kernel_size; ++y) {
    for (int64_t x = 0; x < kernel_size; ++x) {
      kernel[static_cast<size_t>(y * kernel_size + x)] = static_cast<float>(
          row[static_cast<size_t>(y)] * row[static_cast<size_t>(x)] / total);
    }
  }
  return kernel;
}

std::vector<Ltx2UpsamplerTensorSpec> EnumerateLtx2UpsamplerTensors(
    const Ltx2UpsamplerConfig& config) {
  // `named_parameters()` order for LatentUpsampler(dims=3): initial_conv,
  // initial_norm, res_blocks, upsampler, post_upsample_res_blocks, final_conv.
  const std::string p = config.prefix;
  const int64_t in_c = config.in_channels;
  const int64_t mid = config.mid_channels;
  std::vector<Ltx2UpsamplerTensorSpec> specs;

  specs.push_back({p + "initial_conv.weight", {mid, in_c, 3, 3, 3}});
  specs.push_back({p + "initial_conv.bias", {mid}});
  specs.push_back({p + "initial_norm.weight", {mid}});
  specs.push_back({p + "initial_norm.bias", {mid}});

  auto res_block = [&](const std::string& prefix) {
    specs.push_back({prefix + "conv1.weight", {mid, mid, 3, 3, 3}});
    specs.push_back({prefix + "conv1.bias", {mid}});
    specs.push_back({prefix + "norm1.weight", {mid}});
    specs.push_back({prefix + "norm1.bias", {mid}});
    specs.push_back({prefix + "conv2.weight", {mid, mid, 3, 3, 3}});
    specs.push_back({prefix + "conv2.bias", {mid}});
    specs.push_back({prefix + "norm2.weight", {mid}});
    specs.push_back({prefix + "norm2.bias", {mid}});
  };
  for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
    res_block(p + "res_blocks." + std::to_string(i) + ".");
  }

  if (config.temporal_upsample && !config.spatial_upsample) {
    // `torch.nn.Sequential(Conv3d(mid, 2*mid, 3, padding=1), PixelShuffleND(1))`
    // names its conv `upsampler.0` (model.py:68-71). Note the RANK: this is a
    // Conv3d, so the weight is 5-D, where the non-rational SPATIAL arm's
    // identically-named tensor is a 4-D Conv2d kernel (model.py:64-66).
    specs.push_back({p + "upsampler.0.weight",
                     {kLtx2UpsamplerTemporalFactor * mid, mid, 3, 3, 3}});
    specs.push_back({p + "upsampler.0.bias", {kLtx2UpsamplerTemporalFactor * mid}});
  } else if (config.spatial_upsample && !config.temporal_upsample) {
    if (config.rational_resampler) {
      // SpatialRationalResampler names its conv `upsampler.conv`
      // (spatial_rational_resampler.py:36); the blur kernel is a BUFFER, not a
      // parameter, so it never appears here.
      const Ltx2RationalScale rational = Ltx2RationalForScale(config.spatial_scale);
      specs.push_back({p + "upsampler.conv.weight",
                       {rational.num * rational.num * mid, mid, 3, 3}});
      specs.push_back({p + "upsampler.conv.bias", {rational.num * rational.num * mid}});
    } else {
      // `torch.nn.Sequential(Conv2d, PixelShuffleND)` names its conv `upsampler.0`
      // (model.py:64-67).
      specs.push_back({p + "upsampler.0.weight", {4 * mid, mid, 3, 3}});
      specs.push_back({p + "upsampler.0.bias", {4 * mid}});
    }
  }

  for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
    res_block(p + "post_upsample_res_blocks." + std::to_string(i) + ".");
  }
  specs.push_back({p + "final_conv.weight", {in_c, mid, 3, 3, 3}});
  specs.push_back({p + "final_conv.bias", {in_c}});
  return specs;
}

Ltx2LatentVolume Ltx2LatentUpsample(const Ltx2UpsamplerConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const Ltx2LatentVolume& latent) {
  // model.py:73-74 — upstream's own ValueError, raised at CONSTRUCTION there and
  // here at the first forward, which is the earliest this port sees the config.
  Require(config.spatial_upsample || config.temporal_upsample,
          "ltx2 upsampler: either spatial_upsample or temporal_upsample must be True "
          "(model/upsampler/model.py:73-74)");
  // The SPATIOTEMPORAL arm is a different operator, not "the temporal arm with
  // spatial on": model.py:55-59 builds `Conv3d(mid, 8*mid)` + `PixelShuffleND(3)`.
  // Refused BEFORE any weight is touched, so it reports an unported feature
  // rather than a wrong element count on `upsampler.0.weight`.
  if (config.temporal_upsample && config.spatial_upsample) {
    Ltx2RefuseUnportedPipelineFeature(Ltx2UnportedPipelineFeature::kSpatiotemporalUpsampler);
  }
  Require(config.dims == 3,
          "ltx2 upsampler: dims=" + std::to_string(config.dims) +
              " is not ported. The dims=2 arm (model/upsampler/model.py:85-100) builds Conv2d "
              "everywhere, i.e. NO temporal convolution at all, so the 3-D path cannot serve "
              "it. LTX-2.5's upsampler is dims=3. Owed and recorded in "
              ".agents/specs/ltx-2-5.md phase L5.");
  Require(latent.channels == config.in_channels,
          "ltx2 upsampler: latent has " + std::to_string(latent.channels) +
              " channels, config declares " + std::to_string(config.in_channels));

  const std::string p = config.prefix;
  Ltx2LatentVolume result;
  result.batch = latent.batch;
  result.channels = config.in_channels;

  for (int64_t b = 0; b < latent.batch; ++b) {
    Volume x;
    x.channels = latent.channels;
    x.frames = latent.frames;
    x.height = latent.height;
    x.width = latent.width;
    const int64_t stride = latent.channels * latent.frames * latent.height * latent.width;
    x.data.assign(latent.data.begin() + b * stride, latent.data.begin() + (b + 1) * stride);

    // model.py:102-104.
    x = Conv3dPad1(x, config.mid_channels, weights.Get(p + "initial_conv.weight"),
                   weights.Get(p + "initial_conv.bias"));
    GroupNorm(x, weights.Get(p + "initial_norm.weight"), weights.Get(p + "initial_norm.bias"));
    Silu(x.data);

    for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
      x = ResBlockForward(weights, p + "res_blocks." + std::to_string(i) + ".", x);
    }

    if (config.temporal_upsample) {
      // model.py:109-113, and the branch order is upstream's: `if
      // self.temporal_upsample` (:109) is tested BEFORE the resampler check
      // (:114). A full 3-D conv (not per-frame): the temporal arm's
      // `upsampler.0` is a Conv3d (model.py:70), unlike the spatial arm's Conv2d.
      x = Conv3dPad1(x, kLtx2UpsamplerTemporalFactor * config.mid_channels,
                     weights.Get(p + "upsampler.0.weight"), weights.Get(p + "upsampler.0.bias"));
      x = PixelShuffle1d(x, kLtx2UpsamplerTemporalFactor);
      x = DropFirstFrame(x);
    } else if (config.rational_resampler) {
      // SpatialRationalResampler.forward (:40-47): per-frame conv, pixel shuffle
      // by `num`, then an anti-aliased stride-`den` blur.
      const Ltx2RationalScale rational = Ltx2RationalForScale(config.spatial_scale);
      x = Conv2dPad1PerFrame(x, rational.num * rational.num * config.mid_channels,
                             weights.Get(p + "upsampler.conv.weight"),
                             weights.Get(p + "upsampler.conv.bias"));
      x = PixelShuffle2d(x, rational.num, rational.num);
      x = BlurDownsample(x, rational.den, kLtx2BlurKernelSize);
    } else {
      // model.py:117-119 — per-frame Conv2d then PixelShuffleND(2).
      x = Conv2dPad1PerFrame(x, 4 * config.mid_channels, weights.Get(p + "upsampler.0.weight"),
                             weights.Get(p + "upsampler.0.bias"));
      x = PixelShuffle2d(x, 2, 2);
    }

    for (int64_t i = 0; i < config.num_blocks_per_stage; ++i) {
      x = ResBlockForward(weights, p + "post_upsample_res_blocks." + std::to_string(i) + ".", x);
    }
    x = Conv3dPad1(x, config.in_channels, weights.Get(p + "final_conv.weight"),
                   weights.Get(p + "final_conv.bias"));

    result.frames = x.frames;
    result.height = x.height;
    result.width = x.width;
    result.data.insert(result.data.end(), x.data.begin(), x.data.end());
  }
  return result;
}

Ltx2LatentVolume Ltx2UpsampleVideoLatent(const Ltx2UpsamplerConfig& config,
                                         const Ltx2VaeWeights& weights,
                                         const Ltx2LatentVolume& latent,
                                         const std::vector<float>& std_of_means,
                                         const std::vector<float>& mean_of_means) {
  // model.py:140-142, with PerChannelStatistics (video_vae/ops.py:76-84):
  //   un_normalize(x) = x * std + mean
  //   normalize(x)    = (x - mean) / std
  Require(std_of_means.size() == static_cast<size_t>(latent.channels) &&
              mean_of_means.size() == static_cast<size_t>(latent.channels),
          "ltx2 upsample_video: per-channel statistics must carry one value per latent channel");

  Ltx2LatentVolume denormalized = latent;
  const int64_t spatial = latent.frames * latent.height * latent.width;
  for (int64_t b = 0; b < latent.batch; ++b) {
    for (int64_t c = 0; c < latent.channels; ++c) {
      const float std_value = std_of_means[static_cast<size_t>(c)];
      const float mean_value = mean_of_means[static_cast<size_t>(c)];
      for (int64_t i = 0; i < spatial; ++i) {
        float& value =
            denormalized.data[static_cast<size_t>((b * latent.channels + c) * spatial + i)];
        value = value * std_value + mean_value;
      }
    }
  }

  Ltx2LatentVolume upsampled = Ltx2LatentUpsample(config, weights, denormalized);
  const int64_t out_spatial = upsampled.frames * upsampled.height * upsampled.width;
  for (int64_t b = 0; b < upsampled.batch; ++b) {
    for (int64_t c = 0; c < upsampled.channels; ++c) {
      const float std_value = std_of_means[static_cast<size_t>(c)];
      const float mean_value = mean_of_means[static_cast<size_t>(c)];
      for (int64_t i = 0; i < out_spatial; ++i) {
        float& value = upsampled.data[static_cast<size_t>(
            (b * upsampled.channels + c) * out_spatial + i)];
        value = (value - mean_value) / std_value;
      }
    }
  }
  return upsampled;
}

}  // namespace vllm
