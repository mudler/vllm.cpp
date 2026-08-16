// LTX-2.5 CONV VIDEO VAE — ConvVideoDecoder (conv_video_decoder.py) ported 1:1
// from upstream `ltx_core` and gated against it by
// scripts/gen-ltx2-vae-goldens.py, which EXECUTES the upstream module at reduced
// dimensions on CPU.
//
// ─── SCOPE, so nothing is discovered later ───────────────────────────────────
//  * The DIFFUSION decoder (`NADiffusionDecoder` / `DiffusionVideoDecoder`) is
//    NOT ported. Ltx2VideoDecode refuses it BY NAME and never falls back — see
//    the header, and .agents/specs/ltx-2-5.md section 0 item 2.
//  * `attn_res_x` is refused too, for a different reason: at this upstream
//    revision the block cannot be CONSTRUCTED, because `_make_decoder_block`
//    passes `attention_head_dim` to `UNetMidBlock3D`, whose __init__ does not
//    accept it (conv_video_decoder.py:85-96 vs resnet.py:210-222). Upstream
//    raises TypeError; this raises with the same reason named.
//  * `dims == 2` / `dims == (2, 1)` (Conv2d and DualConv3d, convolution.py:27-71)
//    are not ported: the decoder is built with `convolution_dimensions=3`.
//  * The ENCODER half is out of this phase and owed.
//  * The TILED decode path (`tiled_decode`, conv_video_decoder.py:383-484) is NO
//    LONGER owed: it landed in row `LTX25-TILED-DECODE` (#644) and lives next
//    door in ltx2_video_vae_tiled.cpp, reached through `Ltx2VideoDecodeStreaming`.
//    This line named it as debt, and the row's spec cited this exact anchor as
//    the debt it pays, so leaving the two disagreeing was the record contradicting
//    the tree.
//
// ─── DTYPE: THIS IS THE CPU REFERENCE ARM, AND f32 IS NOT WHAT SHIPS ─────────
// Every buffer below is f32, and unlike the audio VAE next door that is NOT an
// upstream-grounded choice — it is the choice a reference arm makes, and it is
// annotated here because AGENTS.md requires an f32 on a model path to carry a
// reason, and because a too-WIDE dtype is the one defect a correctness gate
// structurally cannot report: it stays numerically right, the goldens stay green,
// and the only symptom is twice the bytes moved.
//
// Upstream does the OPPOSITE of what the audio VAE does. `ConvVideoDecoder.forward`
// runs in the CHECKPOINT's dtype: it casts in with `sample.to(weights_dtype)` on
// entry and back with `sample.to(output_dtype)` on exit
// (conv_video_decoder.py:283-286, 355-356). There is no autocast, no float32
// pin, and no spectral-metric argument of the kind that justifies the audio
// tower's f32 (ltx2_audio_vae.cpp:7-12 -> vocoder.py:585-595). So f32 here is
// the reference arm's convention and nothing more.
//
// The golden CANNOT catch this either, and that is worth stating plainly rather
// than leaving for someone to discover: the generator's `fill_from_stream` casts
// every upstream parameter to f32, so the oracle itself runs f32 and a dtype
// comparison against it is vacuous by construction.
//
// ─── THE ARITHMETIC IS f32 TOO, AND USED NOT TO BE (#1008) ───────────────────
// Storage being f32 says nothing about the width the arithmetic runs at, and
// until #1008 this file accumulated every convolution, GEMM, norm and softmax in
// `double` — a width no reference uses anywhere on this path. Upstream's ops are
// plain `nn.Conv3d` / `nn.Conv2d` / `F.normalize` / SDPA, which accumulate in the
// tensor dtype. That was MEASURED rather than assumed: on a reduction engineered
// so the widths separate, `F.conv3d` returns 0.0 for f32 AND for bf16 tensors
// while an f64 accumulator returns 2.5. The case
// "the decode's convolution accumulates in f32" in tests/vllm/models/
// test_ltx2_vae.cpp is that instrument, and it is the only gate here that can
// see the width — for the reason the paragraph above gives.
//
// What deliberately stays f64, each annotated at its site: the pinned config
// epsilons, the once-per-block scalars `sqrt(C)` and `1/sqrt(C)`, and the
// TimestepEmbedding frequency table, which is a constant precompute rather than
// a data path.
//
// PHASE L6 OWES THE PRODUCTION ARM — the bf16/NVFP4 decode that inherits the
// checkpoint dtype the way upstream does. Until it lands, this file is a
// correctness reference, not the shipping path, and no memory or throughput
// number should be taken from it.
#include "vllm/model_executor/models/ltx2_video_vae.h"

#include "vllm/model_executor/models/ltx2_video_vae_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vllm/model_executor/models/minimax_h3.h"
#include "vt/dtype.h"

namespace vllm {

namespace {

// A [C, T, H, W] volume at batch 1.
struct Volume {
  std::vector<float> data;
  int64_t channels = 0, t = 0, h = 0, w = 0;

  int64_t spatial() const { return t * h * w; }
  size_t At(int64_t c, int64_t ti, int64_t hi, int64_t wi) const {
    return static_cast<size_t>(((c * t + ti) * h + hi) * w + wi);
  }
};

int64_t ReflectIndex(int64_t index, int64_t size) {
  // torch's "reflect" excludes the edge sample: [a b c] -> b a b c b.
  if (size == 1) return 0;
  while (index < 0 || index >= size) {
    if (index < 0) index = -index;
    if (index >= size) index = 2 * (size - 1) - index;
  }
  return index;
}

int64_t SpatialIndex(int64_t index, int64_t size, Ltx2PaddingMode mode, bool* zero) {
  *zero = false;
  if (index >= 0 && index < size) return index;
  switch (mode) {
    case Ltx2PaddingMode::kZeros:
      *zero = true;
      return 0;
    case Ltx2PaddingMode::kReflect:
      return ReflectIndex(index, size);
    case Ltx2PaddingMode::kReplicate:
      return std::max<int64_t>(0, std::min<int64_t>(size - 1, index));
  }
  *zero = true;
  return 0;
}

// CausalConv3d (convolution.py:266-317). Two things that are NOT interchangeable
// with MiniMax-H3's causal Conv3d:
//   * the temporal pad REPLICATES FRAME 0 `k_t - 1` times (H3 pads with zeros);
//   * the non-causal branch replicates the FIRST and LAST frame `(k_t - 1) / 2`
//     times each, so the output frame count is the same either way.
// Spatial padding is `k // 2` on each side in `spatial_padding_mode`.
//
// THE STRIDE IS APPLIED AFTER THE PAD, AND THE PAD DOES NOT KNOW ABOUT IT.
// `CausalConv3d.forward` concatenates `k_t - 1` copies of frame 0 and only then
// calls the strided `nn.Conv3d` (convolution.py:305-312), so a stride-2 temporal
// convolution still prepends TWO frames, not one. The video ENCODER is the only
// caller that passes a stride; every decoder call site keeps the defaults.
Volume CausalConv3d(const Volume& in, int64_t out_channels, int64_t kernel, bool causal,
                    Ltx2PaddingMode mode, const std::vector<float>& weight,
                    const std::vector<float>* bias, int64_t stride_t = 1, int64_t stride_h = 1,
                    int64_t stride_w = 1) {
  const int64_t ci = in.channels;
  VT_CHECK(stride_t >= 1 && stride_h >= 1 && stride_w >= 1, "ltx2 conv3d: stride must be positive");
  VT_CHECK(static_cast<int64_t>(in.data.size()) == ci * in.spatial(),
           "ltx2 conv3d: input size does not match [C, T, H, W]");
  VT_CHECK(static_cast<int64_t>(weight.size()) == out_channels * ci * kernel * kernel * kernel,
           "ltx2 conv3d: weight size does not match the kernel");

  const int64_t pad_front = causal ? kernel - 1 : (kernel - 1) / 2;
  const int64_t pad_back = causal ? 0 : (kernel - 1) / 2;
  const int64_t pad_spatial = kernel / 2;
  const int64_t pt = in.t + pad_front + pad_back;
  const int64_t ph = in.h + 2 * pad_spatial;
  const int64_t pw = in.w + 2 * pad_spatial;

  std::vector<float> padded(static_cast<size_t>(ci * pt * ph * pw), 0.0f);
  for (int64_t c = 0; c < ci; ++c) {
    for (int64_t ti = 0; ti < pt; ++ti) {
      // Temporal padding REPLICATES the edge frame, never zeros.
      const int64_t st = std::max<int64_t>(0, std::min<int64_t>(in.t - 1, ti - pad_front));
      for (int64_t hi = 0; hi < ph; ++hi) {
        bool zero_h = false;
        const int64_t sh = SpatialIndex(hi - pad_spatial, in.h, mode, &zero_h);
        for (int64_t wi = 0; wi < pw; ++wi) {
          bool zero_w = false;
          const int64_t sw = SpatialIndex(wi - pad_spatial, in.w, mode, &zero_w);
          if (zero_h || zero_w) continue;
          padded[static_cast<size_t>(((c * pt + ti) * ph + hi) * pw + wi)] =
              in.data[in.At(c, st, sh, sw)];
        }
      }
    }
  }

  Volume out;
  out.channels = out_channels;
  out.t = (pt - kernel) / stride_t + 1;
  out.h = (ph - kernel) / stride_h + 1;
  out.w = (pw - kernel) / stride_w + 1;
  VT_CHECK(pt >= kernel && ph >= kernel && pw >= kernel && out.t > 0 && out.h > 0 && out.w > 0,
           "ltx2 conv3d: empty output");
  out.data.resize(static_cast<size_t>(out_channels * out.spatial()));
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t ti = 0; ti < out.t; ++ti) {
      for (int64_t hi = 0; hi < out.h; ++hi) {
        for (int64_t wi = 0; wi < out.w; ++wi) {
          // f32, because that is the width `nn.Conv3d` accumulates in — MEASURED,
          // not assumed: F.conv3d returns 0.0 on the separable reduction in
          // tests/vllm/models/test_ltx2_vae.cpp for f32 AND for bf16 tensors,
          // where an f64 accumulator returns 2.5 (#1008).
          float acc = bias != nullptr ? (*bias)[static_cast<size_t>(oc)] : 0.0f;
          for (int64_t ic = 0; ic < ci; ++ic) {
            // BLOCKED, one partial sum per input channel, and this is the ORDER
            // as well as the width. A single naive serial f32 sum over all
            // `ci * kernel^3` taps accumulates error with sqrt of the whole
            // length; splitting it into `ci` blocks of `kernel^3` accumulates
            // with sqrt of each. That is not a local optimisation — torch's f32
            // convolution is a blocked GEMM and sums exactly this way, which is
            // why `torch.sum` on the separable reduction returns 2.0999999 where
            // a naive serial f32 sum returns 0.0. Narrowing the width alone,
            // with the naive order kept, pushed the non-causal tiled golden to
            // 5.00679e-06 against a 5e-06 tolerance — MEASURED, and the reason
            // this loop is shaped this way.
            float tap = 0.0f;
            for (int64_t a = 0; a < kernel; ++a) {
              for (int64_t b = 0; b < kernel; ++b) {
                for (int64_t d = 0; d < kernel; ++d) {
                  tap += padded[static_cast<size_t>(
                             ((ic * pt + ti * stride_t + a) * ph + hi * stride_h + b) * pw +
                             wi * stride_w + d)] *
                         weight[static_cast<size_t>(
                             (((oc * ci + ic) * kernel + a) * kernel + b) * kernel + d)];
                }
              }
            }
            acc += tap;
          }
          out.data[out.At(oc, ti, hi, wi)] = acc;
        }
      }
    }
  }
  return out;
}

// make_linear_nd for dims == 3 (convolution.py:84-85): a 1x1x1 Conv3d.
Volume Linear3d(const Volume& in, int64_t out_channels, const std::vector<float>& weight,
                const std::vector<float>& bias) {
  Volume out;
  out.channels = out_channels;
  out.t = in.t;
  out.h = in.h;
  out.w = in.w;
  out.data.resize(static_cast<size_t>(out_channels * in.spatial()));
  const int64_t n = in.spatial();
  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t i = 0; i < n; ++i) {
      // f32: this is an `nn.Conv3d` upstream too (make_linear_nd's dims==3
      // branch, convolution.py:84-85), so it accumulates at the same width as
      // every other conv on the path.
      float acc = bias[static_cast<size_t>(oc)];
      for (int64_t ic = 0; ic < in.channels; ++ic) {
        acc += in.data[static_cast<size_t>(ic * n + i)] *
               weight[static_cast<size_t>(oc * in.channels + ic)];
      }
      out.data[static_cast<size_t>(oc * n + i)] = acc;
    }
  }
  return out;
}

void Silu(std::vector<float>& x) {
  // f32: `F.silu` computes in the activation dtype. `std::exp(-v)` on a float
  // selects the float overload — spelling it `-static_cast<double>(v)` would
  // quietly restore the f64 path this deliberately leaves.
  for (float& v : x) v = v / (1.0f + std::exp(-v));
}

// PixelNorm() with its DEFAULT eps of 1e-8 (normalization.py:22, reached bare
// from video_vae/resnet.py:46 and conv_video_decoder.py:243) — NOT the 1e-6 the
// audio VAE gets through build_normalization_layer.
void PixelNorm(std::vector<float>& x, int64_t channels, int64_t spatial, double eps) {
  for (int64_t i = 0; i < spatial; ++i) {
    // f32: `torch.mean(x**2, dim=...)` runs in the activation dtype
    // (normalization.py:37-40).
    float mean_sq = 0.0f;
    for (int64_t c = 0; c < channels; ++c) {
      const float v = x[static_cast<size_t>(c * spatial + i)];
      mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(channels);
    // The reciprocal is f32 too: upstream is `x / torch.sqrt(mean_sq + eps)`
    // with every term in the tensor dtype (normalization.py:37-40). `eps`
    // remains an f64 PARAMETER because it is a pinned config threshold
    // (Ltx2ConvVideoDecoderConfig::pixel_norm_eps); it is narrowed here, at the
    // one point it enters the arithmetic.
    const float inv = 1.0f / std::sqrt(mean_sq + static_cast<float>(eps));
    for (int64_t c = 0; c < channels; ++c) {
      x[static_cast<size_t>(c * spatial + i)] = x[static_cast<size_t>(c * spatial + i)] * inv;
    }
  }
}

// The fields the shared convolution/normalization primitives need, so ONE set of
// them serves both the decoder and the encoder rather than each half growing its
// own causal pad. Nothing here is a new degree of freedom: every member is read
// straight off whichever config the caller holds.
struct VideoConvSpec {
  Ltx2NormLayer norm_layer = Ltx2NormLayer::kPixelNorm;
  int64_t norm_num_groups = 32;
  double norm_eps = 1e-6;
  double pixel_norm_eps = 1e-8;
  Ltx2PaddingMode spatial_padding_mode = Ltx2PaddingMode::kZeros;
  // The per-CALL causal flag, i.e. what upstream passes as `causal=` rather than
  // what it passes to a constructor. The decoder takes it from `self.causal`
  // (conv_video_decoder.py:307); the encoder never passes it at all and so always
  // gets the `causal: bool = True` DEFAULT (convolution.py:304).
  bool causal = true;
};

VideoConvSpec SpecOf(const Ltx2ConvVideoDecoderConfig& config) {
  VideoConvSpec spec;
  spec.norm_layer = config.norm_layer;
  spec.norm_num_groups = config.norm_num_groups;
  spec.norm_eps = config.norm_eps;
  spec.pixel_norm_eps = config.pixel_norm_eps;
  spec.spatial_padding_mode = config.spatial_padding_mode;
  spec.causal = config.causal;
  return spec;
}

void ApplyNorm(const VideoConvSpec& config, std::vector<float>& x, int64_t channels,
               int64_t spatial, const Ltx2VaeWeights& weights, const std::string& prefix) {
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(x, channels, spatial, config.pixel_norm_eps);
    return;
  }
  MiniMaxH3GroupNorm3d(x, channels, spatial, config.norm_num_groups,
                       weights.Get(prefix + ".weight"), weights.Get(prefix + ".bias"),
                       config.norm_eps);
}

// ---------------------------------------------------------------------------
// PixArtAlphaCombinedTimestepSizeEmbeddings (timestep_embedding.py:118-141) at
// batch 1: Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0) followed
// by TimestepEmbedding(256 -> embedding_dim) with a SiLU between its two linears.
// ---------------------------------------------------------------------------
std::vector<float> TimestepEmbedding(double timestep, int64_t embedding_dim,
                                     const Ltx2VaeWeights& weights, const std::string& prefix) {
  constexpr int64_t kProjChannels = 256;
  constexpr double kMaxPeriod = 10000.0;
  const int64_t half = kProjChannels / 2;
  // DELIBERATE f64 EXCEPTION, and the only one on this path. `proj` is a
  // transcendental CONSTANT table — 256 cos/sin values built once per block from
  // the timestep, never a per-element data-path accumulation — so it is off
  // every hot path, and evaluating it in f64 sits closer to the exact value that
  // upstream's f32 `torch.arange` table approximates. Everything downstream of
  // it is f32 (#1008).
  std::vector<double> proj(static_cast<size_t>(kProjChannels));
  for (int64_t i = 0; i < half; ++i) {
    // downscale_freq_shift = 0, so the divisor is exactly half_dim.
    const double exponent = -std::log(kMaxPeriod) * static_cast<double>(i) / static_cast<double>(half);
    const double angle = timestep * std::exp(exponent);
    // flip_sin_to_cos=True puts COS first (timestep_embedding.py:87-89).
    proj[static_cast<size_t>(i)] = std::cos(angle);
    proj[static_cast<size_t>(half + i)] = std::sin(angle);
  }

  const std::vector<float>& w1 = weights.Get(prefix + ".timestep_embedder.linear_1.weight");
  const std::vector<float>& b1 = weights.Get(prefix + ".timestep_embedder.linear_1.bias");
  const std::vector<float>& w2 = weights.Get(prefix + ".timestep_embedder.linear_2.weight");
  const std::vector<float>& b2 = weights.Get(prefix + ".timestep_embedder.linear_2.bias");
  VT_CHECK(static_cast<int64_t>(w1.size()) == embedding_dim * kProjChannels,
           "ltx2 timestep embedding: linear_1 shape does not match the embedding dim");

  // f32 for both `nn.Linear` accumulators and for the hidden activation between
  // them: upstream's TimestepEmbedder is two plain Linears with a SiLU, all in
  // the activation dtype. The frequency table above stays f64 — see its note.
  std::vector<float> hidden(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    float acc = b1[static_cast<size_t>(o)];
    for (int64_t i = 0; i < kProjChannels; ++i) {
      acc += static_cast<float>(proj[static_cast<size_t>(i)]) *
             w1[static_cast<size_t>(o * kProjChannels + i)];
    }
    hidden[static_cast<size_t>(o)] = acc / (1.0f + std::exp(-acc));  // SiLU
  }
  std::vector<float> out(static_cast<size_t>(embedding_dim));
  for (int64_t o = 0; o < embedding_dim; ++o) {
    float acc = b2[static_cast<size_t>(o)];
    for (int64_t i = 0; i < embedding_dim; ++i) {
      acc += hidden[static_cast<size_t>(i)] * w2[static_cast<size_t>(o * embedding_dim + i)];
    }
    out[static_cast<size_t>(o)] = acc;
  }
  return out;
}

// _feed_spatial_noise (resnet.py:104-119): ONE [H, W] draw, broadcast over batch,
// channels and TIME, scaled per channel. Drawing a full [C, T, H, W] block
// instead still yields a finite, plausible clip.
void FeedSpatialNoise(Volume& x, const std::vector<float>& per_channel_scale,
                      Ltx2NoiseStream* noise) {
  VT_CHECK(noise != nullptr,
           "ltx2 video vae: a block sets inject_noise but no noise stream was supplied");
  const std::vector<float> plane = noise->Draw(x.h * x.w);
  VT_CHECK(static_cast<int64_t>(plane.size()) == x.h * x.w,
           "ltx2 video vae: the noise stream returned the wrong element count");
  for (int64_t c = 0; c < x.channels; ++c) {
    // f32: upstream scales and adds the noise plane in the activation dtype.
    const float scale = per_channel_scale[static_cast<size_t>(c)];
    for (int64_t ti = 0; ti < x.t; ++ti) {
      for (int64_t hi = 0; hi < x.h; ++hi) {
        for (int64_t wi = 0; wi < x.w; ++wi) {
          x.data[x.At(c, ti, hi, wi)] += plane[static_cast<size_t>(hi * x.w + wi)] * scale;
        }
      }
    }
  }
}

// One ada-LN group applied in place: x * (1 + scale) + shift, with the pair taken
// from `table[row]` plus `embed[row]` (resnet.py:135-147).
void ApplyAdaLn(Volume& x, const std::vector<float>& table, const std::vector<float>& embed,
                int64_t rows, int64_t shift_row, int64_t scale_row) {
  const int64_t c = x.channels;
  VT_CHECK(static_cast<int64_t>(table.size()) == rows * c,
           "ltx2 video vae: scale_shift_table does not match the channel count");
  VT_CHECK(static_cast<int64_t>(embed.size()) == rows * c,
           "ltx2 video vae: timestep embedding does not match rows x channels");
  const int64_t n = x.spatial();
  for (int64_t ch = 0; ch < c; ++ch) {
    // f32: upstream adds two f32 tensors and applies `x * (1 + scale) + shift`
    // in the activation dtype (resnet.py:135-147).
    const float shift = table[static_cast<size_t>(shift_row * c + ch)] +
                        embed[static_cast<size_t>(shift_row * c + ch)];
    const float scale = table[static_cast<size_t>(scale_row * c + ch)] +
                        embed[static_cast<size_t>(scale_row * c + ch)];
    for (int64_t i = 0; i < n; ++i) {
      x.data[static_cast<size_t>(ch * n + i)] =
          x.data[static_cast<size_t>(ch * n + i)] * (1.0f + scale) + shift;
    }
  }
}

// ResnetBlock3D.forward (resnet.py:121-186).
Volume ResnetBlock3d(const VideoConvSpec& config, const Ltx2VaeWeights& weights,
                     const std::string& prefix, const Volume& input, int64_t out_channels,
                     bool inject_noise, bool timestep_conditioning,
                     const std::vector<float>* timestep_embed, Ltx2NoiseStream* noise) {
  Volume hidden = input;
  ApplyNorm(config, hidden.data, hidden.channels, hidden.spatial(), weights, prefix + ".norm1");
  if (timestep_conditioning) {
    VT_CHECK(timestep_embed != nullptr,
             "ltx2 video vae: a timestep-conditioned block needs a timestep embedding");
    // ada_values rows are (shift1, scale1, shift2, scale2).
    ApplyAdaLn(hidden, weights.Get(prefix + ".scale_shift_table"), *timestep_embed, 4, 0, 1);
  }
  Silu(hidden.data);
  hidden = CausalConv3d(hidden, out_channels, 3, config.causal, config.spatial_padding_mode,
                        weights.Get(prefix + ".conv1.conv.weight"),
                        &weights.Get(prefix + ".conv1.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(hidden, weights.Get(prefix + ".per_channel_scale1"), noise);
  }

  ApplyNorm(config, hidden.data, hidden.channels, hidden.spatial(), weights, prefix + ".norm2");
  if (timestep_conditioning) {
    ApplyAdaLn(hidden, weights.Get(prefix + ".scale_shift_table"), *timestep_embed, 4, 2, 3);
  }
  Silu(hidden.data);
  hidden = CausalConv3d(hidden, out_channels, 3, config.causal, config.spatial_padding_mode,
                        weights.Get(prefix + ".conv2.conv.weight"),
                        &weights.Get(prefix + ".conv2.conv.bias"));
  if (inject_noise) {
    FeedSpatialNoise(hidden, weights.Get(prefix + ".per_channel_scale2"), noise);
  }

  Volume residual = input;
  if (input.channels != out_channels) {
    // norm3 is GroupNorm with ONE group — a LayerNorm over (C, T, H, W) that
    // works in the (B, C, ...) layout without a rearrange (resnet.py:91-97).
    MiniMaxH3GroupNorm3d(residual.data, residual.channels, residual.spatial(), 1,
                         weights.Get(prefix + ".norm3.weight"), weights.Get(prefix + ".norm3.bias"),
                         config.norm_eps);
    residual = Linear3d(residual, out_channels, weights.Get(prefix + ".conv_shortcut.weight"),
                        weights.Get(prefix + ".conv_shortcut.bias"));
  }
  VT_CHECK(residual.data.size() == hidden.data.size(),
           "ltx2 video vae: resnet residual and main-branch shapes must match");
  for (size_t i = 0; i < hidden.data.size(); ++i) hidden.data[i] += residual.data[i];
  return hidden;
}

// DepthToSpaceUpsample.forward (sampling.py:93-123). The channel unpack is
// `(c p1 p2 p3)` with p1 temporal and p2/p3 spatial, and a temporal stride of 2
// DROPS THE FIRST FRAME afterwards.
Volume DepthToSpaceUpsample(const VideoConvSpec& config, const Ltx2VaeWeights& weights,
                            const std::string& prefix, const Volume& x, int64_t st, int64_t sh,
                            int64_t sw, int64_t reduction, bool residual) {
  const int64_t stride_product = st * sh * sw;
  const int64_t conv_out_channels = stride_product * x.channels / reduction;

  auto expand = [&](const Volume& packed) {
    Volume out;
    out.channels = packed.channels / stride_product;
    out.t = packed.t * st;
    out.h = packed.h * sh;
    out.w = packed.w * sw;
    out.data.resize(static_cast<size_t>(out.channels * out.spatial()));
    for (int64_t c = 0; c < out.channels; ++c) {
      for (int64_t p1 = 0; p1 < st; ++p1) {
        for (int64_t p2 = 0; p2 < sh; ++p2) {
          for (int64_t p3 = 0; p3 < sw; ++p3) {
            const int64_t src_c = ((c * st + p1) * sh + p2) * sw + p3;
            for (int64_t ti = 0; ti < packed.t; ++ti) {
              for (int64_t hi = 0; hi < packed.h; ++hi) {
                for (int64_t wi = 0; wi < packed.w; ++wi) {
                  out.data[out.At(c, ti * st + p1, hi * sh + p2, wi * sw + p3)] =
                      packed.data[packed.At(src_c, ti, hi, wi)];
                }
              }
            }
          }
        }
      }
    }
    return out;
  };
  auto drop_first_frame = [&](const Volume& v) {
    Volume out;
    out.channels = v.channels;
    out.t = v.t - 1;
    out.h = v.h;
    out.w = v.w;
    out.data.resize(static_cast<size_t>(out.channels * out.spatial()));
    for (int64_t c = 0; c < out.channels; ++c) {
      for (int64_t ti = 0; ti < out.t; ++ti) {
        for (int64_t hi = 0; hi < out.h; ++hi) {
          for (int64_t wi = 0; wi < out.w; ++wi) {
            out.data[out.At(c, ti, hi, wi)] = v.data[v.At(c, ti + 1, hi, wi)];
          }
        }
      }
    }
    return out;
  };

  Volume skip;
  if (residual) {
    // The residual expands the INPUT itself and then repeats it up to the output
    // width (sampling.py:98-110).
    Volume expanded = expand(x);
    const int64_t repeat = stride_product / reduction;
    Volume repeated;
    repeated.channels = expanded.channels * repeat;
    repeated.t = expanded.t;
    repeated.h = expanded.h;
    repeated.w = expanded.w;
    repeated.data.resize(static_cast<size_t>(repeated.channels * repeated.spatial()));
    for (int64_t r = 0; r < repeat; ++r) {
      std::copy(expanded.data.begin(), expanded.data.end(),
                repeated.data.begin() +
                    static_cast<ptrdiff_t>(r * expanded.channels * expanded.spatial()));
    }
    skip = st == 2 ? drop_first_frame(repeated) : repeated;
  }

  Volume packed = CausalConv3d(x, conv_out_channels, 3, config.causal, config.spatial_padding_mode,
                               weights.Get(prefix + ".conv.conv.weight"),
                               &weights.Get(prefix + ".conv.conv.bias"));
  Volume out = expand(packed);
  if (st == 2) out = drop_first_frame(out);
  if (residual) {
    VT_CHECK(skip.data.size() == out.data.size(),
             "ltx2 video vae: depth-to-space residual and main-branch shapes must match");
    for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += skip.data[i];
  }
  return out;
}

// AttnBlock3D.forward (attention.py:58-69): SINGLE-HEAD spatial self-attention
// PER FRAME, with frames folded into the batch — there is deliberately no
// cross-frame interaction, so this block does not break temporal causality.
Volume AttnBlock3d(const Ltx2VaeWeights& weights, const std::string& prefix, const Volume& x) {
  const int64_t c = x.channels;
  const int64_t n = x.h * x.w;
  const std::vector<float>& gamma = weights.Get(prefix + ".norm.gamma");
  const std::vector<float>& qkv_w = weights.Get(prefix + ".to_qkv.weight");
  const std::vector<float>& qkv_b = weights.Get(prefix + ".to_qkv.bias");
  const std::vector<float>& proj_w = weights.Get(prefix + ".proj.weight");
  const std::vector<float>& proj_b = weights.Get(prefix + ".proj.bias");
  const double norm_scale = std::sqrt(static_cast<double>(c));
  const double attn_scale = 1.0 / std::sqrt(static_cast<double>(c));

  Volume out = x;
  // f32 activations, not f64. Upstream holds q/k/v and the attention output in
  // the tensor dtype (attention.py:63-67) and never promotes; these six buffers
  // are the block's whole scratch footprint, so the width is bytes as well as
  // arithmetic. `norm_scale` and `attn_scale` above stay f64 — upstream's
  // `channels**0.5` is a Python float evaluated once per block.
  std::vector<float> normed(static_cast<size_t>(c * n));
  std::vector<float> q(static_cast<size_t>(c * n)), k(static_cast<size_t>(c * n)),
      v(static_cast<size_t>(c * n));
  std::vector<float> scores(static_cast<size_t>(n));
  std::vector<float> attended(static_cast<size_t>(c * n));

  for (int64_t frame = 0; frame < x.t; ++frame) {
    // _RMSNorm2D: F.normalize(x, dim=1) * (sqrt(C) * gamma) — an L2 normalize with
    // torch's 1e-12 floor, not a mean-square RMS.
    for (int64_t i = 0; i < n; ++i) {
      // f32: `F.normalize(x, dim=1)` computes its norm in the input dtype
      // (attention.py:23). torch's 1e-12 floor stays f64 — it is a threshold.
      float sum_sq = 0.0f;
      for (int64_t ch = 0; ch < c; ++ch) {
        const float value = x.data[x.At(ch, frame, i / x.w, i % x.w)];
        sum_sq += value * value;
      }
      const float inv = static_cast<float>(
          1.0 / std::max(std::sqrt(static_cast<double>(sum_sq)), kLtx2RmsNorm2dEps));
      // Same left-to-right association the f64 arm used; only the width changes.
      const float norm_scale_f = static_cast<float>(norm_scale);
      for (int64_t ch = 0; ch < c; ++ch) {
        normed[static_cast<size_t>(ch * n + i)] = x.data[x.At(ch, frame, i / x.w, i % x.w)] * inv *
                                                  norm_scale_f * gamma[static_cast<size_t>(ch)];
      }
    }
    // to_qkv is a 1x1 Conv2d emitting [q | k | v] along the channel axis, and the
    // rearrange to tokens keeps that split on the LAST axis (attention.py:63-64).
    // f32: `to_qkv` is a 1x1 nn.Conv2d (attention.py:55), the same accumulator
    // width as every other conv here.
    for (int64_t oc = 0; oc < 3 * c; ++oc) {
      std::vector<float>& dst = oc < c ? q : (oc < 2 * c ? k : v);
      const int64_t row = oc % c;
      for (int64_t i = 0; i < n; ++i) {
        float acc = qkv_b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += normed[static_cast<size_t>(ic * n + i)] * qkv_w[static_cast<size_t>(oc * c + ic)];
        }
        dst[static_cast<size_t>(row * n + i)] = acc;
      }
    }
    // f32: SDPA computes scores, softmax and the value-weighted sum in the
    // tensor dtype (attention.py:65). `attn_scale` stays f64 for the same reason
    // `norm_scale` does.
    const float attn_scale_f = static_cast<float>(attn_scale);
    for (int64_t i = 0; i < n; ++i) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < n; ++j) {
        float dot = 0.0f;
        for (int64_t ch = 0; ch < c; ++ch) {
          dot += q[static_cast<size_t>(ch * n + i)] * k[static_cast<size_t>(ch * n + j)];
        }
        scores[static_cast<size_t>(j)] = dot * attn_scale_f;
        max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
      }
      float sum = 0.0f;
      for (int64_t j = 0; j < n; ++j) {
        scores[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - max_score);
        sum += scores[static_cast<size_t>(j)];
      }
      for (int64_t ch = 0; ch < c; ++ch) {
        float acc = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
          acc += scores[static_cast<size_t>(j)] * v[static_cast<size_t>(ch * n + j)];
        }
        attended[static_cast<size_t>(ch * n + i)] = acc / sum;
      }
    }
    // f32: `proj` is a 1x1 nn.Conv2d (attention.py:56).
    for (int64_t oc = 0; oc < c; ++oc) {
      for (int64_t i = 0; i < n; ++i) {
        float acc = proj_b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < c; ++ic) {
          acc += attended[static_cast<size_t>(ic * n + i)] * proj_w[static_cast<size_t>(oc * c + ic)];
        }
        out.data[out.At(oc, frame, i / x.w, i % x.w)] += acc;
      }
    }
  }
  return out;
}

}  // namespace

Ltx2VideoDecoderKind Ltx2ParseVideoDecoderKind(const std::string& vae_class_name) {
  // model_configurator.py:18-34: the conv decoder is the DEFAULT when the field is
  // absent, and is otherwise selected by the exact class name.
  if (vae_class_name.empty() || vae_class_name == "CausalVideoAutoencoder") {
    return Ltx2VideoDecoderKind::kConv;
  }
  return Ltx2VideoDecoderKind::kDiffusion;
}

Ltx2VideoFrames Ltx2ConvVideoDecode(const Ltx2ConvVideoDecoderConfig& config,
                                    const Ltx2VaeWeights& weights,
                                    const std::vector<float>& latent, int64_t latent_channels,
                                    int64_t latent_t, int64_t latent_h, int64_t latent_w,
                                    Ltx2NoiseStream* noise, const double* timestep) {
  VT_CHECK(latent_channels == config.in_channels,
           "ltx2 video vae: latent channel count does not match in_channels");
  VT_CHECK(static_cast<int64_t>(latent.size()) == latent_channels * latent_t * latent_h * latent_w,
           "ltx2 video vae: latent size does not match [C, T, H, W]");
  const std::string p = config.prefix;
  const VideoConvSpec spec = SpecOf(config);

  Volume x;
  x.channels = latent_channels;
  x.t = latent_t;
  x.h = latent_h;
  x.w = latent_w;
  x.data = latent;

  // --- noise + denormalize (conv_video_decoder.py:286-301) ---
  if (config.timestep_conditioning) {
    VT_CHECK(noise != nullptr,
             "ltx2 video vae: timestep conditioning injects noise but no noise stream was supplied");
    const std::vector<float> drawn = noise->Draw(static_cast<int64_t>(x.data.size()));
    VT_CHECK(drawn.size() == x.data.size(),
             "ltx2 video vae: the noise stream returned the wrong element count");
    // f32: the blend runs in the activation dtype upstream. The two scalars are
    // config values, so they are narrowed once rather than per element.
    const float noise_scale = static_cast<float>(config.decode_noise_scale);
    const float keep_scale = static_cast<float>(1.0 - config.decode_noise_scale);
    for (size_t i = 0; i < x.data.size(); ++i) {
      x.data[i] = drawn[i] * noise_scale + keep_scale * x.data[i];
    }
  }
  {
    const std::vector<float>& std_of_means =
        weights.Get(p + "per_channel_statistics.std-of-means");
    const std::vector<float>& mean_of_means =
        weights.Get(p + "per_channel_statistics.mean-of-means");
    VT_CHECK(static_cast<int64_t>(std_of_means.size()) == latent_channels &&
                 static_cast<int64_t>(mean_of_means.size()) == latent_channels,
             "ltx2 video vae: per-channel statistics must have one value per latent channel");
    const int64_t n = x.spatial();
    // f32: upstream's de-normalize is `latent * std + mean` on f32/bf16 tensors.
    for (int64_t c = 0; c < latent_channels; ++c) {
      const float std_c = std_of_means[static_cast<size_t>(c)];
      const float mean_c = mean_of_means[static_cast<size_t>(c)];
      for (int64_t i = 0; i < n; ++i) {
        x.data[static_cast<size_t>(c * n + i)] =
            x.data[static_cast<size_t>(c * n + i)] * std_c + mean_c;
      }
    }
  }

  const double scaled_timestep =
      config.timestep_conditioning
          ? (timestep != nullptr ? *timestep : config.decode_timestep) *
                static_cast<double>(weights.Get(p + "timestep_scale_multiplier")[0])
          : 0.0;

  // --- conv_in widens the latents to the bottleneck ---
  int64_t multiplier = 1;
  for (const Ltx2VideoDecoderBlock& block : config.decoder_blocks) {
    if (block.name == "compress_time" || block.name == "compress_space" ||
        block.name == "compress_all") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 1;
    } else if (block.name == "res_x_y") {
      multiplier *= block.multiplier != 0 ? block.multiplier : 2;
    }
  }
  // TWO DIFFERENT `causal` FLAGS, and passing `config.causal` here is correct.
  // Upstream builds conv_in with `causal=True` (conv_video_decoder.py:216), but
  // that constructor argument only selects the MODULE — it is what makes conv_in a
  // CausalConv3d at all. The one-sidedness of any given call comes from the
  // separate per-call argument, `self.conv_in(sample, causal=self.causal)`
  // (conv_video_decoder.py:307), and that is `self.causal`, i.e. this config's
  // field. So `config.causal` is the value that belongs here; hardcoding `true`
  // would silently make a non-causal decoder pad one-sidedly.
  x = CausalConv3d(x, config.base_channels * multiplier, 3, config.causal,
                   config.spatial_padding_mode, weights.Get(p + "conv_in.conv.weight"),
                   &weights.Get(p + "conv_in.conv.bias"));

  // --- the reversed block walk (conv_video_decoder.py:222-238, 315-326) ---
  int64_t index = 0;
  for (auto it = config.decoder_blocks.rbegin(); it != config.decoder_blocks.rend(); ++it, ++index) {
    const Ltx2VideoDecoderBlock& block = *it;
    const std::string bp = p + "up_blocks." + std::to_string(index);
    if (block.name == "res_x") {
      std::vector<float> embed;
      const std::vector<float>* embed_ptr = nullptr;
      if (config.timestep_conditioning) {
        embed = TimestepEmbedding(scaled_timestep, x.channels * 4, weights, bp + ".time_embedder");
        embed_ptr = &embed;
      }
      for (int64_t i = 0; i < block.num_layers; ++i) {
        x = ResnetBlock3d(spec, weights, bp + ".res_blocks." + std::to_string(i), x, x.channels,
                          block.inject_noise, config.timestep_conditioning, embed_ptr, noise);
      }
    } else if (block.name == "res_x_y") {
      const int64_t out_channels = x.channels / (block.multiplier != 0 ? block.multiplier : 2);
      // _make_decoder_block forces timestep_conditioning=False for res_x_y
      // (conv_video_decoder.py:107).
      x = ResnetBlock3d(spec, weights, bp, x, out_channels, block.inject_noise,
                        /*timestep_conditioning=*/false, nullptr, noise);
    } else if (block.name == "attn") {
      x = AttnBlock3d(weights, bp, x);
    } else if (block.name == "compress_time" || block.name == "compress_space" ||
               block.name == "compress_all") {
      const int64_t st = block.name == "compress_space" ? 1 : 2;
      const int64_t ss = block.name == "compress_time" ? 1 : 2;
      x = DepthToSpaceUpsample(spec, weights, bp, x, st, ss, ss,
                               block.multiplier != 0 ? block.multiplier : 1,
                               block.name == "compress_all" && block.residual);
    } else if (block.name == "attn_res_x") {
      VT_CHECK(false,
               "ltx2 video vae: the `attn_res_x` decoder block cannot be built — upstream passes "
               "`attention_head_dim` to UNetMidBlock3D, which does not accept it "
               "(conv_video_decoder.py:85-96 vs video_vae/resnet.py:210-222)");
    } else {
      VT_CHECK(false, "ltx2 video vae: unknown decoder block `" + block.name + "`");
    }
  }

  // --- conv_norm_out -> ada-LN -> SiLU -> conv_out ---
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(x.data, x.channels, x.spatial(), config.pixel_norm_eps);
  } else {
    MiniMaxH3GroupNorm3d(x.data, x.channels, x.spatial(), config.norm_num_groups,
                         weights.Get(p + "conv_norm_out.weight"),
                         weights.Get(p + "conv_norm_out.bias"), config.norm_eps);
  }
  if (config.timestep_conditioning) {
    const std::vector<float> embed =
        TimestepEmbedding(scaled_timestep, x.channels * 2, weights, p + "last_time_embedder");
    // ada_values rows are (shift, scale) — two, not the resnet's four.
    ApplyAdaLn(x, weights.Get(p + "last_scale_shift_table"), embed, 2, 0, 1);
  }
  Silu(x.data);
  x = CausalConv3d(x, config.out_channels * config.patch_size * config.patch_size, 3, config.causal,
                   config.spatial_padding_mode, weights.Get(p + "conv_out.conv.weight"),
                   &weights.Get(p + "conv_out.conv.bias"));

  // --- unpatchify (ops.py:35-60): `b (c p r q) f h w -> b c (f p) (h q) (w r)`
  // with p = patch_size_t = 1. NOTE h takes q and w takes r; swapping them
  // transposes every patch.
  const int64_t q = config.patch_size;
  const int64_t r = config.patch_size;
  Ltx2VideoFrames out;
  out.channels = config.out_channels;
  out.frames = x.t;
  out.height = x.h * q;
  out.width = x.w * r;
  out.data.resize(
      static_cast<size_t>(out.channels * out.frames * out.height * out.width));
  for (int64_t c = 0; c < out.channels; ++c) {
    for (int64_t ri = 0; ri < r; ++ri) {
      for (int64_t qi = 0; qi < q; ++qi) {
        const int64_t src_c = (c * r + ri) * q + qi;
        for (int64_t f = 0; f < x.t; ++f) {
          for (int64_t hi = 0; hi < x.h; ++hi) {
            for (int64_t wi = 0; wi < x.w; ++wi) {
              out.data[static_cast<size_t>(
                  ((c * out.frames + f) * out.height + hi * q + qi) * out.width + wi * r + ri)] =
                  x.data[x.At(src_c, f, hi, wi)];
            }
          }
        }
      }
    }
  }
  return out;
}

Ltx2VideoFrames Ltx2VideoDecode(Ltx2VideoDecoderKind kind,
                                const Ltx2ConvVideoDecoderConfig& config,
                                const Ltx2VaeWeights& weights, const std::vector<float>& latent,
                                int64_t latent_channels, int64_t latent_t, int64_t latent_h,
                                int64_t latent_w, Ltx2NoiseStream* noise, const double* timestep) {
  // REFUSE, never downgrade: falling back to the conv decoder would return a
  // lower-quality render as if it were the requested one, and no gate this
  // project owns could detect that (.agents/specs/ltx-2-5.md section 0 item 2).
  VT_CHECK(kind != Ltx2VideoDecoderKind::kDiffusion,
           "ltx2 video vae: this checkpoint asks for the DIFFUSION video decoder "
           "(NADiffusionDecoder / DiffusionVideoDecoder), which is NOT implemented — it needs a "
           "neighborhood-attention kernel and has its own row. It is refused rather than "
           "downgraded to the Conv video VAE, which would silently return a worse render");
  return Ltx2ConvVideoDecode(config, weights, latent, latent_channels, latent_t, latent_h, latent_w,
                             noise, timestep);
}

// ===========================================================================
// THE ENCODER HALF (video_vae.py:39-336), which phase L4 recorded as owed.
//
// It lives in this translation unit deliberately, so that `CausalConv3d`,
// `PixelNorm`, `ApplyNorm`, `ResnetBlock3d` and `AttnBlock3d` are the SAME
// functions the decoder is gated on rather than a second copy of each. The one
// primitive the decoder never needed is a STRIDE on the causal convolution, and
// that was added to the shared function above rather than forked here.
// ===========================================================================

namespace {

// patchify (ops.py:6-32), the 5-D arm with patch_size_t = 1:
//   `b c (f p) (h q) (w r) -> b (c p r q) f h w`
// r is the OUTER spatial factor and q the inner one, and h takes q while w takes
// r. That is the exact inverse of the decoder's unpatchify above; swapping r and
// q transposes every patch and still type-checks.
Volume Patchify(const Volume& in, int64_t patch) {
  if (patch == 1) return in;
  VT_CHECK(in.h % patch == 0 && in.w % patch == 0,
           "ltx2 video encoder: height and width must be whole multiples of patch_size");
  Volume out;
  out.channels = in.channels * patch * patch;
  out.t = in.t;
  out.h = in.h / patch;
  out.w = in.w / patch;
  out.data.resize(static_cast<size_t>(out.channels * out.spatial()));
  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t ri = 0; ri < patch; ++ri) {
      for (int64_t qi = 0; qi < patch; ++qi) {
        const int64_t dst_c = (c * patch + ri) * patch + qi;
        for (int64_t f = 0; f < out.t; ++f) {
          for (int64_t hi = 0; hi < out.h; ++hi) {
            for (int64_t wi = 0; wi < out.w; ++wi) {
              out.data[out.At(dst_c, f, hi, wi)] =
                  in.data[in.At(c, f, hi * patch + qi, wi * patch + ri)];
            }
          }
        }
      }
    }
  }
  return out;
}

// The space-to-depth fold both branches of SpaceToDepthDownsample share:
//   `b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w`   (sampling.py:43-49, 55-61)
Volume SpaceToDepthFold(const Volume& in, int64_t st, int64_t sh, int64_t sw) {
  VT_CHECK(in.t % st == 0 && in.h % sh == 0 && in.w % sw == 0,
           "ltx2 video encoder: space-to-depth needs each axis to be a whole multiple of its "
           "stride");
  Volume out;
  out.channels = in.channels * st * sh * sw;
  out.t = in.t / st;
  out.h = in.h / sh;
  out.w = in.w / sw;
  out.data.resize(static_cast<size_t>(out.channels * out.spatial()));
  for (int64_t c = 0; c < in.channels; ++c) {
    for (int64_t p1 = 0; p1 < st; ++p1) {
      for (int64_t p2 = 0; p2 < sh; ++p2) {
        for (int64_t p3 = 0; p3 < sw; ++p3) {
          const int64_t dst_c = ((c * st + p1) * sh + p2) * sw + p3;
          for (int64_t ti = 0; ti < out.t; ++ti) {
            for (int64_t hi = 0; hi < out.h; ++hi) {
              for (int64_t wi = 0; wi < out.w; ++wi) {
                out.data[out.At(dst_c, ti, hi, wi)] =
                    in.data[in.At(c, ti * st + p1, hi * sh + p2, wi * sw + p3)];
              }
            }
          }
        }
      }
    }
  }
  return out;
}

// SpaceToDepthDownsample.forward (sampling.py:34-65). Three things that fail
// silently and are therefore spelled out:
//  * a temporal stride of 2 DUPLICATES FRAME 0 first (sampling.py:39-40), and the
//    duplication happens BEFORE both the skip fold and the convolution;
//  * the skip is a GROUP MEAN over `group_size` contiguous folded channels
//    (`b (c g) d h w -> b c g d h w` then `.mean(dim=2)`, sampling.py:50-51) —
//    c is the OUTER factor, so group g is contiguous;
//  * the convolution emits `out_channels / prod(stride)` channels and the fold
//    multiplies them back up (sampling.py:27, 55-61).
Volume SpaceToDepthDownsample(const VideoConvSpec& spec, const Ltx2VaeWeights& weights,
                              const std::string& prefix, const Volume& x, int64_t st, int64_t sh,
                              int64_t sw, int64_t out_channels) {
  const int64_t stride_product = st * sh * sw;
  VT_CHECK(out_channels % stride_product == 0,
           "ltx2 video encoder: SpaceToDepthDownsample needs out_channels divisible by the stride "
           "product (sampling.py:27)");
  const int64_t conv_out_channels = out_channels / stride_product;
  const int64_t folded = x.channels * stride_product;
  VT_CHECK(folded % out_channels == 0,
           "ltx2 video encoder: SpaceToDepthDownsample needs in_channels * prod(stride) divisible "
           "by out_channels (sampling.py:23)");
  const int64_t group_size = folded / out_channels;

  Volume grown = x;
  if (st == 2) {
    grown.t = x.t + 1;
    grown.data.resize(static_cast<size_t>(grown.channels * grown.spatial()));
    for (int64_t c = 0; c < grown.channels; ++c) {
      for (int64_t ti = 0; ti < grown.t; ++ti) {
        const int64_t src_t = ti == 0 ? 0 : ti - 1;
        for (int64_t hi = 0; hi < grown.h; ++hi) {
          for (int64_t wi = 0; wi < grown.w; ++wi) {
            grown.data[grown.At(c, ti, hi, wi)] = x.data[x.At(c, src_t, hi, wi)];
          }
        }
      }
    }
  }

  // --- the skip: fold, then average each contiguous group of `group_size` ---
  const Volume folded_in = SpaceToDepthFold(grown, st, sh, sw);
  Volume skip;
  skip.channels = out_channels;
  skip.t = folded_in.t;
  skip.h = folded_in.h;
  skip.w = folded_in.w;
  skip.data.resize(static_cast<size_t>(skip.channels * skip.spatial()));
  const int64_t n = skip.spatial();
  for (int64_t c = 0; c < out_channels; ++c) {
    for (int64_t i = 0; i < n; ++i) {
      // f32: upstream's group mean runs in the activation dtype.
      float acc = 0.0f;
      for (int64_t g = 0; g < group_size; ++g) {
        acc += folded_in.data[static_cast<size_t>((c * group_size + g) * n + i)];
      }
      skip.data[static_cast<size_t>(c * n + i)] = acc / static_cast<float>(group_size);
    }
  }

  // --- the conv branch, at stride 1, on the SAME duplicated input ---
  const Volume convolved =
      CausalConv3d(grown, conv_out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   weights.Get(prefix + ".conv.conv.weight"),
                   &weights.Get(prefix + ".conv.conv.bias"));
  Volume out = SpaceToDepthFold(convolved, st, sh, sw);
  VT_CHECK(out.data.size() == skip.data.size(),
           "ltx2 video encoder: SpaceToDepthDownsample skip and conv shapes must match");
  for (size_t i = 0; i < out.data.size(); ++i) out.data[i] += skip.data[i];
  return out;
}

bool StartsWith(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

VideoConvSpec SpecOf(const Ltx2ConvVideoEncoderConfig& config) {
  VideoConvSpec spec;
  spec.norm_layer = config.norm_layer;
  spec.norm_num_groups = config.norm_num_groups;
  spec.norm_eps = config.norm_eps;
  spec.pixel_norm_eps = config.pixel_norm_eps;
  spec.spatial_padding_mode = config.spatial_padding_mode;
  // The ENCODER never passes `causal=` to anything it calls (video_vae.py:292-299),
  // so every convolution takes the `causal: bool = True` DEFAULT. There is no
  // knob, and inventing one would let a caller build a non-causal encoder upstream
  // cannot produce.
  spec.causal = true;
  return spec;
}

// `_make_encoder_block`'s out_channels arithmetic (video_vae.py:39-145). The
// plain strided convolutions keep `in_channels`; every `*_x_y` and `*_res` kind
// multiplies by `block_config.get("multiplier", 2)`.
int64_t EncoderBlockOutChannels(const Ltx2VideoEncoderBlock& block, int64_t in_channels) {
  const int64_t multiplier = block.multiplier != 0 ? block.multiplier : 2;
  if (block.name == "res_x_y" || block.name == "compress_all_x_y" ||
      block.name == "compress_all_res" || block.name == "compress_space_res" ||
      block.name == "compress_time_res") {
    return in_channels * multiplier;
  }
  return in_channels;
}

}  // namespace

int64_t Ltx2VideoTemporalScaleFactor(const std::vector<Ltx2VideoEncoderBlock>& blocks) {
  int64_t steps = 0;
  for (const Ltx2VideoEncoderBlock& block : blocks) {
    if (StartsWith(block.name, "compress_time") || StartsWith(block.name, "compress_all")) ++steps;
  }
  return int64_t{1} << steps;
}

int64_t Ltx2VideoSpatialScaleFactor(const std::vector<Ltx2VideoEncoderBlock>& blocks,
                                    int64_t patch_size) {
  int64_t steps = 0;
  for (const Ltx2VideoEncoderBlock& block : blocks) {
    if (StartsWith(block.name, "compress_space") || StartsWith(block.name, "compress_all")) ++steps;
  }
  return patch_size * (int64_t{1} << steps);
}

Ltx2LatentVolume Ltx2ConvVideoEncode(const Ltx2ConvVideoEncoderConfig& config,
                                     const Ltx2VaeWeights& weights,
                                     const std::vector<float>& frames, int64_t channels,
                                     int64_t frame_count, int64_t height, int64_t width,
                                     int64_t* out_cropped_frames) {
  VT_CHECK(channels == config.in_channels,
           "ltx2 video encoder: input channel count does not match in_channels");
  VT_CHECK(static_cast<int64_t>(frames.size()) == channels * frame_count * height * width,
           "ltx2 video encoder: input size does not match [C, F, H, W]");
  VT_CHECK(frame_count >= 1, "ltx2 video encoder: at least one frame is required");
  // `latent_log_var="none"` is REFUSED rather than reproduced. Upstream skips the
  // uniform/constant fix-ups and then still runs `torch.chunk(sample, 2, dim=1)`
  // (video_vae.py:335), so the means carry HALF of `out_channels` while
  // `per_channel_statistics` carries `out_channels` — the broadcast in
  // `normalize` (ops.py:81-84) raises. Reproducing "whatever it does" would mean
  // inventing semantics upstream does not have.
  VT_CHECK(config.latent_log_var != Ltx2LogVarianceType::kNone,
           "ltx2 video encoder: latent_log_var=`none` cannot produce a latent — upstream still "
           "chunks the conv_out into two halves (video_vae.py:335), leaving out_channels/2 mean "
           "channels against out_channels per-channel statistics, and PerChannelStatistics."
           "normalize raises on the broadcast (video_vae/ops.py:81-84)");

  const std::string p = config.prefix;
  const VideoConvSpec spec = SpecOf(config);

  // --- the frame-count crop (video_vae.py:276-286) ---
  // Upstream WARNS and crops rather than failing, so a caller that quietly hands
  // an invalid count gets a SHORTER clip, not an error. The count is reported.
  const int64_t temporal_factor = Ltx2VideoTemporalScaleFactor(config.encoder_blocks);
  const int64_t cropped = (frame_count - 1) % temporal_factor;
  const int64_t kept = frame_count - cropped;
  if (out_cropped_frames != nullptr) *out_cropped_frames = cropped;

  Volume x;
  x.channels = channels;
  x.t = kept;
  x.h = height;
  x.w = width;
  x.data.resize(static_cast<size_t>(x.channels * x.spatial()));
  for (int64_t c = 0; c < channels; ++c) {
    for (int64_t f = 0; f < kept; ++f) {
      const size_t src = static_cast<size_t>((c * frame_count + f) * height * width);
      std::copy(frames.begin() + static_cast<ptrdiff_t>(src),
                frames.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(height * width)),
                x.data.begin() + static_cast<ptrdiff_t>(x.At(c, f, 0, 0)));
    }
  }

  // --- patchify -> conv_in (video_vae.py:291-292) ---
  x = Patchify(x, config.patch_size);
  x = CausalConv3d(x, config.out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   weights.Get(p + "conv_in.conv.weight"), &weights.Get(p + "conv_in.conv.bias"));

  // --- the FORWARD block walk (video_vae.py:221-236, 294-295) ---
  int64_t index = 0;
  for (const Ltx2VideoEncoderBlock& block : config.encoder_blocks) {
    const std::string bp = p + "down_blocks." + std::to_string(index);
    const int64_t out_channels = EncoderBlockOutChannels(block, x.channels);
    if (block.name == "res_x") {
      // UNetMidBlock3D, built with neither timestep conditioning nor noise
      // injection: `_make_encoder_block` passes neither (video_vae.py:52-60), so
      // both take their `False` defaults (resnet.py:219-220).
      for (int64_t i = 0; i < block.num_layers; ++i) {
        x = ResnetBlock3d(spec, weights, bp + ".res_blocks." + std::to_string(i), x, x.channels,
                          /*inject_noise=*/false, /*timestep_conditioning=*/false, nullptr,
                          nullptr);
      }
    } else if (block.name == "res_x_y") {
      x = ResnetBlock3d(spec, weights, bp, x, out_channels, /*inject_noise=*/false,
                        /*timestep_conditioning=*/false, nullptr, nullptr);
    } else if (block.name == "attn") {
      x = AttnBlock3d(weights, bp, x);
    } else if (block.name == "compress_time" || block.name == "compress_space" ||
               block.name == "compress_all" || block.name == "compress_all_x_y") {
      // Plain strided CausalConv3d (video_vae.py:72-112). `compress_all_x_y` is
      // the only one of the four that changes the channel count.
      const int64_t st = block.name == "compress_space" ? 1 : 2;
      const int64_t ss = block.name == "compress_time" ? 1 : 2;
      x = CausalConv3d(x, out_channels, 3, spec.causal, spec.spatial_padding_mode,
                       weights.Get(bp + ".conv.weight"), &weights.Get(bp + ".conv.bias"), st, ss,
                       ss);
    } else if (block.name == "compress_all_res" || block.name == "compress_space_res" ||
               block.name == "compress_time_res") {
      const int64_t st = block.name == "compress_space_res" ? 1 : 2;
      const int64_t ss = block.name == "compress_time_res" ? 1 : 2;
      x = SpaceToDepthDownsample(spec, weights, bp, x, st, ss, ss, out_channels);
    } else {
      VT_CHECK(false, "ltx2 video encoder: unknown encoder block `" + block.name + "`");
    }
    ++index;
  }

  // --- conv_norm_out -> SiLU -> conv_out (video_vae.py:239-262, 297-299) ---
  if (config.norm_layer == Ltx2NormLayer::kPixelNorm) {
    PixelNorm(x.data, x.channels, x.spatial(), config.pixel_norm_eps);
  } else {
    MiniMaxH3GroupNorm3d(x.data, x.channels, x.spatial(), config.norm_num_groups,
                         weights.Get(p + "conv_norm_out.weight"),
                         weights.Get(p + "conv_norm_out.bias"), config.norm_eps);
  }
  Silu(x.data);
  int64_t conv_out_channels = config.out_channels;
  if (config.latent_log_var == Ltx2LogVarianceType::kPerChannel) {
    conv_out_channels *= 2;
  } else if (config.latent_log_var == Ltx2LogVarianceType::kUniform ||
             config.latent_log_var == Ltx2LogVarianceType::kConstant) {
    conv_out_channels += 1;
  }
  x = CausalConv3d(x, conv_out_channels, 3, spec.causal, spec.spatial_padding_mode,
                   weights.Get(p + "conv_out.conv.weight"), &weights.Get(p + "conv_out.conv.bias"));

  // --- the log-variance fix-ups and the mean split (video_vae.py:301-336) ---
  // Only the MEANS survive, so the fix-ups matter for exactly one reason: they
  // decide WHICH channels the split calls means. kUniform must drop the single
  // trailing logvar channel; kConstant must drop it too. Getting either wrong
  // shifts the whole latent by one channel.
  VT_CHECK(conv_out_channels >= 2,
           "ltx2 video encoder: conv_out must emit at least 2 channels (video_vae.py:308-312)");
  const int64_t latent_channels = config.out_channels;
  VT_CHECK(x.channels >= latent_channels,
           "ltx2 video encoder: conv_out emitted fewer channels than the latent width");

  const std::vector<float>& std_of_means = weights.Get(p + "per_channel_statistics.std-of-means");
  const std::vector<float>& mean_of_means = weights.Get(p + "per_channel_statistics.mean-of-means");
  VT_CHECK(static_cast<int64_t>(std_of_means.size()) == latent_channels &&
               static_cast<int64_t>(mean_of_means.size()) == latent_channels,
           "ltx2 video encoder: per-channel statistics must have one value per latent channel");

  Ltx2LatentVolume out;
  out.batch = 1;
  out.channels = latent_channels;
  out.frames = x.t;
  out.height = x.h;
  out.width = x.w;
  out.data.resize(static_cast<size_t>(out.elems()));
  const int64_t elems = x.spatial();
  // f32: the encoder's normalize is the decoder de-normalize run backwards, and
  // upstream computes it in the activation dtype on both sides.
  for (int64_t c = 0; c < latent_channels; ++c) {
    const float mean = mean_of_means[static_cast<size_t>(c)];
    const float denom = std_of_means[static_cast<size_t>(c)];
    for (int64_t i = 0; i < elems; ++i) {
      out.data[static_cast<size_t>(c * elems + i)] =
          (x.data[static_cast<size_t>(c * elems + i)] - mean) / denom;
    }
  }
  return out;
}

}  // namespace vllm
