// The shared 1-D BigVGAN vocoder core.
//
// MiniMax-H3 ported these first; LTX-2.5's audio VAE descends from the same
// BigVGAN lineage and reused them rather than copying, because a second copy of
// the alias-free trim geometry is the duplicate that goes wrong quietly -- a fix
// to the pad/trim arithmetic lands in one file, the other keeps its own green
// gate, and the two audio VAEs drift apart with nothing to say so.
//
// IndexTTS-2.5 (#634) is the third consumer, which is what moved this out of
// `minimax_h3.h`: a model's header is not a home for something three lanes
// share, and the `MiniMaxH3` prefix said who ported it rather than who may call
// it. Signals are CHANNEL-MAJOR [C, T] throughout.
//
// `tests/scripts/test_vocoder1d_single_home.py` asserts there is exactly one
// home and one definition of each symbol; no numeric test can see a fork,
// because a fresh copy agrees on the day it is made and only drifts later.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace vocoder1d {


// One 1-D convolution over [C_in, T] with stride/dilation/groups. Weight is
// [C_out, C_in/groups, K]; the input must ALREADY be padded. Accumulates in
// float — the width torch accumulates a float convolution in (#1474) — and
// reports the produced length through `out_len`.
std::vector<float> Conv1d(const std::vector<float>& in, int64_t in_channels,
                                   int64_t in_len, const std::vector<float>& weight,
                                   const std::vector<float>* bias, int64_t out_channels,
                                   int64_t kernel, int64_t stride, int64_t dilation,
                                   int64_t groups, int64_t* out_len);

// torch.nn.functional.conv_transpose1d over [C_in, T]. Weight is
// [C_in, C_out/groups, K]; output length is (T-1)*stride - 2*padding + K.
std::vector<float> ConvTranspose1d(const std::vector<float>& in, int64_t in_channels,
                                            int64_t in_len, const std::vector<float>& weight,
                                            const std::vector<float>* bias, int64_t out_channels,
                                            int64_t kernel, int64_t stride, int64_t padding,
                                            int64_t groups, int64_t* out_len);

// F.pad along the time axis. `replicate` selects mode="replicate"; false is the
// zero pad an ordinary nn.Conv1d `padding=` argument performs.
std::vector<float> Pad1d(const std::vector<float>& in, int64_t channels, int64_t in_len,
                                  int64_t left, int64_t right, bool replicate, int64_t* out_len);

// The stabilizing epsilon in Snake/SnakeBeta's reciprocal, named so it can be
// pinned: upstream writes `1.0 / (beta + 1e-9)` on both sides of this port's
// lineage — LTX-2.5 at audio_vae/vocoder.py:198 (Snake) and :221 (SnakeBeta), and
// MiniMax-H3 in its BigVGAN activation. Mutation proves no reduced-dimension
// golden can tell 1e-9 from 0.0, because beta is O(1) there and never small
// enough for the term to matter; the value still decides whether a real
// checkpoint whose learned beta approaches zero divides or explodes. It is
// therefore held by a source-anchored constant assertion, not by a tensor
// comparison.
inline constexpr double kSnakeEps = 1e-9;

// Snake / SnakeBeta: x + (b + kSnakeEps)^-1 * sin^2(a * x), in place.
// A null `beta` selects plain Snake, which reuses ALPHA as the reciprocal scale
// (LTX-2.5 vocoder.py:198); a non-null one selects SnakeBeta (vocoder.py:221),
// which is what every MiniMax-H3 checkpoint carries. `logscale` exponentiates
// both, which is how the parameters are stored.
void SnakeActivation(std::vector<float>& x, int64_t channels, int64_t length,
                              const std::vector<float>& alpha, const std::vector<float>* beta,
                              bool logscale);

// The anti-aliased activation, `Activation1d`: upsample by `ratio` -> Snake(Beta)
// -> downsample by `ratio`, both through the kaiser-sinc window with REPLICATE
// padding. MiniMax-H3 reaches it through dac_alias_free_act.py +
// dac_alias_free_resample.py; LTX-2.5 through vocoder.py:104-184. The trim
// geometry is the fragile part and the reason this is shared rather than copied.
//
// Build() computes the window once; Apply() is const and may be reused.
struct AliasFreeActivation1d {
  int64_t ratio = 2;
  int64_t kernel_size = 12;
  std::vector<float> filter;

  void Build();

  std::vector<float> Apply(const std::vector<float>& in, int64_t channels, int64_t in_len,
                           const std::vector<float>& alpha, const std::vector<float>* beta,
                           bool logscale, int64_t* out_len) const;
};

// kaiser_sinc_filter1d (dac_alias_free_filter.py:26-60) — built at load time, never
// read from the checkpoint.
std::vector<float> KaiserSincFilter1d(double cutoff, double half_width,
                                               int64_t kernel_size);

// torch `weight_norm`: w = g * v / ||v||, the norm taken over every dimension
// EXCEPT dim 0. A pure function of the stored parameters, so every consumer
// folds it once at load rather than reproducing the parameterization per
// forward — which is what torch itself caches.
//
// `dim0` is the size of dimension 0 of `v`, and naming it that rather than
// `out_channels` is deliberate: for an `nn.Conv1d` weight [C_out, C_in, K] dim 0
// IS the output channel, but for the `nn.ConvTranspose1d` weight
// [C_in, C_out, K] it is the INPUT channel, and torch normalizes over dim 0
// either way. MiniMax-Music3's vocoder carries both spellings in one file
// (minimax_music3_vocoder.py:55 transpose, :42/:44/:89/:98 conv), so a helper
// that assumed "out channels" would fold four of its thirty convolutions over
// the wrong axis while still producing finite, correctly shaped weights.
//
// Second consumer, hence its home here rather than in a model's header:
// MiniMax-H3's audio VAE (`parametrizations.weight.original0/1`, the modern
// spelling) and MiniMax-Music3's vocoder (`weight_g`/`weight_v`, the legacy
// one). Same arithmetic, different era.
std::vector<float> MaterializeWeightNorm(const std::vector<float>& g,
                                         const std::vector<float>& v, int64_t dim0);

}  // namespace vocoder1d
}  // namespace vllm
