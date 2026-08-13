// LTX-2.5 AUDIO VAE — the spectrogram decoder and its BigVGAN vocoder.
//
// LTX-2.5 (Lightricks/LTX-2) decodes one request into frames PLUS a stereo
// waveform. The waveform half is two stages: an `AudioDecoder` that turns audio
// latents back into a log-mel spectrogram, and a `Vocoder` that turns that
// spectrogram into samples. Both are pure-Python `ltx_core` modules upstream, so
// unlike MiniMax-H3's checkpoint remote code they can be executed directly as the
// oracle — which is exactly what scripts/gen-ltx2-vae-goldens.py does, at reduced
// dimensions on CPU, with both sides rebuilding weights from one deterministic
// stream so NO WEIGHT BYTE is checked in.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2, packages/ltx-core/src/ltx_core/
//   OURS                             <-  UPSTREAM
//   Ltx2AudioDecoderForward          <-  model/audio_vae/audio_vae.py:385-400
//   (its ResnetBlock)                <-  model/audio_vae/resnet.py:115-176
//   (its AttnBlock)                  <-  model/audio_vae/attention.py:16-55
//   (its Upsample)                   <-  model/audio_vae/upsample.py:12-55
//   (its CausalConv2d)               <-  model/audio_vae/causal_conv_2d.py:7-64
//   (PixelNorm / GroupNorm)          <-  model/common/normalization.py:14-59
//   (per-channel statistics)         <-  model/audio_vae/ops.py:58-75
//   (the audio patchifier)           <-  components/patchifiers.py:287-330
//   Ltx2VocoderForward               <-  model/audio_vae/vocoder.py:398-438
//   (AMPBlock1 / SnakeBeta)          <-  model/audio_vae/vocoder.py:208-290
//   (Activation1d up/down)           <-  model/audio_vae/vocoder.py:104-184
//   (ResBlock1, legacy arm)          <-  model/audio_vae/resnet.py:12-80
//   Ltx2VocoderWithBweForward        <-  model/audio_vae/vocoder.py:573-630
//   (MelSTFT / _STFTFn)              <-  model/audio_vae/vocoder.py:441-516
//   Ltx2KaiserSincFilter1d           <-  model/audio_vae/vocoder.py:52-70
//   Ltx2HannSincResampleFilter1d     <-  model/audio_vae/vocoder.py:116-128
//
// ─── THE THREE THINGS THAT FAIL SILENTLY ─────────────────────────────────────
//  * CAUSALITY LIVES ON THE HEIGHT AXIS. The shipped `causality_axis` is
//    `height` (model/audio_vae/model_configurator.py:134) and the tensor layout is
//    (batch, channels, TIME, mel_bins) — so dim 2 is time and ALL of its
//    convolution padding goes on the LEFT (top). Padding it symmetrically still
//    produces a plausible spectrogram that simply peeks into the future.
//    MEASURED CAVEAT, so nobody over-claims it: `causality_axis` governs the
//    CONVOLUTIONS ONLY. The decoder's `AttnBlock`s attend over the entire
//    (time, mel) map (attention.py:31-55), so with the shipped mid-block
//    attention on, the decoder as a whole is NOT causal in time — upstream moves
//    every output frame under a last-frame perturbation, and the test asserts
//    exactly that alongside the convolution-only reach.
//  * THE UPSAMPLER DROPS THE FIRST TEMPORAL ELEMENT, not the last
//    (upsample.py:29-42): only the first two interpolated elements depend on a
//    single input element, so undoing the encoder's pad means dropping the head.
//  * THE VOCODER'S ANTI-ALIASING FILTERS ARE COMPUTED, NEVER LOADED. Both the
//    kaiser-sinc window the activations use and the hann-sinc window the BWE
//    resampler uses are built at construction; a checkpoint carries neither. They
//    are gated on their own before the decoders are blamed for a mismatch.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vllm {

// ---------------------------------------------------------------------------
// Parameters keyed by their upstream `state_dict` name, so the module's own
// naming IS the contract. The same bag serves the audio VAE, the vocoder and the
// Conv video VAE (ltx2_video_vae.h), which is why it lives in one place.
// ---------------------------------------------------------------------------
struct Ltx2VaeWeights {
  std::map<std::string, std::vector<float>> tensors;

  const std::vector<float>& Get(const std::string& name) const;
  bool Has(const std::string& name) const { return tensors.count(name) != 0; }
};

// causality_axis.py:4-10. The axis whose padding is one-sided; `kHeight` is the
// shipped default and, in the (b, c, time, mel) layout, it is TIME.
enum class Ltx2CausalityAxis { kNone, kWidth, kHeight, kWidthCompatibility };

// common/normalization.py:7-11. `kPixel` is the shipped default and is
// PARAMETER-FREE, so a pixel-norm checkpoint carries no norm tensors at all.
enum class Ltx2NormType { kGroup, kPixel };

// ---------------------------------------------------------------------------
// AudioDecoder (audio_vae.py:277-494)
// ---------------------------------------------------------------------------

struct Ltx2AudioDecoderConfig {
  // Names and defaults mirror AudioDecoderConfigurator.from_metadata
  // (audio_vae/model_configurator.py:108-141).
  int64_t ch = 128;
  int64_t out_ch = 2;
  std::vector<int64_t> ch_mult = {1, 2, 4};
  int64_t num_res_blocks = 2;
  std::vector<int64_t> attn_resolutions = {8, 16, 32};
  int64_t resolution = 256;
  int64_t z_channels = 8;
  Ltx2NormType norm_type = Ltx2NormType::kPixel;
  Ltx2CausalityAxis causality_axis = Ltx2CausalityAxis::kHeight;
  bool mid_block_add_attention = true;
  // The decoder's TARGET mel-bin count. 0 keeps whatever the latent carried,
  // mirroring `mel_bins=None` (audio_vae.py:422).
  int64_t mel_bins = 0;
  // Only read on the GroupNorm arm; PixelNorm has no parameters at all. Neither
  // is reachable from a checkpoint: `build_normalization_layer` passes `eps=1e-6`
  // as a LITERAL and forwards its own `num_groups` keyword, whose default is 32
  // (normalization.py:44, 56), and no audio_vae call site passes `num_groups`.
  // They are fields here so the gate can pin them.
  //
  // `norm_type = kGroup` is not a dead arm, but it is not what pure defaults give
  // you either. `AudioDecoder.__init__` declares `norm_type = GROUP`
  // (audio_vae.py:294) and, on the very next line, `causality_axis = WIDTH`
  // (audio_vae.py:295) — and `ResnetBlock` refuses that combination with
  // `ValueError: Causal ResnetBlock with GroupNorm is not supported`
  // (resnet.py:130-131), verified by construction against the pinned upstream. A
  // group-norm checkpoint is therefore one that declares `causality_axis: none`
  // alongside it, which is legal and is what the group-norm golden in
  // test_ltx2_vae.cpp runs. Before that arm existed this eps was never READ on
  // any path, and a 100x change moved nothing.
  int64_t num_groups = 32;
  double norm_eps = 1e-6;
  // The audio VAE reaches PixelNorm through `build_normalization_layer`, which
  // passes eps=1e-6 (normalization.py:58). The VIDEO VAE constructs `PixelNorm()`
  // bare and so gets its 1e-8 DEFAULT (normalization.py:22). The two are
  // deliberately different fields with deliberately different defaults; see the
  // honesty note on Ltx2ConvVideoDecoderConfig::pixel_norm_eps.
  double pixel_norm_eps = 1e-6;
  // `state_dict` prefix, so one bag can hold several modules.
  std::string prefix;

  int64_t num_resolutions() const { return static_cast<int64_t>(ch_mult.size()); }
};

// A (channels, frames, mel_bins) spectrogram, channel-major and contiguous.
struct Ltx2AudioSpectrogram {
  int64_t channels = 0;
  int64_t frames = 0;
  int64_t mel_bins = 0;
  std::vector<float> data;
};

// AudioDecoder.forward at batch 1: denormalize -> conv_in -> mid -> up path ->
// norm/SiLU/conv_out -> crop-or-pad to the target shape. `latent` is
// [latent_channels, latent_frames, latent_mel_bins], channel-major.
//
// Batch is fixed at 1 deliberately: the pipeline decodes one request's latents,
// and upstream's per-channel statistics broadcast over the patchified last axis
// rather than over the batch, so nothing here is batch-coupled.
Ltx2AudioSpectrogram Ltx2AudioDecoderForward(const Ltx2AudioDecoderConfig& config,
                                             const Ltx2VaeWeights& weights,
                                             const std::vector<float>& latent,
                                             int64_t latent_channels, int64_t latent_frames,
                                             int64_t latent_mel_bins);

// ---------------------------------------------------------------------------
// Vocoder (vocoder.py:293-438)
// ---------------------------------------------------------------------------

struct Ltx2VocoderConfig {
  // Shapes mirror Vocoder.__init__ (vocoder.py:317-341), but the two ARM
  // selectors deliberately do NOT: `Vocoder.__init__` defaults to `resblock="1"`
  // and `activation="snake"`, i.e. the pre-2.3 legacy arm, while `amp` and
  // `snakebeta` below default to the AMP1/snakebeta arm.
  //
  // That is not a mismatch, it is the LTX-2.5 contract. A 2.5 checkpoint reaches
  // the vocoder through VocoderConfigurator.from_metadata, whose BWE branch
  // REQUIRES both — `check_config_value(vocoder_cfg, "resblock", "AMP1")` and
  // `check_config_value(vocoder_cfg, "activation", "snakebeta")`, and the same
  // pair again for the BWE generator (audio_vae/model_configurator.py:59-64).
  // A default that mirrored `Vocoder.__init__` would therefore be a default no
  // shipping checkpoint can use. The legacy arm stays reachable by setting
  // `amp = false`, which is what the `resblock == "1"` branch selects
  // (model_configurator.py:53), and it is gated on its own.
  std::vector<int64_t> resblock_kernel_sizes = {3, 7, 11};
  std::vector<int64_t> upsample_rates = {6, 5, 2, 2, 2};
  std::vector<int64_t> upsample_kernel_sizes = {16, 15, 8, 4, 4};
  std::vector<std::vector<int64_t>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
  int64_t upsample_initial_channel = 1024;
  // resblock == "AMP1": BigVGAN v2 AMPBlock1 with anti-aliased activations. False
  // selects the legacy ResBlock1 arm (`resblock == "1"`, plain leaky ReLU), which
  // is what pre-2.3 checkpoints carry (audio_vae/model_configurator.py:53).
  bool amp = true;
  // activation == "snakebeta" vs "snake" (vocoder.py:242). Only read when `amp`.
  bool snakebeta = true;
  bool use_tanh_at_final = true;
  bool apply_final_activation = true;
  bool use_bias_at_final = true;
  int64_t output_sampling_rate = 24000;
  // Snake/SnakeBeta store alpha/beta in LOG scale upstream (alpha_logscale=True,
  // vocoder.py:193).
  bool snake_logscale = true;
  std::string prefix;
};

// Vocoder.forward over a STEREO spectrogram [2, frames, mel_bins], channel-major.
// Upstream hardcodes conv_pre's input width to 128 = 2 channels x 64 mel bins
// (vocoder.py:350-358), so the interleave `b s c t -> b (s c) t` is part of the
// contract: flattening channel-last instead time-smears the audio while still
// producing something that plays. Returns [2, samples].
std::vector<float> Ltx2VocoderForward(const Ltx2VocoderConfig& config,
                                      const Ltx2VaeWeights& weights, const std::vector<float>& mel,
                                      int64_t channels, int64_t frames, int64_t mel_bins,
                                      int64_t* out_samples);

// ---------------------------------------------------------------------------
// VocoderWithBWE (vocoder.py:519-630) — the ltx-2.3+ arm the 2.5 checkpoint uses.
// ---------------------------------------------------------------------------

// The floor under the BWE mel BEFORE its log: `torch.clamp(mel, min=1e-5)`
// (vocoder.py:515). Named so it can be pinned, because it sets the floor of the
// log-mel fed to the bwe_generator and REAL SILENCE reaches it in production.
//
// It is NOT a member of the invisible-constant class described in
// ltx2_video_vae.h, and the line here that said it was is corrected rather than
// carried: the ORDINARY BWE arm's mel_basis is non-negative and well-scaled and
// its raw minimum is ~4.4e-3, so that arm alone cannot move under a mutation.
// "ltx2 vae: the BWE mel log clamp is gated where it actually binds" attenuates
// mel_basis until every bin saturates the floor, and against it 1e-5 -> 1e-8 REDS
// at max|diff| = 0.144965 versus the 5e-6 band. The pin below stays anyway, for
// what no golden can see: a regeneration that moves the constant and the expected
// tensors together.
inline constexpr double kLtx2BweMelLogClamp = 1e-5;

struct Ltx2VocoderBweConfig {
  Ltx2VocoderConfig vocoder;        // prefix defaults to `<prefix>vocoder.`
  Ltx2VocoderConfig bwe_generator;  // prefix defaults to `<prefix>bwe_generator.`
  int64_t filter_length = 1024;
  int64_t hop_length = 256;
  int64_t win_length = 1024;
  int64_t n_mel_channels = 64;
  int64_t input_sampling_rate = 24000;
  int64_t output_sampling_rate = 48000;
  std::string prefix;
};

// The full BWE chain: vocoder -> pad to a hop multiple -> causal STFT log-mel ->
// bwe_generator residual -> + hann-sinc resampled skip -> clamp -> trim. Returns
// [2, samples] at `output_sampling_rate`.
std::vector<float> Ltx2VocoderWithBweForward(const Ltx2VocoderBweConfig& config,
                                             const Ltx2VaeWeights& weights,
                                             const std::vector<float>& mel, int64_t channels,
                                             int64_t frames, int64_t mel_bins,
                                             int64_t* out_samples);

// ---------------------------------------------------------------------------
// The two COMPUTED filters. Neither is in any checkpoint; both are gated
// directly, because a wrong window makes every activation wrong at once and the
// resulting mismatch is impossible to localize.
// ---------------------------------------------------------------------------

// kaiser_sinc_filter1d (vocoder.py:52-70). This is BigVGAN's filter, identical to
// the one the MiniMax-H3 audio VAE already ports, so it DELEGATES to that shared
// implementation rather than standing up a second copy; the golden proves the
// shared code also matches LTX's upstream.
std::vector<float> Ltx2KaiserSincFilter1d(double cutoff, double half_width, int64_t kernel_size);

// The BWE resampler's HANN-windowed sinc (vocoder.py:116-128) — a different
// window from the kaiser one above, equivalent to torchaudio's `resample`. Fills
// the geometry the forward pass needs; any out-pointer may be null.
std::vector<float> Ltx2HannSincResampleFilter1d(int64_t ratio, int64_t* kernel_size, int64_t* pad,
                                                int64_t* pad_left, int64_t* pad_right);

}  // namespace vllm
