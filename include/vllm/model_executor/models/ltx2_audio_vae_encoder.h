// LTX-2.5 AUDIO VAE — the ENCODER half and its mel front-end, which phase L4
// recorded as owed.
//
// Reference AUDIO conditioning ("dub it") needs a waveform turned into latents,
// and that is two stages, not one: `AudioProcessor.waveform_to_mel` builds a
// log-mel spectrogram, then `AudioEncoder` compresses it
// (`encode_audio`, audio_vae.py:249-274).
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f, packages/ltx-core/src/ltx_core/
//   OURS                          <-  UPSTREAM
//   Ltx2AudioEncoderForward       <-  model/audio_vae/audio_vae.py:190-246
//   (the downsampling path)       <-  model/audio_vae/downsample.py:60-110
//   (Downsample)                  <-  model/audio_vae/downsample.py:11-57
//   (mid block)                   <-  model/audio_vae/audio_vae.py:22-57
//   (ResnetBlock / AttnBlock)     <-  model/audio_vae/resnet.py:115-176,
//                                     model/audio_vae/attention.py:16-55
//   (the audio patchifier)        <-  components/patchifiers.py:287-305
//   (per-channel statistics)      <-  model/audio_vae/ops.py:58-75
//   Ltx2WaveformToLogMel          <-  model/audio_vae/ops.py:8-55
//   Ltx2SlaneyMelFilterbank       <-  torchaudio.functional.melscale_fbanks
//                                     (reached from ops.py:20-34)
//   Ltx2AudioEncoderConfig        <-  model/audio_vae/model_configurator.py:144-182
//
// Every primitive is SHARED with the decoder: this file's implementation is
// compiled into ltx2_audio_vae.cpp, so `Conv2d`, `ApplyCausalPadding`,
// `PixelNorm`, `ResnetBlock` and `AttnBlock` are the exact functions the decoder
// goldens already hold. Only `Downsample` is new, because the decoder has no
// strided convolution.
//
// ─── THE FOUR THINGS THAT FAIL SILENTLY ──────────────────────────────────────
//  * `Downsample`'s PRE-PAD IS ASYMMETRIC AND ITS CONV IS STRIDED
//    (downsample.py:34-52). On the shipped `height` axis the pad is
//    `(left, right, top, bottom) = (0, 1, 2, 0)` — TWO rows of history on the
//    time axis, ONE column on the mel axis, and nothing on the other side of
//    either. Padding it symmetrically shifts the whole sampling lattice by half
//    a bin and produces a plausible latent, not an error.
//  * THE `attn` LIST IS BUILT INSIDE THE RES-BLOCK LOOP but `curr_res` is halved
//    OUTSIDE it (downsample.py:87-107), so a level either has `num_res_blocks`
//    attention blocks or none, and the resolution a level is tested at is the one
//    it ENTERS with, not the one it leaves with.
//  * `in_ch_mult` IS `(1, *ch_mult)`, so level i reads `ch * ch_mult[i-1]` and
//    writes `ch * ch_mult[i]` (downsample.py:78-85). Reading `ch_mult[i]` on both
//    sides builds an encoder whose first level is already too wide.
//  * THE PATCHIFIER FLATTENS `b c t f -> b t (c f)` (patchifiers.py:301-304), so
//    the per-channel statistics are indexed by `c * mel_bins + f`, NOT by `c`.
//    This is the same packing the decoder's `_denormalize_latents` undoes.
//
// ─── THE MEL FRONT-END IS A ONE-SIDED REFERENCE, AND THAT IS RECORDED ────────
// diffusers has NO counterpart: there is no `torchaudio` import anywhere in
// `src/diffusers`, `AutoencoderKLLTX2Audio.encode` is never called from
// `pipelines/ltx2/*`, and audio conditioning enters diffusers only as a
// pre-computed latent. So `slaney` mel scale, `slaney` normalization, the
// centered reflect-padded STFT and `power=1.0` are attested by `ltx_core` alone.
// Everything else in this file is corroborated by both implementations.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"

namespace vllm {

// ---------------------------------------------------------------------------
// AudioEncoder (audio_vae.py:60-246)
// ---------------------------------------------------------------------------

struct Ltx2AudioEncoderConfig {
  // Names and defaults mirror AudioEncoderConfigurator.from_metadata
  // (audio_vae/model_configurator.py:162-182).
  int64_t ch = 128;
  int64_t in_channels = 2;
  std::vector<int64_t> ch_mult = {1, 2, 4};
  int64_t num_res_blocks = 2;
  // The SHIPPED LTX-2.5 checkpoint sets this EMPTY, so no level carries
  // attention; the configurator's `{8, 16, 32}` is only the fallback when the
  // checkpoint omits the key (model_configurator.py:166).
  std::vector<int64_t> attn_resolutions = {8, 16, 32};
  int64_t resolution = 256;
  int64_t z_channels = 8;
  // `double_z` doubles conv_out's width; only the FIRST half is ever used
  // (audio_vae.py:237), so a checkpoint trained with double_z and read without it
  // still produces a latent — of the wrong channels.
  bool double_z = true;
  bool resamp_with_conv = true;
  // The shipped checkpoint sets this FALSE (its `mid_block_add_attention`), unlike
  // the decoder arm L4 gated, which ships it true.
  bool mid_block_add_attention = true;
  Ltx2NormType norm_type = Ltx2NormType::kPixel;
  Ltx2CausalityAxis causality_axis = Ltx2CausalityAxis::kHeight;
  // Only read on the GroupNorm arm; PixelNorm is parameter-free. Neither is
  // reachable from a checkpoint: `build_normalization_layer` passes `eps=1e-6` as
  // a LITERAL and forwards its own `num_groups` keyword, whose default is 32
  // (normalization.py:44, 56), and no audio_vae call site passes `num_groups` at
  // all — so they are fields only so the gate can pin them.
  //
  // `norm_type = kGroup` is `AudioEncoder.__init__`'s declared default
  // (audio_vae.py:82), but it is NOT what pure defaults give you: the paired
  // default is `causality_axis = WIDTH` (audio_vae.py:83), and `ResnetBlock`
  // refuses GroupNorm on any causal axis with
  // `ValueError: Causal ResnetBlock with GroupNorm is not supported`
  // (resnet.py:130-131) — verified by construction against the pinned upstream.
  // So a group-norm checkpoint is one that declares `causality_axis: none`
  // alongside it, which is legal and is what the group-norm golden in
  // test_ltx2_vae.cpp runs. That arm is what stopped this eps from being a
  // constant no arm ever read.
  int64_t num_groups = 32;
  double norm_eps = 1e-6;
  // Reached through `build_normalization_layer`, which passes eps=1e-6
  // (normalization.py:58) — the SAME value the audio decoder gets and NOT the
  // video VAE's bare 1e-8.
  double pixel_norm_eps = 1e-6;
  std::string prefix;

  int64_t num_resolutions() const { return static_cast<int64_t>(ch_mult.size()); }
};

// AudioEncoder.forward at batch 1 (audio_vae.py:190-203). `spectrogram` is
// [in_channels, frames, mel_bins] channel-major — upstream's
// (batch, channels, TIME, frequency), which is why dim 2 is the causal axis on
// the shipped `height` setting.
//
// Returns the NORMALIZED latent means, [z_channels, frames', mel_bins'],
// i.e. exactly what `Ltx2AudioDecoderForward` de-normalizes on the way back.
Ltx2AudioSpectrogram Ltx2AudioEncoderForward(const Ltx2AudioEncoderConfig& config,
                                             const Ltx2VaeWeights& weights,
                                             const std::vector<float>& spectrogram,
                                             int64_t channels, int64_t frames, int64_t mel_bins);

// ---------------------------------------------------------------------------
// AudioProcessor (audio_vae/ops.py:8-55) — the mel front-end
// ---------------------------------------------------------------------------

// `torch.log(torch.clamp(mel, min=1e-5))` (ops.py:52). A member of the
// invisible-constant class and the one that BINDS IN PRODUCTION: real silence
// drives the linear mel to 0 and this constant alone decides the value the
// encoder then sees (log(1e-5) = -11.5129...). A reduced-dimension golden built
// from a well-scaled random waveform never reaches it, so the gate holds it with
// a source-anchored assertion AND a golden arm whose input is actual silence.
inline constexpr double kLtx2AudioMelLogClamp = 1e-5;

struct Ltx2AudioProcessorConfig {
  // Every field is passed straight to `torchaudio.transforms.MelSpectrogram`
  // (ops.py:20-34). `win_length` is not a separate knob upstream — it is
  // `n_fft` — and `f_min` / `f_max` are hardcoded to 0 and `sample_rate / 2`
  // rather than read from the checkpoint's `preprocessing.mel` block, which is a
  // real (if currently harmless) upstream inconsistency: for the shipped
  // 16 kHz / fmax 8000 config the two agree.
  int64_t target_sample_rate = 16000;
  int64_t mel_bins = 64;
  int64_t mel_hop_length = 160;
  int64_t n_fft = 1024;
};

// `melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm="slaney",
// mel_scale="slaney")`, returned as [n_mels, n_freqs] — TRANSPOSED relative to
// torchaudio's [n_freqs, n_mels], because every consumer here contracts over the
// frequency axis. Exposed on its own because a wrong filterbank makes every mel
// bin wrong at once and the resulting mismatch is impossible to localize.
//
// `sample_rate / 2` is computed with INTEGER division, mirroring torchaudio's
// `torch.linspace(0, sample_rate // 2, n_freqs)`.
std::vector<float> Ltx2SlaneyMelFilterbank(int64_t n_freqs, double f_min, double f_max,
                                           int64_t n_mels, int64_t sample_rate);

// `AudioProcessor.waveform_to_mel` (ops.py:44-55) at any channel count.
// `waveform` is [channels, samples], channel-major, at `sampling_rate`.
//
// A SAMPLE RATE THAT DOES NOT MATCH `target_sample_rate` IS REFUSED BY NAME.
// Upstream resamples with `torchaudio.functional.resample` (ops.py:40), a
// polyphase kaiser resampler for an arbitrary rational ratio; this project ports
// only the integer-ratio hann-sinc variant the BWE stage needs
// (Ltx2HannSincResampleFilter1d). Refusing is deliberate: silently treating
// 44.1 kHz samples as 16 kHz produces audio conditioning that is pitched and
// time-scaled wrong while every shape still checks out.
//
// Returns [channels, frames, mel_bins] channel-major — the layout
// `Ltx2AudioEncoderForward` takes, which is upstream's final
// `permute(0, 1, 3, 2)` (ops.py:55).
std::vector<float> Ltx2WaveformToLogMel(const Ltx2AudioProcessorConfig& config,
                                        const std::vector<float>& waveform, int64_t channels,
                                        int64_t samples, int64_t sampling_rate, int64_t* out_frames);

}  // namespace vllm
