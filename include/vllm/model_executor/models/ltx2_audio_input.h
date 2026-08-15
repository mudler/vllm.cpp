// LTX-2.5 AUDIO-TO-VIDEO — the INPUT path: a file on disk becomes the audio
// latent the DiT's audio stream is conditioned on. Row LTX25-A2V-AUDIO-INPUT
// (#922), spec .agents/specs/ltx25-a2v-audio-input.md.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides) ────────────────────────
// Upstream root: Lightricks/LTX-2 @ fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
//   OURS                          <-  UPSTREAM
//   Ltx2DecodeAudioFile           <-  ltx-pipelines utils/media_io/decode.py:240-300
//                                     (decode_audio_from_file)
//   Ltx2EncodeAudioToLatent       <-  ltx-core model/audio_vae/audio_vae.py:249-274
//                                     (encode_audio), then the truncation at
//                                     ltx-pipelines a2vid_two_stage.py:201-202
//   Ltx2AudioVaeEncoderKeyRules   <-  ltx-core audio_vae/model_configurator.py:194-200
//   Ltx2ParseAudioEncoderConfig   <-  ltx-core audio_vae/model_configurator.py:144-182
//
// The pipeline that drives these is `A2VidPipelineTwoStage`
// (a2vid_two_stage.py:53, called at :143): decode the file, encode it, truncate
// it to the video's duration, then hold it FROZEN through both denoise stages
// while the video is generated around it.
//
// ─── THE THREE THINGS THAT FAIL SILENTLY ─────────────────────────────────────
//  * TRUNCATION IS ONE-SIDED. `a2vid_two_stage.py:202` is
//    `encoded_audio_latent[:, :, : audio_shape.frames]` — it truncates and NEVER
//    pads. Retake's helper `_conform_latent_length`
//    (ltx-pipelines/utils/helpers.py:149-162) truncates OR zero-pads, and A2Vid
//    does not call it. Audio shorter than the video therefore yields a SHORT
//    audio latent, and padding it out to the video duration is a divergence that
//    renders a finite, correctly shaped clip with silence welded onto the end.
//  * THE WINDOW IS APPLIED IN SAMPLES, AFTER THE DECODE (decode.py:286-296),
//    because a codec's frame boundaries need not align with the requested time
//    range. For the uncompressed RIFF container this path accepts there are no
//    codec frames, so upstream's `first_frame_time` is exactly 0.0 and the
//    leading trim reduces to `round(start_time * sample_rate)`. That
//    simplification is sound ONLY for an uncompressed container, which is why
//    compressed ones are refused rather than approximated.
//  * A CHANNEL COUNT THAT DISAGREES WITH THE ENCODER IS REFUSED, not upmixed.
//    Upstream feeds whatever channel count the file carries straight into a
//    conv whose `in_channels` is 2 (model_configurator.py:172), so a mono file
//    raises there. `MiniMaxH3ReadWav` would instead REPEAT mono up to the
//    requested width (minimax_h3.h:1839-1845, mirroring H3's vae.py:305-313),
//    which is that family's contract and not this one's: silently duplicating a
//    mono take into both channels conditions the render on audio the user did
//    not supply. `Ltx2ProbeWavFormat` exists to turn that into a refusal.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_audio_vae.h"
#include "vllm/model_executor/models/ltx2_audio_vae_encoder.h"
#include "vllm/model_executor/models/ltx2_loader.h"  // Ltx2VaeKeyRule, nlohmann::json

namespace vllm {

// ---------------------------------------------------------------------------
// The load path (its own TU: ltx2_audio_vae_encoder_load.cpp)
// ---------------------------------------------------------------------------

// The encoder and its mel front-end are configured from ONE metadata object but
// from four different sub-objects of it, so they are resolved together and
// returned together: `sampling_rate` lives on `audio_vae.model.params` while the
// FFT size is `audio_vae.preprocessing.stft.filter_length`
// (model_configurator.py:156-159).
struct Ltx2AudioEncoderLoad {
  Ltx2AudioEncoderConfig encoder;
  Ltx2AudioProcessorConfig processor;
};

// AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER (model_configurator.py:194-200). Carries
// the `per_channel_statistics.` rule as well as the `encoder.` one, because the
// encoder NORMALIZES with the same statistics the decoder un-normalizes.
std::vector<Ltx2VaeKeyRule> Ltx2AudioVaeEncoderKeyRules();

// Does this checkpoint carry encoder weights at all? Asked of the
// `audio_vae.encoder.` prefix ALONE — the key rules above also match
// `audio_vae.per_channel_statistics.`, which every decoder-only checkpoint
// carries, so "did the filter match anything" is not this question.
bool Ltx2CheckpointHasAudioEncoder(const std::vector<std::string>& names);

// `AudioEncoderConfigurator.from_metadata` (model_configurator.py:144-182), off
// the checkpoint's own `__metadata__` config object.
Ltx2AudioEncoderLoad Ltx2ParseAudioEncoderConfig(const nlohmann::json& config);

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

// A decoded waveform, CHANNEL-MAJOR [channels, samples_per_channel], float in
// [-1, 1]. This is `Audio` (ltx-core types.py:203) minus the batch axis, and it
// is the layout `Ltx2WaveformToLogMel` takes.
struct Ltx2DecodedAudio {
  std::vector<float> samples;
  int64_t channels = 0;
  int64_t samples_per_channel = 0;
  int64_t sample_rate = 0;
};

// The `fmt ` chunk only: what a RIFF file DECLARES, so a mismatch can be refused
// by name before any sample is read. Not a decoder — `MiniMaxH3ReadWav` is.
struct Ltx2WavFormat {
  int64_t channels = 0;
  int64_t sample_rate = 0;
  int64_t bits_per_sample = 0;
};

Ltx2WavFormat Ltx2ProbeWavFormat(const std::string& bytes);

// `decode_audio_from_file` (decode.py:240-300) over an uncompressed RIFF/WAVE
// buffer. `start_time` and `max_duration` are seconds; a `max_duration <= 0`
// means "to the end of the stream", which is upstream's `None` (`:248`).
//
// HARNESS ADAPTATION, and the only one: upstream takes a PATH and opens it
// (`av.open`, decode.py:252). This takes the BYTES plus a `label` used in the
// refusal messages, because the engine already owns the one file-reading helper
// (`ltx2_video.cpp:52`) and a second one would be a parallel path. The label is
// the path the caller opened, so every message still names the file.
//
// `want_channels` is the encoder's `in_channels`; a file that declares anything
// else is REFUSED rather than mixed, for the reason in this header's third
// bullet. `want_sample_rate` is the mel front-end's target; a mismatch is
// REFUSED rather than resampled, because this project ports only the
// integer-ratio hann-sinc resampler and upstream's `ops.py:40` is an
// arbitrary-ratio polyphase kaiser one.
Ltx2DecodedAudio Ltx2DecodeAudioWav(const std::string& bytes, const std::string& label,
                                    int64_t want_channels, int64_t want_sample_rate,
                                    double start_time, double max_duration);

// `encode_audio` (audio_vae.py:249-274) followed by A2Vid's truncation
// (a2vid_two_stage.py:201-202): waveform -> log-mel -> encoder -> keep at most
// `latent_frames` frames. TRUNCATES ONLY; never pads.
//
// Returns the latent CHANNEL-MAJOR as [channels, frames, mel_bins] — the
// unpatchified volume layout `Ltx2AudioPatchify` consumes — and reports the
// frame count it actually produced, which may be FEWER than `latent_frames`.
Ltx2AudioSpectrogram Ltx2EncodeAudioToLatent(const Ltx2AudioEncoderConfig& encoder_config,
                                             const Ltx2AudioProcessorConfig& processor_config,
                                             const Ltx2VaeWeights& weights,
                                             const Ltx2DecodedAudio& audio, int64_t latent_frames);

}  // namespace vllm
