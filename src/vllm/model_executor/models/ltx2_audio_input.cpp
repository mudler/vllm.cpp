// LTX-2.5 AUDIO-TO-VIDEO — ingestion. See ltx2_audio_input.h for the port map
// and for the three failure modes this file exists to make impossible.
//
// Row LTX25-A2V-AUDIO-INPUT (#922). Upstream: Lightricks/LTX-2 @ fd4ded7f.
#include "vllm/model_executor/models/ltx2_audio_input.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/ltx2_pipeline.h"
#include "vllm/model_executor/models/minimax_h3.h"

namespace vllm {
namespace {

[[noreturn]] void Fail(const std::string& why) {
  throw std::runtime_error("ltx2 audio input: " + why);
}

uint32_t ReadLE(const unsigned char* p, size_t n) {
  uint32_t v = 0;
  for (size_t i = 0; i < n; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}

// The WAVE format tag of a chunk we can decode. `WAVE_FORMAT_PCM`; anything else
// is a compressed or float payload this path does not accept.
constexpr uint32_t kWaveFormatPcm = 1;

}  // namespace

Ltx2WavFormat Ltx2ProbeWavFormat(const std::string& bytes) {
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const size_t n = bytes.size();
  // 12 bytes of RIFF header plus an 8-byte chunk header is the least that can
  // carry a `fmt ` chunk at all.
  if (n < 20 || std::memcmp(p, "RIFF", 4) != 0 || std::memcmp(p + 8, "WAVE", 4) != 0) {
    Fail("the audio file is not a RIFF/WAVE buffer. Upstream decodes whatever PyAV can open "
         "(media_io/decode.py:252) and this project vendors no demuxer, so MP3, FLAC, OGG and "
         "MP4 audio are refused rather than misread. Supply 16-bit PCM WAV");
  }
  Ltx2WavFormat out;
  bool have_fmt = false;
  size_t pos = 12;
  while (pos + 8 <= n) {
    const unsigned char* id = p + pos;
    const uint32_t size = ReadLE(p + pos + 4, 4);
    const size_t body = pos + 8;
    if (body + size > n) break;
    if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16) {
      const uint32_t tag = ReadLE(p + body, 2);
      if (tag != kWaveFormatPcm) {
        Fail("the WAV declares format tag " + std::to_string(tag) +
             ", not 1 (WAVE_FORMAT_PCM). A compressed or floating-point payload would be read "
             "as integer samples and produce a plausible, wrong waveform");
      }
      out.channels = static_cast<int64_t>(ReadLE(p + body + 2, 2));
      out.sample_rate = static_cast<int64_t>(ReadLE(p + body + 4, 4));
      out.bits_per_sample = static_cast<int64_t>(ReadLE(p + body + 14, 2));
      have_fmt = true;
    }
    pos = body + size + (size & 1);  // chunks are word-aligned
  }
  if (!have_fmt) Fail("the WAV carries no 'fmt ' chunk");
  return out;
}

Ltx2DecodedAudio Ltx2DecodeAudioWav(const std::string& bytes, const std::string& label,
                                    int64_t want_channels, int64_t want_sample_rate,
                                    double start_time, double max_duration) {
  const Ltx2WavFormat fmt = Ltx2ProbeWavFormat(bytes);

  // Refused BEFORE the decode, because `MiniMaxH3ReadWav` would silently repeat
  // a mono take across both channels (minimax_h3.h:1839-1845). That is H3's
  // contract, not LTX-2's: upstream hands the file's own channel count to a conv
  // declaring `in_channels = 2` (model_configurator.py:172), so a mono file is
  // an error there and must be an error here. Duplicating it would condition the
  // render on audio the caller never supplied and produce a finished clip.
  if (fmt.channels != want_channels) {
    Fail("'" + label + "' carries " + std::to_string(fmt.channels) +
         " audio channel(s) and this checkpoint's audio VAE encoder declares in_channels = " +
         std::to_string(want_channels) +
         " (audio_vae/model_configurator.py:172). Upstream feeds the file's own channel count "
         "straight into that convolution, so a mismatch is an error there too. Refused rather "
         "than up- or down-mixed: either would condition the render on a waveform the caller "
         "did not supply");
  }
  if (fmt.bits_per_sample != 16) {
    Fail("'" + label + "' is " + std::to_string(fmt.bits_per_sample) +
         "-bit PCM; only 16-bit is decoded here");
  }
  if (want_sample_rate > 0 && fmt.sample_rate != want_sample_rate) {
    Fail("'" + label + "' is sampled at " + std::to_string(fmt.sample_rate) +
         " Hz and this checkpoint's mel front-end targets " + std::to_string(want_sample_rate) +
         " Hz (audio_vae/ops.py:19-34). Upstream resamples with "
         "`torchaudio.functional.resample` (ops.py:40), an arbitrary-ratio polyphase kaiser "
         "resampler; this project ports only the integer-ratio hann-sinc variant, so the rate "
         "is refused rather than assumed. Treating these samples as " +
         std::to_string(want_sample_rate) +
         " Hz would pitch- and time-shift the conditioning while every shape still checked out");
  }

  Ltx2DecodedAudio out;
  out.channels = fmt.channels;
  out.sample_rate = fmt.sample_rate;

  // CHANNEL-MAJOR [channels, samples_per_channel], the layout
  // `Ltx2WaveformToLogMel` takes. `want_channels` equals the declared count by
  // the guard above, so the reader's mono-repeat and wide-truncate arms are both
  // no-ops here and only its chunk walking and int16 -> float scaling run.
  int64_t samples_per_channel = 0;
  out.samples = MiniMaxH3ReadWav(bytes, fmt.channels, fmt.sample_rate, &samples_per_channel);

  // ── the time window (decode.py:286-296) ────────────────────────────────────
  // Upstream trims in SAMPLES after the decode because a codec's frame
  // boundaries need not align with the requested range, and its leading trim is
  // `round((start_time - first_frame_time) * sample_rate)`. RIFF is
  // uncompressed, so the first sample IS at t = 0 and `first_frame_time` is
  // exactly 0.0. That is why a compressed container is refused above rather than
  // approximated: there the two expressions differ.
  const double rate = static_cast<double>(fmt.sample_rate);
  int64_t skip = 0;
  if (start_time > 0.0) skip = static_cast<int64_t>(std::llround(start_time * rate));
  if (skip < 0) skip = 0;
  if (skip >= samples_per_channel) {
    // The upstream chain, READ rather than assumed, because a refusal that
    // misstates its reason sends the next reader to the wrong file. Every
    // decoded frame ends before `start_time`, so the decode loop's
    // `if frame_end < start_time: continue` (decode.py:271-272) drops all of
    // them, `if not samples: return None` (decode.py:281-282) fires, and the
    // pipeline's `if decoded_audio is None: raise ValueError`
    // (a2vid_two_stage.py:197-198) is what the caller actually sees.
    Fail("'" + label + "' has " +
         std::to_string(static_cast<double>(samples_per_channel) / rate) +
         " s of audio and audio_start_time is " + std::to_string(start_time) +
         " s, so the requested window begins past the end of the stream. Upstream's decode "
         "drops every frame that ends before start_time and then returns None "
         "(decode.py:271-272, :281-282), and the pipeline raises "
         "(a2vid_two_stage.py:197-198)");
  }
  int64_t keep = samples_per_channel - skip;
  if (max_duration > 0.0) {
    const int64_t cap = static_cast<int64_t>(std::llround(max_duration * rate));
    keep = std::min(keep, cap);
  }
  if (keep <= 0) {
    // A DIFFERENT upstream path from the one above, and saying "the pipeline
    // raises at :198" here would be wrong. `max_samples = round(max_duration *
    // sample_rate)` can be 0, and `audio[..., :0]` (decode.py:295-296) is an
    // `Audio` with zero samples rather than `None` — so `decoded_audio is None`
    // is FALSE and a2vid_two_stage.py:197-198 never fires. Upstream still fails,
    // one or two hops later: the mel front-end and `vae_encode_audio` run on an
    // empty waveform and, if anything survives that, the empty latent trips
    // `AudioLatentTools.create_initial_state`'s shape assertion
    // (tools.py:253-255). We refuse HERE, which is stricter than upstream by the
    // distance between those two points and is stated as such rather than
    // dressed up as a mirror.
    Fail("'" + label + "' yields no audio samples for the requested window (audio_start_time " +
         std::to_string(start_time) + " s, audio_max_duration " + std::to_string(max_duration) +
         " s). Upstream would hand on a ZERO-SAMPLE take here rather than raising "
         "(decode.py:295-296 returns an empty Audio, so a2vid_two_stage.py:197-198 does not "
         "fire) and fail later in the encode or at the latent shape assertion "
         "(tools.py:253-255); this refuses at the window instead, which is STRICTER than "
         "upstream and reports the knob that is wrong");
  }

  if (skip != 0 || keep != samples_per_channel) {
    std::vector<float> windowed(static_cast<size_t>(fmt.channels) * static_cast<size_t>(keep));
    for (int64_t c = 0; c < fmt.channels; ++c) {
      const size_t src = static_cast<size_t>(c * samples_per_channel + skip);
      const size_t dst = static_cast<size_t>(c * keep);
      std::copy(out.samples.begin() + static_cast<ptrdiff_t>(src),
                out.samples.begin() + static_cast<ptrdiff_t>(src + static_cast<size_t>(keep)),
                windowed.begin() + static_cast<ptrdiff_t>(dst));
    }
    out.samples = std::move(windowed);
  }
  out.samples_per_channel = keep;
  return out;
}

Ltx2AudioSpectrogram Ltx2EncodeAudioToLatent(const Ltx2AudioEncoderConfig& encoder_config,
                                             const Ltx2AudioProcessorConfig& processor_config,
                                             const Ltx2VaeWeights& weights,
                                             const Ltx2DecodedAudio& audio,
                                             int64_t latent_frames) {
  if (latent_frames < 1) Fail("the target audio latent resolved to zero frames");

  // `encode_audio` (audio_vae.py:249-274): waveform -> log-mel -> encoder. The
  // processor is built from the ENCODER's own fields upstream (`:264-269`), and
  // here it comes from the same metadata object in one parse, so the two cannot
  // disagree.
  int64_t mel_frames = 0;
  const std::vector<float> mel =
      Ltx2WaveformToLogMel(processor_config, audio.samples, audio.channels,
                           audio.samples_per_channel, audio.sample_rate, &mel_frames);

  Ltx2AudioSpectrogram latent = Ltx2AudioEncoderForward(
      encoder_config, weights, mel, audio.channels, mel_frames, processor_config.mel_bins);

  // ── the truncation (a2vid_two_stage.py:201-202) ────────────────────────────
  // ONE-SIDED, and this is the line that carries it. Upstream slices
  // `[:, :, : audio_shape.frames]` and has no padding arm; the padding helper in
  // the same package (`utils/helpers.py:149-162`) belongs to Retake, which
  // conforms rather than truncates. Growing this into a pad would weld silence
  // onto the end of a short take and still render.
  if (latent.frames > latent_frames) {
    std::vector<float> cut(static_cast<size_t>(latent.channels) *
                           static_cast<size_t>(latent_frames) *
                           static_cast<size_t>(latent.mel_bins));
    for (int64_t c = 0; c < latent.channels; ++c) {
      for (int64_t f = 0; f < latent_frames; ++f) {
        const size_t src = static_cast<size_t>((c * latent.frames + f) * latent.mel_bins);
        const size_t dst = static_cast<size_t>((c * latent_frames + f) * latent.mel_bins);
        std::copy(latent.data.begin() + static_cast<ptrdiff_t>(src),
                  latent.data.begin() +
                      static_cast<ptrdiff_t>(src + static_cast<size_t>(latent.mel_bins)),
                  cut.begin() + static_cast<ptrdiff_t>(dst));
      }
    }
    latent.data = std::move(cut);
    latent.frames = latent_frames;
  }

  // ── and the assertion the truncation leaves standing (tools.py:253-255) ────
  // The slice above can only SHORTEN, so a take shorter than the clip stays
  // short — and upstream then trips
  //   assert initial_latent.shape == self.target_shape.to_torch_shape()
  // in `AudioLatentTools.create_initial_state`. So "audio shorter than the
  // video" is an ERROR upstream, not a short latent, and the two lines have to
  // be read together: truncate-only is not permission to under-fill.
  //
  // Zero-padding to the target here is the tempting repair and it is the wrong
  // one twice over: it diverges from upstream, and it welds silence onto the end
  // of the take while the render finishes normally.
  if (latent.frames < latent_frames) {
    // `latents_per_second` (types.py:174): sample_rate / hop_length /
    // audio_latent_downsample_factor. Derived from the checkpoint's own
    // front-end rather than written as 25.0, so a checkpoint with a different
    // hop reports seconds that are actually its seconds.
    const Ltx2AudioPatchifierParams patch;
    const double per_second = static_cast<double>(processor_config.target_sample_rate) /
                              static_cast<double>(processor_config.mel_hop_length) /
                              static_cast<double>(patch.audio_latent_downsample_factor);
    const double have = static_cast<double>(latent.frames) / per_second;
    const double need = static_cast<double>(latent_frames) / per_second;
    Fail("the audio encodes to " + std::to_string(latent.frames) + " latent frames (" +
         std::to_string(have) + " s) and this clip needs " + std::to_string(latent_frames) +
         " (" + std::to_string(need) +
         " s). Upstream truncates a LONG take and never pads a short one "
         "(a2vid_two_stage.py:202), then asserts the latent matches the target shape "
         "(tools.py:253-255), so a take shorter than the clip is an error there too. Supply at "
         "least " + std::to_string(need) +
         " s of audio, raise audio_max_duration, or shorten the clip. Padding to length would "
         "weld silence onto the end and still render");
  }
  return latent;
}

}  // namespace vllm
