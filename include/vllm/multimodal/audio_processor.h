// Whisper-class AUDIO multimodal input processor — C++ mirror of the vLLM/HF
// audio-processing pipeline for the audio-track vehicle openai/whisper-small
// (audio-track A1). The genuinely-new audio INPUT pipeline: WAV decode -> resample
// -> log-mel `input_features` -> audio placeholder-token expansion -> mm-hash.
//
// Ported from:
//   - transformers WhisperFeatureExtractor (feature_extraction_whisper.py):
//     `_torch_extract_fbank_features` — the torch STFT path that runs when torch
//     is installed (the one the oracle used). hann(n_fft) periodic window,
//     torch.stft(n_fft, hop, center=True, pad=reflect, return_complex),
//     magnitudes=|stft[...,:-1]|**2, mel_filters.T @ magnitudes, log10(clamp 1e-10),
//     x=max(x, x.max()-8), x=(x+4)/4. Waveform padded/truncated to
//     chunk_length*sr=480000 samples (=> 3000 frames).
//   - the mel filterbank (audio_utils.mel_filter_bank, slaney/slaney) is a
//     deterministic config constant, LOADED as a golden [n_freq, n_mels] matrix
//     (construction-from-config is a deferred port; the STFT float ops are the
//     parity variable). Matches the image path's config-constant loading.
//   - vllm/model_executor/models/whisper.py: get_num_audio_tokens:656
//     (= config.max_source_positions), _get_prompt_updates:740 (single placeholder
//     [0] -> [0]*num_audio_tokens).
//   - the mm-hash: MultiModalHasher.hash_kwargs(model_id, audio=<f32 waveform>)
//     (MultiModalHasher::HashAudioF32).
//
// The A1 correctness gate (audio-processor parity) checks the log-mel
// `input_features` rel-L2 within a stated tolerance vs the vLLM 0.25.0 oracle
// fixture, and the placeholder-expanded ids + mm-hash BIT/BYTE-identical.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vllm/multimodal/inputs.h"

namespace vllm::multimodal {

// Decoded mono waveform + its sampling rate (from a canonical PCM16 WAV).
struct DecodedAudio {
  std::vector<float> samples;  // float32 in [-1, 1) (int16 / 32768.0)
  int sampling_rate = 0;
};

// Decode a canonical little-endian PCM16 MONO RIFF/WAVE byte buffer to float32
// (int16 / 32768.0). Only the fmt/data chunks needed by the fixture are parsed;
// throws on non-PCM16 / non-mono / malformed input. Mirrors how the oracle decode
// produced the f32 waveform (int16/32768.0), so C++ and oracle see identical
// samples with zero decode ambiguity.
DecodedAudio DecodeWavPcm16Mono(const uint8_t* wav_bytes, size_t num_bytes);

// The same buffer, but with ANY channel count, reduced to mono by the per-sample
// MEAN over channels -- upstream's own reduction
// (`vllm/multimodal/media/audio.py:207-208` @ `9035151d6`, reached by
// `load_audio`'s `mono=True` default at `:220`; and `ChannelReduction.MEAN` /
// `AudioSpec.target_channels = 1` at `vllm/multimodal/audio.py:69-70`, which
// dots3-note selects at
// `vllm/models/dots3_note/common/processor.py:523-525`).
//
// A SIBLING RATHER THAN A WIDENING, so that `DecodeWavPcm16Mono`'s three
// callers -- parakeet transcription, the ROAD-V1-MM parse path and the voxtral
// e2e gate -- decode the same bytes to the same samples. Both share ONE chunk
// walk, which moves two REFUSAL MESSAGES and no sample; `audio_processor.cpp`
// names both. The mean is accumulated in int32 -- exact over the whole uint16
// channel domain, and no overflow is representable -- and the answer is the
// CORRECTLY-ROUNDED float of that exact mean. It is BIT-IDENTICAL to upstream's
// float32 mean for every power-of-two channel count UP TO 512, C = 1 and C = 2
// included; that bound is TIGHT, and past it the two may differ by half an ulp
// with this arm the more accurate. See `.agents/specs/dots3-note.md` 4.16.2.
// W7c-1, issue #2813.
DecodedAudio DecodeWavPcm16MeanToMono(const uint8_t* wav_bytes, size_t num_bytes);

// The subset of the whisper-small feature-extractor + config the audio path needs.
struct AudioProcessorConfig {
  int n_fft = 400;
  int hop_length = 160;
  int n_mels = 80;              // == feature_size
  int sampling_rate = 16000;
  int chunk_length_s = 30;      // n_samples = chunk_length_s * sampling_rate
  double dither = 0.0;          // whisper-small: no dithering (deterministic)
  int max_source_positions = 1500;  // == num_audio_tokens (encoder output length)
  int32_t audio_placeholder_id = 0;

  std::string model_id = "openai/whisper-small";  // for the mm-hash

  int n_samples_padded() const { return chunk_length_s * sampling_rate; }  // 480000
  int num_freq_bins() const { return 1 + n_fft / 2; }                      // 201
};

class WhisperAudioProcessor {
 public:
  // `mel_filters` is the [num_freq_bins * n_mels] row-major golden filterbank
  // (mel_filters[k * n_mels + m]), matching fe.mel_filters (shape [201, 80]).
  WhisperAudioProcessor(AudioProcessorConfig cfg, std::vector<float> mel_filters);

  const AudioProcessorConfig& config() const { return cfg_; }
  // The [num_freq_bins, n_mels] bank this processor multiplies with. Exposed
  // by dots3-note W7a (#2703) so a gate can compare the SHARED
  // `MelFilterBankSlaney` construction against the committed
  // `voxtral_mel_filters_f32.bin` oracle through the object the front end
  // actually uses, rather than against a second construction beside it.
  const std::vector<float>& mel_filters() const { return mel_filters_; }

  // Compute the log-mel `input_features` [n_mels, n_frames] from a mono waveform
  // at `sample_rate`. If sample_rate != cfg.sampling_rate the (deferred) resample
  // path throws — the whisper-small fixture is already 16 kHz mono (identity),
  // mirroring the image SmartResize/bicubic identity-only deferral.
  AudioKwargs ProcessWaveform(const float* samples, int64_t num_samples,
                              int sample_rate) const;

  // mm-hash of the raw mono f32 waveform (before feature extraction), exactly as
  // vLLM hashes the audio mm item.
  std::string HashAudio(const float* samples, int64_t num_samples) const;

 private:
  AudioProcessorConfig cfg_;
  std::vector<float> mel_filters_;  // [num_freq_bins * n_mels]
};

// Audio placeholder expansion (whisper.py::_get_prompt_updates:740). Replaces each
// `audio_placeholder_id` in `prompt_ids` with `num_audio_tokens` copies of
// `audio_placeholder_id`, consuming one count per placeholder in order. Returns the
// expanded ids and fills `placeholders` with the [offset,length] span of every
// expanded item. Mirrors ExpandImagePlaceholders for the image path.
std::vector<int32_t> ExpandAudioPlaceholders(
    const std::vector<int32_t>& prompt_ids, int32_t audio_placeholder_id,
    const std::vector<int>& num_audio_tokens_per_item,
    std::vector<std::array<int, 2>>* placeholders);

}  // namespace vllm::multimodal
