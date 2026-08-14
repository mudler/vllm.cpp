// SeamlessM4T feature extraction — reference clip to w2v-bert input (#634).
//
// The class `indextts/infer_v2_5.py:173` instantiates. Gated against
// `transformers` executed directly, the same secondary oracle `w2vbert.h`
// already uses.
//
//   waveform x 2^15                 // Kaldi expects 16-bit integers
//   per frame: remove DC, preemphasis 0.97, povey window
//   |rfft(512)|^2 -> kaldi mel(80) -> log(max(x, mel_floor))
//   per utterance: (x - mean) / sqrt(var + 1e-7)      // VAR IS UNBIASED
//   pad frames up to a multiple of stride with `padding_value`
//   stack stride 2                                    // 80 -> 160 columns
//
// Four details, each of which still produces a plausible spectrogram:
//   - the window is POVEY (hann^0.85), not hann.
//   - the mel scale is KALDI and the triangles are built IN MEL SPACE, not the
//     slaney scale this tree's other log-mel front end uses.
//   - DC removal happens PER FRAME and BEFORE preemphasis.
//   - the normalization variance is UNBIASED (ddof=1). At 12 frames the
//     difference between /n and /(n-1) is 4%, which no shape check can see.
//   - an odd frame count is PADDED UP, not truncated down. `__call__` defaults
//     `pad_to_multiple_of=2`, so 23 raw frames become 24 and then 12 stacked
//     ones. Truncating gives 11 -- a whole stacked frame of audio silently
//     dropped -- and that is what this port did first.
#pragma once

#include <cstdint>
#include <vector>

namespace vllm {
namespace models {
namespace w2v_fbank {

struct Config {
  int64_t frame_length = 400;
  int64_t hop = 160;
  int64_t fft_length = 512;
  int64_t mel_bins = 80;
  int64_t stride = 2;
  double preemphasis = 0.97;
  double mel_floor = 1.192092955078125e-07;
  double norm_eps = 1e-7;
  // The value padded frames are filled with; IndexTTS-2.5 passes 1.
  float padding_value = 1.0F;
};

// hann(length, symmetric)^0.85.
std::vector<float> PoveyWindow(int64_t length);

// The kaldi-scale filterbank, [freq_bins, mel_bins], unnormalized, with the
// triangles laid out in mel space.
std::vector<float> KaldiMelFilterbank(int64_t freq_bins, int64_t mel_bins,
                                      double min_hz, double max_hz, double sample_rate);

// The whole extractor. `audio` is mono at 16 kHz. Returns [frames, mel_bins *
// stride], frame-major, and reports `frames`.
//
// `raw_log_mel` optionally captures the log-mel BEFORE normalization,
// [raw_frames, mel_bins]. That quantity is what correctness lives in and it is
// well conditioned; the normalized output is NOT, because bins whose variance
// falls below `norm_eps` are divided by ~sqrt(norm_eps) and any implementation
// difference in them is amplified by three orders of magnitude. Both are gated,
// at tolerances measured for each rather than one number chosen for both.
std::vector<float> Extract(const Config& cfg, const std::vector<float>& audio,
                           double sample_rate, int64_t* frames,
                           std::vector<float>* raw_log_mel = nullptr);

}  // namespace w2v_fbank
}  // namespace models
}  // namespace vllm
