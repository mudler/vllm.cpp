// IndexTTS-2.5 shape contract, read from the SHIPPED config.yaml (#634).
//
// Every value here was taken from
// huggingface.co/IndexTeam/IndexTTS-2.5/resolve/main/config.yaml, not inferred
// from the paper or the recipe page. `tests/vllm/models/test_indextts2_config`
// pins them, so a port built against the wrong dimensions fails at the contract
// rather than at a tensor.
#pragma once

#include <cstdint>

namespace vllm {
namespace models {
namespace indextts2 {

// ── THE MODEL RUNS AT TWO SAMPLE RATES, and conflating them is the first
// mistake available. The talker's mel front end is 24 kHz with 100 mel bins;
// S2Mel and the vocoder work at 22.05 kHz with 80. The OUTPUT is 22.05 kHz.
inline constexpr int64_t kTalkerMelSampleRate = 24000;
inline constexpr int64_t kTalkerMelBins = 100;
inline constexpr int64_t kOutputSampleRate = 22050;
inline constexpr int64_t kS2MelMelBins = 80;
inline constexpr int64_t kHopLength = 256;
inline constexpr int64_t kNFft = 1024;

// ── talker (gpt.pth)
inline constexpr int64_t kTalkerDim = 1280;
inline constexpr int64_t kTalkerLayers = 24;
inline constexpr int64_t kTalkerHeads = 20;
inline constexpr int64_t kNumberTextTokens = 60509;
inline constexpr int64_t kNumberMelCodes = 8194;
inline constexpr int64_t kStartMelToken = 8192;
inline constexpr int64_t kStopMelToken = 8193;
inline constexpr int64_t kMaxMelTokens = 1815;
inline constexpr int64_t kMaxTextTokens = 600;
inline constexpr int64_t kMelLengthCompression = 1024;

// ── semantic codec (codec.pth)
inline constexpr int64_t kCodecCodebookSize = 8192;
inline constexpr int64_t kCodecHiddenSize = 1024;
inline constexpr int64_t kCodecCodebookDim = 8;
inline constexpr int64_t kVocosDim = 384;
inline constexpr int64_t kVocosIntermediateDim = 2048;
inline constexpr int64_t kVocosNumLayers = 12;

// ── S2Mel (s2mel.pth)
inline constexpr int64_t kStyleDim = 192;          // CAMPPlus embedding
inline constexpr int64_t kLengthRegulatorChannels = 512;
inline constexpr int64_t kLengthRegulatorInChannels = 1024;
inline constexpr int64_t kDitHiddenDim = 512;
inline constexpr int64_t kDitNumHeads = 8;
inline constexpr int64_t kDitDepth = 13;
inline constexpr int64_t kDitInChannels = 80;

}  // namespace indextts2
}  // namespace models
}  // namespace vllm
