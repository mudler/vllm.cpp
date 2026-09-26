// diarization.h — diarization seam wrapping parakeet.cpp's C-API
//
// VLLM_WITH_DIARIZATION gates the whole seam. When the parakeet.cpp
// dependency is absent (VLLM_CPP_WITH_DIARIZATION=OFF), the header
// is empty and every function is a no-op stub, so the rest of vllm.cpp
// compiles unchanged.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#ifdef VLLM_WITH_DIARIZATION
#include "parakeet_capi.h"
#endif

namespace vllm::multimodal {

// One speaker segment: who spoke, and when.
struct SpeakerSegment {
    int speaker;   // 0-indexed speaker ID
    float start;   // seconds from audio start
    float end;
};

// One speaker-attributed utterance.
struct SpeakerUtterance {
    int speaker;
    std::string text;
    float start;
    float end;
    float conf;
};

// A loaded diarization model (Nemotron-3-Diarization GGUF).
// Wraps parakeet_ctx from parakeet.cpp's C-API.
class Diarizer {
public:
    // Load a diarization GGUF file. Returns nullptr on failure.
    static std::unique_ptr<Diarizer> FromFile(const std::string& path);

    ~Diarizer();

    // Diarize a mono float32 waveform at `sample_rate`.
    std::vector<SpeakerSegment> Diarize(
        const float* pcm, int64_t n_samples, int sample_rate) const;

    // Diarize a WAV file path.
    std::vector<SpeakerSegment> DiarizeWavFile(const std::string& path) const;

#ifdef VLLM_WITH_DIARIZATION
    // Access the underlying parakeet_ctx for streaming + SAS composition.
    parakeet_ctx* ctx() const { return ctx_; }
#endif

private:
    Diarizer();
#ifdef VLLM_WITH_DIARIZATION
    parakeet_ctx* ctx_ = nullptr;
#endif
};

// Combined ASR + diarization: runs both models and merges the results.
// Takes an ASR engine (from vllm_engine_load with a Parakeet checkpoint)
// and a diarization model. The ASR path goes through the existing
// ParakeetTranscriber; the diarization path goes through parakeet.cpp's
// C-API. The merge uses parakeet.cpp's SAS merge layer.
struct SpeakerAttributedASR {
    // The speaker-attributed utterances.
    std::vector<SpeakerUtterance> utterances;
    // True if at least one utterance was produced.
    bool has_result = false;
};

// Run combined ASR + diarization on a WAV file.
// `asr_dir` is a Parakeet checkpoint directory (HF format).
// `diar_gguf` is a diarization GGUF file path.
SpeakerAttributedASR TranscribeAndDiarize(
    const std::string& wav_path,
    const std::string& asr_dir,
    const std::string& diar_gguf);

// Run combined ASR + diarization on raw PCM.
SpeakerAttributedASR TranscribeAndDiarizePCM(
    const float* pcm, int64_t n_samples, int sample_rate,
    const std::string& asr_dir,
    const std::string& diar_gguf);

}  // namespace vllm::multimodal
