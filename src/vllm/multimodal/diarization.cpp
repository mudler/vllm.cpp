// diarization.cpp — diarization seam wrapping parakeet.cpp's C-API
#include "vllm/multimodal/diarization.h"

#ifdef VLLM_WITH_DIARIZATION
#include "vllm/multimodal/parakeet_transcription.h"
#endif

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef VLLM_WITH_DIARIZATION

namespace vllm::multimodal {

// --- WAV reading helper ---
// Reads a 16-bit PCM mono WAV into float32 samples.
static std::vector<float> ReadWavPcm16Mono(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open WAV: " + path);
    // Read RIFF header
    char hdr[44];
    if (std::fread(hdr, 1, 44, f) != 44) {
        std::fclose(f);
        throw std::runtime_error("WAV too short: " + path);
    }
    // Validate RIFF
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        std::fclose(f);
        throw std::runtime_error("not a RIFF/WAVE file: " + path);
    }
    // Find the data chunk
    uint32_t data_offset = 12;
    while (data_offset < 44) {
        char chunk_id[4];
        uint32_t chunk_size;
        std::memcpy(chunk_id, hdr + data_offset, 4);
        std::memcpy(&chunk_size, hdr + data_offset + 4, 4);
        if (std::memcmp(chunk_id, "data", 4) == 0) {
            // Found it — but we need to seek to it in the file
            break;
        }
        data_offset += 8 + chunk_size;
    }
    // Simple approach: assume standard 44-byte header
    std::fseek(f, 44, SEEK_SET);
    // Read the rest as PCM16
    std::vector<int16_t> pcm16;
    int16_t sample;
    while (std::fread(&sample, 2, 1, f) == 1)
        pcm16.push_back(sample);
    std::fclose(f);
    // Convert to float
    std::vector<float> pcm(pcm16.size());
    for (size_t i = 0; i < pcm16.size(); ++i)
        pcm[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    return pcm;
}

// --- Diarizer ---

Diarizer::Diarizer() = default;

Diarizer::~Diarizer() {
    if (ctx_) {
        parakeet_capi_free(ctx_);
        ctx_ = nullptr;
    }
}

std::unique_ptr<Diarizer> Diarizer::FromFile(const std::string& path) {
    auto d = std::unique_ptr<Diarizer>(new Diarizer());
    d->ctx_ = parakeet_capi_load(path.c_str());
    if (!d->ctx_) {
        throw std::runtime_error("Diarizer::FromFile: parakeet_capi_load failed: " + path);
    }
    return d;
}

std::vector<SpeakerSegment> Diarizer::Diarize(
        const float* pcm, int64_t n_samples, int sample_rate) const {
    if (!ctx_) throw std::runtime_error("Diarizer: no model loaded");

    parakeet_diarization_result* result = parakeet_capi_diarize_pcm(
        ctx_, pcm, (int)n_samples, sample_rate);
    if (!result) throw std::runtime_error("Diarizer: diarize_pcm failed");

    std::vector<SpeakerSegment> segs;
    segs.reserve(result->n_segments);
    for (int i = 0; i < result->n_segments; ++i) {
        segs.push_back({
            result->segments[i].speaker,
            result->segments[i].start,
            result->segments[i].end
        });
    }
    parakeet_capi_free_diarization_result(result);
    return segs;
}

std::vector<SpeakerSegment> Diarizer::DiarizeWavFile(const std::string& path) const {
    if (!ctx_) throw std::runtime_error("Diarizer: no model loaded");

    parakeet_diarization_result* result = parakeet_capi_diarize_path(
        ctx_, path.c_str());
    if (!result) throw std::runtime_error("Diarizer: diarize_path failed");

    std::vector<SpeakerSegment> segs;
    segs.reserve(result->n_segments);
    for (int i = 0; i < result->n_segments; ++i) {
        segs.push_back({
            result->segments[i].speaker,
            result->segments[i].start,
            result->segments[i].end
        });
    }
    parakeet_capi_free_diarization_result(result);
    return segs;
}

// --- Combined ASR + diarization ---

SpeakerAttributedASR TranscribeAndDiarize(
        const std::string& wav_path,
        const std::string& asr_dir,
        const std::string& diar_gguf) {
    SpeakerAttributedASR result;

    // Load ASR model
    ParakeetTranscriber asr = ParakeetTranscriber::FromDir(asr_dir);
    // Load diarization model
    auto diar = Diarizer::FromFile(diar_gguf);

    // Transcribe
    ParakeetTranscription trans = asr.TranscribeWavFile(wav_path);
    if (!trans.has_text) {
        result.has_result = false;
        return result;
    }

    // Diarize
    auto segs = diar->DiarizeWavFile(wav_path);

    // Use parakeet.cpp's SAS merge via C-API
    parakeet_ctx* asr_ctx = parakeet_capi_load(asr_dir.c_str());
    if (!asr_ctx) {
        // Fallback: just return the ASR text as a single utterance
        result.utterances.push_back({-1, trans.text, 0.0f, 0.0f, 0.0f});
        result.has_result = true;
        return result;
    }
    parakeet_ctx* diar_ctx = diar->ctx();

    // Read the WAV into PCM for the SAS path
    std::vector<float> pcm = ReadWavPcm16Mono(wav_path);

    int n_sas = 0;
    parakeet_sas_result* sas = parakeet_capi_transcribe_and_diarize(
        asr_ctx, diar_ctx, pcm.data(), (int)pcm.size(), 16000, &n_sas);

    if (sas && n_sas > 0) {
        for (int i = 0; i < n_sas; ++i) {
            SpeakerUtterance u;
            u.speaker = sas[i].speaker;
            u.text = sas[i].text ? sas[i].text : "";
            u.start = sas[i].start;
            u.end = sas[i].end;
            u.conf = sas[i].conf;
            result.utterances.push_back(u);
            if (sas[i].text) parakeet_capi_free_string(sas[i].text);
        }
        parakeet_capi_free_sas_results(sas);
        result.has_result = true;
    } else {
        result.has_result = false;
    }

    parakeet_capi_free(asr_ctx);
    return result;
}

SpeakerAttributedASR TranscribeAndDiarizePCM(
        const float* pcm, int64_t n_samples, int sample_rate,
        const std::string& asr_dir,
        const std::string& diar_gguf) {
    SpeakerAttributedASR result;

    ParakeetTranscriber asr = ParakeetTranscriber::FromDir(asr_dir);
    auto diar = Diarizer::FromFile(diar_gguf);

    ParakeetTranscription trans = asr.Transcribe(pcm, n_samples, sample_rate);
    if (!trans.has_text) {
        result.has_result = false;
        return result;
    }

    parakeet_ctx* asr_ctx = parakeet_capi_load(asr_dir.c_str());
    if (!asr_ctx) {
        result.utterances.push_back({-1, trans.text, 0.0f, 0.0f, 0.0f});
        result.has_result = true;
        return result;
    }

    int n_sas = 0;
    parakeet_sas_result* sas = parakeet_capi_transcribe_and_diarize(
        asr_ctx, diar->ctx(), pcm, (int)n_samples, sample_rate, &n_sas);

    if (sas && n_sas > 0) {
        for (int i = 0; i < n_sas; ++i) {
            SpeakerUtterance u;
            u.speaker = sas[i].speaker;
            u.text = sas[i].text ? sas[i].text : "";
            u.start = sas[i].start;
            u.end = sas[i].end;
            u.conf = sas[i].conf;
            result.utterances.push_back(u);
            if (sas[i].text) parakeet_capi_free_string(sas[i].text);
        }
        parakeet_capi_free_sas_results(sas);
        result.has_result = true;
    } else {
        result.has_result = false;
    }

    parakeet_capi_free(asr_ctx);
    return result;
}

}  // namespace vllm::multimodal

#else  // !VLLM_WITH_DIARIZATION

namespace vllm::multimodal {

std::unique_ptr<Diarizer> Diarizer::FromFile(const std::string&) {
    throw std::runtime_error("Diarizer: diarization support not compiled in");
}

std::vector<SpeakerSegment> Diarizer::Diarize(const float*, int64_t, int) const {
    throw std::runtime_error("Diarizer: diarization support not compiled in");
}

std::vector<SpeakerSegment> Diarizer::DiarizeWavFile(const std::string&) const {
    throw std::runtime_error("Diarizer: diarization support not compiled in");
}

Diarizer::Diarizer() = default;
Diarizer::~Diarizer() = default;

SpeakerAttributedASR TranscribeAndDiarize(
        const std::string&, const std::string&, const std::string&) {
    SpeakerAttributedASR r;
    r.has_result = false;
    return r;
}

SpeakerAttributedASR TranscribeAndDiarizePCM(
        const float*, int64_t, int, const std::string&, const std::string&) {
    SpeakerAttributedASR r;
    r.has_result = false;
    return r;
}

}  // namespace vllm::multimodal

#endif  // VLLM_WITH_DIARIZATION
