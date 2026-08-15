// The ONE mapping from a parsed `/v1/audio/speech` request to the speech seam
// and back. See speech_api.h for why it is here and not in the server's lambda.
//
// Split from speech_api.cpp so the pure request contract stays free of the
// model-executor dependency the shared RIFF writer brings with it.
#include "vllm/entrypoints/openai/speech_api.h"

#include <string>

#include "vllm/model_executor/models/minimax_h3.h"  // MiniMaxH3WriteWav, the shared writer

namespace vllm::openai {

SpeechResponse SynthesizeSpeechRequest(::vllm::multimodal::SpeechEngine& engine,
                                       const SpeechRequest& request) {
  // The route validated the ENVELOPE; the family validates its own fields and
  // refuses BY NAME, which is why nothing here second-guesses a value.
  ::vllm::multimodal::SpeechGenParams gen;
  gen.text = request.text;
  gen.language = request.language;
  gen.lyrics = request.lyrics;
  gen.description = request.description;
  gen.reference_audio = request.reference_audio;
  gen.reference_sample_rate = request.reference_sample_rate;
  gen.audio_duration_s = request.audio_duration_s;
  gen.num_inference_steps = request.num_inference_steps;
  // NEGATIVE means "the family decides": 0 is a legal guidance scale, so the
  // FLAG and not the value is what selects the default.
  gen.guidance_scale = request.has_guidance_scale ? request.guidance_scale : -1.0;
  gen.seed = request.seed;

  const ::vllm::multimodal::SpeechResult result = engine.Synthesize(gen);
  SpeechResponse out;
  out.sample_rate = result.sample_rate;
  out.channels = result.channels;
  out.samples_per_channel =
      result.channels > 0 ? static_cast<int64_t>(result.samples.size()) / result.channels : 0;
  // The SHARED RIFF writer the H3 and LTX-2.5 audio paths already use, so HTTP
  // and the C ABI cannot emit different bytes for the same waveform.
  out.wav = ::vllm::MiniMaxH3WriteWav(result.samples, out.channels, out.samples_per_channel,
                                      out.sample_rate);
  return out;
}

}  // namespace vllm::openai
