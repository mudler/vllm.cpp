// IndexTTS-2.5 speech-family registration. See indextts2.h for why it refuses.
#include "vllm/model_executor/models/indextts2.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vllm {
namespace models {
namespace {

// Detection INSPECTS the artifact rather than trusting the path spelling, which
// is chosen by whoever repackaged the checkpoint. IndexTTS-2.5 ships a
// `config.yaml` carrying its stage keys (infer_v2_5.py reads `cfg.gpt`,
// `cfg.s2mel_checkpoint` and `cfg.semantic_codec`), so the presence of all three
// is what identifies the family.
bool LooksLikeIndexTts2(const multimodal::SpeechModelParams& params) {
  std::error_code ec;
  const std::filesystem::path config = std::filesystem::path(params.path) / "config.yaml";
  if (!std::filesystem::is_regular_file(config, ec) || ec) return false;

  std::ifstream in(config);
  if (!in) return false;
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();

  // All three, not any: a config carrying only one of them is a different model
  // in the same lineage, and claiming it would be a guess.
  return text.find("gpt:") != std::string::npos &&
         text.find("s2mel_checkpoint") != std::string::npos &&
         text.find("semantic_codec") != std::string::npos;
}

}  // namespace

void RegisterIndexTts2SpeechFamily(multimodal::SpeechRegistry& registry) {
  multimodal::SpeechFamilyRegistration reg;
  reg.name = "indextts2";
  reg.detect = LooksLikeIndexTts2;
  reg.load = [](const multimodal::SpeechModelParams& params)
      -> std::unique_ptr<multimodal::SpeechEngine> {
    // Name every missing piece. "unsupported" with no evidence is what sends the
    // next person reading loader source to work out what was meant.
    throw std::runtime_error(
        "indextts2: recognized an IndexTTS-2.5 checkpoint at '" + params.path +
        "' but the lane is not implemented yet. Ported so far: the GPT-2 talker "
        "backbone (W2) and the shared BigVGAN 1-D core (W1). Still missing: the "
        "mandatory reference-audio conditioning path (w2v-bert-2.0, the MaskGCT "
        "semantic codec, CAMPPlus), the EnhancedCodec, and the S2Mel CFM/DiT "
        "decoder — W3-W5 of .agents/specs/indextts-2-5.md, issue #634. Note "
        "IndexTTS-2 has no text-only synthesis, so the conditioning path is "
        "required rather than optional.");
  };
  registry.Register(std::move(reg));
}

}  // namespace models
}  // namespace vllm
