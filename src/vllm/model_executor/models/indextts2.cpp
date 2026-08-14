// IndexTTS-2.5 speech-family registration. See indextts2.h for why it refuses.
#include "vllm/model_executor/models/indextts2.h"

#include "vllm/model_executor/models/indextts2_config.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <memory>
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


// The seam implementation. Everything it needs is ported and gated; what it
// CANNOT do is turn text into tokens, because the checkpoint's tokenizer is
// tiktoken and this tree has no reader for one -- the constraint already
// recorded for Kimi-Linear. So loading SUCCEEDS, the capability is reachable
// and describable through the seam, and `Synthesize` refuses by naming exactly
// that one missing piece rather than a stale list.
class IndexTts2Engine : public multimodal::SpeechEngine {
 public:
  explicit IndexTts2Engine(std::string path) : path_(std::move(path)) {}

  std::string family() const override { return "indextts2"; }

  // 22050, from the shipped config; NOT the talker's 24 kHz mel front end,
  // which is a different rate in the same model.
  int64_t sample_rate() const override { return indextts2::kOutputSampleRate; }

  // Upstream has no text-only synthesis, so this is true and a server can
  // reject before staging any weights.
  bool requires_reference_audio() const override { return true; }

  multimodal::SpeechResult Synthesize(const multimodal::SpeechGenParams& params) override {
    if (params.reference_audio.empty()) {
      throw std::runtime_error(
          "indextts2: a reference clip is REQUIRED -- IndexTTS-2 has no "
          "text-only synthesis, so an empty clip is a refusal rather than a "
          "default voice");
    }
    throw std::runtime_error(
        "indextts2: cannot tokenize '" + params.text +
        "'. The render path itself is implemented and runs on the real "
        "checkpoints -- see `indextts2::Render`, which takes the talker's mel "
        "codes through the length regulator, the CFM loop over the S2Mel "
        "estimator and BigVGAN. What is missing between text and that path is "
        "the TOKENIZER: this checkpoint ships "
        "`multilingual_zh_ja_yue_char_del.tiktoken` and no `tokenizer.json`, "
        "and this tree has no tiktoken reader (the same constraint recorded for "
        "Kimi-Linear). Issue #634. Correctness against vLLM-Omni additionally "
        "needs the oracle pin, #633.");
  }

 private:
  std::string path_;
};

}  // namespace

void RegisterIndexTts2SpeechFamily(multimodal::SpeechRegistry& registry) {
  multimodal::SpeechFamilyRegistration reg;
  reg.name = "indextts2";
  reg.detect = LooksLikeIndexTts2;
  reg.load = [](const multimodal::SpeechModelParams& params)
      -> std::unique_ptr<multimodal::SpeechEngine> {
    return std::make_unique<IndexTts2Engine>(params.path);
  };
  registry.Register(std::move(reg));
}

}  // namespace models
}  // namespace vllm
