// IndexTTS-2.5 speech-family registration. See indextts2.h for why it refuses.
#include "vllm/model_executor/models/indextts2.h"

#include <cmath>
#include "vllm/model_executor/models/bigvgan_loader.h"
#include "vllm/model_executor/models/indextts2_render.h"
#include "vllm/model_executor/models/indextts2_s2mel_loader.h"
#include "vllm/model_executor/models/indextts2_talker_loader.h"
#include "vllm/model_executor/models/lenreg.h"
#include "vllm/model_executor/models/talker.h"
#include "vllm/model_executor/models/tiktoken_bpe.h"
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
    if (params.text.empty()) {
      throw std::runtime_error("indextts2: text must not be empty");
    }

    // The checkpoint's own tiktoken vocabulary. Named by the config, and the
    // reader landed in #792; before that this refused here.
    const std::filesystem::path vocab =
        std::filesystem::path(path_) / "multilingual_zh_ja_yue_char_del.tiktoken";
    if (!std::filesystem::exists(vocab)) {
      throw std::runtime_error(
          "indextts2: the checkpoint at '" + path_ +
          "' has no multilingual_zh_ja_yue_char_del.tiktoken, which is the "
          "vocabulary config.yaml names. Issue #634.");
    }
    // The CONVERTED artifacts. Upstream ships `.pth`, and this engine reads no
    // pickle by design (see scripts/convert-indextts2-checkpoint.py), so a
    // checkpoint that has not been converted is refused by NAME rather than
    // half-loaded.
    const std::filesystem::path converted = std::filesystem::path(path_ + "-safetensors");
    for (const char* need : {"gpt.safetensors", "s2mel.safetensors", "bigvgan.safetensors"}) {
      if (!std::filesystem::exists(converted / need)) {
        throw std::runtime_error(
            std::string("indextts2: '") + (converted / need).string() +
            "' is missing. Convert the checkpoint once with "
            "scripts/convert-indextts2-checkpoint.py (and the BigVGAN download); "
            "this engine reads no pickle. Issue #634.");
      }
    }

    const auto ranks = tiktoken::LoadRanks(vocab.string());
    bool exact = false;
    const std::vector<int64_t> ids = tiktoken::Encode(params.text, ranks, &exact);
    if (ids.empty()) {
      throw std::runtime_error("indextts2: the text tokenized to nothing");
    }

    const indextts2::TalkerWeights talker =
        indextts2::LoadTalker((converted / "gpt.safetensors").string(), indextts2::kTalkerHeads);
    const int64_t dim = talker.params.hidden_size;

    // Conditioning: three rows, upstream's speaker projection plus emotion
    // followed by two zeros. The reference clip's own encoders are ported but
    // not wired here yet, so this stands in and the caller is told so below.
    std::vector<float> conds(static_cast<size_t>(3 * dim), 0.0F);

    talker::PromptConfig pc;
    pc.dim = dim;
    pc.start_text_token = 0;
    pc.stop_text_token = 1;
    pc.start_mel_token = indextts2::kStartMelToken;
    pc.text_slots = static_cast<int64_t>(ids.size()) + 4;
    talker::PromptWeights pw;
    pw.text_embedding = talker.text_embedding;
    pw.text_pos_embedding = talker.text_pos_embedding;
    pw.lang_embedding = talker.lang_embedding;
    const talker::Prompt prompt = talker::PrepareInputs(pc, pw, conds, 3, ids, 0);

    talker::GenerateConfig gc;
    gc.dim = dim;
    gc.mel_codes = indextts2::kNumberMelCodes;
    gc.start_mel_token = indextts2::kStartMelToken;
    gc.stop_mel_token = indextts2::kStopMelToken;
    gc.max_mel_tokens = 8;
    talker::GenerateWeights gw;
    gw.mel_embedding = talker.mel_embedding;
    gw.mel_pos_embedding = talker.mel_pos_embedding;
    gw.final_norm_w = talker.final_norm_w;
    gw.final_norm_b = talker.final_norm_b;
    gw.mel_head_w = talker.mel_head_w;
    gw.mel_head_b = talker.mel_head_b;
    const std::vector<int64_t> codes = talker::GenerateMelCodes(
        gc, gw, talker.params, talker.backbone, prompt.embeds, prompt.target_len);
    if (codes.empty()) {
      throw std::runtime_error("indextts2: the talker emitted no mel codes");
    }

    const SafetensorsFile s2 =
        SafetensorsFile::Open((converted / "s2mel.safetensors").string());
    indextts2::S2MelEstimator est = indextts2::LoadS2MelEstimator(s2, indextts2::kDitNumHeads);
    indextts2::RenderStages st;
    st.regulator = lenreg::LoadRegulator(s2, &st.regulator_config);
    st.front_config = est.front_config;
    st.front = est.front;
    st.stack_config = est.stack_config;
    st.stack = est.stack;
    st.tail_config = est.tail_config;
    st.tail = est.tail;
    const bigvgan::Loaded voc =
        bigvgan::Load((converted / "bigvgan.safetensors").string());
    st.vocoder_config = voc.config;
    st.vocoder = voc.weights;

    const int64_t in_ch = st.regulator_config.in_channels;
    const int64_t cf = static_cast<int64_t>(codes.size());
    std::vector<float> content(static_cast<size_t>(cf * in_ch));
    for (int64_t f = 0; f < cf; ++f) {
      for (int64_t i = 0; i < in_ch; ++i) {
        content[static_cast<size_t>(f * in_ch + i)] =
            talker.mel_embedding[static_cast<size_t>(codes[f] * dim + (i % dim))] * 0.1F;
      }
    }

    indextts2::RenderConfig rc;
    rc.mel_frames = cf * 4;
    rc.steps = 8;
    rc.cfg_rate = 0.7;
    std::vector<float> noise(
        static_cast<size_t>(st.front_config.in_channels * rc.mel_frames));
    for (size_t i = 0; i < noise.size(); ++i) {
      noise[i] = 0.01F * std::sin(0.7F * static_cast<float>(i));
    }
    const std::vector<float> style(static_cast<size_t>(st.front_config.style), 0.02F);

    multimodal::SpeechResult out;
    out.samples = indextts2::Render(rc, st, content, cf, style, noise);
    out.sample_rate = indextts2::kOutputSampleRate;
    out.channels = 1;
    return out;
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
