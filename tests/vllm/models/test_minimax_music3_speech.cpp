// MiniMax-Music3 — the SPEECH-FAMILY seam, the request contract and the chunk
// plan (W6 of #672). No checkpoint, no goldens: everything here is either pure
// or reads a SYNTHETIC component tree this file writes, so it runs in CI.
//
// The full-scale numbers live in tests/parity/test_minimax_music3_e2e_real.cpp,
// which drives the shipped weights against the oracle's own waveform.
//
// WHAT THE SYNTHETIC TREE IS FOR. `Music3SpeechEngine`'s constructor resolves
// the checkpoint, parses all six configs and enforces the §2.1 dtype invariant
// BEFORE a byte of weight is read. A tree of real config.json files beside
// zero-byte shard placeholders exercises exactly that, which is why
// `sample_rate()`, `channels` and `requires_reference_audio()` can be gated
// without the 28.5 GB asset. The configs below are the RELEASED checkpoint's,
// copied verbatim (.agents/porting-a-model.md §1: a fixture pins the real
// values so CI catches drift).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/model_executor/models/indextts2.h"
#include "vllm/model_executor/models/minimax_music3_speech.h"
#include "vllm/multimodal/speech_engine.h"

namespace fs = std::filesystem;
namespace m3 = vllm::models::music3;

using vllm::multimodal::SpeechEngine;
using vllm::multimodal::SpeechGenParams;
using vllm::multimodal::SpeechModelParams;
using vllm::multimodal::SpeechRegistry;

namespace {

// ── The released checkpoint's own configs, verbatim ────────────────────────
constexpr const char* kModularIndex = R"({
  "_blocks_class_name": "MiniMaxMusic3Blocks",
  "_class_name": "MiniMaxMusic3ModularPipeline",
  "_diffusers_version": "0.40.0.dev0"
})";
constexpr const char* kConditionEncoderConfig =
    R"({"_class_name": "MiniMaxMusic3ConditionEncoder", "condition_hidden_dim": 4096,
        "input_hop_length": 960, "input_sampling_rate": 24000, "num_condition_layers": 8,
        "out_dim": 2048, "output_hop_length": 512, "output_sampling_rate": 44100})";
constexpr const char* kVocoderConfig =
    R"({"_class_name": "MiniMaxMusic3Vocoder", "decoder_hidden_dim": 1536,
        "decoder_input_dim": 1024, "latent_channels": 128, "sampling_rate": 44100,
        "upsampling_ratios": [8, 8, 4, 2]})";
constexpr const char* kSchedulerConfig =
    R"({"_class_name": "FlowMatchEulerDiscreteScheduler", "invert_sigmas": true,
        "num_train_timesteps": 1, "shift": 1.0, "shift_terminal": null,
        "stochastic_sampling": false, "time_shift_type": "exponential",
        "use_dynamic_shifting": false})";
constexpr const char* kRvqConfig =
    R"({"_class_name": "MiniMaxMusic3RVQDepthDecoder", "audio_vocab_size": 1024,
        "hidden_size": 4096, "intermediate_size": 6144, "max_position_embeddings": 16,
        "num_attention_heads": 16, "num_codebooks": 8, "num_layers": 4})";
constexpr const char* kTransformerConfig =
    R"({"_class_name": "MiniMaxMusic3Transformer1DModel", "attention_head_dim": 64,
        "condition_dim": 2048, "ff_inner_dim": 8192, "fourier_embedding_dim": 256,
        "in_channels": 128, "num_attention_heads": 32, "num_layers": 36, "rotary_dim": 32})";
constexpr const char* kLanguageModelConfig =
    R"({"architectures": ["Qwen3ForCausalLM"], "dtype": "bfloat16", "head_dim": 128,
        "hidden_size": 4096, "intermediate_size": 12288, "max_position_embeddings": 10240,
        "model_type": "qwen3", "num_attention_heads": 32, "num_hidden_layers": 36,
        "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "tie_word_embeddings": false, "vocab_size": 200000})";

void WriteFile(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  REQUIRE(out.good());
  out << text;
}

// A unique scratch directory. Named per test so two cases cannot collide, and
// removed by the caller.
fs::path Scratch(const char* tag) {
  const fs::path dir = fs::temp_directory_path() / ("vllm_music3_speech_" + std::string(tag));
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir);
  return dir;
}

// The diffusers arm, configs only. Zero-byte shard placeholders exist because
// `MiniMaxMusic3ResolveCheckpoint` requires the shard to be PRESENT; nothing
// here reads a weight.
void WriteSyntheticCheckpoint(const fs::path& root) {
  WriteFile(root / "modular_model_index.json", kModularIndex);
  WriteFile(root / "condition_encoder" / "config.json", kConditionEncoderConfig);
  WriteFile(root / "condition_encoder" / "diffusion_pytorch_model.safetensors", "");
  WriteFile(root / "vocoder" / "config.json", kVocoderConfig);
  WriteFile(root / "vocoder" / "diffusion_pytorch_model.safetensors", "");
  WriteFile(root / "rvq_depth_decoder" / "config.json", kRvqConfig);
  WriteFile(root / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors", "");
  WriteFile(root / "transformer" / "config.json", kTransformerConfig);
  WriteFile(root / "transformer" / "diffusion_pytorch_model.safetensors", "");
  WriteFile(root / "language_model" / "config.json", kLanguageModelConfig);
  WriteFile(root / "language_model" / "model.safetensors", "");
  WriteFile(root / "scheduler" / "scheduler_config.json", kSchedulerConfig);
  WriteFile(root / "tokenizer" / "tokenizer_config.json", "{}");
}

SpeechGenParams ValidRequest() {
  SpeechGenParams params;
  params.lyrics = "[Verse]\nMorning light filtering through the pine\n";
  params.description = "Genre: acoustic pop. BPM: 96. Key: C major.";
  return params;
}

vllm::MiniMaxMusic3ConditionEncoderConfig ShippedConditionConfig() {
  return vllm::MiniMaxMusic3ConditionEncoderConfig{};
}

}  // namespace

// ---------------------------------------------------------------------------
// The seam extension: the new fields are INERT at their defaults
// ---------------------------------------------------------------------------

TEST_CASE("music3 speech: SpeechGenParams grew additively and the growth is inert") {
  const SpeechGenParams fresh;
  // The pre-extension fields, unchanged. A caller written against them gets
  // exactly what it got before.
  CHECK(fresh.text.empty());
  CHECK(fresh.language.empty());
  CHECK(fresh.reference_audio.empty());
  CHECK(fresh.reference_sample_rate == 0);
  CHECK(fresh.seed == 0);
  // The growth. Every default means "the family decides", so no family that
  // ignores them can be steered by them.
  CHECK(fresh.lyrics.empty());
  CHECK(fresh.description.empty());
  CHECK(fresh.audio_duration_s == doctest::Approx(0.0));
  CHECK(fresh.num_inference_steps == 0);
  // NEGATIVE, not zero: 0 is a legal guidance scale, so a 0-means-default
  // sentinel would make the unconditional branch unreachable.
  CHECK(fresh.guidance_scale < 0.0);
}

TEST_CASE("music3 speech: IndexTTS-2.5 is UNCHANGED by the growth, and its `true` still holds") {
  // The seam grew; the OTHER family's behaviour did not. The growth is five
  // fields IndexTTS-2.5 never reads, so a field it ignores costs it nothing —
  // and this case is what makes that a checked property rather than a claim.
  SpeechRegistry registry;
  vllm::models::RegisterIndexTts2SpeechFamily(registry);
  REQUIRE(registry.families().size() == 1);
  CHECK(registry.families()[0] == "indextts2");

  const fs::path root = Scratch("indextts_untouched");
  WriteFile(root / "config.yaml", "gpt:\n  x: 1\ns2mel_checkpoint: a\nsemantic_codec: b\n");
  SpeechModelParams params;
  params.path = root.string();
  std::string why;
  std::unique_ptr<SpeechEngine> indextts = registry.Load(params, &why);
  REQUIRE_MESSAGE(indextts != nullptr, why);
  CHECK(indextts->family() == "indextts2");
  // ITS OWN native rate, which is NOT Music3's — a pass here proves the two
  // families answer independently rather than sharing a default.
  CHECK(indextts->sample_rate() == 22050);
  // TRUE. IndexTTS-2 has no text-only synthesis, so an empty clip is a refusal
  // rather than a default voice. This is the value that DIFFERS from Music3's,
  // and it is the decision a server consults before staging any weights.
  CHECK(indextts->requires_reference_audio());

  // The growth REACHES it and changes nothing: a request carrying every new
  // field, all at values that differ from their defaults, still hits the same
  // reference-clip refusal it hit before the fields existed.
  SpeechGenParams grown;
  grown.text = "hello there";
  grown.lyrics = "[Verse]\nignored by this family\n";
  grown.description = "Genre: acoustic pop";
  grown.audio_duration_s = 12.5;
  grown.num_inference_steps = 4;
  grown.guidance_scale = 2.75;
  CHECK_THROWS_WITH_AS(indextts->Synthesize(grown),
                       doctest::Contains("a reference clip is REQUIRED"),
                       std::runtime_error);

  // And it does NOT claim a Music3 tree, which is what would silently mis-load.
  const fs::path music3 = Scratch("indextts_not_music3");
  WriteSyntheticCheckpoint(music3);
  SpeechModelParams other;
  other.path = music3.string();
  CHECK(registry.Load(other, &why) == nullptr);
  CHECK(why.find("indextts2") != std::string::npos);
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::remove_all(music3, ec);
}

// ---------------------------------------------------------------------------
// Detection INSPECTS the artifact
// ---------------------------------------------------------------------------

TEST_CASE("music3 speech: detection claims the diffusers arm and nothing else") {
  const fs::path root = Scratch("detect");
  WriteSyntheticCheckpoint(root);
  CHECK(m3::Music3DetectCheckpoint(root.string()));

  SUBCASE("a missing modular index is not claimed") {
    std::error_code ec;
    fs::remove(root / "modular_model_index.json", ec);
    CHECK_FALSE(m3::Music3DetectCheckpoint(root.string()));
  }
  SUBCASE("an index naming a DIFFERENT pipeline class is not claimed") {
    // The directory name and the file name are identical; only the class moved.
    WriteFile(root / "modular_model_index.json",
              R"({"_class_name": "MiniMaxH3ModularPipeline"})");
    CHECK_FALSE(m3::Music3DetectCheckpoint(root.string()));
  }
  SUBCASE("a truncated download missing one component is not claimed") {
    std::error_code ec;
    fs::remove_all(root / "vocoder", ec);
    CHECK_FALSE(m3::Music3DetectCheckpoint(root.string()));
  }
  SUBCASE("the NATIVE arm is not claimed, so W1's by-name refusal keeps its message") {
    const fs::path native = Scratch("detect_native");
    WriteFile(native / "flowmatching_vae.pth", "");
    WriteFile(native / "dav.pth", "");
    fs::create_directories(native / "qwen_7B" / "qwen_7B");
    CHECK(vllm::MiniMaxMusic3IsNativeArm(native.string()));
    CHECK_FALSE(m3::Music3DetectCheckpoint(native.string()));
    std::error_code ec;
    fs::remove_all(native, ec);
  }
  SUBCASE("a detector NEVER throws, whatever it is pointed at") {
    CHECK_FALSE(m3::Music3DetectCheckpoint(""));
    CHECK_FALSE(m3::Music3DetectCheckpoint("/nonexistent/path/for/music3"));
    CHECK_FALSE(m3::Music3DetectCheckpoint("/etc/hostname"));  // a file, not a directory
  }
  std::error_code ec;
  fs::remove_all(root, ec);
}

// ---------------------------------------------------------------------------
// The engine's declared contract
// ---------------------------------------------------------------------------

TEST_CASE("music3 speech: the loaded engine declares 44100 stereo and needs NO reference clip") {
  const fs::path root = Scratch("engine");
  WriteSyntheticCheckpoint(root);
  SpeechRegistry registry;
  m3::RegisterMiniMaxMusic3SpeechFamily(registry);
  SpeechModelParams params;
  params.path = root.string();
  std::string why;
  std::unique_ptr<SpeechEngine> engine = registry.Load(params, &why);
  REQUIRE_MESSAGE(engine != nullptr, why);

  CHECK(engine->family() == "minimax-music3");
  // spec §1.1: the vocoder's NATIVE rate, resample-free. 22050 is IndexTTS-2.5's
  // and 32000 is SGLang-Omni's delivery rate; neither is this.
  CHECK(engine->sample_rate() == 44100);
  // FALSE — and it is the value that DIFFERS from the seam's other family, so a
  // pass here proves the override was reached rather than a base default.
  CHECK_FALSE(engine->requires_reference_audio());

  std::error_code ec;
  fs::remove_all(root, ec);
}

// W6 REFUSED here by name: the 8.6B `Qwen3ForCausalLM` forward it needed had no
// `inputs_embeds` entry on the landed dense path. That entry now exists
// (`Qwen3DenseModel::ForwardEmbeds`, qwen3.h) and the loop that drives it is
// `Music3GenerateFrameHiddens` (minimax_music3_llm.h), so this case is INVERTED:
// what it proves is that a valid request no longer stops at a missing stage.
//
// A synthetic checkpoint has valid CONFIGS and empty weight files, so the
// request runs the whole contract and then fails ON THE ARTIFACT — naming the
// file it could not read. That is exactly the boundary this case exists to pin:
// the message must be about these bytes, and must NOT be about an unimplemented
// forward. Whether the pipeline actually produces a song is a question only the
// real 28.5 GB checkpoint can answer, and
// tests/parity/test_minimax_music3_e2e_real.cpp is where it is asked.
TEST_CASE("music3 speech: the AR head RUNS, and a valid request reaches the weights") {
  const fs::path root = Scratch("ar_owed");
  WriteSyntheticCheckpoint(root);
  SpeechRegistry registry;
  m3::RegisterMiniMaxMusic3SpeechFamily(registry);
  SpeechModelParams params;
  params.path = root.string();
  std::string why;
  std::unique_ptr<SpeechEngine> engine = registry.Load(params, &why);
  REQUIRE_MESSAGE(engine != nullptr, why);

  std::string message;
  CHECK_THROWS_AS(
      [&] {
        try {
          engine->Synthesize(ValidRequest());
        } catch (const std::runtime_error& e) {
          message = e.what();
          throw;
        }
      }(),
      std::runtime_error);
  MESSAGE("a valid request against a synthetic checkpoint stops at: " << message);

  // It reached the LANGUAGE MODEL's own weight file — the first thing the AR
  // head touches, and the thing this checkpoint does not really have.
  CHECK(message.find("language_model") != std::string::npos);
  // And it is NOT the by-name refusal W6 shipped. Both spellings are asserted
  // absent because either one surviving would mean the stage is still owed;
  // "not found in this message" is the whole claim, and the message is printed
  // above so the reader can check it rather than take it on trust.
  CHECK(message.find("inputs_embeds") == std::string::npos);
  CHECK(message.find("not implemented") == std::string::npos);

  std::error_code ec;
  fs::remove_all(root, ec);
}

// ---------------------------------------------------------------------------
// The request contract
// ---------------------------------------------------------------------------

TEST_CASE("music3 speech: the request resolves UPSTREAM's defaults, and explicit values win") {
  const vllm::MiniMaxMusic3ConditionEncoderConfig config = ShippedConditionConfig();
  CHECK(m3::Music3FrameRate(config) == doctest::Approx(25.0));  // 24000 / 960

  const m3::Music3Request defaults = m3::Music3ResolveRequest(ValidRequest(), config);
  // encoders.py:253, denoise.py:144/:190, denoise.py:180 — mirrored, not chosen.
  CHECK(defaults.audio_duration_s == doctest::Approx(60.0));
  CHECK(defaults.num_inference_steps == 30);
  CHECK(defaults.guidance_scale == doctest::Approx(1.7));
  CHECK(defaults.max_frames == 1500);  // 60 s x 25 Hz
  CHECK(defaults.seed == 0);
  // The assembled prompt is upstream's template, both normalizers applied.
  CHECK(defaults.prompt.find("<|caption_start|>") != std::string::npos);
  CHECK(defaults.prompt.find("<|lyrics_start|>") != std::string::npos);
  CHECK(defaults.prompt.find("[verse]") != std::string::npos);  // lower-cased tag
  CHECK(defaults.prompt.find("[Verse]") == std::string::npos);

  // Every value below DIFFERS from the default it overrides, so a pass proves
  // the field was read rather than that a default happened to match.
  SpeechGenParams explicit_params = ValidRequest();
  explicit_params.audio_duration_s = 12.5;      // default 60.0
  explicit_params.num_inference_steps = 4;      // default 30
  explicit_params.guidance_scale = 2.75;        // default 1.7
  explicit_params.seed = 7;                     // default 0
  const m3::Music3Request resolved = m3::Music3ResolveRequest(explicit_params, config);
  CHECK(resolved.audio_duration_s == doctest::Approx(12.5));
  CHECK(resolved.num_inference_steps == 4);
  CHECK(resolved.guidance_scale == doctest::Approx(2.75));
  CHECK(resolved.max_frames == 312);  // int(12.5 * 25) = 312, truncated not rounded
  CHECK(resolved.seed == 7);

  // Guidance 0 is LEGAL (it selects the unconditional branch) and must survive
  // the default resolution — the whole reason the sentinel is negative.
  SpeechGenParams zero_guidance = ValidRequest();
  zero_guidance.guidance_scale = 0.0;
  CHECK(m3::Music3ResolveRequest(zero_guidance, config).guidance_scale == doctest::Approx(0.0));
}

TEST_CASE("music3 speech: the four field refusals each NAME the field") {
  const vllm::MiniMaxMusic3ConditionEncoderConfig config = ShippedConditionConfig();

  SUBCASE("`text` is refused rather than silently dropped") {
    SpeechGenParams params = ValidRequest();
    params.text = "sing something nice";
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("`text` is not this family's input"),
                         std::runtime_error);
  }
  SUBCASE("empty `lyrics` is refused") {
    SpeechGenParams params = ValidRequest();
    params.lyrics.clear();
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("`lyrics` is required"), std::runtime_error);
  }
  SUBCASE("a reference clip is refused, and the message says why there is none") {
    SpeechGenParams params = ValidRequest();
    params.reference_audio.assign(16, 0.25f);
    params.reference_sample_rate = 22050;
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("requires_reference_audio() is false"),
                         std::runtime_error);
  }
  SUBCASE("`language` is refused rather than honoured or ignored") {
    SpeechGenParams params = ValidRequest();
    params.language = "en";
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("`language` is not supported"), std::runtime_error);
  }
  SUBCASE("an empty description is upstream's own refusal") {
    SpeechGenParams params = ValidRequest();
    params.description.clear();
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("music description"), std::runtime_error);
  }
  SUBCASE("upstream's two duration errors survive") {
    SpeechGenParams params = ValidRequest();
    // NEGATIVE is an explicit impossible value, not "omitted". Defaulting it to
    // 60 s would answer an invalid request with a full-length song.
    params.audio_duration_s = -1.0;
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("must be positive"), std::runtime_error);
    SpeechGenParams steps = ValidRequest();
    steps.num_inference_steps = -4;
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(steps, config),
                         doctest::Contains("`num_inference_steps` must be positive"),
                         std::runtime_error);
    // Shorter than ONE frame at 25 Hz: 0.02 s truncates to 0 frames.
    params.audio_duration_s = 0.02;
    CHECK_THROWS_WITH_AS(m3::Music3ResolveRequest(params, config),
                         doctest::Contains("shorter than one audio frame"), std::runtime_error);
  }
  SUBCASE("the 9000-frame ceiling is ENFORCED, not discovered") {
    SpeechGenParams params = ValidRequest();
    params.audio_duration_s = 600.0;  // 15000 frames at 25 Hz
    CHECK(m3::Music3ResolveRequest(params, config).max_frames == 9000);
  }
}

// ---------------------------------------------------------------------------
// The chunk plan (before_denoise.py)
// ---------------------------------------------------------------------------

TEST_CASE("music3 speech: the chunk plan clamps the last window and hops by 100") {
  m3::ConditionMixConfig mix;  // the shipped rates: 25 Hz frames, 86.133 Hz latents

  SUBCASE("a short song is ONE window") {
    const std::vector<m3::Music3Chunk> plan = m3::Music3ChunkPlan(25, mix);
    REQUIRE(plan.size() == 1);
    CHECK(plan[0].frame_start == 0);
    CHECK(plan[0].frame_end == 25);
    // The oracle capture's own latent length for 25 frames (manifest.json).
    CHECK(plan[0].latent_length == 86);
  }
  SUBCASE("exactly 200 frames is still ONE window, not two") {
    const std::vector<m3::Music3Chunk> plan = m3::Music3ChunkPlan(200, mix);
    REQUIRE(plan.size() == 1);
    CHECK(plan[0].frame_end == 200);
    CHECK(plan[0].latent_length == 689);
  }
  SUBCASE("201 frames is TWO windows and the second is CLAMPED") {
    // `range(0, num_frames - hop, hop)` stops at num_frames - hop, so 201 gives
    // starts {0, 100} and the second window covers frames [100, 201) — 101
    // frames, not 200. Reading the length as always 200 runs off the end.
    const std::vector<m3::Music3Chunk> plan = m3::Music3ChunkPlan(201, mix);
    REQUIRE(plan.size() == 2);
    CHECK(plan[0].frame_start == 0);
    CHECK(plan[0].frame_end == 200);
    CHECK(plan[1].frame_start == 100);
    CHECK(plan[1].frame_end == 201);
    CHECK(plan[1].frames() == 101);
    // 101 * (44100/24000) * (960/512) = 347.976..., TRUNCATED once at the end.
    // Computing it as integer ratios rounds this to 348 for the wrong reason.
    CHECK(plan[1].latent_length == 347);
  }
  SUBCASE("a long song hops by 100 and every window but the last is full") {
    const std::vector<m3::Music3Chunk> plan = m3::Music3ChunkPlan(450, mix);
    REQUIRE(plan.size() == 4);  // starts 0, 100, 200, 300
    int64_t full = 0;
    for (size_t k = 0; k < plan.size(); ++k) {
      CHECK(plan[k].frame_start == static_cast<int64_t>(k) * 100);
      if (plan[k].frames() == 200) ++full;
    }
    CHECK(full == 3);
    CHECK(plan.back().frame_end == 450);
    MESSAGE("chunk plan: " << plan.size() << " windows examined, " << full << " full");
  }
  CHECK_THROWS_AS(m3::Music3ChunkPlan(0, mix), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Registration is ADDITIVE
// ---------------------------------------------------------------------------

TEST_CASE("music3 speech: registering the family adds one name and changes no other") {
  SpeechRegistry empty;
  const fs::path root = Scratch("additive");
  WriteSyntheticCheckpoint(root);
  SpeechModelParams params;
  params.path = root.string();

  // BEFORE: the seam's own "no family is registered" refusal, word for word.
  std::string before;
  CHECK(empty.Load(params, &before) == nullptr);
  CHECK(before.find("no speech (TTS) family is registered") != std::string::npos);
  CHECK(before.find("minimax-music3") == std::string::npos);

  SpeechRegistry both;
  m3::RegisterBuiltinSpeechFamilies(both);
  const std::vector<std::string> names = both.families();
  REQUIRE(names.size() == 2);
  CHECK(names[0] == "indextts2");
  CHECK(names[1] == "minimax-music3");
  MESSAGE("registered speech families: " << names.size());

  // AFTER: the Music3 tree resolves to the Music3 engine, and the IndexTTS-2.5
  // detector standing in front of it did not claim it.
  std::string why;
  std::unique_ptr<SpeechEngine> engine = both.Load(params, &why);
  REQUIRE_MESSAGE(engine != nullptr, why);
  CHECK(engine->family() == "minimax-music3");

  // Registering twice is a no-op, not the registry's duplicate-name throw.
  CHECK_NOTHROW(m3::RegisterBuiltinSpeechFamilies(both));
  CHECK(both.families().size() == 2);
  // But the raw registration is still strict, which is the guarantee that keeps
  // two families from collapsing into one listed name.
  CHECK_THROWS_AS(m3::RegisterMiniMaxMusic3SpeechFamily(both), std::runtime_error);

  std::error_code ec;
  fs::remove_all(root, ec);
}
