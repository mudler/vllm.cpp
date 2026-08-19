// M0.10 Task 2: LoadedEngine::FromModelDir routes a `.gguf` path to the GGUF
// loader (not the safetensors directory path). A full end-to-end engine build
// from a real GGUF is dgx-pending; here we prove the branch selection: a
// `.gguf` file is opened as GGUF (a bad-magic .gguf surfaces the reader's
// "magic" error, which the directory path can never produce), and a non-.gguf
// non-directory path still takes the directory branch.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "gguf_builder.h"
#include "vllm/entrypoints/model_loader.h"

using gguf_test::TempFile;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// The smallest GGUF that reaches the entrypoint's architecture dispatch: a
// valid v3 header plus the ONE kv the dispatch reads. No tensors and no vocab
// on purpose — the refusal has to fire before any tokenizer or weight I/O, and
// a fixture that carries them could not tell a refusal at the dispatch apart
// from one further down.
std::string GgufWithArchitecture(const std::string& arch) {
  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", arch));
  return builder.Build();
}

// What FromModelDir threw, as text. CHECK_THROWS_WITH_AS proves a substring is
// PRESENT; the point of #809 is also what must be ABSENT, and only the caught
// message can answer that.
std::string RefusalFor(const std::string& gguf_bytes) {
  TempFile file(gguf_bytes);
  try {
    LoadedEngine::FromModelDir(file.path(), EngineParams{});
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

class TempDir {
 public:
  TempDir() {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllm_model_registry_reject_" + std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() { std::filesystem::remove_all(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST_CASE("FromModelDir routes .gguf files to the GGUF reader") {
  // A .gguf file with a corrupt magic: the GGUF branch is taken, so the
  // reader's "magic" error surfaces (the directory branch cannot produce it).
  TempFile bad(std::string("GGML") + std::string(60, '\0'));
  CHECK_THROWS_WITH_AS(LoadedEngine::FromModelDir(bad.path(), EngineParams{}),
                       doctest::Contains("magic"), std::runtime_error);
}

TEST_CASE("FromModelDir keeps the directory path for non-.gguf inputs") {
  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir("/nonexistent/model/dir", EngineParams{}),
      doctest::Contains("not a directory"), std::runtime_error);
}

// #809. The GGUF architecture dispatch used to have no default: after the
// `deepseek4` and `muse-glimmer` arms it FELL THROUGH to `vllm::HfConfigFromGguf`
// — qwen3_5's config builder, which hard-asserts its own three keys
// (qwen3_5_gguf_weights.cpp). So every GGUF outside those five names was refused
// with "qwen3_5 gguf: unexpected architecture", naming a model unrelated to the
// file the user passed and sending the reader into an unrelated translation
// unit. These cases drive `LoadedEngine::FromModelDir` — the REAL path a `.gguf`
// argument takes — not a hand-built ModelSource.
TEST_CASE("An unsupported GGUF architecture is refused BY NAME, not qwen3_5's") {
  const std::string message = RefusalFor(GgufWithArchitecture("mamba-unported"));
  REQUIRE_FALSE(message.empty());

  // It names the file's OWN architecture...
  CHECK(message.find("mamba-unported") != std::string::npos);
  // ...and the set this build actually supports, so the reader can see whether
  // the file is convertible to something loadable rather than guessing.
  CHECK(message.find("deepseek4") != std::string::npos);
  CHECK(message.find("muse-glimmer") != std::string::npos);
  CHECK(message.find("qwen35") != std::string::npos);
  CHECK(message.find("qwen35moe") != std::string::npos);
  CHECK(message.find("qwen3next") != std::string::npos);
  // And it does NOT hand the reader one model's parser as the explanation.
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("A GGUF with no general.architecture says so, by name") {
  // The old fall-through reported this as qwen3_5's "general.architecture must
  // be a string" too. There is no architecture to select, and the message that
  // says that must not be a specific model's.
  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.name", "no-arch-key"));
  const std::string message = RefusalFor(builder.Build());
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("general.architecture") != std::string::npos);
  CHECK(message.find("deepseek4") != std::string::npos);
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("A supported GGUF architecture still reaches its own builder") {
  // The guard on the change: adding an explicit default must not have moved a
  // SUPPORTED key onto the refusal path. A `qwen35` file with none of the
  // geometry keys reaches qwen3_5's builder and fails on the FIRST MISSING KEY,
  // which only that builder can report — so this proves the arm was taken, not
  // merely that something threw.
  const std::string message = RefusalFor(GgufWithArchitecture("qwen35"));
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("qwen3_5 gguf: missing metadata key") != std::string::npos);
  CHECK(message.find("is not supported by this build") == std::string::npos);
}

TEST_CASE("FromModelDir rejects an unknown dense architecture before loading") {
  // The rejection must fire during architecture resolution, BEFORE any tokenizer
  // or weight I/O — so the arch must be one the registry does NOT know. (Note:
  // LlamaForCausalLM, which this case originally used, is now a SUPPORTED arch,
  // so it would sail past resolve and fail later on a missing tokenizer instead.)
  // Gemma4ForCausalLM is a real-but-still-unported Gemma arch: it is absent from
  // the registry, so Resolve throws the "not supported" error up front.
  TempDir dir;
  nlohmann::json config{
      {"model_type", "gemma"},
      {"architectures", nlohmann::json::array({"Gemma4ForCausalLM"})},
      {"hidden_size", 8},
      {"num_hidden_layers", 1},
      {"num_attention_heads", 1},
      {"vocab_size", 8},
  };
  std::ofstream(dir.path() / "config.json") << config.dump();

  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir(dir.path().string(), EngineParams{}),
      "Model architectures ['Gemma4ForCausalLM'] are not supported for now. "
      "Supported architectures: "
      "dict_keys(['CohereForCausalLM', 'DeepseekV2ForCausalLM', "
      "'DeepseekV4ForCausalLM', "
      "'Gemma2ForCausalLM', 'Gemma3ForCausalLM', "
      "'Gemma4ForConditionalGeneration', 'Gemma4UnifiedForConditionalGeneration', 'GemmaForCausalLM', "
      "'Glm4ForCausalLM', 'Glm4MoeLiteForCausalLM', 'GraniteForCausalLM', "
      "'InternLM2ForCausalLM', 'InternLM3ForCausalLM', "
      "'KimiK3ForConditionalGeneration', 'KimiLinearForCausalLM', "
      "'LagunaForCausalLM', "
      "'LlamaForCausalLM', 'LlamaModel', "
      "'MiniCPM3ForCausalLM', 'MiniCPMForCausalLM', 'MistralForCausalLM', 'MuseGlimmerForCausalLM', 'MuseGlimmerForConditionalGeneration', "
      "'NemotronHForCausalLM', "
      "'OPTForCausalLM', 'Olmo2ForCausalLM', 'Olmo3ForCausalLM', "
      "'ParakeetForCTC', 'ParakeetForRNNT', 'ParakeetForTDT', "
      "'Phi3ForCausalLM', 'PhiForCausalLM', 'Qwen3ForCausalLM', "
      "'Qwen3MoeForCausalLM', 'Qwen3VLForConditionalGeneration', "
      "'Qwen3_5ForCausalLM', 'Qwen3_5ForConditionalGeneration', "
      "'Qwen3_5MoeForCausalLM', "
      "'Qwen3_5MoeForConditionalGeneration', 'StableLmForCausalLM'])",
      std::runtime_error);
}
