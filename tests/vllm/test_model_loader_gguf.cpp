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
      "'Qwen3_5ForConditionalGeneration', "
      "'Qwen3_5MoeForConditionalGeneration', 'StableLmForCausalLM'])",
      std::runtime_error);
}
