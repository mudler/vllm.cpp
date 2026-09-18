// PEFTHelper tests.
//
// UPSTREAM tests re-expressed (${VLLM_SOURCE} @ 555967922):
//   tests/lora/test_peft_helper.py:30-72   test_peft_helper_pass
//   tests/lora/test_peft_helper.py:75-101  test_peft_helper_error (parametrized)
//   tests/lora/test_peft_helper.py:104-116 test_peft_helper_invalid_rank_direct
//
// We write synthetic adapter_config.json files to temp directories instead of
// relying on the upstream fixture (llama32_lora_files), which is a downloaded
// checkpoint. The config content mirrors the upstream fixture's values.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include "vllm/lora/peft_helper.h"

using vllm::lora::PEFTHelper;

namespace {

// RAII temp directory.
class TempDir {
 public:
  TempDir() {
    auto base = std::filesystem::temp_directory_path();
    std::string name = "vllm_peft_test_" +
                       std::to_string(reinterpret_cast<uintptr_t>(this));
    path_ = base / name;
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string path() const { return path_.string(); }

  void WriteConfig(const nlohmann::json& config) {
    std::ofstream f(path_ / "adapter_config.json");
    f << config.dump();
    f.close();
  }

  void WriteConfig(const std::string& json_str) {
    std::ofstream f(path_ / "adapter_config.json");
    f << json_str;
    f.close();
  }

 private:
  std::filesystem::path path_;
};

// A valid config mirroring the upstream llama32 fixture.
nlohmann::json ValidConfig() {
  return {
      {"r", 8},
      {"lora_alpha", 32},
      {"target_modules",
       nlohmann::json::array({"q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",
                              "embed_tokens", "lm_head"})},
      {"bias", "none"},
      {"modules_to_save", nullptr},
      {"use_rslora", false},
      {"use_dora", false},
  };
}

}  // namespace

// test_peft_helper_pass (test_peft_helper.py:30-51).
TEST_CASE("peft_helper: valid config loads and validates") {
  TempDir dir;
  dir.WriteConfig(ValidConfig());

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  peft.ValidateLegal(16);

  CHECK(peft.r == 8);
  CHECK(peft.lora_alpha == 32);

  // target_modules sorted should match the upstream expected list.
  std::vector<std::string> expected = {
      "down_proj", "embed_tokens", "gate_proj", "k_proj",
      "lm_head", "o_proj", "q_proj", "up_proj", "v_proj"};
  std::vector<std::string> sorted_tm = peft.target_modules;
  std::sort(sorted_tm.begin(), sorted_tm.end());
  CHECK(sorted_tm == expected);

  CHECK(peft.vllm_max_position_embeddings.value() == 4096);

  // Scaling = alpha / r = 32 / 8 = 4.0.
  CHECK(peft.vllm_lora_scaling_factor == doctest::Approx(4.0));
}

// test_peft_helper_pass rsLoRA section (test_peft_helper.py:53-72).
TEST_CASE("peft_helper: rsLoRA scaling factor") {
  TempDir dir;
  auto config = ValidConfig();
  config["use_rslora"] = true;
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  peft.ValidateLegal(16);

  double expected = 32.0 / std::sqrt(8.0);
  CHECK(std::abs(peft.vllm_lora_scaling_factor - expected) < 1e-3);
}

// test_peft_helper_error parametrized (test_peft_helper.py:75-101).
TEST_CASE("peft_helper: rank exceeds max_lora_rank") {
  TempDir dir;
  auto config = ValidConfig();
  config["r"] = 1024;
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK_THROWS_AS(peft.ValidateLegal(16), std::invalid_argument);
}

TEST_CASE("peft_helper: DoRA unsupported") {
  TempDir dir;
  auto config = ValidConfig();
  config["use_dora"] = true;
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK_THROWS_AS(peft.ValidateLegal(16), std::invalid_argument);
}

TEST_CASE("peft_helper: modules_to_save unsupported") {
  TempDir dir;
  auto config = ValidConfig();
  config["modules_to_save"] = nlohmann::json::array({"lm_head"});
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK_THROWS_AS(peft.ValidateLegal(16), std::invalid_argument);
}

TEST_CASE("peft_helper: rank zero rejected at construction") {
  TempDir dir;
  auto config = ValidConfig();
  config["r"] = 0;
  dir.WriteConfig(config);

  CHECK_THROWS_AS(PEFTHelper::FromLocalDir(dir.path(), 4096),
                  std::invalid_argument);
}

TEST_CASE("peft_helper: negative rank rejected at construction") {
  TempDir dir;
  auto config = ValidConfig();
  config["r"] = -8;
  dir.WriteConfig(config);

  CHECK_THROWS_AS(PEFTHelper::FromLocalDir(dir.path(), 4096),
                  std::invalid_argument);
}

// test_peft_helper_invalid_rank_direct (test_peft_helper.py:104-116).
TEST_CASE("peft_helper: direct construction with non-positive rank throws") {
  CHECK_THROWS_AS(PEFTHelper(0, 16, {"q_proj"}), std::invalid_argument);
  CHECK_THROWS_AS(PEFTHelper(-1, 16, {"q_proj"}), std::invalid_argument);
  CHECK_THROWS_AS(PEFTHelper(-8, 16, {"q_proj"}), std::invalid_argument);
}

// Extra: bias != "none" is rejected by ValidateLegal.
TEST_CASE("peft_helper: bias not none rejected") {
  TempDir dir;
  auto config = ValidConfig();
  config["bias"] = "all";
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK_THROWS_AS(peft.ValidateLegal(16), std::invalid_argument);
}

// Extra: target_modules as a string (not list) is accepted.
TEST_CASE("peft_helper: target_modules as string") {
  TempDir dir;
  auto config = ValidConfig();
  config["target_modules"] = "q_proj";
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK(peft.target_modules.size() == 1);
  CHECK(peft.target_modules[0] == "q_proj");
}

// Extra: unknown keys in the JSON are ignored (from_dict filtered_dict).
TEST_CASE("peft_helper: unknown keys ignored") {
  TempDir dir;
  auto config = ValidConfig();
  config["unknown_key"] = "ignored";
  config["peft_type"] = "LORA";
  dir.WriteConfig(config);

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK(peft.r == 8);
}

// Extra: missing required fields throws.
TEST_CASE("peft_helper: missing required field throws") {
  TempDir dir;
  // Missing "r".
  std::string json_str =
      R"({"lora_alpha":32,"target_modules":["q_proj"]})";
  dir.WriteConfig(json_str);

  CHECK_THROWS_AS(PEFTHelper::FromLocalDir(dir.path(), 4096),
                  std::invalid_argument);
}

// Extra: FromDict works directly with a JSON object.
TEST_CASE("peft_helper: from_dict with json object") {
  nlohmann::json config = ValidConfig();
  PEFTHelper peft = PEFTHelper::FromDict(config);
  CHECK(peft.r == 8);
  CHECK(peft.lora_alpha == 32);
  CHECK_FALSE(peft.use_rslora);
  CHECK_FALSE(peft.use_dora);
  CHECK_FALSE(peft.modules_to_save.has_value());
  CHECK_FALSE(peft.vllm_max_position_embeddings.has_value());
}

// Extra: max_position_embeddings nullopt is handled.
TEST_CASE("peft_helper: nullopt max_position_embeddings") {
  TempDir dir;
  dir.WriteConfig(ValidConfig());

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), std::nullopt);
  CHECK_FALSE(peft.vllm_max_position_embeddings.has_value());
}
