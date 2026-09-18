// LoRA checkpoint loading + name parsing tests.
//
// UPSTREAM tests re-expressed (${VLLM_SOURCE} @ 555967922):
//   tests/lora/test_lora_checkpoints.py:29-102  test_load_checkpoints
//   tests/lora/test_lora_checkpoints.py:105-135 test_lora_weights_mapping
//   tests/lora/test_lora_checkpoints.py:138-144 test_gemma4_lora_weights_mapping
//   tests/lora/test_lora_checkpoints.py:147-156 test_gemma4_moe_lora_weights_mapping
//
// We build synthetic adapter_model.safetensors files with known tensor data
// instead of relying on the upstream baichuan/chatglm3 fixtures (downloaded
// checkpoints). The safetensors byte format is assembled directly, matching
// the approach in tests/vllm/test_safetensors.cpp.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/lora/lora_model.h"
#include "vllm/lora/peft_helper.h"

using vllm::lora::IsBaseEmbeddingWeights;
using vllm::lora::LoRAModel;
using vllm::lora::ParseFineTunedLoraName;
using vllm::lora::PEFTHelper;
using vllm::lora::WeightsMapper;

namespace {

// --- Temp directory + file helpers ---

class TempDir {
 public:
  TempDir() {
    auto base = std::filesystem::temp_directory_path();
    std::string name = "vllm_lora_ckpt_" +
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

  void WriteFile(const std::string& name, const std::string& bytes) {
    std::ofstream f(path_ / name, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    f.close();
  }

  void WriteConfig(const nlohmann::json& config) {
    std::ofstream f(path_ / "adapter_config.json");
    f << config.dump();
    f.close();
  }

 private:
  std::filesystem::path path_;
};

// --- Safetensors assembly ---

// Little-endian u64 prefix.
std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i)
    s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

// Build a safetensors file from a list of (name, dtype, shape, data_bytes).
struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string data;
};

std::string MakeSafetensors(const std::vector<TensorSpec>& tensors) {
  // Assemble data section and build header simultaneously.
  std::string data;
  nlohmann::json header = nlohmann::json::object();

  for (const auto& t : tensors) {
    size_t offset = data.size();
    data += t.data;
    nlohmann::json entry = nlohmann::json::object();
    entry["dtype"] = t.dtype;
    entry["shape"] = t.shape;
    entry["data_offsets"] =
        nlohmann::json::array({offset, offset + t.data.size()});
    header[t.name] = entry;
  }

  std::string header_str = header.dump();
  return U64Le(header_str.size()) + header_str + data;
}

// F32 tensor data from a float vector.
std::string F32Data(const std::vector<float>& vals) {
  std::string s(vals.size() * 4, '\0');
  std::memcpy(s.data(), vals.data(), s.size());
  return s;
}

// BF16 tensor data from a float vector (truncate to top 16 bits).
std::string BF16Data(const std::vector<float>& vals) {
  std::string s(vals.size() * 2, '\0');
  for (size_t i = 0; i < vals.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &vals[i], 4);
    uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
    std::memcpy(&s[i * 2], &bf16, 2);
  }
  return s;
}

// A valid adapter_config.json with the given target_modules.
nlohmann::json MakeConfig(const std::vector<std::string>& target_modules) {
  return {
      {"r", 8},
      {"lora_alpha", 32},
      {"target_modules", target_modules},
      {"bias", "none"},
      {"modules_to_save", nullptr},
      {"use_rslora", false},
      {"use_dora", false},
  };
}

// Build a simple 2-module LoRA safetensors: q_proj and o_proj.
// Each module has lora_A [rank=8, input=16] and lora_B [output=16, rank=8].
std::string MakeSimpleLoraSafetensors() {
  constexpr int rank = 8;
  constexpr int dim = 16;
  constexpr int count = rank * dim;  // 128 elements per matrix

  std::vector<float> a_vals(count, 1.0f);
  std::vector<float> b_vals(count, 2.0f);

  return MakeSafetensors({
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
       "F32", {rank, dim}, F32Data(a_vals)},
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
       "F32", {dim, rank}, F32Data(b_vals)},
      {"base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight",
       "F32", {rank, dim}, F32Data(a_vals)},
      {"base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight",
       "F32", {dim, rank}, F32Data(b_vals)},
  });
}

}  // namespace

// ---------------------------------------------------------------------------
// parse_fine_tuned_lora_name (utils.py:155-207)
// ---------------------------------------------------------------------------

TEST_CASE("parse_fine_tuned_lora_name: lora_A weight") {
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight");
  CHECK(module == "model.layers.0.self_attn.q_proj");
  CHECK(is_a);
}

TEST_CASE("parse_fine_tuned_lora_name: lora_B weight") {
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight");
  CHECK(module == "model.layers.0.self_attn.q_proj");
  CHECK_FALSE(is_a);
}

TEST_CASE("parse_fine_tuned_lora_name: lora_embedding_A") {
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.embed_tokens.lora_embedding_A");
  CHECK(module == "model.embed_tokens");
  CHECK(is_a);
}

TEST_CASE("parse_fine_tuned_lora_name: lora_embedding_B") {
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.embed_tokens.lora_embedding_B");
  CHECK(module == "model.embed_tokens");
  CHECK_FALSE(is_a);
}

TEST_CASE("parse_fine_tuned_lora_name: no base_model prefix") {
  auto [module, is_a] =
      ParseFineTunedLoraName("model.layers.0.q_proj.lora_A.weight");
  CHECK(module == "model.layers.0.q_proj");
  CHECK(is_a);
}

TEST_CASE("parse_fine_tuned_lora_name: with weights_mapper prefix") {
  WeightsMapper mapper;
  mapper.orig_to_new_prefix = {{"model.", "language_model.model."}};
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.layers.0.q_proj.lora_A.weight", &mapper);
  CHECK(module == "language_model.model.layers.0.q_proj");
  CHECK(is_a);
}

// test_gemma4_lora_weights_mapping (test_lora_checkpoints.py:138-144).
TEST_CASE("parse_fine_tuned_lora_name: gemma4 prefix mapping") {
  WeightsMapper mapper;
  mapper.orig_to_new_prefix = {{"model.language_model.", "model."}};
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.language_model.layers.9.mlp.down_proj.lora_A.weight",
      &mapper);
  CHECK(module == "model.layers.9.mlp.down_proj");
  CHECK(is_a);
}

// test_gemma4_moe_lora_weights_mapping (test_lora_checkpoints.py:147-156).
TEST_CASE("parse_fine_tuned_lora_name: gemma4 moe mapping") {
  WeightsMapper mapper;
  mapper.orig_to_new_prefix = {{"model.language_model.", "model."}};
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.language_model.layers.9.moe.experts."
      "gate_up_proj.lora_B.weight",
      &mapper);
  CHECK(module == "model.layers.9.moe.experts.gate_up_proj");
  CHECK_FALSE(is_a);
}

// test_lora_weights_mapping substr (test_lora_checkpoints.py:105-135).
TEST_CASE("parse_fine_tuned_lora_name: substr mapping") {
  WeightsMapper mapper;
  mapper.orig_to_new_prefix = {{"model.", "language_model.model."}};
  mapper.orig_to_new_substr = {{".layers.", ".baichuan_layers."}};
  auto [module, is_a] = ParseFineTunedLoraName(
      "base_model.model.model.layers.0.self_attn.W_pack.lora_A.weight",
      &mapper);
  CHECK(module == "language_model.model.baichuan_layers.0.self_attn.W_pack");
  CHECK(is_a);
}

TEST_CASE("parse_fine_tuned_lora_name: unsupported name throws") {
  CHECK_THROWS_AS(
      ParseFineTunedLoraName("base_model.model.model.layers.0.weight"),
      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// is_base_embedding_weights (utils.py:210-216)
// ---------------------------------------------------------------------------

TEST_CASE("is_base_embedding_weights: embed_tokens") {
  CHECK(IsBaseEmbeddingWeights(
      "base_model.model.model.embed_tokens.base_layer.weight"));
}

TEST_CASE("is_base_embedding_weights: lm_head") {
  CHECK(IsBaseEmbeddingWeights(
      "base_model.model.lm_head.base_layer.weight"));
}

TEST_CASE("is_base_embedding_weights: lora weight is not base") {
  CHECK_FALSE(IsBaseEmbeddingWeights(
      "base_model.model.model.embed_tokens.lora_embedding_A"));
}

// ---------------------------------------------------------------------------
// from_local_checkpoint (lora_model.py:166-307)
// ---------------------------------------------------------------------------

// test_load_checkpoints pass case (test_lora_checkpoints.py:44-57).
TEST_CASE("from_local_checkpoint: valid safetensors loads") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"q_proj", "o_proj"}));
  dir.WriteFile("adapter_model.safetensors", MakeSimpleLoraSafetensors());

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  LoRAModel model = LoRAModel::FromLocalCheckpoint(
      dir.path(), {"q_proj", "o_proj"}, peft, 1);

  CHECK(model.id == 1);
  CHECK(model.rank == 8);
  CHECK(model.loras.size() == 2);

  // Check q_proj layer.
  auto* q = model.GetLora("model.layers.0.self_attn.q_proj");
  REQUIRE(q != nullptr);
  CHECK(q->rank == 8);
  CHECK(q->lora_alpha == 32);
  CHECK(q->input_dim == 16);
  CHECK(q->output_dim == 16);
  CHECK(q->scaling == doctest::Approx(4.0));  // 32/8
  CHECK(q->lora_a.size() == 128);  // rank * input_dim = 8*16
  CHECK(q->lora_b.size() == 128);  // output_dim * rank = 16*8
  CHECK(q->lora_a[0] == doctest::Approx(1.0f));
  CHECK(q->lora_b[0] == doctest::Approx(2.0f));

  // Check o_proj layer.
  auto* o = model.GetLora("model.layers.0.self_attn.o_proj");
  REQUIRE(o != nullptr);
  CHECK(o->input_dim == 16);
  CHECK(o->output_dim == 16);
}

// test_load_checkpoints fail case (test_lora_checkpoints.py:87-102).
TEST_CASE("from_local_checkpoint: unexpected modules throws") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"q_proj", "o_proj"}));
  // Write safetensors with modules that are NOT in expected.
  std::vector<float> a(128, 1.0f);
  std::vector<float> b(128, 2.0f);
  dir.WriteFile("adapter_model.safetensors",
                MakeSafetensors({
                    {"base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight",
                     "F32", {8, 16}, F32Data(a)},
                    {"base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight",
                     "F32", {16, 8}, F32Data(b)},
                }));

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK_THROWS_AS(
      LoRAModel::FromLocalCheckpoint(dir.path(), {"q_proj", "o_proj"}, peft, 1),
      std::invalid_argument);
}

// test_lora_weights_mapping (test_lora_checkpoints.py:105-135).
TEST_CASE("from_local_checkpoint: weights_mapper remaps module names") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"W_pack", "o_proj"}));
  dir.WriteFile("adapter_model.safetensors", MakeSafetensors({
      {"base_model.model.model.layers.0.self_attn.W_pack.lora_A.weight",
       "F32", {8, 16}, F32Data(std::vector<float>(128, 1.0f))},
      {"base_model.model.model.layers.0.self_attn.W_pack.lora_B.weight",
       "F32", {16, 8}, F32Data(std::vector<float>(128, 2.0f))},
  }));

  WeightsMapper mapper;
  mapper.orig_to_new_prefix = {{"model.", "language_model.model."}};
  mapper.orig_to_new_substr = {{".layers.", ".baichuan_layers."}};

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  LoRAModel model = LoRAModel::FromLocalCheckpoint(
      dir.path(), {"W_pack", "o_proj"}, peft, 1, std::nullopt, &mapper);

  // Every lora name should start with the mapped prefix and contain the
  // mapped substr.
  for (const auto& [name, _] : model.loras) {
    CHECK(name.find("language_model.model.") == 0);
    CHECK(name.find(".baichuan_layers.") != std::string::npos);
  }
}

// Extra: BF16 tensors are converted to F32.
TEST_CASE("from_local_checkpoint: BF16 tensors converted to F32") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"q_proj"}));

  std::vector<float> a_vals(128, 3.0f);
  std::vector<float> b_vals(128, 5.0f);

  dir.WriteFile("adapter_model.safetensors",
                MakeSafetensors({
                    {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                     "BF16", {8, 16}, BF16Data(a_vals)},
                    {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                     "BF16", {16, 8}, BF16Data(b_vals)},
                }));

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  LoRAModel model = LoRAModel::FromLocalCheckpoint(
      dir.path(), {"q_proj"}, peft, 1);

  auto* q = model.GetLora("model.layers.0.self_attn.q_proj");
  REQUIRE(q != nullptr);
  // BF16 3.0 and 5.0 convert exactly to F32 3.0 and 5.0.
  CHECK(q->lora_a[0] == doctest::Approx(3.0f));
  CHECK(q->lora_b[0] == doctest::Approx(5.0f));
  CHECK(q->lora_a.size() == 128);
  CHECK(q->lora_b.size() == 128);
}

// Extra: rsLoRA scaling flows from PEFTHelper into LoRALayerWeights.
TEST_CASE("from_local_checkpoint: rsLoRA scaling propagated") {
  TempDir dir;
  auto config = MakeConfig({"q_proj"});
  config["use_rslora"] = true;
  dir.WriteConfig(config);
  dir.WriteFile("adapter_model.safetensors", MakeSafetensors({
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
       "F32", {8, 16}, F32Data(std::vector<float>(128, 1.0f))},
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
       "F32", {16, 8}, F32Data(std::vector<float>(128, 2.0f))},
  }));

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  LoRAModel model = LoRAModel::FromLocalCheckpoint(
      dir.path(), {"q_proj"}, peft, 1);

  auto* q = model.GetLora("model.layers.0.self_attn.q_proj");
  REQUIRE(q != nullptr);
  double expected = 32.0 / std::sqrt(8.0);
  CHECK(q->scaling == doctest::Approx(expected));
}

// Extra: missing safetensors file throws.
TEST_CASE("from_local_checkpoint: missing file throws") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"q_proj"}));

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  CHECK_THROWS_AS(
      LoRAModel::FromLocalCheckpoint(dir.path(), {"q_proj"}, peft, 1),
      std::invalid_argument);
}

// Extra: check_lora_name and get_lora.
TEST_CASE("from_local_checkpoint: check_lora_name and get_lora") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"q_proj", "o_proj"}));
  dir.WriteFile("adapter_model.safetensors", MakeSimpleLoraSafetensors());

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  LoRAModel model = LoRAModel::FromLocalCheckpoint(
      dir.path(), {"q_proj", "o_proj"}, peft, 1);

  CHECK(model.CheckLoraName("model.layers.0.self_attn.q_proj"));
  CHECK(model.CheckLoraName("model.layers.0.self_attn.o_proj"));
  CHECK_FALSE(model.CheckLoraName("model.layers.0.mlp.gate_proj"));
  CHECK(model.GetLora("nonexistent") == nullptr);
}

// Extra: skip_prefixes skips modules.
TEST_CASE("from_local_checkpoint: skip_prefixes skips modules") {
  TempDir dir;
  dir.WriteConfig(MakeConfig({"q_proj", "o_proj", "mtp"}));

  std::vector<float> a(128, 1.0f);
  std::vector<float> b(128, 2.0f);
  dir.WriteFile("adapter_model.safetensors", MakeSafetensors({
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
       "F32", {8, 16}, F32Data(a)},
      {"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
       "F32", {16, 8}, F32Data(b)},
      {"base_model.model.mtp.layers.0.q_proj.lora_A.weight",
       "F32", {8, 16}, F32Data(a)},
      {"base_model.model.mtp.layers.0.q_proj.lora_B.weight",
       "F32", {16, 8}, F32Data(b)},
  }));

  PEFTHelper peft = PEFTHelper::FromLocalDir(dir.path(), 4096);
  LoRAModel model = LoRAModel::FromLocalCheckpoint(
      dir.path(), {"q_proj", "o_proj", "mtp"}, peft, 1, std::nullopt,
      nullptr, {"mtp."});

  // q_proj should be loaded, mtp should be skipped.
  CHECK(model.loras.size() == 1);
  CHECK(model.CheckLoraName("model.layers.0.self_attn.q_proj"));
  CHECK_FALSE(model.CheckLoraName("mtp.layers.0.q_proj"));
}

// ---------------------------------------------------------------------------
// WeightsMapper order: substr applied before prefix (models/utils.py:108-126)
// ---------------------------------------------------------------------------

TEST_CASE("weights_mapper: substr before prefix order matters") {
  // A prefix replacement creates a substring that the substr rule would match
  // if it ran second. With upstream order (substr first, prefix second), the
  // substr rule does NOT fire because the prefix hasn't created the match yet.
  WeightsMapper mapper;
  mapper.orig_to_new_prefix = {{"model.", "X."}};
  mapper.orig_to_new_substr = {{"X.", "Y."}};

  // "model.layers.0" → substr finds no "X." → prefix: "model."→"X." → "X.layers.0"
  auto result = mapper.MapName("model.layers.0");
  REQUIRE(result.has_value());
  CHECK(*result == "X.layers.0");

  // If order were reversed (prefix first, then substr), the result would be
  // "Y.layers.0" — this test fails on that swap.
  CHECK(*result != "Y.layers.0");
}

// ---------------------------------------------------------------------------
// LoRAModel constructor: lora_model_id validation
// ---------------------------------------------------------------------------

TEST_CASE("lora_model: non-positive id throws") {
  std::unordered_map<std::string, vllm::lora::LoRALayerWeights> empty;
  CHECK_THROWS_AS(LoRAModel(0, 8, std::move(empty)), std::invalid_argument);
  CHECK_THROWS_AS(LoRAModel(-1, 8, std::move(empty)), std::invalid_argument);
}
