// Ported from the Qwen3.5 plain-weight contracts exercised by upstream
// tests/models/language/generation/test_common.py and
// tests/models/utils.py::check_logprobs_close: ordinary BF16/F32 parameters,
// stacked projections, tied embeddings, and a complete real-checkpoint load.
// The local real-model case is an explicit SKIP when Qwen/Qwen3.5-4B is absent.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"

namespace fs = std::filesystem;

namespace {

std::string FindSnapshot() {
  if (const char* explicit_path = std::getenv("QWEN35_PLAIN_MODEL")) {
    if (fs::exists(fs::path(explicit_path) / "config.json")) {
      return explicit_path;
    }
  }
  std::vector<fs::path> roots;
  if (const char* hf_home = std::getenv("HF_HOME")) roots.emplace_back(hf_home);
  if (const char* home = std::getenv("HOME")) {
    roots.emplace_back(fs::path(home) / ".cache/huggingface");
  }
  std::error_code ec;
  for (const fs::path& root : roots) {
    const fs::path snapshots =
        root / "hub/models--Qwen--Qwen3.5-4B/snapshots";
    if (!fs::is_directory(snapshots, ec)) continue;
    for (const auto& entry : fs::directory_iterator(snapshots, ec)) {
      if (fs::exists(entry.path() / "config.json", ec) &&
          fs::exists(entry.path() / "model.safetensors.index.json", ec)) {
        return entry.path().string();
      }
    }
  }
  return {};
}

std::vector<vllm::SafetensorsFile> OpenShards(const std::string& snapshot) {
  std::vector<std::string> paths;
  for (const auto& entry : fs::directory_iterator(snapshot)) {
    if (entry.path().extension() == ".safetensors") {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& path : paths) {
    shards.push_back(vllm::SafetensorsFile::Open(path));
  }
  return shards;
}

void CheckTensor(const vllm::OwnedTensor& tensor, vt::DType dtype,
                 std::initializer_list<int64_t> shape, bool nk = false) {
  REQUIRE_FALSE(tensor.Empty());
  CHECK(tensor.dtype == dtype);
  REQUIRE(tensor.rank == static_cast<int>(shape.size()));
  int index = 0;
  for (int64_t dim : shape) CHECK(tensor.shape[index++] == dim);
  CHECK(tensor.nk == nk);
}

}  // namespace

TEST_CASE("Qwen3.5-4B plain safetensors load stacked and tied weights") {
  const std::string snapshot = FindSnapshot();
  if (snapshot.empty()) {
    MESSAGE("SKIP: Qwen/Qwen3.5-4B is absent; set HF_HOME or "
            "QWEN35_PLAIN_MODEL to run the real plain-weight gate");
    return;
  }

  const vllm::HfConfig config =
      vllm::LoadHfConfig(snapshot + "/config.json");
  REQUIRE(config.architectures ==
          std::vector<std::string>{"Qwen3_5ForConditionalGeneration"});
  REQUIRE(config.hidden_size == 2560);
  REQUIRE(config.num_hidden_layers == 32);
  REQUIRE(config.intermediate_size == 9216);

  std::vector<vllm::SafetensorsFile> shards = OpenShards(snapshot);
  REQUIRE(shards.size() == 2);
  vllm::Qwen3_5DenseWeights weights =
      vllm::LoadQwen3_5Dense(shards, config);

  const int64_t h = config.hidden_size;
  const int64_t i = config.intermediate_size;
  const int64_t q = 2 * config.num_attention_heads * config.head_dim;
  const int64_t kv = config.num_key_value_heads * config.head_dim;
  const int64_t value =
      config.linear_num_value_heads * config.linear_value_head_dim;
  const int64_t conv =
      2 * config.linear_num_key_heads * config.linear_key_head_dim + value;

  CHECK(weights.tied_lm_head);
  CHECK(weights.lm_head.Empty());
  CheckTensor(weights.embed_tokens, vt::DType::kBF16,
              {config.vocab_size, h}, /*nk=*/true);
  CheckTensor(weights.final_norm, vt::DType::kBF16, {h});
  REQUIRE(weights.layers.size() == 32);
  CHECK(vllm::IsPlainBf16Qwen3_5Dense(weights));

  for (const vllm::Qwen3_5DenseLayerWeights& layer : weights.layers) {
    CheckTensor(layer.input_layernorm, vt::DType::kBF16, {h});
    CheckTensor(layer.post_attention_layernorm, vt::DType::kBF16, {h});
    CheckTensor(layer.mlp.gate_up_proj, vt::DType::kBF16, {2 * i, h}, true);
    CheckTensor(layer.mlp.down_proj, vt::DType::kBF16, {h, i}, true);
    CHECK(layer.mlp.gate_proj.Empty());
    CHECK(layer.mlp.up_proj.Empty());
    if (layer.is_linear_attention) {
      CheckTensor(layer.gdn.in_proj_qkvz, vt::DType::kBF16,
                  {conv + value, h}, true);
      CheckTensor(layer.gdn.in_proj_ba, vt::DType::kBF16,
                  {2 * config.linear_num_value_heads, h}, true);
      CheckTensor(layer.gdn.out_proj, vt::DType::kBF16, {h, value}, true);
      CheckTensor(layer.gdn.a_log, vt::DType::kF32,
                  {config.linear_num_value_heads});
      CheckTensor(layer.gdn.dt_bias, vt::DType::kF32,
                  {config.linear_num_value_heads});
    } else {
      CheckTensor(layer.attn.q_proj, vt::DType::kBF16, {q, h}, true);
      CheckTensor(layer.attn.k_proj, vt::DType::kBF16, {kv, h}, true);
      CheckTensor(layer.attn.v_proj, vt::DType::kBF16, {kv, h}, true);
      CheckTensor(layer.attn.o_proj, vt::DType::kBF16,
                  {h, config.num_attention_heads * config.head_dim}, true);
    }
  }
}

TEST_CASE("dense host release keeps a device-resident weight dispatch-visible") {
  vllm::Qwen3_5DenseWeights weights;
  weights.embed_tokens.dtype = vt::DType::kBF16;
  weights.embed_tokens.rank = 2;
  weights.embed_tokens.shape[0] = 4;
  weights.embed_tokens.shape[1] = 8;
  weights.embed_tokens.bytes.resize(4 * 8 * sizeof(uint16_t), 0x5A);
  weights.embed_tokens.d_dev =
      std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});

  const size_t expected = weights.embed_tokens.bytes.size();
  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(weights) == expected);
  CHECK_FALSE(weights.embed_tokens.Empty());
  CHECK_FALSE(weights.embed_tokens.HasHostBytes());
  CHECK(weights.embed_tokens.d_dev != nullptr);
}

TEST_CASE("Qwen3.5-4B direct-device load matches retained-host execution") {
  const std::string snapshot = FindSnapshot();
  if (snapshot.empty()) {
    MESSAGE("SKIP: Qwen/Qwen3.5-4B is absent; set HF_HOME or "
            "QWEN35_PLAIN_MODEL to run the direct-device gate");
    return;
  }
#ifndef VLLM_CPP_CUDA
  MESSAGE("SKIP: direct-device gate requires a CUDA build");
  return;
#else
  const auto generate = [&snapshot](const char* direct) {
    REQUIRE(setenv("VT_DIRECT_DEVICE_LOAD", direct, 1) == 0);
    vllm::SamplingParams sampling;
    sampling.temperature = 0.0;
    sampling.max_tokens = 4;
    sampling.PostInit();
    auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(
        snapshot, vllm::entrypoints::EngineParams{});
    return loaded->engine().generate("The capital of France is", sampling,
                                     std::string("direct-") + direct);
  };

  const vllm::RequestOutput retained = generate("0");
  const vllm::RequestOutput direct = generate("1");
  unsetenv("VT_DIRECT_DEVICE_LOAD");

  REQUIRE(retained.finished);
  REQUIRE(direct.finished);
  REQUIRE(retained.outputs.size() == 1);
  REQUIRE(direct.outputs.size() == 1);
  CHECK(direct.prompt_token_ids == retained.prompt_token_ids);
  CHECK(direct.outputs[0].token_ids == retained.outputs[0].token_ids);
#endif
}
