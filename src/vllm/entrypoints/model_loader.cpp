// See include/vllm/entrypoints/model_loader.h. ORIGINAL packaging helper — the
// shared model-load + engine-stack wiring behind both the OpenAI server and the
// C ABI. Mirrors the M1.8 LLMEngine __init__ (vllm/v1/engine/llm_engine.py @
// e24d1b24) as exercised by examples/server/main.cpp and the test harness.
#include "vllm/entrypoints/model_loader.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm::entrypoints {

namespace fs = std::filesystem;

namespace {

vt::Queue CpuQueue() {
  // T0 runs the reference CPU backend (matches examples/server/main.cpp). The
  // CUDA/dgx path selects a device queue in a later milestone.
  return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
}

std::vector<vllm::SafetensorsFile> LoadShards(const std::string& model_dir) {
  std::vector<std::string> paths;
  for (const auto& e : fs::directory_iterator(model_dir)) {
    if (e.is_regular_file() && e.path().extension() == ".safetensors") {
      paths.push_back(e.path().string());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no *.safetensors shards found in " + model_dir);
  }
  std::sort(paths.begin(), paths.end());
  std::vector<vllm::SafetensorsFile> shards;
  shards.reserve(paths.size());
  for (const std::string& p : paths) {
    shards.push_back(vllm::SafetensorsFile::Open(p));
  }
  return shards;
}

}  // namespace

bool LoadedEngine::EnsureNoneHash() {
  // Idempotent: init_none_hash just (re)assigns the NONE_HASH global. Prefix
  // caching stays inert for prompts shorter than a block, so the (unseeded)
  // NONE_HASH value does not affect determinism here.
  vllm::v1::init_none_hash(vllm::v1::sha256_cbor);
  return true;
}

vllm::SchedulerConfig LoadedEngine::MakeSchedulerConfig(int max_model_len,
                                                        int max_num_seqs) {
  vllm::SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_model_len * max_num_seqs;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  return cfg;
}

vllm::v1::KVCacheConfig LoadedEngine::MakeKvConfig(const HfConfig& c,
                                                   int block_size,
                                                   int num_blocks) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  vllm::v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<vllm::v1::FullAttentionSpec>(block_size, Hkv, Dh,
                                                    vt::DType::kF32));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn"},
      std::make_shared<vllm::v1::MambaSpec>(
          block_size,
          std::vector<std::vector<int64_t>>{{Hv, Dv, Dk}, {conv_dim, Kw - 1}},
          std::vector<vt::DType>{vt::DType::kF32, vt::DType::kF32}));
  return kv;
}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5MoeWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params)
    : hash_ready_(EnsureNoneHash()),
      config_(std::move(config)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      max_model_len_(params.max_model_len > 0
                         ? params.max_model_len
                         : static_cast<int>(config_.max_position_embeddings)),
      kv_cfg_(MakeKvConfig(config_,
                           params.block_size > 0 ? params.block_size : 32,
                           params.num_blocks > 0 ? params.num_blocks : 256)),
      scheduler_(MakeSchedulerConfig(
                     max_model_len_,
                     params.max_num_seqs > 0 ? params.max_num_seqs : 8),
                 kv_cfg_, params.block_size > 0 ? params.block_size : 32,
                 /*enable_caching=*/true),
      runner_(config_, weights_, kv_cfg_, CpuQueue(),
              /*max_num_reqs=*/params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_model_len_,
              /*max_num_batched_tokens=*/
              max_model_len_ * (params.max_num_seqs > 0 ? params.max_num_seqs
                                                        : 8)),
      executor_(runner_),
      engine_core_(scheduler_, executor_),
      input_processor_(tokenizer_, config_),
      output_processor_(&tokenizer_),
      engine_(input_processor_, engine_core_, output_processor_,
              vllm::v1::get_request_block_hasher(
                  params.block_size > 0 ? params.block_size : 32,
                  vllm::v1::sha256_cbor)) {
  (void)hash_ready_;
}

std::unique_ptr<LoadedEngine> LoadedEngine::FromModelDir(
    const std::string& model_dir, const EngineParams& params) {
  const fs::path dir(model_dir);
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("model path is not a directory: " + model_dir);
  }
  const std::string config_path = (dir / "config.json").string();
  const std::string tokenizer_path = (dir / "tokenizer.json").string();

  HfConfig config = vllm::LoadHfConfig(config_path);
  tok::Tokenizer tokenizer = tok::Tokenizer::FromHfJson(tokenizer_path);

  std::vector<vllm::SafetensorsFile> shards = LoadShards(model_dir);
  Qwen3_5MoeWeights weights = vllm::LoadQwen3_5Moe(shards, config);
  shards.clear();  // the mmap'd shards may be released after the load.

  return std::make_unique<LoadedEngine>(std::move(config), std::move(weights),
                                        std::move(tokenizer), params);
}

}  // namespace vllm::entrypoints
