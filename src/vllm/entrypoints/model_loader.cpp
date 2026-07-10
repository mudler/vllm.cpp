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

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm::entrypoints {

namespace fs = std::filesystem;

namespace {

vt::Queue SelectQueue() {
  // M2.2b: run the engine forward on the CUDA device when the backend is
  // available (GB10), so the fp4-resident MoE/lm_head weights hit vt::MatmulNvfp4
  // on-device instead of the CPU dequant reference. Falls back to the reference
  // CPU backend when built without CUDA or when no usable GPU is present at
  // runtime (GetBackend(kCUDA) throws if the probe left it unregistered).
#ifdef VLLM_CPP_CUDA
  try {
    return vt::GetBackend(vt::DeviceType::kCUDA).CreateQueue();
  } catch (const std::exception&) {
    // No usable GPU; fall through to CPU.
  }
#endif
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

// Resolve the per-step token budget (max_num_batched_tokens) for chunked
// prefill. An explicit EngineParams override wins; otherwise a PER-ARCH bounded
// default that does NOT scale with max_num_seqs, so a long/many-request prefill
// is split across steps and the per-step GDN chunked-scan activation stays
// bounded regardless of concurrency (the 27B 8x1024 conc-8 OOM fix — the old
// max_model_len*max_num_seqs product ran the whole 8192-token prefill in one
// step).
//
//  * DENSE arch (27B W4A4): 2048 FLAT — mirrors vLLM's own scheduler default
//    (DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048, vllm/config/scheduler.py:42 @
//    e24d1b24). The dense prefill is expensive per token: at mnbt=8192 one
//    giant mixed step runs several full prompts' prefill eagerly and every
//    decode stream stalls behind it (TTFT ~2x, decode starved). MEASURED (27B
//    NVFP4, GB10, in1024/out128): conc32/np96 mnbt=2048 999.16 tok/s vs 8192
//    895.90 (+11.5%); the conc16/conc32 default-vs-8192 A/Bs are in
//    .agents/parity-ledger.md (2026-07-10).
//  * MoE arch (35B A3B W4A16): keep the GB10-tuned CONCURRENCY-AWARE budget —
//    8192 at high concurrency (>=32), else 4096. Its cheap A3B expert prefill
//    wants the bigger chunk: at the 35B gate (conc-64, in1024/out128)
//    mnbt=8192 is +2.7% over 4096 (itself +8.2% over 2048) — bigger prefill
//    chunks amortize per-token GEMM/attention work across the many running
//    seqs. At LOW concurrency 8192 loses pipelining, so those keep 4096.
//    Memory-safe on GB10's 119GB (35B conc-64 peak 54GB; 16384 OOMs).
//
// Invariants mirrored from SchedulerConfig.verify_max_model_len
// (vllm/config/scheduler.py:87): the budget must be >= max_num_seqs (every
// running seq needs at least one token/step). For tiny models whose whole
// workload is smaller than the default we keep the old
// (max_model_len*max_num_seqs) ceiling so no behavior changes there.
int LoadedEngine::ResolveMaxNumBatchedTokens(const EngineParams& params,
                                             int max_model_len,
                                             bool is_dense_arch) {
  const int seqs = params.max_num_seqs > 0 ? params.max_num_seqs : 8;
  if (params.max_num_batched_tokens > 0) {
    // Explicit override; still honor the >= max_num_seqs invariant.
    return std::max(params.max_num_batched_tokens, seqs);
  }
  int budget = is_dense_arch ? 2048 : (seqs >= 32 ? 8192 : 4096);
  // Never exceed the whole workload's ceiling (tiny-model no-op preservation).
  const long ceiling = static_cast<long>(max_model_len) * seqs;
  if (ceiling > 0 && static_cast<long>(budget) > ceiling) {
    budget = static_cast<int>(ceiling);
  }
  return std::max(budget, seqs);
}

bool LoadedEngine::EnsureNoneHash() {
  // Idempotent: init_none_hash just (re)assigns the NONE_HASH global. Prefix
  // caching stays inert for prompts shorter than a block, so the (unseeded)
  // NONE_HASH value does not affect determinism here.
  vllm::v1::init_none_hash(vllm::v1::sha256_cbor);
  return true;
}

vllm::SchedulerConfig LoadedEngine::MakeSchedulerConfig(
    int max_model_len, int max_num_seqs, int max_num_batched_tokens,
    vllm::SchedulerPolicy policy) {
  vllm::SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  // Bounded per-step budget (chunked prefill). See ResolveMaxNumBatchedTokens.
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;
  // Scheduling policy (fcfs default; kPriority selects the priority queue).
  cfg.policy = policy;
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
      moe_weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      max_model_len_(params.max_model_len > 0
                         ? params.max_model_len
                         : static_cast<int>(config_.max_position_embeddings)),
      kv_cfg_(MakeKvConfig(config_,
                           params.block_size > 0 ? params.block_size : 32,
                           params.num_blocks > 0 ? params.num_blocks : 256)),
      scheduler_(MakeSchedulerConfig(
                     max_model_len_,
                     params.max_num_seqs > 0 ? params.max_num_seqs : 8,
                     ResolveMaxNumBatchedTokens(params, max_model_len_,
                                                IsDenseArch(config_)),
                     params.policy),
                 kv_cfg_, params.block_size > 0 ? params.block_size : 32,
                 /*enable_caching=*/true),
      runner_(config_, *moe_weights_, kv_cfg_, SelectQueue(),
              /*max_num_reqs=*/params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_model_len_,
              /*max_num_batched_tokens=*/
              ResolveMaxNumBatchedTokens(params, max_model_len_,
                                         IsDenseArch(config_))),
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

// DENSE-arch overload (27B). Identical to the MoE constructor except dense_weights_
// carries the model and runner_ is built through the GPUModelRunner dense overload
// (Qwen3_5DenseModel::Forward). Every other member is arch-agnostic.
LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params)
    : hash_ready_(EnsureNoneHash()),
      config_(std::move(config)),
      dense_weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      max_model_len_(params.max_model_len > 0
                         ? params.max_model_len
                         : static_cast<int>(config_.max_position_embeddings)),
      kv_cfg_(MakeKvConfig(config_,
                           params.block_size > 0 ? params.block_size : 32,
                           params.num_blocks > 0 ? params.num_blocks : 256)),
      scheduler_(MakeSchedulerConfig(
                     max_model_len_,
                     params.max_num_seqs > 0 ? params.max_num_seqs : 8,
                     ResolveMaxNumBatchedTokens(params, max_model_len_,
                                                IsDenseArch(config_)),
                     params.policy),
                 kv_cfg_, params.block_size > 0 ? params.block_size : 32,
                 /*enable_caching=*/true),
      runner_(config_, *dense_weights_, kv_cfg_, SelectQueue(),
              /*max_num_reqs=*/params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_model_len_,
              /*max_num_batched_tokens=*/
              ResolveMaxNumBatchedTokens(params, max_model_len_,
                                         IsDenseArch(config_))),
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

bool LoadedEngine::IsDenseArch(const HfConfig& config) {
  // The dense 27B (Qwen3_5ForConditionalGeneration, text_config qwen3_5_text)
  // has no experts; the MoE 35B (Qwen3_5MoeForConditionalGeneration) sets
  // num_experts > 0. num_experts is the structural discriminator between the two
  // Qwen3.5 gate arches sharing this hybrid backbone.
  return config.num_experts == 0;
}

std::unique_ptr<LoadedEngine> LoadedEngine::FromModelDir(
    const std::string& model_dir, const EngineParams& params) {
  const fs::path dir(model_dir);

  // A single `.gguf` file: config + weights + tokenizer all come from the
  // GGUF (M0.10). The engine stack below is unchanged.
  if (fs::is_regular_file(dir) && dir.extension() == ".gguf") {
    vllm::GgufFile gguf = vllm::GgufFile::Open(model_dir);
    HfConfig config = vllm::HfConfigFromGguf(gguf);
    tok::Tokenizer tokenizer = tok::Tokenizer::FromGguf(gguf);
    Qwen3_5MoeWeights weights = vllm::LoadQwen3_5MoeFromGguf(gguf, config);
    return std::make_unique<LoadedEngine>(std::move(config), std::move(weights),
                                          std::move(tokenizer), params);
  }

  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("model path is not a directory: " + model_dir);
  }
  const std::string config_path = (dir / "config.json").string();
  const std::string tokenizer_path = (dir / "tokenizer.json").string();

  HfConfig config = vllm::LoadHfConfig(config_path);
  tok::Tokenizer tokenizer = tok::Tokenizer::FromHfJson(tokenizer_path);

  std::vector<vllm::SafetensorsFile> shards = LoadShards(model_dir);

  // Arch-select: the dense 27B (Qwen3_5ForConditionalGeneration, num_experts==0)
  // routes to LoadQwen3_5Dense + the dense engine constructor; the MoE 35B stays
  // on LoadQwen3_5Moe. Both drive the SAME stack below via the {moe,dense}_weights_
  // pair (the runner already carries either arch).
  if (LoadedEngine::IsDenseArch(config)) {
    Qwen3_5DenseWeights weights = vllm::LoadQwen3_5Dense(shards, config);
    shards.clear();  // the mmap'd shards may be released after the load.
    return std::make_unique<LoadedEngine>(std::move(config), std::move(weights),
                                          std::move(tokenizer), params);
  }

  Qwen3_5MoeWeights weights = vllm::LoadQwen3_5Moe(shards, config);
  shards.clear();  // the mmap'd shards may be released after the load.

  return std::make_unique<LoadedEngine>(std::move(config), std::move(weights),
                                        std::move(tokenizer), params);
}

}  // namespace vllm::entrypoints
