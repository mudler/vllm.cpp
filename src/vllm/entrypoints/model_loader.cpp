// See include/vllm/entrypoints/model_loader.h. ORIGINAL packaging helper — the
// shared model-load + engine-stack wiring behind both the OpenAI server and the
// C ABI. Mirrors the M1.8 LLMEngine __init__ (vllm/v1/engine/llm_engine.py @
// e24d1b24) as exercised by examples/server/main.cpp and the test harness.
#include "vllm/entrypoints/model_loader.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "vt/dtype.h"
#include "vt/tensor.h"
#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
#include "vt/cuda/nvfp4_autotune.h"
#endif

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

bool DirectDeviceLoadRequested() {
  const char* release = std::getenv("VT_RELEASE_HOST_WEIGHTS");
  if (release != nullptr && release[0] == '0') return false;
  const char* direct = std::getenv("VT_DIRECT_DEVICE_LOAD");
  return direct == nullptr || direct[0] != '0';
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

#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
bool EnvironmentEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr || value[0] != '0';
}
#endif

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

bool LoadedEngine::ResolveEnablePrefixCaching(const EngineParams& params,
                                               const ModelInfo& model_info) {
  if (params.enable_prefix_caching.has_value()) {
    return *params.enable_prefix_caching;
  }
  // ModelConfig.is_prefix_caching_supported at the parity pin: generative
  // hybrid and attention-free models default OFF while ordinary decoder-only
  // models default ON. ModelInfo.has_inner_state covers the attention-free
  // family in the native registry; is_hybrid covers Qwen3.5/3.6 GDN.
  return !model_info.is_hybrid && !model_info.has_inner_state;
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

// The construction-time async-scheduling resolution (W3 enable-flip). Mirrors
// vLLM: async_scheduling=None resolves to True when compatible
// (vllm/config/vllm.py:990-1038) gated on the runner advertising the async
// device path, then the house VT_ASYNC_SCHED=0 rollback env force-disables it in
// the same binary. `async_scheduling` stays nullopt on MakeSchedulerConfig, so
// ResolveAsyncScheduling(runner_supports_async) yields runner_supports_async
// (when otherwise compatible).
bool LoadedEngine::ResolveAsyncEnabled(
    const vllm::SchedulerConfig& scheduler_config, bool runner_supports_async) {
  return vllm::AsyncSchedulingEnabled(
      scheduler_config.ResolveAsyncScheduling(runner_supports_async));
}

std::unique_ptr<vllm::v1::Scheduler> LoadedEngine::MakeScheduler(
    bool async_enabled, vllm::SchedulerConfig scheduler_config,
    vllm::v1::KVCacheConfig kv_cache_config, int block_size,
    bool enable_caching) {
  if (async_enabled) {
    // get_scheduler_cls -> AsyncScheduler (scheduler.py:180-189).
    return std::make_unique<vllm::v1::AsyncScheduler>(
        std::move(scheduler_config), std::move(kv_cache_config), block_size,
        enable_caching);
  }
  return std::make_unique<vllm::v1::Scheduler>(
      std::move(scheduler_config), std::move(kv_cache_config), block_size,
      enable_caching);
}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5MoeWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params)
    : LoadedEngine(std::move(config),
                   MakeQwen3_5MoeLoadedModel(std::move(weights)),
                   std::move(tokenizer), params) {}

LoadedEngine::LoadedEngine(HfConfig config, Qwen3_5DenseWeights weights,
                           tok::Tokenizer tokenizer, const EngineParams& params)
    : LoadedEngine(std::move(config),
                   MakeQwen3_5DenseLoadedModel(std::move(weights)),
                   std::move(tokenizer), params) {}

LoadedEngine::LoadedEngine(HfConfig config,
                           std::unique_ptr<LoadedModel> model,
                           tok::Tokenizer tokenizer,
                           const EngineParams& params,
                           vt::Queue* preselected_queue)
    : hash_ready_(EnsureNoneHash()),
      config_(std::move(config)),
      model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      max_model_len_(params.max_model_len > 0
                         ? params.max_model_len
                         : static_cast<int>(config_.max_position_embeddings)),
      max_num_batched_tokens_(ResolveMaxNumBatchedTokens(
          params, max_model_len_, ModelRegistry::IsDenseModel(*model_))),
      prefix_caching_enabled_(ResolveEnablePrefixCaching(
          params, model_->registration().info)),
      kv_cfg_(ModelRegistry::MakeKVCache(
          *model_, config_, params.block_size > 0 ? params.block_size : 32,
          params.num_blocks > 0 ? params.num_blocks : 256)),
      // runner_ FIRST (W3): the async-scheduling flip reads
      // runner_.runner_supports_async().
      runner_(config_, *model_, kv_cfg_,
              preselected_queue != nullptr ? *preselected_queue : SelectQueue(),
              /*max_num_reqs=*/params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_model_len_,
              /*max_num_batched_tokens=*/max_num_batched_tokens_),
      // Resolve the enable-flip from the now-constructed runner + VT_ASYNC_SCHED,
      // then size the batch-queue depth (2 under async scheduling → depth-2
      // step_with_batch_queue; 1 otherwise). Since the 2026-07-17 flip the default
      // (no env) resolves ON (VT_ASYNC_RUNNER default ON), mirroring vLLM;
      // VT_ASYNC_RUNNER=0 / VT_ASYNC_SCHED=0 roll back to the synchronous path.
      async_scheduling_enabled_(ResolveAsyncEnabled(
          MakeSchedulerConfig(
              max_model_len_,
              params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_num_batched_tokens_, params.policy),
          runner_.runner_supports_async())),
      max_concurrent_batches_(MakeSchedulerConfig(
                                  max_model_len_,
                                  params.max_num_seqs > 0 ? params.max_num_seqs
                                                          : 8,
                                  max_num_batched_tokens_, params.policy)
                                  .MaxConcurrentBatches(async_scheduling_enabled_)),
      // AsyncScheduler when the flip resolved ON, else the synchronous Scheduler.
      scheduler_(MakeScheduler(
          async_scheduling_enabled_,
          MakeSchedulerConfig(
              max_model_len_, params.max_num_seqs > 0 ? params.max_num_seqs : 8,
              max_num_batched_tokens_, params.policy),
          kv_cfg_, params.block_size > 0 ? params.block_size : 32,
          /*enable_caching=*/prefix_caching_enabled_)),
      executor_(runner_),
      engine_core_(*scheduler_, executor_),
      input_processor_(tokenizer_, config_),
      output_processor_(&tokenizer_),
      block_hasher_(prefix_caching_enabled_
                        ? vllm::v1::get_request_block_hasher(
                              params.block_size > 0 ? params.block_size : 32,
                              vllm::v1::sha256_cbor)
                        : nullptr),
      engine_(input_processor_, engine_core_, output_processor_, block_hasher_) {
  (void)hash_ready_;
  // Mirror vLLM's "Asynchronous scheduling is enabled/disabled" resolution log
  // (vllm/config/vllm.py:990-1038) so the DGX A/B can audit which arm is live.
  std::cerr << "vllm.cpp: Asynchronous scheduling is "
            << (async_scheduling_enabled_ ? "enabled" : "disabled")
            << " (max_concurrent_batches=" << max_concurrent_batches_ << ")\n";
  WarmupKernels();
}

void LoadedEngine::WarmupKernels() {
#if defined(VLLM_CPP_CUDA) && defined(VT_CUTLASS_NVFP4)
  if (!model_->uses_nvfp4_w4a4() ||
      runner_.device().type != vt::DeviceType::kCUDA ||
      !EnvironmentEnabled("VT_FP4_PRE_SERVE_WARMUP") ||
      !EnvironmentEnabled("VT_FP4_AUTOTUNE") ||
      !EnvironmentEnabled("VT_FP4_PLAN_CACHE")) {
    return;
  }

  int32_t dummy_token = -1;
  for (int32_t token = 0; token < tokenizer_.VocabSize(); ++token) {
    if (tokenizer_.HasToken(token) && !tokenizer_.IsSpecial(token)) {
      dummy_token = token;
      break;
    }
  }
  if (dummy_token < 0) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup could not find a non-special tokenizer token");
  }

  SamplingParams sampling;
  sampling.max_tokens = 1;
  sampling.temperature = 0.0;
  sampling.ignore_eos = true;
  sampling.PostInit();
  std::vector<int32_t> prompt(
      static_cast<size_t>(max_num_batched_tokens_), dummy_token);

  std::cerr << "vllm.cpp: warming FlashInfer-parity NVFP4 profiles at "
            << max_num_batched_tokens_ << " tokens before serving\n";
  vt::cuda::Nvfp4AutotuneWarmupScope warmup(
      static_cast<uint32_t>(max_num_batched_tokens_), runner_.device().index);
  engine_core_.add_request(std::make_unique<vllm::v1::Request>(
      "_vllm_cpp_nvfp4_warmup", std::move(prompt), std::move(sampling),
      /*arrival_time=*/0.0, /*block_hasher=*/nullptr));
  while (scheduler_->get_num_unfinished_requests() > 0) {
    (void)engine_core_.step();
  }
  if (scheduler_->has_finished_requests()) {
    (void)engine_core_.step();
  }
  if (scheduler_->get_num_unfinished_requests() != 0 ||
      scheduler_->has_finished_requests()) {
    throw std::runtime_error(
        "NVFP4 pre-serve warmup left scheduler state behind");
  }
  warmup.Complete();
#endif
}

vllm::v1::AsyncLLM& LoadedEngine::async_engine() {
  std::lock_guard<std::mutex> lock(async_engine_mutex_);
  if (async_engine_ == nullptr) {
    // Thread the resolved depth-2 batch-queue size so the async frontend runs
    // step_with_batch_queue under async scheduling (W3 enable-flip); the sync
    // default keeps max_concurrent_batches_ == 1 (depth-1 step()).
    async_engine_ = std::make_unique<vllm::v1::AsyncLLM>(
        input_processor_, *scheduler_, executor_, output_processor_,
        block_hasher_, /*shutdown_timeout_s=*/0, max_concurrent_batches_);
  }
  return *async_engine_;
}

std::unique_ptr<LoadedEngine> LoadedEngine::FromModelDir(
    const std::string& model_dir, const EngineParams& params) {
  const fs::path dir(model_dir);

  // A single `.gguf` file: config + weights + tokenizer all come from the
  // GGUF (M0.10). The engine stack below is unchanged.
  if (fs::is_regular_file(dir) && dir.extension() == ".gguf") {
    vllm::GgufFile gguf = vllm::GgufFile::Open(model_dir);
    HfConfig config = vllm::HfConfigFromGguf(gguf);
    // Resolve before tokenizer/weight work so unsupported architecture errors
    // are deterministic and match registry.py rather than being masked by a
    // later source-specific missing-tensor/tokenizer error.
    (void)ModelRegistry::Resolve(config);
    tok::Tokenizer tokenizer = tok::Tokenizer::FromGguf(gguf);
    // Dense-vs-MoE GGUF dispatch now happens through the registry: the bench
    // branch's inline `IsDenseArch` split is superseded by
    // `HfConfigFromGguf` mapping the GGUF `general.architecture` key
    // (`qwen35` dense / `qwen35moe` / `qwen3next`) onto the registered
    // architecture ID, which resolves to the owning arch TU's GGUF loader.
    std::unique_ptr<LoadedModel> model =
        ModelRegistry::Load(config, ModelSource::FromGguf(gguf));
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params));
  }

  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    throw std::runtime_error("model path is not a directory: " + model_dir);
  }
  const std::string config_path = (dir / "config.json").string();
  const std::string tokenizer_path = (dir / "tokenizer.json").string();

  HfConfig config = vllm::LoadHfConfig(config_path);
  const ModelRegistration& registration = ModelRegistry::Resolve(config);
  tok::Tokenizer tokenizer = tok::Tokenizer::FromHfJson(tokenizer_path);

  // Shared ownership so a loader may retain the mmap'd shards past the load: the
  // Qwen3.6-35B MoE loader defers its routed-expert host copies and streams them
  // per layer during PrepareMarlinResident (bounds load-phase peak PSS). The
  // deferred-expert closure holds the last reference and releases the shards once
  // the device Marlin resident is built; loaders that don't retain it drop the
  // shards when this local `shards` and the model's ModelSource go out of scope.
  auto shards = std::make_shared<const std::vector<vllm::SafetensorsFile>>(
      LoadShards(model_dir));

  // Live architecture dispatch: consume config.architectures in order and let
  // the matched registration own the weight-name map/loader. Unknown dense
  // configs now reject instead of falling through num_experts == 0.
  if (!registration.factory->is_dense_model || !DirectDeviceLoadRequested()) {
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(
        config, ModelSource::FromSafetensorsOwned(shards));
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params));
  }

  // Select before loading so an eligible discrete-CUDA dense loader stages each
  // completed layer to the exact queue the runner will use. If construction
  // fails before the runner takes over, destroy the selected native stream.
  vt::Queue load_queue = SelectQueue();
  try {
    std::unique_ptr<LoadedModel> model = ModelRegistry::Load(
        config, ModelSource::FromSafetensorsOwned(shards, &load_queue));
    return std::unique_ptr<LoadedEngine>(new LoadedEngine(
        std::move(config), std::move(model), std::move(tokenizer), params,
        &load_queue));
  } catch (...) {
    vt::DestroyQueue(load_queue);
    throw;
  }
}

}  // namespace vllm::entrypoints
