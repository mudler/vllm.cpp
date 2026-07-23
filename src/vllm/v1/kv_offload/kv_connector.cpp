// Ported from: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
//               (OffloadingConnectorScheduler) @ e24d1b24
//               vllm/distributed/kv_transfer/kv_connector/factory.py:27-125
//               (the connector registry + config selection)
// See include/vllm/v1/kv_offload/kv_connector.h for scope, the abstract ABI, and
// the load-bearing-vs-no-op statement (the nullopt third state, block-hash
// striding, load-before-compute).
#include "vllm/v1/kv_offload/kv_connector.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "vllm/v1/request.h"

namespace vllm::v1::kv_offload {

// ---------------------------------------------------------------------------
// KVConnectorFactory — the compile-time registry (factory.py:27-125).
// A Meyers-singleton map so registration from other TUs' static initializers is
// order-independent, exactly like the model registry.
// ---------------------------------------------------------------------------
namespace {
std::map<std::string, KVConnectorBuilder>& registry() {
  static std::map<std::string, KVConnectorBuilder> r;
  return r;
}
}  // namespace

void KVConnectorFactory::Register(const std::string& name,
                                  KVConnectorBuilder builder) {
  auto& r = registry();
  if (r.count(name) != 0) {
    // factory.py:33-36: duplicate registration is rejected.
    throw std::runtime_error("KV connector already registered: " + name);
  }
  r[name] = builder;
}

std::unique_ptr<KVConnector> KVConnectorFactory::Create(
    const KVConnectorContext& ctx) {
  // factory.py:96-125 + config selection. Default-OFF: no config, or a config
  // with no connector name / role, yields nullptr (the scheduler treats a null
  // connector as zero behaviour change).
  if (ctx.config == nullptr || !ctx.config->is_kv_transfer_instance()) {
    return nullptr;
  }
  const std::string& name = *ctx.config->kv_connector;
  auto& r = registry();
  auto it = r.find(name);
  if (it == r.end()) {
    // factory.py:91-92: unsupported connector name.
    throw std::runtime_error("Unsupported KV connector: " + name);
  }
  return it->second(ctx);
}

bool KVConnectorFactory::IsRegistered(const std::string& name) {
  return registry().count(name) != 0;
}

std::vector<std::string> KVConnectorFactory::RegisteredNames() {
  std::vector<std::string> names;
  names.reserve(registry().size());
  for (const auto& [name, _] : registry()) names.push_back(name);
  return names;
}

// ---------------------------------------------------------------------------
// OffloadingConnector.
// ---------------------------------------------------------------------------

OffloadingConnector::OffloadingConnector(TieringOffloadingManager& manager,
                                         OffloadingConnectorConfig config)
    : KVConnector(KVConnectorRole::kScheduler),
      manager_(manager),
      config_(config) {}

OffloadingConnector::OffloadingConnector(
    std::unique_ptr<PrimaryByteView> owned_view,
    std::unique_ptr<TieringOffloadingManager> owned_manager,
    OffloadingConnectorConfig config)
    : KVConnector(KVConnectorRole::kScheduler),
      owned_view_(std::move(owned_view)),
      owned_manager_(std::move(owned_manager)),
      manager_(*owned_manager_),
      config_(config) {}

OffloadingConnector::~OffloadingConnector() = default;

namespace {
// Small helpers for reading integer/bool fields out of kv_connector_extra_config.
int64_t extra_int(const KVTransferConfig& cfg, const std::string& key,
                  int64_t dflt) {
  const std::string v = cfg.get_from_extra_config(key, "");
  if (v.empty()) return dflt;
  return static_cast<int64_t>(std::stoll(v));
}
bool extra_bool(const KVTransferConfig& cfg, const std::string& key,
                bool dflt) {
  const std::string v = cfg.get_from_extra_config(key, "");
  if (v.empty()) return dflt;
  return v == "1" || v == "true" || v == "True";
}
}  // namespace

// factory.py builder for "OffloadingConnector" (offloading_connector.py:46). Reads
// the tiered-offload config from kv_connector_extra_config + the runtime geometry
// from the context, then builds and OWNS the CPU primary + disk secondary stack.
std::unique_ptr<KVConnector> OffloadingConnector::CreateFromConfig(
    const KVConnectorContext& ctx) {
  if (ctx.config == nullptr) {
    throw std::runtime_error("OffloadingConnector: null KVTransferConfig");
  }
  const KVTransferConfig& cfg = *ctx.config;
  if (ctx.page_size_bytes <= 0) {
    throw std::runtime_error(
        "OffloadingConnector: page_size_bytes must be > 0 (set from the "
        "offloaded group's KV cache spec)");
  }
  if (ctx.identity == nullptr) {
    throw std::runtime_error(
        "OffloadingConnector: a CacheIdentity is required for the verified "
        "disk-tier header (§B2)");
  }
  const std::string root_dir = cfg.get_from_extra_config("root_dir", "");
  if (root_dir.empty()) {
    throw std::runtime_error(
        "OffloadingConnector: kv_connector_extra_config['root_dir'] is "
        "required");
  }

  // Block count for the CPU primary tier: explicit num_cpu_blocks, else derived
  // from cpu_bytes_to_use / page (mirrors upstream deriving the count from a byte
  // budget), else a small default.
  int64_t num_cpu_blocks = extra_int(cfg, "num_cpu_blocks", 0);
  if (num_cpu_blocks <= 0) {
    const int64_t cpu_bytes = extra_int(cfg, "cpu_bytes_to_use", 0);
    if (cpu_bytes > 0) num_cpu_blocks = cpu_bytes / ctx.page_size_bytes;
  }
  if (num_cpu_blocks <= 0) num_cpu_blocks = 64;

  const std::string eviction_policy =
      cfg.get_from_extra_config("eviction_policy", "lru");

  OffloadingConnectorConfig ccfg;
  ccfg.group_idx = static_cast<uint32_t>(extra_int(cfg, "group_idx", 0));
  ccfg.offload_block_tokens = static_cast<int>(
      extra_int(cfg, "offload_block_tokens", ctx.block_size));
  ccfg.offload_prompt_only = extra_bool(cfg, "offload_prompt_only", true);

  auto view = std::make_unique<HeapPrimaryByteView>(
      num_cpu_blocks, static_cast<std::size_t>(ctx.page_size_bytes));
  auto primary =
      std::make_unique<CPUOffloadingManager>(num_cpu_blocks, eviction_policy);
  FileSystemTierOptions opts;
  opts.root_dir = root_dir;
  opts.identity = *ctx.identity;
  opts.capacity_bytes = extra_int(cfg, "fs_capacity_bytes", 0);
  opts.eviction_policy = eviction_policy;
  auto secondary = std::make_unique<FileSystemTier>(opts);
  auto manager = std::make_unique<TieringOffloadingManager>(
      std::move(primary), std::move(secondary), *view);

  // The private owning constructor takes ownership so the stack outlives use.
  return std::unique_ptr<KVConnector>(new OffloadingConnector(
      std::move(view), std::move(manager), ccfg));
}

// offloading/scheduler.py:270-284. Offloaded block j keys on the LAST
// fine-grained hash of its factor-group: block_hashes[hbsf*j + hbsf-1].
std::optional<OffloadKey> OffloadingConnector::offload_key(
    const Request& request, int j) const {
  const int hbsf = config_.hash_block_size_factor;
  const int idx = hbsf * j + hbsf - 1;
  if (idx < 0 || idx >= static_cast<int>(request.block_hashes.size())) {
    return std::nullopt;
  }
  return make_offload_key(request.block_hashes[idx], config_.group_idx);
}

void OffloadingConnector::on_new_request(const Request& request) {
  manager_.on_new_request(ReqContext{request.request_id});
}

// base.py:453-486 + offloading/scheduler.py:648-693. Walk the offloaded-block
// keys beyond the locally-computed prefix; count the consecutive HIT run; the
// nullopt third state propagates a promotion still in flight.
MatchResult OffloadingConnector::get_num_new_matched_tokens(
    const Request& request, int num_computed_tokens) {
  const ReqContext ctx{request.request_id};
  const int tokens_per_block = config_.offload_block_tokens;
  if (tokens_per_block <= 0) return MatchResult{0, false};

  // The first offloaded block NOT already covered by the local prefix cache.
  const int start_block = num_computed_tokens / tokens_per_block;

  // Recompute AT LEAST ONE token: mirror get_computed_blocks' max_cache_hit
  // length of num_tokens - 1 (kv_cache_manager.py:227). Cap the external match to
  // whole blocks so total computed <= NumTokens() - 1, keeping block alignment
  // and guaranteeing the scheduler always has a token to schedule.
  const int max_ext_tokens =
      std::max(0, (request.NumTokens() - 1) - num_computed_tokens);
  const int max_hit_blocks = max_ext_tokens / tokens_per_block;

  // Mirror _maximal_prefix_lookup (offloading/scheduler.py:390-410): scan the
  // consecutive prefix; on RETRY/HIT_PENDING set `defer` but KEEP scanning so
  // EVERY prospective prefix block's promotion is initiated THIS step (the disk
  // hits kick off in one pass). A single MISS ends the run. When deferring,
  // return the nullopt third state; next step the same blocks are HITs and the
  // count is returned.
  std::vector<OffloadKey> hit;
  int prospective = 0;  // would-be-hit blocks (HIT or in-flight promotion)
  bool defer = false;
  for (int j = start_block; prospective < max_hit_blocks; ++j) {
    std::optional<OffloadKey> key = offload_key(request, j);
    if (!key.has_value()) break;  // no more full blocks
    const LookupResult r = manager_.lookup(*key, ctx);
    if (r == LookupResult::kHit) {
      hit.push_back(*key);
      prospective += 1;
      continue;
    }
    if (r == LookupResult::kHitPending || r == LookupResult::kRetry) {
      // A disk->CPU promotion is running; still a prospective prefix hit
      // (scheduler.py:744-750). Keep scanning to initiate the rest.
      defer = true;
      prospective += 1;
      continue;
    }
    break;  // kMiss: the hit run ends
  }

  if (defer) {
    // The THIRD state. Do NOT record hit keys — nothing is loadable yet.
    hit_keys_.erase(request.request_id);
    return MatchResult{std::nullopt, false};
  }

  // Protect the hit blocks from eviction while this request is scheduled and
  // record them for update_state_after_alloc (idempotent: overwrite).
  if (!hit.empty()) manager_.touch(hit, ctx);
  hit_keys_[request.request_id] = hit;

  const int num_hit_tokens = static_cast<int>(hit.size()) * tokens_per_block;
  // W4/W5 are synchronous: async flag is always false (see header deviation).
  return MatchResult{num_hit_tokens, false};
}

// base.py:488-507 + offloading/scheduler.py:695-792. Record the load of the hit
// keys into the freshly-allocated GPU blocks. The ext==0 early-out makes the
// scheduler's second (post-load) call a no-op.
void OffloadingConnector::update_state_after_alloc(
    const Request& request, const std::vector<std::vector<int>>& block_ids,
    int num_external_tokens) {
  if (num_external_tokens == 0) return;  // scheduler.py:698-699

  auto it = hit_keys_.find(request.request_id);
  if (it == hit_keys_.end() || it->second.empty()) return;
  const std::vector<OffloadKey>& keys = it->second;

  // The external prefix occupies the GPU blocks immediately AFTER the locally
  // computed ones. num_external_tokens == keys.size() * tokens_per_block, so the
  // external blocks are the [n_local, n_local + keys.size()) slice of the
  // group's allocated block ids.
  if (config_.group_idx >= block_ids.size()) return;
  const std::vector<int>& group_blocks = block_ids[config_.group_idx];
  const int tokens_per_block = config_.offload_block_tokens;
  const int n_local = request.num_computed_tokens / tokens_per_block;
  const int n_ext = static_cast<int>(keys.size());
  if (n_local + n_ext > static_cast<int>(group_blocks.size())) return;

  ConnectorLoadJob job;
  job.req_id = request.request_id;
  job.keys = keys;
  job.gpu_block_ids.assign(group_blocks.begin() + n_local,
                           group_blocks.begin() + n_local + n_ext);
  // Pin the tier blocks for the duration of the load (released by the worker on
  // load completion).
  manager_.prepare_load(keys, ReqContext{request.request_id});
  batch_loads_.push_back(std::move(job));
}

// base.py:509-522. Drain + RESET the per-step batch state.
std::vector<ConnectorLoadJob> OffloadingConnector::build_connector_meta() {
  std::vector<ConnectorLoadJob> out;
  out.swap(batch_loads_);
  return out;
}

// base.py:93-116 / scheduler.py:2371. block_ids ignored (the offloading
// connector tracks its own), collapsing the SupportsHMA single-vs-all-groups
// split. Synchronous store => no deferred free (delay_free stays false).
RequestFinishedResult OffloadingConnector::request_finished_all_groups(
    const Request& request,
    const std::vector<std::vector<int>>& /*block_ids_per_group*/) {
  hit_keys_.erase(request.request_id);
  manager_.on_request_finished(ReqContext{request.request_id});
  return RequestFinishedResult{};
}

void OffloadingConnector::on_schedule_end() { manager_.on_schedule_end(); }

// offloading/scheduler.py:844-1025 (the key set only; byte movement is the
// worker's). offload_prompt_only clamps to the prompt's blocks.
std::vector<OffloadKey> OffloadingConnector::store_keys(
    const Request& request, int num_computed_tokens) const {
  const int tokens_per_block = config_.offload_block_tokens;
  if (tokens_per_block <= 0) return {};
  int limit_tokens = num_computed_tokens;
  if (config_.offload_prompt_only) {
    limit_tokens = std::min(limit_tokens, request.num_prompt_tokens);
  }
  const int num_full_blocks = limit_tokens / tokens_per_block;
  std::vector<OffloadKey> keys;
  keys.reserve(static_cast<size_t>(std::max(0, num_full_blocks)));
  for (int j = 0; j < num_full_blocks; ++j) {
    std::optional<OffloadKey> k = offload_key(request, j);
    if (!k.has_value()) break;
    keys.push_back(*k);
  }
  return keys;
}

}  // namespace vllm::v1::kv_offload

// Self-register the disk-offload connector (factory.py:206). Compile-time
// registration replaces vLLM's dynamic importlib module path (recorded
// deviation). Selected by KVTransferConfig{kv_connector = "OffloadingConnector"}.
REGISTER_KV_CONNECTOR(offloading, "OffloadingConnector",
                      &::vllm::v1::kv_offload::OffloadingConnector::CreateFromConfig)
