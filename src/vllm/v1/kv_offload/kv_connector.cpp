// Ported from: vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
//               (OffloadingConnectorScheduler) @ e24d1b24
// See include/vllm/v1/kv_offload/kv_connector.h for scope and the load-bearing
// semantics (the nullopt third state, block-hash striding, load-before-compute).
#include "vllm/v1/kv_offload/kv_connector.h"

#include <algorithm>

#include "vllm/v1/request.h"

namespace vllm::v1::kv_offload {

OffloadingConnector::OffloadingConnector(TieringOffloadingManager& manager,
                                         OffloadingConnectorConfig config)
    : manager_(manager), config_(config) {}

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
  // W4 is synchronous: async flag is always false (see header deviation).
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

// base.py:542-561 / scheduler.py:2341-2371. block_ids ignored (the offloading
// connector tracks its own), collapsing the SupportsHMA split.
void OffloadingConnector::request_finished(const Request& request) {
  hit_keys_.erase(request.request_id);
  manager_.on_request_finished(ReqContext{request.request_id});
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
