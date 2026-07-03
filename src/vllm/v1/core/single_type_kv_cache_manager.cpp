// Ported from: vllm/v1/core/single_type_kv_cache_manager.py @ e24d1b24
// See include/vllm/v1/core/single_type_kv_cache_manager.h for the scope /
// deferred list and the FullAttention-vs-Mamba difference this task turns on.
#include "vllm/v1/core/single_type_kv_cache_manager.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "vllm/v1/request.h"

namespace vllm::v1 {

namespace {

// Ceiling division for non-negative operands (upstream vllm.utils cdiv).
int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

// ---------------------------------------------------------------------------
// SingleTypeKVCacheManager (base)
// ---------------------------------------------------------------------------

SingleTypeKVCacheManager::SingleTypeKVCacheManager(
    std::shared_ptr<KVCacheSpec> kv_cache_spec, BlockPool& block_pool,
    bool enable_caching, int kv_cache_group_id, int scheduler_block_size)
    : scheduler_block_size(scheduler_block_size),
      block_size(kv_cache_spec->block_size),
      kv_cache_spec(std::move(kv_cache_spec)),
      block_pool(block_pool),
      enable_caching(enable_caching),
      kv_cache_group_id(kv_cache_group_id),
      _null_block(block_pool.null_block) {}

int SingleTypeKVCacheManager::_get_num_evictable_blocks(
    const std::vector<KVCacheBlock*>& blocks) {
  int count = 0;
  for (const KVCacheBlock* blk : blocks) {
    if (blk->ref_cnt == 0 && !blk->is_null) {
      ++count;
    }
  }
  return count;
}

int SingleTypeKVCacheManager::get_num_blocks_to_allocate(
    const std::string& request_id, int num_tokens,
    const std::vector<KVCacheBlock*>& new_computed_blocks,
    int total_computed_tokens, int /*num_tokens_main_model*/,
    bool /*apply_admission_cap*/) {
  int num_required_blocks = cdiv(num_tokens, block_size);
  // apply_admission_cap only matters for recycling-aware specs (SWA / chunked
  // local), which are DEFERRED; _max_admission_blocks_per_request is None here.
  auto it = req_to_blocks.find(request_id);
  int num_req_blocks =
      it != req_to_blocks.end() ? static_cast<int>(it->second.size()) : 0;

  if (num_cached_block.count(request_id) != 0) {
    // Fast-path: a running request won't have any new prefix-cache hits.
    assert(new_computed_blocks.empty());
    return std::max(num_required_blocks - num_req_blocks, 0);
  }

  int num_skipped_tokens = get_num_skipped_tokens(total_computed_tokens);
  int num_local_computed_blocks =
      static_cast<int>(new_computed_blocks.size()) + num_req_blocks;
  int num_skipped_blocks = num_skipped_tokens / block_size;
  int num_new_blocks = std::max(
      num_required_blocks -
          std::max(num_skipped_blocks, num_local_computed_blocks),
      0);

  int num_skipped_new_computed_blocks =
      std::max(0, num_skipped_blocks - num_req_blocks);

  std::vector<KVCacheBlock*> tail(
      new_computed_blocks.begin() +
          std::min<int>(num_skipped_new_computed_blocks,
                        static_cast<int>(new_computed_blocks.size())),
      new_computed_blocks.end());
  int num_evictable_blocks = _get_num_evictable_blocks(tail);
  return num_new_blocks + num_evictable_blocks;
}

void SingleTypeKVCacheManager::add_local_computed_blocks(
    const std::string& request_id,
    const std::vector<KVCacheBlock*>& new_computed_blocks,
    int num_local_computed_tokens, int num_external_computed_tokens) {
  std::vector<KVCacheBlock*>& req_blocks = req_to_blocks[request_id];
  assert(req_blocks.empty());
  int num_total_computed_tokens =
      num_local_computed_tokens + num_external_computed_tokens;
  int num_skipped_tokens = get_num_skipped_tokens(num_total_computed_tokens);
  int num_skipped_blocks = num_skipped_tokens / block_size;

  std::vector<KVCacheBlock*> computed = new_computed_blocks;
  if (num_skipped_blocks > 0) {
    // All new computed blocks may be skipped when
    // num_skipped_blocks > len(new_computed_blocks).
    int drop = std::min<int>(num_skipped_blocks,
                             static_cast<int>(computed.size()));
    computed.erase(computed.begin(), computed.begin() + drop);
  }

  if (enable_caching) {
    block_pool.touch(computed);
  } else {
    // Computed blocks should be empty when prefix caching is disabled.
    assert(computed.empty());
  }

  req_blocks.insert(req_blocks.end(), num_skipped_blocks, _null_block);
  req_blocks.insert(req_blocks.end(), computed.begin(), computed.end());
  num_cached_block[request_id] = static_cast<int>(req_blocks.size());
}

void SingleTypeKVCacheManager::allocate_external_computed_blocks(
    const std::string& request_id, int num_local_computed_tokens,
    int num_external_computed_tokens) {
  int num_total_computed_tokens =
      num_local_computed_tokens + num_external_computed_tokens;
  int num_skipped_tokens = get_num_skipped_tokens(num_total_computed_tokens);
  if (num_skipped_tokens > 0) {
    num_external_computed_tokens =
        std::min(num_total_computed_tokens - num_skipped_tokens,
                 num_external_computed_tokens);
  }
  if (num_external_computed_tokens <= 0) {
    return;
  }

  std::vector<KVCacheBlock*>& req_blocks = req_to_blocks[request_id];
  std::vector<KVCacheBlock*> allocated_blocks = block_pool.get_new_blocks(
      cdiv(num_total_computed_tokens, block_size) -
      static_cast<int>(req_blocks.size()));
  req_blocks.insert(req_blocks.end(), allocated_blocks.begin(),
                    allocated_blocks.end());
  if (kv_cache_spec->kind() == KVCacheSpecKind::kFullAttention) {
    for (KVCacheBlock* b : allocated_blocks) {
      new_block_ids.push_back(b->block_id);
    }
  }
}

std::vector<KVCacheBlock*> SingleTypeKVCacheManager::allocate_new_blocks(
    const std::string& request_id, int num_tokens,
    int /*num_tokens_main_model*/) {
  std::vector<KVCacheBlock*>& req_blocks = req_to_blocks[request_id];
  int num_required_blocks = cdiv(num_tokens, block_size);
  int num_new_blocks =
      num_required_blocks - static_cast<int>(req_blocks.size());
  if (num_new_blocks <= 0) {
    return {};
  }
  std::vector<KVCacheBlock*> new_blocks =
      block_pool.get_new_blocks(num_new_blocks);
  req_blocks.insert(req_blocks.end(), new_blocks.begin(), new_blocks.end());
  if (kv_cache_spec->kind() == KVCacheSpecKind::kFullAttention) {
    for (KVCacheBlock* b : new_blocks) {
      new_block_ids.push_back(b->block_id);
    }
  }
  return new_blocks;
}

std::vector<int> SingleTypeKVCacheManager::take_new_block_ids() {
  std::vector<int> ids;
  ids.swap(new_block_ids);
  return ids;
}

void SingleTypeKVCacheManager::cache_blocks(
    const Request& request, int num_tokens,
    std::optional<int> retention_interval) {
  auto it = num_cached_block.find(request.request_id);
  int num_cached_blocks = it != num_cached_block.end() ? it->second : 0;
  int num_full_blocks = num_tokens / block_size;

  if (num_cached_blocks >= num_full_blocks) {
    return;
  }

  std::optional<std::vector<bool>> block_mask =
      reachable_block_mask(num_cached_blocks, num_full_blocks,
                           scheduler_block_size, retention_interval,
                           request.num_prompt_tokens);
  block_pool.cache_full_blocks(request, req_to_blocks[request.request_id],
                               num_cached_blocks, num_full_blocks, block_size,
                               kv_cache_group_id, block_mask);

  num_cached_block[request.request_id] = num_full_blocks;
}

std::optional<std::vector<bool>> SingleTypeKVCacheManager::reachable_block_mask(
    int /*start_block*/, int /*end_block*/,
    std::optional<int> /*alignment_tokens*/,
    std::optional<int> /*retention_interval*/,
    std::optional<int> /*num_prompt_tokens*/) {
  // Cache every (non-null) block — the full-attention default.
  return std::nullopt;
}

std::vector<KVCacheBlock*> SingleTypeKVCacheManager::pop_blocks_for_free(
    const std::string& request_id) {
  std::vector<KVCacheBlock*> req_blocks;
  auto it = req_to_blocks.find(request_id);
  if (it != req_to_blocks.end()) {
    req_blocks = std::move(it->second);
    req_to_blocks.erase(it);
  }
  num_cached_block.erase(request_id);
  return req_blocks;
}

void SingleTypeKVCacheManager::free(const std::string& request_id) {
  std::vector<KVCacheBlock*> blocks = pop_blocks_for_free(request_id);
  // Free blocks in reverse order so tail blocks are freed first.
  std::reverse(blocks.begin(), blocks.end());
  block_pool.free_blocks(blocks);
}

void SingleTypeKVCacheManager::_remove_blocks_in_range(
    const std::string& request_id, int first_block, int last_block) {
  auto it = req_to_blocks.find(request_id);
  if (it == req_to_blocks.end()) {
    return;
  }
  if (first_block >= last_block) {
    return;
  }
  std::vector<KVCacheBlock*>& blocks = it->second;
  last_block = std::min(last_block, static_cast<int>(blocks.size()));

  std::vector<KVCacheBlock*> freed;
  for (int i = last_block - 1; i >= first_block; --i) {
    if (blocks[i] == _null_block) {
      break;
    }
    freed.push_back(blocks[i]);
    blocks[i] = _null_block;
  }
  if (!freed.empty()) {
    block_pool.free_blocks(freed);
  }
}

void SingleTypeKVCacheManager::remove_skipped_blocks(
    const std::string& request_id, int total_computed_tokens,
    std::optional<int> /*num_prompt_tokens*/) {
  int num_skipped_tokens = get_num_skipped_tokens(total_computed_tokens);
  if (num_skipped_tokens <= 0) {
    // All tokens are inside the attention window (typical full attention).
    return;
  }
  std::vector<KVCacheBlock*>& blocks = req_to_blocks[request_id];
  int num_skipped_blocks = num_skipped_tokens / block_size;
  num_skipped_blocks =
      std::min(num_skipped_blocks, static_cast<int>(blocks.size()));
  _remove_blocks_in_range(request_id, 0, num_skipped_blocks);
}

int SingleTypeKVCacheManager::get_num_skipped_tokens(int /*num_computed_tokens*/) {
  return 0;
}

void SingleTypeKVCacheManager::new_step_starts() {}

// ---------------------------------------------------------------------------
// FullAttentionManager
// ---------------------------------------------------------------------------

std::vector<std::vector<KVCacheBlock*>>
FullAttentionManager::find_longest_cache_hit(
    const std::vector<BlockHash>& block_hashes, int max_length,
    const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
    const KVCacheSpec& kv_cache_spec, bool drop_eagle_block,
    int alignment_tokens, int dcp_world_size, int pcp_world_size) {
  assert(kv_cache_spec.kind() == KVCacheSpecKind::kFullAttention &&
         "FullAttentionManager can only be used for full attention groups");
  std::vector<std::vector<KVCacheBlock*>> computed_blocks(
      kv_cache_group_ids.size());
  int block_size = kv_cache_spec.block_size;
  if (dcp_world_size * pcp_world_size > 1) {
    block_size *= dcp_world_size * pcp_world_size;
  }
  int max_num_blocks = max_length / block_size;
  int n = std::min<int>(max_num_blocks, static_cast<int>(block_hashes.size()));
  for (int i = 0; i < n; ++i) {
    // block_hashes is a chain: on the first miss, later blocks are not computed.
    std::optional<std::vector<KVCacheBlock*>> cached_block =
        block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids);
    if (cached_block.has_value()) {
      for (size_t g = 0; g < computed_blocks.size(); ++g) {
        computed_blocks[g].push_back((*cached_block)[g]);
      }
    } else {
      break;
    }
  }
  if (drop_eagle_block && !computed_blocks[0].empty()) {
    for (auto& computed : computed_blocks) {
      computed.pop_back();
    }
  }
  while (block_size != alignment_tokens &&
         static_cast<int>(computed_blocks[0].size()) * block_size %
                 alignment_tokens !=
             0) {
    for (auto& computed : computed_blocks) {
      computed.pop_back();
    }
  }
  return computed_blocks;
}

int FullAttentionManager::get_num_common_prefix_blocks(
    const std::string& running_request_id) {
  const std::vector<KVCacheBlock*>& blocks = req_to_blocks[running_request_id];
  int num_common_blocks = 0;
  int num_requests = static_cast<int>(req_to_blocks.size());
  for (const KVCacheBlock* block : blocks) {
    if (block->ref_cnt == num_requests) {
      ++num_common_blocks;
    } else {
      break;
    }
  }
  return num_common_blocks;
}

// ---------------------------------------------------------------------------
// MambaManager
// ---------------------------------------------------------------------------

MambaManager::MambaManager(std::shared_ptr<KVCacheSpec> kv_cache_spec,
                           BlockPool& block_pool, bool enable_caching,
                           int kv_cache_group_id, int scheduler_block_size)
    : SingleTypeKVCacheManager(kv_cache_spec, block_pool, enable_caching,
                               kv_cache_group_id, scheduler_block_size) {
  const auto* mamba = dynamic_cast<const MambaSpec*>(kv_cache_spec.get());
  assert(mamba != nullptr && "MambaManager requires a MambaSpec");
  mamba_cache_mode = mamba->mamba_cache_mode;
  num_speculative_blocks = mamba->num_speculative_blocks;
}

std::vector<std::vector<KVCacheBlock*>> MambaManager::find_longest_cache_hit(
    const std::vector<BlockHash>& block_hashes, int max_length,
    const std::vector<int>& kv_cache_group_ids, BlockPool& block_pool,
    const KVCacheSpec& kv_cache_spec, bool /*drop_eagle_block*/,
    int alignment_tokens, int dcp_world_size, int pcp_world_size) {
  assert(kv_cache_spec.kind() == KVCacheSpecKind::kMamba &&
         "MambaManager can only be used for mamba groups");
  assert(dcp_world_size == 1 && "DCP not support mamba now.");
  assert(pcp_world_size == 1 && "PCP not support mamba now.");
  (void)dcp_world_size;
  (void)pcp_world_size;
  std::vector<std::vector<KVCacheBlock*>> computed_blocks(
      kv_cache_group_ids.size());

  int block_size = kv_cache_spec.block_size;
  int max_num_blocks = max_length / block_size;
  // Search from right to left and early stop when a match is found.
  for (int i = max_num_blocks - 1; i >= 0; --i) {
    if (i >= static_cast<int>(block_hashes.size())) {
      continue;
    }
    std::optional<std::vector<KVCacheBlock*>> cached_block =
        block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids);
    if (cached_block.has_value()) {
      // Mamba prefix caching aligns block_size across attention and mamba
      // layers so the hit length is block-aligned.
      if (block_size != alignment_tokens &&
          (i + 1) * block_size % alignment_tokens != 0) {
        continue;
      }
      for (size_t g = 0; g < computed_blocks.size(); ++g) {
        // The hit-length logic assumes hit_length ==
        // len(hit_blocks[0]) * other_block_size, so pad with dummy nulls up to
        // i, then append the single state block.
        computed_blocks[g].insert(computed_blocks[g].end(), i,
                                  block_pool.null_block);
        computed_blocks[g].push_back((*cached_block)[g]);
      }
      break;  // just need the last match — early stopping.
    }
  }
  return computed_blocks;
}

std::optional<std::vector<bool>> MambaManager::reachable_block_mask(
    int start_block, int end_block, std::optional<int> alignment_tokens,
    std::optional<int> retention_interval,
    std::optional<int> num_prompt_tokens) {
  if (!retention_interval.has_value() || !alignment_tokens.has_value()) {
    // Dense caching (default) or no alignment constraint imposed.
    return std::nullopt;
  }
  std::vector<bool> mask(static_cast<size_t>(end_block - start_block), false);

  // (1) Segment-boundary states. A Mamba hit needs the single state block
  // ending on the boundary. Block i ends at token (i + 1) * block_size.
  std::optional<int> segment_tokens =
      *retention_interval == 0 ? std::nullopt : retention_interval;
  if (segment_tokens.has_value()) {
    int per_segment = *segment_tokens / block_size;
    if (per_segment <= 1) {
      // Interval at/below the block size: every block is a boundary.
      return std::nullopt;
    }
    int first_boundary =
        (start_block + per_segment) / per_segment * per_segment - 1;
    for (int i = first_boundary - start_block;
         i < static_cast<int>(mask.size()); i += per_segment) {
      mask[static_cast<size_t>(i)] = true;
    }
  }

  // (2) Replay boundary: keep the state on the latest fine-aligned boundary.
  if (num_prompt_tokens.has_value()) {
    int latest =
        (*num_prompt_tokens - 1) / *alignment_tokens * *alignment_tokens;
    int boundary_block = latest / block_size - 1;
    if (start_block <= boundary_block && boundary_block < end_block) {
      mask[static_cast<size_t>(boundary_block - start_block)] = true;
    }
  }

  return mask;
}

void MambaManager::remove_skipped_blocks(const std::string& request_id,
                                         int num_computed_tokens,
                                         std::optional<int> num_prompt_tokens) {
  // With async scheduling, num_computed_tokens can include draft tokens that
  // may be rejected; assume all rejected so we don't free blocks we may need.
  num_computed_tokens = std::max(0, num_computed_tokens - num_speculative_blocks);

  SingleTypeKVCacheManager::remove_skipped_blocks(request_id, num_computed_tokens,
                                                  num_prompt_tokens);
  if (mamba_cache_mode == "align") {
    // last_state_block_idx refers to the block allocated two steps ago; the
    // block from the previous step copies state into the current block, so the
    // earlier block is no longer needed and is freed here.
    auto it = last_state_block_idx.find(request_id);
    if (it != last_state_block_idx.end() &&
        it->second < cdiv(num_computed_tokens, block_size) - 1) {
      std::vector<KVCacheBlock*>& blocks = req_to_blocks[request_id];
      int idx = it->second;
      if (blocks[static_cast<size_t>(idx)] != _null_block) {
        block_pool.free_blocks({blocks[static_cast<size_t>(idx)]});
        blocks[static_cast<size_t>(idx)] = _null_block;
      }
    }
  }
}

int MambaManager::get_num_common_prefix_blocks(
    const std::string& /*running_request_id*/) {
  // Cascade attention is not supported by mamba.
  return 0;
}

int MambaManager::get_num_blocks_to_allocate(
    const std::string& request_id, int num_tokens,
    const std::vector<KVCacheBlock*>& new_computed_blocks,
    int total_computed_tokens, int num_tokens_main_model,
    bool apply_admission_cap) {
  if (!new_computed_blocks.empty() &&
      new_computed_blocks.back()->block_hash().has_value() &&
      cached_blocks_this_step.count(*new_computed_blocks.back()->block_hash()) !=
          0) {
    // Mamba can't rely on blocks generated by other requests in the current
    // step. Return num_gpu_blocks + 1 so the manager defers this request.
    return static_cast<int>(block_pool.num_gpu_blocks) + 1;
  }
  if (mamba_cache_mode != "align") {
    if (num_speculative_blocks > 0) {
      num_tokens += block_size * num_speculative_blocks;
    }
    return SingleTypeKVCacheManager::get_num_blocks_to_allocate(
        request_id, num_tokens, new_computed_blocks, total_computed_tokens,
        num_tokens_main_model, apply_admission_cap);
  }
  // align mode: no blocks for lookahead tokens (they break the alignment).
  num_tokens = num_tokens_main_model;
  int num_required_blocks = cdiv(num_tokens, block_size) + num_speculative_blocks;
  int num_new_blocks = num_required_blocks -
                       static_cast<int>(new_computed_blocks.size()) -
                       static_cast<int>(req_to_blocks[request_id].size());
  if (num_new_blocks > 0) {
    if (_allocated_block_reqs.count(request_id) != 0) {
      // Old request: at most 1 more block (reuse the speculative blocks).
      num_new_blocks = 1;
    } else {
      // First prefill: 1 running-state block + the speculative blocks.
      num_new_blocks = 1 + num_speculative_blocks;
    }
  }
  int num_evictable_computed_blocks =
      _get_num_evictable_blocks(new_computed_blocks);
  return num_new_blocks + num_evictable_computed_blocks;
}

std::vector<KVCacheBlock*> MambaManager::allocate_new_blocks(
    const std::string& request_id, int num_tokens, int num_tokens_main_model) {
  if (mamba_cache_mode != "align") {
    if (num_speculative_blocks > 0) {
      num_tokens += block_size * num_speculative_blocks;
    }
    return SingleTypeKVCacheManager::allocate_new_blocks(request_id, num_tokens,
                                                         num_tokens_main_model);
  }
  // align mode.
  num_tokens = num_tokens_main_model;
  std::vector<KVCacheBlock*>& req_blocks = req_to_blocks[request_id];
  int num_required_blocks = cdiv(num_tokens, block_size) + num_speculative_blocks;
  if (num_required_blocks <= static_cast<int>(req_blocks.size())) {
    return {};
  }
  int prev_block_len = static_cast<int>(req_blocks.size());
  bool blocks_allocated = _allocated_block_reqs.count(request_id) != 0;
  // Record the last state block.
  if (blocks_allocated) {
    // The running state is always at the last (1 + num_speculative_blocks)
    // block.
    last_state_block_idx[request_id] =
        prev_block_len - 1 - num_speculative_blocks;
  } else if (prev_block_len > 0) {
    // A new request that hits the prefix cache: the last block saves the state.
    last_state_block_idx[request_id] = prev_block_len - 1;
  }

  int num_skipped_blocks = num_required_blocks - num_speculative_blocks - 1;
  // null blocks
  if (prev_block_len < num_skipped_blocks) {
    req_blocks.insert(req_blocks.end(),
                      static_cast<size_t>(num_skipped_blocks - prev_block_len),
                      _null_block);
  }

  if (blocks_allocated) {
    // Reuse previous speculative blocks in this step.
    for (int block_idx = prev_block_len - num_speculative_blocks;
         block_idx < prev_block_len; ++block_idx) {
      if (block_idx < num_skipped_blocks) {
        req_blocks.push_back(req_blocks[static_cast<size_t>(block_idx)]);
        req_blocks[static_cast<size_t>(block_idx)] = _null_block;
      } else {
        break;
      }
    }
  }
  int num_new_blocks = num_required_blocks - static_cast<int>(req_blocks.size());
  if (blocks_allocated) {
    assert(num_new_blocks <= 1);
  } else {
    assert(num_new_blocks <= num_speculative_blocks + 1);
  }
  std::vector<KVCacheBlock*> new_blocks = block_pool.get_new_blocks(num_new_blocks);
  req_blocks.insert(req_blocks.end(), new_blocks.begin(), new_blocks.end());
  _allocated_block_reqs.insert(request_id);
  return std::vector<KVCacheBlock*>(
      req_blocks.begin() + prev_block_len, req_blocks.end());
}

std::vector<KVCacheBlock*> MambaManager::pop_blocks_for_free(
    const std::string& request_id) {
  if (mamba_cache_mode == "align") {
    _allocated_block_reqs.erase(request_id);
    last_state_block_idx.erase(request_id);
  }
  return SingleTypeKVCacheManager::pop_blocks_for_free(request_id);
}

int MambaManager::get_num_skipped_tokens(int num_computed_tokens) {
  // Mamba only keeps the state of the last computed token.
  return num_computed_tokens - 1;
}

void MambaManager::cache_blocks(const Request& request, int num_tokens,
                                std::optional<int> retention_interval) {
  auto before_it = num_cached_block.find(request.request_id);
  int num_cached_blocks_before =
      before_it != num_cached_block.end() ? before_it->second : 0;
  SingleTypeKVCacheManager::cache_blocks(request, num_tokens, retention_interval);
  auto after_it = num_cached_block.find(request.request_id);
  int num_cached_blocks_after =
      after_it != num_cached_block.end() ? after_it->second : 0;
  if (num_cached_blocks_after > num_cached_blocks_before) {
    const std::vector<KVCacheBlock*>& blocks = req_to_blocks[request.request_id];
    for (int i = num_cached_blocks_before; i < num_cached_blocks_after; ++i) {
      KVCacheBlock* block = blocks[static_cast<size_t>(i)];
      // Skip null blocks (align-mode skipped states) and blocks not cached this
      // step (sparse retention leaves intermediate snapshots without a hash).
      if (block->is_null || !block->block_hash().has_value()) {
        continue;
      }
      cached_blocks_this_step.insert(*block->block_hash());
    }
  }
}

void MambaManager::new_step_starts() { cached_blocks_this_step.clear(); }

}  // namespace vllm::v1
