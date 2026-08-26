// See include/vllm/v1/core/hybrid_kv_budget.h for the port anchors and the
// reasoning. This file is the arithmetic only.
#include "vllm/v1/core/hybrid_kv_budget.h"

#include <algorithm>

namespace vllm::v1 {
namespace {

// Upstream `cdiv` (vllm/utils/math_utils.py): ceiling division on positives.
int64_t CeilDiv(int64_t x, int64_t y) { return (x + y - 1) / y; }

}  // namespace

HybridKvBudget ComputeHybridKvBudget(const KVCacheConfig& kv_cfg,
                                     int kernel_block_alignment) {
  HybridKvBudget budget;

  const MambaSpec* mamba = nullptr;
  const AttentionSpec* attn = nullptr;
  for (const KVCacheGroupSpec& group : kv_cfg.kv_cache_groups) {
    const KVCacheSpec* spec = group.kv_cache_spec.get();
    if (spec == nullptr) continue;
    const KVCacheSpecKind kind = spec->kind();
    if (kind == KVCacheSpecKind::kMamba) {
      if (mamba == nullptr) mamba = static_cast<const MambaSpec*>(spec);
      continue;
    }
    // Every non-Mamba kind this tree builds derives from AttentionSpec, and the
    // runner's own target selection keeps the FIRST non-eagle attention group
    // (gpu/runner.cpp, `full_attn_group_id_`). Mirror that choice so the page
    // this budget unifies against is the page the runner allocates.
    if (attn == nullptr && !group.is_eagle_group &&
        (kind == KVCacheSpecKind::kFullAttention ||
         kind == KVCacheSpecKind::kMlaAttention ||
         kind == KVCacheSpecKind::kSlidingWindow ||
         kind == KVCacheSpecKind::kSlidingWindowMla ||
         kind == KVCacheSpecKind::kChunkedLocalAttention)) {
      attn = static_cast<const AttentionSpec*>(spec);
    }
  }

  // No recurrent state: nothing to bound, and the caller must clamp nothing.
  if (mamba == nullptr) return budget;

  budget.mamba_page_bytes = mamba->page_size_bytes();
  budget.slots_per_seq = 1 + mamba->num_speculative_blocks;

  // A pure-recurrent model has no attention page to unify against. Upstream's
  // unification has nothing to do there either; report unbounded rather than
  // invent a page. (Owed, see the row's spec.)
  if (attn == nullptr || budget.mamba_page_bytes <= 0 ||
      budget.slots_per_seq <= 0 || kv_cfg.num_blocks <= 0) {
    return budget;
  }

  const int64_t attn_block_tokens = attn->storage_block_size();
  if (attn_block_tokens <= 0) return budget;
  // Upstream builds a `block_size=1` spec to read this (`interface.py:840-851`),
  // which is the UNPADDED per-token cost; `real_page_size_bytes()` is that same
  // formula before any `page_size_padded` override, so divide it rather than
  // `page_size_bytes()`.
  budget.attn_bytes_per_token =
      attn->real_page_size_bytes() / attn_block_tokens;
  if (budget.attn_bytes_per_token <= 0) return budget;

  // interface.py:875-882 -- `max(min(supported kernel block sizes),
  // cache_config.block_size)`. Our `cache_config.block_size` is the attention
  // spec's own block length.
  const int64_t align = std::max<int64_t>(
      std::max(kernel_block_alignment, 1), attn->block_size);
  // interface.py:896-901.
  budget.unified_block_tokens =
      align * CeilDiv(budget.mamba_page_bytes, align * budget.attn_bytes_per_token);
  if (budget.unified_block_tokens <= 0) return budget;

  // The same bytes, restated in unified pages. `num_blocks * block_size` tokens
  // per attention layer is what our pool holds; one upstream page holds
  // `unified_block_tokens` of them.
  budget.unified_num_blocks =
      (static_cast<int64_t>(kv_cfg.num_blocks) * attn_block_tokens) /
      budget.unified_block_tokens;

  const int64_t seats = budget.unified_num_blocks / budget.slots_per_seq;
  // Clamp into int; a pool this large is already unbounded in practice.
  budget.max_state_seqs = static_cast<int>(
      std::min<int64_t>(seats, static_cast<int64_t>(1) << 30));
  return budget;
}

int ClampMaxNumSeqsToStateBudget(int configured, const HybridKvBudget& budget) {
  if (budget.max_state_seqs == kStateSeqsUnbounded) return configured;
  // Never below one seat. A budget that holds state for zero sequences is a
  // refusal `check_enough_state_memory` owns; silently serving nothing is not a
  // disposition this engine has.
  const int seats = std::max(budget.max_state_seqs, 1);
  return std::min(configured, seats);
}

}  // namespace vllm::v1
