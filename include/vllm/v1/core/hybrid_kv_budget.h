// How many sequences' RECURRENT state a KV budget holds (issue #1983).
//
// Ported from vllm @ 555967922:
//   vllm/platforms/interface.py::Platform.check_and_update_config   :853-935
//   vllm/v1/core/kv_cache_utils.py::_unify_hybrid_kv_cache_specs    :1073-1131
//   vllm/v1/core/kv_cache_utils.py::get_num_blocks                  :993-1011
//   vllm/v1/core/kv_cache_utils.py::_get_kv_cache_config_uniform_page_size
//                                                                   :1399-1416
//   vllm/v1/attention/backends/flash_attn.py
//     ::FlashAttentionBackend.get_supported_kernel_block_sizes      :83-84
//
// WHY THIS EXISTS. `GPUModelRunner::initialize_kv_cache` sizes the GDN state
// pool as `max_num_reqs * (num_spec + 1)` slots and allocates one conv and one
// SSM buffer per GDN layer from it, each zeroed at construction. So the
// allocation is a function of `--max-num-seqs`, and nothing bounds the product:
// not `--kv-cache-memory`, not `--num-blocks`, not `--gpu-memory-utilization`.
// At the Qwen3.8-27B geometry with `num_speculative_tokens = 8` one sequence
// costs 1.356 GiB and `--max-num-seqs 32` costs 43.40 GiB, resident before the
// first request arrives.
//
// `max_num_seqs` sizes NO allocation anywhere in vLLM. It is a scheduler cap.
// Upstream raises the attention block size until one attention page holds one
// mamba page, pads the mamba page up to match, and then draws BOTH from one
// budgeted pool whose tensors are `shared_by` one layer from each group -- so a
// mamba layer's state blocks and an attention layer's KV blocks come out of the
// same `num_blocks` identifier space, and the total is a function of available
// memory. This header ports that PROPERTY (the recurrent allocation is bounded
// by the KV budget) without porting the machinery (a KV-manager group with a
// block table, which is a campaign and not a fix).
//
// THE ARITHMETIC IS UPSTREAM'S, ON THE SPECS WE ALREADY BUILD:
//
//   mamba_page           = mamba_spec->page_size_bytes()      // one layer, one slot
//   attn_bytes_per_token = attn real page / storage_block_size
//   align                = max(16, attn_spec->block_size)     // interface.py:875-882
//   unified_block_tokens = align * cdiv(mamba_page, align * attn_bytes_per_token)
//   unified_num_blocks   = num_blocks * attn_block_size / unified_block_tokens
//   max_state_seqs       = unified_num_blocks / (1 + num_speculative_blocks)
//
// `unified_num_blocks` is a token-count identity, not an estimate: our pool
// holds `num_blocks * block_size` tokens per attention layer, one upstream page
// holds `unified_block_tokens` of them, so the same bytes buy that many upstream
// pages.
//
// NO LAYER COUNT APPEARS. Upstream's per-layer page equality cancels it, which
// is why this file does not read `KVCacheGroupSpec::layer_names` and therefore
// does not depend on the placeholder-layer-name repair owned by
// https://github.com/mudler/vllm.cpp/issues/1963 and
// https://github.com/mudler/vllm.cpp/issues/1966. It is also why this lives in
// its own translation unit: those two rows and this one share no edit surface.
//
// THE BOUND IS AN UPPER BOUND, DELIBERATELY. A running sequence also holds
// attention pages out of the same upstream pool, so upstream's effective
// concurrency is strictly below `unified_num_blocks / slots_per_seq`. Those are
// not charged here, because a STATIC cap tighter than upstream's DYNAMIC
// scheduler would refuse work upstream serves. This is a ceiling on an
// allocation, never a scheduler.
#ifndef VLLM_V1_CORE_HYBRID_KV_BUDGET_H_
#define VLLM_V1_CORE_HYBRID_KV_BUDGET_H_

#include <cstdint>

#include "vllm/v1/kv_cache_interface.h"

namespace vllm::v1 {

// The minimum kernel block size, in tokens, the alignment floor is taken from.
// Upstream reads it from the selected backend's own
// `get_supported_kernel_block_sizes()`; FlashAttention returns `MultipleOf(16)`
// (`vllm/v1/attention/backends/flash_attn.py:83-84`), and 16 is the floor across
// the CUDA backends. Named rather than inlined so a future per-backend seam has
// one place to replace.
inline constexpr int kMinKernelBlockTokens = 16;

// `max_state_seqs == kStateSeqsUnbounded` means "this configuration carries no
// bound", and a caller must then clamp NOTHING. Two shapes reach it, and they
// are different from `max_state_seqs == 0` (which would mean "not even one
// sequence fits" and is never produced):
//   * no Mamba group at all -- an attention-only model has no recurrent state,
//     so there is nothing to bound;
//   * a Mamba group with no attention group -- a pure-recurrent model has no
//     attention page to unify against, and upstream's unification has nothing
//     to do there either. Recorded as owed in the row's spec.
inline constexpr int kStateSeqsUnbounded = -1;

// The recurrent-state seat count a KV budget affords, plus every intermediate
// the derivation used. The intermediates are public because a bound whose
// reader cannot see WHAT was compared against WHAT is a bound nobody can audit
// (`.agents/verification.md`, "make the instrument say what it is measuring").
struct HybridKvBudget {
  // One GDN/Mamba layer, one state slot, in bytes: the spec's own page.
  int64_t mamba_page_bytes = 0;
  // One attention layer, one token, in bytes. Upstream's
  // `attn_page_size_1_token` (`interface.py:840-851`).
  int64_t attn_bytes_per_token = 0;
  // Upstream's raised `cache_config.block_size` (`interface.py:896-901`): the
  // block length at which one attention page holds one mamba page.
  int64_t unified_block_tokens = 0;
  // The paged pool restated in those pages. A token-count identity.
  int64_t unified_num_blocks = 0;
  // `1 + MambaSpec::num_speculative_blocks` -- the k+1 draft-timestep snapshots
  // a speculating request owns (`kv_cache_interface.py:713-718`).
  int slots_per_seq = 0;
  // The answer: how many sequences' recurrent state this budget holds, or
  // `kStateSeqsUnbounded`.
  int max_state_seqs = kStateSeqsUnbounded;
};

// Compute the budget from the config the runner allocates FROM, so the two
// cannot drift. `kernel_block_alignment` is upstream's
// `min(get_supported_kernel_block_sizes())`; pass `kMinKernelBlockTokens`
// unless a backend states otherwise.
//
// Reads the FIRST attention-kind group (mirroring the runner's own target
// selection, which keeps the first non-eagle full-attention/MLA group) and the
// FIRST Mamba group. Throws nothing that the specs' own `page_size_bytes()`
// does not already throw.
HybridKvBudget ComputeHybridKvBudget(
    const KVCacheConfig& kv_cfg,
    int kernel_block_alignment = kMinKernelBlockTokens);

// `min(configured, budget.max_state_seqs)`, with `kStateSeqsUnbounded` passing
// `configured` through. Never returns below 1: a budget that holds state for
// zero sequences is a REFUSAL the #371 guard owns
// (`check_enough_state_memory`), not a zero-seat engine.
int ClampMaxNumSeqsToStateBudget(int configured, const HybridKvBudget& budget);

}  // namespace vllm::v1

#endif  // VLLM_V1_CORE_HYBRID_KV_BUDGET_H_
