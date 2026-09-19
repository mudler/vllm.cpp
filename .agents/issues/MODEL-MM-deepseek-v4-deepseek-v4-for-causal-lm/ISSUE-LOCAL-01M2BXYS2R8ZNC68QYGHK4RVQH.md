ID: ISSUE-LOCAL-01M2BXYS2R8ZNC68QYGHK4RVQH
Title: vllm serve aborts on DeepSeek-V4: the engine hands the KV coordinator hash_block_size 256 for a group paged at 64
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

test_serve_deepseek_v4_mm dies with SIGABRT inside UnitaryKVCacheCoordinator (src/vllm/v1/core/kv_cache_coordinator.cpp:350, assert !enable_caching || hash_block_size == block_size_). Measured on a CPU Debug build at 5f86f9d67 with a temporary probe: groups=1, enable_caching=1, hash_block_size=256, block_size_=64, scheduler_block_size=256, group[0] block_size=64 over 3 layers. LoadedEngine resolves block_size_=256 from DeepseekV4's kv_block_size_floor and builds the KV config at 256, but MakeDeepseekV4KVCache publishes the SWA group at a fixed 64 (kSwaBlockSize, mirroring upstream sparse_swa.py self.block_size = 64); with the fixture's compress_ratios all zero that SWA group is the only group. model_loader.cpp then passes block_size_ (256) as the scheduler block size and scheduler.cpp:277 hardcodes hash_block_size = block_size, so the coordinator receives 256 for a 64-token group. Upstream does two things this tree does not: it re-derives cache_config.block_size = min(g.kv_cache_spec.block_size for g in kv_cache_groups) after the KV cache config is built (vllm/v1/engine/core.py:335-338 @ e126687a9a) and it resolves the pair with resolve_kv_cache_block_sizes and passes scheduler_block_size and hash_block_size separately to the Scheduler (core.py:158-170, scheduler.py:76). Our resolve_kv_cache_block_sizes is ported at src/vllm/v1/core/kv_cache_utils.cpp:640 but has no production caller.

## Resolution

Fixed 2026-09-12 on a CPU Debug build at 5f86f9d67. The engine now mirrors upstream's two-step derivation: LoadedEngine::ResolveSchedulerBlockSizes re-derives the cache block size as the minimum over the BUILT kv_cache_groups (vllm/v1/engine/core.py:335-338 @ e126687a9a) and then calls the already-ported resolve_kv_cache_block_sizes (core.py:158-160), whose result is threaded to the Scheduler as a separate scheduler_block_size and hash_block_size (core.py:158-170, scheduler.py:76,268-270); the request block hasher now takes hash_block_size, as upstream does at core.py:232. scheduler.cpp no longer hardcodes hash_block_size = block_size. RED BEFORE: test_serve_deepseek_v4_mm aborted with SIGABRT at kv_cache_coordinator.cpp:350, doctest reporting 'test cases: 2 | 0 passed | 2 failed', binary md5 8f1cf89ce2fd589b83a0019b9eb5b084. GREEN AFTER: 'test cases: 2 | 2 passed | 0 failed', binary md5 b7a5767653eb84b5d9cd3a50e8c08bff, so a different binary ran. Eleven neighbouring suites stay green: test_scheduler 48/48, test_scheduler_lpm 6/6, test_engine_core 6/6, test_kv_cache_coordinator 21/21, test_prefix_match_unit 8/8, test_prefix_cache_stats 12/12, test_loaded_engine_dense 30/30, test_dspark_draft_routing 7/7, test_deepseek_v4_mm_chat 8/8, test_deepseek_v4_mm_reach 20/20, test_deepseek_v4_mm_loader 17/17. The multi-group shape of a real Flash checkpoint (seven groups at 256/64/4/8) still cannot prefix-cache and is recorded under ## Owed in .agents/specs/deepseek-v4-flash-vision.md; that limit is pre-existing and not introduced here.
