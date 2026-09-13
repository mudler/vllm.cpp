ID: ISSUE-LOCAL-01M2C0JBH6CNPYGQXNF0A1WG0H
Title: A multi-group DeepSeek-V4 cannot prefix-cache: the coordinator's equality assert is LATENT behind the fp8_ds_mla refusal
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-13
Closed: -

## Problem

MEASURED 2026-09-13, and this replaces the abort this issue originally asserted. That claim was a prediction and it is false.

WHAT IS TRUE. A real DeepSeek-V4-Flash checkpoint sets attention.compress_ratios, so MakeDeepseekV4KVCache publishes seven KV cache groups at block sizes {256, 256, 256, 64, 4, 4, 8}, and resolve_kv_cache_block_sizes resolves scheduler_block_size=256 (the LCM) and hash_block_size=4 (the GCD). Both halves are measured through the factory pointer the loader dereferences, in tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp.

WHAT IS FALSE. Such a checkpoint does NOT abort in HybridKVCacheCoordinator, because it never reaches it. ApplyCacheDType runs while kv_cfg_ is being initialized, which precedes scheduler_block_size_ and scheduler_ in the LoadedEngine constructor's initializer list, and RetypeAttentionSpec refuses any MLAAttentionSpec BY NAME at src/vllm/v1/kv_cache_interface.cpp:398: the fp8_ds_mla page formula landed without the store or the read. The compressed-latent groups are exactly the groups a non-zero compress_ratio adds, so the ratios that build the multi-group topology are also what trips that guard. A named refusal is a message, not an abort, so the serve-time outcome today is already acceptable — for a different reason than this issue first claimed.

WHY THE ASSERT IS UNREACHABLE RATHER THAN MERELY UNTRIGGERED. DeepSeek-V4 is the only architecture in this tree that publishes groups with differing block sizes; every other multi-group registry (glm5_next, kimi_linear, nemotron_h, qwen4_exp, qwen3_5_common) hands the same block_size variable to every group it publishes. So no production entry point can reach the coordinator's block_size == hash_block_size assert (src/vllm/v1/core/kv_cache_coordinator.cpp:386) or BlockPool's matching throw (block_pool.cpp:93,220) today.

WHAT A REAL CHECKPOINT NEEDS NEXT. #2455, the fp8_ds_mla store and read, owed to KV-DSV4-MULTICACHE W8. That work is what makes this path reachable at all. The hash-granularity port stays owed behind it and is NOT being landed unreached: a converting view no entry point can reach is dead code. The upstream shape it must mirror is recorded under ## Owed in .agents/specs/deepseek-v4-flash-vision.md so whoever lands W8 does not have to rediscover it.

## Resolution

-
