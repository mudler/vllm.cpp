ID: ISSUE-LOCAL-01M2EF43HVZ05H0HR61R4J0BA2
Title: Measure what a real DeepSeek-V4-Flash-Vision checkpoint does now that the fp8_ds_mla refusal no longer fires on auto
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

The W8 fp8_ds_mla bridge (#2455) landed and ApplyCacheDType now short-circuits on resolved.is_auto, so the MLA cache-dtype refusal at kv_cache_interface.cpp:398 no longer fires on the default path. tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp is now a FALSE RED asserting a refusal that correctly stopped firing, and whether a real 82 GB UD-IQ1_S checkpoint now serves text through vllm-cli is unmeasured.

## Resolution

PARTIAL; the issue stays OPEN because the real-checkpoint legs are still running. MEASURED 2026-09-13 on base 7a62a7fca. (1) The fp8_ds_mla refusal stopped firing because RESOLUTION changed, not because the guard was widened: ApplyCacheDType now returns immediately on resolved.is_auto (W8 slice 6), so RetypeAttentionSpec is never called on the default path, and it still refuses every EXPLICIT --kv-cache-dtype. (2) The wall is the STRICT EQUALITY at kv_cache_coordinator.cpp:386 in the HYBRID coordinator, not the :350 predicate in the unitary one. DeepSeek-V4 registers is_hybrid=false/has_inner_state=false, so prefix caching resolves ON, seven groups take HybridKVCacheCoordinator, and :386 is evaluated. For {256,256,256,64,4,4,8} at scheduler 256 / hash 4 the divisibility guards at :138, :140 and :382 all PASS; only :386 fails. (3) :386 is a bare assert, so -DNDEBUG DELETES it: Release builds construct the engine and proceed into the DEFERRED BlockHashListWithBlockSize path with the invariant violated, while a Debug build of the same fixture ABORTS with SIGABRT at :386. Any load result on this topology is meaningless unless it states CMAKE_BUILD_TYPE and whether NDEBUG was defined. A NAMED REFUSAL would be a better guard than an assert, because a refusal survives NDEBUG; that change is recommended, not made here. (4) test_deepseek_v4_multigroup_kv was a FALSE RED and is inverted rather than deleted: case (2) asserts the new default-path construction, case (3) is NEW and still proves the fp8_ds_mla refusal fires by name on an explicit override. 3 cases / 19 assertions SUCCESS, binary md5 7d4751983fa0338f73968cc2f05bfa1e, proven changed.
