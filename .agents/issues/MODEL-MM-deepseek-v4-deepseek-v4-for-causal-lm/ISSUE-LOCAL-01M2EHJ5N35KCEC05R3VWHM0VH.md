ID: ISSUE-LOCAL-01M2EHJ5N35KCEC05R3VWHM0VH
Title: test_deepseek_v4_multigroup_kv REVERSES with NDEBUG: the multi-group construction case aborts on a default (no-CMAKE_BUILD_TYPE) build
Row: MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: 2026-09-13

## Problem

Case (2) of tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp asserts that a multi-group DeepSeek-V4 checkpoint CONSTRUCTS on the default path. That expectation holds only under -DNDEBUG. The repository's default configure (cmake -S . -B build -G Ninja, CMAKE_BUILD_TYPE empty) defines no NDEBUG, so the bare assert at src/vllm/v1/core/kv_cache_coordinator.cpp:386 (g.kv_cache_spec->block_size == hash_block_size, the DEFERRED hash-granularity marker) is live and the case SIGABRTs, killing the runner. The suite's verdict therefore depends on -DNDEBUG: green in Release, red on a default checkout. A gate whose answer is a build flag is not a gate. The suite must express the same truth in both configurations without deleting or widening the production assert, which belongs to the owed hash-granularity port.

## Resolution

FIXED 2026-09-13 in tests/vllm/entrypoints/test_deepseek_v4_multigroup_kv.cpp. The load that case (2) performs moved into a CHILD PROCESS: a skip-decorated case dsv4_multigroup_construct_child re-execs this binary by name (--no-skip --test-case=), prints a CONSTRUCT= marker, and the parent reads its exit status plus its captured stderr, the shape tests/vllm/v1/test_none_hash_determinism.cpp and tests/vllm/entrypoints/openai/test_serve_hf_model.cpp:371 already use. An abort therefore becomes an observation instead of killing the runner. Both builds run the SAME case and assert the SAME invariant, that NOTHING refuses this topology BY NAME; one constexpr bool kDeferralAssertLive, the only NDEBUG read in the suite, selects which observed death follows, because the product genuinely differs. The production assert at src/vllm/v1/core/kv_cache_coordinator.cpp:386 is UNTOUCHED, and nothing was deleted or widened. RED FIRST, on the HEAD source: default build (CMAKE_BUILD_TYPE empty, NDEBUG present False) SIGABRT at kv_cache_coordinator.cpp:386, 'test cases: 2 | 1 passed | 1 failed | 1 skipped', exit 134, binary md5 aef4caae93fb5ea02a38cf0a007cf129; the same source in Release 'test cases: 3 | 3 passed | 0 failed', md5 ce0a6c3b36b8ad72c1853b8bd84d260f. GREEN AFTER, same two build dirs: default 'test cases: 3 | 3 passed | 0 failed | 1 skipped', assertions 27 | 27 passed, md5 88c1c8a619009f100f5ca23d0302f92d (proven changed from aef4caae93fb5ea02a38cf0a007cf129); Release 'test cases: 3 | 3 passed | 0 failed | 1 skipped', assertions 27 | 27 passed, md5 44dc0f84080f2168aa0b2335f9da65ea (proven changed from ce0a6c3b36b8ad72c1853b8bd84d260f). Cases (1) and (3) are unchanged, so the fp8_ds_mla refusal on an explicit --kv-cache-dtype is still pinned by name. The suite's header comment also stopped claiming that :386 is the only wall: the rc-job b622dd45-d763-41d6-9fe3-c1822100163d measurement on dgx:gpu0 shows a real 82 GB UD-IQ1_S checkpoint now constructing and dying later in DeepseekV4Model::ForwardDevice at src/vllm/model_executor/models/deepseek_v4.cpp:4658, which the comment now records.
