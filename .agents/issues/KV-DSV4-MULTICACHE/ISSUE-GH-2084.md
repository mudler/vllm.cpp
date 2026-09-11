ID: ISSUE-GH-2084
Title: **An EAGLE `AttentionSpec` group on a multi-cache topology gets NO buffer and NO refusal.** `initialize_kv_cache` excludes an eagle group from `attn_group_ids_` (`src/vllm/v1/worker/gpu/runner.cpp`, `is_attention_spec && !group.is_eagle_group`), but W3's narrowed refusal loop never tested `is_eagle_group` -- so such a group kept `why` empty, passed the refusal, and then received no buffer because the allocation loop iterates `attn_group_ids_`. That is a SUBSET of the published topology allocated in silence, verbatim what the refusal's own message says it prevents; the code comment enumerated "three shapes" and this is a fourth. Demonstrated rather than reasoned: setting `is_eagle_group=true` on the indexer-key group in `MakeMultiCacheKvConfig` yields `REQUIRE( 9 == 10 )` -- nine caches allocated instead of ten, no message. **Not reachable today**: `is_eagle_group` is set true in exactly one place in the tree (`tests/vllm/v1/worker/test_runner.cpp`), on a config that is not a multi-cache topology, and no registered factory sets it; the multi-cache path also refuses at the forward until W5, so no wrong tokens were possible. Found while reviewing [#2068](https://github.com/mudler/vllm.cpp/issues/2068). FIXED IN FLOW with W3: the refusal gains an eagle clause and `test_runner`'s refusal case gains a subcase that is RED without it (`CHECK_THROWS_AS ... did NOT throw at all`, 5 assertions failed) and green with it. **Refusing is the direction rather than dropping the `!group.is_eagle_group` filter**, because allocating a draft group as an ordinary named cache decides how speculation shares a multi-cache topology, and that decision belongs to the wave that gates the speculative path
Row: KV-DSV4-MULTICACHE
State: UNKNOWN
Kind: bug
GitHub: 2084
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:798`

### Frozen archive evidence

> | [#2084](https://github.com/mudler/vllm.cpp/issues/2084) | `KV-DSV4-MULTICACHE` | **An EAGLE `AttentionSpec` group on a multi-cache topology gets NO buffer and NO refusal.** `initialize_kv_cache` excludes an eagle group from `attn_group_ids_` (`src/vllm/v1/worker/gpu/runner.cpp`, `is_attention_spec && !group.is_eagle_group`), but W3's narrowed refusal loop never tested `is_eagle_group` -- so such a group kept `why` empty, passed the refusal, and then received no buffer because the allocation loop iterates `attn_group_ids_`. That is a SUBSET of the published topology allocated in silence, verbatim what the refusal's own message says it prevents; the code comment enumerated "three shapes" and this is a fourth. Demonstrated rather than reasoned: setting `is_eagle_group=true` on the indexer-key group in `MakeMultiCacheKvConfig` yields `REQUIRE( 9 == 10 )` -- nine caches allocated instead of ten, no message. **Not reachable today**: `is_eagle_group` is set true in exactly one place in the tree (`tests/vllm/v1/worker/test_runner.cpp`), on a config that is not a multi-cache topology, and no registered factory sets it; the multi-cache path also refuses at the forward until W5, so no wrong tokens were possible. Found while reviewing [#2068](https://github.com/mudler/vllm.cpp/issues/2068). FIXED IN FLOW with W3: the refusal gains an eagle clause and `test_runner`'s refusal case gains a subcase that is RED without it (`CHECK_THROWS_AS ... did NOT throw at all`, 5 assertions failed) and green with it. **Refusing is the direction rather than dropping the `!group.is_eagle_group` filter**, because allocating a draft group as an ordinary named cache decides how speculation shares a multi-cache topology, and that decision belongs to the wave that gates the speculative path | bug |

## Resolution

-
