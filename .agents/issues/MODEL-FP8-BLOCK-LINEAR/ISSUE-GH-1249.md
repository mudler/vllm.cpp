ID: ISSUE-GH-1249
Title: An f32 paged KV cache refuses every arm whose `v_proj` emits bf16. `vt::ReshapeAndCache`'s auto path requires `k`, `v`, `k_cache` and `v_cache` to share one float dtype (`src/vt/ops.cpp:3146`), and the CPU kernel behind it is a byte `memcpy` per token (`src/vt/cpu/cpu_cache.cpp:33-72`), so there is no conversion to widen. That mirrors upstream `reshape_and_cache_flash`, which torch types the same way, and it is correct for the production configuration, where the cache dtype follows the model dtype. It is wrong for `VT_KV_CACHE_F32=1`, which selects an f32 cache while every bf16 model's `v_proj` keeps emitting bf16 — so that lever is unusable on any bf16 arm, not merely on the block-wise FP8 one that found it. Found while wiring #1189 M4, whose G3 reachability case drives the bf16 cache and records in the source that it cannot drive the f32 one. Closing it is a `vt` semantic change (a converting store, or a refusal that names the lever) and needs its own spec and red-first test, so it is NOT fixed in M4's flow; owned by `MODEL-FP8-BLOCK-LINEAR` under its spec's `## Owed`
Row: MODEL-FP8-BLOCK-LINEAR
State: UNKNOWN
Kind: bug
GitHub: 1249
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:402`

### Frozen archive evidence

> | [#1249](https://github.com/mudler/vllm.cpp/issues/1249) | `MODEL-FP8-BLOCK-LINEAR` | An f32 paged KV cache refuses every arm whose `v_proj` emits bf16. `vt::ReshapeAndCache`'s auto path requires `k`, `v`, `k_cache` and `v_cache` to share one float dtype (`src/vt/ops.cpp:3146`), and the CPU kernel behind it is a byte `memcpy` per token (`src/vt/cpu/cpu_cache.cpp:33-72`), so there is no conversion to widen. That mirrors upstream `reshape_and_cache_flash`, which torch types the same way, and it is correct for the production configuration, where the cache dtype follows the model dtype. It is wrong for `VT_KV_CACHE_F32=1`, which selects an f32 cache while every bf16 model's `v_proj` keeps emitting bf16 — so that lever is unusable on any bf16 arm, not merely on the block-wise FP8 one that found it. Found while wiring #1189 M4, whose G3 reachability case drives the bf16 cache and records in the source that it cannot drive the f32 one. Closing it is a `vt` semantic change (a converting store, or a refusal that names the lever) and needs its own spec and red-first test, so it is NOT fixed in M4's flow; owned by `MODEL-FP8-BLOCK-LINEAR` under its spec's `## Owed` | bug |

## Resolution

-
