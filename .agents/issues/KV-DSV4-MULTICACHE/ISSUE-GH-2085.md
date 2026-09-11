ID: ISSUE-GH-2085
Title: **The multi-cache `PagedKvCache` view geometry contradicts the page it is built over.** Each buffer is `num_blocks * spec->page_size_bytes()` while its `FaDims` view is built from `spec->block_size`; for a spec whose page derives from a `storage_block_size` the two disagree. DeepSeek-V4's C4A latent (`block_size` 256, `compress_ratio` 4, so `storage_block_size` 64) has a **37440**-byte page while the view declares `{num_blocks, 256, 512}` = **131072** bytes per block, 3.5x what the page holds. `CheckKvCacheShape` cannot see it: it compares the backend's declared shape against that same view metadata, so it measures self-consistency rather than agreement with the allocation. **INERT today** -- `ModelRegistry::Forward` refuses a multi-cache index before any kernel reads a view. Found while reviewing [#2068](https://github.com/mudler/vllm.cpp/issues/2068). NOT fixed in flow and OWED to **W5** with the store path, listed under `## Owed` in `.agents/specs/kv-dsv4-multicache.md`, because resolving it is entangled with two things W3 cannot settle: the `fp8_ds_mla` 584 B/token layout is not expressible in `PagedKvCache` at all, and `tests/vllm/v1/worker/test_runner.cpp` pins `block_size == 256` for that entry as a literal that the resolution may have to contradict. Given its own `## Owed` entry rather than folded into the W4 non-uniform-`block_size` item, which is about pool budgeting (`KVBytesPerBlock` counting one page per layer) where this is about the view a kernel would index off
Row: KV-DSV4-MULTICACHE
State: UNKNOWN
Kind: bug
GitHub: 2085
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:799`

### Frozen archive evidence

> | [#2085](https://github.com/mudler/vllm.cpp/issues/2085) | `KV-DSV4-MULTICACHE` | **The multi-cache `PagedKvCache` view geometry contradicts the page it is built over.** Each buffer is `num_blocks * spec->page_size_bytes()` while its `FaDims` view is built from `spec->block_size`; for a spec whose page derives from a `storage_block_size` the two disagree. DeepSeek-V4's C4A latent (`block_size` 256, `compress_ratio` 4, so `storage_block_size` 64) has a **37440**-byte page while the view declares `{num_blocks, 256, 512}` = **131072** bytes per block, 3.5x what the page holds. `CheckKvCacheShape` cannot see it: it compares the backend's declared shape against that same view metadata, so it measures self-consistency rather than agreement with the allocation. **INERT today** -- `ModelRegistry::Forward` refuses a multi-cache index before any kernel reads a view. Found while reviewing [#2068](https://github.com/mudler/vllm.cpp/issues/2068). NOT fixed in flow and OWED to **W5** with the store path, listed under `## Owed` in `.agents/specs/kv-dsv4-multicache.md`, because resolving it is entangled with two things W3 cannot settle: the `fp8_ds_mla` 584 B/token layout is not expressible in `PagedKvCache` at all, and `tests/vllm/v1/worker/test_runner.cpp` pins `block_size == 256` for that entry as a literal that the resolution may have to contradict. Given its own `## Owed` entry rather than folded into the W4 non-uniform-`block_size` item, which is about pool budgeting (`KVBytesPerBlock` counting one page per layer) where this is about the view a kernel would index off | bug |

## Resolution

-
