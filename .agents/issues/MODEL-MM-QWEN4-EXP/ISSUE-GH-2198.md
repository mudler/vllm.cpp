ID: ISSUE-GH-2198
Title: **W4's QSA comments cited `tokens_per_state`, a field with ZERO hits over the pinned vLLM tree, and the wave writing the KV-cache spec is exactly who would have gone looking for it.** Fixed IN FLOW under W5c-1 of [#2031](https://github.com/mudler/vllm.cpp/issues/2031). `grep -rn tokens_per_state` over `/home/mudler/_git/vllm/vllm/` at the parity pin `5559679229` returns nothing tree-wide, and neither does a search for the docstring the comments quoted ("Ints > 1 compress multiple tokens into one state"); the anchor they cited, `v1/attention/backends/mla/indexer.py:624-628`, is `_prepare_decode_tensors` and is unrelated to KV sizing. The real field is **`compress_ratio`** — `vllm/v1/kv_cache_interface.py:386` declares it defaulted to 1, `:393-395` is `storage_block_size = block_size // compress_ratio`, `:617` and `:624-625` repeat the pair on `SlidingWindowMLASpec`, and `:424-435` is `MLAAttentionSpec.merge` asserting ONE `compress_ratio` per KV group. This tree was already correct where it matters (`include/vllm/v1/kv_cache_interface.h` spells it `compress_ratio`), so the defect was a CITATION and never a number: the two sites are `src/vllm/model_executor/models/qwen4_exp_qsa.h`'s port-map comment and its `QsaSideCacheSpec` doc comment, both of which now cite `compress_ratio` with the three anchors above and record what was wrong so the correction is not re-derived. `QsaSideCacheSpec::tokens_per_state` KEEPS its name deliberately — it is a LOCAL field with no upstream referent whose arithmetic is right (64 B/token/layer at bf16, pinned by `tests/vllm/models/test_qwen4_exp_qsa.cpp`) and identical to `MLAAttentionSpec::real_page_size_bytes()`, so renaming it would churn W4's TU and suite to fix a citation the comments now carry; a comment beside the field says it has no upstream referent. Found while scoping W5c, whose `MLAAttentionSpec` third group is built with `compress_ratio=4` and whose `block_size % compress_ratio` refusal exists because `storage_block_size()` truncates in silence
Row: MODEL-MM-QWEN4-EXP
State: UNKNOWN
Kind: bug
GitHub: 2198
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:861`

### Frozen archive evidence

> | [#2198](https://github.com/mudler/vllm.cpp/issues/2198) | `MODEL-MM-QWEN4-EXP` | **W4's QSA comments cited `tokens_per_state`, a field with ZERO hits over the pinned vLLM tree, and the wave writing the KV-cache spec is exactly who would have gone looking for it.** Fixed IN FLOW under W5c-1 of [#2031](https://github.com/mudler/vllm.cpp/issues/2031). `grep -rn tokens_per_state` over `/home/mudler/_git/vllm/vllm/` at the parity pin `5559679229` returns nothing tree-wide, and neither does a search for the docstring the comments quoted ("Ints > 1 compress multiple tokens into one state"); the anchor they cited, `v1/attention/backends/mla/indexer.py:624-628`, is `_prepare_decode_tensors` and is unrelated to KV sizing. The real field is **`compress_ratio`** — `vllm/v1/kv_cache_interface.py:386` declares it defaulted to 1, `:393-395` is `storage_block_size = block_size // compress_ratio`, `:617` and `:624-625` repeat the pair on `SlidingWindowMLASpec`, and `:424-435` is `MLAAttentionSpec.merge` asserting ONE `compress_ratio` per KV group. This tree was already correct where it matters (`include/vllm/v1/kv_cache_interface.h` spells it `compress_ratio`), so the defect was a CITATION and never a number: the two sites are `src/vllm/model_executor/models/qwen4_exp_qsa.h`'s port-map comment and its `QsaSideCacheSpec` doc comment, both of which now cite `compress_ratio` with the three anchors above and record what was wrong so the correction is not re-derived. `QsaSideCacheSpec::tokens_per_state` KEEPS its name deliberately — it is a LOCAL field with no upstream referent whose arithmetic is right (64 B/token/layer at bf16, pinned by `tests/vllm/models/test_qwen4_exp_qsa.cpp`) and identical to `MLAAttentionSpec::real_page_size_bytes()`, so renaming it would churn W4's TU and suite to fix a citation the comments now carry; a comment beside the field says it has no upstream referent. Found while scoping W5c, whose `MLAAttentionSpec` third group is built with `compress_ratio=4` and whose `block_size % compress_ratio` refusal exists because `storage_block_size()` truncates in silence | bug |

## Resolution

-
