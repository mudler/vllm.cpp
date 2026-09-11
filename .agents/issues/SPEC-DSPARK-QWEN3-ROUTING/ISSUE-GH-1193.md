ID: ISSUE-GH-1193
Title: A Qwen3 DSpark draft declaring `architectures=["DSparkDraftModel"]` with `model_type` `qwen3` has no route. The pin forces every DSpark draft that is not `Qwen3DSparkModel` or `Gemma4DSparkModel` onto `model_type` `deepseek_v4` (`vllm/config/speculative.py:934-944` @ `555967922`), and vLLM PR 52197 (merged 2026-08-17 at `7075ddac`) replaced that with a leading branch normalizing the pair to `Qwen3DSparkModel`. We diverge from BOTH: the forced rewrite was never ported, so nothing in `src/vllm/entrypoints/model_loader.cpp` reads a draft config's `architectures` key at all, and `SpeculativeConfig::IsDsparkDraft` (`include/vllm/config/speculative.h:120-136`) has no production caller — every reference outside its header is in `tests/vllm/config/test_speculative_dspark.cpp:132-140`, and `ResolveSpecConfig` branches on `cli.method` alone. The checkpoint is real and gateable here: `RadixArk/Qwen3.8-27B-DSpark` at revision `85ef153be924f17ce4bf62726954eeaa4a73e854` carries exactly that config shape in one 2718576122-byte shard, drafting five layers for a 64-layer Qwen3.8-27B target
Row: SPEC-DSPARK-QWEN3-ROUTING
State: UNKNOWN
Kind: bug
GitHub: 1193
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:385`

### Frozen archive evidence

> | [#1193](https://github.com/mudler/vllm.cpp/issues/1193) | `SPEC-DSPARK-QWEN3-ROUTING` | A Qwen3 DSpark draft declaring `architectures=["DSparkDraftModel"]` with `model_type` `qwen3` has no route. The pin forces every DSpark draft that is not `Qwen3DSparkModel` or `Gemma4DSparkModel` onto `model_type` `deepseek_v4` (`vllm/config/speculative.py:934-944` @ `555967922`), and vLLM PR 52197 (merged 2026-08-17 at `7075ddac`) replaced that with a leading branch normalizing the pair to `Qwen3DSparkModel`. We diverge from BOTH: the forced rewrite was never ported, so nothing in `src/vllm/entrypoints/model_loader.cpp` reads a draft config's `architectures` key at all, and `SpeculativeConfig::IsDsparkDraft` (`include/vllm/config/speculative.h:120-136`) has no production caller — every reference outside its header is in `tests/vllm/config/test_speculative_dspark.cpp:132-140`, and `ResolveSpecConfig` branches on `cli.method` alone. The checkpoint is real and gateable here: `RadixArk/Qwen3.8-27B-DSpark` at revision `85ef153be924f17ce4bf62726954eeaa4a73e854` carries exactly that config shape in one 2718576122-byte shard, drafting five layers for a 64-layer Qwen3.8-27B target | bug |

## Resolution

-
