ID: ISSUE-GH-1225
Title: A DSpark speculative length below the draft's block is accepted silently. `ResolveDspark` carries upstream's `k >= dspark_block_size` hard error (`include/vllm/config/speculative.h:179-185`, from `vllm/config/speculative.py:1003-1027` @ `555967922`) and both production call sites pass `std::nullopt` for `n_predict` and for `dspark_block_size` (`src/vllm/entrypoints/model_loader.cpp:881-883` and `:1675-1677`), so the floor reaches no user path and only `tests/vllm/config/test_speculative_dspark.cpp:99-107` drives it. Nothing in our draft path reads the checkpoint's block key — the block layout is sized by `k` alone (`include/vllm/v1/worker/gpu/spec_decode/dspark/speculator.h:56`) and no weight is block-shaped — so a short `k` raises no shape error and drafts a structurally wrong block while the tokens keep flowing. A literal port does NOT close it: `dspark_block_size` appears in no pinned file but `speculative.py`, and neither published Qwen3 draft sets it (`deepseek-ai/dspark_qwen3_4b_block7` and `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b` both carry `block_size: 7`, no `n_predict`), while upstream's `block_size` normalization at `:945-961` is Gemma4-only — so upstream accepts `k=6` on both sides of vllm#52197. Closing it for the lane we ship needs `block_size` as the floor fallback, one tracked divergence argued in the spec and the commit
Row: SPEC-DSPARK-BLOCK-SIZE-GUARD
State: UNKNOWN
Kind: bug
GitHub: 1225
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:398`

### Frozen archive evidence

> | [#1225](https://github.com/mudler/vllm.cpp/issues/1225) | `SPEC-DSPARK-BLOCK-SIZE-GUARD` | A DSpark speculative length below the draft's block is accepted silently. `ResolveDspark` carries upstream's `k >= dspark_block_size` hard error (`include/vllm/config/speculative.h:179-185`, from `vllm/config/speculative.py:1003-1027` @ `555967922`) and both production call sites pass `std::nullopt` for `n_predict` and for `dspark_block_size` (`src/vllm/entrypoints/model_loader.cpp:881-883` and `:1675-1677`), so the floor reaches no user path and only `tests/vllm/config/test_speculative_dspark.cpp:99-107` drives it. Nothing in our draft path reads the checkpoint's block key — the block layout is sized by `k` alone (`include/vllm/v1/worker/gpu/spec_decode/dspark/speculator.h:56`) and no weight is block-shaped — so a short `k` raises no shape error and drafts a structurally wrong block while the tokens keep flowing. A literal port does NOT close it: `dspark_block_size` appears in no pinned file but `speculative.py`, and neither published Qwen3 draft sets it (`deepseek-ai/dspark_qwen3_4b_block7` and `RadixArk/Qwen3.8-27B-DSpark` @ `85ef153b` both carry `block_size: 7`, no `n_predict`), while upstream's `block_size` normalization at `:945-961` is Gemma4-only — so upstream accepts `k=6` on both sides of vllm#52197. Closing it for the lane we ship needs `block_size` as the floor fallback, one tracked divergence argued in the spec and the commit | bug |

## Resolution

-
