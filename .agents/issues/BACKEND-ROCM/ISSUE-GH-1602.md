ID: ISSUE-GH-1602
Title: **`e2a9e035d`, the current tip of `main`, added one `test_runner` case that is RED on a default CPU build.** `tests/vllm/v1/worker/test_runner.cpp:1557` asserts `CHECK_THROWS_WITH_AS(make_runner(), "Block size must be a multiple of 16", std::invalid_argument)`; with `block_size = 8` and no ROCm and no FLASH_ATTN the runner never reaches `CheckKvCacheShape`, because the attention-backend SELECTOR rejects every candidate first and raises `No valid attention backend for device type 0 from {CPU_ATTN: [block_size not supported], FLASH_ATTN: [block_size not supported]}`. Measured at `e2a9e035d` on a Release CPU build: the file is 20 cases / 544 assertions, 19 pass, this one fails, and over a 585-test `ctest` run it is the ONLY `***Failed`. The commit body records its validation as "on gfx1151 (Strix Halo, ROCm 7.2.3)", which is the one configuration where the selector has a backend that reaches the shape check, so the case encodes a build configuration it never declares. Both refusals are correct behaviour and they are DIFFERENT guarantees, so the fix is to assert whichever refusal the built configuration produces, or to gate the case on a build that has a backend supporting the block size and SKIP LOUDLY otherwise — never to widen the assertion to accept any throw. Found by `QUANT-QWEN38-27B-NVFP4-ARM` W5 ([#821](https://github.com/mudler/vllm.cpp/issues/821)) running the full suite for an unrelated loader change, in a file that change does not touch. Related: [#41](https://github.com/mudler/vllm.cpp/issues/41), [#1332](https://github.com/mudler/vllm.cpp/issues/1332)
Row: BACKEND-ROCM
State: UNKNOWN
Kind: bug
GitHub: 1602
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:569`

### Frozen archive evidence

> | [#1602](https://github.com/mudler/vllm.cpp/issues/1602) | `BACKEND-ROCM` | **`e2a9e035d`, the current tip of `main`, added one `test_runner` case that is RED on a default CPU build.** `tests/vllm/v1/worker/test_runner.cpp:1557` asserts `CHECK_THROWS_WITH_AS(make_runner(), "Block size must be a multiple of 16", std::invalid_argument)`; with `block_size = 8` and no ROCm and no FLASH_ATTN the runner never reaches `CheckKvCacheShape`, because the attention-backend SELECTOR rejects every candidate first and raises `No valid attention backend for device type 0 from {CPU_ATTN: [block_size not supported], FLASH_ATTN: [block_size not supported]}`. Measured at `e2a9e035d` on a Release CPU build: the file is 20 cases / 544 assertions, 19 pass, this one fails, and over a 585-test `ctest` run it is the ONLY `***Failed`. The commit body records its validation as "on gfx1151 (Strix Halo, ROCm 7.2.3)", which is the one configuration where the selector has a backend that reaches the shape check, so the case encodes a build configuration it never declares. Both refusals are correct behaviour and they are DIFFERENT guarantees, so the fix is to assert whichever refusal the built configuration produces, or to gate the case on a build that has a backend supporting the block size and SKIP LOUDLY otherwise — never to widen the assertion to accept any throw. Found by `QUANT-QWEN38-27B-NVFP4-ARM` W5 ([#821](https://github.com/mudler/vllm.cpp/issues/821)) running the full suite for an unrelated loader change, in a file that change does not touch. Related: [#41](https://github.com/mudler/vllm.cpp/issues/41), [#1332](https://github.com/mudler/vllm.cpp/issues/1332) | bug |

## Resolution

-
