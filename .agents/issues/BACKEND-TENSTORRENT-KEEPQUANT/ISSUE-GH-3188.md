ID: ISSUE-GH-3188
Title: ReorderVRowsRef test helper has wrong offset formula — heap-buffer-overflow in test_gguf_keep_quant
Row: BACKEND-TENSTORRENT-KEEPQUANT
State: OPEN
Kind: bug
GitHub: 3188
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-14
Updated: 2026-09-14
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-KEEPQUANT`
>
> The test helper `ReorderVRowsRef` in `tests/vllm/test_gguf_keep_quant.cpp:3194` uses `(row_off + g) * cs` as the byte offset, where `cs = head_rows * cols`. This treats `row_off` and the V-unit index `g` as multiples of the head-group stride, but `row_off` is in individual rows and `g` counts V-units (each spanning `head_rows` rows).
>
> With test params (K=512, row_off=3, num_k=2, rpk=3, head_rows=2):
> - Buffer size: 15 * 512 = 7680 floats
> - Buggy max offset: (3+5) * 1024 = 8192 → past end
> - Correct max offset: (3 + 5*2) * 512 = 6656, ending at 7680 → exactly the buffer
>
> The production `ReorderVRows` in `src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:395` uses the correct formula: `base + t * head_stride` where `base = buf.data() + row_off * cols` and `head_stride = head_rows * cols`.
>
> This breaks build-test-cpu, build-test-cpu-arm64-full, and sanitize-cpu (address,undefined).
>
> Introduced by commit `4c8f5d5bb` (PR #3042).

## Resolution

-
