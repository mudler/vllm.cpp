ID: ISSUE-GH-3153
Title: test_rocm_f16_contract fails on clean main: test expects ViewOn to propagate layout markers it deliberately does not
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 3153
Mirror: SYNCED
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: -
>
> `test_rocm_f16_contract` (introduced in #3095) fails on clean main at lines 153-155. The test expects `OwnedTensor::View()` / `OwnedTensor::ViewOn()` to propagate layout markers (`repacked`, `q8_0_aligned`, `elem_kn_repacked`), but the code deliberately does not propagate them — `ViewOn` carries only `weight_value_dtype`.
>
> The `ResidentWeight` CPU-alias arms in `dense_attn_block.h:211-217` and `qwen3_5.cpp:1147-1154` already set `repacked` and `elem_kn_repacked` explicitly after calling `ViewOn`, matching the pre-`ViewOn` behavior. The test was written alongside the code but contradicts it: the code says markers are NOT propagated, the test says they ARE.
>
> This was never caught because `build-test-cpu` was skipping when #3095 merged. It now blocks every PR that runs `build-test-cpu`.
>
> The fix is to correct the test at `tests/vt/test_rocm_f16_contract.cpp:143-166` to verify the actual contract: `weight_value_dtype` IS propagated, layout markers are NOT.

## Resolution

fixed in row/FIX-ROCM-F16-CONTRACT-TEST: test corrected to verify ViewOn carries only weight_value_dtype, not layout markers
