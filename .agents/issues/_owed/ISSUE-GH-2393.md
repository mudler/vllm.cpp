ID: ISSUE-GH-2393
Title: KGATHER: confirm the CUDA block gather on sm_121a, and measure mutation M2 on a device
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 2393
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `MODEL-MM-QWEN4-EXP`
>
> The CUDA dequantizing gather (KGATHER, [`.agents/specs/cuda-quant-gather.md`](https://github.com/mudler/vllm.cpp/blob/main/.agents/specs/cuda-quant-gather.md)) is gated and green **on sm_110 only**.
>
> **What ran.** `thor:gpu0` (Jetson Thor, sm_110, nvcc 13.0.88, aarch64), `rc` job `e53a20f5-f267-4672-8647-3fd82e5f0fe0`, 2026-08-31, both legs built from a staged bundle:
>
> - RED `e0188c50b` — build rc 0, test rc 1: 4 of 5 cases threw `cuda embedding: unsupported table dtype (f32/bf16 only)` at 32 assertions.
> - GREEN `e08bc069d` — build rc 0, test rc 0: **6 of 6 cases, 231 of 231 assertions**, all 18 block encodings bit-exact against the CPU arm in f32 and bf16 out with i32 and i64 ids.
> - M1 (restore the refusal) RED, M4 (element stride) RED, M3 (FMA contraction) SURVIVED as predicted in writing before the run.
>
> **What is owed.**
>
> 1. **sm_121a.** The target architecture is GB10. Nothing here claims it. `dgx:gpu0` job `38a9b799-7caa-4703-bfe2-4f393f6ac06c` is queued with the same script and closes this.
> 2. **Mutation M2 on a device.** M2 drops the Q4_K sub-block minimum. On thor it FAILED TO BUILD — nvcc runs `-Werror=all-warnings` and replacing `m1` with a literal left it unreferenced (`error #177-D`). The job's build guard refused to run the test, because the binary on disk was the previous leg's and would have printed a green describing code nobody compiled. The mutation now reads `m1 * 0.0f`, verified against a control: the new form compiles under `-Werror` at rc 0 and the old form reproduces the exact failure.
>
> The defect class itself is NOT unmeasured — the host transliteration harness reds it at 457,498 of 968,199 elements, max|diff| 4.0295e+06. Only its measurement **on a GPU** is owed.
>
> Neither gap blocks correctness on sm_110; both are named so the sm_110 result is not read as a fleet-wide one.

## Resolution

Commit `c93cb10beb33568caaee8221199220908e433530` records the sm_121a gate green and explicitly discharges issue #2393; GitHub closed it as COMPLETED on 2026-08-31.
