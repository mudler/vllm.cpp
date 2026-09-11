ID: ISSUE-GH-2095
Title: ROCm decode: #1863's 4b1154bc5 baseline (18.4 / 13.0 tok/s) does not reproduce on the same board
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 2095
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-27
Updated: 2026-08-27
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Title
>
> ROCm decode: #1863's `4b1154bc5` baseline (18.4 / 13.0 tok/s) does not reproduce on the same board
>
> ## Body
>
> #1863 records 18.393 and 18.574 tok/s for `Ornith-1.5-9B-Q4_K_M.gguf` at
> commit `4b1154bc5` on this board (RX 9060 XT, gfx1200, ROCm 7.2.3). A repeat
> measurement today does not reproduce that number.
>
> ### Method
>
> Each test runs `vllm-cli` with `--prompt "The capital of France is"
> --max-tokens 32 --temperature 0`: one untimed warm-up run per model, then
> three timed repetitions. The dense case uses the file #1863 names,
> `Ornith-1.5-9B-Q4_K_M.gguf` (5,629,108,992 bytes, matching #1863's stated
> 5.23 GiB). The grouped case uses `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf`,
> a local file not confirmed to be the one #1863's "same session" 14B
> measurement used.
>
> ### Result
>
> | Build | Dense tok/s (3 reps) | Grouped tok/s (3 reps) |
> |---|---|---|
> | `4b1154bc5`, the commit #1863 names | 6.150, 6.148, 6.128 | 4.946, 4.967, 4.980 |
> | Current HEAD (`a73b26968`) | 6.157, 6.226, 6.138 | 4.978, 4.988, 4.990 |
> | #1863's recorded value | 18.393, 18.574 | 13.000, 13.145 |
>
> Each set of three repetitions stays within 2 percent of its own mean. The
> dense case ran the exact file and the exact commit #1863 names, and it still
> returns about one-third of the recorded value.
>
> ### Ruled out
>
> Seven checks, each a controlled same-binary comparison on this board.
>
> 1. **A code change.** `git log 4b1154bc5..HEAD -- src/vt/rocm/` returns 0
>    commits.
> 2. **Build type.** `Release` and `RelWithDebInfo` give the same result.
> 3. **The GPU power state.** `power_dpm_force_performance_level=auto`
>    (`mclk` idle-parked at 96 MHz between calls) and `=high` (`mclk` pinned
>    at 1258 MHz) give the same result.
> 4. **A missing performance flag.** Every ROCm decode flag in the tree
>    already defaults to its fast setting: `VT_ROCM_GDN_POSTCONV_CHUNK`,
>    `VT_ROCM_SKINNY`, `VT_ATTN_DECODE_OPT`, `VT_ATTN_DECODE_GQA`,
>    `VT_ATTN_DECODE_D128`, `VT_ATTN_PREFILL_FLASH_SHAREDK`,
>    `VT_ATTN_PREFILL_SHAREDK_WMMA`, and `VT_GGUF_KEEP_QUANT`. No override
>    changes the result.
> 5. **A different machine.** The developer confirms this is the same board
>    #1863 used, and the only one available.
> 6. **Host load.** `/proc/loadavg` reads 2.02 to 3.30 across the session, the
>    same band #1863 itself states (2.2 to 3.0).
> 7. **Desktop GPU contention.** Closing every Chromium process does not
>    change the result.
>
> ### Open
>
> Two explanations remain, and this record does not decide between them.
>
> - The number #1863 states may not match what its own run produced.
> - Something outside code, build type, GPU power state, flags, and load has
>   changed on this board since #1863's measurement. `dmesg` is not readable
>   in this environment (`Operation not permitted`), so a driver reset, a
>   fault, or a firmware change cannot be checked from here.
>
> ### What this does not affect
>
> The per-kernel percentage findings from the same profiling session stay
> valid. `QuantizeQ8KK` takes 22.66 percent of GPU kernel time on the dense
> case and 36.19 percent on the grouped case, measured with `rocprofv3
> --kernel-trace --stats`. Those are ratios within one run, so they do not
> depend on the absolute throughput this issue leaves open.
>

## Resolution

-
