ID: ISSUE-LOCAL-01M2DW8CXYEWWMJSZZ6GRH48SZ
Title: GB10: with CUDA graphs on, the DevicePool retains about 44 GiB beyond startup within two minutes of c=32 EXL3 DFlash2 serving, and nothing bounds it
Row: ENG-POOL-BEST-FIT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-13
Updated: 2026-09-13
Closed: -

## Problem

Measured on dgx:gpu0 (GB10, 125 GiB unified memory) on 2026-09-13, serving Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw with its DFlash2 EXL3 draft (k=7) at the #3150 merge `39d3af455` (binary md5 c65ac4edb442783ee4030bdbb9bb6c43), with the variadic benchmark's c=32 recipe: `--max-num-seqs 32 --max-model-len 8192 --num-blocks 8192 --max-num-batched-tokens 16384`, the 192-prompt variadic corpus (sha256 b52ef36c...), 128 measured requests at concurrency 32. MemAvailable was sampled every 2 s, and a watchdog killed the server below 14 GiB. Evidence: /workspace/exl3-poolgrowth/20260913-163156/ and /workspace/exl3-poolgrowth/20260913-163739/ on the rc share.

| arm | MemAvailable at ready | minimum under load | outcome |
|---|---|---|---|
| default (CUDA graphs on, DevicePool on) | 58.2 GiB | 13.8 GiB, falling about 22 GiB/min | watchdog kill after 2 min 14 s |
| graphs off + VT_POOL_BYPASS=1 | 58.3 GiB | 34.3 GiB, flat for 27 min | leg completed, 17.2 output tok/s |
| graphs off, DevicePool on | 58.3 GiB | 23.0 GiB, flat for the 3 min it ran | host crashed at 17:09 UTC before the leg finished |
| graphs on + VT_POOL_BYPASS=1 | 58.3 GiB | - | refused: `cudaMalloc: operation not permitted when stream is capturing` |

Reading: every arm returned to 115 GiB after the server exited, so the growth is owned by the process and is invisible in its VmRSS (1.5-2.3 GiB throughout). With CUDA graphs on, the default pool retains about 44 GiB beyond the 58 GiB startup footprint within two minutes, which is the same `MemAvailable` decline #1922 reported per request. The same host telemetry during an earlier c=32 leg recorded a fall from 32 to 16 GiB in 70 s immediately before the host became unreachable at 12:43 UTC. `device_pool_cap_bytes` is 0 (uncapped) on GB10 (`include/vllm/platforms/interface.h:118`), and `DevicePool::Drain` has no caller on the LLM serving path.

Not established: which scratch shapes grow, and whether #3150's reconstruct path (M > 144 adds a [K, min(N, 32768)] fp16 scratch and larger [M, N] outputs per linear) adds size classes the pre-#3150 tree did not. The pre-#3150 binary has not run this diagnostic. The host crashes are NOT all ours: one of the four crashes this session happened while exllamav3 alone was serving at c=1 (job 7137c9c8, 2026-09-13 00:29 UTC), so dgx:gpu0 also fails under sustained load independent of this growth.

vLLM mirror: torch's CUDACachingAllocator frees cached blocks and retries when an allocation fails (`c10/cuda/CUDACachingAllocator.cpp` release_cached_blocks on OOM), and `gpu_memory_utilization` bounds the working set that vLLM profiles. On unified memory a cudaMalloc does not fail before the host runs out, so an uncapped retention pool has no back-pressure at all.

## Resolution

-

## Progress 2026-09-13: attribution runs

Two more runs on dgx:gpu0, same recipe and watchdog. Evidence is in `/workspace/exl3-poolgrowth/20260913-171625/` and `/workspace/exl3-poolgrowth/20260913-182654/`.

**#3150's parent does not cross the watchdog.** The `3cafbcaf` binary, graphs on and default pool, completed the whole c=32 leg at 50.84 output tok/s. MemAvailable went from 58.19 GiB at ready to a plateau of 19.2 GiB. At the same elapsed 130 s it read 24.3 GiB where `39d3af455` read 14.0 GiB and was still falling. #3150 therefore adds more than 10 GiB at c=32 on top of a growth of about 39 GiB that already existed.

**Only part of the growth is DevicePool.** A diagnostic-only trace was applied to `39d3af455` and never landed. It printed the pool's classes each time its driver allocations crossed another GiB (binary md5 897c2b95fc9148f16c33324ad64d89ab). One pool reached 16.65 GiB of driver allocations when the watchdog fired at 13.71 GiB MemAvailable, with 9.67 GiB of that held FREE across 348 size classes. The largest class was 17825792 B, 330 blocks, 5.48 GiB. That size equals an f32 [256, 17408] MLP intermediate at the c=32 verify width of 32 x 8 tokens. It was live 330 in one sample and free 287 in another. The host lost about 43 GiB in the same window, so about 27 GiB grew outside this pool.

**CUDA graphs account for roughly 20 GiB of it.** Graphs off with the pool on plateaued at 23.0 GiB. Graphs off with the pool bypassed held 34.3 GiB. Graphs on with the pool on went below 13.7 GiB.

**Next hypothesis:** graphs are captured lazily during serving, so their memory is not accounted when the KV pool is sized. The pool's free retention is uncapped on top of that. vLLM captures every size in `cudagraph_capture_sizes` during `capture_model` before serving, and accounts graph memory in the profiled budget that `gpu_memory_utilization` sizes the KV cache from (`vllm/v1/worker/gpu_model_runner.py` `capture_model`, `profile_run`). This is not yet read against our runner.
