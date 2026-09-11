ID: ISSUE-GH-357
Title: `vllm-server`: default KV sizing allocates ~27 GB regardless of model size; `--kv-cache-memory` does not constrain it (workaround: `--num-blocks`)
Row: KV-SIZING
State: OPEN
Kind: UNKNOWN
GitHub: 357
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ### Workaround first
>
> If you are hitting an unexplained multi-tens-of-GB allocation at `vllm-server` startup —
> especially on a shared or unified-memory device — pass an explicit block count:
>
> ```
> --num-blocks 64          # or whatever your context x concurrency actually needs
> ```
>
> That alone took startup from *never ready, 27 GB consumed* to **ready in 5 s, 686 MB consumed**
> on the same model. `--kv-cache-memory` did **not** have this effect (below).
>
> ### What I measured
>
> Device: NVIDIA Jetson AGX Thor (sm_110, CC 11.0), 122 GB **unified** LPDDR5X, CUDA 13.2,
> built in `nvcr.io/nvidia/vllm:26.04-py3`, vllm.cpp @ `e3cc4f6`,
> `-DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF`.
>
> Model: `apolloparty/Qwen3-1.7B-NVFP4A16` — **1.4 GB on disk**. Server invocation:
>
> ```
> vllm-server --model <dir> --host 127.0.0.1 --port 30899 \
>             --max-model-len 1024 --max-num-seqs 8 --kv-cache-memory 1073741824
> ```
>
> Sampled every 5 s, with the container capped at `--memory=12g`:
>
> ```
> t=5s   container RSS = 364.2 MiB / 12 GiB      <-- container is tiny
>        system available: 38560 MB -> 11485 MB  <-- 27,075 MB gone
> ```
>
> **~27 GB consumed within 5 seconds, by a 1.4 GB model, and essentially none of it in the
> container's host RSS.** The server never reached `/v1/models`.
>
> The RSS-vs-`free` divergence is the diagnostic that makes this identifiable rather than
> anecdotal: on unified-memory Tegra a `cudaMalloc` is visible in `free` but is **not** charged
> to the container's cgroup, so this reads as a host-side "leak" that no host memory limit
> contains — `--memory=12g` did not stop it, and the container was never OOM-killed.
>
> Swapping only the sizing knob:
>
> ```
> --num-blocks 64   ->  686 MB consumed, server READY at t=5s
> ```
>
> Same model, same everything else.
>
> ### Why I think this is a defect and not my misuse
>
> `include/vllm/entrypoints/model_loader.h:69-83` documents the precedence as:
>
> ```
> //   1. num_blocks > 0            -> used verbatim
> //   2. kv_cache_memory_bytes > 0 -> num_blocks = kv_cache_memory_bytes /
> //                                   KVBytesPerBlock(kv_cfg) (absolute pool size,
> //                                   IGNORES gpu_memory_utilization)
> //   3. otherwise                 -> the gpu_memory_utilization profile path,
> //                                   which needs a device profile run (M3, not
> //                                   yet implemented) and so falls back to 256.
> ```
>
> Two mismatches against what I observe:
>
> - **Knob 2 appears not to bind.** I passed `--kv-cache-memory 1073741824` (1 GiB). For this
>   model a KV block is 2 (K,V) x 28 layers x 8 KV heads x 128 head_dim x 2 B x 32 tokens =
>   **3.5 MiB**, so 1 GiB should resolve to ~292 blocks ≈ 1 GiB of KV. Observed: ~27 GB.
> - **The documented fallback is 256 blocks**, which for this model is **896 MiB**. Observed
>   default behaviour is ~27 GB — roughly 7,700 blocks' worth.
>
> `server_main.cpp:144` separately notes `--gpu-memory-utilization` is *"inert until"* the M3
> profile run lands, which is consistent with knob 3 not being the thing sizing this.
>
> **What I did NOT do:** I did not instrument which allocation is responsible. I am reporting a
> contract-vs-behaviour mismatch and a reproducible workaround, not a root cause. It is possible
> the 27 GB is not the KV pool at all and that `--num-blocks` changes the outcome through some
> other path.
>
> ### Operational consequence
>
> On a device that hosts other CUDA processes, the default sizing was sufficient to disrupt
> them. On our box four unrelated inference services restarted while this server was starting.
> Reporting that as observed behaviour, with the limits of my evidence stated:
>
> - **No host OOM kill was involved** — zero kernel OOM lines on the device across the window.
>   That is consistent with a device-side allocation starving peers into CUDA OOM rather than
>   the host OOM killer selecting a victim.
> - I did **not** instrument the causal chain between the two, so I am describing correlation
>   plus a plausible mechanism, not a proven cause.
>
> The practical point stands regardless: because the allocation is device-side on unified
> memory, **container memory limits do not contain it**, so the usual way of protecting
> co-tenants on a shared box does not work here.
>
> ### Suggested direction (maintainer's call)
>
> Any of these would have prevented the failure mode:
>
> 1. Make knob 2 (`kv_cache_memory_bytes`) actually bind, since it is already documented to.
> 2. Make the knob-3 fallback the documented 256 blocks.
> 3. Fail loudly at startup when the resolved pool exceeds some fraction of free device memory,
>    rather than allocating and hanging.
>
> ### Scope
>
> One device, one model, one build. Not reproduced on another architecture. Reported because
> the workaround is cheap and the failure mode is expensive to diagnose from the outside — it
> presents as a memory leak in a process whose RSS is 364 MiB.
>
> ### Forward note on knob 3
>
> `server_main.cpp:147` sets `double gpu_memory_utilization = 0.92`. That knob is inert today
> (it needs the M3 profile run), so it is not what is sizing the pool here — but if the profile
> path lands with that default, a 0.92 fraction of free device memory on a unified-memory part
> is the same hazard by another route. Mentioning it only so the eventual M3 work does not
> inherit the problem.
>
> ---
>
> Cross-referencing #168 (Jetson AGX Thor sm_110 report) since that is where this device and
> build are described, and #325 / #326 for the unrelated Marlin work on the same box. Filed
> separately from #168 because this is engine KV-pool sizing rather than anything sm_110-specific
> — but see the caveat above: I have one device, so "arch-independent" is read from the code
> path, not measured. Happy to fold it into #168 if you would rather keep one thread.
>

## Resolution

-
