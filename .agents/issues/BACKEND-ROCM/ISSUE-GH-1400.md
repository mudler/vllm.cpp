ID: ISSUE-GH-1400
Title: ROCm decode is 4.25x behind llama.cpp on the same backend API, on a model that fits in VRAM
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1400
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-19
Updated: 2026-08-19
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ROCm decode is **4.25x behind llama.cpp on the same backend API**, measured on a
> model that fits entirely in VRAM. The gap is kernels and dispatch, not memory
> strategy: no offload is involved on either side.
>
> Measured against the pinned llama.cpp oracle (`b10451` / `10bf611e5`) built from
> source for gfx1200, same card, same file, all three arms inside the same 31
> seconds on an idle host.
>
> ## Measurement
>
> `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf`, 7.87 GiB, experts all Q4_K/Q6_K,
> batch 1. RX 9060 XT (gfx1200, 15.92 GiB VRAM), ROCm 7.2.3, gcc 15.2.0, NixOS,
> load average 1.14 -> 1.15 across the three runs.
>
> | | decode tok/s | ms/token | vs ours |
> |---|---|---|---|
> | vllm.cpp ROCm | **11.13 / 11.19** | 88.7 | 1.00x |
> | llama.cpp HIP `b10451` | **47.44 +/- 9.07** | 21.1 | **4.25x** |
> | llama.cpp Vulkan (unpinned) | 79.32 +/- 7.13 | 12.6 | 7.11x |
>
> ## Where our time goes
>
> `rocprofv3 --runtime-trace --stats`, decode isolated by differencing
> `--max-tokens 4` against `--max-tokens 36` and dividing by 32, idle host.
> Dispatch counts were byte-identical across a contended and an idle run, so the
> structure is deterministic and only durations moved.
>
> | | calls/tok | ms/tok | share of GPU |
> |---|---|---|---|
> | `QuantizeQ8KK` | 281 | **21.25** | 35% |
> | `GdnPostConvK` | 30 | 11.24 | 19% |
> | hipBLASLt GEMM | 230 | 13.81 | 23% |
> | copy/fill plumbing | 532 | 1.26 | 2% |
> | real H2D + D2H transfer | 160 | **0.885** | - |
> | total kernel dispatch | 1738 | 59.99 | |
>
> **llama.cpp HIP produces an entire token in 21.1 ms. Our activation quantizer
> alone takes 21.25 ms.**
>
> Full profile and method: #1294.
>
> ## Named first lever
>
> `QuantizeQ8KK` (`src/vt/rocm/rocm_grouped_gemm.hip:147`, arriving with #523) is
> thread-per-256-superblock, launched:
>
> ```cpp
> QuantizeQ8KK<<<(m * nsb + 127) / 128, 128, 0, s>>>(...)
> ```
>
> At decode on the MoE path `m` = T x top_k = 8 and `nsb` = 2048/256 = 8, so
> `m*nsb` = **64 threads -> one workgroup**, 64 of 128 lanes active, on a 32-CU
> GPU, 281 times per token at 75.6 us each. That is a parallelization problem, not
> an algorithmic one: workgroup-per-superblock with a reduction for the scale is
> the obvious shape.
>
> A contributing redundancy is on `main` and is backend-agnostic: `MoeBlock` hands
> the **same** activation buffer to `KqGrouped` twice
> (`src/vllm/model_executor/models/qwen3_5.cpp:6832-6833`, gate then up), so the
> identical activation is quantized twice per layer per token. CPU and CUDA pay it
> too; ROCm is only where it was measured.
>
> ## What this refutes
>
> A memory-strategy explanation for the gap. Separately measured on this card:
>
> - llama.cpp's **HIP** backend **cannot load** the 19.45 GiB
>   `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (`hipMalloc` refuses past VRAM) - the same
>   refusal this engine hits.
> - llama.cpp's **Vulkan** backend loads it: `offloaded 41/41 layers to GPU`,
>   `Vulkan0 model buffer size = 19399.34 MiB` on a 15.92 GiB card, spilling into
>   the 31.35 GiB GTT the amdgpu driver exposes (`mem_info_gtt_total`).
>
> That is a real and separate finding about **fitting** models larger than VRAM.
> It is not why decode is slow: on the 14B, which fits, llama.cpp HIP is still
> 4.25x faster under the identical `hipMalloc` constraint.
>
> `hipMallocManaged` was tested as a route to the same GTT path and **does not
> migrate on this part**. Allocation succeeds past VRAM (18 GiB on a 15.92 GiB
> card) and a device memset on the result works, but a paired same-binary A/B
> against pinned host memory was identical (7.64/7.77 vs 7.63/7.77 tok/s), and
> during a managed run `mem_info_gtt_used` stayed flat at 0.43 GiB while
> `vram_used` filled to 15.76. This is consistent with the existing record at
> `docs/ROCM.md:148` that the managed branch is dead on discrete parts, for the
> XNACK-less reason that record gives.
>
> ## Caveats
>
> - `llama-bench tg32` measures raw generation; our figure comes through the full
>   engine, so some of the gap is harness. Not 4.25x of it.
> - The **HIP** arm is stock `b10451` built from source here, i.e. the pinned
>   oracle. The **Vulkan** arm is a nixpkgs build reporting `build: unknown (0)`;
>   its revision is unrecorded, so 79.32 is indicative context and NOT an oracle
>   comparison.
> - One model, batch 1, one board. No prefill comparison, and occupancy improves
>   on its own at larger `m`, so this decode-shape pathology may not appear in a
>   throughput benchmark.
> - The binary under test is `main` + #523 + the #559/#570 `AttnQkNormRopeGate`
>   fix, and carries a throwaway weight-offload patch that is **inert** with no
>   environment variable set (budget 0 returns immediately), so the measured path
>   is unmodified.
> - The llama.cpp oracle's `gateable` flag still reads `no` pending
>   [#857](https://github.com/mudler/vllm.cpp/issues/857), where the build-and-run
>   measurement for this same pin is recorded.
>

## Resolution

-
