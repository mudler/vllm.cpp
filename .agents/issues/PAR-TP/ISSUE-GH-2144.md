ID: ISSUE-GH-2144
Title: tp>1 prefill is host-bound: per-token fp4 GEMM + per-layer H2D activation staging
Row: PAR-TP
State: OPEN
Kind: UNKNOWN
GitHub: 2144
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-28
Updated: 2026-08-28
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> TP>1 prefill on the 27B NVFP4 checkpoint is host-bound, not GPU-bound. The fp4 device GEMM (`vt_cuda_mlp_shard_runT_fp4` / `MlpGuActFp4` / `MlpDownFp4`, `src/vt/cuda/nccl_communicator.cu`) launches **once per token instead of once per batch**, and every fp4 layer stages its activations through host memory (`Sm70Fp8PackQpn`). Decode hides this (CUDA-graph replay saturates the GPU); prefill exposes it.
>
> Measured with `nsys` on tp=2 (fresh binary, exclusive `$HOME/gpu.lock`, trace `/tmp/prof_tp2.nsys-rep`):
>
> - `MlpGuActFp4` launches with **identical duration (~845 us) and identical gridX (8704) at T=5 and T=1** — per-token, NOT batched. 5 tokens cost ~5x a decode-step MLP.
> - `cudaMemcpyAsync` H2D is **82.6% of CUDA-API time** (15.9 s; 33.6 GB / 9,195 copies; 19.6 GB >10 MB during the request window in ~4.1 s).
> - `Sm70Fp8PackQpn` is 25.4% of kernel time (1.74 s, 2,560 launches) — staging activations through host.
> - Warm prefill: GPU0 2.33 s kernel work over 4.37 s wall = ~53% busy; 459 >2 ms inter-launch stalls + `cudaStreamSynchronize` ~140 ms.
>
> Not a defect in the shipped fp8w row ([#2058](https://github.com/mudler/vllm.cpp/issues/2058), `BACKEND-DISTRIBUTED-TP-CLEAN`); it is the next wave in the same mirror line.
>
> ## Fix directions (to be scoped in the spec)
>
> 1. **Batched T-token fp4 GEMM**: gridX scales with T so a prefill batch costs one launch set, not T of them.
> 2. **Device-resident activations** across the fp4 layers: kill the per-layer host round-trip.
> 3. **Fold `Sm70Fp8PackQpn` staging on-device**.
>
> ## Axis
>
> Prefill wall (TTFT / prefill-only throughput at tp=1 vs tp=2), with the decode line held flat. Established token-exact baseline `' Paris.\nThe capital of Germany is Berlin.'` (10 tokens) stays the correctness gate.

## Resolution

-
