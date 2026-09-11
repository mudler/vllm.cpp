ID: ISSUE-GH-1506
Title: ROCm has no kMatmulBTQuant provider, so every GGUF k-quant silently expands to bf16 at load: 1.73x peak RSS and 4.5x wall on gfx1200
Row: BACKEND-ROCM
State: CLOSED
Kind: UNKNOWN
GitHub: 1506
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-24
Closed: 2026-08-24

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `vt::OpId::kMatmulBTQuant` is registered on `kCPU`
> (`src/vt/cpu/cpu_quant_gemm.cpp:303`) and `kCUDA`
> (`src/vt/cuda/cuda_quant_dot.cu:1991`). It is **not** registered on `kROCM`, and
> `src/vt/rocm/` contains no keep-quant GEMM of any kind (18 `.hip` files, none
> matching `MatmulBTQuant`).
>
> The consequence is not a refusal. `GgufQuantComputeAvailable()`
> (`src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:75`) probes
> `OpRegistered(kMatmulBTQuant, CurrentPlatform().device_type())`, and
> `GgufLoadPolicy::FromEnv` (`:169`) uses that as the default for `keep_quant`.
> On a HIP build the probe is false, so `keep_quant` is false, so
> `RouteGgufTensor` returns `kExpandBf16` for **every** quantized weight in the
> checkpoint. `p.expand_nk` (`:172`) rides the same flag, so the NK-orientation
> win is lost with it.
>
> A ROCm user therefore loads every GGUF k-quant fully dequantized to bf16, with
> no warning and no log line.
>
> ## Measured
>
> gfx1200 (RDNA4), ROCm 7.2.3, `Qwen3.5-4B-Q4_K_M.gguf` (2.74 GB, arch `qwen35`),
> identical prompt and `--max-tokens 4` through `examples/vllm-cli`. Peak RSS
> sampled from `/proc/<pid>/status` `VmHWM`.
>
> | build | device | `keep_quant` | peak RSS | wall |
> |---|---|---|---|---|
> | CPU | `--device cpu` | on (auto) | 7038 MiB | 8.2 s |
> | HIP | `--device auto` | **off (auto)** | **12159 MiB** | 37.1 s |
> | HIP | `--device auto`, `VT_GGUF_KEEP_QUANT=1` | forced on | — | fails |
>
> Output is identical in the first two rows (" Paris.\nA"). The expansion costs
> **1.73x peak RSS and 4.5x wall**.
>
> Forcing the override past the probe produces the missing kernel directly:
>
> ```
> vt: no kernel for op MatmulBTQuant (id 75) on device rocm (type 5)
>     at src/vt/op_provider.cpp:518
> ```
>
> The absolute cost scales with the checkpoint, and it is a load-time cliff rather
> than a slowdown: a ~16 GB k-quant expands to roughly 55-60 GB, which does not
> fit on a 64 GB host that runs the same file comfortably on CPU. The HIP build is
> strictly worse than the CPU build for GGUF today.
>
> ## Scope
>
> The activation half already exists on ROCm — see `6251de146`
> (`perf(rocm)`: resolve the Q8_K quantizer's activation dtype at launch), which
> tunes the Q8_K quantizer this GEMM consumes. What is missing is the weight-side
> `DotSuperblock` family.
>
> Portability is not in question. llama.cpp compiles the same CUDA sources for
> HIP (`ggml/src/ggml-hip/CMakeLists.txt` globs `../ggml-cuda/*.cu`) and wraps
> `__dp4a` with a HIP/GCN-asm branch plus a scalar fallback
> (`ggml/src/ggml-cuda/common.cuh:733`). AMD has had `v_dot4_i32_i8` since
> gfx906. Our CUDA implementation uses 52 `__dp4a`, 3 `__vsub4` and one
> `__shfl_sync`, with no tensor cores, no PTX and no async copy.
>
> ## Related, not duplicated
>
> - #487 — ROCm gfx1200 decode M=1 matmuls go to a hipBLASLt 128x128-tile GEMM
>   instead of a skinny-GEMM kernel, 77% of decode GPU time. Those are the very
>   GEMMs this kernel would replace for quantized weights, so the two rows
>   overlap on the decode path.
> - #1294 — ROCm decode is kernel-bound with the activation quantizer at 35% of
>   GPU time. That is this GEMM's input side.
> - #41 — the ROCm backend thread.
>
> ## Also noticed
>
> The HIP build reports `max_concurrent_batches=1` where CPU reports `2`, so
> async scheduling silently disables on ROCm. Not investigated; filed here only
> so it is not lost.
>

## Resolution

The binding comment https://github.com/mudler/vllm.cpp/issues/1506#issuecomment-5399708825 dated 2026-08-24 records that commit `7bcf2f5e1` registered the ROCm quant operations, retracts the cold-cache wall claim, and moves the surviving residency gap to issue #1870.
