ID: ISSUE-GH-332
Title: ROCm: no decode-graph capture — the hipGraph seam is unimplemented, costing ~3x decode throughput vs vLLM on gfx1200
Row: BACKEND-ROCM
State: OPEN
Kind: perf
GitHub: 332
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> **Board:** AMD Radeon RX 9060 XT, `gfx1200` (Navi 44, RDNA4, discrete). ROCm
> 7.2.3, hipClang/Clang 22.0.0. M0/M1/M2 met on this board, and M4 met for
> Gemma-3-1B-it (#269).
>
> `vt::Backend`'s graph-capture virtuals are implemented only for CUDA.
> `RocmBackend::SupportsGraphCapture()` inherits `false`
> (`src/vt/rocm/rocm_backend.hip:22-26` says so explicitly), and
> `RocmPlatform` does not override `support_static_graph_mode()`, so it inherits
> `false` from `platforms/interface.h:189`. Every decode-graph class gates on
> exactly those two — e.g. `qwen3.cpp:494-498` — so on ROCm every decode step
> pays full host-side kernel-launch cost, while vLLM on the same board replays
> captured hipGraphs.
>
> ## Measurement
>
> Ours (`examples/vllm-bench` on `build-hip`) vs a real vLLM-ROCm oracle built
> from this project's own pinned commit `555967922` (against
> `rocm/vllm-dev:base`, whose ROCm 7.2.3 matches this board's native build), in
> **production config — graphs on, not `--enforce-eager`**. 128 in / 128 out,
> batch 8, bf16, stock upstream checkpoints:
>
> | Model | Layers | hidden x inter | Ours | vLLM | Ratio |
> |---|---|---|---|---|---|
> | Qwen3-0.6B | 28 | 1024 x 3072 | 184.69 tok/s | 552.65 tok/s | 2.99x |
> | Qwen3-1.7B | 28 | 2048 x 6144 | 150.50 tok/s | 286.42 tok/s | **1.90x** |
> | Gemma-3-1B-it | 26 | 1152 x 6912 | 146.43 tok/s | 438.09 tok/s | 2.99x |
>
> Single-stream agrees: our TPOT 13.55 -> 24.09 ms across the two Qwen3 sizes,
> vLLM 5.21 -> 12.34 ms/token, ratio 2.60x -> 1.95x.
>
> vLLM's runs capture 51 piecewise + 35 full hipGraphs before their timed
> section (visible in its own log), so **hipGraph capture demonstrably works on
> this board** — the mapping is available, we just do not use it.
>
> ## The evidence that this is launch overhead and not kernel quality
>
> Qwen3-0.6B vs Qwen3-1.7B is a controlled pair: **identical layer count (28)**,
> so an identical number of kernel launches per decode step, with roughly **4x
> the compute per step** (2x hidden, 2x intermediate). Launch overhead is fixed
> per step; compute is not. The gap falls **2.99x -> 1.90x**. A gap dominated by
> kernel quality would not move under that change; a gap dominated by fixed
> per-step overhead must.
>
> ## What this would and would not fix
>
> Fitting `ratio = alpha * (1 + L/C)` to the two Qwen3 points gives
> **alpha ~= 1.54x** (size-independent: kernel quality, inductor fusion, the
> Triton attention path) and a launch term of **~1.45x at 0.6B falling to ~0.36x
> at 1.7B**. So the realistic outcome is 0.6B 2.99x -> ~1.54x and 1.7B 1.90x ->
> ~1.54x — **a real win, and not parity.** A ~1.5x residual from non-launch
> causes would remain and is a separate problem.
>
> Two caveats, stated rather than buried: two points fitted with a one-line model
> is indicative and not rigorous, and **no `rocprof` trace has been taken on
> either side**, so this is a calibrated prediction rather than a measured
> attribution.
>
> It also does **nothing for Gemma-3**: there is no `Gemma3DecodeGraph` on any
> backend, CUDA included (`grep -rl DecodeGraph` returns Qwen3 dense/MoE,
> DeepSeek-V2/V4, Voxtral, Laguna). Gemma gains only once that sibling is written
> (sweep-gemma.md W7, unclaimed).
>
> ## Scope
>
> Small, and deliberately so:
>
> - `src/vt/rocm/rocm_backend.hip` — six virtuals mirroring
>   `src/vt/cuda/cuda_backend.cu:194-286`. Every CUDA call there has a direct HIP
>   equivalent (`hipStreamBeginCapture`, `hipGraphInstantiate`, `hipGraphLaunch`,
>   `hipGraphExecDestroy`, ...) — the same near-1:1 correspondence that lets
>   upstream compile `csrc/` for both through hipify.
> - `src/vllm/platforms/rocm.cpp` — add the `support_static_graph_mode()`
>   override.
> - `tests/vt/test_rocm_backend.cpp` — a RED-first capture/replay case mirroring
>   `test_cuda_backend.cpp:102-153`: capture a d2d copy, replay, then **mutate
>   the source at the same address and replay again**, which is the assertion
>   that catches a graph baking a stale pointer.
>
> **No model-level edit is needed.** Every decode-graph class already gates
> generically with no `is_cuda()` anywhere, so Qwen3 dense/MoE, DeepSeek-V2/V4,
> Voxtral and Laguna would pick this up for free.
>
> Worth noting this would also be the **second** implementation of that seam —
> Metal (`metal_backend.mm:13`) and Vulkan (`vulkan_backend.cpp:16`) both carry
> the same "stays FALSE" note, so `vt::Backend`'s capture abstraction currently
> has exactly one implementation and is therefore unproven as an abstraction.
> hipGraph is the cheapest available second, since `MTLIndirectCommandBuffer` and
> a pre-recorded `VkCommandBuffer` are genuinely different models.
>
> ## Known risk before anyone starts
>
> `src/vt/rocm/rocm_matmul_hipblaslt.hip:243-255` (`LtWorkspace`) calls
> `hipMalloc`/`hipFree` **lazily inside the GEMM call path**, growing the
> hipBLASLt workspace on demand. Allocation inside a capture region is illegal
> and invalidates the capture — CUDA's own contract comment
> (`cuda_backend.cu:186-190`) names exactly this hazard. The decode-graph
> pre-warm should grow it to high-water mark first, but that needs verifying
> rather than assuming. It fails loudly at `hipStreamEndCapture` rather than
> silently, which is the acceptable direction.
>
> ## Repro
>
> ```sh
> # ours
> ./build-hip/examples/vllm-bench --model <Qwen3-1.7B bf16> \
>   --num-prompts 8 --input-len 128 --output-len 128 --concurrency 8 --temperature 0
>
> # vLLM-ROCm oracle at pin 555967922, production config
> docker run --rm --device=/dev/kfd --device=/dev/dri \
>   --security-opt seccomp=unconfined --shm-size=8g \
>   -v <model-dir>:/models/m:ro vllm-rocm-oracle:555967922-gfx1200 \
>   vllm bench throughput --model /models/m --input-len 128 --output-len 128 \
>   --num-prompts 8 --dtype bfloat16 --max-model-len 2048 --gpu-memory-utilization 0.6
> ```
>
> The oracle container recipe (`rocm/vllm-dev:base` + the pinned commit built from
> source, ~6.5 min, hipify reports zero unsupported CUDA calls) is in
> `.agents/specs/rocm-gfx1200-m2-correctness.md`.
>
> Numbers above are single runs on a board that also drives a display —
> indicative, not the reproduced-idle standard a perf gate requires.
>
> **Spec:** `.agents/specs/rocm-decode-graph.md`, which carries the port map, the
> RED-first test design with the two mutations that must turn it red, the
> pre-registered gate-5 prediction, and a numeric stop threshold fixed in
> advance.

## Resolution

-
