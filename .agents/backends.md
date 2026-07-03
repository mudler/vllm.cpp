# Backend portability strategy (NVIDIA · Apple · Vulkan · Intel · CPU)

**Principle:** the engine (scheduler, KV/block management, persistent batch,
sampler pipeline, serving) is 100% backend-agnostic — in upstream vLLM it
already is; block tables and metadata are plain integers/arrays that kernels
consume. Portability therefore lives in exactly three mirrored surfaces, and
nowhere else:

1. **Platform** — mirror of `vllm/platforms/` (`interface.py::Platform`;
   upstream ships cpu, cuda, rocm, tpu, **xpu**, zen_cpu). Device probing,
   memory model, stream/queue semantics, graph-capture capability.
2. **Attention/GDN backends** — mirror of `vllm/v1/attention/backends/`
   registry: each platform registers its paged-attention + GDN
   implementations behind the same `AttentionBackend/Impl/MetadataBuilder`
   contract.
3. **vt:: op tables** — our compute layer (deviation §9): per-device kernel
   registration for GEMM/norms/rope/activations/MoE/sampling ops.

Adding a platform never touches engine code. This is what "exploit each
platform's inner goods without sacrificing vLLM 1:1 loyalty" means concretely:
loyalty is preserved because the *seams* are vLLM's own seams; the platform-
specific brilliance (unified memory, graph replay, ANE, cooperative matrix)
lives entirely below them.

## Requirements this puts on vt:: (binding for M0.2)

- Device is an open enum (`CUDA`, `CPU`, later `METAL`, `VULKAN`, `XPU`) —
  no `#ifdef CUDA`-shaped assumptions in engine-visible types.
- Memory API distinguishes discrete vs **unified** memory (GB10 and Apple
  Silicon are both unified — same design pays twice).
- Explicit stream/queue handle per device; ops take it; no global stream.
- Optional per-backend **graph/command-capture hook** (CUDA Graphs ↔ Metal
  indirect command buffers ↔ Vulkan pre-recorded command buffers) behind one
  interface; engines see "capture(replayable step)" or nothing.
- Quantized-weight layout is backend-negotiated: NVFP4 W4A4 is
  NVIDIA-specific; Metal/Vulkan/XPU lean on GGUF quant kernels (k-quants,
  and their own fp4/int4 paths where hardware supports it).
- Known M0.6 decisions deferred: backend registry is per-DeviceType (multi-GPU needs per-Device or Alloc(Device,...)); kernel fn-type aliases move to ops.h at second-backend registration; op validation currently pins out=f32 (widen for bf16 CUDA outputs).

## Per-platform status & strategy

| Platform | Upstream equivalent | Strategy | Tier |
|---|---|---|---|
| NVIDIA CUDA (GB10 first) | `platforms/cuda.py` | The gate. Native CUDA kernels + cuBLASLt + CUDA graphs | T0 |
| CPU | `platforms/cpu.py`, `csrc/cpu/` | Correctness-grade reference for CI/parity | T0 (correctness) |
| Intel XPU | **`platforms/xpu.py` exists upstream** | Loyal port: SYCL/oneAPI kernels (upstream uses oneDNN/IPEX paths; we write SYCL against the same contracts). Battlemage/Arc + Data Center Max | T2 |
| Apple Metal | none — vllm.cpp extension | New `platforms/metal` following the mirrored Platform contract; kernel layer per exploration E1/E2 below | T2 |
| Vulkan | none — vllm.cpp extension | New `platforms/vulkan`; hand-written compute shaders, `VK_KHR_cooperative_matrix` for MMA-class GEMM; llama.cpp's Vulkan backend is the maturity reference | T2 |
| AMD ROCm | `platforms/rocm.py` | Not scoped yet; falls out of the same seams when wanted | T3 |

Extensions (Metal, Vulkan) are recorded as deviations in
[porting-inventory.md](porting-inventory.md) §9 — upstream has no such
platforms; we add them *through* upstream's own abstraction so a future
upstream platform PR ports mechanically.

## Explorations (open questions with decision criteria)

- **E1 — MLX as the Apple kernel layer.** MLX is Apple's C++ array framework
  (Metal kernels, unified memory, C API via mlx-c). Attractive: mature
  GEMM/attention kernels, custom-MSL-kernel API, active development. Risks:
  **lazy-evaluation graph semantics** vs our eager persistent-batch design
  (same impedance class that disqualified ggml — needs a forced-eval eager
  wrapper and profiling to see what that costs); dependency weight vs the
  "we own the code" posture; MLX does **not** use the ANE (GPU only).
  Decide by benchmark: MLX-backed vt ops vs hand-written MSL (E2) on paged
  attention + GDN + MoE decode for a gate-class model. Criteria: decode
  tokens/s at high concurrency, memory control (paged KV must be raw
  buffers we manage), build/maintenance cost.
- **E2 — Hand-written Metal (llama.cpp-style).** Full control, zero deps,
  proven by llama.cpp on Apple Silicon; higher initial kernel cost. Default
  if E1's eager-wrapper tax is real.
- **E3 — ANE.** Reachable only via CoreML compiled models: static shapes,
  no paged KV, no custom ops — a poor fit for autoregressive decode with
  dynamic batching. Realistic roles: encoder/embedding/pooling models (T2
  inventory families), or fixed-shape draft models for spec decode. Treat as
  a specialized accelerator for select model classes, not a serving backend.
  Revisit when Apple opens lower-level ANE access.
- **E4 — Intel kernel sourcing.** SYCL portability vs Level Zero directness;
  whether upstream's XPU attention contracts translate 1:1 to our SYCL
  kernels. Explore when T2 scheduling begins.

## Non-negotiables

- The NVIDIA gate (see [gates.md](gates.md)) is not delayed by portability
  work: M0.2 bakes the interface requirements above (cheap now, unpayable
  later); backend implementations are post-MVP.
- A new backend lands like any port: parity harness (same golden dumps —
  they are backend-independent), behavioral suites unchanged, benchmark
  honesty per [workflow.md](workflow.md).
