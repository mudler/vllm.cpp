# ROCm grouped quant expert GEMM (kMatmulBTQuantGrouped) — spike

**Issue:** #41 (ROCm lane); the named blocker for MoE-bearing models after the
GDN slice (#334–#345) and the MoE chain (#348, #509).
**Status:** spike — no code yet.

## The gap, verified

On discrete ROCm the MoE path now resolves everything except the expert GEMM:
`kMoeRouterTopK` + `kMoeSiluMul` (#348), `kSharedExpertGate`/`kMoeCombine`/
`kMoeCombineGate` (#509) are native and gated. The remaining throw is
`kMatmulBTQuantGrouped` — the keep-quant grouped expert GEMM that runs the
stacked `[E*N,K]` expert towers. Without it, MoE-bearing models
(Qwen3.5-27B-class GDN-MoE, DeepSeek-V4 GGUF) throw on discrete ROCm.

`test_bench`/`test_capi` flipped green once the chain ops landed (they don't
reach the grouped GEMM). `test_loaded_engine_dense` still fails — but on the
**async-scheduling assertion** (`runner_supports_async()=false` on ROCm), a
lane capability gap unrelated to this op.

## What the donor actually is

`src/vt/cuda/cuda_quant_dot.cu` (2069 lines). The grouped GEMM is
`QuantDotGemmGroupedKernel` (:746) + a fused SwiGLU variant (:799) +
Q8_0-specific kernels (:1404/:1441). Structure:

1. **Shared activation quant**: input rows quantized to Q8_K once
   (`QuantizeRowQ8_K`, CPU ref `cpu_quant_act.cpp:88`; a `QuantizeQ8_0Kernel`
   device quantizer exists for the Q8_0 path).
2. **Per-format integer dot superblocks**: `DotSuperblock<W>` specializations
   (`:655`+) for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, IQ2_XXS/IQ3_XXS —
   each dequantizes a keep-quant weight superblock and dots against the Q8_K
   activation block. This is the bulk and the only genuinely tricky part.
3. **Grouped dispatch**: warp-per-(p,j), `__shfl_down_sync` reduction
   (HIP-compatible as-is), expert row selected by `expert_ids[p]`.

The CPU reference (`cpu_quant_dot.cpp` VecDot family) is complete and is the
gate oracle. HIP needs no torch; the donor's torch surface is only the host
glue.

## Port plan (per-format PRs, red-first, CPU-oracle gated)

- **W0: Q8_K activation quant + Q8_0 dot + grouped skeleton.** Smallest
  end-to-end slice that runs a real (if low-value) grouped GEMM; establishes
  the registration, the Q8_K quantizer port, and the cross-device gate vs
  `VecDotQ8_0Q8_0` (cpu_quant_dot.cpp:88). RED: op unregistered today.
- **W1: Q4_0 + Q4_K** (`VecDotQ4_0Q8_0` :50, `VecDotQ4_KQ8_K` :203) — the
  dominant GGUF expert formats.
- **W2: Q5_K/Q6_K/Q2_K** and the fused SwiGLU variant (the ds4 epilogue).
- **W3: IQ2/IQ3** — lowest-value, last.

Each family: hand-port from the donor's `DotSuperblock`, cross-device case vs
the CPU VecDot oracle (NMSE ≤ 5e-4 — the same band the CUDA lane uses, since
the integer core is bit-exact and only the float scale sum reassociates),
focused + full gate.

## Testability constraint (honest)

The op is only reachable end-to-end on MoE models I cannot fit on this box
(Qwen3.5-27B needs a multi-GB GGUF; the 0.8B has no experts). So the gate is
the **CPU reference at the op level** (cross-device, both groupings, the
broadcast-activation arm), and the model-level e2e stays PENDING a host with
the checkpoint — that is a real constraint, stated, not papered over.

## What is deliberately not in scope

- The fused SwiGLU grouped kernel (W2, a perf/composition variant).
- ggml's SIMD-table IQ formats' fastest paths (port the reference math first).
- Any perf tuning — correctness first; the win over "no path at all" is
  binary.

## Stop conditions

- A format's dot cannot be made NMSE-clean vs the CPU VecDot oracle → stop and
  post the failing evidence on #41 rather than ship a wrong quant path.
- A model-level e2e claim is ever made from op-level-only evidence → it must
  not be; the constraint above holds.
