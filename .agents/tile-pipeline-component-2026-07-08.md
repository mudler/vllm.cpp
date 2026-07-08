# vt::tile — the portable async-pipeline / tile component ("what Triton does, internally")

**Decision (user, 2026-07-08):** *"can't we have a component here that does what
triton does, and use it internally in vllm.cpp?"* — YES. Build a **portable C++
tile + async-pipeline abstraction**, grounded 1:1 in CUTLASS/CuTe, and write the
hot kernels against it. This supersedes both the one-off hand-transcription
(doesn't scale) and the Triton-AOT compile-target (violates mission.md "no
Python at build time" + discipline.md "Kernel-DSL is a reference, never a
compile-target"). It is the canon-consistent way to get Triton-grade codegen.

## What "does what Triton does" means here (the crux)

Triton's value on the hot path is NOT the DSL — it's the **compiler passes**:
software **pipelining** (multi-stage async gmem→smem double-buffering + barrier
scheduling), **swizzled smem layout** (bank-conflict-free), and **MMA lowering**.
On Blackwell it emits `cp.async` (Ampere) → **TMA + `mbarrier`** (Hopper/BW).
Our ad-hoc hand `cp.async` went NEGATIVE because we bolted it on per-kernel
without the double-buffer/scheduling discipline. The fix is to encode that
expertise ONCE as reusable primitives — which is exactly what CUTLASS/CuTe are
(and what flashinfer/vLLM's fast kernels are built on).

## Why this satisfies the canon

- **Pure C++, no build-time Python** (mission.md) — hand-written primitives,
  compiled by nvcc. No Triton/DSL at build or run time.
- **Not a compile-target** (discipline.md 33-47) — we READ CUTLASS/CuTe/CuTe-DSL
  as the 1:1 porting reference and transcribe to portable C++, citing file:line.
- **Portable seam** — the ABSTRACTION (tile, copy, mma, pipeline-state) is
  backend-neutral; each backend supplies its ATOMS (CUDA: cp.async/TMA+mbarrier;
  Metal: simdgroup_matrix; Vulkan: cooperative-matrix; CPU: scalar ref). Same
  model as `vt::` already uses. Multi-backend contract intact.
- **Scales** — adding a fast kernel = compose primitives, not hand-roll staging.

## Grounding (upstream, cited)

- **What vLLM runs on GB10 (sm_121):** confirmed by dispatch + profile. The
  CuTe-DSL GDN path (`vllm/.../mamba/ops/gdn_chunk_cutedsl/`) is **opt-in +
  SM10.x-only** (`qwen_gdn_linear_attn.py:152-210` `supports_cutedsl` needs
  SM10.x + cuda≥13); sm_121 falls to **FLA Triton
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`** (matches the captured
  profile name). So the ALGORITHM to match = FLA blockdim64
  (`vllm/.../fla/ops/chunk_delta_h.py`); the CODEGEN to match = Triton's
  Blackwell async pipeline. The CuTe-DSL `kernel_h.py` (same math, explicit
  `cpasync.TmaCopyOp` + `mbarrier_init` + `num_stages`, 128B swizzle) is the
  cleanest C++ port reference for that pipeline structure.
- **Rung-1 primitive (sm80 cp.async multistage):** `cute/arch/copy_sm80.hpp:40-193`
  (`SM80_CP_ASYNC_CACHEGLOBAL/ALWAYS_ZFILL`, `cp_async_fence`, `cp_async_wait<N>`).
  → `include/vt/cuda/tile/cp_async.cuh` (ported, this change).
- **Ring state:** `cutlass/pipeline/sm90_pipeline.hpp:171-250` `PipelineState<Stages>`.
- **Rung-2 primitive (TMA + mbarrier):** `cutlass/pipeline/sm90_pipeline.hpp`
  `PipelineTmaAsync` (producer_acquire/commit, consumer_wait/release) +
  `cute/arch/copy_sm90_tma.hpp`. Deferred — see rung strategy.

## Rung strategy (the user's "transcribe first, escalate if needed")

1. **Rung 1 — sm80 cp.async multistage.** Low risk, well-understood, portable-ish
   atom. Port the primitive (done: `cp_async.cuh`), then rewrite GDN `delta_h`
   (blockdim64) as a FAITHFUL structural port staged through it (this time the
   real BV×64 register tiles + K-split + N-stage double-buffer, not our ad-hoc
   tiling). Token-exact + gate A/B.
2. **Rung 2 — TMA + mbarrier** (`PipelineTmaAsync`). Only if Rung 1 measurably
   lags — because Blackwell's real codegen is TMA-based. Higher complexity
   (mbarrier init/phase, transaction bytes, warp specialization). Justified by
   evidence, grounded in the CuTe-DSL `kernel_h.py` which uses exactly this.

## First target & success bar

`GdnChunkDeltaHWmmaKernel` (the ~3.7% blockdim64 bucket where ad-hoc cp.async
went neutral/negative). Bar: token-exact (test_ops_gdn) AND a measurable gate
A/B win (in1024/out128 conc32) vs the current hand kernel — same-binary toggle
`VT_GDN_TILE_PIPE`. If Rung 1 stays neutral, escalate to Rung 2 before concluding.

## Shelved (not deleted): Triton AOT toolchain

`perf/triton-fastpath` branch (proven end-to-end on dgx, token-exact rmsnorm).
Kept OFF as a **measurement oracle** — it can compile FLA's exact kernel to cubin
to give us the ground-truth "what Triton's codegen achieves" number to target,
without shipping it. Not merged; violates the canon for shipping.

## Campaign sequence (user, 2026-07-08: "after that we chase the other levers then reassess")

1. **delta_h faithful register-tiled port** (H in accumulator registers, smem freed
   for the cp.async ring) — the true 1:1 FLA blockdim64. Rung-1 staging-add-on was
   token-exact but NEUTRAL: our H-in-64KiB-smem starves the ring; FLA holds H in
   registers (`b_h1..b_h4`). IN PROGRESS.
2. **chunk_o** via `vt::tile` (no persistent state → ring fits, more parallel).
3. **recompute_w_u** via `vt::tile`.
4. **fused norm+quant chains** (~8%) — the TDR/fused-recipe framework (fusion, not
   just pipelining). Reuses the component where staging applies.
5. **Reassess** — re-profile vLLM vs ours, re-rank the residual gap.
