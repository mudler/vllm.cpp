# GDN gated-RMSNorm decode kernel-efficiency port (KERNEL-EW-NORM-ACT) — 2026-07-17

**Row:** `KERNEL-EW-NORM-ACT` (norm/act family; the kernel physically lives in
`cuda_gdn.cu` with the other GDN kernels). **Claim:** `CLAIM-EW-NORM-GATED-1`
(perf/parity sub-lever feeding `SERVE-GATE-ONLINE`; the norm/act row stays `DONE`,
no row-state change). **Kind:** the c2/c8-attribution-identified batch-independent
gated-norm kernel-glue lever (`+0.40 ms/step` vs vLLM's fused gated norm,
[c2-c8 attribution](c2-c8-attribution-2026-07-16.md), evidence
`dgx:~/work/vllm.cpp-c2c8-attribution/beb8497`: "RMSNorm-gated 0.403/fused/+0.40").
**Verdict:** CONFIRMED (isolated ≥1.3× at every decode shape, c16 2.04×) and PORTED
BIT-IDENTICAL (0-ulp) behind `VT_RMSNORM_GATED_FAST`, default flipped ON.

## Scope

- In: the GDN gated RMSNorm launches — `RmsNormGated` over `Dv=128` at
  `[T*Hv, 128]` (27B `Hv=48`, 35B `Hv=32`), 48 launches/step (one per GDN layer),
  batch-independent. Add a bit-identical fast kernel behind `VT_RMSNORM_GATED_FAST`
  (default ON, `=0` rollback), mirroring the RMSNorm decode-fast bit-safety
  technique (`cuda_ops.cu` `RmsNormRowFastKernel`, 348d12d). No model/scheduler
  change.
- Out: plain `RmsNorm` (`KERNEL-EW-NORM-ACT` decode-fast, separate flag), FP4/FP8
  quant kernels, the packed-decode / recurrence / conv GDN kernels, the
  attention-preamble fused q/k norm.
- Prohibited files (other claims): `src/vt/cuda/cuda_ops.cu`,
  `cuda_matmul_nvfp4.cu`, `gdn_packed_*`, `gdn_packed_reg_tile.*`, the recurrence
  kernels in `cuda_gdn.cu`, runner/scheduler. Owned: the `RmsNormGatedRowKernel`
  launcher + the new fast kernel/launcher + flag header + tests.

## Upstream chain (what vLLM does for the gated norm)

- vLLM's gated norm is the Inductor `triton_poi_*mean_mul_pow_rsqrt*silu*` fusion
  (RMSNorm + silu/sigmoid gate) over the FLA `RMSNormGated`
  (`layers/fla/ops/layernorm_guard.py` `layer_norm_fwd_kernel` /
  `layernorm_guard.py:57` `out.to(dtype)`; `norm_before_gate=True`,
  `group_size=None` — the only config Qwen GDN uses). The portable CUDA mirror is
  the csrc-style one-block-per-row RMSNorm with the gate applied
  (`csrc/layernorm_kernels.cu:106-173` pattern), which our shipped
  `RmsNormGatedRowKernel` (`cuda_gdn.cu:851`) already realizes.

## Our baseline

Shipped `RmsNormGatedRowKernel<Tin,Tout>` (`src/vt/cuda/cuda_gdn.cu:851`): one block
per row, `kBlock=256` threads, scalar `Load()` (bf16→f32), shared-memory f32 tree
reduction of the per-element squares, `inv = 1.0f/sqrtf(var/d + eps)`, then a second
pass `out = ((x*inv)*w)*act` with `act = sigmoid ? 1/(1+exp(-z)) : silu(z)`. It
reloads `x` in the normalize pass and, for the production `d=Dv=128` shape, leaves
the upper half (threads 128-255) of every block idle in both passes. Correct and
oracle-exact (part of the 235/235 through-stack reference), but it is the SAME slow
pattern the plain-RMSNorm decode-fast (`RmsNormRowFastKernel`, cuda_ops.cu, 348d12d)
already superseded. Dispatch: `RmsNormGatedKernelCuda` (`cuda_gdn.cu:878`) over
f32/bf16 in × f32/bf16 out; the bf16/bf16 `d==128` case is the decode hot path.

## Port map

| File | Change |
|---|---|
| `src/vt/cuda/rmsnorm_gated_fast.h` (new) | `RmsNormGatedFastFlagIsOn` predicate (default ON, `=0` rollback), house convention. |
| `src/vt/cuda/cuda_gdn.cu` | new `RmsNormGatedRowFastKernel` + `TryLaunchRmsNormGatedFast` launcher; wire it into `RmsNormGatedKernelCuda` before the shipped dispatch; `#include` the flag header. Shipped `RmsNormGatedRowKernel` untouched. |
| `tests/vt/test_rmsnorm_gated_fast.cpp` (new) + `tests/CMakeLists.txt` | CPU flag-parse contract. |
| `tests/vt/test_ops_gdn.cpp` | `RunRmsNormGatedFastContiguous` + `RunRmsNormGatedFastPaddedRank3` helpers + the `CUDA rmsnorm_gated decode-fast … 0-ulp` TEST_CASE. |
| records | kernel-matrix `KERNEL-EW-NORM-ACT` cell, ledger row, coordination `CLAIM-EW-NORM-GATED-1`, state, README, docs/BENCHMARKS. |

## Why bit-identical is MANDATORY (a875397 lesson)

The gated norm sits in the same 27B forward as the fast plain-RMSNorm + GDN decode
cubin, all converging on the greedy RAZOR near-tie at output token 6 (198 prod vs
271 emu, ~zero logit gap). A ≤1-ulp-close gated kernel, accumulated over the
layers alongside the other kernels' own roundings, can tip that tie and fail the
production stream — exactly the failure that forced the a875397 RMSNorm revert.
Making the fast gated output the SAME BITS as shipped removes the perturbation by
construction: `{fast-RMSNorm + fast-gated + cubin}` logits == the passing baseline,
so it CANNOT cross the tie.

## The kernel (`RmsNormGatedRowFastKernel`, `cuda_gdn.cu`)

Regime: unlike the plain RMSNorm decode-fast (block-starved, ~16 rows ⇒ 1024
threads to hide latency), the gated norm launches `T*Hv` (~512–768 at c16) blocks
and is NOT block-starved. The shipped `RmsNormGatedRowKernel` inefficiency is that
it launches `kBlock=256` threads for a 128-element row — the upper half of EVERY
block is idle in both passes — and it reloads `x` in the normalize pass. The fast
kernel fixes both: `kGatedFastBlock=128` threads (one per `Dv` element, no idle
half ⇒ 2× the occupancy headroom) and `x` cached in a register (loaded once).

Bit-identity, term by term vs shipped `RmsNormGatedRowKernel` (`cuda_gdn.cu`):
1. **variance ORDER.** shipped: 256 threads, per-thread `acc = Σ_m sq[tid+256m]`;
   for `d==128` that is the SINGLE term `sq[tid]` (tid<128) or 0 (tid≥128), then the
   shared-tree `for s=128; s>0; s>>=1`. Fast: 128 threads set `partial[tid]=sq[tid]`
   and the tree runs from `s=64`; shipped's extra `s=128` step only adds the
   provably-zero `partial[128..255]`, so omitting it is byte-for-byte identical, and
   the `s=64..1` pairing is the same. `f32(bf16 x)^2` is the same squared value on
   both sides.
2. **inv.** `1.0f / sqrtf(partial[0]/d + eps)` verbatim (NOT `rsqrtf`).
3. **normalize + act.** shipped `Load(xrow,j)*inv*Load(w,j)*act` == `((x*inv)*w)*act`,
   `act = sigmoid_gate ? 1/(1+exp(-z)) : Silu(z)=z/(1+exp(-z))`; reproduced with the
   same left-to-right multiply order, same act, same `__float2bfloat16` store. The gate
   addressing `zrow = gate + (row/gate_group)*gate_outer + (row%gate_group)*d` is
   identical (contiguous rank-2 gate_group=1 and padded rank-3 gate_group=Hv both
   covered by the parity test).

Only the launch geometry (128 vs 256 threads) and the single register-cached `x`
load differ; the arithmetic is identical.

Guard (`TryLaunchRmsNormGatedFast`): bf16 in/out/gate/weight AND `d==128` (the only
production gated-norm shape). Out-of-scope keeps `RmsNormGatedRowKernel`.

## Flag

`VT_RMSNORM_GATED_FAST` — house predicate header
[`rmsnorm_gated_fast.h`](../../src/vt/cuda/rmsnorm_gated_fast.h),
`RmsNormGatedFastFlagIsOn` (DEFAULT ON; `=0` first-char rolls back), mirroring
`VT_RMSNORM_DECODE_FAST` / `VT_GDN_PACKED_DECODE_TRITON`. CPU-unit-tested in
[`test_rmsnorm_gated_fast.cpp`](../../tests/vt/test_rmsnorm_gated_fast.cpp).

## Tests to port / add

- CPU flag-parse contract (portable, every platform): `test_rmsnorm_gated_fast.cpp`
  (default-ON / `=0`-rollback), 10 assertions.
- CUDA bit-exact 0-ulp parity `fast==shipped`
  ([`test_ops_gdn.cpp`](../../tests/vt/test_ops_gdn.cpp)
  `CUDA rmsnorm_gated decode-fast … matches rollback kernel 0-ulp`): explicit-env A/B
  at the real decode shapes — contiguous rank-2 `[T*Hv,128]` (rows 48/96/512/768) AND
  padded rank-3 `[T,Hv,128]` strided gate (T∈{1,2,16}, Hv∈{48,32}), silu + sigmoid,
  adversarial x/z ∈ [−6,6]. `CheckClose(..., 0.0f, 0.0f)`.

## Gates (DGX, clean -Werror build, CUTLASS sm120a NVFP4 + FA2 sm_121a hard-verified, one flock; evidence `dgx:~/work/vllm.cpp-ewnorm-gated`)

- **Gate 1 (CPU):** clean `-Werror` build (CUDA build on dgx, 0 warnings); flag test
  10/10; full ctest; tools `unittest discover` 164; record + doc-checkpoint checkers.
- **Gate 2 (bit-exact):** `test_ops_gdn` gated decode-fast `fast==shipped` **0-ulp
  BIT-EXACT** — 140/140 assertions (contiguous + padded gate, silu/sigmoid, c1–c16).
- **Gate 3 (isolated perf, nsys pure-kernel, `/tmp/gated_mb`):** avg ns fast/shipped:
  **rows=768 (27B c16) 3285.7/6702.8 = 2.04×**, rows=96 (c2) 1549.7/2141.0 = 1.38×,
  rows=48 (c1) 1608.4/2114.9 = 1.31×; 4-shape aggregate avg 2201/4116 = 1.87× (median
  2400/4832 = 2.01×). ALL ≥ 1.3×.
- **Gate 4 (ENGINE token gates, full production default set = async ON + GDN cubin ON
  + RMSNorm-fast ON + gated-fast ON):** `test_qwen27_paged_engine` **235/235**
  (token 6 = 198) + `test_qwen36_paged_engine` **315/315**; both `VT_RMSNORM_GATED_FAST=0`
  rollback arms **235/235 + 315/315**; full GDN suite 50/50 (2483/2483). Bit-identity ⇒
  passes by construction.

## Dependencies

None blocking. Sits alongside the already-landed plain-RMSNorm decode-fast
(`VT_RMSNORM_DECODE_FAST`, `CLAIM-EW-NORM-ACT-3`), GDN decode cubin
(`VT_GDN_PACKED_DECODE_TRITON`), and async scheduling (`ENG-ASYNC-SCHED`), all
default ON; the full production default set is gated together (Gate 4). Feeds the
next `SERVE-GATE-ONLINE` binding grid. Does not touch any other claim's files.

## Work breakdown

1. Flag header + CPU flag test (RED→GREEN) + CMake row.
2. `RmsNormGatedRowFastKernel` + launcher + dispatch wiring in `cuda_gdn.cu`.
3. CUDA 0-ulp bit-exact A/B test (contiguous + padded gate, silu/sigmoid, c1-c16).
4. CPU gates (clean `-Werror` build, ctest, tools, checkers).
5. DGX gate series under one flock: bit-exact, isolated microbench, ENGINE token
   gates (default + rollback arms).
6. Flip decision + records + push.

## Risks/decisions

- **Bit-identity is the acceptance floor, not ≤1-ulp** (a875397): decided to
  reproduce shipped's exact op sequence so `fast+cubin+fast-RMSNorm ≡ baseline` by
  construction. Verified 0-ulp (Gate 2, 140/140).
- **Scope decision `d==128`**: the gated norm is `Dv=128` for both gate models
  (27B Hv=48, 35B Hv=32); the guard is `d==128` exactly, which keeps the
  bit-identity proof airtight (single-term partials) and matches the only
  production shape. The task's "1024≤H≤8192" guard was templated from the plain
  RMSNorm and does NOT apply here (that was the block-starved 1024-thread regime;
  the gated norm launches T*Hv blocks and is not block-starved). Recorded deviation.
- **Thread-count decision 128 not 256/1024**: one thread per element removes the
  idle upper half (the shipped waste) while preserving the exact reduction order;
  fewer threads would require a strided per-thread accumulation that changes the
  f32 sum order (not bit-identical), so 128 is the natural bit-identical choice.

## Decision

Gates 1–4 PASS AND isolated ≥1.3× at every decode shape (c16 2.04×). Per the
parity-enabler policy (binding grids measure production defaults) the
`VT_RMSNORM_GATED_FAST` default is **flipped ON** (`=0` rollback); bit-identity makes
the flip safe (`fast+cubin+fast-RMSNorm ≡ the passing baseline` by construction). The
next binding grid re-measures the production default. This closes the batch-independent
gated-norm glue component of the c2–c4 kernel-glue floor identified by the c2/c8
attribution.
