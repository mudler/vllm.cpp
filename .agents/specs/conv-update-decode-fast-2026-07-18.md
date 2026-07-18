# GDN decode conv-update kernel-efficiency port (KERNEL-SSM-MAMBA) — 2026-07-18

**Row:** `KERNEL-SSM-MAMBA` (Qwen GDN causal-conv / state path). **Claim:**
`CLAIM-CONV-UPDATE-FAST-1` (decode-glue headroom sub-lever feeding
`SERVE-GATE-ONLINE`; the SSM/Mamba row stays `INVENTORIED`, no row-state change —
this adds a bit-identical fast variant of the existing conv-update kernel). **Kind:**
the c16-trace-identified remaining GDN scan lever (#5): our shipped conv-update
`~0.584 µs/step` vs vLLM `~0.432 µs/step`. **Verdict:** see the flip decision below
(gated on the isolated ≥1.3× microbench bar per the task contract).

Also in this change (`CLAIM-CONV-UPDATE-FAST-1`, row `KERNEL-GEMM-NVFP4-W4A4`): the
two bit-identical FP4-quant/SiLU fast kernels (`VT_FP4_QUANT_FAST`,
`VT_SILU_FP4_FAST`, landed OPT-IN at 861b518) are flipped **DEFAULT ON** per the
parity-enabler policy — byte-exact ⇒ never-slower + token-safe by construction.

## Scope

- In: the GDN decode `causal_conv1d_update` per-step conv-state roll —
  `CausalConv1dUpdateKernel` (`cuda_gdn.cu:549`), one thread per (token, channel),
  one launch/step, batch-independent. Add a bit-identical fast kernel behind
  `VT_CONV_UPDATE_FAST` (opt-in `=1` unless the isolated microbench passes ≥1.3×),
  mirroring the RMSNorm/gated decode-fast bit-safety technique
  (`cuda_ops.cu RmsNormRowFastKernel` 348d12d; `cuda_gdn.cu RmsNormGatedRowFastKernel`
  9ecd9d0). No model/scheduler/cache-layout change.
- In (FP4/SiLU): flip the `fp4_quant_fast.h` predicates DEFAULT ON (bit-identical,
  proven byte-exact at 861b518); invert the CPU flag test RED→GREEN and the CUDA
  bit-identity test's baseline arm (scalar = `=0`).
- Out: the prefill `CausalConv1dFwd`/`CausalConv1dFwdTiled` (VT_CONV_TILED, separate),
  the packed-decode / recurrence / delta_h / gated-norm GDN kernels, l2norm, the SSM
  cache gather/scatter, runner/scheduler.
- Prohibited files (other claims): `cuda_ops.cu`, `gdn_packed_*`,
  `gdn_packed_reg_tile.*`, the recurrence/prefill kernels in `cuda_gdn.cu`,
  runner/scheduler. Owned: the `CausalConv1dUpdateKernel` launcher path + the new
  fast kernel/launcher + `conv_update_fast.h` + its tests; `fp4_quant_fast.h` + its
  CPU/CUDA flag tests.

## Upstream chain (what vLLM does for the decode conv-update)

- vLLM's decode conv-update is the FLA Triton `_causal_conv1d_update_kernel`
  (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:15-192` @ pin `e24d1b24`),
  a `BLOCK_N`-channel program that loads each conv-state column ONCE into registers
  (`col0..col3`, specialized per `KERNEL_WIDTH` 2-5, `:158-192`), computes the conv,
  and rolls the state — reusing the register-held columns for both the accumulation
  and the state write. The channel axis is the contiguous (`stride_conv_state_dim`)
  dimension, so its per-column load across `BLOCK_N` channels coalesces.
- Qwen GDN uses `KERNEL_WIDTH = linear_conv_kernel_dim = 4` (state width `k-1 = 3`),
  `conv_dim = 2*key_dim + value_dim`, conv_state cache bf16 by default
  (`model_registry.cpp:307,335,311`). One conv-update launch per decode step.

## Our baseline

Shipped `CausalConv1dUpdateKernel<Tin,Tout,TState>` (`cuda_gdn.cu:549`): a flat
grid-stride kernel, one thread per flat index `idx = bt*c_dim + c`, deriving
`(token bt, channel c)` via two int64 `idx/c_dim` + `idx%c_dim` (expensive 64-bit
div/mod). Each thread reads its `k-1`-element state row, accumulates the conv
(`acc += w[j]*state[j]` then `w[width]*x`), stores `silu?/acc`, then RE-READS
`state[j+1]` from global memory to roll the row left and appends the raw `x`.
Correct and oracle-exact (part of the through-stack reference), but it (1) pays two
int64 div/mod per thread and (2) re-loads the state row it already read.

## Port map

| File | Change |
|---|---|
| `src/vt/cuda/conv_update_fast.h` (new) | `ConvUpdateFastFlagIsOn` predicate + rationale (bit-identity term-by-term, the two numerics-neutral mechanics). |
| `src/vt/cuda/cuda_gdn.cu` | new `CausalConv1dUpdateFastKernel<Tin,Tout,TState,WIDTH>` (2D grid: `blockIdx.y`=token, x-dim=channel → no div/mod; state row register-cached, WIDTH-templated so the loops unroll and it stays register-resident) + `TryLaunchConvUpdateFast` (flag + width∈{1,2,3,4} + `nt≤kMaxGridY` guard) wired into `LaunchConvUpdate` before the shipped launch; `#include` the flag header. Shipped kernel untouched. |
| `tests/vt/test_conv_update_fast.cpp` (new) + `tests/CMakeLists.txt` | CPU flag-parse contract (default OFF / '1'-opt-in). |
| `tests/vt/test_ops_gdn.cpp` | `RunConvUpdateFastCase` helper + the `CUDA causal_conv1d_update decode-fast … 0-ulp` TEST_CASE (both `out` + rolled conv_state byte-exact; k∈{3,4,5}, bf16+f32 state, both in/out dtypes, ±bias, silu/identity, compact + scattered cache_idx incl. NULL-block). |
| `src/vt/cuda/fp4_quant_fast.h` + `tests/vt/test_fp4_quant_fast.cpp` + `tests/vt/test_ops_nvfp4_fp4.cpp` | flip `VT_FP4_QUANT_FAST`/`VT_SILU_FP4_FAST` DEFAULT ON ('0'-rollback); flag test RED→GREEN; CUDA bit-identity baseline arm → `=0`. |

## Bit-identity argument (0-ulp, by construction)

`CausalConv1dUpdateFastKernel` keeps the shipped float op sequence EXACTLY: same
`bias` init, same left-to-right `acc += Load(w,c*k+j) * state[j]` accumulation (with
`state[j]` the SAME cached bits `Load(srow,j)` produces), same `w[width]*x` tail,
same `silu?Silu(acc):acc` epilogue, same `Store` round-to-dtype; the roll writes the
same `state[j+1]` bits (cached) and the same raw `x`. The ONLY differences are
numerics-neutral: (1) index math via a 2D grid instead of int64 div/mod (index
arithmetic is not part of the numeric output), and (2) loading the state row once
into registers instead of re-reading `state[j+1]` in the roll. Both leave every
loaded value and every stored byte identical ⇒ `out` and rolled `conv_state` are
byte-identical to shipped. Same for the FP4/SiLU fast kernels (proven byte-exact at
861b518). Per the a875397 all-bit-identical ⇒ safe lesson, the combined default set
stays token-exact.

## Gates

1. CPU flag-parse contracts (every platform): `test_conv_update_fast` (default OFF /
   '1'), `test_fp4_quant_fast` (RED→GREEN, default ON / '0'). Full tools + ctest.
2. CUDA 0-ulp bit-identity: `test_ops_gdn` conv-update decode-fast fast==shipped
   byte-exact (both `out` + conv_state); `test_ops_nvfp4_fp4` FP4/SiLU byte-exact.
3. Isolated microbench (nsys, one flock): conv-update fast vs shipped µs/step at the
   dominant 27B decode shape → flip decision (≥1.3× ⇒ default ON, else opt-in).
4. Full prospective default set (async + cubin + RMSNorm-fast + gated-fast +
   FP4-fast + SiLU-fast + conv-fast-if-flipped) `test_qwen27_paged_engine` 235/235
   (tok6=198) + `test_qwen36_paged_engine` 315/315, plus the `=0` rollback arms.
5. Clean `-Werror` CUDA rebuild, memcheck, doc/record checkers.

## Flip decision

Recorded at the closing checkpoint from gate 3 (isolated µs / ×speedup) — default ON
iff ≥1.3× + bit-exact, else lands opt-in (`VT_CONV_UPDATE_FAST=1`). The orchestrator
runs the binding re-grid to measure the in-situ effect.
