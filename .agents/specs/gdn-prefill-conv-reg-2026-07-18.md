# GDN prefill causal-conv1d + fused post-conv kernel-efficiency port (KERNEL-SSM-MAMBA) — 2026-07-18

**Row:** `KERNEL-SSM-MAMBA` (Qwen GDN causal-conv / state path). **Claim:**
`CLAIM-GDN-PREFILL-CONV-1` (35B low-batch prefill sub-lever; the SSM/Mamba row stays
`INVENTORIED` — this adds bit-identical fast variants of the existing PREFILL conv +
post-conv kernels, no row-state change). **Kind:** the prefill-attribution GDN lever
— nsys 35B prefill attributed `CausalConv1dFwdTiled` 5.9% + `GdnPostConv` 5.5% =
11.4% of prefill GPU time. **Verdict:** conv register-window kernel ships DEFAULT ON
(bit-exact + consistently faster); post-conv split ships OPT-IN (bit-exact, measured
near-neutral). See "Measured" below.

## Scope

- In: the GDN PREFILL `causal_conv1d` forward (`CausalConv1dFwdTiledKernel` /
  `CausalConv1dFwdKernel`, `cuda_gdn.cu`) and the fused post-conv prep
  (`GdnPostConvKernel`). Add bit-identical fast kernels:
  - `CausalConv1dFwdRegKernel` behind `VT_CONV_REG` (DEFAULT ON, `=0` → tiled): the
    register-resident sliding-window mirror of vLLM's FLA conv fwd.
  - `GdnPostConvSplitKernel` behind `VT_GDN_POSTCONV_SPLIT` (OPT-IN, unset → shipped
    megablock): the per-V-head split mirror of vLLM's fused post-conv grid.
- Out: the decode `causal_conv1d_update` (VT_CONV_UPDATE_FAST, sibling
  `CLAIM-CONV-UPDATE-FAST-1`), the packed-decode / recurrence / delta_h / gated-norm
  GDN kernels, l2norm, the SSM cache gather/scatter, runner/scheduler. The bf16
  post-conv-activation streaming (VT_GDN_IN_BF16, halved conv traffic) is task #40's
  sibling lever and stays separate.
- Owned files: the `CausalConv1dFwd`/`GdnPostConv` launcher paths + the two new
  kernels/launchers in `cuda_gdn.cu` + `src/vt/cuda/gdn_prefill_conv.h` + its tests.

## Upstream chain (what vLLM runs for the prefill conv + post-conv)

- Conv fwd: FLA Triton `_causal_conv1d_fwd_kernel`
  (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:397-452` @ pin `e24d1b24`).
  Key structure: (1) PRE-LOAD the `KERNEL_WIDTH` per-channel weights into registers
  once (`w_col0..w_col3`, lines 399-409); (2) keep the `(k-1)` history taps in a
  REGISTER sliding window (`col0..col3`, lines 414-452) so each `x` element is loaded
  from global EXACTLY ONCE, reused across the `k` taps, then the window shifts —
  no shared memory, no barriers; (3) parallelize the token axis into `BLOCK_M`
  chunks (`chunk_offset`, `token_offset = BLOCK_M*chunk_offset`, line 123) so a long
  single-sequence prefill saturates the GPU. Accumulation is
  `acc = bias; for j: acc += matrix_x * matrix_w` (line 442), taps oldest→newest.
- Fused post-conv: `_fused_post_conv_kernel`
  (`vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py:19-149`,
  `fused_post_conv_prep` grid `(cdiv(L,BLOCK_T), H+HV)`, line 214). `program_id(1)`
  in `[0,H)` does the q/k L2-norm; in `[H,H+HV)` EACH V head is its own program doing
  the V copy + `g = -exp(A_log)*softplus(a+dt_bias)` / `beta = sigmoid(b)` gating.

## Our before-state

- `CausalConv1dFwdTiledKernel` (`cuda_gdn.cu`, VT_CONV_TILED default ON) stages an
  `[BLOCK_M+width, BLOCK_N]` x-tile through SHARED memory with two `__syncthreads()`
  per token-tile and RE-LOADS `w[c*k+j]` from GLOBAL on every tap of every output.
- `GdnPostConvKernel` grid `(T, Hk+1)`: the `head==Hk` block packs the ENTIRE
  `value_dim = Hv*Dv` V copy + all `Hv` gating scalars into ONE block per token.

## What changed

- `CausalConv1dFwdRegKernel<Tin,Tout,THas>` + `LaunchConvFwdReg` + `ConvRegEnabled()`:
  one thread per channel; weights pre-loaded to registers; `(k-1)`-tap register
  sliding window (each x loaded once, coalesced across channels); grid
  `(cdiv(c,128), n, gridZ)` where `gridZ` chunks the token axis (`kConvRegM=32`) ONLY
  for `n <= kConvRegChunkMaxSeqs=4` (few long sequences — the c1-c4 low-batch prefill
  where one block/(channel-tile,seq) under-occupies); bounded by `cdiv(total,M)`
  (exact for `n==1`). For `n>4`, `gridZ==1` and each block streams its whole sequence.
- `GdnPostConvSplitKernel<Tqkv,Tconv,Tgate>`: grid `(T, Hk+Hv)`, each V head its own
  block; the `head<Hk` q/k L2-norm branch is byte-for-byte the shipped code.
- Dispatch: `ConvFwdKernelCuda` → reg (default) else tiled else scalar;
  `GdnPostConvKernelCuda` selects split (opt-in) vs megablock and sets grid.y.

## Bit-exactness argument

Both kernels preserve the shipped per-output float op ORDER over the SAME f32 values:
- Conv: `acc = bias; for j in [0,k): acc += w[c*k+j]*x_tap[j]` (j=0 oldest..k-1 newest),
  same silu/identity epilogue + round-to-store, same `(K-1)` state write-back (reads
  raw x / shifted conv_state directly, never a value the window mutated). The register
  window holds the identical `Load(x,...)`/`conv_state` bytes the scalar/tiled kernels
  read; a conv is a fixed sum of `k<=4` f32 products, so preserving order ⇒ 0-ulp.
- Post-conv: q/k L2-norm branch is byte-identical; V copy and per-head gating are the
  same per-element ops, only split across grid.y blocks. No arithmetic reorders.

## Tests to port / add

- `src/vt/cuda/gdn_prefill_conv.h` — pure `ConvRegFlagIsOn` / `GdnPostConvSplitFlagIsOn`
  predicates (CPU-tier). Test: `tests/vt/test_gdn_prefill_conv.cpp` (VT_CONV_REG
  default-ON `0`-rollback; VT_GDN_POSTCONV_SPLIT default-OFF opt-in).
- CUDA byte-exact A/B (DGX-gated, `tests/vt/test_ops_gdn.cpp`):
  `RunConvFwdRegByteExactCase` (reg vs tiled, `out`+`conv_state` raw-byte equal over
  k∈{3,4,5}, bf16+f32, chunked/no-chunk multi-seq, T<K-1, i8 mask) and
  `RunGdnPostConvSplitByteExactCase` (split vs megablock, all five outputs raw-byte
  equal over the qkv×conv×gate dtype cube at 35B dims).

## Gates + Measured (DGX GB10, 35B `nvidia/Qwen3.6-35B-A3B-NVFP4`, one flock series)

- CPU: clean `-Werror` build; ctest 122/122 (test_async_llm concurrency flake passes
  on rerun); tools unittest 164/164; `test_gdn_prefill_conv` 19/19.
- Bit-exact CUDA: `test_ops_gdn` 53 cases / 3081 assertions; the two new byte-exact
  cases 268 assertions GPU; `compute-sanitizer memcheck` 0 errors.
- Token-exact (final defaults, reg ON / split OFF): 27B **235/235**, 35B **315/315**.
- nsys `--cuda-graph-trace=node` kernel A/B (evidence `dgx:~/work/prefill-attr-conv-35b`):
  - Conv `CausalConv1dFwdRegKernel` vs `CausalConv1dFwdTiledKernel`:
    c1 337,112 → **321,148 ns/call (−4.7%)**; c6 1,036,438 → **960,313 ns/call (−7.3%)**.
    Consistently faster ⇒ DEFAULT ON.
  - Post-conv split vs megablock: c1 83,128 → 79,947 (−3.8%); c6 384,066 → 401,971
    (**+4.7%**). Near-neutral / concurrency-dependent ⇒ OPT-IN (GdnPostConv time is
    dominated by the already-grid-parallel per-head q/k L2-norm, not the V-megablock).
  - TTFT c1 A/B (3 reps): reg-ON mean 186.23 ms vs reg-OFF 186.96 ms (**−0.39%,
    within run-noise** — the conv kernel is ~2.5% of GPU time, so a 5-7% kernel win is
    below TTFT noise; the kernel-level nsys A/B is the binding evidence).

## Finding

The prefill conv + post-conv are BANDWIDTH-bound (conv ~215 GB/s of GB10's ~273 GB/s
peak, f32 in/out). The register-window structural mirror closes the launch/compute
overhead (redundant weight loads, `__syncthreads`) for a real but modest 5-7%
kernel-time win; the larger conv gap vs vLLM (if any) is TRAFFIC — the bf16
post-conv-activation path (VT_GDN_IN_BF16, task #40), which halves the bytes moved —
not kernel structure. The post-conv split does not attack its dominant cost (q/k
L2-norm) and ships opt-in.
