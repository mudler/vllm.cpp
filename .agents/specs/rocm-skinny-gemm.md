# ROCm wvSplitK skinny GEMM — review rework (PR #506)

The original #506 ported upstream's `wvSplitK_hf_sml_` kernel body faithfully
(the reviewer verified the anchors and the arithmetic) but dropped the dispatch
preconditions around it. This spec records the rework; the kernel math is
unchanged.

## The three guards, restored (all verified against the pin `555967922`)

1. **Feature-dim guards** (upstream `utils.py:181` `m > 8 and 0 < n <= 5` with
   `m = weight.shape[0]`, plus `skinny_gemms.cu:1217` `M_in % _YTILE == 0`):
   with the donor's naming mapped onto ours (`out[M_tokens, N_features]`), the
   kernel's YTILE=2 stores write `C[m + y + n*M_features]` unguarded for `y<2` —
   on odd N the last wave writes `C[N]` (two bytes past the buffer), and at
   N==1 the first wave already writes out of bounds. Our dispatch now requires
   `N > 8 && (N % 2) == 0`, everything else falls through to the BLAS path.
2. **Arch guard**: the port carries only the wave32 reduction arm
   (`__shfl_xor(x,16)`); upstream branches to `ROW_BCAST15/31` on gfx9
   (wave64). The gfx9 arm is NOT ported, so dispatch now refuses non-wave32
   architectures via `CapabilityFromGcnArch(DeviceArchName())` (gfx11xx/gfx12xx
   only), rather than compiling and silently producing wrong sums on gfx9.
3. The `N > 8` lower bound (upstream `m > 8`) — folded into (1).

## The test, ported for real this time

`tests/kernels/quantization/test_rocm_skinny_gemms.py::test_rocm_wvsplitk_kernel`
@ pin — preserved: the applicable NKM factor list (tokens 1–4 = our template
arms), the xavier on/off scaling, and the **elementwise** tolerance
(`atol = eps_bf16 * sqrt(K)`, `rtol = 1e-2`; torch assert_close semantics) in
place of the aggregate NMSE. Added guard-boundary cases the upstream suite
implies: features ≤ 8 and odd features must route to BLAS and stay correct,
odd K declines, and a K%512 ≠ 0 shape exercises the K-tail. Every case runs
into a **sentinel-padded output buffer** (0xDEAD guard band) so any residual
out-of-bounds store fails the test outright. Deferred with reason recorded:
fp16 (port is bf16-only), bias (the `vt::MatmulBT` seam has no bias operand),
padded strides (our dispatch precondition is contiguous rows).

Mutation proof: with the `N % 2` guard removed, the odd-features case corrupts
the sentinel band and the case fails; with it restored, green.

## Boundaries

- Kernel body unchanged from the reviewed port.
- The gfx9 (wave64) arm remains owed — a future port of the ROW_BCAST
  reduction, gated the same way.
- `VT_ROCM_SKINNY=0` remains the A/B rollback; the allowlist carries it once,
  in main's re-sorted layout.

## Issue #1183 repair

[Issue #1183](https://github.com/mudler/vllm.cpp/issues/1183) found that the
architecture guard cached the first device's architecture for the process.

### Diagnosis

`SkinnyGemmArchOk(int device_index)` stored `DeviceArchName(device_index)` in a
function-static string. A gfx11 call initialized that string as eligible. A
later gfx9 or unknown device then reused the gfx11 result and could reach the
wave32-only kernel. The four devices on the repair host are gfx1100, so the
test uses a controlled resolver instead of claiming heterogeneous hardware.

### Decision

The production call and the test now use the same HIP-free predicate. A
per-thread vector keys each result by resolver and device index. The first call
for a key resolves and parses the architecture. Later calls read one boolean
without a HIP query or a process-wide mutex. The resolver key prevents the test
resolver from contaminating a production result in the same process.

The guard continues to accept only gfx11 and gfx12. It refuses gfx9 and every
unknown architecture. The shape, dtype, rollback, fallback, and GetBlas rules
remain unchanged.

### Rejected alternatives

- A single first-device value repeats the defect and is unsafe after a device
  hop.
- An uncached `DeviceArchName` call adds a HIP property query to every skinny
  GEMM dispatch.
- A process-wide keyed map needs synchronization on the decode path.

### Evidence

The test-only refactor first preserved the faulty cache. This command compiled
the production predicate and its deterministic device-hop test:

```sh
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib flock /home/vikash/gpu.lock cmake --build build-hip --target test_rocm_arch -j2
```

The build exited 0. The next command exited 1 before the fix:

```sh
env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib flock /home/vikash/gpu.lock build-hip/tests/test_rocm_arch '--test-case=skinny GEMM architecture eligibility follows device hops'
```

The gfx9 and unknown checks received `true`. Resolver counts were `{1,0,0,0}`
instead of `{1,1,1,1}`. After the keyed cache change, the same test passed 1 of
1 cases and 6 of 6 assertions with exit 0.

The mutation replaced the keyed cache call with one function-static resolved
architecture. The build exited 0, and the same focused test exited 1 with the
same three assertion failures. After restoration, SHA-256 values for the
header, production caller, and test matched their pre-mutation values. A fresh
rebuild and focused run then passed 1 of 1 cases and 6 of 6 assertions.

### Outcome

Architecture eligibility follows the requested device on every device-hop
sequence. A repeated key does not query the resolver again. The gfx9 wave64 arm
remains refused and owed as recorded in `## Boundaries`.
