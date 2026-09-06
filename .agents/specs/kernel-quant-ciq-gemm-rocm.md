# Spec: KERNEL-QUANT-CIQ-GEMM-ROCM

- Issue: [#2109](https://github.com/mudler/vllm.cpp/issues/2109)
- Row: `KERNEL-QUANT-CIQ-GEMM-ROCM` — the performance half of the ROCm
  keep-quant GEMM (mirror of `KERNEL-QUANT-CIQ-GEMM-CUDA`, which closed the
  same gap for NVFP4). Owning feature row: `BACKEND-ROCM` (#41).
- Base: `bcade48d6f7e6666f88ffaaf3d2b78af24c35d7d` (`upstream/main`,
  2026-09-05).
- Pull request shape: separate spec and implementation pull requests
  (developer decision 2026-09-05, recorded in
  `.agents/developer-preferences.md`). This pull request lands the spec only.

## Scope

Give the ROCm keep-quant GEMM prefill arm (`KQuantGemmK` / `Q8_0GemmK` /
`GroupedKQ8K` in `src/vt/rocm/rocm_grouped_gemm.hip:323-403,446-496`) a
tensor-core inner loop on **RDNA4 only: gfx1200 and gfx1201**. The scalar
`__dp4a` warp-reduction path stays the fallback for every other ROCm target.

Keep the existing block-dequant math (`DotQ4K`/`DotQ6K`/`DotQ8_0`, lines
179-290) and the `EnsureQuantScratch` stream-ordered pool (lines 421-441)
unchanged. Out of scope, and not attempted:

- **gfx1100 (RDNA3).** llama.cpp dispatches RDNA3 through a separate WMMA tile
  branch with different, non-`_gfx12` builtins (`mma.cuh:729`). `VikashLoomba`
  audited current `main` for gfx1100 on 2026-09-05 (issue #2109 comment) and
  proposed a fresh RDNA3 spec. This row does not claim that hardware or that
  work.
- **gfx1151 (RDNA3.5, Strix Halo).** Takes the RDNA3 branch, not the RDNA4
  one, per `localai-org-maint-bot`'s 2026-09-02 correction on #2109. This
  session has no Strix Halo hardware, so no W0 probe can be run against it
  here. Owed to whoever holds that box.
- **The decode/GEMV (n=1) path.** Already covered by PR #2086
  (`ROCM-KQUANT-NWARPS-DECODE`, merged), which explicitly leaves the
  WMMA/row-packing/MoE work `## Owed`. This row is the GEMM/n>1 (prefill) arm
  only.
- **MFMA and hipBLASLt.** Neither is reachable on this hardware; see below.
- **Q4_0/Q2_K/Q3_K/IQ2_\*/IQ3_\*/MXFP4 keep-quant formats.** Unrelated gap
  (#1940), untouched by this row.
- **`GFX1100-TG200`'s campaign** (external fork `ghazni101/vllm.cpp`, its own
  GPU lock). That campaign targets gfx1100 and already owns the correctness
  half of this kernel (`kROCM` provider registration, landed via its F1
  stage, PR #2782). This row's gfx1200/gfx1201 tensor-core work does not
  touch the same launch sites that campaign is mid-flight on, and does not
  claim any of its scope.

## Mechanism verified on target (W0)

**Reconstruction note.** `joral` first ran this probe and posted the MFMA
correction on issue #2109 on 2026-08-27. Both the probe source and this
spec's earlier draft were local-only and lost to a system crash/restore
before either landed on `main`. This session (2026-09-05) rebuilt the probe
from the description in that comment and reran it on the same hardware
before writing this spec, so W0 stands on fresh evidence rather than a
memory of the original run.

`__builtin_amdgcn_mfma_i32_16x16x32_i8` needs the `mai-insts` target feature,
which exists only on CDNA (for example gfx90a). Compiling for
`--offload-arch=gfx1200` with this project's toolchain (ROCm 7.2, HIP
7.2.53211, clang 22) fails:

```text
error: '__builtin_amdgcn_mfma_i32_16x16x32_i8' needs target feature mai-insts
```

The same source compiles for `--offload-arch=gfx90a`. Neither gfx1200 nor
gfx1201 carries `mai-insts`, so no MFMA instruction is reachable on this
row's target hardware.

The RDNA4 path is rocwmma/WMMA. Reran on the actual RX 9060 XT (gfx1200)
with `rocwmma` (already available in this project's ROCm toolchain):

```text
hip=no error mismatches=0/256 first=16 (expected 16)
W0_PROBE_OK (reconstructed)
```

`fragment<matrix_a,16,16,16,int8_t,row_major>` times
`fragment<matrix_b,16,16,16,int8_t,col_major>` into
`fragment<accumulator,16,16,16,int32_t>` produces the exact integer dot
product on this hardware, through the toolchain this project actually builds
with. This run used an independently checkable all-ones fixture (expected
value 16 for every output cell) rather than the original run's fixture
(reported `first=136`); the mechanism proven is the same.

An in-tree precedent already exists for the rocwmma include and namespace
pattern: `src/vt/rocm/rocm_paged_attn.hip` already uses
`fragment<matrix_a, WM, WN, WK, bfloat16_t, row_major>` for attention. The
new int8 kernel follows that pattern rather than introducing a second one.

## Upstream anchor

llama.cpp, pin `b10451` per `.agents/upstream-sync.md`. Line numbers below
are cross-checked against a local checkout at tag `b10688`; the conditional
compilation this cites has not changed across that span.

- `ggml/src/ggml-cuda/common.cuh:265` — `AMD_MFMA_AVAILABLE` gates on
  `defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)`.
- `ggml/src/ggml-cuda/common.cuh:76` — "RDNA removes MFMA, dp4a, xnack, acc
  registers, wave size is 32."
- `ggml/src/ggml-cuda/vendors/hip.h:189-208` — `CDNA` is defined only for
  `__gfx908__`, `__gfx90a__`, `__gfx942__`, `__gfx950__`.
- `ggml/src/ggml-cuda/common.cuh:269` — `AMD_WMMA_AVAILABLE` gates on
  `defined(RDNA4) || defined(RDNA3)`.
- `ggml/src/ggml-cuda/vendors/hip.h:211-221` — `RDNA4` comes from `__GFX12__`
  (gfx1200, gfx1201).
- `ggml/src/ggml-cuda/mma.cuh:697` — the RDNA4 tile branch:
  `AMD_MFMA_AVAILABLE || (AMD_WMMA_AVAILABLE && RDNA4)`, using the
  `_gfx12`-suffixed builtins (for example
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`).
- `ggml/src/ggml-cuda/mmq.cuh:181,197,472` — the quantized matmul kernel
  selects on the same `AMD_MFMA_AVAILABLE`/`AMD_WMMA_AVAILABLE` pair.

## Design

Replace the scalar `Dp4a()` warp-reduction inner loop in `KQuantGemmK` (and
its `Q8_0GemmK`/`GroupedKQ8K` siblings) with a WMMA int8 tile, gated to
compile only under `--offload-arch=gfx1200`/`gfx1201` (mirroring how
`rocm_paged_attn.hip` already gates its own WMMA arms). Every Q8_K-family
superblock is 256 elements; a 16-wide WMMA K-tile divides it into exactly 16
tiles with no ragged remainder, accumulated in one `fragment<accumulator,
16,16,16,int32_t>` before the per-superblock scale is applied. `M`/`N` tail
handling for a row count not a multiple of 16 is unresolved and owed to the
implementation wave, not this spec.

Per-superblock Q8_K scale dequant is fused in the epilogue after
accumulation, following the same shape as the existing scalar path
(`DotQ4K`/`DotQ6K`/`DotQ8_0`) rather than inside the tensor-core tile itself,
because the risk below is that no library path carries the per-superblock
scale layout through the tile op.

## Risks

- **hipBLASLt may not support the per-superblock Q8_K block-scale layout**
  (16 independent scales per 256-element superblock). This is the design's
  assumption already, not a fallback to check later: write the tile directly
  against `rocwmma` with explicit scale dequant in the epilogue, matching
  llama.cpp's own approach in `mma.cuh`.
- **Reduction order changes relative to the scalar `dp4a` path** could move
  low bits. Record near-tie adjudication per the ratified band doctrine
  (`.agents/specs/rocm-m4-oracle.md`) if bit-exactness cannot be shown
  directly, rather than asserting identity the change cannot prove.
- **Row-count tail handling** (`M`/`N` not a multiple of 16) is unresolved in
  this spec and must be designed in the implementation wave before any
  launch-site change lands.

## Tests

- `test_rocm_quant_dot.cpp` (mirrors `test_cuda_quant_dot`), NMSE <= 5e-4,
  unchanged as the correctness gate for every quant-path lever on this
  kernel (issue #2109 gate (b)).
- `test_backend_cross_device`, NMSE <= 5e-4 vs the CPU oracle (gate (a)).
- `rocprofv3 --kernel-trace` on the Ornith-1.5-9B-Q4_K_M trace workload
  (gate (c)): target total kernel time <= 20,000 ms (oracle 18,812 ms;
  current scalar arm 108,817 ms).
- `ctest -R 'rocm|cross_device'`, zero regression (gate (d)).

## Owed

- gfx1100 (RDNA3) WMMA tile: `VikashLoomba`'s audit on #2109, a separate row
  and spec.
- gfx1151 (RDNA3.5) WMMA tile: needs Strix Halo hardware to verify; a
  separate row and spec.
- `M`/`N` tail handling for a non-16-multiple row count: unresolved here,
  owed to the implementation wave.
- hipBLASLt per-superblock scale support: unmeasured; recorded as an open
  question for whoever implements W1, not assumed either way beyond the
  Risks section above.

## Stop conditions

- `NEEDS_DECISION`: any request to extend this row's scope to gfx1100 or
  gfx1151, or to touch `GFX1100-TG200`'s launch sites.
- 20 failed attempts within the implementation wave: stop, report the
  measured position and the next traceable hypothesis.

## Now

`SPIKE`. W0 (mechanism verified on target, gfx1200) is done and reproduced
fresh in this spec. W1 (the WMMA tile kernel with the fused scale epilogue)
and W2 (launch-site replacement) are not started. This pull request lands
the spec only; no product code changes in this change.
