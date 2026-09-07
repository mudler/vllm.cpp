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

## Upstream chain

llama.cpp, pin `b10451` per `.agents/upstream-sync.md`. Line numbers below
are cross-checked against a local checkout at tag `b10688`; the conditional
compilation this cites has not changed across that span. This is the
COMPLETE chain from source to the executed instruction: the arch-detection
macros, the tile-selection branch they feed, and the MMQ call site that
reaches it — every link a change here has to reconcile against, not only the
kernel this row ports.

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

## Our baseline

`KQuantGemmK<OutT, Fmt>` (`src/vt/rocm/rocm_grouped_gemm.hip:446-482` at this
row's base SHA) — a scalar warp-per-output-element GEMM: one warp owns one
`(i,j)` output cell, strides its 32 lanes over the row's superblocks, and
reduces each superblock's dot product with the `__ockl_sdot4`-backed `Dp4a()`
helper (four signed `int8` multiply-adds per instruction, not a tensor-core
op). Measured baseline (issue #2109, same-tool `rocprofv3` attribution,
`Ornith-1.5-9B-Q4_K_M`, `-n 128 -ngl 99`, this same RX 9060 XT): prefill
(pp512) 73.7 tok/s against the oracle's 1691 (22.9x slower); decode (tg128)
17.44 against 48.51 (2.78x slower); total GPU time 108,817 ms against the
oracle's 18,812 ms. This row targets the prefill (`m>1`) arm of that gap;
the decode (`m==1`) arm already has its own row
(`ROCM-KQUANT-NWARPS-DECODE`, PR #2086, merged) and is untouched here. Own
measured op-level baseline (`examples/quant-gemm-bench`, RX 9060 XT): Q6_K
115-122 GFLOP/s, Q4_K 302-448 GFLOP/s — see `## Now` for the WMMA arm's
numbers against each.

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

## Design: cooperative activation share (issue #3034)

A follow-on wave, after the wider-block A/B (#3032/#3033, measured and
rejected, geomean -4.3%) and reading llama.cpp's actual `mul_mat_q` source
(confirmed, not inferred: `ggml-cuda/mmq.cuh`, `mmq-config-rdna4.cuh`,
`mmq-load-tiles.cuh`) to find out why widening alone did not help.

**The precise redundancy, traced through this kernel's own index math.**
`tile = blockIdx.x * WarpsPerBlock + threadIdx.y`, `it = tile / n_tiles`,
`jt = tile % n_tiles`. Because `n_tiles` (weight-row tile count, e.g. 192 at
N=3072) is always far larger than `WarpsPerBlock`, every warp in one block
shares the SAME `it` (activation rows) and gets a DIFFERENT, consecutive
`jt` (weight rows). So within one block, every warp already reads the
identical 16 activation rows — and does so independently, via its own
`load_matrix_sync` call straight to global memory, every superblock. The
rejected wide-block experiment widened this same redundancy (more warps
sharing one `it`) without adding any sharing to fix it, which is exactly
why it did not help: it made the redundant pattern wider, not cheaper.

**The design**, scoped deliberately smaller than matching llama.cpp's full
128x128 cooperative tile (that would also share the WEIGHT tile across a
wider `jt` range per warp, which our current per-warp weight assignment
does not need — weight rows are already disjoint across warps in one
block, so there is no analogous weight-side redundancy to remove):

1. Reuse `kQuantWmmaWideWarpsPerBlock == 8` (already in the tree from
   #3033, currently unused/rejected on its own). At 8 warps, one block
   already spans a 128 weight-row range, matching llama.cpp's `I=128` on
   that one axis, for free.
2. Add cooperative activation staging: once per superblock, before any
   warp's WMMA calls for that superblock, ALL 8 warps' threads together
   copy the shared `it`'s `BlockQ8_K` bytes (`qs`, `bsums`, `d`) for this
   one superblock into ONE shared buffer, `__syncthreads()`, then every
   warp's `load_matrix_sync` (and the Q4_K min-correction's `bsums` read)
   reads from that shared copy instead of calling global memory
   independently 8 times.
3. LDS budget check: the existing 8-warp arm already uses ~50 KiB
   (`w_stage` 32 KiB + `row_scales`/`row_mins` 2 KiB + `raw_tile` 16 KiB
   for Q4_K, the larger of the two formats). One superblock's worth of
   shared `BlockQ8_K` bytes for 16 rows is `qs` (256 B) + `bsums` (32 B) +
   `d` (4 B) per row x 16 rows ~= 4.7 KiB. Total ~55 KiB, under the 64 KiB
   limit — no redesign of the existing weight-side staging is needed to
   fit this.

**What this measurement isolates.** If this wins, sharing was the lever
the wider-block-alone experiment could not reach, and it is worth checking
whether widening `J` (the activation range one warp loops over, still 16
today) compounds it further. If it does not win, the redundant reads were
likely already served by cache rather than costing real bandwidth, and the
remaining gap points somewhere else — raw WMMA instruction throughput or
shared-memory bank-conflict patterns, not data reuse. Either result is
recorded, not assumed.

## Port map

| Upstream | Local |
|---|---|
| `mma.cuh:697` RDNA4 tile branch (`_gfx12` WMMA builtins) | `KQuantGemmKWmmaQ6K`/`KQuantGemmKWmmaQ4K` (`rocm_grouped_gemm.hip`), via `rocwmma`'s `fragment`/`mma_sync` rather than the raw builtins — this project's own in-tree precedent (`rocm_paged_attn.hip`), not a second pattern |
| `common.cuh:265,269`, `vendors/hip.h:189-221` arch macros | `Gfx12QuantWmmaHostOk` (`rocm_grouped_gemm.hip`) at the host call site, reusing `vt::rocm::GcnArchNameIsGfx12PrefillWmma` (`include/vt/rocm/rocm_arch.h`) rather than the compile-time macro (issue #785's trap for a host-side WMMA gate) |
| `DotQ4K`/`DotQ6K` block-dequant math (already ported 1:1, unchanged by this row) | `src/vt/rocm/rocm_grouped_gemm.hip:230-321` |
| Q6_K's 16-wide scale group | `DequantQ6KGroup16` |
| Q4_K's 32-wide scale/min sub-block | `DequantQ4KTile16` + `UnpackQ4KScalesMins` |

## Dependencies

- `rocwmma` (already vendored in this project's ROCm toolchain; no new
  external dependency).
- gfx1200/gfx1201 hardware to compile the device-pass WMMA branch and to
  runtime-gate the host dispatch; every other ROCm target is unaffected and
  untested by this row.
- ROCm 7.2 / HIP 7.2.53211 / clang 22, the toolchain W0 was proven against;
  not verified on another ROCm release.
- The in-tree WMMA precedent `rocm_paged_attn.hip` (include/namespace
  pattern, and the `#785` host-vs-device-macro fix this row's host gate
  reuses directly rather than re-deriving).

## Work breakdown

- **W0** (done): mechanism verified on target — rocwmma int8 tile produces
  the exact integer dot product on gfx1200.
- **W1** (done): `KQuantGemmKWmmaQ6K`, hardware-verified, mutation-proven.
- **W1.1/W1.2** (done): two performance passes on the same kernel (fill-once
  staging, then batched store/fold), 11.7x-16.3x over scalar.
- **W1.3** (done): `KQuantGemmKWmmaQ4K`, hardware-verified, mutation-proven
  on both its correction terms.
- **W1.4** (done): the issue's own gate (c) recipe run against the pinned
  oracle; narrowly satisfied against its stated threshold, oracle-figure
  reconciliation left open.
- **W1.5** (done): `M`/`N` tail handling for a non-16-multiple row count —
  see `## Now`. Closes the gap where a real (essentially never 16-aligned)
  prompt fell back to scalar for the whole call, not only its ragged edge.
- **W2 does not exist as originally framed.** The pre-W1.5 spec named a W2
  ("launch-site replacement", the scalar arm's eventual retirement once
  every format is covered). W1.5's tail-fill design makes the scalar kernel
  a PERMANENT dependency (the remainder cells, and every `m<16`/decode
  shape) rather than a stopgap awaiting format coverage, so that framing no
  longer applies — see `## Now`.

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
- **Row-count tail handling** (`M`/`N` not a multiple of 16) — RESOLVED in
  the implementation wave: see `## Now`. The host gate relaxed from exact
  alignment to `m >= 16 && n >= 16`, and the scalar `KQuantGemmK` grew a
  `skip_m`/`skip_n` guard so it fills in the WMMA corner's remainder without
  recomputing it.

## Tests to port

- `test_rocm_quant_dot.cpp` (mirrors `test_cuda_quant_dot`): named by the
  issue as gate (b)'s vehicle; not authored — the row's actual correctness
  coverage landed instead as two new `TEST_CASE`s in the existing
  `test_backend_cross_device.cpp` (one per format, f32 and bf16 out each,
  CPU-oracle NMSE and dispatch-count reachability), which already exercises
  this kernel end-to-end and did not need a second, format-mirrored file.
- `test_backend_cross_device.cpp` — the CPU-oracle NMSE cases above (gate
  (a)), plus the pre-existing non-grouped/grouped keep-quant suites this row
  does not modify.
- `examples/quant-gemm-bench` (extended this wave to run any registered
  device, not only CPU) — the op-level GFLOP/s A/B, not a `ctest` gate but
  the vehicle for every performance table in `## Now`.

## Gates

- (a) `test_backend_cross_device` NMSE <= 5e-4 vs the CPU oracle: MET (both
  formats, f32 and bf16 out).
- (b) Correctness coverage for every quant-path lever on this kernel: MET,
  via the cases named above rather than a separate mirrored file.
- (c) `rocprofv3 --kernel-trace` on the Ornith-1.5-9B-Q4_K_M trace workload,
  target total kernel time <= 20,000 ms (oracle 18,812 ms; scalar arm
  108,817 ms): narrowly SATISFIED against the stated absolute threshold on
  a faithful same-tool same-checkpoint same-pinned-oracle reproduction (see
  `## Gate (c), the issue's own recipe` below) — NOT reconciled against the
  issue's original absolute figures, which stay an open question.
- (d) `ctest -R 'rocm|cross_device'`, zero regression: MET (one pre-existing
  unrelated failure, #1513/#1954, present before and after this row).

## Owed

- gfx1100 (RDNA3) WMMA tile: `VikashLoomba`'s audit on #2109, a separate row
  and spec.
- gfx1151 (RDNA3.5) WMMA tile: needs Strix Halo hardware to verify; a
  separate row and spec.
- hipBLASLt per-superblock scale support: unmeasured; recorded as an open
  question for whoever implements W1, not assumed either way beyond the
  Risks section above.
- Q5_K WMMA tile: same 32-wide scale/min shape `KQuantGemmKWmmaQ4K` already
  carries, plus a high-bit plane (like Q6_K's `qh`) neither existing kernel
  handles. Deprioritized below Q4_K by real-model evidence: the checkpoint
  this row gates against (`Q4_K_M`) never invokes Q5_K at all — it is a
  `Q5_K_M`/`Q5_K_S`-only format — so porting it would not move this row's own
  gate (c) measurement, only a differently-quantized checkpoint's.
- Cross-warp data reuse in the WMMA kernel body (issue #3032). **CONFIRMED
  by reading the actual source**, not inferred: llama.cpp's Q4_K config at
  this row's shapes (`ggml/src/ggml-cuda/mmq-config-rdna4.cuh:127`,
  `GGML_TYPE_Q4_K, 256, ..., 128, 128, ...`) gives ONE BLOCK an I=128 (weight
  rows) x J=128 (activation columns) output tile — 8-64x the area our
  4-8-warp block covers, since each of OUR warps independently owns only a
  16x16 tile. That whole I x J tile is loaded ONCE per block, not once per
  warp: `ggml_cuda_mmq_load_tiles_q4_K`
  (`ggml/src/ggml-cuda/mmq-load-tiles.cuh:703-741`) stripes the I=128 rows
  across every warp's threads (`i0 += nrows*nwarps`, each warp taking a
  DIFFERENT row range), dequantizing each row exactly once into one shared
  `x_tile`; `mmq_get_nbytes_shared` (`mmq.cuh:1379-1383`) sizes the
  activation-tile allocation by `J` alone, not `J*nwarps`, confirming one
  shared activation copy too. The compute phase
  (`ggml_cuda_mmq_write_back_mma`, `mmq.cuh:476-500`) then splits the I
  dimension across warps (`rows_per_warp = I/nwarps`), each warp looping the
  FULL J range doing 16x16 MMA ops against that one shared load. This
  kernel's per-warp-independent design (own `w_stage`/`raw_tile` slice, own
  `load_matrix_sync` call, zero sharing even when two warps in one block
  share the same M-tile) is the actual mechanism gap the wider-block
  experiment above could not touch, because widening never introduced
  sharing — it only added more independent warps to the same per-warp-load
  pattern. Matching this (cooperative tile load + row-split compute over a
  much larger shared tile) is a real kernel redesign, materially bigger than
  the block-width experiment, and is the next traceable step — not attempted
  in this row.

  **UPDATE: partially done.** `KQuantGemmKWmmaQ6KBigTile`/`Q4KBigTile`
  (below) implement exactly this at `ItGroup=3` (48 rows) instead of
  llama.cpp's 128, measured a real +16.8% geomean / -14% real-model win,
  and are correctness-verified. The scope was deliberately narrower than
  full parity: this row's kernels still stage one full 256-wide
  superblock at a time for BOTH operands, where llama.cpp stages 32
  (weight) and 128 (activation) K-elements at a time (confirmed by
  reading `ggml_cuda_mmq_get_nbytes_shared_x`, `mmq.cuh:415-419`, and
  `ggml_cuda_mmq_get_sram_stride`, `mmq.cuh:132-153` — not a smaller
  encoding, nearly identical bytes/element, just a narrower chunk staged
  more often). Finer K-chunking on our side — the next traceable step —
  would shrink both operands' per-load footprint and could afford a much
  wider `ItGroup` (closer to llama.cpp's 8) within the same 64 KiB
  budget, at the cost of more frequent, smaller syncs; this is a bigger
  change than BigTile was (it touches the weight-side staging every wave
  of this row has left untouched) and is not attempted here.

## Stop conditions

- `NEEDS_DECISION`: any request to extend this row's scope to gfx1100 or
  gfx1151, or to touch `GFX1100-TG200`'s launch sites.
- 20 failed attempts within the implementation wave: stop, report the
  measured position and the next traceable hypothesis.

## Now

`ACTIVE` (moved from `SPIKE`: real, hardware-verified kernels now sit on the
production `MatmulBTQuant` dispatch path, not only a mechanism probe).
W0 (mechanism verified on target, gfx1200) is done. W1 has landed on
`row/KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4-w1`: `KQuantGemmKWmmaQ6K`, a rocWMMA
int8 tile arm for the Q6_K prefill GEMM, gated to gfx1200/gfx1201 and to
tile-aligned M/N (both multiples of 16); every other shape and architecture
keeps the scalar arm. Hardware-verified on the RX 9060 XT (gfx1200):
correct against the CPU oracle (f32 and bf16 out), reachability-witnessed,
mutation-proven RED-before-GREEN, zero regression on `ctest -R
'rocm|cross_device'` (one pre-existing unrelated failure, #1513/#1954, not
touched by this row).

**Op-level A/B, same tool both arms (`examples/quant-gemm-bench`, extended in
this wave to run on any registered device, not only CPU), RX 9060 XT
(gfx1200), Q6_K prefill (M=128, tile-aligned N/K), best-of-6, idle host:**

| Shape | scalar (`VT_ROCM_QUANT_WMMA=0`) | WMMA v1 (per-group shared-mem bounce) | WMMA v2 (fill once per superblock) |
|---|---|---|---|
| N=3072 K=2048 | 120.41 GFLOP/s | 630.42 GFLOP/s (5.24x) | 1439.55 GFLOP/s (11.95x) |
| N=12288 K=2048 | 122.48 GFLOP/s | 558.44 GFLOP/s (4.56x) | 1514.70 GFLOP/s (12.37x) |
| N=2048 K=6144 | 115.38 GFLOP/s | 833.77 GFLOP/s (7.23x) | 1467.55 GFLOP/s (12.72x) |

v1 staged one scale-group's dequantized tile per iteration (16 barriers/
superblock for that step alone, only 16 of 32 lanes active while filling). v2
fills all 16 groups' tiles once per superblock, spread over all 32 lanes, and
removes those 16 barriers down to 1. That single change bought a further
~2.3-2.7x on top of v1's already-measured 4.5-7.2x, and the WMMA arm now
EXCEEDS this card's own Q8_0 scalar dp4a ceiling (1258-1339 GFLOP/s) — a
result v1 did not reach.

**Batching the remaining per-group store-sync-fold-sync round trip, same
tool and shapes, RX 9060 XT, best-of-6:**

| Batch width | N=3072 K=2048 | N=12288 K=2048 | N=2048 K=6144 |
|---|---|---|---|
| 1 (v2, above) | 1439.55 GFLOP/s | 1514.70 GFLOP/s | 1467.55 GFLOP/s |
| 2 | 1709.58 GFLOP/s (+18.8%) | 1995.02 GFLOP/s (+31.7%) | 1689.45 GFLOP/s (+15.1%) |
| 4 | 1335.06-1347.51 GFLOP/s (-6-7%) | 1408.20-1409.09 GFLOP/s (-7%) | 1419.29-1424.70 GFLOP/s (-3%) |
| 8 | 768.51 GFLOP/s (-47%) | 888.49 GFLOP/s (-41%) | 791.03 GFLOP/s (-46%) |

Batching 2 groups' WMMA compute+store per barrier (instead of 1) is a real
further win on top of v2; batching 4 or 8 REGRESSES relative to 2, and 8 is
worse than not batching at all. Widening `raw_tile` to hold more groups grows
shared-memory occupancy pressure faster than it saves barriers — the
regression at 4 and 8 is evidence for that shape, not measured to a specific
cause. **v3 (batch=2) is the row's current state**, RED-before-GREEN mutation-
proven and `ctest`-clean identically to v1/v2 above; batch=4 and batch=8 were
measured, rejected, and are not carried forward — this table is why, so the
same two shapes are not retried without new evidence.

Total measured gain over the original scalar arm on this card: **11.7x to
16.3x** (1689-1995 GFLOP/s vs 115-122 GFLOP/s), depending on shape.

**First real-model measurement, `rocprofv3 --kernel-trace --stats`, the
actual `Ornith-1.5-9B-Q4_K_M.gguf` checkpoint, RX 9060 XT, `vllm-cli --device
auto --max-tokens 1`, a 288-token prompt (16-aligned, so every layer's Q6_K
prefill call takes the WMMA arm), same prompt both legs, `VT_ROCM_QUANT_WMMA`
the only variable:**

| | scalar (`=0`) | WMMA (v3, batch=2) | ratio |
|---|---:|---:|---:|
| `KQuantGemmK<uint16_t, 2>` / `KQuantGemmKWmmaQ6K` (20 calls, this row's kernel) | 4322.5 ms | 201.3 ms | **21.5x** |
| Total kernel time, whole forward pass | 11973.9 ms | 7795.1 ms | **1.54x** |

This is the row's own kernel, isolated by same-prompt/same-tool/only-the-
env-toggle-differs A/B, at the exact production call site (`MatmulBTQuant`
reached through the model's real forward pass, not a synthetic harness). It
is NOT yet the issue's own gate (c) recipe: that recipe is `-n 128 -ngl 99`
against a longer, differently-shaped prompt (llama.cpp's own pp512/tg128
convention) whose absolute totals (108,817 ms scalar / 18,812 ms oracle) are
a different workload and are not directly comparable to the 11973.9/7795.1 ms
above — reporting the two side by side would compare different quantities.
The 1.54x total-kernel-time reduction is strong, additional, real-model
evidence toward gate (c); replicating the issue's own exact recipe (its
prompt, its `-n 128 -ngl 99`, and the oracle side) is the remaining step to
close it and is not done in this paragraph.

**Q4_K WMMA tile (`KQuantGemmKWmmaQ4K`) landed in the same wave**, at the
developer's direction after the real-model trace above showed Q4_K carrying
59.2% of this checkpoint's total kernel time (7090.3 ms of 11973.9 ms
scalar-arm total across 108 calls) against Q6_K's 36.1% (20 calls) — bigger
in absolute terms than the kernel this row started with, so closing it
mattered more than tuning Q6_K further. Q4_K's layout differs from Q6_K's in
two ways this kernel carries: its nibbles are UNSIGNED (no -32 recentering —
the scalar `DotQ4K` has none either), and its scale granularity is 32-wide
(twice the 16-wide WMMA tile) with a SECOND per-sub-block correction
(`dmin * sumi`, using the activation's precomputed `bsums` against the
weight's mins) that needs no tensor-core work at all and is folded into the
same one-f32-expression-per-superblock epilogue as a separate integer
accumulator. `kGroupGemmBatch == 2` batches exactly one Q4_K sub-block (two
16-wide K-tiles) per barrier, by construction rather than by re-tuning.

Same hardware-verification shape as Q6_K: correct against the CPU oracle
(f32 and bf16 out), and BOTH correction terms mutation-proven separately
(breaking the scale term reds NMSE 0.639; breaking the min term reds NMSE
0.0027 — smaller because the min term is the smaller correction, but still
~5.5x over the 5e-4 bound) — reverting each restores an identical source
hash. `ctest -R 'rocm|cross_device'` carries the same single pre-existing
failure as every commit in this row.

**Op-level A/B (`examples/quant-gemm-bench`), Q4_K prefill:**

| Shape | scalar | WMMA | ratio |
|---|---:|---:|---:|
| N=3072 K=2048 | 390.5 GFLOP/s | 1134.4 GFLOP/s | 2.91x |
| N=12288 K=2048 | 447.5 GFLOP/s | 1331.1 GFLOP/s | 2.97x |
| N=2048 K=6144 | 302.7 GFLOP/s | 1094.9 GFLOP/s | 3.62x |

Smaller than Q6_K's ratio: Q4_K's scalar arm was already lighter per element
(a plain nibble mask, versus Q6_K's heavier ql+qh reconstruction) and the
WMMA arm carries extra per-cell work (the min-correction loop, the scale/min
unpack) that Q6_K's does not. Real, still substantial in absolute terms given
Q4_K's share of total kernel time.

**Second real-model measurement, same recipe as above, same 288-token
prompt, same binary, both kernels active:**

| | scalar (`=0`) | WMMA (both kernels) | ratio |
|---|---:|---:|---:|
| `KQuantGemmKWmmaQ4K<float>` (64 calls) | 4700.3 ms | 1199.5 ms | 3.92x |
| `KQuantGemmKWmmaQ4K<uint16_t>` (44 calls) | 2380.2 ms | 475.9 ms | 5.00x |
| `KQuantGemmKWmmaQ6K<uint16_t>` (20 calls) | 4311.9 ms | 203.5 ms | 21.2x |
| Total kernel time, whole forward pass | 11954.5 ms | 2447.1 ms | **4.89x** |
| Wall clock (`vllm-cli`, includes model load) | 12.358 s | 2.844 s | 4.35x |

Same caveat as the Q6_K-only measurement: this is not the issue's own `-n 128
-ngl 99` recipe, so the absolute totals are not directly comparable to its
108,817 ms / 18,812 ms figures. It is the strongest real-model evidence this
row has produced toward gate (c) — closing gate (c) itself still needs the
issue's exact recipe run, oracle side included.

## Gate (c), the issue's own recipe

**Built and ran the pinned oracle at the exact SHA the spec's `## Upstream
chain` cites** (`10bf611e533d81f739128304991c5e133c6aebd8`, tag `b10451`),
in a linked worktree (`/tmp/llama-cpp-b10451`) so the developer's own
checkout was never touched, `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1200`, on the
same RX 9060 XT. `llama-bench -m Ornith-1.5-9B-Q4_K_M.gguf -p 512 -n 128
-ngl 99` reproduces the issue's own numbers closely: pp512 1773-1862 tok/s
(issue: 1691), tg128 48.65-49.05 tok/s (issue: 48.51) — same tool, same
checkpoint, same recipe, and the oracle stands.

**`rocprofv3 --kernel-trace --stats` total kernel time, `-r 5` (llama-bench's
own default rep count), idle host verified (`rocm-smi`: 44°C, 0% GPU, no
other `/dev/kfd` holder; `uptime` load average 2.66-3.14, which is this
project's own recorded honest-caveat shape for "idle GPU, busy host" rather
than a fully quiet machine):**

| | Oracle (llama.cpp `b10451`) | Ours, scalar (`VT_ROCM_QUANT_WMMA=0`) | Ours, WMMA (this row) |
|---|---:|---:|---:|
| Total (`-r 5`) | 13,581.8 ms | 179,037.6 ms | 93,576.5 ms |
| Per-run average | 2,716.4 ms | 35,807.5 ms | 18,715.3 ms |
| Ratio vs oracle | 1.00x | 13.18x slower | 6.89x slower |

Single-run (`-r 1`, closer in shape to how the issue's own table reads):
oracle 2,972.3 ms, scalar 36,352.0 ms, WMMA 19,128.8 ms. **This row's WMMA
total is under the gate's stated 20,000 ms threshold on both single-run and
per-run-average readings, on the real checkpoint, the pinned oracle
revision, and the production call site** — the strongest form of evidence
this spec has for gate (c). Scalar-to-WMMA improvement on this full `-p 512
-n 128` workload (prefill AND decode combined, so Q4_K/Q6_K's isolated
kernel gains are diluted by every OTHER kernel in the forward pass): 1.90x
(`-r 1`) to 1.91x (`-r 5`), consistent between the two.

**What does NOT reconcile, stated rather than elided:** the oracle side
measured here (2,716-2,972 ms) is 6.3-6.9x SMALLER than the issue's cited
18,812 ms, and the scalar side (35,807-36,352 ms) is 3.0-3.1x smaller than
its cited 108,817 ms — different ratios, so this is not simply "the issue
used more repetitions" (that would scale both sides by the same factor).
Candidates not distinguished here: a different exact prompt (the issue does
not record one; this measurement's is a real 512-token English paragraph,
built and pinned to exactly 512 tokens against this checkpoint's own
tokenizer, `/tmp/prompt512.txt`, not preserved in this commit), ROCm/driver
drift on this box between 2026-08-27 and this measurement, or a recipe
detail the issue's one-line citation does not carry. Closing this gap is
follow-up work, not assumed away: **gate (c) is reported here as narrowly
SATISFIED against its own stated absolute threshold, on a faithful same-tool
same-checkpoint same-pinned-oracle reproduction — not as reconciled against
the issue's original absolute figures**, which stay an open question this
paragraph names rather than answers.

**M/N tail (W1.5) landed.** The host gate relaxed from exact 16-alignment to
`m >= 16 && n >= 16`; the WMMA kernels already floored `m`/`n` internally
and needed no change. `KQuantGemmK` grew a `skip_m`/`skip_n` guard (every
pre-existing call site passes `0, 0`, an always-false guard — unchanged
behavior there) so the tail-fill pass computes only the remainder the WMMA
corner left untouched, never recomputing it. Hardware-verified on a
deliberately doubly-misaligned shape (M=37, N=50, neither a multiple of
16) for both formats: correct against the CPU oracle, WMMA reachability
still holds for the aligned corner, and RED-first mutation-proven (`&&` to
`||` in the skip guard leaves real cells unwritten — NMSE explodes to
0.17/3.6e65 — and reverting restores an identical source hash). Real-model
confirmation: the ORIGINAL 291-token prompt (`prompt_tokens=291`, not a
multiple of 16, the exact shape that fell back to scalar entirely before
this change) now measures 2704.7 ms total kernel time — within 10.5% of
the perfectly-aligned 288-token result (2447.1 ms) and a 4.4x improvement
over the pre-fix scalar total for this same prompt. **W2 is not this row's
retirement of the scalar arm** — the tail-fill pass and the `m<16`/decode
paths need the scalar kernel permanently, by design, not as a stopgap
awaiting format coverage; that framing predates the tail fix and no longer
applies. `## Owed` above is current: Q5_K (a separate row) and the
gfx1100/gfx1151 WMMA tiles (separate rows) are what remains.

**Independent review repair (two LOW findings, both closed on
`row/KERNEL-QUANT-CIQ-GEMM-ROCM-RDNA4-w1`).** An independent review of this
row's implementation wave (PR #2991) found two non-blocking gaps in the
W1.5 tail-fill landing, both fixed in the same follow-up commit:

1. **Stale `docs/ENVIRONMENT.md` row.** `VT_ROCM_QUANT_WMMA`'s doc entry
   still described the Q6_K-only, exact-16-alignment gate from before the
   Q4_K and W1.5 commits in this same branch. Corrected to name both
   kernels (`KQuantGemmKWmmaQ6K`/`KQuantGemmKWmmaQ4K`) and the actual gate
   (`m >= 16 && n >= 16`, not exact alignment). A stale comment making the
   same claim at `QuantWmmaEnabled`'s definition (`rocm_grouped_gemm.hip`)
   was corrected alongside it — same fact, same staleness, same file
   already in scope.

2. **Tail-fill launch grid sized to the full domain.** Both tail-fill call
   sites in `MatmulBTQuantKernelRocm` launched `KQuantGemmK` over the
   entire `m*n` grid and relied on an `i < skip_m && j < skip_n` guard to
   no-op every warp landing inside the WMMA corner. For a wide production
   `N` (this row's own `N=12288` shape) and a large aligned corner, the
   no-op warps outnumber the real ones by roughly the corner's own area —
   an unmeasured, real launch-overhead cost the row had not isolated.
   Fixed by replacing the skip guard with an index remap: a launched
   index `t` now enumerates ONLY the remainder — the bottom strip
   `[skip_m,m)x[0,n)` (contiguous at the buffer's own row stride `n`, so
   no separate stride parameter), then the right strip
   `[0,skip_m)x[skip_n,n)` — and the host sizes the tail-fill grid to
   `bottom_strip + right_strip`, not `m*n`. The pre-existing full-grid path
   (`skip_m == 0 && skip_n == 0`, every call site outside the tail arm)
   takes an unchanged branch in the same kernel.

   Hardware-verified on the RX 9060 XT: the existing M=37/N=50
   (both-misaligned) case still passes, plus two NEW asymmetric cases this
   row had not isolated — M=37/N=48 (bottom-strip-only remainder, N exactly
   aligned) and M=32/N=50 (right-strip-only remainder, M exactly aligned)
   — both formats, CPU-oracle NMSE (measured 0.0 for every shape and
   format against the fixed kernel — bit-identical to the CPU oracle, not
   merely under tolerance) plus the existing dispatch-count reachability
   check.

   **Mutation-proving the new mechanism, not the retired guard.** The
   guard-based guarantee this row's spec already recorded (`&&` to `||`)
   no longer applies once the mechanism changed; re-asserting it would
   have proven nothing. Two mutations were tried against the NEW
   index-remap, and the first is recorded because it under-proves rather
   than over-proves: an off-by-one at the bottom/right boundary
   (`t < bottom_strip` to `t <= bottom_strip`) reds only 0 to 3.18e-4 NMSE
   depending on shape — under this suite's 5e-4 bound — because it
   miscomputes exactly ONE cell out of the FULL `m*n` output the whole-
   buffer NMSE is normalized against (and on the M=37/N=48 shape, whose
   remainder is bottom-strip-only, it reds NOTHING at all: that shape's
   `right_strip` is 0, so `t` never reaches the mutated boundary). That
   result is evidence about this test's proof strength at this scale, not
   a clean pass — recorded rather than discarded. The second mutation —
   the right-strip remap's divisor (`t2 / (n - skip_n)` to `t2 / n`,
   `t2 % (n - skip_n)` to `t2 % n`, a forgot-to-narrow-the-divisor bug in
   the same class) — reds unambiguously: NMSE `nan`/`-nan` on both the
   combined M=37/N=50 case and the N-misaligned M=32/N=50 case, all
   formats (out-of-bounds device reads through a corrupted `i`/`j`).
   Reverting either mutation restores an identical source hash
   (`sha256sum` verified) and an identical GREEN result.

   **Op-level A/B, same tool, `examples/quant-gemm-bench` extended with a
   `M=132, N=12288, K=2048` shape** (M=132 deliberately un-aligns M by one
   tile past 8*16 at this row's own widest production N), same-binary
   toggle by reverting only `rocm_grouped_gemm.hip` to its pre-repair
   content and rebuilding (not `VT_ROCM_QUANT_WMMA`, which selects the
   scalar arm entirely rather than isolating the tail-fill launch), best-
   of-6, idle host:

   | Format | before (full-grid tail launch) | after (remainder-sized tail launch) | ratio |
   |---|---:|---:|---:|
   | Q6_K | 1449.0-1449.8 GFLOP/s | 1678.1-1680.0 GFLOP/s | +15.8-15.9% |
   | Q4_K | 1809.3-1826.7 GFLOP/s | 2179.7-2187.6 GFLOP/s | +19.3-20.9% |

   Real, not negligible, and reported at the actual measured range rather
   than a single cherry-picked run. Q8_0 (a different, unrelated launch
   path that carries no tail-fill mechanism) was measured alongside as a
   control and is flat before/after (~2065-2083 GFLOP/s), as expected.

**Wider WMMA block (issue #3032): MEASURED AND REJECTED.** A same-tool
`rocprofv3` trace of `KQuantGemmKWmmaQ4K`/`Q6K` against llama.cpp's own
`mul_mat_q` on the identical model/prompt found this kernel 4.9x (Q6_K) to
10.6x (Q4_K) slower per-kernel, and llama.cpp's launch configuration uses
double the warps per block (256 vs 128 threads) and 16-48x fewer blocks per
launch yet finishes 15-45x faster. The obvious hypothesis — that packing
more of this kernel's already-independent warps into fewer, bigger blocks
would close some of that gap — does not hold: each warp here owns a fully
independent output tile with its own shared-memory slice, so widening only
changes how many independent warps share one launch's LDS budget, not how
much work any one warp does.

Templated `KQuantGemmKWmmaQ6K`/`KQuantGemmKWmmaQ4K` on `WarpsPerBlock` and
added an 8-warp instantiation behind `VT_ROCM_QUANT_WMMA_WIDE=1` (default
off). Hardware-verified correct (`ctest -R rocm|cross_device`, both configs,
46/46 cases, 84066/84066 assertions, zero regression). Op-level A/B
(`examples/quant-gemm-bench`, RX 9060 XT, best-of-4, idle host), 4-warp
(current default) vs 8-warp, all six Q4_K/Q6_K prefill shapes:

| Shape | 4-warp (current) | 8-warp | ratio |
|---|---:|---:|---:|
| Q4_K N=3072 K=2048 | 1948.5 GFLOP/s | 1827.9 GFLOP/s | -6.2% |
| Q4_K N=12288 K=2048 | 2242.7 GFLOP/s | 2027.9 GFLOP/s | -9.6% |
| Q4_K N=2048 K=6144 | 1941.3 GFLOP/s | 1966.1 GFLOP/s | +1.3% |
| Q6_K N=3072 K=2048 | 1874.0 GFLOP/s | 1775.9 GFLOP/s | -5.2% |
| Q6_K N=12288 K=2048 | 2159.1 GFLOP/s | 1981.4 GFLOP/s | -8.2% |
| Q6_K N=2048 K=6144 | 1885.0 GFLOP/s | 1933.5 GFLOP/s | +2.6% |

Geomean -4.3%: a net regression, not a win. `VT_ROCM_QUANT_WMMA_WIDE` stays
in the tree default-off, the same posture `VT_ROCM_Q6K_SMALL_PRIVATE` above
ships with, as a ready-made A/B for re-checking this specific axis on
different hardware (gfx1201) or a future toolchain revision rather than
re-deriving it from scratch — not carried forward as an open question. The
sharper, still-open hypothesis this measurement points at — cross-warp data
reuse llama.cpp's kernel may have that this one's per-warp-independent
design does not — is recorded in `## Owed` above, unread and unconfirmed,
not assumed.

**Cooperative activation share (issue #3034): MEASURED AND REJECTED, and
worse than the wider-block-alone result above.** Implemented
`KQuantGemmKWmmaQ6KShared`/`KQuantGemmKWmmaQ4KShared` per the `## Design:
cooperative activation share` section: the 8-warp block cooperatively
stages the shared `it`'s 16-row `BlockQ8_K` bytes into shared memory once
per superblock, gated on `n_tiles % 8 == 0` (falls back to the plain
per-warp kernel otherwise) behind `VT_ROCM_QUANT_WMMA_SHARE_ACT=1`
(default off). Hardware-verified correct: `ctest -R rocm|cross_device`,
46/46 cases, 84066/84066 assertions, zero regression, both configs.

Two copy implementations were tried, in order, because the first one's
result was ambiguous enough to check rather than trust:

1. A byte-at-a-time copy computing `row = byte / sizeof(BlockQ8_K)` per
   byte. Best-of-4 geomean across the six shapes: -18.7% vs the plain
   8-warp arm.
2. Suspecting the per-byte division, rewrote as an outer loop over the 16
   rows (unrolled) with an inner word-sized (`int`, `sizeof(BlockQ8_K)` is
   exactly 73 words) copy, division-free. Barely moved: -10.9% vs the
   plain 8-warp arm, because the real cost of that version was ~73-of-256
   threads active per row-step, not the division it removed.
3. Rewrote again as ONE flat loop over all 1168 words (16 rows'
   worth), keeping every thread active every iteration: still `idx /
   kWordsPerRow` and `idx % kWordsPerRow`, but by the compile-time
   constant 73, which the compiler lowers to a multiply-shift rather than
   a genuine divide.

Op-level A/B (`examples/quant-gemm-bench`, RX 9060 XT, best-of-4, idle
host), version 3 (the fully-parallel, division-by-constant copy) against
both prior arms:

| Shape | 4-warp default | 8-warp, no share (#3033) | 8-warp + share |
|---|---:|---:|---:|
| Q4_K N=3072 K=2048 | 1964.7 | 1830.4 | 1620.4 |
| Q4_K N=12288 K=2048 | 2247.0 | 2015.4 | 1772.8 |
| Q4_K N=2048 K=6144 | 1949.9 | 1953.9 | 1726.3 |
| Q6_K N=3072 K=2048 | 1883.1 | 1769.0 | 1564.8 |
| Q6_K N=12288 K=2048 | 2174.1 | 1973.3 | 1757.0 |
| Q6_K N=2048 K=6144 | 1876.8 | 1930.1 | 1704.4 |

Geomean: -16.0% vs the 4-warp default, **-11.5% vs the already-rejected
8-warp-no-share arm** — sharing did not just fail to help, it cost more
than the 8-warp width regression by itself. The result held essentially
unchanged (within noise) across all three copy implementations once the
worst inefficiency was fixed, which is itself evidence: this is not a
copy-implementation artifact, it is the cost of the cooperative-staging
mechanism itself (the extra shared-memory round trip and its
`__syncthreads()`, once per superblock, `nsb` times per kernel call)
outweighing whatever redundant-global-read cost it removes.

**What this settles and what it does not.** The confirmed redundancy
(every warp in a block independently re-reading the same activation rows)
is real and traced precisely in `## Owed` above — but removing it this
way costs more than it saves, which is itself evidence that those
redundant reads were likely already well-served by cache (this card's
Infinity Cache) rather than costing real DRAM bandwidth. Two of the three
axes this row has now measured (block width, activation sharing) are
closed as net negatives. `VT_ROCM_QUANT_WMMA_SHARE_ACT` stays in the tree,
default off, for the same reason `VT_ROCM_QUANT_WMMA_WIDE` does: a
ready-made, already-correct A/B for different hardware or a future
toolchain, not an open question to re-litigate here. The remaining
untested hypothesis is llama.cpp's own per-kernel efficiency at the
instruction/ISA level (WMMA throughput, register allocation, shared-memory
bank-conflict patterns in its actual generated code) rather than tiling or
data reuse — not confirmed, not assumed, the next traceable step if this
row continues.

**Test-coverage gap found and fixed.** Both this row's existing WMMA
correctness tests ("keep-quant Q6_K/Q4_K WMMA tile arm...") use N=48
(`n_tiles=3`), which never satisfies `n_tiles % WarpsPerBlock(8) == 0` --
the precondition Shared (and BigTile, below) both require. Every "green"
result reported for Shared above was real (the assertions did pass), but
the kernel under test was silently the plain fallback, not Shared itself,
because the existing tests structurally never reach it. Added
per-variant dispatch counters (`g_kq_wmma_share_dispatches`/
`_share_q4k_`/`_bigtile_`/`_bigtile_q4k_`, distinct from the generic ones
the old tests use, which cannot tell variants apart) and two new test
cases at N=128/M=80 (n_tiles=8 satisfies the precondition; m_tiles=5 is
deliberately not a multiple of ItGroup=3, exercising BigTile's ragged
last-block skip) that prove the SPECIFIC kernel launched via
`CHECK(after > before)` on its own counter, not inferred from shapes.
Re-ran Shared under this real test: still correct (NMSE clean), so its
rejection above stands on genuine evidence, just not the evidence
originally cited for it.

**ISA-level check (the ratified next step above): CONFIRMED to rule out
register allocation, not to explain the gap.** Read the actual compiled
GCN assembly on both sides (`-save-temps` for ours; llama.cpp's own
`.so` unbundled per-`ggml_type` via `clang-offload-bundler` and read
through `llvm-readobj --notes` for the AMDGPU kernel metadata) rather
than inferring from source:

| Kernel | VGPRs | Spill |
|---|---:|---|
| Ours `KQuantGemmKWmmaQ4K` (shipping default) | 192 | 60 B, one-time setup/epilogue only (traced: outside the per-superblock loop, not a recurring cost) |
| Ours `KQuantGemmKWmmaQ6K` (shipping default) | 142 | none |
| llama.cpp `mul_mat_q<Q4_K, I=128, J=128>` | 247 | none |
| llama.cpp `mul_mat_q<Q6_K, I=128, J=128>` | 202 | none |

llama.cpp uses MORE registers than us in both formats and still wins by a
wide margin -- ruling out "fewer registers, more occupancy" as their
advantage. Our one real spill is confirmed cosmetic (one-time, not
per-iteration). This corroborates the structural (tile-size/reuse) story
rather than pointing at a codegen defect on either side.

**BigTile (second cut, issue #3034): MEASURED AND ACCEPTED.** Reading
llama.cpp's actual shared-memory formula
(`ggml_cuda_mmq_get_nbytes_shared_x`, `mmq.cuh:415-419`) resolved why its
128x128 tile fits this card's 64 KiB LDS budget when a naive scale-up of
our own approach does not (86.5 KiB, see the Shared section above): it is
NOT a smaller encoding -- their `block_q8_1_mmq` (144 B / 128 K-elements
~= 1.125 B/element) and our `BlockQ8_K` (292 B / 256 K-elements ~= 1.14
B/element) have essentially identical density. The difference is
granularity: their weight tile stages only `MMQ_TILE_NE_K=32`
K-elements/row at a time (`mmq.cuh:132-136`) and their activation tile
stages only 128 K-elements/row at a time, against our 256-wide
superblock-at-once staging for both operands -- computed exactly for
I=128/J=128/Q4_K: `nbs_x` (38.9 KiB) + `nbs_y` (18.0 KiB, padded) +
`nbs_ids` (0.5 KiB) = 56.5 KiB, comfortably under budget with 7.5 KiB to
spare. They pay the staging/sync cost up to 8x more often per unit of K
depth than we do, in exchange for a peak footprint small enough to afford
a much wider tile.

Implemented `KQuantGemmKWmmaQ6KBigTile`/`Q4KBigTile<OutT, WarpsPerBlock,
ItGroup>`, keeping this row's existing 256-wide superblock staging
granularity (unlike llama.cpp, not chunked finer -- see `## Owed`) and
choosing the largest `ItGroup` (activation-row groups reused per staged
load) that still fits: `ItGroup=3` (48 rows, ~14 KiB) alongside the
existing ~50 KiB per-warp weight-side footprint at 8 warps. Grid is 2D
(`blockIdx.x` gives each warp a fixed `jt`, matching the existing design;
`blockIdx.y` gives the whole block a shared `it_base` spanning `ItGroup`
groups, looped inside each warp) rather than the previous flat 1D `tile`
encoding, because a 2D grid needs no `n_tiles % WarpsPerBlock == 0`
precondition on its OWN axis-mixing (removed for the `it`/M direction;
kept for `jt`/N, because a warp whose `jt` falls outside `n_tiles` must
never exist in a launched block -- its early `return` would desync the
block's later `__syncthreads()` calls from warps that keep running).
Weight dequant (and, for Q4_K, `UnpackQ4KScalesMins`) now execute ONCE
per superblock per warp and serve all `ItGroup` iterations, not just
one -- weight-side reuse Shared did not have either.

Hardware-verified correct via the new dedicated tests above (not the
stale N=48 ones): `ctest -R rocm|cross_device`, 48/48 cases, 84084/84084
assertions, zero regression, both configs.

Op-level A/B (`examples/quant-gemm-bench`, RX 9060 XT, best-of-4, idle
host), BigTile vs the shipping 4-warp default:

| Shape | Default | BigTile | ratio |
|---|---:|---:|---:|
| Q4_K N=3072 K=2048 | 1964.7 | 1996.6 | +1.6% |
| Q4_K N=12288 K=2048 | 2247.0 | 2766.2 | +23.1% |
| Q4_K N=2048 K=6144 | 1949.9 | 2377.6 | +21.9% |
| Q6_K N=3072 K=2048 | 1883.1 | 2158.0 | +14.6% |
| Q6_K N=12288 K=2048 | 2174.1 | 2775.2 | +27.6% |
| Q6_K N=2048 K=6144 | 1876.8 | 2130.9 | +13.5% |

Geomean **+16.8%**, and +22.9%/+38.9% against the two rejected arms
(wider-block-alone, activation-share-alone) respectively -- consistent
with the diagnosis that neither axis alone cleared the staging cost, and
this row's own ragged "tail" shape (M=132, not 16-aligned) shows the same
order of improvement (+19.1%/+19.9%), so this is not an artifact of
perfectly-aligned synthetic shapes.

**Real-model confirmation**, same isolated-prefill recipe as gate (c)
above (`Ornith-1.5-9B-Q4_K_M.gguf`, 512-token prompt, `rocprofv3`,
isravale, idle, 3 reps):

| | Default (4-warp) | BigTile | ratio |
|---|---:|---:|---:|
| Prefill total kernel time (whole forward pass) | 3012.9 ms | 2590.9 ms | -14.0% |
| ...vs oracle pp512 (274.3 ms) | 10.98x slower | 9.45x slower | |
| Quant-GEMM kernels only (Q4_K+Q6_K) | 1982.7 ms | 1552.0 ms | -21.7% |
| ...vs llama.cpp `mul_mat_q` (222.3 ms) | 8.92x slower | 6.98x slower | |

The synthetic op-level win survives contact with the real checkpoint,
diluted by the other kernels in the forward pass (`GdnScanK`, attention,
the dense bf16 GEMM) exactly as expected -- not inflated, not an
artifact. Still default OFF behind `VT_ROCM_QUANT_WMMA_WIDE=1
VT_ROCM_QUANT_WMMA_BIGTILE=1`; whether to flip the default is a decision
for after review, not made in this wave.
