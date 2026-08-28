# ROCm KQuantGemmK: warps-per-row for small-`nsb` decode dispatches

Row: `BACKEND-ROCM`. Issue: [#1910](https://github.com/mudler/vllm.cpp/issues/1910).
Branch: `row/ROCM-KQUANT-NWARPS-DECODE`.

## Problem, restated from the issue

`KQuantGemmK` (`src/vt/rocm/rocm_grouped_gemm.hip:355-378`) gives one warp to
each `(i, j)` output element. Each lane strides over `nsb = K / 256`
superblocks (`for (sb = lane; sb < nsb; sb += 32)`), then the warp always runs
a fixed 5-round `__shfl_down_sync` reduction regardless of how many lanes did
real work. At `K = 4096` (`nsb = 16`), lanes 16-31 execute the loop body zero
times. Measured on `4b1154bc5` (RX 9060 XT, gfx1200, ROCm 7.2.3): 195 of 259
decode (`m=1`) dispatches carry `nsb = 16`; `KQuantGemmK`'s three
instantiations total 22.937 ms/token, 54.3% of decode GPU time, and the
matmul family (`KQuantGemmK` + `wvSplitKSml`) is 1.585x behind llama.cpp
`b10451`'s `mul_mat_vec_q` on the identical workload. Two shapes are **not**
affected and a fix must not regress them: `n=4096 k=12288` (`nsb=48`, packs
all 32 lanes already).

## Upstream anchor — and why the issue's own hypothesis is wrong

The issue speculates the fix "likely" packs multiple `(i, j)` outputs into one
warp when `nsb < 32` ("choose the lane->work map from nsb"). Reading the
oracle (`AGENTS.md` "Run the oracle and read its source") shows this is **not**
what llama.cpp does on our hardware family, and building it would diverge from
the oracle rather than mirror it.

llama.cpp pin `10bf611e533d81f739128304991c5e133c6aebd8` (tag `b10451`,
`.agents/oracles/llama-cpp.md`), `ggml/src/ggml-cuda/mmvq.cu`:

- **The multi-output-per-warp path exists (`rows_per_cuda_block`) but is
  wired OFF for every RDNA table.** `calc_rows_per_block()` (`:460-478`) only
  returns >1 for `MMVQ_PARAMETERS_GENERIC/GCN/TURING`; `RDNA2/RDNA3_0/RDNA4`
  fall through to the unconditional `return 1;` at `:477`. Separately,
  `should_use_small_k()`'s lambda (`:863-903`) forces `use = false` whenever
  `GGML_CUDA_CC_IS_RDNA(cc)` (`:898`), independent of the table check. Two
  independent gates both refuse row-packing on our hardware family.
  **Correction, checked via `gh api` blame + PR history against the same
  pin**: this is not a measured architectural finding. `calc_rows_per_block`'s
  `GENERIC || GCN` condition never included any RDNA table from its first
  diff (PR #20635, 2026-03-16, NVIDIA-only benchmarks). The
  `GGML_CUDA_CC_IS_RDNA(cc) -> use = false` line first appears in PR #20885
  (batched small-K, RTX 5090 only) and was carried unchanged into PR #20905's
  `should_use_small_k`; that PR's seven-comment review thread has one AMD
  contributor spot-checking CDNA (AMD's datacenter line) as "ok" and never
  mentions RDNA. No AMD RDNA hardware appears in either PR. **Row-packing on
  RDNA is untested upstream, not proven worse** — do not cite this exclusion
  as evidence row-packing would regress on our board. The `nwarps` scaling
  this row ports instead has a real, independent RDNA4 benchmark backing it
  (next bullet), which is why it is the port target regardless.
- **What RDNA4 actually does is scale `nwarps` per output row.**
  `calc_nwarps()` (`:354-458`), the `MMVQ_PARAMETERS_RDNA4` branch (`:385-408`):
  for `ncols_dst == 1` (single-token decode, our regime) it returns **8** for
  `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS` — our three
  formats (`Q4_K`/`Q5_K`/`Q6_K`) are all on that list — and **1** for
  everything else (comment: "Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*)
  regress due to register pressure and lookup table contention at higher
  thread counts"). RDNA3 (`:409-428`) is a stricter whitelist — Q4_K/Q5_K are
  explicitly *not* on it, only `Q6_K` at `nwarps=2`.
- **Mechanism**: `rows_per_cuda_block` stays 1 (one row per thread block), but
  the block now has `nwarps` warps. `blocks_per_iter = vdr*nwarps*warp_size/qi`
  (`:507`) spreads the row's `blocks_per_row_x` (our `nsb`) superblocks across
  all `nwarps*warp_size` lanes rather than one warp's 32. Each warp accumulates
  its own partial sum (`:594-614`), writes it to
  `__shared__ tmp_shared[nwarps-1][...][warp_size]` (`:616-632`), then warp 0
  sums the `nwarps-1` shared partials into its own before the final
  `warp_reduce_sum` shuffle (`:640-660`).
- **The real win is occupancy, not per-warp lane utilization.** With `nsb`
  small, most of the `nwarps*32` lanes still execute the loop body zero or one
  times — spreading 16 superblocks over 256 lanes is not less idle in
  aggregate than 16-of-32. The benefit is that one output row now occupies 8
  warps' worth of the SM's resident-warp budget instead of 1, which hides the
  K-quant weight-fetch latency better on a kernel that is memory-bound at
  `m=1`. This is why `should_use_small_k` frames the trigger as "the full
  thread block covers all K blocks in a single loop iteration" (`:864-865`,
  `:872`: `blocks_per_row_x < nwarps * blocks_per_iter_1warp`) — it is a
  latency-hiding decision, not a SIMD-utilization one.

**Consequence for this row**: the port target is "more warps cooperate on one
output row via a shared-memory partial-sum reduction, gated by format and
triggered when a single warp would finish `nsb` in one loop iteration" — not
"more outputs per warp." The exact `nwarps` value and which formats qualify on
RDNA4 are llama.cpp's own measured whitelist for *their* `vec_dot_q4_K_q8_1`
bodies, not ours; `DotQ4K`/`DotQ5K`/`DotQ6K` (`src/vt/rocm/...`) are a
different implementation with a different register footprint, so llama.cpp's
whitelist is a **starting hypothesis to measure**, not a value to copy
blindly. This spec's `## Design` requires re-deriving it against our own
kernel body's occupancy behavior before committing to `nwarps=8`.

## Scope

- **In scope**: `KQuantGemmK` (`rocm_grouped_gemm.hip:355-378`) and its host
  launcher `MatmulBTQuantKernelRocm` (`:446-496`) — the dense, non-grouped
  decode path the issue measured. Formats: Q4_K, Q5_K, Q6_K (the three the
  issue's shape table covers).
- **Out of scope, listed under `## Owed`**: `GroupedKQ8K` (`:323-349`,
  `MatmulBTQuantGroupedKernelRocm` `:502+`), the MoE path — same defect class,
  same fix shape, but the issue explicitly states it "was not measured" and a
  fix there needs its own dispatch shape evidence first. `Q8_0GemmK`
  (`:381-403`) is a different quant format with a different oracle whitelist
  entry (`Q8_0` qualifies on RDNA4 too, per the anchor above, but the issue's
  measurement never dispatched it, so it is not claimed here).
- Prefill (`m > 1`) dispatches are unaffected by this row: `nsb` does not
  change with `m`, and a warp-per-row kernel is not what prefill uses (larger
  `m` batches already keep lanes busy via `rows_per_cuda_block`-style batching
  in other kernels, or amortize the fixed reduction cost over more work per
  lane — out of scope to re-derive here).

## Design

1. Add an `nwarps`-parameterized cooperative-reduction variant of
   `KQuantGemmK`: `dim3 block(32, nwarps)`, each of the `nwarps` warps handles
   the SAME `(i, j)` (not independent ones), strides its lanes over `nsb`
   starting at an offset derived from `threadIdx.y` (mirroring
   `kbx = tid / (qi/vdr)` with `tid = warp_size*threadIdx.y + threadIdx.x`,
   `blocks_per_iter = nwarps*32` since our per-lane work is exactly one
   dot-product per stride step, i.e. `vdr=1, qi=32` in llama.cpp's terms),
   accumulates a partial sum, writes it to
   `__shared__ float partial[nwarps-1][32]` when `threadIdx.y > 0`,
   `__syncthreads()`, then `threadIdx.y == 0` sums the `nwarps-1` shared
   partials into its own before the existing 5-round `__shfl_down_sync`.
2. Host-side trigger, mirroring `should_use_small_k`'s condition
   (`nsb < nwarps_candidate * 32`, i.e. a single warp would finish `nsb` in
   one iteration): dispatch the cooperative variant only when
   `nsb <= 32` (covers the issue's `nsb=16` case; `nsb=48` keeps the existing
   one-warp-per-output kernel, satisfying "must not regress those shapes"
   without needing a second threshold).
3. Per-format `nwarps` is a **measured** constant per `## Upstream anchor`'s
   caution, not copied from llama.cpp's table. Start from the RDNA4 whitelist
   value (8) for Q4_K/Q5_K/Q6_K as the first candidate, sweep `{2, 4, 8}` on
   the gate model's actual dispatch shapes (`n=12288,k=4096`; `n=248320,k=4096`
   Q6_K lm_head; `n=1024,k=4096`; `n=8192,k=4096`), and record whichever wins
   per format — this may not be 8 for all three formats given a different
   `vec_dot` body.
4. `GroupedKQ8K` is explicitly NOT touched by this row (see `## Scope`); do
   not generalize the change to the grouped launcher speculatively.

## Tests

- Port the existing K-quant matmul correctness suite's ROCm case(s) (find via
  `grep -rn KQuantGemmK tests/`) run at `nsb <= 32` AND `nsb > 32` — both
  branches of the new dispatch must stay token/byte-identical against the CPU
  K-quant reference, since this changes reduction order (partial sums combine
  in a different order across warps) and floating-point reassociation can
  change rounding. If exact byte-identity does not survive the reassociation,
  the existing NMSE bar for this kernel family applies instead — state which,
  do not assume.
- A red-first case that fails on old `KQuantGemmK` for `nsb=16` by asserting
  which lanes actually execute the dot-product loop body (a `#ifdef`-gated
  counter or the existing `VT_DUMP_KQ_SHAPES` idea from the issue, ported into
  a test rather than left as a revert-only instrumentation).
- Regression: full ROCm ctest zero-delta vs the pre-row baseline for every
  case not touching `nsb <= 32` K-quant dispatch.

## Gate

- Correctness first: token-exact (or the kernel family's existing tolerance)
  on the gate model(s) already used by `.agents/oracles/llama-cpp.md`'s
  gateable pin, at both `nsb <= 32` and `nsb > 32` shapes.
- Speed: `rocprofv3 --kernel-trace --stats`, same differencing method as the
  issue (`--max-tokens 4` vs `--max-tokens 36` over the same 32-token
  workload, same model, same host), reproduced 2-3x idle per
  `AGENTS.md` §Gates. Report `KQuantGemmK`'s new ms/token share and the
  decode-total delta; do not claim a ceiling and do not extrapolate past what
  was measured on gfx1200.
- The `nsb=48` shapes (`n=4096 k=12288`, Q5_K/Q6_K) must show zero regression
  (same kernel, same launch config as before this row).

## Risks

- The `nwarps` sweep in `## Design` item 3 may show our `DotQ4K/Q5K/Q6K`
  bodies do not benefit the way llama.cpp's `vec_dot_q*_K_q8_1` do (different
  register/LUT pressure) — if so, record the measured optimum (which may be
  `nwarps=1`, i.e. no win) rather than forcing 8.
- Shared-memory cross-warp reduction changes floating-point summation order;
  verify against the existing correctness bar before claiming a speed number.
- The 593 us/call outlier the issue leaves unattributed (likely lm_head,
  `n=248320 k=4096` Q6_K) is IN this row's measured shape set, so this row's
  gate should resolve that attribution as a side effect — record it, do not
  treat it as closed unless the correlation is actually run.

## Owed

- `GroupedKQ8K` / `MatmulBTQuantGroupedKernelRocm` (MoE path, `:547`) — same
  defect class, unmeasured, needs its own dispatch-shape evidence before a
  fix is scoped (per the issue's own "What is NOT established").
- `Q8_0GemmK` — same defect class per the oracle anchor (Q8_0 is on the RDNA4
  whitelist), not measured by the issue, not touched here.
- **Q4_K and Q5_K keep the single-warp arm**, and the next hypothesis for them
  is row-packing, not a different `nwarps`. Every width in {2, 4, 8} lost at
  every `nsb <= 32` shape (see `## Outcome`), and the mechanism — a superblock
  header re-read once per warp, and a quant read broken into eighths — gets
  worse as the split widens, so there is no unmeasured width left to try. What
  is genuinely untried is `rows_per_cuda_block`: per the correction in
  `## Upstream anchor`, upstream's RDNA exclusion of row-packing was never
  benchmarked on RDNA hardware, so "llama.cpp turns it off" is not evidence it
  loses here. That needs its own row, spec and measurement.
- **`test_backend_cross_device` is RED on gfx1200 before this row and after
  it**, at `:2067` `CHECK(got == ref_b)` in the case "MoeSiluMul matches the CPU
  oracle within NMSE <= 5e-4", 1 of 26 cases and 1 of 80253 assertions. #1954
  records the same assertion at `:2063`, which is where it sat before this
  row's `CAPTURE` stringification repair in
  `tests/vt/test_backend_cross_device.cpp` moved it down four lines. It is
  **not this row's defect and this row does not own the fix**. Proven
  pre-existing by reverting only this row's two source files to the parent
  `5888abf11` and rebuilding, which reproduces the identical failure at 24 of 25
  cases and 1 of 80195 assertions. The control was run twice, once by the review
  and once again when this note was written, because a cited control is not a
  run one. Filed as
  [#1954](https://github.com/mudler/vllm.cpp/issues/1954), the gfx1200
  counterpart of the CUDA-only bookkeeping in
  [#1802](https://github.com/mudler/vllm.cpp/issues/1802), which carries this
  same test name and this same assertion on sm_110.
  [#907](https://github.com/mudler/vllm.cpp/issues/907) is the same defect
  family on sm_121a and **NOT the same test**: it records `test_cuda_ops`'s
  "CUDA silu_and_mul matches CPU" failing at 439 of 440 assertions, a different
  test and a different assertion. An earlier draft of this section, of the
  `.agents/environment.md` row and of the #1954 index row claimed #907 carried
  the same test and assertion, which is false; the corrected body of #1954 is
  the source of truth for that distinction. `.agents/environment.md` now
  carries a gfx1200 known-red row for it, so the next reader of this gate reads
  one expected red rather than a regression. This row's own gate is therefore
  reported as one pre-existing failure, never as a clean pass.
- **`KQuantGemmKCoopQ6K`'s multi-iteration barrier is unreachable and
  untested.** `iters = (nsb + 31) / 32` and `KQuantDecodeCoopWarps` returns a
  cooperative width only for `nsb <= 32`, so `iters` is always exactly 1 and the
  second `__syncthreads()` in the loop body never separates two iterations.
  Deleting it leaves the suite green at 58 of 58, measured. The barrier is kept
  because it becomes load-bearing the moment the `nsb` bound moves, and the
  claim beside it in `src/vt/rocm/rocm_grouped_gemm.hip` now says so instead of
  describing a race the dispatch cannot produce. What is owed is a gate: either
  a dispatch that reaches `nsb > 32` cooperatively, or a test that instantiates
  the kernel at a trip count above 1. Neither belongs to this row, because both
  need the `nsb > 32` measurement `## Gate` deliberately excluded.
- **`DotQ6KIsumRange` duplicates `DotQ6K`'s arithmetic** rather than `DotQ6K`
  calling it at the full range, because the ranged loop costs the donor's
  unrolling (642.59 us/call against 371.90 on `n=4096 k=12288`, measured). The
  duplication is held honest by the byte-equality case rather than by a
  comment, but one body computing Q6_K would be better than two, and a
  compile-time range parameter is the obvious way to get there.

## Outcome

`DONE`. What landed is narrower than `## Design` proposed, because the sweep
`## Design` item 3 required contradicted the design's own premise for two of
the three formats.

### What landed

A cooperative arm, `KQuantGemmKCoopQ6K<OutT, 8>`, dispatched for **Q6_K only**
at `m == 1 && nsb <= 32`. Eight warps share one output row; warp `w` takes
sub-block range `[w, w+1)` of each superblock its lane holds. `DotQ4K`,
`DotQ5K` and `DotQ6K` are byte-for-byte the pre-row bodies, so every arm this
row did not claim runs the code it ran before.

### The sweep, and why Q4_K and Q5_K were rejected

One `rocprofv3 --kernel-trace --stats` run per `(format, shape)` so the
per-kernel average attributes to exactly one shape; 100 timed iterations after
20 warm-up, RX 9060 XT / gfx1200 / ROCm 7.2.3, f32 output. us/call:

| fmt | n x k | pre-row | nw=1 | nw=2 | nw=4 | nw=8 |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | 12288 x 4096 | **154.93** | 158.21 | 296.71 | 244.06 | 268.74 |
| Q4_K | 8192 x 4096 | **104.50** | 105.60 | 201.16 | 167.71 | 183.82 |
| Q4_K | 1024 x 4096 | **16.65** | 16.79 | 29.22 | 24.56 | 28.12 |
| Q4_K | 248320 x 4096 | **2475.78** | 2495.02 | 4464.84 | 3885.22 | 4287.58 |
| Q5_K | 12288 x 4096 | 180.92 | **180.06** | 314.26 | 284.74 | 296.28 |
| Q5_K | 248320 x 4096 | **2788.14** | 2825.68 | 4546.94 | 4203.04 | 4302.89 |
| Q6_K | 12288 x 4096 | 393.20 | 453.24 | 464.91 | 496.86 | **389.04** |
| Q6_K | 1024 x 4096 | 40.13 | 60.35 | 45.95 | 47.82 | **37.67** |
| Q6_K | 248320 x 4096 | 6625.34 | 6072.71 | 6298.22 | 6765.27 | **5615.01** |
| Q6_K | 4096 x 12288 (nsb=48) | **371.90** | 642.59 | 640.73 | 649.08 | 659.92 |

**Q4_K and Q5_K lose 1.5x to 1.8x at every width and every `nsb <= 32` shape,
and the loss does not shrink as `nwarps` falls.** `## Risks` anticipated a
measured optimum of 1 and that is the answer for both. The mechanism is
traffic, not occupancy: splitting a superblock makes every warp re-read its
16-byte header and turns one contiguous 128-byte quant read into eight
32-byte ones. For a GEMV that is already bandwidth-bound, the bytes cost more
than the extra resident warps buy. `## Upstream anchor`'s framing — "the real
win is occupancy, not per-warp lane utilization" — is therefore right about
where a win would have to come from and wrong that it arrives.

Q6_K escapes because its per-lane cost is not a read. `KQuantGemmK<bf16, Q6_K>`
reports **272 bytes of scratch per thread and 152 VGPRs** in the kernel trace:
the donor rebuilds all 256 weights into `int8_t aux8[kQK_K]` and spills it.
Ranged to one sub-block the array is 32 bytes, and the cooperative kernel
reports **0 scratch and 72 VGPRs**. That is what pays for the extra traffic,
and it is why the same restriction does not help the 4-bit formats, whose dot
bodies never spilled.

### Rejected: making `DotQ6K` call the ranged core

The obvious de-duplication — `DotQ6K` = `DotQ6KIsumRange(0, 8)` — costs
**642.59 us/call against 371.90** on `n=4096 k=12288` (1.73x, the `nw=1` column
above). The donor's two fixed 128-wide chunks unroll; a loop over a runtime
group range does not. `DotQ6K` therefore keeps the donor body and the ranged
core is a second implementation, listed under `## Owed`.

### Correctness: byte-identical, not within tolerance

The cross-warp reduction carries the **integer** dp4a accumulator, so the
partials reassemble exactly and the one f32 expression per superblock is
unchanged. `-ffp-contract=off` (already in the HIP flags) removes the
remaining way two spellings could diverge. Two gates hold it:

- Host-side, 20000 random block quadruples x 4 splits: the ranged cores equal
  the donor bodies in **0 of 80000** combinations mismatched.
- On hardware, through `vt::MatmulBTQuant`: the same activation row at `m == 1`
  (cooperative) and `m == 2` (single-warp) returns **identical bytes**, for
  Q4_K/Q5_K/Q6_K at `nsb` 16 and 48. Byte equality, not the family's NMSE bar
  — the NMSE bar is checked too, but it is not what this row rests on.

### Speed, end to end

`rocprofv3 --kernel-trace --stats`, `--max-tokens 4` differenced against
`--max-tokens 36` over 32 tokens, `Ornith-1.5-9B-Q4_K_M.gguf`, prompt "The
capital of France is", batch 1, `build-hip/examples/vllm-cli`. Three runs per
arm, spread under 0.9%:

| | pre-row | this row | ratio |
|---|---:|---:|---:|
| `KQuantGemmK<bf16, Q6_K>` family | 12.296 ms/tok | 11.154 | **1.102x** |
| `KQuantGemm*` total | 22.566 | 21.459 | 1.052x |
| total decode GPU time | 54.340 | 53.311 | 1.019x |

The pre-row run reproduces the issue's headline row closely: 583.9 us/call
over 21 calls/token against the issue's 593 and 21.

### The 593 us/call outlier, attributed

`## Risks` asked for this and the data resolves it. Bucketing the pre-row
kernel trace by grid size, the single `KQuantGemmK<bf16, Q6_K>` instantiation
the issue reported as "593 us/call" is an **average over three shapes**:

| calls/tok | n x k | nsb | us/call |
|---:|---|---:|---:|
| 16 | 4096 x 12288 | 48 | 345.9 |
| 4 | 1024 x 4096 | 16 | 41.3 |
| 1 | **248320 x 4096** (lm_head) | 16 | **6562.7** |

The issue's guess that lm_head is the expensive one is right, but 593 us/call
was never lm_head's cost — it understated it **11x**. lm_head alone is 6.56 of
that row's 12.26 ms/token, 53% of it, and 12% of all decode GPU time in one
dispatch per token. After this row the two `nsb = 16` buckets move to the
cooperative arm: lm_head 6562.7 -> **5425.3** us/call (1.21x) and n=1024
41.3 -> **27.1** (1.52x). The `nsb = 48` bucket keeps its kernel and its launch
config and measures 345.9 -> 349.6, inside run-to-run spread — the
no-regression condition `## Gate` names.

### Standing gaps

- `QuantizeQ8KK` is now the largest single decode kernel at 12.87 ms/token,
  24.2% of decode, 129 calls/token. It is issue #1876's kernel, not this row's.
- The matmul family is still behind llama.cpp `b10451`. This row moves total
  decode 1.019x. No ceiling is claimed and no number here extrapolates off
  gfx1200.

## Now

`DONE` — Q6_K cooperative decode arm landed, measured and gated.
