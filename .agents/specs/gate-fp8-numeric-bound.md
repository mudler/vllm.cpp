# GATE-FP8-NUMERIC-BOUND — the numerical bound and the scale-variance probe

**Row:** `GATE-FP8-NUMERIC-BOUND`. A spec-owned row, like the rest of this
milestone chain (`VT-QUANT-FP8-GROUP`, `VT-MATMUL-FP8-BLOCK-REF`,
`VT-MATMUL-FP8-BLOCK-CUDA`, `MODEL-FP8-BLOCK-WEIGHT`, `MODEL-FP8-BLOCK-LINEAR`,
`MODEL-FP8-BLOCK-MERGED`, `GATE-QWEN38-27B-FP8-BLOCK`); the capability it
instruments is recorded in `QUANT-FP8-GENERIC`
([`quantization-matrix.md`](../quantization-matrix.md)).
**Issue:** [#1777](https://github.com/mudler/vllm.cpp/issues/1777).
**Parent:** [#1189](https://github.com/mudler/vllm.cpp/issues/1189), whose
`## Gate design` is this row's brief. **This row does not close it**; see
[`## Owed`](#owed).
**Related:** [`gate-qwen38-27b-fp8-block.md`](gate-qwen38-27b-fp8-block.md)
established layer 1 on 2026-08-23 and recorded layers 2 and 3d as owed by
whoever closes #1189; [`vt-matmul-fp8-block-cuda.md`](vt-matmul-fp8-block-cuda.md)
owns the device arm and the one-ULP-per-K-block association difference this
bound has to tolerate; [#1437](https://github.com/mudler/vllm.cpp/issues/1437)
and [#1453](https://github.com/mudler/vllm.cpp/pull/1453) established the sm120
complete-scale-block refusal.
**Lifecycle:** `DONE`.
**Owner:** unassigned

## Scope

Two of the three layers of #1189's `## Gate design`, and the honest state of the
third:

1. **Layer 2, a numerical lower bound on per-projection outputs, tight enough to
   fail a x1.10 scale perturbation.** This is the row's main work.
2. **Layer 3d, the per-block scale-variance probe**, whose degenerate
   per-tensor reading is exactly 1.0, carried in one snapshot beside the
   dispatch counter as `Fp8BlockStats`.
3. **Layer 3c in-tree, on the split arm.** The MERGED arm has asserted one fp8
   byte per element and the `cdiv` scale grid at the GEMM boundary since M6; the
   SPLIT arm asserted neither, and built an `[N,K]` tensor view over whatever
   buffer the weight carried.

Out of scope, each named so nothing here reads wider than it is:

- **Closing #1189.** Its scope is six milestones plus a gate, and this row
  touches two lines of one section of it.
- **Every speed axis.** Nothing here is measured for time and no number below
  is a performance figure.
- **The device arm's own numerical comparison.**
  `tests/vt/test_ops_matmul_fp8_block_cuda.cpp` G2/G7 compares the CUTLASS
  kernel against this same CPU reference on seven shapes and needs a lease. This
  row runs on the CPU tier and does not substitute for it.
- **Any load-time REFUSAL based on the probe.** A spread of 1.0 is suspicious,
  not illegal; see [`## Design`](#design).
- **A new oracle.** The reference is the fixture's existing independent `double`
  `RefBlockGemm`, which mirrors upstream's `native_w8a8_block_matmul`.

## What was already established, measured rather than assumed

The task this row was dispatched with named the shape assertion as missing. It
is not, and the check was made against the tree rather than against the brief.
Both halves exist on `origin/main` at `21abaf169`:

| Half of the shape assertion | Where it already fires |
|---|---|
| the scale tensor's `[cdiv(N, block_n), cdiv(K, block_k)]` shape | `src/vt/ops.cpp::MatmulFp8BlockScaled`, `VT_CHECK(b_scale.shape[0] == n_tiles && b_scale.shape[1] == k_tiles, ...)`, with the a-scale `[M, cdiv(K, block_k)]` beside it. Pinned by `tests/vt/test_ops_matmul_fp8_block_cpu.cpp` G5, subcase "a b_scale sized by FLOOR instead of cdiv, on N and on K" |
| the sm120 `N % 128 == 0 && K % 128 == 0` complete-scale-block constraint | `src/vt/cuda/fp8_block_scaled_dispatch.h::Fp8BlockScaledRefusalFor`, called at `src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu` **before anything is allocated or launched**, so the user gets a message naming the dimension and its granularity rather than CUTLASS's `Invalid status`. Pinned by `test_fp8_block_scaled_dispatch.cpp` G4 and `test_ops_matmul_fp8_block_cuda.cpp` G6 |

The CUDA translation unit says so itself at the refusal site: "`vt::ops.cpp` has
already checked ranks, dtypes, contiguity, devices and the two `cdiv` scale
shapes; what is left is what THIS kernel cannot implement". Re-implementing
either would have been a second, weaker copy of a check that already bites, and
this row does not.

**What was genuinely absent is narrower and is what landed**: the SPLIT arm's
own operand assertions. `ResidentFp8BlockPacked` handed out an `[N, K]` i8
tensor over `w.packed.bytes` without asking whether that buffer holds `N*K`
bytes, and `ResidentFp8BlockScale` uploaded a grid without asking whether its
shape is the `cdiv` one. The loader checks both
(`dense_weight_loaders.h::LoadFp8BlockRaw`) and the merged builder checks both
(`CheckFp8BlockMergeable`), so the hole is only reachable by a weight that
arrives at this seam another way — which is exactly what a synthetic fixture, a
second loader or a future quantizer is.

## Upstream anchors

Read at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, asserted as
the HEAD of the local checkout before the first read.

| What | Where |
|---|---|
| the reference this bound compares against, structure for structure | `tests/kernels/quant_utils.py::native_w8a8_block_matmul` (`:91-153`), mirrored by `tests/support/fp8_block_fixture.h::RefBlockGemm` |
| upstream's own criterion for the CUTLASS-vs-reference pair, `rel_diff < 0.001` | `tests/kernels/quantization/test_block_fp8.py::test_w8a8_block_fp8_cutlass_matmul` (`:156-200`) |
| the block scale registers as `weight_scale_inv`, `[cdiv(N,128), cdiv(K,128)]` f32 | `layers/quantization/fp8.py:378-379`; allocated `utils/fp8_utils.py:1279-1296` |
| the scales apply in the MAINLOOP, once per K-block, into an f32 accumulator | `utils/fp8_utils.py:826-838` |
| `out_dtype` on this path is the MODEL dtype, not f32 | `kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:104` -> `fp8.py:284,391-392` |
| the `cdiv` tiling on both weight axes, so a short final block still owns a scale | `utils/fp8_utils.py:930,935-936` |

## Design

### The statistic, and why the obvious one cannot carry a bound

Per output element, `|got - ref| / max(|ref|, floor)`, worst over the
projection, with `floor = 0.01 * max|ref|` over that same projection.

The floor is not cosmetic and it was measured before it was chosen. With a bare
`max(|ref|, 1e-6)` denominator — the shape the two existing bounds in
`test_fp8_block_linear.cpp` use, on one small shape where it happens to be
stable — the same grid reads an **unperturbed** worst relative error of
**6.58e-3 at f32 output on `{M=32, N=512, K=7168}`, 835x the floored reading of
7.88e-6 on identical data**, because one output element cancels to near zero and
its ratio then says nothing about any scale. Under the x1.10 perturbation the
same statistic reads **609**. A number that moves by three orders on the clean
arm for a reason that has nothing to do with the quantity being bounded cannot
carry a bound: any constant that admits the clean reading also admits a large
class of real defects.

The floored statistic is stable across every shape in the grid, which is what
makes the two constants below meaningful rather than fitted to one case.

### The two constants, and the two-sided assertion that keeps them honest

```
out_dtype   unperturbed worst   x1.10 worst   bound   headroom / bite
bf16         3.82e-3             1.034e-1      2e-2    5.24x  / 5.17x
f32          6.20e-6             1.000e-1      1e-4    16.1x  / 1000x
```

Both columns are the hardest reading each regime produced over the whole grid,
not a representative one.

**bf16 is the load-bearing constant**, because bf16 is the model dtype and every
wired call site passes it. Its reading is pinned from below by the store: this
tree's `F32ToBF16` truncates, so `2^-8 = 3.906e-3` is a structural floor that
the measured `3.82e-3` sits just under. It cannot drift up under an unrelated
change, and a future reader who tightens `kBoundBf16` will hit that floor at
about 1.28x of slack from the 4x margin assertion.

**f32 measures the arithmetic without the store's rounding on top**, and its
margin is four orders wide.

**The bound is asserted in BOTH directions in the same case.** `clean < bound`
AND `bumped > bound`, per shape, per dtype, plus a 4x margin on each side over
the whole grid. A one-directional bound stops biting the moment somebody widens
it and nothing in the tree says so; this repository has recorded that failure
for tolerances that bound nothing and for counts that cannot discriminate. The
mutation table below measures it: widening `kBoundBf16` from `2e-2` to `1.0`
fails 6 assertions.

**Loose enough for the one legitimate difference.** Our CUTLASS collective
associates the two scale multiplies left to right where the reference forms
their product first, so the arms may differ by up to one f32 ULP per K-block
([`vt-matmul-fp8-block-cuda.md`](vt-matmul-fp8-block-cuda.md)). At the widest
shape in the grid that is 56 K-blocks x `2^-24`, about **3.3e-6** relative — 30x
under `kBoundF32` and four orders under `kBoundBf16`. It is a DEVICE difference
and does not arise on this CPU tier at all; it is written down because the same
constants are what a device-side comparison would have to clear. Upstream's own
`rel_diff < 0.001` for exactly this pair sits between the two constants, as it
should.

### The grid

Six shapes, every `K` a multiple of `block_k` because the dynamic per-token
per-group activation quant requires it (`fp8_utils.py:596-599`):

| M, N, K | block | Why it is in the grid |
|---|---|---|
| 1, 128, 128 | 128,128 | decode. All 2736 dispatches of the 2026-08-23 token gate ran at `M = 1` |
| 4, 192, 128 | 128,128 | the GDN `in_proj_qkv`: a RAGGED N, where `cdiv` leaves a short final block and a floor tiling drops it |
| 4, 256, 128 | 128,128 | the merged attention QKV |
| 3, 192, 256 | 64,128 | `block_n != block_k`, so a body that swapped the two misindexes the grid. Both are 128 in every checkpoint in play, which is why the fixture has to break the tie |
| 8, 512, 1024 | 128,128 | 8 K-tiles, so a per-K-block scale has somewhere to be wrong |
| 8, 512, 7168 | 128,128 | upstream's own CUTLASS-test K at the nearest servable N. `N = 576` is refused on sm120 (#1453), and `M = 8` rather than upstream's 32 keeps the `double` reference inside the suite's time budget |

### The perturbation

One cell of the grid — the LAST block row, first K-column — multiplied by
**1.10**, the exact factor
`tests/parity/test_qwen27n_fp8_tower_paged_engine.cpp` measured passing a token
gate 16/16. Never cell `(0,0)`, so a per-tensor collapse to the first cell, an
epilogue-folded scalar and a transposed index are each blind to it. The
reference is always computed from the UNPERTURBED weight, so what the case
measures is the error a kernel applying a wrong scale would produce.

### The scale spread, and why it is a ratio

`Fp8BlockScaleSpread(w) = max/min` over the weight's own grid. #1189 pins the
degenerate reading at **exactly 1.0**, and a ratio is what reads 1.0 when every
cell is equal — a variance reads 0.0 — so the issue's own pinned value picks the
statistic.

It answers one question no other instrument in this tree can. A weight whose
`weight_scale_inv` grid has become one number repeated is a per-TENSOR fp8
weight wearing a block-wise grid: it produces plausible tokens, it moves exactly
the same bytes, and it dispatches exactly the same GEMMs, so
`REFERENCE_TIER_LINES`, the dispatch counter and `bytes_per_element` all read
clean on it.

**Three readings, not two, and the third is why.** The counters separate a grid
of ONE cell from a multi-cell grid that reads 1.0. A single-cell grid has no
per-block structure to have lost, and charging it to the suspicious count would
have made the count useless on this tree's own fixtures: 11 of the model
fixture's 13 grids hold one cell, because that config's hidden size is 128 and
one `[128,128]` block covers a whole projection. On
`Qwen/Qwen3.8-27B-FP8` the case does not arise — its narrowest quantized
projection is still several blocks wide — but a probe that reads 11 on a healthy
synthetic model is a probe nobody will believe on a real one.

Two edges are deliberately NOT 1.0. An EMPTY grid returns 1.0 and is counted as
single-cell. A grid whose minimum is not positive returns **infinity**, so a
zero or negative scale is never mistaken for a per-tensor collapse.

**It is a counter, not a refusal.** A checkpoint that legitimately quantized one
narrow projection into a single block would be REFUSED by a rule and is merely
COUNTED by a probe, and #1189 groups this with the dispatch counter and the
bytes-moved assertion — instruments a gate reads. The loud failure is in the
gate, not in the loader: the committed cases assert `collapsed == 0` on a
healthy model and `collapsed == 2` on a collapsed one, and a gate run reads the
same snapshot.

### One snapshot

`ReadFp8BlockStats()` returns `{gemms, scale_grids, single_cell_scale_grids,
collapsed_scale_grids}` in one call, so a reader cannot pair a GEMM count taken
before a forward with a grid count taken after it. `vt::cuda::Fp8BlockScaledStats`
is read the same way and for the same reason. `BlockGemmCount()` is unchanged
and its existing callers are untouched.

The grid is recorded ONCE per weight, on the upload that makes it
device-resident — never per GEMM. On the merged path it is recorded per SHARD
and not per merged operand, because the concatenated grid's spread reads > 1.0
whenever any two shards differ, which is precisely what a per-shard collapse
hides: three collapsed projections concatenate into a grid with three distinct
values.

## Risks

| Risk | Control |
|---|---|
| the bound passes in both regimes, so it bounds nothing | the case asserts BOTH directions per shape per dtype, plus a 4x margin over the grid; mutation M3 measures that widening the bound to 1.0 fails 6 assertions |
| the bound is fitted to one shape and does not hold on another | six shapes spanning decode, ragged N, `block_n != block_k`, 1 to 56 K-tiles; the assertion is per shape, and the margin assertions take the worst and the tightest over all of them |
| the statistic is dominated by a cancelled element rather than by the scales | the 1%-of-max floor, chosen after MEASURING that the unfloored statistic moves by 835x on the clean arm alone |
| the bound fires on the legitimate CUTLASS association difference | 3.3e-6 at the widest shape against `kBoundF32 = 1e-4`; upstream's own `rel_diff < 0.001` for the same pair sits between the two constants |
| the reference and the arm share a helper, so the comparison proves consistency rather than correctness | `RefBlockGemm` is the fixture's independent `double` implementation: a different loop nest, a `double` accumulator and an e4m3 decode unpacked from the bit fields, none of it shared with `src/` |
| an all-zero or degenerate output passes any ratio | a vacuity guard per projection: every output element non-zero |
| the probe reads 1.0 on healthy synthetic weights and is disbelieved | single-cell grids counted apart; the model-level case asserts 11 single-cell, 0 collapsed |
| the model fixture can only discriminate 2 grids, so the model-level probe case is thin | stated in the case, and a dedicated subcase carries the discrimination on a 32-cell `{512, 1024}` grid through `Apply` |
| the new operand assertions refuse a weight the loader legitimately produces | the loader asserts the same two properties itself (`LoadFp8BlockRaw`), so any weight it produces passes; the full `ctest` suite is the control |
| the suite is slow enough to matter under the sanitizers | 4.7 s wall on this box; the widest shape is `M = 8` rather than upstream's 32 for that reason |

## Tests

`tests/vllm/model_executor/models/test_fp8_block_numeric_bound.cpp` — **CPU
tier, runs on every machine**, registered in `tests/CMakeLists.txt` beside its
M4/M6 neighbours and therefore in `ctest`.

- **N1** the per-projection numerical bound. Six shapes x two out-dtypes x two
  arms, each entered through `layers::Fp8BlockLinearMethod::Apply` — the
  production per-projection entry point, the same call
  `ModelRegistry::Forward` makes and the one `test_fp8_block_linear.cpp` G3 pins
  by dispatch count. `clean < bound` and `bumped > bound` per case, then a 4x
  margin on the worst clean and the tightest bumped reading over the whole grid.
- **N2** the scale-variance probe. A genuine grid reads > 1.0; a collapsed one
  reads **exactly** 1.0 (`==`, not an approximation: a tolerance here would
  accept a merely near-uniform grid, which is a different and unasserted claim);
  a single-cell grid reads 1.0 and is counted apart; a non-positive minimum
  reads infinity. Then the counters, twice: one 32-cell projection healthy and
  collapsed through `Apply`, and one whole `ModelRegistry::Forward` healthy and
  collapsed.
- **N3** the split arm's GEMM-boundary refusals, each by name and each with the
  message asserted, plus the well-formed weight served first so the refusals are
  not vacuous.

The lane is verified rather than assumed: `.github/workflows/ci.yml`'s
`build-test-cpu` job runs `ctest --test-dir build --output-on-failure` over the
whole suite, and `sanitize-cpu` runs it again under the sanitizers.
`cuda-fat-build` deliberately does not run `ctest`, so no claim here rests on
it.

## Gates

| Gate | Command |
|---|---|
| this row | `ctest -R test_fp8_block_numeric_bound --output-on-failure` |
| the M4 wiring, unchanged | `ctest -R test_fp8_block_linear --output-on-failure` |
| the M6 merge, unchanged | `ctest -R test_fp8_block_merged --output-on-failure` |
| the M2 reference arm, unchanged | `ctest -R test_ops_matmul_fp8_block_cpu --output-on-failure` |
| the M3 loader, unchanged | `ctest -R test_fp8_block_weight_load --output-on-failure` |
| the host-tier CUDA dispatch decision, unchanged | `ctest -R test_fp8_block_scaled_dispatch` |
| the whole suite, because this edits a header ten call sites include | `ctest --test-dir build` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |

## Evidence

Taken in a linked worktree at `/dev/shm`, off `origin/main` at `21abaf169`, no
`CMAKE_BUILD_TYPE` (matching `build-test-cpu`), no CUDA toolkit and no device on
the host. **Nothing here is a device measurement and nothing here is a speed
measurement.**

**GREEN.** `test_fp8_block_numeric_bound` reports **3 cases, 154 assertions, 0
failed, `Status: SUCCESS!`** in 4.7 s.

**GREEN, whole suite.** `ctest --test-dir build -j 8 --output-on-failure` on the
first merge result reports **100% tests passed, 0 tests failed out of 594**, in
1467 s, with three checkpoint-gated cases skipped
(`test_modelopt_mixed_precision_checkpoint`, `test_voxtral_e2e`,
`test_qwen35_paged_engine`). That is the control this change needs and not a
formality: `dense_fp8_block_gemm.h` is included by every Qwen3.5 dense
translation unit, and the two new operand assertions run on every block-wise
weight the tree builds. `test_cpu_x86_llamacpp_floor` PASSED inside that run and
FAILED in a `scripts/agent-preflight.sh` run taken while another session held
eight busy-loops on this box, which is [#618](https://github.com/mudler/vllm.cpp/issues/618)
behaving exactly as the index describes it -- measured as environmental here
rather than assumed, because the same case passed on the same tree minutes
apart. A later preflight on the second merge result reports **All gates green**.

**The readings the two constants were set from**, printed by a measurement pass
that added a `MESSAGE` beside each assertion and was then restored from tar and
verified by `sha256sum -c`:

| shape | bf16 clean | bf16 x1.10 | f32 clean | f32 x1.10 |
|---|---|---|---|---|
| `1, 128, 128` | 3.769e-3 | 1.0338e-1 | 2.175e-7 | 1.000e-1 |
| `4, 192, 128` | 3.769e-3 | 1.0406e-1 | 2.175e-7 | 1.000e-1 |
| `4, 256, 128` | 3.769e-3 | 1.0410e-1 | 2.175e-7 | 1.000e-1 |
| `3, 192, 256` (64x128) | 3.611e-3 | 7.207e-1 | 1.141e-6 | 7.187e-1 |
| `8, 512, 1024` | 3.816e-3 | 1.8105e0 | 4.072e-6 | 1.8076e0 |
| `8, 512, 7168` | 3.820e-3 | 8.457e-1 | 6.202e-6 | 8.456e-1 |
| **worst / tightest** | **3.820e-3** | **1.0338e-1** | **6.202e-6** | **1.000e-1** |

So the separation is **27x** between the two bf16 regimes and **16000x**
between the two f32 regimes, and the constants sit near the geometric middle of
each. The bf16 margin is 5.2x in each direction. That is the number to argue
with if this bound is ever called too loose or too tight, and it is not
enormous; the f32 margin is.

### Mutation results

Every mutation printed `compile_rc` AND evidence that it applied, because a
mutation that fails to build and a mutation that never applied both read as a
passing test. The harness refuses to run when the target string does not occur
exactly once. The tree was snapshotted to **tar** and restored from tar, never
with `git checkout -- .`, and every restore was verified by `sha256sum -c`
against a digest taken beforehand; the three touched files were `touch`ed after
each restore so `ninja` could not skip the rebuild and leave a mutated binary to
be re-run. The harness lives at a uniquely named path in a private subdirectory,
because the session scratchpad is shared and a replaced running script executes
another session's commands — the failure
[`vt-matmul-fp8-block-cuda.md`](vt-matmul-fp8-block-cuda.md) records.

| # | Mutation | `compile_rc` | Result |
|---|---|---|---|
| M1 | the CPU block GEMM applies **x1.10** to every weight scale (`src/vt/cpu/cpu_ops.cpp`, the mainloop line) | 0 | **RED, 14 assertions**, 1 of 3 cases. This is the acceptance criterion, measured: the bound is RED under a x1.10 scale perturbation and GREEN without it |
| M2 | the same line applies **x1.02** | 0 | **RED, 14 assertions**, the same 14 as M1. The OTHER factor the token gate demonstrably could not see is caught too |
| M3 | `kBoundBf16` widened from `2e-2` to `1.0` | 0 | **RED, 6 assertions.** The two-sided half is what fires; a one-directional bound would have gone green |
| M4 | the resident upload COLLAPSES every grid to its first cell — a per-tensor weight wearing a block-wise grid | 0 | **RED, 14 assertions, 2 of 3 cases.** N2's collapsed counters AND N1's values both catch it |
| M5 | `RecordFp8BlockScaleGrid` compares the spread against `0.0F` instead of `1.0F`, so nothing is ever counted collapsed | 0 | **RED, 2 assertions.** The probe's own accounting |
| M6 | the packed operand's one-byte-per-element check is defanged | 0 | **RED, 1 assertion**, and the assertion COUNT drops 154 -> 152: the `CHECK_THROWS_AS` no longer throws, so the two message assertions after it never run. Without the check the short buffer is read out of bounds and the call returns a value |
| M7 | the scale grid check accepts `<=` the `cdiv` shape, so a FLOOR tiling passes again | 0 | **RED, 2 assertions** |

Two results are worth keeping.

**M2 is the one that was not predicted, and the prediction was wrong in an
interesting direction.** A x1.02 perturbation produces about 2% relative error,
which is `kBoundBf16` exactly, so the bf16 half was expected to be marginal at
best and probably to pass. It does not pass on either half and it does not pass
marginally: the bf16 readings come in at **2.32e-2 to 2.39e-2** against the
`2e-2` bound, because the truncating bf16 store compounds ON TOP of the scale
error rather than being absorbed by it, and the f32 readings come in at
**2.00e-2** against `1e-4`, 200x over. All twelve per-shape assertions fail on
their own and both margin assertions fail beside them. The bound therefore
catches BOTH factors the 2026-08-23 token gate structurally could not, and that
is a measurement rather than a claim.

The honest caveat is that x1.02 is caught by 1.16x on the bf16 half. That is the
edge of what this bound resolves at the model dtype, it is stated so that nobody
reads the M2 row as headroom, and the f32 half is what makes the result robust
rather than lucky.

**M6 shows what the missing check cost.** With it defanged the suite does not
merely fail an assertion — it dereferences past the end of the packed buffer and
carries on. That is the shape of the hole: not a wrong answer that a value
comparison would find, but unowned memory read by the first GEMM.

### The one gate that is red, and why it is not this change

Recorded on the merge result rather than assumed: `build-test-cpu` and both
`sanitize-cpu` legs are red on `origin/main` itself on one case at
`tests/.../test_runner.cpp:1544`/`:1557`
([#1602](https://github.com/mudler/vllm.cpp/issues/1602),
[#1608](https://github.com/mudler/vllm.cpp/issues/1608)),
`test_cpu_x86_llamacpp_floor` is the load-dependent
[#618](https://github.com/mudler/vllm.cpp/issues/618), the two `windows-msvc`
legs are pull-request-only with no `main` baseline, and `agent-record` is red on
`main` with `FileNotFoundError: 'hugo'` from `SiteGuardTests` — a missing runner
binary rather than a record defect. None of them is in a file this change edits.

## Stop conditions

Report `NEEDS_DECISION` rather than proceeding if any of these holds.

- The bound cannot be made to separate the clean arm from the x1.10 arm on some
  shape. Report the numbers rather than shipping a bound that does not bite; a
  bound that passes both ways is the defect this row exists to avoid.
- Making the bound pass requires weakening an existing assertion. Nothing here
  may be made green by deleting or widening a check.
- The probe's degenerate reading has to be something other than exactly 1.0.
  #1189 pins that value, and a different reading is a change to the issue's own
  gate design rather than a detail of this row.

## Outcome

**What was measured.** Seven mutations, in both directions, on a tree restored
from tar and verified by digest after each one. The bound is RED at x1.10 and at
x1.02 and GREEN unperturbed, over six shapes and two output dtypes; the probe
reads exactly 1.0 on a per-tensor collapse and is RED when a collapsing load
path is simulated in product code; the two new operand assertions are each RED
when defanged. The full `ctest` suite is the control on the header ten call
sites include.

**Why each default has the value it has.**

- **`kBoundBf16 = 2e-2`.** The geometric middle of a measured 27x separation,
  and 5.1x above a structural `2^-8` truncation floor that cannot drift up.
- **`kBoundF32 = 1e-4`.** 16x above the worst measured clean reading and three
  orders under the tightest perturbed one; upstream's own `rel_diff < 0.001`
  for this pair sits just above it.
- **A 1%-of-max floor in the denominator.** Measured, not assumed: without it
  the clean f32 reading moves by 835x on one shape for a reason unrelated to
  any scale.
- **x1.10 as the perturbation.** The exact factor a token gate on this family
  was measured passing 16/16.
- **`max/min` rather than a variance.** #1189 pins the degenerate reading at
  exactly 1.0 and only a ratio reads 1.0.
- **A counter rather than a refusal.** A uniform grid is suspicious, not
  illegal, and #1189 groups this instrument with the dispatch counter.
- **The CPU tier.** `vt::MatmulFp8BlockScaled` has a CPU reference arm, so all
  of this runs on `build-test-cpu` on every pull request with no lease.

**What was rejected, and why.**

- **Re-implementing the shape assertion.** It already fires in two places and
  the CUDA TU's own comment says so. A duplicate would have been a second,
  weaker copy.
- **A load-time refusal on a collapsed grid.** It would refuse a legitimate
  checkpoint and it is not what #1189 asks for.
- **Charging single-cell grids to the collapsed count.** It reads 11 on a
  healthy fixture, which is a probe nobody would believe.
- **The unfloored relative statistic.** Measured unusable; the numbers are
  above.
- **Upstream's `M = 32` at the widest shape.** The `double` reference is
  `O(M*N*K)` and `M = 8` keeps the suite at 4.7 s. The device suite runs the
  `M = 32` case where it belongs.
- **Closing #1189.** See below.

**What this does NOT establish.** It is a CPU-tier bound on the CPU reference
arm. It says nothing about the CUTLASS kernel's own numbers — that is
`test_ops_matmul_fp8_block_cuda.cpp` G2/G7 and it needs a lease. It is not a
token gate and does not touch one. It produces no speed number of any kind. And
it bounds the projections in its grid, not a class of shapes.

## Now

`DONE`. The bound, the probe and the split arm's operand assertions are
committed and green, with the mutation evidence above. #1189 stays OPEN.

## Owed

- **[#1189](https://github.com/mudler/vllm.cpp/issues/1189) is NOT closed by
  this row, deliberately.** Its `## Gate design` has all three layers
  established after this change — layer 1 by
  [`gate-qwen38-27b-fp8-block.md`](gate-qwen38-27b-fp8-block.md), layer 2 and the
  scale-variance probe here, the dispatch counter and the bytes-moved assertion
  by the 2026-08-23 gate run and by M4/M6 — but the ISSUE is wider than its gate
  design. It owns six milestones and their `## Owed` sections are live: two of
  the three tile configurations are unexercised on the model path, the
  column-major and TMA-aligned activation-scale layouts are unimplemented, fp16
  output is unsupported, and every other blockwise arm upstream lists is
  unported. Whoever closes it owes a reading of all of that, not of this
  section. Nothing in this change's commit or pull-request body carries a
  closing keyword, because those fire from prose.
- **The bound does not run on a DEVICE.** The x1.10 discrimination is
  established for the CPU reference arm. The CUTLASS arm is compared against the
  same reference on seven shapes under
  [#1437](https://github.com/mudler/vllm.cpp/issues/1437) at `rel_diff < 0.001`,
  which is tighter than `kBoundF32` and would therefore also fail x1.10 — but
  that implication is REASONED here and not measured, and no lane in this
  repository runs a device test. Taking it needs a lease and belongs to
  `VT-MATMUL-FP8-BLOCK-CUDA`.
- **The sm120 complete-scale-block constraint is still discovered at the first
  GEMM, not at `Prepare`.** `RefuseUnrunnableQwen3_5DenseFp8Block` asks whether
  the device has a block-wise arm; it does not ask whether that arm can serve
  each projection's `(N, K)`. A checkpoint with a ragged-N block projection —
  DSV3's `kv_a_proj_with_mqa`, `N = 576` — therefore loads, passes `Prepare`,
  captures a graph and then throws by name at the first GEMM. The message is
  correct and names the dimension; the TIMING is one rung later than it could
  be. Closing it means exposing the arch-specific predicate as a queryable
  capability, which is a `vt` API change on
  `VT-MATMUL-FP8-BLOCK-CUDA`'s surface rather than this row's, and it is left
  unfixed rather than half-fixed. Tracked in
  [#1777](https://github.com/mudler/vllm.cpp/issues/1777).
- **The model-level probe case discriminates only 2 of 13 grids**, because 11 of
  the fixture's grids hold one cell. The dedicated 32-cell subcase carries the
  discrimination and the counts are asserted, so this is bounded rather than
  hidden; a fixture whose hidden size is a multiple of 256 would fix it and
  changing the shared M4/M6 fixture is not this row's scope.
- **No speed axis.** None is measured, claimed or derivable here.
