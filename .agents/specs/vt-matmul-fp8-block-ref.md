# VT-MATMUL-FP8-BLOCK-REF — the block-scaled FP8 GEMM, CPU reference arm

Issue: [#1189](https://github.com/mudler/vllm.cpp/issues/1189), milestone **M2**.
Row: `VT-MATMUL-FP8-BLOCK-REF`.
Pinned oracle: vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), asserted as the HEAD of the local checkout before
any `file:line` below was read.

## Scope

Add one `vt` op with a CPU kernel only:

```c++
void MatmulFp8BlockScaled(Queue& q, Tensor& out,
                          const Tensor& a_fp8, const Tensor& a_scale,
                          const Tensor& b_fp8, const Tensor& b_scale,
                          int block_n, int block_k);
```

`a_fp8` is `[M,K]` i8 carrying raw fp8-e4m3fn bytes, as `vt::QuantFp8Group`
emits. `a_scale` is `[M, cdiv(K, block_k)]` f32, as `vt::QuantFp8Group` emits.
`b_fp8` is `[N,K]` i8, the on-disk weight. `b_scale` is
`[cdiv(N, block_n), cdiv(K, block_k)]` f32, the on-disk `weight_scale_inv`.
`out` is `[M,N]` f32 or bf16.

This is the numerical oracle every later block-FP8 kernel is measured against.
It is milestone M2 of #1189 and it deliberately stops there.

**Out of scope, each owned by another milestone of #1189**: `Fp8BlockWeight`,
the loader rung and the config reader (M3); `layers::Fp8BlockLinearMethod` and
the Qwen3.5 wiring (M4); the mainloop-scaled CUTLASS kernel for `sm_121a` (M5);
merged `gate_up` and QKV (M6). No CUDA arm lands here: M5 owns it, and a
device-free host cannot gate one.

## Which implementation actually runs, and why it does not change the answer

M1 established that upstream's Triton source is not what executes for the
activation quant, and that the two arms disagree in polarity. The same question
has to be asked here before any arithmetic is mirrored, and the executing chain
was read end to end rather than inferred from the Python.

The dispatch decision, in order:

| Step | Where |
|---|---|
| CUDA block-FP8 kernel priority list: FlashInfer, DeepGEMM, **CUTLASS**, Marlin, Triton, Humming | `vllm/model_executor/kernels/linear/__init__.py:355-377` |
| DeepGEMM is auto-disabled for `qwen3_5_text` on device-capability family 120 | `vllm/utils/deep_gemm.py:27-46` |
| the selected kernel's apply: `ops.cutlass_scaled_mm(A, B.T, scale_a=As, scale_b=Bs.T)` | `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:312-326` |
| `cutlass_scaled_mm` routes every `sm >= 120` device to the sm120 entry | `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:220-226` |
| the sm120 entry hands `cutlass_scaled_mm_blockwise_sm120_fp8` to the shared dispatcher | `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu:13-20` |
| the dispatcher takes the blockwise branch whenever a scale is not per-tensor or per-row, requires both scales **f32** and 2-D, checks the shapes with **`ceil_div`**, and refuses a bias | `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:15-18,39-55` |
| the blockwise kernel's accumulator and scale element type are both `float`, and the two scale pointers are **mainloop** arguments | `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh:56-58,218-235` |
| CUTLASS 4.5.0, the mainloop line itself: `accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i) * tCrScaleBViewAsC(i)` | `include/cutlass/gemm/collective/sm120_mma_tma_blockwise_scaling.hpp:714-717` |

So the executing kernel is CUTLASS, and unlike M1 it **agrees** with the
reference in both placement and polarity: a per-K-block partial product is
formed in an f32 accumulator and **multiplied** by `a_s * b_s` before it is
added to the running sum. Mirroring `native_w8a8_block_matmul` is therefore
mirroring the executing arm, and that is a measurement rather than an
assumption: upstream compares the CUTLASS arm against exactly this reference at
`rel_diff < 0.001` (`tests/kernels/quantization/test_block_fp8.py:194-200`), a
gate no polarity or placement error survives.

One difference is real and is not a divergence. The reference forms the scale
product first, `s = As_tiles[i] * Bs[j][i]` then `partial * s`
(`tests/kernels/quant_utils.py:150-151`); CUTLASS associates left to right,
`(partial * a_s) * b_s` (`sm120_mma_tma_blockwise_scaling.hpp:715`). The two
differ by at most one f32 ULP per K-block and upstream's own 1e-3 relative gate
is what admits it. We mirror the reference's association, because this op **is**
the reference port and M5's kernel will be measured against it.

## The constraint that decides correctness

**The scales apply in the mainloop, once per K-block, into an f32 accumulator —
not in the epilogue.** Written out (`fp8_utils.py:826-836`, which is the Triton
arm spelling the same structure the CUTLASS mainloop implements):

```text
accumulator = zeros(f32)
for k_block:
    a_s = a_scale[m, k_block]          per token, per K-group
    b_s = b_scale[n_block, k_block]    per (N-block, K-block)
    accumulator += dot(a_tile, b_tile) * a_s * b_s
```

Our existing per-tensor FP8 path folds one scalar `alpha` into the epilogue
(`vt::MatmulFp8Cutlass`, `src/vt/cpu/cpu_ops.cpp` `MatmulFp8CutlassKernel`:
`alpha * acc` after the whole K reduction). **An epilogue-only application
cannot express a per-K-block scale at all**: it has exactly one degree of
freedom per output element, and the block scheme has `cdiv(K, block_k)` of them.
This is a correctness constraint, not an optimisation choice, and it is the
reason M2 is a new op rather than a parameter of `kMatmulFp8Cutlass`.

G4 is the instrument for it, and it is constructed so that no epilogue-folded
alpha can pass: it builds a case whose two K-blocks carry different scales and
whose correct output is not `alpha * (plain fp8 GEMM)` for any `alpha`.

## Upstream anchors

| What | Where |
|---|---|
| the reference this op ports, whole | `tests/kernels/quant_utils.py:91-154` |
| the fp32 compute type and the `.to(output_dtype)` store | `quant_utils.py:98,109-110,153` |
| `ceil` tiling of both dimensions | `quant_utils.py:123-124` |
| the `Bs` shape assertions, `n_tiles`/`k_tiles` | `quant_utils.py:125-126` |
| the `As` last-dimension `ceil` assertion | `quant_utils.py:115-116` |
| the mainloop accumulate, scales multiplied per K-block | `quant_utils.py:145-151` |
| the production wrapper's shape contract, `cdiv` on N and K | `vllm/model_executor/layers/quantization/utils/fp8_utils.py:928-936` |
| the Triton mainloop spelling the same structure | `fp8_utils.py:826-836` |
| the ported test, its grid, its tolerance | `tests/kernels/quantization/test_block_fp8.py:48-55,123-153` |
| the ragged-shape case upstream calls out by name (DSV3 `kv_a_proj_with_mqa`) | `test_block_fp8.py:156-200` |
| the activation quant that produces `a_fp8`/`a_scale` | `.agents/specs/vt-quant-fp8-group.md`, landed `ad5f175e7` |

## Design

### Numerics

```text
for each output element (m, n):
    acc = 0                                        f32
    for kt in [0, cdiv(K, block_k)):
        part = 0                                   f32, a SEPARATE accumulator
        for k in the k-tile (ragged final tile is short):
            part += f32(a_fp8[m,k]) * f32(b_fp8[n,k])
        acc += part * (a_scale[m, kt] * b_scale[n / block_n, kt])
    out[m, n] = acc                                stored to out's dtype
```

`part` is a separate register that is scaled and then folded into `acc`. That is
the whole point: collapsing it into `acc` is what an epilogue alpha does, and it
is unrepresentable.

`n / block_n` is integer division of the **output column index**, mirroring
`offs_bsn = offs_bn // group_n` (`fp8_utils.py:823`) and the reference's tiling
at `quant_utils.py:131-143`. `block_n` and `block_k` are validated positive
before either divides anything, because `x % 0` and `x / 0` are undefined
behaviour and a zero must refuse rather than trap.

### Ragged edges

`cdiv`, not floor, on both dimensions. Upstream's production wrapper asserts
`triton.cdiv(N, block_n) == Bs.shape[0]` and
`triton.cdiv(K, block_k) == Bs.shape[1]` (`fp8_utils.py:935-936`), so a final
short block is legal and must work. The final K-tile runs to `K`, not to
`(kt+1)*block_k`; the final N-block is short and its scale row still exists.
`N=576` (`4*128 + 64`) and `K=3884` (`30*128 + 44`) are the shapes from
upstream's own grid that expose an integer-division bug, and both are in the
ported grid, separately and together.

### Memory format

`out` is f32 or bf16 and nothing here is f32 by default. The accumulator is f32
because upstream's `ElementAccumulator` is `float`
(`scaled_mm_blockwise_sm120_fp8_dispatch.cuh:56`) and the reference's
`compute_type` is `torch.float32` (`quant_utils.py:98`); the **store** rounds to
whatever `out` carries, exactly as `.to(output_dtype)` does
(`quant_utils.py:153`). Both scale tensors are f32 and the op refuses any other
dtype, because upstream refuses it too
(`scaled_mm_helper.hpp:15-18`). This op therefore widens nothing: it is the same
polarity `.agents/porting.md` requires.

### Structure

`vt::MatmulNvfp4Fp4` (`include/vt/ops.h:1477`) is the closest existing
signature — a block-scaled GEMM taking two packed operands and two scale
streams — and `MatmulFp8BlockScaled` follows its shape, with the scalar `alpha`
replaced by the block geometry it cannot express. `OpId::kMatmulFp8BlockScaled`
is appended before `kCount`, the additive convention documented at
`include/vt/ops.h:363-368`, so no existing op's id shifts.

The kernel is a **correctness reference, not a performance path**, in the same
sense as `MatmulFp8CutlassKernel` beside it: a naive nest that makes the
block-FP8 seam resolvable on a CPU queue so M3 and M4 can be gated without a
GPU. It makes no speed claim.

## Risks

| Risk | Control |
|---|---|
| a later reader folds the scales into the epilogue, "simplifying" the inner accumulator away | the prose above and in the header, plus G4, which is unpassable by any single-alpha form |
| a ragged N or K uses floor tiling and silently drops or misindexes a block | `cdiv` everywhere; G2 runs `N=576`, `K=3884` separately and together, and G5 refuses a floor-sized scale tensor by name |
| the `n / block_n` index is computed from the tile index rather than the column, which agrees for round N and diverges for ragged N | G2's `N=576` case, where the two disagree in the final block |
| an all-zero output passes every value comparison | G2, G3 and G4 each carry a `nonzero == numel` vacuity guard |
| a scale dtype silently widens or narrows | G5 refuses a non-f32 scale by name |
| the test's reference is the implementation rewritten, so agreement proves nothing | the G2/G3 reference accumulates in `double` with a different loop nest and an exhaustively derived fp8 decode table, and its bound is an f32 forward-error bound rather than a fudge factor |
| `block_n`/`block_k` of zero divides by zero | validated positive before use; G5 asserts the refusal |
| nothing dispatches the op, so it is dead on arrival | acknowledged and named under `## Owed`: this is the staged-slice exception of `.agents/reachability.md`, and M4 owns the wiring |

## Tests

`tests/vt/test_ops_matmul_fp8_block_cpu.cpp`, registered in
`tests/CMakeLists.txt`.

- **G1** the registration itself: `OpRegistered(kMatmulFp8BlockScaled, kCPU)`,
  and `OpName` is not `unknown`. A refusal case cannot stand in for this — M1
  measured a deleted registration passing every refusal test it had.
- **G2** the ported upstream case, `test_block_fp8.py:123-153`, against an
  independently written `double` reference, carrying upstream's own
  `rel_diff < 0.001` criterion verbatim **and** a tighter per-element f32
  forward-error bound, because our arm is the reference rather than a kernel
  measured against it. Vacuity guard.
- **G3** the ragged grid, run as part of G2's table and called out here because
  it is the reason the grid is what it is: `N=576`, `K=3884`, and the pair
  together, plus upstream's dedicated DSV3 case `M=32, N=576, K=7168`
  (`test_block_fp8.py:156-200`).
- **G4** **the mainloop constraint, made unpassable by an epilogue**. Two
  K-blocks with deliberately different scales and a hand-computed expected
  output; then the same operands with the two K-block scales *swapped*, which
  leaves every per-tensor summary of the scales identical and changes the
  correct answer. A kernel that folds one alpha produces the same output for
  both and fails. Vacuity guard.
- **G5** the refusals, each by name: a non-f32 scale, a `b_scale` sized by
  floor instead of `cdiv` on N and on K, an `a_scale` with the wrong number of
  groups or rows, a zero or negative `block_n`/`block_k`, a rank mismatch, a
  non-contiguous operand, a device mismatch, a non-i8 packed operand, and an
  `out` that is neither f32 nor bf16.
- **G6** the M1 seam: `vt::QuantFp8Group` feeding `vt::MatmulFp8BlockScaled` on
  a CPU queue, end to end, which is the pair a block-FP8 linear method will run.
  This is a *composition* test, not a reachability claim; nothing in production
  calls it yet and `## Owed` says so.

### The adaptation of the upstream grid, and why it is unavoidable

Upstream's grid is `itertools.product(M, N, K, ...)` with `M=[1,7,8,83,4096]`,
`N=[128,512,576,7168,13824]`, `K=[256,3884,4096,13824,16384]`
(`test_block_fp8.py:48-55`) — 125 combinations, the largest of which is
`4096x13824x16384`, about 9.3e11 multiply-accumulates. That is a GPU grid. A
naive CPU reference nest cannot run it, and running it twice (op and reference)
is worse.

The adaptation preserves the **axes**, not the product: every value of every
axis appears at least once, the ragged values appear separately and together,
and the total is under 1e9 MACs. Which values are paired with which is the only
thing dropped, and the parameters, dtypes, tolerance and failure criterion are
preserved exactly. The grid is written out in the test with this reasoning
beside it.

## Gates

| Gate | Command |
|---|---|
| focused | `ctest -R test_ops_matmul_fp8_block_cpu --output-on-failure` |
| op provider totality | `ctest -R test_op_provider` |
| the M1 sibling, unchanged | `ctest -R test_ops_quant_fp8_group_cpu` |
| the per-tensor sibling, unchanged | `ctest -R test_ops_fp8_cpu` |
| record | `scripts/agent-preflight.sh --fail-on-skip` |

No GPU lease is taken and none is needed: the op has a CPU arm only.

## Owed

- **Nothing reaches this op.** `vt::MatmulFp8BlockScaled` is dispatched by no
  production entry point at this merge commit: `include/vllm.h` does not expose
  it, no loader builds an `Fp8BlockWeight`, and `ModelRegistry::Forward` has no
  block-FP8 linear method to call it from. The wiring is owned by #1189
  milestone M4 (`layers::Fp8BlockLinearMethod` and the Qwen3.5 dense forward),
  which needs M3 first. This is the staged-slice exception of
  `.agents/reachability.md`, named here, in the commit body, and in the pull
  request body.
- **The CUDA arm.** There is none. Milestone M5 owns the mainloop-scaled CUTLASS
  kernel for `sm_121a`, needs a GPU, and will be measured against this op.
- **The `swap_ab` path.** Upstream's sm120 blockwise dispatch swaps the operands
  for some shapes (`scaled_mm_blockwise_sm120_fp8_dispatch.cuh:221-235`), which
  changes the scale-pointer assignment but not the mathematical result. It is a
  kernel-scheduling concern with no reference-arm meaning. Owed by M5.
- **The column-major and TMA-aligned activation-scale layouts**
  (`fp8_utils.py:610-628`), which the CUTLASS kernel reads and this reference
  does not. Already owed by M5 in `.agents/specs/vt-quant-fp8-group.md`; repeated
  here because M2 is the first op that could have consumed them.
- **`bias`.** Upstream refuses a bias on the blockwise path outright
  (`scaled_mm_helper.hpp:55`). This op takes none, which mirrors the refusal
  rather than deferring a feature.

## Stop conditions

Stop and report `NEEDS_DECISION` if any of the following holds.

- The pinned oracle's checkout is not at
  `5559679229bc961848b121ccdeaa8fa5d79bec98`. Every anchor above was read at
  that revision, asserted before the first read.
- G4 cannot be made to fail an epilogue-folded implementation. If a single alpha
  can reproduce the mainloop result on the constructed case, the case is wrong
  and the constraint is untested, and widening the case is not the fix.
- The op cannot express upstream's contract without a parameter that no caller
  in this tree passes.

Stop and report `NEEDS_CONTEXT` if the work requires a GPU lease. The row is
scoped so that it does not.

## Evidence

Taken on the merged tree, `origin/main` at `2d26da5a1`.

**RED.** With the test present and the implementation reversed out — the reverse
of the implementation diff applied to the four source files, `git diff --stat`
printing `4 files changed, 200 deletions(-)` so the revert is proven to have
landed — the focused build fails, `compile_rc=1`, with 29 errors: 23
``'MatmulFp8BlockScaled' is not a member of 'vt'`` and 6
``'kMatmulFp8BlockScaled' is not a member of 'vt::OpId'``. Restoring the four
files returns `git status --porcelain` to empty and the build to green.

**GREEN.** `test_ops_matmul_fp8_block_cpu` reports 6 cases, 80 assertions, 0
failed, in 3.5 s under `ctest`. Per block, each run through a `-tc` prefix
filter that contains no comma, because doctest splits `-tc` on commas and a
name that contains one yields `0 cases ran` under a `SUCCESS!` banner:

| Block | Cases | Assertions |
|---|---:|---:|
| G1 registration and name | 1 | 2 |
| G2 the ported grid | 1 | 22 |
| G3 the ragged edges | 1 | 13 |
| G4 the mainloop constraint | 1 | 14 |
| G5 the refusals | 1 | 25 |
| G6 the M1 seam | 1 | 4 |
| **sum** | **6** | **80** |

The buckets sum to the whole-run count, so no block is silently empty and no
filter selected nothing.

The other declared gates on the same tree: `test_op_provider` passed,
`test_ops_quant_fp8_group_cpu` passed (M1 unchanged),
`test_ops_fp8_cpu` passed (the per-tensor sibling unchanged).
`scripts/agent-preflight.sh --fail-on-skip` reports **All gates green** with 80
`ok` results, no `FAIL` and no `SKIP` before the verdict line.
`test_cpu_x86_llamacpp_floor` (#618) passed on this run rather than reporting
`NO_QUIET_WINDOW`, so this row needed no pristine-baseline reproduction.

### The ragged edge, specifically

`N=576` is `4*128 + 64` and `K=3884` is `30*128 + 44`, the two non-round shapes
from upstream's own grid (`test_block_fp8.py:49-50`). G3 runs each separately
and both together, plus upstream's dedicated DSV3 `kv_a_proj_with_mqa` case
`M=32, N=576, K=7168` (`test_block_fp8.py:156-200`). This is load-bearing and
measured, not asserted: mutation M2 (floor `k_tiles`) and mutation M3 (a floor
N-block index) each leave **G2 entirely green** and fail only G3, because every
`N` and every `K` in G2 is a multiple of 128. A grid of round shapes passes
while being wrong.

### Mutation results

Every mutation printed `compile_rc` **and** `git diff --stat` before the run,
because a mutation that fails to build and a mutation that never applied both
read as a passing test, and M1 of #1189 hit each of those once. Each was
restored with `git checkout --` and the restore verified.

| Mutation | `compile_rc` | Result |
|---|---|---|
| the scales folded into ONE epilogue alpha, `part` collapsed into `acc` | 0 | 4 of 6 cases fail, 26 of 80 assertions. G4's `got != got_swapped` fires, which is the epilogue signature exactly |
| `k_tiles = k / block_k`, floor instead of cdiv, in the kernel | 0 | **only G3 fails**, 4 assertions. G2 stays green: every K in it is a multiple of 128 |
| the b_scale row index forced to 0 | **1** | proves nothing: `-Werror=unused-parameter` on `block_n` |
| the same defect with the parameter kept live: `nb = min(col / block_n, n / block_n - 1)`, a floor tile count that makes a short final N-block reuse the previous scale row | 0 | 3 of 6 cases fail, 13 assertions: G3, G4, G6. **G2 stays green**, for the same reason |
| `a_scale` dropped from the scale product | 0 | 4 of 6 cases fail, 24 assertions |
| the CPU registration deleted, the kernel symbol kept live | 0 | all 6 cases fail after 7 assertions. G1's `REQUIRE(OpRegistered(...))` is what catches it |
| `n_tiles = n / block_n`, floor, in the wrapper's validation | 0 | 4 of 6 cases fail. G5's floor-sized `b_scale` refusal stops firing and G3's well-formed ragged calls are refused instead |
| the `block_n > 0 && block_k > 0` check removed | 0 | **SIGFPE, core dumped, `run_rc=136`**. The check converts undefined behaviour into a named refusal |
| the `a_scale` shape check widened to a rank check | 0 | G5's narrow and tall `a_scale` refusals both stop firing, 2 assertions |
| the store zeroed, `acc * 0.0F` | 0 | 4 of 6 cases fail, 29 assertions, including G4's two vacuity guards — the guards are live |

Three results are worth keeping.

**The epilogue mutation is the row's whole claim, and it is now measured.**
Folding the per-block scales into a single alpha compiles, runs, and produces the
identical number for a scale tensor and for that same tensor with its two
K-block entries swapped. G4 is the only block that can see that, and it does.

**A grid of round shapes is blind to the ragged bug in both directions.** Two
different floor-vs-ceil mutations — one on the K tiling in the kernel, one on the
N-block index — left G2 completely green and were caught only by G3. That is the
argument for keeping `N=576` and `K=3884` in the grid rather than the shapes the
target checkpoint happens to use, which are all multiples of 128.

**A refusal removed can crash rather than report.** With the block-size
positivity check deleted, the process took SIGFPE on integer division and doctest
had already printed `assertions: 52 | 52 passed | 0 failed` before it died. The
run's exit status is the verdict; its printed summary is not.

### A false red, recorded because it cost a cycle

Mid-run, a direct invocation of the test binary reproduced the zeroed-store
mutation's exact signature — 4 failed cases, 29 failed assertions — against a
tree whose sources were clean. The binary was the one the mutation harness had
last linked; the whole-tree build that followed had not reached the test target
before the run. Nothing was wrong with the code. The control is the one
`.agents/verification.md` already names: rebuild explicitly, require
`ninja: no work to do`, and check that the executable's mtime is newer than
every source it links, before believing any verdict about it.
