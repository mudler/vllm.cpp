# PERF-FP8-SMALL-M-DISPATCH — the per-tensor FP8 CUTLASS GEMM gets vLLM's small-M tile ladder back

Row: `KERNEL-GEMM-FP8` — the owning kernel-family row
([`.agents/kernel-matrix.md`](../kernel-matrix.md)). `PERF-FP8-SMALL-M-DISPATCH`
is this work's branch name, not a matrix row, exactly as
`PERF-FP8-ALPHA-FOLD` is for [#402](https://github.com/mudler/vllm.cpp/issues/402).

Issues: [#1866](https://github.com/mudler/vllm.cpp/issues/1866) (primary).
Context: [#1857](https://github.com/mudler/vllm.cpp/issues/1857) (the corrected
kernel head-to-head that attributed +3.04 ms/step to the FP8 tower).

Base SHA: `1724be38e`

Upstream pin: [`.agents/upstream-sync.md`](../upstream-sync.md),
vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98`. Every `file:line` below was
read at that revision with `git show`, in a checkout that is never modified.

Why `KERNEL-GEMM-FP8` and not `KV-FP8`: the 208 modules this is about are the
checkpoint's static per-tensor FP8 **linear projections**, whose kernel family
is `KERNEL-GEMM-FP8` ("FP8/INT8 scaled-mm plus static activation quant"). That
row already owns `src/vt/cuda/cuda_matmul_fp8_cutlass.cu`, the cuBLASLt fp8
lane and the fp8 plan cache, and already carries the two sibling levers
(`PERF-FP8-ALPHA-FOLD`, `FIX-FP8-PLAN-CAPTURE`). `KV-FP8` is the KV-cache
dtype and touches none of this.

## Now

The port is written and CPU-gated; the CUDA compile and every device number are
`## Owed` and operator-run. No speed number is claimed here.

This section states no lifecycle token, because this change moves none. The
owning row `KERNEL-GEMM-FP8` stays `ANCHOR-BACKFILL`, exactly as it was — a
sub-lever landing under it is not a state transition, and writing `ACTIVE` here
would put a second, unowned lifecycle value beside the matrix's.

## Scope

`E1` Port the two sm120 FP8 configs this tree deliberately omitted —
`sm120_fp8_config_M16` and `sm120_fp8_config_M32`, each with its custom
CUTLASS `EpilogueTile` — into `src/vt/cuda/cuda_matmul_fp8_cutlass.cu`, and
extend `Fp8GemmDispatch` from its two-way ladder to upstream's four-way one.

`E2` Move the ladder itself into a CUDA-free header,
`src/vt/cuda/fp8_per_tensor_dispatch.h`, so the decision that selects a tile is
red-first testable and mutation-provable on a host with no GPU and no `nvcc` —
the arrangement `fp8_block_scaled_dispatch.h` already uses for the *blockwise*
FP8 sibling, for the same stated reason.

`E3` `VT_FP8_CUTLASS_SMALL_M=0` restores the previous two-way ladder as a
same-binary A/B arm, documented in `docs/ENVIRONMENT.md` in this change.

`E4` Correct the record this change falsifies. Three places in the tree say our
cuBLASLt fp8 lane is "the native equivalent of vLLM's cuBLASLt fp8 dense path
(the `nvjet_sm121_qqtst_*` kernels)". At the pin vLLM does not use cuBLASLt for
this GEMM at all, and #1857's artifact-verified profile measured our lane
resolving to `sm89_xmma_gemm_e4m3f32_e4m3f32_f32_tn_n_tilesize32x64x64`.

`E5` Under the existing `VT_GEMM_ALGO_LOG=1` diagnostic, enumerate and print
the cuBLASLt heuristic's **candidate list** for each fp8 plan, not only the
selected algo. This is the instrument that decides the remaining hypothesis
(below) in one operator run instead of another round trip. Default OFF, so the
shipped path is byte-identical with the flag unset.

`E6` **Change `scripts/check-gemv-invocation-consistency.py`'s A(2)
invariant**, because `E5` adds a second `requestedAlgoCount` argument and A(2)
as written cannot express one. This is a semantic checker change and is scoped
here rather than discovered in the diff, as
[`CLAUDE.md`](../../CLAUDE.md) `## Changing the rules or a checker` requires.

What the new rule requires, at every `requestedAlgoCount` site in
`src/vt/cuda/cuda_matmul.cu`: the argument must be a name on the explicit
`ALGO_POLICY_NAMES` allowlist in the checker, **and** that name must be
declared `constexpr int <name> = ...` in the same file. Both conditions, not
either. Adding a policy constant therefore costs a deliberate edit to the
checker itself.

It is a **STRENGTHENING, not a widening.** The old rule rejected only a bare
numeric literal, so *any* identifier passed — including a runtime variable
holding a swept algo count — while the OK line still claimed every site routed
through `kGemvHeuristicAlgos`. The new rule admits exactly two names and
demands a compile-time declaration for each.

The direction is measured rather than read, by two mutations on
`test_check_gemv_invocation_consistency`'s 21 cases, green 21/21 on the shipped
tree:

| mutation | red |
|---|---:|
| `unknown_algo_count_sites()` neutered to report no site — the literal "accept any identifier" | **2/21**, both real assertion failures |
| the whole A(2) body rolled back to `a2c7dca45^` with the new tests kept | **6/21** |

Recorded because the fresh review reported 8/21 for the first mutation and that
number does NOT reproduce here; the two above were rerun with `__pycache__`
cleared and the checker restored byte-identical (`md5sum` unchanged, `git
status` clean) each time. The added clause carries real detection weight and
more than one case sees it, which is the claim this bullet needs; 8 is not the
count.

NOT in scope, and deliberately: flipping `VT_DENSE_CUBLASLT_FP8` to default
OFF, and adding a measured cuBLASLt algo sweep. Both are decisions the
measurement this change enables should make, and neither can be made here —
see `## Owed`.

## Upstream chain

The GEMM vLLM actually runs for a per-tensor static FP8 linear on GB10, top to
bottom at the pin:

1. `vllm/model_executor/kernels/linear/__init__.py:325-334` — the CUDA FP8
   backend order is Marlin -> FlashInfer (sm100+) -> **Cutlass** ->
   PerTensorTorch -> ChannelWiseTorch -> Humming.
2. `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:265` —
   `ops.cutlass_scaled_mm(A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias)`.
   `As`/`Bs` are device tensors, not host floats.
3. `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:222-225` —
   `if (version_num >= 120) cutlass_scaled_mm_sm120(...)`. `sm_121a` reports
   121, so this is the branch GB10 takes.
4. `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh:143-176`
   — `cutlass_gemm_sm120_fp8_dispatch`, the **four-way M ladder**:

   | M | config | TileShape | EpilogueTile | schedule |
   |---|---|---|---|---|
   | `M <= 16` | `sm120_fp8_config_M16` (`:127-138`) | `Shape<_16,_64,_128>` | `Shape<_16,_32>` | Pingpong |
   | `M <= 32` | `sm120_fp8_config_M32` (`:112-123`) | `Shape<_32,_64,_128>` | `Shape<_32,_32>` | Pingpong |
   | `M <= 256` | `sm120_fp8_config_M64` (`:94-108`) | `Shape<_64,_64,_128>` | auto | Pingpong |
   | else | `sm120_fp8_config_default` (`:81-90`) | `Shape<_128,_128,_128>` | auto | `KernelScheduleAuto` |

   The two small-M configs are built through a separate wrapper,
   `cutlass_3x_gemm_sm120_custom` (`:18-77`), whose only difference from the
   plain `cutlass_3x_gemm_sm120` is that it passes an explicit `EpilogueTile`
   into `CollectiveBuilder` where the plain one passes `EpilogueTileAuto`.

**vLLM never queries a cuBLASLt heuristic.**
`git grep -n "cublasLt\|nvjet\|AlgoGetHeuristic" 5559679229 -- csrc vllm`
returns nothing. The only cuBLAS at the pin is plain `cublasHgemm` /
`cublasGemmEx` in the GPTQ kernels. Any cuBLASLt selection upstream reaches is
inside `torch._scaled_mm`, on the PerTensorTorch backend, which is *fourth* in
the dispatch order and not what a Cutlass-capable device takes.

**Upstream runs no FP8 algorithm sweep either.**
`vllm/model_executor/warmup/kernel_warmup.py:189` `flashinfer_autotune` is
called from `:120`, itself from `vllm/v1/worker/gpu_worker.py:704`, and it
tunes **FlashInfer** ops with one dummy run at
`scheduler_config.max_num_batched_tokens` (`:238-242`). The only explicit
small-M sweep at the pin is the **bf16 router** GEMM (`:47-72`,
`_LL_BF16_WARMUP_M_RANGE = range(1, 17)`). The FP8 tower's small-M behaviour
upstream is the static tile ladder above, not a measurement.

That is the finding that redirects #1866. The issue asked us to mirror
SGLang's `[AutoTuner]: Tuning fp8_gemm` sweep. SGLang is not the mirror source
where vLLM implements the path, and vLLM implements this one — with a ladder we
ported and then truncated.

## Our baseline

`src/vt/cuda/cuda_matmul_fp8_cutlass.cu` is a 1:1 lift of
`cutlass_scaled_mm_sm120_fp8`, and its own comment records the omission
verbatim:

```
// M>256: KernelScheduleAuto, EpilogueScheduleAuto, 128x128x128 (sm120_fp8_config
// _default). M<=256: KernelTmaWarpSpecializedPingpong, 64x64x128 (config_M64 —
// "SM120 Cooperative kernel requires Tile M >= 128; for smaller tiles use
// Pingpong"). vLLM's M16/M32 custom-EpilogueTile refinements are perf-only for
// tiny M and are covered correctly (predicated) by the M64 pingpong tile.
```

"Perf-only for tiny M" is true and is the whole problem: **tiny M is decode.**
`Fp8GemmDispatch` sends every M from 1 to 256 to the 64x64x128 tile, so the
spec-verify batch of 9 query rows computes a 64-row tile and throws 55 of them
away, where vLLM computes a 16-row tile with a 16x32 epilogue.

The second half of the baseline is which arm runs at all.
`include/vllm/model_executor/models/dense_fp8_gemm.h` routes the fp8 dense
projections through `vt::MatmulFp8CublasLt` by default
(`DenseCublasLtFp8Enabled()`, `VT_DENSE_CUBLASLT_FP8` default ON), and the
CUTLASS arm is reached on `VT_DENSE_CUBLASLT_FP8=0` or when cuBLASLt reports no
fp8 heuristic for a shape. The justification recorded for that default is that
cuBLASLt gives us "vLLM's measured-FASTER `nvjet_sm121_qqtst` fp8 kernels".

**That justification is refuted.** #1857's build7 profile — the artifact-verified
build (`-DVLLM_CPP_CUTLASS_FETCH=ON`, FA2 manifest `[121a]`) — measured our
default fp8 tower running

| ms/step | calls/step | avg us | kernel |
|---:|---:|---:|---|
| 26.92 | 96.8 | 278.3 | `sm89_xmma_gemm_e4m3f32_e4m3f32_f32_tn_n_tilesize32x64x64_stage5...` |
| 12.91 | 64.6 | 199.8 | `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5...` |

= 39.90 ms/step against SGLang's 36.86 for the same tower, **+3.04 ms/step,
64% of the +4.72 ms/step (+4.1%) step delta**. Not one `nvjet` kernel. Both the
f32-D and the bf16-D arms take the sm89 family.

So the default arm's premise is false, and the alternative arm was ported
without the configs the decode regime needs. Neither of those is an
autotuner's absence.

## Port map

| upstream (at `5559679229`) | ours |
|---|---|
| `scaled_mm_sm120_fp8_dispatch.cuh:18-77` `cutlass_3x_gemm_sm120_custom` | the `EpilogueTile` member added to `Fp8GemmSm120`'s `Config`, defaulted to `EpilogueTileAuto` |
| `:127-138` `sm120_fp8_config_M16` | `sm120_fp8_config_M16` |
| `:112-123` `sm120_fp8_config_M32` | `sm120_fp8_config_M32` |
| `:155-179` the four-way `if` ladder | `Fp8Sm120ConfigForM` in `src/vt/cuda/fp8_per_tensor_dispatch.h`, consumed by `Fp8GemmDispatch` |

Two deviations, both pre-existing and unchanged by this port:

- We fold the two per-tensor scales into one device `alpha` scalar and use
  CUTLASS's default `LinearCombination` epilogue, where upstream applies a
  `ScaledEpilogue` EVT. For per-tensor (scalar) scales the two collapse to the
  same accumulator multiply; the file's header records this.
- Our `ElementC` is `OutType` where upstream's is `void`. Untouched here.

The env gate `VT_FP8_CUTLASS_SMALL_M` has no upstream counterpart, and it is
not pretending to: it is the rollback arm this tree requires of an unmeasured
kernel-selection change, the same role `VT_MARLIN_DENSE=0` and
`VT_FP8_PLAN_CACHE=0` play.

## Tests to port

Upstream has no unit test for the ladder itself — it is a compile-time
`if`-chain inside a `.cuh`, exercised only through
`tests/kernels/quantization/test_cutlass_scaled_mm.py`'s shape sweep, which
needs a device. So the ported test is the *decision*, in the shape this tree
already uses for the blockwise sibling:

`tests/vt/test_fp8_per_tensor_dispatch.cpp` pins `Fp8Sm120ConfigForM` against
the upstream ladder at every boundary and one value either side of it (0, 1, 9,
16, 17, 32, 33, 256, 257), pins the rollback predicate's parse table, pins that
the rollback arm reproduces the pre-change two-way ladder exactly, and pins the
config names that the diagnostic prints.

The device-side numerical case — the same shape computed under the M16, M32 and
M64 configs must agree — is `## Owed`. It belongs in
`tests/vt/test_ops_fp8_cutlass.cpp` beside the existing cached-vs-fresh
byte-exactness case, and it needs a GPU.

## Gates

CPU tier, runnable on any host and run for this change:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4 --target test_fp8_per_tensor_dispatch
./build/tests/vt/test_fp8_per_tensor_dispatch
python3 scripts/check-gemv-invocation-consistency.py
python3 scripts/check-env-doc.py
python3 scripts/check-agent-record.py
scripts/agent-preflight.sh --staged
```

The first CUDA compile is NOT owed after all, and CI discharged it — for the
shipped translation unit, on a head that is not this one. Said exactly, because
the earlier wording said "green on this branch's head" and that was never true
of the head it was written on:

- The `cuda-fat-build` job configures `-DVLLM_CPP_CUTLASS_FETCH=ON` with
  `-DVLLM_CPP_CUDA_ARCHITECTURES='80;86;87;89;90a;100a;103a;110;120a;121a'`
  and builds the `vllm` target, so it compiles this TU with CUTLASS present for
  the two arches the fp8 gate covers.
- That job is `success` on head `d9bf525c0`, in run `32802716762`.
- **That run's own conclusion is `cancelled`, not `success`**, and this tree
  reads a cancelled run as a FAIL. The green cited here is therefore the ONE
  job and never the run. What was cancelled is `sanitize-cpu
  (address,undefined)`, killed at 2h20m by this branch's own next push, plus
  the `baseline-summary` job that waits on it.
- `git diff --stat d9bf525c0..HEAD` changes no compiled CUDA. It touches
  `src/vt/cuda/` only in comment lines.
- The head's OWN `cuda-fat-build` is PENDING at the time this is written.
  Pending is not green, and this row does not claim it as one.

Under those four readings the risk this row named largest is discharged for the
shipped TU: the M16/M32 `CollectiveBuilder` instantiations, which upstream
builds through its own `cutlass_3x_gemm_sm120_custom` wrapper with
`ElementC = void`, do instantiate under ours with `ElementC = OutType`.

A compile is not an execution, and the rest of the device tier stays `## Owed`,
operator-run under an `rc` lease on `dgx:gpu0`:

- SACRED 27B and 35B token gates on both arms of `VT_FP8_CUTLASS_SMALL_M`;
- the same-binary `VT_DENSE_CUBLASLT_FP8` A/B at c1 decode, which is the
  measurement #1866 actually needs.

## Dependencies

- CUTLASS >= 4.5.0 in the build (`-DVLLM_CPP_CUTLASS_FETCH=ON` or
  `-DVLLM_CPP_CUTLASS_DIR=`). The TU is already gated on
  `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a) and this change adds no new gate.
- Nothing else. No new op, no new registration, no ABI change.

## Work breakdown

- **W1** the pure ladder header + its red-first test (CPU, done here).
- **W2** the two CUTLASS configs + the custom `EpilogueTile` plumbing + the
  dispatch rewire (written here, compile OWED).
- **W3** the `VT_GEMM_ALGO_LOG` candidate enumeration on the cuBLASLt lane
  (written here, default OFF).
- **W4** the record corrections (done here).
- **W5** the device compile, the token gates and the A/B — OWED, operator.
- **W6** the default decision that W5 enables — OWED, and explicitly not taken
  here.

## Design

`Fp8Sm120ConfigForM(int64_t m, bool small_m_enabled)` returns one of
`{kM16, kM32, kM64, kDefault}`. With `small_m_enabled` true it is upstream's
ladder verbatim; with it false the two small-M rungs collapse into `kM64`,
which is byte-for-byte the ladder this tree shipped before. The `.cu` holds
only a `switch` over that enum, so the branch that can be *wrong* is on the
tier that has a test runner.

Counters mirror `fp8_block_scaled_dispatch.h`: one relaxed atomic per config,
so a gate can assert which rung a real model step took rather than assuming it.

### Numerics: why this is expected to be bit-identical, and what would break it

All four configs use `TileShape_K = _128` and none configures split-K or a
stream-K scheduler. Each output element is therefore accumulated over the whole
of K, in `K/128` tile steps, in the same order, in the same f32 accumulator,
whichever M-tile computes it. Changing the M and N tile extents changes **which
CTA owns an output tile**, not the sequence of adds that produces its value.
The epilogue is `alpha * acc` with a device scalar `alpha` in every config, and
`EpilogueTile` partitions the store, not the arithmetic.

So the claim here is exact equality, not a near-tie band — and it is stated as
an expectation to be *measured*, because this code has not compiled yet. It is
deliberately not defended by a tolerance: a per-element tolerance on a bf16
store is exactly the instrument that cannot tell a store-width change from a
reduction-order change, which is the trap
`.agents/specs/perf-fp8-alpha-fold.md` records against the bf16-D arm and which
`RequireBf16DSplitKOne` exists to refuse. If the device run shows the M16 or
M32 rung moving a single token against the M64 rung, that is a reduction-order
difference this argument did not predict, and the correct response is
`NEEDS_DECISION` with the measured divergence — not a widened threshold.

The rollback arm makes that measurable in one binary: `VT_FP8_CUTLASS_SMALL_M=0`
is the M64 ladder, `=1`/unset is the four-way one, same weights, same prompts.

### What this does NOT claim

It does not claim 3.04 ms/step, or any part of it. It does not claim the
CUTLASS arm beats the cuBLASLt arm at decode — before this change that
comparison was not even fair, because the CUTLASS arm was running a 4x-too-tall
tile at every decode M. It makes the comparison fair and hands it to the
operator.

## Risks/decisions

- **The CUDA had never been compiled when this was written, and CI has since
  compiled it.** The authoring host has no `nvcc`, so the named risk was that a
  CUTLASS `CollectiveBuilder` instantiation upstream compiles under its own
  wrapper (`ElementC = void`) might not instantiate under ours
  (`ElementC = OutType`). `cuda-fat-build` is `success` at `120a;121a` with
  `VLLM_CPP_CUTLASS_FETCH=ON` on head `d9bf525c0` — the job, not its run, whose
  conclusion is `cancelled` — and nothing since that head changes compiled
  CUDA, so that risk is retired for the shipped TU. The current head's own
  `cuda-fat-build` is PENDING; see `## Gates`. What a compile CANNOT
  say is whether the kernel runs and what it computes, and that is still owed.
- **A default that does not move.** With `VT_DENSE_CUBLASLT_FP8` still ON, a
  production decode step does not reach the new configs. This is stated in the
  commit and PR bodies and listed under `## Owed` with the row that owns the
  flip. The alternative — flipping a shipped default on an uncompiled,
  unmeasured path — is worse.
- **The remaining hypothesis is not closed.** If the cuBLASLt lane stays the
  default, the sm89 selection still costs whatever it costs. W3's candidate
  enumeration is what turns "cuBLASLt has no sm121 fp8 kernels for our
  descriptor" from a guess into a reading. Note what the tree already knows and
  what it does not: the *scale form* is measured dead as a cause
  (`src/vt/cuda/fp8_plan_cache.h` records that `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`
  with `SCALAR_32F` "returns the identical algo as the host scalar", GB10,
  2026-08-12), while `CUBLASLT_MATMUL_DESC_FAST_ACCUM` has never been probed
  here — and upstream does not set it either (`use_fast_accum` is not passed at
  `scaled_mm/pytorch.py:81-88`, and the only two hits in the tree hardcode
  `False`), so setting it would be an invention rather than a mirror.
- **`requestedAlgoCount` stays 1 on the production query, and A(2) is
  edited.** W3 issues its own separate, diagnostic-only heuristic query, so the
  shipped call site keeps `/*requestedAlgoCount=*/kGemvHeuristicAlgos`
  byte-for-byte and invocation parity is unmoved. The invariant is NOT
  satisfied as written, and an earlier draft of this bullet claimed it was:
  `scripts/check-gemv-invocation-consistency.py` A(2) is rewritten by this
  change, in scope as `E6` above — the argument must now be a name on
  `ALGO_POLICY_NAMES` *and* be declared `constexpr int` in the same file. That
  is a strengthening of the rule rather than a widening of it, and the
  direction is measured, not asserted: mutating `unknown_algo_count_sites()`
  to accept anything reds 2 of that checker's 21 test cases, and rolling the
  whole A(2) body back to `a2c7dca45^` reds 6 (see `E6`; the fresh review's
  8/21 did not reproduce). What is widened is
  the ADMITTED SET, by exactly one diagnostic-only name, and widening that set
  is the deliberate edit A(2) exists to force.

## Evidence

Recorded in the PR body and the commit; the CPU commands and their exit status
are in `## Gates` above. No device evidence exists for this change and none is
claimed.

## Owed

- [#1866](https://github.com/mudler/vllm.cpp/issues/1866) stays OPEN after this
  change. What it asked for (an autotune sweep) is not the vLLM mirror; what it
  is *about* (the +3.04 ms/step FP8 tower) is not closed by a port that the
  default arm does not reach. It closes when W5 and W6 land.
- ~~The first CUDA compile of `src/vt/cuda/cuda_matmul_fp8_cutlass.cu` at
  `121a`.~~ DONE, by CI, on head `d9bf525c0` and not on this one: the
  `cuda-fat-build` JOB is `success` in run `32802716762`, building `vllm` with
  `VLLM_CPP_CUTLASS_FETCH=ON` across `120a;121a`. That RUN's conclusion is
  `cancelled`, so only the job is cited. `git diff --stat d9bf525c0..HEAD`
  changes no compiled CUDA, which is what carries the result forward; the
  current head's own `cuda-fat-build` is PENDING, and pending is not green.
  Struck rather than deleted, because the risk it retired is the one this row
  called its largest.
- The device numerical case: the same shape under M16, M32 and M64 must agree
  bit-for-bit, in `tests/vt/test_ops_fp8_cutlass.cpp`. OPERATOR.
- **The dispatch counters count CAPTURES, not decode steps, and the owed device
  case has to account for that or it will undercount badly.** From the fresh
  review. `Fp8PerTensorCountDispatch` is a host-side `std::atomic<uint64_t>`
  incremented where the host picks a rung
  (`src/vt/cuda/fp8_per_tensor_dispatch.h`), while the fp8 decode path is
  reached inside a CUDA-graph capture. A graph replay does no host dispatch, so
  the counter moves ONCE per captured graph and not once per replayed step —
  the same reading that makes `nsys` and `ncu` host-side counters read zero
  under graph replay. A device case that asserts "kM16 fired N times for N
  decode steps" will therefore read 1, or 0, and the failure will look like an
  unreached rung rather than a mismeasured one. The case must either run with
  capture disabled, or assert per-capture counts against the captured graph
  count, and it must say in the test which of the two it did.
- SACRED 27B / 35B token gates on both `VT_FP8_CUTLASS_SMALL_M` arms. OPERATOR.
- The `VT_DENSE_CUBLASLT_FP8` c1-decode A/B on the 27B fp8 tower, with the
  CUTLASS arm now carrying the small-M ladder. OPERATOR. This is the number
  #1866 needs.
- The `VT_GEMM_ALGO_LOG=1` candidate dump from one warm 27B decode, answering
  whether a `nvjet_sm121` algo is enumerated at all for our TN fp8 descriptor.
  OPERATOR.
- Depending on those two readings, either flip `VT_DENSE_CUBLASLT_FP8` to
  default OFF (mirroring vLLM's own backend order) or add a measured cuBLASLt
  algo sweep. Not decided here.

## Stop conditions

- A token moves between the M16/M32 rungs and the M64 rung: STOP,
  `NEEDS_DECISION`, with the measured divergence. Do not widen a tolerance.
- The CUTLASS instantiation cannot be built for `121a`: STOP, report the
  compiler diagnostic, do not fall back to a hand-written tile.
- The measurement shows the CUTLASS arm slower than cuBLASLt at decode even
  with the ladder: that is a valid negative. Record it, leave the default where
  it is, and the cuBLASLt sweep becomes the next hypothesis.

## Outcome

Pending. This section is written when the row reaches `DONE`.
