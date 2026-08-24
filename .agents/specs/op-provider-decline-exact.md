# `OpProviderStats::declines` becomes exact from the first decline

Row `KERNEL-ACCEL-PROVIDER-DECLINE-EXACT`, issue
[#1584](https://github.com/mudler/vllm.cpp/issues/1584). The seam is
`KERNEL-ACCEL-PROVIDER-SELECT`
([kernel-matrix.md](../kernel-matrix.md)); this row repairs one stated guarantee
of it and adds nothing to the matrix.

## Now

Landed. **The CUDA arm of `## Owed` is executed** — `thor:gpu0`, sm_110,
2026-08-23, [#1692](https://github.com/mudler/vllm.cpp/issues/1692). `## 12`
carries the readings, including a negative one that corrects `## 9`. That
negative reading is now repaired: `## 13` makes the FULL unfiltered suite red
when the CUDA call site is reverted
([#1812](https://github.com/mudler/vllm.cpp/issues/1812)). **The Metal arm is
executed too**, on a `macos-15` GitHub runner, 2026-08-23 (`## 14`) — that
runner does expose a real Metal device and a working MLX device, which `## 14.1`
establishes from evidence before anything was designed, and the arm is now run by
`macos-metal-mlx` in `.github/workflows/ci.yml` on every push to `main` rather
than only when somebody cuts a release.

## 1. Scope

In scope:

- `include/vt/op_provider.h` — a resolver that does not count, and prose that
  matches what the code does.
- `src/vt/op_provider.cpp` — the two spellings share one body.
- The two live shape-gated providers that cache the fallback pointer:
  `src/vt/cuda/cuda_attention_cross.cu` and `src/vt/metal/metal_mlx_provider.mm`.
- `tests/vt/test_op_provider.cpp` — the red-before case and the new function's
  own contract.
- `tests/vt/test_ops_attention_cross.cpp` — removal of the `WarmDeclineOnce`
  workaround that #1555 added to dodge this defect.

Out of scope, deliberately:

- `src/vt/vulkan/vulkan_ops.cpp:950,1067,1488,1509` and
  `src/vt/tenstorrent/tenstorrent_ops.cpp:1341`. They call `GetOpFallback`
  per call and without caching, so they count correctly today. The design below
  is chosen so that they do not have to change and what they report does not
  move.
- Any change to `NoteOpDecline`, `GetOpProviderStats`, `ResetOpProviderStats`
  or the `declines` field itself.
- The `qlen_cap_declines` counter in
  `src/vllm/v1/worker/gpu/cudagraph_dispatch.h:220`. Same English word,
  unrelated counter.

## 2. The defect, stated as code rather than as prose

`include/vt/op_provider.h:178-184` tells a provider author to cache the fallback
pointer once and call `NoteOpDecline` per decline, and states the guarantee:
"so `OpProviderStats::declines` stays exact".

It does not stay exact. `GetOpFallback` increments the same counter at
`src/vt/op_provider.cpp:709`, and the prescribed cache is a function-local
static, so the FIRST decline of a process takes two increments and every later
one takes one. Over N declines the counter reads N+1.

The magnitude is one. The consequence is not, because `declines` is the only
discriminator that separates a served call from a forwarded one for a provider
that is always the SELECTED one. A gate asserting `declines == N` therefore
passes or fails on whether some earlier case in the same binary already forced
the static to resolve. `tests/vt/test_op_provider.cpp:19-21` states that its own
cases are order-independent; the cases #1555 added to
`tests/vt/test_ops_attention_cross.cpp` were not, and needed a warm-up call to
become so.

## 3. Anchors

No upstream mirror: `vt::OpProvider` is a vllm.cpp original (inventory deviation
§9.1). The shape it generalizes is vLLM's own runtime chain — flashinfer tactic
registries and cuBLASLt/CUTLASS per-call heuristics — which select per call and
do not expose a decline counter, so vLLM answers nothing here and this is a
local-seam decision. Local anchors:

Line numbers are read at the base commit `73ada0df8`, because this row moves
several of them and an anchor recorded against the head is stale before the
pull request merges. Grep the symbol, not the line.

| What | Symbol | At `73ada0df8` |
|---|---|---|
| The false guarantee | the `NoteOpDecline` doc comment | `include/vt/op_provider.h:178-184` |
| The second increment | `slot.declines.fetch_add` inside `GetOpFallback` | `src/vt/op_provider.cpp:709` |
| The single-counter helper | `NoteOpDecline` | `src/vt/op_provider.cpp:727-729` |
| Live caching site (CUDA) | `BlockedFallback` / `AttentionCrossBlockedCuda` | `src/vt/cuda/cuda_attention_cross.cu:586-620` |
| Live caching site (Metal) | `MlxFallback` / `MlxMatmulKernel` | `src/vt/metal/metal_mlx_provider.mm:207-241` |
| The workaround | `WarmDeclineOnce` | `tests/vt/test_ops_attention_cross.cpp:511-543` |
| Non-caching callers, unchanged | `GetOpFallback(` | `src/vt/vulkan/vulkan_ops.cpp:950,1067,1488,1509`, `src/vt/tenstorrent/tenstorrent_ops.cpp:1341` |

## 4. Design

Add `GetOpFallbackUncounted(op, device, declining_provider)`: the same
resolution as `GetOpFallback` with the `fetch_add` omitted. Both spellings share
one body, so the resolution order, the reference-tier install, the drain and
every throw stay byte-identical; only the increment is conditional, and it stays
in its current position — after the reference-tier install attempt and BEFORE
the "nothing below" `VT_CHECK`, so a decline that throws still counts, which
`tests/vt/test_op_provider.cpp:156-157` depends on.

The two caching providers switch to it. `GetOpFallback` keeps counting, so the
five non-caching call sites are untouched.

The header's prose then matches the code: `GetOpFallback` counts one decline,
`GetOpFallbackUncounted` counts none and must be paired with `NoteOpDecline`.

### Why the other candidate loses

Candidate 1 in #1584 is to drop the `fetch_add` from `GetOpFallback` and make
`NoteOpDecline` the sole counter. It loses on the direction of its failure mode.

- It requires five call sites to be edited in this change, and it changes what
  the Vulkan and Tenstorrent backends report if any is missed. One of them is
  already gated on an exact value: `tests/vt/test_vulkan_backend.cpp:2901`
  asserts `after.declines == before.declines + 1` on the `kGdnPrefill` decline.
- Its failure mode for a FUTURE caller is silent UNDER-counting: a provider that
  declines and forwards, and reports zero declines, is indistinguishable from a
  provider that served the call. That is exactly the Risk 4 the seam exists to
  make visible, and the reason
  `tests/vllm/models/test_opt_paged_engine.cpp:201` calls `declines == 0` "the
  load-bearing half".
- The chosen design's failure mode for a future caller is that they use the
  ordinary spelling and pay a provider-stack walk per decline. That is a
  performance regression, which a benchmark can see, rather than an
  observability hole, which nothing can see.

The safe default therefore stays on the ordinary name, and the opt-out is
spelled `Uncounted` at the call site.

## 5. Every consumer of `declines` and what it reports after this change

| Consumer | Path | After |
|---|---|---|
| `tests/vt/test_op_provider.cpp:157` (`>= 3`) | direct `GetOpFallback` x3 | unchanged |
| `tests/vt/test_vulkan_backend.cpp:2901` (`before + 1`) | Vulkan `kGdnPrefill` per-call `GetOpFallback` | unchanged |
| `tests/vt/test_backend_cross_device.cpp:1213` (`== 0`) | no decline | unchanged |
| `tests/parity/test_qwen3_paged_engine.cpp:400` (`== 0`) | no decline | unchanged |
| `tests/parity/test_qwen35_paged_engine.cpp:354` (`== 0`) | no decline | unchanged |
| `tests/parity/test_mistral_paged_engine.cpp:337` (`== 0`) | no decline | unchanged |
| `tests/vllm/models/test_opt_paged_engine.cpp:283` (`== 0`) | no decline | unchanged |
| `tests/vt/test_metal_backend.cpp:392` (`== 0`) | MSL arm, no decline | unchanged |
| `tests/vt/test_metal_backend.cpp:416,476` (`>= 1`) | MLX caching provider | value drops by 1 on the first decline of the process; the bound is `>=`, so the assertion is unchanged |
| `tests/vt/test_ops_attention_cross.cpp:652,668,681,709,729,749,775,855,856,890,891` (`== 0`) | no decline | unchanged |
| `tests/vt/test_ops_attention_cross.cpp:773,800` (`== 1`) | CUDA caching provider | exactly 1 WITHOUT the warm-up call, which is what this row buys |
| `GetOpProviderStats` (public seam API) | reader | reports one fewer for the first decline of a caching provider — the defect |
| `VT_OP_PROVIDER_STATS=1` announce line (`op_provider.cpp:262-269`) | reader | does not print `declines`; unchanged |
| `include/vllm.h` | — | does not expose `declines`; unchanged |

Four backends were named in #1584 as at risk. Measured: with this design the
number is **zero** — no Vulkan, Tenstorrent, CPU or non-caching CUDA reading
moves. Exactly two providers change what they report, both by exactly one, both
in the direction of the header's stated contract: the CUDA
`vt-cross-blocked` attention provider and the Metal `mlx` matmul provider.

## 6. The red-before, and why it is subtle

The double count happens once per function-local static, not once per process
globally, but the two are the same thing for a static that only one case
reaches. The red-before is therefore built to be red in BOTH regimes rather than
only under a filter:

- a private (op, device) slot no other case touches;
- a private `CachedFallback()` static that only this case can warm;
- `ResetOpProviderStats` immediately before the counted window, so the static
  resolves INSIDE it;
- an EXACT assertion, `declines == 3` after three declines.

Before the fix that reads 4 in a full run and 4 under `doctest -tc=`. A count
that is short in the other direction (a dropped `NoteOpDecline`) reads 1 or 0
and fails the same assertion, so the case is two-sided.

`doctest -tc=` splits its filter on commas, and a filter that matches nothing
prints `0 cases ran` and `SUCCESS!`. Every case name added by this row is
comma-free, and every filtered run below quotes its `test cases:` line so that a
zero-case run cannot read as a pass.

## 7. Reachability

`GetOpFallbackUncounted` lands with two production call sites, both in the same
commit, neither a test and neither an example:

- `vt::AttentionCross` (`src/vt/ops.cpp`) -> the selected `vt-cross-blocked`
  provider `AttentionCrossBlockedCuda` -> `BlockedFallback()` ->
  `GetOpFallbackUncounted`. Reached from `ModelRegistry::Forward` through the
  MiniMax-Music3 DiT and depth decoder.
- `vt::Matmul` / `vt::MatmulBT` -> the selected `mlx` provider ->
  `MlxFallback()` -> `GetOpFallbackUncounted`, on a `VLLM_CPP_MLX` build.

Both entry points are GPU-only. The authoring host has neither a CUDA device nor
a Mac, so the reachability MUTATION (delete the production call site, rerun the
focused gate) cannot be executed here for either. ~~This is recorded in
`## Owed` rather than claimed.~~ **Both are now executed elsewhere:** the CUDA
one on `thor:gpu0` (`## 12.4` M3) and the Metal one on a `macos-15` runner
(`## 14.4` M2).

## 8. Gates

| Gate | Command | State |
|---|---|---|
| G1 red-before | `build/tests/test_op_provider -tc='op provider: the cached fallback pattern counts exactly one decline per decline'` on the pre-fix tree | must be RED with `declines == 4` |
| G2 focused | the same case plus `op provider: GetOpFallbackUncounted resolves the same provider without counting`, in a full run and under a per-case filter | GREEN |
| G3 suite | `ctest --test-dir build -R test_op_provider` | GREEN |
| G4 full CPU | `ctest --test-dir build` | GREEN modulo the known-red set |
| G5 preflight | `scripts/agent-preflight.sh --staged` | GREEN |
| G6 CUDA arm | `test_ops_attention_cross` on a CUDA device | GREEN on `thor:gpu0` — 20 cases, 156 assertions, `Status: SUCCESS!`, 0 `SKIP`. See `## 12` |
| G6b CUDA red-before | the two `declines == 1` cases under `-tc=`, with the double count reintroduced | RED with `CHECK( 2 == 1 )` both ways. **The FULL run is NOT a gate for this** — see `## 12.2` |
| G6c CUDA reachability | delete the `BlockedFallback()` call, rerun `test_ops_attention_cross` | RED — 8 cases, 13 assertions. See `## 12.4` |
| G7 Metal arm | `test_metal_backend` on a `VLLM_CPP_MLX` build | RUN on a `macos-15` runner — see `## 14.2`. RED as a whole on [#1823](https://github.com/mudler/vllm.cpp/issues/1823), which is not this row's; the two steps that are this row's are GREEN |
| G7b Metal red-before | `GetOpFallbackUncounted` → `GetOpFallback` in `MlxFallback`, dispatched as a `ci.yml` run on a probe branch | RED — `## 14.4` M1 |
| G7c Metal reachability | the `MlxFallback()(...)` forward disabled, same dispatch | RED — `## 14.4` M2 |
| G7d Metal is executed by CI | `macos-metal-mlx` builds `test_metal_backend`, runs the exactness case in its own process, resolves the per-case ctest entry, then runs the suite | LANDED — `## 14.3` |
| G8 the FULL run falsifies the CUDA call site | `build-cuda/tests/test_ops_attention_cross`, UNFILTERED, with `GetOpFallbackUncounted` → `GetOpFallback` in `cuda_attention_cross.cu` | must be RED. `## 13.4` |
| G8b restored | the same binary rebuilt from the restored source | GREEN, `git status --porcelain` empty. `## 13.4` |

## 9. Risks

- **The `.cu` and `.mm` edits are compile-unverified on this host.** No `nvcc`
  and no AppleClang. The CUDA arm is compiled by the `cmake --build
  build-cuda-fat --target vllm` step of `.github/workflows/ci.yml:842-854`, and
  it now also compiles and RUNS on `thor:gpu0` (`## 12`). The claim that the MLX
  arm "is compiled by no CI job in this repository" was true when written and is
  no longer: [#1765](https://github.com/mudler/vllm.cpp/issues/1765) added the
  never-linked `vllm_metal_mlx_provider_syntax_check` object library, so every
  Linux non-MSVC configure compiles `metal_mlx_provider.mm` against the real
  `vt::` seam. What is still owed for Metal is EXECUTION on a Mac, not a
  compile.
- **Removing `WarmDeclineOnce` cannot be verified here.** ~~If the fix were
  wrong, the CUDA case would read 2 and go red on the first GPU run.~~
  **MEASURED FALSE for a full run, and this is the row's most useful negative
  result.** Both halves of the defect were reintroduced on the device and the
  full `test_ops_attention_cross` stayed green at 156/156 each time, because the
  suite warms the static itself. Only a per-case `-tc=` run reads 2. `## 12.2`
  has the mechanism and `## 12.3` has the contrast.

## 10. Owed

| ID | What | Issue |
|---|---|---|
| ~~O1~~ | ~~Execute `test_ops_attention_cross` on a CUDA device~~ **DISCHARGED 2026-08-23 on `thor:gpu0`: green after, red before, and the reachability mutation, all in `## 12`** | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) |
| ~~O2~~ | ~~Execute `test_metal_backend` on a `VLLM_CPP_MLX` build and take the reachability mutation on `MlxFallback()`~~ **DISCHARGED 2026-08-23 on a `macos-15` GitHub runner: green after, red before, and the reachability mutation, all in `## 14`.** The premise that no Mac is reachable was right about the `rc` fleet and wrong about the runner pool. The observation that the two pre-existing assertions are `declines >= 1` and cannot see the off-by-one stands, and is what `## 14.2`'s new `== 1` case exists for | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) |
| ~~O3~~ | ~~Register the two `declines == 1` cases as per-case ctest entries, or reset `BlockedFallback()`'s static between cases.~~ **DISCHARGED, and by neither of those two.** `## 13` measures the FIRST decline of the process from a doctest listener, before any case runs, so the unfiltered suite reds when the CUDA call site is reverted | [#1812](https://github.com/mudler/vllm.cpp/issues/1812) |

## 11. Stop conditions

- Stop and return `NEEDS_DECISION` if any existing `declines` assertion has to
  be relaxed or deleted to make this green. Relaxing an exact count to make a
  counter change fit is the failure this row exists to remove.
- Stop if the fix requires editing the Vulkan or Tenstorrent call sites. That is
  the blast radius #1584 warned about, and it means the chosen design was wrong.

## 12. The CUDA arm, executed on `thor:gpu0`

`thor:gpu0` — NVIDIA Thor, sm_110, driver 595.78 — inside two `rc run` leases on
2026-08-23: `d452b91f-a12d-4ce5-8035-beab829fbd8c` and
`43a27be9-d819-4f34-8ef4-8e530e672ed9`. Worker pod `rc-worker-kk96r`, `boot_id`
`e2112cac-660b-434e-911d-33cbd29b9176`, aarch64, root, `/tmp` at 115 GiB free.
The job installed CUDA 13.0.88 itself, because the toolkit is not in the worker
image. Tree `bacb71109`, staged as a `git archive` tarball and guarded by a
`FATAL_CLONE` check that the staged `cuda_attention_cross.cu` carries
`GetOpFallbackUncounted` and that `WarmDeclineOnce` is gone from the suite.

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF
cmake --build build-cuda -j 4 --target test_op_provider test_ops_attention_cross
```

Thor is the only sm_110 device on this fleet. Any CUDA device would have
answered this row, because it is a correctness question with no timing axis.

### 12.1 Green after, and the suite stops being uninformative

| Suite | CPU-only build | `thor:gpu0`, CUDA |
|---|---|---|
| `test_ops_attention_cross` | 20 cases, 32 assertions, 21 `SKIP` lines | **20 cases, 20 passed, 156 assertions, 156 passed, `Status: SUCCESS!`, 0 `SKIP` lines** |
| `test_op_provider` | — | 14 cases, 14 passed, 446 assertions, 446 passed, `Status: SUCCESS!`, 0 `SKIP` lines |

`ctest -j1 -R 'test_op_provider|test_ops_attention_cross'` passes 2/2. The
CPU-only column is quoted from #1692 and was NOT re-measured in this run; the
CUDA column is measured. The assertion count is the number that matters: on the
authoring host all 32 were the skip guard, and 156 assertions with zero `SKIP`
lines is the suite finally doing its job.

**The provider that engaged says so in its own output.** With
`VT_OP_PROVIDER_STATS=1`, and unchanged under the per-case filter of `## 12.3`:

```text
[vt op-provider] op=19 device=1 selected=vt-cross-blocked priority=10 registered=2 caps=0.0/unprobed
[vt op-provider] op=19 device=0 selected=vt-native priority=0 registered=1 caps=0.0/unprobed
```

`op=19` is `OpId::kAttentionCross` and `device=1` is `DeviceType::kCUDA`, so the
blocked provider is the selected one on the device with `vt-native` beneath it.
A green suite without that line would be measuring the CPU tier.

### 12.2 What the full run cannot falsify

The first thing this arm produced was a NEGATIVE result about the gate, and it
is why `## 9` Risk 2 is struck through above.

Reintroducing the defect in the seam — `GetOpFallbackUncounted` delegating with
`/*count=*/true` — turns `test_op_provider` red and leaves
`test_ops_attention_cross` **green in a full run**, 156/156. Reverting the CUDA
call site instead (`GetOpFallbackUncounted` → `GetOpFallback` in
`cuda_attention_cross.cu`, which is this row's entire CUDA edit) also leaves the
full run green.

The cause is in the suite, and the reachability mutation of `## 12.4` is what
shows it: with the `BlockedFallback()` call site disabled, the cases that go red
FIRST are the plain `attention-cross:` ones at `test_ops_attention_cross.cpp:326`
and `:334`. Those predate the blocked provider and assert nothing about
`declines`. They run CUDA `AttentionCross` on geometries `BlockedShape` rejects,
so they decline and forward, and each of them resolves `BlockedFallback()`'s
function-local static outside any counted window. By the time the
`declines == 1` cases run, the static is warm and the extra increment has
already been erased by a `ResetOpProviderStats`.

So the suite warms itself, exactly as `WarmDeclineOnce` warmed it by hand. The
file's own comment is the accurate one — these cases are exact "standalone and
under `-tc=`". Nothing in `tests/CMakeLists.txt` runs them that way, which is
O3 and [#1812](https://github.com/mudler/vllm.cpp/issues/1812).

### 12.3 The red-before that does falsify it: one case, one process

`-tc=` splits its filter on commas and the depth-decoder case name contains one,
so it is matched with a trailing wildcard. Every run quotes its `test cases:`
line, because a filter that matches nothing prints `0 cases ran` and `SUCCESS!`.

| Tree | `…the DEPTH-DECODER shape declines*` | `…an unhandled head_dim declines once per call` |
|---|---|---|
| baseline | `test cases: 1 \| 1 passed`, `assertions: 6 \| 6 passed`, `SUCCESS!` | `test cases: 1 \| 1 passed`, `assertions: 4 \| 4 passed`, `SUCCESS!` |
| M1, seam double count | `1 \| 0 passed \| 1 failed`, `6 \| 5 passed \| 1 failed`, `FAILURE!`, **`CHECK( 2 == 1 )`** | `1 \| 0 passed \| 1 failed`, `4 \| 3 passed \| 1 failed`, `FAILURE!`, **`CHECK( 2 == 1 )`** |
| M2, CUDA call site reverted | same, **`CHECK( 2 == 1 )`** | same, **`CHECK( 2 == 1 )`** |
| restored | `1 \| 1 passed`, `6 \| 6 passed`, `SUCCESS!` | `1 \| 1 passed`, `4 \| 4 passed`, `SUCCESS!` |

M2 is the one that answers this row. It touches nothing but the CUDA call site,
and it moves both assertions from 1 to 2, so `BlockedFallback` is genuinely on
the uncounted path rather than merely compiled beside it.

### 12.4 The mutation table

Every mutation applied as exactly one hunk, verified by `git diff --stat` and a
`^@@` count against a scratch commit of the staged tree, and every rebuild
printed its `compile_rc` BEFORE any test ran.

| ID | Change | hunks | `compile_rc` | Effect |
|---|---|---|---|---|
| M1 | `op_provider.cpp`: `ResolveFallback(…, /*count=*/false)` → `true` | 1 | 0 | `test_op_provider` RED: `CHECK( 4 == 3 )` at `:203`, plus `1 == 0`, `2 == 1`, `3 == 1`, `4 == 2` in the `GetOpFallbackUncounted` case. Full `test_ops_attention_cross` GREEN. Filtered cases RED |
| M2 | `cuda_attention_cross.cu`: `GetOpFallbackUncounted` → `GetOpFallback` | 1 | 0 | `test_op_provider` GREEN (446/446), full attention suite GREEN, filtered cases RED with `CHECK( 2 == 1 )` |
| M3 | `cuda_attention_cross.cu`: the `BlockedFallback()(…)` call no longer forwards | 1 | 0 | `test_ops_attention_cross` RED: 8 cases, 13 assertions, first at `:326`/`:334` |

M3 is the `.agents/reachability.md` mutation. It is expressed as
`if (BlockedFallback() == nullptr) return;` rather than as a deleted line so
that the symbol stays referenced: an anonymous-namespace function that becomes
unused fails `-Werror=unused-function`, and a mutation that does not build reads
as a passing test.

**Restore, and one honest caveat.** After each mutation the file was restored
with `git checkout --`, `touch`ed so ninja could not skip the TU, and rebuilt;
`git status --porcelain` was empty every time and the suites returned to their
baseline counts. The BINARY hash returns to baseline only for the g++ arm:
`op_provider.cpp.o` hashed back to `3725972…` byte for byte after M1, while
`cuda_attention_cross.cu.o` came back as `6a6d4a0…` and `662477e…` from an
identical clean source. **nvcc's output for this TU is not bit-reproducible
here**, so a hash comparison cannot prove a `.cu` restore and the green re-run
is what does.

## 13. The gate that could not fail, repaired ([#1812](https://github.com/mudler/vllm.cpp/issues/1812))

`## 12.2` is the finding this section closes. Both of the exact `declines == 1`
routing assertions stayed GREEN in a full `test_ops_attention_cross` run with
this row's entire CUDA edit reverted, and `tests/CMakeLists.txt` registers one
ctest entry per suite, so the full run is the only run anything performs. The
assertions were correct, exact and load-bearing on paper, and nothing in this
repository could watch them move.

### 13.1 Why the existing shape cannot be repaired in place

The double count is **once per function-local static**, not once per call. It is
therefore observable only around the call that RESOLVES `BlockedFallback()`'s
static, and invisible to every measurement taken afterwards. Three consequences
follow, and each one closes a candidate repair:

- A window is not the problem, so a wider or narrower window does not fix it.
  The two cases ALREADY reset immediately before their counted window
  (`ProviderStatsGuard`), and they still read 1 either way.
- A **delta across a reset the case performs** fails for the same reason. After
  the static is warm, the delta is 1 whether the resolver counts or not.
- **Ordering is not a repair.** doctest's default `--order-by=file` is a default,
  `--order-by=rand` exists, and a case placed first is one file edit away from
  no longer being first.

### 13.2 The design: measure the first decline before doctest runs a case

A doctest **listener** fires `test_run_start()` once, before the case loop
(`third_party/doctest/doctest.h:5983`), and is skipped only for the
`--list-*`/`--count` query modes. A listener registered with
`DOCTEST_REGISTER_LISTENER` is always active whatever `-r=` selects, so it
cannot be switched off from the command line.

`tests/vt/test_ops_attention_cross.cpp` registers one. It runs a single
`vt::AttentionCross` on the cheapest geometry `BlockedShape` rejects
(`{tq=1, s=8, hq=1, hk=1, d=32}`, asserted rather than assumed), through the same
`RunCuda` helper every case below uses, so the window is the same reset → one
call → read. It records four things — whether it fired, whether there was a
device, the `declines` it saw, and any exception, kept rather than swallowed.

Being FIRST is what the design buys, and it is bought by construction rather
than by ordering: `test_run_start()` precedes every case, so the probe's call is
the call that resolves the static, and its window is a counted one.

One new case asserts the reading:

```
attention-cross blocked: the FIRST decline of the process counts exactly ONE
```

It asserts the instrument's own precondition (`REQUIRE(p.ran)`) before its
reading, because a listener that never fired leaves a zeroed struct that would
otherwise read as an absent device. Then `CHECK(p.declines == 1)`. Exact, and
two-sided: 2 is #1584's double count, 0 is a dropped `NoteOpDecline`.

**Nothing existing is weakened, and one thing had to be added to avoid weakening
it.** The two `declines == 1` assertions are untouched. But the probe warms the
static under a `-tc=` filter too, which would have taken away the one regime in
which those two cases WERE falsifiable (`## 12.3`). Each therefore also asserts
`FirstDecline().declines == 1` beside its own count, so a filtered run of either
case stays red on a regression. The assertions get strictly harder to satisfy.

### 13.3 The candidates that lose

| Candidate | Why it loses |
|---|---|
| Per-case ctest entries with `-tc=` (#1812 candidate 1) | It does not satisfy the requirement, which is that the FULL run reds. It also puts the case NAME in a second file: `-tc=` splits on commas so the depth-decoder name needs a trailing wildcard, a filter that matches nothing prints `0 cases ran` and `SUCCESS!`, and a renamed case silently returns to that state unless a wrapper also asserts `test cases: 1`. A gate whose failure mode is a silent pass is the defect this row exists to remove |
| Reset `BlockedFallback()`'s static between cases (#1812 candidate 2) | It edits a production hot path to suit a gate, and the seam gets a test-only entry point. The static's whole purpose is that the provider stack is immutable after registration |
| A delta across a reset the case performs | Cannot work. Measured, not argued: the two cases already reset before their window and are still blind (`## 12.2`) |
| A private op slot / private stats scope for a copy of the pattern | That is `tests/vt/test_op_provider.cpp`'s red-before, and it already exists. It tests a COPY of the caching pattern, so it stayed GREEN under M2 — the mutation that reverts `cuda_attention_cross.cu` — which is exactly the mutation that has to red |
| Reordering the cases | doctest does not guarantee case order, and `--order-by=rand` inverts it |

### 13.4 Evidence, executed on `thor:gpu0`

`thor:gpu0` — NVIDIA Thor, sm_110 (`compute_cap 11.0`), driver 595.78 — inside
`rc run` job `09a2f31e-65bc-463d-ac35-674def5add20` on 2026-08-23. Worker pod
`rc-worker-kk96r`, `boot_id` `e2112cac-660b-434e-911d-33cbd29b9176`, aarch64,
root, 99 GiB free on `/tmp`. The job installed CUDA 13.0.88 itself. Tree
`1d066423f`, staged as a `git archive` tarball behind a `FATAL_CLONE` check that
the staged `cuda_attention_cross.cu` carries `GetOpFallbackUncounted`, that the
suite carries the listener, `MeasureFirstDecline` and the new case, and that
`WarmDeclineOnce` survives only as comment prose (that last guard refused a
CORRECT tree on the first submission, because the removal's own explanation
names the symbol; it now ignores comment lines).

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF
cmake --build build-cuda -j 4 --target test_op_provider test_ops_attention_cross
```

`configure_rc=0`, `-- CUDA target architectures: 110`, 33 `.cu.o` objects, and
the binary links `libcudart.so.13`/`libcublasLt.so.13`. Every rebuild printed
its `compile_rc` before any test ran, and each was `0`.

**The provider that engaged says so**, under `VT_OP_PROVIDER_STATS=1`:

```text
[vt op-provider] op=19 device=1 selected=vt-cross-blocked priority=10 registered=2 caps=0.0/unprobed
[vt op-provider] op=19 device=0 selected=vt-native priority=0 registered=1 caps=0.0/unprobed
```

#### The acceptance result

| Tree | FULL `test_ops_attention_cross`, unfiltered | `SKIP` lines |
|---|---|---|
| baseline | `test cases: 21 \| 21 passed`, `assertions: 161 \| 161 passed`, `Status: SUCCESS!` | 0 |
| **M2, this row's whole CUDA edit reverted** | **`21 \| 18 passed \| 3 failed`, `161 \| 158 passed \| 3 failed`, `Status: FAILURE!`** | 0 |
| restored | `21 \| 21 passed`, `161 \| 161 passed`, `Status: SUCCESS!` | 0 |

That is the row. Before this change the same mutation left the full run GREEN at
156/156 (`## 12.2`); it now reds it. `ctest -R test_ops_attention_cross` reports
`rc=8`, `504 - test_ops_attention_cross (Failed)`, so what CI executes sees it.

The three failing assertions are `CHECK( 2 == 1 )` in the new case AND in both
pre-existing exact cases, which is the point of `## 13.2`'s last paragraph: the
two old assertions did not get easier, they got harder.

#### The mutation table

Every mutation applied as exactly one hunk, verified by `git diff --stat` and a
`^@@` count against a scratch commit of the staged tree.

| ID | Change | hunks | `compile_rc` | FULL run | Filtered runs |
|---|---|---|---|---|---|
| M2 | `cuda_attention_cross.cu`: `GetOpFallbackUncounted` → `GetOpFallback` | 1 | 0 | **RED** 21/18/3, 161/158/3 | all three RED, `CHECK( 2 == 1 )`, `test cases: 1` each |
| M1 | `op_provider.cpp`: `ResolveFallback(…, /*count=*/false)` → `true` | 1 | 0 | **RED** 21/18/3, 161/158/3 | — ; `test_op_provider` RED 14/12/2, 446/441/5 |
| M4 | the `DOCTEST_REGISTER_LISTENER` line removed, so the probe never fires | 1 | 0 | **RED** 21/18/3, `assertions: 159` | — |

M4 is the instrument's own precondition, and it fails in the right place:
`REQUIRE( false )` on `p.ran` in the new case, and `CHECK( 0 == 1 )` in the two
supplemented cases reading a zeroed struct. The assertion total drops from 161
to 159 because the `REQUIRE` aborts its case. A probe that silently did not run
would otherwise have read as a device that was simply absent.

#### Restores

After each mutation the file was restored with `git checkout --`, `touch`ed so
ninja could not skip the TU, and rebuilt. `git status --porcelain` was empty
after all three (`porcelain_lines=0`), and the final re-run returned to
baseline: `test_ops_attention_cross` `21 \| 21 passed`, `161 \| 161 passed`,
`SUCCESS!`, 0 `SKIP`; `test_op_provider` `14 \| 14 passed`, `446 \| 446
passed`, `SUCCESS!`, 0 `SKIP`; `ctest` 2/2, `final_ctest_rc=0`.

**One honest caveat, carried from `## 12.4`.** nvcc's output for
`cuda_attention_cross.cu` is not bit-reproducible on this host, so a `.cu`
restore cannot be proved by a binary hash. The green re-run and the empty
`git status --porcelain` are what prove it, and this run relies on exactly
those two.

#### What the CPU build reports

Measured on the authoring host (x86-64, no CUDA), baseline vs this change:

| Tree | `test_ops_attention_cross` on CPU | `SKIP` lines |
|---|---|---|
| `af320abb2` | `20 \| 20 passed`, `32 \| 32 passed`, `SUCCESS!` | 20 |
| this change | `21 \| 21 passed`, `33 \| 33 passed`, `SUCCESS!` | 21 |

The delta is exactly the new case and its one `REQUIRE(p.ran)` — which still
runs without a device, because the listener fires either way and records that it
did. A CPU green proves nothing about the count, and it is quoted here only so
the numbers are not a surprise later.

## 14. The Metal arm, executed on a `macos-15` GitHub runner

The CUDA arm needed a device this fleet has. The Metal arm needed a Mac, and
`## 10` O2 recorded that none is reachable — which was right about `rc devices`
and wrong about the runner pool. GitHub's `macos-15` runners are the Mac, and
they were already in this repository.

### 14.1 The device question, answered before anything was designed

`MlxSupports` gates on `MetalContext::Available()`, so a device-less runner
registers no MLX provider and every MLX case would SKIP rather than exercise it.
`assertions: 0` is a skip wearing a pass, and this row exists to remove exactly
that shape, so the question was settled from evidence first.

**Answer: yes, and MLX computes on it.** Release run
[31466516224](https://github.com/mudler/vllm.cpp/actions/runs/31466516224), job
`93700466685` (`mlx_arm64`, image `macos-15-arm64` 20260727.0256.1, 2026-08-11)
executed `test_metal_backend`:

```text
[doctest] test cases:     25 |     25 passed | 0 failed | 3 skipped
[doctest] assertions: 112327 | 112327 passed | 0 failed |
[doctest] Status: SUCCESS!
```

The 3 skips are the file's own `doctest::skip(true)` benchmarks — `Metal
strided-read bandwidth probe`, `Metal GEMM microbenchmark`, `Metal GEMM per-row
diagnostic` — and not a device guard. The GEMM case printed real MLX numerics on
that run: `mlx-vs-cpu=1.66168e-13` and `mlx-vs-msl=1.48631e-13` on the square f32
arm, against `mlx-vs-msl=0` on the decode arm MLX declines. A silently-absent MLX
cannot produce a non-zero MLX-vs-MSL difference on one shape and a zero on
another.

What that run could NOT cover is this row: `944d7d947` landed after it.

### 14.2 The case that can see the off-by-one

`test_metal_backend`'s two pre-existing MLX decline assertions are
`declines >= 1` (`:416` and `:476` at the time #1692 was filed), and #1692 says
plainly that they cannot see the off-by-one in either direction. They still
cannot. This row adds one that can, `MLX counts EXACTLY one decline for the
first decline of the process`:

- it asserts `declines == 1` after ONE declining `MatmulBT`, and `== 1` again
  after a second, and that the two readings are EQUAL — "exact from the FIRST
  decline" is a statement about `d1 == d2`, and the counting resolver makes
  `d1` 2 and `d2` 1;
- it REQUIREs the `mlx` provider to be registered rather than skipping when it
  is not, so a device-less runner reds instead of reporting `assertions: 0`;
- it checks the NMSE of the declined result against the CPU oracle, which is
  the reachability half: the fallback must have RUN, not merely resolved.

It sits FIRST in line order, above every case that issues a Metal matmul.
doctest's default is `--order-by=file`, which for a single-file binary is line
order (`third_party/doctest/doctest.h:5476` `fileOrderComparator`, defaulted at
`:5714`), and no case above it issues a `Matmul`/`MatmulBT` — `:179` only asks
`OpRegistered`. So it owns the FIRST decline of the process and `MlxFallback`'s
statics are cold when it runs.

**That is the CUDA arm's defect avoided rather than repeated.** `## 12.2`
measures that the full `test_ops_attention_cross` stayed green with the entire
CUDA half of this row reverted, because an earlier case warmed the static outside
a counted window (#1812).

**And ordering alone is NOT what makes this one falsifiable.** `## 13.1` — landed
for the CUDA arm while this section was being measured — argues that correctly:
`--order-by=file` is a default, `--order-by=rand` exists, and a case that is first
is one file edit away from not being. What carries the guarantee here is the
DEDICATED PROCESS: `tests/CMakeLists.txt` registers the case as its own ctest
entry, which is O3's first candidate repair applied where a single entry is
enough, and `macos-metal-mlx` runs it as its own step. Line order is a second,
weaker property that makes the full suite red as well; it is stated, and it is not
leaned on.

**The stronger shape now exists in this tree and was not adopted here.** `## 13.2`
measures the first decline from a doctest LISTENER that fires before any case, so
being first is bought by construction rather than by a line number. Adopting the
same listener in `tests/vt/test_metal_backend.cpp` is the better end state and is
recorded here as such. It was not done in this change because the measured gate
already reds in a configuration CI executes, and because the whole-suite arm it
would strengthen is red on #1823 regardless — so the addition would buy nothing
observable today and would grow this row past what #1692 asks.

`--test-case=` splits on commas, so the name carries none. The CMake `COMMAND`
argument is QUOTED, and that was a live defect for one commit: unquoted, CMake
splits it on its spaces into `--test-case=*EXACTLY`, `one` and `decline*`,
doctest matches nothing, prints `0 cases ran` and `Status: SUCCESS!`, and the
entry is green forever. Caught before any runner allocated, fixed in
`744abeca3`, and the job now asserts `test cases: 1 | 1 passed` out of a
`ctest -V` run precisely because an entry's exit status cannot report it.

### 14.3 What `macos-metal-mlx` runs now

`.github/workflows/ci.yml`'s `macos-metal-mlx` configured
`VLLM_CPP_BUILD_TESTS=OFF` and built `--target vllm`. It now builds
`test_metal_backend` too and executes it three ways, in this order:

1. **the exactness case in its own process** — this row's gate, FIRST so that an
   unrelated red elsewhere in the suite cannot skip it (`## 14.5` is why that
   order is not hypothetical);
2. **`ctest -V -R '^test_metal_backend_mlx_decline_exact$'`** — that the CMake
   entry RESOLVES, asserted from its doctest summary rather than from its exit
   status, and scoped to that entry alone so the question stays answerable while
   `#1823` keeps the whole-suite entry red;
3. **the full suite**, with the summary PARSED rather than trusted.

The full-suite step keeps floors of 25 cases and 100000 assertions, `0 failed` on
both summary lines, `Status: SUCCESS!`, and the skip count echoed but not gated
so that adding a benchmark does not red the lane.

Exit status is deliberately not the verdict at any of the three: doctest prints
`assertions: 0 | 0 passed` and `Status: SUCCESS!` for a run that executed
nothing, and a filter that matches nothing prints `0 cases ran` and exits 0.

### 14.4 The mutation table

No Linux host can run this job, so red-before and green-after are dispatched
`ci.yml` runs on probe branches (`gh workflow run ci.yml --ref <branch>`), the
technique `.agents/specs/gate-metal-mlx-compile.md` `§12.6` established. Job-level
conclusions are read with `gh run view --json jobs`, because a superseded run
reports `cancelled` at run level.

| Run | Branch / SHA | Mutation | `macos-metal-mlx` | The step that decided it |
|---|---|---|---|---|
| [`32669918527`](https://github.com/mudler/vllm.cpp/actions/runs/32669918527) | `row/…-METAL` `ae262aaab` | none | **failure**, on `#1823` alone | exactness step **success**: `test cases: 1 \| 1 passed \| 0 failed \| 28 skipped`, `assertions: 9 \| 9 passed \| 0 failed`, `Status: SUCCESS!`, `filtered run rc=0`, `SKIP lines: 0`, and `MLX decline accounting: first=1 second=1 provider=mlx NMSE vs CPU=2.86646e-06`. The suite step then read 26 cases / 25 passed / **1 failed** / 112336 assertions, `Status: FAILURE!`, on `:170` — `## 14.5` |
| [`32669919787`](https://github.com/mudler/vllm.cpp/actions/runs/32669919787) | `probe/metal-1692-red-m1d` `542db5fec` | **M1** — `GetOpFallbackUncounted` → `GetOpFallback` in `MlxFallback`, 2 hunks, this row's whole Metal edit | **failure** | exactness step **failure**: `:423 ERROR: CHECK( d1 == 1ull ) is NOT correct!` and `:433 ERROR: CHECK( d1 == d2 ) is NOT correct!`, `MLX decline accounting: first=2 second=1`, `test cases: 1 \| 0 passed \| 1 failed`, `assertions: 9 \| 7 passed \| 2 failed`, `filtered run rc=1` |
| [`32669921278`](https://github.com/mudler/vllm.cpp/actions/runs/32669921278) | `probe/metal-1692-red-m2d` `e447210f7` | **M2** — the `MlxFallback()(q, out, a, b)` forward disabled in both kernels, 2 hunks, expressed as `if (MlxFallback(op) == nullptr) MlxFallback(op)(…)` so every symbol and parameter stays referenced | **failure** | exactness step **failure**: `:443` and `:444 ERROR: CHECK( nmse <= 5e-4 ) is NOT correct!`, `NMSE vs CPU=1`, `test cases: 1 \| 0 passed \| 1 failed`, `assertions: 9 \| 7 passed \| 2 failed` |
| [`32671380955`](https://github.com/mudler/vllm.cpp/actions/runs/32671380955) | `row/…-METAL` `875f74fb6` | none | **failure**, on `#1823` alone | exactness step **success** (same three lines and the same accounting message); ctest step **success** — `1/1 Test #415: test_metal_backend_mlx_decline_exact ... Passed`, `100% tests passed out of 1`, `test cases: 1 \| 1 passed`; suite step **failure** with the floor parser reached and printing `cases total=26 passed=25 failed=1 skipped=3`, `assertions total=112336 passed=112335 failed=1`, `FAIL: a case or assertion failed` |
| [`32672532769`](https://github.com/mudler/vllm.cpp/actions/runs/32672532769) | `row/…-METAL` `56457cf83`, the LANDING tree, after merging `origin/main` | none | **failure**, on `#1823` alone | identical to the row above, on the merged tree: exactness **success**, ctest **success** (`1/1 Test #415 … Passed`, `100% tests passed out of 1`, `0.10 sec`), suite **failure** with `cases total=26 passed=25 failed=1 skipped=3`. The three `run:` scripts are byte-identical to `875f74fb6`'s (sha256 per step), so this run answers only whether the merge changed the answer. It did not |

**M1 is the red-before this row owes.** It moves the FIRST reading from 1 to 2
while leaving the second at 1, which is the #1584 defect stated as two numbers,
and it does so in the configuration CI runs rather than only under a hand-typed
filter. **M2 is `.agents/reachability.md`'s mutation.** Its accounting reading
stays `first=1 second=1` and only the numeric checks red, which separates
"resolved a fallback pointer" from "called it" — the production forward in
`MlxMatmulBTKernel` is what the test enters through.

Every mutation is 2 hunks, verified with `git diff --stat` and a `^@@` count
before the commit, and the compile step ran and succeeded on all four (step 5
`success`), so no red here is a compiler red wearing a test's face.

**Two of this job's own assertions were wrong and were caught by running them,
which is the argument for this whole row in miniature.** The CMake `COMMAND`
filter was unquoted (`744abeca3`), and the ctest verdict was asserted as `tests
passed, 0 tests failed out of 1` — the wording CTest uses when something DID
fail. Run `32670585579` reported `ctest rc=0`, the entry `Passed` and `100%
tests passed out of 1`, and the step still went red on that grep. Both are
written from the log this runner produces rather than from memory.

**One thing measured and not assumed: `--no-colors=1` works.** doctest disables
colour when stdout is not a tty, but the greps in this job read the summary lines
and an ANSI escape would defeat them silently (a green 79-gate run once read as
`0 ok / 1 FAIL` here for that reason). All three logs carry bare
`[doctest] test cases: …` lines.

**And the runner queue.** These three dispatches waited roughly 15 minutes for a
`macos-15` runner and the job itself ran in about 5 minutes, against the 45-50
minutes `gate-metal-mlx-compile.md` `§12.6` measured on 2026-08-23. The queue is
variable, not a constant, and neither figure is a floor.

### 14.5 What the first execution found, immediately

The suite step went RED on its first run, on a case this row did not touch:
`test_metal_backend.cpp:170`, `SelectAttentionBackendName(p) == "FLASH_ATTN"`
THREW `"No valid attention backend for device type 2 from {FLASH_ATTN: [compute
capability not supported]}"`. 26 cases, 25 passed, 1 failed, 112336 assertions,
`Status: FAILURE!`.

`FlashAttentionBackend::supports_compute_capability` is upstream's NVIDIA
`capability >= (8,0)`; `MetalPlatform::get_device_capability` answers with the
MTLGPUFamilyApple GENERATION, deliberately. Apple family 9 on the M4 gate box
clears an SM-8.0 bar by coincidence and the runner reports lower. `369ea7fd4`
landed 2026-08-19 and is not an ancestor of `7020de936`, so the case had been red
for four days with no lane able to run it — the [#1765](https://github.com/mudler/vllm.cpp/issues/1765)
exposure window turned into an actual break.

Filed as [#1823](https://github.com/mudler/vllm.cpp/issues/1823), owned by
`BACKEND-ATTN-REGISTRY`, listed under `## Found in flow, filed, not fixed here`
in `.agents/specs/attn-validate-configuration.md`. NOT fixed here: every
candidate repair changes what backend selection means for four device types.

**So `macos-metal-mlx` is RED on `main` from this commit**, and that is not
hidden behind `continue-on-error`, a skip or a floor tuned around it. A lane that
had never executed anything found a real break the moment it did; hiding it would
rebuild the defect this row and #1765 both exist to remove. It is also why the
exactness step runs FIRST — on run
[32668677681](https://github.com/mudler/vllm.cpp/actions/runs/32668677681) it
reported `skipped` behind the suite failure, which would have made this row's own
gate hostage to an unrelated case.
