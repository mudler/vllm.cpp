# `OpProviderStats::declines` becomes exact from the first decline

Row `KERNEL-ACCEL-PROVIDER-DECLINE-EXACT`, issue
[#1584](https://github.com/mudler/vllm.cpp/issues/1584). The seam is
`KERNEL-ACCEL-PROVIDER-SELECT`
([kernel-matrix.md](../kernel-matrix.md)); this row repairs one stated guarantee
of it and adds nothing to the matrix.

## Now

Landed. **The CUDA arm of `## Owed` is executed** — `thor:gpu0`, sm_110,
2026-08-23, [#1692](https://github.com/mudler/vllm.cpp/issues/1692). `## 12`
carries the readings, including a negative one that corrects `## 9`. The Metal
arm is still unrun, because it needs a Mac.

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
focused gate) cannot be executed here for either. This is recorded in `## Owed`
rather than claimed.

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
| G7 Metal arm | `test_metal_backend` on a `VLLM_CPP_MLX` build | NOT RUN — see `## Owed` |

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
| O2 | Execute `test_metal_backend` on a `VLLM_CPP_MLX` build and take the reachability mutation on `MlxFallback()`. No Mac is reachable from this fleet, and `rc devices` lists none. The `.mm` file is COMPILED on Linux since [#1765](https://github.com/mudler/vllm.cpp/issues/1765); what is unrun is the MLX matmul provider itself. Its two assertions are `declines >= 1`, so they cannot see the off-by-one in either direction — what a run proves is that the arm still declines and forwards | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) |
| O3 | Register the two `declines == 1` cases as per-case ctest entries, or reset `BlockedFallback()`'s static between cases. `## 12.2` measures that the full-suite run — which is the only thing ctest and CI execute — stays green with this row's entire CUDA edit reverted, so nothing in the tree can catch a regression of it | [#1812](https://github.com/mudler/vllm.cpp/issues/1812) |

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
