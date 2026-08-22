# `OpProviderStats::declines` becomes exact from the first decline

Row `KERNEL-ACCEL-PROVIDER-DECLINE-EXACT`, issue
[#1584](https://github.com/mudler/vllm.cpp/issues/1584). The seam is
`KERNEL-ACCEL-PROVIDER-SELECT`
([kernel-matrix.md](../kernel-matrix.md)); this row repairs one stated guarantee
of it and adds nothing to the matrix.

## Now

Spec committed, implementation in the same pull request. The CPU arm of the gate
runs on the authoring host. The CUDA and Metal arms do not (see `## Owed`).

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
| G6 CUDA arm | `test_ops_attention_cross` on a CUDA device | NOT RUN — see `## Owed` |
| G7 Metal arm | `test_metal_backend` on a `VLLM_CPP_MLX` build | NOT RUN — see `## Owed` |

## 9. Risks

- **The `.cu` and `.mm` edits are compile-unverified on this host.** No `nvcc`
  and no AppleClang. The CUDA arm is compiled by the `cmake --build
  build-cuda-fat --target vllm` step of `.github/workflows/ci.yml:842-854`; the
  MLX arm is compiled by no CI job in this repository, because MLX needs
  `MLX_ROOT`. Both edits are a one-token rename of a call whose signature is
  identical.
- **Removing `WarmDeclineOnce` cannot be verified here.** If the fix were wrong,
  the CUDA case would read 2 and go red on the first GPU run. That is the
  correct direction: keeping the warm-up would make the CUDA gate incapable of
  ever falsifying this row's claim.

## 10. Owed

| ID | What | Issue |
|---|---|---|
| O1 | Execute `test_ops_attention_cross` on a CUDA device and `test_metal_backend` on a `VLLM_CPP_MLX` build after this change, and take the reachability mutation on `BlockedFallback()` / `MlxFallback()`. Neither device exists on the authoring host. `test_ops_attention_cross` is not merely unrun there -- it is uninformative: 20 cases, 20 SKIP, 32 assertions, every one the skip guard | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) |

## 11. Stop conditions

- Stop and return `NEEDS_DECISION` if any existing `declines` assertion has to
  be relaxed or deleted to make this green. Relaxing an exact count to make a
  counter change fit is the failure this row exists to remove.
- Stop if the fix requires editing the Vulkan or Tenstorrent call sites. That is
  the blast radius #1584 warned about, and it means the chosen design was wrong.
