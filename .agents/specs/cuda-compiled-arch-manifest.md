# SPEC — the FA2 capability predicate consults what was compiled

Issue: [#1357](https://github.com/mudler/vllm.cpp/issues/1357).
Umbrella: [#1332](https://github.com/mudler/vllm.cpp/issues/1332) M2.
Owning rows: `BACKEND-CUDA-COMP-FA` (the FA2 arms), `BACKEND-PLATFORM` (the
predicate seam).

## Now

Neither row changes lifecycle state. This closes one over-claiming predicate on
a live consumer; it does not widen an arm or add a backend.

## The defect

`CudaPlatform::supports_fa2_attention()` returns `true` for every CUDA device.
`qwen3_5.cpp:5163` consumes it to choose bf16 FlashAttention-2 over the f32
graph-captured fallback, so the claim decides `attn_dt` on the default decode
path. The default build compiles FA2 for the intersection of
`VLLM_CPP_CUDA_ARCHITECTURES` (`121a`) with the feature table's `fa2` row
(`8.0,8.6,8.7,8.9,12.0a,12.1a`) — one architecture. Run that build on an sm_86
card and the model takes a path with no code for the device.

**A predicate over the DEVICE cannot answer a question about the BINARY.** That
is [#1332](https://github.com/mudler/vllm.cpp/issues/1332)'s invariant, and this
is the tree's own instance of it.

## Why this and not more of `validate_configuration`

Considered and rejected. The attention-backend selector's result reaches
`attn_backend_names_`, a debug log and `CheckKvCacheShape`, and no kernel;
`dense_attn::AttnBlock` calls `vt::PagedAttention` unconditionally. Adding
capability declarations there would add surface to a mechanism that routes
nothing. `supports_fa2_attention()` has a real consumer today, so fixing it
changes what executes. That is the whole difference, and it is why this row is
narrow.

## Design

**The manifest is generated from the variable that emits the flags, never
written by hand.** `vt_cuda_feature_archs(VT_FA2_ARCHS "fa2")`
(`CMakeLists.txt:499`) already computes the FA2 architecture set, and
`CMakeLists.txt:2279` passes that SAME variable to `vt_cuda_set_source_gencode`,
which turns it into the `-gencode` options nvcc receives. `configure_file`
renders it into a header, mirroring the existing `include/vllm/version.h.in`
pattern. Nothing tracks anything: one variable, two consumers.

A hand-maintained list is exactly `CUDA_SUPPORTED_ARCHS`, the shape that caused
this. There is no second list to keep in sync, and the manifest is generated per
build into the build tree, so it is not a file any pull request edits.

**`vt_cuda_archs_denormalize` strips `+PTX`** (`CudaArchFeatures.cmake:190-198`),
so `VT_FA2_ARCHS` names only targets with real SASS. That is the correct input
here, and it is load-bearing — see the PTX rule below.

**The matching rule is deliberately conservative, and its polarity is the
argument.** An arch counts as compiled when the running device matches a manifest
entry:

- exact `(major, minor)`, suffix included — an `a` target is arch-specific and
  matches only its own arch;
- same major, no suffix, manifest minor `<=` device minor — CUDA SASS
  minor-version compatibility, the rule that makes an `sm_80` cubin run on
  sm_86.

Everything else is not compiled. **PTX is deliberately NOT a yes.** A compute-only
target is what burned the reference engine: its FA2 binary carries `sm_80` SASS
plus `compute_80` PTX, its predicate said yes on a GB10, and every launch failed
the driver JIT with `cudaErrorUnsupportedPtxVersion`. Treating "a JIT could
theoretically produce code" as "this will run" is the defect, not the fix.

The polarity matters more than the precision. A false negative drops to the f32
graph-captured fallback: correct output, slower. A false positive launches a
kernel with no code for the device. Where the rule is uncertain it must answer
no, and this one does.

## What this changes, and where it changes nothing

It changes which dtype `qwen3_5.cpp:5160-5176` selects on any build whose FA2
arch set does not contain the running device. **On the gate hardware nothing
changes**, because a GB10 build requests `121a`, the feature table's `fa2` row
contains `12.1a`, so `VT_FA2_ARCHS` contains `121a` and the device reports
capability 12,1 — an exact match including the suffix. That is asserted rather
than asserted-about: the test pins the exact GB10 triple.

## Tests

`tests/vllm/platforms/test_cuda_arch_manifest.cpp`, CPU-runnable, because the
matcher is a pure function over an INJECTED manifest string and an injected
capability. No GPU, no CUDA build.

Red-first case: a fixture manifest that lacks the simulated device's arch must
make the predicate answer false. Plus the GB10 exact-match control; the
`a`-suffix rule (a `121a` manifest does not serve a base sm_121 request and vice
versa); minor-version compatibility (`80` serves sm_86, `86` does not serve
sm_80); an empty manifest, which is the "FA2 not built at all" case; and the PTX
rule, asserted as a decision rather than assumed.

## Gates

`scripts/agent-preflight.sh --fail-on-skip`, plus the focused test and
`test_platform`. The CUDA leg cannot be built or run here, so the `cuda.cpp` call
site is proven by the matcher's tests plus a mutation, and that limit is stated
rather than papered over.

## Evidence

CPU Debug build, `-DVLLM_CPP_CUDA=OFF`. Every run is a direct binary invocation,
never through a pipe.

**RED before.** The matcher was first implemented as the status quo — literally
`return true`, which is what `supports_fa2_attention()` did — so the first run
measured the DEFECT rather than a missing symbol:

```
[doctest] test cases:  6 |  0 passed |  6 failed | 0 skipped
[doctest] assertions: 25 | 14 passed | 11 failed |
[doctest] Status: FAILURE!            (compile_rc=0, run exit 1)
```

**GREEN after.**

| Binary | cases | assertions | status |
|---|---:|---:|---|
| `test_cuda_arch_manifest` | 6 | 40 | SUCCESS |
| `test_platform` | 12 | 95 | SUCCESS |
| `test_cuda_fa2_arch_manifest.py` | 6 | — | OK |

**The gate hardware keeps its FA2 path, proved rather than asserted.**
`tests/scripts/test_cuda_fa2_arch_manifest.py` drives the REAL
`cmake/CudaArchFeatures.cmake` through `cmake -P`, so it needs no CUDA toolkit
and no GPU: requesting `121a` resolves the manifest to exactly `121a`, and a
device reporting 12,1 matches it including the suffix. The same probe shows the
ten-SM release request narrowing to `{80, 86, 87, 89, 120a, 121a}` — `90a`,
`100a`, `103a` and `110` dropped, because the `fa2` feature row names no kernel
body for them — and shows a Hopper-only build resolving EMPTY.

**Mutations.** Restored from a `tar` snapshot, never `git checkout`. `compile_rc`
and `git diff --stat` printed for each.

| # | Mutation | `compile_rc` | Result |
|---|---|---:|---|
| MM1 | hard-code the unconditional `return true` again | 0 | RED 1/6 cases pass |
| MM2 | ignore the arch-specific suffix | 0 | RED 5/6 |
| MM3 | drop SASS minor-version compatibility | 0 | RED 4/6 |
| MM4 | **reachability**: sever the production call site from the manifest | 0 | **GREEN — a NULL RESULT, see below** |

**MM4 measured nothing, and is reported as nothing.**
`src/vllm/platforms/cuda.cpp` is compiled only inside the CUDA block
(`CMakeLists.txt:1635`), so on a CPU host the mutated translation unit is not in
the build at all: `ninja -t targets all | grep -c 'platforms/cuda.cpp'` returns
`0` and no `cuda.cpp.o` exists. Its green is the "mutation that never entered the
build reads as a pass" shape, not evidence that the call site is unreached. The
wiring is therefore proven here by inspection only; the CI `cuda-fat-build` job
compiles that translation unit and is what actually exercises it. Stated rather
than papered over, and carried in `## Owed`.

**One mutation stranded a change and was caught.** A 10-minute shell timeout
killed the MM3 runner between its build and its restore, leaving the mutation in
the tree. It was found by grepping for the `MUTATION` marker rather than by
trusting the script, restored from the snapshot, and the runner was then hardened
with an `EXIT`/`INT`/`TERM` trap. A second defect surfaced with it: this host's
`find` is `bfs`, which rejects the `-newermt '-1 day'` form the restore used, so
the post-restore `touch` had been failing silently — and a `tar` restore rewinds
mtimes, which is exactly how ninja is made to skip a rebuild and hand back a
binary built from mutated source. The restore now touches the files explicitly,
and a full rebuild plus re-run confirmed green before the pass continued.

## Owed

- `SelectAttentionBackendName` still routes nothing, and
  `validate_configuration` and the `supports_*` predicates are still unported.
  Both owned by [#1332](https://github.com/mudler/vllm.cpp/issues/1332).
- The manifest covers the `fa2` feature only. Every other `VT_CUDA_FEATURE_TABLE`
  row has the same exposure the moment a predicate consults it; the generator is
  written so a second feature is one line, but no second feature is wired here.
- No CUDA build ran in this change. The matcher is proven on host; the `cuda.cpp`
  wiring is proven by mutation, not by execution on a device whose arch is
  absent from the manifest. Owner: `BACKEND-CUDA-COMP-FA`.

## Stop conditions

- Stop if the manifest cannot be derived from `VT_FA2_ARCHS` and would have to be
  written by hand. An honest blocker beats a list that goes stale.
- Stop if the conservative rule would disable FA2 on the gate hardware, because
  that is a performance regression on the one device we measure.
