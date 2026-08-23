# Spec — the one Metal TU that Linux can compile, compiled where a merge can see it

Row `GATE-METAL-MLX-COMPILE`, issue
[#1765](https://github.com/mudler/vllm.cpp/issues/1765). Filed out of
[#1692](https://github.com/mudler/vllm.cpp/issues/1692), which owns the two
GPU/Mac **runtime** arms of `KERNEL-ACCEL-PROVIDER-DECLINE-EXACT` and states
one thing about the build that this row measured and found wrong.

## 1. Scope

In scope:

- One never-linked OBJECT library that compiles
  `src/vt/metal/metal_mlx_provider.mm` on a host with no Mac, no Metal and no
  MLX, in the shape `vllm_rocm_platform_syntax_check` already uses for the ROCm
  platform leg. That target sits at `CMakeLists.txt:1726` after this row's own
  60-line insertion at `:1490` moved it down; grep the symbol, not the line.
  It was `:1665` before this row and `:1725` before the rebase onto
  `ff8f72807` that added one line above it, which is the staleness this file is
  arguing about, happening to this file.
- The stub headers that make that possible, under `src/vt/metal/stubs/`, in the
  shape `src/vt/cuda/flash_attn/stubs/` already uses for ATen and c10.

Out of scope, deliberately:

- `metal_ops.mm`, `metal_backend.mm`, `metal_context.mm`. They carry real
  Objective-C (`§2` measures how much), so no C++ compiler reaches them and this
  technique does not extend to them. **Wave 2 (`§12`, 2026-08-23) covers them by
  a different route** — a `macos-15` job on the post-merge lane — and re-measures
  the claim in this bullet rather than inheriting it.
- The MLX API surface. The stubs are ours, so the gate cannot fail for an MLX
  reason. `§4` states that limit as a limit. Wave 2 closes it (`§12.3`): the
  macOS job builds against the real `mlx` wheel.
- Either runtime arm of #1692. A GPU and a Mac; neither is here, and neither
  becomes cheaper because of this row.
- `src/vt/cuda/cuda_attention_cross.cu`. Already compile-gated pre-merge by the
  `-DVLLM_CPP_CUDA=ON` job at `.github/workflows/ci.yml:840-854`, so the second
  #1584 call site needs nothing from this row.

## 2. The defect, measured

#1692 states that `metal_mlx_provider.mm` is "compiled by NO job in this
repository". **That is not accurate, and the correction is the whole shape of
this row.** `.github/workflows/release.yml:347` `mlx_arm64` runs on `macos-15`,
installs `mlx==0.32.0`, and calls `scripts/build-macos-release.sh
macos-arm64-metal-mlx preview ... "$mlx_root" 0.32.0 ...`, which configures
`-DVLLM_CPP_MLX=ON` and compiles the file for real against real MLX.

The defect is the trigger, not the absence. `release.yml:3-6`:

```yaml
on:
  workflow_dispatch: {}
  push:
    tags: ['v*']
```

A `v*` tag or a manual dispatch. Never a pull request, never a push to `main`.
And the pre-merge lane has no Apple runner to fall back on. All 17 `runs-on`
lines in `.github/workflows/ci.yml` are `ubuntu-latest` (14),
`ubuntu-24.04-arm` (1) or `windows-2022` (2). The file does contain the string
`macos` exactly once, at `ci.yml:199`, and it is not a runner: it is
`python3 tests/scripts/test_release_macos_metadata.py`, a metadata test of the
release manifest that compiles nothing.

So the only build of this file happens **after** the change has landed, and only
when somebody cuts a release. A break presents as a blocked release rather than
as a red check, and the author who caused it has moved on.

**The exposure window is not hypothetical, and it is open right now.** The last
successful `release.yml` run is `31466516224` at head `7020de936`, tag `v0.0.2`,
2026-08-11; its `mlx_arm64` job concluded `success`. The two runs after it
(2026-08-12) failed, and there has been none since. Measured against
`origin/main` at `8eecc05a9`, 2026-08-23:

| Since the file was last compiled (`7020de936`) | Count |
|---|---|
| commits on `main` | 938 |
| commits touching `src/vt/` or `include/vt/` | 120 |
| commits touching `op_provider.{h,cpp}` — the seam this file calls | **11** |
| commits touching `metal_mlx_provider.mm` itself | 1 — `944d7d947`, the #1584 fix |

So the #1584 edit to `MlxFallback` has been compiled **nowhere**, and eleven
changes to the seam under it have landed in the twelve days since anything last
built it. This row is not a precaution against a hypothetical; it is the gate
that would have caught the change that motivated it, on the day it landed.

**One thing #1692 is owed less of than it says, and one it is not.** The same
script runs what it builds: `scripts/build-macos-release.sh:44-46` builds
`server test_metal_backend` and then executes `"$build_dir/tests/test_metal_backend"`.
So the Metal arm of #1692's O1 is not "never executed anywhere" — it executed
on 2026-08-11, before the change it is owed for. Whether the `macos-15` runner
exposes a Metal device, and therefore whether the MLX provider registered at all
rather than skipping, is **not established here** and is left to #1692. What is
established is the trigger: even that execution is tag-gated and post-merge.

Measured on a configured CPU tree at base `8eecc05a9`
(`cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON`):

| Query | Count |
|---|---|
| `metal_mlx_provider` in `build/build.ninja` | 0 |
| `metal_mlx_provider` in `build/compile_commands.json` | 0 |
| **any** `.mm` file in `build/compile_commands.json` | 0 |
| `src/vt/op_provider.cpp` in `build/compile_commands.json` (control) | 3 |

Zero edges, not few. Nothing in a Linux build can detect any change to that
file, which is why #1584's edit to `MlxFallback` landed unbuilt.

**Why this file and not the other three.** `metal_mlx_provider.mm` is the only
Metal TU that is plain C++ wearing an ObjC++ extension. Counting the lines that
carry an Objective-C construct — a message send, an `@`-keyword, an `id<...>`:

| TU | ObjC-construct lines |
|---|---|
| `metal_mlx_provider.mm` | **0** |
| `metal_ops.mm` | 10 |
| `metal_backend.mm` | 5 |
| `metal_context.mm` | 19 |

Its two `#import`s are `<Foundation/Foundation.h>` and `<Metal/Metal.h>`, and it
uses nothing from either: the Metal handles arrive as `void*`, because
`metal_context.h:22` says so on purpose — "This header is deliberately PLAIN C++
(no Objective-C types) so the op TUs and the tests can include it" — and
`metal_buffers.h:37` carries the `MTLBuffer` as a `void* buffer` with the ObjC
type in a comment. The seam this file actually depends on is `vt/op_provider.h`
and `vt/ops.h`, and those are ordinary portable C++.

## 3. Design

`add_library(vllm_metal_mlx_provider_syntax_check OBJECT
src/vt/metal/metal_mlx_provider.mm)`, never linked anywhere, built by `all`.
Four choices in it are not obvious.

**`LANGUAGE CXX` on a `.mm`.** CMake maps `.mm` to the `OBJCXX` language, which
would need `enable_language(OBJCXX)` and an ObjC++ compiler that a Linux runner
does not have. `set_source_files_properties(... PROPERTIES LANGUAGE CXX)` makes
CMake emit `-x c++` instead. That property is directory-scoped, so it could in
principle reach the real `vllm` target — which is why the whole block is guarded
`if(NOT VLLM_CPP_MLX ...)`. On the build that actually ships the provider the
property is never set, and `mlx_arm64` compiles the file as OBJCXX exactly as it
does today.

**`-Wno-deprecated`, and nothing wider.** g++ rejects ObjC's `#import` in C++
mode as `error: #import is a deprecated GCC extension [-Werror=deprecated]`.
Measured, rather than assumed, that this flag is narrow: with
`-Wall -Wextra -Werror -Wno-deprecated`, a call to a `[[deprecated]]` function
still fails as `-Werror=deprecated-declarations`. The suppression buys the
`#import` spelling and gives up no diagnostic this TU could otherwise draw. The
alternative — rewriting the two `#import`s to `#include` — was rejected because
it edits a production file whose real compiler is not here, to satisfy a gate.

**`NOT MSVC`.** In MSVC `#import` means "import a type library", so cl.exe would
hard-error on the same two lines. The two `windows-2022` jobs must not acquire a
new red from a bit-rot guard, so they simply do not build it. gcc and clang do.

**Stubs, under `src/vt/metal/stubs/`.** `Foundation/Foundation.h` and
`Metal/Metal.h` are empty, which is exactly faithful because the file uses
nothing from them. `mlx/{allocator,array,dtype,ops,stream,transforms}.h` declare
only the MLX surface the file names, with no definitions — a syntax gate needs
declarations, not a library. The location and the idea are
`src/vt/cuda/flash_attn/stubs/{ATen,c10}`, which exist so a vendored CUDA TU
compiles without torch.

## 4. What this proves, and what it does not

It proves the file compiles against the **real** `vt::` seam headers. That is the
#1584 defect class exactly: a rename, a signature change or a removed
declaration in `include/vt/op_provider.h` now fails a Linux build.

It proves **nothing** about the MLX API. The stubs are written from the call
sites, so they cannot disagree with them; if MLX 0.33 changes `array::set_data`,
this gate stays green and `mlx_arm64` goes red at the next release. It proves
nothing about Foundation, Metal, AppleClang, ObjC++ codegen, or any runtime
behaviour.

Stating it the other way round: this closes the gap where **our** refactor
breaks the file, and leaves open the gap where **their** release does. The first
is the one that has already happened.

## 5. Tests

The gate is a build target, so its test is the build, and its red-before is a
mutation of the guarded file. `§6` G1-G3.

No `tests/` file changes. A doctest case cannot express "this TU compiles",
and a checker that greps `CMakeLists.txt` for the target name would be a
tautology — it would assert its own expectation out of the file it reads.

## 6. Gates

| Gate | Command | State |
|---|---|---|
| G0 build-graph, before | `grep -c metal_mlx_provider build/compile_commands.json` at base | 0 — measured, `§2` |
| G1 red-before | break the `.mm`, `cmake --build build` at base | GREEN, i.e. undetected |
| G2 red-after | the same break, `cmake --build build` with this change | RED |
| G3 green-after | restore the `.mm`, `cmake --build build` | GREEN |
| G4 full CPU build | `cmake --build build -j 12` | GREEN |
| G5 full CPU ctest | `ctest --test-dir build` | GREEN modulo the known-red set |
| G6 preflight | `scripts/agent-preflight.sh --staged` | GREEN |
| G7 MSVC | the two `windows-2022` jobs | unchanged by construction — `NOT MSVC` |
| G8 Apple MLX build | `mlx_arm64` on the next tag | unchanged by construction — `NOT VLLM_CPP_MLX` |

## 7. Risks / decisions

- **gcc 16 is not measured here.** `build-newest-gcc` runs the `gcc:16`
  container (`ci.yml:1125`) and this host has gcc 13.3.0. If gcc 16 spells the
  `#import` diagnostic differently, that job reds and the fix is one flag. The
  risk is bounded to a flag because the TU is otherwise ordinary C++.
- **The stubs can drift.** They are written to the call sites, so drift is
  invisible to this gate by construction (`§4`). Recorded, not mitigated: the
  mitigation is `mlx_arm64`, which already exists.
- **A stub is not the dependency.** A reader could take a green
  `vllm_metal_mlx_provider_syntax_check` for "the MLX provider builds". The
  target name says `syntax_check`, and the CMake comment says what it does not
  cover.

## 8. Stop conditions

- Stop and return `NEEDS_DECISION` if the file cannot be compiled without
  editing it. A gate bought by changing the guarded source is not a gate.
- Stop if the target requires `enable_language(OBJCXX)` anywhere, which would
  make a Linux configure depend on an ObjC++ compiler.

## 9. Now

**Wave 1** (2026-08-23, `15298f033`): spec and implementation in one pull
request. G0-G6 measured on this host. G7 and G8 hold by construction and are
verified by reading the guard, not by running an MSVC or an Apple build.

**Wave 2** (`§12`): the `macos-metal-mlx` job on `ci.yml`'s post-merge lane,
covering all four Metal TUs against the real SDK. Its red-before is measured on
this host (`§12.1`); its green-after is measured on the lane itself
(`§12.5`), because no Linux host can run it.

## 10. Owed

| ID | What | Issue |
|---|---|---|
| O1 | The two **runtime** arms: `test_ops_attention_cross` on a CUDA device and `test_metal_backend` on a `VLLM_CPP_MLX` build, plus the `.agents/reachability.md` mutation on `BlockedFallback()` / `MlxFallback()`. Unchanged by this row, which buys compile coverage only. `§2` narrows the Metal half — that binary IS executed by `build-macos-release.sh:46`, but last did so at `v0.0.2` on 2026-08-11, before the change it is owed for, and whether the runner had a Metal device at all is unestablished | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) |
| O2 | ~~The MLX **API** surface.~~ **DISCHARGED by wave 2** (`§12.3`): `macos-metal-mlx` builds the provider against the real `mlx==0.32.0` wheel on every push to `main` and every 4-hourly baseline, not only at release time | [#1765](https://github.com/mudler/vllm.cpp/issues/1765) |
| O3 | ~~The other three Metal TUs.~~ **DISCHARGED by wave 2** (`§12`): all four are compiled by `macos-metal-mlx`, against the real Apple SDK. What remains owed is not coverage but latency — the lane is post-merge, so a break is named a commit or two after it lands rather than before (`§12.4`) | [#1765](https://github.com/mudler/vllm.cpp/issues/1765) |

## 12. Wave 2 — the other three Metal TUs, on the only runner that can compile them

`§10` O3 recorded `metal_ops.mm`, `metal_backend.mm` and `metal_context.mm` as
owed and owned by nobody. This wave discharges O3 and O2. It uses a different
technique, because `§1` is still right that the wave-1 technique cannot reach
them, and this section measures that rather than repeating it.

### 12.1 The gap, measured at `be432e8e3`

`cmake -S . -B build -G Ninja -DVLLM_CPP_BUILD_TESTS=ON
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`, then counting occurrences:

| TU | `build.ninja` | `compile_commands.json` |
|---|---|---|
| `metal_backend.mm` | 0 | 0 |
| `metal_context.mm` | 0 | 0 |
| `metal_ops.mm` | 0 | 0 |
| `metal_mlx_provider.mm` (wave 1) | 3 | 3 |
| `src/vt/op_provider.cpp` (control) | 6 | 6 |

**Red-before, one #1584-class seam rename per file**, applied at once so a single
build answers for all three: `RegisterBackend(` (from `include/vt/backend.h`) in
`metal_backend.mm`, 1 hunk; `VT_CHECK(` (from `include/vt/dtype.h`) in
`metal_context.mm`, 6 hunks; `RegisterOp(` (from `include/vt/ops.h`) in
`metal_ops.mm`, 19 hunks. Every one of them makes the file reference a symbol
that does not exist.

- `cmake --build build -j 12` → **rc 0**, 1667 edges, 9m24.65s wall, `grep -c
  'error:'` = **0**. The build log mentions `metal_backend` 0 times,
  `metal_context` 0, `metal_ops` 0, and `metal_mlx_provider` once
  (`[1079/1667] Building CXX object
  CMakeFiles/vllm_metal_mlx_provider_syntax_check.dir/...`) — the wave-1 control,
  proving the measurement instrument works and only these three are invisible.
- `ctest --test-dir build -j 4` → 593/595 passed, 3 skipped, rc 8 on
  `test_serve_low_tools` and `test_async_llm`; both pass serially (rc 0), which
  is `-j` starvation under `verification.md`, not a regression, and neither can
  be caused by a file in no build rule.
- Restored with `git checkout --` plus `touch`; `sha256sum -c` OK on all three.

So the full CPU build and the full gate are **green over three broken files**.
That is the defect, stated as a measurement.

**Why no C++ compiler can close it.** Compiling each file with wave 1's exact
recipe (`-x c++`, the stub include path, `-Wall -Wextra -Werror
-Wno-deprecated`, `-fsyntax-only`):

| TU | rc | `error:` lines | first diagnostic |
|---|---|---|---|
| `metal_backend.mm` | 1 | 22 | `‘MTLDevice’ was not declared in this scope` |
| `metal_context.mm` | 1 | 46 | `stray ‘@’ in program` |
| `metal_ops.mm` | 1 | 166 | `‘MTLCommandBuffer’ was not declared in this scope` |

No stub fixes `stray ‘@’`. These files need an Objective-C++ front end, and the
development host has none: `g++ -x objective-c++` → `cannot execute
‘cc1objplus’`, no `clang` anywhere under `/`, `/opt`, `/usr/local` or `$HOME`,
no `/usr/include/objc`.

### 12.2 Why C, and not A or B

Three designs were costed before one was written. The developer chose **C** on
2026-08-23.

- **A — a Linux clang ObjC++ syntax check.** Covers 3 of 4 TUs, at seam level
  only. It needs an ObjC++ compiler installed here and in CI, and it needs stubs
  that **invent** roughly 30 Apple selectors, 4 dot-syntax properties
  (`dev.hasUnifiedMemory`, `dev.maxThreadsPerThreadgroup`,
  `dev.maxThreadgroupMemoryLength`, `opts.mathMode`), 14 Apple types and 5
  constants — none of them checkable on this host. It would also compile a
  different ObjC dialect than AppleClang: a GNUstep runtime, and
  `metal_context.mm:61`'s `@available(macOS 15.0, *)` is a Darwin construct.
  REJECTED: more fabricated surface for less coverage. `§8`'s stop condition
  against making a Linux configure depend on an ObjC++ compiler stands.
- **B — the same macOS job, on every pull request.** Covers 4 of 4 against the
  real SDK and is pre-merge. REJECTED on recurring cost: a macOS runner is
  roughly 10x a Linux one, and this repository takes ~55 pushes/day (`ci.yml:22`).
- **C — the macOS job, on `push` to `main` plus the 4-hourly baseline.**
  CHOSEN. Same 4-of-4 coverage as B and the same zero invented surface, at none
  of B's per-pull-request cost. It shrinks the exposure window from *"until
  somebody cuts a release"* — 955 commits between `7020de936` (v0.0.2,
  2026-08-11) and `be432e8e3`, 2 of them editing these TUs and 28 editing seam
  headers they include (`vt/ops.h` 23, `vt/backend.h` 4, `vt/dtype.h` 1) — to the
  commits since the last completed run. What it gives up is stated in `12.4`.

### 12.3 Design

One job, `macos-metal-mlx` in `.github/workflows/ci.yml`. It is
`release.yml`'s proven `metal_arm64`/`mlx_arm64` shape with the release-only
parts removed: `macos-15`, `actions/checkout`, the same `pip install
'mlx==0.32.0'` and the same `importlib.metadata` resolution of `MLX_ROOT`, and
`scripts/build-macos-release.sh`'s configure flags. `--target vllm` is the whole
difference — this lane compiles, it does not package or execute.

**MLX is ON, and the reason is a measurement.** On release run `31466516224`,
`metal_arm64` took 5m59s and `mlx_arm64` 6m09s. Ten seconds and one `pip
install` buy the fourth TU compiled against the **real** MLX headers, which the
wave-1 stubs are blind to by construction (`§4`). A pip failure reds the job
under its own step name, so it can never be read as a verdict on the code.

**The postcondition is asserted, not assumed.** A green build proves nothing if
`VLLM_CPP_METAL` resolved OFF, if a TU left `target_sources`, or if the object
layout moved: the compiler would have read none of these files and the job would
publish success for a lane covering nothing. The last step therefore requires
all four of `build-metal/CMakeFiles/vllm.dir/src/vt/metal/*.mm.o` to exist and
names the missing one.

**The verdict has to arrive somewhere.** The job joins `baseline-summary`'s
`needs:` and `scripts/main-baseline.py`'s `EXPECTED_JOBS`, with the count pin in
`tests/scripts/test_main_baseline.py` moved 11 → 12. Leaving it out would repeat
#503 exactly: a compiling gate that the published baseline never graded, printing
green because it never ran it.

### 12.4 What this proves, and what it does not

It proves that the four Metal TUs compile against the **real** Apple SDK, the
real Metal and Foundation headers, AppleClang's ObjC++ front end and the real
`mlx` wheel — everything `§4` said wave 1 could not reach. Nothing about it is
stubbed, so `src/vt/metal/stubs/README.md`'s limit is unchanged and untouched:
that document still describes the wave-1 Linux target and is still exactly right
about it.

It is **post-merge**. It cannot stop a break from landing; it names one within a
commit or two of landing instead of at the next release. The wave-1 Linux target
stays, because it is the only Metal signal a pull request gets at all, and it is
free.

It executes nothing. `test_metal_backend` on a real Metal device, and the CUDA
arm beside it, remain #1692's (`§10` O1).

### 12.6 The gate, measured on the lane itself

No Linux host can run this job, so its red-before and green-after are dispatched
runs of `ci.yml` (`gh workflow run ci.yml --ref <branch>`) rather than local
builds. One probe branch per file, each carrying the shipped job and exactly one
mutated file.

| Run | Branch | Mutation | `macos-metal-mlx` | Evidence |
|---|---|---|---|---|
| [`32647402016`](https://github.com/mudler/vllm.cpp/actions/runs/32647402016) | `row/GATE-METAL-MLX-COMPILE-W2` | none | **success**, 5m03s | `compiled build-metal/CMakeFiles/vllm.dir/src/vt/metal/{metal_context,metal_backend,metal_ops,metal_mlx_provider}.mm.o` — all four, and `-- MLX GEMM provider enabled: .../site-packages/mlx/lib/libmlx.dylib` after `Successfully installed mlx-0.32.0 mlx-metal-0.32.0` |
| [`32647406515`](https://github.com/mudler/vllm.cpp/actions/runs/32647406515) | `probe/metal-w2-red-backend` | `RegisterBackend(`, 1 hunk | **failure** | `FAILED: [code=1] CMakeFiles/vllm.dir/src/vt/metal/metal_backend.mm.o`, `src/vt/metal/metal_backend.mm:141:5: error: use of undeclared identifier 'RegisterBackend_RENAMED_BY_MUTATION'` |
| [`32650354752`](https://github.com/mudler/vllm.cpp/actions/runs/32650354752) | `probe/metal-w2-red-context` | `VT_CHECK(`, 6 hunks | **failure** | `FAILED: [code=1] .../metal_context.mm.o`, `src/vt/metal/metal_context.mm:44:5: error: use of undeclared identifier 'VT_CHECK_RENAMED_BY_MUTATION'` |
| [`32650358209`](https://github.com/mudler/vllm.cpp/actions/runs/32650358209) | `probe/metal-w2-red-ops` | `RegisterOp(`, 19 hunks | **failure** | `FAILED: [code=1] .../metal_ops.mm.o`, `src/vt/metal/metal_ops.mm:1092:5: error: use of undeclared identifier 'RegisterOp_RENAMED_BY_MUTATION'` |

Each red fails at step 5 and **skips** step 6, so the postcondition assertion is
not what produced the red — the compiler is. The same three breaks leave the
full Linux build and `ctest` green (`§12.1`). That contrast, per file, is the
whole claim.

**The expected first red did not happen, and that is a result.** 955 commits
after the last macOS build, `main` plus this change still compiles all four TUs
against the real SDK on the first attempt. The drift measured in `§12.2` was
real exposure; it had not yet been converted into a break.

**One thing the runs also measured, unasked:** the macOS runner queue. The four
dispatches waited 45 minutes and 50 minutes for a `macos-15` runner while every
Linux job in the same runs also queued. A pre-merge job of this shape would have
added that latency to every pull request, which is design B's cost expressed in
wall time rather than in dollars.

### 12.5 Gates

| Gate | Command | State |
|---|---|---|
| W1 build graph | occurrences of each `.mm` in `build.ninja` / `compile_commands.json` at `be432e8e3` | 0/0/0 vs 3 (wave 1) and 6 (control) — `§12.1` |
| W2 red-before, Linux | three seam renames, `cmake --build build -j 12` | rc 0, GREEN, i.e. UNDETECTED — `§12.1` |
| W3 red-before, gate | `ctest --test-dir build -j 4` over the same break | 593/595, the 2 reds green on a serial re-run — `§12.1` |
| W4 `-x c++` is closed | wave-1 recipe, `-fsyntax-only`, per file | rc 1, 22/46/166 errors — `§12.1` |
| W5 red-after, CI | one seam rename per file on a probe branch, `gh workflow run ci.yml --ref <probe>` | RED three times out of three, each naming its own file — `§12.6` |
| W6 green-after, CI | the unmutated row branch, same dispatch | GREEN, all four objects compiled — `§12.6` |
| W7 baseline wiring | `python3 tests/scripts/test_main_baseline.py` | 65 tests, rc 0 |
| W8 wiring mutation | drop `- macos-metal-mlx` from `baseline-summary`'s `needs:` | RED on `test_expected_jobs_is_pinned_against_the_workflow_needs_list` |
| W9 preflight | `scripts/agent-preflight.sh` | rc 0 |
| W10 PR lane cost | `macos-metal-mlx` on the pull-request run of this change | `completed/skipped`, 0 s, `runner_name: null` — run `32647478561` |

## 11. Outcome

**Wave 1** put the one Metal TU a Linux compiler can read into every build, and
`§4` states what that does and does not prove.

**Wave 2** put the other three, plus the provider's real MLX dependency, into a
lane that runs within a commit of a merge instead of at the next release tag.
Measured rather than assumed at both ends: three seam renames are invisible to
the full Linux build and `ctest` (`§12.1`), and each of them reds
`macos-metal-mlx` naming its own file (`§12.6`).

**What was rejected.** A Linux clang ObjC++ syntax check (design A) would have
invented ~30 Apple selectors, 14 types, 5 constants and 4 dot-syntax properties,
none checkable on the host that wrote them, and compiled a GNUstep dialect
rather than AppleClang's — for 3 of 4 TUs and seam-level coverage only. The same
macOS job pre-merge (design B) buys the one property C lacks, at ~10x a Linux
runner on ~55 pushes/day plus the 45-50 minute macOS queue measured in `§12.6`.

**Why each default has its value.** `--target vllm` because the row owes
compilation, and executing `test_metal_backend` on a runner whose Metal device is
unestablished is #1692's question. MLX ON because release run `31466516224`
prices it at ten seconds and it is the only lane that ever sees the real MLX API.
The object-existence step because a build that compiled nothing would otherwise
publish success. The `baseline-summary` wiring because #503 already proved that a
gate the baseline cannot see reports green by never running.

**What this does not do, stated once more so no reader has to infer it:** it is
post-merge. A break still lands. It is named a commit or two later instead of at
the next release, and the wave-1 Linux target remains the only Metal signal a
pull request gets.
