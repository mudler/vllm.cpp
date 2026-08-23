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
  technique does not extend to them.
- The MLX API surface. The stubs are ours, so the gate cannot fail for an MLX
  reason. `§4` states that limit as a limit.
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

Spec and implementation in one pull request. G0-G6 measured on this host. G7 and
G8 hold by construction and are verified by reading the guard, not by running an
MSVC or an Apple build.

## 10. Owed

| ID | What | Issue |
|---|---|---|
| O1 | The two **runtime** arms: `test_ops_attention_cross` on a CUDA device and `test_metal_backend` on a `VLLM_CPP_MLX` build, plus the `.agents/reachability.md` mutation on `BlockedFallback()` / `MlxFallback()`. Unchanged by this row, which buys compile coverage only. `§2` narrows the Metal half — that binary IS executed by `build-macos-release.sh:46`, but last did so at `v0.0.2` on 2026-08-11, before the change it is owed for, and whether the runner had a Metal device at all is unestablished | [#1692](https://github.com/mudler/vllm.cpp/issues/1692) |
| O2 | The MLX **API** surface. Covered only by `mlx_arm64` at release time; this gate is blind to it by construction (`§4`) | [#1765](https://github.com/mudler/vllm.cpp/issues/1765) |
| O3 | The other three Metal TUs, which carry real Objective-C and are reached by no pre-merge job either (`§2`). This row does not extend to them and no row yet owns them | [#1765](https://github.com/mudler/vllm.cpp/issues/1765) |

## 11. Outcome

To be recorded when the row reaches `DONE`.
