# BACKEND-ROCM-HIP-WERROR — give the HIP compile the `-Werror` every other project language already has

Issue: [#2713](https://github.com/mudler/vllm.cpp/issues/2713).
Base: `ca07f6e94`.
Measured on: `strix:gpu0` (gfx1151, ROCm/HIP 7.2.53211-97f5574fe2), `rc` job
`60f7861d-789f-4cb8-b6fa-b057da83347d`, 2026-09-03.

## Scope

`vllm_cpp_set_warnings` (`cmake/CompilerWarnings.cmake`) hands `-Werror` to
`CXX`, `OBJCXX` and `CUDA` through `COMPILE_LANGUAGE` generator expressions. It
has no `HIP` branch. `CMakeLists.txt:404` calls `enable_language(HIP)` and
`CMakeLists.txt:1711` adds 23 `.hip` sources to the `vllm` target, which
`CMakeLists.txt:1484` passes to `vllm_cpp_set_warnings`. So HIP is a distinct
compile language in this build and it is the only project language whose
diagnostics cannot fail a build.

IN SCOPE:

- One `$<$<COMPILE_LANGUAGE:HIP>:...>` line in `cmake/CompilerWarnings.cmake`.
- Every diagnostic that line then raises, cleared at its source.

OUT OF SCOPE:

- `CXX`, `OBJCXX` and `CUDA`, whose arms are unchanged.
- Any behaviour of any kernel. Every repair below is a deletion of code nothing
  calls, an explicit parenthesisation that matches the existing parse, an
  explicit cast that matches the existing comparison, or a `(void)` on a value
  the call site already discards on purpose.
- The `EnsureQuantScratch` / `LtWorkspace` synchronisation defects
  ([#2712](https://github.com/mudler/vllm.cpp/issues/2712)), which touch two of
  the same files and land on their own branch.

## The count, first

The issue asks for the number before the repair, because a large number would
make the flag a build break for everyone rather than a detector. Two arms were
built on `strix:gpu0`, each a fresh configure into a fresh build directory,
each compiling all 23 HIP objects (`ninja rc=0`, 23 `Building HIP object` lines,
23 objects produced):

| Arm | HIP flags | raw warning lines | unique `file:line:col:flag` |
|---|---|---|---|
| A | as-committed (clang defaults, no `-Wall`) | 6 | **3** |
| B | `-Wall -Wextra`, `-Werror` deliberately absent | 46 | **19** |

Zero of the 19 are in a ROCm or system header. All 19 are project code. The
build system was asked directly, not the cmake source: on arm A a HIP compile
line carries `-Wall` 0 times and `-Werror` 0 times; on arm B it carries `-Wall`
once and `-Werror` still 0 times. That is #2713's claim measured rather than
read.

Arm B by warning class:

| Class | Unique | Disposition |
|---|---|---|
| `-Wunused-function` | 10 | dead code, deleted |
| `-Wunused-value` | 3 | ignored `nodiscard` `hipError_t`, made explicit |
| `-Wunused-const-variable` | 2 | dead constants, deleted |
| `-Wlogical-op-parentheses` | 2 | parse already correct, parenthesised |
| `-Wunused-parameter` | 1 | dead kernel parameter |
| `-Wsign-compare` | 1 | explicit cast |

19 is small, so the repair is "add the flag and clear them", not "report the
size and stop".

## The ten `-Wunused-function` reports are real, and the line numbers prove it

A `__device__` function is not emitted in the host pass, so "clang reports a used
`__device__` helper as unused in the host pass" is the obvious false-positive
hypothesis, and it is wrong here. Each of `rocm_gdn_conv.hip`,
`rocm_gdn_fused.hip` and `rocm_gdn_postconv.hip` declares three `Ld` overloads
and three `St` overloads on consecutive lines — `float`, `__half`,
`__hip_bfloat16`. Exactly one `Ld` line and one `St` line is reported in each
file, and in all three files it is the `__half` overload. The `float` and
`__hip_bfloat16` overloads on the neighbouring lines are silent because the
kernels instantiate them. A pass artefact would report all six.

The other four are single-mention symbols: `DotFp8Row`, `LoadRow8Bf16`,
`StoreRow8Bf16` and one `GridFor` appear in their file exactly once, at their own
definition.

## Design

Add the HIP arm beside the three that exist, using `${_vllm_cpp_werror}` rather
than a literal `-Werror`. The literal is what the `OBJCXX` line uses, and it is
wrong for HIP: `${_vllm_cpp_werror}` is emptied on the sanitizer lanes, and a HIP
build under a sanitizer has the same reason to keep the diagnostics visible and
non-fatal that the CXX arm has.

`${_vllm_cpp_array_bounds}` is not carried over. It exists for GCC 15 and later;
the HIP compiler is clang.

## Tests

The gate for a compile-flag change is the compile. It runs on `strix:gpu0`,
because no other host here has a HIP toolchain.

1. **RED** — the flag added, the 19 diagnostics not yet cleared. The build must
   fail, and it must fail on those 19.
2. **GREEN** — the flag added, the 19 cleared. All 23 HIP objects build.
3. **MUTATION** — reinstate the exact defect the issue names: a bare
   `hipError_t` call, in a `.hip` translation unit, through this project's own
   cmake target. It must compile clean on the base tree and fail the build on
   this one. Rebuilt, not reasoned about.

## Stop conditions

- Stop and report the number if a repair for any of the 19 would change what a
  kernel computes. None does; that is asserted per site, not in aggregate.
- Stop if the count on a second `gfx` target is materially larger. Not measured;
  see `## Owed`.

## Owed

- The count is from one architecture (`gfx1151`) and one ROCm release (7.2.5).
  `-Wall -Wextra` output is target-independent for these six classes, but that is
  reasoning, not a measurement, and no second target was built.
- CI cannot rerun this gate. No CI lane has a HIP toolchain, so the flag is
  enforced only where somebody builds the ROCm backend. The regression this
  leaves open is deletion of the genex, which nothing here detects.

## Evidence

`rc` job `c74c0cdf-804f-4666-a155-00fa0f59b963` on `strix:gpu0`, 2026-09-03,
four arms, each a fresh configure into a fresh build directory, each asked to
build the same 23 HIP objects with `ninja -k 0`. The arms were archived from
commits whose messages were later reworded to repair an `Assisted-by` value; the
trees are byte-identical (`da7b2417…` for RED, `38d96fce…` for GREEN), so the
result binds to the commits on this branch.

| Arm | Tree | `-Werror` on a HIP compile line | ninja rc | objects | distinct errors | Result |
|---|---|---|---|---|---|---|
| RED | flag added, nothing cleared | 1 | 1 | 14/23 | 15 | BUILD_FAILED |
| GREEN | this branch | 1 | 0 | 23/23 | 0 | BUILD_OK |
| MUT | this branch + a bare `hipError_t` | 1 | 1 | — | ≥1 | BUILD_FAILED |
| CTRL | base tree + the SAME bare `hipError_t` | 0 | 0 | 23/23 | 0 | BUILD_OK |

MUT and CTRL apply the identical one-line mutation to `rocm_backend.hip`. On this
branch it reads `error: ignoring return value of type 'hipError_t' declared with
'nodiscard' attribute [-Werror,-Wunused-value]`; on the base tree the same line
reads `warning:` and the build goes green. That is #2713's claim mutated and
rebuilt, with the control that shows the new flag is the detector rather than
something else in the diff.

**RED reports 15, not 19, and the four missing ones are not a discrepancy.** All
four — `rocm_paged_attn.hip:30`, `:335`, `:448`, `:451` — are in the single
translation unit that hard-fails on the `-Wlogical-op-parentheses` error at
`:2301`, and all four are end-of-translation-unit diagnostics
(`-Wunused-function`, `-Wunused-const-variable`) that clang never reaches once
the frontend has errored. The GREEN arm is what shows all 19 were cleared: 23 of
23 objects, zero errors and zero warnings.

The job's "red classes" tally line printed nothing. Its regex expects `[-Wflag]`
and the `-Werror` form is `[-Werror,-Wflag]`. The per-error list above it is
complete and is the record; the tally is cosmetic and was not relied on.
