# BACKEND-ROCM-STRIX-BUILD-FIX — clear the ROCm 5.7 / clang-17 build failures on Strix (gfx1151)

Issue: `ISSUE-LOCAL-01M336D0AJDJP96NB83NK50HFG`.
Base: `87cfcbf54`.
Measured on: `strix:gpu0` (gfx1151, ROCm 5.7, clang-17), `rc` job, 2026-09-21.

## Scope

The ROCm HIP build fails on Strix (gfx1151, ROCm 5.7, clang-17) with distinct
errors across source and build configuration. Commit `af394746a` armed
`-Werror` for the HIP compile language and `78725a3ce` cleared the 19
diagnostics raised on ROCm 7.2 / clang-20. Clang-17 raises additional
diagnostics not present on clang-20, and ROCm 5.7 lacks APIs and headers that
newer releases provide. The build was confirmed working with all workarounds
applied via `sed` in a build script: 618/618 objects, first-c1 gate passed
(5/5 stability, ~10 tok/s throughput).

PR #3220 (KhazAkar, `row/BACKEND-ROCM-HIP71-BUILD`) addresses the dead-code
subset for HIP 7.1 but is stale (213-file diff from branch divergence) and does
not address the compatibility or build-configuration issues.

IN SCOPE:

- Dead code in `src/vt/rocm/rocm_skinny_gemm.hip`: `on_gfx1151()` (line 55,
  `__device__` function calling host API `hipDeviceGetAttribute`),
  `YtileUnrl` struct (line 82), `SelectYtileUnrl()` (line 83) — no caller since
  #2787 (`984f7265c`). Also the duplicate `mindiv` definition (lines 67-78,
  exact copy of lines 41-52).
- Unused `m_split` variable in `src/vt/rocm/rocm_paged_attn.hip:1333` —
  declared but never read.
- `__shfl_down_sync` / `__shfl_sync` with mask parameter (15 call sites in
  `rocm_grouped_gemm.hip`, `rocm_gdn_fused.hip`, `rocm_quant_dot.hip`) —
  not defined in ROCm 5.7 / clang-17 HIP headers.
- hipBLASLt not installed on all ROCm systems — `rocm_matmul_hipblaslt.hip:15`
  includes `<hipblaslt/hipblaslt.h>` unconditionally. Make the include and the
  source file optional behind a CMake `find_library` check.
- `__HIP_CLANG_ONLY__` — `.hip` files include `hip/hip_bf16.h` and
  `hip/hip_fp16.h` before `hip/hip_runtime.h` (18+ files). On ROCm 5.7 this
  causes `__ockl_*` undeclared errors. Fix the include order so
  `hip_runtime.h` precedes the type headers.
- Multiple definition link errors — `__clang_hip_runtime_wrapper.h` defines
  bfloat16 conversion functions as non-inline in every TU on clang-17. Add
  `-Wl,--allow-multiple-definition` to the HIP linker flags when targeting
  ROCm 5.7, or find and fix the root cause in the include chain.

OUT OF SCOPE:

- CMake 3.28 rejecting `hipcc` — environment-specific (CMake version on Strix).
  Document the `CMAKE_HIP_COMPILER=clang-17` workaround in `docs/ROCM.md`.
- `--rocm-device-lib-path` pointing to a nonexistent path — environment-specific
  (non-standard ROCm layout at `/usr`). Document the required flag in
  `docs/ROCM.md`.
- Any kernel behavior change. Every fix is a deletion of dead code, an include
  reorder, a compatibility wrapper, or a build-configuration guard.
- The Q4_K hang (#2511), which is a separate issue tracked under
  `rocm-gfx1151-q4k-hang.md`.
- PR #3220's stale branch — this spec supersedes the dead-code subset of that
  PR and extends it to the compatibility and build-configuration issues.

## Design

### Dead code removal

Delete `on_gfx1151()`, `YtileUnrl`, `SelectYtileUnrl()`, and the duplicate
`mindiv` (lines 67-78) from `rocm_skinny_gemm.hip`. The gfx1151 tile table is
recoverable from git history (`git log -S'SelectYtileUnrl'`). Delete the unused
`m_split` declaration at `rocm_paged_attn.hip:1333`.

### `__shfl_down_sync` / `__shfl_sync` compatibility

The `_sync` variants are CUDA-compatibility wrappers. HIP's native API is
`__shfl_down(val, offset)` / `__shfl(val, lane)` (no mask, no `_sync`). On
ROCm 5.7 / clang-17, the `_sync` wrappers are not defined in the HIP headers.

Approach: add a compatibility header (e.g. `include/vt/rocm/hip_shfl_compat.h`)
that defines `__shfl_down_sync` and `__shfl_sync` as macros wrapping the native
`__shfl_down` / `__shfl` when they are not already defined. Guard with
`#ifndef __shfl_down_sync`. Include it from the three affected `.hip` files.

Alternatively, if a HIP version check is available and reliable, use
`#if HIP_API_VERSION < N` to select the API form. The implementer determines
which approach is cleaner and verifies it builds on both ROCm 5.7 (Strix) and
the CI's ROCm version.

### hipBLASLt optionality

In `CMakeLists.txt`, wrap the hipBLASLt source and include with
`find_library(HIPBLASLT_LIB hipblaslt)` and guard with `if(HIPBLASLT_LIB)`.
In `rocm_matmul_hipblaslt.hip`, guard the `#include <hipblaslt/hipblaslt.h>`
with a preprocessor check (`#ifdef VLLM_CPP_HAS_HIPBLASLT`) and provide a
stub or early-return path when hipBLASLt is absent. The CMake check defines
`VLLM_CPP_HAS_HIPBLASLT` when the library is found.

Note: the stub must define `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` (with
`MATMUL_`), not `HIPBLASLT_PREF_MAX_WORKSPACE_BYTES`. This was the root cause
of a previous stub error.

### Include order fix

In all `.hip` files that include `hip/hip_bf16.h` or `hip/hip_fp16.h` before
`hip/hip_runtime.h`, swap the order so `hip_runtime.h` comes first. This is
a mechanical change affecting 18+ files. The implementer verifies the build
is green on both ROCm 5.7 and the CI's ROCm version.

### Multiple definition linker fix

Add `-Wl,--allow-multiple-definition` to the HIP linker flags in
`CMakeLists.txt`, guarded by a ROCm version check or applied unconditionally
(the flag is harmless when there are no duplicate definitions). Alternatively,
investigate whether reordering includes or adding `inline` to the offending
functions in a compatibility header eliminates the root cause.

## Tests

- The CPU build (`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`) must
  remain green: `ctest --test-dir build -j` with zero regressions.
- The HIP build on Strix (`strix:gpu0`, ROCm 5.7, clang-17) must compile
  618/618 objects and link cleanly.
- The first-c1 gate on Strix must pass: `vllm-cli --model Qwen3-4B --prompt
  "The capital of France is" --max-tokens 1 --temperature 0` exits 0 with
  output " Paris".
- No CI regression: all existing CI checks must remain green (excluding
  pre-existing failures on `main`).

## Gates

- **Correctness**: first-c1 token-exact gate on Strix (gfx1151, ROCm 5.7).
  Model: Qwen3-4B BF16. Prompt: "The capital of France is". Expected: " Paris".
  5/5 stability runs.
- **Build**: 618/618 HIP objects compiled, 0 errors, 0 warnings under `-Werror`.
  Link succeeds.
- **CI**: all checks green or failing only from inherited causes.

## Evidence

- Build script with all 10 workarounds confirmed working: `/tmp/strix-full-build-gate-v16.sh`
- Build result: 618/618 objects, ~8 min build time
- First-c1: exit 0, output " Paris", 0.264s
- Stability: 5/5 PASS, 0.124-0.487s per run
- Throughput: 10.334, 9.168, 9.580 tok/s (avg ~9.7 tok/s)
- Memory: `strix_rocm_build_recipe` (all 10 fixes documented)

## Risks

- The include-order swap touches 18+ files. A missed file or wrong order
  could break the build on newer ROCm. Mitigation: test on both Strix (5.7)
  and a newer ROCm if available; CI runs on newer ROCm configurations.
- The `__shfl_down_sync` compatibility wrapper might mask a genuine API
  difference. Mitigation: the wrapper only activates when the native macro is
  absent, so newer ROCm uses its own definition.
- hipBLASLt optionality changes the build surface. Mitigation: guard with
  CMake `find_library`, and the stub path is only active when the library is
  absent.

## Stop conditions

- The include-order swap breaks the CI build on a newer ROCm version and the
  fix is non-obvious. Stop and escalate.
- The `__shfl_down_sync` compatibility wrapper produces wrong results at
  runtime (not just a compile issue). Stop and escalate.
- hipBLASLt optionality changes the behavior of the matmul path on systems
  that DO have hipBLASLt. Stop and verify byte-identical output.

## Git integration

One pull request (repository default). Spec committed before implementation.

## Now

Implementing the Strix ROCm 5.7 / clang-17 build fixes. The worktree is
`/home/mudler/worktrees/strix-build` on branch `row/BACKEND-ROCM-STRIX-BUILD`.
Issue: `ISSUE-LOCAL-01M336D0AJDJP96NB83NK50HFG`.
