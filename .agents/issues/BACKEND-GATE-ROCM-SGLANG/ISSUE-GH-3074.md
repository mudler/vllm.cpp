ID: ISSUE-GH-3074
Title: fix(BACKEND-GATE-ROCM-SGLANG): isolate offload inspection from installed kernels
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3074
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-08
Updated: 2026-09-08
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-GATE-ROCM-SGLANG`
>
> Owner: current Strix benchmark campaign operator. Found during #3053 preparation retry fb77c7b0-4fa8-4e69-80d9-115845f6922e. The aab36a6a4 HIP identity fix passes, kernel compilation and initial sgl_kernel import pass, but upstream tests fail collection. The extension-targets.log proves llvm-objdump --offloading extracts host/HIP bundles beside the installed common_ops extension. Pinned SGLang load_utils.py glob common_ops.* accepts names containing .so and attempts only the first candidate, which is an extracted non-importable bundle.
>
> Fix the preparation caller to inspect an exact verified copy in an isolated scratch directory, preserve the installed extension/package bytes and candidate inventory, retain gfx1151-only validation, and add a CLI regression with an inspector that creates adjacent artifacts. No upstream loader patch or pin change is needed. Use committed design before a fresh implementation and independent mutation review. Evidence directory: /mnt/nas_share/rc/strix-four-engine-3053.X94a3J/prepare-aab36a6a-rebuilt-03/logs. #3053 owns downstream kernel tests, AITER/model startup and benchmarks.

## Resolution

-
