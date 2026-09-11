ID: ISSUE-GH-1478
Title: Native Windows CUDA build rejects unguarded GCC diagnostic pragmas
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: UNKNOWN
GitHub: 1478
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> The first native Windows SM86 build of `SPEC-DFLASH2` W3 passes its repaired CUDA translation units, then fails while compiling `src/vllm/platforms/cuda.cpp` with MSVC `/WX`.
>
> Observed errors:
>
> ```text
> src/vllm/platforms/cuda.cpp(231): warning C4068: unknown pragma 'GCC'
> src/vllm/platforms/cuda.cpp(232): warning C4068: unknown pragma 'GCC'
> src/vllm/platforms/cuda.cpp(243): warning C4068: unknown pragma 'GCC'
> error C2220: warning treated as error
> ```
>
> Commit `cea682956` added unguarded `#pragma GCC diagnostic` directives around the CUDA platform registrar. This repository treats warnings as errors and supports native Windows CUDA, so the GCC-only pragmas require a compiler guard. A separate Qwen3.8 development branch already carries the expected `#if defined(__GNUC__)` shape, but that branch is not main and cannot satisfy this row's immutable build.
>
> Owner: `SPEC-DFLASH2` W3 repair flow. Add a focused compile/contract regression, preserve the GCC suppression, and run the Windows CUDA focused build through link before closing.

## Resolution

-
