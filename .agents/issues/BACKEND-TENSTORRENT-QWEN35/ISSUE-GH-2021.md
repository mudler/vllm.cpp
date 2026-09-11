ID: ISSUE-GH-2021
Title: qwen3_5.cpp: -Werror=nonnull on GCC 15 breaks every CPU build (Tenstorrent debug dump path)
Row: BACKEND-TENSTORRENT-QWEN35
State: CLOSED
Kind: bug
GitHub: 2021
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-26
Updated: 2026-08-31
Closed: 2026-08-31

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-QWEN35`
>
> A fresh CPU build of `main` (`2a42cb369`) fails under GCC 15.2.0 with
> `-Werror=nonnull`:
>
> ```
> src/vllm/model_executor/models/qwen3_5.cpp: In function
> 'vllm::{anonymous}::GdnQkvzOutput vllm::{anonymous}::ProjectGdnQkvz(...)':
> src/vllm/model_executor/models/qwen3_5.cpp:4019:24: error: argument 1 null
> where non-null expected [-Werror=nonnull]
>  4019 |             std::fwrite(vec.data(), 4, vec.size(), f2);
>       |             ~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
> ```
>
> [`qwen3_5.cpp:4014-4021`](https://github.com/mudler/vllm.cpp/blob/main/src/vllm/model_executor/models/qwen3_5.cpp#L4014-L4021)
> (the Tenstorrent `DebugDeviceReadbackF32` dump path, landed in
> `eec5aa16f`/#1715): `vec` is a `std::vector<float>` that can be empty, in
> which case `vec.data()` may return `nullptr`, and `std::fwrite`'s first
> parameter is declared `nonnull`. GCC 15 flags this statically regardless of
> the actual runtime size (a size-0 `fwrite` on a null pointer is well-defined
> by the C standard, but the `nonnull` attribute makes the call itself
> undefined behavior by the letter of the standard, which is what the warning
> is about).
>
> ## Impact
>
> This is a hard build break — `-Werror` promotes it to a compile failure — on
> any CPU build using GCC 15, which blocks every downstream CPU verification
> (unit tests, gates) regardless of what else changed. Found while building
> `main` fresh for an unrelated row (#1934).
>
> ## Fix
>
> Guard the write on `!vec.empty()`, mirroring the pattern the raw-bytes write
> three lines above already uses (`if (f)` guards the file handle; this needs
> the same shape for the data). Debug-only code path (gated behind the `call ==
> 0` / `qdir` debug-dump condition), so there is no behavior to preserve beyond
> "do not write when there is nothing to write", which is already true of an
> empty vector today logically, just not provably so to the compiler.
>
> Owned by `BACKEND-TENSTORRENT-QWEN35`.

## Resolution

The binding comment https://github.com/mudler/vllm.cpp/issues/2021#issuecomment-5483631362 dated 2026-08-31 verifies commit `7f368c62a` guards the empty-vector write and closes the GCC 15 failure.
