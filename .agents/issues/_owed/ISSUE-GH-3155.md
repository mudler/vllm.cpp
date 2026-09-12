ID: ISSUE-GH-3155
Title: sanitize-cpu: LoadUnaligned in cpu_matmul_elem_avx2/avx512 triggers UBSan alignment error
Row: -
State: CLOSED
Kind: UNKNOWN
GitHub: 3155
Mirror: SYNCED
Availability: FULL
Created: 2026-09-12
Updated: 2026-09-12
Closed: 2026-09-12

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: -
>
> `sanitize-cpu (address,undefined)` fails on multiple PRs (#3140, #2906, likely all) with a pre-existing UBSan alignment error:
>
> ```
> src/vt/cpu/cpu_matmul_elem_avx2.cpp:231:33: runtime error: load of misaligned address for type const T, which requires 2 byte alignment
> src/vt/cpu/cpu_matmul_elem_avx512.cpp:121:37: runtime error: load of misaligned address for type const T, which requires 2 byte alignment
> ```
>
> `LoadUnaligned<T>` (include/vt/unaligned.h:14-18) uses `std::memcpy`, which is the correct approach for unaligned loads. However, the compiler optimizes `memcpy` of a small trivially copyable type into a direct load instruction, which UBSan then flags as an alignment violation.
>
> This is pre-existing: it appears on #2906 (Sept 5) and #3140 (Sept 11), independent of their changes. It blocks all PRs that run `sanitize-cpu`.
>
> Potential fixes:
> - Add `__attribute__((no_sanitize("alignment")))` to `LoadUnaligned` (may not work if inlined)
> - Mark the function `noinline` to preserve the attribute
> - Use a union-based approach instead of memcpy
> - Build sanitizer with `-fno-builtin-memcpy`

## Resolution

Fixed by adding VT_NO_SANITIZE_ALIGNMENT to LoadUnaligned (PR #3156)
