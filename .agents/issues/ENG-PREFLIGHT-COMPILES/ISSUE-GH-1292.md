ID: ISSUE-GH-1292
Title: main does not build on gcc 15.x — the -Warray-bounds error demotion is gated at GCC >= 16
Row: ENG-PREFLIGHT-COMPILES
State: CLOSED
Kind: bug
GitHub: 1292
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-18
Updated: 2026-08-21
Closed: 2026-08-21

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `cmake/CompilerWarnings.cmake:39-42` demotes `-Warray-bounds` from fatal to
> visible for GCC >= 16 only. Its comment states the intent for everything below:
>
> > Everything <= 15 is unchanged and still fails the build on a real out-of-bounds.
>
> On gcc 15.2.0 the same false-positive class fires, so the tree does not compile
> at all.
>
> ## Reproduction
>
> Current `main` (`4ee5f4a6`), gcc 15.2.0, project flags:
>
> ```
> g++ -Iinclude -Isrc -isystem third_party -O3 -DNDEBUG -std=c++20 -fPIC \
>     -ffp-contract=off -Wall -Wextra -Werror -c \
>     src/vllm/model_executor/models/ltx2_samplers.cpp
> ```
>
> ```
> src/vllm/model_executor/models/ltx2_samplers.cpp:161:41: error: array subscript -1 is outside array bounds of 'float [2305843009213693951]' [-Werror=array-bounds=]
> src/vllm/model_executor/models/ltx2_samplers.cpp:163:16: error: array subscript -1 is outside array bounds of 'float [2305843009213693951]' [-Werror=array-bounds=]
> ```
>
> A full `cmake --build` stops at 146/1552.
>
> ## Why it is a false positive
>
> The bound `float [2305843009213693951]` is `SIZE_MAX/4` — the allocator's
> unconstrained size range, not a real object. GCC attributes the subscript check
> to `std::vector<float>`'s inlined copy-construct plus `_M_allocate`, the same
> inlining-attribution mechanism the existing comment documents for
> `_Sp_counted_base::_M_release()`, reached here through the other libstdc++
> container.
>
> The subscripts are `sigmas.back()` at `ltx2_samplers.cpp:161,163`, guarded three
> lines above by:
>
> ```cpp
> VT_CHECK(sigmas_in.size() >= 2, "ltx2 res2s loop: a schedule needs at least two sigmas, got " + ...);
> std::vector<float> sigmas = sigmas_in;
> const bool terminal_zero = sigmas.back() == 0.0f;
> ```
>
> `sigmas` is a copy of a container proven non-empty, so `.back()` is well
> defined. GCC does not propagate the throwing check through the inlined copy.
>
> ## Why CI does not see it
>
> The compiler matrix has exactly two points:
>
> | Lane | Compiler |
> |---|---|
> | every ordinary Linux job | gcc 13 (ubuntu-latest distro `g++`) |
> | `build-newest-gcc` | gcc 16 (`container: gcc:16`) |
>
> gcc 14 and 15 are uncovered. gcc 16 is already exempt via the guard, and gcc 13
> does not emit the diagnostic, so the failure falls exactly between the two
> lanes. gcc 15 is the current toolchain on Arch, NixOS unstable and Fedora 42.
>
> Introduced by `4d774864` (`LTX25-RES2S-LOOP`, 2026-08-17), which added the loop;
> the guard predates it and was written for the `shared_ptr`/json case.
>
> ## Fix
>
> Widen the existing guard to `VERSION_GREATER_EQUAL 15`. The diagnostic stays
> visible as a warning, which is what the guard is for. Verified: with
> `-Wno-error=array-bounds` appended the same two diagnostics appear as warnings
> and the object builds.
>
> Whether CI should also cover gcc 14/15 is a separate question and is
> deliberately not bundled here.
>
> ## Environment
>
> - gcc 15.2.0, x86_64, NixOS
> - `main` at `4ee5f4a6`
>

## Resolution

GitHub records closing pull request #1293 (https://github.com/mudler/vllm.cpp/pull/1293) merged on 2026-08-21 as commit `e05911ff2bc8d7041dfa2b5ee14cb34827412c06`. GitHub closed issue #1292 on 2026-08-21.
