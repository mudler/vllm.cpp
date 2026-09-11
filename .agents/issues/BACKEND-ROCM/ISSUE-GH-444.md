ID: ISSUE-GH-444
Title: ROCm: main does not build for gfx1200/gfx1201 — rocm_paged_attn.hip includes rocwmma unconditionally on arch, not availability
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: 444
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-12
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> ## Summary
>
> `main` does not build for ROCm in this repo's own `nix develop .#rocm-shell`
> when the target arch is gfx1200/gfx1201. The build stops at:
>
> ```
> src/vt/rocm/rocm_paged_attn.hip:8:10: fatal error: 'rocwmma/rocwmma.hpp' file not found
>     8 | #include <rocwmma/rocwmma.hpp>
> ```
>
> Introduced by `9302732f` ("feat(gemma4/rocm): FP8 resident MoE + SharedK-WMMA",
> 2026-08-11). Reproduced on **pristine `upstream/main` at `0f2b12ed`** in a clean
> worktree with no other commits applied, so this is not a local-branch artifact.
>
> ## Reproduce
>
> ```sh
> git worktree add /tmp/mainhip upstream/main
> cd /tmp/mainhip
> nix develop .#rocm-shell -c cmake -S . -B build-hip -DVLLM_CPP_HIP=ON \
>   -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200 -DROCM_PATH=$ROCM_PATH -DCMAKE_BUILD_TYPE=Release
> nix develop .#rocm-shell -c cmake --build build-hip --target vllm -j8
> ```
>
> Board: AMD Radeon RX 9060 XT, gfx1200, Navi 44, RDNA4, discrete. ROCm 7.2.3,
> hipClang/Clang 22.0.0, NixOS.
>
> ## Why it happens
>
> The include is gated on **architecture**, not on **availability**
> (`src/vt/rocm/rocm_paged_attn.hip:6-10`):
>
> ```c
> // rocWMMA only on gfx12 (R9700). Fat builds also emit gfx1036 — must not include there.
> #if defined(__gfx1200__) || defined(__gfx1201__)
> #include <rocwmma/rocwmma.hpp>
> #define VT_ROCWMMA_OK 1
> #endif
> ```
>
> Targeting gfx1200 satisfies the guard, so the include fires whether or not
> rocWMMA exists. On a stock ROCm install it ships alongside HIP and this is
> invisible. In this repo's dev shell it is absent — the `clr` derivation the shell
> provides carries no `rocwmma/`, nothing in `/nix/store` matches `*rocwmma*`, and
> `flake.nix` does not mention it. Nothing in `CMakeLists.txt` or `cmake/` detects
> or requires it either, so configure succeeds and the failure lands mid-build.
>
> `docs/ROCM.md`, `flake.nix` and `.agents/environment.md` do not list rocWMMA as a
> prerequisite, so a contributor following the documented ROCm path has no warning.
>
> ## Please do NOT fix this with `__has_include` alone
>
> `__has_include(<rocwmma/rocwmma.hpp>)` does work here — verified under `hipcc`,
> it correctly reports absent — but swapping the arch guard for it would turn a
> loud build failure into a **silent correctness bug**.
>
> `VT_ROCWMMA_OK` also selects the kernel body. At
> `rocm_paged_attn.hip:900`, `PagedAttnPrefillWmmaWave` compiles to a no-op when
> the macro is undefined — every parameter cast to `(void)` and an immediate
> `return;` (that path exists for fat builds that also emit gfx1036, where the
> kernel is never dispatched). If availability rather than arch decided the macro,
> a gfx1200 build without rocWMMA would compile cleanly and produce a paged
> attention kernel that writes nothing.
>
> `grep -rn VT_ROCWMMA_OK src/` shows the macro is confined to
> `rocm_paged_attn.hip`, so no host-side dispatch code can currently gate on it
> either.
>
> Safer shapes, in rough order of preference:
>
> 1. **Detect in CMake and fail configure** with a message naming the package,
>    when a gfx12 target is requested and rocWMMA is missing. Loud, early, and
>    keeps the arch guard's meaning intact.
> 2. **Add rocWMMA to `flake.nix`'s `rocm-shell`** so the documented path works out
>    of the box. Complements (1) rather than replacing it — it fixes this shell,
>    not other environments.
> 3. **Detect availability *and* route dispatch away from the WMMA kernel**, so the
>    no-op body is unreachable rather than merely unused. More work, and only worth
>    it if a rocWMMA-free gfx12 build is meant to be supported at all.
>
> At minimum, documenting rocWMMA as a gfx12 build prerequisite in `docs/ROCM.md`
> would stop the next person losing the same hour.
>
> ## Scope
>
> - Only `src/vt/rocm/rocm_paged_attn.hip` includes rocWMMA (`grep -rln rocwmma src/`).
> - Blocks any ROCm build targeting gfx1200/gfx1201 in an environment without
>   rocWMMA, including this repo's own dev shell.
> - Other gfx targets are unaffected: the guard is false, so the include never
>   fires.
> - No open issue mentions rocWMMA (searched `rocwmma in:title,body`).
>

## Resolution

-
