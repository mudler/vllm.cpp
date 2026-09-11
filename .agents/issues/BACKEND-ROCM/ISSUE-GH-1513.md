ID: ISSUE-GH-1513
Title: ROCm MoeSiluMul bf16 is not bit-exact against the CPU reference on gfx1200 (test_backend_cross_device:2063)
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1513
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> `test_backend_cross_device` fails one assertion on gfx1200 / ROCm 7.2.3:
>
> ```
> tests/vt/test_backend_cross_device.cpp:2063: ERROR: CHECK( got == ref_b ) is NOT correct!
> [doctest] test cases:  21 |  20 passed | 1 failed | 0 skipped
> [doctest] assertions: 360 | 359 passed | 1 failed |
> ```
>
> Line 2063 is the **bf16** arm of the `kMoeSiluMul` case, which asserts exact
> equality against the CPU reference on the stated grounds that a single multiply
> plus an RNE store is exact on both sides:
>
> ```cpp
> vt::MoeSiluMul(q, tout, tg, tu);
> std::vector<uint16_t> got(n);
> doutb.Download(got.data());
> CHECK(got == ref_b);  // single multiply + RNE store both sides: exact
> ```
>
> So either the ROCm `kMoeSiluMul` bf16 path is not doing a single multiply with
> an RNE store, or that comment's premise does not hold on this device. The f32
> arm of the same case passes.
>
> `kMoeSiluMul` is registered for ROCm at `src/vt/rocm/rocm_ops.hip:182`; the
> kernel is in `src/vt/rocm/rocm_moe_router.hip`.
>
> ## Pre-existing, with a control
>
> Found while gating an unrelated row (#1506, the ROCm keep-quant GEMM). It is
> **not** caused by that work, and the control is empirical rather than
> argumentative: removing the new `src/vt/rocm/rocm_quant_dot.hip` from the build
> entirely and rebuilding reproduces the identical failure, same line, same
> 359/360. The new TU also contains zero references to `MoeSiluMul`.
>
> Not fixed in that flow because it is not a small clear fix: it is a numeric
> divergence in a different kernel family that wants its own investigation and
> its own red-first gate. Owned by `BACKEND-ROCM`.
>
> ## Repro
>
> ```
> nix develop .#rocm-shell --command bash -c \
>   'cmake -S . -B build-hip -G Ninja -DVLLM_CPP_HIP=ON \
>      -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200 -DROCM_PATH=$ROCM_PATH \
>      -DCMAKE_BUILD_TYPE=Release && \
>    cmake --build build-hip --target test_backend_cross_device -j 6 && \
>    ./build-hip/tests/test_backend_cross_device'
> ```
>

## Resolution

-
