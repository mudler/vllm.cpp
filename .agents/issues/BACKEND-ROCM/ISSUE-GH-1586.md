ID: ISSUE-GH-1586
Title: Adopt the ROCm 7.14 container toolchain and open the gfx1100 optimization campaign
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 1586
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-28
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Context
>
> The local development box carries an AMD Radeon RX 7900 XTX (gfx1100, RDNA3,
> 24 GiB) on kernel 6.8.0-138. The host has ROCm 7.2.4 installed. The developer
> directed adoption of ROCm 7.14.0 through container images instead of a host
> install, and directed that `ninja` live in the container rather than on the
> host.
>
> ROCm 7.14.0 is the first production TheRock release (15 July 2026). The
> compatibility matrix lists gfx1100 as supported.
>
> ## Verified setup
>
> - Image `rocm-dev:7.14.0`: hipcc at `/opt/rocm/core-7.14/lib/llvm/bin`,
>   cmake 3.28.3, ninja 1.11.1, python 3.12.3.
> - Device passthrough with `--device=/dev/kfd --device=/dev/dri` works;
>   `rocminfo` reports gfx1100 inside the container.
> - Configure is green with `-DVLLM_CPP_HIP=ON` on a read-only mount of the
>   repository; hipBLAS and hipBLASLt resolve from the image. The TheRock
>   layout needs the compiler-root hints that `CMakeLists.txt` derives from
>   `ROCM_PATH`.
> - OpenSSL is absent from the image, so `--model org/repo` cannot reach
>   `https://` endpoints until the image adds `libssl-dev` or the build sets
>   `-DVLLM_CPP_BUILD_BORINGSSL=ON`.
>
> ## Scope
>
> 1. Baseline: full build on 7.14, then `ctest --test-dir <build> -R
>    'rocm|cross_device'` under `flock ${GPU_LOCK}` on the host.
> 2. Re-run every focused gate that earlier records ran on 7.2.x-era
>    toolchains. A toolchain swap invalidates carried build evidence.
> 3. Open the optimization campaign on the registered ROCm ops (matmul,
>    paged attention, GDN family, MoE router, sampling): trace both sides
>    with the same tool on an identical workload before any throughput
>    claim, per `.agents/benchmarking.md`.
>
> ## Out of scope
>
> - A pinned vLLM-ROCm oracle. `BACKEND-GATE-ROCM-VLLM` stays `INVENTORIED`
>   until a revision is recorded under `.agents/oracles/`. The local image
>   `vllm/vllm-openai-rocm:latest` is a floating tag and is not an oracle.
> - Any lifecycle transition of `BACKEND-ROCM`. This issue opens work; it
>   does not move the row.
>
> ## Records
>
> Toolchain facts and the container workflow are recorded in
> `.agents/developer-preferences.md` (untracked) in the local workspace.
>

## Resolution

-
