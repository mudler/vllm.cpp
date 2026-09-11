ID: ISSUE-GH-132
Title: ROCm: -O0 RmsNorm triggers a CLR HostcallListener teardown deadlock
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: 132
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-07
Updated: 2026-08-10
Closed: 2026-08-10

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Summary
>
> The documented M0/M1 command from #41 builds and executes successfully on `gfx1100`, but `test_backend_cross_device` can hang **after all test cases and assertions pass**. The failure is a ROCm 7.14 CLR `HostcallListener` startup/termination race, exposed because the default no-build-type CMake invocation compiles `rocm_rmsnorm.hip` at `-O0` and ROCm consequently marks the kernels as dynamic-stack users.
>
> This is split from #41 as the concrete gfx1100 M1 blocker. It covers only the shutdown failure; the broader gfx1100/discrete ROCm kernel campaign remains tracked by #41.
>
> I searched the open issues and PRs for `HostcallListener`, `hostcall`, `uses_dynamic_stack`, `test_backend_cross_device` hangs, and the `libamdhip64` finalizer stack and found no existing report. The relevant ROCm source and HIP CMake blocks are unchanged on current `main` at [`19ddceb8`](https://github.com/mudler/vllm.cpp/commit/19ddceb8878ca5ff5b995a930b1db9d4ad20af8b).
>
> ## Environment
>
> - 4× AMD Radeon RX 7900 XTX, all `gfx1100`
> - Linux Mint 22.3, kernel `7.0.0-28-generic`
> - ROCm 7.14.0
> - HIP `7.14.60850-0000000`, AMD Clang 23
> - amdgpu `6.19.14.31400000`
> - CMake 3.28.3
> - Runtime validation commit: [`6a69045b`](https://github.com/mudler/vllm.cpp/commit/6a69045bbc003ee5804faf652bb672a44aae78fa)
> - Current-main source check: [`19ddceb8`](https://github.com/mudler/vllm.cpp/commit/19ddceb8878ca5ff5b995a930b1db9d4ad20af8b)
>
> The runtime validation used a clean checkout. `HIP_PATH`, `HIP_CLANG_PATH`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, and architecture overrides were unset for the documented first run.
>
> ## Reproduction
>
> The exact documented sequence:
>
> ```sh
> cmake -S . -B build-hip -DVLLM_CPP_HIP=ON
> cmake --build build-hip -j
> ctest --test-dir build-hip -R 'rocm|cross_device'
> ```
>
> Configure and build complete successfully. CMake detects all four devices as `gfx1100`. The ROCm-specific tests themselves pass:
>
> - `test_rocm_arch`: 7/7 cases, 40/40 assertions
> - `test_rocm_backend`: 5/5 cases, 1044/1044 assertions
>
> `test_backend_cross_device` prints:
>
> ```text
> [doctest] test cases: 11 | 11 passed | 0 failed | 0 skipped
> [doctest] assertions: 39 | 39 passed | 0 failed |
> [doctest] Status: SUCCESS!
> ```
>
> but the process sometimes never exits. A single-architecture build with `CMAKE_HIP_ARCHITECTURES=gfx1100` reproduces it, so repeated auto-detected architecture flags are not causal.
>
> For a short, higher-probability reproducer on this 48-thread host:
>
> ```sh
> VLLM_CPP_CPU_THREADS=48 timeout 1s build-hip/tests/test_backend_cross_device
> ```
>
> The timeout occurs after the success summary.
>
> ## Root cause
>
> ### 1. Default HIP compilation activates an unnecessary hostcall path
>
> The current HIP setup adds `-ffp-contract=off` but no optimization level ([`CMakeLists.txt:245-276`](https://github.com/mudler/vllm.cpp/blob/19ddceb8878ca5ff5b995a930b1db9d4ad20af8b/CMakeLists.txt#L245-L276)); the ROCm sources are attached at [`CMakeLists.txt:1070-1082`](https://github.com/mudler/vllm.cpp/blob/19ddceb8878ca5ff5b995a930b1db9d4ad20af8b/CMakeLists.txt#L1070-L1082). With no `CMAKE_BUILD_TYPE`, the actual `rocm_rmsnorm.hip` compile command has no `-O` option.
>
> Extracting the `gfx1100` code object from the resulting `rocm_rmsnorm.hip.o` and inspecting its AMDHSA metadata shows this for all eight RmsNorm specializations:
>
> ```text
> .uses_dynamic_stack: true
> .value_kind: hidden_hostcall_buffer
> .value_kind: hidden_heap_v1
> ```
>
> `AMD_LOG_LEVEL=4` then shows the first RmsNorm launch taking this path even though the kernel performs no hostcall:
>
> ```text
> Created hostcall buffer ... numPackets == 3072, size == 12656696
> Launched hostcall listener
> Registered hostcall buffer
> ShaderName : __amd_rocclr_initHeap
> ...
> Terminated hostcall listener
> ```
>
> ### 2. CLR loses the listener termination state
>
> The installed runtime maps to ROCm CLR source commit [`d0a36fc`](https://github.com/ROCm/clr/blob/d0a36fc388d76a8e6659ded35638f3cbbf999def/rocclr/device/devhostcall.cpp#L268-L329). Its handshake is racy:
>
> 1. `initSignal()` starts the listener asynchronously.
> 2. When the listener eventually enters `consumePackets()`, it unconditionally writes `kHostThreadActive.state = kInit` ([line 285](https://github.com/ROCm/clr/blob/d0a36fc388d76a8e6659ded35638f3cbbf999def/rocclr/device/devhostcall.cpp#L282-L286)).
> 3. Under host CPU saturation, `terminate()` can run first, write `kExit`, and signal `SIGNAL_DONE` ([lines 321-329](https://github.com/ROCm/clr/blob/d0a36fc388d76a8e6659ded35638f3cbbf999def/rocclr/device/devhostcall.cpp#L321-L329)).
> 4. The delayed listener overwrites `kExit` with `kInit`, observes `SIGNAL_DONE`, and returns without restoring `kExit` ([lines 305-307](https://github.com/ROCm/clr/blob/d0a36fc388d76a8e6659ded35638f3cbbf999def/rocclr/device/devhostcall.cpp#L305-L307)).
> 5. At DSO finalization, `Init::~Init()` sees stale `kInit`, changes it to `kDestroy`, and spins forever waiting for a listener thread that has already exited ([lines 268-280](https://github.com/ROCm/clr/blob/d0a36fc388d76a8e6659ded35638f3cbbf999def/rocclr/device/devhostcall.cpp#L268-L280)).
>
> The shared state is `volatile`, not atomic or lock-protected. The same source retains `FIXME_lmoriche: fix termination handshake` inside `terminate()`.
>
> GDB on the hung process shows only the main thread plus two HSA async-event threads. The main thread is in `__cxa_finalize`:
>
> ```text
> #0  libamdhip64.so.7 + 0x6c2b05
> #1  __cxa_finalize
> #2  libamdhip64.so.7
>
> mov    (%rdi),%eax
> cmp    $0x2,%eax
> je     <same load>
> *rdi == 2  # Init::State::kDestroy
> ```
>
> ## Controlled A/B
>
> Using `VLLM_CPP_CPU_THREADS=48`, I relinked the same copied library/test with **only** `rocm_rmsnorm.hip.o` changed:
>
> | RmsNorm device object | Metadata | Result |
> |---|---|---:|
> | Original default `-O0` | dynamic stack; hidden hostcall + heap | **14/20 teardown timeouts**, all after 39/39 assertions |
> | Recompiled with `-O1` | `.uses_dynamic_stack: false`; no hidden hostcall/heap | **20/20 clean exits**, 39/39 assertions each |
>
> With `-O1`, `AMD_LOG_LEVEL=4` shows no hostcall buffer, listener, or `__amd_rocclr_initHeap` for RmsNorm.
>
> I also reproduced the same finalizer loop with a standalone raw-HIP program containing a dynamic-stack kernel and 48 busy host workers. It links neither vllm.cpp nor CTest/doctest. It hung after returning from `main` in 3/8 runs; GDB showed the identical `libamdhip64 + 0x6c2b05` loop and only the two HSA event threads. With no busy workers it exited 60/60 times. This isolates the underlying defect to the CLR listener handshake rather than the test framework.
>
> ## Expected behavior
>
> After the test body reports success, the process must terminate normally. A short-lived dynamic-stack kernel must not leave CLR finalization waiting for an already-finished hostcall listener.
>
> ## Proposed fix boundary
>
> Two layers are involved:
>
> 1. **vllm.cpp mitigation:** compile ROCm device kernels with at least `-O1` even when the host build has no `CMAKE_BUILD_TYPE` or is a debug build. For the current RmsNorm kernel this removes the unnecessary dynamic-stack/hostcall/heap path and is validated by the 20/20 A/B above.
> 2. **ROCm definitive fix:** synchronize/atomically order listener startup and termination, prevent a delayed listener from overwriting `kExit`, set `kExit` on the `SIGNAL_DONE` return path, and avoid an unbounded destructor wait for a dead thread. A separate upstream CLR report can link back here.
>
> Reducing `VLLM_CPP_CPU_THREADS` or adding a sleep suppresses the timing window but does not fix the bug and should not be the test resolution.
>
> ## Acceptance criteria
>
> - The documented M0/M1 command terminates reliably on `gfx1100` while retaining 39/39 cross-device assertions.
> - Default/debug ROCm builds do not attach an unnecessary hostcall buffer/hidden heap to the RmsNorm kernel.
> - The resolution is exercised repeatedly on a high-thread-count host rather than accepted from one successful race outcome.
>
> Related: #41
>

## Resolution

Commit `fb6bb8b32e2aadbc61a961e53727f239283eccd9` dated 2026-08-09 keeps `-O0` off the ROCm CLR hostcall path and names issue #132 in its subject. GitHub closed issue #132 on 2026-08-10.
