ID: ISSUE-GH-3100
Title: fix(BACKEND-ROCM): bind resource operations to the requested device
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 3100
Mirror: DIVERGED
Availability: FULL
Created: 2026-09-09
Updated: 2026-09-09
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-ROCM`
>
> ROCm resource methods use the ambient HIP device instead of their recorded device index. At base `6db4bef906859e864c82523c01107473f7dcca29`, `src/vt/rocm/rocm_backend.hip:220-284` calls `hipMalloc`, `hipStreamCreate`, `hipMemsetAsync`, `hipMemcpyAsync`, and `hipStreamSynchronize` without selecting `device_` or the queue device. The backend registers no `DeviceResourceOps` override for `vt::Alloc(Device, ...)`. The introducing source history is `50b0709b3`.
>
> A native MoE boundary test on two local gfx1100 devices exposed this assumption. Device-1-labeled buffers were constructed while device 0 remained current. The operator run returned 1 with three failed comparisons in the null-default-stream case. The other six cases passed. Untouched `0x5555` buffers and invalid reference values identify invalid test resource placement, not a valid native-kernel numerical result. An isolated allocation-device probe remains required before accepting the precise runtime diagnosis.
>
> Owner: `BACKEND-ROCM`, resource and queue device ownership. The new MoE component test must explicitly allocate, copy, and build its reference on each intended device before testing native operation device selection. That scoped harness adaptation does not establish full-model execution on multiple GPUs.
>
> Acceptance requires a committed spec, a red test through the shared resource API with a deliberately different ambient device, verified pointer and stream ownership, preserved ambient-device restoration, and correct default-stream copy and synchronization. Cover two devices, host threads, existing single-device behavior, and graph lifetimes. Run the pinned primary oracle for applicable resource behavior, fresh mutation review, and operator GPU gates. Keep native provider and resource ownership claims separate.
>
> Local reproduction: `/home/vikash/vllm.cpp-rdna3-moe-impl/build-rdna3-moe-hip/evidence/native-boundary-1.log` and its operator receipt. Binary SHA256: `124546db69241940fd5ee88fb26cf71d6053a1dab7ad425915d0422dfc4854f2`. The run held `/home/vikash/gpu.lock`, exposed local devices 0 and 1, and ran on 9 September 2026 UTC.
>

## Resolution

-
