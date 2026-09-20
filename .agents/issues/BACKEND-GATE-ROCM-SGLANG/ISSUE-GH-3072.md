ID: ISSUE-GH-3072
Title: fix(BACKEND-GATE-ROCM-SGLANG): query shared memory through pinned HIP driver
Row: BACKEND-GATE-ROCM-SGLANG
State: OPEN
Kind: UNKNOWN
GitHub: 3072
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
> Owner: current Strix benchmark campaign operator. Found during #3053 restoration. The reviewed preparation CLI at c8d019447 queries torch.cuda.get_device_properties(0).shared_memory_per_block, but the pinned AMD Torch 2.9.1 wheel does not expose that Python property. After repairing MIOpen, Strix job c13a24ee-d1ad-49a7-ba08-e96e814273c0 fails identity-dependencies with AttributeError before kernel compilation. Evidence: /mnt/nas_share/rc/strix-four-engine-3053.X94a3J/prepare-c8d019447-rebuilt-02/logs/identity-dependencies.log.
>
> The pinned Triton AMD backend already returns max_shared_mem from HIP hipGetDeviceProperties: triton/backends/amd/driver.c lines 199-209; driver.py lines 161-177 and 700-705. Validate that executing path on Strix, then use the measured HIP value without changing the Torch/Triton pin, hard-coding 65536 as a measurement, or weakening the existing 64KiB identity gate. Add a regression that executes the production identity probe with Torch properties lacking the unsupported member. Preserve failure behavior for missing, invalid, or insufficient device limits. A committed design supplement precedes implementation; a fresh implementer and independent mutation reviewer are required. #3053 owns downstream preparation, model startup, and benchmarks.

## Resolution

-
