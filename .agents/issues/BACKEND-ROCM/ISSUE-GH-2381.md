ID: ISSUE-GH-2381
Title: Benchmark: record AMD GPU clocks in leased ROCm runs
Row: BACKEND-ROCM
State: OPEN
Kind: UNKNOWN
GitHub: 2381
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-31
Updated: 2026-08-31
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BENCH-ROCM-CLOCK-STATE`
>
> `tools/bench/gpu_clock_state.py` records NVIDIA clock state through `nvidia-smi`. It cannot attribute a benchmark on `strix:gpu0`, where the worker exposes AMD sysfs state and no `rocm-smi` command.
>
> The measured Strix interface is:
>
> ```text
> /sys/class/drm/card1/device/pp_dpm_sclk
> 0: 600Mhz
> 1: 621Mhz *
> 2: 2900Mhz
> /sys/class/drm/card1/device/gpu_busy_percent
> 0
> ```
>
> The existing DFlash2 harness can run without a clock summary only by refusing the result. A custom sampler does not satisfy the repository benchmark protocol.
>
> Add an AMD provider to the committed helper. Keep the current NVIDIA output and decisions unchanged. The AMD record must include the resolved DRM device, active SCLK samples, maximum SCLK, busy samples, boot ID, driver identity, and the sample count inside each measured window. AMD-specific checks must not invent NVIDIA persistence-mode or throttle fields.
>
> Acceptance:
>
> - Unit tests cover AMD sysfs discovery, parsing, device ambiguity, field changes, idle windows, and cross-arm comparisons.
> - Existing NVIDIA tests remain unchanged and green.
> - A leased Strix workload produces a clock record with at least 30 busy samples.
> - The DFlash2 benchmark harness accepts the AMD record only when its vendor-specific clock rules pass.
> - The final evidence states the exact workload, source revision, model hashes, device identity, boot ID, and contention state.

## Resolution

-
