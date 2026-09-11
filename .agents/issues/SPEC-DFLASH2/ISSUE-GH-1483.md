ID: ISSUE-GH-1483
Title: CUDA-enabled DFlash2 runner reach test selects CUDA for CPU fixture
Row: SPEC-DFLASH2
State: CLOSED
Kind: UNKNOWN
GitHub: 1483
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-20
Updated: 2026-08-20
Closed: 2026-08-20

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## Problem
>
> `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` constructs synthetic host-resident model weights but leaves `EngineParams.device` at `kAuto`. In a CUDA-enabled build on Windows with an RTX 3090, platform selection chooses CUDA and the test fast-fails with exit code `0xC0000409` immediately after the DFlash2 startup notice.
>
> The neighboring API-server test already documents the required rule: synthetic host fixtures must pin the engine to CPU because `auto` selects CUDA in a CUDA build.
>
> ## Reproduction
>
> - Windows 11, Visual Studio 2022, CUDA 13.2, SM86
> - Build `test_dflash2_runner_reach` with `VLLM_CPP_CUDA=ON`
> - Run the resulting executable on RTX 3090
> - Observed: process exits `0xC0000409`
>
> The actual DFlash2 CUDA kernel suites pass on the same build/device: grouped convolution 8/8, top-k 10/10, selector edges 7/7. This isolates the failure to the synthetic reachability harness rather than those kernels.
>
> ## Expected fix
>
> Pin `DflashSpecParams(...).device` to CPU and assert the selected runner device, while preserving the production-entry reachability and call-site deletion mutation guarantees.
>
> Owner: `SPEC-DFLASH2-W3`.

## Resolution

GitHub records `state_reason: not_planned`, closed by `Don-Chad` on 2026-08-20. This disposition claims no test repair.
