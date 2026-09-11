ID: ISSUE-GH-1495
Title: Windows fast-fails in host memory probe on Linux fopen mode
Row: SPEC-DFLASH2
State: CLOSED
Kind: UNKNOWN
GitHub: 1495
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
> `vllm::v1::host_available_memory_bytes()` opens Linux `/proc/meminfo` with mode `re`. The `e` close-on-exec extension is not a valid MSVC `fopen` mode. On native Windows, the Universal CRT invokes its invalid-parameter handler and terminates the process with fast-fail `0xC0000409`.
>
> This is reached during every `LoadedEngine` construction through the recurrent-state budget probe. It made the CUDA-enabled DFlash2 W3 production-reachability test terminate immediately after runner construction, before doctest could report a result.
>
> ## Reproduction
>
> Windows 11, Visual Studio 2022, CUDA 13.2:
>
> 1. Construct a `LoadedEngine` in `test_dflash2_runner_reach`.
> 2. Runner initialization and `ModelRegistry::Prepare` complete.
> 3. Constructor enters the recurrent memory probe.
> 4. Process terminates at `std::fopen(/proc/meminfo, re)` with `0xC0000409`.
>
> Increasing PE stack reserve and heap-holding the engine do not change the failure.
>
> ## Expected
>
> The host-memory probe must be platform-safe. On Linux it should retain `/proc/meminfo` behavior; on Windows it should either use a native available-memory API or return the documented unknown value `0` without invoking an invalid CRT parameter path.
>
> Owner: `SPEC-DFLASH2-W3` in-flow portability repair.

## Resolution

GitHub records `state_reason: not_planned`, closed by `Don-Chad` on 2026-08-20. This disposition claims no host-memory implementation.
