ID: ISSUE-GH-1627
Title: TT backend never advertises SupportsAsyncSampledTokenReadback: async scheduling resolves OFF, test_qwen3_dense_async_serving FATALs on every cached checkpoint
Row: BACKEND-TENSTORRENT-HOST-FREE-FORWARD
State: OPEN
Kind: bug
GitHub: 1627
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-21
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `BACKEND-TENSTORRENT-HOST-FREE-FORWARD`
>
> ## What fails
>
> `tests/parity/test_qwen3_dense_async_serving.cpp` (the ROW-SERVE-ASYNC-DENSE-MIRROR battery) FATALs on Tenstorrent for every case whose checkpoint is cached on the box — Qwen3-0.6B, Mistral-7B-v0.3, and the 4B case where cached — at the anti-vacuous-pass guard:
>
> ```
> REQUIRE(loaded->async_scheduling_enabled());     // tests/parity/test_qwen3_dense_async_serving.cpp:124
> REQUIRE(loaded->max_concurrent_batches() == 2);
> ```
>
> The engine logs `Asynchronous scheduling is disabled (max_concurrent_batches=1)` at load. The remaining cases SKIP as checkpoint-absent (llama-3.2-1B, internlm2), so the battery reads 3 FATAL / 5 skip on TT.
>
> ## Mechanism
>
> `LoadedEngine` resolves async scheduling from `runner_.runner_supports_async()` (`src/vllm/entrypoints/model_loader.cpp:1598-1605`), which the shared runner derives from the backend capability `vt::Backend::SupportsAsyncSampledTokenReadback()` (`src/vllm/v1/worker/gpu/runner.cpp:109-112`, declared `include/vt/backend.h:186` with a `false` default). Only the CPU and CUDA backends override it (`src/vt/cpu/cpu_backend.cpp:38`, the CUDA `async_device_mirror()` path). The Tenstorrent backend has **no override**, so on TT `runner_supports_async()` is false, `ResolveAsyncScheduling` yields OFF, and the REQUIRE fails exactly as designed: the depth-2 path the gate exists to exercise never engages.
>
> ## Not caused by the R5 host-free flip (#1604)
>
> Pre-existing: `git grep -c SupportsAsyncSampledTokenReadback 52e328789 -- src/vt/tenstorrent/` returns zero hits, and the flip commits (`b86e3705f`, `f85492992`) touch only `src/vt/tenstorrent/*`, `src/vllm/platforms/tenstorrent.cpp`, and `tests/vt/test_tenstorrent_backend.cpp` — none of the async resolution path. Captured-vs-eager decode mode is orthogonal to the readback capability; the battery would FATAL identically on the pre-flip default.
>
> ## Why this is not an in-flow fix
>
> Advertising the capability is a design row, not a one-liner. The async input-combine host-reads the sampled token id between steps (`combine_sampled_and_draft_tokens`), which is exactly the per-step host touch the TT host-free decode loop exists to avoid; CUDA answers it with a device-mirrored id (`async_device_mirror`), and TT needs an equivalent decided against the tt-metal allocator/ownership model, plus the #323-class stale-host-ids guard re-proven on device. That needs its own spec, oracle comparison, and perf gates (the win/loss vs host-free is an open question), so it cannot ride the #1604 change that merely tripped over it.
>
> ## Owed by
>
> Filed from the `BACKEND-TENSTORRENT-HOST-FREE-FORWARD` row (#1604); listed under `## Owed` in `.agents/specs/tenstorrent-host-free-forward.md`. The enabling work (TT `SupportsAsyncSampledTokenReadback` + device-mirrored sampled ids + this battery green on TT) is the scope.

## Resolution

-
