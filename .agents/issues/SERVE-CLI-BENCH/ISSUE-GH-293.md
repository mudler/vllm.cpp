ID: ISSUE-GH-293
Title: perf(cpu): stop benchmark AsyncLLM polling from stealing a worker core
Row: SERVE-CLI-BENCH
State: OPEN
Kind: UNKNOWN
GitHub: 293
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-10
Updated: 2026-08-10
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: SERVE-CLI-BENCH
>
> The Raspberry Pi 5 W0 refresh for issue #284 reranked the current CPU gap before kernel work. On current main, vllm-bench uses the production AsyncLLM frontend and scans every active collector with get_output_nowait, then std::this_thread::yield when no output is ready. With VLLM_CPP_CPU_THREADS=4 on the four-core Cortex-A76, the benchmark frontend and engine/output-handler threads contend with the four compute workers.
>
> Fresh diagnostic evidence on the same Pi/model shows the current T4 arm at about 48.7 seconds E2E and 1.37 per-stream decode tok/s, while the same current binary at T3 recovers about 26.96 seconds and 2.53 tok/s with exact tokens. The matched profile ranks threadpool Barrier at 42.21 percent. This proves a CPU thread-budget/oversubscription problem, but does not yet isolate busy polling itself: the T4-to-T3 control changes worker count, and the old-binary control spans other code changes.
>
> Scope:
> - add a current-source synchronous/blocking or event-driven benchmark control that isolates frontend polling from compute-worker count;
> - port the minimum event/wait primitive needed to wait for any active collector without a spin/yield scan, or implement an explicit CPU thread budget if profiling proves that is the correct production behavior;
> - preserve deterministic request admission, DELTA collection, token identity and metric definitions;
> - same-binary rollback/control and CPU tests first;
> - QEMU-build AArch64 locally, execute/profile only on rich@rpi5fan.lan;
> - compare recursively against the exact llama.cpp b9892 same-file denominator.
>
> Out of scope: assembly, model/loader/kernel arithmetic, CUDA/Vulkan, building on the Pi, service changes, or weakening correctness.
>
> Done when a current-source control identifies the mechanism, the retained C++ change restores or improves T4 throughput without degrading T3/other hosts, and the full Pi model gate plus focused AsyncLLM/benchmark tests pass.

## Resolution

-
