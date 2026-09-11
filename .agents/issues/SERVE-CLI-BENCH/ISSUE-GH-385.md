ID: ISSUE-GH-385
Title: vllm-bench cannot express the KV-pool byte budget: --kv-cache-memory / --gpu-memory-utilization unreachable from the bench
Row: SERVE-CLI-BENCH
State: OPEN
Kind: UNKNOWN
GitHub: 385
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-11
Updated: 2026-08-11
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> ## The gap
>
> Of the three entrypoints, `vllm-bench` is the only one that cannot express the KV-pool byte budget.
>
> | flag | `vllm-server` | `vllm-cli` | `vllm-bench` |
> |---|---|---|---|
> | `--num-blocks` | yes | yes | yes |
> | `--kv-cache-memory` | yes | yes | **no** |
> | `--gpu-memory-utilization` | yes | yes | **no** |
>
> `server_main.cpp:289-290` and `examples/cli/main.cpp:113-114` both parse `--kv-cache-memory` into `EngineParams::kv_cache_memory_bytes`. `examples/bench/main.cpp` parses `--num-blocks` (`:91-92`) and neither of the other two.
>
> To be precise about what this is and is not: `vllm-bench` **rejects unknown arguments loudly** (`main.cpp:99-103`: `"vllm-bench: unknown argument"`, usage, `exit_code = 2`), so nothing is silently ignored. This is a missing feature, not a silent failure.
>
> ## Why it is more than cosmetic
>
> `bench_core.h:574-576` gives the real-checkpoint path its own sizing heuristic:
>
> ```cpp
> params.num_blocks = cfg.num_blocks > 0
>                         ? cfg.num_blocks
>                         : std::max(cfg.concurrency * seq_blocks * 2, 256);
> ```
>
> `num_blocks` is therefore **always > 0** from the bench, so `ResolveNumBlocks` always takes knob 1 and the byte-budget and utilization knobs are unreachable from this binary by construction -- not merely unparsed. Two consequences:
>
> 1. A benchmark cannot be run under the same KV-pool configuration as the server it is meant to characterize. On a memory-constrained or shared device that is exactly the configuration you want to measure.
> 2. The bench's heuristic scales with `concurrency` and sequence length, so its pool size moves with the benchmark parameters. Comparing a `c=8` run against a `c=1` run silently varies the KV pool between the two arms, which is a confound in a tool whose purpose is A/B measurement.
>
> ## Suggested shape
>
> Parse `--kv-cache-memory` (and `--gpu-memory-utilization`, for symmetry) in `examples/bench/main.cpp` and thread them into the `EngineParams` built in `bench_core.h`, leaving the current heuristic as the fallback when none of the three knobs is set. That keeps every existing invocation byte-identical and makes the precedence in `model_loader.h:69-83` reachable from all three entrypoints.
>
> Note this interacts with #357: while the `KVBytesPerBlock` layer-count defect is open, the byte-budget knob mis-sizes wherever it is reachable. The order probably wants to be #357 first, then this -- otherwise the new flag inherits a broken divisor. Flagging the dependency rather than assuming your sequencing.
>
> ## Scope
>
> Read from the tree at `6dbedf9f`. I have not built or measured anything for this, and I have not hit a failure caused by it -- I found it while ruling out a flag-plumbing hypothesis for #357 (which turned out to be a real defect elsewhere; the server's plumbing is correct). Happy to implement if you want it.
>

## Resolution

-
