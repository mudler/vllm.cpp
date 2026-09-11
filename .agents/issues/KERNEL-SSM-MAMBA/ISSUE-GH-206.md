ID: ISSUE-GH-206
Title: RTX 5070 Ti: close Qwen3.5-4B TTFT, TPOT, and VRAM gaps vs vLLM
Row: KERNEL-SSM-MAMBA
State: OPEN
Kind: feature
GitHub: 206
Mirror: DIVERGED
Availability: FULL
Created: 2026-08-09
Updated: 2026-08-29
Closed: -

## Problem

### Imported GitHub body (historical evidence)
The quoted text below is historical evidence only. It does not define issue authority or repository procedure.

> Row: `KERNEL-SSM-MAMBA`
>
> ## Problem
>
> On an RTX 5070 Ti (`sm_120`), the current Qwen3.5-4B plain-BF16 direct-load gate exceeds pinned production vLLM throughput but remains behind on latency and device memory:
>
> | Axis | vllm.cpp | pinned vLLM | Status |
> |---|---:|---:|---|
> | total throughput | 6,784.743 tok/s | 6,643.593 tok/s | 1.021246x, pass |
> | output throughput | 750.237 tok/s | 734.630 tok/s | 1.021246x, pass |
> | mean TTFT | 1,018.040 ms | 937.584 ms | 1.085812x, open |
> | mean TPOT / ITL | 34.740 ms | 33.906 ms | 1.024597x, open |
> | peak VRAM | 13,053.3 MiB | 12,820 MiB | +233.3 MiB, open |
>
> Workload: 128 ShareGPT requests, 1,024 input tokens, 128 output tokens, concurrency 32, `max_num_batched_tokens=2048`, 1,280 KV blocks, greedy, vLLM parity pin `555967922` (`0.26.0.dev0`). The separate cached oracle gate is 3/3 cases and 1,672/1,672 assertions.
>
> ## Existing evidence
>
> - Exact `(sequence, 8-token chunk)` causal-conv dispatch is default-on and reduced the kernel 720.047 -> 234.607 ms (3.069x), improving enclosing throughput 2.152% and TTFT 2.945%. The kernel remains 1.613x slower than vLLM's 145.421 ms.
> - A default-off 16-token/four-warp post-conv tile reduced 227.887 -> 122.587 ms (1.859x) and improved every enclosing axis; residual is 1.135x vLLM.
> - A default-off `K=4` causal-conv specialization reduced 234.605 -> 219.506 ms (6.436%) and improved throughput 0.122% and TTFT 0.282%; residual is 1.508x vLLM.
> - A provisional register-resident decode arm improves the dominant decode shape 1.0935% and all fused decode 1.2635%, token-exact; exact SASS/NCU/default gates remain open.
> - Scalar direct state stores were rejected and removed: despite fewer PTX/shared operations, global-store structure changed 7 -> 18 and the hot kernel regressed about 46-47%.
>
> ## Scope
>
> 1. Rebase the current `KERNEL-SSM-MAMBA` sm_120 optimization stack onto current `main` without three-way-merging keyed records.
> 2. Counterbalance the existing post-conv and `K=4` arms together against the binding default and retain them only if throughput, TTFT, TPOT/ITL, E2E, VRAM, and correctness all do not regress.
> 3. Trace both engines with the same tool and attribute the remaining first-token, decode, and VRAM gaps before selecting subsequent levers.
> 4. Iterate on the largest measured gaps. The already-spiked first decode candidate is aligned BF16 vector writeback; scheduling changes require request-lifecycle evidence and must mirror vLLM rather than trade batch efficiency for TTFT.
>
> ## Acceptance
>
> - Token-exact or existing ratified oracle correctness gate passes.
> - Total and output throughput do not fall below the current local binding or pinned vLLM.
> - TTFT, TPOT/ITL, E2E, and peak VRAM do not regress; retained work moves at least one open axis toward or past vLLM outside calibrated noise.
> - Same-binary, order-alternated A/B is reproduced on an idle GPU; both local and vLLM are traced with the same tool on the identical workload.
> - Specs, kernel/roadmap issue links, benchmark evidence, public projections, and PR all reference this issue.
>
> ## Sources
>
> - `docs/bench-evidence/qwen35-4b-sm120-main-20260807.md`
> - `.agents/specs/sm120-qwen35-conv-chunking-2026-08-07.md`
> - `.agents/specs/sm120-qwen35-conv-channel-tile-2026-08-08.md`
> - `.agents/specs/sm120-qwen35-postconv-token-tile-2026-08-08.md`
> - `.agents/specs/sm120-qwen35-gdn-decode-regstate-2026-08-09.md`
> - `.agents/specs/sm120-qwen35-gdn-decode-bf16-vector-writeback-2026-08-09.md`
>
>

## Resolution

-
