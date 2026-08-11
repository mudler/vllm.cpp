# sm_120 Qwen3.5 BF16 pre-serve prefill warmup

**Lifecycle:** `REJECTED/REMOVED`; directionally positive but below hard gates

**Owner rows:** `ROAD-V1-C2-LOCAL-BF16`, `KERNEL-SSM-MAMBA`

**Scope:** Qwen3.5 plain-BF16 CUDA initialization only. No default, release,
server-startup SLA, NVFP4, CPU, 27B/35B or pinned-vLLM performance claim is
authorized by this spike.

## Ground and falsifiable hypothesis

Corrected atomic/pretoken comparison `da449a88f` leaves local TTFT 1016.513 ms
versus pinned vLLM 936.589 ms. A current accepted-stack diagnostic at
`6c7e64b90` is exact and reports 6865.95 tok/s, 1008.71 ms TTFT, 34.33 ms TPOT
and 5368.08 ms E2E. Its internal means are 133.406 ms intake, 401.197 ms queue
and 473.969 ms prefill. Pinned vLLM reports 338.960 ms queue and 449.291 ms
prefill; its comparable inferred intake is 154.011 ms.

The gap is isolated to the first request wave, not steady service:

| request IDs | local queue + prefill | vLLM queue + prefill | local - vLLM |
|---|---:|---:|---:|
| 0-31 | 2086.086 ms | 1625.343 ms | **+460.743 ms** |
| 32-63 | 478.327 ms | 490.471 ms | -12.144 ms |
| 64-95 | 473.912 ms | 510.115 ms | -36.203 ms |
| 96-127 | 462.336 ms | 527.074 ms | -64.738 ms |

Thus every post-first wave is already faster; the mean deficit is first-touch
startup cost. This matches the execution-chain difference. Pinned vLLM always
runs `_dummy_run(self.max_num_tokens, is_profile=True)` and synchronizes during
memory profiling (`gpu_worker.py:468-495`, `gpu_model_runner.py:6411-6483`),
then warms declared compile sizes and kernels before serving
(`gpu_worker.py:671-708`). Local `LoadedEngine::WarmupKernels` is invoked before
the async frontend exists, but returns unless the model is NVFP4 W4A4 and three
FP4 flags are on (`src/vllm/entrypoints/model_loader.cpp:862-918`). Plain BF16
therefore serves its first real prefill cold.

**Hypothesis:** one pre-serve dummy BF16 prefill at the configured
`max_num_batched_tokens` moves kernel/library/pool first-touch work out of user
TTFT, matching vLLM initialization. It must reduce the first-wave mean by at
least 250 ms and aggregate TTFT by at least 5%, without changing later-wave
queue+prefill, throughput, TPOT/E2E, tokens, steady memory or scheduler state.

## Exact experimental contract

Add strict opt-in `VT_BF16_PRE_SERVE_WARMUP=1`; only exact `"1"` selects it.
Eligibility requires CUDA, a loaded non-NVFP4 Qwen3.5 model, positive
`max_num_batched_tokens`, and the synchronous pre-serve phase before
`async_engine()` construction. Unset, `0`, invalid values, CPU, NVFP4 and every
other model retain byte-identical initialization.

Reuse the existing `WarmupKernels` dummy-request lifecycle rather than adding a
benchmark-only warmup. Choose one valid non-special tokenizer token, construct a
prompt of exactly `max_num_batched_tokens`, request one greedy token with EOS
ignored, drive `engine_core_.step()` until completion, drain the finished marker,
and assert both unfinished and finished scheduler state are empty. Use a distinct
request id. Do not start AsyncLLM, alter benchmark timers, change scheduling,
retain output, or special-case the benchmark executable. The warmup must use the
production model/runner/KV paths and must release its blocks before serving.

## Red-first, mutation, and correctness gates

Extract a portable selector/lifecycle seam and first capture RED for the missing
BF16 arm. Tests cover strict parsing, BF16-vs-NVFP4/model/backend eligibility,
exact prompt length, one request, greedy one-token parameters, step-until-done,
the extra finished drain, and empty final scheduler state. Mutations must kill a
permissive selector, BF16 hardwired false, wrong prompt length, skipped final
drain, more than one generated token, and benchmark-only routing. Existing FP4
warmup tests and defaults remain unchanged.

Operator CUDA gates: focused warmup/loader tests, Qwen GDN and scheduler tests,
clean `-Werror`, then exact 128x128 Qwen output hash
`be20ffbceb61f0264ca21d972bfc5fc51e855e64f2b945de71669cae666aa702` in
both arms. A second request after warmup must reuse a clean scheduler/block state.

## Measurement and disposition

Use one immutable binary, one GPU lock and the standard cgroup. Run
`COLD-a -> WARM-a -> WARM-b -> COLD-b`; only the selector differs. Record load
time separately because moving work before serving is a startup trade, not free
compute. Capture `VT_TTFT_DUMP=1` and total/output throughput, TTFT, TPOT/ITL,
E2E, GPU peak/resident memory and host PSS. All tokens must match before timing.

Accept as an opt-in only if both WARM raw first-wave queue+prefill means beat
both COLD controls, their mean improves at least 250 ms, aggregate TTFT improves
at least 5%, IDs 32-127 do not regress beyond 1%, and throughput, TPOT, E2E,
steady VRAM/PSS and correctness do not regress. Report added load time and peak
initialization memory. A pass remains default-OFF until 27B/35B correctness and
startup-memory gates justify mirroring vLLM's default. Any miss removes product
and tests while retaining this outcome. The next structural target on failure is
the first-wave chunk scheduling/shape sequence, not another decode microkernel.

## Outcome

**REJECTED and removed, 2026-08-10.** Implementation `ef1f404f1` plus review
repair `120b36a0e` passed fresh mutation re-review and the operator's rebuilt
5-case / 42-assertion portable gate. The first exact GPU pair produced identical
token SHA-256
`be20ffbceb61f0264ca21d972bfc5fc51e855e64f2b945de71669cae666aa702`:

| Arm | total / output tok/s | TTFT | TPOT | E2E | first-wave queue+prefill |
|---|---:|---:|---:|---:|---:|
| COLD | 6864.21 / 759.03 | 1008.58 ms | 34.34 ms | 5369.49 ms | 2084.065 ms |
| WARM | 6902.89 / 763.30 | 973.79 ms | 34.37 ms | 5339.26 ms | 1941.388 ms |

Warmup is directionally positive (+0.564% total throughput, -3.45% TTFT,
-0.563% E2E), but the first-wave gain is only **142.676 ms** versus the required
250 ms and aggregate TTFT improves only **3.45%** versus the required 5%.
TPOT also rises 0.03 ms. One leg is sufficient to reject on both hard magnitude
gates; repetitions cannot turn this measured arm into an accepted result without
changing the prespecified thresholds. Raw roots are
`/tmp/qwen35-bf16-warmup-{cold,warm}-120b36a0e.*`. Product, selector and tests
are removed. The remaining TTFT target is the first-wave chunk scheduling/shape
sequence after first-touch warmup, as this spec prescribed.
