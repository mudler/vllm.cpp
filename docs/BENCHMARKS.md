# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-12** (pushed `bce2627` completes the exact small-M FP4
W1 component, trace, and binding 27B oracle campaigns. W1 is structurally
effective but does not pass its no-regression or every-axis gates. The fresh
27B median total-throughput ratios are 0.9680/0.9317/0.9403/0.9516/0.9944/
1.0073× at c1-c32, with 4/4/5/4/4/12 of 20 performance axes and 2/4 memory
axes passing. W2, the traced full FlashInfer tactic family, is next; 35B still
waits for complete 27B closure. External KV-cache/LMCache interoperability
remains explicit roadmap inventory and has no benchmark yet).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **BELOW FLOOR — VALID `bce2627` 27B CHECKPOINT** | Exact greedy/cache-off/closed-loop input-1024→output-128 c1-c32 evidence has **2,016/2,016** successful timed requests, three interleaved repetitions, six verified memory returns, model gate **1/1**, and passing paired-trace status. Median total-throughput ratios are **0.9680/0.9317/0.9403/0.9516/0.9944/1.0073×**; only **4/4/5/4/4/12 of 20** throughput+latency axes pass. Median mean-TPOT normalized ratios are **0.9669/0.9230/0.9294/0.9189/0.9591/0.9733×**. Memory passes **2/4**: ours/vLLM median PSS **48,175,118/28,106,819 KiB**, RSS **48,177,520/28,443,844 KiB**, GPU **39,483/72,672 MiB**, and available-memory drop **66,975,336/80,648,668 KiB**. The canonical 27-only runs/ratios/trace-status hashes are `06a4bd7a…e41d` / `1e9643e9…c4b9` / `ef9ce611…3a14`; ours nsys/kernel are `94048795…afde` / `8ce71db7…3f03`; vLLM trace/kernel are `c8931f0a…d3ba` / `2afd6a57…16086`. 35B was not run. | Port and independently gate FP4 W2's full 32-tactic family, rerun the same component/trace/27B campaign, and continue until every 27B axis passes. Only then run 35B. Pool teardown remains separately required. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **W1 MEASURED — POSITIVE AT c8/c32, STRICT GATES FAIL; W2 ACTIVE** | W1 retains all prior CPU/TSan/capture/memcheck/model proof. Its same-binary exact/legacy AB/BA/AB completes **2,016/2,016** requests: total-throughput ratios c1-c32 are **1.0002/1.0012/1.0006/1.0093/0.9996/1.0072×**, with **10/16/18/18/5/19 of 20** axes and only **1/4** memory axes passing. Thus the real c8/c32 gains do not satisfy the no-regression gate. Paired traces prove distinct exact M=1/2/4/8/16 plans and reduce aggregate FP4 kernel time **34,754.7→34,391.0 ms (-1.05%)** plus all-kernel time **107,121.5→105,159.8 ms (-1.83%)**; profiler throughput inverts to 0.9873× and is diagnostic only. Component summary/artifact hashes are `de20915a…6239` / `b0f4b432…dceb`; trace summary/artifact hashes are `d83826db…9c44` / `d7591e84…4811`. The fresh oracle trace still spends **25.1%** in the absent 128x32x256 Stream-K/static pair (94,144 + 119,280 calls), while ours exposes only four wide persistent tactics. | W1 is classified and remains default because its binding checkpoint materially improves the prior 27B grid without a correctness failure. Port W2's exact 8-tile x 2-orientation x 2-scheduler family with high-water workspace and forced-ID tests; repeat every gate. Do not run 35B yet. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0's clean build, default/fallback pointer/model gates, +2.1239% A/B, trace and both-model access checks remain valid. W1 adds persistent full non-spec/prefill metadata plus indexed BF16↔F32 gather/scatter. Its exact cache-off c16/48 AB/BA/AB repetitions average **781.799 indexed vs 776.946 fallback tok/s (+0.6246%)**, improve **20/20** axes, have 0.846%/0.530% CV, and return memory in all six legs. Paired traces reduce `cudaMemcpyAsync` **163,540→7,508**, D2D calls **142,717→1,231**, and D2D volume **49,088.289→1,855.918 MB**. CUDA/model/turnover/sanitizer gates pass; inherited pools still fail strict teardown (27B **47,290,056 B/101**, 35B **36,822,413,188 B/1,236**) with W0 caches absent. The exact `bce2627` oracle remains below floor and freshly confirms the separate FP4 full-tactic gap. | Keep device-residency W2 scoped while FP4 W2 owns the confirmed residual. Re-rank after the next exact oracle; separately repair model/pool/queue teardown. |

The stream-usage path changes host-side JSON/SSE serialization, not model
kernels. Its performance disposition is nevertheless `PENDING` because the
final frame participates in whole-request throughput and the feature directly
unblocks the binding online latency campaign. Retained ITLs are inter-choice
timings: pinned vLLM's single-slot collector deliberately merges DELTA outputs
when the producer gets ahead, while native usage remains the exact token-count
oracle. The gate accepts fewer than 127 ITLs for that case but still rejects
extra intervals, partial requests, count drift, errors, or unsaturated load.

The audit found that the historical direct-library comparisons were not exact
same-workload runs. Pinned `vllm bench throughput` hard-codes
`temperature=1.0`, while the vllm.cpp arms used greedy `temperature=0`; the 27B
vLLM run also resolved `max_num_batched_tokens=8192`, versus 2048 for vllm.cpp.
They remain useful historical diagnostics, but the strict performance gate is
reopened for both models. Correctness remains established separately by the
commit-bound real-model greedy gates and exact native request counts.

## Historical CUDA engine checkpoints — reopened, non-binding

These values are preserved so the optimization history remains reproducible.
They cannot establish production-vLLM parity under the current protocol because
their oracle sampling differed; the 27B scheduler budget differed as well.

| Model / point | Historical build and workload | vllm.cpp | vLLM | Diagnostic ratio | Disposition |
|---|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B NVFP4, c64 / 200 prompts | Triton-AOT GDN; input 1024, output 128 | 3345.9 tok/s | 3282.0 tok/s | 1.0195× | Non-binding: ours temperature 0, vLLM temperature 1. Correctness remains 16/16 token-exact in its separate greedy gate. [Record](../.agents/state.md#L1740). |
| Qwen3.6-27B NVFP4, c16 / 96 prompts | Triton-AOT + default FA-2 prefill; input 1024, output 128, seed 0 | 764.28 tok/s total; 84.89 output | 758.84 tok/s total; 84.32 output | 1.0072× total; 1.0068× output | Non-binding: temperature 0/1 and token budget 2048/8192. Historical repetition spread retained in the [ledger](../.agents/parity-ledger.md#L284). |
| Qwen3.6-27B NVFP4, c32 / 192 prompts | Same historical build | 1051.24 tok/s total; 116.77 output | 1043.86 tok/s total; 115.98 output | 1.0071× total; 1.0068× output | Non-binding for the same two configuration mismatches; historical peak memory was 61.8 vs 76.2 GB. |

The historical 35B result requires `-DVLLM_CPP_TRITON=ON`; its default pure-C++
build measured 0.99× in the same old campaign. The 27B values require the
default-on vendored FA-2 prefill route. Same-binary component A/Bs still support
the individual kernel choices, but fresh exact oracle runs are required for an
end-to-end acceptance claim, and binding server-to-server latency remains open.

## Reproduce the current online checkpoint

Run from a clean, merged checkout on `dgx.casa`; the driver owns one
uncontended `/tmp/gpu` lock for each whole-model series and refuses stale or
partial evidence. Snapshot paths below are the pinned gate checkpoints.

```sh
SHA=$(git rev-parse HEAD)
CLAIM_ROOT="$HOME/work/vllm.cpp-online-gate"
EVIDENCE="$CLAIM_ROOT/evidence/$SHA"
CLIENT="$HOME/venvs/vllm-oracle/bin/vllm"
BUILD="$HOME/work/vllm.cpp-online-build"
M27=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots" -name config.json -print -quit)")
M35=$(dirname "$(find "$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots" -name config.json -print -quit)")

scripts/dgx-online-serving.sh --dry-run \
  --claim-root "$CLAIM_ROOT" --client "$CLIENT" --vllm-cpp-sha "$SHA"
# Execute the exact corpus commands recorded in $EVIDENCE/manifest.json,
# then build the clean SHA in $BUILD before either measured arm.
scripts/dgx-online-serving.sh --execute --model 27 --snapshot "$M27" \
  --source-corpus "$EVIDENCE/corpus/27" --evidence "$EVIDENCE" \
  --build-dir "$BUILD" --client "$CLIENT"
scripts/dgx-online-serving.sh --execute --model 35 --snapshot "$M35" \
  --source-corpus "$EVIDENCE/corpus/35" --evidence "$EVIDENCE" \
  --build-dir "$BUILD" --client "$CLIENT"
```

The full acceptance contract, including corpus generation, zero-residency
cache proof, warmup, repetitions, every-axis validation, memory return, and
paired traces, is in the
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md) and
[benchmark protocol](../.agents/benchmark-protocol.md).
