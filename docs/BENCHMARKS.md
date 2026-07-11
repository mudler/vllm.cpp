# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-11** (`KV-DEVICE-RESIDENCY` W1 indexed GDN state-I/O
passes its repeated 27B component A/B, both-model fallback gates and structural
trace; inherited process-lifetime pools and the fresh exact vLLM gate remain
open).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **BELOW FLOOR — VALID 27B BASELINE** | Pushed `f065fce` completed the exact cache-off/closed-loop c1–c32 27B series: 2,016/2,016 timed requests, three repetitions, six verified memory returns, a passing commit-bound model gate, and a passing corrected trace contract. Mean total-throughput ratios are **0.9616/0.9258/0.9364/0.9450/0.9775/0.9722×**; only **4/4/5/3/3/3 of 20** timing axes pass. Memory passes **2/4** axes: mean PSS is 45.97 GiB ours vs 26.83 GiB vLLM, while reported GPU memory is 27,509 vs 72,725 MiB. Throughput CV is below 0.7% on both engines. W0 and W1 independently win their same-binary c16/48 component A/Bs by **2.1239%** and **0.6246%**, each on 20/20 axes; neither is a new vLLM denominator or multiplied into this grid. | Run fresh exact current-SHA direct-library and online 27B closure. If any axis remains below floor, execute W2 direct indexed convolution-state update and remeasure; then run all exact 35B axes. The inherited pool-teardown debt remains a separate required row gate. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0's clean build, default/fallback pointer/model gates, +2.1239% A/B, trace and both-model access checks remain valid. W1 adds persistent full non-spec/prefill metadata plus indexed BF16↔F32 gather/scatter. Its exact cache-off c16/48 AB/BA/AB repetitions average **781.799 indexed vs 776.946 fallback tok/s (+0.6246%)**, improve **20/20** axes, have 0.846%/0.530% CV, and return memory in all six legs. Mean reported GPU peaks are 38,081/38,400 MiB. Paired traces reduce `cudaMemcpyAsync` **163,540→7,508**, D2D calls **142,717→1,231**, and D2D volume **49,088.289→1,855.918 MB**; gather/scatter each run 9,394 times. Profiler throughput is inverted at 768.249/794.593 tok/s, so it is retained only as structural evidence; the unprofiled interleaved series owns the speed ratio. CUDA op tests, a focused **7/7, 0-error** memcheck, indexed/fallback 27B+35B model tests, 16-concurrent exact-count turnover smoke, current local serial CTest **105/105**, CPU model turnover, and focused leak-disabled ASan+UBSan pass. Two earlier remote full-suite attempts exposed the unrelated intermittent C-API callback-count flake at 104/105. A strict local model LSan run reports **58,624 B/153** process-lifetime scratch-pool allocations; the indexed op is leak-clean. Evidence manifest SHA is `34285a91…a5b`, summary `4c68b1dc…033`, and runtime manifest `736842d6…41f`. W0 full leak checking still fails on inherited pools: 27B **47,290,056 B/101 allocations**; 35B **36,822,413,188 B/1,236 allocations**, with W0 caches absent. | Fresh 27B oracle closure is next; W2 remains scoped if a gap survives. Separately repair model/pool/queue teardown before zero-leak/lifecycle closure. |

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
