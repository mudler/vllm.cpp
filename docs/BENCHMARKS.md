# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-11** (the exact pushed-`a531e05` 27B online campaign and
paired traces are complete. All 12 performance groups are binding-eligible,
but the every-axis gate is red; c32 HTTP worker capacity and small-M FP4 tactic
keying are confirmed repair targets. The 35B campaign waits for 27B closure).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **BELOW FLOOR — VALID `a531e05` 27B CHECKPOINT** | Exact greedy/cache-off/closed-loop input-1024→output-128 c1–c32 evidence has 2,016/2,016 successful timed requests, three interleaved repetitions, six verified memory returns, model gate **1/1**, and passing paired-trace status. Median total-throughput ratios are **0.9679/0.9338/0.9482/0.9547/1.0028/0.9268×**; only **4/2/5/3/10/8 of 20** throughput+latency axes pass. Median mean-TPOT ratios are **0.9665/0.9204/0.9342/0.9204/0.9676/0.9858×**. Memory passes **2/4**: ours/vLLM median PSS **48,175,251/28,075,085 KiB**, RSS **48,177,656/28,401,904 KiB**, GPU **39,067/72,704 MiB**, and available-memory drop **66,305,132/80,687,860 KiB**. Campaign/trace-status hashes are `24d78fbc…e9d2a` / `1c702ef9…142a`; ours nsys/kernel `22d5a0f4…47d1` / `ab7d0131…d6a3`; vLLM trace/kernel `83fd0f41…2a66` / `7056183f…e417`. 35B was not run. | Repair the confirmed HTTP c32 capacity fault, then the confirmed FP4 bucket/tactic mismatch; re-run the exact 27B ladder after each iteration. Only once every 27B axis passes, run the exact 35B campaign. Pool teardown remains separately required. |
| `SERVE-ASYNC-LLM` HTTP capacity | **CONFIRMED REGRESSION — REPAIR CLAIMED** | cpp-httplib starts 19 workers on the 20-CPU DGX and grows only when `idle_thread_count_ == 0` at enqueue. A streaming SSE socket owns its worker through completion/keepalive. In current c32 repetitions 1 and 3, one accepted socket remained unread with **2,131 receive-queue bytes and zero bytes sent** for about **205–207 s**, while the other 31 streams decoded normally; the healthy repetition grew enough workers and reached **1087.15 tok/s**. This is server admission capacity, not a model-kernel stall. | Add deterministic stream-capacity sizing/backlog coverage under `CLAIM-SERVE-HTTP-POOL-1`, preserve disconnect abort, then repeat c32 with fresh servers and the full ladder. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **CONFIRMED GAP — SPIKE NEXT** | `NextPow2M` clamps every M=1/2/4/8/16 call to one M=16 key. Because the exact ladder starts at c1, its M=1 tactic is reused through c16. Three fresh-server traces started directly at c16 retune real M=16 and give TPOT **161.747/161.719/161.729 ms**, effectively the standard vLLM mean **161.698 ms**, versus **167.484 ms** in our ascending standard series. The paired vLLM trace also resolves 128×32×256 Stream-K/static-persistent FP4 tactics absent from our four wide persistent candidates. | Commit a full FlashInfer dependency-chain spike, port its exact hybrid M buckets with single-flight tuning and an A/B toggle, then separately expand the tactic family. Each iteration gets AB/BA/AB and exact oracle evidence. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0's clean build, default/fallback pointer/model gates, +2.1239% A/B, trace and both-model access checks remain valid. W1 adds persistent full non-spec/prefill metadata plus indexed BF16↔F32 gather/scatter. Its exact cache-off c16/48 AB/BA/AB repetitions average **781.799 indexed vs 776.946 fallback tok/s (+0.6246%)**, improve **20/20** axes, have 0.846%/0.530% CV, and return memory in all six legs. Paired traces reduce `cudaMemcpyAsync` **163,540→7,508**, D2D calls **142,717→1,231**, and D2D volume **49,088.289→1,855.918 MB**. CUDA/model/turnover/sanitizer gates pass; inherited pools still fail strict teardown (27B **47,290,056 B/101**, 35B **36,822,413,188 B/1,236**) with W0 caches absent. The exact `a531e05` oracle checkpoint proves a residual gap, but direct-c16 retuning and the c32 socket trace localize higher-priority causes outside W2. | Keep W2 scoped but do not implement it speculatively. First close HTTP capacity and FP4 dispatch, then re-rank the residual trace. Separately repair model/pool/queue teardown. |

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
