# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-12** (the staged W2 implementation now mirrors the full
32-tactic SM12 family, merged gate/up CT scaling and the traced one-input
SiLU→NVFP4 quant producer. Focused CPU/CUDA, sanitizer and native 27B
correctness gates pass. Its unprofiled c16 fused/fallback AB/BA/AB is
**809.60/797.47 tok/s (+1.521%)**, but only 17/20 timing and 0/4 sampled memory
axes pass; the paired trace is +2.084% with 20/20 timing axes and the expected
launch/time reduction, but remains diagnostic. The binding public checkpoint
is still pushed `bce2627`: 27B median total-throughput ratios are
0.9680/0.9317/0.9403/0.9516/0.9944/1.0073× at c1-c32, with
4/4/5/4/4/12 of 20 performance axes and 2/4 memory axes passing. A clean
pushed-SHA W2 oracle campaign is next. The 27B-only BF16 GDN output default is
separately +0.799% at c16 but 16/20 timing and 2/4 memory, so its row also stays
open; 35B retains f32 and still waits for complete 27B
closure. External KV-cache/LMCache interoperability
remains explicit roadmap inventory and has no benchmark yet).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **BELOW FLOOR — VALID `bce2627` 27B CHECKPOINT** | Exact greedy/cache-off/closed-loop input-1024→output-128 c1-c32 evidence has **2,016/2,016** successful timed requests, three interleaved repetitions, six verified memory returns, model gate **1/1**, and passing paired-trace status. Median total-throughput ratios are **0.9680/0.9317/0.9403/0.9516/0.9944/1.0073×**; only **4/4/5/4/4/12 of 20** throughput+latency axes pass. Median mean-TPOT normalized ratios are **0.9669/0.9230/0.9294/0.9189/0.9591/0.9733×**. Memory passes **2/4**: ours/vLLM median PSS **48,175,118/28,106,819 KiB**, RSS **48,177,520/28,443,844 KiB**, GPU **39,483/72,672 MiB**, and available-memory drop **66,975,336/80,648,668 KiB**. The canonical 27-only runs/ratios/trace-status hashes are `06a4bd7a…e41d` / `1e9643e9…c4b9` / `ef9ce611…3a14`; ours nsys/kernel are `94048795…afde` / `8ce71db7…3f03`; vLLM trace/kernel are `c8931f0a…d3ba` / `2afd6a57…16086`. 35B was not run. | Commit/push the correctness-gated W2 implementation, rebuild from that immutable SHA and rerun this exact component/trace/27B campaign until every axis passes. Only then run 35B. Pool teardown remains separately required. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **W2 IMPLEMENTED / CORRECTNESS-GREEN; COMPONENT POSITIVE, STRICT GATE OPEN** | W1's exact-bucket classification remains valid. W2 now ports all **32** stable FlashInfer tactic IDs in split CUTLASS 4.5 TUs, high-water workspace, merged gate/up resident weights, maximum logical-shard CT divisors/one alpha and the traced one-input `SiluAndMulFp4Quant` producer. `VT_FP4_FULL_TACTICS=0`, `VT_FP4_MERGED_GATE_UP=0` and `VT_FP4_MERGED_SILU_QUANT=0` isolate each iteration. CPU focused tests pass **12/12 / 885/885**; final staged CUDA passes **14/14 / 18,619/18,619**; focused sanitizer passes **1/1 / 16/16** with zero errors/leaks; dense 27B passes **9/9 prefill + 16/16 greedy**, and paged shipping/fallback each pass **235/235 and 16/16**. Final staged binary/log hashes are `36779505…e9786`, `338a059b…dbc4`, `dc90e5fa…3546` and `738d15a5…fcbe` / `61b23535…911b` / `3d2b984f…6595` / `04a3c872…d29c1`; before/after GPU process snapshots are empty. The unprofiled c16 fusion A/B runs 815.625/812.912/800.256 versus 792.337/798.232/801.833 tok/s: means **809.597/797.467 = 1.015211×**, **17/20** timing axes, **0/4** sampled memory axes, all six returns. Summary/driver SHA-256 are `cb5e5204…b0ab89` / `3cda0d4c…d7bb0`. A bounded paired trace is **817.020/800.338 = 1.020843×**, 20/20 timing axes; fused activation+quant executes 8,557 times / 4.802 s versus fallback SiLU 8,390 / 7.054 s plus 8,013 additional quant calls / 2.643 s. Trace summary SHA is `9933724b…a1318`; fused/fallback nsys are `094615a1…be22c` / `5efe621c…0a080`. Profile timing is structural, not the binding speed result. | Rebuild from the clean pushed SHA, run the exact c1-c32 27B vLLM oracle campaign and keep iterating until all performance and memory axes pass. Do not run 35B yet. |
| `KERNEL-GDN-AOT-BF16` 27B output dtype | **27B DEFAULT / CORRECTNESS-GREEN; COMPONENT STRICT GATE OPEN** | The already-vendored BF16 `chunk_o` path now carries the 27B recurrence output, z projection and gated-norm weight by default, matching vLLM's model dtype and restoring the native 16/16 stream with the full FP4 stack. `VT_GDN_OUT_BF16=0` restores f32. Every 35B path, including GGUF, retains its prior f32 default. The c16/96-request BF16/f32 AB/BA/AB runs 789.183/790.691/787.963 versus 780.660/782.203/786.207 tok/s: means **789.279/783.023 = 1.007989×**, CV 0.141%/0.299%, **16/20** timing and **2/4** memory axes, all six returns. The misses are median ITL/TPOT, p90 ITL and p99 TTFT; summary SHA is `ee6d25c…c930b`. | Keep correctness-faithful BF16 for 27B, but retain the row `ACTIVE`; pair its trace/pool evidence and classify it against the clean exact vLLM campaign. Do not infer any 35B result. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0's clean build, default/fallback pointer/model gates, +2.1239% A/B, trace and both-model access checks remain valid. W1 adds persistent full non-spec/prefill metadata plus indexed BF16↔F32 gather/scatter. Its exact cache-off c16/48 AB/BA/AB repetitions average **781.799 indexed vs 776.946 fallback tok/s (+0.6246%)**, improve **20/20** axes, have 0.846%/0.530% CV, and return memory in all six legs. Paired traces reduce `cudaMemcpyAsync` **163,540→7,508**, D2D calls **142,717→1,231**, and D2D volume **49,088.289→1,855.918 MB**. CUDA/model/turnover/sanitizer gates pass; inherited pools still fail strict teardown (27B **47,290,056 B/101**, 35B **36,822,413,188 B/1,236**) with W0 caches absent. The exact `bce2627` oracle remains below floor; the FP4 chain it identified is now implemented/component-gated and awaits its immutable-SHA oracle rerun. | Keep device-residency W2 scoped while FP4 W2 runs the next exact oracle. Re-rank from that residual; separately repair model/pool/queue teardown. |

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
