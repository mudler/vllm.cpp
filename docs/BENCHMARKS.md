# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-12** (clean pushed `b5c6e4f` completes the exact W2 27B
campaign. All 12 groups, 2,016/2,016 timed requests, six memory returns, model
gate and paired traces validate. Median total-throughput ratios are
0.9933/0.9520/0.9657/0.9760/1.0213/1.0218× at c1-c32, with
4/4/5/4/17/14 of 20 performance axes and 2/4 memory axes passing. W2 improves
every point and wins c16/c32 total throughput, but c1-c8 and mean TPOT/ITL stay
below the strict floor. The trace shows vLLM's dominant 128x32x256
Stream-K/static-persistent FP4 pair while our live planner predominantly
selects 128x128x128/256x128x128, promoting W3 pre-serve warmup/selection/cache
parity. Runtime inspection now proves pip-vLLM uses pre-serve in-memory tuning
with a stream sync and 1-ms GPU delay before ten eager event repeats; its file
cache is disabled for key collisions. W3-A's exact delayed timing mirror is
implementation-staged with `VT_FP4_AUTOTUNE_DELAY=0` restoring W2. Immutable
`71f1e89` focused delayed/off, both 27B correctness arms and delayed memcheck
pass; delayed real-model selection includes the oracle's ID 6/4 narrow pair.
The timed performance A/B is **PENDING**, so no new number is published. The
27B-only BF16 GDN output default remains correctness-required; 35B
retains f32 and still waits for complete 27B closure. External KV-cache/LMCache interoperability
remains explicit roadmap inventory and has no benchmark yet).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **BELOW FLOOR — VALID CLEAN `b5c6e4f` 27B CHECKPOINT** | Exact greedy/cache-off/closed-loop input-1024→output-128 c1-c32 evidence has **2,016/2,016** successful timed requests, three interleaved repetitions, six verified memory returns, model gate **1/1**, and passing paired-trace status. Median total-throughput ratios are **0.9933/0.9520/0.9657/0.9760/1.0213/1.0218×**; **4/4/5/4/17/14 of 20** throughput+latency axes pass. Median mean-TPOT normalized ratios are **0.9915/0.9417/0.9476/0.9404/0.9827/0.9837×**. Memory passes **2/4**: ours/vLLM median PSS **48,272,873/28,096,858 KiB**, RSS **48,275,264/28,424,060 KiB**, GPU **38,746/72,608 MiB**, and available-memory drop **66,089,528/80,435,540 KiB**. The canonical 27-only runs/ratios/trace-status hashes are `0056bf62…c5c59` / `632e087b…192c` / `0190a7e1…ad3e`; ours nsys/kernel are `f0599533…9e57` / `d2367ab4…392e`; vLLM trace/kernel are `db996f39…b41` / `caf8ac9f…258b`. Trace-run output digests differ locally and are stable in vLLM; the separate 16/16 commit-bound model gate owns correctness. W3-A has commit-bound correctness/safety but no timed performance result, so `b5c6e4f` remains the denominator and 35B was not run. | Run the W3-A delayed/off c16 component A/B, add W3-B pre-serve in-memory all-bucket warmup if needed, and prove stable oracle-selected 128x32x256 plans. Gate optional W3-C persistence separately because pip-vLLM disables its file cache. Then rerun this exact 27B campaign until every axis passes. Only then run 35B. Pool teardown remains separately required. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **W3-A IMMUTABLE CORRECTNESS/SAFETY GREEN / PERFORMANCE A/B PENDING; W2 BINDING BELOW FLOOR** | W1's exact-bucket classification and W2's 32 tactics, merged gate/up scaling, one-input fusion and prior gates remain green. Clean `b5c6e4f` remains the binding performance result: totals **0.9933/0.9520/0.9657/0.9760/1.0213/1.0218×**, axes **4/4/5/4/17/14**, memory **2/4**. Immutable `71f1e89` clean-builds on CUDA 13.0.88/sm_121a/CUTLASS 4.5. Fresh delayed/off focused processes each pass **14/14 / 18,619/18,619**; delayed/off native 27B each pass **235/235** and the full **16/16** vLLM stream; delayed memcheck passes **1/1 / 16,389/16,389 / 0 errors**. Delayed M=9 selects ID 6 Stream-K for output and merged gate/up and ID 4 static for Q; M=1 selects ID 6/4 on output/gate-up. Both arms are numerically green, while plan differences remain timing-sensitive. Model delayed/off log SHA are `8065b47e…7a61d` / `3b3fcb6a…7a61d`; memcheck SHA is `60d704a9…75c81`; GPU/lock return idle. No throughput or memory comparison is inferred. | Run the real-27B delayed/off c16/96 AB/BA/AB with selected-plan logs and all 20+4 axes. Then implement W3-B pre-serve in-memory all-bucket warmup if W3-A alone does not close selection stability. W3-C persistence stays optional because pip-vLLM disables its file cache. Repeat exact 27B only after the component is accepted; do not run 35B. |
| `KERNEL-GDN-AOT-BF16` 27B output dtype | **27B DEFAULT / CORRECTNESS-GREEN; STRICT GATE OPEN** | The BF16 `chunk_o` path carries the 27B recurrence output, z projection and gated-norm weight by default, matching vLLM and restoring the native 16/16 stream; `VT_GDN_OUT_BF16=0` restores f32 and every 35B path retains f32. Its BF16/f32 component remains **1.007989×**, **16/20** timing and **2/4** memory. Clean `b5c6e4f` now classifies the exact oracle: c16 total throughput passes at **1.021341×**, but normalized mean TPOT/ITL is **0.982670×**; c1-c8 remain decode-shaped gaps. The paired trace contains vLLM's fused recurrent decode and the local GDN kernels, while FP4 tactic selection is the higher-priority traced mismatch. | Keep correctness-faithful BF16 for 27B and retain the row `ACTIVE`; re-rank its residual after W3 FP4 selection parity, then address GDN decode/pool evidence if still red. Do not infer any 35B result. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0's clean build, default/fallback pointer/model gates, +2.1239% A/B, trace and both-model access checks remain valid. W1 indexed BF16↔F32 state I/O averages **+0.6246%**, improves **20/20** axes and sharply reduces D2D traffic. CUDA/model/turnover/sanitizer gates pass; inherited pools still fail strict teardown (27B **47,290,056 B/101**, 35B **36,822,413,188 B/1,236**). Clean `b5c6e4f` re-ranks the residual: W2 FP4 availability/topology materially improves the grid, but the trace still shows FP4 tactic-selection mismatch and the end-to-end host-memory axes remain red. | Keep device-residency W2 scoped while FP4 W3 repairs selection parity; separately repair model/pool/queue teardown. |

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

### Reproduce the W3-A correctness checkpoint

Use immutable `71f1e894d0c5e496607d08cfe9089a9944128271` in one clean detached
checkout on `dgx.casa`. The two test arms use the same binary and one uncontended GPU lock;
the default delayed arm must report `delay=1000us`, while the fallback must
report `delay=off`. This is a correctness/selection preflight, not a benchmark
result.

```sh
BUILD="$HOME/work/vllm.cpp-nvfp4-small-m/w3-a-build"
cmake -S . -B "$BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" --target test_ops_nvfp4_fp4 \
  test_qwen27_paged_engine server --parallel "$(nproc)"
flock /tmp/gpu sh -c '
  VT_FP4_AUTOTUNE_VERBOSE=1 ctest --test-dir "'"$BUILD"'" \
    -R "^(test_ops_nvfp4_fp4|test_qwen27_paged_engine)$" --output-on-failure &&
  VT_FP4_AUTOTUNE_VERBOSE=1 VT_FP4_AUTOTUNE_DELAY=0 \
    ctest --test-dir "'"$BUILD"'" \
    -R "^(test_ops_nvfp4_fp4|test_qwen27_paged_engine)$" --output-on-failure
'
```

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
# Do not execute this 35B command until the validator reports every 27B axis
# passing; it is retained here as the eventual second-model gate recipe.
scripts/dgx-online-serving.sh --execute --model 35 --snapshot "$M35" \
  --source-corpus "$EVIDENCE/corpus/35" --evidence "$EVIDENCE" \
  --build-dir "$BUILD" --client "$CLIENT"
```

The full acceptance contract, including corpus generation, zero-residency
cache proof, warmup, repetitions, every-axis validation, memory return, and
paired traces, is in the
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md) and
[benchmark protocol](../.agents/benchmark-protocol.md).
