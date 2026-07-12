# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-12** (W3-B's corrected three-arm trace is lifecycle-clean
and structurally closes the FP4 tactic-family gap; clean pushed `b5c6e4f`
remains the binding exact W2 27B campaign. All 12 groups, 2,016/2,016 timed
requests, six memory returns, model
gate and paired traces validate. Median total-throughput ratios are
0.9933/0.9520/0.9657/0.9760/1.0213/1.0218× at c1-c32, with
4/4/5/4/17/14 of 20 performance axes and 2/4 memory axes passing. W2 improves
every point and wins c16/c32 total throughput, but c1-c8 and mean TPOT/ITL stay
below the strict floor. The trace shows vLLM's dominant 128x32x256
Stream-K/static-persistent FP4 pair while our live planner predominantly
selects 128x128x128/256x128x128, promoting W3 pre-serve warmup/selection/cache
parity. Runtime inspection proves pip-vLLM uses pre-serve in-memory tuning with
a stream sync and 1-ms GPU delay before ten eager event repeats; its file cache
is disabled for key collisions. W3-A's exact delayed timing mirror uses
`VT_FP4_AUTOTUNE_DELAY=0` to restore W2. Immutable
`71f1e89` focused delayed/off, both 27B correctness arms and delayed memcheck
pass; delayed real-model selection includes the oracle's ID 6/4 narrow pair.
Its c16/96 delayed/off AB/BA/AB averages **809.932/803.379 tok/s =
1.008156×**, but standalone acceptance **FAILED** at **13/20 timing** and
**2/4 memory** axes; only **5/35** common delayed plan keys keep one ID across
all three fresh processes. This component changes no production-vLLM
denominator. W3-B is now immutable build/correctness/access-safety-green at
clean pushed `d7cdf66`: CPU/registry/loader contracts pass, exact/legacy CUDA
each pass **14/14 and 26,819/26,819**, native 27B passes **235/235 + 16/16**,
and memcheck passes **24,586/24,586 with zero errors**. Model and HTTP startup
each materialize **80/80** profiles (16 buckets × five real shapes), report zero
lazy misses, and warmup completes at log line 3 before HTTP listening at line
6. Its exact c16/96 shipping-prewarm/lazy AB/BA/AB completes **576/576** timed
requests and six memory returns. Means are **808.457/808.220 tok/s =
1.000293×**, but strict acceptance **FAILED** at **15/20 timing** and **2/4
memory** axes. Prewarm exposes all 80 keys in each fresh process, but only
**20/80** retain one tactic ID (lazy: **9/30**), so selection remains unstable.
The untimed preflight does show the intended first-use effect: mean first chunk
**0.779/5.662 s** and full request **14.929/20.249 s** for prewarm/lazy. The
production-faithful default stays without steady-state speed credit and no
denominator changes. Trace attempt 1 remains **VOID**. Corrected attempt 2
succeeds under one uncontended whole-series lock: prewarm, lazy and pinned
vLLM each complete **144/144** retained c16/48 requests after a separate
warmup, all six before/after cache inventories remain exactly **49 files**
with digest `b1789458…7523`, all three memory returns pass, and no process or
Nsight session remains. Node-traced prewarm/lazy/vLLM FP4 sums are
**110.623/114.229/109.932 s**. Prewarm removes the lazy arm's **480 / 0.492 s**
retained delay launches and is within **0.63%** of vLLM; its 128x32x256
Stream-K/static pair is **70.333/70.986 s** and **218,434/220,465** calls
versus vLLM, although the scheduler split differs. Profiled means are
**804.860/810.250/798.324 tok/s** and prewarm's normalized mean TPOT is only
**0.9673x**. These rates are diagnostic because CUDA-graph node tracing
perturbs execution; the kernel mix closes the original wide-tactic mismatch,
while the fresh exact W3-B grid remains mandatory. That clean grid is now
**ACTIVE** at pushed `3cc490c` under one uncontended model-wide lock: the
commit-bound 27B gate and corpus/cache preconditions passed, ours repetition 1
retained c1-c32 plus a verified return, and vLLM repetition 1 is in progress.
No partial rate is binding; the disposition waits for all 36 timed groups, six
returns and the paired trace. Corrected evidence
`~/work/vllm.cpp-nvfp4-small-m/d7cdf66…/w3b/trace-ab-oracle-r2`;
driver/provenance/tree SHA `af29681e…22fbf` / `dff465b3…bbe3` /
`2aab1197…a137`; prewarm/lazy/vLLM reports
`a73d6032…1194a` / `9d74b6c8…37ea2` / `f89ffd4a…2d1b8`. The
27B-only BF16 GDN output default remains correctness-required; 35B
retains f32 and still waits for complete 27B closure. External KV-cache/LMCache interoperability
remains explicit roadmap inventory and has no benchmark yet).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **ACTIVE CLEAN `3cc490c` 27B CAMPAIGN; BINDING FLOOR STILL `b5c6e4f`** | The prior exact greedy/cache-off/closed-loop input-1024→output-128 c1-c32 evidence remains binding at **2,016/2,016** requests, totals **0.9933/0.9520/0.9657/0.9760/1.0213/1.0218×**, **4/4/5/4/17/14 of 20** performance axes and **2/4** memory axes. W3-B's corrected trace remains lifecycle-clean and structurally FP4-equivalent. The replacement clean pushed-`3cc490c` campaign now owns one uncontended model-wide lock: model gate **1/1**, deterministic corpus/cache inventory and first ours cache eviction passed; ours repetition 1 retained all c1-c32 results and returned memory, and vLLM repetition 1 is running. This is a **PENDING/ACTIVE** benchmark attempt: no partial throughput, latency or memory ratio is binding, `b5c6e4f` is not replaced, and 35B is not running. | Finish all three interleaved ours/vLLM ladders, six memory/cache returns and the paired trace; generate the canonical 27-only summary. If any 27B axis remains red, rank and repair the next executed-kernel gap before another exact grid. Gate optional W3-C separately; only then run 35B. |
| `SERVE-ASYNC-LLM` HTTP capacity | **GPU-CLASSIFIED — HEALTHY / STEADY-STATE NEUTRAL; ROW GATING** | Production replaces cpp-httplib's racy 19→76 dynamic pool with a fixed **`max_num_seqs + 4`** floor (36 workers at c32); `VLLM_CPP_HTTP_FIXED_POOL=0` selects the legacy arm in the same binary. The c32 fixed/legacy AB/BA/AB means are **1097.031/1097.290 tok/s = 0.999764×**, with **0.541%/0.311% CV** and 8/20 fixed axes. All **1,152/1,152** requests and six memory returns pass; neither arm reproduces the rare historical stall. The fresh exact fixed ladder completes all three c32 legs without a queued/unread socket and narrows the current c32 oracle ratio to 0.9910×. Fixed/legacy mean GPU peaks are **39,198/38,993 MiB**; fixed PSS/RSS are slightly lower. CPU evidence remains Release/help, API **100/100**, ASan+UBSan **1/1**, and TSan **1/1**. Summary/artifact hashes are `3ce27a16…18ee9` / `27bc7f7d…53df6d`. | The bounded A/B proves no steady-state speed win and did not sample the legacy rare tail, so the broader row remains `GATING`. No more HTTP tuning is inferred: repair the confirmed FP4 path and use the exact full-grid gate to classify the remaining performance gap. |
| `KV-EXTERNAL-CACHE` / LMCache | **ROADMAP INVENTORY — NOT BENCHMARKED** | Pinned vLLM's config roles, scheduler/worker connector lifecycle, dynamic module override, load-failure policy and built-in LMCache MP/in-process connectors are now explicit source inventory, along with the official LMCache shared-prefix quickstart. vllm.cpp has no connector ABI or LMCache execution path yet, so no hit rate, TTFT, transfer-throughput, memory or reliability result exists. | Write the full spike, port a deterministic fake-provider conformance seam, then gate LMCache MP two-engine store/retrieve and Qwen3.6 hybrid behavior before the in-process leaf. Required axes: token correctness, hit/recompute behavior, TTFT, transfer GB/s, host/GPU memory, failures and metrics. |
| `KERNEL-GEMM-NVFP4-W4A4` small-M dispatch | **W3-B EXACT GRID ACTIVE / FP4 STRUCTURAL PARITY** | W1/W2, immutable W3-A and corrected W3-B trace evidence remain unchanged. Clean pushed `3cc490c` is now executing the exact c1-c32 W3-B oracle gate under one uncontended model-wide lock. The commit-bound model gate and corpus/cache preconditions pass; ours repetition 1 completed all points and memory return, and vLLM repetition 1 is running. No partial rate is accepted and the prior `b5c6e4f` floor remains binding until the complete summary and trace validate. | Finish the active exact grid. If decode remains red, use the corrected trace's remaining dependent launch/traffic differences rather than revisiting tactic availability. W3-C persistence stays optional; do not run 35B. |
| `KERNEL-GDN-AOT-BF16` 27B output dtype | **27B DEFAULT / CORRECTNESS-GREEN; STRICT GATE OPEN** | The BF16 `chunk_o` path carries the 27B recurrence output, z projection and gated-norm weight by default, matching vLLM and restoring the native 16/16 stream; `VT_GDN_OUT_BF16=0` restores f32 and every 35B path retains f32. Its BF16/f32 component remains **1.007989×**, **16/20** timing and **2/4** memory. Clean `b5c6e4f` classifies the exact oracle: c16 total throughput passes at **1.021341×**, but normalized mean TPOT/ITL is **0.982670×**; c1-c8 remain decode-shaped gaps. Corrected W3-B tracing now matches FP4 aggregate time while diagnostic mean TPOT stays red; no GDN attribution follows from total category sums alone. | Keep correctness-faithful BF16 for 27B and retain the row `ACTIVE`; use the fresh exact W3-B grid to re-rank GDN decode/pool evidence against the other remaining traced gaps. Do not infer any 35B result. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — W0+W1 A/B/TRACE/CORRECTNESS PASS; ZERO-LEAK FAIL** | W0's clean build, default/fallback pointer/model gates, +2.1239% A/B, trace and both-model access checks remain valid. W1 indexed BF16↔F32 state I/O averages **+0.6246%**, improves **20/20** axes and sharply reduces D2D traffic. CUDA/model/turnover/sanitizer gates pass; inherited pools still fail strict teardown (27B **47,290,056 B/101**, 35B **36,822,413,188 B/1,236**). Corrected W3-B tracing now closes the FP4 tactic-family mismatch, while the binding host-memory axes remain red. | Keep device-residency W2 scoped until the exact W3-B grid re-ranks the residual; separately repair model/pool/queue teardown. |

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

### Reproduce the W3-A correctness and component checkpoints

Use immutable `71f1e894d0c5e496607d08cfe9089a9944128271` in one clean detached
checkout on `dgx.casa`. The two test arms use the same binary and one uncontended GPU lock;
the default delayed arm must report `delay=1000us`, while the fallback must
report `delay=off`. The test command is the correctness/selection preflight.
The archived component driver below then runs the exact c16/96 AB/BA/AB; copy
it to a fresh path and change only its `ev=` destination because it refuses to
overwrite evidence.

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

# Exact measured component driver (SHA-256 425f8521...e9ae).
W3="$HOME/work/vllm.cpp-nvfp4-small-m/71f1e894d0c5e496607d08cfe9089a9944128271/w3"
cp "$W3/component-ab/driver.sh" /tmp/w3a-component-repro.sh
sed -i "s|^ev=.*|ev=$W3/component-ab-repro|" /tmp/w3a-component-repro.sh
cd "$W3/source"
/tmp/w3a-component-repro.sh
```

### Reproduce the W3-B immutable and component checkpoints

Run this from clean pushed `d7cdf66` (or a later code-identical checkpoint) on
`dgx.casa`. The model log must contain exactly one
`profiles_requested=80 profiles_tuned=80 cached_plans=80` completion, no
`lazy-miss after pre-serve warmup`, and the 16/16 production stream.

```sh
SHA=$(git rev-parse HEAD)
BUILD="$HOME/work/vllm.cpp-nvfp4-small-m/$SHA/w3b/build"
cmake -S . -B "$BUILD" -DVLLM_CPP_CUDA=ON -DVLLM_CPP_TRITON=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" --target test_ops_nvfp4_fp4 \
  test_qwen27_paged_engine server --parallel "$(nproc)"
flock /tmp/gpu sh -c '
  "'"$BUILD"'"/tests/test_ops_nvfp4_fp4 &&
  VT_FP4_EXACT_BUCKETS=0 "'"$BUILD"'"/tests/test_ops_nvfp4_fp4 &&
  VT_FP4_AUTOTUNE_VERBOSE=1 \
    "'"$BUILD"'"/tests/test_qwen27_paged_engine
'

# Exact measured W3-B component driver (SHA-256 e996e6dd...662).
W3="$HOME/work/vllm.cpp-nvfp4-small-m/d7cdf66db0cfcc53d68d49613623ec6cd3807641/w3b"
cp "$W3/component-ab/driver.sh" /tmp/w3b-component-repro.sh
sed -i "s|^ev=.*|ev=$W3/component-ab-repro|" /tmp/w3b-component-repro.sh
cd "$W3/source"
/tmp/w3b-component-repro.sh

# Corrected paired trace (SHA-256 af29681e...22fbf). This is intentionally
# tied to immutable d7cdf66 and writes a new evidence root.
cp "$W3/trace-ab-oracle-r2/driver.sh" /tmp/w3b-trace-repro.sh
sed -i "s|^ev=.*|ev=$W3/trace-ab-oracle-r2-repro|" /tmp/w3b-trace-repro.sh
cd "$W3/source"
/tmp/w3b-trace-repro.sh
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
