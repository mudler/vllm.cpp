# Benchmarks

This is the public, current-state benchmark scoreboard for vllm.cpp. It keeps
only binding results, current component dispositions, pending gates, and exact
reproduction entry points. Detailed attempt history, artifact hashes, and
failure forensics live in the [append-only parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), and linked feature specs.

Last updated: **2026-07-13**. The binding 27B result remains immutable
`3f256ab`; parity against vLLM v0.25.0 is **FAILED / open at 55/124 axes**.
The W3-H1d schema-v4 trace repair is implemented and CPU-gated; a fresh
immutable DGX run is pending. No newer performance number exists.

## Binding 27B online gate

| Item | Binding value |
|---|---|
| Model | Qwen3.6-27B NVFP4 |
| Hardware | NVIDIA GB10 / DGX Spark, sm_121a |
| vllm.cpp source | `3f256abdbb558e162bf8a2196284deb119648560` |
| Reference | vLLM v0.25.0, tag `702f4814fe54fabff350d43cb753ae3e47c0c276`, FlashInfer 0.6.13 |
| Workload | Cache off, input 1,024 → output 128, greedy, closed loop, c1/c2/c4/c8/c16/c32 |
| Repetitions | Three interleaved repetitions per point |
| Evidence completeness | 12/12 performance groups, 2/2 memory groups, 124/124 axes eligible |
| Stability | Maximum total-throughput CV **0.189%** |
| Disposition | **FAILED: 55/124 pass, 69/124 fail** |

All ratios are direction-normalized: throughput is ours/vLLM, while latency
ratios are vLLM/ours, so **1.0 or higher passes**. Values are medians of the
three interleaved repetitions.

| Concurrency | Axes passing | Total tok/s ours / vLLM (ratio) | Output tok/s ours / vLLM (ratio) | Mean TTFT | Mean TPOT / ITL | Mean E2EL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5/20 | 81.645106 / 82.178953 (**0.993504×**) | 9.071678 / 9.130995 (**0.993504×**) | 1.038340× | 0.992092× | 0.993513× |
| 2 | 4/20 | 150.561023 / 157.744007 (**0.954464×**) | 16.729003 / 17.527112 (**0.954464×**) | 1.196031× | 0.942815× | 0.954468× |
| 4 | 5/20 | 280.291354 / 290.025183 (**0.966438×**) | 31.143484 / 32.225020 (**0.966438×**) | 1.066496× | 0.954755× | 0.964313× |
| 8 | 4/20 | 495.699906 / 505.466352 (**0.980678×**) | 55.077767 / 56.162928 (**0.980678×**) | 1.382124× | 0.941853× | 0.980621× |
| 16 | 17/20 | 812.302839 / 790.263558 (**1.027889×**) | 90.255871 / 87.807062 (**1.027889×**) | 1.432626× | 0.987450× | 1.027464× |
| 32 | 18/20 | 1121.954512 / 1079.407095 (**1.039417×**) | 124.661612 / 119.934122 (**1.039417×**) | 1.446098× | 1.002666× | 1.039521× |

| Memory axis | Ours | vLLM | Normalized ratio | Result |
|---|---:|---:|---:|---|
| Peak PSS | 48,175,537 KiB | 28,167,719 KiB | 0.584689× | **FAIL** |
| Peak RSS | 48,177,860 KiB | 28,534,276 KiB | 0.592269× | **FAIL** |
| Peak GPU memory | 38,561 MiB | 70,531 MiB | 1.829076× | PASS |
| Peak `MemAvailable` drop | 65,901,992 KiB | 80,911,844 KiB | 1.227760× | PASS |

Total throughput passes at c16/c32, but the project gate requires every total
and output throughput, request-rate, TTFT, TPOT/ITL, E2EL, and memory axis to
match or beat vLLM. The low-concurrency and host-memory gaps are therefore
binding failures, not noise or “near parity.” No 35B performance command is
authorized until all 124 27B axes pass.

## Current checkpoint

| Track | Disposition | Current evidence | Next binding gate |
|---|---|---|---|
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | `3f256ab` binds at **55/124**; no later result supersedes it | Repair the selected hot path and rerun the exact 27B grid |
| W3-H1d lossless trace | **ACTIVE — CPU PASS / DGX PENDING** | Schema v4 requires 3 independent sessions × 4 indexed reports, `repeat:4:sync`, device synchronization before stop, grouped/distinct UUIDs, one exact 1,107-kernel replay per report, zero eager work, exact frozen plans, and no severity-2 diagnostic. Focused contracts pass **31/31**; the final CUDA-off suite passes **106/106** | Run from a new pushed SHA and immutable evidence root; all 12 reports must pass before W3-H2 |
| W3-H2 vectorized BF16→FP4 I/O | **PENDING / NOT IMPLEMENTED** | The scalar implementation remains active; retained traces are diagnostic only | Implement only if W3-H1d produces complete lossless evidence and still selects this residual |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes, but no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | The cited external comparison mismatched prefix-cache, KV dtype/capacity, MTP, repetitions, and required axes. Its large headline gap does not bind | After cache-off parity, compare equivalent vllm.cpp, vLLM v0.25.0, and SGLang v0.5.15 cache-on workloads; the faster equivalent engine binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI, two-engine store/retrieve, hybrid-cache behavior, metrics, and failure policy are roadmap inventory only | Write the spike, port a deterministic fake provider, then gate LMCache MP before in-process mode |
| Stream-usage serialization | **PENDING / GATING** | CPU and sanitizer contracts pass; native 128-token counts are present in all binding requests | Run its serialization A/B after the model hot path closes |
| Async HTTP capacity | **IMPLEMENTED / STEADY-STATE NEUTRAL** | Fixed/legacy c32 mean ratio **0.999764×**, 8/20 axes; 1,152/1,152 requests and all lifecycles pass | Keep the safe fixed worker floor; do not treat it as a speed lever |

The previous H1 trace attempts through clean `b2c940c` are all **VOID** and are
intentionally not narrated here. They produced no accepted speed result. The
latest attempt proved that each isolated report contains one complete graph
replay and zero eager CUDA rows, but all four reports still carried Nsight's
severity-2 possible-event-loss diagnostic and the old schema expected one
combined report. That observation selected schema v4; it did not validate or
reclassify the old evidence. Exact roots, hashes, and failure causes remain in
the [parity ledger](../.agents/parity-ledger.md) and
[state log](../.agents/state.md).

## Current component dispositions

These same-binary components explain which implementations are retained, but
none substitutes for the end-to-end 124-axis gate.

| Component | Current result | Disposition |
|---|---|---|
| Frozen NVFP4 plan cache | Exact vLLM v0.25.0 fixture loads 64/64 plans with zero tuning/misses in read-only benchmark mode | **PASS as benchmark control**, not a speed claim |
| Packed full-attention QKV | Closes the trace topology at **208.192 vs 208 FP4 GEMMs/forward**; isolated c16 mean total throughput +0.5049%, 14/20 timing + 2/4 memory | Retained for structural parity; strict component **FAILED** |
| Direct swizzled activation scales | c2/c16 mean ratios **1.004483× / 1.005044×**; 39/40 timing + 1/8 memory | Correctness/trace pass; strict component **FAILED**, no speed credit |
| Model-owned device alpha | c2/c16 mean ratios **1.001967× / 1.000144×**; 27/40 timing + 3/8 memory | Correctness/trace pass; strict component **FAILED**, no speed credit |
| FA2 ratio-6 split-KV decode | c2/c16 mean ratios **1.017668× / 1.006548×**; 35/40 timing + 5/8 memory | Correctness/dispatch pass; strict component **FAILED**, no speed credit |
| 27B BF16 GDN output | Mean ratio **1.007989×**; 16/20 timing + 2/4 memory | Retained for vLLM/token correctness; strict gate open |
| Fixed HTTP worker floor | Fixed/legacy ratio **0.999764×**; 8/20 axes | Safety/capacity fix retained; performance neutral |
| Device-resident indexed GDN state | Indexed/fallback mean +0.6246%, 20/20 timing axes, large copy reduction | Correctness/A-B/trace pass; strict zero-leak lifecycle still open |

Historical vLLM 0.24.0 grids and workload-mismatched direct-library numbers
are omitted from this snapshot. They remain reproducible history, but cannot
establish current production-vLLM parity. See the ledger for those values.

## Reproduce the binding result

The immutable evidence can be re-aggregated without GPU work. Exit **1** is
expected because the evidence is complete but the every-axis gate fails. Exit
2 indicates malformed evidence or a harness error.

```sh
SOURCE="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"
CHECK="/tmp/vllm-cpp-3f256ab-summary-$USER"
cp -a --reflink=auto "$SOURCE" "$CHECK"
rm -rf "$CHECK/summary-27"

set +e
PYTHONPATH="$PWD" python3 tools/bench/online_gate_summary.py \
  --evidence "$CHECK" --model 27
rc=$?
set -e
test "$rc" -eq 1
sha256sum "$CHECK"/summary-27/{all-runs.json,ratios.json,report.md}
```

## Reproduce the pending W3-H1d trace gate

Use a clean, pushed schema-v4 commit in a new detached DGX source and a new
SHA-owned root. Plan in an empty evidence directory before copying the frozen
corpus. The driver owns and records the exact target build and one uncontended
GPU lock. Never reuse an earlier H1 root.

```sh
set -euo pipefail
SHA=$(git rev-parse HEAD)
ROOT="$HOME/work/vllm.cpp-executed-path-refresh-h1d/$SHA"
BINDING="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"

"$ROOT/source/scripts/dgx-online-serving.sh" --dry-run \
  --claim-root "$ROOT" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"

mkdir -p "$ROOT/evidence/$SHA/corpus"
cp -a "$BINDING/corpus/27" "$ROOT/evidence/$SHA/corpus/27"

cmake -S "$ROOT/source" -B "$ROOT/build-cuda" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$HOME/venvs/vllm-oracle/bin/ninja" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_BUILD_TESTS=ON \
  -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a \
  -DVLLM_CPP_FLASH_ATTN=ON \
  -DVLLM_CPP_TRITON=ON \
  -DVLLM_CPP_TRITON_REGEN=OFF \
  -DVLLM_CPP_BENCH_PROFILE_CONTROL=ON \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$ROOT/configure.log"

export VT_FP4_PERSISTENT_CACHE=1
export VT_FP4_FLASHINFER_CACHE_PATH="$ROOT/source/tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
export VT_FP4_AUTOTUNE_CACHE_READONLY=1
export VT_FP4_AUTOTUNE_CACHE_PATH="$ROOT/evidence/$SHA/readonly-native-must-not-exist.json"
export VT_FP4_AUTOTUNE_DELAY_US=5000
export VT_FP4_PLAN_CACHE=1
export VT_FP4_AUTOTUNE=1
export VT_FP4_FULL_TACTICS=1
export VT_FP4_PRE_SERVE_WARMUP=1

"$ROOT/source/scripts/dgx-online-serving.sh" --trace-only --model 27 \
  --snapshot "$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a" \
  --source-corpus "$ROOT/evidence/$SHA/corpus/27" \
  --evidence "$ROOT/evidence/$SHA" \
  --build-dir "$ROOT/build-cuda" \
  --configure-log "$ROOT/configure.log" \
  --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
```

Acceptance requires all 12 reports to be lossless, zero-eager, exact-plan,
launch/workload-reconciled, and mutually signature-identical. A valid trace is
still diagnostic: only a later 48/48 same-binary component gate may authorize
the exact 27B grid.

## Benchmark policy

- Correctness is a precondition and cannot be traded for speed.
- Ours and the reference use the same model, requests, sampling, token budget,
  cache policy, concurrency, and hardware.
- A benchmark series runs on an idle GPU under one ownership window and is
  repeated enough to distinguish signal from run noise.
- Partial, contended, stale-denominator, or diagnostically incomplete results
  are `VOID`; they never contribute to an accepted ratio.
- Every 27B throughput, latency, and memory axis must pass before 35B
  performance or broader roadmap execution.

The complete contract is in the
[benchmark protocol](../.agents/benchmark-protocol.md) and
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md).
