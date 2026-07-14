# Benchmarks

This is the public, current-state benchmark scoreboard for vllm.cpp. It keeps
only binding results, current component dispositions, pending gates, and exact
reproduction entry points. Detailed attempt history, artifact hashes, and
failure forensics live in the [append-only parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), and linked feature specs.

Last updated: **2026-07-13**. The binding 27B result remains immutable
`3f256ab`; parity against vLLM v0.25.0 is **FAILED / open at 55/124 axes**.
W3-H's accepted trace selects fused SiLU→FP4 as the largest positive mapped
residual. W3-I1 is now **structurally accepted** at clean `15c6b89`: the exact
CUDA build, operator/sanitizer/model gates, commit-bound SASS, and paired
27B/35B traces pass. Its packed graph body plus required scale zeroing is
**36.68% shorter** than the fallback fused slice. That is diagnostic evidence,
not an end-to-end ratio; the complete c2/c16 component is still pending and no
binding speed number changed.

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
| W3-H1d complete trace | **PASS — DIAGNOSTIC TRACE ACCEPTED** | Clean `c498a413` passes exact 154/154 build, 27B 1/1 correctness, frozen plans, three 48/48 + 16/16 local sessions, 12/12 lossless reports, and the paired vLLM trace. Validator `7112864` writes passing status SHA `84d15970…6e66` | Retain as the executed-path attribution baseline; any later trace must meet the same fail-closed contract |
| W3-H2 vectorized normal BF16→FP4 I/O | **DEFERRED / NOT IMPLEMENTED** | Diagnostic residual is **+0.313930 ms/window**, but the fused producer is larger in 12/12 reports | Do not implement W3-H2 first |
| W3-I fused SiLU→FP4 producer | **ACTIVE / STRUCTURE PASS; PERFORMANCE PENDING** | Clean `15c6b89`: CUDA build **158/158 targets**; candidate OFF/ON operator **22/22 + 26,916/26,916** each; strict no-pool memcheck **0 errors / 0 leaks**; both 27B arms **235/235 + 16/16**; 35B **2/2 + 315/315**. SASS is **816 instructions / 36 registers / two 256-bit loads / one 64-bit output store**. Paired trace proves **896/896** eligible graph calls use the packed body, **896** matching zero nodes, unchanged capture/runtime lifecycle, and zero W3-I calls on 35B. No same-binary rate exists | Run all c2/c16 **40 timing + 8 memory** axes with the immutable driver below |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes, but no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | The cited external comparison mismatched prefix-cache, KV dtype/capacity, MTP, repetitions, and required axes. Its large headline gap does not bind | After cache-off parity, compare equivalent vllm.cpp, vLLM v0.25.0, and SGLang v0.5.15 cache-on workloads; the faster equivalent engine binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI, two-engine store/retrieve, hybrid-cache behavior, metrics, and failure policy are roadmap inventory only | Write the spike, port a deterministic fake provider, then gate LMCache MP before in-process mode |
| Stream-usage serialization | **PENDING / GATING** | CPU and sanitizer contracts pass; native 128-token counts are present in all binding requests | Run its serialization A/B after the model hot path closes |
| Async HTTP capacity | **IMPLEMENTED / STEADY-STATE NEUTRAL** | Fixed/legacy c32 mean ratio **0.999764×**, 8/20 axes; 1,152/1,152 requests and all lifecycles pass | Keep the safe fixed worker floor; do not treat it as a speed lever |

The accepted trace root is
`~/work/vllm.cpp-executed-path-refresh-h1d/c498a4131af7e6cf0ac678841212af80f4f12d53`.
It contains three distinct Nsight sessions and 12/12 lossless single-replay
reports, each with the exact 1,107-node graph and zero eager work, plus the
paired vLLM trace. The three local semantic output-array digests differ; the
model gate still passes and the gate spec explicitly keeps batch-invariance
digests diagnostic. Passing status SHA is `84d15970…6e66`; canonical node
multiset SHA is `c357867c…68b`. Superseded attempts and detailed hashes remain only in the
[parity ledger](../.agents/parity-ledger.md) and [state log](../.agents/state.md).

### W3-I1 immutable structural result

The current root is
`~/work/vllm.cpp-nvfp4-fused/15c6b8933d982019aa8965d218deb0eb1d9dc3f4-r2`.
The fallback/candidate Nsight report SHA-256 values are `488770b1…5b6b` /
`a1621e3c…47e3`; their SQLite SHA-256 values are `fa812990…454e` /
`1310981c…f9c6`. The commit-bound object/SASS SHA-256 values are
`d6ca771b…65bb` / `662f2c54…4102`.

| Graph slice across 14 forwards | Fallback | Packed candidate |
|---|---:|---:|
| Logical fused-producer calls | 896 | 896 |
| Fused body | 6.064064 ms | 2.747840 ms |
| Required explicit scale zeroing | integrated in scalar body | 1.091968 ms / 896 nodes |
| Comparable fused slice | **6.064064 ms** | **3.839808 ms** |

The comparable slice falls by **2.224256 ms / 36.68%**, or about
**0.158875 ms per forward**. Both traces record 14 graph launches, identical
allocation/free/synchronization counts, and only the same seven expected
`cudaMemcpyAsync` calls during capture; graph copies are identical and contain
no D2H. The 35B candidate trace (report/SQLite SHA `dd92270b…9ca7` /
`7a4abf1c…f8fa`) passes correctness and contains zero W3-I kernel calls.
Whole-process profiler duration is intentionally not treated as a rate.

Run the pending component with the preflighted immutable driver. It owns one
`/tmp/gpu` lock for both model gates and all 12 AB/BA/AB legs; exit 0 means all
48 axes pass, while exit 1 after a complete evidence tree means strict failure.

```sh
ssh dgx.casa \
  '$HOME/work/vllm.cpp-nvfp4-fused/15c6b8933d982019aa8965d218deb0eb1d9dc3f4-r2/w3i-component-driver.sh'
```

Driver SHA-256 is `0f08750f…ae1b`; its frozen native-plan document is
`2590fc94…199d`. Partial output is never binding.

### Executed-path diagnostic ranking

These are cross-profiler attribution values, not a benchmark ratio or speed
credit. Ours is the median of 12 exact one-replay reports; vLLM is the clean
1,476-window mean from the paired Torch trace. The canonical derived ranking
is `~/work/vllm.cpp-trace-validator/71128642ce04c191f559ea4ccabe4b7e33a66b0f/c498a413-residual-ranking.json`,
SHA-256 `7c323248…2456`.

| Mapped family | Ours ms/window | vLLM ms/window | Ours − vLLM |
|---|---:|---:|---:|
| Fused SiLU→FP4 producer | 0.595536 | 0.238182 | **+0.357354** |
| Normal BF16→FP4 producer | 0.655728 | 0.341798 | **+0.313930** |
| FA2 main | 4.816544 | 4.685569 | **+0.130975** |
| FA2 combine | 0.082480 | 0.082735 | −0.000255 |
| FP4 GEMMs | 54.120144 | 54.430211 | −0.310067 |
| GDN recurrence | 12.969872 | 19.266843 | −6.296971 |
| All GPU kernels | 124.574592 | 128.153486 | −3.578894 |

The fused-producer delta exceeds the normal-producer delta in every one of the
12 reports, so the W3-H contract displaced normal W3-H2. W3-I0 completed the
required source/generated-code/SASS/test inventory and W3-I1 now passes its
immutable structural gate; it still contributes no benchmark ratio. The
negative whole-window delta is consistent with c16 total throughput already
passing; it does not
explain or close the binding low-concurrency and host-memory failures.

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

## Reproduce the accepted schema-v5 W3-H1d trace

The accepted status was reconstructed from the immutable `c498a413` artifacts
with validator `71128642ce04c191f559ea4ccabe4b7e33a66b0f`; its SHA-256 is
`84d15970d5a68e8a6307949a78eb33fbe5db3104c70129abd3d2ae0bb3696e66`.
The full fresh-run recipe below creates a new SHA-owned root; never append a
new execution to `c498a413`.

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

Acceptance requires all 12 reports to be complete, zero-eager, exact-plan,
launch/workload/event-counter-reconciled, completion-ordered, and mutually
signature-identical. The exact capture-boundary warning may be recorded only
under that complete pinned-version contract; every other severity-2 diagnostic
still fails. A valid trace is diagnostic: only a later 48/48 same-binary
component gate may authorize the exact 27B grid.

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
