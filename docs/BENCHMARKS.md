# Benchmarks

This is the public, current-state benchmark scoreboard for vllm.cpp. It keeps
only binding results, current component dispositions, pending gates, and exact
reproduction entry points. Detailed attempt history, artifact hashes, and
failure forensics live in the [append-only parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), and linked feature specs.

Last updated: **2026-07-14**. The binding 27B result remains immutable
`3f256ab`; parity against vLLM v0.25.0 is **FAILED / open at 55/124 axes**.
The post-W3-I scan is complete: the largest binding low-concurrency symptom is
c2 decode TPOT (**114.841 vs 108.274 ms**, ours **6.1% slower**) while ours has
better TTFT. The next checkpoint is an exact vLLM async-scheduler ON/OFF c2
control on the same corpus; its timing and paired trace are **PENDING**, so W3
has no speed credit yet. Host PSS/RSS is independently traced to a persistent
**22.920 GiB** CPU weight mirror plus source-page residency during load. W3-I
remains default-off after its **30/48** component failure.

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
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | `3f256ab` binds at **55/124**; no later result supersedes it | Run the exact c2 async ON/OFF oracle control, then gate the selected lever before rerunning the grid |
| vLLM async-scheduler credit | **PENDING / DIAGNOSTIC PREPARED** | Binding logs confirm v0.25.0 resolved async ON. The profiler supports explicit `default`/`on`/`off` modes and records the resolved mode; local contract tests pass | Paired c2 timing plus Torch traces under one uncontended GPU lock; promote W3 only if the credit explains the decode gap |
| Host-weight ownership | **FAILED / ROOT CAUSE DIAGNOSED** | Exact selected-tensor accounting finds **24,610,136,064 B / 22.920 GiB** retained in host `OwnedTensor` storage; mmap pages overlap that copy during load | Direct-to-final-device streaming design and all-axis memory A/B after the speed lever is selected |
| W3-H1d complete trace | **PASS — DIAGNOSTIC TRACE ACCEPTED** | Clean `c498a413`, 12/12 lossless local reports and paired vLLM trace; status SHA `84d15970…6e66` | Retain as the c16 executed-path baseline; low-batch traces supersede it only under the same fail-closed contract |
| W3-I fused SiLU→FP4 producer | **STRUCTURE PASS / COMPONENT FAILED** | Clean `15c6b89`; 612/612 requests, **27/40 timing + 3/8 memory**, c2/c16 totals **1.002457× / 0.999771×** | Keep default-off; no speed credit or exact grid |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes, but no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | The cited external comparison mismatched prefix-cache, KV dtype/capacity, MTP, repetitions, and required axes. Its large headline gap does not bind | After cache-off parity, compare equivalent vllm.cpp, vLLM v0.25.0, and SGLang v0.5.15 cache-on workloads; the faster equivalent engine binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI, two-engine store/retrieve, hybrid-cache behavior, metrics, and failure policy are roadmap inventory only | Write the spike, port a deterministic fake provider, then gate LMCache MP before in-process mode |
| Stream-usage serialization | **PENDING / GATING** | CPU and sanitizer contracts pass; native 128-token counts are present in all binding requests | Run its serialization A/B after the model hot path closes |
| Async HTTP capacity | **IMPLEMENTED / STEADY-STATE NEUTRAL** | Fixed/legacy c32 mean ratio **0.999764×**, 8/20 axes; 1,152/1,152 requests and all lifecycles pass | Keep the safe fixed worker floor; do not treat it as a speed lever |

## Current diagnostic context

The accepted c16 execution baseline remains
`~/work/vllm.cpp-executed-path-refresh-h1d/c498a4131af7e6cf0ac678841212af80f4f12d53`
(status `84d15970…6e66`). It shows ours already faster in aggregate GPU-kernel
time at c16, so it cannot explain the c2 decode gap. The completed residual
scan ranks the live evidence as follows:

| Finding | Binding interpretation |
|---|---|
| vLLM depth-2 async scheduling + GPU-resident sampled-token path | Only unmeasured structural difference large enough to explain 4–6%; exact ON/OFF control is next |
| RMSNorm/generated partitions | Largest unmapped GPU candidate; ours spends **2.094864 ms/window** across 129 RMSNorm nodes, but oracle partitions need exact node/adjacency mapping before a speed claim |
| Normal BF16→FP4 | Grounded **+0.313930 ms/window** residual; estimated end-to-end ceiling is only about **0.25%** |
| Host weight ownership | **22.920 GiB** persistent host mirror plus load-time mmap residency; independent memory repair, not a decode-speed hypothesis |

W3-I's immutable component remains under
`~/work/vllm.cpp-nvfp4-fused/15c6b8933d982019aa8965d218deb0eb1d9dc3f4-r2/evidence/component-ab-c2-c16-fused-vec-flashinfer-r2`.
It completed 12/12 legs and 612/612 requests but failed **30/48** axes; summary
SHA is `b7cfa029…7c17`. Full per-attempt forensics and prior component results
remain in the [parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), and linked feature specs rather than this
scoreboard.

## Run the pending async-credit traces

Run both oracle modes under one uncontended lock; the online AB/BA/AB timing
series uses the same c2 corpus and server flags before either trace is
interpreted. These commands create diagnostic traces, not binding speed credit:

```sh
MODEL="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a"
CORPUS="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560/corpus/27/vllm/c2-r1.jsonl"
OUT="$HOME/work/vllm-async-credit/702f4814fe54/c2"

flock /tmp/gpu bash -c '
  set -e
  for mode in on off; do
    "$HOME/venvs/vllm-oracle/bin/python" \
      tools/bench/profile_vllm_online_gate.py \
      --model "'$MODEL'" --corpus "'$CORPUS'" \
      --profile-dir "'$OUT'/trace-$mode" \
      --metadata "'$OUT'/trace-$mode.json" \
      --num-prompts 6 --max-concurrency 2 \
      --max-num-seqs 32 --max-num-batched-tokens 2048 \
      --repetitions 3 --async-scheduling "$mode"
  done
'
```

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
