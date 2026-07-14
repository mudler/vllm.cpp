# Benchmarks

This is the public current-state scoreboard for vllm.cpp. It contains the
binding result, the active performance diagnosis, pending gates, and current
reproduction entry points. Attempt chronology and failure forensics live in the
[append-only parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), linked specs, and Git.

Last updated: **2026-07-14**. Qwen3.6-27B parity against vLLM v0.25.0 is
**FAILED / open at 55/124 axes**. The active implementation leaf is now exact:
our 48 GDN layers issue four BF16 input projections each, while vLLM packs them
into qkvz and ba. That explains all **96** extra BF16 GEMM launches. The
[merged-projection spike](../.agents/specs/gdn-merged-input-projections.md) has
a `GATING` BA-only W1 implementation. Its F32-output merged and split arms are
token-identical, while a BF16-output preflight fails the known near-tie. The
immutable trace and 40+8-axis component remain **PENDING**, so no accepted
number changes.

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

Ratios are direction-normalized: throughput is ours/vLLM, latency is
vLLM/ours, and **1.0 or higher passes**. Values are medians of three
interleaved repetitions.

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

Total throughput passes at c16/c32, but every throughput, request-rate,
TTFT, TPOT/ITL, E2EL, and memory axis must pass. At c2, ours has better TTFT
but decode TPOT is **114.841 vs 108.274 ms** (**6.1% slower**). No 35B
performance command is authorized until the 27B result reaches 124/124.

## Current checkpoint

| Track | Disposition | Current evidence | Next binding gate |
|---|---|---|---|
| `SERVE-GATE-ONLINE` | **FAILED / GATING** | Immutable `3f256ab` remains **55/124** | Complete the two merged-projection component checkpoints, then rerun the exact grid only if authorized |
| `KERNEL-GEMM-BF16` | **GATING / W1 BA IMPLEMENTED** | One raw-NK `[b,a]` owner, split rollback views, one merged F32-output GEMM, and stride-aware F32/BF16 consumers are production-build green. Frozen-fixture merged/split 27B logs are byte-identical at **235/235 + 16/16**; packed-view capture/replay memcheck is **590/590, 0 errors/leaks**; 35B is **315/315**. The BF16-output arm is **FAILED at 233/235** and selects the rejected emulation continuation | Repeat from the pushed SHA; prove 145-vs-193 structure, resolve exact BF16 rounding parity, then run c2/c16 AB/BA/AB before qkvz |
| RMSNorm/generated partitions | **PENDING / QUEUED** | Equal 177-call structure remains the next positive diagnostic residual after the merge | Whole-chain spike only after the merged-projection checkpoints |
| Host-weight ownership | **FAILED / DIAGNOSED** | **24,610,136,064 B / 22.920 GiB** retained in host weight tensors plus overlapping source mmap pages | Direct-to-final-device streaming design and all-axis memory A/B |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes; no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | No equivalent cache-on vllm.cpp/vLLM/SGLang campaign exists | After cache-off parity, gate equivalent vLLM v0.25.0 and SGLang v0.5.15; the faster reference binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI and two-engine store/retrieve remain roadmap inventory | Spike fake-provider semantics, then gate LMCache MP before in-process mode |

Closed controls do not remain active leaves: async scheduling measured neutral
for speed, and the prior fused-producer candidate remains default-off after its
strict component failure. Their exact results and reproduction history remain
in the append-only record rather than this live scoreboard.

The current implementation evidence is a mutable-source preflight, not binding
performance evidence. It lives under
`~/work/vllm.cpp-gdn-ba/evidence/precommit-final-{focused-20260714T071303Z,models-20260714T071327Z}`.
The focused CTest/memcheck hashes are `5fd62a85…f461` / `a3d61cb9…fb87`;
merged 27B, split 27B, and 35B logs are `e7c243cd…e7c8` /
`e7c243cd…e7c8` / `328e02e9…d348`, using frozen fixture
`e81e9181…7edd`. The rejected BF16-output log is `09078b76…b050`.
Focused ASan+UBSan is **2/2** with the existing process-lifetime CPU pool
excluded; a leak-enabled diagnostic attributes **16,960 B / 30 allocations**
to that inherited pool and finds no invalid access.
These results authorize an immutable rerun; they earn no speed ratio.

## Evidence selecting merged GDN projections

The immutable completed root is
`~/work/vllm.cpp-executed-path-c2/179a0fc2afc1c33b63d14de8e50d3fde976c7356`.
Its status is `complete-diagnostic`.

| Artifact | SHA-256 |
|---|---|
| c2 summary | `0ef6a1240d33c16410cd4e43b30ca8667a6d92e6eee8506d7bd03388fe010273` |
| c2 manifest | `2556cfd032fae2201d9f8deb818343731b7dc99d9f8e6329da9b793262712f21` |
| status | `9e0143fa1b9c74e218e486fedd0606850708619a0e859dafe94957e24a507b57` |
| artifact set | `cc248ad2b5bf08f85b0d6b178de70682a104917e16c59c9adf34d661217f823a` |
| fresh oracle trace | `2b3bf41269fd19ef65c5c3e06f067af73d7d997de3b6be17a2af785b6a86785c` |

All **12/12** local B=2 graph ranges are invariant at 1,011 kernels + 7
memcpy + 1 memset. The oracle contains **1,522** invariant steady B=2 windows
at 1,160 kernels plus two bounded B=1 drains. Both engines execute the same
**128 Stream-K + 80 static-persistent** FP4 tactic split.

| BF16 projection structure per B=2 window | vllm.cpp | vLLM v0.25.0 |
|---|---:|---:|
| qkv / packed qkvz | 48 | 48 |
| z | 48 | included in qkvz |
| b+a / packed ba | 96 | 48 |
| lm_head | 1 | 1 |
| Total | **193** | **97** |

The source arithmetic independently gives `(4-2) × 48 = 96`; this is not a
profiler-name classification artifact. Diagnostic family medians are
**51.662672 vs 48.798042 ms** (+2.864630 ms), with a shape-level decomposition
ranking BA at about 1.882 ms and qkvz at about 0.476 ms. These durations cross
Nsight/Torch-profiler domains and are not a speed ratio. The accepted spike
therefore gates BA and qkvz separately and forbids duplicate weight owners or
split-copy kernels.

## Verify or reproduce the current checkpoint

Verify the durable diagnostic without GPU work:

```sh
RAW_SHA=179a0fc2afc1c33b63d14de8e50d3fde976c7356
ROOT="$HOME/work/vllm.cpp-executed-path-c2/$RAW_SHA/evidence/$RAW_SHA/trace/27"
sha256sum "$ROOT"/{c2-summary.json,c2-manifest.json,status-c2.json}
# Expected prefixes: 0ef6a124…0273 / 2556cfd0…2f21 / 9e0143fa…7b57
```

Repeat the implementation gate from the pushed SHA before tracing; this is not
a benchmark claim:

```sh
cmake -S . -B build-gdn-merge -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a
cmake --build build-gdn-merge --target test_ops_gdn \
  test_qwen27_dense_forward test_qwen27_paged_forward \
  test_qwen27_paged_engine test_qwen36_paged_engine server --parallel
flock /tmp/gpu -c '
  ctest --test-dir build-gdn-merge \
    -R "(test_ops_gdn|test_qwen27|test_qwen36_paged_engine)" \
    --output-on-failure &&
  VT_GDN_MERGED_BA=0 ./build-gdn-merge/tests/test_qwen27_paged_engine
'
```

The first trace must use `nsys --cuda-graph-trace=node`. BA default must show
145 BF16 GEMMs per B=2 window and BA-off 193, with no new cast/copy node; after
qkvz, the corresponding target is 97. Exact immutable trace and AB/BA/AB
commands, toggles, model fixture, hashes, and acceptance rules are in the
[merged-projection spike](../.agents/specs/gdn-merged-input-projections.md).

Re-aggregate the binding result without GPU work; exit **1** is expected for a
complete every-axis failure, while exit 2 means malformed evidence:

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
```

## Benchmark policy

- Correctness is a precondition and cannot be traded for speed.
- Ours and every reference use the same model, requests, sampling, token
  budget, cache policy, concurrency, and hardware.
- A benchmark series runs on an idle GPU under one ownership window and is
  repeated enough to distinguish signal from run noise.
- Partial, contended, stale-denominator, or diagnostically incomplete results
  are `VOID`; they never contribute to an accepted ratio.
- Every 27B throughput, latency, and memory axis must pass before 35B
  performance or broader roadmap execution.

The complete contract is in the
[benchmark protocol](../.agents/benchmark-protocol.md) and
[online serving gate spec](../.agents/specs/cuda-online-serving-gate.md).
