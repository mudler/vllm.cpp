# Benchmarks

This is the public current-state scoreboard for vllm.cpp. It contains the
binding result, the active performance diagnosis, pending gates, and current
reproduction entry points. Attempt chronology and failure forensics live in the
[parity ledger](../.agents/parity-ledger.md),
[state log](../.agents/state.md), linked specs, and Git. Those raw records are
append-only within the current era and are frozen under `.agents/completed/`
when the era is rolled up; this page never accumulates their run-by-run history.

Last updated: **2026-07-14**. Qwen3.6-27B parity against vLLM v0.25.0 is
**FAILED / open at 55/124 axes**. The active implementation leaf is now exact:
our 48 GDN layers issue four BF16 input projections each, while vLLM packs them
into qkvz and ba. That explains all **96** extra BF16 GEMM launches. The
[merged-projection spike](../.agents/specs/gdn-merged-input-projections.md) has
a `GATING` BA-only W1 implementation. Clean pushed `581d335` closes its core
correctness/safety repetition; BF16 output still fails the known near-tie.
Immutable trace `0091cd1`, finalized by pushed `8a1f923`, now has durable
status **`complete-structural`**. All 12 merged ranges are exactly
**963 / 145 BF16** and all 12 split ranges are **1,011 / 193**; every selected
non-BF16 family is unchanged. The fresh vLLM controls retain 1,522/1,521
internally invariant steady windows, accepted names/geometry/families/tactics,
and the two exact inspected Torch/Inductor register signatures. This proves an
exact 48-BF16-only removal but remains non-binding. The 40+8-axis component is
**PENDING**, so no accepted performance number changes.

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
| `KERNEL-GEMM-BF16` | **GATING / W1 BA STRUCTURE PASS** | `0091cd1` is durably `complete-structural`: 24/24 exact local ranges, merged **963 / 145 BF16** versus split **1,011 / 193**, exactly 48 BF16-only removals, unchanged non-BF16 families, and 1,522/1,521 invariant vLLM control windows. `benchmark_binding=false`; BF16 output remains **FAILED at 233/235** | Finish GGUF/legacy inertness and BF16 rounding, then run c2/c16 AB/BA/AB before qkvz |
| RMSNorm/generated partitions | **PENDING / QUEUED** | Equal 177-call structure remains the next positive diagnostic residual after the merge | Whole-chain spike only after the merged-projection checkpoints |
| Host-weight ownership | **FAILED / DIAGNOSED** | **24,610,136,064 B / 22.920 GiB** retained in host weight tensors plus overlapping source mmap pages | Direct-to-final-device streaming design and all-axis memory A/B |
| Qwen3.6-35B-A3B performance | **BLOCKED / NOT RUN** | Correctness passes; no current v0.25.0 performance denominator exists | Run only after 27B reaches 124/124 |
| SGLang shared-prefix floor | **PENDING / NO ACCEPTED NUMBER** | No equivalent cache-on vllm.cpp/vLLM/SGLang campaign exists | After cache-off parity, gate equivalent vLLM v0.25.0 and SGLang v0.5.15; the faster reference binds each axis |
| External KV / LMCache | **NOT IMPLEMENTED / NOT BENCHMARKED** | Connector ABI and two-engine store/retrieve remain roadmap inventory | Spike fake-provider semantics, then gate LMCache MP before in-process mode |

Closed controls do not remain active leaves: async scheduling measured neutral
for speed, and the prior fused-producer candidate remains default-off after its
strict component failure. Their exact results and reproduction history remain
in the append-only record rather than this live scoreboard.

The completed core correctness/safety root is
`~/work/vllm.cpp-gdn-ba/immutable-581d335fec2e5a96d9ccbb38c1ec001c39ac1789`.
Status / artifact-list SHA-256 values are `3895e658…4cf6` / `ed2bf8d8…895b`.
Focused CTest, merged/split 27B, 35B, and memcheck log hashes are
`4cf699ad…759b`, `c2a6f93f…cf96` (both arms), `b926716e…9875`, and
`a3d61cb9…fb87`; fixture `e81e9181…7edd` loaded 64 plans and the forbidden
native cache stayed absent. The rejected BF16-output preflight log remains
`09078b76…b050`. This closes immutable core correctness/safety only and earns no
speed ratio.

The completed structural root is
`~/work/vllm.cpp-gdn-ba-trace/0091cd192d9a6baa2197a4f3bdb0561bd859baf5`.
All 24 local ranges pass their exact contracts. The merged/split oracle traces
have SHA-256 `b8d26d4c…fc59` / `cef841ce…ede5`; they contain 1,522 / 1,521
internally invariant steady B=2 windows at 1,160 kernels and ordered-name SHA
`858915dd…fad0`. Their full launch signatures are `17e1037e…14ed` /
`f7a3ca1f…cadf`. Pushed finalizer commit `8a1f923` wrote the marker last with
status `complete-structural`. Summary / manifest / marker / artifact-set /
finalizer SHA-256 values are `03601168…54d5` / `b203f0d2…5412` /
`72328c48…63e` / `b93fd633…70a2` / `57395e99…b146`. The summary proves merged
963/145 versus split 1,011/193, exact 48/48 deltas, unchanged non-BF16 family
counts and `benchmark_binding=false`; it grants no speed ratio.

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

Verify the pushed-SHA core correctness/safety checkpoint without rerunning the GPU:

```sh
SHA=581d335fec2e5a96d9ccbb38c1ec001c39ac1789
ROOT="$HOME/work/vllm.cpp-gdn-ba/immutable-$SHA"
test "$(git -C "$ROOT/src" rev-parse HEAD)" = "$SHA"
test -z "$(git -C "$ROOT/src" status --porcelain)"
sha256sum "$ROOT/evidence"/{status.txt,sha256sums.txt}
# Expected: 3895e658…4cf6 / ed2bf8d8…895b
```

Verify the durable structural checkpoint without GPU work:

```sh
RAW_SHA=0091cd192d9a6baa2197a4f3bdb0561bd859baf5
ROOT="$HOME/work/vllm.cpp-gdn-ba-trace/$RAW_SHA"
TRACE="$ROOT/evidence/$RAW_SHA/trace/27"
sha256sum "$TRACE"/{gdn-ba-summary.json,gdn-ba-manifest.json,status-gdn-ba.json}
# Expected: 03601168…54d5 / b203f0d2…5412 / 72328c48…63e
```

For a fresh reproduction, create a new SHA-owned root, write the plan before
copying the frozen corpus, configure the exact trace build, then run the
same one-lock driver and finalizer:

```sh
set -o pipefail
SHA=$(git rev-parse HEAD)
ROOT="$HOME/work/vllm.cpp-gdn-ba-trace/$SHA"
SOURCE="$ROOT/source"
EVIDENCE="$ROOT/evidence/$SHA"
BINDING="$HOME/work/vllm.cpp-online-gate/evidence/3f256abdbb558e162bf8a2196284deb119648560"
SNAPSHOT="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a"
test -z "$(git status --porcelain)"
git worktree add --detach "$SOURCE" "$SHA"
"$SOURCE/scripts/dgx-online-serving.sh" --dry-run \
  --claim-root "$ROOT" --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA"
mkdir -p "$EVIDENCE/corpus"
cp -a "$BINDING/corpus/27" "$EVIDENCE/corpus/27"
cmake -S "$SOURCE" -B "$ROOT/build-cuda" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$HOME/venvs/vllm-oracle/bin/ninja" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_BUILD_TESTS=ON -DVLLM_CPP_SERVER=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES=121a -DVLLM_CPP_FLASH_ATTN=ON \
  -DVLLM_CPP_TRITON=ON -DVLLM_CPP_TRITON_REGEN=OFF \
  -DVLLM_CPP_BENCH_PROFILE_CONTROL=ON \
  -DVLLM_CPP_CUTLASS_DIR="$HOME/venvs/vllm-oracle/lib/python3.12/site-packages/flashinfer/data/cutlass" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "$ROOT/configure.log"
"$SOURCE/scripts/dgx-online-serving.sh" --trace-only --model 27 \
  --snapshot "$SNAPSHOT" --source-corpus "$EVIDENCE/corpus/27" \
  --evidence "$EVIDENCE" --build-dir "$ROOT/build-cuda" \
  --configure-log "$ROOT/configure.log" --trace-concurrency 2 \
  --gdn-ba-mode both --client "$HOME/venvs/vllm-oracle/bin/vllm" \
  --vllm-cpp-sha "$SHA" 2>&1 | tee "$ROOT/gdn-ba-run.log"
python3 "$SOURCE/tools/bench/finalize_gdn_ba_trace.py" \
  --evidence "$EVIDENCE" --source-commit "$SHA" \
  --run-log "$ROOT/gdn-ba-run.log"
```

Acceptance requires 145 versus 193 BF16 GEMMs, total nodes 963 versus 1,011,
unchanged non-BF16 families and memcpy/memset counts. Exact build/corpus
requirements and acceptance rules are in the
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
