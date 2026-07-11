# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-11** (exact `f065fce` 27B baseline complete and below
floor; `KV-DEVICE-RESIDENCY` W0 default/fallback GB10 model gates pass).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU/sanitizer gates pass. At `31d053f`, all 2,016 standard timed 27B requests across three complete paired ladders retained exact native 128-token counts, closing the prior missing-usage symptom; this does not close its performance/A-B gate. | Complete the serialization A/B and fresh 27B+35B every-axis campaigns after the online hot-path gap is repaired. |
| `SERVE-GATE-ONLINE` | **BELOW FLOOR — VALID 27B BASELINE** | Pushed `f065fce` completed the exact cache-off/closed-loop c1–c32 27B series: 2,016/2,016 timed requests, three repetitions, six verified memory returns, a passing commit-bound model gate, and a passing corrected trace contract. Mean total-throughput ratios are **0.9616/0.9258/0.9364/0.9450/0.9775/0.9722×**; only **4/4/5/3/3/3 of 20** timing axes pass. Memory passes **2/4** axes: mean PSS is 45.97 GiB ours vs 26.83 GiB vLLM, while reported GPU memory is 27,509 vs 72,725 MiB. Throughput CV is below 0.7% on both engines. Old `31d053f`, `ed6247d`, and historical offline ratios remain void for their recorded configuration mismatches; they are not blended into these values. | Run merged W0's repeated same-binary A/B and corrected trace, then W1 indexed state I/O and fresh exact direct-library/online 27B closure before 35B. |
| `KV-DEVICE-RESIDENCY` | **ACTIVE — CUDA CORRECTNESS PASS; PERF OPEN** | W0 replaces CUDA host-vector ownership for full-attention KV and GDN state with stable backend allocations, stream-ordered zeroing, and a `VT_DEVICE_KV_CACHE=0` fallback. CPU/CI pass. A clean CUDA 13.0.88/sm_121a build reaches 100%; both 35B+27B tests pass **2/2** in default mode (79.91s/31.04s) and **2/2** in fallback mode (66.23s/16.50s). Default tests assert device-pointer residency; the GPU is process-free and its series lock released. Evidence log SHA-256: `051a83cb…2f66`. These timings are cold/warm sequential correctness diagnostics, not an A/B speed result. The exact pre-W0 trace records 153,394 async copies and 20.809/20.806 GiB H2D/D2H state rows. | Complete sanitizer/lifecycle/memory, corrected trace, and repeated interleaved W0-vs-fallback A/B gates; then implement W1 to collapse the remaining device-to-device row loop. |

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
