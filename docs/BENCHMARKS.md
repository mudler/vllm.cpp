# Benchmarks

This is the user-facing benchmark checkpoint for vllm.cpp. It separates
accepted, reproducible results from work that is pending, failed, or void.
Detailed commands, per-repetition values, hashes, and parity rationale remain
in the [append-only parity ledger](../.agents/parity-ledger.md) and the linked
feature specs.

Last updated: **2026-07-11** (`SERVE-GATE-ONLINE` merged-DELTA validator checkpoint).

## Current checkpoint

| Track | Disposition | Evidence now | Next binding gate |
|---|---|---|---|
| `SERVE-STREAM-USAGE` | **PENDING — GATING** | Completion and chat parse `stream_options`, emit final/continuous usage from native token IDs, validate non-stream requests, and expose force-usage mode. CPU, TSan, and ASan+UBSan gates pass. The interrupted `a40a9e3` 27B campaign retained 1,488/1,488 successful requests with exact native 128-token counts, closing the missing-usage symptom; it is not a performance result. | Complete fresh 27B and 35B online campaigns plus the planned same-binary serialization A/B. |
| `SERVE-GATE-ONLINE` | **VOID — no binding ratio** | `a40a9e3` completed two full interleaved 27B ours/vLLM ladders and ours rep3 through c16. At c16-r3, 96/96 requests completed with exact native counts; 12 legally merged two adjacent DELTA outputs and retained 126 rather than 127 ITLs. The old validator rejected this pinned-vLLM behavior, so c32-r3, vLLM rep3, traces, and finalization never ran. Cleanup and lock release passed. [Diagnostic record](../.agents/state.md). | Merge the corrected inter-chunk validator, regenerate SHA-bound evidence, and rerun 27B then 35B. No latency, throughput, memory, or ratio from `a40a9e3` may be reused. |

The stream-usage path changes host-side JSON/SSE serialization, not model
kernels. Its performance disposition is nevertheless `PENDING` because the
final frame participates in whole-request throughput and the feature directly
unblocks the binding online latency campaign. Retained ITLs are inter-choice
timings: pinned vLLM's single-slot collector deliberately merges DELTA outputs
when the producer gets ahead, while native usage remains the exact token-count
oracle. The gate accepts fewer than 127 ITLs for that case but still rejects
extra intervals, partial requests, count drift, errors, or unsaturated load.

## Accepted CUDA engine throughput

These are the accepted offline engine checkpoints against fresh, production
(CUDA graphs + `torch.compile`) vLLM on the same GB10. They do **not** replace
the still-open HTTP online TTFT/TPOT/ITL gate above.

| Model / point | Build and workload | vllm.cpp | vLLM | Ratio | Other accepted evidence |
|---|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B NVFP4, c64 / 200 prompts | Triton-AOT GDN; input 1024, output 128, greedy | 3345.9 tok/s | 3282.0 tok/s | **1.0195×** | 16/16 token-exact; repetition ratio 1.0192–1.0198; peak memory 52.8 vs ~80.6 GB. [Record](../.agents/state.md#L1740). |
| Qwen3.6-27B NVFP4, c16 / 96 prompts | Triton-AOT + default FA-2 prefill; input 1024, output 128, seed 0, greedy | 764.28 tok/s total; 84.89 output | 758.84 tok/s total; 84.32 output | **1.0072× total; 1.0068× output** | 16/16 token-exact; 7 local reps, 6/7 ≥1.0×, worst 0.996 disclosed. [Ledger](../.agents/parity-ledger.md#L284). |
| Qwen3.6-27B NVFP4, c32 / 192 prompts | Same build/workload | 1051.24 tok/s total; 116.77 output | 1043.86 tok/s total; 115.98 output | **1.0071× total; 1.0068× output** | 16/16 token-exact; 5/5 local reps ≥1.0×; peak memory 61.8 vs 76.2 GB. [Ledger](../.agents/parity-ledger.md#L284). |

The 35B ratio requires `-DVLLM_CPP_TRITON=ON`; the default pure-C++ build was
0.99× at that checkpoint. The accepted 27B path also requires the default-on
vendored FA-2 prefill route. Reported TTFT/TPOT improvements in those offline
runs are implementation A/B evidence; binding server-to-server latency ratios
remain intentionally absent until `SERVE-GATE-ONLINE` closes.

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
