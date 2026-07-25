# Qwen3.5-4B main repair evidence, 2026-07-25

This is the immutable evidence index for the local discrete-Blackwell
`LOAD-SAFETENSORS-DIRECT-DENSE` checkpoint after transplanting the relevant
H32 Triton-AOT work onto `main`, enabling the existing decode graph for plain
BF16, and extending the existing FA2 split-KV decode adapter to the model's
exact 16-query-head / 4-KV-head topology.

The checkpoint is **GATING / speed-pending**. Correctness and the direct-loader
memory goal pass. The stable current-vLLM denominator still wins total/output
throughput and TPOT/ITL, so this is not a parity claim.

## Workload and environment

- GPU: NVIDIA GeForce RTX 5070 Ti, sm_120, 16 GiB.
- Model: cached `Qwen/Qwen3.5-4B` snapshot
  `851bf...` under `.hf-cache`.
- Dataset: `/tmp/qwen35-4b-sharegpt-1024.json`, SHA-256
  `9ea13603767c62c267e3f381fbccf42d0c9ca0c393655c37533eadca7aefca0c`.
- Workload: 128 requests, 131,784 actual input tokens, 128 output tokens per
  request, concurrency 32, greedy, prefix caching disabled.
- Project build: `RelWithDebInfo`, CUDA arch `120a`, CUTLASS, FlashAttention-2,
  Triton AOT and AOT regeneration enabled.
- Oracle: local vLLM 0.25.0 from `.venv-vllm`, identical request corpus and
  output length.
- All A/B and full comparison series held one `flock /tmp/gpu`.

## Binding result

The final 18-leg project/direct-OFF/vLLM comparison is
`/tmp/qwen35-main-final-fa2-20260725`. Its aggregate SHA-256 is
`dfba09cb5a01d873a36492b7a05b382e234c10759abd5831a497f96eaf5475f9`.
The same-lock vLLM middle repetition was a 2.1% slow outlier, so its raw
three-run mean is retained as evidence but is not the binding denominator.
The immediate clean-shell vLLM confirmation is
`/tmp/qwen35-vllm-confirm2-20260725`; all three runs are stable and use the
same model, corpus, sampling and concurrency.

| Axis | Direct ON, mean | Direct OFF, mean | vLLM 0.25 stable mean | ON vs vLLM |
|---|---:|---:|---:|---:|
| Total throughput (tok/s) | 5769.993 | 5660.703 | 5849.802 | 0.986357x |
| Output throughput (tok/s) | 638.030 | 625.943 | 646.855 | 0.986357x |
| Requests/s | 4.9867 | 4.8900 | 5.0536 | 0.986765x |
| Mean TTFT (ms) | 834.910 | 943.427 | 1047.130 | PASS, 20.27% lower |
| Mean TPOT/ITL (ms) | 43.720 | 43.837 | 38.550 | FAIL, 13.41% higher |
| Peak PSS (GiB) | 2.406 | 8.592 | 7.662 | PASS |
| Stable PSS (GiB) | 0.759 | 8.589 | 4.029 | PASS |
| Peak VRAM (MiB) | 12850.7 | 12843.3 | 12942.7 | PASS vs vLLM; ON is +7.3 MiB vs OFF |

Direct ON and OFF are request/output-token identical for 128/128 requests in
all three paired repetitions. The project arm is also deterministic across
its three repetitions. vLLM is not byte-identical across all three runs on
this near-tie workload, so project-vLLM correctness is grounded in the real
model gate and the established near-tie contract rather than forced benchmark
token identity.

The stable vLLM total-throughput repetitions are 5840.304, 5862.944 and
5846.159 tok/s. The discarded out-of-shell confirmation root
`/tmp/qwen35-vllm-confirm-20260725` is **VOID** because the CUDA/Triton runtime
was unavailable; it contributes no number.

## Repair attribution

### H32 packed recurrence

Same-binary root `/tmp/qwen35-h32-main-ab-locked-20260725`:

- AOT ON: 5645.537 total tok/s, 624.267 output tok/s, 44.833 ms TPOT.
- AOT rollback: 5397.747 total tok/s, 596.870 output tok/s, 47.203 ms TPOT.
- Gain: +4.5906% total/output throughput.

The focused default/rollback flag suite passes 10/10. The full GDN suite passes
66/66 cases and 4242/4242 assertions. AOT and rollback have the same established
119/128 request identity pattern as the earlier H32 repair because their
floating-point reduction order differs; operator coverage supplies the numeric
equivalence proof.

### Plain-BF16 decode graph

Same-binary root `/tmp/qwen35-4b-main-graph-ab-20260725`:

- Graph: 5668.707 total tok/s, 626.827 output tok/s, 44.593 ms TPOT.
- Eager rollback: 5646.837 total tok/s, 624.410 output tok/s, 44.803 ms TPOT.
- Gain: +0.3873% total throughput.
- Identity: 128/128 requests and 16,384/16,384 generated tokens for every pair.

The real-checkpoint test now exercises capture and replay with graph enabled,
direct loading enabled, direct loading disabled and eager rollback. It passes
3/3 cases and 1672/1672 assertions.

### Ratio-4 FA2 decode

Same-binary root `/tmp/qwen35-4b-main-fa2-r4-ab-20260725`:

- FA2: 5757.830 total tok/s, 636.683 output tok/s, 43.803 ms TPOT.
- Generic fallback: 5667.133 total tok/s, 626.657 output tok/s, 44.623 ms TPOT.
- Gain: +1.6004% total/output throughput.

The composed-reference attention suite passes 25/25 cases and 454,474/454,474
assertions, including the ratio-4 ladder, finite-window case and unsupported
geometry fallback.

## Execution traces

The final local node-mode trace is
`/tmp/qwen35-main-final-fa2-profile-20260725`, SHA-256
`c619a4b698196e44ceb70bd2cc5fcf0ebd2711950b03386d720ee97bcadf6fb4`.
It contains 453 graph launches, 5,148 graph-node API events, 200,972 graph-child
kernel events and 1,680 distinct graph-node IDs. It is attribution-complete,
not a whole-graph-only capture.

The ratio-4 FA2 main kernel executes 3,568 times at 0.643 s total and
180.28 us/call. The matched vLLM node trace executes its FA2 kernel at
178.40 us/call. Local graph-child GPU time is 9.000 s, below the matched vLLM
trace's 9.199 s, while binding TPOT remains 43.72 vs 38.55 ms.

CUDA API attribution identifies the exact residual mechanism on this discrete
GPU. The project trace has 1,090 `cudaStreamSynchronize` calls. Of those, 497
follow the sampled-ID 256-byte D2H and consume **20.975 s total,
42.20 ms/call**; the variable-size tail adds another 0.24 s. These are
`GPUModelRunner::sample_tokens_async`'s discrete-GPU host bookkeeping path:
it synchronizes the main stream immediately so it can update the host
`last_sampled_tokens` before the next batch. The other 531 synchronizations
follow the 16-byte embedding bounds flag and cost only **12.6 ms total,
23.7 us/call**, so embedding validation is not the gap.

The matched vLLM trace instead waits on sampled output through CUDA events
(`cudaEventSynchronize`, 1,342 calls) after the next batch has been launched,
preserving depth-2 overlap. The correct repair is a discrete-CUDA
device-resident sampled-token mapping plus device combine that survives
InputBatch removal/condensation. Simply deleting the synchronize would race or
feed the wrong request after a row move, so no unsafe shortcut is landed here.

## Reproduction

```sh
nix develop .#cuda --command cmake --build build-nix-cuda-transplant-triton -j4

flock /tmp/gpu env HF_HOME="$PWD/.hf-cache" \
  build-nix-cuda-transplant-triton/tests/test_qwen35_plain_weights --no-skip

REQUIRE_TRITON_AOT=1 \
CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh \
  /tmp/qwen35-main-final-fa2-<commit>
```

The next performance change must port the sampled-token device mapping for
discrete CUDA, prove row-condensation and token identity, then rerun the
complete matched series. No 4B result implies support or speed for the 27B/35B
gate checkpoints.
