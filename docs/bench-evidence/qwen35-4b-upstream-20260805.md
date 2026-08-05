# Qwen3.5-4B revalidation after the upstream update, 2026-08-05

Immutable evidence index for the remeasurement on the `upstream/main` tree at
`59674cf1d`. The measurement worktree commit was `312af21a9`, whose tree is
byte-identical to that upstream commit; the evidence checkpoint is published as
one cherry-picked commit on a fresh branch based directly on `59674cf1d`. The
upstream advance contained four commits, including the device-resident sampled
token fix, TTFT instrumentation work, and the merged GCC 15 repair from PR #28.

## Headline

**The local engine did not move.** Direct-load throughput is 6611.207 tok/s,
0.99997x the prior 2026-08-03 series. The current pinned vLLM run is 6630.481
tok/s, 1.00089x its prior run, so the binding ratio is now **0.9971x** rather
than 0.9980x. That change is denominator noise, not a local regression.

TTFT and host memory still pass. Total/output/request throughput, TPOT/ITL,
and peak VRAM remain below the strict vLLM floor. This local 4B diagnostic does
not establish correctness or performance for the 27B or 35B gate models.

## What was measured

- Ours: measurement commit `312af21a9051af1ff9869a183d8e11fe68905933`,
  rebuilt from a tree byte-identical to `upstream/main` at `59674cf1d`.
- Oracle: `.venv-vllm-pin`, vLLM
  `0.23.1rc1.dev1511+g555967922`, built from the required parity pin.
- Device: RTX 5070 Ti, discrete Blackwell `sm_120a`.
- Build: `build-nix-cuda-transplant-triton`, CUDA enabled, FlashAttention
  enabled, Triton AOT regenerated for `sm_120`.
- Workload: cached `Qwen/Qwen3.5-4B`, 128 requests, 128 output tokens,
  concurrency 32, maximum 2048 batched tokens, greedy sampling.
- Three memory and three performance repetitions for direct ON, vLLM, and
  direct OFF, interleaved under one `flock /tmp/gpu`. All 18 legs observed 0%
  GPU utilization before starting.

## Binding result

| Axis | ours | vLLM at pin | ratio | Disposition |
|---|---:|---:|---:|---|
| Total throughput (tok/s) | 6611.207 | 6630.481 | 0.9971x | FAIL |
| Output throughput (tok/s) | 731.050 | 733.180 | 0.9971x | FAIL |
| Requests/s | 5.710 | 5.728 | 0.9969x | FAIL |
| Mean TTFT (ms) | 730.403 | 946.214 | 0.7719x | PASS |
| Mean TPOT (ms) | 38.143 | 33.924 | 1.1244x | FAIL |
| Mean ITL (ms) | 38.143 | 33.924 | 1.1244x | FAIL |
| Peak PSS (GiB) | 2.531 | 8.093 | 0.3127x | PASS |
| Stable PSS (GiB) | 0.739 | 4.422 | 0.1671x | PASS |
| Peak VRAM (MiB) | 12850 | 12832 | 1.0014x | FAIL |

Direct ON remains useful independently of the reference comparison: versus
direct OFF it improves total throughput by 1.86%, lowers mean TTFT by 12.15%,
and lowers peak/stable PSS by 70.56%/91.40%.

Local direct-ON and direct-OFF output is stable: 128/128 requests match within
each arm, between the two arms, and against the same historical arm in every
repetition. vLLM repetitions 1 and 2 match each other and the historical
series 128/128; repetition 3 matches 95/128. Ours versus vLLM matches
89/89/98 requests. This is the existing near-tie sampling branch behavior, not
local output drift.

## Execution traces

Fresh node-level Nsight Systems traces were captured on the identical full
workload with `--cuda-graph-trace=node --trace=cuda,nvtx`:

- ours: `/tmp/qwen35-upstream-312af21a9-ours.nsys-rep`
- vLLM: `/tmp/qwen35-upstream-312af21a9-vllm.nsys-rep`

Both exports contain graph child-kernel rows, so attribution is not the
incomplete whole-graph view. The structural hot-path names remain visible:
ours is led by CUTLASS bf16 256x128 (40.0%), CUTLASS bf16 128x256 (10.4%),
WMMA bf16 (9.5%), and packed GDN decode (6.7%); vLLM is led by CUTLASS bf16
256x128 (25.6%), a warmup/prefill FlashAttention kernel (15.9%), WMMA bf16
(12.7%), and its remaining CUTLASS/GDN families. The percentages include
warmup, JIT, and graph capture and therefore are not steady-state lever
weights. Profile-instrumented throughput was 6550.65 tok/s for ours and
6592.83 tok/s for vLLM; the unprofiled 18-leg series above is binding.

## Reproduction

```bash
nix develop .#cuda --command cmake --build build-nix-cuda-transplant-triton -j 16

nix develop .#cuda --command bash -c \
  'REQUIRE_TRITON_AOT=1 \
   CPP_BENCH="$PWD/build-nix-cuda-transplant-triton/examples/vllm-bench" \
   CMAKE_CACHE="$PWD/build-nix-cuda-transplant-triton/CMakeCache.txt" \
   VLLM_PYTHON="$PWD/.venv-vllm-pin/bin/python" \
   VLLM_CUDA_HOME="$PWD/.venv-vllm-pin/lib/python3.12/site-packages/nvidia/cu13" \
   flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh \
   /tmp/qwen35-upstream-312af21a9'

python3 tools/bench/summarize_qwen35_4b_compare.py \
  --root /tmp/qwen35-upstream-312af21a9 \
  --historical-root /tmp/qwen35-gcc15fix-4e43aa5ee \
  --output /tmp/qwen35-upstream-312af21a9/aggregate.json
```

Raw result root: `/tmp/qwen35-upstream-312af21a9`. Machine-readable summary:
`/tmp/qwen35-upstream-312af21a9/aggregate.json`.
