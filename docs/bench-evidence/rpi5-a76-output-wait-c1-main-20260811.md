# Raspberry Pi 5 output-wait C1 current-main confirmation, 2026-08-11

This is the post-rebase operator checkpoint for
[`SERVE-CLI-BENCH` issue #293](https://github.com/mudler/vllm.cpp/issues/293)
and the prerequisite discriminator for
[`KERNEL-GEMM-CPU-ELEM-A76` issue #284](https://github.com/mudler/vllm.cpp/issues/284).
It repeats the concurrency-one `poll` / `blocking-c1` control after rebasing
the reviewed implementation onto current `upstream/main`. The
[original C1 evidence](rpi5-a76-output-wait-c1-20260810.md) remains the record
for its older source and binary; this checkpoint does not overwrite it.

## Decision

The current-main result confirms that the C1 polling hypothesis is positive:

- at four compute threads, `blocking-c1` / `poll` is 2.149x median per-stream
  decode, 2.151x output throughput, 1.973x prefill and 0.465x E2E latency;
- at three compute threads, the control remains neutral: 1.003x decode,
  0.999x prefill and 0.998x E2E latency;
- all 12 processes exit zero and emit the accepted token-file SHA-256
  `0ec98eabb23e4148d540fcf79a2fe61678fb90fe462cdf28134af7a42fe6a826`;
  every blocking arm reports exactly 64 blocking calls and every poll arm
  reports zero; and
- T4 blocking uses 45.1% fewer user cycles, 63.0% fewer user instructions and
  53.4% less task-clock than polling. The T3 wall-time-neutral control uses
  5.1%, 5.7% and 25.0% less respectively.

The accepted llama.cpp floor does not move. T4 polling has 13.33-15.96%
spread across the six performance axes, well above the sub-3% binding gate.
Every interleaved T4 pair improves in the expected direction and the T4
blocking/T3 arms are stable, so the causal conclusion survives the rebase; the
unstable denominator still prevents promoting a replacement performance
floor. `blocking-c1` remains a diagnostic restricted to concurrency one,
`poll` remains the default, and C2's lost-wakeup-safe general event wait is
still the next implementation.

## Same-binary result

Values are medians of three separate, order-alternated processes. Higher is
better for throughput; lower is better for latency and RSS. Final rows are
`blocking-c1 / poll`.

| Threads | Mode | Prefill tok/s | Decode tok/s | Output tok/s | TTFT ms | TPOT ms | E2E ms | Peak RSS KiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | poll | 5.0046 | 1.2028 | 1.1404 | 3,396.86 | 831.37 | 56,121.80 | 2,978,704 |
| 4 | blocking-c1 | 9.8763 | 2.5848 | 2.4528 | 1,721.29 | 386.88 | 26,092.51 | 2,978,464 |
| 4 | blocking / poll | **1.973x** | **2.149x** | **2.151x** | **0.507x** | **0.465x** | **0.465x** | 0.9999x |
| 3 | poll | 8.1281 | 2.5294 | 2.3705 | 2,091.51 | 395.34 | 26,998.68 | 2,977,952 |
| 3 | blocking-c1 | 8.1176 | 2.5364 | 2.3762 | 2,094.21 | 394.26 | 26,933.80 | 2,978,192 |
| 3 | blocking / poll | 0.999x | 1.003x | 1.002x | 1.001x | 0.997x | 0.998x | 1.0001x |

The T4 polling maximum-minus-minimum spread about the median is 15.35%
prefill, 14.01% decode, 13.64% output, 15.96% TTFT, 13.91% TPOT and 13.33%
E2E. The T4 blocking arm is 0.023-2.04% across those axes; every T3 arm is at
or below 0.343%. Peak-RSS spread is below 0.026% in every arm.

## PMU result

`perf stat` wrapped each complete process. These are median totals from the
same legs; user cycles and instructions were requested explicitly.

| Threads | Mode | Cycles | Instructions | Task-clock ns | IPC from median totals |
|---:|---|---:|---:|---:|---:|
| 4 | poll | 455,038,839,265 | 708,685,367,178 | 224,842,433,752 | 1.557 |
| 4 | blocking-c1 | 249,764,922,813 | 262,507,520,194 | 104,800,580,477 | 1.051 |
| 4 | blocking / poll | 0.549x | 0.370x | 0.466x | - |
| 3 | poll | 204,253,742,508 | 258,064,105,230 | 108,440,568,102 | 1.264 |
| 3 | blocking-c1 | 193,827,363,627 | 243,472,258,058 | 81,301,312,384 | 1.256 |
| 3 | blocking / poll | 0.949x | 0.943x | 0.750x | - |

The T4 poll arm consumes about 3.96 CPUs for more than twice the blocking
wall time. At T3, polling consumes about 3.94 CPUs while blocking consumes
about 2.96 CPUs for the same wall time. This independently reproduces the
frontend-contention discriminator; it does not establish a general wait
implementation or a cross-engine floor.

## Source, build and gates

- Source: `960647bf5e0467c0121aa9d69b85c0e0b192b10c`, based on fetched
  `upstream/main` `4ba051406`; the C1 implementation had a fresh post-rebase
  review before this operator checkpoint.
- AArch64 Release `vllm-bench`: SHA-256
  `d03b612eaa59590f4c5df25b3bc128e298aa89bdee6eb5bf5633d3ff18e8f058`.
  Exported `vllm-cpu-kernel-bench`: SHA-256
  `80ab743acdee0d2bc749a8c871672f8852e45ce56c271f6ae9a5673171b774a9`.
- Native scoped CTest: 5/5 (`test_engine_core_proc`,
  `test_output_processor`, `test_async_llm`, `test_bench` and
  `test_cpu_threadpool`).
- QEMU/buildx AArch64 gate: quant 23/23 cases and 150,350 assertions;
  `test_bench` 8/8 cases and 63 assertions; A76 smoke selected `a76-asm` and
  returned checksum `0xbb60dafd79d08ed6`. QEMU timings are void.
- Model: `Qwen3.5-2B-UD-Q8_K_XL.gguf`, SHA-256
  `a53988df91157d78acaf3c95e22db179d13f6236061bdb86576494dc99b1bc3b`.
- Workload: one request, 17 actual input tokens, 64 output tokens,
  concurrency one, seed zero, temperature zero, cores 0-3,
  `VT_CPU_Q8_DOT=auto`, with `VLLM_CPP_CPU_THREADS` set to four or three.

## Host and raw evidence

The Pi snapshots identify four Cortex-A76 r4p1 cores, Linux
`6.18.34+rpt-rpi-2712`, the `ondemand` governor, 2.1-2.4 GHz captured
frequencies, 49.9-62.6 C temperatures and `throttled=0x0` throughout. The
highest pre-series process was the SSH session at 0.4% CPU; individual pre-leg
snapshots peaked at 0.2%, and no contender reached 1% CPU.

The immutable raw set remains at
`~/vllm-cpp-assembly/evidence/a76-c1-960647bf5-20260811-operator/` on
`rich@rpi5fan.lan`. Its `SHA256SUMS` has 98 entries and SHA-256
`2d4f26091f02589f63e818639f503539b7281e06eda4eb68c5883a4c6dc6a149`.
All 98 entries were verified again after copying the set locally; the manifest
contains absolute Pi paths, so the local verification remapped only that fixed
directory prefix. Each leg contains JSON, exact token IDs, text output,
stderr, `perf stat`, process `getrusage`, and before/after host snapshots.

The remote harness and standard-library RSS wrapper were independently
rehashed as
`02b75b2a69bc6dc5150071a252ee53862aa8cf37d80b79294ab38a4f85b51445`
and
`ad310102e1d88fa6395c2c41a9594f62d195891bdb1b79e322cd31611db805f5`.
Every raw host snapshot repeats the expected benchmark/model hashes. Every
JSON field, text report, exit status, embedded token stream and token file was
checked across all 12 legs.

The core command remains:

```sh
perf stat -x, -e cycles:u,instructions:u,task-clock,context-switches,cpu-migrations -- \
  taskset -c 0-3 env VLLM_CPP_CPU_THREADS=<3|4> VT_CPU_Q8_DOT=auto \
  vllm-bench --model Qwen3.5-2B-UD-Q8_K_XL.gguf \
    --num-prompts 1 --input-len 16 --output-len 64 --concurrency 1 \
    --seed 0 --temperature 0 --output-wait <poll|blocking-c1> \
    --output-token-ids <leg>.tokens.json --output-json <leg>.json
```

For continuity only, the latest T4 blocking medians remain about 0.353x the
prior llama.cpp b9892 prefill denominator, 0.662x decode and 1.536x E2E
latency. The two harnesses still use different prompt content, so those values
are diagnostic and do not replace the accepted same-file floor.
