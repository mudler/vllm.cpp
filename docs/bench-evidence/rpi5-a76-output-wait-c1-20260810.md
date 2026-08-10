# Raspberry Pi 5 output-wait C1 discriminator, 2026-08-10

This is the current-source causal control for
[`SERVE-CLI-BENCH` issue #293](https://github.com/mudler/vllm.cpp/issues/293)
and the prerequisite discriminator requested by
[`KERNEL-GEMM-CPU-ELEM-A76` issue #284](https://github.com/mudler/vllm.cpp/issues/284).
It compares the existing yielding poll loop with a concurrency-one-only
blocking wait in the same binary. It does not implement the general
multi-request event mechanism and does not change the default.

## Decision

The C1 polling hypothesis is **positive**, pending the required fresh mutation
review and operator gate:

- at four compute threads, `blocking-c1` improves median per-stream decode
  2.193x, output throughput 2.185x and prefill 2.004x; median E2E falls from
  56.993 to 26.081 seconds;
- at three compute threads, the same control is neutral: decode is 1.003x,
  prefill 1.001x and E2E is 0.997x relative to polling;
- all 12 runs emit the accepted token-file SHA-256
  `0ec98eabb23e4148d540fcf79a2fe61678fb90fe462cdf28134af7a42fe6a826`;
  the blocking arm records exactly 64 blocking calls per run and polling
  records zero; and
- T4 PMU medians fall by 46.1% cycles, 76.5% instructions and 54.2% task-clock
  under the blocking control. At T3 the corresponding changes are 5.1%, 5.5%
  and 25.0%, with no material wall-time change.

This establishes that the benchmark consumer's scan-and-yield policy causes
the four-core collapse. It does not make `blocking-c1` a general solution:
blocking an arbitrary collector at concurrency greater than one can delay a
ready peer, so that mode deliberately rejects `--concurrency` above one.

The existing llama.cpp floor remains unchanged. The T4 polling arm is itself
pathological and has 5.9-10.7% spread across latency/throughput axes, above the
3% acceptance limit. The causal effect occurs in every interleaved pair and
the blocking/T3 arms are stable below 0.52%, but these medians are not promoted
to a replacement binding floor. C2 must implement and independently review the
lost-wakeup-safe general event/epoch wait before the A76 full-model gate and
fresh kernel profile resume.

## Same-binary result

All values are medians of three separate, interleaved processes. Higher is
better for tok/s; lower is better for latency and RSS. `blocking / poll` is the
ratio in every final column.

| Threads | Mode | Prefill tok/s | Decode tok/s | Output tok/s | TTFT ms | TPOT ms | E2E ms | Peak RSS KiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | poll | 4.9219 | 1.1797 | 1.1229 | 3,453.92 | 847.69 | 56,992.80 | 2,978,448 |
| 4 | blocking-c1 | 9.8644 | 2.5865 | 2.4539 | 1,723.37 | 386.62 | 26,080.51 | 2,978,704 |
| 4 | blocking / poll | **2.004x** | **2.193x** | **2.185x** | **0.499x** | **0.456x** | **0.458x** | 1.0001x |
| 3 | poll | 8.1268 | 2.5353 | 2.3755 | 2,091.85 | 394.43 | 26,941.99 | 2,978,192 |
| 3 | blocking-c1 | 8.1389 | 2.5424 | 2.3816 | 2,088.74 | 393.33 | 26,872.68 | 2,978,704 |
| 3 | blocking / poll | 1.001x | 1.003x | 1.003x | 0.999x | 0.997x | 0.997x | 1.0002x |

The individual T4 results show that the conclusion does not depend on the
median-selected process:

| Rep | Mode | Prefill tok/s | Decode tok/s | TTFT ms | TPOT ms | E2E ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | poll | 5.2644 | 1.1683 | 3,229.23 | 855.94 | 57,153.65 |
| 1 | blocking-c1 | 9.8609 | 2.5744 | 1,723.97 | 388.44 | 26,195.70 |
| 2 | blocking-c1 | 9.8644 | 2.5865 | 1,723.37 | 386.62 | 26,080.51 |
| 2 | poll | 4.9219 | 1.2518 | 3,453.92 | 798.86 | 53,782.22 |
| 3 | poll | 4.7375 | 1.1797 | 3,588.39 | 847.69 | 56,992.80 |
| 3 | blocking-c1 | 9.9050 | 2.5876 | 1,716.30 | 386.45 | 26,062.78 |

Maximum-minus-minimum spread about the median is 0.45-0.51% for the T4
blocking performance axes, 0.14-0.43% for both T3 arms, and 5.92-10.71% for
the T4 polling axes. Peak RSS varies by at most 0.02% in every arm.

## PMU result

`perf stat` wrapped the entire process at every leg. These are median totals;
the stat run is counting-only, so a lost-sample count is not applicable.

| Threads | Mode | Cycles | Instructions | Task-clock ns | Instructions/cycle from median totals |
|---:|---|---:|---:|---:|---:|
| 4 | poll | 462,788,342,577 | 1,162,875,406,457 | 228,289,315,838 | 2.513 |
| 4 | blocking-c1 | 249,435,618,828 | 273,661,479,884 | 104,649,636,414 | 1.097 |
| 4 | blocking / poll | 0.539x | 0.235x | 0.458x | - |
| 3 | poll | 203,695,408,808 | 265,302,921,340 | 108,157,222,000 | 1.302 |
| 3 | blocking-c1 | 193,367,448,020 | 250,728,593,259 | 81,105,074,218 | 1.297 |
| 3 | blocking / poll | 0.949x | 0.945x | 0.750x | - |

The T3 poll arm consumes about 3.93 CPUs for the same wall time as the
2.96-CPU blocking arm. At T4, polling consumes about 3.97 CPUs for more than
twice as long and retires over four times the instructions. That matched
control is the direct evidence for frontend contention; the earlier W0 barrier
profile alone did not identify the responsible thread.

## Correctness, build and host

- Source head: `bdbfffcbf8b02f4c54fb1a0000c7dab7c10f5c3d`, based on operator
  integration head `9ec8cae23` and fetched `upstream/main` `7d45913a7`.
- AArch64 Release `vllm-bench`: SHA-256
  `c6bdb1a46f433e40bc18dddf43cdc85855c1e6ad4e74ec81b28d6fa5982329f6`,
  GNU build ID `d01ab3ed05056ce805ded8bd1a132b07bb452b65`.
- Model: `Qwen3.5-2B-UD-Q8_K_XL.gguf`, 2,834,940,160 bytes, SHA-256
  `a53988df91157d78acaf3c95e22db179d13f6236061bdb86576494dc99b1bc3b`.
- Workload: one request, 17 actual input tokens, 64 output tokens,
  concurrency one, seed zero, temperature zero, cores 0-3,
  `VT_CPU_Q8_DOT=auto`, and `VLLM_CPP_CPU_THREADS` 4 or 3.
- Host: `rpi5fan`, four Cortex-A76 cores, Linux
  `6.18.34+rpt-rpi-2712`, `ondemand` governor; captured frequencies
  1.9-2.4 GHz, temperatures 49.9-63.1 C, and `throttled=0x0` before and after
  every leg. No process used at least 1% CPU before any leg.
- Native focused gate: 6/6 cases, 47 assertions. Full scoped native CTest:
  5/5 (`test_bench`, `test_async_llm`, `test_output_processor`,
  `test_engine_core_proc`, `test_cpu_threadpool`).
- QEMU/buildx AArch64 gate: quant 23/23 cases and 150,350 assertions;
  benchmark 6/6 cases and 47 assertions; A76 assembly smoke selected
  `a76-asm` and completed with checksum `0xbb60dafd79d08ed6`.
- Dispatch mutation: changing the `blocking-c1` branch to `poll` fails the
  three intended blocking-call assertions; the source was restored to SHA-256
  `5d5bcb3330c1164a10608d5c60b2fbbf4972681e3a6cb1437053eb0404e4efbd`.
- CLI rejection gate exits 2 with `benchmark output wait 'blocking-c1'
  requires --concurrency 1`.

## Raw evidence and recipe

The immutable raw set is on the Pi at
`~/vllm-cpp-assembly/evidence/a76-c1-bdbfffcbf-20260810-rerun1/`.
Its `SHA256SUMS` contains 98 entries and has SHA-256
`a1ae17387fff08ea454fd19ce33c7fd21f8b8ca14aa2ffb4c59b7cacef77be78`.
Every entry was verified on the Pi and again after copying the set locally.
Each leg contains JSON, exact token IDs, stdout/stderr, `perf stat`, fresh
process `getrusage`, and before/after host snapshots. The harness and
standard-library RSS wrapper have SHA-256
`35166849604a534ebbc7d0969bbc7624bcb8931bbbc30db38a5ea92f3ab599f6`
and `ad310102e1d88fa6395c2c41a9594f62d195891bdb1b79e322cd31611db805f5`.

The first harness attempt stopped before inference because the Pi image has no
GNU `/usr/bin/time`; that partial set remains separately at
`~/vllm-cpp-assembly/evidence/a76-c1-bdbfffcbf-20260810/`. The accepted rerun
uses a fresh standard-library Python process per leg and
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`; the binary and workload are
unchanged.

The core command in each interleaved leg was:

```sh
perf stat -x, -e cycles:u,instructions:u,task-clock,context-switches,cpu-migrations -- \
  taskset -c 0-3 env VLLM_CPP_CPU_THREADS=<3|4> VT_CPU_Q8_DOT=auto \
  vllm-bench --model Qwen3.5-2B-UD-Q8_K_XL.gguf \
    --num-prompts 1 --input-len 16 --output-len 64 --concurrency 1 \
    --seed 0 --temperature 0 --output-wait <poll|blocking-c1> \
    --output-token-ids <leg>.tokens.json --output-json <leg>.json
```

For diagnostic continuity only, the C1 T4 blocking medians are 0.353x
llama.cpp b9892 prefill, 0.662x decode, 0.651x output-equivalent throughput and
1.535x E2E latency using the previous shape-matched denominator. Prompt content
still differs between the two harnesses, so this is not a new cross-engine
binding comparison and no llama.cpp floor is moved.
