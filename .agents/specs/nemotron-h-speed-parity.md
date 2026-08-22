# Nemotron-3.5-Lightning-30B speed parity against the pinned vLLM oracle

Row: `MODEL-NEMOTRON-H-ABI-A2P`.
Issue: [#1250](https://github.com/mudler/vllm.cpp/issues/1250).
Depends on: [#1157](https://github.com/mudler/vllm.cpp/issues/1157) / PR #1221,
[#810](https://github.com/mudler/vllm.cpp/issues/810).

## Scope

Produce the FIRST speed numbers for `NemotronHForCausalLM` on GB10, against the
pinned vLLM oracle, on the workload the A3 token gate already covers. There are
no Nemotron numbers anywhere today and none are claimed:
`docs/BENCHMARKS.md` carries the row as "**No speed number, by the unit's own
rule**".

In scope:

- This spec.
- The `--num-blocks`, `--kv-cache-memory` and `--repeat` flags on
  `examples/nemotron_h_gen`, which the measurement needs and which the driver
  did not have.
- The measurement itself, its recipe, and its evidence, recorded in
  [`benchmark-record.md`](../benchmark-record.md) and `docs/BENCHMARKS.md`.
- One appended row in [`issue-index.md`](../issue-index.md) for #1250.

Out of scope:

- Any repair of the two arms that make the number what it is. A2-Q2b owns the
  device `lm_head`, and A2-Q1 (PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289)) owns the 46 FP8 W8A8 mamba
  projections. [#940](https://github.com/mudler/vllm.cpp/issues/940) is CLOSED (2026-08-16) and is not the live
  pointer. Measuring a gap is not closing it.
- Any lifecycle change. The row does not move state here; it gains a number.
- Any advance of the parity pin.

## What must be true before a number is taken

**Correctness first, and on the DEVICE.** The declared token-exact gate is
`examples/nemotron_h_gen` against
`tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json`, captured from
the pinned oracle at `temperature 0.0, max_tokens 32` over three prompts. The
device leg reads `TOKEN MATCH: 96/96 over 3 prompt(s) (full rows=3, short
rows=0, mode=decode)` `STRICT PASS` under the #1157 fix, and 4/24 on the same
binary with the fix reverted. **The speed legs and the token comparison are the
SAME run**: `--repeat` reruns the battery over one engine load and every leg is
compared against the golden, so no timing leg is an ungated one. A binary whose
tokens were not checked produces no accepted number here.

**The measured tree carries the #1157 fix.** PR #1221 landed as `0ea5d249f`,
verified by CONTENT on `main` rather than by the API, so the measured base is a
SHA at or after it and every result records that SHA. Without the fix every
decode step on the CUDA path embeds the same placeholder id.

**The build is not degraded.** `fp4-mma`, `cutlass-nvfp4`, `cutlass-fp8`,
`marlin-nvfp4` and `fa2` must each read `ENABLED for [121a]` in the configure
log. A line reading `DISABLED` or `[121]` VOIDS the result rather than failing
it, because the binary then measured a different stack. The harness counts the
lines that are not `ENABLED for [121a]` and exits rather than continuing.

## The denominator

vLLM at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`, in its PRODUCTION
configuration. `enforce_eager=False`, so CUDA graphs are on.
**`--enforce-eager` is never the denominator.** Both sides run the identical
checkpoint directory, the identical pre-tokenized prompts, the identical token
count (32, greedy, `ignore_eos`), batch 1, sequential, `max_model_len 512`,
`max_num_seqs 8`.

## KV sizing, which is not comparable by default

`--gpu-memory-utilization 0.92` is accepted on our side and does NOT size the
pool: the profile run that turns a fraction into a block count is unimplemented
([#83](https://github.com/mudler/vllm.cpp/issues/83)) and the pool falls back to
**256 blocks**. Both legs of the correctness run logged that warning. A
comparison whose two sides hold different KV budgets is not like-for-like on the
memory axis, so:

- Our side passes `--num-blocks 256` EXPLICITLY. It is the same pool the token
  gate ran with, so the correctness result carries over unchanged, and it is now
  stated rather than inherited from a fallback.
- `block_size` defaults to 32 here (`include/vllm.h`), so the pool is
  256 x 32 = **8192 KV tokens**.
- The oracle is given `block_size=32, num_gpu_blocks_override=256`, the same
  8192 tokens, and the resolved cache config is READ BACK from the running
  engine and printed. If the override is refused the driver falls back to the
  production default pool and prints `ORACLE_KV_NOT_MATCHED`, and the memory
  axis is then reported with that asymmetry named rather than quietly.

At batch 1 with sequences of at most 45 tokens both pools vastly exceed demand,
so this choice binds the MEMORY axis and not the throughput axis. It is recorded
because a reader cannot tell which from a ratio.

## What is already known about the answer, and must not be papered over

The gate run measured engine load 264.4 s and 342.61 / 328.19 / 327.51 s for 32
tokens each, about **10.3 s per output token** at batch 1. Two named arms cause
it, and neither has landed:

- **The NVFP4 `lm_head` still executes HOST-side.** `src/vllm/model_executor/models/nemotron_h.cpp::NemotronHHostLmHead`
  refuses it on a non-CPU queue, so the last step of every forward is a host
  projection and the forward returns `HostLogits`. Owned by A2-Q2b.
- **The 46 FP8 W8A8 mamba `in_proj`/`out_proj` projections still execute
  HOST-side.** They are 36.6% of decode bytes and 27.6% of GEMM FLOPs. Owned by
  A2-Q1, PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289).

So the expected result is a poor ratio with a known cause. **It is recorded as
the current state with those two arms named, and never as a ceiling.** An
apparent same-architecture limit is an unresolved implementation difference.

## Risks

- **The oracle has never RUN a model inside a lease.**
  [#1185](https://github.com/mudler/vllm.cpp/issues/1185) established build,
  install, import and `torch.cuda.is_available()` only, and
  [`mtp-k-gt-1.md`](mtp-k-gt-1.md) records the previous attempt consuming the
  host in the step AFTER `torch.compile` and REBOOTING the box at
  `gpu_memory_utilization` 0.75 and again at 0.30. The oracle leg therefore runs
  LAST, after our numbers are on `/workspace`, under a watchdog that kills the
  run below a `MemAvailable` floor rather than merely sampling it. If it cannot
  run, the numerator-only figures are recorded as UNGATED and never as a ratio.
- **`gpu_memory_utilization` does not bound HOST RAM on GB10**, and
  `nvidia-smi` sees only the device side. Both instruments are blind to the
  unified pool, so the host sampler records `MemTotal - min(MemAvailable)`
  across the window and that is the peak this spec means.
- **The SM clock differs between boots and does not announce it.** Each leg
  records a window with `tools/bench/gpu_clock_state.py`, boot id included. A
  figure without one is unattributed and is said to be.
- **Contention voids rather than degrades.** The box state is sampled and
  recorded before the first leg. `rc run` holds the whole device, so the lease
  is the exclusion; the record exists so a reader can check it.

## Gates

```sh
scripts/agent-preflight.sh --fail-on-skip
```

The measurement itself is not reproducible in CI, which has no fleet device.
The `rc` job and its persisted logs are the evidence.

## Stop conditions

- Stop if a required feature line is not `ENABLED for [121a]`. The result is
  VOID, not slow.
- Stop if the token comparison on any timing leg is not full-width and exact.
  A speed number on unverified tokens is not a result.
- Stop if the oracle cannot be run. Record numerator-only figures as ungated and
  say precisely why. Never substitute a different vLLM for the pin.
- Return `NEEDS_DECISION` rather than benchmarking a configuration that avoids
  the host-side arms, because that would measure a model nobody can run.

## What the correctness evidence covers, and what it does not

The 96/96 belongs to ONE configuration: the three golden prompts, 32 tokens
each, greedy with `ignore_eos`, batch 1 sequential, `max_model_len 512`, the
256-block pool. The timing legs run that configuration and no other. A
deviation in KV sizing, batching or sampling does not inherit the correctness
evidence, and any such leg is reported as ungated rather than allowed to borrow
coverage the gate does not give it.

## What the first lease measured, and the two instruments it corrected

Job `81a0cfb1-eaa3-46fc-b64f-d9fa0cab5ecf` on `dgx:gpu0`, 2026-08-18, produced no
speed number and two corrections that the number depends on. Both are
[#1253](https://github.com/mudler/vllm.cpp/issues/1253).

**`nvcc --version` is not the toolkit postcondition.** The worker ships a PARTIAL
CUDA 13.0: the compiler answers and the cuBLAS development component is absent.
The harness installed nothing because `nvcc` answered, CMake then printed all
five feature lines `ENABLED for [121a]` and failed at generate with
`Target "vllm" links to CUDA::cublasLt but the target was not found`. The
postcondition is `libcublasLt.so`, `cublasLt.h` and `Python.h`. This is #1185's
"an environment repair must be unconditional and assert its postcondition"
applied to a postcondition chosen wrong, which is a different defect from
skipping the repair.

**The oracle's recorded hazard is not what stopped it.** The model run did NOT
reboot the box: peak host use was 28,534 MB of 122,502 MB and the watchdog never
fired. It died on `Python.h: No such file or directory` inside `torch._inductor`,
because Triton compiles `cuda_utils.c` at runtime and the worker has no
`python3-dev`. That surfaces as `Engine core initialization failed ... Failed
core proc(s): {}`, an empty proc set naming nothing, which reads as an oracle
limitation and is a missing `-dev` package.

**Contention has one usable instrument here.** `nvidia-smi
--query-gpu=memory.used` reads `[N/A]` on GB10, so the compute-apps list is the
only device-side one. At this lease's start it reported PID 10495 holding
36,396 MiB from the PREVIOUS lease, later reaped by the controller. A benchmark
that samples only `memory.used` on this box is blind to exactly the state that
would void it, so the leg sampler records the compute-apps list instead.

## Owed

- A clock window that satisfies `gpu_clock_state compare`. Both arms failed its
  majority-busy floor and vLLM failed its spread ceiling, so the ratios are
  recorded WITH that refusal and are not clock-attributed.
- The oracle at its DEFAULT `gpu_memory_utilization`, which the box could not
  hold; and with it, whether the oracle reproduces its own golden 96/96 when its
  resolved `block_size` is the one the golden was captured at.
- The re-measurement after A2-Q1 (PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289), blocked by
  [#1388](https://github.com/mudler/vllm.cpp/issues/1388)) and A2-Q2b land, which is the next traceable hypothesis rather than a ceiling.

## Now

**Result: MEASURED on every declared axis, with two refusals recorded beside the
ratios.** Both legs ran on `dgx:gpu0` through `rc run`, same box, same boot id,
same checkpoint, same prompts, same 32 greedy tokens with `ignore_eos`, batch 1
sequential, `--repeat 2` over one load each. Tree
`5325b7b970b67f97a77834e907fc34fb2990b71e`. **`enforce_eager=False`: CUDA graphs
were on and were never disabled.**

| axis | ours | pinned vLLM | ratio |
|---|---|---|---|
| per output token, warm (n=5) | 10.3194 s | 0.014369 s | **718.2x slower** |
| output throughput, batch 1 | 0.09691 tok/s | 69.595 tok/s | **0.001392x** |
| engine load | 280.9 s | 596.3 s | **0.4711x, we are 2.12x FASTER** |
| peak host memory | 44,616 MB | 70,974 MB | 0.629x raw, not like-for-like |
| KV pool | 8192 tokens | 644,096 tokens | 78.6x, could NOT be matched |

**Correctness.** Ours `TOKEN MATCH: 96/96 ... mode=decode`, `STRICT PASS`, on
BOTH timing legs, 192/192 tokens. The oracle read 180/192 against its OWN
committed golden, deterministically, because its resolved `block_size` moved to
512 under the memory configuration the box forced.

**Refusal 1, the clock gate.** `gpu_clock_state compare` exits 1: ours 6.31%
busy and vLLM 31.05% busy against a 50% floor, vLLM spread 5.14% against a 5.0%
ceiling, ours throttled `SwPowerCap`, persistence `Disabled` on both. Same boot,
both medians 2411 MHz, `median_offset_pct` 0.0. The tool's own basis is 0.7548
points of kernel time per point of clock, so it cannot reach a 718x gap.

**Refusal 2, the KV pool.** vLLM's hybrid allocator resolves its own block size
(512 here, 4192 at other settings) against our 32, so the pools cannot be
matched and the memory axis is not like-for-like.

**The bottleneck is named by an instrument.** `nvidia-smi` reported GPU
utilization 0% in 2,019 of 2,155 samples of our window. It is the 23 of 52
layers that bounce to the host per token plus the host NVFP4 `lm_head`. Next
traceable hypothesis: A2-Q1 (PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289)), then A2-Q2b, each of which
must RAISE that busy fraction to be believed. **A2-Q1's arm has since been
measured and it does both, and it is still UNGATED** — see the confirmation
section in [`benchmark-record.md`](../benchmark-record.md).

**The lever is confirmed and still ungated.** A three-leg discriminator
(2026-08-19) moved the 23 Mamba2 layers on-device: warm 10.1502 -> 1.3947
s/token, **7.28x**, with the busy fraction rising 7.86% -> 10.2% as this spec
predicted. Both device legs read **95/96 `DIVERGENCE`** while the host arm reads
96/96 on the same binary, so no parity number is accepted from them: correctness
precedes performance. PR [#1289](https://github.com/mudler/vllm.cpp/pull/1289) is DRAFT pending [#1388](https://github.com/mudler/vllm.cpp/issues/1388).
At ~10.2% busy the decode is still ~90% GPU-idle, so A2-Q1 banks 7.28x and does
not close the gap; A2-Q2b's `lm_head` is next and owes the same test.

Full recipe, hashes, contention and clock state:
[`benchmark-record.md`](../benchmark-record.md).
