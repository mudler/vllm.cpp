# The ROCm hyper-connection grouped norm on gfx1151: 35.68% -> 0.56%, 1.49x decode

Row `MODEL-MM-QWEN4-EXP`, issue `ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F`, spec
[`qwen4-exp-rocm-hc-norm.md`](../../.agents/specs/qwen4-exp-rocm-hc-norm.md).
Measured 2026-09-13 on `strix:gpu0` (gfx1151, Radeon 8060S, **wavefront size 32**,
ROCm 7.2.4 / HIP 7.2.53211) under exclusive `rc` leases.

## What changed

`src/vt/rocm/rocm_qwen4_exp.hip`'s `HcGroupedNormKernel` went from ONE THREAD per
`(token, hc stream)` group accumulating in `double` to ONE BLOCK per group with a
`warpSize`-derived cross-lane tree in f32 -- the shape and the width vLLM's own
`_grouped_gemma_rmsnorm_kernel` computes at the active parity pin `e126687a9a`.

## The four leases

| wave | `rc` job | what |
|---|---|---|
| 1 | `6b5fe700-74e6-45a8-adc5-12c8eff19516` | battery, red-first probe, M0, mutations M2/M3/M4/M6, full cross-device suite |
| 2 | `ee83eded-7827-43f4-95f8-0c514f271c48` | M5b, the decode A/B, both `rocprofv3` captures |
| 3 | `182ea36d-15fb-44e5-8dec-b5f605f6d650` | M1c |
| 4 | `c4823c13-06e0-4c64-9429-7f0fd0088bb7` | deep re-rank of both archived traces, `#3040` bound |

Job scripts are `job-wave{1,2,3,4}.sh` here; `rcjob-wave*.log.gz` are the
leases' own stdout with the base64 archive blocks stripped.

## Decode throughput -- NO PROFILER, four interleaved launches

`vllm-cli --model /tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
--prompt 'The capital of France is' --max-tokens 64 --temperature 0
--max-num-seqs 1 --repeat 4`, alternating FIX, BASE, FIX, BASE on one boot.
**Run 1 of every launch is discarded**: this arm stages lazily and reads
1.27-1.43 tok/s there.

| arm | six usable runs (tok/s) | median | min-max | spread |
|---|---|---:|---|---:|
| BASE `60990ee78` | 5.364 5.442 5.450 / 5.436 5.431 5.433 | **5.4345** | 5.364-5.450 | 1.60% |
| FIX `63925eb7f` | 8.129 7.970 8.108 / 8.124 8.114 8.093 | **8.111** | 7.970-8.129 | 2.00% |

**1.49x.** Per token 184.01 ms -> 123.29 ms, 60.72 ms removed. The spread is a
tenth of the effect and the two blocks of each arm agree, so this is not drift.

## Where the time came from

`rocprofv3 --kernel-trace` from `cwd = /tmp` with `ROCPROF_TMPDIR=/tmp/rpt`,
windowed by `slice-decode-window.py` (both arms selected `(128, 187, 3005)`, 59
steady steps, 177296 rows) and ranked by the committed
`scripts/rocm-rank-kernels.py`, which validated each window at ONE dispatch
count, `[3005]`.

| | BASE | FIX |
|---|---:|---:|
| `HcGroupedNormKernel` share | **35.68%** | **0.56%** |
| its total over 59 steps | 3625.8 ms | 37.9 ms |
| its dispatches per step | 97.0 | 97.0 |
| its mean duration | 633.5 us | **6.6 us** |
| kernel busy per step | 172.22 ms | 113.63 ms |
| step wall median | 185.20 ms | 124.73 ms |

The dispatch count does not move, so this is a per-dispatch cost change (96x) and
not a kernel that stopped being called. Full tables in
`{BASE,FIX}_prof-ranked-top60.txt`.

## THE `#3040` BOUND IS VACUOUS ON THESE TRACES

Supplied from each capture's own stderr, the worst case exceeds the budget:

| | adjusted rows | worst-case share of the kernel budget |
|---|---|---:|
| BASE | 615 of 177296 (0.3469%) | **242.55%** |
| FIX | 626 of 177296 (0.3531%) | **1264.49%** |

It is computed by assuming every adjusted row cost the trace MAXIMUM, and these
traces have maxima of 40.076 ms and 135.432 ms. **So the ranking is
corroboration and the unprofiled tok/s A/B is the claim.** They agree: the
trace's step-wall median says 60.47 ms removed where the A/B says 60.72 ms, by
instruments sharing no arithmetic. No per-row exactness is claimed. For this
kernel the FIX rows are clean anyway -- mean 6.6 us, min 1.1, max 16.4.

## Correctness

`test_qwen4_exp_rocm_reductions -tc='*ROCM W7*'`: 5 cases / **45 assertions** / 0
failed, worst case using 9.5% of its derived budget. Full ROCm cross-device
suite on the FIX: **61 cases / 84841 assertions / 0 failed**, `-tc='*DSA*'`
unmoved at 2 / 273.

All five structural mutations convict; two more waves were needed because two of
this row's own mutations were written wrong (one a semantic no-op, one that did
not compile) and are reported VOID rather than as survivors. The spec's §7.3
carries the table.

## NO CROSS-ENGINE RATIO IS CLAIMED

`qwen4_exp` on ROCm has no token-exact gate
([`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG`](../../.agents/issues/MODEL-MM-QWEN4-EXP/ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG.md)),
and that ABSENT gate is the whole reason. **THE REASON THIS SECTION FIRST GAVE IS
FALSE AND IS REPLACED** (corrected 2026-09-13; no number above is touched). It
read: "llama.cpp aborts on this architecture and the pinned vLLM has no ROCm
build here." Only the STOCK `llama-cpp` pin `b10451` aborts. The scoped
[`llama-cpp-qwen4exp`](../../.agents/oracles/llama-cpp-qwen4exp.md) oracle
(PR #27742 at `035e22731a`, `gateable = yes`) builds with HIP for `gfx1151` and
decoded this same artifact on this same board at 25.877 tok/s
([evidence](qwen4exp-llamacpp-denominator-gfx1151-20260913.md)), so a
DENOMINATOR exists here. And the pinned vLLM at `e126687a9a` ships a complete
`amd/` backend beside its `nvidia/` one and registers
`Qwen4ExpForConditionalGeneration` at `registry.py:580`, so the primary oracle
CAN define token-exact; what is missing is a RUN, for the three blockers that
issue names. A denominator without a gate is still not divided:
`AGENTS.md` §Gates, and the [#2497](https://github.com/mudler/vllm.cpp/issues/2497)
retraction. **Every number above is a within-engine A/B on one box, one boot,
one artifact.**

## Not measured

Wave 2's own tar never printed -- the job wedged on a bare `wait` that included
its heartbeat loop and was killed after its work was archived -- so its M5b and
speed logs survive in `rcjob-wave2.log.gz` and in `/workspace/q4exp-hcnorm/w2-*`
rather than as separate files here. Wave64 is reasoned and not run: gfx1151 is
the only AMD part in this fleet. No ROCm race instrument was run against the new
shared-memory shape.
