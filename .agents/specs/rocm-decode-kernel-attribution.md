# Per-kernel decode-time attribution on ROCm

Row: `BACKEND-ROCM`
Issue: `ISSUE-LOCAL-01M2DCDVCQNGQB4F3AHXQRYK8H` (row-owned)

## 1. The gap

On CUDA we can rank a decode step by kernel. A `dgx:gpu0` session ran `nsys`
over a decode and found `HcGroupedNormKernel` at **40.7% of all GPU kernel
time** -- 626.94 ms over 1,439 instances -- and that single number became a
scoped optimization row.

On ROCm nothing ranks anything:

- `VT_OP_PROVIDER_STATS` (`src/vt/rocm/rocm_backend.hip:460`) counts which
  provider was **selected**. It does not time anything.
- The clock instrument samples `sclk`/`busy_percent` for a whole process. On the
  qwen4_exp leg ~35 s of a ~46 s run is model load, and its pooled window read
  39.1% busy, which cannot resolve a per-kernel share.
- `rocprofv3` is recorded as unusable on this board under
  [#3040](https://github.com/mudler/vllm.cpp/issues/3040).

The cost of the gap is concrete. `cc0e827dd` closed the qwen4_exp PLE placement
question and had to end with "where our decode step time goes is unestablished,
and I did not guess". Qwen3.8-Flash-Next decodes at **5.0-5.3 tok/s** on
`strix:gpu0` and we cannot say what that time is spent on.

## 2. What #3040 actually measured

#3040 is real and it is narrower than its own summary sentence.

The evidence README at
`docs/bench-evidence/strix-kernel-trace-3015-20260907/README.md` says:

> Do not derive kernel durations, time shares, or host idle bounds from these
> timestamps.

Its basis is that rocprofiler-sdk 1.1.0's
[`profiling_time.hpp:82`](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/tracing/profiling_time.hpp#L82)
**swaps inverted timestamps and clamps timestamps outside the CPU bounds** by
default rather than failing, and that the kernel-only capture emitted **62**
such warnings.

Read against the capture it describes, that is a row-level advisory, not a
global invalidation. Measured directly on the committed
`diag-kernel-control-asaj36ta--trace--e5367aefafa8--1_kernel_trace.csv.gz`:

| Property | Value |
|---|---:|
| `KERNEL_DISPATCH` rows | 85,737 |
| rows with `End < Start` | **0** |
| rows with `End == Start` | **0** |
| min / median / max duration | 7 ns / 52.301 us / 5.212 ms |
| sum of kernel durations | 11.749 s |
| first-start to last-end span | 12.551 s |
| implied kernel occupancy | **93.6%** |
| swap warnings reported | 62 (**0.072%** of rows) |

A trace whose durations were untrustworthy would not sum to 93.6% of its own
wall span with zero inversions. The SDK repaired the 62 rows it warned about and
delivered a monotonic table. What #3040 correctly forbids is treating any
individual row as exact; what it does not establish is that a **ranking over
tens of thousands of rows** is unusable.

**The residual risk is bounded and small.** The delivered CSV does not mark
which rows were adjusted, so the worst case must be assumed rather than
inspected: if all 62 adjusted rows were maximally wrong and each actually cost
the largest duration in the trace (5.212 ms), the total error is 0.323 s against
an 11.749 s budget -- **2.75%**. No conclusion in this spec depends on a margin
narrower than that, and every table states the bound beside it.

This spec therefore does not "overrule" #3040. It narrows it, with the arithmetic
that narrows it, and leaves #3040 open for the one thing it still owns: a
per-row exactness claim.

## 3. The instrument

**`rocprofv3 --kernel-trace`, plus offline windowing by the per-token sampler
dispatch.** Nothing is added to the engine.

### 3.1 Why the alternatives were rejected

**HIP events around dispatches** (the `VT_OP_PROVIDER_STATS` hook at
`rocm_backend.hip:460` shows where such a probe would live). Rejected. An event
pair per op serialises the queue it measures, and this repository already has the
finding that *a probe that synchronizes cannot see a race*. It would also change
the binary, so the profiled arm and the control arm would no longer be the same
bytes, and the distortion could not be separated from the change.

**ROCTX / `hipUserObj` ranges.** Rejected for this wave for the same reason at
smaller magnitude, and because it needs source changes to place the ranges. It
stays the fallback if §3.2's window marker is ever unavailable. `rocprofiler-sdk`
does ship `rocprofiler-sdk-roctx`, so the option is live.

**A working profiler is strictly better than anything hand-rolled**, so the
profiler path is tried first and only abandoned on evidence.

### 3.2 Decode-only windows without touching the engine

Greedy decode emits exactly one sampler dispatch per generated token. On the
committed gfx1151 capture the sampler is `ArgmaxK`, and slicing the trace between
consecutive `ArgmaxK` dispatches yields:

- 64 marks for 64 generated tokens;
- **63 of 63** inter-mark segments containing **exactly 1330** dispatches;
- step wall times of 182.37 to 184.88 ms, a 1.4% spread;
- 1,945 rows before the first mark, which are model load and prefill and are
  excluded by construction.

The constant dispatch count is what makes this a **self-validating** window: a
mis-segmented trace produces ragged segments, and the tool prints the set of
segment sizes so a reader sees a single value or sees the failure. The first
decode step is discarded as well, because our arm stages ~72 GiB lazily on the
first generation.

This addresses the failure mode the prior clock instrument had. Its window
contained ~35 s of model load, so a sub-10% effect could not be resolved. These
windows contain decode and nothing else.

### 3.3 Distortion, measured rather than assumed

`rocprofv3` intercepts dispatches and therefore costs something. Because no
in-process probe is added, the profiled binary is byte-identical to the control
binary, and the distortion is measurable as one number: tok/s with the profiler
attached against tok/s without it, same binary, same artifact, same prompt,
same token count, alternating legs. The tool reports the ratio and every table
carries it.

The instrument distorts in two further ways that it must state and not hide:

1. It measures **kernel execution**, not host time. A step whose wall time
   exceeds its kernel-busy sum is host-bound in the difference, and the tool
   prints kernel-busy as a percentage of step wall so that gap is visible rather
   than silently attributed to kernels.
2. It cannot see **overlap**. Durations sum per kernel name; on a single queue
   that sum is the step, and the printed occupancy percentage is what tells a
   reader whether that assumption held.

## 4. Scope

In scope: `scripts/rocm-rank-kernels.py`, its tests, and one evidence file
carrying a ranked table for a `qwen4_exp` decode step on gfx1151.

Out of scope: any optimization, any default change, any change under `src/` or
`include/`. This wave measures. A dominant kernel, if one appears, is scoped in
§6 and implemented by a later row.

**No cross-engine ratio appears anywhere.** The `qwen4_exp` ROCm arm has no
declared token-exact gate (`ISSUE-LOCAL-01M2D6MV5RNSSM2GZVZKCZA4EG` -- absent,
not failing), `AGENTS.md` §Gates requires that gate before a performance result
is accepted, and #2497 already carries a retraction for dividing early. Every
figure is this engine against itself.

## 5. Gates

- `scripts/rocm-rank-kernels.py` refuses a trace whose sampler marks are fewer
  than the requested discard plus three, and refuses rather than silently
  ranking when segments are ragged.
- Tests run against the committed gfx1151 trace, so the gate needs no GPU.
- Each mutation deletes or inverts one guarantee, requires the focused suite to
  fail, and restores the tree byte-for-byte.
- The full ROCm cross-device suite stays green at 61 cases / 84841 assertions,
  `-tc=*DSA*` at 2/273. Case AND assertion counts are printed beside every
  `-tc=` line, because a selector matching zero cases prints `Status: SUCCESS!`
  and exits 0, and this row has shipped two such greens.

## 5a. What the measurement found, 13 September 2026

The instrument was built, it works, and **the qwen4_exp table this row exists to
produce now exists**. Getting there took two sessions, and the two halves are
worth keeping apart.

### 5a.1 First session: the profiler wrote nothing

**§3.3's distortion question is ANSWERED and the answer is "almost none".**
Profiled decode 4.804 tok/s against unprofiled 4.823 tok/s, same binary, same
artifact, alternating legs: **-0.4%**. Attaching the profiler does not change
what it measures.

Nothing was written, though. After the traced process exited, rocprofiler-sdk
failed to `mmap` with `EINVAL` (`ring_buffer.cpp:106`), logged it at FATAL,
aborted, and deadlocked in its own signal handler, surviving SIGTERM for 29
minutes. Four leases, both artifacts, both output formats. That session
correctly identified the ENVIRONMENT as the difference and named the
2026-09-07 `podman` image as the thing to reproduce.

### 5a.2 Second session: it is a temp DIRECTORY, and no image is required

**`rocprofv3` spills its records to `$CWD/.rocprofv3/`, and an `rc` job starts
in `/`.** `output_config.hpp:81` defaults `tmp_directory` to `"%cwd%"`,
`--output-directory` moves the result files only, and the CLI never sets
`ROCPROF_TMPDIR`. The spill under `/` does not read back, `ring_buffer::load`
reads a zero size into an unchecked variable, and `mmap` of length **zero** is
`EINVAL` by definition -- which `strace` shows and the FATAL message omits.

Setting `ROCPROF_TMPDIR` under `/tmp`, or running from a `/tmp` working
directory, writes a complete trace on the bare worker. A 50-dispatch HIP program
reproduces both the failure and the fix in 0.04 s, which is how four hypotheses
were killed in one lease. Three of the first session's suspects are FALSIFIED:
the locked-memory limit, seccomp and capabilities, and a missing runtime
package. The `podman` image is NOT required.

`ISSUE-LOCAL-01M2DQC3M3VHKDEZYPA8BSRCAV` is CLOSED on that evidence, and
`.agents/environment.md` carries the recipe so the next session does not spend a
day on it.

### 5a.3 The table

Qwen3.8-Flash-Next UD-IQ1_S on gfx1151, 660,273 dispatch rows, ranked over the
59 steady decode steps of the third repeat:

| | |
|---|---:|
| step wall, median | 197.50 ms |
| kernel busy per step | 173.24 ms (**87.7%**; the rest is HOST time) |
| dispatches per step | **3437**, one value across all 59 steps |
| `HcGroupedNormKernel` | **35.10%** |
| `Cijk_Alik_Bljk_SB_MT32x32x8_SN_1LDSB0_AP` | 17.27% |
| `wvSplitKSml` | 11.68% |
| `QsaGatherAttentionKernel` | 6.55% |
| `GdnScanK` | 5.01% |

Two kernels carry 52.4% and five carry 75.6%. **`HcGroupedNormKernel` is the
same kernel a `dgx:gpu0` `nsys` session found at 40.7% of CUDA decode time**
(§1): two backends, two architectures, one kernel at the top.

Full account: [`rocm-kernel-attrib-gfx1151-20260913.md`](../../docs/bench-evidence/rocm-kernel-attrib-gfx1151-20260913.md)
sections 8 and 9.

So §7's third risk was named correctly and sized wrongly: the 2026-09-07 stall
was not specific to the extra trace modes, and it was not a profiler defect
either. It was where the tool put its scratch file.

## 6. Owed

- The ranked table's top item gets its own row and spec with a falsifiable
  predicted gain. Nothing is optimized in this wave. On qwen4_exp that item is
  **`HcGroupedNormKernel` at 35.10%**, and the same kernel is 40.7% of CUDA
  decode time on `dgx:gpu0`, so the row it earns is not ROCm-only.
- **The #3040 swap-warning count for the qwen4_exp capture.** The tool printed
  `NOT SUPPLIED` rather than assuming zero. The count is in the archived
  `capture.log` on `/workspace` and no conclusion in §5a.3 rests on a margin
  near the 2.75% bound.
- **756 `__amd_rocclr_copyBuffer` dispatches per decode token**, 0.70% of GPU
  time inside a step that is 12.3% host-bound. Settling whether they are the
  host cost needs a HIP API trace, which this wave did not take.
- **Why a spill under `/` specifically fails.** `/` is writable and the doubled
  separator is accepted, so the mechanism is unidentified. The remedy does not
  depend on it.
- Per-row timestamp exactness stays with
  [#3040](https://github.com/mudler/vllm.cpp/issues/3040). This spec bounds the
  aggregate error at 2.75% of the kernel budget; it does not close #3040.
- A matched llama.cpp kernel trace is still owed under #3040 and is not produced
  here, because comparing the two engines needs the token gate this arm lacks.

## 7. Risks

- **gfx1151 faults about 2 of 5 identical greedy runs** with an illegal memory
  access (`ISSUE-LOCAL-01M2BY2M2ATNVR3XQKV2DB1BJD`). Every leg is repeated and a
  crash is checked against that signature before it is attributed to the
  profiler.
- The 2026-09-07 **full** trace (`--kernel-trace` plus HIP-runtime, memory-copy
  and scratch-memory tracing) reported a GPU hang. Only `--kernel-trace` is
  used, which is the configuration that completed 64 tokens on that same board.
- The sampler name is architecture-dependent. The tool takes it as a parameter
  and validates the segmentation, so a wrong name is a refusal and never a
  wrong table.
