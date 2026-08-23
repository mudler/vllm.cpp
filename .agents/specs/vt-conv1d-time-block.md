# VT-CONV1D-TIME-BLOCK — what limits the vocoder's parallel decomposition, measured

Row `VT-CONV1D-TIME-BLOCK`. Issue
[#1664](https://github.com/mudler/vllm.cpp/issues/1664). Lane
[#672](https://github.com/mudler/vllm.cpp/issues/672); the owed item is
[#1334](https://github.com/mudler/vllm.cpp/issues/1334) §18.9, *"the vocoder's
PARALLEL DECOMPOSITION is now the lever"*.

## Now

`DONE` pending review. The ablation is taken (§2a) and it refuted the candidate
the row was written to test; §2b prices the two changes and §2c is the paired
re-take that settles the second one. The gates and the mutation evidence are
§6, what the row declines to close is §7, and §9 records the outcome.

**The fresh review returned `FAIL` on the RECORDS and found the code correct.**
It attacked bit-identity from several directions, could not break it, and found
the guarantee stronger than the row claimed. What it found instead: the headline
curve was labelled with the wrong arm and the 4.11x ratio was composed across
two jobs. **Both are now MEASURED rather than labelled**: job `16b594ec` sweeps
the shipped arm at 11.54x of 14 and pairs it against the baseline at 4.08x in
one job on one boot (§2b, §9), which closes
[#1683](https://github.com/mudler/vllm.cpp/issues/1683) and opens
[#1770](https://github.com/mudler/vllm.cpp/issues/1770). The geometry gate transcribed its shapes
by hand and no model suite entered the blocked path arithmetically (§6c,
[#1684](https://github.com/mudler/vllm.cpp/issues/1684)), the header stated a
false premise for the tile alignment (§3b, §6b), and three counts in §6 had gone
stale. Every one is repaired or recorded above.

## 0. Scope

**In scope.** The CPU provider of `vt::Conv1d`
(`src/vt/cpu/cpu_conv1d_general.cpp`), its parallel decomposition, and the
instrument that decides what limits it.

**Out of scope, deliberately.**

- `vt::ConvTranspose1d`. §18.5 prices it at **~6 % of the chain's wall** at
  `-O3`, and its scatter form is input-driven: a time block of INPUT positions
  writes an output range that overlaps its neighbour's by `dilation * (kernel -
  1)`, so two blocks would contribute to one cell and the per-cell order would
  have to be argued rather than inherited. That is a different row.
- The **arithmetic**. Nothing here changes what is computed. §3 states the
  bit-identity argument and §6 states how it is gated.
- The **device arm**. `VLLM_CPP_VOCODER_DEVICE` still defaults to `kCPU`
  (`vocoder1d.cpp:98`) and this row does not move that default.
- `vt::cpu::ParallelForRows` itself. §4 argues why the decomposition belongs in
  the kernel's index mapping rather than in the shared pool primitive, and what
  that buys in blast radius.

## 1. The gap

`vocoder.decode_window` is **122.169 s of a 449.969 s run** at 20 s / 30 steps
on `thor:gpu0`, about **27 %**, and it is the largest term no row owns.

§18.8b measured the window at **2.16x per core** and **1.365x on 14 threads**,
which is a scaling of **6.76x (before) and 4.27x (after) of a possible 14x**.
The faster kernel scales worse. §18.8b attributed that to a shared-bandwidth
limit and **named the attribution as an inference**: *"no bandwidth counter was
read, and none is available on this worker."*

`Conv1dKernel` (`cpu_conv1d_general.cpp:152`) partitions
`rows = batch * out_channels` and nothing else. Every thread therefore sweeps
the whole input tensor for its own slice of output channels, and the reuse
available across output channels is not taken. At 344 latent frames the b3
residual activation is **67.6 MiB** against **96** output channels; the b1
residual is **33.8 MiB** against **384**.

## 2. What was not established, and the ablation for each

`ncu` is refused on this fleet — `ERR_NVGPUCTRPERM` even from the root `rc`
worker, `NVreg_RestrictProfilingToAdminUsers` on the host driver module — and
Thor is aarch64. So each candidate gets an ablation, and **a counter that is
unavailable is reported as unavailable rather than replaced by an estimate**.

| # | Candidate | The ablation that separates it |
|---|---|---|
| 1 | Activation residency | `--control`: the SAME geometry with the time axis cut into declared pieces, all output channels of a piece before the next. Identical arithmetic, identical code path, identical position tile; only the footprint one piece touches changes. Flat row ⇒ not the limit. A knee ⇒ the limit is AT a cache level and the knee names it |
| 2 | `Threadpool::Barrier` / dispatch | `--dispatch`: one empty `ParallelForRows` priced at each thread count, against the **62** dispatches the window makes (31 convolutions x 2 streams). Converts the worry into arithmetic. The residency control is BIASED AGAINST ITSELF here — 128 positions per piece is 172x more dispatches at the b3 geometry — so a control that still wins has refuted this twice |
| 3 | Work-partition granularity | Every row prints the chunk grid `ParallelForRows` will build, transcribed from `cpu_threadpool.cpp:413-458`. `conv_out` has ONE output row and therefore runs inline on the caller at every thread count; that is visible rather than suspected |
| 4 | The threads are not running | `getrusage` user+system CPU over the wall clock of the same window. Near 1.0 ⇒ not dispatched. Near the thread count while the speedup does not follow ⇒ dispatched and stalled |
| 5 | The CPU clock falls as cores light up | Not measurable from inside the process. The job script samples `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` around every leg and reports min/median/max, plus the governor. **A 14-core leg at a lower clock than a 1-core leg is a scaling loss that has nothing to do with the code** |

The instrument is `tools/bench/conv1d_scaling_probe.cpp`, registered as
`vllm_conv1d_scaling_probe` and as no test. The window's own curve comes from
the existing `vllm_music3_vocoder_conv_ab`, which drives `VocoderDecode` itself.

**The two instruments are never multiplied together.** §18.9 records a **1.80x
disagreement** between this kernel bench and the e2e bucket at 344 latents
(54.091 s against 97.4463 s through the same call), and §18.4 prices the
kernel-to-window gap at a further factor of ~2. A ratio from the probe is a
ratio ON the probe.

## 2a. THE ANSWER — measured, and it is not the convolution

**`rc` job `706f15ef-8add-4e8e-a976-954af66e90f5` on `thor:gpu0`**, worker
`rc-worker-m4d7t`, `Linux 6.8.12-1021-tegra` aarch64, 14 cores, boot id
`fabedc13-97a1-4cb9-909f-217a425d3f70`, `--max-runtime 150m`, 2026-08-22. One
lease, one boot, no `ssh`, no file mutex. Release `-O3`, CPU only, built inside
the lease from `1745d9017`. Box facts that the rest of this section rests on,
read rather than assumed: **L1d 64 KiB 4-way and L2 1 MiB 8-way, both PRIVATE
per core, and NO shared last-level cache in `sysfs` at all**; governor
`schedutil`, `scaling_max_freq` 2 601 000 kHz.

### The window does not scale, and the op does

| threads | `vocoder.decode_window`, 20 latents | speedup | `vt::Conv1d`, one of each geometry, 86 latents | speedup |
|---|---|---|---|---|
| 1 | 9.6374 s | 1.00x | 4.24978 s | 1.00x |
| 2 | 6.2616 s | 1.54x | 2.20365 s | 1.93x |
| 4 | 4.6068 s | 2.09x | 1.08414 s | 3.92x |
| 8 | 3.7876 s | 2.54x | 0.53646 s | 7.92x |
| 14 | 3.4247 s | **2.81x** | 0.34177 s | **12.44x** |

**The convolution scales 12.44x of 14 while the window it sits in scales
2.81x.** `user/wall` on the op is 13.91, 13.82, 13.52 and 13.45 at the four
heaviest geometries, so the pool is running and productive. Every arm printed
one fingerprint per length across every thread count, so nothing here trades
correctness for the ratio.

### Each candidate, and what killed it

| # | Candidate | Verdict | The measurement |
|---|---|---|---|
| 1 | Activation residency | **NOT the limit** | The residency sweep is nearly flat where it matters: at 14 threads `b2_res_conv1` reads 1.008x/1.021x/0.997x/0.989x across a 165x footprint range and `b3_res_conv1` reads 0.977x/1.019x/1.132x/1.067x. The largest reading anywhere at 14 threads is `b1_res_conv1` at **1.307x**. Real, and worth about 6 % of a window the convolution is 20 % of |
| 2 | Barrier / dispatch | **DEAD BY ARITHMETIC** | One empty `ParallelForRows` costs 11.845 µs at 14 threads. The window makes **62**. That is **0.734 ms against 3.4247 s — 0.02 %** |
| 3 | Granularity | **NOT the limit, with one real exception** | The chunk grid is 48-55 chunks on 14 threads at every heavy geometry. The exception is `conv_out`: ONE output row, so `chunks = 1` and `user/wall = 1.00` at every thread count. It is 0.2 % of the op total, so it is a defect and not the gap |
| 4 | The pool is not running | **DEAD** | `user/wall` 13.45-13.91 at 14 threads |
| 5 | The CPU clock falls as cores light up | **DEAD** | `scaling_cur_freq` sampled every 2 s across all 14 CPUs: max **2 601 000 kHz at every leg**, and the MEDIAN over 14 CPUs is 2 601 000 at 8 and at 14 threads — i.e. with the box full, the typical core is at the cap. (At 1 and 2 threads the median reads the 972 000 kHz idle floor, because 12 of the 14 samples are idle cores. That is the statistic reporting what it was asked, not a throttle) |

**No bandwidth counter was read and none is claimed.** `perf` is NOT INSTALLED
on the `thor:gpu0` worker and `perf_event_paranoid` is 2; `ncu` is refused on
this fleet. Candidate 1 is settled by ablation instead, and the ablation says
residency is worth 1.0-1.31x rather than the several-fold factor a
bandwidth-saturation story needs.

### So where does the window go? The split says it in one line

`vocoder.decode_window` split into leaves through `profile::Timer` on the
production path, 14 threads. **PROVENANCE, because the two halves of this
section come from different boxes:** the scaling curve and the five verdicts
above are `thor:gpu0` under the lease named at the top. The split below is the
AUTHORING host (x86-64, 20 cores, `uptime` load 18.5 — not an idle box) at 20
latents, because the instrument did not exist when the Thor job was submitted.
It is quoted as a SHARE and never as a duration, and §2b re-takes it on Thor.

| leaf | seconds | calls | share |
|---|---|---|---|
| **`vocoder.snake`** | 4.231 | 58 | **70.70 %** |
| `vocoder.conv1d` | 1.197 | 54 | 20.00 % |
| `vocoder.conv_transpose` | 0.489 | 8 | 8.17 % |
| `vocoder.pad` | 0.053 | 28 | 0.89 % |
| `vocoder.copy` | 0.006 | 32 | 0.10 % |
| `vocoder.residual_add` | 0.004 | 24 | 0.07 % |
| `vocoder.tanh` | 0.000 | 2 | 0.00 % |
| sum(leaf) | 5.982 | — | 99.95 % |

`vocoder1d::SnakeActivation` is a **serial** loop — no partition of any kind —
over every element of every activation in the chain, computing a `double`
`std::sin` per element. Amdahl's law over it predicts the window's whole curve:
solving `p/12.44 + (1 - p) = 3.4247/9.6374` gives **p = 0.70**, and the split
independently reads the parallel part at 0.70 of the T=1 window if the snake is
the serial term. Two instruments, one number.

**§18.8b's shared-bandwidth candidate is therefore refuted rather than
refined.** The scaling it measured was real, its reading of the cause was not,
and the reason it could not see this is stated plainly: it timed the WINDOW and
reasoned about the KERNEL, with nothing in between. The instrument that was
missing was a split, not a counter.

## 2b. The A/B/C on `thor:gpu0` — what each change is worth

**`rc` job `3ca07477-f3b7-402a-bcc4-b6af55f30a66`**, same worker
`rc-worker-m4d7t`, **same boot id `fabedc13-97a1-4cb9-909f-217a425d3f70`** as
§2a, so every ratio on this page is inside one boot. Three SOURCE TREES built
inside the lease into three binaries, hashed, and the run refuses to time
anything if two hash the same:

| arm | tree | `vllm_music3_vocoder_conv_ab` sha256 |
|---|---|---|
| A | `3b00897fe` — instrumented baseline, no perf change | `4e9b30b65ffd2aad…` |
| B | `fd99a0d7f` — A + the parallel snake | `5f6c9960a2ef74b3…` |
| C | `cf9296496` — B + the conv decomposition, blocked UNCONDITIONALLY | (third distinct hash) |

**Correctness first, on arm C, before any speed number was read** — five suites,
`test cases:` / `assertions:` / `Status:` in full: `test_ops_conv1d_general`
14/19 617/`SUCCESS!`, `test_host_parallel` 11/968/`SUCCESS!`, `test_vocoder1d`
10/58/`SUCCESS!`, `test_bigvgan` 6/65/`SUCCESS!`,
`test_minimax_music3_acoustic` 36/345/`SUCCESS!`. All rc 0.

**And every arm is bit-identical at full scale.** Across all three arms, every
round and every thread count from 1 to 14, each length produced ONE waveform
fingerprint: `0xcdfc4309a0070783` at 20 latents and `0xc2d5eaf095d1c483` at 86.

### The window's scaling curve, before and after — ON ARM C, WHICH DOES NOT SHIP

20 latents, best of 3, `VLLM_CPP_CPU_THREADS` swept:

| threads | arm A | speedup | arm C, blocked UNCONDITIONALLY | speedup |
|---|---|---|---|---|
| 1 | 9.6374 s | 1.00x | 9.4392 s | 1.00x |
| 2 | 6.2616 s | 1.54x | 4.8859 s | 1.93x |
| 4 | 4.6068 s | 2.09x | 2.4640 s | 3.83x |
| 8 | 3.7876 s | 2.54x | 1.2884 s | 7.33x |
| 14 | 3.4247 s | **2.81x** | 0.8223 s | **11.48x** |

**2.81x of 14 becomes 11.48x of 14**, and the arm-to-arm ratio at the shipped
default is **3.4478 / 0.8223 = 4.19x** (arm A re-measured in the same job).

**READ THE COLUMN HEADER, because the shipped tree is not in this table.** The
right-hand arm is C — `cf9296496`, blocked UNCONDITIONALLY — and the shipped
arm is D, which added the `out_channels * kernel <= in_len` condition in
`06ba79d1b`. When this table was written, arm D had exactly ONE measured
operating point, 86 latents at 14 threads (§2c), and no thread sweep of it
existed. The subsection below is that sweep, taken in a later job on a later
boot, and it also replaces the "D should be at least C" inference this
paragraph used to carry
([#1683](https://github.com/mudler/vllm.cpp/issues/1683)).

### The SHIPPED arm's own curve, and the paired A-vs-D — `rc` job `16b594ec`, a SECOND boot

**`rc` job `16b594ec-7987-4cae-b377-414adbe0f944`** on `thor:gpu0`, worker
`rc-worker-kk96r`, `Linux 6.8.12-1021-tegra` aarch64, 14 cores, boot id
`e2112cac-660b-434e-911d-33cbd29b9176`, `--max-runtime 170m`, 2026-08-23.
`ONE BOOT: OK` — the job reads the boot id before and after and compares them,
so every ratio in this subsection is inside one boot. Governor `schedutil`,
`scaling_max_freq` 2 601 000 kHz. Release `-O3`, CPU-only, built inside the
lease. The whole log is committed at
[`docs/bench-evidence/vt-conv1d-time-block-1683-thor-20260823.log`](../../docs/bench-evidence/vt-conv1d-time-block-1683-thor-20260823.log),
because §2c's complaint about the earlier pair was that no job log was in the
tree and the load window could not be recovered from it.

**This is a DIFFERENT BOOT from §2a, §2b and §2c**, which all ran on
`fabedc13-97a1-4cb9-909f-217a425d3f70` and worker `rc-worker-m4d7t`. That is why
all three arms are rebuilt and re-measured here rather than one: every ratio
below is paired inside this boot, and no number here is divided by a number from
the other one. The two boots agree closely on the two points they share — arm A
at 20 latents on one thread reads 9.8952 s here against 9.6374 s there (2.7 %),
and arm C at 20 latents on 14 threads reads 0.8047 s here against 0.8223 s there
(2.1 %) — which is well inside the 12.79 % cross-boot spread #543 records, but
is not a licence to mix them.

**Three trees that differ ONLY in the row's own files.** All three start from
`origin/main` at `8eecc05a9`, and the job prints the sha256 of each file it
replaced:

| arm | tree | what it is |
|---|---|---|
| A | `8eecc05a9` + `cpu_conv1d_general.cpp` and `vocoder1d.cpp` from `3b00897fe` | the instrumented baseline: no parallel snake, no conv decomposition |
| C | `8eecc05a9` + `cpu_conv1d_general.cpp` and `cpu_conv1d_block.h` from `cf9296496` | blocked UNCONDITIONALLY |
| D | `8eecc05a9`, UNMODIFIED | **the tree that ships** |

`CONFIGURE_RC` and `BUILD_RC` are 0 for all three, the three window binaries and
the three probe binaries all hash differently, and the job refuses with
`FATAL_CLONE` before it times anything if any two agree.

**Correctness first, on the SHIPPED arm, before any speed number was read** —
`test cases:` / `assertions:` / `Status:` in full: `test_ops_conv1d_general`
14/19696/`SUCCESS!`, `test_host_parallel` 11/968/`SUCCESS!`, `test_vocoder1d`
11/66/`SUCCESS!`, `test_bigvgan` 6/65/`SUCCESS!`,
`test_minimax_music3_acoustic` 39/386/`SUCCESS!`. All rc 0.
`test_ops_conv1d_general` also printed its four CUDA `[SKIP]` lines, which are
the CPU-only build and not a result.

**The settle, and what it could and could not reach.** The three builds took the
one-minute load average to 8.45. The job then waited, printing the load every
60 s: 5.06, 4.07, 4.19, 3.75, 3.57, 3.00, 3.06, 3.16, 3.38, 3.21, 2.83, 2.95,
3.40, 3.31, **2.82**. It was written to stop early below 1.5 and it never got
there, so it hit its own 900 s ceiling and said so (`SETTLE_CEILING_HIT
load1=2.82`). **1.5 was the wrong number for this box:** the job's own reading
before it built anything was 2.83, so the settle returned the worker to its
pre-job floor and the ceiling fired only because the gate was set below a floor
nobody had measured. Every timed leg after that ran alone — the script is
strictly serial — so the load averages later in the log are the benchmark's own
decaying utilisation and not a second job.

#### The sweep — 20 latents, best-of-3 per point, three alternated rounds

The arms alternate at every thread count, and the order reverses on even rounds
(`A C D`, then `D C A`). Each cell is the MEDIAN of the three rounds' best-of-3,
and 20 latents with best-of-3 is §2b's own statistic so that the columns are
comparable to the table above.

| threads | arm A | speedup | arm C (unconditional) | speedup | **arm D (SHIPPED)** | **speedup** |
|---|---|---|---|---|---|---|
| 1 | 9.8952 s | 1.00x | 9.2368 s | 1.00x | 9.2816 s | 1.00x |
| 2 | 6.3894 s | 1.55x | 4.7914 s | 1.93x | 4.7975 s | 1.93x |
| 4 | 4.6380 s | 2.13x | 2.4112 s | 3.83x | 2.4406 s | 3.80x |
| 8 | 3.8322 s | 2.58x | 1.2361 s | 7.47x | 1.2510 s | 7.42x |
| 14 | 3.4713 s | **2.85x** | 0.8047 s | **11.48x** | 0.8044 s | **11.54x** |

**2.85x of 14 becomes 11.54x on the arm that ships**, and the arm-to-arm ratio
at the shipped default is 3.4713 / 0.8044 = **4.32x** at this length. Arm C
re-measured in this boot reads 11.48x, the same value §2b recorded for it in the
other one.

Each cell's three rounds spread by 0.37 % to 3.34 %, and the raw legs are in the
committed log. Every leg of every arm at every thread count printed
`0xcdfc4309a0070783`, and every 86-latent leg printed `0xc2d5eaf095d1c483` —
two fingerprints in the whole job, which is bit-identity across all three arms.

#### The paired A-vs-D — 86 latents, 14 threads, seven alternated rounds

This is the leg #1683 asked for: the numerator and the denominator in ONE job,
alternated, after the settle, so the ratio stops being composed across two.

| round | order | arm A | arm C | **arm D** |
|---|---|---|---|---|
| 1 | A C D | 14.3613 s | 3.3681 s | 3.4030 s |
| 2 | D C A | 14.3082 s | 3.4099 s | 3.5158 s |
| 3 | A C D | 14.3519 s | 3.3811 s | 3.5072 s |
| 4 | D C A | 14.2748 s | 3.4683 s | 3.4285 s |
| 5 | A C D | 14.3855 s | 3.4078 s | 3.3942 s |
| 6 | D C A | 14.2744 s | 3.3899 s | 3.5302 s |
| 7 | A C D | 14.2568 s | 3.3838 s | 3.5211 s |
| **median** | | **14.3082 s** | **3.3899 s** | **3.5072 s** |

**A-against-D is 4.08x, paired.** The per-round ratios are 4.220, 4.070, 4.092,
4.164, 4.238, 4.044 and 4.049; their median is **4.092x** and the median of the
two medians is **4.080x**. The loudest pair, 4.238x, is kept rather than quoted.
The composed 4.11x this replaces was therefore about 0.7 % high, which is the
first statement anyone can make about that number rather than about its
provenance. Medians rather than best-of, for §2c's reason: the arms have
different spreads.

#### Arm D against arm C — the condition costs nothing and buys nothing here

**At 20 latents the two arms are within 1.2 % at every thread count**: C/D reads
0.9952, 0.9987, 0.9880, 0.9881 and 1.0004 at 1, 2, 4, 8 and 14 threads. At 8
threads the two arms' raw ranges do not overlap (C 1.2307-1.2421 s, D
1.2463-1.2545 s), so that 1.2 % is a real reading rather than noise — and it is
1.2 % the wrong way round for the "D should be at least C" inference.

**At 86 latents the paired median puts arm D 3 % BEHIND arm C** (3.3899 against
3.5072, 0.9699x on the per-round median). **That gap cannot be the condition,
and the arithmetic says so.** At 86 latents the rule decides differently on
exactly four shapes — `dec_in_proj`, `conv_in`, `b0_res_conv1` and
`b0_res_conv2`. `vocoder.conv1d` makes 54 calls per window, so those four run 2,
2, 6 and 6 times, and the per-call deltas measured by the op probe below bound
the condition's whole effect on the window at **0.36 ms, about 0.01 % of a 3.5 s
window**. What the 3 % is instead: arm D's seven legs are bimodal, at
3.394-3.429 s and 3.507-3.530 s, with no correlation to the order it ran in,
while arm C's seven sit inside 3.368-3.468 s. On best-of the two arms are
3.3681 s against 3.3942 s, 0.992x. The leaf split and the op probe in the same
job both read the two arms as a tie. **The 3 % is reported because it was
measured, and it is NOT attributed to the condition, because a 0.36 ms lever
cannot move a 117 ms gap.** The residual is unexplained variance in the window
legs and is owed
([#1770](https://github.com/mudler/vllm.cpp/issues/1770)).

#### The BEHAVIOURAL control — because a hash is not the proof (#1516)

Op-level probe, `vllm_conv1d_scaling_probe`, 14 threads, 86 latents,
`--repeats=2`, three alternated rounds, medians:

| geometry | rule on D | arm A | arm C | arm D | C/D | A/D | `user/wall` A / C / D |
|---|---|---|---|---|---|---|---|
| `dec_in_proj` k1 | declined | 0.00020 s | 0.00013 s | 0.00015 s | 0.867x | 1.333x | 15.40 / 9.50 / 13.18 |
| `conv_in` k7 | declined | 0.01677 s | 0.01633 s | 0.01649 s | 0.990x | 1.017x | 13.84 / 13.85 / 13.76 |
| `b0_res_conv1` k7 | declined | 0.03840 s | 0.04073 s | 0.03824 s | **1.065x** | 1.004x | 13.85 / 13.85 / 13.88 |
| `b0_res_conv2` k1 | declined | 0.01105 s | 0.00897 s | 0.01134 s | **0.791x** | 0.974x | 13.72 / 13.70 / 13.82 |
| `b1_res_conv1` k7 | taken | 0.07276 s | 0.07203 s | 0.07146 s | 1.008x | 1.018x | 13.85 / 13.79 / 13.85 |
| `b1_res_conv2` k1 | taken | 0.02511 s | 0.01711 s | 0.01660 s | 1.031x | **1.513x** | 13.85 / 13.63 / 13.75 |
| `b2_res_conv1` k7 | taken | 0.08671 s | 0.06910 s | 0.06969 s | 0.992x | 1.244x | 13.73 / 13.88 / 13.84 |
| `b2_res_conv2` k1 | taken | 0.02969 s | 0.01779 s | 0.01761 s | 1.010x | **1.686x** | 13.68 / 13.85 / 13.92 |
| `b3_res_conv1` k7 | taken | 0.04150 s | 0.03461 s | 0.03426 s | 1.010x | 1.211x | 13.70 / 13.88 / 13.90 |
| `b3_res_conv2` k1 | taken | 0.01376 s | 0.00821 s | 0.00891 s | 0.921x | **1.544x** | 13.49 / 13.82 / 13.71 |
| `conv_out` k7 | taken | 0.00866 s | 0.00066 s | 0.00072 s | 0.917x | **12.03x** | **1.00** / **14.22** / **13.06** |
| TOTAL, one of each | | 0.34630 s | 0.28663 s | 0.28764 s | 0.996x | 1.204x | |

**`conv_out`'s `user/wall` is the control that a hash cannot give.** It reads
**1.00 on arm A and 14.22 and 13.06 on arms C and D** — the `rows == 1` inline
path, which ran the whole convolution on the caller at every thread count, being
reached on the two arms that carry the second axis and not on the one that does
not. Three binaries with three hashes prove only three build directories
(#1516); this is the arms behaving as three arms. The `chunks` column is NOT a
control here, because the probe derives it from `out_channels` and the thread
count, so it is identical on all three arms by construction.

**The declined-versus-taken pattern the condition predicts is only HALF there.**
Every shape the rule takes gains against the baseline — 1.21x to 1.69x on the
k1 residual convolutions and 12.03x on `conv_out` — which is §2c's result
reproduced on a second boot. What does not reproduce is the evidence the
condition was derived FROM. §2b read arm C at **0.82x and 0.89x** on
`b0_res_conv1` and `b0_res_conv2`. Here `b0_res_conv1` keeps its direction and
loses three quarters of its size (1.065x, so arm C is 6.5 % slower), and
`b0_res_conv2` REVERSES (0.791x, so arm C is 21 % faster on a shape the rule
declines). Over the two shapes together arm C reads 0.04970 s against arm D's
0.04958 s, a tie. So the shipped condition is measured NEUTRAL on this box, by
three instruments in one job, and the reading that justified it is not
reproducible ([#1770](https://github.com/mudler/vllm.cpp/issues/1770), §7).

#### The split, all three arms, one length

86 latents, 14 threads, two rounds, `vocoder.decode_window` in leaves:

| leaf | arm A | arm C | arm D |
|---|---|---|---|
| `vocoder.snake` | 11.463 / 11.457 s | 0.980 / 1.046 s | 0.976 / 0.968 s |
| `vocoder.conv1d` | 2.096 / 2.091 s | 1.719 / 2.036 s | 1.733 / 1.710 s |
| `vocoder.conv_transpose` | 0.610 / 0.607 s | 0.523 / 0.623 s | 0.529 / 0.526 s |
| `vocoder.pad` | 0.084 / 0.080 s | 0.077 / 0.080 s | 0.079 / 0.074 s |
| `vocoder.copy` | 0.048 / 0.049 s | 0.047 / 0.054 s | 0.050 / 0.045 s |
| `vocoder.residual_add` | 0.067 / 0.068 s | 0.064 / 0.066 s | 0.066 / 0.066 s |
| TOTAL | 14.383 / 14.363 s | 3.421 / 3.919 s | 3.444 / 3.397 s |

`vocoder.snake` **11.46 s → 0.97 s, 11.8x** on the shipped arm, against §2b's
11.96x for arm C on the other boot.

**What this job does NOT carry.** It samples
`/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` BETWEEN legs rather than
during them, so most of its readings are the idle clock (972 000 to
1 728 000 kHz) and only one sample caught 2 601 000 kHz. Candidate 5 of §2 —
the CPU clock falling as cores light up — is therefore inherited from §2a's
during-the-leg sampling and is not re-refuted here (§7).

### The split says the same thing twice

`thor:gpu0`, 86 latents, 14 threads, arm A against arm C:

| leaf | arm A | share | arm C | share |
|---|---|---|---|---|
| `vocoder.snake` | **11.418 s** | **79.35 %** | 0.955 s | 27.35 % |
| `vocoder.conv1d` | 2.212 s | 15.37 % | 1.727 s | 49.43 % |
| `vocoder.conv_transpose` | 0.542 s | 3.77 % | 0.604 s | 17.30 % |
| `vocoder.pad` | 0.085 s | 0.59 % | 0.079 s | 2.27 % |
| `vocoder.copy` | 0.049 s | 0.34 % | 0.048 s | 1.37 % |
| `vocoder.residual_add` | 0.069 s | 0.48 % | 0.066 s | 1.90 % |
| sum(leaf) / TOTAL | 14.376 / 14.390 s | 99.90 % | 3.480 / 3.493 s | 99.63 % |

`vocoder.snake` **11.418 s → 0.955 s, 11.96x**.

**And the serial claim is confirmed by a second, independent reading.** On arm A
the snake costs 2.653 s at 20 latents on ONE thread and 11.418 s at 86 latents
on FOURTEEN — 0.13265 and 0.13277 seconds per latent frame. A loop with no
partition costs the same wall clock however many cores are idle beside it, and
that is what two legs at different lengths and different thread counts read, to
four significant figures.

### The conv decomposition, priced on its own

The op-level probe, arm A against arm C at 14 threads, `latents=86`, paired in
one job:

| geometry | arm A | arm C | ratio |
|---|---|---|---|
| `conv_in` k7 | 0.01660 s | 0.01671 s | 0.99x |
| **`b0_res_conv1` k7** | 0.03805 s | 0.04656 s | **0.82x** |
| **`b0_res_conv2` k1** | 0.01117 s | 0.01261 s | **0.89x** |
| `b1_res_conv1` k7 | 0.07551 s | 0.07616 s | 0.99x |
| `b1_res_conv2` k1 | 0.02582 s | 0.02009 s | 1.29x |
| `b2_res_conv1` k7 | 0.10550 s | 0.08399 s | 1.26x |
| `b2_res_conv2` k1 | 0.03118 s | 0.01748 s | 1.78x |
| `b3_res_conv1` k7 | 0.04609 s | 0.03489 s | 1.32x |
| `b3_res_conv2` k1 | 0.01715 s | 0.00841 s | 2.04x |
| **`conv_out` k7** | 0.00913 s | 0.00087 s | **10.49x** |
| TOTAL, one of each | 0.37634 s | 0.31857 s | 1.18x |

`conv_out`'s `user/wall` goes from **0.98 to 15.18**: that is the `rows == 1`
inline path, which ran the whole convolution on the caller at every thread
count, being reached for the first time.

**The two b0 losses are the reason §3b acquired a condition.** Where the weight
tensor is 16.5 MiB against a 2.1 MiB activation, re-reading the weights once per
block costs more than reading the activation 768 times saves. Arm D
(`0f738d6ec`) blocks only where `out_channels * kernel <= in_len`, and §2c pairs
it against arm B.

**Why the whole-window A/B/C rounds do NOT settle B against C.** They ran at
`uptime` load 8.84, the decaying residue of three back-to-back builds in the
same lease, and the two arms landed inside that noise of each other — B
0.8852/0.8877 and C 0.9309/0.8365 at 20 latents. That is recorded as a defect in
the SCHEDULE of the job rather than as a result, and §2c is the re-take with a
settle period and seven alternated rounds.

## 2c. The paired B-vs-D re-take, and the rule holds on every geometry

**`rc` job `214f5f70-9ed4-460b-82c8-3ca62411877e`**, `thor:gpu0`, worker
`rc-worker-m4d7t`, **same boot id `fabedc13-97a1-4cb9-909f-217a425d3f70`** as
§2a and §2b. Two trees, two binaries, distinct hashes, and the schedule defect
§2b names is repaired rather than argued away: the job **sleeps 300 s after the
builds and prints `uptime` on both sides of the wait**, then alternates B and D
seven times.

Arm B is `fd99a0d7f` (the parallel snake alone). Arm D is `0f738d6ec` — B plus
the conv decomposition **conditioned** on `out_channels * kernel <= in_len`.

### The window, 86 latents, 14 threads, seven alternated rounds

| round | arm B | arm D |
|---|---|---|
| 1 | 3.7293 s | 3.7612 s |
| 2 | 3.7042 s | 3.4755 s |
| 3 | 4.0695 s | 3.4620 s |
| 4 | 3.9063 s | 3.4989 s |
| 5 | 3.7452 s | 3.7632 s |
| 6 | 3.7319 s | 3.8783 s |
| 7 | 3.7183 s | 3.4770 s |
| **median** | **3.7319 s** | **3.4989 s** |

**1.067x**, and against arm A's 14.3895 s at the same length the shipped arm is
**4.11x**. Medians rather than best-of, because the two arms have different
spreads and the loudest single pair (round 3, 1.176x) is kept rather than
quoted.

**THE 1.067x IS PAIRED; THE 4.11x WAS COMPOSED ACROSS TWO JOBS, AND IT IS
SUPERSEDED.** The numerator, 3.4989 s, is arm D's median here — 300 s settle,
seven alternated rounds. The denominator, 14.3895 s, was arm A's 86-latent leg
in job `3ca07477`, the job whose whole-window rounds §2b records as running at
`uptime` load 8.84 and calls "a defect in the SCHEDULE of the job rather than as
a result". Same boot id and same worker, but a different job, not alternated
against D, and not under the settle, and whether that leg fell inside the
load-8.84 window cannot be recovered, because that job's log was never
committed. **The paired replacement is 4.08x**, taken in job `16b594ec` with
both arms alternated in one job on one boot after a settle whose load is printed
every 60 s (§2b). Quote 4.08x. The 4.11x is kept here as the composed figure it
was, and the two differ by 0.7 %
([#1683](https://github.com/mudler/vllm.cpp/issues/1683)).

### The paired split, both arms, same length

| leaf | arm B | arm D | ratio |
|---|---|---|---|
| `vocoder.conv1d` | 2.192 / 2.205 s | 1.720 / 1.758 s | **1.27x / 1.25x** |
| `vocoder.snake` | 1.043 / 1.023 s | 0.962 / 0.991 s | 1.08x / 1.03x |
| `vocoder.conv_transpose` | 0.541 / 0.540 s | 0.602 / 0.607 s | 0.90x / 0.89x |
| TOTAL | 3.987 / 3.985 s | 3.492 / 3.573 s | 1.14x / 1.12x |

### Per geometry — and this is what the condition was for

Op-level probe, paired, three alternated rounds. Round 2, and the medians agree:

| geometry | rule | arm B | arm D | ratio |
|---|---|---|---|---|
| `dec_in_proj` k1 | declined | 0.00018 s | 0.00018 s | 1.00x |
| `conv_in` k7 | declined | 0.01562 s | 0.01647 s | 0.95x |
| `b0_res_conv1` k7 | declined | 0.03803 s | 0.03865 s | 0.98x |
| `b0_res_conv2` k1 | declined | 0.01108 s | 0.01122 s | 0.99x |
| `b1_res_conv1` k7 | taken | 0.07797 s | 0.07287 s | 1.07x |
| `b1_res_conv2` k1 | taken | 0.02447 s | 0.01686 s | **1.45x** |
| `b2_res_conv1` k7 | taken | 0.08204 s | 0.06898 s | 1.19x |
| `b2_res_conv2` k1 | taken | 0.02902 s | 0.01777 s | **1.63x** |
| `b3_res_conv1` k7 | taken | 0.04124 s | 0.03437 s | 1.20x |
| `b3_res_conv2` k1 | taken | 0.01433 s | 0.00851 s | **1.68x** |
| `conv_out` k7 | taken | 0.00841 s | 0.00055 s | **15.3x** |
| TOTAL, median of 3 | | 0.34977 s | 0.29552 s | **1.18x** |

**Nothing regresses.** The four shapes the rule declines are ties inside the
run's own spread; every shape it takes gains. Against §2b's unconditional arm,
where the same two b0 shapes read **0.82x and 0.89x**, the condition is what
turned a mixed result into a monotone one.

`conv_out`'s `user/wall` goes from **1.00 to 12.26**. That is the `rows == 1`
inline path — the whole convolution on the caller at every thread count — being
reached for the first time, and it is the clearest single reading that the
second axis is live at a production shape.

**Bit-identity throughout.** Every leg of every round on both arms printed
`0xc2d5eaf095d1c483`.

## 3. What the row changes, and why each is bit-identical BY CONSTRUCTION

The measurement moved the row's lever, so there are TWO changes and they are
not the same size. The larger one is the one §2a found; the smaller one is the
one the row was written to make, kept because it is measured and because it
reaches a shape the row partition could not.

### 3a. The activation function gets a partition — the row's lever

`vocoder1d::SnakeActivation` had none. It now partitions the CHANNEL axis
through `host_parallel::ForOutputRows`, the same seam every other host-reference
body in this tree uses.

**It needs none of the argument §3b needs.** Output element `(c, t)` is a
function of input element `(c, t)` and of `alpha[c]` alone. There is no
reduction, so there is no summation order, no accumulator, no reassociation and
no tolerance. The `double` intermediates, the `std::sin` and `kSnakeEps` are
UNTOUCHED — narrowing them is a numerics change an oracle owns and this row does
not make.

The size guard is the shared one, so a small activation still runs inline; and
`channels == 1` stays inline, which no shipped consumer geometry hits.

### 3b. The convolution gets a second axis

**Blocked over the time axis, and parallel over (time block, output row).**

The shipped nest is `for oc: for t0 tile: for ic: for k`. The proposed nest is
`for t block: for oc: for t0 tile: for ic: for k`, with the parallel grid over
the flattened `(block, oc)` pair rather than over `oc` alone.

Fix any single output cell and read what it receives, in order: the bias, then
`(ic=0,k=0)`, `(ic=0,k=1)`, ... — **the identical sequence of IEEE-754
additions of the identical products, in the identical order**, in the identical
f32 accumulator. Nothing splits a reduction, nothing accumulates atomically,
and no cell is touched by two units of work. This is §18.3's argument, applied
to a second scheduling change rather than restated for it.

Two conditions make the equality hold at the level of the code path as well as
the arithmetic, and both are gated:

1. **The block length is a multiple of `kConv1dPosTile`.** Position tiles then
   land on exactly the boundaries they land on today, so a tile that takes the
   constant-trip-count fast path today takes it after the change. §18.4 prices
   that path at up to 5x, so a change that silently moved a tile onto the
   chunked path would report a speed result about a different kernel.
   **THIS IS A PERFORMANCE INVARIANT AND NOT PART OF THE BIT-IDENTITY
   ARGUMENT**, which `src/vt/cpu/cpu_conv1d_block.h` used to say it was. The
   argument above never mentions `t0`: a cell is seeded and swept `ic` ascending
   then `k` ascending whatever tile it falls in, so re-cutting the tiles cannot
   move a bit — and §6b measures exactly that, with the misalignment mutation
   reddening 1 014 assertions of which every single one is an alignment property
   check and none is arithmetic. The gate on the property stays, for the speed
   reason rather than the correctness one.
2. **The trailing block carries the remainder.** `length` is not in general a
   multiple of the block, and the last block is short exactly as the last
   position tile is short today.

`conv_out` is the case that shows the decomposition is not only about cache: it
has **one** output row, so it is inline and serial at every thread count today,
and a `(block, oc)` grid gives it `ceil(length / block)` units of work.

## 4. The seam

`vt::cpu::ParallelForRows` **is** the seam and it is not modified. What changes
is the index space the kernel hands it: `rows` becomes `blocks * rows`, and the
body decodes the pair. That is the same relationship `cpu_paged_attn.cpp:211`
already has with the primitive (`total_q`, a flattened pair) and
`cpu_attn_relpos.cpp:89` (`hq * t`).

**This is a deliberate choice about blast radius, not an avoidance of the
seam.** A new pooled primitive would be inherited by every one of the primitive's
consumers — `cpu_layernorm.cpp` (5 sites), `cpu_quant_gemm.cpp` (3),
`cpu_paged_attn.cpp`, `cpu_attn_relpos.cpp`, `cpu_conv2d.cpp`,
`cpu_conv3d.cpp`, `cpu_conv1d_depthwise.cpp`, `cpu_quant_repack_arm.cpp`,
`cpu_ops.cpp`, `tenstorrent_ops.cpp` and `ltx2_video_vae.cpp` — and would owe a
gate on each. The blocking factor that is right for a convolution over a 67.6
MiB activation is not a property those consumers share, and inventing one
shared knob for them would be the parallel path this rule exists to prevent,
wearing a seam's name.

**Consumers this row does affect, and how each is gated** — every model that
reaches `vt::Conv1d` on the CPU device:

| Consumer | Reached from | Gate |
|---|---|---|
| MiniMax-Music3 vocoder | `vocoder1d::Conv1d` | `test_vocoder1d`, `test_minimax_music3_acoustic` |
| LTX-2.5 audio VAE | `ltx2_audio_vae.cpp` | `test_ltx2_audio_vae` — **13 audio arms at `5e-6`, the tightest tolerance any consumer holds** |
| IndexTTS-2.5 BigVGAN | `bigvgan.cpp` | `test_bigvgan` |
| MiniMax-H3 audio VAE | `minimax_h3_audio_vae.cpp` | `test_minimax_h3_audio_vae` |
| the op itself | `vt::Conv1d` | `test_ops_conv1d_general`, `test_host_parallel` |

## 5. Risks

- **R1 — a residency win that does not exist on the box that matters.** The
  authoring host is an x86 part with a very large last-level cache, where the
  whole activation is already resident and the control is flat BY
  CONSTRUCTION. Every ratio this row quotes is from `thor:gpu0`, and the
  authoring host is used for compile and correctness only.
- **R2 — the block length becomes a tuned constant nobody can re-derive.**
  Mitigated by the residency sweep: the block is chosen at the knee the sweep
  measures, and the sweep is a shipped executable.
- **R3 — the ratio is a code-path ratio.** Mitigated by §3's multiple-of-32
  condition and gated by a test that asserts it.
- **R4 — the answer is candidate 5.** If the 14-core clock is materially below
  the 1-core clock, part of §18.8b's 4.27x is thermal and is not addressable
  from this repository. It is then **reported as such** and the remaining
  attributable part is what this row acts on. Leases cannot pin clocks
  (`LGC_RC=4`, [#1354](https://github.com/mudler/vllm.cpp/issues/1354)), so
  sampling is the only instrument available.

## 6. Gates and evidence

### 6a. The suites — counts and `Status:` in full, because `assertions: 0` is a skip wearing a pass

Authoring host, x86-64 20 cores, Release `-O3`, CPU only, at the row's head.
Every binary's sha256 was taken before it ran.

| suite | result | rc |
|---|---|---|
| `test_ops_conv1d_general` | 14 cases, 19 615 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_host_parallel` | 11 cases, 968 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_vocoder1d` | 10 cases, 58 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_bigvgan` | 6 cases, 65 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_minimax_music3_acoustic` | 39 cases, 386 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_ltx2_vae` | 44 cases, 3 131 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_minimax_h3` | 79 cases, 57 395 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_indextts2_pipeline` | 8 cases, 433 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_indextts2_family` | 7 cases, 22 assertions, 0 failed, `SUCCESS!` | 0 |
| `test_ops_conv1d_depthwise` | 5 cases, 1 184 assertions, 0 failed, `SUCCESS!` | 0 |

RE-TAKEN AT THE REVIEW-REPAIR HEAD, and the row's earlier printing of
`test_minimax_music3_acoustic` as 36/345 is corrected rather than left standing:
that suite runs 39/386 once `origin/main` is merged in, and it was recorded
before that merge. The two conv suites also move at this head, because the
repair adds cases to both — `test_ops_conv1d_general` 14 cases / **19 696**
assertions (was 19 615: the derived geometry sweep replaces six hand-copied
shapes with 27+27 evaluated ones) and `test_vocoder1d` **11** cases / **66**
assertions (was 10/58: the block-boundary case, §6c).

`test_ops_conv1d_general` prints four `[SKIP]` lines and they are read rather
than ignored: this host has no CUDA toolkit, so every CPU-vs-CUDA arm in that
file — including #1334's two cancellation arms — did NOT run. The device
provider is unchanged by this row and §7 carries the re-measure as owed.

### 6b. Red before green — every mutation with its compile rc, its hunk count and its binary sha256

Baseline binaries: `test_ops_conv1d_general` `0f886491b735fdea…`,
`test_host_parallel` `63a79608a7f270de…`.

| # | Mutation | compile_rc | hunks | binary sha256 | result |
|---|---|---|---|---|---|
| M1 | `Conv1dTimeBlock` drops the position-tile rounding | 0 | 1 | `08c97a3952bc6b57…` | **2 cases / 1 343 assertions FAILED**, rc 1 |
| M2b | the trailing time block is dropped (floor for ceil) | 0 | 1 | `463a2bab3160077c…` | **2 cases / 2 assertions FAILED**, rc 1 |
| M3 | the snake partition drops the last channel of every chunk | 0 | 1 | `82b2f989663a4dc6…` | **1 case / 20 assertions FAILED**, rc 1 |
| M4 | the snake body is never dispatched | 0 | 1 | `daa17366f828bc71…` | **1 case / 20 assertions FAILED**, rc 1 |
| M5b | the registered CPU `vt::Conv1d` provider refuses every call | 0 | 1 | per suite, all distinct | **every consumer suite red** — see 6c |
| — | restored | 0 | 0 | `0f886491b735fdea…`, `63a79608a7f270de…` | back to baseline, byte for byte |

**THE TABLE ABOVE PREDATES THE FUNCTION IT DESCRIBES, and that is corrected
rather than glossed.** It was taken at `8c67da748`; `06ba79d1b` then added rule
(1) to `Conv1dTimeBlock`, so M1's recorded "2 cases / 1 343 assertions" is a
count against a function that no longer exists. **Re-taken at the review-repair
head**, on the shipped function and against the derived geometry gate: one line
removed, `compile_rc` 0, binary `e9e3124ff3c76b85…` against the baseline
`ab93820e4bb17ac4…`, `test_ops_conv1d_general` **2 cases / 1 014 assertions
FAILED**, rc 1, and `test_vocoder1d` **1 case / 1 assertion FAILED**, rc 1 —
the new block-boundary case's `REQUIRE(block % kConv1dPosTile == 0)`. Both
binaries hash back to baseline after restore.

**AND EVERY ONE OF THOSE 1 014 IS AN ALIGNMENT PROPERTY CHECK, WHICH IS THE
POINT.** Read by name: 976 are `CHECK(aligned)` in the tile-boundary sweep and
38 are `CHECK(block % kConv1dPosTile == 0)` in the two shipped-geometry loops.
**Zero arithmetic assertions move** — not the 19 615 that were there before the
repair, not either engineered cancellation case, not `test_host_parallel`, which
stays 11 cases / 968 assertions / `SUCCESS!`, rc 0. That is the measurement
behind §3b's corrected claim: misaligning the block cannot move a bit, and the
alignment is a PERFORMANCE invariant that this gate pins for a performance
reason.

**M2 is recorded as a defect in the INSTRUMENT, not omitted.** The first
attempt at M2 removed the clamp on the last block, which writes past the output
row. That is undefined behaviour rather than a wrong answer: the binary built
(`2eb599bc05546129…`) and the suite had not finished after 201 s, so it was
killed and reads `test_rc=137`. A hang is a detection nobody can grade, so M2b
replaces it with a mutation that computes FEWER cells and writes none out of
range, and that one reds cleanly on the two block-crossing cancellation cases.

### 6c. Reachability, measured rather than argued

A scheduling change is invisible to every arithmetic assertion BY DESIGN, so
two separate instruments carry the reachability claim.

**The op is reached from all four consumers' production paths.** M5b makes the
registered CPU provider refuse every call, and the suites red in exactly the
places that prove it: `test_ops_conv1d_general` 5 cases, `test_host_parallel` 3,
`test_vocoder1d` 3, `test_bigvgan` 3, `test_minimax_music3_acoustic` 6,
`test_ltx2_vae` 5. `test_indextts2_family` and `test_ops_conv1d_depthwise` stay
green, which is correct — neither reaches this op — and is the control that
says the red is a signal and not a broken build. **The list of six/two above is
UNDERSOLD and is corrected here:** the fresh reviewer ran all ten suites under
the same mutation and got **eight red, two green**, which is a stronger result
than the row claimed, not a weaker one.

**The second axis is entered at the SHIPPED geometries, and the gate now READS
those geometries.** It walks `MiniMaxMusic3VocoderConfig` and the residual-stack
constants `kVocoderResidualDilations` / `kVocoderResidualUnits`, reproducing the
recurrence in `minimax_music3_acoustic.cpp` `VocoderDecode` / `VocoderBlock` /
`VocoderResidualUnit`, and evaluates `Conv1dTimeBlock` at every convolution the
chain makes: at a 344-latent window 22 shapes are taken and 5 declined, and the
case asserts `blocks > 1`, the tile alignment and the slice budget on every taken
one and `block == length` on every declined one. It runs the same walk at a
20-latent window and asserts strictly more shapes are declined, which is the rule
flipping with the window rather than with a list of names.

**The previous revision transcribed six shapes by hand from
`minimax_music3_loader.h:253-265` and could not have noticed the source moving,
and that is measured rather than asserted.** Mutating the loader's
`upsampling_ratios` to `{1, 1, 1, 1}` — a checkpoint geometry that collapses the
production shapes back to one block — reds the DERIVED gate at 1 case / 16
assertions, rc 1, and leaves the hand-transcribed one at **14 cases / 19 615
assertions / `SUCCESS!`, rc 0**: completely blind, on the exact drift the case
exists to catch. `compile_rc` 0 on both legs, binaries `3ec7121712179109…`
(derived) and `169faecba6c898d6…` (transcribed), and the tree restored to
`ab93820e4bb17ac4…` byte for byte.

**AND A CONSUMER NOW ENTERS THE BLOCKED PATH ARITHMETICALLY.** Mutation **M7b**
— sign-flip every output cell where `blocks > 1`, exactly and only the axis this
row adds — was run over all ten suites and left **eight green** at the row's
head: only `test_ops_conv1d_general` and `test_host_parallel` reddened. Every
audio consumer reached `vt::Conv1d` at SINGLE-BLOCK shapes. That is repaired on
the shared-core side by `tests/vllm/models/test_vocoder1d.cpp`'s `vocoder1d
Conv1d is exact ACROSS a time block boundary`, which enters through
`vllm::vocoder1d::Conv1d` — the body all four audio models call — at 32
channels, kernel 7, 10 000 positions, asserts the block length is SHORTER than
the output length so the case cannot silently become single-block, and reads
`out[oc][t] == 7t + 21 + oc` exactly, every partial sum being an integer below
2^24. Re-run at the repaired head, M7b reds it: `compile_rc` 0, one line, binary
`622ad67775df5e6c…`, **1 case / 4 assertions FAILED, rc 1**. The four MODEL
suites still reach the provider at single-block shapes only and M7b still leaves
them green — recorded as owed
([#1684](https://github.com/mudler/vllm.cpp/issues/1684), §7).

**And the snake's dispatch is mutated directly.** M4 replaces the
`ForOutputRows` call with one that runs the body over an empty range; the gate
reds. A green there would have meant the case was measuring a class.

## 7. Owed

Named here rather than left to a profile, because each is a real gap this row
declines to close.

- **The two b0 losses the blocking condition was derived from do not
  reproduce, and the condition measures NEUTRAL**
  ([#1770](https://github.com/mudler/vllm.cpp/issues/1770)). §2b priced arm C at
  0.82x and 0.89x on `b0_res_conv1` and `b0_res_conv2`, and §3b's condition
  exists to decline exactly those two shapes. Job `16b594ec` on a second boot
  keeps the first direction at a quarter of the size (1.065x) and REVERSES the
  second (0.791x), and over the two shapes together the arms tie. Three
  instruments in that job — the 20-latent thread sweep, the 86-latent paired
  window and the op probe — all read the condition as worth nothing on this box.
  Nothing is broken and the shipped arm is correct; what is unsupported is that
  the condition buys anything. Settling it needs a per-geometry spread rather
  than a median of three rounds, at both lengths, which is a fresh lease.
- **This job's CPU-clock sampling is BETWEEN legs, not during them.** §2a
  sampled `scaling_cur_freq` every 2 s inside each leg and killed candidate 5 on
  that evidence. Job `16b594ec` samples around each thread-count block instead,
  so most of its readings are the idle clock and only one sample caught
  2 601 000 kHz. Candidate 5 is inherited from §2a rather than re-measured, and
  the §2b sweep therefore carries no per-leg clock attribution of its own.
- **No per-MODEL suite exercises a `blocks > 1` shape**
  ([#1684](https://github.com/mudler/vllm.cpp/issues/1684)). §6c carries the
  measurement and the partial repair: the shared vocoder core now has an
  arithmetic case that crosses a block boundary, and `test_bigvgan`,
  `test_minimax_music3_acoustic`, `test_ltx2_vae`, `test_minimax_h3` and both
  IndexTTS-2.5 suites still reach the provider at single-block shapes only, so
  M7b leaves them green. Closing it means lengthening each consumer's
  reduced-dimension fixture until its convolutions cross a boundary, which moves
  those fixtures' goldens — a fixture change per model, not a test addition, and
  not something to fold into a review repair.
- **The CUDA provider is not re-measured against either change.** The authoring
  host has no CUDA toolkit, so `test_ops_conv1d_general`'s four CPU-vs-CUDA arms
  printed `[SKIP]` and the `thor:gpu0` job was built CPU-only. Neither change
  touches `cuda_conv1d_general.cu`, and the `memcmp` argument is §18.3's — the
  per-cell order is unchanged on the host — but an argument is not a
  measurement. It needs a CUDA build, which §20.2 records as reachable inside a
  lease (`apt` installs `cuda-nvcc-13-0`).
- **`vt::ConvTranspose1d` keeps the row-only partition.** It was 8.17 % of the
  window before this row and is a larger share after it, and its scatter form
  makes a time block's output overlap its neighbour's by
  `dilation * (kernel - 1)`, so the per-cell order would have to be argued
  rather than inherited. Deliberately a different row.
- **The snake returns about 7x, not 14x**, on the authoring host. Why it stops
  there is not measured: candidates are the `std::sin` call's own throughput,
  the pool's chunk grid at 96 channels, and the pass being a pure stream over an
  activation that does not fit any cache here. Not chased, because after this
  row it is no longer the largest term.
- **`vocoder.pad` and `vocoder.copy` are unpartitioned**, at 0.89 % and 0.10 %
  on the AUTHORING HOST at 20 latents (§2a) and at 0.59 % and 0.34 % on
  `thor:gpu0` at 86 latents and 14 threads (§2b). Both readings are real and
  neither is quotable without its host and its window length, which is why they
  are written with them here. They are named so that the next reader does not
  re-derive that they are small.
- **The block byte budget was not swept per geometry ON `thor:gpu0`.** The
  residency instrument sweeps four footprints at each geometry and
  `kConv1dSliceBytes` was chosen at half of the measured 1 MiB private L2 rather
  than at a per-geometry optimum. A finer sweep may move it; the constant is
  derived from a stated budget, so moving it is a one-line change with a shipped
  instrument behind it.
- **The snake's arithmetic is untouched and unexamined.** It evaluates a
  `double` `std::sin` per element. Whether upstream's `Snake1d` needs that width
  is a numerics question an oracle owns, and narrowing it would re-gate four
  models — exactly the shape #1474 was.

## 8. Stop conditions

Stop and report rather than widen scope if: the ablation says the limit is
candidate 4 or 5, which this row cannot act on; a decomposition change cannot
be made without moving a per-cell summation order; the blast radius exceeds
`vt::Conv1d`'s CPU provider; or `thor:gpu0` is unavailable, in which case no
ratio is quoted at all rather than quoted from another box.

## 9. Outcome

**What was measured, and on which arm.** The window's scaling on `thor:gpu0`
goes from **2.85x of 14 threads to 11.54x ON THE ARM THAT SHIPS**, swept at 1,
2, 4, 8 and 14 threads with the arms alternated at every count, in job
`16b594ec` (§2b). The unconditional arm C, re-measured in the same job, reads
11.48x — the value §2b recorded for it on the earlier boot — so the headline
survives the correction that produced it. Against the instrumented baseline the
shipped arm is **4.08x**, paired: both arms in one job, alternated over seven
rounds after a settle whose load is printed every 60 s. That replaces the 4.11x
composed across two jobs, and the two differ by 0.7 %. Against arm B, paired
and alternated seven times with a 300 s settle, arm D is **1.067x** (§2c). Every
arm is bit-identical to every other at full scale, in both jobs: two
fingerprints in the whole of `16b594ec`, one per length, across three arms, five
thread counts and every round.

**What was rejected, and why.**

- **The premise of the row.** §18.8b's shared-bandwidth candidate is refuted:
  `vt::Conv1d` already scaled **12.44x of 14** before this row touched it, and
  the residency ablation reads 0.98-1.31x rather than the several-fold factor a
  bandwidth story needs. The window's problem was Amdahl's law over an
  activation function with no partition, which no instrument in that section
  could see because it timed the window and reasoned about the kernel with
  nothing in between.
- **Blocking the convolution unconditionally.** Measured **0.82x and 0.89x** on
  the two b0 shapes, where the weight tensor is 16.5 MiB against a 2.1 MiB
  activation. Shipped conditionally instead. **That rejection is now weaker than
  the sentence above it.** On a second boot the first shape reads 1.065x and the
  second REVERSES to 0.791x, the two together tie, and the conditioned arm
  measures within 1.2 % of the unconditional one at every thread count at 20
  latents (§2b). The condition costs nothing and is not shown to buy anything on
  this box ([#1770](https://github.com/mudler/vllm.cpp/issues/1770), §7).
- **A new pooled primitive for the decomposition.** Rejected on blast radius: it
  would be inherited by eleven other call sites whose right blocking factor is
  not a property they share (§4).
- **`vt::ConvTranspose1d`**, and **narrowing the snake's `double`**. Both are
  out of scope with a reason, not overlooked (§0, §3a).

**Why each default has its value.**

- `kConv1dSliceBytes` is **512 KiB**: half of `thor:gpu0`'s measured 1 MiB
  PRIVATE L2, because the weight rows of the output channels in flight stream
  through the same L2 and a slice sized to fill it evicts itself. The box has no
  shared last-level cache at all, which is why the budget is a per-core one.
- The block is a **multiple of `kConv1dPosTile`** so the position tiles land
  where they land today, which is what keeps the code path — not only the
  arithmetic — unchanged.
- The blocking condition carries **no constant**: it is `weights <= activation`
  with the common `in_per_group` divided out, and it flips with the window
  length rather than naming shapes.
