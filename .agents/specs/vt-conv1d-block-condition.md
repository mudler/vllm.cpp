# VT-CONV1D-BLOCK-CONDITION — does the blocking condition buy anything, measured against its own noise floor

Row `VT-CONV1D-BLOCK-CONDITION`. Issue
[#1770](https://github.com/mudler/vllm.cpp/issues/1770). Spun out of
[`vt-conv1d-time-block.md`](vt-conv1d-time-block.md) `## Owed`, which owns the
condition and records the reading this row is about.

**The index link for #1770 already exists** and names `VT-CONV1D-TIME-BLOCK` as
its owner. `.agents/issue-index.md` is append-only and
`scripts/check-agent-record.py` refuses a second row for an issue that already
has one, so this row appends nothing to the index. That is the same shape
`VT-CONV1D-MODEL-BLOCK` used for #1684 (`161d808ce`).

## Now

`DONE` pending review. The design and the decision rule in §2c were committed
BEFORE the measurement (`252225d9e`), which is the whole point of this row: the
reading it re-takes was turned into a shipped conditional after the fact, and a
rule written after a number cannot be falsified by it.

**The pre-registered rule returned "the condition GOES", and it is gone.** §8
carries the measurement, the noise floor it is measured against, and what each
of the three lengths says. The verdict is not that the condition was harmful.
It is that over 31 paired rounds at three window lengths, on the shapes it
decides, it is worth **1.0086x, 0.9721x and 0.9700x** — every one of them inside
the distribution the same instrument produces from a difference of NOTHING.

## 0. Scope

**In scope.** One `if` — rule (1) of `vt::cpu::Conv1dTimeBlock`
(`src/vt/cpu/cpu_conv1d_block.h`), `out_channels * kernel > in_len` returns
`length`, which declines the time-block decomposition for the shapes where the
weight tensor is larger than the activation. Whether it stays, goes, or is
re-derived, and the measurement that decides.

**Out of scope.**

- Rule (2), the multiple-of-`kConv1dPosTile` alignment. It is a performance
  invariant with its own property gate and nothing here touches it.
- `kConv1dSliceBytes`. The 512 KiB budget is a separate owed item of
  `vt-conv1d-time-block.md` and moving it would confound this measurement.
- The arithmetic. Nothing here may move a bit; §3 is why, §4 is how it is
  gated.
- `vt::ConvTranspose1d`, the snake, and the CUDA provider. Each is owed
  elsewhere.

## 1. The gap

`vt-conv1d-time-block.md` §9 records **"blocking the convolution
unconditionally"** as REJECTED, on §2b's reading of the unconditional arm C
against the pre-row baseline arm A at 86 latents: `b0_res_conv1` **0.82x** and
`b0_res_conv2` **0.89x**. §3b's condition exists to decline exactly those two
shapes and the header of `cpu_conv1d_block.h` quotes both numbers as its
justification.

Job `16b594ec` re-took them on a second boot, same instrument, same length,
three alternated rounds, medians: `b0_res_conv1` **1.065x** — same direction, a
quarter of the size — and `b0_res_conv2` **0.791x**, REVERSED. Over the two
shapes together the arms tie, 0.04970 s against 0.04958 s. Three instruments in
that job read the condition as neutral, and the window arithmetic bounds its
whole effect at 86 latents at **0.36 ms against a 3.5 s window**.

**So a shipped conditional rests on a measurement that did not reproduce.** A
conditional fast path is a correctness surface — the fresh review of #1678
named it as where a silent divergence would hide — and one that buys nothing
measurable is a liability. Equally, a real loss that reproduces is a reason to
keep it, and the row that ships it deserves numbers that hold.

**Why the earlier readings could not settle this, structurally.** Both were
medians of three rounds of a best-of-2, reported as a single ratio per
geometry. A median of three has no spread, so neither job could say whether a
6 % reading was an effect or an artefact, and **neither job measured what the
instrument reports when there is no difference at all.** That missing quantity
is the whole of this row.

## 2. Design

### 2a. Three arms, and the third one is the point

Three source trees, three build directories, built inside one lease from one
base SHA, differing in one `if` and in nothing else:

| arm | tree | what it is |
|---|---|---|
| **D** | `origin/main`, UNMODIFIED | the shipped tree: rule (1) present |
| **C** | D with rule (1)'s two lines deleted | blocked UNCONDITIONALLY — the candidate change |
| **D2** | byte-identical source to D, second build directory | **the A/A control: the instrument's own noise floor** |

Arm C is constructed from `main` by deleting the rule rather than by checking
out `cf9296496`'s files, because the question is what the change I would
actually make is worth, not what a five-commit-old tree measured.

**D2 is the arm both earlier jobs lacked.** It differs from D in nothing that
can affect a time, so every `D2/D` ratio the probe reports is the instrument
measuring zero. Run with the same rounds, the same alternation and the same
statistic as `C/D`, it gives each geometry a NULL DISTRIBUTION on this box, on
this boot, in this job. `C/D` then has something to be large against. A hash
guard proves three build directories (#1516); this proves what a difference of
nothing looks like.

If D and D2 hash identically the control is stronger, not weaker — it is then
the same bytes run twice — and the job records which happened rather than
requiring either.

### 2b. Rounds, alternation and the statistic

The op probe `vllm_conv1d_scaling_probe` at 14 threads,
`--no-control --no-dispatch --repeats=2`, one invocation per (round, arm).

| length | rounds per arm | why |
|---|---|---|
| 86 | **31** | the disputed regime: the rule declines BOTH b0 shapes here |
| 20 | **31** | the second length #1770 asks for |
| 344 | **15** | the production window length of `vt-conv1d-time-block.md` §1, where the rule declines only `b0_res_conv1` (`b0_res_conv2` needs `in_len >= 768`, so it is TAKEN from 96 latents up) |

The three arms alternate within each round and the order reverses on even
rounds (`C D D2`, then `D2 D C`). A round is one invocation of each arm
back-to-back, so a drift over the job affects all three arms of a round almost
equally, which is what makes a PAIRED ratio the right quantity.

**The statistic is the paired per-round ratio, and the spread is reported
rather than collapsed.** For each geometry and length: the 31 paired `C/D`
ratios, their median, their [min, max] and their interquartile range; and the
same 31 paired `D2/D` ratios. Medians rather than best-of, because the arms may
have different spreads (`vt-conv1d-time-block.md` §2c's reason). The raw legs go
in the committed log so the statistic can be recomputed.

### 2c. THE DECISION RULE, PRE-REGISTERED

Written here, before the job runs, so that the measurement can refuse the answer
this row would find convenient. Let `M` be the median of the paired `C/D`
ratios for a geometry and `[lo, hi]` the full range of the paired `D2/D` ratios
for the SAME geometry, at the same length.

1. **A loss is REAL for a geometry** when `M > hi` — arm C's central reading is
   outside everything the instrument produced from a difference of nothing.
2. **A loss is NOT ESTABLISHED** when `M` lies inside `[lo, hi]`. The effect is
   at or below this instrument's noise floor and no shape-level rule may be
   derived from it.
3. **The condition GOES** when no geometry the rule declines has a real loss,
   AND the sum over the declined geometries is not a real loss by the same test
   applied to the summed walls. Removing it then deletes a branch, a predicate,
   a documented exception, and a divergence surface, and buys nothing lost.
4. **The condition STAYS** when any declined geometry has a real loss. §9 of
   `vt-conv1d-time-block.md` is then re-derived with these numbers in place of
   the ones that did not hold, and the header of `cpu_conv1d_block.h` is
   corrected to quote the reproducible reading.
5. **`NEEDS_DECISION`** when the three lengths disagree in DIRECTION on the same
   geometry under rule 1 — a real loss at one length and a real gain at another
   — because neither answer is then defensible on measurement, and the row
   reports what would settle it and what it would cost.

Rule 3 is deliberately not "C is faster". Neutral is enough to remove, because
the condition's cost is not a time: it is a second code path through a kernel
four audio models depend on.

**The window is a check, not the decision.** `vllm_music3_vocoder_conv_ab` at 86
latents and 14 threads, 11 alternated rounds over C, D and D2, is run for one
reason: job `16b594ec` read a 3 % C-against-D window gap it could not attribute
and arm D's seven legs were bimodal. D2 gives that bimodality a control. **No
op-probe ratio is ever multiplied onto a window bucket** — `vt-conv1d-time-block.md`
§2 records a 1.80x disagreement between exactly these two instruments.

### 2d. If the condition goes, what changes

`Conv1dTimeBlock` loses its first two lines. `kConv1dSliceBytes`, the tile
alignment, `Conv1dKernel` and every caller are untouched. The header's rule (1)
paragraph is replaced by the measurement that removed it, and
`vt-conv1d-time-block.md` §9's REJECTED verdict is corrected in the same change,
because it rests on evidence that did not hold.

`tests/vt/test_ops_conv1d_general.cpp` reads `Conv1dTimeBlock` directly at
`:791`, `:942`, `:985`, `:1008` and `:1054`. Any case that asserts a declined
shape returns `length` is asserting rule (1) and moves with it, RED-FIRST: the
case is changed to the new expectation and shown red on the unmodified kernel
before the kernel changes.

## 3. Risks

- **A bit moves.** The strongest reason to think it cannot: the blocked and
  unblocked paths ALREADY produce identical bits, on every shape, today. A
  cell's reduction is `seed`, then `ic` ascending, then `k` ascending, and that
  sequence does not mention `t0`; #1678's review proved it by mutation, where
  misaligning the block left all 19 615 arithmetic assertions green. Removing
  the condition moves shapes from one bit-identical path to the other. It should
  be free. **§4 proves it rather than assuming it**, because "should be free" is
  the sentence this repository has been wrong behind before.
- **More shapes take the blocked path, so the blocked-path gates become more
  load-bearing.** Four per-model suites now enter that axis (`c98ffd4d0`,
  row `VT-CONV1D-MODEL-BLOCK`) plus the op suite and `test_host_parallel`. All
  of them run, on the arm that would ship.
- **The measurement is taken on ONE box.** `thor:gpu0`, sm_110 aarch64, 14
  cores, 64 KiB L1d and 1 MiB PRIVATE L2 with no shared last level. Every
  ratio here is a ratio on that box and is written with it. The condition is a
  cache-budget argument, so a machine with a shared L3 could read differently;
  that is recorded as owed rather than glossed.
- **A third reading that lands between the first two settles nothing.** Which is
  why the null distribution is measured in the same job. Without it, this row
  would be the third reading.
- **The box's own floor is not idle.** `16b594ec` read `load1` 2.83 BEFORE it
  built anything and could not settle below 2.82 in 900 s. The settle gate is
  therefore written against a floor MEASURED at the top of this job rather than
  against a constant, and it prints what it waited for.

## 4. Gates and evidence

**Correctness before any speed number is read**, on the arm that would ship, and
`test cases:` / `assertions:` / `Status:` quoted in full for each — a bare
`assertions: 0` is a skip wearing a pass:

`test_ops_conv1d_general`, `test_host_parallel`, `test_vocoder1d`,
`test_bigvgan`, `test_minimax_music3_acoustic`, `test_ltx2_vae`,
`test_minimax_h3`. `test_ops_conv1d_general` prints four CUDA `[SKIP]` lines on
a CPU-only build; they are the build, not a result.

**Bit-identity, at the shapes the change moves.** The probe prints an FNV-1a
fingerprint over the raw output bytes of every geometry. Across arms C, D and D2
and every round and length, each geometry must print ONE value. The four shapes
the rule decides differently on — `dec_in_proj`, `conv_in`, `b0_res_conv1`,
`b0_res_conv2` — are precisely the ones where C and D run different code, so
their fingerprints agreeing IS the bit-identity proof for this change. The
window binary's waveform fingerprint carries the same claim end to end.

**The behavioural control, because a hash is not the proof (#1516).**
`conv_out`'s `user/wall` separates an arm that carries the second axis from one
that does not — 1.00 inline against 13-14 pooled. `chunks` is NOT a control: the
probe derives it from `out_channels` and the thread count, so it is identical on
all arms by construction.

**One boot.** The boot id is read at the top and at the bottom of the job and
compared. Every ratio is inside one boot; #543 puts the cross-boot difference at
up to 12.79 % and `clocks.sm` reads `[N/A]` here, so nothing on the box detects
one.

**The log is committed** to `docs/bench-evidence/`, as `16b594ec` was.

## 5. Reachability

`Conv1dTimeBlock` is called from `Conv1dKernel`
(`src/vt/cpu/cpu_conv1d_general.cpp:162`), which is the registered CPU provider
for `vt::Conv1d` and is reached by every audio model's decoder through its own
entry point — the four suites `c98ffd4d0` added enter it that way. This row
changes the value that call returns for four shapes; it adds no surface and
needs no new wiring.

**The mutation that proves the gates see it is M7b, and the fresh review re-ran
it at THIS head rather than inheriting `vt-conv1d-model-block.md` §4b's result
at the old one.** `on[t0 + i] = blocks > 1 ? -acc[i] : acc[i]`, one hunk,
compile rc 0: **all seven suites red** — `test_ops_conv1d_general` 2 cases,
`test_host_parallel` 1 (5 assertions), `test_vocoder1d` 1 (4),
`test_bigvgan` 1, `test_minimax_music3_acoustic` 1 (4), `test_ltx2_vae` 1,
`test_minimax_h3` 1 — which matches §4b's "seven red after" on the tree that
ships.

**Deleting the production call site alone reds NOTHING, and that is expected
rather than a defect.** `block_len = length` in `Conv1dKernel` was mutated with
compile rc 0 and left all seven suites green, because the blocked and unblocked
paths are bit-identical by construction, so no arithmetic assertion can separate
them — that is the whole reason §4 of `vt-conv1d-time-block.md` gates the
FUNCTION's answer as a property. M7b is the instrument that can see the axis,
because it changes what the blocked path computes rather than whether it is
taken.

## 6. Owed

- The condition is measured on `thor:gpu0` only. A box with a shared
  last-level cache is a different cache-budget argument.
- `kConv1dSliceBytes` is now the ONLY thing standing between a large-weight
  shape and a blocked path, and it was never swept per geometry on this box
  (`vt-conv1d-time-block.md` §7).
- **The block-length gate cannot see a block one tile too LARGE, and that is a
  named trade-off rather than an oversight.** The fresh review of the repair
  mutated `block = budget / row_bytes` to `... + 1`: compile rc 0, the answer
  changes on 89 of 1125 swept shapes, and `test_ops_conv1d_general` stayed
  GREEN. The reason is structural. `Conv1dTimeBlock` is conservative by one row
  — it charges `block` rows of activation where the slice actually spans
  `block - 1` — so a block one tile larger can still fit the TRUE span, and the
  upper bound the gate asserts is about the true span. Pinning the function's own
  `block <= budget / row_bytes` would make the test recompute the implementation,
  which is the tautology this repository has been bitten by before. What does
  catch it today is `test_minimax_music3_acoustic.cpp` `REQUIRE(block2 < kLength)`,
  because at that geometry the larger block turns a multi-block shape
  single-block. Recorded so the next reader knows the gate's edge rather than
  discovering it.
- **1 435 of 2 560 multi-block swept shapes sit at `block == kConv1dPosTile`**,
  where the floor clamp may have fired, so the UPPER bound stays guarded and does
  not reach them. The maximality bound is unguarded and reaches all 2 560. Of the
  1 435, the review derived that 295 are genuine maxima where the upper bound
  would also hold; closing that would need the test to distinguish the floor arm
  from a genuine answer, which is the same tautology trade-off as above.
- **No `.agents/benchmark-record.md` entry was written for job `b0fc900b`,
  deliberately.** Two earlier rows in this lane wrote one, so the silence needs a
  reason rather than a habit. That file is a 2.2 MB append-only ARCHIVE that
  `scripts/roll-benchmark-record.py` fills by rolling narrative out of
  `docs/BENCHMARKS.md`; it is not a per-row obligation, and AGENTS.md's `Records`
  section is explicit that a surface every pull request must write is a lock.
  This row's per-row surface is this spec, its two job logs are committed under
  `docs/bench-evidence/`, and `docs/benchmarks/open-gaps.md` carries the closure.
  Recorded as a decision so the next reader does not read it as an omission.
- **Three window lengths, not a sweep.** 20, 86 and 344 latents span the regime
  the condition operates in — it declines five shapes at 20, four at 86 and
  three at 344 — but they are three points and not a curve. A length between
  them is interpolated rather than measured.
- ~~**The probe's own `--control` residency sweep**~~ — RUN, in job `5a11acdd`,
  and §8.3b carries it. What it does not cover is a piece length equal to the
  kernel's own block at each geometry; it sweeps 128 / 512 / 2048 / 8192 and the
  kernel picks 160 at the b0 shapes.

## 7. Stop conditions

Stop and report rather than widen scope if: `thor:gpu0` is unavailable, in which
case NO ratio is quoted from another box; a bit moves anywhere; the D2 control
shows a spread so wide that no reading at any length can clear it, which is a
result about the instrument and is reported as one; or the pre-registered rule
returns `NEEDS_DECISION`, which is an acceptable outcome and is reported as the
answer rather than resolved by preference.

## 8. Outcome

### The job

`rc` job **`b0fc900b-2ce3-49ca-ac24-df4788e46fc8`** on `thor:gpu0`, worker
`rc-worker-kk96r`, `Linux 6.8.12-1021-tegra` aarch64, 14 cores, boot id
**`e2112cac-660b-434e-911d-33cbd29b9176` read before and after, `ONE BOOT: OK`**,
`--max-runtime 170m`, 23 minutes wall, 2026-08-23. Governor `schedutil`,
`scaling_max_freq` 2 601 000 kHz. Release `-O3`, CPU-only, three trees built
inside the lease from `origin/main` at `ea9b7e30e`. The whole log is committed at
[`docs/bench-evidence/vt-conv1d-block-condition-1770-thor-20260823.log`](../../docs/bench-evidence/vt-conv1d-block-condition-1770-thor-20260823.log).

**This is the SAME worker and the SAME boot as job `16b594ec`**, and that is
stated rather than glossed: this row confirms `16b594ec` and it cannot by itself
adjudicate the boot `fabedc13` that `vt-conv1d-time-block.md` §2b ran on. What it
can do — and what §8.3 does — is measure how wide a reading this instrument
produces from no difference at all, and §2b's two numbers are inside it.

**The settle worked because it was measured rather than assumed.** The floor read
`3.69` before anything was built; the gate was set at floor + 0.5 = `4.19`; the
builds took the load to `9.30` and it decayed `5.87`, `4.28`, `3.99` and
`SETTLE_REACHED` in 180 s. `16b594ec`'s constant `1.5` was below this box's own
floor and burned its 900 s ceiling for nothing.

### 8.1 Correctness, and the two arms behaving as two arms

**Bit-identity: PASS.** Across arms C, D and D2, all **231** probe invocations — 77 rounds
times three arms — and three lengths, each of the eleven geometries printed
exactly ONE fingerprint — **33 of
33 (length, geometry) cells, one value each**. The window binary printed
`0xc2d5eaf095d1c483` on all 33 of its legs, which is the same value
`16b594ec` recorded at 86 latents. Removing the condition moves four shapes from
one bit-identical path to the other and moves no bit.

**The suites, on arm C — the arm that would ship — `test cases:` /
`assertions:` / `Status:` in full, against arm D as the positive control:**

| suite | arm C | arm D |
|---|---|---|
| `test_ops_conv1d_general` | 14 / **19696, 9 FAILED** / `FAILURE!` | 14 / 19696 / `SUCCESS!` |
| `test_host_parallel` | 11 / 968 / `SUCCESS!` | 11 / 968 / `SUCCESS!` |
| `test_vocoder1d` | 11 / 66 / `SUCCESS!` | 11 / 66 / `SUCCESS!` |
| `test_bigvgan` | 7 / 85 / `SUCCESS!` | 7 / 85 / `SUCCESS!` |
| `test_minimax_music3_acoustic` | 40 / 395 / `SUCCESS!` | 40 / 395 / `SUCCESS!` |
| `test_ltx2_vae` | 45 / 3152 / `SUCCESS!` | 45 / 3152 / `SUCCESS!` |
| `test_minimax_h3` | 80 / 57416 / `SUCCESS!` | 80 / 57416 / `SUCCESS!` |

`test_ops_conv1d_general` also printed its four CUDA `[SKIP]` lines, which are
the CPU-only build and not a result.

**THE NINE FAILURES ARE THE RED-FIRST EVIDENCE, and they are all one assertion.**
Every one is `CHECK(block == sh.length)` in the DECLINED branch of "the shipped
vocoder geometries are MULTI-BLOCK" — four at 344 latents (`conv_in`, and
`b0_*_conv1` at each of the three dilations, block 96/160/128/96 against lengths
344/2752) and five at 20 latents. **Not one arithmetic assertion moved anywhere**,
including LTX-2.5's thirteen audio arms at `5e-6` and Music3's closed form. So
the test that gates rule (1) reds when rule (1) is removed, and nothing else
does: the change is gated, and it is gated in exactly one place.

**And the arms behave as three arms, which no hash can say (#1516).** The
strongest control here is not `conv_out`'s `user/wall` — every arm carries the
blocking axis for `conv_out`, so it reads 12.6-13.9 on all three and cannot
separate them. It is the PATTERN in §8.2: arm C differs from arm D at exactly the
geometries where the rule decides differently (`b0_res_conv2` at 86 latents,
C>1 in 2 of 31 rounds against 16 of 31 in the null) and ties with it at every geometry where the rule
decides the same way — within **1.03 %** on the SIX taken residual shapes, and
**3.39 %** over all seven taken shapes, the seventh being `conv_out`, whose
0.36-2.41 null makes it the noisiest cell in the table. Two build
directories give two hashes; only that pattern gives two kernels.

### 8.2 The measurement, and what settles it

The op probe at 14 threads, `--repeats=2`, arms alternated and the order
reversed on even rounds. Each cell is the median of the PAIRED per-round ratio.
`C/D` is the candidate; **`D2/D` is the same statistic over an arm that differs
from `D` in nothing, and it is the quantity neither earlier job had.**

**86 latents, 31 paired rounds — the disputed regime, where the rule declines
both b0 shapes:**

| geometry | rule | C/D median | C/D IQR | D2/D median | D2/D range | D2/D IQR | rounds C/D > 1 (null) |
|---|---|---|---|---|---|---|---|
| `dec_in_proj` k1 | declined | 1.0000 | [0.833, 1.167] | 1.0000 | [0.667, 1.769] | [0.778, 1.133] | 11/31 (11/31) |
| `conv_in` k7 | declined | 1.0063 | [0.986, 1.101] | 0.9981 | [0.829, 1.116] | [0.972, 1.021] | 17/31 (13/31) |
| `b0_res_conv1` k7 | declined | **1.0698** | [1.0497, 1.0854] | 0.9943 | [0.884, 1.327] | [0.963, 1.014] | **28/31** (12/31) |
| `b0_res_conv2` k1 | declined | **0.8129** | [0.800, 0.839] | 1.0036 | [0.962, 1.378] | [0.994, 1.014] | **2/31** (16/31) |
| the seven TAKEN shapes | taken | 0.990-1.034 | overlapping | 0.986-1.019 | | | 11-18/31 |
| **SUM over the four DECLINED** | | **1.0086** | | 0.9968 | [0.922, 1.194] | | |
| ALL ELEVEN | | 1.0009 | | 0.9957 | | | |

**20 latents, 31 paired rounds** (the rule declines five shapes here, `b1_res_conv1`
joining): every geometry `NOT ESTABLISHED`; `b1_res_conv1` reads **0.9521**,
IQR [0.941, 0.974], disjoint from its null's [0.984, 1.028] — a ~5 % GAIN on a
shape the rule declines. **SUM over the five declined: C/D 0.9721**, null median
1.0104, range [0.832, 1.263].

**344 latents, 15 paired rounds — the production window length**, where
`b0_res_conv2` is TAKEN (it needs `in_len >= 768`) and only three shapes are
declined: `conv_in` reads **0.9636**, IQR [0.945, 0.980], disjoint from its
null's [0.985, 1.030] — a ~3.6 % GAIN the rule declines. `b0_res_conv1` reads
**0.9708**, a gain, having been a 7 % LOSS at 86 latents. **SUM over the three
declined: C/D 0.9700**, null median 1.0020, range [0.952, 1.238].

**Applying §2c to that, clause by clause.**

1. No declined geometry at any length has `C/D median > D2/D max`, so **no real
   loss is established anywhere**.
2. `b0_res_conv2` at 86 latents clears the null in the other direction: a **REAL
   GAIN**, and the rule declines it.
3. The SUM over the declined geometries is inside its own null at all three
   lengths, and its median is on the GAIN side at two of them. **The condition
   GOES.**
5. Rule 5 does not fire: `NEEDS_DECISION` needed a real loss at one length and a
   real gain at another on the same geometry under clause 1, and clause 1
   establishes no loss at all.

**What the stricter, POST-HOC statistic says, reported because it is the honest
reading and because it does not change the answer.** Comparing interquartile
ranges rather than full ranges — a test not pre-registered, and named as such —
four per-shape effects are real and reproducible: `b0_res_conv1` at 86 latents is
a genuine **7 % LOSS** (28 of 31 rounds above 1, against 12 of 31 in the null),
and `b0_res_conv2` at 86 (**19 % gain**), `b1_res_conv1` at 20 (**5 % gain**) and
`conv_in` at 344 (**3.6 % gain**) are genuine gains. **So the rule is right about
exactly one of the shapes it decides, at one of the three lengths, and wrong
about three others.** Its net effect on the shapes it decides is zero at 86 and
about 3 % the wrong way at 20 and 344. A predicate with that record is not
earning a branch through a kernel four audio models depend on.

### 8.3 THE NOISE FLOOR, which is the finding that settles #1770

**Measured, at 86 latents and 14 threads, as the paired `D2/D` distribution over
31 rounds of an arm that differs from the shipped one in nothing at all:**

- Its MEDIAN is well behaved: 0.986 to 1.020 across the eleven geometries, where
  1.000 is the truth.
- Its RANGE is not: **[0.884, 1.327] on `b0_res_conv1`, [0.962, 1.378] on
  `b0_res_conv2`, [0.549, 1.825] on `b3_res_conv2` and [0.363, 2.407] on
  `conv_out`.**
- Its IQR is tight, typically within 1-5 %.

**So a single paired reading from this instrument carries tens of percent of
excursion, and only a quantile over ~30 rounds is stable to a few percent.**

That is the direct answer to why two jobs disagreed. `vt-conv1d-time-block.md`
§2b read arm C at **0.82x and 0.89x** against the baseline on the two b0 shapes,
which is C running 1.22x and 1.12x the baseline's wall. **Both of those numbers
sit inside the null distribution measured here for those same two geometries** —
1.22 against a null reaching 1.327 on `b0_res_conv1`, 1.12 against a null
reaching 1.378 on `b0_res_conv2`. §2b was a median of THREE rounds. A median of
three cannot separate a 20 % effect from nothing on this instrument, and the
shipped condition was derived from exactly that.

`16b594ec`'s readings, by contrast, REPRODUCE: it read **1.065** and **0.791**
where this job reads **1.0698** and **0.8129** at ten times the rounds. Both
jobs are on boot `e2112cac`.

### 8.2b WHAT THE PRE-REGISTERED RULE COULD NOT HAVE DONE, found by the fresh review

**Clause 1 compares a median of 31 paired ratios against the full range of 31
individual null draws, and that is a low-variance statistic against a
high-variance one.** The review re-derived what it would have taken to fire on
`b0_res_conv1` at 86 latents: a `C/D` median above **1.327**, the null's maximum
— a 33 % loss, where the reading this row was auditing was 22 %. **So clause 4,
"the condition STAYS", was structurally unreachable for the effect the rule was
written to adjudicate.** That is a defect in the rule, and it is recorded here
rather than argued away.

**It does not change the verdict, and that is measurable rather than a
judgement.** Three independent things carry it:

- The **post-hoc IQR test in §8.2 finds the 7 % loss** the range test missed, and
  §8.2 reports it as real. The verdict was never resting on that loss being
  absent.
- **The absolute milliseconds go the other way at two lengths out of three.** On
  the shapes the condition decides, the medians are C 66.53 ms against D 65.86 ms
  at 86 latents (**+0.67 ms** for removal), **−1.11 ms** at 20, and **−8.22 ms**
  at 344, the production window length. A predicate that costs 0.67 ms at one
  length and saves 8.22 ms at another is not paying for itself.
- The **end-to-end window** reads `C/D` 0.9976 against a null `D2/D` of 0.9988
  (§8.4), and the **residency control reproduces every direction with no arms at
  all** (§8.3b).

**What the next row should take from this.** Do not re-use a full-range null
test. The null's INTERQUARTILE range is the comparable spread for a median
statistic; its full range is dominated by the few contended legs that a median
is chosen to ignore. Pre-registering the wrong statistic is better than
pre-registering none — the failure is visible here because the rule was written
down first — but it is still the wrong statistic.

### 8.3b THE RESIDENCY CONTROL — the same answer from an instrument with NO ARMS

`rc` job **`5a11acdd-5277-4aa2-90d7-eea970e7ccf8`** on `thor:gpu0`, worker
`rc-worker-kk96r`, **the same boot `e2112cac`**, log committed at
[`docs/bench-evidence/vt-conv1d-block-condition-1770-thor-control-20260823.log`](../../docs/bench-evidence/vt-conv1d-block-condition-1770-thor-control-20260823.log).
[#1770](https://github.com/mudler/vllm.cpp/issues/1770) named this and §8.2 did
not run it, so it was taken afterwards. It reuses arm D's probe binary from
`b0fc900b`, and the job asserts the reuse rather than assuming it: sha256
`3ffdeb30…`, identical to the value `b0fc900b` recorded for arm D.

**Why it is a second instrument rather than a repeat.** `--control` cuts the time
axis of ONE geometry into declared pieces, all output channels of a piece before
the next, on ONE binary. Identical arithmetic, identical code path, identical
position tile; only the activation footprint one unit of work touches changes.
There are no arms, no second build directory and no arm ordering, so nothing it
reports can be an artefact of any of those. It is also biased AGAINST the small
pieces — R pieces cost R pool dispatches and R separate buffers, which the
production arm does not pay — so a win at a short piece is a LOWER bound on what
blocking buys.

`vs_whole` against the whole-length call, median of 9 rounds at 86 latents and 5
at 344:

| length | geometry | rule | piece 128 | piece 512 |
|---|---|---|---|---|
| 86 | `b0_res_conv1` k7 | declined | **0.973** [0.825, 1.022] | 1.009 |
| 86 | `b0_res_conv2` k1 | declined | **1.705** [1.518, 1.793] | 1.027 |
| 344 | `conv_in` k7 | declined | **1.050** [1.011, 1.084] | — |
| 344 | `b0_res_conv1` k7 | declined | **1.057** [1.044, 1.078] | 0.968 |
| 344 | `b0_res_conv2` k1 | taken | 1.225 [1.041, 1.269] | 0.811 |

**Every direction §8.2 measured with arms is reproduced here without any.**
Blocking `b0_res_conv1` at 86 latents is a small loss and blocking it at 344 is a
gain — the length flip, from an instrument that cannot have an arm-ordering bug.
Blocking `conv_in` at 344 is a gain the rule declined. And `b0_res_conv2` at 86
latents, the shape whose reversal opened this issue, is **1.705x** — the rule
declined a shape that time-blocking makes seventy per cent faster, and the
control says so with a spread of [1.518, 1.793] over nine rounds that does not
come near 1.

**So rule (1)'s premise is not merely unsupported, it is contradicted.** The rule
was "block only when the weights are the smaller tensor". At `b0_res_conv2` and
86 latents the weights are 2.25 MiB against a 2.02 MiB activation — weights
larger, so the rule declines — and cutting the time axis there is the largest
single win anywhere in this row's measurements.

**What this instrument does NOT say.** Its piece lengths are 128 and 512 and the
kernel's own block at these geometries is 160, so the rows above are the same
question at a neighbouring footprint rather than the kernel's exact decision. It
is corroboration of a direction, not a second measurement of a magnitude, and no
number here is multiplied onto anything. `dec_in_proj` at 344 reads 1.032 with a
range of [0.974, 1.044] on a 0.00009 s wall, which is a geometry too small to
carry a claim, and it is quoted only so that the table is not filtered.

### 8.4 The window, and the 3 % that #1770 left unattributed

86 latents, 14 threads, 11 alternated rounds over C, D and D2:

| arm | median | range |
|---|---|---|
| C | 3.4781 s | [3.2324, 3.6638] |
| D (shipped) | 3.4852 s | [3.4677, 4.0861] |
| D2 (the null) | 3.4809 s | [3.4625, 3.6712] |

Paired: **C/D 0.9976, D2/D 0.9988** — the arms are within 0.3 %, and the null is
within 0.1 % of the candidate. `16b594ec`'s **3 % C-against-D window gap does not
reproduce**, and its arm-D bimodality is not a property of arm D: D2's legs are
bimodal in the same way and C's spread is wider than either. That closes the
residue #1770 recorded as unexplained variance. **No probe ratio is multiplied
onto this bucket** — §2 of the owning spec records a 1.80x disagreement between
these two instruments.

### 8.5 What was rejected

- **Keeping the condition.** No loss is established at any length, its net is
  zero-to-negative on the shapes it decides, and it costs a branch, a predicate,
  two parameters, a documented exception, and a second code path through the
  provider four audio models reach.
- **A third plain reading.** §1 says why: a number between the first two settles
  nothing. The row spent its rounds on a null distribution instead, and that is
  what produced an answer rather than a third opinion.
- **Sampling `scaling_cur_freq` during each leg.** `vt-conv1d-time-block.md` §7
  owes that against §2a's method. It is not needed for THIS comparison and the
  reason is structural: a clock excursion that is not the code appears in `D2/D`
  too, so the null absorbs it by construction. That is a claim about the
  comparison and not about the absolute numbers, which are not quoted as
  absolutes anywhere here.
- **Shipping arm C byte-for-byte.** The measured arm kept the signature and
  neutered the branch with two `(void)` casts, to keep the arm minimal. What
  ships DELETES the branch and the two parameters that were its only readers, so
  the function's inputs now state exactly what its answer depends on.

  **That gap between what was measured and what ships is closed by exhaustion
  rather than by argument.** Both forms were transcribed verbatim into one
  translation unit and evaluated over the cross product of 13 channel counts, 7
  kernels, 4 strides, 5 dilations, 11 lengths and 5 output-channel counts —
  **100 100 input tuples, 0 disagreements**, at `-Wall -Wextra -Werror`, compile
  rc 0, exit 0. The two forms are the same function. The difference is a
  signature, and the speed measurement transfers.

### 8.6 Why each default has its value now

- `Conv1dTimeBlock` carries **ONE rule**: the largest multiple of
  `kConv1dPosTile` whose activation slice fits `kConv1dSliceBytes`. A shape whose
  whole activation already fits returns `length`, which is one block.
- The **512 KiB budget is unchanged** and is still half of `thor:gpu0`'s measured
  1 MiB private L2. This row did not touch it; §6 records that it is now the only
  thing standing between a large-weight shape and a blocked path.
- The **tile multiple is unchanged** and is still a performance invariant rather
  than the bit-identity argument.
