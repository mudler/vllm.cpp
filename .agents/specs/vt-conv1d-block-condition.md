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

`ACTIVE`. The design and the decision rule below are committed BEFORE the
measurement, which is the whole point of this row: the reading it re-takes was
turned into a shipped conditional after the fact, and a rule written after a
number cannot be falsified by it.

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
needs no new wiring. The mutation that proves the gates see it is M7b, already
in `vt-conv1d-model-block.md` §4b.

## 6. Owed

- The condition is measured on `thor:gpu0` only. A box with a shared
  last-level cache is a different cache-budget argument.
- If the condition goes, `kConv1dSliceBytes` becomes the ONLY thing standing
  between a large-weight shape and a blocked path, and it was never swept per
  geometry on this box (`vt-conv1d-time-block.md` §7).

## 7. Stop conditions

Stop and report rather than widen scope if: `thor:gpu0` is unavailable, in which
case NO ratio is quoted from another box; a bit moves anywhere; the D2 control
shows a spread so wide that no reading at any length can clear it, which is a
result about the instrument and is reported as one; or the pre-registered rule
returns `NEEDS_DECISION`, which is an acceptable outcome and is reported as the
answer rather than resolved by preference.

## 8. Outcome

Written when the row reaches `DONE`.
