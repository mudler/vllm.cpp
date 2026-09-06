# Spec: the device-resident async spec sampler (SPEC-DFLASH2, A2)

Row: `SPEC-DFLASH2`.
Issue: [#2802](https://github.com/mudler/vllm.cpp/issues/2802), which owns wave
A2-2 and is this spec's live issue.
[#2644](https://github.com/mudler/vllm.cpp/issues/2644) owns wave A2-1, which
has landed; it replaced #2116 and still owns that wave's `## Owed` entries.
#2116 and #2108 no longer resolve, authenticated or anonymously; they stay in
the text below as provenance, and nothing should be read from their numbers.
Discharges: `## Owed` **A2** of
[`spec-decode-async-scheduling.md`](spec-decode-async-scheduling.md), the wave
that turned async SCHEDULING on for the Eagle-type family and deliberately left
the runner's input and sampler halves synchronous.

**This spec was committed as the scoping half of the split shape and carried
no product code.** Wave A2-1 then landed on the same branch, so the spec now
ships beside an implementation; `## Now` states the position and
`## What was measured` states which head each measurement belongs to.

The spec exists because its measurement turned the row's premise around: the
veto at the two `src/vllm/v1/worker/gpu/runner.cpp::async_input_combine_`
assignments — GPUModelRunner has TWO constructors and each carries its own —
is not a stale guard that a mirror argument can lift. It is cited by SYMBOL and
not by line because this citation has now gone stale three times: it read
`:472`/`:537`, then `:480`/`:553`, and the lines were 485 and 563 by the time a
review checked. `scripts/check-symbol-anchors.py` gates the symbol form; nothing
can gate a line number. When the spec was written it was load-bearing on two
independent counts, one of them invisible to every token gate in this tree.
A2-1 discharged that invisible count (reason A, the combine's arithmetic) and
built G2 to hold it. **The veto still stands on reason B**, which A2-1 did not
touch and which A2-2 and A2-3 own.

## Scope

In: the four pieces A2 names, staged so each has a gate that can fail.

1. Device-resident rejection sampling on the verify step.
2. The propose loop's drafts staying on device.
3. The draft-aware `combine_sampled_and_draft_tokens` at `num_logits = 1 + k`.
4. The optimistic `prev_num_draft_len` with the deferred correction.

Out: any change to `async_sched_supported_` (W7 already resolved async
scheduling ON for the Eagle-type family and that half is unaffected); the
stochastic rejection path; structured output under spec (`## Owed` A3 of the
W7 spec); the `async_tokens_to_discard` producer (A4).

Explicitly out: **flipping the veto ahead of pieces 1 and 2.** Piece 3 landed
as A2-1 and did not make the flip safe on its own; the post-A2-1 measurement
below is the reason, and it is not a judgement call.

## What was measured

Mutation **M** is the veto itself, deleted at both construction sites verbatim
— the two `async_input_combine_ =` assignments (`git diff --stat` = one file,
2 insertions, 2 deletions):

```diff
-  async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value() &&
+  async_input_combine_ = AsyncRunnerEnvDefault() &&
                          QueueSupportsAsyncInputCombine(queue_);
```

**M has been run on two different heads and it does NOT produce the same
result on both.** A2-1 changed what it proves. Both runs are recorded below,
each against the head that produced it, because the pre-A2-1 run is the
evidence that made this row's premise turn around and the post-A2-1 run is the
evidence that holds on the tree a reader has in front of them. Neither
supersedes the other, and a row from one must never be quoted against the other.

Both runs used a CPU build
(`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF`), target
`test_mtp_depth`. No GPU was taken and no device measurement is claimed
anywhere in this spec.

### M on `a0f12b727`, the PRE-A2-1 head (this spec's own commit)

The head is identified by the md5 this section has always recorded:
`md5sum src/vllm/v1/worker/gpu/runner.cpp` reads
`ac6a4b61cbb9956299154020112651e7`, which is
`git cat-file blob a0f12b727:src/vllm/v1/worker/gpu/runner.cpp` exactly. The
veto sat at `runner.cpp:472` and `:537` on that head. The tree was restored
byte-for-byte after the run and `git status --porcelain` reported no change to
that file. The same mutation was run once before this branch merged, at base
`331eda888`, and produced the same four rows and the same trace difference.

| run | result | exit |
|---|---|---|
| baseline, whole binary | 10 cases, 10 passed, 123 assertions | 0 |
| **M**, whole binary | 10 cases, 9 passed, 1 failed, 118 assertions | 1 |
| baseline, `-tc="*IDENTICAL*"` | 1 passed, 15 assertions | 0 |
| **M**, `-tc="*IDENTICAL*"` | 0 passed, 1 failed, 10 assertions | 1 |

On that head **M** refused at `runner.cpp:1833`, the async draft fill:

```
vt: async draft fill: no drafts proposed for request 'req'
(placeholders scheduled without a matching propose)  runner.cpp:1833
```

and, separately and silently, corrupted every verify block — the reason-A table
in the subsection below.

### M on `99d223421`, the A2-1 landing head (the head this spec ships with)

Re-run for this landing, because the section previously claimed the rows above
were taken here and they were not. `md5sum src/vllm/v1/worker/gpu/runner.cpp`
reads `d97ab753448957cc34d21454f9fa6b75` before the mutation and again after the
restore, and `git status --porcelain` printed nothing. The rebuild recompiled
`runner.cpp` on both the mutation and the restore (ninja reported 3 steps, not
`no work to do`), so neither number came off a stale object. The veto sat at
`runner.cpp:472` and `:537` on this head as well; the record repair that
followed this run moved them to `:480` and `:553`.

| run | result | exit |
|---|---|---|
| baseline, whole binary | 10 cases, 10 passed, 123 assertions | 0 |
| **M**, whole binary | 10 cases, **5 passed, 5 failed, 29 assertions** | 1 |
| **M**, `-tc="*IDENTICAL*"` | 0 passed, 1 failed, 1 assertion | 1 |

**M** now refuses somewhere else entirely, at a check A2-1 itself added:

```
vt: async input combine: this step scheduled draft tokens, but the draft
buffer the combine scatters from is not wired yet (SPEC-DFLASH2 A2-3, #2644)
  runner.cpp:2384
```

All five failing cases throw that one exception; `runner.cpp:1833` does not
appear in the output at all. The `VT_CHECK(step.num_draft_tokens == 0, ...)`
sits ahead of all three combine arms, so it fires before the combine can run.

**Reason A's corruption is therefore no longer reachable through the combine.**
Under `VT_SPEC_TRACE=1` the baseline emits 201 `draft=` lines and **M** emits
zero: every speculative case refuses before a verify block is ever traced, so
there is nothing to compare and nothing to corrupt. The reason-A table below
cannot be reproduced on this head, and that is A2-1 working, not evidence going
missing.

What survives is that **M is still red, and still red for reason B.** The
refusal it now hits is the one guarding the unwired draft buffer, which is
A2-3's half of reason B rather than A2-2's; the sampler still has no verify arm
either, and that half simply no longer gets the chance to refuse first.

### The CUDA arms' null `cu_num_logits`, measured on the A2-3 REPAIR head

A fresh review returned FAIL on A2-3 because both CUDA arms passed
`/*cu_num_logits=*/nullptr` where the host arm passed `step.cu_num_logits`. The
kernel reads a null as arange, so `num_logits` is 1, `num_draft_tokens` is 0 and
`logits_start` is `query_end - 1` — the last draft slot on a verify step. This is
the mutation that made that executable rather than argued.

**The instrument is the HOST arm, because it is the only one this box builds**
(no `nvcc`). Passing an arange `cu_num_logits` to it is byte-for-byte the null's
kernel meaning, stated by `cuda_combine_tokens.cu:69-79` itself. So the mutation
is: delete the veto at both construction sites (which is **M**, and gives the
combine a real verify step with a correctly sized buffer), then swap the host
arm's `step.cu_num_logits` for `arange(num_reqs + 1)`. Both variants were built
and run at the repair head; `runner.cpp` was restored between them and its md5
`8267057a5cb0b28b4101551e349d290c` read the same before and after.

| run | cases | assertions | exit | `[SPECTRACE]` lines |
|---|---|---|---|---|
| **M** alone (the repaired shape) | 9 of 10 passed | 118 passed, 0 failed | 1 | 192 |
| **M** + arange (the pre-repair shape) | 9 of 10 passed | 118 passed, 0 failed | 1 | 185 |

**The suite cannot tell them apart, and the traces can.** Under the pre-repair
shape each verify block's last draft slot holds the PREVIOUS step's emitted
token, with the `emit=` column unchanged over the same rows:

| position | repaired `draft=[...]` | pre-repair shape | previous step's `emit` |
|---|---|---|---|
| 4 | `6` | `5` | `5` |
| 5 | `11` | `3` | `3` |
| 6 | `7` | `9` | `9` |
| 7 | `5` | `18` | `18` |

which is the reason-A table below, position for position, twelve waves later and
on a second call site. The `k=3` blocks show it in the last slot only:
`[ 6 18 5 ]` becomes `[ 6 18 22 ]`, `[ 10 3 5 ]` becomes `[ 10 3 12 ]`, and 12 is
the previous step's emit. Nothing raises and both arms pass 118 of 118 identity
assertions.

**"Only ACCEPTANCE moves" was measured over THOSE ROWS and it is false over the
whole run.** A round-3 review took the flattened streams and they differ: at
`k=1`, positions 9 and 11, the pre-repair shape reads `ns=2 acc=1 draft=[ 1 ]
emit=[ 1 1 ]` where the repaired shape reads `ns=1 acc=0 emit=[ 1 ]`. The
corrupted slot holds the COMMITTED token, so it matches and is ACCEPTED, and
acceptance therefore RISES in this fixture rather than falling. Flattened, the
pre-repair run emits **198 tokens against the repaired run's 197** — one
inserted `1`, every other position identical, which is verify being lossless —
and `[SPECTRACE]` prints 185 lines against 192.

Two things follow and neither is a gate. Tokens-per-step at a fixed step budget
DOES differ between the shapes, so any statement that the defect is invisible to
every axis has to be scoped to the per-row `emit=` comparison. And the
discriminator is unreachable on a committed configuration: seeing it at all
needs mutation **M**, the veto deleted at both `async_input_combine_`
assignments, so it can be measured and cannot be committed as a gate. The
committed instruments stay what they are — `execute_model`'s `cu_num_logits`
size refusal, which the CPU tier builds and which a review proved non-vacuous
above `num_reqs == 1`, and G2's id comparison in `test_combine_tokens.cpp`.

### Reason B: the async sampler has no verify arm at all — CLOSED BY A2-2

**Historical from here down.** This subsection records the state A2-1 left and
the evidence for it. A2-2 gave `sample_tokens_async` its verify arm, so the
`file:line` and the grep below no longer describe the tree; they describe the
head this section was written against. What SURVIVES A2-2 is the other half of
the paragraph — `pending_drafts_` is still host-resident and the propose still
consumes host `num_sampled` / `num_rejected` — and that half is A2-3's, which is
why the veto still stands. Read `## Now` for the current position.

`GPUModelRunner::sample_tokens_async` (`runner.cpp:4039-4212`) contains
no reference to `sample_tokens_with_rejection`, to `propose_drafts`, to
`spec_on()` or to `num_draft_tokens`; a call-shaped grep over the function body
returns nothing. Its device arm samples one id per request straight off
`assemble_sample_logits`, which under spec returns the EXPANDED
`sum(1 + k_i)` rows (`runner.cpp:2331`, `step_num_logits`). So with the veto
lifted a spec engine on the `AsyncLLM` front would sample the verify rows as if
they were decode rows, never verify a draft, and never propose the next block.
`pending_drafts_` stays empty, the async scheduler's `-1` placeholders have
nothing to fill, and the fill refuses. `Sampler::forward`'s
`sampled_ids_out->Numel() == n` check (`src/vllm/v1/sample/sampler.cpp:382`)
is the second refusal waiting behind it. On the landing head the A2-3 draft-
buffer check refuses ahead of both, which is why the trace above is empty.

**This reason was not in the comment the veto carried** on the pre-A2-1 head,
which named only the input combine. A reader repairing that one piece would
still be several waves away from a flag that can be flipped, which is exactly
the misreading this spec exists to stop. A2-1's landing commit rewrote both
comment blocks so they name the reasons that still hold.

### Reason A: the corruption a token gate cannot see — FIXED BY A2-1

**Historical.** This subsection records the pre-A2-1 defect and the evidence
for it. A2-1 fixed it, and the post-A2-1 M run above can no longer reproduce
it. It is kept because it is why G2 exists and why R1 is written the way it is.

On the pre-A2-1 head `combine_sampled_and_draft_tokens`
(`src/vllm/v1/worker/gpu/prepare_inputs.cpp:285-332`) fixed
`num_logits = num_new_sampled_tokens` and wrote the committed token at
`query_end - num_logits`. With `num_new_sampled_tokens == 1` that address is
`query_end - 1`, which on a verify step is the LAST DRAFT SLOT, not the
committed-token slot.

Under **M** on `a0f12b727`, with `VT_SPEC_TRACE=1`, every verify block in every
arm had its last draft replaced by the previous step's emitted token:

| position | baseline `draft=[...]` | under M | previous step's `emit` |
|---|---|---|---|
| 4 | `6 18` | `6 5` | `5` |
| 5 | `11 7` | `11 3` | `3` |
| 6 | `7 0` | `7 9` | `9` |
| 7 | `5 20` | `5 18` | `18` |

**The emitted tokens never moved.** The `emit=` column was identical in both
runs, and the identity assertions of the arm carrying this corruption passed.
Speculative decoding is lossless, so a wrong draft costs acceptance and nothing
else: `tests/vllm/v1/spec_decode/test_dflash_causality.cpp:19` calls this "the
single defect in this row that raises nothing", and it already fired once as
[#1366](https://github.com/mudler/vllm.cpp/issues/1366). There it fired again,
under a mutation that a whole-binary run still reported as one failed case for
an entirely unrelated reason.

Two consequences follow, and both still bind the waves below even though the
defect itself is fixed.

- The corruption was a property of the INPUT COMBINE, not of the scheduler.
  The sync-scheduler arm (`VT_ASYNC_SCHED=0`) carried it identically, because
  `async_input_combine_` is a runner-level lever that `VT_ASYNC_SCHED` does not
  reach.
- **No COMMITTED acceptance gate on this fixture can see this class of defect,
  and the "zero dynamic range" reason for that was too strong.** The synthetic
  MTP head in `test_mtp_depth` accepts nothing in the ordinary baseline — every
  traced block reads `ns=1 acc=0` — but the round-3 measurement above shows the
  corrupted shape reaching `ns=2 acc=1` at two `k=1` positions and emitting one
  extra token over the run, because the committed token spliced into a draft slot
  matches itself. So the signal is not identically zero; it is only reachable
  under mutation **M**, its sign is an artefact of what the corruption happens to
  overwrite, and it says nothing about WHICH id is wrong. A draft-token
  comparison, not an acceptance ratio, is therefore still the instrument. That is
  what G2 is.

## Upstream anchors

Pin `5559679229bc961848b121ccdeaa8fa5d79bec98`.

- `vllm/config/speculative.py:63-65` — `DFlashModelTypes` is inside
  `EagleModelTypes`, so `vllm/config/vllm.py:1076-1087` never disables async
  scheduling for `method="dflash"` and `:1111-1112` resolves it True. This is
  the mirror obligation the issue names, and W7 already discharged its
  scheduler half.
- `vllm/v1/worker/gpu/input_batch.py:304-406` — the combine kernel and its
  wrapper. `num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS` (`:322`),
  the committed-token store at `logits_start` (`:345-348`) guarded by
  `first_logit_seq_pos >= prefill_len`, and the draft scatter at
  `query_end - num_draft_tokens + block` (`:350-361`) reading a DEVICE
  `draft_tokens` buffer. Piece 3 is this kernel; piece 2 is what makes its
  `draft_tokens` argument real for us.
- `vllm/v1/worker/gpu_model_runner.py:264-305` and `:307-320` — sampled ids
  stay device-resident and the host wait is deferred.
- `vllm/v1/worker/gpu_model_runner.py:1356-1396` and `:1795-1907` — the
  optimistic `prev_num_draft_len` correction and the on-device draft scatter.

Our current divergence, stated once so no wave re-derives it: our rejection
sampler runs its kernel on device but downloads and synchronizes inside
`forward` (`src/vllm/v1/spec_decode/rejection_sampler.cpp:100-106`), and
`propose_drafts` consumes the host `num_sampled` / `num_rejected` it produces
(`runner.cpp:2803-2815`). Upstream needs neither host value in step. That one
download is what pieces 1 and 2 remove, and it is the reason the veto cannot be
lifted a piece at a time.

## Work breakdown

Ordered by dependency. Each wave states what can fail on the CPU tier and what
cannot, because
[#2108](https://github.com/mudler/vllm.cpp/issues/2108) means no CI runner has
a GPU and a device case that is not visibly skipped is a skip wearing a pass.

- **A2-1 — the draft-aware combine.** Port `num_draft_tokens = num_logits -
  NUM_NEW_SAMPLED_TOKENS`, the `first_logit_seq_pos >= prefill_len` guard and
  the draft scatter. CPU-gateable in full: the function is host code with a
  CUDA counterpart (`src/vt/cuda/cuda_combine_tokens.cu`), and
  `tests/vllm/v1/worker/test_combine_tokens.cpp` already owns its unit tier.
  **Lands unreached** while the veto stands, so it carries the `## Owed`
  disclosure AGENTS.md requires, or it waits for A2-2. Prefer waiting.
- **A2-2 — device-resident verify.** Split `RejectionSampler::forward` so the
  accept walk's outputs stay on device, and give `sample_tokens_async` a verify
  arm that routes on `exec_state_.step.num_draft_tokens > 0` exactly as
  `sample_tokens` does at `runner.cpp:2797`. This is the wave that closes
  reason B. The CPU tier can gate the ROUTING and the token identity; it
  cannot gate the overlap.
- **A2-3 — device-resident propose. LANDED** (#2911). `pending_drafts_` stops
  being the drafts' residence and becomes only the out-of-band delivery to the
  scheduler; the residence is `InputBatch::draft_tokens`, a per-req_state
  `[max_num_reqs, k]` buffer that both the combine and the placeholder fill read,
  written through the single `set_draft_tokens` seam. The fill no longer needs
  host drafts, and it is REACHED and mutation-gated. It did NOT on its own make
  A2-1's draft scatter reachable, and the `## Owed` entries say why: the scatter
  needs a step that arrives with `num_logits > 1`, and only A2-5's veto flip
  produces one. The sentence that stood here predicted otherwise; the measurement
  in `## Now` is what replaced the prediction.
- **A2-4 — the optimistic correction.** `prev_num_draft_len` plus the deferred
  correction, replacing the structural rule W7 recorded at `runner.cpp:1760-1773`
  (which is correct precisely because our rejection result IS host-visible in
  step, and stops being correct when A2-2 lands).
- **A2-5 — flip the veto**, and only here. The flip is one line at two sites;
  every wave above exists so that line is safe.

## Gates

- **G1 (CPU, every wave).** `test_mtp_depth`, `test_dflash2_runner_reach`,
  `test_dflash_causality`, `test_rejection_sampler`, `test_combine_tokens`,
  `test_engine_core_proc`, `test_runner`. Green before and after.
- **G2 (CPU, A2-1 onward, and this is the gate the measurement above buys).**
  The draft tokens that reach the verify step must equal the drafts the
  proposer produced, asserted on the token ids and not on an acceptance ratio
  or on the emitted text. Nothing in the tree asserts this today, which is why
  reason A's corruption passed every identity check in the run above. A wave
  that lands without G2 has no instrument for its own defect.
- **G3 (GPU, A2-5).** TPOT and output-throughput A/B at c=1 and c=8 on the
  #1574 DFlash2 workload, operator-run under an `rc` lease, against the same
  binary with `VT_ASYNC_RUNNER=0`. Spread at c=8 is about 5.9% and at c=1 about
  1.6% (`dflash2-batch-propose.md`), so a delta under the floor is reported as
  under the floor and never as a ratio.
- **G4 (GPU, before A2-5 is worth scheduling).** The `nsys` idle read the issue
  specifies, `--cuda-graph-trace=node` so graphed kernels are not hidden. GPU
  busy at or above 95% of wall refutes the throughput half outright and leaves
  the mirror obligation standing alone.

## Risks

- **R1 — the acceptance-only defect, again.** Every wave here can change what
  the draft proposes without changing what comes out. G2 is the only answer;
  a token gate is not one. Recorded because this row has already paid for it
  once (#1366).
- **R2 — a tautological comparison.** Comparing `VT_ASYNC_RUNNER=0` against
  the default on a spec engine compares one path with itself today, because the
  veto forces both onto the sync host input path. Any A/B written before A2-5
  must assert that the two arms actually diverged.
- **R3 — the size is unmeasured, in both directions.** The only recorded
  host-time figure is the non-spec `~3.25 ms/step`
  (`include/vllm/v1/worker/gpu/runner.h:435`), roughly 1.6% of a c=8 step and
  under the floor. The spec-step host work (rejection over 8x9 = 72 rows at
  c=8, then the propose) is larger and has never been measured. G4 bounds it
  before A2-5 is scheduled. Nothing in this row asserts a speedup.
- **R4 — no GPU in CI.** A2-2 through A2-4 have device arms that the CPU tier
  degenerates. Each device case must be a VISIBLE skip with a non-zero
  assertion count on the arm that does run.

## Stop conditions

- G4 reading GPU busy at or above 95% turns this into a parity-only row. The
  mirror obligation stands, the throughput ceiling is zero, and the waves are
  scheduled accordingly rather than dropped.
- A wave needing V2-runner semantics this tree does not have,
  `NEEDS_DECISION` naming the dependency.
- Any correctness gate red that a scoped fix cannot green, `BLOCKED` with the
  failing case named.

## Owed

- **The G2 instrument itself. DISCHARGED by A2-1** (#2644), and the entry stays
  because the reason it was owed is the reason the wave exists. It could not be
  written against the pre-A2-1 tree without either flipping the veto or landing
  A2-1 first, since the drafts that reached the verify step were correct there
  by construction. It now lives in
  `tests/vllm/v1/worker/test_combine_tokens.cpp` as seven `G2:` cases that drive
  the draft-aware combine directly and compare the drafts landing in the verify
  rows against the proposer's ids, one id at a time. It was red before the
  change, with the whole draft-aware body reverted to its pre-A2-1 shape:
  `14 test cases, 6 failed, 17 assertions failed`, and G2.2 fired on BOTH its
  assertions, `CHECK( 5 == 6 )` at `:326` and `CHECK( 5 != 5 )` at `:327` —
  reason A's committed token sitting in the last draft slot.

  Which of those two lines fires is a property of the defect's shape and not of
  the gate, so neither line on its own is "the reason-A assertion". The
  committed store runs BEFORE the draft scatter, so a defect that leaves the
  scatter's bound intact has its corrupted slot rewritten by the scatter and is
  caught by another case instead. G2.2 is sound; the claim to make about it is
  that the case fires, not that a chosen line does.
- **A2-3's two CUDA combine arms are WIRED and UNBUILT.** No `nvcc` on the CPU
  box this wave ran on, and no `rc` lease was taken because the wave is
  CPU-gateable in full and the fleet was needed for measurement. No `.cu` file is
  edited — `LaunchCombineSampledAndDraftTokens` already took `draft_tokens`, its
  stride and `cu_num_logits` — so the unbuilt surface is what sits inside the two
  `#ifdef VLLM_CPP_CUDA` blocks in `runner.cpp`: the two launch calls and the two
  `stage_upload`s beside them. The device mirror's ALLOCATIONS and its entries in
  the destructor's free list are outside the `#ifdef` and are compiled here.

  **THE CLAIM THIS ENTRY USED TO MAKE WAS FALSE, and a fresh review returned FAIL
  on it.** It said the two device arms were "the same call with a different
  residence" with the "risk bounded to a compile failure". They were the same
  BUFFER with a different `cu_num_logits`: the host arm passed the real
  per-request `step.cu_num_logits` and both device arms passed
  `/*cu_num_logits=*/nullptr`. `src/vt/cuda/cuda_combine_tokens.cu:69-79` reads a
  null as arange, giving `num_logits == 1`, `num_draft_tokens == 0` and
  `logits_start == query_end - 1` — which on a verify step IS the last draft
  slot. So the first step after A2-5 lifts the veto, on GB10's integrated
  production default, the device arms would have written the committed token over
  the last draft and scattered NOTHING, while the host arm wrote it at
  `query_end - (1 + k)` and scattered k drafts. That is reason A returning by a
  second route, it compiles cleanly, and it is invisible to every token gate here.

  **FIXED, not documented.** Both arms now read the same `cu_num_logits` the host
  arm does — the UMA arm takes the host pointer, like `prefill_len` beside it; the
  mirror arm uploads it into a new `AsyncDeviceInputs::cu_num_logits` exactly as
  it uploads `query_start_loc`, the same `num_reqs + 1` extent — through ONE local
  so the two cannot be edited apart. `step.cu_num_logits` is the arange on the
  non-speculative path (`prepare_inputs.cpp:213`), so the change is inert for
  every engine running today. A refusal was rejected as the repair: it would be
  unbuilt here too, and it would leave A2-5 making this exact edit at these exact
  two call sites with no mutation to check it against. A built assertion covers
  the remaining silent route — every arm's only source of a row's `num_logits` is
  this vector, so `execute_model` now refuses a combine whose
  `step.cu_num_logits` does not hold `num_reqs + 1` entries, ahead of the arm
  selection and outside the `#ifdef`. THE MEASUREMENT is under `## What was
  measured`. Owner: row `SPEC-DFLASH2`, issue #2911.
- **The fill's FIRST refusal was deleted by A2-3, not restated, and the header
  said otherwise. FIXED.** `prepare_inputs.h` claimed the fill carried "the
  pre-A2-3 pair of refusals, restated on the buffer". It carried one of them.
  The deleted one was
  `VT_CHECK(own_it != own.end(), "no drafts proposed for request '…'
  (placeholders scheduled without a matching propose)")` over a map built from
  `pending_drafts_`; because `take_draft_token_ids` MOVES that object out, it
  meant "a propose ran for this request since the last time anyone looked". A2-3
  rehomed the same MESSAGE onto `input_batch_.req_id_to_index.find(req_id)`,
  which asks whether the request is in the persistent batch — nearly always true.

  Freshness then rested only on `num_valid_draft_tokens`, which survives the pull
  BY DESIGN and is zeroed only by `clear_draft_tokens`. Any step that skips the
  propose leaves the count standing and `FillDraftsForRow` splices the PREVIOUS
  verify step's drafts into this step's placeholders — silently, verify being
  lossless. Pre-A2-3 that input threw by name. It is live as soon as the async
  decode arm's gap (the entry below) is fixed for step one but a later step skips
  a propose, or the scheduler and the runner disagree by one.

  RESTORED rather than documented. `ProposedDraftLedger` (`prepare_inputs.h`)
  records the req_ids the most recent propose wrote; `set_draft_tokens` calls
  `Record`, `clear_draft_tokens` calls `Clear`, and the fill calls `IsFresh` then
  `Consume`, clearing it after use — exactly as `take_draft_token_ids` consumes
  `pending_drafts_`, which is where the old freshness came from. The refusal keeps
  the pre-A2-3 wording so a reader who has seen it before recognises it, and what
  `FillDraftsForRow` can and cannot answer is now stated on that function.

  **THE REACH AND THE RULE ARE GATED SEPARATELY, and neither stands in for the
  other.** REACH: forcing the runner's refusal false reds `test_mtp_depth` 5 of
  10 and `test_dflash2_runner_reach` 9 of 10 — the same counts the producer-write
  mutation gives, because it is the same production filled step, so the line runs
  on every step the fill patches. RULE: `Consume` made a no-op reds
  `test_draft_fill.cpp:241-242` on two `CHECK_FALSE(ledger.IsFresh(...))`, and
  `Record` merging instead of replacing reds `:259` and `:261`. **What NOTHING in
  this tree gates is the two together**: deleting the refusal outright is SILENT
  across `test_mtp_depth`, `test_dflash2_runner_reach`, `test_engine_core_proc`,
  `test_runner` and `test_dflash_causality`, because no CPU-tier suite drives a
  step that skips its propose through `execute_model`. Recorded rather than
  claimed. Mutation **M** is the nearest thing to that step, and at the repair
  head it throws the restored refusal by name (`## Now`). Owner: row
  `SPEC-DFLASH2`, issue #2911.
- **`test_verify_route_predicate.cpp` described a refusal that no longer exists
  and re-derived the predicate it was written to keep single. FIXED.** The case
  said the site refuses because "the draft buffer … is not wired yet (A2-3,
  #2644)" — false since A2-3 wired it, and the wrong issue, because #2911 owns
  A2-3 and #2644 owned A2-1. It then hand-copied
  `const bool refuses = !StepRoutesToVerify(StepTotal(mixed));` and asserted it
  false, which is a SECOND expression of the rule this file exists to keep single
  (#2710) and, since A2-3 made the site's condition
  `!StepRoutesToVerify(...) || <the buffer covers the pool>`, was not even the
  site's rule any more. The case now states the current condition, reads the
  route once, and drops the partition claim; `test_verify_route_predicate` goes
  from 21 assertions to 20 and stays green. Owner: row `SPEC-DFLASH2`, issue
  #2911.
- **The device mirror's destructor freed seven pointers with no drain, and said
  it was safe. FIXED.** `~GPUModelRunner` called `b.Free(p)` over the
  `AsyncDeviceInputs` list with no `Synchronize` and no event wait, and the
  struct's header argued the safety from "the runner and therefore its queue are
  going away". That is not an argument: the runner does not own `queue_`, does
  not destroy it and does not drain it — only `async_copy_queue_` is destroyed
  here, and before the frees. A kernel queued against those buffers is still
  queued. It is the same defect `20d225e54` repaired one level down in
  `RejectionSamplerDeviceOutput::Release`, which is on `main` and sets the
  standard.

  PRE-EXISTING for six of the pointers; A2-3 added the seventh (`draft_tokens`)
  AND wrote the rejected reasoning down as a lifetime proof, which is what turns
  a carried gap into a contradiction between two files. `~GPUModelRunner` now
  drains `queue_` before the first `Free`, and the header says the drain is the
  guarantee. WHAT IT COSTS: one full main-queue drain per runner destruction —
  once in a process's life for the ordinary single-runner server, paid at
  TEARDOWN with no request in flight, so it is off every serving path. Same trade
  as the `Release` repair: an observable stall instead of an unobservable wrong
  answer. Owner: row `SPEC-DFLASH2`, issue #2911.
- **`sample_tokens_async`'s DECODE arm never proposes, and A2-5 is blocked on
  it.** FOUND BY A2-3 (#2911), which is the first wave whose measurement could
  see it. A2-2 gave the async sampler a verify arm and put
  `propose_after_verify` inside it, so a step that DRAFTS proposes correctly. The
  arm that runs when `StepRoutesToVerify` is false has no `spec_on()` branch at
  all, where the synchronous `sample_tokens` ends its decode path with one
  (`runner.cpp`, the `if (spec_on()) { ... propose_drafts(...); }` tail). That
  arm is every spec engine's FIRST step, before any drafts exist, so with the
  veto lifted a spec engine on the `AsyncLLM` front proposes nothing on step one
  and the next step's placeholder fill refuses by name.

  This is why mutation **M** is still red at the A2-3 head, and the refusal it
  hits has moved again — from the combine to the fill. It is NOT a draft-buffer
  gap: A2-3 wired the buffer and M no longer produces a single "async input
  combine:" line. It is an unreached arm that was landed incomplete, and it is
  out of A2-3's scope, which the dispatching brief drew around the draft buffer
  and the fill. **A2-5 cannot flip the veto until this is closed**, and whichever
  wave closes it owes the mixed-step coverage the fill's own rule now has.
  Owner: row `SPEC-DFLASH2`, issue #2911.
- **A2-1's draft lane is STILL UNREACHED after A2-3, and A2-5 owns the last
  step.** A2-3 (#2911) made the scatter's `draft_tokens` argument REAL — every
  call site now passes the per-req_state `InputBatch::draft_tokens` and its
  stride, and the host arm passes the true per-request `step.cu_num_logits` — so
  the buffer exists, is produced by one seam and is read by two consumers. What
  the scatter still lacks is a step that ARRIVES with `num_logits > 1`, and only
  lifting the `async_input_combine_` veto produces one. A2-3 was scoped not to
  touch it. The disclosure AGENTS.md requires is carried by this entry, the
  landing commit body and the pull request body. Owner: row `SPEC-DFLASH2`,
  issue #2911.
- **A2-1's draft lane was UNREACHED at the A2-1 head, and A2-2 plus A2-3 owned
  the wiring.** HISTORICAL; superseded by the two entries above. The
  draft-aware halves of `combine_sampled_and_draft_tokens`
  (`src/vllm/v1/worker/gpu/prepare_inputs.cpp`) and of
  `vt::cuda::LaunchCombineSampledAndDraftTokens`
  (`src/vt/cuda/cuda_combine_tokens.cu`) landed with no production step able to
  reach them: `async_input_combine_` is vetoed for every speculative engine at
  BOTH construction sites, the two
  `src/vllm/v1/worker/gpu/runner.cpp::async_input_combine_` assignments, so every call site passes
  an arange `cu_num_logits` and an empty draft buffer, and `num_draft_tokens` is
  always 0. The call sites refuse
  loudly (`VT_CHECK(step.num_draft_tokens == 0, ...)`) rather than combine a
  verify step against a draft buffer that is not wired. A2-3 supplies that
  buffer, A2-5 lifts the veto, and the disclosure AGENTS.md requires is carried
  by this entry, the landing commit body and the pull request body. Owner: row
  `SPEC-DFLASH2`, issue #2644.
- **The CUDA arm of A2-1 was not compiled.** No `nvcc` on the CPU box this wave
  ran on, and no `rc` lease was taken because the wave is CPU-gateable in full.
  `src/vt/cuda/cuda_combine_tokens.cu` therefore carries the same edit as the
  host loop, INCLUDING its refusal (`__trap()` where the host throws a
  `VT_CHECK`, since a kernel cannot throw), read against it line for line and
  unbuilt. It is a no-op at every present call site (both pass null
  `draft_tokens` and null `cu_num_logits`, so the kernel degenerates to the
  pre-A2-1 single splice), which bounds the risk to a compile failure rather
  than a behaviour change. The `__trap()` itself is unexercised for the same
  reason, and A2-3 is the first wave that can run it. One host check has no
  device counterpart and is recorded in `include/vt/cuda/combine_tokens.h`
  rather than dropped: the host bounds the draft row against
  `draft_tokens.size()`, and no length reaches the kernel.

  **A2-3 must read that gap's CONSEQUENCE, not just the gap.** The kernel's
  scatter reads `draft_tokens[req_state_idx * draft_tokens_stride + b]` with
  nothing bounding it, so an over-long row is an unchecked out-of-bounds DEVICE
  READ, and it has two outcomes. The loud one is an illegal memory access. The
  quiet one is the dangerous one: the read lands inside another allocation,
  garbage arrives in the draft slots, and because speculative decoding is
  lossless a wrong draft costs acceptance and NOTHING ELSE — invisible to every
  token gate in this tree, which is reason A's class exactly. The concrete way
  to hit it is A2-3's own: if A2-3 sizes the draft buffer by the ACTIVE REQUEST
  COUNT while `req_state_idx` is a req_state POOL SLOT (the indirection the
  header documents, the one that "matters after an abort/finish reorder"), a
  high slot indexes past the allocation with every host check satisfied. A2-3
  either passes a length and refuses on it device-side as the host does, or
  sizes the buffer by `num_req_states` rather than by `num_reqs`. Owner: row
  `SPEC-DFLASH2`, issue #2644.
- **The async input-combine refusal has ZERO test coverage.** THE FORM THIS
  ENTRY DESCRIBED NO LONGER EXISTS: A2-2 replaced the per-site
  `VT_CHECK(step.num_draft_tokens == 0, ...)` refusals with ONE site, in the
  async input-combine block of
  `src/vllm/v1/worker/gpu/runner.cpp::execute_model`, written as a
  `VT_CHECK(step.num_draft_tokens >= 0, ...)` guard followed by a
  `VT_CHECK(!StepRoutesToVerify(step.num_draft_tokens), ...)` refusal, so the
  refusal and the route now share one predicate. `grep -rn "num_draft_tokens ==
  0" src/ include/` returns only comments and upstream citations. A2-2 made this
  record stale and this correction rides in the change that did, per AGENTS.md
  §Records.

  The GAP is unchanged. Neutralizing the refusal leaves every runner suite green,
  which is expected while the wave is unreached: the refusal cannot be reached
  without the route it refuses. A fresh review traced the refusal predicate
  against the route predicate and found they cannot disagree in either direction
  for a non-negative count, and the `>= 0` guard is what keeps the negative case
  refused rather than admitted. Recorded rather than covered, because a test that
  reached an unreached refusal would have to fake the route it is asserting
  about. A2-3 makes the route real and owns the coverage with it. Owner: row
  `SPEC-DFLASH2`, issue #2644.
- **`assert()` is a no-op in this build, so the `num_new_sampled_tokens in
  (0, 1)` mirror at `src/vllm/v1/worker/gpu/prepare_inputs.cpp:330` checks
  nothing.** The Release build is `-O3 -DNDEBUG`, which compiles the assertion
  out, so the line reads as a guard and is documentation. It mirrors upstream's
  own assert (`input_batch.py:376-378`), which is why it is written as one.
  PRE-EXISTING: the line predates A2-1 and this wave neither added nor moved
  it; A2-1 is only where a fresh review named it. Not fixed here, because
  turning it into a `VT_CHECK` changes a refusal's behaviour in released builds
  and that is its own change with its own red-before. Owner: row
  `SPEC-DFLASH2`, issue #2644.
- **A2-2's async VERIFY ARM is UNREACHED, and A2-5 owns the wiring.** The arm
  `sample_tokens_async` gained (`src/vllm/v1/worker/gpu/runner.cpp`) routes on
  `StepRoutesToVerify(exec_state_.step.num_draft_tokens)` — the same expression
  `sample_tokens` routes on and the same one, negated, that the async input
  combine refuses on. Nothing reaches it: the function returns
  `sample_tokens(...)` unchanged when `!async_input_combine_`, and the veto at
  the two `src/vllm/v1/worker/gpu/runner.cpp::async_input_combine_` assignments
  keeps `async_input_combine_` false for every
  speculative engine. The reachability mutation was run and it is silent, as
  expected: the whole `if` disabled (`if (false && StepRoutesToVerify(...))`),
  `test_mtp_depth` 10/10, `test_dflash2_runner_reach` 10/10 and `test_runner`
  38/38 all stay green, exit 0. No CPU-tier test can reach it either, and that
  is not a gap in the tests: with `async_input_combine_` forced on, a
  speculative step refuses one stage EARLIER, at the A2-3 draft-buffer
  `VT_CHECK` in `execute_model`, so the route cannot be made real without A2-3.
  A2-3 supplies that buffer and A2-5 lifts the veto. Owner: row `SPEC-DFLASH2`,
  issue [#2802](https://github.com/mudler/vllm.cpp/issues/2802).
- **The COPY-QUEUE download route (`VerifyDownload::kCopyQueueEvent`) is
  unreached for the same reason, and the CPU tier could not measure it even if
  it were.** It is the payload of the wave — the accept walk's two D2H copies
  move off the main queue onto the async copy queue, ordered by a fork event and
  waited through a completion event, so the accept walk's copy stops being a
  main-stream operation and its wait becomes a copy-queue event. The step still
  stalls once: the host waits that event IN STEP, and the next bullet is the
  entry that owns the distinction. On the CPU backend `Copy` is a memcpy and every event is a
  null-handle no-op (`vt::Backend` base implementations), so the route is
  observably token-identical and observably nothing else. The overlap is G3/G4
  at A2-5 and needs a GPU; nothing in this wave claims a throughput or latency
  result. Owner: row `SPEC-DFLASH2`, issue #2802.
- **A2-2 makes the first drain REMOVABLE; it does not remove it, and the
  distinction is the one a reader will get wrong.** On the copy-queue route the
  host still waits IN STEP, because the write-back and `propose_drafts` both
  need `num_sampled` and the emitted ids on the host. Waiting on the ready event
  is not free merely because it is an event: the copy waits the fork event, so
  by the time it completes the main queue has drained anyway. What A2-2 changed
  is the SHAPE — the copy is no longer a main-stream operation, and the wait is
  on a copy-queue event a later wave can move past the propose. A2-3 (the propose
  stops consuming host `num_sampled` / `num_rejected`) and A2-4 (the optimistic
  `prev_num_draft_len` with the deferred correction) are what actually move it.
  Until then the step still stalls once, and NO wave before A2-5 may report a
  latency or throughput result for this. `RejectionSampler::forward` also keeps
  its own `Synchronize`, deliberately, because that is what keeps the synchronous
  sampler byte-identical. The speculator's own draft download and `Synchronize`
  inside `propose_drafts` is untouched and is A2-3's. Owner: row `SPEC-DFLASH2`,
  issue #2802.
- **THE `logits` BUFFER IS NOT OWNED BY THE ACCEPT WALK, AND A2-4/A2-5 ARE WHAT
  MAKE THAT DANGEROUS.** A fresh review found the ownership paragraph on
  `RejectionSamplerDeviceOutput` false: the walk reads six buffers and the object
  owned four. Five are now owned, including the per-row argmax scratch this
  repair moved out of the CUDA backend (see the entry below). The sixth cannot
  be: `logits` is the FORWARD's own output buffer on the device path —
  `include/vllm/v1/worker/gpu/runner.h::assemble_sample_logits` hands the sampler
  `exec_state_.logits.device_tensor` directly — and the next step's forward
  writes it. The argmax kernel reads it AFTER `verify` returns.

  Today every caller is safe because the wait is still in step. The failure this
  entry books is precise: **the moment A2-4 or A2-5 moves the wait past the next
  forward, that forward overwrites the rows the argmax kernel has not read yet.
  Speculative decoding is lossless, so the result is a garbage accept prefix and
  wrong emitted ids with no exception anywhere** — reason A's class, invisible to
  every token gate in this tree (#1366). A2-4 either keeps the wait ahead of the
  next forward or double-buffers the logits; it may not simply move the wait.
  The contract is written per buffer in
  `include/vllm/v1/spec_decode/rejection_sampler.h` so the obligation is on the
  type rather than in this file alone. Owner: row `SPEC-DFLASH2`, issue #2802.
- **The CUDA arm of this repair was NOT COMPILED.** No `nvcc` on the CPU box and
  no `rc` lease was taken, so `src/vt/cuda/cuda_sample.cu` carries its edit read
  line for line and unbuilt. The edit is a deletion plus a substitution: the
  file-scope `g_reject_argmax` / `g_reject_argmax_cap` / `EnsureRejectArgmaxScratch`
  are gone, and both kernels take the caller's `target_argmax` tensor instead.
  That global was the defect — `EnsureRejectArgmaxScratch` `cudaFree`d it to grow
  it, and `GreedyRejectionSampleCuda` returns with both kernels still queued, so
  a second caller with more rows freed the buffer the first caller's queued
  accept kernel reads. It was reachable with two runners in one process and
  reachable from one engine's next step as soon as the wait moves. The grow path
  was also a device-wide synchronize inside a function that advertises waiting on
  nothing. `tests/vt/test_cuda_ops.cpp` (CUDA-only, also unbuilt here) carries the
  matching call-site change. Owner: row `SPEC-DFLASH2`, issue #2802.
- **The copy-queue route's D2H destination is now PAGE-LOCKED, and the CUDA half
  of that is unbuilt too.** It was a plain `std::vector<int32_t>`, and a
  device-to-host `cudaMemcpyAsync` into pageable memory is host-synchronous —
  driver-staged, returning only when the bytes are across — so the fork and ready
  events around it were decorative and no later wave could have obtained overlap
  from them. Upstream copies into torch CPU memory
  (`vllm/v1/worker/gpu/async_utils.py:124-125` @ pin 5559679229) and this tree's
  own async sampled-id route, the one this route mirrors, already uses
  `AllocPinned` (`src/vllm/v1/worker/gpu/async_output.cpp`). The destination is
  now `include/vllm/v1/worker/gpu/runner.h::PinnedGrowStaging`, which grows and
  never frees the block it replaces, because freeing a block a queued copy still
  writes is the same defect one level up. On CPU `AllocPinned` forwards to
  `Alloc`, so the CPU tier gates the tokens and nothing else; whether the copy
  actually overlaps is G3/G4 at A2-5 and needs a GPU.

  WHAT IT RETAINS, stated precisely because page-locked memory is the scarce
  kind. One block per distinct LARGER step shape — the ask is
  `rows * width + rows` — so a serving ramp that adds one request at a time
  retains one block per request added, not "a handful over a process lifetime" as
  the first version of this record and the header comment both said. The total is
  still bounded: the sizes are strictly increasing and the shape is bounded by
  the batch, so the sum is on the order of N^2/2 int32 for a batch of N. It is
  not a leak and it is not a defect; the imprecise claim was, and the header now
  says the bounded thing.

  THE RETAIN RULE'S MUTATION HAS NO STABLE FAILING-ASSERTION COUNT, and the
  landing commit body's "four assertions" should not be read as one. Making `Get`
  free the previous block before allocating reds the two `live_blocks()`
  assertions DETERMINISTICALLY (`test_runner.cpp:3414` `CHECK( 1 == 2 )` and
  `:3423` `CHECK( 1 == 3 )`). The other two assertions in that case read the
  RETAINED block, so under the mutation they read freed memory and fail only when
  the allocator has reused the bytes: this repair measured 4 failing assertions
  (`:3418` `CHECK( 1003444615 == 100 )` and `:3419` `CHECK( -1902570769 == 103 )`
  both fired) and the fresh review measured 3 from the same mutation. Quote the
  two `live_blocks()` lines, which are the gate. Owner: row `SPEC-DFLASH2`,
  issue #2802.
- **`propose_after_verify` DIVERGES from upstream for a row that both carries
  drafts and is chunked-prefilling, and A2-2 doubled that divergence's reach.**
  `RejectionSampler::finalize` zeroes both counts for such a row, mirroring
  `vllm/v1/worker/gpu/input_batch.py:425-434`. The runner then writes
  `num_accepted_tokens[i] = max(num_sampled, 1) = 1` (the GDN slot select's own
  rule), and `propose_after_verify` re-derives the propose's inputs from THAT:
  `num_sampled[i] = 1`, `num_rejected[i] = k`. Upstream passes the kernel's `0`
  and `0` straight through to the propose
  (`vllm/v1/worker/gpu/model_runner.py:1144` -> `:1533-1546`), so we hand the
  proposer a rollback of `k` where upstream hands it none.

  PRE-EXISTING: this is a byte-faithful lift of `e64f00560:runner.cpp:3560-3574`
  and A2-2 neither wrote nor moved the arithmetic. What A2-2 did was give it a
  SECOND entry point, so the same divergence now reaches from
  `sample_tokens_async` as well. NOT FIXED HERE, deliberately: the fix is for the
  propose to consume the sampler's `num_sampled` / `num_rejected` instead of
  re-deriving them from `num_accepted_tokens`, which changes what the REACHABLE
  synchronous propose receives and belongs with A2-3, the wave that owns the
  propose's inputs. The sampler half of the shape is covered
  (`tests/vllm/v1/spec_decode/test_rejection_sampler.cpp`, both the `forward` and
  the split routes drive a row that carries drafts and is prefilling); the runner
  half has no test, and `grep propose_after_verify tests/` returns nothing.
  Owner: row `SPEC-DFLASH2`, issue #2802.
- **The event choreography is gated for WELL-FORMEDNESS ONLY, and the CPU tier
  cannot gate its structure at all.** `test_rejection_sampler.cpp` issues the
  runner's five calls in the runner's order — fork event recorded on one queue,
  a second queue made to wait it, both D2H copies issued there, ready event
  recorded on that queue, host blocked on that event alone — and asserts the
  result is token-identical to `forward`. That is what it gates: the calls
  compile against the real signatures, are well formed against the real buffers,
  and lose no tokens.

  A SECOND REVIEW FOUND THE STRONGER CLAIM THIS ENTRY USED TO MAKE FALSE, and it
  is corrected rather than softened. `CpuBackend::CreateQueue`
  (`src/vt/cpu/cpu_backend.cpp`) returns `Queue{Device{kCPU, 0}, nullptr}`, which
  is byte-identical to the queue the test builds by hand apart from the `id`
  field no backend call reads. So on this tier there is no second queue: `copy_q`
  and `main_q` are the same device and the same null handle, every event op is
  the `vt::Backend` base no-op, and `Copy` is a memcpy. The reviewer swapped the
  copy onto the main queue and the case stayed green — 17 cases, 275 assertions,
  exit 0. "The main queue is never synchronized" is therefore NOT a property this
  test holds, and it no longer says it does; the case now asserts the identity of
  the two queues so the limit is executable rather than prose. The two-queue
  structure, and any overlap it buys, is G3/G4 at A2-5 and needs a GPU. Owner:
  row `SPEC-DFLASH2`, issue #2802.
- **THE DESTRUCTOR'S TIMING IS NOW PART OF THE INVARIANT, AND A2-4 MUST STILL
  DESTROY THE OBJECT AFTER ITS WAIT.** A second review found the per-buffer
  invariant on `RejectionSamplerDeviceOutput` said WHICH buffers it owns and
  never said WHEN it may free them: `Release` called `Free` on all three device
  buffers with no `Synchronize` and no event wait, which is the `g_reject_argmax`
  defect relocated from the backend into the object's own scope.

  The failure is concrete and it is A2-4's. On the copy-queue route `dev_out` is
  a BLOCK-SCOPED LOCAL destroyed at the close of the `else` in
  `GPUModelRunner::sample_tokens_async`. A2-4's stated job is to move
  `SynchronizeEvent(verify_ready_event_)` past `propose_drafts`. Move it past
  that closing brace and the destructor frees `sampled_`, `num_sampled_` and
  `target_argmax_` while the D2H copy is still writing them: a garbage accept
  prefix, wrong emitted ids, nothing raised, and — because the verify is lossless
  — invisible to every token gate in this tree (#1366). That is the `logits`
  entry's sibling, and it was booked nowhere.

  FIXED, not documented, and the choice is deliberate. `Release` now drains
  before it frees: it synchronizes the queue `verify` was given, and the queue of
  the last `CopyToHost` when that is a different one, and only then frees.
  Documentation would have left A2-4 one editing mistake away from a wrong answer
  that no gate here can see. The drain turns that same mistake into a stall,
  which the G4 `nsys` read A2-5 must take anyway would show. Trading an
  unobservable wrong answer for an observable slow one is the right direction in
  a tree whose recurring failure is reason A's class.

  IT IS A NET AND NOT A LICENCE. A2-4 still has to destroy the object AFTER the
  wait it moves, or it pays a full drain in the destructor and gets no overlap
  from the wave at all. The net makes the mistake slow; it does not make it
  correct. It also creates one new and much cheaper obligation, written on the
  type: the two queues are borrowed, so they must outlive the object. Both are
  runner members today, and a queue destroyed before the work on it is undefined
  on CUDA regardless of this type. Gated on the CPU tier by two cases in
  `tests/vllm/v1/spec_decode/test_rejection_sampler.cpp` that swap in a
  forwarding `vt::Backend` and assert the CALL ORDER — both queue drains before
  the first `Free`, and no drain after it. Red-before, by deleting the two
  `Synchronize` calls: 2 cases and 3 assertions failed, `CHECK( free ==
  sync:38 )`, `CHECK( free == sync:39 )` and `CHECK( free == sync:40 )`, exit 1,
  with every token case in the file still green. Owner: row `SPEC-DFLASH2`,
  issue #2802.
- **`src/vt/cuda/cuda_sample.cu` STILL CARRIES TWO of the grow-only scratches
  this wave deleted the third of, and they are on the ORDINARY decode path.**
  `g_argmax_scratch` and `g_sample_scratch` (`:187-188`) are file-scope, grown by
  `EnsureArgScratch` (`:174`, called at `:230` and `:350`) with a `cudaFree` plus
  a larger `cudaMalloc`, and read by `GreedyArgmaxCuda` (`:233-236`) and
  `RandomSampleCuda` (`:354-357`),
  both of which return with their two kernels still queued. F1's argument applies
  verbatim: a later call with more rows frees the buffer an earlier call's queued
  `ArgmaxFinalKernel` reads, the loud outcome is an illegal access, the quiet one
  is a wrong token id with nothing raised, and the grow's `cudaFree` is a
  device-wide synchronize inside a function that advertises waiting on nothing.
  PRE-EXISTING and UNTOUCHED by A2-2, which is why it is filed rather than fixed
  here — but the repair deleted one instance of a pattern and would otherwise
  have left two identical ones unnamed in the same file. Closing it needs a GPU:
  the class is invisible on the CPU tier and on any unified-memory backend,
  where the kernels have already finished. Owner: no row yet; tracked by issue
  [#2916](https://github.com/mudler/vllm.cpp/issues/2916).
- **`~GPUModelRunner` destroys the copy queue and the two verify events BEFORE
  it drains.** `DestroyQueue(async_copy_queue_)` and the two `DestroyEvent` calls
  run ahead of the mirror's `b.Synchronize(queue_)`, so A2-3's drain protects the
  mirror's buffers and protects neither of them. PRE-EXISTING and untouched here:
  A2-3's round-3 repair owns the drain's CLAIM, not the destructor's ordering,
  and reordering the teardown is a separate change with its own mutation. It is
  the same class as the `#2916` entry above and closing it needs the same GPU
  tier, because on the CPU build `async_device_inputs_` is always null and the
  block never executes at all. Owner: this row, wave A2-4, which is the wave that
  moves waits in this destructor.
- **The drain is COMPILED here and EXECUTED on neither tier.** Deleting it leaves
  all eleven G1 targets green, on the CPU build because
  `async_device_mirror()`'s body is inside `#ifdef VLLM_CPP_CUDA` and on the CUDA
  build because this box has no `nvcc`. Its correctness rests on reading — sole
  destruction path, right queue, correctly scoped, no double drain, no deadlock —
  and it stays owed a device run.
- **The `nsys` read (G4).** RUN, by the operator holding the device; this wave
  took no lease and measured no device, so the number below is recorded from that
  run and not from this one, and the run's own recipe and evidence belong with it.
  The decode GPU reads **96.01% busy**, and this spec's threshold is that at or
  above 95% the throughput half of the A2 premise is refuted outright. A2-4 and
  A2-5 are STOPPED on it, and A2-3 lands for its structural value alone: it
  discharges A2-1's `## Owed` and removes a reason-A hazard. No wave in A2 may
  quote a throughput result against this reading. What is still owed here is the
  run's recorded recipe, revisions and contention state beside the number.

## Now

`SPEC-DFLASH2` stays `ACTIVE`. The veto STANDS at both construction sites, the
two `src/vllm/v1/worker/gpu/runner.cpp::async_input_combine_` assignments, and
its comment names the reasons that still hold rather than the one A2-1 removed.

**A2-1 has landed** (#2644): `combine_sampled_and_draft_tokens` takes its
per-request `num_logits` from `cu_num_logits`, carries the
`first_logit_seq_pos >= prefill_len` guard, and scatters the drafts over
`[query_end - num_draft_tokens, query_end)`, on both the host and the CUDA side,
which refuse the same shapes. G2 exists and is green; G1 was green before and
after. The draft lane is UNREACHED, as the `## Owed` entry above discloses.

**G2 IS FUNCTION-LEVEL AND THIS SENTENCE IS NOT COVERAGE**, which a round-3
review found it reading as. `test_combine_tokens.cpp:233` says so itself: it
drives `combine_sampled_and_draft_tokens` directly because the runner cannot
reach it while the veto stands. It gates the RULE. The nearest end-to-end
statement is the producer-side round trip A2-3's round-3 repair added
(`WriteDraftRow` written, `FillDraftsForRow` read back, in `test_draft_fill`)
plus the count-write mutation that reds `test_mtp_depth` 5 of 10 and
`test_dflash2_runner_reach` 9 of 10 through `execute_model`. Neither makes the
combine's draft scatter reached, and nothing here should be read as saying it is.

**A2-2 has landed** (#2802). `RejectionSampler` is split: `verify` issues the
accept walk and returns a `RejectionSamplerDeviceOutput` whose buffers are still
on the device with nothing waited on, `CopyToHost` issues the two D2H copies on
whatever queue the caller names, and `finalize` is the pure-host reduction.
`forward` is those three in a row on one queue and is byte-for-byte its
pre-split self. A fresh review then returned FAIL on eight findings and they are
repaired on the same branch: the accept walk's per-row argmax buffer moved out of
a CUDA process-global and into the device result, which now states its ownership
PER BUFFER and names `logits` as the one it does not own; the copy-queue route's
D2H destination is page-locked; and the drain claim is stated as a change of
SHAPE everywhere it appears, because the host still waits in step. The two
findings that are not repaired here — the `logits` lifetime obligation A2-4/A2-5
inherit, and `propose_after_verify`'s chunked-prefill divergence — are `## Owed`
entries above, each with the failure written out. `sample_tokens_async` has a
verify arm routing on
`StepRoutesToVerify(exec_state_.step.num_draft_tokens)`, the one expression
`sample_tokens` and the input-combine refusal also read, and it takes the
copy-queue download route, so the accept walk's copy leaves the main queue and
its wait becomes a copy-queue event. The host still waits in step; the `## Owed`
entry on the removable drain says what A2-4 and A2-5 have to do to move it.

**Reason B is closed in the code and NOT yet in the run.** The arm exists and is
correct; nothing reaches it while the veto stands, exactly as A2-1's draft lane
is unreached, and both `## Owed` entries say so. What A2-2 changed about the
veto is that lifting it would no longer sample verify rows as decode rows — the
half of reason B this wave owns. The other half was A2-3's, and A2-3 has since
landed: the paragraph that stood here said `pending_drafts_` was still
host-resident and that the A2-3 `VT_CHECK` still refused every speculative step
ahead of everything else, and neither is true at the head this file now ships
with.

**A2-3 has landed** (#2911). `InputBatch` carries the drafts' residence: a
per-req_state `draft_tokens` buffer of `[max_num_reqs, num_speculative_steps]`
row-major, mirroring `RequestStates.draft_tokens` (states.py:71-77), with a
`num_valid_draft_tokens` count beside it that has NO upstream counterpart. A2-3
anchored that count on `gpu_model_runner.py:883-895` and the anchor is wrong,
checked at the pin: those lines are the LEGACY runner's n-gram-GPU async D2H
buffer set, gated on `speculative_config.use_ngram_gpu()`, not a per-req_state
count paralleling `states.draft_tokens`. The new runner keeps no such count and
reads each row's draft length off the scheduler's own
`scheduled_spec_decode_tokens` (`vllm/v1/worker/gpu/model_runner.py:941-949`);
we cannot, because under async scheduling that list is `-1` placeholders and how
many drafts this runner actually proposed is precisely what the fill needs. The
claim is dropped rather than re-anchored. It is zeroed for a slot
at admission, moves with the request through `condense`, swaps through
`swap_states`, and is sized by the req_state POOL rather than by `num_reqs` —
which is the sizing that makes the CUDA scatter's unbounded row read safe, the
choice this spec's `## Owed` told the wave to make. One producer writes it,
`GPUModelRunner::set_draft_tokens`, which every `propose_drafts*` arm now ends
in; it also keeps `pending_drafts_` for the out-of-band pull, exactly as upstream
writes `req_states.draft_tokens` and the `DraftTokensHandler` together
(model_runner.py:1548,1553-1556).

Two consumers read it. The async placeholder fill is REACHED and gated: it
stopped reading `pending_drafts_` (which `take_draft_token_ids` moves out, so it
could never be a residence two in-step readers share) and now applies
`FillDraftsForRow`, the fill's rule stated once in `prepare_inputs.h`. All three
combine call sites pass the buffer and its stride, so A2-1's scatter has a real
argument for the first time.

**What was mutated, because a claim without one is a reading.** Deleting the
producer's write reds `test_mtp_depth` (5 of 10 cases, exit 1) and
`test_dflash2_runner_reach` (9 of 10, exit 1), both throwing the fill's refusal
by name at `prepare_inputs.cpp` — the fill is reached from `execute_model` and
the gate can see it. Sizing the buffer one row short of the pool reds the sizing
case in `test_input_batch` on `CHECK( 21 == 24 )`, twice in one case
(`test_input_batch.cpp:299` and `:308`), and leaves `test_mtp_depth` GREEN —
which is the `num_reqs == 1` blindness made executable and the reason that
assertion is on the SIZE and not on behaviour. **The `CHECK( 3 == 24 )` this
paragraph and A2-3's commit body both quoted was the WRONG mutation's value**; 3
is what a buffer sized by `num_reqs` after a single admission yields, and one row
short of the pool yields 21. The commit body is history and cannot be repaired;
this is the correction, measured at the repair head. Taking the fill's width from
the stride instead of the placeholder count reds two `test_draft_fill` cases, one
of them the MIXED `num_reqs == 3` step.

**BOTH OF THOSE MUTATIONS PROVE A BLINDNESS, and that is the point of them.** The
sizing mutation leaves every behavioural suite green and the width mutation is
seen only by `test_draft_fill`. The CUDA scatter's unbounded row read has NO
behavioural gate anywhere in this tree, on any tier, so the sizing assertion is
the whole instrument for it and it asserts a SIZE deliberately. Do not read
either mutation as evidence that a behavioural suite would catch a resizing.

**The draft scatter is still UNREACHED, and A2-5 is the wave that reaches it.**
A2-3 supplied the buffer; it did not lift the veto, which it was scoped not to
touch. Mutation **M** at the repair head is red for a THIRD reason, and the move
is the result worth recording: `test_mtp_depth` goes to 9 of 10 passed / 118
assertions (against 5 of 10 failed / 29 assertions at the A2-2 head), `async
input combine:` does not appear in the output at all, and the single failure is
the fill refusing because `sample_tokens_async`'s DECODE arm never proposes. The
combine now accepts a verify step. Both remaining gaps are `## Owed` entries
above.

The refusal M hits has MOVED AGAIN with the fill's freshness repair below, and
the count has not: it is now `runner.cpp`'s restored `async draft fill: no drafts
proposed for request 'req' (placeholders scheduled without a matching propose)`
rather than `prepare_inputs.cpp`'s `proposed 0 drafts but the scheduler placed 2
placeholders`. Same case, same 9 of 10 / 118 assertions / exit 1, and the new
message names the decode-arm gap directly instead of describing its downstream
symptom.

**A2-3's THIRD review round is repaired on the same branch, and the wave claims
no throughput result.** Gate G4 has run on the operator's device — 96.01% busy on
the decode GPU, against this spec's own threshold that a reading at or above 95%
refutes the throughput half outright — and A2-4 and A2-5 are STOPPED on it. The
round-3 repair took no lease and measured no device; every number below is from
the CPU tier. A2-3 lands for structural value only: it discharges A2-1's
`## Owed` and removes a reason-A hazard, and nothing in it is offered as a step
toward the row's 11% gap.

Three of the round-3 findings changed the code.

- **The ledger's consume was fail-OPEN and is now tied to the STEP.**
  `proposed_drafts_.Consume()` sat inside
  `if (use_async_scheduling_ && !sched_spec->empty())`, so a step with no
  placeholders consumed nothing while `Record` stayed unconditional on them. Step
  N proposes, step N+1 has no placeholders and never proposes (the async decode
  arm, `## Owed` #2920), step N+2 has placeholders and `IsFresh` answers off step
  N — a propose two steps old spliced into a later step, which
  `prepare_inputs.h` asserts the design makes unrepresentable. The mechanism it
  replaced never behaved that way: `take_draft_token_ids` moves `pending_drafts_`
  out on EVERY deferred-batch step, gated on `check_for_draft_tokens_` alone
  (`core_proc.cpp:234`). The loop and the consume are now one function,
  `FillDraftsForStep`, called on every async-scheduling step INCLUDING the empty
  one. **Nothing distinguished the two placements before**: the fail-open shape
  left `test_draft_fill`, `test_mtp_depth`, `test_dflash2_runner_reach`,
  `test_runner` and `test_engine_core_proc` all green. It now reds one
  `test_draft_fill` case — `CHECK_THROWS` at the third fill in the sequence
  Record / empty step / placeholders — and still reds nothing else, which is the
  blindness stated rather than removed.
- **The producer's draft-token PAYLOAD had no gate anywhere.** Deleting the
  `for (int c = 0; c < stride; ++c)` scatter from `set_draft_tokens`, leaving the
  count write, left all eleven G1 targets green: every draft in the process could
  be zeroed end to end and nothing red. Payload and count are now one named rule,
  `WriteDraftRow`, gated as a ROUND TRIP against `FillDraftsForRow` over a buffer
  seeded with `100*r + c`. Deleting the payload write now reds 2 of 15
  `test_draft_fill` cases and STILL reds nothing else (7 other targets measured
  green under it), so the round trip is the whole instrument. Deleting the COUNT
  write still reds `test_mtp_depth` 5 of 10 and `test_dflash2_runner_reach` 9 of
  10 by name at the fill's refusal, and deleting either production CALL SITE reds
  the same two suites identically — the reach is proven, the payload's EFFECT is
  not observed anywhere but in the round trip.
- **Three claims were corrected rather than defended.** "`emit=` byte-identical"
  and "only ACCEPTANCE falls" are true over the four rows the table lists and
  false over the run; the measured numbers are in `## What was measured` above.
  The destructor's drain is compiled on the CPU tier and EXECUTED on neither,
  because `async_device_mirror()`'s body is entirely inside `#ifdef
  VLLM_CPP_CUDA` and `async_device_inputs_` is therefore always null here; the
  comment now says so and rests the drain on reading. And the two device arms do
  NOT "read one local so they cannot be edited apart": the mirror arm passes
  `dev->cu_num_logits`, which is correct only while its `stage_upload` is present
  and correctly sized.

A2-4 is the next wave, and the decode-arm gap above is what A2-5 is blocked on.
