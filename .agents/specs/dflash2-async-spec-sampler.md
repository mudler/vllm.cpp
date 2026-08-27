# Spec: the device-resident async spec sampler (SPEC-DFLASH2, A2)

Row: `SPEC-DFLASH2`.
Issue: [#2116](https://github.com/mudler/vllm.cpp/issues/2116).
Discharges: `## Owed` **A2** of
[`spec-decode-async-scheduling.md`](spec-decode-async-scheduling.md), the wave
that turned async SCHEDULING on for the Eagle-type family and deliberately left
the runner's input and sampler halves synchronous.

**This spec commits no product code.** It is the scoping half of the split
shape, and it exists because the measurement below turned the row's premise
around: the veto at `src/vllm/v1/worker/gpu/runner.cpp:470` is not a stale
guard that a mirror argument can lift. It is load-bearing for correctness on
two independent counts, and one of them is invisible to every token gate in
this tree.

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

Explicitly out: **flipping the veto ahead of pieces 1 to 3.** The measurement
below is the reason, and it is not a judgement call.

## What was measured

Measured on this branch with `f9af269f9` merged in, CPU build
(`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=OFF`), target
`test_mtp_depth`. No GPU was taken and no device measurement is claimed
anywhere in this spec.

The mutation, call it **M**, is the veto itself, deleted at both construction
sites verbatim (`runner.cpp:470` and `:535`; `git diff --stat` = one file,
2 insertions, 2 deletions):

```diff
-  async_input_combine_ = AsyncRunnerEnvDefault() && !spec_config_.has_value() &&
+  async_input_combine_ = AsyncRunnerEnvDefault() &&
                          QueueSupportsAsyncInputCombine(queue_);
```

| run | result | exit |
|---|---|---|
| baseline, whole binary | 10 cases, 10 passed, 123 assertions | 0 |
| **M**, whole binary | 10 cases, 9 passed, 1 failed, 118 assertions | 1 |
| baseline, `-tc="*IDENTICAL*"` | 1 passed, 15 assertions | 0 |
| **M**, `-tc="*IDENTICAL*"` | 0 passed, 1 failed, 10 assertions | 1 |

The tree was restored byte-for-byte after the run:
`md5sum src/vllm/v1/worker/gpu/runner.cpp` reads
`ac6a4b61cbb9956299154020112651e7` before the mutation and after the restore,
and `git status --porcelain` reports no change to that file. The same mutation
was run once before this branch merged, at base `331eda888`, and produced the
same four rows and the same trace difference; the numbers above are the later
run, on the head this spec lands with.

### Reason B: the async sampler has no verify arm at all

**M** throws in production code, on the depth-2 front, before any verify step
is ever reached:

```
vt: async draft fill: no drafts proposed for request 'req'
(placeholders scheduled without a matching propose)  runner.cpp:1833
```

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
is the second refusal waiting behind it.

**This reason is not in the comment the veto carries.** `runner.cpp:445-455`
names only the input combine. A reader repairing that one piece would still be
several waves away from a flag that can be flipped, which is exactly the
misreading this spec exists to stop.

### Reason A: the corruption a token gate cannot see

The comment's own stated reason also still holds, and it holds in the shape the
row already warned about. `combine_sampled_and_draft_tokens`
(`src/vllm/v1/worker/gpu/prepare_inputs.cpp:285-332`) fixes
`num_logits = num_new_sampled_tokens` and writes the committed token at
`query_end - num_logits`. With `num_new_sampled_tokens == 1` that address is
`query_end - 1`, which on a verify step is the LAST DRAFT SLOT, not the
committed-token slot.

Under **M**, with `VT_SPEC_TRACE=1`, every verify block in every arm has its
last draft replaced by the previous step's emitted token:

| position | baseline `draft=[...]` | under M | previous step's `emit` |
|---|---|---|---|
| 4 | `6 18` | `6 5` | `5` |
| 5 | `11 7` | `11 3` | `3` |
| 6 | `7 0` | `7 9` | `9` |
| 7 | `5 20` | `5 18` | `18` |

**The emitted tokens never move.** The `emit=` column is identical in both
runs, and the identity assertions of the arm that carries this corruption pass.
Speculative decoding is lossless, so a wrong draft costs acceptance and nothing
else: `tests/vllm/v1/spec_decode/test_dflash_causality.cpp:19` calls this "the
single defect in this row that raises nothing", and it already fired once as
[#1366](https://github.com/mudler/vllm.cpp/issues/1366). Here it fired again,
under a mutation that a whole-binary run still reports as one failed case for
an entirely unrelated reason.

Two consequences follow for the waves below.

- The corruption is a property of the INPUT COMBINE, not of the scheduler.
  The sync-scheduler arm (`VT_ASYNC_SCHED=0`) carries it identically, because
  `async_input_combine_` is a runner-level lever that `VT_ASYNC_SCHED` does not
  reach.
- **No acceptance gate on this fixture can see it.** The synthetic MTP head in
  `test_mtp_depth` accepts nothing: every traced block reads `ns=1 acc=0` in
  the baseline as well. The acceptance signal has zero dynamic range here, so
  the draft-token comparison above, not an acceptance ratio, is the only CPU
  instrument that discriminated.

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
- **A2-3 — device-resident propose.** `pending_drafts_` becomes a device
  buffer the combine reads, and the placeholder fill at `runner.cpp:1815-1847`
  stops needing host drafts. This is the wave that makes A2-1's draft scatter
  reachable.
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

- **The G2 instrument itself.** It cannot be written against the current tree
  without either flipping the veto or landing A2-1 first, because the drafts
  that reach the verify step are correct today by construction. It is owed by
  A2-1 and is that wave's red-before test.
- **The `nsys` read (G4).** Needs `dgx:gpu0` under an `rc` lease. Not taken
  here; the task that produced this spec was explicitly denied a device.

## Now

`SPEC-DFLASH2` stays `ACTIVE`. This spec is scoping only and changes no
product behavior. The veto at `runner.cpp:470` STANDS, its comment now names
both reasons rather than one, and #2116 is recorded against it. A2-1 is the
next wave and is blocked on nothing but the decision to schedule it.
