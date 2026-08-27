# SPEC-DFLASH2 — the mixed-step graph loss: make its population readable before anyone fixes it

**Row:** `SPEC-DFLASH2` (engine-matrix, speculative decoding) — wave W13.
**Issue:** [#2117](https://github.com/mudler/vllm.cpp/issues/2117), and it closes
[#2112](https://github.com/mudler/vllm.cpp/issues/2112) in flow because #2112 is
the instrument this wave needs and #2117 cannot be settled without it.
**Parent row spec:** [dflash2-spec-decode.md](dflash2-spec-decode.md).
**Sibling:** [dflash2-batch-propose.md](dflash2-batch-propose.md), whose E6 and
`## Owed` O7 are what #2112 blocks.
**Kind:** wave spec, committed before the implementation in the same pull
request (the W9 shape recorded under `## Git integration` for this row).

## Now

`DONE` — landed as `f9af269f9` ([#2125](https://github.com/mudler/vllm.cpp/pull/2125)).
`row/2117` is merged and deleted. #2117 itself stays OPEN: this wave built the
instrument and corrected the cost model, and mechanism 1 is unfixed and now
measurable (`## Owed` O1). Mechanism 2 has since been run and is settled against
the direction the issue expected — see `## Device runs` run 2.

## Outcome

**What was measured.** Nothing on a device by this wave's implementer. The one
device number here is the operator's, and it arrived after the wave landed: the
`VT_SPEC_GRAPH_MAX_QLENS=0` arm at c=8 reads **-34.5%**, so the query-length cap
is protecting throughput rather than costing it and #2117's mechanism 2 is
retired with the opposite sign to the one it proposed.

**What was rejected, and why.** All three of #2117's candidate fixes, each for a
different reason recorded in `## The three candidates`: (a) PIECEWISE is a real
primary-oracle gap whose speed argument the `vt::BreakableGraph` seam's own
admissibility rule refutes in the prefill regime; (b) is blocked on an unlanded
mirror gap (`decode_threshold = 1` at `runner.cpp:1691`) and its second half is a
CUDA change no CPU box can gate; (c) is a divergence from the mirror source. The
governing reason for landing none of them is arithmetic rather than taste:
mechanism 1's marginal cost is 1.0% to 2.2% against a 5.9% spread, so it can be
neither confirmed nor refuted at c=8 and a fix aimed at it would have been
unfalsifiable in both directions.

**Why the defaults have their values.** `VT_GRAPH_STATS` defaults OFF so the
tree is byte-identical for anyone who does not ask, and its period is read fresh
rather than latched so the readout cannot depend on which caller ran first in a
process. `ragged_steps` is split three ways rather than counted flat because the
flat number was consistent with BOTH of #2117's own predictions and therefore
discriminated neither. No second percentage is printed beside `ragged_pct`,
because two percentages on one line is how the wrong one gets quoted — the cost
of that choice is the paragraph under `## Device runs` run 1, and it is the
cheaper of the two failures.

**One process note worth keeping.** The first attempt at the reachability
mutation deleted the call site alone, which failed the build on `-Werror` for an
unused function — and the STALE binaries then printed SUCCESS for both suites. A
mutation whose build failed reads as a passing test.

## Scope, and what this wave deliberately does NOT do

#2117 describes a real cliff and asks nobody in particular to fix it. It offers
three candidate fixes. This wave implements NONE of them, and the reason is a
correction to the issue's own arithmetic, argued in `## The cost model is wrong`
below: at the #1574 c=8 shape mechanism 1 is bounded at about **1% to 3%** of
throughput, not the 3% to 9% the issue estimates, which puts it **under** the
rung's own 5.9% resolution floor. A fix nobody can measure is a fix nobody can
review, and two of the three candidates are CUDA-path changes that cannot be
gated on a CPU box at all.

What this wave lands is the one thing that makes the question decidable, and it
is the thing the issue's own `## What would settle it` names second and marks
BLOCKED:

1. A production readout for `GraphDispatchStats` and `DflashBlockRouteStats`
   (#2112), env-gated, host-only, emitted from `GPUModelRunner::execute_model`.
2. A **three-way split of `ragged_steps`**, because the flat counter cannot
   answer the question #2117 asks. #2117 predicts admission raggedness at 3% to
   7%; it also says "a share far above 10% means something other than admission
   is making steps ragged, and #1943 is the first suspect". A single
   `ragged_steps` number is consistent with both, so it discriminates nothing.
   The split names the cause at the point the step is classified.

## The cost model is wrong, and that is this wave's first finding

#2117 multiplies frequency by the mixed step's **total** cost:

> the mixed steps are about 3% to 7% of all steps, each costing perhaps 2 to 3
> times a decode step, for about 3% to 9% extra

A mixed step at `max_num_batched_tokens = 2048` carries about 1976 prefill
tokens. Those tokens are prefill work the engine owes whatever lane it runs on;
they are not the defect. The defect's cost is the **marginal** excess of a
degraded mixed step over an undegraded one, which is two terms and only two:

- **The eager term.** The step loses its CUDA graph, so it pays kernel-launch
  overhead it would not otherwise pay. The tree's own figure is at
  `src/vllm/v1/worker/gpu/runner.cpp:2003-2004`: "at conc-64 kernel-launch
  overhead is ~24% of the decode wall". That is a c=64 figure and an upper
  bound at c=8, and it is a fraction of a DECODE step, not of a 2048-token step.
- **The attention-lane term.** The eight decode rows ride the `num_splits=1`
  prefill ladder instead of the split-KV decode lane. `dflash2-spec-as-decode.md`
  prices the whole of that at "+9 ms" on a step of about 113 ms to 122 ms, so
  about 8% of a decode step, and in a mixed step the straggler blocks overlap
  the prefill blocks that are running anyway, which can only make it smaller.

Take both terms at their unreduced decode-step values, about 24% + 8% = 32% of
a decode step, applied to 3% to 7% of steps: **1.0% to 2.2%**. Even doubling
that for a mixed step being longer than a decode step in wall time leaves the
whole mechanism under 5%, and the c=8 rung's spread is 5.9%
(`dflash2-batch-propose.md`). **Mechanism 1 is not measurable at c=8 as a
throughput delta.** It cannot be confirmed, and — this is the part that matters
for review — it cannot be *refuted* either, so a fix landed against it would be
unfalsifiable in both directions.

It is not nothing, and it grows with concurrency in both terms: the eager
penalty rises toward its 24% c=64 figure, and higher concurrency admits more
prompts per unit time so the mixed-step share rises too. That is an argument for
measuring the population directly, which is what this wave builds, and against
guessing at the fix now.

## The three candidates, and why none of them lands here

### (a) PIECEWISE fallback for a ragged step

vLLM degrades a ragged step to `PIECEWISE`, not to eager
(`vllm/v1/cudagraph_dispatcher.py:307-324` @ pin `5559679229`: `FULL` is tried
first, a miss falls to a RELAXED `PIECEWISE` key at `:313-318`, `NONE` only
after that). Ours goes fully eager, and `cudagraph_dispatch.h:204-207` already
names PIECEWISE as uncovered. So it is a genuine primary-oracle porting gap and
it is owed.

**It is not the cheap win.** Two reasons, both already written down in this
tree:

1. **The seam forbids the speed claim.** `include/vt/breakable_graph.h:21-26`
   states the rule this row would have to satisfy: "A speed claim from this seam
   is admissible only after naming a path that is BOTH currently eager AND
   currently host-bound", and it records why — "GB10 measured prefill idle
   between launches at 3.8% with GPU-busy above 96%, and the 27B prefill gap at
   92.5% non-GEMM glue GPU work". A 2048-token mixed step is the prefill regime.
   It is eager, and it is not host-bound. The half of the mixed step PIECEWISE
   would recover is the half that is already GPU-saturated.
2. **The construction cost is a campaign, not a slice.** vLLM gets its split
   from Dynamo and FX; we have no compiler. The `vt::BreakableGraph` seam exists
   (ENG-CUDAGRAPH-BREAK, nine drivers migrated) but every production driver
   opens it `kFull` and only for a uniform decode shape. A ragged step's token
   count varies per step by construction, so a PIECEWISE arm needs padded token
   buckets for ragged shapes, persistent staging for those shapes, and a break
   point at every layer's attention in each of the nine drivers. That is several
   waves against a lever the seam's own admissibility rule says is inert here.

### (b) Keep the mixed step's decode rows on the decode attention lane

This is the half with a measured number attached (+9 ms/step), and it is also a
vLLM mirror: upstream splits a mixed batch with `split_decodes_and_prefills` and
runs the decode sub-batch on the decode kernel.

**It has a precondition we do not satisfy, and that precondition is itself a
recorded mirror gap.** `GPUModelRunner::execute_model` calls
`reorder_batch_to_split_decodes_and_prefills(input_batch_, scheduler_output)`
at `runner.cpp:1691` with the DEFAULT `decode_threshold = 1`. vLLM passes
`self.reorder_batch_threshold` (`gpu_model_runner.py:1126-1130`), which
`_init_reorder_batch_threshold` (`backend.py:657-687`) raises to
`max(1, 1 + (2 if parallel_drafting else 1) * num_speculative_tokens)` for a
backend that supports spec-as-decode. We already mirror that formula —
`SpecAsDecodeReorderThreshold`, `include/vllm/v1/attention/backend.h:180-184` —
and then never feed it to the reorder. At k=8 with parallel drafting the
threshold should be 17 and is 1, so a 9-token verify row is classified
`long_extend` rather than `decode` and is interleaved with chunked-prefill
continuation rows, which leaves no decode/prefill boundary for any split to cut
at.

So (b) is: fix the threshold (host-only, small), then split the paged-attention
launch into a decode prefix and a prefill suffix (`query_start_loc` for the
decode prefix needs no rebasing; the prefill suffix needs a rebased copy and
offset query/out pointers). The second half is a CUDA dispatch change whose
failure mode is silently wrong numerics, and **this wave has no GPU**. Landing
it blind against a predicted win that sits under the measurement floor is the
trap `.agents/verification.md` exists to prevent. Owed, with the threshold fix
named as its first slice.

### (c) Separate prefill steps from decode steps under speculation

SGLang's shape (`get_next_batch_to_run` returns the prefill batch first,
`enable_mixed_chunk` asserted off under any speculative algorithm). vLLM mixes
exactly as we do (`vllm/v1/core/sched/scheduler.py:428-438`, `:471-473`,
`:667-671`, `:1055-1056`). Per AGENTS.md this is a **divergence from the mirror
source, not a porting gap**, so it needs its own justification and its own
sanction. **REJECTED for this wave**, and it should not be reconsidered before
the readout this wave lands says how large the population actually is.

## Design

### D1 — the three-way ragged split

`GraphDispatchStats::ragged_steps` counts every step no decode graph can serve
and says nothing about why. Three causes are mixed into it, and #2117's whole
frequency question is which one dominates:

| new counter | the step |
|---|---|
| `ragged_mixed_steps` | carries BOTH a prefill row and a decode/verify row. **This is mechanism 1's population, exactly.** |
| `ragged_prefill_only_steps` | carries only prefill rows. Ordinary prefill; no decode row is degraded, so it is not mechanism 1. |
| `ragged_spec_only_steps` | carries no prefill row and is still not uniform, i.e. the verify widths differ across requests. This is #1943's shape and #2117's "something other than admission". |

Invariant, asserted rather than documented:
`ragged_mixed + ragged_prefill_only + ragged_spec_only == ragged_steps`.

The classification is per request and reads only host data the step already
holds: request `i` is a decode/verify row iff its query length equals
`num_draft_tokens_per_req[i] + 1`, which is the same per-request test
`GraphEligibleQueryLen` applies, and a prefill row otherwise. `drafts_per_req`
is EMPTY on a non-speculative engine, and then `drafts_i` is 0 and a query
length of 1 is a decode row, which is the right answer for that engine too.
Cost is one pass over `query_start_loc`, no allocation.

### D2 — the readout (#2112)

`VT_GRAPH_STATS=N` (unset or `0` is OFF and byte-identical to today) emits, once
every `N` steps from `GPUModelRunner::execute_model`, two lines on stderr beside
the `[spec-phase]` family that already prints from that file:

```
[graph-dispatch] steps=… uniform=… uniform_spec=… clamped_spec=… ragged=… ragged_mixed=… ragged_prefill=… ragged_spec=… ragged_pct=… spec_as_decode=… capture_shapes=… qlen_cap_declines=…
[dflash-route] paged_seam=… block_kernel=… combined=… last_combined_q=… last_combined_k=…
```

Both families, always, when the dump is due. `[dflash-route]` printing zeros on
an engine with no DFlash draft is information — it says the lane never ran — and
it is the #2089 lesson applied to the readout: an instrument that prints only
when it moved cannot report that a lane is dead.

`qlen_cap_declines` is in the line because #2117's mechanism 2 has a free
same-binary A/B (`VT_SPEC_GRAPH_MAX_QLENS=0`) whose result is uninterpretable
without it: no movement in throughput plus a nonzero decline count means
something different from no movement plus a zero count.

The formatting is a pure function so a CPU gate can pin the field set, and the
emission is a separate call so deleting the production call site reds the
reachability case and nothing else.

## Reachability

The production entry point is `GPUModelRunner::execute_model`, the shared step
path every registered model reaches and the one the server drives. The
reachability gate drives a real `LoadedEngine` with a DFlash2 draft through
`GenerateAndCatch` and reads the lines off **real fd 2**, using the
`CaptureStderr` helper the DFlash2 runner fixture already uses for the
`VT_SPEC_TRACE` line for exactly this reason. The mutation is
`.agents/reachability.md`'s: delete the call site in `execute_model` and the
reachability case reds while the pure-formatter case stays green.

## Tests

- `tests/vllm/v1/worker/gpu/test_cudagraph_dispatch.cpp` — D1's counters and the
  sum invariant, and D2's formatter field set, from hand-built stats. Pins the
  classification arithmetic without a runner.
- `tests/vllm/v1/spec_decode/test_dflash2_runner_reach.cpp` — D2 reached from
  `execute_model` on the production path, both lines, with the numbers in the
  line agreeing with `GetGraphDispatchStats()` read after the run.

## Gates

```sh
cmake --build build --target test_cudagraph_dispatch test_dflash2_runner_reach
./build/tests/test_cudagraph_dispatch
./build/tests/test_dflash2_runner_reach
python3 scripts/agent-integration.py --base origin/main
```

Assertion counts are recorded for every leg, because `assertions: 0` is a skip
wearing a pass.

## Risks

- **R1.** The readout writes to stderr on the per-step path. Mitigated: the
  period is read ONCE into a function-local static, the default is off, and the
  due-check is one integer modulo.
- **R2.** The three-way classification runs on every ragged step whether or not
  the readout is on, because a counter that only counts when someone is looking
  is not a counter. Cost is O(num_reqs) with no allocation, on a step that is by
  construction doing at least one prefill row's work.
- **R3.** The implementer of this wave took no GPU and claims no measurement of
  their own. The `## The cost model is wrong` section is arithmetic over
  measurements already recorded in this tree, and it is labelled as such. The one
  device number in `## Device runs` is the OPERATOR's, taken on their lease and
  attributed there by date, binary and rung; recording somebody else's
  measurement is not claiming one, and losing it because no one on the CPU side
  could reproduce it would be the worse failure.

## Owed

- **O1.** #2117 mechanism 1 remains OPEN. This wave gives it an instrument and a
  corrected cost model; it does not fix it. The next step is the issue's own
  experiment 2, now runnable: read `ragged_mixed / steps` at c=4 and c=8 with
  `VT_GRAPH_STATS=100`. Recipe in `## Device runs`.
- **O2.** Candidate (b)'s first slice, the spec-aware reorder threshold
  (`runner.cpp:1691` should pass
  `SpecAsDecodeReorderThreshold(num_spec(), parallel_drafting)`), is a precise
  primary-oracle mirror gap and is NOT landed here. It changes production batch
  order for mixed speculative steps and its only consumers today are the GDN
  segmentation and the ordering contract, so it needs a hybrid-model gate this
  wave did not build. Owned by this row.
- **O3.** Candidate (a), the PIECEWISE arm, stays owed exactly where
  `cudagraph_dispatch.h:204-207` already names it.
- **O4.** #2108 still applies: no CI runner has a GPU, so every gate case here
  is a CPU case. Of the device recipes below, run 2 has been executed by the
  operator and runs 1 and 3 are unrun; none of them is gateable in CI.

## Device runs

Recorded as commands so the next holder of a lease can run them without
re-deriving anything. Each says what it discriminates. **The implementer of this
wave took no GPU and claims no measurement.** Run 2 below has since been executed
by the operator on their own lease and is recorded here with its attribution;
runs 1 and 3 are unrun.

1. **Mechanism 1's frequency and cause.** #1574 shape, ctx 2048, 1024 in /
   512 out, `--max-num-seqs 16`, DFlash2 k=8, at c=4 and c=8, with
   `VT_GRAPH_STATS=100`. Read `ragged_mixed / steps`.

   **Score `ragged_mixed / steps`, never `ragged_pct`.** That field is
   `100.0 * ragged_steps / total` (`cudagraph_dispatch.cpp`), so its numerator is
   ALL ragged steps and on a prefill-heavy rung the prefill-only bucket dominates
   it. This wave's own reachability run is the clearest case:
   `ragged=1 ragged_prefill=1 ragged_mixed=0 ragged_pct=12.5` — a line reporting
   12.5% whose honest mechanism-1 share is ZERO. The readout deliberately does
   not print a second percentage, because two percentages on one line is how the
   wrong one gets quoted; the cost of that choice is this paragraph.

   - 3% to 7% and `ragged_spec` near zero: mechanism 1 is confirmed as the
     admission cliff at the predicted size, which by `## The cost model is
     wrong` is still under the c=8 floor, so the fix has to be justified on
     mirror grounds or measured at c=32/64 instead.
   - `ragged_spec` materially nonzero: #1943 is making steps ragged
     independently of admission, and it is the larger term.
   - `ragged_mixed` near zero: mechanism 1 is refuted and #2117's first half
     closes.
2. **Mechanism 2 — RUN, and it came back with the opposite sign.** Measured by
   the operator on their own lease on 2026-08-27 20:12Z, binary `3e541640c`
   (`build19`, the D1 head of [#2102](https://github.com/mudler/vllm.cpp/pull/2102)),
   ctx 2048, 1024 in / 512 out, `--num-blocks 3744`, c=8, 32/32 ok, with
   `VT_SPEC_GRAPH_MAX_QLENS=0` injected into the server environment by a copy of
   the D1 ladder script:

   ```
   ours,2048,1024,512,8,49.92,149.71,3128.33,9027.87,143.63
   ```

   **49.92 tok/s against the same binary's default-cap 76.23, which is -34.5%**,
   5.8x the rung's own 5.9% spread. The raw rows, the artifact-gated sizes and
   both arms' provenance are on
   [#2117](https://github.com/mudler/vllm.cpp/issues/2117) as
   `issuecomment-5445437181`; this spec carries the rule, the issue carries the
   datum.

   **So the query-length cap is PROTECTING us, not costing us.** Removing the
   bound lets every distinct uniform speculative query length open its own
   capture, and that costs far more than the eager steps the cap avoids. The
   hypothesis #2117 raised — that we are silently falling out of the graph at
   c=8 because of the cap — is retired in the direction nobody expected, and
   `SpecQueryLenCap()`'s default of 2 stands.

   **THE RULE THIS REPLACES WAS INVERTED, and that is worth stating rather than
   quietly editing.** It read "nonzero declines in the default arm plus movement
   above the floor means the cap was biting", and applied literally to -34.5% —
   which is movement, far above the floor — it returns "the cap was biting",
   whose action is to widen or remove the bound. That is the precise harm the
   entry existed to prevent. The defect was budgeting only for zero or positive
   movement, so a large NEGATIVE result fell through a rule that never named a
   sign.

   **The corrected rule is sign-first.** The SIGN decides the action;
   `qlen_cap_declines` only explains the mechanism, and it was not readable from
   a server until `f9af269f9`, so the run above cannot yet say WHICH of the two
   readings holds:
   - **Throughput FALLS without the cap** (the measured case): the bound is
     load-bearing and must not be widened. Then read `qlen_cap_declines` in the
     default arm to learn why. Near zero means the cap never bit and the whole
     loss is capture churn in the unbounded arm — every speculative step captures
     at its exact `S`, never padded, over a two-slot ring. Materially nonzero
     means the cap DID refuse steps to eager and that was still the cheaper
     trade, which makes the open question the bound's VALUE and not its
     existence.
   - **Throughput RISES above the floor without the cap**: the cap was biting in
     the sense the issue meant, and the bound should be widened toward the
     measured working set.
   - **No movement**: the mechanism is retired whatever the declines say.
3. **NOT this.** Do not re-run the `mnbt=2048` against `8192` A/B recorded at
   `model_loader.cpp:1100-1103`. Its stated reason is mechanism 1 in its own
   words, so it reproduces the cliff and concludes 2048 is correct, which is how
   the defect stayed hidden. #2117 says this explicitly.
