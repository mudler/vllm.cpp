# Spec: async scheduling for the Eagle-type speculative family (SPEC-DFLASH2 W7)

Issue: [#1824](https://github.com/mudler/vllm.cpp/issues/1824). Owning row:
`SPEC-DFLASH2` (engine-matrix.md:177, `ACTIVE`). The deferral this wave
discharges was RECORDED under `SPEC-MTP`
([mtp-spec-decode.md](mtp-spec-decode.md) §2.8 "STILL DEFERRED: ... the async
draft-in-output path (`update_draft_token_ids_in_output`, scheduler.py:1959)"),
but `SPEC-MTP` closed `DONE` on 2026-07-26 with that deferral on the record, and
the row that OWES the result this wave unblocks is `SPEC-DFLASH2`: its W6 speed
gate ([#1574](https://github.com/mudler/vllm.cpp/issues/1574)) names the forced
synchronous scheduling as the largest host-side divergence at c1 spec decode
(~360 steps for 2048 tokens, every host-side scheduling cost serialized into
each step, while our acceptance per step is BETTER than both competitors').
Reopening a `DONE` row to carry new engine work would be a lifecycle edit that
buys nothing; the wave therefore lands as `SPEC-DFLASH2` W7 and this paragraph
is the required "say why".

## Scope

Mirror upstream's polarity: async scheduling stays ON when the configured
speculative method is one upstream keeps async for, and the engine runs the
draft-in-output flow — the scheduler schedules placeholder (`-1`) draft tokens
for step N+1 before step N's output is processed, and the WORKER substitutes
the real draft values it already holds. Concretely:

1. **Enable flip** (`src/vllm/entrypoints/model_loader.cpp::LoadedEngine`,
   the `async_scheduling_enabled_` initializer): drop the blanket
   `!resolved_spec_config_.has_value() &&` and refuse async only for the
   methods upstream refuses.
2. **AsyncScheduler spec placeholders**
   (`src/vllm/v1/core/sched/async_scheduler.cpp::update_after_schedule`):
   assign `request.spec_token_ids = [-1] * num_spec_tokens_to_schedule` after
   each schedule, mirroring `async_scheduler.py:24-25,43-45` — the half that
   was landed as a COUNT only (the header's own DEFERRED list names it).
3. **`Scheduler::update_draft_token_ids_in_output`**
   (`scheduler.py:2072-2107` at the pin): trim the worker's drafts to the
   scheduled count, grammar-validate when the request advances a grammar, pad
   the removed tail with `-1`, rewrite `scheduler_output.
   scheduled_spec_decode_tokens`, and record
   `scheduler_output.num_invalid_spec_tokens`.
4. **Engine-core draft polarity** (`src/vllm/v1/engine/core.cpp`): `post_step`
   pulls drafts only when async scheduling is OFF (`core.py:617`); the
   deferred-grammar branch of `step_with_batch_queue` pulls them and calls
   `update_draft_token_ids_in_output` (`core.py:718-731`). `post_step` reads the mode off a new
   `Scheduler::async_scheduling()` virtual (upstream's `self.async_scheduling`
   flag, `core.py:232`) — the resolution product IS the scheduler class, and
   `EngineCore` gains no flag.
5. **Worker fill + computed-token correction**
   (`src/vllm/v1/worker/gpu/runner.cpp::execute_model`): when async scheduling
   is on and a speculator is configured, replace the scheduler's `-1`
   placeholders with the runner's own last proposed drafts before the splice,
   and correct the scheduler's optimistic `num_computed_tokens` for the
   previous step's rejections (§Design D3/D4). The runner learns the resolved
   mode through a `set_async_scheduling` setter called by `LoadedEngine`.
6. **Async sampler routing** (`runner.cpp::sample_tokens_async`): with a
   speculator configured, degenerate to the synchronous
   `sample_tokens` wrapped in a `ReadyModelRunnerOutput` — the rejection
   sampler and the propose loop are host-synchronous today, and the
   device-resident spec sampler is recorded owed (§Owed A2), not faked.
7. **Preemption hygiene** (`scheduler.cpp::preempt_request`): clear
   `spec_token_ids` on preemption, mirroring `scheduler.py:1217-1218` — today's
   sync path carries the same latent staleness; the placeholder path makes it
   live.
8. **Rollback guard** (`scheduler.cpp::update_from_output`): skip the
   spec-rollback block while `request.async_tokens_to_discard > 0`
   (`scheduler.py:1670-1675`), so a stale in-flight frame cannot underflow the
   counters.

OUT of scope, named so the next reader can tell staging from silence:

- The sync scheduler's first-decode-step padding (`scheduler.py:827-843,1022`,
  `pad_spec_decode`) — a cudagraph-uniformity optimization, not a correctness
  requirement; stays deferred exactly as I2 left it.
- The dynamic-SD lookup (`scheduler.py:1123-1125`): `num_spec_tokens_to_schedule`
  is the flat `num_spec_tokens` here.
- The device-resident async spec sampler (upstream's optimistic
  `prev_num_draft_len` + GPU correction + on-device draft scatter,
  `gpu_model_runner.py:1356-1396,1795-1907`): §Owed A2.
- `disable_padded_drafter_batch`: we do not carry the field; upstream's
  default (padded ON) is the only arm we implement, which is also the only arm
  compatible with async.

## Upstream anchors

Two revisions, both readable in `/home/mudler/_git/vllm`:

- The parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98` (checked out; all
  bare `file:line` below are at the pin).
- The DFlash2 merge `b389ac29465b33f9e9c534df221ea3c129e9793f` (fetched object,
  read with `git show`); NOT an ancestor of the pin and the pin is not an
  ancestor of it — they are parallel heads and the pin governs wherever both
  define the behavior.

| What | Anchor |
|---|---|
| Enable polarity (default-ON unless the method is refused) | `vllm/config/vllm.py:1064-1112` at the pin: refused iff `method not in EagleModelTypes ∪ NgramGPUTypes ∪ {"dspark"}`; also refuses pooling models and `disable_padded_drafter_batch`. At `b389ac2946` (`vllm/config/vllm.py:1275-1304`) the same list gains `"draft_model"` — the pin governs, so `draft_model` stays sync here, noted in the code |
| The method families | `vllm/config/speculative.py:60-65` at the pin: `EagleModelTypes = eagle, eagle3, extract_hidden_states, MTPModelTypes ("mtp", "qwen3_5_mtp", …), DFlashModelTypes ("dflash")`. DFlash2 has NO method literal of its own at either head — the DFlash2 draft rides method `"dflash"` (`speculative.py:952-956` at `b389ac2946`), which matches OUR classification (`include/vllm/config/speculative.h` — DFlash2 is a draft-architecture distinction under `method == "dflash"`) |
| Placeholder assignment | `vllm/v1/core/sched/async_scheduler.py:14-17,20-25,36-45` (`_spec_token_placeholders`, `num_output_placeholders += num_sampled + cur_num_spec`, `request.spec_token_ids = placeholders`, "We will update the actual spec token ids in the worker process") |
| `num_spec_tokens_to_schedule` | `vllm/v1/core/sched/output.py:244,263`; set at `scheduler.py:1123-1156` |
| Scheduling placeholders (already landed at I2, re-verified live) | `scheduler.py:472-491` (max-tokens early-continue), `:504-507` (`num_tokens_with_spec + num_output_placeholders − num_computed`), `:623-639` (`num_scheduled_spec_tokens` subtracts `num_output_placeholders`) — our `scheduler.cpp:483-506,599-626` |
| Rejection rollback under async | `scheduler.py:1666-1698`: `num_computed_tokens -= num_rejected`, `num_output_placeholders -= num_rejected`, the whole block gated on `request.async_tokens_to_discard == 0` — ours has the first two (`scheduler.cpp:952-976`), lacks the discard guard |
| `update_draft_token_ids_in_output` | `scheduler.py:2072-2107` |
| Engine-core wiring | `core.py:168-170` (`check_for_draft_tokens`), `:232` (`self.async_scheduling`), `:617-620` (post_step guard), `:718-731` (deferred-grammar draft update) |
| Worker substitution (the part we adapt, §Design D3) | `gpu_model_runner.py:1795-1907` (`_prepare_input_ids` draft scatter from the device tensor), `:1356-1396` (optimistic `prev_num_draft_len` correction), `gpu_input_batch.py:505-528` ("token_ids_cpu assigned from spec_token_ids are placeholders and will be overwritten") |
| Preempt clears drafts | `scheduler.py:1203-1225` (`:1217-1218`) |
| Upstream tests | `tests/v1/core/test_async_scheduler.py:332` (`test_no_placeholder_underflow_on_discarded_spec_frame`); `tests/v1/e2e/general/test_async_scheduling.py` (`check_outputs_equal` across async ON/OFF with spec configs — the token-identity contract, GPU-model there, CPU-model here) |

## Design

**D1 — the enable condition mirrors the pin's method families.** A new
`SpeculativeConfig::async_scheduling_compatible()`
(`include/vllm/config/speculative.h`) answers the pin's
`vllm.py:1076-1087` predicate over the methods THIS engine implements:
`"mtp"`, `"dflash"`, `"dspark"` (and the `"eagle"`/`"eagle3"` strings the
config type admits) are compatible; `"ngram"` (host n-gram — upstream's async
lane is `ngram_gpu`, a different implementation we do not have) and
`"draft_model"` (refused at the pin; allowed only at `b389ac2946`, which the
pin outranks) are not. `model_loader.cpp` becomes
`async_scheduling_enabled_((!resolved_spec_config_.has_value() ||
resolved_spec_config_->async_scheduling_compatible()) && ResolveAsyncEnabled(...))`.
`MakeScheduler` forwards `speculative_config` into `AsyncScheduler` (today the
async arm drops it, which was correct only while async forced no-spec).

**D2 — the scheduler side is upstream's, verbatim.** Placeholder assignment
in `AsyncScheduler::update_after_schedule` (the count half already landed at
I2 and is live; this adds only the `spec_token_ids` assignment, guarded on
`num_spec_tokens_to_schedule > 0` so the no-speculator path stays
byte-identical); `SchedulerOutput::num_spec_tokens_to_schedule` and
`num_invalid_spec_tokens`; `update_draft_token_ids_in_output`; the
`async_tokens_to_discard` rollback guard; the preempt-time `spec_token_ids`
clear.

**D3 — the worker fill is host-side, and that is an adaptation with a named
reason.** Upstream's worker keeps the drafts as a DEVICE tensor and scatters
them into `input_ids` on device, because its propose runs without a host sync.
OUR propose (`runner.cpp::propose_drafts` and its dflash/dspark/ngram
branches) materializes the drafts on the HOST every step
(`pending_drafts_`, a `DraftTokenIds`), synchronously with sampling. The
host-resident drafts make the faithful analog of "update the actual spec token
ids in the worker process" a host-side substitution: in
`GPUModelRunner::execute_model`, under `use_async_scheduling_ && spec_on()`,
build a patched copy of `scheduled_spec_decode_tokens` whose values come from
the runner's own last `pending_drafts_` (trimmed to the scheduled count;
looking THROUGH the stash without consuming it, so `take_draft_token_ids`
still serves the engine's deferred-grammar pull), and hand THAT to the
`update_req_spec_token_ids` splice. Placeholders never reach the embedding.
A placeholder row whose drafts are missing from the stash is a refusal
(`VT_CHECK` naming the request), not a silent `-1` embed: placeholders are
only ever assigned to requests that were scheduled and sampled the previous
step (async_scheduler.py:36-45 iterates the scheduled set; preemption clears
them per D2), so a miss is a bug and must say so.

**D4 — the computed-token correction is exact where upstream's is
optimistic, same adaptation reason.** Under async the scheduler's
`CachedRequestData.num_computed_tokens` for a request whose PREVIOUS step
scheduled drafts may still include that step's rejected drafts (the rollback
runs in `update_from_output`, which under the batch queue happens after the
next `schedule()`). Upstream cannot know the true value host-side without
killing its overlap, so it assumes all-accepted and corrects on device
(`gpu_model_runner.py:1356-1396`). Our runner DID the rejection on the host
last step, so the true value is derivable structurally:
for a cached request present in the persistent batch whose previous step
scheduled drafts (recorded per-slot at splice time), the corrected
`num_computed_tokens_cpu` is `num_tokens_no_spec − 1` — the position of the
newest committed token, which equals the scheduler's value when the rollback
already ran (depth-1 `LLMEngine::step` order) and subtracts exactly
`num_rejected` when it has not (depth-2 batch-queue order). Both engine
orders are live in production (`LLMEngine::step` for vllm-bench/C-ABI-sync,
`AsyncLLM`'s `step_with_batch_queue` for the server), which is why the
correction must be structural rather than "subtract last step's rejections":
a temporal rule is wrong in whichever order it was not written for.
The relation `sent − corrected ∈ {0, prev_num_rejected}` is asserted.

**D5 — spec sampling under async stays the synchronous path.** In
`sample_tokens_async`, `spec_on()` routes to
`ReadyModelRunnerOutput(sample_tokens(grammar_output))`. This keeps rejection,
write-backs, telemetry, propose, and the device-mirror upload on the one
proven path — token-exactness with the sync scheduler is then by construction
(identical per-step inputs and identical sampling code; only scheduler
bookkeeping timing differs). The costs, stated: the sampled-id D2H overlap
(the ~3.25 ms/step capture) is not realized on spec steps, and the
`combine_sampled_and_draft_tokens` splice must be SKIPPED under
`spec_on()` in `execute_model` — the host arrays are fresh (the sync sampler
wrote them), and the combine's `num_new_sampled_tokens == 1` arithmetic would
overwrite the LAST DRAFT position of a `1+k` row with `last_sampled`
(`prepare_inputs.cpp:285-333` computes `logits_start = query_end − 1`), which
is exactly the async-input-combine corruption I5e RCA'd. Extending the combine
kernel to draft-aware `num_logits = 1 + k` is part of §Owed A2.

**D6 — what async buys with D5 in place, honestly.** The scheduler-side win
the issue names — schedule N+1 and issue its forward before step N's host-side
`update_from_output` — is real in the depth-2 loop; the GPU-idle capture on
spec steps is NOT realized until A2 lands (the propose loop's own host sync
bounds it today anyway). The CPU tier gates the CONTRACT (token identity,
placeholder arithmetic, reachability); the TPOT delta is the operator's owed
GPU A/B (§Gates G3) and no number is claimed here.

## Risks

- **R1 — placeholder/rollback double-count.** The same `num_rejected` rewinds
  `num_computed_tokens` and `num_output_placeholders`; an error leaves the
  budget formula `num_tokens_with_spec + placeholders − computed` drifting and
  either stalls the request or schedules past `max_tokens`. Gated by the
  scheduler tests (multi-step arithmetic checked to steady state) and by the
  mutation set (§Mutations M2/M3).
- **R2 — the fill substituting the wrong step's drafts.** The stash is
  overwritten every sample; a stale row would verify yesterday's drafts and
  silently deflate acceptance while staying token-correct (accept-iff-equal
  masks it). The engine-level stub test pins the VALUES the worker receives,
  not just the counts.
- **R3 — depth-1 vs depth-2 ordering.** D4's structural correction is the
  defense; the LoadedEngine identity gate runs BOTH entries (`engine()` and
  `async_engine()`).
- **R4 — GDN spec state under async.** The GDN rollback keys off
  `num_accepted_tokens` and the spec slot remap, all worker-side and exact;
  the identity gate runs on the GDN-hybrid synthetic model, so a divergence
  reds as a token mismatch.

## Tests

All CPU, red-first, focused suites named per case:

1. `tests/vllm/v1/test_async_scheduler.cpp` — placeholders: after a spec
   schedule, `request.spec_token_ids == [-1]*k` and
   `SchedulerOutput.num_spec_tokens_to_schedule == k`; next schedule verifies
   the placeholders (`1+k` tokens, placeholder values in
   `scheduled_spec_decode_tokens`) and the multi-step arithmetic reaches
   steady state under partial acceptance; the ported
   `test_no_placeholder_underflow_on_discarded_spec_frame`.
2. `tests/vllm/v1/test_scheduler.cpp` — `update_draft_token_ids_in_output`:
   trims to the scheduled count, leaves shorter rows alone, skips
   finished/unknown requests, rewrites the map in place.
3. `tests/vllm/v1/test_engine_core_proc.cpp` — a drafting `RunnerStub` under
   AsyncScheduler + depth-2: the engine does NOT pull drafts out-of-band
   (`take_draft_token_ids` count == 0 under async, > 0 under sync), the stub
   receives real draft VALUES via its own fill, and the output token stream is
   identical to the sync run of the same stub.
4. `tests/vllm/v1/spec_decode/test_mtp_depth.cpp` (the production-entry gate,
   real `LoadedEngine` over the synthetic GDN-hybrid model + real MTP head) —
   (a) a spec engine now RESOLVES async ON (the enable flip, red against the
   current forced-sync line); (b) token identity: sync arm
   (`VT_ASYNC_SCHED=0`) vs async arm (default), both `engine().generate` and
   `async_engine().generate`, exact token equality, plus the existing depth
   witnesses on every arm so a clamped drafter cannot pass as identity.
5. `tests/vllm/test_scheduler_config.cpp` /
   `tests/vllm/config/test_speculative*.cpp` —
   `async_scheduling_compatible()` truth table: mtp/dflash/dspark true,
   ngram/draft_model false.

## Gates

- **G1 (this wave, CPU):** all suites above green; full
  `scripts/agent-preflight.sh --staged` green (known-flaky exception
  `test_cpu_x86_llamacpp_floor`, #618).
- **G2 (this wave, CPU):** spec-OFF byte-identity — the no-speculator path
  through every touched file is inert (placeholder assignment guarded on
  `num_spec_tokens_to_schedule > 0`, fill guarded on `spec_on()`, post_step
  guard change is a strict narrowing).
- **G3 (OWED, GPU, operator-run):** the c1 DFlash2 TPOT A/B, async-ON vs
  `VT_ASYNC_SCHED=0`, same binary, idle box, on the #1574 workload — the
  measurement this wave exists for. NOT run here: no GPU lease is authorized
  for this session. Recorded under §Owed A1.

## Mutations (the reviewer's set, written by the implementer as required)

- **M1 (reachability):** revert the `model_loader.cpp` enable condition to
  `!resolved_spec_config_.has_value() && …` — test 4a goes red (the engine
  resolves sync). This is the production-call-site deletion for the flip.
- **M2:** delete `request->num_output_placeholders -= num_rejected` in
  `update_from_output` — scheduler steady-state case and the identity gate go
  red (budget drift).
- **M3:** delete the `async_tokens_to_discard == 0` guard — the ported
  underflow test goes red.
- **M4:** make the worker fill a no-op (splice the scheduler's map verbatim)
  — the LoadedEngine identity gate is the discriminator: it reds loudly
  (`vt: embedding: id out of range`, a `-1` placeholder reaching the embed).
  The engine-level stub test (`test_engine_core_proc`'s W7 case) does NOT red
  under this mutation: the stub performs its own fill and cannot see the
  production runner's fill.
- **M5:** delete D4's computed correction — the depth-2 identity run reds
  (positions shift by the rejected count).
- **M6:** in `update_draft_token_ids_in_output`, drop the trim — test 2 reds.

## Owed

- **A1 — the GPU A/B (G3).** TPOT/output-tput delta async-ON vs OFF at c1 on
  the #1574 DFlash2 workload, operator-run under an `rc` lease. This wave's
  claim stops at "the contract is token-exact and reachable"; no speed number
  is asserted without it.
- **A2 — the device-resident async spec sampler.** Upstream's overlap on spec
  steps needs: device-resident rejection sampling, the propose loop's drafts
  staying on device, the draft-aware `combine_sampled_and_draft_tokens`
  (`num_logits = 1 + k`), and the optimistic `prev_num_draft_len` + deferred
  correction. Owned by `SPEC-DFLASH2` follow-on; issue to be filed if G3 shows
  the host-sync bound dominating.
- **A3 — structured output × spec × async.** The multi-row grammar bitmask
  under spec decode is deferred upstream-inventory-wide (porting-inventory
  §6), and `update_draft_token_ids_in_output`'s grammar `validate_tokens` arm
  is deferred WITH it, exactly as the sync `update_draft_token_ids`'s arm
  already was (no per-request validate seam exists in this tree). The ported
  function's trim / -1-pad / `num_invalid_spec_tokens` halves are live and
  unit-tested — a worker can deliver fewer drafts than were scheduled without
  any grammar involved.
- **A4 — the `async_tokens_to_discard` producer.** No production path in this
  tree sets `async_tokens_to_discard` (`include/vllm/v1/request.h:271`) above
  zero: upstream's producer is the reset-prefix-cache force-preempt path,
  which is not ported. The W7 rollback guard's false branch
  (`scheduler.cpp` `async_tokens_to_discard == 0`) is therefore
  production-unreachable until that path lands, and a boundary mutation there
  (`== 0` → `<= 1`) survives every suite. The porter of the
  reset-prefix-cache force-preempt path owns this gate.

## Stop conditions

- The port requiring V2-runner semantics we do not have → `NEEDS_DECISION`
  naming the dependency (not hit: upstream's mechanism is V1
  `gpu_model_runner.py`, and the adaptations D3/D4 are host-side).
- Any correctness gate red that a scoped fix cannot make green without
  widening scope → `BLOCKED` with the failing case named.

## Now

Implementation LANDED in this pull request (spec committed first; commit order
is the proof). CPU tier green: the scheduler placeholder/rollback suites, the
engine-core placeholder/worker-fill contract, and the LoadedEngine token
identity across sync/async through both fronts on the synthetic GDN-hybrid
model with a real MTP head. `SPEC-DFLASH2` stays `ACTIVE`; A1 (the GPU TPOT
A/B) and A2 (the device-resident spec sampler) are the open follow-ons.
