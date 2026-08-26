# SPEC-DFLASH2 — the draft context store's capacity: sized from `max_model_len`, and a request that outgrows it falls back instead of killing the engine ([#1919](https://github.com/mudler/vllm.cpp/issues/1919))

Row: `SPEC-DFLASH2`. Issue:
[#1919](https://github.com/mudler/vllm.cpp/issues/1919). Parent waves:
[`dflash2-device-propose.md`](dflash2-device-propose.md) (the device-resident
store) and [`dflash2-draft-block-fa2.md`](dflash2-draft-block-fa2.md) (W11, the
paged seam that made the store's pages the draft block's write target).

## The finding

Measured 2026-08-25 by the #1574 long-context round, on `main` @ `9aea9efec`,
DFlash2 K=8, `--max-model-len 12288 --kv-cache-memory 6GiB --max-num-seqs 1`,
one chat request with a ~8K-token prompt:

```
vt: AppendContextKVDevice: paged store capacity exceeded (raise kDflashMaxCtxSlots)
    at src/vllm/model_executor/models/qwen3_dflash.cpp:1152
api-server: 500 … what=EngineCore encountered an issue …
```

and then EVERY later request on that server, of any size:

```
500 … [request submitted to a stopped AsyncLLM]
```

The 10-prompt short suite on the same server had completed 10/10 immediately
before, so the engine was serving normally until the long prompt arrived. The
HTTP process stayed alive and `/health`-answering while `EngineCore` was dead.

Three separate defects sit behind that one line, and this wave closes all three.

1. **A recoverable condition kills the engine.** The store's capacity refusal is
   a `VT_CHECK` thrown from inside
   `GPUModelRunner::propose_drafts_block`, which runs inside the EngineCore step.
   Nothing between the throw and `AsyncLLM` treats it as anything but a fatal
   engine error, so one oversized prompt takes the server down permanently. The
   condition is not fatal: it says only that the SPECULATOR cannot serve this
   request. The target can.
2. **The capacity is a compile-time constant unrelated to the advertised
   context.** `kDflashMaxCtxSlots = 4096`
   (`src/vllm/model_executor/models/qwen3_dflash.cpp:1006`) with
   `s->max_pages = kDflashMaxCtxSlots / kDflashPageSize` (`:1110`). The engine
   advertises and admits `--max-model-len 12288`, and would advertise the
   checkpoint's 262144, then dies mid-forward at 4096. The advertised context and
   the servable context disagree by construction.
3. **Nothing says so at startup.** `--max-model-len 12288` is accepted without
   comment. The error text asks the operator to "raise `kDflashMaxCtxSlots`",
   i.e. to recompile, and it arrives only after the server is already dead.

## Upstream anchors

Read at the parity pin `5559679229` (`.agents/upstream-sync.md`) and, for the
DFlash2 container, at its merge `b389ac2946` ("[Spec Decode] DFlash2: local
convolution + candidate selector", #52816). Paths below are upstream paths at
`5559679229`.

**Upstream's DFlash draft has no private context store and therefore no private
cap.** Its context K/V is written into the ENGINE's own paged KV cache, in a
dedicated draft KV-cache group, through the shared `BlockTables` slot mappings:

- `vllm/model_executor/models/qwen3_dflash.py:604-620` —
  `precompute_and_store_context_kv` ends in a per-layer
  `attn.impl.do_kv_cache_update(attn, all_k_final[i], all_v[i], kv_cache, slot_mapping)`.
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:400-420` — the context
  slots come from `self._context_slot_mappings[...]`, filled by
  `prepare_dflash_inputs` from `self.block_tables.input_block_tables[gid]`.

**Its capacity is therefore derived from `max_model_len`, not from a constant.**

- `vllm/v1/worker/gpu/model_runner.py:426` —
  `block_table_max_model_len = self.max_model_len`.
- `vllm/v1/worker/gpu/model_runner.py:444-446` —
  `max_num_blocks = cdiv(block_table_max_model_len, spec.block_size * self.dcp_size)`.
- `vllm/v1/worker/gpu/spec_decode/speculator.py:84` —
  `self.draft_max_seq_len = self.max_model_len`.
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:331-333` — per step,
  `self.draft_max_seq_len = min(max_seq_len + self.num_query_per_req, self.max_model_len)`.
  **This is the `+ num_query_per_req` headroom term**: the draft's (1+k) query
  block is addressed past the committed context, exactly as our store's W11
  paged route writes the block at slots `[C, C+Tq)`.

**And when the speculator cannot serve a request, upstream emits an EMPTY draft
for that request and lets the target run alone. It never raises.** Two proposers
state it in the same words:

- `vllm/v1/spec_decode/ngram_proposer.py:156-159` —
  `if num_tokens >= self.max_model_len: # Skip requests that have already reached the max model length. continue`
  (the request is dropped from `valid_ngram_requests`, so its draft comes back
  empty).
- `vllm/v1/spec_decode/suffix_decoding.py:59-62` — the same comment, spelled as
  `draft_token_ids.append([]); continue`.

Where the limit is hit inside a kernel rather than at the request level, upstream
CLAMPS and masks; it still does not raise:

- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:565` —
  `clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)`.
- `vllm/v1/worker/gpu/spec_decode/speculator.py:237` —
  `draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)`.
- `vllm/v1/spec_decode/utils.py:258,266-267` — `torch.clamp(new_positions,
  max=max_model_len - 1)` and `new_slot_mapping.masked_fill_(exceeds_max_model_len,
  PADDING_SLOT_ID)`.

So the mirrored behaviour is: **fall back to the non-speculative path for the
request the speculator cannot serve**, not refuse the request and not stop the
engine.

## Decision: fall back, and make the fallback STICKY per request

The issue's own suggestion offers "refuse THAT request (or fall back)". Upstream
gives only one answer and it is the fallback, for the reason the section above
records: a prompt inside `max_model_len` is a prompt the TARGET can serve, and
the speculator is an accelerator, not an admission control. Refusing would turn a
servable request into a 400/500 that vLLM and SGLang both answered on this box.

The fallback is **sticky for the lifetime of the request occupying the batch
slot**, and that is forced by our own accumulation invariant rather than chosen
for taste. `propose_drafts_block` keeps `dflash_ctx_len_[i]` in lockstep with the
store's `num_ctx` and asserts both against the target's committed positions
(`runner.cpp:2866,2877`). A step that declines to append breaks that lockstep, so
every later step for the same request must also decline. The flag is cleared
where the store is rebuilt — when a reused dense slot changes occupant
(`runner.cpp:2839-2846`) — so a later, shorter request on the same row speculates
normally. Upstream's own skip is monotone in the same way: `num_tokens` only
grows.

## Scope

1. **Size the store from the engine's context**
   (`qwen3_dflash.{h,cpp}`). `kDflashMaxCtxSlots` stops being the capacity.
   `Qwen3DFlashModel::ResolveCtxStoreSizing(config, max_model_len,
   num_query_per_req)` resolves it, and `MakeDeviceKVStore` takes the resolved
   slot count as a REQUIRED argument (no defaulted overload — a defaulted one
   would let a call site keep 4096 silently, which is the defect).
2. **A memory cap, because `max_model_len` alone can be absurd, and the cap is
   an AGGREGATE.** The store is a per-request, per-draft-layer bf16 pool
   `[max_pages, page, Hkv, Dh]` for K and for V, allocated eagerly at full
   `max_pages`, and the runner builds one per BATCH ROW. So the quantity the
   device actually pays is `bytes_per_request * max_num_reqs`, and
   `gpu_memory_utilization` accounts none of it. A per-request budget bounds the
   wrong number: at 256 MiB per request it is an 8 GiB peak at the
   `--max-num-seqs 32` [`docs/USAGE.md`](../../docs/USAGE.md) itself shows,
   which is the unbounded-device-residency shape
   ([#1647](https://github.com/mudler/vllm.cpp/issues/1647)) one indirection
   further out. `kDflashCtxTotalBudgetBytes` is therefore the TOTAL across
   concurrent requests and the per-request share is `total / max_num_reqs`;
   `VT_DFLASH_CTX_MAX_TOKENS` still overrides the cap in TOKENS per request.
3. **Announce the effective speculative context once, at startup**
   (`GPUModelRunner::set_dflash_draft`). One line naming the resolved limit, the
   bytes it costs per request, **the aggregate that costs at this engine's
   `max_num_seqs`, and that `gpu_memory_utilization` does not count it**,
   whether it was capped, and what happens beyond it.
4. **Fall back instead of throwing** (`propose_drafts_block`). Per request, the
   step that would overrun the store disables speculation for that request and
   emits an empty draft; the target serves it alone. A one-line notice names the
   request and the limit the first time it happens for that request.
5. **Keep the `VT_CHECK` in `ScatterProjectedContextRows`.** It stops being
   reachable from production and becomes what it should always have been: an
   internal invariant. Its text stops naming a recompile.

Out of scope: moving the draft's context K/V into the engine's own paged
allocator, which is what upstream actually does and what would remove the private
pool entirely. That is a much larger port; recorded under `## Owed`.

## Design

### The resolver

```
DflashCtxStoreSizing ResolveCtxStoreSizing(const HfConfig& draft_config,
                                           int64_t max_model_len,
                                           int64_t num_query_per_req,
                                           int64_t max_num_reqs)
```

- `bytes_per_slot = num_hidden_layers * (num_key_value_heads * head_dim) * 2 /*bf16*/ * 2 /*K and V*/`
- `want_slots  = RoundUpToPage(max_model_len + num_query_per_req)` — the
  `+ num_query_per_req` mirrors `dflash/speculator.py:331-333` and is what the
  W11 paged route needs, since it writes the block at slots `[C, C+Tq)` and
  `ClassifyDflashBlockAttn` declines the fast route when `ctx_len + tq >
  max_pages * block_size` (`qwen3_dflash_internal.h:91`).
- `budget_slots = RoundDownToPage(budget_bytes / (bytes_per_slot * max_num_reqs))`,
  floored at one page so a store can always hold at least one block. The
  `* max_num_reqs` is what makes `kDflashCtxTotalBudgetBytes` bound the DEVICE
  rather than one row of it: the runner builds one store per batch row, so a
  budget divided per request bounds a number nothing pays.
- `slots = min(want_slots, budget_slots)`; `capped = slots < want_slots`.
- `VT_DFLASH_CTX_MAX_TOKENS`, when set and positive, REPLACES `budget_slots`
  before the page rounding, per request. It is the operator's control and the
  gate's driver: it lets a CPU fixture reach the capped regime without
  allocating a budget's worth of memory. The aggregate arithmetic is then the
  operator's own, and the startup line still states the product.
- `bytes_total = bytes_per_request * max_num_reqs`, and `budget_bytes` is
  reported as the aggregate too, because that is the quantity the budget bounds.

The default is 8 GiB TOTAL, which is a CHOICE and not a measurement. It is
deliberately the aggregate the previous 256 MiB per-request shape already allowed
at the `--max-num-seqs 32` [`docs/USAGE.md`](../../docs/USAGE.md) shows, so the
default is behaviour-preserving at that documented configuration, spends less
below it and refuses to spend more above it. What changed is that the number now
bounds what the device actually holds.

The struct carries `slots`, `want_slots`, `budget_slots`, `bytes_per_slot`,
`bytes_per_request`, `max_num_reqs`, `bytes_total`, `budget_bytes` and `capped`,
so the startup line can state what it compared against what rather than printing
one number (`.agents/verification.md`, "make the instrument say what it is
measuring").

### The runner

`set_dflash_draft` resolves the sizing once from `input_batch_.max_model_len`,
`k + 1` and `input_batch_.max_num_reqs`, stores it in `dflash_ctx_sizing_`, and
prints the announcement — including the AGGREGATE and the fact that
`gpu_memory_utilization` does not count it. Every `MakeDeviceKVStore` call in
`propose_drafts_block` passes `dflash_ctx_sizing_.slots`.

`propose_drafts_block` gains, per request:

- a sticky `dflash_ctx_disabled_[i]`, cleared beside the store rebuild when the
  slot's occupant changes;
- `if (dflash_ctx_disabled_[i]) continue;` placed BEFORE the position-continuity
  and `num_ctx`-agreement checks, because a disabled row stops maintaining both;
- after those checks and BEFORE the append,
  `if (L + append + num_query_per_req > DeviceKVCapacity(store))` → set the flag,
  narrate once, `continue`.

`continue` leaves `out.draft_token_ids[i]` at its initialised `{}`, which is
already how the runner spells "this request proposes nothing this step" for a
discarded (still-prefilling) row. Under SYNCHRONOUS scheduling that empty list
is the whole fallback: `update_draft_token_ids` installs it, the scheduler
schedules no spec slots for the request next step, and it decodes one token per
step on the target alone — upstream's skip, exactly.

### The async-scheduling seam, which this wave found by measuring

The empty list is NOT available under async scheduling, and the reason is
structural. It was found by running the fallback case, not by reading:
`test_dflash2_ctx_capacity`'s capped arm came back

```
vt: async draft fill: request 'req-first' proposed 0 drafts but the scheduler
    placed 3 placeholders at src/vllm/v1/worker/gpu/runner.cpp:1410
```

and took the engine down a second way — the same outage, one seam further in.

`AsyncScheduler::update_after_schedule` assigns `num_spec_tokens_to_schedule`
placeholder drafts to EVERY non-prefill-chunk scheduled request, one step BEFORE
the propose that fills them (`src/vllm/v1/core/sched/async_scheduler.cpp:63-67`,
mirroring `async_scheduler.py:_update_after_schedule` :36-42 at the pin). Under
async, `Scheduler::update_draft_token_ids` is deliberately never called
(`src/vllm/v1/engine/core.cpp:120-123`), so the request state keeps those
placeholders and, **in this engine as it stands**, the count cannot shrink in
response to a propose. By then the scheduler has budgeted `1 + k` verify
positions for the request, and delivering fewer draft tokens than it budgeted is
what `execute_model`'s async draft fill refuses.

That last clause is a property of this tree, not of the design: upstream's worker
DOES shrink the count under async, by mutating the `SchedulerOutput` directly
rather than by routing an empty draft back through the scheduler. The next
section says which mechanism, and why this wave does not port it.

So the async arm keeps the draft's SHAPE and neutralises its CONTENT, using the
draft's own **mask token id** — a real vocabulary entry the draft block already
feeds itself, one the target effectively never emits, and one this runner's fill
can put in front of an embedding gather. Upstream's `-1` placeholder cannot be:
it is overwritten on device by `_prepare_input_ids` before anything embeds it
(`vllm/v1/worker/gpu_model_runner.py`), and this runner has no such overwrite.

**Upstream has no analogue on this path, and saying so is worth more than a
near-match.** Its DFlash draft keeps no private context store — the context K/V
goes into the engine's own paged KV cache
(`vllm/model_executor/models/qwen3_dflash.py:604-620`) — so its proposer cannot
fail for capacity and carries no fallback state at all. The nearest upstream
mechanism for a proposer that delivered FEWER drafts than the scheduler
optimistically budgeted is `update_scheduler_for_invalid_drafts`
(`vllm/v1/spec_decode/ngram_proposer_gpu.py:475-515` at pin `5559679229`, called
from `vllm/v1/worker/gpu_model_runner.py:1333-1344` and gated on
`speculative_config.use_ngram_gpu()`), and it does NOT neutralise the tokens. It
TRIMS the schedule in the worker: `num_scheduled_tokens` and
`total_num_scheduled_tokens` are decremented by `scheduled_k - valid_k`, the
request is popped out of `scheduled_spec_decode_tokens` entirely at
`valid_k == 0`, and the caller keeps `original_num_spec_per_req` beforehand so
`prev_num_draft_len` still carries the optimistic count for the rejection
correction.

**This wave deliberately does not port that trim**
([#1943](https://github.com/mudler/vllm.cpp/issues/1943), owned by
`SPEC-DFLASH2`, listed under `## Owed`). Three reasons, in order of weight.
Upstream gates the trim on `use_ngram_gpu()` and applies it to neither the eagle
nor the DFlash family, so porting it here is a GENERALISATION of an upstream
mechanism rather than a transcription of upstream's DFlash arm, and that is a
design decision with its own spec rather than a line in a bugfix. It moves the
scheduler/worker contract — `SchedulerOutput` mutated from the worker, the
scheduled-token counts, the spec-token map and the rejection correction all
together — and it needs a red-before gate on the SCHEDULE, because a token gate
cannot see it: the verify is lossless, so the emitted stream is identical whether
the request pays `1 + k` or `1`. And the alternative already recorded under
`## Owed`, moving the draft's context K/V into the engine's own paged allocator,
subsumes it: it deletes the private pool, the private cap and the fallback state
together.

**The cost of not porting it is stated rather than hidden.** A fallen-back
request keeps being scheduled `1 + k` verify positions on every later step and
accepts (essentially) none of them, so at `k = 8` it spends about 9x the target
compute per emitted token, for the rest of that request's life, where a
non-speculative request spends 1x. That is a throughput cost on the path this
campaign exists to make faster, and it is exactly what upstream's trim avoids.

**Neither arm can emit a wrong token.** The verify is lossless: a draft token is
accepted only where it equals what the target itself would have emitted, so a
neutralised draft costs acceptance and nothing else. The gate measures this
rather than asserting it — the capped and uncapped arms are compared token for
token, and they agree.

The async arm is therefore SLOWER than the sync arm for a fallen-back request,
because it still pays the (1+k) verify. That cost is the one measured above, it
belongs to the one-step-ahead reservation rather than to the neutralisation
itself, and it is what [#1943](https://github.com/mudler/vllm.cpp/issues/1943)
owes.

The condition is `L + append + Tq > capacity` rather than `> capacity` on the
append alone. When the sizing is UNCAPPED the two forms are the same statement as
upstream's, because `capacity == max_model_len + Tq` makes it exactly
`L + append > max_model_len`. When the sizing IS capped it costs one block's
worth of context and keeps the W11 fast route, which the other form would silently
drop to the bespoke kernel.

## Risks / decisions

- **A too-low cap silently costs acceptance, not correctness.** A request that
  falls back emits the target's own tokens; the verify is lossless, so only
  throughput moves. That is precisely the invisible-defect class W8 fought
  (`runner.cpp:2870-2877`), which is why the fallback narrates on stderr and why
  the startup line states the limit up front.
- **The tokens of a fallen-back request are NOT the tokens speculation would
  produce.** Speculative decoding here is token-identical to non-speculative
  decode by construction (lossless verify), so the drafts change and the OUTPUT
  does not. The gate pins that: the fallback case asserts the emitted token
  stream and the drafted blocks separately.
- **`VT_DFLASH_CTX_MAX_TOKENS` is an env knob, not a CLI flag.** It joins the
  `VT_*` family this file's neighbours already use (`VT_FA2_DFLASH_BLOCK`,
  `VT_SPEC_TRACE`). A CLI surface for it is `## Owed`.
- **The 8 GiB default budget is a choice, not a measurement.** It is stated in
  the startup line and overridable, so an operator can see it and move it. What
  is NOT a choice is that some cap exists (#1647), nor that the cap is the
  AGGREGATE: the store is per batch row, so a per-request budget bounds a number
  nothing pays, and the review of this wave found exactly that — 256 MiB per
  request is 8 GiB at `--max-num-seqs 32`, unaccounted by
  `gpu_memory_utilization` and large enough to move a concurrency ladder that
  does not know it is there.

## Tests and gates

New binary `tests/vllm/v1/spec_decode/test_dflash2_ctx_capacity.cpp`, driving the
shared `dflash2_runner_fixture.h` engine.

- **G1 — the issue, reproduced through the production front (RED before).** A
  `LoadedEngine` with `max_model_len` above 4096, a prompt above 4096 tokens,
  driven through `eng.async_engine().generate(...)`. Then a SECOND, short request
  on the SAME engine, which must succeed. On the pre-fix tree the first request
  throws `AppendContextKVDevice: paged store capacity exceeded` and the second
  throws `request submitted to a stopped AsyncLLM`; that second failure is the
  red this wave exists to remove. The assertion that matters is the SECOND
  request, not the polite handling of the first.
- **G2 — the fallback, and the server surviving it.** `VT_DFLASH_CTX_MAX_TOKENS`
  set small enough that a short prompt outgrows the store mid-generation. The
  request must COMPLETE (not error), the propose trace must stop producing blocks
  for it, and a second request must succeed.
- **G3 — the resolver, unit.** `ResolveCtxStoreSizing` derives `want_slots` from
  `max_model_len + num_query_per_req`, rounds to the page, caps at the budget,
  reports `capped` truthfully, and never returns fewer than one page.
- **G4 — the announcement.** The startup line appears exactly once per wiring and
  names the resolved limit, the AGGREGATE that limit costs at this engine's
  `max_num_seqs`, and that `gpu_memory_utilization` does not count it; in the
  capped regime it says so.
- **G5 — the budget bounds the DEVICE, not one row.** Two resolutions of the same
  geometry at `max_num_reqs` 1 and 32, both in the capped regime so the budget is
  the thing under test. Raising the concurrency must SHRINK the per-request store
  rather than multiply the aggregate; the aggregate is held to the 8 GiB total
  within one page at each concurrency; the per-request share at 32 is the 256 MiB
  the old per-request budget handed out unconditionally, which is what makes the
  default behaviour-preserving there; and a nonsense `max_num_reqs` floors at one
  instead of handing a single request the whole aggregate. A per-request budget
  answers `many.slots == one.slots`, so it reds.
- **The G1 control, measured rather than assumed.** A 4300-token prompt is above
  the old 4096 cap; a 4000-token prompt on the IDENTICAL engine configuration is
  below it. The 4000-token run passes 13/13 on the pre-fix tree and the
  4300-token run dies, so the discriminator is the cap and not the prompt's size,
  the KV pool, or the prefill budget. Reaching that control cost one wrong
  premise: at the default per-step token budget BOTH prompts stalled with no
  error at all, because a prompt needing more than one prefill step was never
  scheduled. G1 therefore names `max_num_batched_tokens` and `num_blocks`
  explicitly, so its red is the cap and nothing else.
- **The pre-fix failure is a HANG, not a throw, for the in-flight request**, and
  the instrument has to say so. `AsyncLLM::generate` blocks until a terminal
  output arrives, and the request in flight when the EngineCore dies never gets
  one. G1 therefore drains with a deadline and records `timed_out` as a state of
  its own, distinct from both `finished` and `threw`; a test that hangs has
  measured nothing.
## Evidence

### Red, on the pre-fix tree

G1 at 4300 prompt tokens, through `LoadedEngine` -> `AsyncLLM` -> `EngineCore` ->
`GPUModelRunner::propose_drafts_block`:

```
first request:  threw: EngineCore encountered an issue. [vt: AppendContextKVDevice:
                paged store capacity exceeded (raise kDflashMaxCtxSlots) at
                src/vllm/model_executor/models/qwen3_dflash.cpp:1152]
second request: threw: EngineCore encountered an issue. [request submitted to a
                stopped AsyncLLM]
```

That second line is #1919's own symptom and the assertion this wave exists to
turn green. The same binary, same engine configuration, at 4000 prompt tokens
passes 13/13 — so the discriminator is the 4096 cap, not the prompt's size, the
KV pool, or the prefill budget.

Green after: `test_dflash2_ctx_capacity` 6 cases, 77 assertions, 0 failed —
measured on the review-repair head rather than carried forward from an earlier
one. The figure this section first carried, 5 cases / 55 assertions, was taken
before `39c43be20` added assertions and before the repair added G5, so it was a
count of the parent read as a count of the head.
`ctest -R "dflash|mtp"` green on every suite that this change touches
(`test_qwen3_dflash2_draft`, `test_qwen3_dflash_decode_graph_seam`,
`test_dflash_propose`, `test_dflash2_runner_reach`,
`test_dflash2_draft_phase_trace`, `test_dflash2_argmax_guard`,
`test_mtp_depth`).

### Mutations

Applied one at a time to the working tree, each rebuilt and re-run, each restored
from a pre-taken byte copy and re-hashed. The harness aborts rather than reports
when an anchor does not match exactly once or when the mutant does not build,
because both of those read as a passing test.

| # | Mutation | Gate | Result |
|---|---|---|---|
| M1 | the fallback branch never fires | G2 | RED |
| M2 | the async arm's shape preservation removed | G2 | RED |
| M3 | `ResolveCtxStoreSizing` pinned back to 4096 | G1 + G3 | RED (both) |
| M4 | the startup announcement's text removed | G4 | RED |
| M5 | **reachability**: the resolver's answer unwired from `MakeDeviceKVStore` (call site takes a literal) | G1 | RED |
| M6 | `budget_bytes` reported from the PRE-floor count | G3 | RED |
| M7 | the budget divided per request again instead of across `max_num_reqs` | G5 | RED |
| M8 | the announcement stops stating the aggregate | G4 | RED |
| M9 | `bytes_total` stops being `bytes_per_request * max_num_reqs` | G5 | RED |
| M10 | the cap is the aggregate but `budget_bytes` REPORTS a per-request number | G5 | RED |

M3 and M5 were re-run on the review-repair head, because the resolver's signature
moved and a later commit can silently disarm an earlier commit's mutation proof:
both are still RED (M3 4/6 cases failed, M5 2/6).

**M10 was GREEN on its first pass, and that was a finding rather than a harness
fault.** G5 as first written recomputed `budget_slots * bytes_per_slot *
max_num_reqs` and compared THAT against the budget, never the reported
`budget_bytes` field — so a resolver that caps by the aggregate and reports a
per-request number passed every bound while the startup line it feeds said
256 MiB for an 8 GiB budget. That is the same "a struct field that lies to
whoever prints it" shape M6 exists for, one field further on. G5 now holds
`budget_bytes` against the product directly.

**M1 and M5 were GREEN on the first pass, and that was a real finding rather than
a harness fault.** G1 as first written asserted only that both requests finish.
With the store pinned back to 4096, a 4300-token prompt still finishes — the
FALLBACK catches it and the request decodes on the target alone. That is the
right outcome for a request that genuinely does not fit and exactly the wrong one
at `max_model_len 6144`, and it is invisible from outside because the verify is
lossless and only acceptance falls. So G1 now also asserts that the fallback did
NOT fire and that blocks were drafted, which is what makes M3 and M5 red. M1 was
simply pointed at the wrong gate: it is the FALLBACK's mutation, and the fallback
is exercised by G2, not by G1.

- **Mutations** (byte-identical restore against pre-taken hashes): the fallback
  branch, the async arm's shape preservation, the sizing derivation (pin it back
  to 4096), the reachability mutation on the production call site, and the four
  the aggregate budget adds (M7-M10).

Full gate: `scripts/agent-preflight.sh --staged`, plus `ctest` over the spec
suites. CPU only; this wave claims no GPU measurement.

## Owed

- **The GPU repro, operator-run.** An 8K-token prompt served on a DFlash2 K=8
  server at `--max-model-len 12288`, followed by a second request that succeeds
  — the exact #1574 recipe that found this. This wave claims nothing measured on
  a device; the CPU fixture proves the mechanism, not the deployment. Tracked by
  [#1919](https://github.com/mudler/vllm.cpp/issues/1919).
- **The upstream shape.** Moving the draft's context K/V into the engine's own
  paged allocator as a KV-cache group, which is what `do_kv_cache_update` plus
  `BlockTables` gives upstream and what removes the private pool, the private cap
  and this whole class of refusal. Owned by `SPEC-DFLASH2`, tracked by
  [#1919](https://github.com/mudler/vllm.cpp/issues/1919).
- **A CLI surface for the cap.** `VT_DFLASH_CTX_MAX_TOKENS` is reachable only
  through the environment. Owned by `SPEC-DFLASH2`, tracked by
  [#1919](https://github.com/mudler/vllm.cpp/issues/1919).
- **The schedule TRIM for a fallen-back request.** Under async scheduling this
  wave neutralises the draft's content and keeps its shape, so a fallen-back
  request keeps paying a full `1 + k` verify at ~zero acceptance for the rest of
  its life — about 9x the target compute per emitted token at `k = 8`. Upstream
  trims the schedule in the worker instead
  (`update_scheduler_for_invalid_drafts`,
  `vllm/v1/spec_decode/ngram_proposer_gpu.py:475-515`, gated on
  `use_ngram_gpu()`). Not ported here because it generalises an upstream
  mechanism onto a family upstream does not apply it to, moves the
  scheduler/worker contract, and needs a red-before gate on the SCHEDULE rather
  than on the tokens. Owned by `SPEC-DFLASH2`, tracked by
  [#1943](https://github.com/mudler/vllm.cpp/issues/1943).

## Coordination, not scope

[#1945](https://github.com/mudler/vllm.cpp/issues/1945) is adjacent to this
wave and is deliberately NOT touched by it. It reports that the DFlash2
per-request capture returns graph-baked scratch to the shared `DevicePool`
while those addresses stay live inside the captured graph
(the `st.g_logits` / `st.g_final_hidden` `reset()` groups in
`Qwen3DFlashModel`'s draft-block graph driver,
`src/vllm/model_executor/models/qwen3_dflash.cpp` — named by symbol rather than
by line, because a line anchor in a file this wave also edits goes stale inside
one pull request), and that both of
its accidental protections vanish at concurrency. It shares this wave's subject
— per-request device memory on the DFlash2 draft path — and it is a different
defect with a different owner: this wave sizes and bounds the context STORE,
#1945 is about the lifetime of CAPTURE scratch. They meet at concurrency,
because the aggregate budget above is resolved from `max_num_reqs` and #1945's
protections are the ones that disappear as `max_num_reqs` rises, so a
concurrency ladder run on this code should read the two together.

## Now

`ACTIVE` — spec committed, implementation in the same pull request.

## Stop conditions

- The mirrored behaviour is unclear at the pin: `NEEDS_DECISION`, do not invent
  an admission policy.
- The long-prompt fixture cannot run inside a CPU test budget: `NEEDS_DECISION`
  on the shape of the red, never a silent downgrade to a unit-only gate.
