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
2. **A memory cap, because `max_model_len` alone can be absurd.** The store is a
   per-request, per-draft-layer bf16 pool `[max_pages, page, Hkv, Dh]` for K and
   for V. At the published draft's geometry that is kilobytes per context token,
   so a 262144-token `max_model_len` would allocate about a gigabyte per
   concurrent request, and this box has OOM-rebooted from unbounded device
   residency before ([#1647](https://github.com/mudler/vllm.cpp/issues/1647)).
   The resolver therefore caps the slot count at a per-request byte budget
   (`kDflashCtxDefaultBudgetBytes`, 256 MiB), overridable in TOKENS by
   `VT_DFLASH_CTX_MAX_TOKENS`.
3. **Announce the effective speculative context once, at startup**
   (`GPUModelRunner::set_dflash_draft`). One line naming the resolved limit, the
   bytes it costs per request, whether it was capped, and what happens beyond it.
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
                                           int64_t num_query_per_req)
```

- `bytes_per_slot = num_hidden_layers * (num_key_value_heads * head_dim) * 2 /*bf16*/ * 2 /*K and V*/`
- `want_slots  = RoundUpToPage(max_model_len + num_query_per_req)` — the
  `+ num_query_per_req` mirrors `dflash/speculator.py:331-333` and is what the
  W11 paged route needs, since it writes the block at slots `[C, C+Tq)` and
  `ClassifyDflashBlockAttn` declines the fast route when `ctx_len + tq >
  max_pages * block_size` (`qwen3_dflash_internal.h:91`).
- `budget_slots = RoundDownToPage(budget_bytes / bytes_per_slot)`, floored at one
  page so a store can always hold at least one block.
- `slots = min(want_slots, budget_slots)`; `capped = slots < want_slots`.
- `VT_DFLASH_CTX_MAX_TOKENS`, when set and positive, REPLACES `budget_slots`
  before the page rounding. It is the operator's control and the gate's driver:
  it lets a CPU fixture reach the capped regime without allocating a budget's
  worth of memory.

The struct carries `slots`, `want_slots`, `budget_slots`, `bytes_per_slot`,
`bytes_per_request`, `budget_bytes` and `capped`, so the startup line can state
what it compared against what rather than printing one number
(`.agents/verification.md`, "make the instrument say what it is measuring").

### The runner

`set_dflash_draft` resolves the sizing once from `input_batch_.max_model_len` and
`k + 1`, stores it in `dflash_ctx_sizing_`, and prints the announcement. Every
`MakeDeviceKVStore` call in `propose_drafts_block` passes
`dflash_ctx_sizing_.slots`.

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
discarded (still-prefilling) row. So the fallback needs no new plumbing
downstream: an empty draft list is a plain non-speculative decode step for that
request.

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
- **The 256 MiB default budget is a choice, not a measurement.** It is stated in
  the startup line and overridable, so an operator can see it and move it. What
  is NOT a choice is that some cap exists: #1647.

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
  names the resolved limit; in the capped regime it says so.
- **Mutations** (scratch copy, byte-identical restore against pre-taken hashes):
  the fallback branch, the sizing derivation (pin it back to 4096), and the
  reachability mutation on the production call site.

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

## Now

`ACTIVE` — spec committed, implementation in the same pull request.

## Stop conditions

- The mirrored behaviour is unclear at the pin: `NEEDS_DECISION`, do not invent
  an admission policy.
- The long-prompt fixture cannot run inside a CPU test budget: `NEEDS_DECISION`
  on the shape of the red, never a silent downgrade to a unit-only gate.
