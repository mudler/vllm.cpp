# SPEC-DFLASH2 — the draft context belongs to the REQUEST, not to the batch ROW ([#2008](https://github.com/mudler/vllm.cpp/issues/2008))

Row: `SPEC-DFLASH2`. Issue:
[#2008](https://github.com/mudler/vllm.cpp/issues/2008). Parent waves:
[`dflash2-device-propose.md`](dflash2-device-propose.md) (W8, the device-resident
per-request store and the two invariants this defect trips) and
[`dflash2-ctx-store-capacity.md`](dflash2-ctx-store-capacity.md) (#1919, the
fallback branch a naive repair here would be mistaken for).

## The finding

Measured 2026-08-26 by the operator on `integ4/1574` @ `3d137890f`, artifact-gated
(`flash_fwd=1792`, `SpecDecodeFA2Bf16=1`, FA2 manifest `[121a]`), on an idle
leased GB10. DFlash2 K=8, `--num-blocks 3744 --max-num-seqs 16 --max-model-len
8192 --no-enable-prefix-caching`, `vllm bench serve --backend openai-chat`,
input 1024 / output 512:

| rung | result |
|---|---|
| c=1 | 24.70 out tok/s, TPOT 37.90 ms, **8/8 ok** |
| c=2 | **VOID — ok=1, failed=7** |

```
engine-fatal: EngineCore busy loop threw: vt: propose_drafts_block: context position
discontinuity (accumulation out of sync with the target's committed positions)
at src/vllm/v1/worker/gpu/runner.cpp:2961
```

and every later request on that server returns 500 `[request submitted to a
stopped AsyncLLM]`. Memory was not a factor; the box had ample headroom at c=2.

The operator then ran the isolation. **With `--speculative-config` omitted and
everything else identical, the same two concurrent requests both complete**
(`req0: ok completion=32`, `req1: ok completion=32`, `fatal_lines=0`). So
batching, scheduling, the paged KV cache, the block tables and the sampler all
serve two sequences correctly. The defect is confined to the DFlash2 draft's
per-request context accumulation.

**`ok=1` is the signature of the mechanism below**, not an incidental count. Two
requests are admitted together; the first completes; the second dies on the very
next step; the six that had not started yet then meet a stopped engine.

## Why the accumulation desynchronises

`GPUModelRunner` holds the draft context in four arrays indexed by **batch row**
(`include/vllm/v1/worker/gpu/runner.h:852-870`):

```cpp
std::vector<std::shared_ptr<vllm::DflashDeviceKVStore>> dflash_kv_store_;
std::vector<int32_t> dflash_ctx_len_;
std::vector<std::string> dflash_ctx_reqid_;
std::vector<bool> dflash_ctx_disabled_;
```

A row index is **not** stable for a request's lifetime in this tree.
`InputBatch::condense` slides a live request down into the hole a finished
neighbour left (`src/vllm/v1/worker/gpu/input_batch.cpp:611-760`; the row move is
`:686-706`), and `InputBatch::swap_states` exchanges two live rows
(`:762-847`). Both permute every per-slot array they know about — `req_ids`,
`num_computed_tokens_cpu`, `num_accepted_tokens`, `last_sampled_tokens`,
`prefill_len`, the block-table rows (`:706` `block_table.move_row`), the
index-keyed sampling maps. They know nothing about the four above, because those
live in `GPUModelRunner`, not in `InputBatch`.

So when the first of two concurrent requests finishes, condense moves the
survivor from row 1 into row 0, and the survivor meets row 0's bookkeeping:

1. `dflash_ctx_reqid_[0]` still names the departed request, so the reuse test at
   `src/vllm/v1/worker/gpu/runner.cpp:2895-2906` reads a changed occupant.
2. It therefore allocates a **fresh empty store** and sets
   `dflash_ctx_len_[0] = 0` — discarding a context the survivor is still using.
3. The survivor is mid-decode at absolute position L > 0, so the invariant at
   `runner.cpp:2939-2945` sees `step.positions[rows[0]] == L_target != 0` and
   refuses.

**The invariant is correct and stays.** It is the only reason this is a loud
failure rather than a silent one: the alternative to the throw is drafting from a
context belonging to a different request. What is wrong is that the state it
guards is keyed by something the batch is free to change underneath it.

This tree already names this exact bug class for a different array, in a comment
sitting above the log that was written to solve it
(`include/vllm/v1/worker/gpu/input_batch.h:240-252`):

> Upstream needs no equivalent because it never condenses: `states.py:132`
> returns a finished request's slot to a free list and the slot index is stable
> for the request's lifetime. This log is the price of our condensed dense batch,
> not a deviation in what the state MEANS.

`last_sampled_tokens` got a `LastSampledOp` log so its device mirror could follow
a row move. The DFlash2 arrays never got the equivalent treatment, and nothing
in the tree tied them to a request.

### Why c=1 works and hides it

At concurrency 1 the only row is row 0. A request finishes, row 0 empties, the
next request is admitted into row 0, the reuse test resets, and the reset is
**correct** — the new occupant is a fresh prefill whose first position is 0, so
the invariant passes. The condense move that breaks the state only exists when a
second live request has to be slid down over a departed one.

### The `P == 1` capture gate is a consequence, not the cause

`src/vllm/model_executor/models/qwen3_dflash.cpp`, in `ForwardBlockLogitsWithDeviceKV`, admits the capture-safe
paged path only when `P == 1`; above that a fallback re-materialises each
request's context from the paged store every propose step. That is a
**performance** boundary. It is downstream of this defect — no batch ever reaches
`P == 2` for more than a step or two today, because the first row move kills the
engine. It is out of scope here and stays owed.

## Upstream

Read at `b389ac29465b33f9e9c534df221ea3c129e9793f` in `/home/mudler/_git/vllm`,
which is **beyond our parity pin `5559679229`** — beyond-pin vLLM, not a secondary
oracle. The clone is shallow and checked out at the pin, so every anchor below
was read with `git show b389ac2946:<path>` and every line number is for the blob
at that revision, not for what is on disk.

**Upstream has no host-side per-row draft context length at all.** The DFlash and
DFlash2 speculators are stateless across steps with respect to context: every
step re-derives the draft's whole KV addressing from the **target's own
`positions` array**, and writes draft KV into a paged KV cache group at the slot
addressed by that absolute position.

- The draft's context lives in a paged KV cache group sharing the target
  runner's `BlockTables`
  (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:181-193`,
  `vllm/v1/worker/gpu/spec_decode/speculator.py:217-223`). Context KV is
  re-inserted per step by `precompute_and_store_context_kv`
  (`dflash/speculator.py:434-438`).
- Addressing is by absolute position, never by a running append cursor:
  `ctx_pos = target_positions[ctx_start + j]` then
  `ctx_block_num = ctx_pos // (block_size * CP_SIZE)`
  (`dflash/speculator.py:562-590`).
- The one anchor is re-read from the target every step:
  `last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)`
  (`dflash/speculator.py:553`), with the rejected suffix trimmed at `:542-544`.
  The draft block's positions are `last_valid_pos + 1 + query_off` (`:593`).
- **Every tensor that outlives a step is indexed by the persistent request
  slot**, resolved inside the kernel as
  `req_state_idx = tl.load(idx_mapping_ptr + req_idx)`
  (`dflash/speculator.py:536`) and carried forward through `sample_idx_mapping`
  (`:639`). DFlash2's own cross-step state, `_cached_candidate_ids`, is written
  at `cache_base = (req_state * num_steps + step) * top_k`
  (`dflash2/speculator.py:95`), never at a row index. Its step-scoped
  `_selector_scores` is row-indexed and written-then-read inside one step
  (`dflash2/speculator.py:120-129`, `:213`, `:215`).
- The V2 runner those speculators live in **has no `condense` and no
  `swap_states`**; a finished request's slot returns to a free list and the slot
  is stable for the request's lifetime (`vllm/v1/worker/gpu/states.py:29,100,132`),
  with the batch-row view produced by a per-step gather
  (`vllm/v1/worker/gpu/block_table.py:143-170`).
- In the **legacy V1** runner, where rows *are* condensed, the draft's block-table
  row is moved with the request:
  `vllm/v1/worker/gpu_input_batch.py:786` `self.block_table.move_row(...)` fans
  out to every KV group including the draft's
  (`vllm/v1/worker/block_table.py:367-373`). `DFlashProposer` itself holds only
  step-scoped buffers (`vllm/v1/spec_decode/dflash.py:49-72`, `:138`, `:142`,
  `:292-299`).
- Upstream's own test already pins the row-to-slot indirection:
  `tests/v1/spec_decode/test_dflash_prepare_inputs.py:46` passes
  `idx_mapping=torch.tensor([2])` and `:139` asserts `sample_idx_mapping[:3] ==
  [2,2,2]` — a batch row deliberately mapped to a non-identity persistent slot.

**Both upstream shapes key the draft context to the REQUEST.** V2 does it with a
stable slot plus an `idx_mapping` indirection; V1 does it by moving the draft's
block-table row along with the request. Ours does neither. That is the whole
defect.

## Design

**Key the four arrays by request id.** One `std::unordered_map<std::string,
DflashReqCtx>` on the runner, holding the store, the context length and the
disabled flag; the row loop resolves a pointer per row once per step and every
existing use reads through it.

This is upstream's V2 invariant expressed in our structure, not a workaround for
it. Upstream's persistent draft state is indexed by a key that survives any
reordering of the batch; a request id is such a key here, and it is the same key
`InputBatch::req_id_to_index` already uses. Every row permutation the batch can
perform — today's `condense` and `swap_states`, and any future one — becomes a
no-op for the draft context, which is the property that makes upstream's V2
speculator indifferent to row order in the first place.

```cpp
struct DflashReqCtx {
  std::shared_ptr<vllm::DflashDeviceKVStore> store;
  int32_t ctx_len = 0;
  bool disabled = false;
};
std::unordered_map<std::string, DflashReqCtx> dflash_ctx_;
```

Three consequences, all of them simplifications:

1. The reuse test at `runner.cpp:2895-2906` **disappears**. "Has this row's
   occupant changed" was only ever a proxy for "is this state this request's";
   with the map the question cannot be asked wrongly. A first sight of a request
   id constructs the entry, which is the reset.
2. `dflash_ctx_disabled_`'s own comment — "the flag is a property of the REQUEST,
   not of the row" (`runner.cpp:2904-2905`) — becomes literally true instead of
   approximately true.
3. The **decode-first reorder** is fixed for free, and this is a second live
   trigger rather than a hypothetical one. `reorder_batch_to_split_decodes_and_prefills`
   runs UNCONDITIONALLY on every step (`runner.cpp:1324`) and swaps live rows
   through `swap_states` (`:207`) to put decode -> short_extend -> long_extend ->
   prefill in that order. It is a no-op only while the batch's arrival order
   already satisfies that ordering — which is the common case, because condense
   keeps older decoding requests at low rows and a new arrival appends at the end
   as a prefill, so `req_regions == target_regions` and no swap is emitted. That
   is why the #2008 measurement met the condense move first and not this. A batch
   whose regions are out of order — an older row still prefilling while a newer
   one decodes — does emit swaps, and on the pre-change code those swapped two
   live requests' draft contexts onto each other's rows.

   An earlier draft of this spec called that path "inert for a Qwen3 DFlash2
   target today". That was wrong, and it is corrected here rather than quietly:
   the reorder has no model-family gate at its call site.

The entries are pruned each step against `InputBatch`'s own membership, so a
finished or preempted request releases its device store on the step after it
leaves the batch. `InputBatch` is the authority on residency, not
`exec_state_.req_ids`, which lists only the rows scheduled this step.

### What was rejected

**Adopting upstream's paged shape** — registering the draft's context as a KV
cache group and letting `MultiGroupBlockTable::move_row` carry it, which is
exactly how upstream's legacy V1 path stays correct — is the right end state and
is not this change. It replaces `DflashDeviceKVStore` entirely, changes how the
draft's capacity is budgeted (#1919's sizing, and #2007's two-pool split), and
removes the private store the W8/W11 device-propose and paged-attention fast
paths are written against. It is a wave, not a bug fix, and it would land the
concurrency repair behind a rewrite. Recorded under `## Owed`.

**Permuting the arrays from a `LastSampledOp`-style log** was the other candidate
and is worse on two counts. It keeps the row-indexed representation whose only
defect is that it is row-indexed, so every future batch mutation owes it another
entry; and the existing log is drained and cleared by
`replay_last_sampled_ops` (`runner.cpp:3456-3492`, cleared at `:1537` and
`:3431`), so a second consumer would race the first for the same buffer.

## Risks

- **A masking repair passes the obvious gate.** Marking the moved row disabled
  (the #1919 fallback at `runner.cpp:2917-2921`) `continue`s *before* the
  invariant, so it makes the throw go away, and the tokens are **identical** —
  the verify is lossless, so a request that stops speculating emits exactly what
  it emitted before, only slower. A token gate cannot see the difference. The
  gate below therefore asserts that the survivor keeps proposing, and mutation A′
  measures that this refuses the repair rather than asserting that it would.
- **Per-step hashing on the propose path.** One string lookup per request per
  step, bounded by `max_num_seqs`. At the measured TPOT of 37.90 ms this is
  unmeasurable; it is stated so the ladder below can falsify it.
- **A leaked store is a device allocation.** Pruning is part of the change, not a
  follow-up.

## Tests

`tests/vllm/v1/spec_decode/test_dflash2_concurrency.cpp`, its own binary for the
reason every fixture binary here has one: `VT_SPEC_TRACE` is latched once per
process by a function-local static on the first propose.

Two concurrent requests are driven through the **synchronous production front**
(`LLMEngine::add_request` + `step()`), so the step boundaries the trace reports
are the ones the scheduler took. The neighbour is added first (row 0) and asks
for one token; the survivor is added second (row 1) and asks for ten. The
neighbour finishes first, condense slides the survivor into row 0, and the next
propose meets the defect.

Four legs:

1. **The engine does not throw.** Red-before this change with
   `propose_drafts_block: context position discontinuity`.
2. **Concurrency was actually reached** — some step reports `rows=2`. Without
   this the case would pass on a build that served the two requests one after the
   other and never exercised a move.
3. **The survivor never stops proposing** — no `[spec-propose] NO proposing rows
   this step` line (`runner.cpp:3259-3262`). This is the leg that refuses the
   fallback masking repair.
4. **It proposes at every step it is alive for** — a count, not an absence. The
   concurrent run's solo-row proposes are the solo run's minus the one step it
   shared with the neighbour.

Plus token-exactness of the survivor's output against the solo control, which is
necessary and not sufficient.

### The leg that is deliberately absent, and why

The obvious fifth leg is to compare the drafted BLOCKS either side of the move
against the solo run's, on the argument that a repair which resets the context
instead of moving it drafts from an empty context and is caught while every token
stays unchanged. **That leg was written, run, and removed: it is a tautology on
this fixture.** With the production invariant deleted and the context reset at
every row move, this draft still emits `12 12 12` at every step of both runs. The
synthetic draft's block is constant in the context — seeded-noise weights over a
24-token vocabulary, and the selector walk collapses to one id — so the
comparison asserts a constant against itself, and nine passing `CHECK`s measured
nothing. It is recorded here rather than silently dropped because the next person
to reach for that leg will find the same fixture.
(`test_dflash2_runner_reach`'s value-sensitivity case is unaffected: it moves the
drafts by changing the selector's WEIGHTS, not the context.)

What pins the survivor's context to the survivor is therefore the **two
production invariants this change leaves alone**, standing with legs 1 and 3. No
throw means that for every proposing row `positions[rows[0]] == ctx_len` **and**
`ctx_len == DeviceKVNumCtx(store)`; still proposing means the row reached those
invariants rather than skipping them down the disabled path. Together they say
the survivor proposed with a context length equal to its own committed position,
held by a store containing exactly that many rows. This is why the row's method
depends on the invariants staying, and not only as a matter of policy.

### Mutation evidence

| Mutation, built on the PRE-CHANGE code | Result |
|---|---|
| **A′ — the fallback repair.** On a position mismatch, mark the row disabled and fall back instead of asserting. Makes the throw stop; emits identical tokens. | **CAUGHT.** Case 1 goes green, and leg 3 reads `none_lines == 9` — the survivor stopped proposing for all nine remaining steps. This is the token-invisible repair, refused by measurement. |
| **B′ — the "silence the check" repair.** Delete the position invariant and let the reset context stand. | **NOT CAUGHT** by this file, which is how the tautology in leg 5 was found. Recorded under `## Owed`: the invariant is load-bearing and nothing in this row's gate holds it. |
| **Reachability.** Delete the `propose_drafts_block` call site in `propose_drafts_dflash`. | See `## Gates`. |

## Gates

```sh
cmake -S . -B build -DVLLM_CPP_BUILD_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

Focused: `ctest --test-dir build -R 'dflash' --output-on-failure`.

Reachability mutation (`.agents/reachability.md`): delete the production call
site and rerun the focused gate. The capability enters through
`LoadedEngine` -> `LLMEngine::add_request`/`step` -> `EngineCore::step` ->
`GPUModelRunner::execute_model` -> `propose_drafts_block`; no test constructs the
runner or the store by hand.

### The end-to-end measurement, and what falsifies it

The concurrency ladder at the #2008 flags: `--num-blocks 3744 --max-num-seqs 16
--max-model-len 8192 --no-enable-prefix-caching --speculative-config
'{"method":"dflash","model":"/draft","num_speculative_tokens":8}'`, `vllm bench
serve --backend openai-chat`, in 1024 / out 512, rungs c = 1, 2, 4, 8, 16.
**The operator runs this. Predicted before the run:**

| c | predicted out tok/s | predicted TPOT (ms) | vLLM | SGLang |
|---:|---:|---:|---:|---:|
| 1 | 24.7 (unchanged) | 37.9 (unchanged) | 24.36 | 25.20 |
| 2 | 38-46 | 40-48 | 38.59 | 44.93 |
| 4 | 60-78 | 45-58 | 64.25 | 77.06 |
| 8 | 75-105 | 60-85 | 80.95 | 109.24 |
| 16 | 85-135 | 90-150 | 99.87 | 142.61 |

The bands are wide on purpose and the reason is stated rather than hidden: this
change makes the rungs *exist*, and what they measure once they do is the
`P == 1` capture gate in `qwen3_dflash.cpp::ForwardBlockLogitsWithDeviceKV` (the line was `:1577` when this was written and has since moved; `.agents/porting.md` asks for the symbol, not the line) and the two-pool allocation
of #2007, neither of which this row touches. A rung that lands at the bottom of
its band is that fallback path being measured for the first time, not this fix
underperforming.

**This must NOT merge on any of these:**

- **c=1 regresses at all** — 24.7 out tok/s or TPOT 37.9 ms moving outside noise
  in the wrong direction. c=1 never takes a condense move with a live neighbour,
  so this change must be inert there. A c=1 regression means the per-step map
  lookup or the pruning scan is on a hotter path than claimed, and the claim is
  wrong.
- **Any rung still VOIDs**, on this or any other invariant. A ladder that reaches
  c=4 and dies at c=8 is a second defect this change did not find, and it stays
  open rather than merging behind a partial result.
- **c=2 comes in below c=1.** Two sequences that each go slower than one sequence
  alone would mean the draft is serialising the batch, which is a different
  defect from the one diagnosed here.
- **A rung completes with `failed > 0`**, or any `engine-fatal` line, at any
  concurrency.

A rung that lands *below its band but above c=1, with `failed == 0` and no fatal
line*, is not a falsifier: it is #2007 and the `P == 1` gate being measured, and
those are named, owned and out of scope. The distinction is exactly which claim
each result contradicts, and this row claims only that concurrency **works**.

## Owed

- The draft context as a real KV cache group carried by
  `MultiGroupBlockTable::move_row`, which is upstream's own shape on both its
  paths. Tracked by the row; not attempted here.
- The `P == 1` capture gate in
  `src/vllm/model_executor/models/qwen3_dflash.cpp::ForwardBlockLogitsWithDeviceKV`
  (recorded here as `:1577`; it has since moved, which is the anchor decay
  `.agents/porting.md` asks the symbol name to prevent). Above one proposing
  row the paged capture-safe route is refused and a fallback re-materialises each
  request's context every propose step. This is the first change that lets a
  batch reach `P > 1` at all, so it is also the first that makes this cost
  measurable.
- [#2007](https://github.com/mudler/vllm.cpp/issues/2007) — attention and
  recurrent state allocated from two pools, which is why c=32 at k=8 is
  unservable. Interacts with the ladder above and is deliberately not widened
  into.
- [**#2009**](https://github.com/mudler/vllm.cpp/issues/2009) — **the position
  invariant at `runner.cpp:2939-2945` is itself ungated.**
  Deleting it (mutation B′ above) leaves this row's own gate green, and this row
  does not close that. It is the guard the whole draft-context accumulation rests
  on and the one the operator required be kept, so "nothing would notice if it
  went" is a real gap. Gating it needs a fixture whose draft is sensitive to its
  context, which this one is not — see the absent leg above. Its own issue.

- **UNVERIFIED, and labelled so deliberately: a prefix-cache hit or a resumed
  request may trip the same invariant at c=1.** The reasoning is that a request
  admitted with `num_computed_tokens > 0` has no draft context for the tokens the
  cache supplied, so its first propose would read `step.positions[rows[0]] > 0`
  against `L == 0` and the same `VT_CHECK` would refuse. **It was not
  reproduced.** A throwaway probe on this fixture forced
  `EngineParams::enable_prefix_caching = true` (which
  `ResolveEnablePrefixCaching` honours verbatim,
  `src/vllm/entrypoints/model_loader.cpp:1075-1078`) and issued the same
  20-token prompt twice; both requests completed, nothing threw, and the engine's
  own `prefix_cache_metrics()` reported **`queries=40 hits=0`**. The cache never
  engaged, so the probe measured nothing about the hypothesis — the target here
  is a GDN hybrid, the family upstream defaults prefix caching OFF for. This
  stays a reasoned hypothesis and is NOT reported as a defect. Confirming it
  needs a decoder-only DFlash2-capable target, which this fixture is not.

  Recorded because the answer, if it is real, is upstream's empty-draft path
  (`ngram_proposer.py:156-159`) — which is also the shape of the masking branch
  this row's gate exists to refuse, so it would have to be a separate change with
  its own discriminating gate either way.

## Now

`SPEC-DFLASH2` stays `ACTIVE`.
