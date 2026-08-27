# SPEC-DFLASH2 — prefix caching and the DFlash2 draft context

Issue: [#2042](https://github.com/mudler/vllm.cpp/issues/2042)
Row: `SPEC-DFLASH2` ([engine-matrix.md](../engine-matrix.md))
Parity pin: `5559679229bc961848b121ccdeaa8fa5d79bec98`
([upstream-sync.md](../upstream-sync.md))
**Built on `main` at `7781ce3ce`, which carries both
[#2010](https://github.com/mudler/vllm.cpp/pull/2010) (`dde045419`, issue
[#2008](https://github.com/mudler/vllm.cpp/issues/2008)) and
[#2047](https://github.com/mudler/vllm.cpp/pull/2047) (`7781ce3ce`, issue
[#2029](https://github.com/mudler/vllm.cpp/issues/2029)).** #2010 is a hard
prerequisite for correctness, not for sequencing — see `## Coordination`.

## Now

`SPEC-DFLASH2` stays `ACTIVE`. This wave changes no lifecycle state. It removes
one fatal refusal and makes `--enable-prefix-caching` usable on the speculative
path.

## The failure

`--enable-prefix-caching` with a DFlash2 draft kills EngineCore on the first
request that takes a prefix-cache hit, at concurrency 1:

```
engine-fatal: EngineCore busy loop threw: vt: propose_drafts_block: context
  position discontinuity (accumulation out of sync with the target's committed
  positions) at src/vllm/v1/worker/gpu/runner.cpp:2959
```

Every later request on that server returns 500 `[request submitted to a stopped
AsyncLLM]`. Measured on `3d895a202`, `sm_121a`, dgx:gpu0 under an `rc` lease,
DFlash2 k=8, 1024 in / 512 out: the same binary serves 8/8 with
`--no-enable-prefix-caching` and `ok=0 failed=8` with it on.

## The question this wave had to answer first: DEFECT or DETECTOR

The issue left it open whether the `VT_CHECK` is refusing a state that is
genuinely inconsistent, or whether it is itself too strict. It is the
**DETECTOR**, and three independent facts settle it.

1. **The state really is inconsistent.** With prefix caching on, a request is
   admitted with `request->num_computed_tokens` already equal to the cached
   prefix length (`src/vllm/v1/core/sched/scheduler.cpp`, the waiting-admission
   `get_computed_blocks` arm), and the worker turns that straight into absolute
   positions — `positions[t] = num_computed_tokens_cpu[r] + query_pos[t]`
   (`src/vllm/v1/worker/gpu/prepare_inputs.cpp`). The target never runs those
   tokens, so no aux hidden state exists for them, so
   `CombineAuxFeaturesDevice` produces no combined feature for them and
   `AppendContextKVDeviceRows` is never called for them. The draft's private
   store therefore holds **zero** context rows while the target has committed N.
   That is not a counter that drifted: the second `VT_CHECK` in the same block
   (`L == DeviceKVNumCtx`) agrees with the counter, because both are 0.
2. **Weakening the check cannot make the state consistent.** The draft attends
   over its own context store. Letting the propose run would speculate off an
   empty context while claiming an N-token one — the exact
   invisible-defect class (`#1838` W8) this invariant was added to make loud:
   the verify is lossless, so tokens stay correct and only ACCEPTANCE falls, and
   no token gate can see it.
3. **Upstream has no such state, because it keeps no private store.** The
   DFlash draft's context K/V goes into the engine's own paged KV cache through
   `attn.impl.do_kv_cache_update(attn, all_k_final[i], all_v[i], kv_cache,
   slot_mapping)` (`vllm/model_executor/models/qwen3_dflash.py:601-619` at the
   pin), and the slot mapping is derived from the TARGET's block table
   (`vllm/v1/spec_decode/dflash.py:145-153`, `block_table_ptr=cad.block_table_tensor`),
   with attention over `cad.seq_lens`, the full sequence length. So a
   prefix-cache hit hands the draft its context for free: the draft layers' KV
   for those blocks was written by whichever request last computed that token
   prefix, and the block hash keys on token ids, which is exactly what makes it
   reusable.

So the divergence that produces the crash is our **private, dense, non-paged
draft context store** — the same divergence that produced
[#1919](https://github.com/mudler/vllm.cpp/issues/1919),
[#1943](https://github.com/mudler/vllm.cpp/issues/1943) and
[#2008](https://github.com/mudler/vllm.cpp/issues/2008), and whose removal is
already owed under
[dflash2-ctx-store-capacity.md](dflash2-ctx-store-capacity.md) `## Owed`.

## READ THIS BEFORE MERGING: what this buys, and what it costs

This is not a free TTFT win, and the cost lands on the axis the campaign cares
about most.

**It buys** prefix caching's time-to-first-token half, which is where the win is:
the target skips the cached prefill. That is what makes our arm comparable to
SGLang's radix-cache-enabled arm at all.

**It costs** the decode half for every request that takes a hit. Such a request
**stops speculating for the rest of its life**. On a workload with a shared
system prompt — which is the workload prefix caching exists for, and the workload
`--scheduling-policy lpm` is designed to exploit — that is most requests after
the first. So in effect this makes prefix caching and DFlash2 **mutually
exclusive**: you get the TTFT of one or the throughput of the other, per request,
and the engine no longer dies choosing.

**It can therefore be net-negative on output throughput** for a served workload
with high cache-hit rates, and nobody should merge it expecting otherwise. What
it unambiguously fixes is that the configuration is currently a **crash**.

**The re-baselined numbers say which axis this lands on, and it is the worse
one.** At c=1 our TTFT is **1332 ms** against vLLM's **1331** and SGLang's
**940**: we are already at vLLM parity on TTFT, so prefix caching is not closing
a gap against vLLM — it is the only route toward SGLang's number, and SGLang
reaches it with its radix cache ON and without paying what this costs. Meanwhile
at c=8 we run **63.3 out tok/s against vLLM's 80.0**, both arms with ~5%
run-to-run spread, so that **21% throughput deficit is real and controlled**.
This change spends throughput to buy TTFT, on the axis where the deficit already
is. That is the argument for measuring both numbers before drawing any
conclusion from it, and the argument for the paged store owed below being the
answer rather than this.

The trade is announced per request on stderr, recorded in `docs/FEATURES.md`, and
removed by the paged context store owed under #1919. The device measurement that
would quantify it is owed below and cannot be taken here.

A cheaper partial that would keep speculation — a TRUNCATED draft context
anchored at the cache boundary, with a per-request position base — is recorded
under `## Owed` and deliberately **not** taken, because it moves draft acceptance
and acceptance is not measurable without a device.

## Scope

**IN.** A request whose target-side context begins beyond anything the draft
store has ever held, and can never obtain, stops speculating and runs on the
target alone, announced by name. Nothing else changes.

**OUT, and named so nobody reads this as covering them.**

- [#2008](https://github.com/mudler/vllm.cpp/issues/2008) /
  [#2010](https://github.com/mudler/vllm.cpp/pull/2010) — the row-indexed draft
  context and the batch permutations that invalidate it. That is this wave's
  **prerequisite**, not its scope; see `## Coordination`.
- [#2029](https://github.com/mudler/vllm.cpp/issues/2029) /
  [#2047](https://github.com/mudler/vllm.cpp/pull/2047) — the speculation-OFF
  half. Prefix caching used to fail there too, separately, with `cudaMalloc:
  operation not permitted when stream is capturing`, because
  `PreGrowForCapture` sat behind `if (dbuf)`. **That landed as `7781ce3ce` and is
  in this wave's base**, so the two halves are now complementary rather than
  parallel: #2047 makes prefix caching survive with speculation off, this wave
  makes it survive with speculation on, and **neither alone makes
  `--enable-prefix-caching` usable**.
- [#2028](https://github.com/mudler/vllm.cpp/issues/2028) — the c>=8 illegal
  memory access.
- Moving the draft context into the paged allocator. That is the upstream shape
  and the real repair; it is owed under
  [dflash2-ctx-store-capacity.md](dflash2-ctx-store-capacity.md) and tracked by
  [#1919](https://github.com/mudler/vllm.cpp/issues/1919).

## Design

One change, in `GPUModelRunner::propose_drafts_block`
(`src/vllm/v1/worker/gpu/runner.cpp`).

#2010 replaces four row-indexed arrays with `dflash_ctx_`, an
`unordered_map<request id, DflashReqCtx>` pruned each step against
`InputBatch`'s own membership. Its resolve loop opens each row with

```cpp
const bool first_sight = c.store == nullptr;
```

**That predicate is the whole of this wave's foundation.** "This runner has never
held context for this request" is precisely the question the cache-hit
classification needs, and under the row-indexed form it could not be asked: a
request the batch had merely MOVED presented to `propose_drafts_block` exactly as
a never-seen request does — no context, target already past position 0. Keyed by
request id, the moved request keeps its entry and only a genuinely first-seen
request reaches the classification.

Inside that branch, the request's first committed position this step is read from
`step.positions` (not from `num_computed_tokens_cpu`, which the spec-decode arm of
`execute_model` corrects in place before this runs; `step.positions` is what the
invariant compares against and is therefore the same fact). If it is `> 0`, the
missing features are not in this step and were never computed, so no future step
can supply them: the speculator cannot serve this request.

Upstream's answer for a proposer that cannot serve a request is an EMPTY draft
and the target running alone — `vllm/v1/spec_decode/ngram_proposer.py:150-159`
and `vllm/v1/spec_decode/suffix_decoding.py:55-62` both `continue` past the
request rather than raise. That is already this file's `DflashReqCtx::disabled`
mechanism, landed for #1919, including its async arm. This wave sets the same
flag from a second place, with its own message naming prefix caching and its
cost, and reuses the existing fallback for everything after that.

The flag is set once, at admission, and stays for the request's life, because
the missing prefix features are gone for good.

### What the invariant keeps

Both `VT_CHECK`s stay exactly as they are, and stay armed for every row this
classification does not claim:

| state | after this change |
|---|---|
| first-seen request, prefix-cache or connector hit | falls back, announced |
| request moved by `condense` / `swap_states` | keeps its context (#2010); classification not reached |
| request resumed after preemption | falls back, announced — see `## Risks` |
| accumulation drift on a maintained row | **fatal**, unchanged |
| append dead / double-run (`L != DeviceKVNumCtx`) | **fatal**, unchanged |
| context outgrew the store (#1919) | falls back, unchanged |

## Risks

- **Masking.** The whole risk is that the fallback swallows a state that should
  be fatal. #2010's request-id keying removes the one confusable state, and G5 +
  mutation M3 make that executable rather than argued: drop the freshness gate
  and G5 reddens on `CHECK(stderr.find("was admitted with") == npos)`.
- **Preemption.** A request removed from the batch and resumed loses its entry to
  #2010's prune, so it is first-seen again with a non-zero position and falls
  back. Before this change that combination was **fatal**, so this is strictly
  better; it is named rather than claimed as designed-for, and it is another
  population the paged store would remove.
- **Silent speculation loss.** A fallen-back request under async scheduling keeps
  being scheduled `1 + k` verify positions at ~zero acceptance
  ([#1943](https://github.com/mudler/vllm.cpp/issues/1943)). This wave adds a
  second population to that known cost and does not change it.

## Tests and gates

New binary `tests/vllm/v1/spec_decode/test_dflash2_prefix_cache.cpp`, driving
the shared `dflash2_runner_fixture.h` engine through the production `AsyncLLM`
front, on CPU. Target `max_model_len` 256, 64-token pre-tokenized prompts, so the
default 32-token KV block gives a real one-block hit; the shared
`kMaxModelLen = 32` fixture cannot produce a hit at all and would gate nothing.

- **G1 — the repro.** One engine, `enable_prefix_caching = true`, the SAME
  prompt twice. Pre-fix the second request never completes, because EngineCore
  died mid-step; post-fix both finish.
- **G2 — the tokens do not move.** The same two requests with prefix caching OFF
  must produce byte-identical token ids, and the two requests within the
  caching-ON run must agree with each other. The verify is lossless, so dropping
  speculation may change what was DRAFTED and must not change what was EMITTED.
- **G3 — the fallback is scoped, not global.** The FIRST request (a cache MISS)
  must still speculate: strictly fewer drafted blocks than the control, and not
  zero. A fix that disabled speculation everywhere would pass G1 and G2 and be
  invisible to a token gate.
- **G4 — the notice** names the request that hit, not the one that missed, names
  prefix caching and the cost, fires once, and is not #1919's capacity line.
- **G5 — the CONTROL, and the anti-masking gate.** Two requests in flight on a
  two-row batch with prefix caching OFF, the short one finishing first so
  `condense` moves the long one down. It is GREEN on the #2010 base before this
  change and green after: what it asserts is that no fallback of ours is
  announced for a request the batch merely moved.

Focused gate: `ctest -R test_dflash2_prefix_cache`.
Row gate: `ctest -R "dflash|mtp"`.
Full gate: `scripts/agent-preflight.sh`.

### Mutation proof

Each mutation is applied to a scratch copy, the focused gate is rerun, and the
tree is restored and re-hashed.

| # | mutation | result |
|---|---|---|
| M1 | delete the fallback branch | `rc=1`, 3/5 cases fail — the engine dies again |
| M2 | fall back for EVERY first-seen row (`first_pos >= 0`) | `rc=1`, 3/5 cases fail — G3 and G5 catch the over-reach |
| M3 | drop the `first_sight` gate | `rc=1`, 3/5 cases fail, G5 on `CHECK(stderr.find("was admitted with") == npos)` |
| M4 | delete the production `propose_drafts_block` call site | `rc=1`, **5/5** cases fail |

M3 is the one worth reading twice. Without the freshness gate the classification
also fires for a request the batch MOVED, which is #2008's population: the loud
refusal #2010 fixed would have become a silent acceptance loss instead. That is
the masking this wave must not commit, made executable.

### Evidence

Measured in this worktree, CPU build (`-DVLLM_CPP_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`),
**base `main` `7781ce3ce`**, which carries #2010 (`dde045419`) and #2047
(`7781ce3ce`). Every figure below was re-measured on that tree after the rebuild;
none is carried across from the pre-rebuild branch.

| leg | command | result |
|---|---|---|
| red-before | `./build/tests/test_dflash2_prefix_cache` with `runner.cpp` reverted to `origin/main` | `rc=1`, 5 cases / **3 failed** (G1, G2, G4), 57 assertions / 6 failed, `context position discontinuity` **x7** |
| green-after | same, with the classification | `rc=0`, 5 cases / **5 passed**, **69 assertions** / 0 failed |
| row gate | `ctest -R "dflash\|mtp" -j 4` | `rc=0`, **34/34** |
| adjacent | `ctest -R "runner\|scheduler\|spec\|decode_graph_seam" -j 4` | `rc=0`, **33/33** |
| full gate | `scripts/agent-preflight.sh` | recorded in the pull request |

The adjacent suite deliberately includes the decode-graph seam, because #2047
moved it in this wave's base and its CI cycle was not read before that merge.

**The red leg is the load-bearing result.** #2010 is a complete fix for #2008 and
it does **not** fix this: with only its change present, the engine still throws
`propose_drafts_block: context position discontinuity` on the second request,
seven times in one run. G3 and G5 are GREEN on that base, which is what makes
them controls rather than gates — G5 in particular is green before and after, and
only mutation M3 moves it, on
`CHECK(r.stderr_text.find("was admitted with") == npos)`.

The tree was restored to
`sha256 a5eaf1ea4fa2dc56121a5472dddfd9c1ba5fc3e0e9ee0d2b8ac7d91126dedf0c` and
re-verified after every mutation. That hash is also the byte-for-byte match
between the pre-rebuild branch and the rebuilt one, which is independent
confirmation that #2010's squash landed the content the stacked branch was
gated against.

## Coordination

**#2010 is a prerequisite for CORRECTNESS, not for sequencing, and it has
landed** (`dde045419`). Applied to the row-indexed state, the same classification
cannot distinguish a prefix-cache hit from #2008's moved request, so it would
convert that crash into a silent acceptance loss — measured, not argued: on the
pre-#2010 shape, deleting the row fixup that stood in for #2010 turned exactly
that one assertion red and nothing else.

A first draft of this wave carried its own row-permutation fixup for the same
defect and filed it as
[#2066](https://github.com/mudler/vllm.cpp/issues/2066). #2066 is a **duplicate
of #2008** — same arrays, same refusal, same function, and #2008 additionally
names `swap_states`, which the duplicate missed — and it is closed as such. The
fixup is withdrawn in favour of #2010's request-id keying, which is the better
repair: it makes every batch permutation a no-op rather than one that has to be
undone, and it releases the device store when a request leaves the batch.

Because the repository squash-merges, #2010's squash destroyed this branch's
merge base. The branch was therefore **rebuilt** on `main` `7781ce3ce` and its
two commits re-applied, rather than merged, and every number in `### Evidence`
was re-measured on that tree rather than carried across.

**One caveat about the base, recorded rather than implied.** `7781ce3ce` (#2047)
was merged on explicit developer instruction without reading its CI cycle: its
local gates were green, but `cuda-fat-build` and the sanitizers were unread on
the merged tree. Nothing in this wave touches `qwen3_5.cpp` or the decode-graph
seam, and every gate below is CPU, so that gap is not this wave's to close — it
is named so a later reader knows which tree these numbers came from.

## Stop conditions

- If the CPU fixture cannot reach a prefix-cache hit at all, report
  `NEEDS_DECISION` rather than asserting the mechanism from a test that never
  entered it. It did reach one: `first_pos = 32`, one full block.
- No GPU is taken. Nothing here claims a measured deployment result; the device
  repro is owed.

## Owed

- **The GPU repro AND the throughput cost.** A DFlash2 k=8 server with
  `--enable-prefix-caching --scheduling-policy lpm` serving the 1024/512 rung at
  c=1: `ok=8 failed=0`, the TTFT against the re-baselined **1332 ms (ours) /
  1331 ms (vLLM) / 940 ms (SGLang)** three-way, **and the output tok/s against
  the 24.84 non-caching arm**, which is the number that says whether this trade
  is worth taking on a served workload.
  Note that `vllm bench serve` issues a warm-up request using the first dataset
  prompt, which is why the first benchmarked request hits the cache. Owned by
  `SPEC-DFLASH2`, tracked by [#2042](https://github.com/mudler/vllm.cpp/issues/2042).
- **The upstream shape.** The draft context K/V in the engine's paged allocator,
  which removes the private store, the private cap, the #1919 fallback and this
  one together — and with them the trade above. Owned by `SPEC-DFLASH2`, tracked
  by [#1919](https://github.com/mudler/vllm.cpp/issues/1919).
- **The truncated-context partial.** Keeping speculation for a cache-hit request
  by anchoring its draft context at the cache boundary and carrying a per-request
  position base. Correct by the lossless verify, unmeasurable without a device.
  Owned by `SPEC-DFLASH2`.

## Outcome

**Measured.** The invariant is the detector, and the evidence is in the code
rather than in the argument: the CPU fixture reproduces #2042's exact refusal on
the second request — on #2010's head, so it is not the row-indexing defect — and
the fix turns it into a served request plus one stderr line.

**The gate that matters most is not G1.** A liveness check is easy to satisfy.
G3 and G5 say the fallback is SCOPED: the verify is lossless, so a fix that
quietly stopped speculating everywhere would emit identical tokens and pass every
token gate in this tree. The block count and the announcement are the only
instruments that can see it.

**Rejected: making the check pass.** Widening or deleting the `VT_CHECK` was
considered and refused. The propose would then run against an empty context
while claiming an N-token one, which costs acceptance and nothing visible.

**Rejected for now: the truncated-context partial**, for the reason given above.

**Withdrawn: this wave's own fix for #2008.** It was written before #2010 was
visible from this base, it worked, and it is redundant. Keeping a redundant fix
to preserve a mutation proof would be the wrong trade; the proof was rebuilt on
#2010's own predicate instead, and it is stronger there.
