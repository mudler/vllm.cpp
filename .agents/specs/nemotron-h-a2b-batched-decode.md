# A2-B — NemotronH decodes a BATCH: the ordering contract, the mamba split, and the end of the `num_reqs <= 1` refusal

**Issue:** [#1395](https://github.com/mudler/vllm.cpp/issues/1395).
**Umbrella:** [#810](https://github.com/mudler/vllm.cpp/issues/810).
**Parent row:** `MODEL-TEXT-nemotron-h-nemotron-hfor-causal-lm`
([#517](https://github.com/mudler/vllm.cpp/issues/517)).
**Predecessor:** [`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md)
— A2-P states at `:64`, `:68`, `:81` and `:146` exactly what it left here.
**Governing spec:** [`nemotron-h-abi-e2e.md`](nemotron-h-abi-e2e.md) §1 `A2b`.
**Siblings:** [`nemotron-h-a2q1-fp8-mamba.md`](nemotron-h-a2q1-fp8-mamba.md),
[`nemotron-h-a2q2-nvfp4-moe-lmhead.md`](nemotron-h-a2q2-nvfp4-moe-lmhead.md).
**Base:** `origin/main` @ `5f68e60df22670a714f31d6362695b012b2598e2`.
**Pinned oracle:** `${VLLM_SOURCE}` = `/home/mudler/_git/vllm` @
`5559679229bc961848b121ccdeaa8fa5d79bec98`, `git rev-parse HEAD` verified while
writing this spec against the `parity-pin` block in
[`upstream-sync.md`](../upstream-sync.md), and `git remote -v` verified to be
`https://github.com/vllm-project/vllm` rather than a fork. Every `file:line`
below was re-derived in that checkout at that SHA; §Upstream chain records the
one anchor the filing issue got off by one.
**Lifecycle at this commit:** unchanged. A spec commit changes no lifecycle state
and owes no `STATUS`/`BENCHMARKS`/`NOW` write. §Records owed lists what the
implementing change owes.

**No product code is written by this spec, deliberately.** AGENTS.md permits the
split for "a large campaign benefits from agreement on the scope before
implementation waves start", and there is a second, concrete reason:
`row/…A2Q2B` is editing `nemotron_h_device.cpp` for the device `lm_head` at the
time of writing, and A2-B's surface is the same file.

---

## 0. What A2-B is, in one paragraph

`ForwardNemotronHForCausalLM` refuses every step carrying more than one request
— `src/vllm/model_executor/models/nemotron_h_registry.cpp:159-167`. A2-B lifts
that refusal. **The lift is much smaller than the refusal's wording suggests,
and this spec's first job is to say so precisely**, because A2-P did not merely
leave a placeholder: its §4.1 decision put the per-request indexing machinery in
at `num_reqs == 1`, and the runner has carried the four-way decode-first reorder
since M1.8. What is genuinely missing is narrower and sharper: **our
decode/prefill split implements the wrong one of upstream's two modes for this
architecture's backend**, nothing verifies the ordering contract the paged
forward silently assumes, and no gate in the tree has ever run this model with
two requests in one step.

A2-B is the **contract** unit. It is not a performance unit — but it is the unit
after which a throughput comparison against vLLM's production configuration
becomes *possible*, which is why #1395 files it as a blocker on every Nemotron
speed number rather than as a feature.

---

## Scope

### The G-SAFE clause, verbatim

`src/vllm/model_executor/models/nemotron_h_registry.cpp:159-167`:

```cpp
VT_CHECK(
    input.num_reqs <= 1,
    "Model architecture NemotronHForCausalLM: BATCHED decode is not ported "
    "(issue #810, .agents/specs/nemotron-h-a2p-paged-forward.md A2-B). ...");
```

| Clause | A2-B | Why |
|---|---|---|
| `input.num_reqs <= 1` | **CONSUMED — the clause is dropped** | this unit is that clause |
| `input.gdn_meta.num_spec_decodes == 0` (`nemotron_h_device.cpp:1216-1219`) | **STAYS, untouched** | the MTP head is #517 W5; a speculative row is not a batching row and A2-B must not absorb it |
| the `ModelAs<NemotronHLoadedModel>` checked downcast (`:174`) | **STAYS, untouched** | #775; unrelated to the request count |

> **The registry refusal is REPLACED, not simply deleted.** Dropping
> `num_reqs <= 1` removes the only thing standing between this forward and a
> batch it has never been run on. §Design D4 specifies what takes its place: a
> positive assertion, inside `BuildNemotronHPagedStep`, that the batch satisfies
> the decode-first ordering contract the forward already reads
> `gm.num_decodes` / `gm.num_prefills` as if it did. **A reviewer who finds the
> count clause removed and nothing asserted in its place returns FAIL.**

### In and out

| In A2-B | Out of A2-B |
|---|---|
| a mamba-flavoured decode/prefill split mirroring `mamba_attn.py:445-469` (the promotion `:445-461`, the call `:463-469`), including `treat_short_extends_as_decodes=False` and the single-token-prefill promotion | GDN's (Qwen3-Next's) existing `SplitDecodesAndPrefills` semantics, which are correct for `gdn_attn.py:213` and must stay byte-identical for that caller |
| a positive ordering assertion replacing the count refusal | any change to the runner's reorder, which already mirrors `utils.py:665` and is not this unit's to rewrite |
| the `is_prefilling` per-request signal that `split_decodes_and_prefills` needs under `treat_short_extends_as_decodes=False`, plumbed onto `v1::CommonAttentionMetadata` from data the runner already computes | `require_uniform` — **upstream's mamba backend never passes it** (§Upstream chain), so mirroring it here would be inventing a mode |
| a red-first multi-request gate through `ModelRegistry::Forward`, A/B against the same requests run singly | speculative rows (`num_spec_decodes > 0`), which stay refused — #517 W5 |
| the ported upstream tests for the split's modes, with their parameters and failure cases | the FP8 mamba arm (A2-Q1, [#940](https://github.com/mudler/vllm.cpp/issues/940) — **closed** 2026-08-16, kept only as provenance; the live pointer is PR #1289) and the NVFP4 `lm_head` arm (A2-Q2b) |
| the CUDA-graph question answered by measurement (§Design D5) | writing a NemotronH decode graph driver, and any change to `GraphEligibleQueryLen` |
| a throughput-comparison *possibility* | any throughput, latency or memory number. **A2-B records none.** The measurement is the umbrella's next unit, and quoting one from this row is how a correctness unit acquires an unearned speed claim |
| mamba prefix caching for the multi-request case only insofar as it is refused by name | implementing `mamba_cache_mode == "all"` (upstream `mamba_mixer2.py` prefix-cache branch), which A2-P also left unported |

### What A2-B explicitly does NOT deliver

Stated so the implementer cannot silently widen scope, and so a reviewer has a
list to check against:

1. **No speed number, on any axis.** Not a ratio, not a token/s, not a "for
   reference" figure in the PR body.
2. **No speculative decode.** `num_spec_decodes > 0` still refuses by name.
3. **No prefix caching for the recurrent state.**
4. **No device `lm_head` and no FP8 mamba.** The host arms stay where A2-P left
   them; the `want`/`gathered` tail is already per-row general and A2-B touches
   it only if a gate proves otherwise.
5. **No parallelism inside the mamba per-request loop.** The loop stays serial
   host compute. Making it parallel is a performance change with its own
   correctness surface and it is not this unit.
6. **No new `require_uniform` mode**, and no change to `SplitDecodesAndPrefills`
   as GDN calls it.

---

## Upstream chain

Every line re-derived at `5559679229bc961848b121ccdeaa8fa5d79bec98`, and each
asserted **unique** with a `grep -n` over its own file rather than merely
present, because a wrong line number still holds plausible unrelated code.

| Concern | Upstream `file:line` | Verified |
|---|---|---|
| decode threshold for every mamba-family backend | `vllm/v1/attention/backends/mamba_attn.py:87` — `reorder_batch_threshold: int = 1` | **CONFIRMED, and unique** — `grep -n 'reorder_batch_threshold: int = 1'` returns exactly `87` |
| how the threshold is established | `mamba_attn.py:200` — `self._init_reorder_batch_threshold(1, self.use_spec_decode)` | **CONFIRMED, and unique** — the only `_init_reorder_batch_threshold` in the file |
| the split itself | `vllm/v1/attention/backends/utils.py:564` — `def split_decodes_and_prefills(` | **CONFIRMED** — `grep -n 'def split_decodes_and_prefills'` returns exactly `564` |
| the batch ordering contract | the same docstring, `utils.py:574-575` — "The batch is expected to be ordered as: `decode → short_extend → long_extend → prefill`" | **CONFIRMED.** `:571-572` is the preceding sentence ("Assuming a reordered batch, finds the boundary between prefill and decode requests"), not this one. Note the arrow is `→`, not `->` |
| per-request counts in the metadata | `mamba_attn.py:31` `num_prefills: int`, `:33` `num_decodes: int` | **CONFIRMED, both unique** at the top-level indent |
| the call site | `mamba_attn.py:464` | **CORRECTED.** #1395 cites `:463`. `:463` is the assignment `num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (`; the call `split_decodes_and_prefills(` is on `:464`, and `464` is the only call site in the file (`:22` is the import). Off by one, and the anchor to cite is `mamba_attn.py:464-468` because the argument list is what matters |
| **the argument this architecture actually passes** | `mamba_attn.py:467` — `treat_short_extends_as_decodes=False` | **CONFIRMED.** This is the single most important anchor in the unit |
| the single-token-prefill promotion, immediately before the call | `mamba_attn.py:445-461` | **CONFIRMED, and re-derived.** `:445-448` read and assert the two inputs, `:450` `single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)`, `:452` `has_prior_state = seq_lens_cpu > 1`, `:453` combines them and `:458` flips the rows OUT of `is_prefilling` before the split runs. Each name asserted unique in the file: `single_token_prefill_rows` at `:450`/`:453` only, `has_prior_state` at `:452`/`:453` only. **An earlier draft of this spec cited `:455-462`, which is wrong in the way this table exists to catch**: the two lines quoted as evidence sit at `:450` and `:452`, outside it, and `:462` is blank |
| what `treat_short_extends_as_decodes=False` costs | `utils.py:623-625` — `:623` guards, `:624` is `assert common_attn_metadata.is_prefilling is not None`, `:625` is `is_prefill \|= common_attn_metadata.is_prefilling` | **CONFIRMED.** The `\|=` is escaped because an unescaped pipe silently splits this table cell into two columns, which it did until this repair |
| the GDN backend, for contrast | `vllm/v1/attention/backends/gdn_attn.py:213` — `split_decodes_and_prefills(m, decode_threshold=1)`, both flags left at their defaults | **CONFIRMED** |
| the reorder | `utils.py:665` `def reorder_batch_to_split_decodes_and_prefills(`, driven from `vllm/v1/worker/gpu_model_runner.py:1127` | **CONFIRMED** |
| **who PRODUCES `is_prefilling`** | `vllm/v1/worker/gpu/model_runner.py:975` — `is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np` | **CONFIRMED, and unique** — `grep -n 'is_prefilling_np ='` returns exactly `975`. Anchored because the Port map's ★ note turns on it, and an unanchored producer is one nobody is asked to check |
| the field itself, and its sibling | `vllm/v1/attention/backend.py:459-462` `is_prefilling`, `:464-468` `seq_lens_cpu_upper_bound` | **CONFIRMED.** The promotion at `mamba_attn.py:452` reads the **second** one. Our `include/vllm/v1/attention/backend.h:30-31` defers **both**; the Port map records the substitution |
| the model | `vllm/model_executor/models/nemotron_h.py` — `NemotronHMambaDecoderLayer:360`, `self.mixer = MambaMixer2(:373)`, `NemotronHAttention:409`, layer map `:532-534` | **CONFIRMED.** Nemotron-H is still in the OLD layout at this pin; it is not under `vllm/models/` |
| the mixer that consumes the split | `vllm/model_executor/layers/mamba/mamba_mixer2.py:751-767` (`num_prefills`, `has_prefill`, `has_decode`, then the decode-first `torch.split`) and `:808-812` (output split) | **CONFIRMED** |
| CUDA-graph capture is DECODE-ONLY for mamba | `mamba_attn.py:204-221` `build_for_cudagraph_capture`: `:213-219` asserts `m.max_query_len <= 1 + self.num_spec_tokens`, and the exact-equality `assert m.max_query_len == 1 + self.num_spec_tokens  # decode-only` is `:221`; `_cudagraph_support = AttentionCGSupport.UNIFORM_BATCH` at `:88` | **CONFIRMED.** The equality assert is `:221`, so a range ending at `:220` excludes the line it is cited for |

### ★ `require_uniform` — the issue's framing is half right, and the correction matters

#1395 and the dispatch both say `require_uniform` and
`treat_short_extends_as_decodes` "both change which requests count as decodes".
That is true of the **function**. It is not true of **this architecture's
caller**.

Re-derived at the pin with `grep -rn 'split_decodes_and_prefills' vllm/`, then
each hit classified by reading it (the bare-name grep also catches the two
`def`s, the imports and three prose mentions, so the raw hit count is not the
call count). **There are 16 call sites**, and `require_uniform` is passed at
**8** of them:

| Call site | `require_uniform=` |
|---|---|
| `flashinfer.py:1129` | `True` (literal) |
| `minimax_m3/common/indexer.py:287` | `True` (literal) |
| `minimax_m3/common/sparse_attention.py:242` | `True` (literal) |
| `minimax_m3/nvidia/indexer_msa.py:156` | `True` (literal) |
| `mla/indexer.py:791` | `not self.use_flattening` |
| `mla_attention.py:1952` | `self.query_len_support != QueryLenSupport.VARLEN` |
| `hpc_attn.py:186` | `self.reorder_batch_threshold > 1` |
| `sparse_mla_attention.py:177` | `self.require_uniform_decodes` |

Only **4** pass a literal `True`; the other four pass a predicate, so "passes
`require_uniform=True`" is the wrong description of half of them. The remaining
eight call sites — `mamba_attn.py:464`, `linear_attn.py:80` and `:208`,
`gdn_attn.py:213`, `turboquant_attn.py:262`, `mla/sparse_swa.py:518`,
`deepseek_v4/compressor.py:135`, `deepseek_v4/sparse_mla.py:251` — do not pass it
at all. **`mamba_attn.py` passes it nowhere**, so it holds its `False` default on
every mamba step. That conclusion is what the unit rests on, and it is unchanged
by the corrected counts.

The spec therefore says what each does and what we do about it, as required:

| Mode | Upstream default | What `mamba_attn.py` passes | A2-B |
|---|---|---|---|
| `decode_threshold` | `1` | `decode_threshold=decode_threshold`, itself `reorder_batch_threshold` = `1` unless spec decode widens it (`:200`) | **mirror the value 1.** The spec-decode widening is out of scope because speculative rows stay refused |
| `require_uniform` | `False` | **not passed ⇒ `False`** | **mirror by NOT implementing it.** Adding a mode this architecture's backend cannot reach would be an invention with no oracle, and AGENTS.md's "mirror every applicable mode" turns on *applicable*. Recorded here so the absence is a decision with a reason, not an omission. Uniformity enters this architecture through the **CUDA-graph** path instead (`mamba_attn.py:204-221`), which §Design D5 handles |
| `treat_short_extends_as_decodes` | `True` | **`False`, explicitly (`:467`)** | **mirror it.** This is the divergence, and §Our baseline shows we currently implement `True` |

### What `treat_short_extends_as_decodes=False` means, concretely

A **short extend** is a request whose query length this step is `<= threshold`
but which is *still prefilling* — a chunked prefill whose final chunk happens to
be one token, or a request the scheduler clamped. Under `True` it is counted as a
decode. Under `False` it is counted as a prefill.

For a Mamba2 mixer that distinction is not bookkeeping. A decode row enters
`causal_conv1d_update` + `selective_state_update` reading and writing the same
state slot in place; a prefill row enters `causal_conv1d_fn` +
`mamba_chunk_scan_combined_varlen` with `has_initial_states_p` deciding whether
it continues. Misclassifying a still-prefilling row as a decode runs the
single-step recurrence over a row that has more than one token of context to
absorb, and — this is the part that matters for the gate — **it produces fluent
output, not an error.**

Upstream then softens the flag in one exact direction at `:445-461`: a row that
is prefilling, has query length exactly 1, **and** has `seq_len > 1` (so it does
have prior state) is promoted back to a decode, with the comment that `ReplaySSM`
handles it as a single-token flush. The two pieces are one behaviour and must be
ported together. Porting `treat_short_extends_as_decodes=False` without the
promotion is strictly wrong at the pin, and porting the promotion without the
flag is a no-op.

### ★★ Which population actually DISCRIMINATES the two modes

This is the single most consequential correction in this spec, because the
obvious answer is wrong and the gate built on the obvious answer cannot fail.

**A short extend is not it.** This spec's own definition requires prior context
(`has_context = num_computed > 0`, `runner.cpp:144` and `:155-157`), so a short
extend necessarily has `seq_len > 1`. At `decode_threshold = 1` it also has
`query_len == 1`. Those are exactly the two conditions of the promotion at
`mamba_attn.py:450-453`, so on a short extend **the promotion always fires** and
flips the row back out of `is_prefilling` *before* the split reads it. Traced
through `utils.py:594-635` for `query_lens = [1, 1]`:

| population at `R = 2` | mode `True` (what we implement) | mode `False` + promotion (upstream) |
|---|---|---|
| two plain decodes | `(2, 0, 2, 0)` | `(2, 0, 2, 0)` — same |
| decode + **short extend** (`is_prefilling`, `seq_len > 1`) | `(2, 0, 2, 0)` | `(2, 0, 2, 0)` — **same**, because the promotion fires |
| decode + **single-token first-chunk prefill** (`is_prefilling`, `seq_len == 1`) | `(2, 0, 2, 0)` | **`(1, 1, 1, 1)` — differs** |

**So the discriminating population in A2-B's scope is a still-prefilling row with
`query_len == 1` and NO prior state (`seq_len == 1`)** — a one-token prompt, or a
first chunk the scheduler clamped to one token. `has_prior_state` is `False`
there, the promotion does not fire, and `is_prefill |= is_prefilling`
(`utils.py:625`) keeps it a prefill. Our runner classifies that row as region 3,
"pure prefill (first chunk)", at `runner.cpp:149-151` — i.e. explicitly **not** a
short extend.

The short-extend arm is still worth running, and G1 keeps it as arm (c4), but
for the
opposite assertion: it proves the **promotion** is ported, because under mode
`False` a short extend must come out classified as a **decode**. A short-extend
arm that reds is a missing promotion, not a missing flag.

### ★★★ A DIFFERING SPLIT IS NOT A DIFFERING FORWARD: (c3) needs a DIRTY state slot

★★ establishes that (c3) is the only population whose **four-tuple** differs
between the two modes. G0 requires a **token** mismatch, and that is one level
further down. **An earlier revision of this spec stopped at the four-tuple**, and
so specified a flagship arm that is GREEN on a fresh runner — the same defect as
the short-extend draft it replaced, one level deeper. This section traces the
distance and states the condition that closes it.

**Every consumer of `nd` / `np` in the forward.** The population is the complete
output of `grep -n '\bnd\b\|\bnp\b'` over
`src/vllm/model_executor/models/nemotron_h_device.cpp`, so it is a closed list
rather than a sample:

| site | what it does | can it change a token? |
|---|---|---|
| `:1201-1203` | `VT_CHECK(nd + np == R)` | **no** — a refusal, and it holds under both modes |
| `:1204-1206` | `VT_CHECK(num_decode_tokens + num_prefill_tokens == T)` | **no** — a refusal, and it holds under both modes |
| `:1233` | `if (r < nd)` ⇒ `init[r] = 1` | only through `init` |
| `:1239-1242` | `VT_CHECK(prefill_has_initial_state->size() == np)` | **no** — a refusal |
| `:1244` | `init[r] = (*gm.prefill_has_initial_state)[r - nd]` | only through `init` |
| `:1261-1263` | the `[NH-DIAG]` print | **no** |

Nothing else reads the split. `non_spec_query_start_loc` and
`non_spec_state_indices_tensor` are the builder's own inputs verbatim on the
non-spec path — `gdn_attn.cpp:161-162`, `non_spec_state_indices = state_indices`
and `non_spec_query_start_loc = m.query_start_loc` — and so do not depend on it;
and the attention half reads `attn_meta` only.

**So the split reaches the arithmetic through exactly one wire: `init`.** `init`
becomes `sdi.state_has_initial` (built `:1225-1246`, uploaded `:1254`, field
declared `:1170`), and that field has exactly one reader — `:1453`, the mask
argument of the two `vt::GdnStateGather` calls at `:1454-1455`. The kernel's
store is `StoreF32(working, dst, keep ? LoadF32(cache, src) : 0.0f)`
(`src/vt/cpu/cpu_ops.cpp:2416`, and `:2423` on the widened-cache path, with
`keep` read at `:2408-2410`). **The mask chooses between the cache slot's bytes
and zeros, and nothing else.** The mixer is then told `state.has_initial = true`
unconditionally (`:1663`), and A2-P's own comment at
`nemotron_h_device.cpp:1423-1424` states the consequence outright:

> "it is EXACT. With a zeroed state row, `has_initial_state = 1` and
> `has_initial_state = 0` compute the identical answer"

On (c3)'s row the mask is `0` under mode `False` — the builder sets it as
`context_lens[r] > 0` (`gdn_attn.cpp:326`) and a no-prior-state row has
`context_len == 0` — and `1` under mode `True`, because `r < nd`. **The two modes
therefore produce different tokens if and only if cache slot `idx[r]` is
non-zero.**

**The GDN state cache is allocated ZEROED**: `host_data_.assign(bytes, 0)` at
`src/vllm/v1/worker/gpu/runner.cpp:484`, or `Memset(..., 0, bytes)` at `:491` on
the backend-resident arm. **So on a fresh runner arm (c3) is GREEN under both
modes**, and a green there is evidence about the fixture's slot, not about the
flag.

**What makes the difference observable is a DIRTY slot**, and nothing zeroes one
on release. `ScatterNemotronHState` (`:1459-1470`) writes a slot for its owning
request; `remap_gdn_state_slots` returns a finished request's slot to
`gdn_free_slots_` (`runner.cpp:1153-1160`) still carrying that request's final
conv and SSM rows, and issues it to the next new sequence at `:1177-1178`. Those
two loops run in **one** call, reclaim before assign, and the pool is LIFO
(`back()` / `pop_back()`), so the slot a request frees is the very next slot
issued. The construction is therefore deterministic rather than incidental.

**Scope that determinism honestly: it holds for EXACTLY ONE departure per
step.** The reclaim loop iterates `gdn_slot_of_req_`, which is a
`std::unordered_map` (`include/vllm/v1/worker/gpu/runner.h:563`), so with two or
more requests departing in the same step the order in which their slots are
pushed onto `gdn_free_slots_` (`:564`) is unspecified and *which* freed slot
lands at `back()` is unspecified with it. The fixture below retires one request
while another keeps decoding, so the case never arises — and step 3's reuse
assertion is what would detect it if a later edit made it arise. Do not extend
the fixture to retire two requests and keep relying on the identity of the
reclaimed slot.

**★ Arm (c3)'s row must be scheduled onto a RECLAIMED slot.** "A one-token prompt
scheduled alongside a decoding request" is necessary and **not** sufficient, and
that phrasing on its own specifies the green arm. The fixture is:

1. run a request `X` for at least one step that reaches the mamba block, so
   `ScatterNemotronHState` writes `X`'s slot with a non-zero state;
2. in the step where `X` is no longer scheduled, submit the (c3) one-token
   prompt alongside a still-decoding request. `remap_gdn_state_slots` reclaims
   `X`'s slot and issues it to the new sequence in that same call;
3. **assert in the test that the (c3) row's `non_spec_state_indices_tensor`
   entry is the slot `X` held**, and that the slot's gathered state is non-zero.
   Without that assertion the arm silently degrades to the fresh-slot arm and
   reports a green that means nothing.

Under mode `False` the row is handed zeros, which is correct for a sequence with
no prior state. Under mode `True` it is handed `X`'s final recurrent state — the
previous tenant, decoded as this sequence's own context. That is fluent wrong
output.

**This is B-M4's mechanism, observed from the other side.** B-M4 drops the
zeroing so that even the correct mode leaks the previous tenant; (c3) keeps the
zeroing and asks whether the mode selects it. **They share one fixture
requirement**, so an implementer who builds B-M4's slot-reuse fixture has built
(c3)'s, and one that cannot dirty a slot disarms both.

---

## Our baseline

Measured in this worktree at `origin/main` @ `5f68e60df`, by reading the tree
rather than by inheriting A2-P's description of it. **Three of the four things
A2-B was expected to build already exist**, and the one that does not is not the
one the issue points at.

### ★ The correction A2-P's `:64` owes

A2-P's scope table says at
[`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md)`:64` that
the `num_reqs <= 1` clause stays because "nothing in this unit … indexes
per-request state by anything but slot 0". (`:81` is a **different** In/Out row
and is not stale: it correctly assigns "batching, decode/prefill reordering,
`num_reqs > 1`" to A2-B.) **That describes
A2-P's scope, and the landed implementation is strictly more general than its own
scope line.** A2-P's §4.1
([`nemotron-h-a2p-paged-forward.md`](nemotron-h-a2p-paged-forward.md)`:348-354`)
made the opposite decision on purpose — "**Index through them anyway.** A forward
that hardcodes slot 0 passes every gate A2-P owns and then fails silently under
A2-B" — and the code carries it, in the same words at the site:
`nemotron_h_device.cpp:1221-1224`, "INDEX THROUGH THE VECTORS EVEN AT ONE
REQUEST". **Two sources, quoted separately**; an earlier draft ran them together
as one verbatim quotation attributed to §4.1, and neither text reads that way.
The enumeration the dispatch asked for, site by site:

| Site | What it does today | What A2-B changes |
|---|---|---|
| `nemotron_h_device.cpp:1225-1246` — the `idx` / `init` build in `BuildNemotronHPagedStep` (the function itself is `:1173-1279`) | loops `for r in [0,R)`, reads `gm.non_spec_state_indices_tensor[r]`, range-checks it against `state_slots`, sets `init[r] = 1` for `r < nd` and reads `gm.prefill_has_initial_state[r - nd]` otherwise | **nothing structural.** The `r < nd` branch is where the ordering contract becomes load-bearing, and D4's assertion is added HERE |
| `:1248-1255` — the `NemotronHPagedStep` upload | uploads `slot_mapping [T]`, `block_table [R, cols]`, `seq_lens [R]`, `query_start_loc [R+1]`, `state_idx [R]`, `state_has_initial [R]` | **nothing.** Already `R`-shaped end to end |
| `:1443-1457` `GatherNemotronHState` | allocates `[R, Cd, Kw-1]` and `[R, Hh, P, N]`, calls `vt::GdnStateGather(…, sdi.state_idx.t(), &hinit)` twice | **nothing.** `GdnStateGather` takes the index vector and the mask; both are extent `R` |
| `:1459-1471` `ScatterNemotronHState` | `vt::GdnStateScatter` writes only the rows named by `state_idx`; its own comment says this "keeps two concurrent sequences from overwriting each other once A2-B lifts the request count" | **nothing.** This is the property A2-B must now *prove*, and B-M3 is the mutation that proves it |
| `:1630-1707` — the mamba layer body (the per-request loop is `:1650-1690`) | downloads `conv_all` / `ssm_all` at `[R, …]`, loops `for r in [0,R)`, slices `r*conv_row` and `r*ssm_row`, reads `[t0,t1)` from `gdn_meta.non_spec_query_start_loc`, calls `NemotronHMamba2Mixer` per request, writes back per request, re-uploads and scatters | **nothing structural.** Two real defects fall out at `R > 1` and are listed below |
| `:1639-1644`, `:1692-1699` — the `[NH-DIAG]` `GATHERED` and `WROTE` prints | `DiagL2(conv_all, 0, conv_row)` — request **0 only**, at every `R` | **fix.** A diagnostic that silently reports one request of a batch is the instrument that makes a batching defect invisible during triage. Either print per request or name the request in the format string |
| `:1294-1399` — the attention half, called from the layer loop at `:1620-1621`; the embedding is `:1528-1571` | `NemotronHAttnBlockPaged` takes `sdi.block_table [R, cols]`, `sdi.seq_lens [R]`, `sdi.query_start_loc [R+1]` into `vt::PagedAttention` (`:1389-1390`), which is varlen by construction; `pa.query_start_loc_host = meta.query_start_loc.data()` (`:1387`, the only occurrence in the file) is the runner's own `R+1` vector | **nothing to write; something to PROVE.** See D1 |
| `:1765-1785` — the logits tail | `logits_indices` is a per-row index list and an empty one means "every row" | **nothing.** Already general |

**So the per-request state indexing is done, and the honest statement of this
unit is that A2-B *verifies and hardens* it rather than building it.** That
changes the shape of the work — most of the budget is gate, not code — and it
changes what a reviewer looks for.

Two real `R > 1` defects the audit did find in that body, both in the mamba
block:

- **D-B1, `init[r] = 1 for r < nd`.** Correct only if the first `nd` rows really
  are the decodes. Nothing in the forward checks it, and under
  `treat_short_extends_as_decodes=False` the set of rows that *are* decodes
  changes. D4 asserts it.
- **D-B2, the diagnostic.** Above.

### The runner already delivers an ordered batch

`src/vllm/v1/worker/gpu/runner.cpp:126-214`,
`reorder_batch_to_split_decodes_and_prefills`, ported from `utils.py:665`. It is
called **unconditionally** at `runner.cpp:1257`, before any metadata is built,
with the default `decode_threshold = 1` (`include/vllm/v1/worker/gpu/runner.h:103`).
It classifies four-way into `decode(0) → short_extend(1) → long_extend(2) →
prefill(3)` (`runner.cpp:147-162`) on exactly upstream's predicates:
`has_context = num_computed > 0`, `is_below = num_scheduled <= decode_threshold`,
`done_prefilling = num_computed >= num_prompt`.

**So the answer to "do we reorder, or require the runner to present an ordered
batch" is settled by what exists: the runner reorders, and A2-B requires the
ordered batch.** A2-B adds no reorder of its own. It adds the assertion that the
requirement held, which is what turns an assumption into a contract.

Note the runner's classifier already computes `done_prefilling` at
`runner.cpp:146` — the exact signal `is_prefilling` is. It is computed and then
discarded. **It is also not computed at all when `R == 1`**; see the Port map.

### ★ The one thing that is genuinely wrong: our split implements the other mode

`src/vllm/v1/attention/backends/gdn_attn.cpp:12-13`, verbatim:

```
// Ported from utils.py::split_decodes_and_prefills @ e24d1b24 (lines 564-633),
// T0 subset: require_uniform=False, treat_short_extends_as_decodes=True.
```

`SplitDecodesAndPrefills` (`gdn_attn.cpp:14-54`) has no `is_prefilling` term at
all: it takes the `max_query_len <= decode_threshold ⇒ all decodes` early return
(`:23-25`), then argmax's the first `query_len > threshold`. That is exactly
`treat_short_extends_as_decodes=True`.

That port is **correct for GDN** — `gdn_attn.py:213` passes the defaults — and it
is called for the GDN metadata at `gdn_attn.cpp:156` with
`decode_threshold = 1`. NemotronH's `GDNAttentionMetadata` comes off that same
builder. So today NemotronH's `num_decodes` / `num_prefills` are computed with
Qwen3-Next's flag settings, not with its own backend's.

At `num_reqs <= 1` the difference is unobservable: a single request is either the
whole decode side or the whole prefill side under either mode. **The refusal has
been hiding a real mirroring gap, not only an unbuilt feature.** That is the
finding that most changes how this unit should be reviewed.

`v1::CommonAttentionMetadata` (`include/vllm/v1/attention/backend.h:107-140`) has
`query_start_loc`, `seq_lens`, `num_computed_tokens_cpu`, `num_reqs`,
`num_actual_tokens`, `max_query_len`, `max_seq_len`, `block_table_tensor`,
`slot_mapping` — and **no `is_prefilling`**. `grep -rn is_prefilling src/ include/ tests/`
returns hits in exactly two places, and neither is a field on this struct:
`v1/engine/output_processor` (`output_processor.h:164`, `output_processor.cpp:407`,
`:408`, `:438`, `:458`, `:460`) carries an unrelated per-request flag of the same
name, and `include/vllm/v1/attention/backend.h:30` is the DEFERRED-fields comment
this spec already cites above — it names `is_prefilling` as **not** ported, which
is the claim, not a counterexample to it. **There is no `is_prefilling` declared
anywhere in `src/`, `include/` or `tests/`**, and that is what the grep shows.

### The interlock test that exists

`tests/vllm/models/test_nemotron_h_paged_forward.cpp:1356-1390`, "NemotronH
paged: G-SAFE still refuses a BATCHED step by name", asserts the refusal
contains `NemotronHForCausalLM`, `BATCHED decode is not ported`, `A2-B` and
`#810`. **A2-B rewrites this case rather than deleting it** (§Gates G5).

The harness the multi-request gate needs is already there: `Fixture`,
`RunnerGreedy` (`:418`), `MakeNewReq` (`:375`), `NewStep` (`:387`), `DecodeStep`
(`:399-414`, which already takes a **vector** of ids), and every existing case constructs
`GPUModelRunner(..., /*max_num_reqs=*/2, ...)`.

---

## Port map

| Upstream | Local | Change |
|---|---|---|
| `utils.py:564-635` under `treat_short_extends_as_decodes=False` + `mamba_attn.py:445-461` | new, beside `SplitDecodesAndPrefills` in `src/vllm/v1/attention/backends/gdn_attn.cpp` | **ADD a second entry point**, do not re-flag the existing one. GDN's caller must stay byte-identical (§Risks R1). Name it for what it is — the mamba-family split — and cite `mamba_attn.py:464-468` in its header |
| `utils.py:623-625` `is_prefilling`, declared at `vllm/v1/attention/backend.py:459-462`, produced at `vllm/v1/worker/gpu/model_runner.py:975` | `v1::CommonAttentionMetadata` in `include/vllm/v1/attention/backend.h` | **ADD** `std::vector<int32_t> is_prefilling` (or `std::vector<uint8_t>`; mirror upstream's per-request bool). **Where it comes from: see ★ below — NOT from the reorder classifier.** Empty ⇒ the mamba split must refuse rather than assume, because an empty vector read as "nothing is prefilling" is the silent-wrong-answer shape |
| `mamba_attn.py:447-448` + `vllm/v1/attention/backend.py:464-468` `seq_lens_cpu_upper_bound` | `v1::CommonAttentionMetadata::seq_lens_cpu` (`backend.h:117`) | **SUBSTITUTE, and record it — this is a tracked exception, not an oversight.** The promotion predicate `has_prior_state = seq_lens_cpu > 1` (`:452`) reads **`seq_lens_cpu_upper_bound`**, not `seq_lens_cpu`. `include/vllm/v1/attention/backend.h:30-31` lists **both** fields as DEFERRED, so neither exists locally today. Since ★★ makes `seq_len > 1` the quantity that discriminates the two modes, this field is load-bearing and cannot be waved through. **Decision: use our existing `seq_lens_cpu`, and say why.** Upstream's docstring says the upper bound is "Precise for prefill rows and for all rows outside async spec decode; optimistic for async-spec decode rows"; A2-B refuses every speculative row (`num_spec_decodes == 0`, `nemotron_h_device.cpp:1216-1219`), so the one population the two fields disagree on is the one this unit does not admit. **If a later row lifts the speculative refusal, this substitution stops being sound and the real field is owed** — recorded under `## Owed` |
| `mamba_attn.py:87`, `:200` | the mamba split's `decode_threshold` argument | value `1`, with the anchor in the comment. No new constant: `gdn_attn.h:75` already defaults to 1 |
| `mamba_attn.py:31,33` → `mamba_mixer2.py:751-767` | `GDNAttentionMetadata::num_decodes` / `num_prefills`, consumed at `nemotron_h_device.cpp:1199-1200` | **REWIRE** the NemotronH path to the mamba split. Scope the rewire to this architecture; nothing that reaches GDN's own models may change |
| `nemotron_h_registry.cpp:159-167` | same | **REPLACE** per D4 |
| `mamba_attn.py:204-221` | — | **NO port.** D5 records the measurement instead |

### ★ Where `is_prefilling` must come from, and where it must NOT

**An earlier draft of this spec said to populate it "from the `done_prefilling`
the reorder classifier already computes at `runner.cpp:147`". That is a hazard,
and it is written out here so the implementer does not walk into it.**

`reorder_batch_to_split_decodes_and_prefills` early-returns before computing
anything:

```cpp
// src/vllm/v1/worker/gpu/runner.cpp:129-132
const int num_reqs = input_batch.num_reqs();
if (num_reqs <= 1) {
  return false;
}
```

`done_prefilling` is computed at `:146`, **inside the per-request loop below that
return**. So at `R == 1` it is never computed at all. An `is_prefilling` vector
sourced from there would be **empty on every single-request step** — and R2 makes
the mamba split refuse on an empty vector, so plumbing it as instructed would
make the split refuse *every existing NemotronH gate in the tree*. The unit would
appear to break single-request decode, and the cause would be three files away
from the symptom.

**It must therefore be computed unconditionally, per request, outside the
reorder**, in the same place and from the same inputs the classifier uses —
`num_computed_tokens_cpu` and `num_prompt_tokens`, both already on
`input_batch` — and populated for all `R >= 1`. The reorder may then read the
same value rather than recomputing it, but the reorder is not its producer.

Upstream agrees on the shape and not on the arithmetic: it derives the signal as
`num_computed_prefill_tokens_np < prefill_len_np`
(`vllm/v1/worker/gpu/model_runner.py:975`), over *prefill* counters rather than
over total computed tokens against the prompt length.
`vllm/v1/attention/backend.py:460-462` documents the field as
"`num_computed_tokens < num_prompt_tokens`", which is the equivalence our
classifier's `done_prefilling` relies on, so the substitution is defensible —
**but it is an equivalence to be checked, not assumed**, and the implementer
checks it against `:975` explicitly rather than against the docstring alone.

---

## Design

### D1 — Per-request KV page indexing: what changes, and where

**Nothing changes. The obligation is a proof, not an edit.**

`NemotronHAttnBlockPaged` (`nemotron_h_device.cpp:1294-1399`) writes K/V with
`vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, sdi.slot_mapping.t())` (`:1374`) — the
slot map is per **token** (`[T]`) and carries no request assumption — and reads
with `vt::PagedAttention(..., sdi.block_table.t(), sdi.seq_lens.t(),
sdi.query_start_loc.t(), pa)` (`:1389`), which is varlen over `[R, cols]`, `[R]`
and `[R+1]`. The four `VT_CHECK`s at `:1179-1190` already scale in `R`.

Two things a reviewer must confirm rather than assume, both because "the shapes
are right" is not "the kernel serves this population":

1. **The mixed-batch arm.** A batch with some rows at `query_len == 1` and some
   at `query_len > 1` must be served by `vt::PagedAttention` on **each backend
   the gate runs**, not only on the one that happens to dispatch a varlen
   launcher. `pa.causal` is a single flag for the whole batch, which is correct
   for both populations (causal masking is per query position), but that is an
   argument, and an argument is not a measurement.
2. **`pa.query_start_loc_host`** (`:1387`) points at `meta.query_start_loc.data()`, the
   runner's own vector, which outlives the call and is `R+1` long. The
   host-resident grid-sizing path (`ops.h` `PagedAttentionArgs`) is the one that
   avoids a per-layer D2H sync; it must be exercised at `R > 1`, because a
   fallback to the D2H path would be correct and silent.

If either fails, that is a finding for the op, not a reason to widen A2-B. File
it and refuse by name.

### D2 — Per-request recurrent state

Enumerated in §Our baseline. Summary: `GatherNemotronHState`,
`ScatterNemotronHState`, the `idx`/`init` build and the mamba layer loop are all
already `R`-general; A2-B changes **D-B1** (the ordering assumption behind
`init[r] = 1 for r < nd`, handled by D4) and **D-B2** (the request-0 diagnostic).

The `state.has_initial = true`-always decision A2-P documented at
`nemotron_h_device.cpp:1401-1433` stays, and it is *more* important at `R > 1`,
not less: the gather's zeroing is what keeps a fresh request from reading the
previous tenant of a reused slot, and coupling the mixer's flag to the mask would
make that zeroing unobservable. B-M4 is the mutation that keeps it honest.

**★ What D4's ordering assertion can and cannot prove.** D4's second bullet — no
request after index `nd` has a query length `<= 1` unless the split classified it
as a prefill for the `is_prefilling` reason — **re-derives the classification
from the classifier it is checking.** The split sets
`first_prefill = argmax(is_prefill)` where
`is_prefill = (query_lens > threshold) | is_prefilling` (`utils.py:621`, `:625`,
`:630`), so on an **ordered** batch both halves of the assertion hold by
construction and it can never fire. It is therefore **not** what asserts D-B1
(`init[r] = 1 for r < nd`), and §Design must not be read as saying it is: only
**B-M5**, through tokens on arms G1(c2) and G1(c3), catches a broken `r < nd`
branch. A reviewer who sees this assertion green must not read that as proof of
D-B1.

It is **not** vacuous, and G3 is where it earns its place: an **unordered** batch
takes the `query_lens[0] > threshold` early return (`gdn_attn.cpp:31-33`,
`utils.py:607-609`) and yields `nd = 0`, so a decode row sitting after index 0
then violates the first bullet and the assertion fires. That is the case D4 is
built for, and it is why the assertion is a contract rather than a tautology.

**One property is newly load-bearing and must be asserted directly rather than
inferred from tokens:** two requests in one step must land on **distinct** state
slots. `BuildNemotronHPagedStep` range-checks each `s` against `state_slots`
(`:1229-1231`) but never checks the values are distinct. Duplicate slots would
make two requests share a recurrent state — fluent output, wrong tokens. Add the
distinctness check where the range check is, and gate it (B-M6).

### D3 — Reordering: the runner delivers, A2-B requires and verifies

Settled by §Our baseline: `runner.cpp:1257` reorders every step,
unconditionally, on upstream's predicates. A2-B adds **no** reorder.

What A2-B adds is the mamba-flavoured split (Port map row 1) and the assertion
that the batch it is handed satisfies the ordering the split's argmax assumes.
Upstream's own docstring is the specification: "Assuming a reordered batch,
finds the boundary" (`utils.py:571`; `:570` is the opening `"""`). Both our port and upstream's original
return `{0, num_reqs, 0, num_tokens}` the moment `query_lens[0] > threshold`
(`gdn_attn.cpp:31-33`, `utils.py:607-609`) — an *unordered* batch is therefore
not detected, it is silently reclassified as all-prefill.

### D4 — What replaces the G-SAFE clause

**Recommendation, and the implementer should not deviate without a
`NEEDS_DECISION`:**

- `nemotron_h_registry.cpp` — drop `input.num_reqs <= 1` entirely. Do **not**
  narrow it to a smaller count; a count is not the property.
- `BuildNemotronHPagedStep` (`nemotron_h_device.cpp:1173-1279`; the `nd`/`np`
  check is `:1201-1203` and the `idx`/`init` build `:1225-1246`) — add, beside the
  existing `nd + np == R` check, a positive assertion of the contract the
  function's own comment already claims:
  - every one of the first `nd` requests has `query_start_loc[r+1] -
    query_start_loc[r] <= 1`, and no request after index `nd` has a query length
    `<= 1` **unless** the mamba split classified it as a prefill for the
    `is_prefilling` reason (which is the `treat_short_extends_as_decodes=False`
    case and is legitimate). **On an ordered batch this holds by construction and
    cannot fire — see D2's ★. It earns its place on G3's unordered batch, and it
    is not a proof of D-B1;**
  - the `R` state slots are pairwise distinct (D2). **This half is the one that
    is genuinely new**: `BuildNemotronHPagedStep` range-checks each `s` against
    `state_slots` (`nemotron_h_device.cpp:1229-1231`) and never checks the values
    are distinct, so nothing in the tree asserts it today;
  - the message names the architecture and names #1395, per the same discipline
    A2-P's refusal follows.
- `num_spec_decodes == 0` (`:1216-1219`) is untouched and remains the speculative
  refusal.

Who owns the remainder: speculative rows → #517 W5 / #810; mamba prefix caching
→ #810, still unported and still unrefused by name, which is itself worth an
issue if the implementer finds no existing one.

### D5 — Interactions

| With | Interaction | Disposition |
|---|---|---|
| **A2-Q1** (FP8 mamba, [#940](https://github.com/mudler/vllm.cpp/issues/940) **(closed)**, PR #1289 — **LANDING BUT HELD `DRAFT`**, not landed and not abandoned; see D6) | swaps the arm **inside** the per-request mamba loop; the loop, the gather and the scatter are unchanged by it. No seam conflict — but a **textual** conflict in `nemotron_h_device.cpp` is near-certain | order is free. Whoever lands second rebases and re-runs the focused gate on the merge result, not on either parent. `merge-tree` reporting clean is not the same as the merge building. **A2-B must not wait for it and must not assume it** — §Gates G6 says what A2-B measures on each side of that arm |
| **A2-Q2b** (device `lm_head`, in flight) | rewrites the `want`/`gathered` tail (`:1765-1785`). That tail is already per-row general | order is free; same rebase rule. A2-B must not pre-empt the tail |
| **CUDA graph capture** | `GraphEligibleQueryLen` (`src/vllm/v1/worker/gpu/cudagraph_dispatch.h:161`) admits `R > 1` uniform decode steps, and `runner.cpp:1510` calls it on the shared path for **whatever model the step routes to**. Lifting the refusal therefore newly admits NemotronH multi-request decode steps into the eligible population | **measure, do not assume.** NemotronH registers no decode graph driver, so the expectation is that the predicate names a length and nothing captures. That expectation is exactly the "absent hook looks like an armed instrument" shape. The gate reads `GraphDispatchStats` (`cudagraph_dispatch.h:187+`) on a multi-request run and **records the counters in the PR body**, whatever they say. If something does capture, A2-B refuses graph capture for this architecture by name and files the driver as owed — it does not write one |
| **`mamba_attn.py:204-221`** | upstream's mamba CG capture is decode-only and asserts `max_query_len == 1 + num_spec_tokens` | consistent with refusing; recorded so the refusal has an upstream anchor rather than being a local preference |
| **#1217** (`device_token_ids` has no per-model opt-in) | A2-P's `## Owed`. The decode-row splice is per row and its correctness at `R > 1` has never been observed | A2-B's multi-request gate is the first thing in the tree that can see it. If it diverges, that is #1217 evidence, filed and referenced — **not** silently repaired inside A2-B |


### D6 — ★ A2-Q1's device mamba arm FAILS its token gate, and only the host arm is gated

Measured on `dgx:gpu0` (GB10, sm_121a), run `20260819T200231Z`, logs at
`/workspace/a2d1-discriminate/20260819T200231Z`. A three-leg discriminator
settled two things, and both belong in this spec because both change what A2-B
may assume.

**Correctness — BOTH device arms diverge; only the HOST arm passes.**

| leg | arm | token gate | warm s/tok | GPU busy |
|---|---|---|---|---|
| `a3_hostmamba` | host bounce (the shipped default) | **96/96 STRICT PASS** | 10.1502 | 7.86% |
| `a3_off` | device `Mamba2ChunkScan` | **95/96 DIVERGENCE** | 1.3898 | 10.18% |
| `a3_on` | device `Mamba2StateUpdate` | **95/96 DIVERGENCE** | 1.3947 | 10.27% |

**Read the leg names carefully: the leg NAMED `off` still has the device arm ON**
— `off` distinguishes the two device kernels, not device from host. The correct
one-line statement is "both device arms diverge, only the host arm passes", and
the earlier phrasing "95/96 with the arm on, 96/96 with the arm off" is
confusable and must not be reproduced. Only the last token of the longest prompt
differs. PR #1289 is held `DRAFT` on that.

That divergence is **A2-Q1's to close, not A2-B's**, and A2-B must not absorb it.
It has one consequence here, and it is a gate-hygiene one: a 95/96 is exactly the
shape a batched gate can produce by accident, and an implementer who runs G1 on a
tree carrying either device arm will see a mismatch that is not theirs.
**Therefore every A2-B correctness leg runs on the HOST mamba arm — the shipped
default, and the only arm that passes 96/96** — stated explicitly in the recipe,
and any leg run on a device arm is reported as a separate labelled pair. An
unlabelled 95/96 in an A2-B PR body is not evidence about batching.

**Speed — the device arm is worth 7.28x per output token on a WARM basis**
(10.1502 s → 1.3898 s). Cite the warm-basis ratio and say that it is warm-basis;
a ratio that mixes the cold first prompt into one side's exclusion rule is a
different number and has already been quoted as this one.

**★ That 7.28x is an UNGATED PROJECTION, and `718.2x` remains the GATED figure.**
AGENTS.md is explicit — "Correctness always comes first. Establish the declared
token-exact gate before you accept a performance result. Never trade correctness
for throughput." Both device legs fail their token gate, so the ratio measured on
them is not an accepted result and **cannot supersede a gated baseline**. That is
precisely why #1289 is `DRAFT`.

Three things follow:

1. **`718.2x` is the live gated baseline against vLLM. Quote it, and label the
   device-arm figure as a projection pending #1289.** An earlier draft of this
   spec said the 718x baseline was "dead" and treated `108.2x` as the live
   denominator; that inverted the polarity of the rule above, retiring a gated
   number in favour of an ungated one. A number quoted often enough starts being
   treated as measured, and `108.2x` is the one at risk here.
2. **A2-Q1 banks a large multiple and does NOT close the gap.** At ~10.2% GPU
   busy the decode is still ~90% GPU-idle even on the device arms, so A2-B must
   not be written as though the remaining gap is small once #1289 clears.
3. **A per-output-token cost that a kernel arm cuts 7.28x is dominated by
   HOST-SIDE work, and that is precisely the term batching amortises
   differently.** Our mamba block is a serial per-request host loop
   (`nemotron_h_device.cpp:1650-1690`, and §Scope item 5 keeps it serial), so at
   `R` requests the host term scales with `R` while a batched device GEMM would
   not. **The consequence is a prediction, and it is falsifiable** — but only once
   the two rates are named, because "tokens/s per step" is not a defined rate and
   "did not improve" is not decidable without a noise floor:

   | quantity | definition | prediction at `R = 2` vs two `R = 1` runs |
   |---|---|---|
   | **aggregate output tok/s** | total output tokens generated by the step, divided by the wall time of the step, summed over the run | **improves** |
   | **per-request output tok/s** | for each request, its own output tokens divided by that request's own wall time, reported per request and never imputed from the aggregate | **does not improve, and may worsen** |

   **Spread rule**: `>= 5` steady-state repetitions per arm on an idle,
   clock-pinned host; report median with min and max; a difference counts as
   "improved" or "worsened" only when the two arms' min-max ranges do not
   overlap. Overlapping ranges are reported as **not resolvable against the noise
   floor**, which is a result. That is R7 restated with a mechanism and a
   measurement.

So the honest statement of what A2-B's throughput gate measures, when the
umbrella opens that axis:

- **today it measures on the HOST arm, against the gated `718.2x`**, because that
  is the only configuration whose token gate passes and AGENTS.md forbids
  accepting a performance result before the token-exact gate is established. The
  device arm's figure may be carried alongside, **named as an ungated projection
  pending #1289**, and never as the denominator or as a superseding baseline;
- it measures on a device arm only **after** #1289 clears its 95/96 and that arm
  lands. Until then the correctness precondition for that measurement is unmet
  and the axis stays **PENDING #1289** — which is a result, recorded, not
  silence;
- it reports the **step** and **per-request** rates separately, because with a
  serial host mamba loop the two move in opposite directions and a single
  aggregate hides it;
- it never imputes a per-request number from an aggregate. The per-item record
  either exists or the claim is not made.

**A2-B still records no number itself.** D6 exists so that the row that does
cannot inherit an ungated projection as if it were a baseline, or an unlabelled
arm state, from this one.


---

## Tests to port

Preserving parameters, modes, fixtures, tolerances, failure cases and the
revision anchor `5559679229bc961848b121ccdeaa8fa5d79bec98`. Document only an
unavoidable harness adaptation.

| Upstream | What it pins | Local |
|---|---|---|
| the `split_decodes_and_prefills` cases under `treat_short_extends_as_decodes=False` | that a short extend counts as a **prefill**, and that the `is_prefilling` assert (`utils.py:624`) fires when the signal is absent | extend `tests/vllm/v1/attention/test_gdn_metadata_builder.cpp`, whose `:224` already pins the ordering contract for the `True` mode. The `False` twin sits beside it, and the `True` cases must stay byte-identical — that is the regression guard for R1 |
| `mamba_attn.py:445-461` promotion | a prefilling row with `query_len == 1` **and** `seq_len > 1` is promoted back to a decode; one with `seq_len == 1` (no prior state) is **not** | a table-driven case over the four combinations of (`is_prefilling`, `seq_len > 1`). All four, not the happy pair |
| `mamba_attn.py:87` / `:200` | threshold 1 | assertion on the value the NemotronH path passes |
| the mamba mixer's decode-first splits (`mamba_mixer2.py:758-767`, `:808-812`) | decode rows occupy the **leading** token range | our equivalent is the `[t0,t1)` slice at `nemotron_h_device.cpp:1669-1681` (`t0`/`t1` are read at `:1670-1671` from `gdn_meta.non_spec_query_start_loc`); a case asserting a two-request step's per-request output lands at the right token offsets |

The upstream harness is Python and torch-tensor based; the adaptation is
host-vector based and is stated in each case's comment, as A2-P did for
`test_mamba_utils.py:2136`.

---

## Gates

**Correctness first. A2-B records no throughput, latency or memory number on any
axis** (§Scope, item 1).

### G0 — the red, first, and what "for the intended reason" means here

The RED is not the refusal. `ModelRegistry::Forward` on a two-request input
already throws today (`test_nemotron_h_paged_forward.cpp:1356`), and a test that
merely stops throwing has proven that a `VT_CHECK` was deleted.

> **The red-first test is G1 with the `num_reqs <= 1` clause already removed and
> nothing else changed.** It must fail with a **token mismatch** between the
> batched step and the same requests run singly.

**Exactly one arm is expected to red, and the others are expected to pass.** G0
is G1, G1 has several arms, and §Our baseline establishes that the `idx`/`init`
build, `GatherNemotronHState`, `ScatterNemotronHState` and the mamba per-request
loop are all already `R`-general on the untouched forward. So:

| G0 arm | expected on the untouched forward | why |
|---|---|---|
| G1(a), G1(c1), G1(c2), G1(c4) | **GREEN** | the per-request machinery already exists (§Our baseline), and none of these populations discriminates the two split modes (★★) |
| G1(b), the oracle leg | **GREEN** | same reason |
| **G1(c3)** ★, **on a reclaimed state slot** | **RED, with a token mismatch** | the only population whose four-tuple differs, run in the only condition under which that four-tuple reaches the tokens (★★★). This is the whole of G0 |
| **G1(c3)** on a FRESH state slot | **GREEN, and it proves nothing** | the split still differs, but `init` is the only wire it reaches the arithmetic on, and over a zeroed slot mask `0` and mask `1` compute the identical answer (`nemotron_h_device.cpp:1423-1424`). Named as an arm so it cannot be mistaken for the flagship |

**A green on (a), (b), (c1), (c2) and (c4) is the predicted result, not a
rebuttal of anything.** Reading it as one is the trap this section exists to
close: an earlier draft asked for a red from "G1" undifferentiated, which no arm
but (c3) can deliver, and an implementer following it would have reported stop
condition 2 on the unit's own flagship.

If **(c3)** passes on the untouched forward, work the hypotheses in this order,
because the first one is both the likeliest and the one a correctly shaped
fixture still hits:

1. **The state slot was clean.** ★★★: the split's only wire into the arithmetic
   is `init`, and over a zeroed slot both mask values compute the identical
   answer. A fixture built exactly as G1(c3)'s prose prescribed before this
   revision — a one-token prompt scheduled alongside a decoding request, on a
   fresh runner — is green here **for a reason that is not a finding**. Check
   the slot before anything else.
2. **The row was not the (c3) row.** A row that does not carry
   `query_len == 1` **and** `seq_len == 1` **and** `is_prefilling` is not arm
   (c3).
3. **Only then** is it a real surprise, and the unit's shape changes.

Report, beside the transcript: the constructed metadata (`query_start_loc`,
`seq_lens`, `is_prefilling`), the (c3) row's state-slot index, whether that slot
was previously written and by which request, and the L2 of the gathered state
row. The first two hypotheses are distinguishable only from those.

Capture that transcript — the `[doctest]` `test cases:` / `assertions:` /
`Status:` lines and the mismatching token indices — and put it in the PR body. A
test never seen failing has proven nothing.

### G1 — the batched gate. **More than one request in one step.**

Through `ModelRegistry::Forward`, from a real `GPUModelRunner`. Not
`NemotronHPagedForward`, not a fabricated `ModelForwardInput`.

Three comparisons, all required:

- **(a) against the same requests run singly.** Two prompts, `P` and `Q`. Arm A:
  one runner, both requests scheduled in the same step, prefill then `>= 3`
  decode steps. Arm B: two runners (or one runner used twice with reset state),
  each running one prompt alone. **Every token of `P` and every token of `Q`
  must match across the arms.** This is the comparison a single-request gate
  structurally cannot make, and it is the one that catches cross-request
  contamination, a mis-split and a duplicate state slot.
- **(b) against the pinned oracle**, `555967922`, same checkpoint, same prompts,
  same token counts, greedy. Recorded with the exact build and run recipe,
  revisions, model hash and contention state, per AGENTS.md.
- **(c) mixed populations, not only uniform decode.** Four arms, and the third
  is the flagship — the other three cannot discriminate the two split modes at
  all (see ★★ above for the derivation):

  | arm | population | what it asserts |
  |---|---|---|
  | (c1) | two plain decodes | the batched path at all. **Classifies identically under both modes** |
  | (c2) | one decode + one multi-token prefill | the decode-first token split. **Identical under both modes** |
  | (c3) ★ | one decode + one **still-prefilling, single-token, NO-prior-state** row (`query_len == 1`, `seq_len == 1`), **on a RECLAIMED state slot** | **THE mode difference.** Upstream splits `(1, 1, 1, 1)`; our `True`-mode port splits `(2, 0, 2, 0)`, which sets the gather mask to `1` and hands the row the previous tenant's recurrent state instead of zeros. **The reclaimed slot is part of the arm, not a detail of the harness** — over a never-written slot the two masks compute the identical answer and this arm is green (★★★) |
  | (c4) | one decode + one **short extend** (`query_len == 1`, `seq_len > 1`) | **the PROMOTION**, not the flag. Under mode `False` this row must classify as a **decode**, because `mamba_attn.py:450-453` promotes it before the split. A red here is an unported promotion |

  Arm (c3) is the population arm (a) structurally cannot reach and the only one
  that fails today. Constructing it needs a one-token prompt (or a first chunk
  clamped to one token) scheduled alongside a decoding request — a two-request
  step of two ordinary prompts will not produce it — **and that row must land on
  a state slot a finished request already dirtied.** ★★★ gives the three-step
  construction and the assertion that proves the slot was reused; it is the same
  slot-reuse fixture **B-M4** needs, and neither can red without it.

`RunnerGreedy` (`:418`) is single-id; a batched sibling is needed. `DecodeStep`
(`:399-414`) already takes a vector of ids.

### G2 — the split, directly

Unit cases on the mamba split (§Tests to port), asserting the four-tuple against
upstream's values for each mode. Direct, because a token gate over a small
synthetic model can agree by luck on a classification error.

### G3 — the ordering contract

D4's assertion, gated: an out-of-order batch must **refuse by name**, not be
silently reclassified as all-prefill.

### G4 — CUDA-graph dispatch counters

`GraphDispatchStats` read on a multi-request run, values in the PR body
whichever way they fall (D5).

### G5 — the rewritten interlock case

`test_nemotron_h_paged_forward.cpp:1356` is **rewritten, not deleted**: it now
asserts the speculative refusal and the ordering refusal by name, with `#1395`
and `#810` in the message. A reviewer who finds the case deleted returns FAIL.

### G6 — the arm state is part of every recipe

Every correctness leg names, in the recipe and in the PR body, **which mamba arm
it ran on** — the host bounce, or one of the two device kernels. **The A2-B legs
run on the HOST arm** (D6), because it is the only one whose token gate passes
96/96. A leg run on a device arm is reported as a separate labelled pair. Name
the arm, never "on"/"off": the leg named `a3_off` is a device leg, and that
labelling has already been misread once. An unlabelled 95/96 is not evidence
about batching, and A2-B records no throughput number on any axis regardless of
the arm.

### What each gate CANNOT see

| Gate | Blind to |
|---|---|
| any token gate | **a dtype that is too WIDE.** An f32 conv or SSM page is *more* precise: tokens match, goldens pass, the path moves twice the bytes. The memory format is compared against the oracle explicitly, read from the **running** engine's resolved config, not from source |
| any token gate | **a dequant fallback.** A silently dequantized NVFP4 expert produces correct tokens; only the memory format and the load accounting can see it |
| any token gate | **a dropped mechanism whose argmax is unchanged.** Hence G2's direct assertions |
| **G1 arm (a), and arms (c1), (c2), (c4)** | **the entire `treat_short_extends_as_decodes` difference.** Two decodes classify identically under both modes; so do a decode plus a multi-token prefill; and so does a decode plus a short extend, because the promotion at `mamba_attn.py:450-453` always fires on one. This is why **(c3)** is required and not optional, and why an earlier draft of this spec — which named the short extend as "the population that exists only because of `treat_short_extends_as_decodes=False`" — specified a flagship arm that could not fail |
| **G1 arm (c3), on tokens alone** | whether a misclassification changed the *answer*. A wrong split produces fluent output; on a small synthetic model it can agree by luck. G2 asserts the four-tuple directly, and G2 — not G1 — is the gate that must red for B-M1 |
| **G1 arm (c3) on a FRESH state slot** | **the entire mode difference, again.** The split's only wire into the arithmetic is the gather mask, and over a zeroed slot mask `0` and mask `1` compute the identical answer (`nemotron_h_device.cpp:1423-1424`). The arm is well-formed, the four-tuples differ, and the tokens match. ★★★ is why the reclaimed slot is part of the arm's definition |
| **any `num_reqs == 1` gate** | everything in this unit. Named so no A2-P result is quoted as A2-B evidence |
| G1 | whether the graph path changed. Hence G4 |
| a passing G1 alone | whether the code was **reached**. Hence B-M8 |

### Mutations

Applied **alone**, in a scratch copy, rebuilt, run, tree restored to the
**baseline sha** — the restore is the control that catches `shutil.copy2`
preserving mtime so ninja skips the rebuild.

| # | Mutation | Must RED |
|---|---|---|
| B-M1 | the mamba split's `treat_short_extends_as_decodes` behaviour flipped back to `True` | **G2**, which asserts the four-tuple directly and is deterministic. **G1(c3) is expected to red too and is reported either way** — it is the only token arm that can see this mutation, **and only when its row sits on a reclaimed state slot** (★★★). If it survives, report the row's slot and its previous tenant alongside, because a survival on a clean slot is a fixture result and a survival on a dirty one bounds what the token gate can see. **G1(c1), (c2) and (c4) will NOT red, by construction** (see ★★); their staying green is the control, not a defect |
| B-M2 | the `mamba_attn.py:445-461` promotion dropped | G2's promotion cases, and **G1(c4)** — the short-extend arm is the one the promotion governs. Report G1(c3) either way; it should NOT move, because the promotion never fires on a no-prior-state row. If G1(c4) survives, say so, because that bounds what the token gate can see |
| B-M3 | `GdnStateScatter` widened to write **every** slot rather than only the named rows | G1(a) — this is the cross-request contamination the scatter's own comment claims to prevent |
| B-M4 | the fresh-request state zeroing dropped | G1(a) on the first step after a slot is reused — **and G1(c3), which runs on a reclaimed slot by construction (★★★) and is the same mechanism seen from the other side**. If both survive, the fixture is not actually reusing a dirtied slot, which also disarms G0's flagship; fix the fixture before recording a surviving mutation. If they survive on a fixture whose slot reuse is asserted, the gate is blind to the loudest trap in the unit and the row owes a direct assertion on the zeroed rows |
| B-M5 | `init[r] = 1 for r < nd` changed to `init[r] = 1` for **all** `r` | **G1(c2)** and **G1(c3)** — the arms that carry a row classified as a prefill, which is the only place the `r < nd` branch differs. If both survive, the mixed-population arms are not reaching the prefill classification and the harness is not the one described. D4's assertion cannot catch this one (see D2) |
| B-M6 | both requests pointed at the **same** state slot | D2's distinctness assertion. **G1 must also RED**; report as a pair — a distinctness check that reds while the tokens stay correct means the fixture is not actually sharing state |
| B-M7 | the ordering assertion removed and a deliberately unordered batch fed | G3 |
| B-M8 | the production call site deleted — `ForwardNemotronHForCausalLM`'s paged branch removed, host arm left in place | the focused gate must RED. A gate that stays green without the call site measures a class, not a capability (AGENTS.md §"Nothing lands dead"; method at [`reachability.md`](../reachability.md)) |
| B-M9 | the conv page dtype widened to f32 | the memory-format assertion. **The token gate must NOT red** — that asymmetry IS the demonstration that a token gate cannot see a too-wide dtype. Report as a pair, not as a failure |

**Report per mutation, all of it, every time:**

- the exact `[doctest]` `test cases:`, `assertions:` **and** `Status:` lines;
- a **non-zero case count**. `assertions: 0` is a skip wearing a pass, and the
  `assertions:` line can read `0 failed` while cases threw;
- `git diff --stat`, proving the edit applied. A mutation that never applied
  reads as a passing test;
- the compile exit code **and** error count. A mutation that fails to build reads
  as a passing test;
- a binary sha256 distinct from baseline. **This is necessary and never
  sufficient**: this project's build is not byte-reproducible — the same source
  SHA has produced two different binary hashes on this box — so a *distinct*
  sha256 is consistent with the mutation never having applied. `git diff --stat`
  is the item that actually carries the claim; the sha256 only catches the
  opposite error, a rebuild that did not happen (the `shutil.copy2` mtime trap
  above).

**The scratch mutation path must be SESSION-UNIQUE.** The scratchpad on this box
is shared across concurrent sessions, and two agents have already overwritten
each other's mutation script mid-run there. Put the scratch tree and every script
under a per-session directory, and take a `sha256` of the script *before* the run
that you re-check *after* it. A mutation report from a script another session
rewrote is not evidence about this tree.

**Never put a comma in a `TEST_CASE` name.** doctest's `-tc` splits filters on
commas, selects zero cases, prints `SUCCESS!` and exits 0. Assert the case count
moved; do not read the word `SUCCESS`.

**Never grep a run log for `\bok\b`.** ANSI colour codes defeat the word
boundary, and a green run has already read as `0 ok / 1 FAIL` in this tree.

### Reachability

The gate enters through `ModelRegistry::Forward` from a `GPUModelRunner` built on
a NemotronH `KVCacheConfig`, which is a production entry point. B-M8 is the
proof. Method: [`reachability.md`](../reachability.md).

### Gate hosts

Inherited from A2-P §5.7 and not re-derived here: `dgx.casa` (GB10, sm_121a) is
the primary host and is a **fleet device** — claim it with `rc run` / `rc hold`,
never `ssh` plus `flock`, because the fleet cannot see that mutex and a bypass
makes the box report free while somebody is on it. Thor (sm_110) is the portable
leg and is not a substitute for Marlin work
([#962](https://github.com/mudler/vllm.cpp/issues/962)). The local x86_64 box is
a development arm, not a gate host.

---

## Dependencies

| Depends on | State |
|---|---|
| A2-P, landed | **MET.** `5f68e60df` carries the paged forward and the narrowed G-SAFE clause |
| the runner's decode-first reorder | **MET.** `runner.cpp:126-214`, unconditional at `:1257` |
| `vt::GdnStateGather` / `GdnStateScatter` indexed forms | **MET.** Both already take the `[R]` index vector |
| `vt::PagedAttention` mixed-batch arm on each gate backend | **UNVERIFIED.** D1; verify before implementing, and file rather than widen if it fails |
| a NemotronH checkpoint on the gate host | inherited from A2-P §5.7; the driver is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| A2-Q1 (#940 **(closed)** / PR #1289), A2-Q2b | **NOT** dependencies for the correctness gate. Textual rebase only (D5). A2-Q1's device mamba arms are **held `DRAFT`: both diverge at 95/96, only the host arm passes 96/96** (D6), and every A2-B correctness leg runs on the HOST arm, named in the recipe |
| the throughput axis #1395 says A2-B unblocks | **PENDING [#1289](https://github.com/mudler/vllm.cpp/pull/1289)** for the device arm. The gated baseline against vLLM remains **`718.2x`** on the host arm; the device arm's `7.28x` warm-basis gain is an **ungated projection** while both device legs fail their token gate (D6) |

---

## Work breakdown

Non-overlapping, and W1 is deliberately a read rather than an edit.

| W | Work | Output |
|---|---|---|
| W1 | verify D1 (the `vt::PagedAttention` mixed-batch arm) and re-derive the §Our baseline audit at the implementer's own base | a finding, in the PR body. If D1 fails, stop and return `NEEDS_DECISION` |
| W2 | the mamba split + the `is_prefilling` metadata field + the ported upstream cases (G2) | red-first, then green; GDN's `SplitDecodesAndPrefills` cases byte-identical |
| W3 | rewire the NemotronH path to the mamba split; D2's distinctness check; D-B2's diagnostic fix | |
| W4 | drop the registry clause, add D4's ordering assertion, rewrite the interlock case (G5) | G0's red captured first |
| W5 | the batched gate G1(a)(b) and all four (c) arms, **(c3) first** because it is the one that fails today + G4 counters | |
| W6 | mutations B-M1…B-M9, each with the full report | |
| W7 | records (§Records owed) | rides in the same PR |

---

## Risks and decisions

- **R1 — changing `SplitDecodesAndPrefills` in place would silently change
  Qwen3-Next.** It is shared, and its current semantics are correct for its
  current caller. **Decision: add a second entry point; leave the existing one
  and its cases byte-identical.** A reviewer diffs `gdn_attn.cpp` for exactly
  this.
- **R2 — an `is_prefilling` vector that is empty reads as "nothing is
  prefilling".** That is the whole `treat_short_extends_as_decodes=False`
  behaviour disabled, silently, on any path that forgets to populate it.
  **Decision: the mamba split refuses on an empty vector**, mirroring the
  `assert common_attn_metadata.is_prefilling is not None` at `utils.py:624`
  (guarded by `if not treat_short_extends_as_decodes:` at `:623`). Upstream
  asserts; so do we.
- **R3 — misclassification produces fluent wrong tokens, never an error.** It is
  the same failure class the G-SAFE clause was built for, one level down.
  **G1(c3), G2 and B-M1 are the only things that can see it** — and of those, G2
  is the one that must red, because a token gate over a small synthetic model can
  agree by luck on a classification error (★★), and because G1(c3) can only see
  it at all when its row sits on a reclaimed state slot (★★★).
- **R4 — a token gate at `num_reqs == 1` will pass throughout.** Every existing
  NemotronH case is single-request. **None of them is evidence for A2-B**, and a
  PR body that quotes them as if they were should be rejected.
- **R5 — the CUDA-graph population changes as a side effect of deleting a
  refusal.** D5; G4 measures it rather than reasoning about it.
- **R6 — #1217's `device_token_ids` splice has never been observed at `R > 1`.**
  If G1 diverges only on the device path with a null-pointer CPU control passing,
  that is the #1217 signature and it is filed, not repaired here.
- **R7 — the serial per-request mamba loop makes a batched step slower per token
  than a single-request one.** Expected, and **not** a defect of this unit. A2-B
  records no speed number precisely so that this cannot be reported as a
  regression or as a win. D6 now gives this a measured mechanism rather than an
  expectation: the per-output-token cost is host-dominated (a device mamba arm
  cuts it 7.28x on a warm basis, though that arm fails its token gate and the
  figure is an ungated projection), so `R` serial host mixer calls are the term
  batching does not amortise.
- **R8 — `merge-tree` clean is not "the merge builds".** With A2-Q1 and A2-Q2b
  live on the same file, the focused gate runs on the merge result.
- **R9 — a tree carrying either of A2-Q1's device mamba arms produces a 95/96
  that is not A2-B's.** D6. **Both** device arms diverge; only the host arm
  passes 96/96. Every A2-B correctness leg names its arm in the recipe, and an
  unlabelled near-miss is not evidence about batching. Do not read the leg named
  `a3_off` as "device arm off" — it is a device leg.
- **R10 — the ungated `108.2x` projection will be quoted as if it retired the
  gated `718.2x`.** D6, and it has already happened once in this spec's own
  drafting. Both device legs fail their token gate, so `718.2x` remains the
  accepted figure and `108.2x` is a projection pending #1289. A number quoted
  often becomes treated as measured; grep its origin before carrying it.
- **R11 — G1(c3) is green on a clean state slot, and reads as "the flag is
  already mirrored".** The split reaches the forward's arithmetic through exactly
  one wire, the `vt::GdnStateGather` mask, and over a zeroed slot both mask
  values compute the identical answer (★★★,
  `nemotron_h_device.cpp:1423-1424`). The GDN state cache is allocated zeroed
  (`runner.cpp:484`, `:491`), so the natural fixture — a one-token prompt beside
  a decoding request on a fresh runner — is well-formed, differs in its
  four-tuple, and passes. **Decision: the reclaimed state slot is part of arm
  (c3)'s definition and the test asserts the reuse**, and stop condition 2 puts
  the slot ahead of the fixture shape in the diagnosis order. This is the same
  failure the short-extend draft had, one level down, and it survived one
  repair.

### Stop conditions

Stop and report rather than widening:

1. **D1 fails** — `vt::PagedAttention` does not serve a mixed batch on a gate
   backend. Return `NEEDS_DECISION`; do not write an attention kernel.
2. **G0's red does not appear on arm (c3)** — the discriminating arm passes on an
   untouched forward. This is **not** a stop condition until the two ordinary
   explanations are excluded, in this order (G0 states them in full):
   **first, the state slot was clean** — the split reaches the arithmetic only
   through the gather mask, and over a zeroed slot both mask values compute the
   identical answer (★★★), so a (c3) arm on a fresh runner is green for a reason
   that is not a finding; **second, the row was not the (c3) row** — it must
   carry `query_len == 1`, `seq_len == 1` and `is_prefilling` together.
   Report the transcript, the metadata the arm actually built, the row's
   state-slot index, and whether that slot had a previous tenant. A green on any
   other arm is the predicted result and is **not** a stop condition (G0). Do not
   proceed as if (c3) had failed, and do not weaken the gate to manufacture a
   red.
3. **A mutation cannot be made to red** after the harness has been checked
   against the four green-but-proves-nothing shapes (zero cases, comma filter,
   failed build, unapplied diff). Report the surviving mutation as an open gap.
4. **The oracle leg cannot run** on the gate host — record `PENDING` naming the
   external blocker. A pending result is a result; silence is not.
5. **Graph capture engages** for a multi-request NemotronH step. Refuse capture
   by name, file the driver as owed, and do not write one.
6. **A fix needs its own spec** — a shared-seam change or a checker-semantics
   change. AGENTS.md routes that to the normal row, spec and fresh-review path,
   not to this flow.

Attempt budgets control scheduling and never stop a correctable finding.

---

## Records owed on landing

The implementing change moves this row's lifecycle state, so it owes, in the
**same** change:

- `docs/STATUS.md`;
- `docs/BENCHMARKS.md` — **pending, failed or void is a result; silence is not.**
  A2-B records no speed number, and *that* is what BENCHMARKS records: the axis
  is now measurable and unmeasured, with #1395 naming why every prior Nemotron
  ratio was bounded;
- `docs/FEATURES.md` — the `NemotronHForCausalLM` row;
- this spec's `## Now`, and the row + checklist entry + rollup at
  `.agents/model-matrix.md:285` (`scripts/check-model-checklist.py` enforces the
  rollup);
- `docs/USAGE.md` if the reachable capability's checkpoint documentation changes;
  run `scripts/check-doc-checkpoint.py` rather than assuming either way;
- `scripts/runner-routing-allowlist.txt:26` — **verified stale-free at this
  commit; the expected result is NO EDIT.** A2-B does not change the entry's
  disposition (it is pending A2-Q2b's device `lm_head`), and its prose makes no
  claim the request count can falsify: it describes the paged forward, the NVFP4
  `lm_head` refusal at `nemotron_h.cpp:1031-1034` and A2-Q2b, and
  `grep -i single scripts/runner-routing-allowlist.txt` returns nothing.
  Re-run that grep at the implementing commit rather than assuming either way; an
  obligation with nothing to act on is one an implementer discharges by inventing
  a change.

`.agents/NOW.md` is authored at operator cadence and is not a per-row lifecycle
write.

**If the implementing change edits this spec's `## Gates` section to carry
runnable gate evidence — a named test binary with case and assertion counts and
an exit status — it moves this row into the runnable population and must re-pin
`RUNNABLE_BASELINE` in `scripts/check-gate-commands.py` in the SAME change.**
Not doing so reds `main` on `tests/scripts/test_check_gate_commands.py`; that has
already happened once, on [#1376](https://github.com/mudler/vllm.cpp/issues/1376).

---

## Now

**State at this commit: SPEC ONLY. No product code is written, no gate has run,
and no lifecycle state moves.**

What this spec established that was not known when #1395 was filed:

1. The pinned oracle's mamba backend passes
   `treat_short_extends_as_decodes=False` (`mamba_attn.py:467`) and never passes
   `require_uniform`. Our `SplitDecodesAndPrefills` implements the opposite flag.
   **That is a live mirroring gap the `num_reqs <= 1` refusal has been hiding**,
   and it is the substance of the unit.
1a. **A short extend does NOT discriminate the two modes**, although it is the
   population the flag is named for. The promotion at `mamba_attn.py:450-453`
   always fires on one, because a short extend has prior context by definition
   (`runner.cpp:144`) and therefore `seq_len > 1`. The only discriminating
   population in A2-B's scope is a still-prefilling row with `query_len == 1` and
   **no** prior state (`seq_len == 1`). A gate built on the short extend cannot
   fail, and the first draft of this spec specified exactly that gate (★★).
1b. **A differing split is not a differing forward.** The split reaches the
   NemotronH forward's arithmetic through exactly one wire — `init`, which is
   the `vt::GdnStateGather` mask — and over a ZEROED state slot mask `0` and
   mask `1` compute the identical answer, which A2-P's own comment states at
   `nemotron_h_device.cpp:1423-1424`. The GDN state cache is allocated zeroed
   (`runner.cpp:484`, `:491`). **So arm (c3) discriminates the two modes only
   when its row lands on a slot a finished request already dirtied** — which the
   runner's LIFO free list makes deterministic (`runner.cpp:1153-1160`,
   `:1177-1178`). The reclaimed slot is part of the arm, and the second draft of
   this spec specified a flagship arm that was green without it (★★★).
2. A2-P's `:64` under-describes what A2-P landed. The per-request state indexing
   exists; A2-B verifies and hardens it rather than building it.
3. The runner already reorders decode-first, unconditionally.
4. `#1395`'s call-site anchor is `mamba_attn.py:464`, not `:463`.
5. A2-Q1's device mamba arms are **held `DRAFT`**: on GB10 **both** device legs
   read 95/96 and only the host arm reads 96/96. The device arm is worth
   **7.28x per output token on a warm basis**, but that is an **ungated
   projection** — `718.2x` remains the gated baseline, because AGENTS.md forbids
   accepting a performance result before the token gate is established. At ~10.2%
   GPU busy the decode stays ~90% GPU-idle, so A2-Q1 banks a multiple and does
   not close the gap. The host-side term remains the thing A2-B's batching does
   *not* amortise (D6).

## Owed

- The upstream-test port for the `treat_short_extends_as_decodes=False` modes is
  owed **by this row**, in the implementing change, not deferred.
- Mamba prefix caching (`mamba_cache_mode == "all"`) remains unported and, unlike
  batching and speculation, is **not refused by name** anywhere. If no issue
  tracks it when the implementer looks, file one and name this row as its owner.
- [#1217](https://github.com/mudler/vllm.cpp/issues/1217) —
  `ModelForwardInput::device_token_ids` has no per-model opt-in. A2-P's `## Owed`
  carries it; A2-B's G1 is the first gate that can observe it at `R > 1`.
- The throughput comparison #1395 names as blocked. A2-B unblocks it and does not
  perform it.
- `require_uniform` is classified **not applicable** rather than deferred, on the
  finding that `mamba_attn.py` never passes it (§Upstream chain). That
  classification is only as current as the pin. **Re-derive it at the next pin
  advance, and again if a spec-decode row is ever admitted here** — `:200`
  widens `decode_threshold` past 1 under spec decode, and `require_uniform`'s
  behaviour is only inert while the threshold is 1. Insurance, not a known gap.
- `seq_lens_cpu_upper_bound` (`vllm/v1/attention/backend.py:464-468`) is
  **substituted** by our `seq_lens_cpu`, sound only because A2-B refuses
  speculative rows (Port map). If a later row lifts that refusal, the real field
  is owed before the promotion predicate can be trusted.

## Outcome

Not yet written. Per AGENTS.md this section is filled when the unit reaches
`DONE`: what was measured, what was rejected and why, and why each default has
its value — including, explicitly, the D1 result, the G4 graph-dispatch counters
whichever way they fell, and the disposition of every mutation that did not red.
