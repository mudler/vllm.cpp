# ENG-RECURRENT-MULTISTATE — a recurrent layer carries N states, because upstream's does

Issue: [#2131](https://github.com/mudler/vllm.cpp/issues/2131).
Row: `ENG-RECURRENT-MULTISTATE`.
Kind: ENGINE. This is shared machinery every recurrent model in the tree runs
through, not a model port.

**The index row for #2131 already exists**, appended when the issue was filed and
keyed to `MODEL-MM-QWEN4-EXP`. `.agents/issue-index.md` is append-only under
`merge=union`, and a second row for the same issue number is refused as exactly
the duplicate two branches appending one issue would produce — verified by
running `scripts/check-agent-record.py` against an appended row and reading its
refusal. So this row appends nothing there, and this document is where the issue
is linked from the work.

vLLM registers `MambaSpec`, `MambaBase` and the recurrent half of the GPU runner
at the parity pin, so vLLM is the mirror source for every decision here and no
secondary oracle is admissible. Pin: `5559679229bc961848b121ccdeaa8fa5d79bec98`,
verified with `git -C /home/mudler/_git/vllm log -1` on 2026-08-28.

## Now

`ACTIVE`. This document is written before the code it scopes, and the wave it
scopes is the FIRST of at least two: it makes a recurrent group carry N states,
and it leaves per-layer state heterogeneity inside one group to a later wave.
`## Owed` names both, with the upstream anchor for each.

## Scope

`GPUModelRunner::initialize_kv_cache` refuses any recurrent group whose
`MambaSpec` does not carry EXACTLY two shapes and two dtypes:

```cpp
VT_CHECK(mamba_spec->shapes.size() == 2 && mamba_spec->dtypes.size() == 2,
         "runner: recurrent MambaSpec must contain conv then temporal state");
conv_state_shape = mamba_spec->shapes[0];
ssm_state_shape  = mamba_spec->shapes[1];
```

and `GdnStateCache` carries exactly two named tensors, `conv_state` and
`ssm_state`. Between them, a recurrent layer in this tree cannot hold a third
state at all.

In scope: the state COUNT and the per-state dtype, end to end — spec read,
allocation, byte accounting, and the view carrier the models read.

Out of scope, each named under `## Owed` rather than dropped: per-layer state
heterogeneity within one recurrent group, a state count of ONE, a SECOND
recurrent group, and any model that publishes N >= 3.

## The finding: vLLM never had a two-state assumption, and we invented one

The issue's premise is that upstream may not express this either. It does, it
expresses it fully generally, and it SHIPS three different values of N at the
pin. Read at `5559679229`:

| Upstream | Anchor | What it says |
|---|---|---|
| the carrier | `vllm/model_executor/layers/mamba/abstract.py:26` | `kv_cache: tuple[torch.Tensor, ...]` — an ordered tuple of unbounded length, NOT a named `(conv, ssm)` pair |
| the unpack | `abstract.py:29-43` `bind_kv_cache` | `for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype())`, slicing one page at a running byte offset. N states, each with its OWN shape and its OWN dtype |
| the contract | `abstract.py:46-52` | "For mamba layers this is **usually** a (conv_state, ssm_state) tuple". Two is a convention the docstring itself hedges |
| N == 1 | `vllm/model_executor/layers/mamba/short_conv.py:87` | `self.kv_cache = (torch.tensor([]),)` |
| N == 5 | `vllm/model_executor/layers/mamba/mamba_mixer2.py:517-520` | `_n_state = 5 if self.use_replayssm else 2`, and `:722-724` `x_cache, dt_cache, B_cache = self.kv_cache[2:]` |
| N == 5 shapes | `vllm/model_executor/layers/mamba/mamba_utils.py:202-221` | the three appended shapes are rank 3, rank **2** and rank 3 — a rank change inside one layer's state set |
| N == 5 dtypes | `mamba_utils.py:84-93` | `(*base_dtypes, activation_dtype, torch.float32, activation_dtype)` — a `float32` beside two activation dtypes |
| the runner | `vllm/v1/worker/gpu_model_runner.py:7429-7440` | allocates `num_blocks * page_size_bytes` RAW int8 and hands the layer one untyped page. The runner never learns N, and cannot |
| the spec | `vllm/v1/kv_cache_interface.py:698-707` | `page_size_bytes` is `sum(prod(shape) * get_dtype_size(dtype))` over the zip — already N-general |

Our `MambaSpec` (`include/vllm/v1/kv_cache_interface.h`) already mirrors the last
row: it holds `std::vector<std::vector<int64_t>> shapes` and
`std::vector<vt::DType> dtypes`, and `MambaSpec::page_size_bytes` sums over both.
`vllm::v1::recurrent_state_bytes` reads nothing but `page_size_bytes()`. **The
two-shape assumption exists in exactly two places, the runner and the state
carrier, and both are local inventions.** That is why this is a repair and not a
feature.

## Design

Mirror `bind_kv_cache`. The recurrent cache becomes an ORDERED LIST of states
whose length, per-state shape and per-state dtype all come from the group's own
`MambaSpec`.

1. **`GdnStateCache` grows `std::vector<vt::Tensor> states`** — the mirror of
   `kv_cache: tuple[torch.Tensor, ...]`. `conv_state` and `ssm_state` stay, and
   are `states[0]` and `states[1]`. Every existing consumer — `qwen3_5.cpp`,
   `kimi_linear_device.cpp` and the `nemotron_h` pair `nemotron_h_device.cpp` /
   `nemotron_h_forward.h` — reads those two names and is untouched. **That is
   THREE families, and this line said four
   ([#2203](https://github.com/mudler/vllm.cpp/issues/2203), fixed in flow under
   W5c-1 of [#2031](https://github.com/mudler/vllm.cpp/issues/2031)).** The
   removed fourth name was `gemma4_mm.cpp`, which reads NEITHER field — zero
   occurrences of `conv_state`, zero of `ssm_state` — and whose only two
   mentions of the type are an include comment and
   `std::vector<GdnStateCache> no_gdn_state;` (`gemma4_mm.cpp:221`), passed
   EMPTY: the file that proves Gemma-4 has no recurrent arm, cited as proving
   the opposite. `muse_glimmer_mm.cpp:340` and `qwen3_vl.cpp:621` carry the same
   empty-vector shape. Grepping the FIELD name over-counts the other way —
   `glm5_next_kda.cpp` matches `conv_state` 13 times on
   `Glm5NextKdaCache::conv_state`, a `std::vector<float>` KDA sequence state
   (`glm5_next_kda.h:314`) and not this `vt::Tensor` (`qwen3_5.h:111`), with
   zero occurrences of `GdnStateCache`. Grep the TYPE. The paragraph's CLAIM is
   unaffected: three untouched consumers is still why `conv_state` and
   `ssm_state` stay as names.
2. **The runner's recurrent geometry becomes vectors over N.** One
   `CacheBuffer` per (recurrent layer, state), allocated in SPEC ORDER, which is
   the order `bind_kv_cache` slices in. `kv_cache_allocated_bytes` sums every
   one of them, so the memory the runner reports stays the memory it took.
3. **The refusal widens from `== 2` to `>= 2`, and keeps a length agreement
   between `shapes` and `dtypes`.** The widening is justified by the upstream
   anchor and by expressibility, and by nothing else. It fixes no bug: at the
   base tree a two-shape/one-dtype spec was ALREADY refused twice over, by the
   conjunctive `shapes.size() == 2 && dtypes.size() == 2` at
   `src/vllm/v1/worker/gpu/runner.cpp:916-917` before any `dtypes[1]` was read,
   and by the `shapes.size() != dtypes.size()` throw at
   `src/vllm/v1/kv_cache_interface.cpp:210-213` ahead of its own zip. An earlier
   draft of this row claimed the length check was new and closed an
   out-of-bounds read; it was not, and it did not.
4. **The per-state dtype predicate widens from `{F16, BF16, F32}` to any
   non-block-quantized `vt::DType`.** `bind_kv_cache` imposes no dtype
   constraint at all; the local floating-only rule was justified by "all-zero
   bytes are `+0.0f` for every supported floating storage type", which is
   equally true of an integer zero. The real constraint is that a block-quant
   dtype has no per-element size, and that is what the widened predicate names.
   This is what makes an INTEGER state expressible — a `qwen4_exp` PLE layer's
   n-gram history holds `input_ids.long()`, i.e. token ids and not activations.

Widening an assertion is a semantic checker change, so it lands red-first: the
new test is RED at the base tree for BOTH halves (the count and the dtype), and
the widening is justified by the upstream anchor rather than by making a gate
green.

### What this wave deliberately does NOT do

`gdn_group_id_` stays a scalar and the `recurrent_seen > 1` refusal stays. Both
therefore REMAIN owed, and both ARE on the path a `qwen4_exp` PLE topology takes.
[#2131](https://github.com/mudler/vllm.cpp/issues/2131) reads the one-group limit
as the second half of the same blocker, and that reading is correct. An earlier
draft of this document claimed otherwise, on a mistaken reading of the upstream
grouping; the correction is recorded here rather than quietly dropped, because
the wrong version was what a reader would have planned the next wave against.

Upstream serves per-layer state heterogeneity two ways, and only ONE of them
keeps heterogeneous recurrent layers in a single group:

- **The uniform-TYPE path, which does.** `UniformTypeKVCacheSpecs`
  (`vllm/v1/kv_cache_interface.py:817`) merges layers of the same type into one
  group, and `MambaSpec.is_uniform_with_collection` (`:732-739`) tests only
  `isinstance(spec, MambaSpec)` and an equal `num_speculative_blocks`. It ignores
  `shapes` and `dtypes` entirely, so two recurrent layers with different state
  sets ARE uniform to it, and `_get_kv_cache_groups_uniform_type`
  (`kv_cache_utils.py:1039-1053`) returns exactly one group. But
  `get_kv_cache_groups` reaches that branch only at `:1786`, via
  `UniformTypeKVCacheSpecs.from_specs`, which asks the FIRST spec whether it is
  uniform with the WHOLE model. A single attention layer makes the answer False.
- **The page-size PADDING path, which does not.**
  `unify_kv_cache_spec_page_size` (`:1070`) only equalises the page SIZE: for a
  `MambaSpec` it sets `page_size_padded=max_page_size` (`:1099-1110`) and returns
  a spec that is otherwise unchanged. Its output feeds
  `_get_kv_cache_groups_uniform_page_size` (`:1140`), whose grouping is
  `same_type_layers[layer_spec].append(layer_name)` (`:1210`) — a dict keyed by
  the SPEC OBJECT. `MambaSpec` is `@dataclass(frozen=True)`
  (`kv_cache_interface.py:689`), so that key is field-wise equality INCLUDING
  `shapes` and `dtypes`. Equal page sizes are not an equal key, and the odd layer
  lands in a group of its own.

`qwen4_exp` is hybrid — `src/vllm/model_executor/models/qwen4_exp_gguf_weights.cpp:152-162`
builds a 3 × `linear_attention` : 1 × `full_attention` schedule out of
`full_attention_interval` — so it falls to the padding path and gets MORE THAN
ONE recurrent group. Measured by running upstream's own two functions at the pin
`5559679229`, on two identical GDN layers plus one PLE-shaped layer carrying a
third conv state and an `int64` n-gram history:

```
page sizes before padding: {'gdn0': 1536, 'gdn1': 1536, 'ple2': 2360}
page sizes after padding : {'gdn0': 2360, 'gdn1': 2360, 'ple2': 2360}
NUMBER OF RECURRENT GROUPS after padding: 3
DISTINCT MambaSpec dict keys after padding: 2
uniform-type from_specs on the RECURRENT-ONLY dict: True
uniform-type from_specs on the HYBRID model: False
```

The last two lines are the whole argument: were `qwen4_exp` wholly recurrent,
upstream would hand it one group; because it is hybrid, it takes the padding
path, and padding equalises the pages while the grouping still splits — two
distinct spec keys, three groups once the equal-group-size split at `:1210` and
below runs. A PLE topology therefore needs BOTH halves, per-layer state sets and
more than one recurrent group. Neither is in this wave, and `## Owed` carries
both.

## Risks

- **Silent byte drift on the existing arms.** Four model families flow through
  these lines. Mitigated by an existing literal byte-neutrality case
  (`test_runner.cpp`, "the Qwen3.5 allocation is BYTE-IDENTICAL after #810") and
  by running the recurrent suites before and after and comparing case and
  assertion counts exactly.
- **A cosmetic generalization.** A vector that is only ever length 2 proves
  nothing. Mitigated by a mutation that reverts the loop to `states[0..1]` and
  must RED the new case, and by shapes chosen so the third state genuinely
  changes the allocated bytes, the view count and the reported total.
- **The reverse: the OLD path stops being exercised.** Mitigated by a mutation
  inside the two-state path that must RED an EXISTING recurrent suite.

## Tests

`tests/vllm/v1/worker/test_runner.cpp`:

- a THREE-state recurrent group, with a third state of a different rank, a
  different element count and a different dtype from either of the first two:
  three buffers, three views, `page_size_bytes` identity over all three, and
  `kv_cache_allocated_bytes` counting the third.
- an INTEGER third state (`kI64`), which is what a token-id history is.
- a spec whose `shapes` and `dtypes` disagree in length is REFUSED.
- a block-quantized state dtype is REFUSED.

## Gates

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVLLM_CPP_BUILD_EXAMPLES=OFF
ninja -C build -j 6 test_runner test_qwen27_paged_forward test_nemotron_h_paged_forward test_kimi_linear_paged
./build/tests/test_runner
./build/tests/test_qwen27_paged_forward
./build/tests/test_nemotron_h_paged_forward
./build/tests/test_kimi_linear_paged
scripts/agent-preflight.sh --fail-on-skip
```

The three model suites are the regression gate named by the issue. Their case
and assertion counts are recorded in `## Outcome` before and after, because a
count that moved is the only thing that can see a case that stopped running.

## Stop conditions

Return `NEEDS_DECISION` rather than redesigning silently if the per-layer
heterogeneity turns out to be reachable only by changing `GdnStateCache`'s two
named fields, because that is a four-model blast radius and a different review.

## Outcome

Landed as the FIRST wave. The runner reads the state COUNT and every per-state
dtype off the group's own `MambaSpec`; nothing in the runner or the state carrier
names two any more.

### Regression counts, before and after, on the same tree

Read from the suite output, not from an exit code. The base is `8997c62b3`.

| Suite | Before | After |
|---|---|---|
| `test_runner` | 29 cases / 831 assertions / rc 0 | 31 cases / 884 assertions / rc 0 |
| `test_qwen27_paged_forward` | 31 / 770 / rc 0 | 31 / 770 / rc 0 |
| `test_nemotron_h_paged_forward` | 13 / 3269 / rc 0 | 13 / 3269 / rc 0 |
| `test_kimi_linear_paged` | 8 / 206 / rc 0 | 8 / 206 / rc 0 |

The three model suites are byte-identical in both numbers. `test_runner` moves by
exactly the two cases this row adds.

### RED, before the change

The three-state case threw the production refusal it was written against:

```
test_runner.cpp:823: ERROR: test case THREW exception:
  vt: runner: recurrent MambaSpec must contain conv then temporal state
  at src/vllm/v1/worker/gpu/runner.cpp:916
[doctest] test cases: 31 | 30 passed | 1 failed
```

### Mutation record

Each mutation was sha256-proven applied, its BUILD status was read before any
test result, and the tree was restored byte-for-byte and re-measured afterwards.
The three mutations below were measured against `runner.cpp` at
`c01eb6ee8d522d7cd7816b97584e87152d03b1fa14a0e33551f57dfad2644527`, which is
the file as this row first landed it. The review repair that followed edited a
COMMENT in that file and nothing else, so the hash at the head differs while
the measured behaviour does not; `git diff` over the repair commit is the check
that says so. The M1 re-measurement recorded above was taken on the repaired
file, at `e538172d207f07e3b325dc2fd980d3e257368387b87b8378caa0bf0114c0332a`.

| # | Mutation | sha256 of `runner.cpp` | Build | Result |
|---|---|---|---|---|
| M1 | the widened refusal back to `shapes.size() == 2` | `f6c8d819…` | rc 0 | `test_runner` RED, 1 case, at the old message. The three model suites stay GREEN, so the mutation is scoped to the new arm |
| M2 | the VIEW loop reads `state_dtypes[i < 2 ? i : 0]` — the THIRD state's dtype mishandled, the first two untouched | `f4d7c05f…` | rc 0 | `test_runner` RED, 9 assertions, all on `states[2].dtype` and `states[2].Bytes()`. The three model suites stay GREEN. This is what makes the third state load-bearing rather than decorative: it is the only state whose answer moves |
| M3 | `gs.ssm_state = gs.states[0]` — the OLD two-state path, inside the same generalized loop | `ed8b76cf…` | rc 0 | `test_nemotron_h_paged_forward` RED (11 of 13 cases, 23 assertions), `test_kimi_linear_paged` RED (5 of 8, 11 assertions), `test_runner` rc 139. The existing arms genuinely run through the new loop |

M2 is the one that answers "is this cosmetic". The fixture's third state is a
different RANK (1-D against 2-D and 3-D), a different ELEMENT COUNT (7 against
192 and 256) and a different DTYPE (`kI64` against two `kF32`) from either of the
first two, so no implementation that reuses `shapes[0]`, `dtypes[0]` or a factor
of 2 can produce its bytes.

### The refusal case needed its MESSAGE, and a review found it did not have one

`TEST_CASE("runner: a malformed recurrent MambaSpec is REFUSED by name")` landed
asserting only `CHECK_THROWS`. Despite "by name" it asserted no name. MEASURED
under M1 — the widened refusal reverted to `shapes.size() == 2` — with the
as-landed case: it reads

```
[doctest] test cases: 1 | 1 passed | 0 failed | 30 skipped
[doctest] assertions: 3 | 3 passed | 0 failed |
```

GREEN, while the rest of `test_runner` reads 30 passed / 1 failed. All three
inputs still throw under the old refusal, at the old message, for a reason that
has nothing to do with what each subcase is named after. The case could not tell
the widened refusal from the one it replaced, so it gated nothing this row did.

Repaired by asserting the message rather than the throw. Each subcase now names
the substring only the widened code can produce, and the empty string stands in
for "nothing was thrown" so a silent acceptance fails the same check. Re-measured
under the same M1, on the same tree:

```
test_runner.cpp:921: ERROR: CHECK( msg.find("with one dtype per shape") ... )
  logged: refusal: vt: runner: recurrent MambaSpec must contain conv then
          temporal state at src/vllm/v1/worker/gpu/runner.cpp:951
[doctest] test cases: 1 | 0 passed | 1 failed | 30 skipped
[doctest] assertions: 3 | 0 passed | 3 failed |
```

All three subcases red, each on the OLD message. Green at the head with the same
3 assertions.

### The multi-cache recurrent allocation site is UNEXERCISED

`alloc_recurrent_layer_states` has two call sites, and only one is reached by any
test in this tree. Measured by deleting each in turn and rebuilding:

| Deleted call site | `test_runner` | `test_qwen27_paged_forward` | `test_nemotron_h_paged_forward` | `test_kimi_linear_paged` |
|---|---|---|---|---|
| the legacy single-topology path: the `is_gdn` arm of the `else` branch | rc 139, 10 of 13 reached cases failed | 31 / 770 / rc 0 | rc 139, 5 of 5 reached cases failed | rc 1, 2 of 8 failed |
| inside `if (multi_cache_topology)`: its `membership_by_name && has_mamba_group` recurrent loop | 31 / 884 / rc 0 | 31 / 770 / rc 0 | 13 / 3269 / rc 0 | 8 / 206 / rc 0 |

Both call sites are named by their enclosing predicate, not by a line number.
An earlier version of this table cited `runner.cpp:1259` and `runner.cpp:1339`,
and this row's own comment expansion in that same file moved both calls five
lines down in the same commit, so the record was already stale at the head that
carried it. The predicate survives a comment edit. The line number does not.

The first row is the control, and it proves the deletion harness is live: an
unreached site and a dead harness look identical without it. The second row is
the finding. Nothing in the tree combines a multi-cache topology with a mamba
group, so the branch's own comment ("no model shipping today reaches it") is
true of the tests as well. The debt predates this row — the site arrives with
KV-DSV4-MULTICACHE ([#2068](https://github.com/mudler/vllm.cpp/issues/2068)) —
but this row is what routes it through the shared helper, so its N-generality is
what is now unexercised there. Recorded under `## Owed`.

The first row also re-confirms, independently of M3, that
`test_qwen27_paged_forward` does not gate this seam: it is the one suite that
stays green while the runner loses the allocation every recurrent layer needs.

### A gate the issue named that does not gate this

[#2131](https://github.com/mudler/vllm.cpp/issues/2131) names
`test_qwen27_paged_forward` as the regression gate for this change. MEASURED: it
is not one. Under M3 — the runner handing every recurrent layer its conv state
where the temporal state belongs — that suite reads 31 cases / 770 assertions /
rc 0, unchanged, while `test_nemotron_h_paged_forward` and
`test_kimi_linear_paged` both go red and `test_runner` faults. The suite builds
its own `GdnStateCache` views rather than reading the runner's, so it cannot see
a defect in the runner's state assignment. The real regression gate for this seam
is those other three, and this row used all four.

### The correction W5c made to this bullet

Measured at the same pin, `5559679229`, and it does not overturn the numbers in
`### What this wave deliberately does NOT do`. Those numbers are correct **about
upstream's grouping FUNCTIONS**, and the premise they were fed is what is wrong:
they were run on "two identical GDN layers plus one PLE-shaped layer carrying a
third conv state and an `int64` n-gram history", and **upstream never constructs
that input**.

| Read at `5559679229` | What it says |
|---|---|
| `vllm/model_executor/models/interfaces.py:809-812` | `get_mamba_state_shape_from_config(cls, vllm_config)` is a CLASSMETHOD taking the CONFIG and nothing else. There is no `layer_idx` to vary a shape by |
| the same name, tree-wide | 19 definitions at the pin: this protocol declaration plus **18 implementations**. Not one of them takes a layer index, and each returns ONE shape tuple for the whole model |
| `vllm/v1/worker/mamba_utils.py:441` | `assert all(mamba_specs[0] == spec for spec in mamba_specs)` — every `MambaSpec` in the model must be EQUAL, field for field, `shapes` and `dtypes` included |
| `vllm/v1/core/kv_cache_utils.py:1101-1109` | when a `MambaSpec`'s page is smaller than the max, upstream sets `page_size_padded=max_page_size` and keeps the spec otherwise unchanged. It PADS. It does not split |

So a heterogeneous per-layer recurrent spec set is not a shape upstream is
reluctant to group — it is a shape upstream cannot produce, and `mamba_utils`
asserts against it one layer below. `MakeQwen4ExpKVCache` mirrors that: ONE
`MambaSpec` carrying `[gdn_conv, temporal, ple_conv, ngram]` on every one of the
36 linear-attention layers, and the 35 that never read the last two pay
184336 B per sequence each — 49.2 MiB at the default `max_num_seqs` of 8, which
is 0.09% of the GB10 headroom the `qwen4_exp` row's `## Hardware` accounts.
Derived from the published shapes; nothing has allocated it on a device.

Both halves therefore stay owed as ENGINE debt and neither blocks W5c.

## Owed

- **Per-layer recurrent specs, in MORE THAN ONE recurrent group.** This row
  closes neither, and it is generic engine debt rather than a `qwen4_exp`
  blocker. **CORRECTED at W5c of
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031)** — see
  `### The correction W5c made to this bullet` below, which is where the
  measurement and the anchors live. The sentence this replaces read "A
  `qwen4_exp` PLE topology needs both halves", on the premise that only ONE of
  its linear-attention layers carries the PLE conv and the n-gram history, so
  its `MambaSpec` differs from its siblings'. That premise describes a
  per-layer spec set upstream never constructs.

  What the earlier measurement showed remains TRUE OF THE FUNCTIONS and is kept
  for the row that eventually needs them: fed a heterogeneous per-layer spec
  set, upstream takes the PADDING path — `vllm/v1/core/kv_cache_utils.py`
  equalises the page size only, and the grouping key at `:1210` is the frozen
  `MambaSpec` itself, `shapes` and `dtypes` included, so two distinct spec keys
  gave three groups. What is false is that any model reaches those functions
  with that input, because `get_mamba_state_shape_from_config` declares ONE
  shape model-wide and `mamba_utils.py:441` asserts every spec equal.

  Seams to mirror WHEN a model needs it: the existing
  `KVCacheConfig::per_layer_attn_specs` for the per-layer spec, and a LIST of
  recurrent group ids in place of the scalar `gdn_group_id_` for the second
  group. `ComputeHybridKvBudget` would need the same widening — it keeps the
  FIRST mamba group and the FIRST attention group
  (`src/vllm/v1/core/hybrid_kv_budget.cpp:26` and `:33-39`) — although the
  paged BYTE divisor `KVBytesPerBlock` is already group-general and does count
  every attention group by its own layer list. Owned by a successor of this row
  rather than by W5c, which does not need it; tracked by
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131).
- **A recurrent group of ONE state.** Upstream's `ShortConv`
  (`short_conv.py:87`) has no temporal state. `GdnStateCache::ssm_state` is a
  named field every consumer reads, so N == 1 needs those consumers to stop
  assuming it, which this wave does not touch. Refused with a message naming the
  missing part.
- **A SECOND recurrent group.** `recurrent_seen > 1` still refuses, and
  `gdn_group_id_` is still a scalar. **CORRECTED at W5c of
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031): this is NOT on
  `qwen4_exp`'s path.** The sentence it replaces said the opposite — "This IS on
  a PLE topology's path, not beside it" — and it is kept here rather than
  deleted because it is what a reader would have planned the next wave against,
  which is the same reason its own predecessor was kept. `qwen4_exp` publishes
  ONE uniform recurrent group, and a scalar `gdn_group_id_` carries it. What
  remains genuinely owed is generic: `ComputeHybridKvBudget` reads only the
  FIRST mamba group (`src/vllm/v1/core/hybrid_kv_budget.cpp:26`), so a model
  that did publish two would be budgeted for one. No registry publishes that
  shape today.
- **`test_qwen27_paged_forward` does not gate the runner's recurrent state
  assignment**, measured above. Either it should enter through the runner's own
  `GdnStateCache`, or the issue text and any future dispatch should stop naming
  it as this seam's gate. Tracked by
  [#2131](https://github.com/mudler/vllm.cpp/issues/2131) until a row picks it
  up.
- **`GdnStateCache::states` is filled by the runner only.** The host-path
  scaffolds in `qwen3_5.cpp` and several test fixtures build the two named
  fields and leave the list empty. Inert while nothing outside the runner reads
  it; a consumer that starts reading `states` owes those builders the
  assignment.
- ~~**The multi-cache recurrent allocation site is UNEXERCISED.**~~ **CLOSED by
  W5c-1 of [#2031](https://github.com/mudler/vllm.cpp/issues/2031).** The
  measurement above stands: at this row's head, deleting the
  `alloc_recurrent_layer_states` call inside `if (multi_cache_topology)`, in its
  `membership_by_name && has_mamba_group` recurrent loop, left all four suites
  fully green. `test_runner.cpp`'s
  "runner: a multi-cache topology ALLOCATES its N-state recurrent group" is the
  fixture that was missing — a `qwen4_exp`-shaped topology publishing a paged
  K+V group, a recurrent group and an `MLAAttentionSpec` indexer side cache —
  and RE-MEASURED with the same deletion it now reads `test_runner` 32 cases /
  902 assertions / **rc 1**, failing only the new case at
  `REQUIRE(runner.gdn_state().size() == 3)`, while the three model suites stay
  byte-identically green. The existing "keeps its recurrent group" case could
  not see it because everything it asserts — `layer_kv_class_`,
  `gdn_group_id_`, the per-layer index lists — is computed BEFORE the
  allocation.
- ~~**Nothing publishes N >= 3.**~~ **CLOSED by W5c-1 of
  [#2031](https://github.com/mudler/vllm.cpp/issues/2031)**, which is the wave
  this bullet named. `MakeQwen4ExpKVCache` publishes **N == 4** —
  `[gdn_conv, temporal, ple_conv, ngram]`, the last of them `kI64` because it
  holds token ids — on every one of `qwen4_exp`'s 36 linear-attention layers,
  reached through the production `make_kv_cache` registry hook and gated by
  `tests/vllm/models/test_qwen4_exp_kv_cache.cpp`. Both halves of the widening
  this row landed are therefore now used by a shipped registry: the COUNT
  (4 > 2) and the DTYPE (an integer state, which the old floating-only
  predicate made inexpressible).
