# `MODEL-DSV4-DSA-COMPOSE` — the DeepSeek-V4 DSA composition

Issue: [#2286](https://github.com/mudler/vllm.cpp/issues/2286).
Owning row: `MODEL-DSV4-DSA-COMPOSE` (`.agents/model-matrix.md`).
Oracle: vLLM at the parity pin `5559679229`, `vllm/models/deepseek_v4/`.

## Now

`READY` — this spec is the deliverable of the scoping wave. No implementation
has started. W1 cannot begin until `KV-DSV4-MULTICACHE` **W5** lands, and W5 has
no owner today (#2302; W3, which an earlier revision of this spec named, landed
as `ca3dcda21` on 2026-08-27). See `## Dependencies`.

## Scope

Assemble the DSA primitives into `AttentionBlock` so the DeepSeek-V4 forward
stops refusing on the 21 `compress_ratio == 4` layers.

**In scope:** the composition — layer-shape dispatch, the compressor's two
stages, the indexer's placement and its `qr`-sourced query, the overlapped
window with role selection, and boundary-only emission into the compressed KV
cache.

**Out of scope, with owners:** the two kernel primitives themselves
(`KERNEL-ATTN-DSA-SPARSE-INDEX`, `KERNEL-ATTN-DSA-COMPRESSOR`, both `SPIKE`);
the cache topology (`KV-DSV4-MULTICACHE`, #1925); residency (#2283); the
attention sink, which is a loaded per-head weight and not cache state
(`attention.py:218-222`).

**The boundary with `KV-DSV4-MULTICACHE` W5, which this row overlapped when it
was created** (#2302). W5's scope lived in a single wave-table cell reading "the
DSA-sparse attention path reading the published caches" and removing the
`!is_indexer && !is_comp` refusal -- the ALGORITHM, not the plumbing -- and it has
never had a design section, so this row was specced over it. The split, recorded
in both specs:

| row | owns |
|---|---|
| `KV-DSV4-MULTICACHE` W5 | the caches REACHING the model and each layer routing to its own: `attn_kv` consumed rather than `(void)`-ed, and `ModelRegistry::Forward`'s `input.multi_kv` guard (`model_registry.cpp`) stopping its refusal |
| **this row** | what RUNS on those caches: the three layer shapes, the compressor's two stages, the `coff` role selection and boundary emission, the indexer's `qr`-sourced query, and the `!is_indexer && !is_comp` refusal at `deepseek_v4.cpp:786-787` |

W5 lands first and deliberately does NOT remove the DSA refusal; it makes a cache
reachable for this row. Neither row is gateable end-to-end alone: W5's
synthetic-config gate proves routing, and the token-exact oracle gate above 512
tokens belongs here.

## Upstream chain

Read at `5559679229`. **The path is `vllm/models/deepseek_v4/`, NOT
`vllm/model_executor/models/`** — a `find` under `model_executor` returns
nothing here and reads as "upstream does not implement it", which is wrong.

| upstream | what it defines |
|---|---|
| `attention.py:345-391` | `DeepseekV4Attention.forward` — GEMMs + RMSNorm, then `attention_impl`, then `_o_proj` |
| `attention.py:454-533` | `attention_impl` — **the composition**, three layer shapes |
| `attention.py:689-856` | `DeepseekV4Indexer`, whose query comes from `qr` |
| `attention.py:721-726`, `:835` | the indexer query projected from `qr` (`q_lora_rank`), not the hidden state |
| `compressor.py:240-248` | `overlap = compress_ratio == 4`; `coff = 1 + overlap` |
| `compressor.py:324-430` | `DeepseekCompressor.forward` — `save_partial_states`, then the boundary-gated fused compress |
| `common/ops/fused_compress_quant_cache.py:~164-183` | the window gather and **the role selection** |

## Design

### D1. Three layer shapes, selected by `compress_ratio`

`attention_impl` has exactly three arms, and every one ends at
`forward_mqa(q, kv, positions, out)` then `_o_proj`:

| shape | `compress_ratio` | count on V4-Flash | what runs beside `wq_b` + kv-insert |
|---|---|---|---|
| indexer + compressor | 4 | 21 | `indexer(...)` and `compressor(...)` |
| compressor only | 128 | 20 | `compressor(...)` |
| SWA-only | 0 | 5 | neither |

Counts are `config.json`'s `compress_ratios` histogram `{0: 5, 4: 21, 128: 20}`
= 46 entries.

**RESOLVED, and it was already answered elsewhere in this row family.**
`dsv4-dsa-geometry.md` read the artifact's own `config.json` and records that the
46 entries are **43 layers + 3 MTP blocks**: layers 0 and 1 are `0`, layers 2..42
alternate `4` and `128`, and the MTP tail is `0`. So `num_hidden_layers == 43`,
2 layers are dense, 21 carry an indexer and 20 a compressor-only -- which is
exactly the "41 of 43 carry a compressor, 21 carry an indexer" the row already
recorded. There is no contradiction: 43 counts LAYERS, 46 counts the config list
INCLUDING the MTP tail, and the trellis shard count matches the layers.

W1 therefore inherits 43 and does not need to reconcile anything. The entry is
kept rather than deleted because #2186 raised it as open and a reader who saw
that deserves to find the answer here, with its source, rather than a silent
deletion.

### D2. The 3-way stream overlap is performance, not correctness

`attention_impl` dispatches through `execute_in_parallel` /
`maybe_execute_in_parallel` with `enable=aux_streams is not None`, and ROCm
(`aux_stream_list is None`) **runs the same work sequentially**. So a sequential
first wave MIRRORS upstream rather than diverging from it, and the overlap is a
later wave with its own measurement.

This is the single biggest scope reduction available here, and it is stated so
that a later reader does not "restore" the overlap believing correctness
depended on it.

### D3. The compressor is two stages, and the second is boundary-gated

Every step: `save_partial_states` writes `kv` and `score + ape[position]` into
the state cache at the token's slot.

Only at a compress boundary: a fused `compress → RMSNorm → RoPE → FP8 quant →
KV-cache write`. The RoPE is exactly specified and load-bearing — GPT-J style,
`is_neox_style=False` (interleaved pairs, **not** split-half), `cos_sin_cache`
laid out `[max_pos, rope_head_dim]` with cos in the first half and sin in the
second, applied to the **last** `rope_head_dim` elements, at position
`(positions // compress_ratio) * compress_ratio`.

### D4. `coff` is a ROLE, chosen by offset within the gathering window

The mechanism the forward's refusal calls "never recoverable from the tensor
alone":

```python
if (position + 1) % COMPRESS_RATIO != 0:   # emit at BOUNDARY tokens only
    return
start  = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
```

The state cache holds **two head-sized rows per token**, and the gather picks
which half to read from the token's offset inside the window gathering it. A
token in the overlap belongs to two windows and has a **different role in
each**.

**Our loader already materializes the `coff` width correctly** (#1970). The
forward refuses because `AttentionBlock` indexes the COLLAPSED geometry. So W1
is a forward change, not a loader change — and the refusal's own text is the
specification of what to build.

### W1 design — the compressor-only shape is COMPOSITION, not new kernels

Traced before estimating, because every earlier estimate on this row family moved
once the tree was read.

**Every primitive W1 needs already exists, on CPU and CUDA.**

| the shape needs | what exists |
|---|---|
| the window pass | `vt::MlaDecodeAttention` with `window_size` (`left == sliding_window - 1`) -- landed by `KV-DSV4-MULTICACHE` W5 |
| the compressed-history pass | the SAME op's SELECTED-SLOT arm, `topk_indices` + `valid_counts` |
| combining the two | `vt::MergeAttnStates` -- an LSE merge with both `+inf` and both-`-inf` edge cases ported |
| the pool | `CompressorPoolNorm` -- per-column softmax over the window, then RMSNorm |
| the APE save | `CompressorSaveScoreApe` |
| the per-head sink | `MlaDecodeAttentionArgs::attn_sink`, landed by W5 |

So W1 composes: save state each step, pool at a boundary into the compressed
cache, then TWO attention passes merged by their LSEs -- rather than the single
fused two-cache kernel upstream calls
(`flash_mla_with_kvcache(k_cache=swa, extra_k_cache=compressed, ...)`). The
composition is mathematically the same; only the kernel fusion differs, and that
is a performance question for a later wave, not a correctness one.

**`compress_ratio == 128` FIRST because `coff == 1` there.** `overlap` is
`compress_ratio == 4`, so the 128 shape has no overlapping windows and no
`head_offset` role selection -- the mechanism W5-4 of `dsv4-dsa-compose.md`
describes. It exercises the state cache, the boundary gate and the two-pass merge
without the hardest part.

#### The `c128a` selection is ARITHMETIC, not a learned top-k

The last unknown in W1's shape, and it resolves in W1's favour. The compressed
pass needs an index list, and the name upstream gives it --
`c128a_global_decode_topk_indices` -- reads like the Lightning Indexer's output.
It is not.

`sparse_mla.py:126-129` calls the field "Pre-computed C128A metadata
(compress_ratio == 128 only). Decode: global slot ids + valid-entry counts
**(fused from positions)**", and `_build_c128a_metadata` asserts
`cm.positions is not None` because positions are its only input. The selection is
therefore arithmetic over the current position -- which compressed windows have
CLOSED -- and carries no learned component at all.

That is what makes `compress_ratio == 128` the right first shape. It needs:

- the compressor cycle (landed: `CompressorStepCycle`),
- a window pass and a compressed pass merged by LSE (proven equivalent above),
- and an index list computable from `positions` alone.

The Lightning Indexer, which DOES learn its selection, belongs only to the
`compress_ratio == 4` layers and therefore to W3. A reader who assumed "topk
implies indexer" would have pulled W3's hardest dependency into W1 for no reason.

#### THE SINK MUST ENTER EXACTLY ONE PASS

The trap, written down before anyone hits it. A sink is one extra logit in the
DENOMINATOR. `MergeAttnStates` combines two states by their LSEs, and each pass's
LSE is `log sum exp(its scores)`. **If both passes seed the denominator with the
sink, the merged denominator counts it TWICE**, and the result is a plausible,
slightly-too-small attention output that no token gate would catch.

This is the same defect family as the split double-count `MlaDecodeAttentionArgs`
already documents: a sink added per split rather than in the final reduction. It
has now appeared twice in this design, which is why it is stated as a rule --
**the sink belongs to exactly one contributor to any merged denominator** --
rather than as a note about one kernel.

The gate must therefore compare a two-pass merged result against a SINGLE-pass
reference over the union of both key sets, with a non-zero sink, at a length
where both passes are non-empty. A gate where either pass is empty cannot see a
double-count.

#### The composition primitive now exists, and the rule is executable

`MergeWindowAndCompressed` in `src/vllm/model_executor/models/deepseek_v4_dsa.cpp`
is that composition: it attends the compressed rows with NO sink, and merges
against the window pass's output and LSE. The window pass carries the sink, so
exactly one contributor seeds the denominator. `PagedCausalMlaAttention` grew an
optional `out_lse` for this, because a merge needs both sides' LSEs.

Two constraints are asserted rather than assumed. Every query sees EVERY
compressed row -- a closed window is history, so no causal bound applies among
them -- and `VT_CHECK(num_tokens == 1 || num_heads == 1)` holds the point where
the two LSE layouts coincide, since `MergeAttnStates` wants `[H, T]` and the
decode op emits `[T, H]`. A general prefill step needs a transpose there and
does not get one yet; it is listed under `## Owed

- **The compressor state's relationship to PREFIX CACHING**, before the runner
  carries it. See the section above: `has_inner_state` is architecture-wide and
  gates prefix caching, while the compressor state is one arm's, and a prefix hit
  would leave that state missing the rows the skipped tokens owed it.`.

The gate this section demanded is `tests/vllm/models/test_deepseek_v4_paged_equiv.cpp`,
"W1: two LSE-merged passes equal one pass over the union". Three mutations prove
it discriminates: seeding the compressed pass with the same sink (the
double-count itself), returning the window output unmerged, and bounding the
compressed rows causally. Each was built before it was read -- a mutation that
fails to compile leaves a stale binary reporting a pass.

## Our baseline

What this tree has TODAY, so a later reader does not re-derive it:

| piece | state |
|---|---|
| the DSA refusal | `src/vllm/model_executor/models/deepseek_v4.cpp:~738` — refuses BY NAME on any layer whose loaded DSA geometry the forward does not index, listing the offending tensors |
| the `coff` width in the loader | ALREADY CORRECT (#1970). The loader materializes each DSA tensor at the width upstream derives for the layer; `AttentionBlock` indexes the COLLAPSED synthetic geometry instead. **This is why the row is a forward change** |
| the one arm that caches | `DeepseekV4ForwardGgufCached` runs `dsa_dense` with indexer and compressor forced OFF on every layer (`deepseek_v4.cpp:677-679`) and refuses otherwise (`:786-787`) |
| the kernel primitives | `KERNEL-ATTN-DSA-SPARSE-INDEX`, `KERNEL-ATTN-DSA-COMPRESSOR` — both `SPIKE`, neither integrated |
| the cache | `KV-DSV4-MULTICACHE` W1+W2 landed: the spec classes exist and all seven groups / 167 entries are published. W3-W5 owed |
| residency | the artifact does not load yet; #2283 owes the measurement |

The loader accepting a geometry the forward refuses is DELIBERATE
(`MODEL-DSV4-EXL3` option C, #1970), so every non-DSA capability of the artifact
stays reachable rather than blocked behind a path none of them use.

## Port map

| upstream | our destination |
|---|---|
| `attention.py:454-533` `attention_impl` | the layer-shape dispatch in `dense_attn::AttnBlock` / `AttentionBlock` (`deepseek_v4.cpp`), replacing the refusal shape by shape |
| `compressor.py:324-380` `save_partial_states` | the per-step state-cache write (kv, and score + APE) |
| `compressor.py:380-430` fused compress | the boundary-gated `compress -> RMSNorm -> RoPE -> FP8 quant -> KV write` |
| `common/ops/fused_compress_quant_cache.py:~164-183` | the window gather and the `head_offset` role selection |
| `attention.py:689-856` `DeepseekV4Indexer` | the indexer call, with its query projected from `qr` |
| `attention.py:721-726`, `:835` | that `qr` sourcing specifically -- NOT the hidden state |

Every arm routes through the existing shared seams (`ModelRegistry::Forward`,
`dense_attn::AttnBlock`); this row adds no parallel path.

## Tests to port

Upstream tests that touch this path at the pin, to port with their parameters,
fixtures and tolerances preserved:

| upstream test | covers |
|---|---|
| `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` | the indexer's slot mapping — the closest upstream test to the role/window mechanism |
| `tests/v1/attention/test_indexer_dcp_localize.py` | indexer localization |
| `tests/v1/core/test_contiguous_kv_packing.py` | the packing the compressed cache depends on |

**These do not by themselves gate the composition**, and saying so here prevents
a later wave from treating a green port of them as sufficient. The composition's
gate is token-exactness against the pinned oracle ABOVE 512 tokens
(see `## Gates`); the ported tests cover the pieces, not the assembly.


## Dependencies

`KV-DSV4-MULTICACHE` (#1925), `ACTIVE`:

| wave | state |
|---|---|
| W1 (#1960, `c1e6f3fb9`) | LANDED — `SlidingWindowMLASpec`, the four `MLAAttentionSpec` fields |
| W2 (#1973, `6b18829bc`) | LANDED — all seven groups / 167 entries published; runner refuses an unallocated published group |
| W3 (#2068, `ca3dcda21`) | LANDED 2026-08-27 — the runner allocates a buffer for EVERY published cache instead of one per hidden layer, and `ModelForwardInput` gained the third channel |
| W4 | proposal, **no owner** — non-uniform `block_size` |
| W5 | proposal, **no owner** — consumption |
| W6-W7 | proposals, **no owner** |

**W1 of this row cannot start before that row's W5.** The composition writes a
separate compressed cache beside a sliding-window raw cache, and it cannot reach
a cache the forward is not handed. This is a hard ordering, not a preference.

**W5, not W3** (#2302). An earlier revision of this spec named W3, which had
already landed when it was written. The wall today is the one the code names
itself, in `ModelRegistry::Forward`
(`src/vllm/model_executor/models/model_registry.cpp`, the `input.multi_kv` guard at the top of `ModelRegistry::Forward`):

> `... and no registered forward consumes a cache set keyed by layer name.
> Refusing rather than discarding an allocated KV topology in silence (row
> KV-DSV4-MULTICACHE W5 owns the consuming forward; #1925, #2068)`

A DeepSeek-V4 engine therefore constructs, publishes and ALLOCATES all 167
buffers today, and refuses at the first forward. **This makes the ordering
harder than the earlier revision claimed, not softer:** W3 had an owner and
landed, while W4 through W7 are proposals with no owner at all. Nothing in this
row can begin until W5 acquires one.

The error is recorded rather than quietly corrected because its cause is
reusable: two stale records agreed with each other and neither was the tree.
#1925's index row predates W3, and `kv-dsv4-multicache.md` `## Now` opened with
"W3 (#2068) is claimed" while its own closing paragraph already said the engine
allocates all 167 buffers and refuses naming W5. AGENTS.md `## History is git`
is explicit -- "Before you conclude anything about past work, read the spec and
run `git log -S`" -- and `git log --oneline --grep '2068'` shows `ca3dcda21`
immediately. It was not run.

## W1's layer step LANDED

`CompressorLayerStep` composes one `compress_ratio == 128` layer's decode step:
the pool-score projection, `CompressorStepCycle` driving the carried state,
appending whatever closed, the window pass carrying the sink and keeping its LSE,
and `MergeWindowAndCompressed` folding in the compressed history with NO sink. It
refuses `compress_ratio == 4` by name, since that is `coff == 2` and W3's.

Nothing calls it yet, so the resolver's refusal stays exactly where it is.

**Three holes were found in its gate by mutation, and two of them looked like
coverage.** A six-token run at ratio 128 closes NO window, so `emitted` is always
empty and dropping the appended rows survived untouched; a 128-token step that
crosses `(127 + 1) % 128 == 0` is the only shape in which that half is
observable. And checking finiteness plus "differs from window-only" could not see
the SINK being dropped from the window pass, because the window-only reference
kept its own sink and the outputs merely differed more. Comparing the
no-rows-closed case bit for bit against the sinked window pass is what pins it.

The four mutations that now run red: the state reset each step, the emitted rows
dropped, `coff == 2` accepted, and the window pass losing its sink.

## The runner cannot simply hold the compressor state, and the reason is prefix caching

The remaining step before the resolver's refusal may narrow is the runner carrying
the compressor state across steps, since the gate supplies it by hand.
`DeepseekV4LoadedModel` persists and could hold it, so the change looks like three
lines. It is not, and the constraint was found before writing them.

**`has_inner_state` gates prefix caching.**
`LoadedEngine::ResolveEnablePrefixCaching` returns
`!is_hybrid && !has_inner_state` (`model_loader.cpp:1179`), so declaring inner
state turns prefix caching OFF by default. This row's registration currently
declares `false`, and flipping it is model-wide: it would disable prefix caching
for the GGUF arm too, which carries no compressor state and would lose the
optimisation for nothing. The flag describes an ARCHITECTURE; the compressor
state belongs to one arm.

**And the interaction is real, not merely a flag.** A prefix-cache hit skips
recomputing tokens whose KV is already cached. The compressor's pooled history is
DERIVED from those tokens, so on a hit the carried state would be missing exactly
the rows the skipped tokens would have contributed, and the layer would attend a
compressed history with holes in it. That is the silent-plausible-output failure
this row exists to avoid, and it would appear only on cache hits.

Three resolutions exist -- the state is REBUILDABLE from a cached prefix, a hit
INVALIDATES it, or the arm declares incompatibility per-arm -- and choosing among
them is scheduler-facing and not this row's. **But choosing is not required to be
safe, because the mismatch is DETECTABLE.** The state knows how many tokens it has
pooled, and a step knows the `kv_base` it resumes at; if they disagree, tokens
were skipped that the state needed.

`CompressorLayerStep` therefore refuses on `seen != kv_base`, naming both numbers.
Refusing is never wrong here, only limiting, so the arm is correct under prefix
caching today and the policy choice above stays open rather than blocking.

Gated both directions, because a guard can fail either way: a state that has seen
one token resuming at `kv_base = 7` refuses, a fresh state at 0 is accepted, and
the consistent continuation at 1 is accepted -- so the guard tracks the state
rather than pinning `kv_base` to zero. Two mutations run red, one disabling the
guard and one making it over-fire.

## W1's arm is REACHED from production

`DeepseekV4ForwardExl3Paged` is the paged non-GGUF entry the arm needed. It binds
the EXL3 tower and leaves `gguf` null, so `dsa_dense` is false and the compressor
predicate is live -- the stateless `DeepseekV4ForwardExl3` beside it already had
that shape and simply carried no pages.

Gated in `test_deepseek_v4_exl3_loader`: a two-layer fixture whose layer 1 is
`compress_ratio == 128`, driven for two single-token steps, must come back with
that layer's carried state holding one row per step while layer 0's stays empty.
Only the composed arm produces that; a dense fallback leaves it empty. Three
mutations run red -- the call site deleted, the guard disabled so a compressor
layer proceeds without state, and the entry dropping the state before it reaches
the backend.

So `CompressorLayerStep` is no longer a function that merely works. What remains
before the resolver's refusal may narrow is the runner carrying this state across
steps, since the gate supplies it by hand.

## The compressor arm is wired, and the GGUF forward CANNOT reach it

`V4Backend::compressor` carries the per-layer state and `AttentionBlock`'s paged
arm routes a `compress_ratio == 128` layer through `CompressorLayerStep` when it
is supplied. `DeepseekV4ForwardGgufPaged` surfaces it as an optional argument, and
null keeps the existing refusal.

**That public entry can never reach it, and two tests written against it failed
before the assumption was caught.** `dsa_dense` is `(be.gguf != nullptr)` and
`is_comp` is `has_compressor(layer) && !dsa_dense`, so on the GGUF arm EVERY layer
is dense regardless of `compress_ratios`. A ratio-128 layer driven through that
forward leaves the carried state empty and the guard silent, because neither is
consulted. It is what the arm IS: the GGUF converter never carried compressor
tensors, and `dsa_dense` says so once for the whole arm.

So the arm is reachable only from a NON-GGUF paged forward, and this tree has no
public one. That is now the concrete blocker for W1's reachability, ahead of the
refusal narrowing, and it is smaller than it sounds: the EXL3 arm already loads
and composes, it simply has no paged entry point yet.

## The refusal narrows LAST, not first

Attempted 2026-08-31 and REVERTED, because the attempt is the natural first move
and it is wrong.

`ResolveDeepseekV4SwaPages` refuses every compressor layer. Narrowing it to refuse
only `compress_ratio == 4` -- W3's `coff == 2` overlapped windows plus the
Lightning Indexer -- looks like progress, since W1's `MergeWindowAndCompressed`
has landed and `compress_ratio == 128` is `coff == 1`. It builds, and
`test_deepseek_v4_paged_equiv` catches it immediately.

The primitive exists; NOTHING CALLS IT. So a narrowed refusal does not enable the
composition, it removes the guard in front of the arm that cannot do it: those 20
layers would attend the RAW PREFIX and emit entirely plausible tokens from the
wrong key set. A loud refusal becomes a silent wrong answer, which is the exact
trade this row exists to prevent.

**The order is: wire the forward, gate it, THEN narrow the refusal.** The refusal
is not the work; it is what makes the missing work visible.

## What the REAL artifact needs, counted

From the artifact's own `config.json`, `compress_ratios` over 46 layers:

| shape | layers | wave |
|---|---|---|
| dense (`0`) | 5 | already handled |
| `compress_ratio == 128` (`coff == 1`) | 20 | W1 |
| `compress_ratio == 4` (`coff == 2` + indexer) | 21 | W3 |

So finishing W1 covers 20 of the 41 compressor layers and does NOT make the real
artifact loadable. The other 21 need W3, including the Lightning Indexer's learned
top-k, which is the hardest mechanism in this row. Any plan that reads
"DSA-COMPOSE finishes" as one more wave is mis-sized by the harder half.

## Work breakdown

| wave | scope | depends on |
|---|---|---|
| W0 | this spec | — |
| W1 | reconcile 43 vs 46; layer-shape dispatch in `AttentionBlock`, SEQUENTIAL, replacing the refusal for the `compress_ratio == 128` (compressor-only) shape first | multicache W5 |
| W2 | the compressor's two stages: `save_partial_states`, then boundary-gated compress/norm/RoPE/quant/store | W1 |
| W3 | the overlapped window and role selection; the `compress_ratio == 4` shape; the indexer's `qr`-sourced query | W2, and both kernel `SPIKE` rows promoted |
| W4 | the stream overlap, as a measured performance wave | W3 |

W1 deliberately takes the **compressor-only** shape first: it is 20 of the 46
layers, needs no indexer and no `coff` overlap (`coff == 1` there), and so
exercises the dispatch and the cache path without the hardest mechanism.

## Gates

Every wave: `scripts/agent-preflight.sh`, plus the DeepSeek-V4 suites.

The correctness gate is **token-exact against the pinned vLLM on an identical
workload**, per AGENTS.md. It is NOT gateable below 512 tokens: the one arm that
caches today runs `dsa_dense` with the indexer and compressor forced OFF
(`deepseek_v4.cpp:677-679`), which is exact only while `seq_len <= index_topk`
(= 512). **A token gate at or below 512 tokens cannot detect a defect in this
row**, so every gate here must exceed it.

Each wave must show the refusal moving: the layer shapes it implements stop
refusing, and every shape it does not implement **still refuses by name**. A
wave that makes the refusal disappear without implementing the path is the
failure mode this row is most exposed to.

## Risks

- **The gate needs the real artifact.** DeepSeek-V4-Flash must load first
  (#2283), on a leased `dgx:gpu0`. Until then W1 gates on a fixture and the
  token-exactness claim stays owed.
- **`46` vs `43`** — inheriting the wrong count silently mis-shapes the
  dispatch. W1 reconciles it before writing code.
- **The two kernel rows are `SPIKE`.** W3 needs both promoted; a spike is not an
  integrated primitive.

## Owed

- The `43` vs `46` layer-count reconciliation, raised by #2186 and still open.
- Promotion of `KERNEL-ATTN-DSA-SPARSE-INDEX` and `KERNEL-ATTN-DSA-COMPRESSOR`
  out of `SPIKE`.
- The `[T, H]` to `[H, T]` LSE transpose `MergeWindowAndCompressed` refuses, so a
  PREFILL step with more than one token and more than one head can compose. A
  decode step is unaffected: it carries one token.
- The `AttentionBlock` compressor arm itself. The primitive above is reached only
  by its gate until that arm calls it, and the `compress_ratio == 128` refusal
  stays in place until then.

## Stop conditions

- Stop if `KV-DSV4-MULTICACHE` W5 does not land: W1 has no cache to READ. W3
  (the allocation and the forward channel) landed as `ca3dcda21`; W5 is the
  consuming forward, and it has no owner (#2302).
- Stop before claiming any speed number. This row makes the model RUN; a
  throughput comparison against SparkInfer's 44-47 tok/s additionally needs
  `nvfp4_ds_mla` and K5 speculative decoding, neither of which exists here.
