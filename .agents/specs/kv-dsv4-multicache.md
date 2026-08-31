# KV-DSV4-MULTICACHE — give DeepSeek-V4 the cache topology it actually needs, so a decode step stops being a prefill

Issue: [#1925](https://github.com/mudler/vllm.cpp/issues/1925).
Row: `KV-DSV4-MULTICACHE` ([`.agents/engine-matrix.md`](../engine-matrix.md)).
Kind: SCOPING SPIKE. This document scopes work. It implements none of it, and it
changes no product code.

vLLM registers `DeepseekV4ForCausalLM` at the parity pin, so vLLM is the mirror
source for every structural decision here and no secondary oracle is admissible
for the cache topology. Pin: `5559679229bc961848b121ccdeaa8fa5d79bec98`, verified
with `git -C /home/mudler/_git/vllm log -1` on 2026-08-25 and matching the anchor
recorded at `include/vllm/model_executor/models/deepseek_v4.h:13`.

## Now

`ACTIVE` — W1 ([#1960](https://github.com/mudler/vllm.cpp/issues/1960)) landed
as `c1e6f3fb9`: the KV-cache spec hierarchy gained `SlidingWindowMLASpec`, the
four DeepSeek-V4 fields on `MLAAttentionSpec`, both `storage_block_size()`
overrides, both `real_page_size_bytes` special cases and the alignment-padding
helper, and published none of it.

W2 ([#1973](https://github.com/mudler/vllm.cpp/issues/1973)) landed as
`6b18829bc`: `MakeDeepseekV4KVCache` publishes DeepSeek-V4's seven groups / 167
cache entries, `GPUModelRunner::initialize_kv_cache` REFUSES a published group it
does not allocate rather than dropping it in silence, and `spec_equal`'s missing
MLA arms ([#1974](https://github.com/mudler/vllm.cpp/issues/1974)) are fixed in
flow. Nothing consumed the topology, so that refusal was what a DeepSeek-V4
engine hit.

W3 ([#2068](https://github.com/mudler/vllm.cpp/issues/2068)) LANDED as
`ca3dcda21` (2026-08-27). Its design is `### W3 design — the runner carries every
published group, and a third forward channel`. It makes the runner allocate a buffer for every published cache
instead of one per hidden layer, generalizes `full_attn_group_id_` /
`gdn_group_id_` and the three-valued `LayerKvClass`, and adds the third
`ModelForwardInput` channel. **This is the wave that touches every model**, so
its obligation is byte-neutrality for the uniform case and its full gate includes
the SACRED `test_qwen35_paged_engine` regression. **Still nothing reads a
cache** — a DeepSeek-V4 engine now constructs and allocates all 167 buffers and
its first forward refuses, naming W5. W4 through W7 remain proposals with no
owner.

**The opening of this section read "W3 is claimed" until 2026-08-29**, while the
paragraph above already described W3's landed behaviour. A reader who stopped at
the first sentence concluded W3 was the wall and wrote it into another row's spec
([#2302](https://github.com/mudler/vllm.cpp/issues/2302)). The wall is W5, and it
has no owner.

## Scope

In: the DeepSeek-V4 KV-cache topology as `MakeDeepseekV4KVCache` publishes it,
the KV-cache interface changes that topology needs, the runner allocation and
forward-input changes that carry it, and `DeepseekV4Model::Forward` /
`ForwardDevice` consuming `attn_kv` instead of discarding it.

Out: the DSA sparse forward math itself (indexer top-k selection and compressor
pooling already exist as host references — see `## What already exists`), the
EXL3 quantization arm (`MODEL-DSV4-EXL3`), MTP, and any speed lever. This row
makes a decode step a decode step. It does not make it fast.

## Upstream chain

The executing chain on upstream's side, so a later wave knows which files it must
read rather than guess at. Every path is under `/home/mudler/_git/vllm` at the pin.

1. **Declaration.** Each cache is an `AttentionLayerBase` that registers itself in
   `compilation_config.static_forward_context` under its own prefix and answers
   `get_kv_cache_spec(vllm_config)`. Four such classes serve DeepSeek-V4:
   `DeepseekV4Attention` (`vllm/models/deepseek_v4/attention.py:626-645`),
   `DeepseekV4IndexerCache` (`:648-690`), `DeepseekV4SWACache`
   (`vllm/v1/attention/backends/mla/sparse_swa.py:56-104`) and
   `CompressorStateCache` (`vllm/models/deepseek_v4/compressor.py:150-205`).
2. **Sizing.** `MLAAttentionSpec` (`vllm/v1/kv_cache_interface.py:381-415`) and
   `SlidingWindowMLASpec` (`:611-642`) turn those declarations into page bytes,
   via `storage_block_size = block_size // compress_ratio` and
   `_apply_alignment_padding` (`:345-351`).
3. **Per-token format.** The 584-byte `fp8_ds_mla` token is produced by
   `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:238-297`
   (store) and consumed on the read side; the dtype resolution that selects it is
   `_resolve_dsv4_kv_cache_dtype` (`attention.py:89-119`).
4. **Backends.** `DeepseekSparseSWABackend` (`sparse_swa.py:110+`),
   `CompressorBackend` (`compressor.py:58+`), `DeepseekV4IndexerBackend`
   (`vllm/v1/attention/backends/mla/indexer.py`), and the platform sparse-MLA
   subclasses selected per `attention.py:122-131`.

Not yet read to a conclusion, and named so a wave starts there rather than
rediscovering it: how the block table maps one request onto four caches with four
different `block_size` values, which lives in the KV-cache manager and the
per-backend metadata builders rather than in the four files above.

## Our baseline

`DeepseekV4Model::Forward` opens with `(void)attn_meta;` and `(void)attn_kv;`
(`src/vllm/model_executor/models/deepseek_v4.cpp:2886-2887`), and
`ForwardDevice` does the same (`:2959-2960`). Both are reached from
`ModelRegistry::Forward` through the registry hook
(`src/vllm/model_executor/models/deepseek_v4_registry.cpp:99-112`). The paged KV
cache the runner allocated and handed in is therefore discarded on the server
path, and each of those forwards runs `ForwardComposeImpl` over the whole token
list it was given.

The cache the runner hands in is a placeholder by its own admission
(`deepseek_v4_registry.cpp:126-148`):

> STUB (W3): V4's TRUE KV topology is the fp8_ds_mla UE8M0 576B-paged latent
> (attention.py:89) PLUS the DSA indexer/compressor caches — a multi-cache
> geometry not yet representable. [...] Never exercised this pass — the forward
> VT_CHECKs pending.

One incremental arm exists. `DeepseekV4ForwardGgufCached`
(`src/vllm/model_executor/models/deepseek_v4.cpp:2654`, declared
`include/vllm/model_executor/models/deepseek_v4.h:497`) keeps a
`DeepseekV4KvCache` (`deepseek_v4.h:474-489`) and appends each token's latent.
Its only call site in the tree is `examples/deepseek_v4_gen/main.cpp:198`, and
that file reaches it by including the internal header
`vllm/model_executor/models/deepseek_v4.h` (`main.cpp:26`) rather than
`include/vllm.h`. Under AGENTS.md "Nothing lands dead" an example's internals are
not a production entry point, and under "Shared seams" an example is an ABI
client. This capability satisfies neither rule today, and that is a second
finding of this spike rather than a restatement of the first.

The row's own record already names the gap. `.agents/specs/deepseek-v4-flash.md:1491`
lists as residual "(c) the paged-engine integration (the stateless full-recompute
driver is the run vehicle)", and `:1247-1249` says the engine's incremental paged
decode "does not apply" so the run "needs a manual greedy loop". What did not
exist until now is an analysis of what "not yet representable" means, which is
`## Why our KV interface cannot represent it`.

## The geometry, derived from source

### The layer population

`DeepSeek-V4-Flash` has 43 hidden layers and a `compress_ratios` array of length
44 — one entry per layer plus a trailing `0` for the MTP/nextn layer. The real
array is transcribed verbatim in this tree at
`tests/vllm/models/test_deepseek_v4_pro_variant.cpp:68-77`:

```
0, 0, 4, 128, 4, 128, ... 4, 128, 4, 0
```

Upstream guards the same off-by-one with `if layer_id < config.num_hidden_layers`
and then `self.compress_ratio = max(1, config.compress_ratios[layer_id])`
(`vllm/models/deepseek_v4/attention.py:205-212`). Note the `max(1, ...)`: ratio
`0` becomes `1` upstream, and `1` is the value every downstream branch tests
against.

That array partitions the 43 layers into upstream's three decode layer types
(`vllm/v1/attention/backends/mla/sparse_swa.py:33-53`):

| ratio | upstream type | Flash layers | count |
|---|---|---|---|
| 0 → 1 | `_LAYER_TYPE_SWAONLY` | 0, 1 | 2 |
| 4 | `_LAYER_TYPE_C4A` | 2, 4, 6, … 42 | 21 |
| 128 | `_LAYER_TYPE_C128A` | 3, 5, 7, … 41 | 20 |

Our parser reproduces the same partition: `has_compressor(l)` is
`compress_ratio(l) != 0` and `has_indexer(l)` is `compress_ratio(l) == 4`
(`include/vllm/model_executor/models/deepseek_v4.h:126-131`), and
`test_deepseek_v4_pro_variant.cpp:227-240` asserts 41 compressor layers and 21
indexer layers against the counts derived from the real checkpoint index. The
issue's framing of "the 21 layers that have an indexer and the 41 that have a
compressor" is therefore correct and already gated.

### What each layer caches, upstream

This is the part the stub calls "a multi-cache geometry" and never enumerates.
Each cache below is a distinct `AttentionLayerBase` registered under its own
prefix in `compilation_config.static_forward_context`, so the KV-cache manager
sees each as a separate *layer name* — upstream does not need a "second cache on
one layer" concept, it needs one model layer to contribute several names.

**(a) The sliding-window cache — every attention layer, all 43.**
`DeepseekV4Attention.__init__` constructs `self.swa_cache_layer =
DeepseekV4SWACache(...)` unconditionally at `attention.py:315-321`, prefix
`{layer}.swa_cache`. Its spec is a `SlidingWindowMLASpec` with
`block_size=64`, `num_kv_heads=1`, `head_size=head_dim` (512),
`sliding_window=config.sliding_window` (128), `alignment=576` under
`fp8_ds_mla` (`sparse_swa.py:86-101`). The 64 is not free: the comment at
`sparse_swa.py:76-83` says SWA and C4A blocks "share the same physical tensor,
[so] they must use the same page size", and `[256//4, head_dim] = [64, head_dim]`
is what fixes it.

**(b) The compressed MLA latent — the 41 layers with ratio > 1.**
`DeepseekV4Attention.get_kv_cache_spec` returns `None` when `compress_ratio <= 1`
(`attention.py:626-630`) — layers 0 and 1 have no MLA cache at all — and
otherwise an `MLAAttentionSpec` with `num_kv_heads=1`, `head_size=self.head_dim`
(512), `dtype=torch.uint8` under `fp8_ds_mla`, `compress_ratio` carried through,
`alignment=576`, `model_version="deepseek_v4"` (`attention.py:631-645`).

**(c) The indexer key cache — the 21 layers with ratio == 4.**
`self.indexer` is constructed only when `compress_ratio == 4`
(`attention.py:273-292`), and the indexer owns `self.k_cache =
DeepseekV4IndexerCache(...)` at prefix `{layer}.indexer.k_cache`
(`attention.py:761-767`). Its width is byte-derived, not semantic
(`attention.py:750-760`): FP8 gives `head_dim + head_dim//quant_block*4 =
128 + 4 = 132`; MXFP4 gives `head_dim//2 + head_dim//32 = 64 + 4 = 68`. Its spec
is an `MLAAttentionSpec` with `dtype=torch.uint8`, `compress_ratio=4`,
`alignment` 576/512 (`attention.py:669-684`).

**(d) The compressor state caches — 41 + 21 of them.**
Every `DeepseekCompressor` owns a `CompressorStateCache`
(`compressor.py:290-295`) whose `state_dim = 2 * coff * head_dim`, `coff = 1 +
(compress_ratio == 4)`, dtype **f32**, and whose `block_size` is **4** for
ratio 4 and **8** for ratio 128 (`compressor.py:168-183`), with
`sliding_window = coff * compress_ratio`. Its spec is a `SlidingWindowMLASpec`
(`compressor.py:188-200`). There are two populations of them:

- the attention layer's own compressor, built when `compress_ratio > 1`
  (`attention.py:333-343`), `head_dim=512` → `state_dim` 2048 (ratio 4) or
  1024 (ratio 128) — **41 caches**;
- the indexer's compressor, built inside `DeepseekV4Indexer`
  (`attention.py:768-777`) with `head_dim=128` → `state_dim` 512 — **21 caches**.

**Totals for Flash: 43 + 41 + 21 + 41 + 21 = 167 KV-cache entries across 43
layers, in two spec classes, at four different `block_size` values (4, 8, 64,
and `cache_config.block_size`).** That number is the scope of this row in one
figure, and it is derived by counting the construction sites above against the
`compress_ratios` array of `### The layer population` — not measured from a running engine, which
nothing here can do.

### The per-token byte layout

Upstream's page cost for the V4 latent is not the element formula. Both
`MLAAttentionSpec.real_page_size_bytes` (`vllm/v1/kv_cache_interface.py:396-405`)
and `SlidingWindowMLASpec.real_page_size_bytes` (`:627-635`) special-case
`model_version == "deepseek_v4"` and `cache_dtype_str == "fp8_ds_mla"` to
`storage_block_size * 584`, with the comment "448B NoPE + 128B RoPE + 8B fp8
scale = 584B per token".

**We already hold this layout exactly.** `Fp8DsMlaLayout`
(`include/vllm/model_executor/models/deepseek_v4_compressor.h:113-125`) records
`nope_head_dim=448`, `rope_head_dim=64`, `quant_block=64`, `n_nope_blocks=7`,
`token_stride_bytes = 448*1 + 64*2 = 576`, `scale_dim = 8` (7 real + 1 pad).
`576 + 8 = 584`. The encoder and decoder are landed beside it
(`deepseek_v4_compressor.h:134-160`), ported from
`fused_compress_quant_cache.py:238-297` and SGLang's `dequant_k_cache.py`. What
is missing is not the layout. It is a spec that can *publish* it.

### The two multipliers our spec cannot express

`storage_block_size` is `block_size // compress_ratio` on both upstream classes
(`kv_cache_interface.py:394-395` and `:624-625`). A compressed cache stores one
row per `compress_ratio` tokens, so a page holds `block_size/4` or
`block_size/128` rows, not `block_size`. `alignment` then rounds the page up
(`_apply_alignment_padding`, `:345-351`).

Our `KVCacheSpec::storage_block_size()` returns `block_size` and is virtual
(`include/vllm/v1/kv_cache_interface.h:121-122`), and
`AttentionSpec::real_page_size_bytes` is already written in terms of it
(`src/vllm/v1/kv_cache_interface.cpp:72`). So the *hook* exists. What does not
exist is any spec that overrides it, or any field to override it from.

### The attention sink

`self.attn_sink` is an `nn.Parameter` of the padded head count, initialized to
`-inf` (`attention.py:218-222`), i.e. per-head and per-layer, loaded from the
checkpoint. It is a weight, not cache state: it does not grow with context and
nothing writes it at decode. We already carry it as
`DeepseekV4LayerHostWeights::attn_sink` (`deepseek_v4.h:159`). **It is out of
scope for this row**, and the issue's inclusion of it under "what needs caching"
is the one item of its framing this spike does not confirm. Recorded here so the
next reader does not go looking for a sink cache.

## Why our KV interface cannot represent it

"Multi-cache geometry" is the stub's phrase. Here is the analysis. Five distinct
things are missing, and only the first is about spec classes.

**(1) No spec class can express a compressed or aligned MLA page.**
`MLAAttentionSpec` (`include/vllm/v1/kv_cache_interface.h:242-261`) adds **no
fields** over `FullAttentionSpec` — its constructor forces `num_kv_heads=1` and
`head_size_v == head_size`, and only `real_page_size_bytes()` and `kind()`
differ. There is no `compress_ratio`, no `alignment`, no `cache_dtype_str`, no
`model_version`. Upstream's has all four (`kv_cache_interface.py:381-388`).
`SlidingWindowMLASpec` **does not exist at all**: the enumerator
`kSlidingWindowMla` is declared (`kv_cache_interface.h:85-96`) with no struct
behind it, and the port's own deferral list names the class as omitted
(`:46-52`). Since (a) and (d) of `### What each layer caches, upstream` — 105 of the 167 caches — are
`SlidingWindowMLASpec`, this is the largest single hole.

**(2) A group carries one spec, and a layer belongs to one group.**
`KVCacheGroupSpec` holds a single `std::shared_ptr<KVCacheSpec>`
(`kv_cache_interface.h:358-369`). The only multi-tensor spec in the file is
`MambaSpec`, whose `shapes`/`dtypes` vectors (`:325-346`) are the one place a
layer gets two differently-shaped buffers — and the runner hard-requires exactly
two (`src/vllm/v1/worker/gpu/runner.cpp:641-644`, "conv then temporal state").
Naming one layer in two attention-kind groups does not work either, for reason
(3).

**(3) The runner keeps at most ONE attention group and ONE recurrent group, and
silently drops every other kind.** The selection loop
(`src/vllm/v1/worker/gpu/runner.cpp:577-597`) sets `full_attn_group_id_` on the
first non-eagle `kFullAttention`/`kMlaAttention` group and `gdn_group_id_` on a
`kMamba` group. `kSlidingWindow`, `kChunkedLocalAttention` and anything else
match no branch: **no buffer is allocated and no diagnostic is emitted.** Both
ids are plain `int`s (`include/vllm/v1/worker/gpu/runner.h:571-572`). The
per-layer taxonomy is a three-valued enum — `kNone`, `kFullAttention`,
`kRecurrent` (`runner.h:366-370`) — and the allocation loop
(`runner.cpp:882-1005`) has exactly two allocating arms: one `CacheBuffer` per
attention layer (`:986-990`) or one `ssm_buf_` + one `conv_buf_` per recurrent
layer (`:919-929`).

**So the precise answer to the issue's question is: it is all four at once.**
Multiple caches per layer (up to four on a C4A layer), heterogeneous per-layer
shapes, a non-uniform layer set (three different subsets, none of them all 43),
*and* non-uniform `block_size` across caches. The last is the one that is easiest
to miss and hardest to fix: `HybridKVCacheCoordinator` asserts every group's
`block_size` equals the hash block size and calls differing block sizes
explicitly deferred (`src/vllm/v1/core/kv_cache_coordinator.cpp:340-346`), while
upstream's V4 needs 4, 8, 64 and the configured block size in one model. (That
assertion is an `assert`, so it is inert under `NDEBUG` — a release build would
not refuse, it would mis-index.)

**(4) `per_layer_attn_specs` is the wrong shape for this.** The non-upstream seam
at `kv_cache_interface.h:377-398` gives Gemma-4 a per-layer `head_dim`. It is
`vector<shared_ptr<AttentionSpec>>` — **one** entry per layer, typed
`AttentionSpec`, and when non-empty it must have exactly `num_hidden_layers`
entries (`runner.cpp:869-873`). It solves heterogeneous shape and nothing else.
It cannot carry a second cache, and it cannot be sparse by intent.

**(5) The forward has no channel to receive them.** `ModelForwardInput` carries
exactly two cache references, `std::vector<PagedKvCache>& attn_kv` and
`std::vector<GdnStateCache>& gdn_state`
(`include/vllm/model_executor/models/model_registry.h:303-315`). There is no
third. Even a correctly published topology would have nowhere to arrive.

One further refusal sits in the path: `RetypeAttentionSpec` refuses any
`MLAAttentionSpec` under a non-`auto` `--kv-cache-dtype` by name
(`src/vllm/v1/kv_cache_interface.cpp:173-177`), pointing at
`fp8_ds_mla` as the unwired formula. `fp8_ds_mla` is the layout V4 actually
ships, so today the real dtype cannot even be requested.

## What already exists that can be reused

The protocol forbids hand-rolling a parallel path around a shared seam, so this
section is load-bearing: it says which seam each piece routes through.

**Reusable as-is:**

- **The fp8_ds_mla per-token codec.** `Fp8DsMlaLayout` / `Fp8DsMlaEncodeToken` /
  `Fp8DsMlaDecodeToken` (`deepseek_v4_compressor.h:113-160`) already produce the
  584-byte token. Nothing here needs writing, only publishing and calling.
- **The DSA forward primitives.** The compressor pooling
  (`CompressorPoolNorm`, `deepseek_v4_compressor.h:100-106`) and the indexer
  top-k are landed host references with CUDA arms
  (`KERNEL-DSV4-W7-DEVICE`, `.agents/kernel-matrix.md:151`). This row consumes
  them; it does not port them.
- **`storage_block_size()` as a virtual hook** (`kv_cache_interface.h:121-122`),
  already threaded into the page formula
  (`src/vllm/v1/kv_cache_interface.cpp:72`).
- **`page_size_padded`** (`kv_cache_interface.h:131-176`) can carry what
  upstream's `alignment` computes, exactly as upstream stores the padded result
  into the same field (`kv_cache_interface.py:345-351`).
- **The by-name layer membership machinery.** `GroupLayerMask` /
  `LayerIndexOfName` (`runner.cpp:339-379`) can already decode real per-layer
  names into a mask. Today it is called only for the GDN group and the target
  attention group, and every registry except `nemotron_h_registry.cpp:270-273`
  publishes placeholder names (`"fa"`, `"gdn"`, `"mla"`, `"kda"`). DeepSeek-V4
  would be the second model to publish real names, not the first.

**Instructive but NOT a fit:**

- **The GDN/Mamba hybrids** (`nemotron_h_registry.cpp:252-315`,
  `qwen3_5_common.cpp:36-107`, `kimi_linear_registry.cpp:128-163`) are the
  closest existing precedent for "a layer keeps state that is not K/V", and they
  show the two halves of the answer: `MambaSpec` publishes `shapes` + `dtypes`
  with **independent dtypes per tensor**, and the runner allocates that state
  **per sequence slot** rather than per block (`runner.cpp:547-556`,
  `:919-929`), exposed through a second struct `GdnStateCache`
  (`include/vllm/model_executor/models/qwen3_5.h:90-96`) on its own
  `ModelForwardInput` channel.

  The **shape** of that answer transfers and the **mechanism** does not. V4's
  compressor state is paged and block-indexed upstream, not slot-indexed: it has
  a `block_size` (4 or 8), a `sliding_window`, and it deliberately shares a
  physical tensor with the KV blocks (`compressor.py:174-178`). Sizing it per
  request slot would be a parallel path, not a reuse. What *does* transfer is the
  precedent that a model may publish a non-K/V spec and receive it on its own
  forward channel — which is exactly what "Why our KV interface cannot represent it" (5) says is missing.

- **The sliding-window precedent points the wrong way and this matters.** No
  registry in the tree ever constructs a `SlidingWindowSpec`. Gemma-3 says why in
  its own words (`src/vllm/model_executor/models/gemma3_registry.cpp:105-109`):
  the window is "masked at the attention kernel (per-layer window_size), NOT by a
  smaller SlidingWindowSpec cache — a memory-only vLLM optimization not needed
  for correctness". `laguna_registry.cpp:115` repeats it. **That reasoning does
  not carry to DeepSeek-V4.** V4's SWA cache is not a smaller copy of a full
  cache; it is the *only* cache on layers 0 and 1 (item (b) of `### What each layer caches, upstream`), it holds a different
  128-token window of a different quantity from the compressed latent, and its
  64-token page size is fixed by physical tensor sharing with the C4A blocks. A
  reviewer who reads the Gemma-3 comment and generalizes it to V4 gets a wrong
  model, so the difference is written here.

## Port map

| upstream (at the pin) | ours |
|---|---|
| `vllm/v1/kv_cache_interface.py:611-642` `SlidingWindowMLASpec` | NEW struct in `include/vllm/v1/kv_cache_interface.h` (W1) |
| `:381-395` `MLAAttentionSpec` `compress_ratio` / `alignment` / `cache_dtype_str` / `model_version` + `storage_block_size` | fields on `MLAAttentionSpec` (`kv_cache_interface.h:242-261`) (W1) |
| `:396-405`, `:627-635` the `deepseek_v4` + `fp8_ds_mla` 584-byte page formula | `real_page_size_bytes()` overrides in `src/vllm/v1/kv_cache_interface.cpp` (W1) |
| `:345-351` `_apply_alignment_padding` | page-size padding helper writing `page_size_padded` (W1) |
| `attention.py:315-321,626-645,669-684,761-777` + `compressor.py:290-295` the four declaration sites | `MakeDeepseekV4KVCache` (`src/vllm/model_executor/models/deepseek_v4_registry.cpp:126-148`) (W2) |
| `vllm/v1/worker/gpu_model_runner.py` per-layer KV allocation over N groups | `src/vllm/v1/worker/gpu/runner.cpp:577-597,882-1005` (W3) |
| `attention.py:122-131` sparse-MLA forward consuming the caches | `DeepseekV4Model::Forward` / `ForwardDevice` (`deepseek_v4.cpp:2886,2959`) (W5) |
| `fused_compress_quant_cache.py:238-297` store / SGLang `dequant_k_cache.py` read | ALREADY LANDED as `Fp8DsMlaEncodeToken` / `Fp8DsMlaDecodeToken` (`include/vllm/model_executor/models/deepseek_v4_compressor.h:134-160`) — reused, not re-ported |

## Tests to port

Upstream's own tests for this surface, with what each binds:

- `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py` — the indexer
  cache's slot mapping under `compress_ratio`. This is the closest upstream test
  to the defect and should be ported at W2/W5. Preserve its parameters and its
  compress-ratio cases.
- `tests/v1/test_kv_cache_spec_registry.py` — spec construction and page sizing,
  the parameters `KV-SLIDING-WINDOW-SPEC` and `KV-CHUNKED-LOCAL-SPEC` already
  cite; the `SlidingWindowMLASpec` and `MLAAttentionSpec` cases are W1's.
- `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py` — the fused
  q-norm/RoPE/KV-insert kernel, i.e. the write side of the MLA cache. W5.
- `tests/models/test_deepseek_v4_mega_moe.py` — carried for the revision anchor;
  it is a MoE test and binds nothing here.

Adaptations must be documented per AGENTS.md. The expected one: upstream
constructs specs through `VllmConfig`, which is not ported, so the C++ cases
construct the spec directly, exactly as the existing
`tests/vllm/v1/test_kv_cache_manager.cpp` cases do.

## Dependencies

- **`KV-MLA-SPEC`** (`.agents/engine-matrix.md`, `INVENTORIED`) — the
  `MLAAttentionSpec` field set W1 extends. This row consumes it rather than
  forking a second MLA spec.
- **`KERNEL-DSV4-W7-DEVICE`** (`.agents/kernel-matrix.md:151`) — the DSA
  compressor/indexer op families W5 calls. Landed for the host arm.
- **A multi-GB10 `rc` lease** for the `## Gates` G2 token gate. `PENDING`; no
  lease was requested for this spike and none is needed before W5.
- **The NVFP4 or fp8 DeepSeek-V4-Flash checkpoint staged** for G2. Not staged;
  the two artifacts staged today (GGUF IQ2_XXS, EXL3) are the two vLLM cannot
  read — see `## Gates`.

## Work breakdown

Waves are proposals. Each is one unit of work with its own issue, red-first
test and fresh review. Sizes are relative effort (S/M/L), not estimates in time,
and no wave has an owner.

| wave | what | gate runs on | size |
|---|---|---|---|
| **W1** | `SlidingWindowMLASpec` + the four `MLAAttentionSpec` fields (`compress_ratio`, `alignment`, `cache_dtype_str`, `model_version`), `storage_block_size()` overrides, the two `real_page_size_bytes` special cases, `_apply_alignment_padding`. Pure allocation metadata — no model, no runner. | **CPU** | S |
| **W2** | `MakeDeepseekV4KVCache` publishes the real topology: one group per (spec class × compress ratio), real per-layer names, the 167 entries enumerated under `## The geometry, derived from source`. Nothing consumes it yet; the gate is the published spec set. | **CPU** | M |
| **W3** | The runner carries more than one attention group and more than one cache per layer: generalize `full_attn_group_id_`/`gdn_group_id_` (`runner.h:571-572`) and the three-valued `LayerKvClass` (`runner.h:366-370`), and add the third forward channel that `## Why our KV interface cannot represent it` item (5) says is absent. **This is the wave that touches every model**, so its obligation is byte-neutrality for the uniform case, on the model of the `per_layer_attn_specs` contract (`kv_cache_interface.h:384-393`). | **CPU** | L |
| **W4** | Non-uniform `block_size` across groups: `HybridKVCacheCoordinator`'s deferral (`kv_cache_coordinator.cpp:340-346`) and the block-table geometry (`runner.cpp:311-319`). May land inside W3 if W3's design needs it; kept separate because it is where a wrong answer is silent under `NDEBUG`. | **CPU** | M |
| **W5** | `Forward`/`ForwardDevice` CONSUME `attn_kv`: the published caches reach the model keyed by layer name and each layer routes to its own, replacing `(void)attn_kv` and the `ModelRegistry::Forward` refusal (its `input.multi_kv` guard, now the free function `MultiKvRefusalApplies` in `model_registry.cpp`; cited as `:430-440` until [#2353](https://github.com/mudler/vllm.cpp/issues/2353), which measured that those lines are `MultiKvCacheIndex::Find` and replaced the range with the symbol). **The cache plumbing only** -- see the scope boundary below, and `### W5 design` for the design ([#2323](https://github.com/mudler/vllm.cpp/issues/2323)). | CPU at synthetic config; **GPU** for the real geometry | L |
| **W6** | Reachability + ABI: the capability reachable from `ModelRegistry::Forward` and exposed through `include/vllm.h`, so `examples/deepseek_v4_gen` stops including an internal header (`## Our baseline`). | **CPU** | S |
| **W7** | The oracle gate of `## Gates`. | **GPU**, ≥2 GB10 | M |

**W5's scope was NARROWED on 2026-08-29, and the boundary is written here
because two rows would otherwise claim the same code.** Its one-line entry above
originally read "the DSA-sparse attention path reading the published caches" and
"Removes the `VT_CHECK(!is_indexer && !is_comp, ...)`", which is the DSA
ALGORITHM rather than the cache plumbing. `MODEL-DSV4-DSA-COMPOSE`
([#2286](https://github.com/mudler/vllm.cpp/issues/2286)) was then created against
the forward's own "no owning row" refusal and specced that algorithm in detail --
overlapping this wave, because W5 has never had a design section and its scope
lived in one table cell.

The split, so each row owns code the other does not:

| row | owns |
|---|---|
| `KV-DSV4-MULTICACHE` W5 | the caches REACHING the model and each layer routing to its own: `attn_kv` consumed rather than `(void)`-ed, and `ModelRegistry::Forward` stopping its refusal |
| `MODEL-DSV4-DSA-COMPOSE` | what RUNS on those caches: the three layer shapes, the compressor's two stages, the `coff` role selection and boundary emission, the indexer's `qr`-sourced query, and the `!is_indexer && !is_comp` refusal at `deepseek_v4.cpp:786-787` |

W5 therefore lands first and does NOT remove the DSA refusal; it makes a cache
reachable for the row that will. Neither row can be gated end-to-end alone, and
that is stated rather than discovered: W5's synthetic-config gate proves routing,
and the token-exact oracle gate belongs to `MODEL-DSV4-DSA-COMPOSE` above 512
tokens.

**A SECOND row consumes this W5, and it is recorded here so the count of
consumers is not rediscovered as a surprise.**
`MODEL-TEXT-GLM-MOE-DSA` (`.agents/specs/glm-dsa-latest-deepseek.md` §3.7)
also has a wave numbered W5, "the indexer KV side cache", and its own text
makes it conditional: *"This is `KV-DSV4-MULTICACHE`'s work and this row
consumes it; if that row does not schedule it, this row's W5 is where it lands
and the ownership is recorded in both places before a line is written."*

**That condition resolved FALSE on 2026-08-30, so W5 is STRUCK from the GLM row
and re-sequenced there as a consumption site — struck, not deferred, because the
wave does not belong to that row at any date.** This row scheduled W5: it has a `### W5 design` section (W5-1 through
W5-6) tracked by [#2323](https://github.com/mudler/vllm.cpp/issues/2323), and
its first implementation commit — W5-2's gated dispatch — is already written.
GLM-5.3 is therefore a CONSUMER of the same by-name multi-KV channel that
DeepSeek-V4 consumes, on the shape `MODEL-MM-GLM53-FLASH` already proved
(`glm5_next_kv.{h,cpp}`, `ModelFactory::consumes_multi_kv_cache`), and it
acquires no independent cache-plumbing scope.

**The concrete thing that would have gone wrong** is worth keeping, because it
is the failure this boundary exists to prevent. The GLM row's W5 lists as a
test *"the refusal at `dots3_note_device.cpp` is deleted"* — and W5-2 above
says, in this row's own words, that **deleting the refusal is the one thing W5
must not do**. Two rows working the same wave from two specs would have had one
of them delete the guard the other built. A third point: that refusal is not
GLM's to delete under any owner, because it guards `Dots3NoteForCausalLM` and
belongs to `MODEL-DOTS3-NOTE` ([#699](https://github.com/mudler/vllm.cpp/issues/699)),
whose own comment already says the cache "is `KV-DSV4-MULTICACHE` (#1925), not
this row".

W1, W2, W4 and W6 are fully CPU-gateable. W3 is CPU-gateable for its own
guarantees but its byte-neutrality obligation reaches every model, so its full
gate includes the SACRED `test_qwen35_paged_engine` regression. W5 and W7 need
the GPU.

### W5 design - the forward consumes the caches, and the refusal becomes a dispatch ([#2323](https://github.com/mudler/vllm.cpp/issues/2323))

W5 had no design section until now, and that absence had a cost worth recording:
its whole scope lived in one wave-table cell, so `MODEL-DSV4-DSA-COMPOSE`
([#2286](https://github.com/mudler/vllm.cpp/issues/2286)) was specced over it and
had to be split back apart (#2302). The boundary above is binding on this
section: **W5 is the plumbing, and it deliberately does not remove the DSA
refusal.**

#### W5-1. What the tree does today

W3 (`ca3dcda21`) made the runner allocate a buffer for EVERY published cache and
hand them to the forward keyed by the name each was published under. A
DeepSeek-V4 engine therefore constructs, publishes and ALLOCATES all 167 caches
today, and then refuses at the first forward
(`src/vllm/model_executor/models/model_registry.cpp:430-440`):

> `... and no registered forward consumes a cache set keyed by layer name.
> Refusing rather than discarding an allocated KV topology in silence (row
> KV-DSV4-MULTICACHE W5 owns the consuming forward; #1925, #2068)`

That refusal fires **unconditionally on `input.multi_kv != nullptr`**, and it is
doing its job: `DeepseekV4Model::Forward` and `::ForwardDevice` still open with
`(void)attn_meta; (void)attn_kv;` (`deepseek_v4.cpp:3033-3034`, `:3105-3106`) and
recompute the whole prefix per token, so letting the step run would report a
decode rate for a full-recompute path.

#### W5-2. The refusal becomes a DISPATCH, never a deletion

**Deleting the refusal is the one thing W5 must not do.** It would restore
exactly the silent-discard failure W3 built it to prevent, and restore it for
every FUTURE model that publishes a topology it cannot consume - a
wrong-answer-not-a-crash, invisible to a token gate because the tokens stay right
while the decode recomputes.

So the forward declares the capability, and `ModelFactory` already has the
pattern with its rationale written down (`model_registry.h`, beside
`supports_weight_offload`):

> Declaring the capability makes that case LOUD instead: the engine refuses a
> configured offload against a model that does not claim support, naming the
> architecture. A new model inherits false and is refused until someone wires it.

W5 adds one more flag in that shape, defaulting **false**:

```cpp
// KV-DSV4-MULTICACHE W5: whether THIS model's forward consumes a cache set
// keyed by layer name (`ModelForwardInput::multi_kv`) rather than the
// positional `attn_kv` convention.
bool consumes_multi_kv = false;
```

`ModelRegistry::Forward` then refuses only when a name-keyed set arrived AND the
registration does not claim it. Every other model keeps its exact behaviour by
construction, because `multi_kv` is `nullptr` for every topology the positional
convention expresses.

#### W5-3. How a layer finds its caches

By name, which is what W3's channel exists for and what upstream does - every
cache upstream is registered under its own prefix and `get_kv_cache_spec()`
returns a `dict[str, KVCacheSpec]` keyed by it. `MultiKvCacheIndex::Find(name)`
returns the index into `attn_kv`, or -1.

The forward therefore resolves each layer's caches by the same names
`MakeDeepseekV4KVCache` published, and **-1 is a refusal, not a fallback**: a
name that does not resolve means the published topology and the consuming
forward disagree, and continuing would silently drop a cache. That is the same
polarity as W2's refusal and W3's, and it is the third time this row has needed
it.

`Find` is linear over 167 entries and a forward looks a name up once per layer
per role; W3 recorded that as a deliberate decision rather than an oversight, and
W5 does not change it. If profiling later shows it, an index is a follow-up with
a measurement behind it.

#### W5-4. What W5 does NOT reach, and why the gate is synthetic

DeepSeek-V4-Flash has 21 `compress_ratio == 4` layers, and their DSA algorithm
belongs to `MODEL-DSV4-DSA-COMPOSE`. **W5 leaves
`VT_CHECK(!is_indexer && !is_comp, ...)` (`deepseek_v4.cpp:786-787`) exactly
where it is.** So the real artifact still does not run after W5, and saying so
here prevents the next reader from expecting it.

What W5 makes true is that a DeepSeek-V4 config WITHOUT DSA layers decodes from
the published caches instead of recomputing the prefix - which is the whole
point of the topology and is gateable on CPU at a synthetic config, as the wave
table says.

`DeepseekV4ForwardGgufCached` (`deepseek_v4.cpp:2804`) is the precedent to build
on: it already caches, with indexer and compressor forced OFF on every layer
(`:677-679`).

#### W5-5. The bridge: the two cache representations, and the shortcut that must not be taken

The wave's substance is that two representations have to meet, and naming them
precisely is what stops the wrong one being chosen:

| side | shape |
|---|---|
| runner (`PagedKvCache`, `qwen3_5.h:78-92`) | PAGED -- `{data, dtype, num_blocks, block_size, num_kv_heads, head_size, fp8_kind}`, addressed through a block table, `bf16` or `fp8` |
| model (`DeepseekV4KvCache`, `deepseek_v4.h:513-528`) | CONTIGUOUS -- `deck[layer]` is a flat `[len * head_dim]` of **f32** that GROWS, plus `len` |

Three ways to join them, and only one of them is real.

**(a) Attend over the paged cache directly.** What upstream does, and what the
primitives here already support: `vt::ReshapeAndCache` (`ops.h:4628`) writes a
step's K/V into pages, `vt::PagedAttention` (`ops.h:4975`) reads `[0, ctx)` back
out, and `dense_attn::AttnBlock` already drives exactly that pair
(`dense_attn_block.h:662`). This is the wave.

**(b) Copy paged -> contiguous each step, then run the existing forward.**
**This is the trap, and it must be named rather than left to be discovered.** It
is by far the easiest thing to write, it produces IDENTICAL TOKENS, and it makes
every token gate green -- while the decode stays O(context) per token, because the
copy is exactly the work the paged cache exists to avoid. The engine would then
report a decode rate for a path that is asymptotically no better than the
full-recompute one it replaced. That is the same shape as
`## Why our KV interface cannot represent it`: a wrong-answer-not-a-crash, only
here the wrong answer is a NUMBER rather than a token. It is also why W5-8's
gate asserts the saving by COUNTING WORK rather than by timing -- a timing gate on
a small synthetic config would not separate (a) from (b), and a token gate cannot
separate them at all.

**(c) Alias the paged storage from the deck.** Not available: pages are not
contiguous, which is the entire point of paging.

**CORRECTION 2026-08-30.** An earlier revision of this section said (a) is "not a
wiring change... a PORT of DeepSeek-V4's MLA attention onto the paged seam", on
the evidence that `deepseek_v4.cpp` contains no `ReshapeAndCache` or
`PagedAttention`. That evidence was real and the conclusion drawn from it was
wrong: it searched for the DENSE paged primitives, and MLA does not use them.

**A paged MLA seam already exists, is implemented on CPU and CUDA, and has two
users.**

| piece | where |
|---|---|
| `ForwardMlaAttentionBlock(Dev, MlaBlockDims, MlaBlockWeights, hidden, positions, Tensor& kv_cache, const Tensor& slot_mapping, MlaBlockMetadata, TritonMLAImpl&, out)` | `src/vllm/model_executor/layers/attention/mla_attention.cpp:353` |
| `vt::ConcatAndCacheMla` -- the paged MLA WRITE | `cpu/cpu_cache.cpp`, `cuda/cuda_cache.cu`, called at `mla_attention.cpp:754` |
| `vt::GatherMlaCache` -- the paged MLA read | `cpu/cpu_mla_prefill.cpp`, `cuda/cuda_mla_prefill.cu` |
| existing users | Kimi-Linear (`kimi_linear*.cpp`), dots3-note (`dots3_note_attn.h`, `dots3_note_device.cpp`) |

So W5 is **routing DeepSeek-V4 onto an existing shared seam**, not building one.
That is also what AGENTS.md `## Shared seams` REQUIRES rather than merely
permits -- "Never write a parallel path by hand" -- and V4's bespoke MLA is
already such a path.

**WHAT THE SEAM CANNOT REPRESENT TODAY, and it is exactly one thing.**
`include/vllm/model_executor/models/mla_attention.h` contains **zero**
occurrences of `sink`, and DeepSeek-V4 carries a per-head attention sink loaded
from the checkpoint (`attention.py:218-222`; ours at `deepseek_v4.h`,
`attn_sink[n_heads]`). AGENTS.md's rule for that case is explicit: "Extend a
shared seam when it cannot represent the upstream behavior. Otherwise, record
one exact tracked exception."

So the shape of W5 is:

1. extend `MlaBlockWeights` / `ForwardMlaAttentionBlock` with the per-head sink,
   inert by default so Kimi-Linear and dots3-note stay byte-identical;
2. route DeepSeek-V4's NON-DSA layers through it;
3. leave the DSA layers refusing -- the indexer and compressor belong to
   `MODEL-DSV4-DSA-COMPOSE` (#2286).

**THE EXACT CHAIN THE SINK HAS TO TRAVEL**, traced so the next session does not
re-derive it:

```
ForwardMlaAttentionBlock            layers/attention/mla_attention.cpp:353
  -> TritonMLAImpl::forward_mqa     v1/attention/backend.cpp:290
    -> vt::MlaDecodeAttention       ops.h (MlaDecodeAttentionArgs, ops.h:1777)
```

`MlaDecodeAttentionArgs` carries `scale`, `num_kv_splits` and the block-table
metadata, and **no sink**; `include/vt/ops.h` has zero `sink` occurrences
anywhere. So the extension is four edits, in this order:

1. a sink field on `MlaDecodeAttentionArgs`, defaulting to an absent/`-inf`
   sentinel so every existing caller is bit-identical;
2. the CPU and CUDA `MlaDecodeAttention` kernels honouring it -- the sink is one
   extra logit **in the denominator only**, exactly as
   `deepseek_v4_dsa.cpp:121-139 SoftmaxWithSink` already does it on the host, so
   the reference semantics are already written and gated in this tree;
3. the same for the prefill half;
4. a `attn_sink` tensor on `MlaBlockWeights`, threaded through
   `ForwardMlaAttentionBlock` -- null for Kimi-Linear and dots3-note, which must
   stay byte-identical.

**The online-softmax detail that makes (2) non-trivial.** `MlaDecodeAttention`
splits the KV over `num_kv_splits` and reduces; a sink added per split would be
counted once per split rather than once per row. It belongs in the FINAL
reduction, and the gate has to cover `num_kv_splits > 1` or it will not see the
difference -- `num_kv_splits = 1` is the batch-invariant path and would pass
either way.


**The correction does not move the trap.** Option (b) -- copy paged to contiguous
each step -- remains available, still produces identical tokens, still passes a
token gate, and still leaves the decode O(context) per token. It is arguably MORE
tempting now that (a) is smaller, because "just adapt the existing forward" looks
close. The work-counting gate below is what separates them.


#### W5-6. The SECOND thing the seam cannot represent: DeepSeek-V4's output projection

The sink was the first. Tracing the routing further found the second, and it is
structural rather than a missing field.

`ForwardMlaAttentionBlock` applies the output projection ITSELF, as step 6, and
requires a dense one: `RequireWeight(w.o_proj, "o_proj")` then
`vt::MatmulBT(d.q, out, attn_flat, w.o_proj)`, with
`o_proj [hidden, num_heads*v_head_dim]` (`mla_attention.cpp:958-960`,
`mla_attention.h:338`).

DeepSeek-V4 does not have one. Its output side is a **grouped LoRA**:
`wo_a [n_groups, o_lora_rank, in_per_group]` applied as a block-diagonal bmm,
then `wo_b [H, n_groups*o_lora_rank]` (`deepseek_v4.h:183-191`). That is a
low-rank, per-group factorization; expanding it into a dense
`[hidden, num_heads*v_head_dim]` would reproduce the arithmetic while discarding
the entire saving the factorization exists for.

**UPSTREAM ALREADY SPLITS THESE, and that is the answer rather than a workaround.**
`DeepseekV4Attention.forward` (`attention.py:345-391`) runs
`self.attention_impl(...)` to fill `o_padded`, slices it, and only then calls
`return self._o_proj(o, positions)` -- a SEPARATE, platform-specific method. The
attention block and the output projection are distinct steps upstream, and it is
our block that fused them.

So the routing shape is: expose the attention output BEFORE the output
projection, let each model apply its own. That is additive for the seam's current
users -- Kimi-Linear and dots3-note keep calling the fused entry point and stay
byte-identical -- and it is the mirror of upstream rather than a divergence from
it.

**Order matters here.** This is a bigger change than the sink, it touches the
seam's public shape, and it must land BEFORE any DeepSeek-V4 routing: a V4 layer
cannot reach `ForwardMlaAttentionBlock` at all while step 6 requires a weight the
model does not have.

#### W5-7. The routing target is the OP, not the block -- and that shrinks the wave again

Tracing the routing to the point of writing it found that DeepSeek-V4 should not
go through `ForwardMlaAttentionBlock` at all.

**V4's attention is already in post-absorption form.** The block exists to do the
V2-style work: project q and kv, apply RoPE, ABSORB `kv_b_proj` into the query,
attend, then project out. V4 has no `kv_b_proj`, no `kv_lora_rank` and no
separate `v_head_dim` -- `deepseek_v4.h:74` records the whole geometry as
`head_dim = 512 (= 448 v/nope + 64 rope)`, and the forward attends a per-token
`deck` of `[T, 512]` latents shared across heads (`deepseek_v4.cpp:~900-930`).
That is MQA over a latent, which is precisely what MLA decode IS once absorbed.

**And `vt::MlaDecodeAttention` is geometry-general.** Its validator
(`ops.cpp:4039-4053`) takes `query [batch, num_q_heads, head_size]`,
`out [batch, num_q_heads, v_head_dim]` and
`kv_cache [num_blocks, block_size, head_size]` -- "MLA has no K/V and no head
axis" -- with no width hardcoded anywhere. V4's `head_size = 512`,
`v_head_dim = 448` satisfies it as written.

So the routing is:

1. write V4's per-token latent into a PAGED cache each step, instead of appending
   to `DeepseekV4KvCache::deck`;
2. call `vt::MlaDecodeAttention` with the runner's `block_table` and `seq_lens` --
   the op already supports the attention sink (W5 step one);
3. keep V4's own grouped-LoRA output projection, unchanged.

That is materially smaller than routing through the block, and it needs no
further seam extension.

**AN HONEST CONSEQUENCE, recorded rather than buried.** The `o_proj` split landed
in the previous commit is therefore NOT on this row's critical path: V4 bypasses
the block, so it never reaches step 6. The split remains a correct improvement --
it mirrors upstream, where `_o_proj` is a separate platform-specific step, and it
unblocks any future model whose output projection is not a dense matrix -- but it
was built for a route this row will not take. It was written before this check
was made, which is the cost of estimating a step before tracing it to the point
of writing it.

That is the fourth time this row's cost has moved after reading the tree rather
than a record (#2302's wrong dependency; "from scratch" versus an existing seam;
the `o_proj` requirement; and now the block versus the op). Each move made the
work smaller.

##### W5-7a. Step one in detail: the latent write needs no new op and no copy

`vt::ConcatAndCacheMla(q, kv_c, k_pe, kv_cache, slot_mapping)` already expresses
V4's cache write exactly, and the mapping is worth writing down because it is not
obvious from the names:

| the op wants | V4 supplies |
|---|---|
| `kv_c [num_tokens, kv_lora_rank]` | the latent's first **448** columns (nope/v) |
| `k_pe [num_tokens, qk_rope_head_dim]` | the latent's last **64** columns (rope) |
| `kv_cache [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]` | `head_dim == 512` |
| `slot_mapping [num_slots] i64` | from the runner's metadata |

**And it needs no copy.** The op's own contract says indexing is driven by the
tensor STRIDES -- "a strided cache view or a split-projection source view is
handled without a copy" -- so V4's CONTIGUOUS `[T, 512]` deck is passed as two
views over the same buffer: offset 0 width 448, and offset 448 width 64, both
with row stride 512. Nothing is materialized to split nope from rope.

**One constraint to respect rather than discover.** The op takes the "auto" path
only: the cache dtype must equal `kv_c`'s, and the `fp8_ds_mla` and int4 layouts
are REFUSED loudly rather than silently mis-written. V4's published compressed
latent is `fp8_ds_mla` at 584 B/token (`## The geometry`), so step one lands on
the plain arm first and the fp8 layout is a later wave -- `KV-DSV4-MULTICACHE`
already lists it, and this is where the two meet.

So step one is a call, two views and a dtype check, not a kernel.

#### W5-8. Gate

Red-first, on CPU at a synthetic config:

1. A config with no DSA layers publishes its caches, the forward consumes them,
   and a multi-token decode is **token-identical** to the same prompt decoded by
   the full-recompute path. Caching a value must not change it.
2. **The saving is real, not nominal**: the cached decode must not recompute the
   prefix. Asserted by counting work rather than by timing, so it cannot flake.
3. A registration that does NOT claim `consumes_multi_kv` still refuses by name
   when a name-keyed set arrives - mutation-proven by clearing the flag and
   seeing exactly that case go red.
4. An unresolvable layer name refuses rather than proceeding.
5. Every model whose topology the positional convention expresses is
   byte-identical, which the existing suites already assert and which case 3
   protects by construction.

**NOT gateable at or below 512 tokens** for the DSA arms, per `## Gates` - the
one arm that caches today is exact only while `seq_len <= index_topk`. W5's own
gate is a non-DSA config, so the bound does not bind it, and it is named here so
nobody carries a 512-token result into the DSA row.

### W1 design — allocation metadata ([#1960](https://github.com/mudler/vllm.cpp/issues/1960))

Committed before the implementation. Every value below was read from
`/home/mudler/_git/vllm` at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`,
and where this section and `## The geometry, derived from source` disagree,
upstream wins and the disagreement is named.

**What lands.** Two spec classes, in
`include/vllm/v1/kv_cache_interface.h` and `src/vllm/v1/kv_cache_interface.cpp`,
plus the registry entry for the new kind in
`src/vllm/v1/kv_cache_spec_registry.cpp`.

1. **The four fields on `MLAAttentionSpec`**, mirroring
   `vllm/v1/kv_cache_interface.py:381-388` name for name:
   `cache_dtype_str` (`str | None`), `alignment` (`int | None`, "Default to None
   for no padding"), `compress_ratio` (`int`, default `1`, "Default to 1 for no
   compression") and `model_version` (`str | None`). Upstream's own comment calls
   the last three "DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these
   at defaults", which is exactly the byte-neutrality argument below.

2. **`MLAAttentionSpec::storage_block_size()`** = `block_size / compress_ratio`
   (`kv_cache_interface.py:393-395`). The hook is already virtual on the base
   (`include/vllm/v1/kv_cache_interface.h:121-122`) and already threaded into
   the MLA page formula (`src/vllm/v1/kv_cache_interface.cpp:72`), so this wave
   supplies the override and the field it reads, not the seam.

3. **`MLAAttentionSpec::real_page_size_bytes()`**, mirroring
   `kv_cache_interface.py:396-410` **including its branch order**:
   `cache_dtype_str == "fp8_ds_mla"` is tested FIRST, and inside it
   `model_version == "deepseek_v4"` returns `storage_block_size * 584` while
   anything else returns `block_size * 656` (the V3.2 main-MLA layout — note it
   is `block_size`, not `storage_block_size`, upstream). Only then does the
   quantization mode matter.

   **The branch order is load-bearing and is the one thing a careless port gets
   wrong.** `DeepseekV4Attention.get_kv_cache_spec` passes
   `kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype)`
   (`vllm/models/deepseek_v4/attention.py:644`), and `get_kv_quant_mode`
   returns `FP8_PER_TENSOR` for any string starting with `fp8`
   (`kv_cache_interface.py:70-71`) — so **every V4 MLA spec carries a non-NONE
   quant mode**. Our port throws on non-NONE quant modes
   (`src/vllm/v1/kv_cache_interface.cpp:64-71`). Testing the quant mode before
   `cache_dtype_str` would therefore throw on every real V4 spec. Upstream
   never reaches its quant branch for `fp8_ds_mla`; neither may we.

4. **`SlidingWindowMLASpec`**, new, deriving from `SlidingWindowSpec` exactly as
   upstream does (`kv_cache_interface.py:611`), carrying the same four fields
   (`:614-618`), the same `storage_block_size` (`:623-625`) and
   `real_page_size_bytes` (`:627-635`). Its formula is **not** the parent's:
   upstream returns `storage_block_size * num_kv_heads * head_size *
   dtype_size` — `head_size` alone, no `head_size_v`, and `storage_block_size`
   rather than `block_size` — because the cache holds one vector instead of
   K + V (`compressor.py:193` says so in its own comment). Its `kind()` is the
   already-declared `KVCacheSpecKind::kSlidingWindowMla`. Upstream also asserts
   `model_version in (None, "deepseek_v4")` before the element formula
   (`:634-636`); we mirror that as a `VT_CHECK`.

5. **`_apply_alignment_padding`** (`kv_cache_interface.py:345-351`): when
   `alignment` is set, `page_size_padded = round_up(real_page_size_bytes,
   alignment)`, written only when it differs from the actual size.
   `round_up(x, y) = ((x + y - 1) // y) * y` (`vllm/utils/math_utils.py:20-22`).
   `page_size_padded` already exists on `AttentionSpec` and is already what
   `page_size_bytes()` returns when set, so upstream's field is our field.

6. **Registry.** Upstream registers `SlidingWindowMLASpec` against
   `SlidingWindowManager` with `uniform_type_base_spec=SlidingWindowMLASpec`
   — **itself**, not `SlidingWindowSpec`
   (`vllm/v1/core/single_type_kv_cache_manager.py:1834-1838`), unlike
   `MLAAttentionSpec` which registers against `FullAttentionSpec` (`:1824-1827`).
   That difference means a `SlidingWindowMLASpec` group is never uniform with a
   plain `SlidingWindowSpec` group, and it is mirrored here. The
   sliding-window-equality arm of `are_uniform_kv_cache_specs` gains the
   matching branch, mirroring
   `SlidingWindowMLASpec.is_uniform_with_collection` (`:679-686`).

**Three C++ deviations, named rather than hidden.**

- Upstream runs `_apply_alignment_padding` from `__post_init__`, where `self` is
  already the most-derived type. C++ has no equivalent: a virtual called from a
  base constructor dispatches to the base. The helper is therefore called from
  the constructor **body** of each of the two leaf classes, where the dynamic
  type is that class ([class.cdtor]/4), and a comment states that any future
  subclass overriding `real_page_size_bytes` must re-run it. `HiddenStateCacheSpec`
  (`kv_cache_interface.py:452`) is the upstream subclass that would hit this; it
  is not ported.
- Upstream's `SlidingWindowMLASpec.__post_init__` does not call
  `SlidingWindowSpec.__post_init__`, so its `head_size_v` stays `None`. Our
  `SlidingWindowSpec` constructor defaults `head_size_v` to `head_size`. No
  formula on the class reads `head_size_v`, so the two are behaviorally
  identical; recorded so a reader diffing the classes does not treat it as a
  port defect.
- `compress_ratio == 0` is a `ZeroDivisionError` upstream and a
  divide-by-zero here. Upstream's callers cannot produce it — the model applies
  `max(1, config.compress_ratios[layer_id])` (`attention.py:212`) — so the
  refusal is a `VT_CHECK` in the constructor rather than a behavior upstream
  defines. `alignment <= 0` is refused the same way.

**Byte-neutrality, proved rather than asserted.** Seven call sites construct
`MLAAttentionSpec` today: `deepseek_v2_registry.cpp:173`,
`deepseek_v4_registry.cpp:145`, `glm4_moe_lite_registry.cpp:176`,
`kimi_k3_registry.cpp:121`, `kimi_linear_registry.cpp:148`,
`minicpm3_registry.cpp:117` and `dots3_note.cpp:697`. **All seven pass exactly
three positional arguments** (`block_size`, `head_size`, dtype). The four new
constructor parameters are appended after the existing ones with upstream's own
defaults, so every one of those calls is unchanged source and produces
`compress_ratio == 1`, `alignment == nullopt`, `cache_dtype_str == nullopt`,
`model_version == nullopt` — which makes `storage_block_size() == block_size`
and sends `real_page_size_bytes()` down the same element formula it takes today.
The gate for this is a test that pins the computed `page_size_bytes()`,
`storage_block_size()` and `page_size_padded` for the MLA geometries those seven
models use, with the numbers stated as literals, so a change to any of them is a
red test rather than a silently different pool.

**Nothing is published, and here is how that is ensured.** No file outside
`tests/` constructs a `SlidingWindowMLASpec`, and no registry passes any of the
four new arguments — verified by `grep -rn 'SlidingWindowMLASpec' src include
examples` returning only the definition, and by the seven `MLAAttentionSpec`
call sites above being byte-identical source. This matters because
`runner.cpp:577-597` drops a group whose kind matches no branch with **no
diagnostic**, so a `kSlidingWindowMla` group published before W3 would allocate
a subset of the topology and say nothing (`## Risks/decisions`). W1 deliberately
leaves that hole open rather than papering over it: the fix is W3's
generalization, not a W1 warning that would have to be removed again.

**Tests** (`tests/vllm/v1/test_kv_cache_interface.cpp`), each red before the
implementation, with expected values derived from an upstream construction site
rather than from this document's prose:

The configured `cache_config.block_size` is **256** on this geometry, and that
is not a guess: `sparse_swa.py:76-83` says "The C4A KV block shape
`[256//4, head_dim] = [64, head_dim]` determines the SWA block size of 64 tokens
per block", and `compressor.py:174-178` repeats the same `[256//4, head_dim] =
[64, 584]` derivation. Both comments only hold at 256.

| case | upstream source of the expectation | `storage_block_size` | real | padded |
|---|---|---:|---:|---:|
| SWA cache, `SlidingWindowMLASpec` | `sparse_swa.py:86-101`: `block_size=64`, `head_size=512`, `window=128`, uint8, `fp8_ds_mla`, `deepseek_v4`, `alignment=576` | 64 | `64*584 = 37376` | `37440` |
| C4A latent, `MLAAttentionSpec` | `attention.py:631-645`: `block_size=256`, `head_size=512`, ratio 4, uint8, `fp8_ds_mla`, `deepseek_v4`, `alignment=576`, `kv_quant_mode=FP8_PER_TENSOR` | 64 | `64*584 = 37376` | `37440` |
| C128A latent, `MLAAttentionSpec` | same site, ratio 128 | 2 | `2*584 = 1168` | `1728` |
| indexer key cache, `MLAAttentionSpec` | `attention.py:669-684`: `head_size=132` (`head_dim + head_dim//quant_block*4 = 128 + 4`, `:756-759`), ratio 4, uint8, **no** `cache_dtype_str`, **no** `model_version`, `alignment=576` | 64 | `64*1*132*1 = 8448` | `8640` |
| C4 attention-compressor state, `SlidingWindowMLASpec` | `compressor.py:188-200` at ratio 4: `block_size=4`, `head_size=2*coff*512 = 2048`, f32, `sliding_window=coff*4 = 8`, `alignment=576` | 4 | `4*2048*4 = 32768` | `32832` |
| C4 indexer-compressor state | same site, `head_size=2*coff*128 = 512` (`attention.py:768-777` builds it at `head_dim=128`) | 4 | `4*512*4 = 8192` | `8640` |
| C128 compressor state | same site at ratio 128: `block_size=8`, `head_size=2*1*512 = 1024`, `sliding_window=128` | 8 | `8*1024*4 = 32768` | `32832` |
| V3.2 main MLA | `kv_cache_interface.py:402-405` | — | `block_size*656`, **not** `storage_block_size*656` | — |
| quant-mode branch order | `attention.py:644` + `kv_cache_interface.py:70-71` | — | `fp8_ds_mla` + `FP8_PER_TENSOR` returns 584-per-token and does **not** throw | — |
| byte-neutrality | the seven existing `MLAAttentionSpec` call sites | `== block_size` | unchanged, pinned as literals | unset |

**The first two rows are the same number, and that is the port's own
self-check.** Upstream fixes the SWA block size at 64 *because* SWA and C4A
share one physical tensor and must therefore have one page size
(`sparse_swa.py:76-83`). Our two classes reach `37440` by different routes — a
`SlidingWindowMLASpec` at `block_size=64, compress_ratio=1` and an
`MLAAttentionSpec` at `block_size=256, compress_ratio=4` — so the equality holds
only if both `storage_block_size` overrides, both 584-byte branches and the
shared 576-byte padding are all right at once. A test asserts the equality
directly, not just the two values.

**One observation W1 did not act on, named so W2 does not trip over it.**
`spec_equal` (`src/vllm/v1/core/kv_cache_coordinator.cpp:17-67`) switches on
`kind()` and its `default:` arm returns `false`. `kMlaAttention` already falls
into that arm today, and `kSlidingWindowMla` now joins it — so two structurally
identical MLA specs compare unequal there. This is not reachable today: that
helper batches groups that share a spec, and every MLA model in the tree
publishes exactly one MLA group, so it is never called on two of them. W1 leaves
it alone deliberately, because adding a `kMlaAttention` arm would change how
existing models group and this wave's obligation is that nothing about them
moves. **W2 publishes several MLA groups per model and must fix this first**,
and the fix has to state which fields upstream's frozen-dataclass `__eq__`
compares, including the four new ones. Filed as an observation from a read of
the switch, not traced to a failing case.

**W1 evidence.** Measured in `/home/mudler/.cache/sdd/mudler-vllm.cpp/kv-w1`
at the implementation commit, CPU Release build, `ninja -C build`.

| what | result |
|---|---|
| red before | `ninja rc=1` at step `501/504`; first error `'SlidingWindowMLASpec' has not been declared in 'vllm::v1'`, then the four missing `MLAAttentionSpec` members |
| green after | `ninja rc=0` at `150/150`; `test_kv_cache_interface` 43 cases / 225 assertions, 0 failed; the 17 new W1 cases alone carry 138 assertions |
| affected suites | 22 suites, `ctest rc=0`, 100% passed; every one reports a NON-ZERO doctest assertion count (largest `test_single_type_kv_cache_manager` 77643, `test_dots3_note_scaffold` 110818, `test_runner` 544, `test_model_registry` 941) |
| full `ctest` | NOT run. The disk stood at 98% used / 9.2 GiB free and the tree has 593 test targets, so the 22 suites that include `kv_cache_interface.h` or `kv_cache_spec_registry.h` were built and run instead. Stated rather than implied. |
| nothing published | `grep -rn SlidingWindowMLASpec src include examples benchmarks` returns only the class definition, its two method bodies and the registry entry; `grep -rn kSlidingWindowMla src include` outside those files returns nothing; all seven `MLAAttentionSpec` call sites still pass exactly three positional arguments |

Five mutations, each rebuilt from source and each recorded with ninja's rc AND
its step count, because a build that failed would re-run the previous binary and
read as a pass. Every one was restored byte-for-byte (sha256 before == after).

| mutation | ninja | run | verdict |
|---|---|---|---|
| `RoundUp` rounds DOWN (`(x / y) * y`) | rc=0, 3 steps | 6 of 17 cases red, 17 assertions failed | RED |
| `MLAAttentionSpec::storage_block_size` returns `block_size` | rc=0, 149 steps | 6 cases red, 24 assertions failed | RED |
| `kFp8DsMlaV4TokenBytes` 584 -> 576 | rc=0, 149 steps | 6 cases red, 9 assertions failed | RED |
| the quant-mode guard moved AHEAD of the `fp8_ds_mla` branch | rc=0, 3 steps | 5 cases red, 4 assertions failed | RED |
| `kFp8DsMlaV32TokenBytes` 656 -> 576 | rc=0, 3 steps | 1 case red, 3 assertions failed | RED |

The 584 and branch-order mutations report FEWER total assertions than the clean
run (127 and 119 against 138). That is not a weaker gate: a `REQUIRE` and a
throw each abort their case, so the assertions after them never execute. The
`REQUIRE(page_size_padded.has_value())` placement is what makes the 584 mutation
abort rather than silently continue past a padding that stopped happening.

**The `## Owed` item dated to W1 is re-dated by this design, and that is a
finding rather than a scope change.** The refusal at
`src/vllm/v1/kv_cache_interface.cpp:173-177` blocks an `MLAAttentionSpec` under
a non-`auto` `--kv-cache-dtype`. Lifting it needs `ParseCacheDType` to accept
`"fp8_ds_mla"` (`include/vllm/v1/kv_cache_dtype.h:87-90` refuses it by name
today), and accepting it would immediately size MLA pages at 584 bytes per token
for a store path that still writes a bf16 latent — the exact "wrong tokens
rather than a crash" failure that comment was written to prevent. W1 supplies
the page formula; it does not supply the store. The refusal therefore stays and
its message is sharpened to say which half now exists, and the obligation moves
to **W5**, the wave that lands the read and write side.

### W2 design — publishing the topology, and the refusal that makes it safe ([#1973](https://github.com/mudler/vllm.cpp/issues/1973))

Committed before the implementation. Every value below was read from
`/home/mudler/_git/vllm` at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`,
and where this section and `## The geometry, derived from source` disagree,
upstream wins and the disagreement is named.

**The blocking question this wave had to answer first.** W1 declined to publish
because `src/vllm/v1/worker/gpu/runner.cpp:577-597` drops an unrecognised group
with no diagnostic. That reading is CORRECT and the situation is worse than it
records, which is why the answer is not "publish anyway" and not "defer W2 behind
W3" but "publish, and make the drop impossible to take silently".

Traced end to end rather than read off one loop:

1. `MakeDeepseekV4KVCache` is `ModelFactory::make_kv_cache`
   (`deepseek_v4_registry.cpp:120`), reached from a production entry point —
   `LoadedEngine`'s constructor through `MakeKVCacheResolved` /
   `MakeKVCacheMaybeSpec` (`src/vllm/entrypoints/model_loader.cpp:1394-1404`,
   `:1681`) and `ModelRegistry::MakeKVCache`
   (`src/vllm/model_executor/models/model_registry.cpp:380-386`). What it
   publishes reaches `GPUModelRunner::initialize_kv_cache` unfiltered.
2. The selection loop has two arms and no `else`: the FIRST non-eagle
   `kFullAttention`/`kMlaAttention` group becomes `full_attn_group_id_`, a
   `kMamba` group becomes `gdn_group_id_`. A `kSlidingWindowMla` group matches
   nothing. A SECOND `kMlaAttention` group is passed over by the
   `full_attn_group_id_ < 0` guard. Both ids are plain `int`s
   (`runner.h:571-572`).
3. **The allocation loop then does not even see the groups.**
   `membership_by_name` is set only when the model has a recurrent group
   (`runner.cpp:820-845` — `gdn_layer_mask` is computed inside
   `if (has_mamba_group)` and `membership_by_name = gdn_layer_mask.has_value()`).
   DeepSeek-V4 has no Mamba group, so the loop falls into the historical
   predicate `is_full_attn = !is_gdn` and allocates ONE `CacheBuffer` per HIDDEN
   LAYER, every one sized from the target group's `page_size_bytes()`
   (`runner.cpp:895-995`). Publishing the seven groups below without the refusal
   allocates 43 buffers of ONE page for a model that needs 167 buffers of seven
   different pages — and reports nothing.

That is a silently short KV allocation, which is a wrong-tokens failure and not
a crash. `AGENTS.md` requires an unimplemented arm to refuse with a message that
names the missing part, so the refusal is policy, not taste.

**What the refusal costs, stated rather than implied.** DeepSeek-V4 on the
server path stops constructing an engine. Today it loads and produces correct
tokens by full recompute, because `Forward`/`ForwardDevice` discard `attn_kv`
(`deepseek_v4.cpp:2886-2887`, `:2959-2960`) — so what is lost is a path that has
no decode step and whose tok/s `### The speed question, answered` already says is
not a decode rate. `examples/deepseek_v4_gen` does not go through the runner
(`DeepseekV4ForwardGgufCached`, `main.cpp:198`) and is unaffected. Recorded here
because it is a user-visible behaviour change that a reader must be able to find
without reading the diff.

**What lands.**

1. **The runner refuses a published group it does not allocate.** After the
   selection loop, every group is one of exactly four things: the target
   attention group, the recurrent group, the single `fa_draft` draft-KV slot
   (the second `kFullAttention` group, allocated at `runner.cpp:1115-1150`), or
   a defect. The fourth case becomes a `VT_CHECK` naming the group index, its
   spec kind and its first layer name. Byte-neutral for every model shipping
   today: each publishes exactly the groups the runner consumes, which the
   byte-neutrality pins below assert directly rather than infer.

   The draft slot is tolerated on its KIND rather than on `spec_on()`, mirroring
   the allocation loop's own predicate (`runner.cpp:1128-1130` skips any group
   whose kind is not `kFullAttention`). A `kFullAttention` group published with
   speculation OFF is therefore still tolerated and still unallocated; tightening
   that is W3's, and it is listed under `## Owed` rather than left to be found.

2. **`MakeDeepseekV4KVCache` publishes the real topology.** Seven groups, 167
   entries, one group per distinct published spec — which is upstream's own
   grouping rule and, for DeepSeek-V4-Flash, resolves to seven.

   **Which dtype arm, and why it is not the one our cache-dtype resolution can
   reach.** `DeepseekV4Attention.use_fp8_ds_mla_layout` is
   `ClassVar[bool] = True` on the base class (`attention.py:140`) and only the
   FlashInfer-sparse SM120 subclass sets it `False`
   (`nvidia/flashinfer_sparse.py:163`); `_resolve_dsv4_kv_cache_dtype` then
   writes `cache_config.cache_dtype = "fp8_ds_mla"` back onto the cache config
   and returns `torch.uint8` (`attention.py:89-119`). The default DeepSeek-V4
   cache format upstream IS `fp8_ds_mla`, so that is the arm this wave mirrors:
   `cache_dtype_str = "fp8_ds_mla"`, `alignment = 576`, 1-byte storage. Our
   `ParseCacheDType` refuses the string `fp8_ds_mla` by name
   (`include/vllm/v1/kv_cache_dtype.h:87-90`) and our factory signature carries
   no cache dtype at all, so the arm is published unconditionally rather than
   selected — the `alignment = 512` non-`fp8_ds_mla` arm is NOT published, and
   that omission is named under `## Owed` against W5, which owns lifting the
   refusal together with the store path.

3. **`spec_equal` gains the two MLA arms**
   ([#1974](https://github.com/mudler/vllm.cpp/issues/1974)) — the observation
   W1 recorded and correctly declined to act on. Verified: every upstream spec
   class is `@dataclass(frozen=True, kw_only=True)`
   (`kv_cache_interface.py:380-381`, `:610-611`), so `__eq__` is generated over
   all fields and two identical `MLAAttentionSpec`s are equal, while
   `src/vllm/v1/core/kv_cache_coordinator.cpp:17-67` returns `false` for them
   from its `default:` arm. Fixed in flow.

**The seven groups, with the upstream site each value came from.**

| # | group | spec class | upstream site | `block_size` | `head_size` | dtype | window | layers |
|---:|---|---|---|---:|---:|---|---:|---:|
| 1 | C4A compressed latent | `MLAAttentionSpec` | `attention.py:631-645` | `cache_config.block_size` | 512 | u8 | — | 21 |
| 2 | C128A compressed latent | `MLAAttentionSpec` | same site, ratio 128 | `cache_config.block_size` | 512 | u8 | — | 20 |
| 3 | indexer key cache | `MLAAttentionSpec` | `attention.py:669-684` | `cache_config.block_size` | 132 | u8 | — | 21 |
| 4 | SWA cache | `SlidingWindowMLASpec` | `sparse_swa.py:86-101` | 64 | 512 | u8 | 128 | 43 |
| 5 | C4 attention-compressor state | `SlidingWindowMLASpec` | `compressor.py:188-200` | 4 | 2048 | f32 | 8 | 21 |
| 6 | C4 indexer-compressor state | `SlidingWindowMLASpec` | same site at `head_dim=128` (`attention.py:768-777`) | 4 | 512 | f32 | 8 | 21 |
| 7 | C128 compressor state | `SlidingWindowMLASpec` | same site at ratio 128 | 8 | 1024 | f32 | 128 | 20 |

21 + 20 + 21 + 43 + 21 + 21 + 20 = **167**.

**Four upstream details a careless port gets wrong, each read rather than
assumed.**

- **The prefix is `attn`, not `self_attn`.** `DeepseekV4DecoderLayer` builds its
  attention as `prefix=f"{prefix}.attn"`
  (`vllm/models/deepseek_v4/nvidia/model.py:808-813`) under
  `prefix=f"{prefix}.layers"` (`:1015`) and `maybe_prefix(prefix, "model")`
  (`:1409`). Every published name is therefore `model.layers.<N>.attn...`, and
  `LayerIndexOfName` (`runner.cpp:343-359`) resolves each to `<N>`. This is the
  same class of trap as W1's branch order: the `## The geometry, derived from
  source` section of this document says `{layer}.swa_cache` without fixing what
  `{layer}` is, and the answer is not the `self_attn` every other architecture
  in this tree uses.
- **`kv_quant_mode` is passed on two of the four construction sites and not the
  other two.** The SWA cache (`sparse_swa.py:100`) and the compressed latent
  (`attention.py:644`) pass `get_kv_quant_mode(...)`, which returns
  `FP8_PER_TENSOR` for any string starting with `fp8`
  (`kv_cache_interface.py:70-71`). The indexer key cache (`attention.py:669-684`)
  and the compressor state (`compressor.py:188-200`) pass nothing and default to
  `NONE`. Mirrored exactly. This is what makes the published topology exercise
  W1's branch-order guarantee on the real geometry: those two specs carry a
  non-NONE quant mode AND `cache_dtype_str == "fp8_ds_mla"`, and our
  `real_page_size_bytes` must reach the 584-byte branch before the quant-mode
  guard throws.
- **The indexer key cache and the compressor state carry NO `model_version` and
  NO `cache_dtype_str`.** Both take the element formula, not the 584-byte
  branch: `64 * 1 * 132 * 1 = 8448` and `4 * 1 * 2048 * 4 = 32768`. Only
  `alignment` is passed on those two sites.
- **The indexer key width is 132, not 68.** `use_fp4_kv` reads
  `attention_config.use_fp4_indexer_cache`, whose default is `False`
  (`vllm/config/attention.py:64`), so the FP8 branch
  `head_dim + head_dim // quant_block_size * 4 = 128 + 128 // 128 * 4 = 132`
  is the default (`attention.py:738`, `:751-760`). The MXFP4 68-byte arm is not
  published and is named under `## Owed`.

**The published page sizes, as literals, at `block_size = 256`.** The
configured block size is 256 on this geometry for the reason
`sparse_swa.py:76-83` and `compressor.py:174-178` both state: the C4A KV block
shape `[256//4, head_dim] = [64, head_dim]` is what fixes the SWA block size at
64 and the compressor block sizes at 4 and 8. Every number below is the value
W1's formulas produce from the constructor arguments in the table above, and the
test states them as literals so a change to any of them is a red test rather
than a silently different pool.

| # | `storage_block_size` | `real_page_size_bytes` | `page_size_bytes` (padded) |
|---:|---:|---:|---:|
| 1 | 64 | `64*584 = 37376` | 37440 |
| 2 | 2 | `2*584 = 1168` | 1728 |
| 3 | 64 | `64*1*132*1 = 8448` | 8640 |
| 4 | 64 | `64*584 = 37376` | 37440 |
| 5 | 4 | `4*1*2048*4 = 32768` | 32832 |
| 6 | 4 | `4*1*512*4 = 8192` | 8640 |
| 7 | 8 | `8*1*1024*4 = 32768` | 32832 |

Groups 1 and 4 reach 37440 by different routes — an `MLAAttentionSpec` at
`block_size=256, compress_ratio=4` and a `SlidingWindowMLASpec` at
`block_size=64, compress_ratio=1` — and upstream fixes the SWA block size at 64
precisely so that they agree (`sparse_swa.py:76-83`, the shared physical
tensor). The test asserts the equality directly, not only the two values.

**`block_size` is refused rather than mis-sized.** `storage_block_size` is
`block_size / compress_ratio`, so a `block_size` that is not a multiple of 128
gives the C128A group a `storage_block_size` of 0 and a zero-byte page.
Upstream has no such check because its own comments derive the whole geometry at
256; ours refuses by name in `MakeDeepseekV4KVCache`, naming 128 and the two
upstream comments. That refusal is the reason a wrong engine block size is a
loud failure here rather than a zero-sized pool.

**Byte-neutrality, proved rather than asserted.** This wave changes no spec
class and no page formula, so no existing page size can move; what could move is
the runner's behaviour for every model and `spec_equal`'s partition. Both are
pinned directly:

- The tolerated group shapes are enumerated in a test: `{target attention}`,
  `{target attention, recurrent}`, `{target attention, recurrent, fa_draft}`
  all construct, and each of `kSlidingWindowMla`, `kSlidingWindow`,
  `kChunkedLocalAttention` and a second `kMlaAttention` group refuses.
- Every KV-cache factory shipping today publishes exactly the groups the runner
  consumes — asserted as literals over the seven `MLAAttentionSpec` call sites'
  registries plus the hybrid registries, so the refusal cannot fire for them.
- `spec_equal`'s existing arms keep their answers; the two new arms are red-first
  from a pair of identical specs that compares unequal today.

**Nothing published is consumed, and the commit says so.** No `attn_kv` reaches
`DeepseekV4Model::Forward`, no runner allocates any of the seven groups, and the
refusal is what makes that visible rather than silent. Reachability is owed to
W3 and W5 under [#1925](https://github.com/mudler/vllm.cpp/issues/1925), and
listed under `## Owed`.

### The speed question, answered

**No. A server-side tok/s measurement of DeepSeek-V4 is not meaningful before
this lands, and the reason is stronger than "it would be slow".**

Three separate facts, each read from the tree:

1. **On the server path there is no decode step.** `Forward` and `ForwardDevice`
   discard `attn_kv` (`deepseek_v4.cpp:2886-2887`, `:2959-2960`) and call
   `ForwardComposeImpl` over the token list they were handed. Generating token
   *n* recomputes tokens 1..*n*. Per-token cost grows with position, so the
   quantity such a run reports is not a decode rate at all — it is a decreasing
   function of context length that happens to have `tok/s` as its unit. Against a
   fixed autoregressive bar it does not measure a slow implementation; it
   measures a different quantity.

2. **The one arm that does cache is not the architecture.** `V4Backend::gguf !=
   nullptr` sets `dsa_dense`, which forces `is_indexer` and `is_comp` false on
   every layer regardless of `compress_ratios`
   (`deepseek_v4.cpp:677-679`), and the KV path refuses to run otherwise —
   `VT_CHECK(!is_indexer && !is_comp, "kv-cache incremental decode requires
   dense MLA (no indexer/compressor)")` (`:786-787`). The comment at `:664-676`
   states the exactness condition precisely: dense MLA is exact "whenever
   `seq_len <= index_topk` (=512)", because top-k over ≤512 tokens is the full
   causal set. **Above 512 tokens that arm attends over a token set upstream does
   not attend over**, so it is not merely slower or faster than the sparse
   architecture — it is a different computation with a different cost curve. Any
   long-context comparison on it compares two architectures.

3. **The arm that cached is not reachable from the server anyway**
   (`examples/deepseek_v4_gen/main.cpp:198`, `## Our baseline`), so the 13.0 tok/s AR figure the
   EXL3 spec records (`.agents/specs/model-dsv4-exl3.md:159`) was taken on a
   vehicle the server does not have.

What *is* meaningful before this lands: a measurement on the example driver,
explicitly labelled as dense-MLA-not-DSA and bounded to ≤512 tokens of context,
with the bar's own context length stated beside it. That is a narrow claim about
a narrow arm. It is not a server decode rate and must not be recorded as one.
`MODEL-DSV4-EXL3`'s W3b denominator run should state which vehicle it used, since
the two vehicles do not compute the same function.

Deliberately not stated: how much faster a cached path would be. Nothing here
measured anything, no GPU was leased for this spike, and an estimate would be the
"a number quoted often becomes treated as measured" failure.

### W2 gate result — G1's oracle side was NOT run, and is PENDING on a named resource ([#1973](https://github.com/mudler/vllm.cpp/issues/1973))

`## Gates` declares **G1** as the gate available at W2 and calls it "the gate
that would have caught the stub". G1 has two sides: dump upstream's 167-entry
spec set by instantiating the four `get_kv_cache_spec` sites, and dump ours from
`MakeDeepseekV4KVCache`, then compare entry for entry.

**Only our side was executed.** `tests/vllm/models/test_deepseek_v4_scaffold.cpp`
runs `MakeDeepseekV4KVCache` and pins every published entry as a LITERAL — group
count, layer names, `block_size`, `storage_block_size`, `head_size`, dtype,
`sliding_window`, `compress_ratio`, `alignment`, `cache_dtype_str`,
`model_version`, `kv_quant_mode`, `real_page_size_bytes`, `page_size_bytes` —
and its case comment calls itself G1 "on the half that needs no checkpoint, no
GPU and no forward". That framing is worth tightening rather than repeating: the
expected values are READ from upstream construction sites at the pin, and
`RealConfig()` is itself a transcription of the artifact's `config.json`. Both
sides of that comparison are therefore source inspection. It is real evidence
against a stub and against drift, and it is not the oracle execution `AGENTS.md`
asks for when it says to check every change in two ways.

MEASURED on 2026-08-26, on the host this wave was gated on, rather than assumed:

- **`vllm.models.deepseek_v4.attention` does not import here.** `cbor2`,
  `pyzmq`, `msgspec` and `cloudpickle` are absent from the ambient environment
  and were installed into a scratch virtual environment; the import then stops
  at `ImportError: cannot import name 'ALLOWED_LAYER_TYPES' from
  'transformers.configuration_utils'`, because the installed `transformers`
  predates the API the pin uses. Repairing that is a dependency resolution
  against the pin, not a step this wave can take incidentally.
- **No DeepSeek-V4-Flash `config.json` is present.** Neither the host's
  HuggingFace cache nor the NAS mount carries one, so the config G1 is a pure
  function of is not on this box either. `docs/USAGE.md` names the artifact repo
  `0xSero/deepseek-v4-flash-0731-spark` @
  `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`.

So G1's result at W2 is **PENDING on a named resource** — a vLLM environment at
the pin that imports `vllm.models.deepseek_v4`, plus that repo's `config.json`.
It is not satisfied, and it is not waived. Listed under `## Owed`.

### W3 design — the runner carries every published group, and a third forward channel ([#2068](https://github.com/mudler/vllm.cpp/issues/2068))

Committed before the implementation. Upstream paths are under
`/home/mudler/_git/vllm` at the pin `5559679229bc961848b121ccdeaa8fa5d79bec98`.

**What W3 changes, and what it deliberately does not.** It makes
`GPUModelRunner::initialize_kv_cache` allocate a buffer for **every published
cache** rather than one per hidden layer, generalizes the two group ids and the
three-valued `LayerKvClass` that record the routing, and adds the third
`ModelForwardInput` channel that `## Why our KV interface cannot represent it`
item (5) says is absent. It does **not** make any model read a cache (W5), and
it does not touch the block-table geometry or the uniform-`block_size`
assertion in `HybridKVCacheCoordinator` (W4).

**Upstream has no equivalent generalization to port, and that is the reason the
design is ours.** Upstream's runner allocates per registered layer NAME from the
start: `get_kv_cache_spec()` walks every `AttentionLayerBase` in
`compilation_config.static_forward_context` and returns a `dict[str, KVCacheSpec]`
keyed by prefix (`vllm/v1/worker/gpu_model_runner.py:7785-7801`), and one C4A
layer contributes four separate keys —
`{layer}`, `{layer}.swa_cache`, `{layer}.indexer.k_cache`,
`{layer}.compressor.state_cache`. There is no "one cache per layer" assumption
anywhere to relax. Ours indexes buffers by layer POSITION, so what is mirrored
here is the KEY: the third channel carries the upstream
`static_forward_context` name beside each cache, because that is what upstream
addresses a cache by and it is the only key that can distinguish four caches on
one layer.

#### 1. The entry predicate, and why byte-neutrality is by construction

The generalized path is entered **only** when the published topology is one the
old path could not carry. The predicate is exactly the set W2's refusal already
computes (`src/vllm/v1/worker/gpu/runner.cpp:630-690`): after selecting the
target attention group, the recurrent group and the single `fa_draft` slot, is
any published group left over?

- **No leftovers ⇒ the legacy path runs, unchanged.** Every model shipping
  today lands here, so its allocation, its views, its backend selection, its
  `layer_kv_class()` and its `ModelForwardInput` are byte-for-byte what they are
  before this change. This is the `per_layer_attn_specs` contract
  (`include/vllm/v1/kv_cache_interface.h:538-556`) applied to the group set
  instead of the layer list: empty means "nothing new", and nothing new means no
  new code runs.
- **Leftovers ⇒ the multi-cache path runs.** DeepSeek-V4 is the only producer in
  the tree.

**The shipped population, re-derived by sweep at base `c714b0234` rather than
inherited from W2's record:** 36 group-emplacement sites across 31 factory files
— **25 `FullAttentionSpec`, 7 `MLAAttentionSpec`, 3 `MambaSpec`, 1
`SlidingWindowMLASpec`**. By runtime shape: 27 single-group factory FILES,
`kimi_linear` (MLA + Mamba), `nemotron_h` (FA + Mamba), `qwen3_5_common`
(FA + Mamba + a `"fa_draft"` FA behind `if (num_spec > 0)`) and `deepseek_v4`
(up to seven groups from two `emplace_back` sites inside the `add_mla` /
`add_swa_mla` lambdas). The count differs from W2's `34 across 32` because W2
itself added the two DeepSeek-V4 sites. Only `deepseek_v4` has leftovers.

**FILES, not architectures, and the difference is 7.** Several single-group
files back more than one `REGISTER_VLLM_MODEL`: `gemma4`/`gemma4_unified`,
`olmo2`/`olmo3`, `llama_dense`/`internlm3_llama`,
`muse_glimmer`/`muse_glimmer_mm`, the three parakeets, and
`llama_model_embedding` reusing `MakeLlamaForCausalLMKVCache`. Counted by
REGISTERED ARCHITECTURE the population is **42 total = 34 single-group + 7
multi-group + 1 that publishes nothing** (`qwen4_exp`, whose `MakeQwen4ExpKVCache`
throws by name). The 7 multi-group archs are the four `qwen3_5_*` registrations
sharing `qwen3_5_common`, plus `nemotron_h`, `kimi_linear` and `deepseek_v4`.
Both denominators are stated because the byte-neutrality argument is about
FACTORIES (the code that emplaces groups) while the blast radius is about
ARCHITECTURES (what a user can actually load).

**Every reachable `deepseek_v4` config is multi-cache, and seven is the MAXIMUM
rather than the count.** `add_mla` / `add_swa_mla` return early on an empty name
list, so the published group count is a function of the checkpoint's
`compress_ratios`: it ranges over **1..7**. The floor is not a uniform topology
— a config whose every layer has `ratio == 1` publishes ONE group, the SWA
group, and that group is a `SlidingWindowMLASpec`, so `full_attn_group_id_`
never binds to it and it is itself the leftover that turns the entry predicate
on. There is no DeepSeek-V4 config that takes the legacy path.

#### 2. The generalized ids

`attn_group_ids_` (`std::vector<int>`) holds every non-eagle group whose spec is
an `AttentionSpec`, in publication order; `recurrent_group_ids_` holds every
`kMamba` group. `full_attn_group_id_` and `gdn_group_id_` survive as the FIRST
of each — the same value they hold today on every uniform topology, since today
they are already "the first non-eagle attention group" and "a Mamba group". They
are kept rather than replaced because eleven call sites outside
`initialize_kv_cache` read them and none of them means anything different.

#### 3. `LayerKvClass` gains a fourth value, and a per-layer index list

`kMultiCache = 3` means: **this layer's caches are not described by the
positional `attn_kv[fa_idx]` convention** — read `layer_attn_kv_indices()[l]`
instead, which lists this layer's indices into `attn_kv()`. A DeepSeek-V4 C4A
layer has four entries there; layers 0 and 1 have one (the SWA cache alone,
`attention.py:626-630` returning `None` for `compress_ratio <= 1`) and are still
`kMultiCache`, because one cache reached by name is not the same thing as one
cache reached by position. `layer_attn_kv_indices()` is **EMPTY on the uniform
path**, which is the same empty-means-unchanged contract as (1).

#### 4. Allocation over every published group

For each attention group `g` and each of its published layer names, in
publication order: one `CacheBuffer` of `num_blocks * spec->page_size_bytes()`,
and one `PagedKvCache` view carrying **that group's own** `block_size`,
`num_kv_heads`, `head_size` and `dtype`. The per-entry `block_size` is the one
new field on the runner's internal `FaDims`; in the legacy path every entry gets
the single `fa_block_size` it uses today, so the view is byte-identical.

`page_size_bytes()` is the single allocation accessor every `AttentionSpec`
inherits and **none of them overrides**, so the rule here is
`dynamic_cast<const AttentionSpec*>` rather than a kind whitelist. It is defined
once, on `AttentionSpec` itself (`vllm/v1/kv_cache_interface.py:196-201`), and
all eleven `AttentionSpec` subclasses at the pin take it as inherited. The
anchor is stated **with its class** rather than by line alone because the name
is not unique in that file: `page_size_bytes` is also defined on `KVCacheSpec`
(`:109`), `MambaSpec` (`:699`) and `UniformTypeKVCacheSpecs` (`:828`), beside
the confusable `unpadded_page_size_bytes` (`:185`) and `real_page_size_bytes`
(`:204`). The VALUE it returns is emphatically kind-**dependent** — five
subclasses override `real_page_size_bytes`, and `_apply_alignment_padding`
(`:345-351`) is typed `MLAAttentionSpec | SlidingWindowMLASpec`, an MLA-only
hook. A uniform ACCESSOR over a kind-dependent value is precisely what makes
the cast a sound stand-in for a whitelist; "kind-independent", which an earlier
draft of this section and of `runner.cpp` said, would have been an argument
against it.
That is also what makes the path additive for `kSlidingWindow` and
`kChunkedLocalAttention`, which no registry builds today.

A `kMamba` group in a multi-cache topology keeps the existing recurrent
allocation, driven by `gdn_group_id_` and its by-name layer mask; its layers stay
`kRecurrent`. Nothing in the tree publishes that shape, and it is supported
rather than refused because expressing it costs one loop split and refusing it
would be a hole the next hybrid falls into.

`fa_page_size_bytes()` keeps reporting the TARGET group's page, and
`kv_cache_allocated_paged_bytes()` sums every buffer allocated, which on the
multi-cache path is all 167 of them.

#### 5. `is_mla` means "fused single-vector cache", not "kind is kMlaAttention"

The view loop's `is_mla` flag selects the fused 3-dim expected KV shape and the
tolerant MLA backend resolution over the dense NHD 5-dim and the loud dense
resolution. `SlidingWindowMLASpec` holds one vector rather than K + V —
`compressor.py:193` says exactly that beside the constructor, and W1's
`real_page_size_bytes` multiplies `head_size` alone for it — so it belongs on the
fused side. The flag therefore becomes `kMlaAttention || kSlidingWindowMla`.
Unreachable on the legacy path, because a `kSlidingWindowMla` group there is a
leftover and the leftover set is what selects the other path.

#### 6. The third channel

```
struct MultiKvCacheIndex {   // model_registry.h, beside MultiModalForwardInput
  const std::vector<std::string>* layer_names;   // parallel to attn_kv
  const std::vector<int32_t>* group_ids;         // parallel to attn_kv
  const std::vector<int32_t>* layer_indices;     // parallel to attn_kv
  int64_t Find(std::string_view layer_name) const;   // -1 when absent
};
```

and `const MultiKvCacheIndex* multi_kv = nullptr;` on `ModelForwardInput`, set
after aggregate construction exactly as `device_token_ids` is, so no positional
initializer moves. `nullptr` on every uniform model, which is what keeps every
existing forward byte-identical.

`layer_names[i]` is upstream's `static_forward_context` key verbatim — the same
string `MakeDeepseekV4KVCache` publishes, e.g. `model.layers.7.attn.indexer.k_cache`.
`Find` is a linear scan; the list is 167 entries and a forward looks a name up
once per layer per role, so an index structure would be premature. Recorded so
it is a decision rather than an oversight.

#### 7. The channel is READ by production code, and that is what keeps it alive

A channel nobody reads is dead code with a disclosure attached. `ModelRegistry::Forward`
(`src/vllm/model_executor/models/model_registry.cpp:376-379`) — the shared
decode seam AGENTS.md routes every forward through — therefore **refuses** when a
multi-cache index arrives, because no registered forward consumes one yet. The
refusal reads the channel's payload rather than its nullness: it names how many
caches arrived, how many distinct groups they came from, and the first layer
name, so a mutation that empties the channel changes the message.

**What this changes for a DeepSeek-V4 run, stated rather than implied.** Before
W3, `LoadedEngine` construction throws inside `GPUModelRunner::initialize_kv_cache`.
After W3 it constructs and allocates all 167 buffers, and the FIRST forward
refuses naming W5. The engine still cannot serve, and it is one seam further
along, with the allocation now genuinely exercised on a production path rather
than gated as a function. (At the default `--block-size` 32 neither refusal is
what a run reads: `check_ratio_fits(128)` in the factory throws first, as
`### W2 design` records. `--block-size` 128 or 256 reaches this path.)

#### 8. W2's refusal is kept, and here is what still reaches it

It becomes unreachable for DeepSeek-V4 and stays reachable for a topology the
generalized path cannot represent:

- a group whose published layer names do not ALL resolve to distinct in-range
  layer indices (`GroupLayerMask` is all-or-nothing by design);
- a second `kMamba` group;
- a group whose spec is neither an `AttentionSpec` nor a `MambaSpec`.

It is not deleted, and the reason is not caution: those three shapes are
reachable from any future registry and a silently short KV allocation is a
wrong-tokens failure rather than a crash. Its message gains the three cases
above so it names what it now means.

#### 9. Tests, each red before the implementation

In `tests/vllm/v1/worker/test_runner.cpp` (the runner's own production
constructor — `LoadedEngine` builds a `GPUModelRunner` through it and
`initialize_kv_cache` is private) and
`tests/vllm/models/test_deepseek_v4_scaffold.cpp` (through
`reg.factory->make_kv_cache`, the pointer `MakeKVCacheResolved` dereferences).

| case | what it pins |
|---|---|
| a DeepSeek-V4-shaped multi-cache config allocates EVERY published cache | `attn_kv().size()` == the sum of the groups' name counts, not the hidden-layer count |
| each cache's view geometry is its OWN group's | `block_size`, `num_kv_heads`, `head_size`, `dtype` per entry, as literals |
| per-layer routing | `layer_kv_class()` == `kMultiCache` on named layers, `kNone` on unnamed; `layer_attn_kv_indices()[l]` lists exactly that layer's caches |
| the generalized ids | `attn_group_ids()` lists every attention group; `full_attn_group_id()` is still its first |
| byte-neutrality, the four shipped shapes | the four `## the group shapes shipped today still construct` subcases keep `attn_kv().size()`, `layer_kv_class()`, `fa_page_size_bytes()` and `kv_cache_allocated_paged_bytes()` at LITERAL values |
| the legacy path is not entered by the new code | `layer_attn_kv_indices()` EMPTY and `multi_kv` null for every uniform config |
| the third channel arrives populated | names parallel to `attn_kv`, `Find` resolves a published name and returns -1 for an absent one |
| `ModelRegistry::Forward` refuses a multi-cache index | the message names the count, the group count and the first name |
| W2's refusal still fires | an unresolvable-name group, a second `kMamba` group |
| the real topology, end to end | `reg.factory->make_kv_cache(RealConfig(), 256, N)` fed to the runner constructor allocates 167 buffers whose page sizes are the literals W2 pinned |

**Reachability mutation** (`.agents/reachability.md`): repoint the entry
predicate at "always uniform" in a scratch copy and the focused gate goes red,
because the 167 buffers become 43. That is the production call site, not a
`-Werror unused-function` artefact.

**The full gate includes the SACRED `test_qwen35_paged_engine` regression**,
because the byte-neutrality obligation of this wave reaches every model.

**W3 evidence.** Measured in
`/home/mudler/.cache/sdd/mudler-vllm.cpp/kv-w3` on the implementation tree,
CPU Release, `cmake -DVLLM_CPP_CUDA=OFF`, named targets only.

| what | result |
|---|---|
| red before, API | `ninja rc=1` at step `511/513`; `'class vllm::v1::GPUModelRunner' has no member named 'attn_kv_layer_names'`, `... 'multi_kv_index'`, `... 'layer_attn_kv_indices'`, `... 'attn_group_ids'; did you mean 'gdn_group_id'?`, `... 'recurrent_group_ids'`, and `'kMultiCache' is not a member of 'LKC'` |
| red before, BEHAVIOUR (the API landed, the logic did not) | `ninja rc=0` at `128/128`; 5 cases red / 2 assertions failed, four of them the W2 refusal firing: `runner: 6 published KV cache group(s) get NO cache from this runner ... group 3 kind=kSlidingWindowMla layers=43 first='model.layers.0.attn.swa_cache' page_size_bytes=37440` |
| green after | `ninja rc=0` at `3/3`; `test_runner` 27 cases / 784 assertions, 0 failed |
| affected suites | the 49 buildable suites that include `worker/gpu/runner.h` or `models/model_registry.h` plus the KV-cache and speculative suites: `ninja rc=0` at `84/84`, `ctest rc=0`, 48 passed and 1 Skipped. **Four report ZERO doctest assertions** and are checkpoint-gated skips wearing a pass: `test_deepseek_v2_paged_engine`, `test_gemma4_registry_e2e`, `test_qwen3vl_registry_e2e` and `test_qwen35_paged_engine`. The 45 that carry assertions are the evidence, led by `test_dots3_note_scaffold` 110819, `test_single_type_kv_cache_manager` 77643, `test_qwen3_8_text_only` 67855, `test_nemotron_h_scaffold` 38311, `test_muse_glimmer_wiring` 10317, `test_nemotron_h_paged_forward` 3269, `test_model_registry` 958, `test_runner` 784, `test_deepseek_v4_scaffold` 669 |
| SACRED `test_qwen35_paged_engine` | **SKIPPED (exit 77), which is NOT a pass.** Its own message: `qwen3.5-0.8B: models--Qwen--Qwen3.5-0.8B snapshot at the pinned revision 2fc06364 not cached — this gate runs where the ROCm oracle was captured (gfx1100)`. Neither the checkpoint nor the NAS mount is on this host and the gate's own host is a different box, so it is PENDING on a named resource rather than satisfied |
| full `ctest` | NOT run. The tree has 593 test targets and a bare `ninja -C build` links every one of them; the disk stood at 77-78 GiB free and this box has hit ENOSPC on that before. Stated rather than implied |
| byte-neutrality, structurally | `git diff -w` on `runner.cpp` is `241 insertions(+), 18 deletions(-)` against `341/118` without `-w`: the whole legacy allocation loop is unchanged content that moved one indentation level into an `else` |

Twelve mutations, each verified by grep to have LANDED before building, each
recorded with ninja's rc AND its step count because a build that failed would
re-run the previous binary and read as a pass, each restored with sha256 verified
equal and rebuilt. The gate is `test_runner` unless stated.

| mutation | ninja | run | verdict |
|---|---|---|---|
| **REACHABILITY** the entry predicate always answers "uniform" | rc=0, 3 steps | 3 cases red, 22 assertions failed, then SIGSEGV | RED |
| **REACHABILITY** `forward_input.multi_kv = &multi_kv_index_` deleted | rc=0, 3 steps | 1 case red, 4 assertions failed | RED |
| **REACHABILITY** `.make_kv_cache` repointed at a pre-W2 one-group placeholder | rc=0, 4 steps | `test_runner` 1 case / 1 assertion; `test_deepseek_v4_scaffold` 3 cases / 6 assertions | RED |
| every entry is sized from the TARGET group's page | rc=0, 3 steps | 2 cases red, 3 assertions failed | RED |
| the view uses the single `fa_block_size` | rc=0, 3 steps | 1 case red, 7 assertions failed | RED |
| a multi-cache layer is classed `kFullAttention` | rc=0, 3 steps | 3 cases red, 49 assertions failed | RED |
| the group refusal is always true | rc=0, 3 steps | 1 case red, 14 assertions failed | RED |
| the channel is published on the UNIFORM path too | rc=0, 3 steps | 21 cases red | RED |
| the group predicate is a kind whitelist instead of `AttentionSpec` | rc=0, 3 steps | 4 cases red, 15 assertions failed | RED |
| the multi-cache path drops its recurrent group | rc=0, 3 steps | 1 case red, 2 assertions failed | RED |
| every entry is published under the same name | rc=0, 3 steps | 3 cases red, 23 assertions failed | RED |
| `ModelRegistry::Forward`'s refusal is disabled | rc=0, 3 steps | 1 case red, 4 assertions failed | RED |

The first reachability mutation SIGSEGVs rather than failing cleanly, and that is
recorded rather than smoothed: with the predicate off, the ten-cache config
allocates four buffers and the views and the block table no longer describe the
same object. It is still a red — the gate does not go green — and it is the shape
of the failure the predicate prevents.


### W3 repair-round evidence

The fresh review PASSED the code change and returned six findings. Measured on
the REPAIR head, which carries both `origin/main` merges (`94238fc52` and
`f586757d8`, the second bringing #2008's request-scoped draft context and with it
a relevant suite the implementation round did not have,
`test_dflash2_concurrency`). Same host, same CPU Release configuration, named
targets only.

| what | result |
|---|---|
| red before the eagle clause (#2084) | `ninja` reached its link step at `2/2` — the exit code was NOT captured on that one invocation (a `zsh` `PIPESTATUS` slip), and it is recorded that way rather than asserted; the stronger proof that the binary was not stale is that the run REPORTED THE NEW SUBCASE BY NAME. `test_runner` 1 case red / 5 assertions failed, led by `CHECK_THROWS_AS( construct(kv), std::runtime_error ) did NOT throw at all!` — the silent-subset shape arriving as a NON-REFUSAL rather than as a wrong count. The mutation row below is the properly instrumented form of the same red |
| green after | `ninja rc=0` at `3/3`; `test_runner` **27 cases / 791 assertions**, 0 failed. The `+7` against the implementation round's 784 is exactly the new subcase's seven checks; the case count is unchanged because a `SUBCASE` adds no case |
| the eagle clause, mutated back out of the LANDED tree | mutation verified to have landed by sha256 delta (`868e3dc0…` → `4ef5bbb0…`); `ninja rc=0` at `3/3`; **1 case red / 5 assertions failed**; restored, sha256 verified equal to `868e3dc0…`, rebuilt `rc=0` at `3/3`, back to 27/791 |
| affected suites, re-derived on THIS head | 71 targets — the 68 buildable suites including `worker/gpu/runner.h`, `models/model_registry.h` or `v1/kv_cache_interface.h`, plus `test_dflash2_concurrency`, `test_qwen35_paged_engine` and `test_deepseek_v2_paged_engine`. Run TWICE, and the second run is the one that counts: first at the repair tree before the third `origin/main` merge (`ninja rc=0` at `131/131` then `5/5`), then again on the merged head **`223dc1446`** after #2071 landed `dots3_note.cpp` and `test_ops_mla_prefill.cpp` into it (`ninja rc=0` at `64/64`). **Both runs: 70 exit 0, 1 exits 77, the same three zero-assertion suites.** Re-run rather than carried forward, because the finding this round repairs is precisely a gate result quoted against a tree it was not measured on. Disk 90-95 GiB free throughout |
| zero-assertion passes | **THREE**, named rather than counted as evidence: `test_gemma4_registry_e2e`, `test_qwen3vl_registry_e2e`, `test_deepseek_v2_paged_engine`. Fewer than the implementation round's four only because `test_qwen35_paged_engine` is on its own line below |
| SACRED `test_qwen35_paged_engine` | **STILL SKIPPED (exit 77), which is NOT a pass**, and unchanged by this round: `qwen3.5-0.8B: models--Qwen--Qwen3.5-0.8B snapshot at the pinned revision 2fc06364 not cached — this gate runs where the ROCm oracle was captured (gfx1100)`. PENDING on a named resource |
| baselines preserved | `test_deepseek_v4_scaffold` 8/669, `test_model_registry` 24/958, `test_kv_cache_coordinator` 21/142, `test_kv_cache_interface` 43/225, `test_dflash2_concurrency` 2/21 — each the count it held before the repair. `test_runner` is the ONLY suite whose count moved, and the delta is accounted above |
| `scripts/agent-preflight.sh` | rc=0, no FAIL line, both plain and `--staged` |
| `test_dots3_note_scaffold`, `test_ops_mla_prefill` | the two suites #2071 moves, re-run on the merged head: 26 cases / 110819 assertions and 7 cases / 329772 assertions, both `rc=0` |
| full `ctest` | STILL NOT run, for the same reason: the tree has 601 test targets and a bare `ninja -C build` links every one |

**What each finding cost.** Finding 1 (#2084) is the only one that changed
behaviour — one `else if` clause, one subcase, and the "three shapes" comment.
Finding 2 rewrote a comment that claimed a refusal the code does not perform and
added an `## Owed` item, rather than a `VT_CHECK` that no test surface can drive
red. Finding 3 re-anchored `page_size_bytes` from `:337-351` to `:196-201` and
replaced "kind-independent", which the cited region actually argues against;
`compressor.py:194`→`:193` and `:288-293`→`:290-295` were corrected in all five
files that carry them rather than only in W3's own prose, because a repaired
anchor beside four unrepaired copies of the same one is worse than either.
Findings 4 and 5 are record repairs, made while the append-only index row is
still correctable. Finding 6 is #2085, owed with its own line.



## Gates

vLLM implements `DeepseekV4ForCausalLM` at the pin, so vLLM is the primary
oracle and no secondary oracle is admissible for this row. Two gates, at
different costs.

**G1 — the topology gate. CPU, no checkpoint, available at W2.** Upstream's
cache topology is a pure function of the config: `get_kv_cache_spec` on each of
the four construction sites (`attention.py:626-645`, `:669-684`,
`compressor.py:188-200`, `sparse_swa.py:86-101`) reads only `compress_ratios`,
`head_dim`, `sliding_window`, `index_head_dim` and the cache dtype. So the full
167-entry spec set of `### What each layer caches, upstream` — per entry: layer index, prefix, spec class,
`block_size`, `storage_block_size`, `head_size`, dtype, `sliding_window`,
`page_size_bytes` — can be produced on both sides from the same `config.json`
and compared exactly, with no weights, no GPU and no forward. It needs only a
Python script that instantiates upstream's layers on the meta device and dumps
the specs, and `MakeDeepseekV4KVCache` on ours.

This is the gate that would have caught the stub. It is cheap, it is exact, and
it should be red before W2 and green after. It proves the topology and nothing
about the attention math, which is G2's job.

**G2 — the token gate. Blocked on hardware, and the record already says so.**
`.agents/specs/deepseek-v4-flash.md` records the ratified decision under its
`## Risks/decisions` (unique phrase "no vLLM DeepSeek-V4 GGUF plugin"): "with no
vLLM DeepSeek-V4 GGUF plugin, the gate is cross-engine coherence +
self-consistency + the per-layer ds4-oracle diff, NOT vLLM-token-exact". The two
constraints behind it are independent and both still hold:

- **Format.** vLLM loads neither our GGUF (IQ2_XXS, 86.33 GiB) nor the EXL3
  artifact, so the two checkpoints that fit one GB10 are exactly the two vLLM
  cannot read.
- **Memory.** The checkpoint vLLM *can* read is 156.7 GiB NVFP4
  (`include/vllm/model_executor/models/deepseek_v4.h:29-33`, from the real index
  `total_size`), against one GB10's 119 GiB pool. `deepseek-v4-flash.md:875-876`
  calls W8 "multi-Spark-blocked" for this reason.

So a token-exact G2 needs: the NVFP4 or fp8 checkpoint, **at least two GB10s**
under one `rc` lease, greedy decode on both sides at identical prompt/tokens/
sampling, and a context length above 512 — because at or below 512 the dense
fallback is exact (`### The speed question, answered`, point 2) and the gate would pass without exercising the sparse
path this row exists to build. **A gate that runs only at ≤512 tokens cannot
detect the defect it is meant to detect**, and that constraint is the single most
important thing this section records.

**A third, cheaper thing that is not a gate against vLLM but belongs in W5:** the
cached decode must be token-identical to full recompute at the tiny synthetic
config with the indexer and compressor **on** — the equivalence
`DeepseekV4ForwardGgufCached` already claims for the dense arm
(`deepseek_v4.h:491-496`), extended to the arm that VT_CHECKs today. That is the
red-first test for W5 and it runs on CPU.

## Risks/decisions

- **W3 is a shared-seam change that touches every model.** Its byte-neutrality
  obligation is the whole risk, and the `per_layer_attn_specs` contract
  (`kv_cache_interface.h:384-393`) is the precedent for how to state and gate it.
- **The `NDEBUG` hole.** The block-size uniformity check is an `assert`
  (`kv_cache_coordinator.cpp:340-346`), so a release build with non-uniform block
  sizes mis-indexes rather than refusing. W4 must convert this to a `VT_CHECK`
  before, not after, it widens the invariant.
- **The runner's silent drop.** A group whose kind matches no branch at
  `runner.cpp:577-597` is dropped with no diagnostic. Publishing the real
  topology at W2 before W3 lands would therefore allocate a *subset* of it and
  say nothing. W2's gate must assert the published set, not the allocated set,
  and W2 must not be taken as making anything reachable.
- **`kv_cache_tensors` appears to be dead** (`kv_cache_interface.h:374`), which
  matters only if a wave assumes it is the allocation source. Flagged as an
  observation from a grep, not traced to a conclusion.

## Stop conditions

Return `NEEDS_DECISION` rather than widening scope if: W3's generalization
cannot be made byte-neutral for uniform models; or the topology gate G1 shows our
config parse and upstream's disagree about the layer partition (that would be a
`ParseDeepseekV4Params` defect and its own row, not this one). Return
`NEEDS_CONTEXT` if no multi-GB10 lease is available and a wave's gate needs G2.

## Owed

- **SETTLED, AND THE RECORDED EXACTNESS BOUND IS TOO GENEROUS BY 4x** (#2323).
  The previous revision of this entry raised it as an open question; the
  composition has now been traced end to end and the answer is confirmed.

  `forward_mqa` issues ONE kernel call
  (`nvidia/flashmla.py`, `flash_mla_with_kvcache`):

  ```python
  out, _ = flash_mla_with_kvcache(
      k_cache=swa_cache,                                 # PRIMARY: the sliding window
      indices=swa_indices,
      extra_k_cache=kv_cache if not swa_only else None,  # compressed latent, SPARSE layers only
      extra_indices_in_kvcache=topk_indices,
      out=output.unsqueeze(1))
  ```

  with `swa_only = self.compress_ratio <= 1`. So upstream attends the SLIDING
  WINDOW always, and the selected compressed history ONLY on sparse layers. A
  `compress_ratio <= 1` layer therefore attends **128 tokens and nothing else**
  (`sliding_window = 128`, `sparse_swa.py:86-101`).

  Our forward attends the FULL context on every layer (`deepseek_v4.cpp`, the
  dense-causal `sel` arm over `[0, kv_base + t]`), and the paged arm reproduces
  it. So:

  | layers | ours | upstream | diverges above |
  |---|---|---|---|
  | `ratio <= 1` (5 of 46) | full context | 128-token window | **128 tokens** |
  | `ratio == 4 / 128` | full context | window + top-k compressed | already refused (#2286) |

  **This row's recorded bound -- "exact only while `seq_len <= index_topk`
  (=512)" -- is therefore wrong for the SWA-only layers, where the binding
  constraint is 128.** Any DeepSeek-V4 token-exactness claim above 128 tokens is
  measuring a different computation from upstream on those five layers, and that
  includes this row's own W5 gate, which runs far below 128 and so cannot see the
  difference either way.

  What is owed is the sliding window itself on the SWA-only path, and a bound in
  the records that says 128 rather than 512.


- **The last link of W5 -- `Forward` resolving pages from `multi_kv` -- is NOT
  mechanical, and the reason is the topology rather than the plumbing** (#2323).
  `DeepseekV4ForwardGgufPaged` and its paged `AttentionBlock` arm have landed and
  are gated token-for-token against full recompute, but they assume ONE page
  tensor per layer. A real DeepSeek-V4 config does not have that:

  - `MakeDeepseekV4KVCache` publishes the compressed latent under
    `model.layers.{l}.attn`, and **skips it entirely for `compress_ratio == 1`**
    (`deepseek_v4_registry.cpp`: `if (ratio == 1) continue;`, mirroring
    `attention.py:626-630` returning `None`). Those layers carry only a SWA
    cache, so `MultiKvCacheIndex::Find` returns -1 for them by design.
  - The 21 `compress_ratio == 4` layers additionally carry an indexer key cache
    and compressor states, and their algorithm belongs to
    `MODEL-DSV4-DSA-COMPOSE` ([#2286](https://github.com/mudler/vllm.cpp/issues/2286)).

  So the wiring has to express THREE layer shapes against the published names,
  not map a flat list. That is a design step, and doing it by widening the paged
  forward's per-layer assumption until it stops throwing would produce a forward
  that reads whichever cache it happened to find -- the silent-wrong-context
  failure this row has refused twice already.

  **A `0` versus `1` discrepancy was flagged here and is WITHDRAWN**, because the
  second side had not been read. The registry NORMALIZES before validating --
  `const int64_t ratio = raw < 1 ? 1 : raw;`, commented as "`max(1,
  config.compress_ratios[layer_id])` -- upstream's own guard ... Our parser keeps
  the raw 0 for layers 0 and 1". So raw `0` and raw `1` both mean "no MLA latent
  cache", the histogram's `{0: 5, ...}` is the raw value and correct, and the
  registry's "accepts 1, 4 or 128" is the normalized one. Nothing is wrong.

  Left in rather than deleted, because the flag was landed and a reader who saw
  it deserves to see it withdrawn -- and because it is a fair example of the rule
  this row keeps relearning: a discrepancy between two records is not a defect
  until BOTH sides have been read in the tree. Reading one side and inferring the
  other is how the four earlier wrong estimates on this row were made.


- **W5's dispatch mechanism has landed UNREACHED, and this entry is the record
  AGENTS.md requires for that** (#2323). `ModelFactory::consumes_multi_kv` and
  `MultiKvRefusalApplies` turn `ModelRegistry::Forward`'s blanket refusal into a
  gated dispatch, and **no model sets the flag yet**, so the `true` arm is
  reachable only from `test_multi_kv_refusal`. Nothing regresses -- every model
  inherits `false` and hits the same refusal it hit before, which
  `test_runner`'s "a multi-cache forward is REFUSED, naming the channel" still
  asserts -- but the capability is not yet a capability.

  What makes it one is the rest of W5: `DeepseekV4Model::Forward` and
  `::ForwardDevice` consuming `attn_kv` by name instead of `(void)`-ing it, and
  the DeepSeek-V4 registration then declaring `consumes_multi_kv = true`. The
  bridge that work has to cross is named here so the next reader does not
  rediscover it: the runner hands over `PagedKvCache` entries keyed by published
  name, while `DeepseekV4ForwardGgufCached` (`deepseek_v4.cpp:2804`) takes a
  model-owned `DeepseekV4KvCache&`. Those two representations have to meet, and
  that is the substance of the wave rather than an incidental detail.

  Owned by `KV-DSV4-MULTICACHE` W5, tracked by
  [#2323](https://github.com/mudler/vllm.cpp/issues/2323). It blocks
  `MODEL-DSV4-DSA-COMPOSE` ([#2286](https://github.com/mudler/vllm.cpp/issues/2286)),
  which cannot start until a cache reaches the forward.


- [#1925](https://github.com/mudler/vllm.cpp/issues/1925) — the defect this
  document scopes. Owned by this row. Not fixed in flow: it is a multi-wave
  capability across the KV interface, the runner and the model, and this spike
  was scoped to produce a document.
- The reachability defect of "Our baseline" — `DeepseekV4ForwardGgufCached` is reached only
  from `examples/deepseek_v4_gen/main.cpp:198`, through the internal header
  included at `main.cpp:26`. Owned by this row, falls due at **W6**. Filed under
  #1925 rather than separately because closing #1925 through
  `ModelRegistry::Forward` is what closes it.
- The `MLAAttentionSpec` + non-`auto` `--kv-cache-dtype` refusal
  (`src/vllm/v1/kv_cache_interface.cpp:173-177`) blocks requesting `fp8_ds_mla`,
  the layout V4 actually ships. Owned by this row. Dated **W1** when this spike
  wrote it; **re-dated to W5** by `### W1 design — allocation metadata`, which
  records why lifting it at W1 would size a page for a store that does not
  exist. W1 landed the page formula the refusal names as missing.
- [#1960](https://github.com/mudler/vllm.cpp/issues/1960) — W1. Owned by this
  row, closed by W1.
- [#1973](https://github.com/mudler/vllm.cpp/issues/1973) — W2. Owned by this
  row, closed by W2.
- [#1974](https://github.com/mudler/vllm.cpp/issues/1974) — `spec_equal`'s
  missing `kMlaAttention` / `kSlidingWindowMla` arms. Owned by this row, FIXED
  IN FLOW with W2 and closed by it. Listed rather than omitted because the index
  row has to name an owner.
- **The published topology is ALLOCATED by W3 and still UNREAD.** W2 wrote this
  item as "UNREACHED": nothing allocated the seven groups and nothing read them.
  W3 closes the first half — the runner now builds a `PagedKvCache` for each of
  the 167 entries and hands them to the forward through
  `ModelForwardInput::multi_kv` — and leaves the second: `DeepseekV4Model::Forward`
  still discards `attn_kv`, and `ModelRegistry::Forward` refuses rather than let a
  multi-cache topology be silently ignored. Owned by this row, remainder falls due
  at **W5**. Tracked under
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925),
  [#1973](https://github.com/mudler/vllm.cpp/issues/1973) and
  [#2068](https://github.com/mudler/vllm.cpp/issues/2068).
- **The runner tolerates a second `kFullAttention` group on its KIND, not on
  `spec_on()`.** W2's refusal mirrors the draft-KV allocation loop's own
  predicate (`src/vllm/v1/worker/gpu/runner.cpp:1128-1130`), so a `fa_draft`
  group published with speculation OFF is still tolerated and still unallocated.
  Owned by this row. W2 dated this **W3**; **W3 did NOT close it**, and that is
  recorded rather than quietly dropped. W3's generalized path is entered only when
  a topology has groups the legacy path leaves over, and a `fa_draft` group
  published with speculation OFF is still absorbed by the kind-based draft-slot
  arm, so the legacy path still runs and the group is still unallocated. Closing
  it means making the draft slot depend on `spec_on()`, which changes behaviour on
  the SACRED speculative path and belongs to a wave that gates that path rather
  than to one whose obligation is that nothing about it moves. **Re-dated to
  W4.**
- **Only the `fp8_ds_mla` arm of the topology is published.** Upstream selects
  between a 576B-aligned `fp8_ds_mla` geometry and a 512B-aligned plain
  bf16/fp8 geometry on `use_fp8_ds_mla_layout` (`attention.py:140`,
  `nvidia/flashinfer_sparse.py:163`); W2 publishes the default `fp8_ds_mla` arm
  unconditionally because our factory signature carries no cache dtype and
  `ParseCacheDType` refuses the string by name
  (`include/vllm/v1/kv_cache_dtype.h:87-90`). The second arm, and the MXFP4
  68-byte indexer width (`attention.py:751-755`), are owed to **W5** together
  with the store path.
- **G1, this row's own topology gate, has NOT been run.** `## Gates` calls it
  "the gate that would have caught the stub", and W2 published the topology
  gated by literals read from upstream source rather than by a dump from the
  running oracle. PENDING on a named resource — a vLLM environment at the pin
  that imports `vllm.models.deepseek_v4`, plus the artifact repo's
  `config.json`, neither of which is on the host this wave was gated on. The
  blockers are measured in `### W2 gate result — G1's oracle side was NOT run,
  and is PENDING on a named resource`. Owned by this row; falls due at the next wave that has
  that environment, and W7 needs it anyway. Tracked under
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925).

- [#2068](https://github.com/mudler/vllm.cpp/issues/2068) — W3. Owned by this
  row, closed by W3.
- [#2076](https://github.com/mudler/vllm.cpp/issues/2076) — `ENG-MOE-LOADSTREAM`
  cites `model_registry.cpp:411` for `ModelSource::FromSafetensorsOwned`, a
  symbol that at base `c714b0234` sits at line **211** in a **392**-line file, so
  the cited line is **19** lines past the end of the file. PRE-EXISTING; W3's 63
  added lines pushed the file past 411 (and moved the symbol to **213**, which is
  the value the repaired citation carries), so the anchor moved from the `broken`
  bucket to the `stale` one and the ratchet fired. Owned by this row, FIXED IN
  FLOW with W3 and closed by it. Listed rather than omitted because the index row
  has to name an owner. **The three numbers in this entry were wrong when first
  written** — "402-line file", "198 lines past the end", and line 213 attributed
  to the base rather than to W3's head — and 198 is in fact the distance from the
  symbol at W3's head to 411, not any distance to the end of the file. Corrected
  here, in `.agents/issue-index.md` and in the pull request body before the squash
  made the index row uncorrectable.
- **The third forward channel is CARRIED and READ, but no model CONSUMES it.**
  `ModelForwardInput::multi_kv` reaches every registered forward and
  `ModelRegistry::Forward` refuses when one arrives, because no forward knows
  what to do with a cache set keyed by layer name. Owned by this row, falls due
  at **W5**, which replaces the refusal with `DeepseekV4Model::Forward` /
  `ForwardDevice` reading the caches instead of `(void)attn_kv`. Tracked under
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925) and
  [#2068](https://github.com/mudler/vllm.cpp/issues/2068).
- **The KV sizing helpers still count one page per hidden layer.**
  `KVBytesPerBlock` and `recurrent_state_bytes` (FIX-KV-GROUP-LAYER-COUNT,
  [#1963](https://github.com/mudler/vllm.cpp/issues/1963) /
  [#1966](https://github.com/mudler/vllm.cpp/issues/1966)) derive the pool budget
  from the layer count, which is 43 where W3 now allocates 167 buffers at four
  different page sizes. Nothing regresses today, because a DeepSeek-V4 engine
  refuses at its first forward and no other model has a multi-cache topology, but
  a `--kv-cache-memory` budget on the multi-cache path would be wrong. Owned by
  this row, falls due at **W4** with the rest of the non-uniform-`block_size`
  geometry. Tracked under [#2068](https://github.com/mudler/vllm.cpp/issues/2068).
- [#2084](https://github.com/mudler/vllm.cpp/issues/2084) — an EAGLE
  `AttentionSpec` group on a multi-cache topology passed W3's narrowed refusal
  and then received no buffer, because `attn_group_ids_` excludes
  `is_eagle_group` and the refusal loop never tested it: nine of ten published
  caches allocated, in silence. A fourth shape, where the comment said three.
  Owned by this row, FIXED IN FLOW with W3 and closed by it — the refusal gained
  an eagle clause and `test_runner`'s refusal case gained a subcase that is red
  without it. Listed rather than omitted because the index row has to name an
  owner.
- [#2085](https://github.com/mudler/vllm.cpp/issues/2085) — **the multi-cache
  view geometry contradicts the page it is built over.** Each buffer is
  `num_blocks * spec->page_size_bytes()` while its `FaDims` view is built from
  `spec->block_size`, and for a spec whose page derives from a
  `storage_block_size` the two disagree: the DeepSeek-V4 C4A latent
  (`block_size` 256, `compress_ratio` 4) has a **37440**-byte page while the view
  declares `{num_blocks, 256, 512}` = **131072** bytes per block, 3.5x what the
  page holds. `CheckKvCacheShape` cannot see it, because it compares the
  backend's declared shape against that same view metadata and so measures
  self-consistency rather than agreement with the allocation. INERT today:
  `ModelRegistry::Forward` refuses a multi-cache index before any kernel reads a
  view. Owned by this row, falls due at **W5** with the store path, because
  resolving it is entangled with two things W3 cannot settle — the `fp8_ds_mla`
  584 B/token layout is not expressible in `PagedKvCache` at all, and
  `tests/vllm/v1/worker/test_runner.cpp:1881` pins `block_size == 256` for that
  entry as a literal, a value the resolution may have to contradict. Given its
  own entry rather than folded into the W4 non-uniform-`block_size` item above:
  that item is about POOL BUDGETING (`KVBytesPerBlock` counting one page per
  layer) and this one is about the VIEW a kernel would index off.
- **Speculation turns ITSELF off on a multi-cache topology, and says nothing.**
  The draft-KV block is guarded by `!multi_cache_topology` so it cannot
  double-allocate a group the generalized loop already allocated; with
  `multi_cache_topology && spec_on()` the consequence is that `draft_attn_buf_`
  stays empty and `propose_drafts_block` returns early on
  `draft_attn_kv_.empty()`. That costs throughput and never tokens — no drafts
  means ordinary decode, which is the output the speculative path is required to
  produce — but it is a silent degradation, and W3's comment originally called
  the guard a refusal it is not. The comment now states the behaviour; making it
  a real `VT_CHECK` is owed, and is not done at W3 because no test surface can
  reach `spec_on()` on this path (the weights-based `GPUModelRunner` ctor takes
  no `SpeculativeConfig`), so the refusal could not be driven red. Owned by this
  row, falls due at **W4** with the `fa_draft`-on-`spec_on()` item above, which
  is the same seam. Tracked under
  [#2068](https://github.com/mudler/vllm.cpp/issues/2068).

## Evidence

Every structural claim in this document was read from a file at the SHA below.
Nothing was measured, no GPU was leased, and no build was run.

| what | where | read |
|---|---|---|
| our tree | `/home/mudler/_git/vllm.cpp`, `origin/main` at `a73b26968` | 2026-08-25 |
| pinned vLLM | `/home/mudler/_git/vllm` at `5559679229bc961848b121ccdeaa8fa5d79bec98` | 2026-08-25 |

Claims that are INFERENCE rather than reading, marked as such where they appear:

- The 167-entry total (`### What each layer caches, upstream`) is a count over construction sites against the
  `compress_ratios` array, not an observation of a running engine.
- "`kv_cache_tensors` is dead" (`## Risks/decisions`) is from a grep for its references, not from
  tracing every allocation path.
- The `NDEBUG` consequence (`## Risks/decisions`) follows from the construct being `assert`; no
  release build was run to observe it.
