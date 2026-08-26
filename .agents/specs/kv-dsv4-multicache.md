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

`READY` — this spec is committed and no wave has started. Nothing is claimed,
nothing is implemented, and no product code changed. The deliverable of this row
so far is this document plus the issue that owns it. The waves in
`## Work breakdown` are proposals, not commitments, and no wave has an owner.

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
(`compressor.py:288-293`) whose `state_dim = 2 * coff * head_dim`, `coff = 1 +
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
| `attention.py:315-321,626-645,669-684,761-777` + `compressor.py:288-293` the four declaration sites | `MakeDeepseekV4KVCache` (`src/vllm/model_executor/models/deepseek_v4_registry.cpp:126-148`) (W2) |
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
| **W5** | `Forward`/`ForwardDevice` consume `attn_kv`: the DSA-sparse attention path reading the published caches, replacing `(void)attn_kv`. Removes the `VT_CHECK(!is_indexer && !is_comp, ...)` at `deepseek_v4.cpp:786-787`. | CPU at synthetic config; **GPU** for the real geometry | L |
| **W6** | Reachability + ABI: the capability reachable from `ModelRegistry::Forward` and exposed through `include/vllm.h`, so `examples/deepseek_v4_gen` stops including an internal header (`## Our baseline`). | **CPU** | S |
| **W7** | The oracle gate of `## Gates`. | **GPU**, ≥2 GB10 | M |

W1, W2, W4 and W6 are fully CPU-gateable. W3 is CPU-gateable for its own
guarantees but its byte-neutrality obligation reaches every model, so its full
gate includes the SACRED `test_qwen35_paged_engine` regression. W5 and W7 need
the GPU.

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
  the layout V4 actually ships. Owned by this row, falls due at **W1**.

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
