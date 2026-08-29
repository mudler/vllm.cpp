# `MODEL-DSV4-DSA-COMPOSE` — the DeepSeek-V4 DSA composition

Issue: [#2286](https://github.com/mudler/vllm.cpp/issues/2286).
Owning row: `MODEL-DSV4-DSA-COMPOSE` (`.agents/model-matrix.md`).
Oracle: vLLM at the parity pin `5559679229`, `vllm/models/deepseek_v4/`.

## Now

`READY` — this spec is the deliverable of the scoping wave. No implementation
has started. W1 cannot begin until `KV-DSV4-MULTICACHE` W3 lands (see
`## Dependencies`).

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
= 46 entries. **46 is not 43**, and the row's older records say "43 layers" /
"41 of 43"; 43 is the trellis shard count (`exl3-layer-000..042`). W1 must
reconcile which number each claim means rather than inherit either (#2186
raised this and it is still open).

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
| W3 | OWED — the third `ModelForwardInput` channel |
| W4 | OWED — non-uniform `block_size` |
| W5 | OWED — consumption |

**W1 of this row cannot start before that row's W3.** The composition writes a
separate compressed cache beside a sliding-window raw cache, and it cannot reach
a cache the forward is not handed. This is a hard ordering, not a preference.

## Work breakdown

| wave | scope | depends on |
|---|---|---|
| W0 | this spec | — |
| W1 | reconcile 43 vs 46; layer-shape dispatch in `AttentionBlock`, SEQUENTIAL, replacing the refusal for the `compress_ratio == 128` (compressor-only) shape first | multicache W3 |
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

## Stop conditions

- Stop if `KV-DSV4-MULTICACHE` W3 does not land: W1 has no cache to write to.
- Stop before claiming any speed number. This row makes the model RUN; a
  throughput comparison against SparkInfer's 44-47 tok/s additionally needs
  `nvfp4_ds_mla` and K5 speculative decoding, neither of which exists here.
