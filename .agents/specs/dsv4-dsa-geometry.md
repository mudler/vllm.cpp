# DSV4-DSA-GEOMETRY — what the real DeepSeek-V4-Flash DSA tensors are, and why a reshape does not reach them

Status: **SCOPING ONLY. No code lands from this document.** It answers the four
questions `MODEL-DSV4-EXL3` `## Owed` needed answered before the real 99.5 GiB
EXL3 artifact could load, and it reports that the answer is not the one that
entry assumed. The design choice that follows is a `NEEDS_DECISION` returned to
the operator, not a choice this document makes.

Issue: [#1961](https://github.com/mudler/vllm.cpp/issues/1961)
Owning row: `MODEL-DSV4-EXL3`
Oracle: vLLM, primary, at the parity pin `5559679229bc961848b121ccdeaa8fa5d79bec98`
(`.agents/upstream-sync.md`), checked out at `/home/mudler/_git/vllm`. Every
`file:line` below is read at that pin. No secondary oracle is used or needed:
vLLM registers and implements this architecture in full.

Local anchors: every `file:line` into THIS tree is read at the head of
`row/DSV4-DSA-GEOMETRY` as it lands, NOT at `c00625141` where the measurement
was taken. Seven of the eight distinct local citations below went stale INSIDE
this pull request, because #1970's implementation landed in
`deepseek_v4.cpp` and `deepseek_v4_weights.cpp` beside this document — the same
mechanism that put a wrong `attention.py` line into seven places on this row.
Where #1970 also changed the BEHAVIOUR a paragraph reports, the paragraph says
so instead of being re-pointed at a line that now reads the other way.

## The measurement, taken first

`compress_ratios` in the artifact's own `config.json`
(`/mnt/nas_share/rc/ckpt/dsv4-flash-0731-spark-exl3/config.json`) is a
**per-layer list**, not a scalar: `[0, 0, 4, 128, 4, 128, ..., 4, 0, 0, 0]` over
46 entries (43 layers + 3 MTP). Layers 0 and 1 are `0`; layers 2..42 alternate
`4`, `128`; the MTP tail is `0`. That is **21 layers at `cr == 4`**, **20 at
`cr == 128`**, **2 dense** — which is exactly the "41 of 43 carry a compressor,
21 carry an indexer" the row recorded, now with the reason attached.

Read from the real shard headers (headers only, `carried-00{1..5}.safetensors`),
the DSA tensors split cleanly by that value:

| tensor | `cr == 4` layers | `cr == 128` layers | our loader wants |
|---|---|---|---|
| `attn.compressor.ape` | `[4, 1024]` | `[128, 512]` | `{cr, 512}` |
| `attn.compressor.wgate.weight` | `[1024, 4096]` | `[512, 4096]` | `{512, 4096}` |
| `attn.compressor.wkv.weight` | `[1024, 4096]` | `[512, 4096]` | accounted only |
| `attn.indexer.compressor.ape` | `[4, 256]` | — | accounted only |
| `attn.indexer.compressor.wgate.weight` | `[256, 4096]` | — | accounted only |
| `attn.indexer.compressor.wkv.weight` | `[256, 4096]` | — | `{128, 4096}` |
| `attn.indexer.weights_proj.weight` | `[64, 4096]` | — | `{64, 4096}` ✓ |
| `attn.indexer.wq_b.weight` | `[8192, 1024]` | — | `{8192, 4096}` |

So **four** tensors refuse, not the three `## Owed` names, and they refuse on the
21 `cr == 4` layers only. `compressor.ape` is the one the row had not counted:
`RequireShape` (`src/vllm/model_executor/models/deepseek_v4_weights.cpp:402-411`)
compared `{cr, hd}` = `{4, 512}` against the stored `[4, 1024]` — the loader as
it stood at `c00625141`, before #1970 made the expected width `{cr, coff * hd}`.
Every `cr == 128`
layer already satisfies our expectations byte for byte, and layers 0 and 1 carry
no compressor at all.

## Q1 — what are the two halves of the doubled dimension?

They are **the two overlapping compression windows a token belongs to**, not a
gate/value pair, not an interleave, and not a fusion.

Upstream derives the width from one line:

```
vllm/models/deepseek_v4/compressor.py:247-248
    self.overlap = compress_ratio == 4
    self.coff = 1 + self.overlap
```

and spends it on the projection and the APE table:

```
vllm/models/deepseek_v4/compressor.py:279-287
    self.fused_wkv_wgate = MergedColumnParallelLinear(
        self.hidden_size,
        [self.coff * self.head_dim, self.coff * self.head_dim],   # wkv | wgate
        ...
vllm/models/deepseek_v4/compressor.py:270-277
    self.ape = nn.Parameter(torch.empty((compress_ratio, self.coff * self.head_dim), ...))
```

`coff * head_dim` is `2 * 512 = 1024` for the main compressor at `cr == 4` and
`2 * 128 = 256` for the indexer's own compressor — every measured width above,
with no residue. At `cr == 128`, `overlap` is `False`, `coff` is `1`, and the
widths collapse to `head_dim`, which is why 20 of the 41 compressor layers load
today.

`CompressorStateCache` states the same fact independently, and pins the domain:

```
vllm/models/deepseek_v4/compressor.py:171-173
    assert compress_ratio in [4, 128]
    coff = 1 + (compress_ratio == 4)
    self.sliding_window = coff * compress_ratio
```

The pooling window is `coff * compress_ratio` tokens wide while a compressed row
is emitted every `compress_ratio` tokens. At `cr == 4` that is an 8-token window
stepping 4, so **consecutive windows overlap by exactly half** and every token is
pooled twice — once as a member of the older half of a window, once as a member
of the newer half. The two halves of `wkv`/`wgate`/`ape` are the two projections
that serve those two roles.

## Q2 — how does upstream index or split them?

It never splits the weight. The split that `packed_modules_mapping` performs
(`vllm/models/deepseek_v4/nvidia/model.py:1157-1158`, and identically in
`amd/model.py:706-707`, `xpu/model.py:1166-1167`) is the *opposite* operation —
it **merges** the checkpoint's separate `compressor.wkv` and `compressor.wgate`
into one `MergedColumnParallelLinear` so the GEMM runs once. Our checkpoint
stores them unfused, which is the storage form upstream loads from.

The `coff` halves are selected at **gather time, by window position**:

```
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:164-183
    # in _fused_kv_compress_norm_rope_insert_sparse_attn (def at :114). Neither
    # line below picks out its own line in the file: the indexer kernel repeats
    # them at :712 and :730, the mxfp4-indexer kernel at :891 and :909.
    if (position + 1) % COMPRESS_RATIO != 0:
        return                                              # boundary tokens only
    start  = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)   # the coff*cr window
    ...
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
```

`HEAD_SIZE` here is `head_dim` (512), not the stored width. A row in the older
half of the window (`tokens < cr`) reads its state at offset 0; a row in the
newer half reads at offset `head_dim`. The full `coff * head_dim` row is written
once per token by `save_partial_states`
(`vllm/models/deepseek_v4/common/ops/save_partial_states.py:86-101`, where
`HEAD_SIZE` is `kv.shape[-1]`, the *full* `coff * head_dim`, and the score half
gets `ape[position % cr]` added), and is read back twice, half at a time, by two
different windows.

So the layout is a plain concatenation along the output dimension, addressed by a
role the token only acquires relative to the window doing the gathering. Nothing
about the halves is recoverable from the tensor alone.

## Q3 — does our host forward want one half, both, or something else?

**Neither.** Our host forward does not implement the composition these tensors
belong to, so there is no half to hand it.

`AttentionBlock` (`src/vllm/model_executor/models/deepseek_v4.cpp:827-857`)
composes the compressor as: for **every** token `t`, softmax-pool a **fixed
`win = 2`** window of the **MLA's own `kraw` latent** and overwrite `latent[t]`
in place. The file says so itself at `:32-33`. Upstream, at the same point,
emits a row **only at `(position + 1) % cr == 0`**, pools **`coff * cr` = 8 or
128** rows, of a **separate `compressor.wkv` projection** that our loader
deliberately does not materialize, into a **separate compressed KV cache** that
sits beside the raw one.

The differences are not parameters of one algorithm:

| axis | upstream | ours |
|---|---|---|
| compressor KV source | `compressor.wkv`, its own projection | reuses MLA `kraw` |
| window width | `coff * cr` (8 or 128) | fixed 2 |
| emission | boundary tokens only, 1 row per `cr` | every token |
| destination | separate compressed cache | overwrites the dense latent |
| `coff` half selection | `head_offset` by window position | absent |
| indexer `wq_b` input | `qr`, the q-LoRA latent (`DeepseekV4Indexer.forward`, `attention.py:835`) | `x`, the hidden state (`deepseek_v4.cpp:915`) |
| indexer K | `indexer.compressor`, a pooled compressor | plain `Gemm(idx_wk, x)` (`:917`) |
| indexer selects among | compressed rows | raw rows |
| attention keys | SWA(128) raw ∪ selected compressed, one softmax | all raw rows, dense causal |

The `wq_b` row is worth separating from the rest, because it is **not a width
problem at all** and it is a defect independent of this decision. Upstream builds
it as `ReplicatedLinear(self.q_lora_rank, self.head_dim * self.n_head)`
(`vllm/models/deepseek_v4/attention.py:721-726`) and calls it on `qr` in
`DeepseekV4Indexer.forward` (`:835`). The stored `[8192, 1024]` is therefore `[inh*ihd, q_lora_rank]` at its
natural size — `64*128` by `q_lora_rank == 1024` — and nothing about it is
doubled. At `c00625141` our loader asked for `[inh*ihd, H]` = `[8192, 4096]`;
#1970 has since moved that K to `q_lora_rank`
(`deepseek_v4_weights.cpp:995`), so the LOADER half is repaired. Our forward
still feeds it `x` (`deepseek_v4.cpp:915`), so we still project the indexer
query from the wrong space.

No gate has ever seen that, and the reason is NOT that `H` and `q_lora_rank`
coincide at the synthetic geometry. They do not: `dsv4_exl3_fixture.h:141`
sets `kHidden` to 256 and `:149` sets `kQLora` to 128. The reason is that the
collapsed fixture WRITES `wq_b` at `K = H`, to match what our forward feeds it
— so the two agree by construction and the disagreement never appears. It
takes a fixture that writes the real geometry everywhere else and collapses
this one tensor to make it visible, which is what
`FixtureOptions::collapsed_indexer_wq_b` now does.

## Q4 — is `dsa_dense` a mode upstream has?

Upstream has a per-layer dense mode, and it is **config-driven, not
source-driven**:

```
vllm/models/deepseek_v4/attention.py:209   self.compress_ratio = max(1, config.compress_ratios[layer_id])
vllm/models/deepseek_v4/attention.py:334   if self.compress_ratio > 1:      # compressor exists
vllm/models/deepseek_v4/attention.py:274   if self.compress_ratio == 4:     # indexer exists
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:263   swa_only = self.compress_ratio <= 1
    # DeepseekV4FlashInferMLAAttention.forward_mqa; the same line is at :686
    # and :793 on DeepseekV4FlashInferSM120Attention.
```

`dsa_dense = (be.gguf != nullptr)` (`deepseek_v4.cpp:776`) keys off the
**weight source**. Upstream keys off `compress_ratios[layer_id]`. The two agree
on nothing: on this checkpoint the config-driven predicate is true for 2 layers
and false for 41, while ours is true for all 43 whenever the source is GGUF. So
`dsa_dense` is our workaround, and widening it to the EXL3 source is wrong for
the reason the task states and for a second one below.

**And upstream's dense layers are not dense.** `swa_only` means *sliding-window
only*, at `config.sliding_window == 128` (`attention.py:204`). Even layers 0 and
1 attend a 128-token window, never the full prefix. Our forward has no sliding
window anywhere (`grep -n sliding_window src/.../deepseek_v4.cpp` returns only
prose).

## The claim `dsa_dense` rests on is false, and that is the finding

`deepseek_v4.cpp:763-775` justifies forcing the DSA path off with: dense MLA "is
EXACT, not an approximation, whenever `seq_len <= index_topk` (=512): the indexer
cannot select more tokens than exist, so top-k over ≤512 tokens IS the full
causal set". [#1925](https://github.com/mudler/vllm.cpp/issues/1925) repeats it.

The premise is right about the indexer and wrong about the attention. On a
`cr > 1` layer, one kernel takes **one softmax over the union** of the raw
sliding window and the selected compressed rows:

```
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:769-782
    # DeepseekV4FlashInferSM120Attention._forward_decode; the same call is at
    # :486, :511 and :888.
    flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=swa_cache,          sparse_indices=swa_indices,
        compressed_kv_cache=extra_cache, extra_sparse_indices=extra_sparse_indices,
        sinks=self.attn_sink, ...)
```

The compressed rows are **pooled aggregates of `coff * cr` raw rows**, not raw
rows, so no selection over them can reproduce attention over raw rows. Their
number does not depend on `index_topk` either: rows are emitted at every
`(position + 1) % cr == 0`, so a 10-token prefill at `cr == 4` already has two,
and the short-context branch (`attention.py:813-830`) explicitly still builds the
K cache and selects **all** candidates — its comment says "we still need to build
k cache". Upstream at `seq_len == 10, cr == 4` attends 10 raw keys **and** 2
compressed keys. We attend 10.

Upstream's attention here is hierarchical: recent tokens at full resolution,
older tokens pooled `cr:1`, jointly normalized. Dense causal attention is not
that at any sequence length. **A token gate below 512 cannot detect this, and
neither can one above it.** The consequence reaches past this row: the GGUF arm
runs `dsa_dense` on the real geometry today, so the shipping GGUF DeepSeek-V4
path is already not upstream's attention on 41 of 43 layers.

## Why this is `NEEDS_DECISION` and not a patch

The loader-side half is small and unambiguous — derive the expected widths as
upstream does (`coff = 1 + (cr == 4)`) and take `wq_b`'s K from `q_lora_rank`.
But landing only that is worse than the refusal it removes, and #1970 corrected
what "worse" means here. This paragraph used to say `Gemm`'s host arm is a
`MatVec` with no length check, so the wrong stride would be read SILENTLY. It is
not: `deepseek_v4.cpp:413` is an unconditional `VT_CHECK` and the keep-quant arm
checks the shape too. The moment `comp_wgate` materializes at `[1024, 4096]` and
`AttentionBlock` calls `Gemm(..., T, hd, H)` with `hd == 512`, what happens is an
ANONYMOUS `vt: MatVec weight size mismatch` from the middle of a forward, on a
checkpoint that loaded successfully, naming no tensor, no layer and nothing
missing. That is a worse DIAGNOSTIC than the loader refusal it replaced, not a
worse numerical outcome, so the refusal must not move without the forward moving
with it.

Three shapes the decision can take. This document recommends none.

- **(A) Port upstream's DSA.** Separate `compressor.wkv`; the `coff`-overlapped
  window with `head_offset` role selection; boundary emission; a compressed KV
  cache beside a SWA(128) raw cache; the indexer on `qr` over compressed rows;
  one joint softmax over the union. The only path to token parity with vLLM on
  this checkpoint. It is a multi-wave model port and it needs the cache topology
  [#1960](https://github.com/mudler/vllm.cpp/issues/1960) and
  [#1925](https://github.com/mudler/vllm.cpp/issues/1925) are already scoping —
  105 of V4's 167 cache entries cannot be sized at all today. Our primitives
  (`CompressorSaveScoreApe`, `CompressorPoolNorm`) are already generic over
  width and window and would carry over unchanged; the gap is the composition,
  not the math.
- **(B) A per-layer dense selector both arms read**, mirroring
  `compress_ratios[layer_id]` rather than the weight source. Honest as a record
  and it fixes the arm-divergence the row's equivalence gate would otherwise
  suffer, but on 41 layers it is still not upstream's attention, so it cannot be
  gated as parity and must not be described as exact.
- **(C) Loader accepts, forward refuses by name.** Materialize every DSA tensor
  at its real width and move the refusal into `AttentionBlock`. The artifact
  then loads and its non-DSA capabilities — the EXL3 tower at real scale, MoE,
  MTP, W2 residency — become reachable, with the DSA layers refusing instead of
  mis-indexing. Under `## Nothing lands dead` this is a staged slice and owes a
  named owner for the wiring.

(A) is what "mirror vLLM" means here. (C) is what unblocks the row this week.
They are not exclusive — (C) is a strict prefix of (A) — but which one is dispatched,
and what the row's equivalence gate compares against once the two arms stop
sharing an attention path, exceeds a helper's authority.

## Owed

- The decision above.
- `MODEL-DSV4-EXL3` `## Owed` names three refusing tensors; there are four.
  `attn.compressor.ape` is missing from it, and `indexer.wq_b` is listed there
  as a width problem when it is a wrong-input-space problem.
- The `indexer.wq_b` input-space defect (`x` where upstream uses `qr`) is real at
  any geometry and is not fixed by any of (A)/(B)/(C) on its own.
- The GGUF arm's `dsa_dense` exactness claim (`deepseek_v4.cpp:763-775`) is
  false and is quoted onward by #1925. Whoever takes the decision owns
  correcting both prose sites.

## Now

`SCOPING`. No lifecycle state moved. No product code changed.
