# `Qwen4ExpForConditionalGeneration` (Qwen3.8-Flash-Next)

**Campaign row:** `MODEL-MM-QWEN4-EXP` (the ID carried by the branch, the issue and
the append-only index row)
**Model-matrix target row:** `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation`,
the deterministic ID the row contract requires. Both name the same work; the index
row is append-only and cannot be re-keyed, so both are recorded rather than one
silently replaced.
**Issue:** [#1978](https://github.com/mudler/vllm.cpp/issues/1978)
**State:** `READY` (spec only; no product code lands under this row's first pull request)
**Motivating checkpoint:** `Qwen/Qwen3.8-Flash-Next`, released 2026-08-24, read live 2026-08-26

## Scope

Port `Qwen4ExpForConditionalGeneration` / `model_type: qwen4_exp`. The card calls it
"this experimental preview of the architecture that will underpin Qwen4"; the
`Qwen3.8` in the name is marketing continuity and not a shape relationship. It is a
180B-total / 6B-activated multimodal (image-text-to-text) hybrid: 48 layers in a
repeating `3 x linear_attention -> 1 x qwen_sparse_attention` pattern, 512-expert MoE
at top-10 plus one shared expert, a 20M-entry n-gram embedding table injected at
layer 2, a 4-branch gated residual stream, and a 1-layer MTP head.

In scope: text generation and the image/video path, every published quantized arm,
and the GGUF k-quant arms this repository requires of any model port.

Out of scope for the first implementation wave, each named under `## Owed` rather
than dropped: MTP depth > 1, the 1M-token RoPE extension the card advertises above
the native 262144, and any throughput claim.

## Why this needs a spec before code

Three of this row's decisions are expensive to reverse and cheap to get wrong, and
all three have already been made incorrectly once by an agent reading a related
record. They are settled here so a fresh implementer does not re-derive them.

1. **This is not a Qwen3.8 row.** `.agents/specs/qwen38-27b-bf16-gate.md` records
   `Qwen/Qwen3.8-27B` as the Qwen3.6-27B shape retrained, differing in exactly one
   config key. That precedent does not extend here. `qwen4_exp` shares an ancestor
   with `qwen3_5` and diverges in four load-bearing places.
2. **QSA's twin in vLLM is MiniMax-M3, not DeepSeek-V4.** See `## Design`. Building
   it on the DSA/MLA path is the wrong port, and DSA is the path an agent reaches
   for first because this tree already has it.
3. **The oracle split is a direction, not a default.** See below.

## Oracles

**vLLM implements nothing here.** Read live at `origin/main` = `6a5e8f5979`,
2026-08-26: no `qwen4*` path, no registry entry, and a repository-wide GitHub search
for `qwen4` returns zero results. `vllm-omni` likewise. This is absence from vLLM
`main`, not staleness in our pin (`555967922`), so a pin advance does not reach it.

**Developer direction, 2026-08-26: transformers is the oracle for the ALGORITHM,
vLLM supplies the OPS.** Recorded verbatim because it is the axis the whole row
hangs on: "use transformers as oracle for algorithmic side. but use ops from vllm so
we account for optimized path."

This is the correct reading of what each upstream is, and not a split of
convenience. transformers [#48337](https://github.com/huggingface/transformers/pull/48337)
(MERGED 2026-08-26, 5211 lines) is a semantics reference that says so in its own
code: `Qwen4ExpTextQSAIndexer.forward` loops in Python over `(batch_idx, query_idx)`
and carries the comment "we only allow eager and sdpa". Ported as written it yields
a correct model at an indefensible speed. AGENTS.md's "Mirror vLLM" polarity
continues to bind every primitive vLLM implements, even though vLLM has never
assembled this particular model from them.

Therefore: **every component resolves against exactly one oracle, named in the
`## Design` table. An implementer who cannot name the oracle for the line they are
writing has found a gap in this spec and returns `NEEDS_CONTEXT`.**

SGLang [#36497](https://github.com/sgl-project/sglang/pull/36497) is OPEN and is not
admissible while it stays open. Re-check it at each wave; if it merges it becomes a
second op source under the `sglang` registry id, still ranked below vLLM.

### The transformers lane pin (ACCEPTED 2026-08-26)

`.agents/oracles/transformers.md` pins transformers to **5.14.1**, deliberately tied
to what the pinned vLLM environment resolves, on the stated ground that an
independent pin "would let the oracle environment hold two different `transformers`
at once, which is the drift this registry exists to stop".

**5.14.1 does not contain `Qwen4Exp`**, so this row cannot run its algorithmic
oracle under the existing pin.

The exception argued here is narrow: the invariant guards against a vLLM environment
and its transformers drifting apart, and for `qwen4_exp` there is no vLLM
implementation to drift from. A lane-scoped second pin therefore cannot create the
inconsistency the rule exists to prevent. It is recorded in the oracle file as a
lane exception naming this row and this issue, and it expires the moment vLLM
registers `qwen4_exp`, at which point the row reconciles onto vLLM and transformers
demotes to the preprocessing role it holds everywhere else.

**Accepted by the developer on 2026-08-26**, having been put as an explicit
accept-or-reject rather than passed as housekeeping, because it changes the
semantics of a registry invariant.

**The lane pin is `transformers` 5.16.0, and it is a real release, not a branch
SHA.** That was not the expected outcome and it is better than one. `Qwen4Exp`
merged to `main` at 12:03:40Z on 2026-08-26 and `v5.16.0` was published at
12:35:15Z, 32 minutes later. Bounded rather than assumed, by fetching the model
source at each tag on 2026-08-26: `v5.16.0` returns HTTP 200 and `v5.15.0` returns
HTTP 404, so 5.16.0 is the FIRST release containing the architecture, which is the
tightest pin available. Its `auto_mappings.py` carries 5 `qwen4_exp` occurrences, so
the registration landed with the model rather than trailing it.

The version string is **unmeasured**: it is the release that provably contains the
model, not a `transformers.__version__` read off a running oracle. Resolving the
runtime string is owed to the first wave that stands one up. Full record and the
`oracle-pin-lane` block: [`../oracles/transformers.md`](../oracles/transformers.md).

### Gateability

`gateable = no` at the time of writing, and the reason is memory rather than
software: see `## Hardware`. The oracle must demonstrably build **and run the
model**, and no published artifact fits any fleet device. The first wave's real
deliverable is the arm that makes an oracle run possible at all.

## Upstream chain

| Source | Revision | Role |
|---|---|---|
| `huggingface/transformers` | **`v5.16.0`** (lane pin; first release containing `qwen4_exp`, landed by `#48337` merged 2026-08-26) | algorithm; `models/qwen4_exp/modular_qwen4_exp.py` is the authored delta, `modeling_qwen4_exp.py` the generated expansion |
| `vllm-project/vllm` | `origin/main` `6a5e8f5979` (survey only; the parity pin stays `555967922`) | ops |
| `Qwen/Qwen3.8-Flash-Next` | HF `main`, read 2026-08-26 | config and weights |

Read the **modular** file, not the generated one. It is 1186 lines against 2707 and
it is the file that states what is inherited unchanged, which is most of the model.

## Our baseline

What this tree already has, and therefore what the port does NOT rebuild. Stated
first because the delta only means something against it, and because the size of
this list is the reason the row is tractable at all.

- **The Qwen3.5 family end to end.** `src/vllm/model_executor/models/qwen3_5*.cpp`
  carries the dense and MoE backbones, the GGUF weights path, the MTP draft
  (`Qwen3_5MTPModel`) and the runner integration. `Qwen4ExpTextModel` inherits from
  `Qwen3_5MoeTextModel`, so this is the base the upstream delta is written against.
- **GDN linear attention with a Triton-AOT fast path.** `src/vt/cuda/cuda_gdn.cu`.
  The AOT specializations are pinned to `K=V=128, Hg=16, H in {48,32}` and this
  model's `linear_key_head_dim` / `linear_value_head_dim` / `linear_num_key_heads` /
  `linear_num_value_heads` are `128 / 128 / 16 / 48`. An exact hit, not a near miss.
- **A working sparse-attention indexer**, `deepseek_v4_dsa.cpp` +
  `deepseek_v4_compressor.h` + `src/vt/cuda/cuda_deepseek_v4.cu`. Useful for its
  compressor and its cache plumbing; **not** the right base for QSA's selection
  path, see `## Port map`.
- **Hyper-connection residual streams**, `deepseek_v4_mhc.cpp`, ported 1:1 from
  vLLM's `kernels/mhc/`. Different math from Gated Residual, same fused shape.
- **Interleaved mRoPE**, `layers/rotary_embedding/mrope.cpp`, which this model needs
  (`mrope_section [11, 11, 10]`, `partial_rotary_factor` 0.25 over `head_dim` 256).
- **The Qwen3.5-Moe vision tower**, which upstream reuses here **unchanged**.
- MoE with grouped GEMM, and the GGUF k-quant loader stack.

## Design

`Qwen4ExpTextModel` inherits from `Qwen3_5MoeTextModel` and leaves the rotary
embedding, MLP, experts, TopK router and the **entire vision tower** unchanged
(`class Qwen4ExpVisionModel(Qwen3_5MoeVisionModel): pass`). This tree already has all
of that. The port is the delta below.

## Port map

| Component | Algorithm oracle | Op oracle (vLLM) | This tree |
|---|---|---|---|
| GDN linear attention | `Qwen4ExpTextGatedDeltaNet` | `layers/mamba/gdn/qwen_gdn_linear_attn.py` | **HAVE.** `K=V=128, Hg=16, Hv=48` is an exact match for the AOT gate in `TryTritonPackedDecode` / the delta_h dispatch (`src/vt/cuda/cuda_gdn.cu`, pinned to `K=V=128, Hg=16, H in {48,32}`) |
| Grouped RMSNorm | `Qwen4ExpTextRMSNorm(group_size=)` | `layers/layernorm.py` **`RMSNormGated`** (`group_size`), NOT the plain `RMSNorm` | new, small; see the correction below |
| QSA block scoring + top-k | `Qwen4ExpTextQSAIndexer` | **DeepSeek-V4 C4 indexer lane**: `fp8_mqa_logits` / `top_k_per_row`, `v1/attention/backends/mla/indexer.py`. NOT MiniMax-M3, see below | new |
| QSA pooled-key build | indexer forward | the **Triton** `head_dim=128` compress/norm/RoPE/store kernel with `OVERLAP=False` and the pool replaced by a mean. NOT the CuteDSL `SparseAttnCompressNormRopeStoreC4Kernel`, which refuses `overlap=False` and pools by learned softmax over 8 | partial: `deepseek_v4_compressor.h` |
| Indexer side cache | `Cache.update_indexer` | `MLAAttentionSpec(num_kv_heads=1, head_size=128, tokens_per_state=4)` + `get_compressed_slot_mapping`, as-is; M3 supplies only the registration precedent | new KV spec |
| Gated Residual | `Qwen4ExpTextGatedResidual` | `layers/mhc.py`, `kernels/mhc/*` (**different math**, same fused shape) | partial: `deepseek_v4_mhc.cpp` |
| MoE 512 / top-10 + 1 shared / intermediate 640 | `Qwen4ExpTextSparseMoeBlock` | FusedMoE, grouped GEMM | HAVE, shape change only |
| MTP, 1 layer, `hybrid: true` | config `mtp` | `qwen3_5_mtp.py` | HAVE, needs extension |
| PLE dilated depthwise conv | `Qwen4ExpTextPLELayer._short_conv` | **NONE** | new, no vLLM op |
| N-gram hashed embedding | `Qwen4ExpTextNGramEmbedding` | **NONE** | new, no vLLM op |
| Vision tower | `Qwen4ExpVisionModel` = `Qwen3_5MoeVisionModel` | qwen3_5 vision | HAVE. `deepstack_visual_indexes: []`, so no deepstack |

**Exactly two components have no vLLM op**, and they are the two where transformers
is the sole source and we author the kernel ourselves. Everything else has an
optimized vLLM form to mirror, and mirroring it is mandatory rather than optional.

### QSA maps to DeepSeek-V4's C4 indexer lane, NOT to MiniMax-M3

**This reverses the call this spec was first written with, and the reversal is the
most important thing in the document.** The original reading was that QSA, being plain
GQA rather than MLA, had to map onto vLLM's non-MLA block-sparse case (MiniMax-M3) and
not onto DeepSeek's DSA. That reasoning was wrong, and it was wrong for a specific,
checkable reason recorded here so it is not repeated: **`MLAAttentionSpec` is not an
MLA claim.** M3's own indexer cache uses it while being a plain-GQA model, and the
comment beside it says why -- "Key-only: MLAAttentionSpec budgets one vector/token (not
2x for K+V)". It is a budget shape, not an architecture assertion. Once that prop is
removed, the GQA-versus-MLA argument for preferring M3 collapses entirely.

Verified at vLLM `origin/main` = `6a5e8f5979`.

**Nine independent structural matches with DeepSeek-V4**, and `compress_ratio == 4` is
literally the same number:

| QSA (transformers v5.16.0) | DeepSeek-V4 (vLLM) |
|---|---|
| index MQA, 1 key head, dim 128 | index MQA, 1 key head, dim 128 |
| score `relu(q.k)` summed over index heads | `(score.relu() * weights).sum(dim=0)` |
| scale `1/sqrt(indexer_head_dim)` | `softmax_scale = head_dim ** -0.5` |
| one score set per query token, no head axis | `topk_indices_buffer[num_tokens, topk]` |
| pool `compress_ratio` tokens into one key | boundary `(position + 1) % COMPRESS_RATIO == 0` |
| `k_layernorm` on the pooled key | RMSNorm on the compressed key |
| RoPE at the **block-start** position | `compressed_pos = (position // CR) * CR` |
| candidates = `visible // compress_ratio` | `len_per_token = (start_pos + 1 + offset) // CR` |
| one stored state per 4 tokens | `MLAAttentionSpec(tokens_per_state=compress_ratio)` |

`tokens_per_state` is a first-class KV-cache field upstream, documented as "Ints > 1
compress multiple tokens into one state (DSv4 sparse MLA)". It is exactly what QSA's
side cache needs, and it does not exist on the M3 path.

**Why M3 is not merely a worse fit but a different algorithm.** Its score is
`tl.max(qk, axis=1)` over 128 **raw** token dots -- no pooling stage, no relu, no head
reduction -- and it asserts `num_idx_heads == num_kv_heads` with the comment "no topk
index reduce", so it produces one independent block set **per KV head** where QSA
produces one set per token. Its `SPARSE_BLOCK_SIZE = 128` is not a tunable: the file
states "One sparse block == one KV page", and both the score and the attend index
`block_table[blk]` on that identity. Moving it to 4 would force a KV page size of 4 and
break `tl.dot`, whose tile needs at least 16.

**M3 still contributes exactly one thing, and it is a wiring precedent rather than an
algorithm:** the demonstration that a plain-GQA model can own a key-only side cache
through `MLAAttentionSpec` and a private indexer backend registered into
`static_forward_context`. Take that pattern; take no kernel.

**The genuinely new work is the CONSUMER, and nothing upstream supplies it.** Every
DSv4 sparse consumer attends to the **compressed** MLA KV, one state per four tokens.
M3's consumers attend to raw tokens but only at page granularity. QSA attends to **raw
tokens selected at ratio-4 granularity**, which no vLLM consumer does. The port has to
expand block id `b` into tokens `[4b, 4b+4)`, append the ragged tail, and run dense GQA
(24 query heads over 2 KV heads, `head_dim` 256) across the gathered set.

**Two silent-failure traps follow, and both would pass a naive gate.**

1. Wiring QSA's top-k straight into a DSv4 sparse-MLA consumer attends to a **pooled**
   key and value instead of the four real tokens. It still produces plausible output.
   A short-prompt token gate cannot catch it, because at context <= `indexer_budget`
   every candidate is selected and the only remaining difference is the value pooling.
   Any QSA gate must therefore run past 2048 tokens of context to be worth anything --
   a requirement this spec did not previously state and which changes what `## Gates`
   has to demand.
2. `SparseAttnCompressNormRopeStoreC4Kernel` does **not** mean-pool, despite being the
   closest-named kernel. It is a learned softmax-weighted pool over an **overlapping
   window of 8** driven by a score channel QSA's checkpoint does not have, and the
   CuteDSL variant refuses `overlap=False` at compile time. Its scaffolding is a direct
   match -- boundary predicate, block-start RoPE, paged store -- but the pooling
   operator must be replaced with an unweighted mean over a non-overlapping window of
   4. The **Triton** `head_dim=128` variant, where `OVERLAP` is a plain `constexpr`, is
   the correct starting point; the CuteDSL C4 one is not.

Also reconcile, and do not inherit: DSv4 has a `weights_proj` producing per-head logit
weights that QSA has no tensor for (QSA's weight is the constant `1/sqrt(128)`), and
its RoPE is GPT-J-style over a trailing contiguous span, whereas QSA uses interleaved
mRoPE over the **leading** 64 dims with the NoPE dims trailing -- the halves are
swapped end for end.

### Two structural consequences beyond the module list

- **The residual stream is `hc_count * hidden_size` = 4 x 2560 = 10240 wide through
  the whole stack.** `Qwen4ExpTextGatedResidual` reads it through a grouped RMSNorm
  and a low-rank (`hc_lowrank` = 320) SiLU-then-sigmoid gate, collapses to 2560 for
  the block, and writes back with a per-branch scalar gate
  (`2 * sigmoid(block_inject_weight(x) / hc_count)`). This is a change to the
  per-layer loop and to every residual buffer, not a drop-in module. The
  `Qwen4ExpTextModel` also holds one `use_combine=False` mixer that collapses the
  stream at the end.
- **`number_of_conv_states = 3` on a PLE layer** (GDN conv, PLE conv, and the n-gram
  token history, which upstream stores as a third conv state precisely because the
  manipulations are identical). The KV-cache spec grows a third conv stream plus the
  indexer side cache. Adjacent to [#1963](https://github.com/mudler/vllm.cpp/issues/1963)
  and [#1966](https://github.com/mudler/vllm.cpp/issues/1966).

### The n-gram embedding is integer-exact or it is silently wrong

Derived from the lane pin and then **verified against the published checkpoint** by
range-reading the safetensors payload, so these are read values and not predictions.

`config.seed` is absent from the published config, so the dataclass default **1234**
applies. That was confirmed rather than assumed: reconstructing the splitmix64 chain
at seed 1234 gives `layer_multipliers = [23703573157769, 20109073645365,
8052911324071]`, and a range read of that buffer out of
`model-00005-of-00131.safetensors` returns those three values exactly.

Head vocab sizes are the successive primes after `ngram_vocab_size_base - 1`, so head
0 is 20000003 and head 15 is 20000171; `total_vocab_size = 320001446`, padded to
**320001536** (90 unaddressable rows), giving `320001536 x 160 = 51,200,245,760`
parameters. `ngram_heads_vocab_sizes` and `ngram_heads_offsets` were range-read from
the checkpoint and match the derivation entry for entry.

**The three C++ divergence sites, ranked.** All are silent.

1. **`_splitmix64` must be `uint64_t` throughout.** Its `>> 30 / 27 / 31` are logical
   shifts on a non-negative Python int; on a signed `int64_t` they become arithmetic
   shifts and the multiplier is wrong. The value has its top bit set about half the
   time, so this fires immediately.
2. **`_splitmix64(value) % half_bound` must be an unsigned modulo.** The dividend
   routinely exceeds 2^63; a signed modulo yields a negative residue.
3. **Shard reassembly must be NUMERIC, not lexicographic.** The table ships as 128
   shards, `shard_0 .. shard_127`, each bf16 `[2500012, 160]`. Sorting the key strings
   gives `shard_0, shard_1, shard_10, shard_100, ...` and silently permutes a 95 GiB
   table. Verified against the checkpoint index: 128 keys, contiguous 0..127, and a
   lexicographic sort does produce that wrong order.

The forward itself is int64-exact and does NOT depend on Python bignum:
`layer_multipliers[i]` is a 0-dim int64 tensor, so the product is int64 arithmetic,
and it is bounded below 2^63 by construction because `multiplier_max * vocab_size <=
2^63 - 1`. **That bound holds only while every token id is below `vocab_size`.** An
out-of-range id overflows int64 and diverges silently, so the loader must not admit
one. Because `mixed` is therefore always non-negative, Python's `%` and C's truncating
`%` agree, and a port may use `int64_t %` without a sign correction.

Two further traps found in the cache path. The history of the previous
`ngram_size - 1` token ids is stored in the linear-attention cache as conv state 2 and
its dtype is **int64**, taken from the first tensor written; a port that stores it as
a float rounds token ids. And upstream's `update_conv_state` pads with **0**, which is
a valid token id, so the model works around it with an explicit EOS left-pad. Pad with
EOS, never with zero.

`split_ngram_parts` is **not used in the forward at all**. It is a checkpoint-layout
parameter consumed only by the weight conversion mapping, and saying so here stops the
next reader hunting for it in the model code.

**The PLE sits on decoder layer 1, not layer 2.** `ple_layer_ids` is 1-indexed and the
lookup is `config.ple_layer_ids.index(layer_idx + 1)`, so `[2]` selects 0-based layer
1. Confirmed from the checkpoint index: every PLE tensor is under
`model.language_model.layers.1.ple.`, and no other layer has one.

### PLE: a strided-history conv with no vLLM op, confirmed

**The dilated depthwise conv has no counterpart anywhere in vLLM, and the search is a
confirmed negative rather than an unfound one.** At `origin/main` = `6a5e8f5979`,
`git grep -in dilat` returns 17 lines tree-wide and **zero** in
`vllm/model_executor/layers/mamba/`, **zero** in `csrc/`, and **zero** in `tests/`.
`layers/conv.py` defines only `Conv2dLayer` and `Conv3dLayer`; there is no
`Conv1dLayer`, and the Transformers-backend auto-replacement maps only `nn.Conv2d` and
`nn.Conv3d`, leaving any `nn.Conv1d` as a bare PyTorch module. Upstream reached the
same conclusion from the other side and hand-rolled it, with the comment "We cannot use
the usual functions/kernels here for the short conv as the conv1d has dilation".

`causal_conv1d_fn` / `causal_conv1d_update` are disqualified on four independent
counts: they take no dilation argument; their Triton state loads unroll the taps at
unit stride in the kernel source; `state_len` is `width - 1` throughout the shape
plumbing where PLE needs `(width - 1) * dilation`; and vLLM has no `state_idx` concept,
so one layer cannot own three independently addressed conv states.

**The conv is strided history, not a local window.** `kernel_size=4`, `dilation=3`,
so output position `t` reads tokens at lags **{9, 6, 3, 0}** — a span of 10 tokens for
4 multiply-accumulates per channel, and the lag-0 tap makes it causal. The state is
therefore a genuine 9-deep ring buffer read at stride 3, and it cannot be compressed to
3 columns even though any single step touches only three of them.

Cost: 9 columns x 10240 channels = **~180 KiB per sequence at bf16 for this one
layer**. That is a real KV-budget line item, not a rounding error, and it belongs in
the `## Hardware` accounting once measured.

What the state holds is the **normed** conv input (`norm_conv`'s output), not the raw
hidden state and not the conv output. The layer forks: the skip term is the
**un-normed** `gated_value`, and only the normed copy enters the conv.

**The signed-sqrt gate has a trap in the clamp order.** It is
`gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`, so the clamp applies **before** the
square root and the floor on the output magnitude is `sqrt(1e-6) = 1e-3`, not `1e-6`.
Tiny scores are **amplified** to +/-1e-3 rather than squashed. Exactly zero maps to
zero, because `sign(0) = 0`, so the function is genuinely discontinuous at the origin
and that is reachable on a fully masked row. Mirror it; do not tidy it. A port that
clamps after the sqrt is wrong by three orders of magnitude in that band.

**A GDN state-length disagreement to reconcile at the seam.** Upstream sizes the GDN
conv state as `linear_conv_kernel_dim` = **4**, where vLLM's shape calculators use
`conv_kernel - 1`. The two conventions differ by one column, and nothing will announce
the mismatch.

**`ple_layer_ids` is one-indexed by design, not by accident.** The docstring says
"One-indexed", the config validator rejects ids outside `[1, num_hidden_layers]` and
resolves the layer type as `layer_types[layer_id - 1]`, and
`test_ple_layers_must_use_linear_attention` pins it. Do not "fix" it.

Padding is a paired obligation: the activations are masked **and** `ple_input_ids` has
its padded positions overwritten with EOS before the layer runs, because the n-gram
hash reads token ids rather than activations. Masking only the activations leaks
padding into the hash. `conv_mask` is `None` in steady-state decode, so the masking
lines are prefill-only.

One item is **AMBIGUOUS and must not be resolved from upstream**: the conv state
written during a chunked prefill whose first chunk is shorter than 9. Upstream
zero-pads on the left, which is arithmetically identical to what a single-shot prefill
would do, and its cache never reuses a prefix from another sequence. A prefix-caching
scheduler has to decide whether a cache hit restores the true 9-column state or re-pads
with zeros. That is our design question, not upstream's.

### Gated Residual: what our MHC actually gives us

The reuse verdict is sharper than "same shape, different math". Buffer plumbing is
largely reusable: the `[T, hc, H]` manifold, the layer-0 broadcast widen (upstream's
`hidden_states.repeat(1, 1, hc_count)` is exactly our broadcast), the per-token loop,
and the read/collapse/write-back cadence twice per layer. Three specifics:

- **`MhcPost` is bit-exact reusable for Qwen's write-back with the comb matrix set to
  identity.** The sum collapses to one non-zero term, so there is no reduction-order
  difference. It is 3x wasteful on the residual read and must not ship that way, but
  it gives a bring-up bridge from a kernel that is already gated, which is a free
  mutation target for the fresh reviewer.
- **`MhcSinkhorn` is dead here.** Qwen has no doubly-stochastic mixing and no comb
  matrix at all, so the carried `res_mix [T, hc, hc]` buffer goes with it. Qwen's
  streams couple only on the READ path, through the shared low-rank projection.
- `MhcPre` and `HcHeadCollapse` share a skeleton and no arithmetic: their norm is
  weight-free and global where Qwen's is grouped and weighted, their projection is one
  dense matrix where Qwen's is a two-stage low-rank `10240 -> 320 -> 10240`, their gate
  is per-stream scalar where Qwen's is per-element, and their reduce is a **sum** where
  Qwen's is a **mean**. Qwen also needs no separate head-collapse op: the final
  collapse is the same class with the injection branch switched off.

**`Qwen4ExpTextModel` has no final RMSNorm.** The mixer's own `hc_norm` is the last
normalization before `lm_head`. A port that copies our DeepSeek-V4 tail will insert one
that does not exist. Stated because that tail is the natural thing to copy.

**Weight parameterization differs from vLLM's op form.** Upstream applies
`output * (1.0 + weight)` with `weight` zero-initialized; vLLM's grouped norm applies
`out * weight` with `weight` ones-initialized. They coincide under a load-time
`w_vllm = 1.0 + w_hf`. Miss it and every `hc_norm` gets a near-zero scale, which reads
as a checkpoint bug rather than a port bug.

**Correction to the port map above.** vLLM's grouped RMSNorm is on **`RMSNormGated`**,
not the plain `RMSNorm`, whose only related knob is `var_hidden_size` -- a prefix
reduction that cannot express per-group norms. Verified directly: `RMSNorm` opens at
`layernorm.py:37` and `RMSNormGated` at `:172`, and the `group_size` parameter is at
`:187`. A porter reaching for `RMSNorm` finds nothing. Separately, `RMSNormGated`'s
`forward_cuda` dispatches to a flash-linear-attention Triton kernel rather than the
native reference, and that kernel's grouped numerics are **unverified**; whoever writes
the device arm owes that check.

The hyper-connection tower is **~640 M dense parameters** at this config (two modules
per layer x 48, plus the mixer), unquantized in the published scheme and read twice per
layer. That is a memory and bandwidth line item, not only a correctness one.

## Dependencies

Shared seams this row must route through rather than around, per AGENTS.md
"Shared seams". Each is named so a reviewer can check the routing instead of
inferring it.

- `ModelRegistry::Forward` and `dense_attn::AttnBlock` for decode.
- `vt::FusedChain` for model fusion; `layers::MlpGateUpMethodBase` and
  `vt::MergedGemmGroup` for the mergeable MLP projections.
- `include/vllm.h` for every shipped capability. Examples and servers stay thin
  ABI clients and never include an internal header.
- `vllm::HfConfigFromGguf` and the `qwen3_5` GGUF builder, which currently
  hard-asserts its own architecture and will refuse `qwen4_exp` by name until this
  row extends it.
- `src/vllm/v1/kv_cache_interface.*` for the third conv stream and the indexer side
  cache. This is the seam [#1963](https://github.com/mudler/vllm.cpp/issues/1963)
  and [#1966](https://github.com/mudler/vllm.cpp/issues/1966) are moving; coordinate
  rather than fork.

New files go beside their vLLM counterparts and mirror the upstream file structure,
per AGENTS.md. Where the upstream counterpart is MiniMax-M3 rather than a Qwen file,
mirror the op's home and say so in the file header.

## Work breakdown

Waves are separable and each is independently reviewable. Every wave lands reachable
from a production entry point, or names what is unreached with its owning row and
issue per AGENTS.md "Nothing lands dead".

- **W0, this pull request.** Spec, records, oracle exception proposal. No product
  code.
- **W1, config and registration.** `qwen4_exp` config resolution including the
  `full_attention` -> `qwen_sparse_attention` rewrite upstream performs in
  `__post_init__`, every `validate_architecture` rejection, and a refusal naming any
  unimplemented arm. Reachable through the loader.
- **W2, the two components with no vLLM op.** N-gram hashed embedding and the PLE
  layer with its dilated depthwise conv. Gated against transformers goldens on
  integer equality for the ID construction. First because they are the highest
  silent-wrongness risk and because they are independent of the attention work.
- **W3, gated residual.** The 10240-wide stream through the per-layer loop, both
  `use_combine` arms, and the final mixer.
- **W4, QSA.** Indexer side cache and KV spec, pooled-key build, block scoring and
  top-k, block-sparse consumer. Mirrors MiniMax-M3's op shape.
- **W5, assembly and the load plan.** Full model forward, vision path, MTP.
- **W6, the first runnable arm**, and the row's real unblock. Split by the blocker
  analysis above rather than by guesswork. **W6a** authors the `qwen4_exp` GGUF
  architecture -- one dispatch row plus its own config builder TU, never reusing
  `HfConfigFromGguf`, which asserts its own architecture by name -- and emits Q4_0 on
  every K=640 / K=320 reduction dim so the file can be opened at all. **W6b** is Route A:
  F16 table, mmap borrow, prefault off, CPU device, producing the token baseline.
  **W6c** is Route B: the dequantizing gather op plus the `kEmbeddingTable` keep-quant
  policy change, in that order, which is what makes the arm the developer actually chose
  reachable on CUDA.

Waves W2 through W4 have no ordering dependency on each other and can be dispatched
in parallel to separate worktrees. W5 is a barrier.

## Hardware

Usable budget on GB10 is about 119 GB. Read live from the HF API, 2026-08-26:

| Artifact | On disk | Verdict |
|---|---|---|
| `Qwen/Qwen3.8-Flash-Next` BF16 | ~360 GB (`BF16 = 179,999,981,424` params) | no |
| `Qwen/Qwen3.8-Flash-Next-FP8` (official) | ~180 GB | no |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | ~128 GB; NVFP4 backbone with the n-gram table kept at **FP8, 51.2 GB** | no, over budget before KV |
| `unsloth/Qwen3.8-Flash-Next-GGUF` | **README only, zero weight files** | does not exist |

No GGUF exists and no existing tool can produce one, because llama.cpp has no
`qwen4_exp` architecture either. Per AGENTS.md the quantized arms are a standing
requirement, so this row owes them and owes authoring the arch on our side.

**The architecture hands us the lever.** Its card argues n-gram embedding is "more
amenable to offloading than MoE", and the arithmetic agrees: the per-token cost is
`(ngram_size - 1) * heads_per_ngram` = 16 lookups of `ple_embed_dim / ngram_heads` =
160 dims. **51.2B of the 180B parameters, 28% of the model, is a table touched 16 times per
token** (51.2 GB at FP8, 102.4 GB at bf16, ~31 GB at Q4_K_M). Making it non-resident is the intended design point. RadixArk reached the
same split independently.

| Arm | Backbone (125B) | N-gram (51B) | Resident | Fits |
|---|---|---|---|---|
| Q8_0 throughout | ~133 GB | ~54 GB | ~191 GB | no |
| Q4_K_M throughout | ~76 GB | ~31 GB | ~109 GB | yes, ~10 GB for KV and activations |
| **Q4_K_M backbone, n-gram table non-resident** | ~76 GB | 0 | **~76 GB** | yes, with room |

**Developer decision, 2026-08-26: the first runnable arm is the third row — a
Q4_K_M backbone with the n-gram table non-resident.** Q8_0 was raised and does not
fit: at ~191 GB it exceeds the budget by a wider margin than BF16 exceeds it on a
box half this size, and no partial-Q8 split reaches 119 GB while keeping the
backbone at 8 bits. Q4_K_M-throughout fits on paper at ~109 GB but leaves about
10 GB for KV and activations on a model whose native context is 262144, which is
not a margin. The chosen arm is also the only one that matches what the
architecture was built for, so the offload is a design point rather than a
concession.

This promotes the non-resident table from a note to a **first-class deliverable**
of W6. It is not free: GB10 is unified memory, so the existing host-pinned offload
seam (`ENG-WEIGHT-OFFLOAD`, mirroring vLLM's `cpu_offload_gb`) does not by itself
solve this there, and the mechanism has to be disk-backed or genuinely unloaded.
Establish that before designing around it.

These are sizing estimates from published parameter counts, not measurements. They
decide which arm to attempt first and nothing else. GB10 is **unified** memory, so
"offload to host" is not a move there; non-resident means disk-backed and page-cached,
and its cost is unmeasured. Establish it before it is designed around.

### The chosen arm has a hard blocker, and it is not the offload

Verified in this tree, 2026-08-26. The developer chose the Q4_K_M backbone with a
non-resident n-gram table, and that arm **does not load today**. The reason is not the
offload machinery and not the memory budget.

**This tree cannot keep a gather table quantized, by construction.** `KeepQuantKDim`
returns `-1` for `GgufTensorRole::kEmbeddingTable`, so the keep-quant branch is
unreachable for a gather table regardless of shape or encoding, and the qwen3_5 loader
asserts it by name: "the embedding table cannot keep quant blocks". A Q4_K or Q8_0
n-gram table therefore **expands to bf16 at load: 51.2B params become 102.4 GB of
anonymous memory** on a ~119 GiB box. The arm dies before the first forward. The reason
is already recorded in a header comment upstream of both -- "a gather, not a GEMM ... A
quantized-gather op is a follow-up row" -- and **no such row exists**. That sentence is
the whole blocker and it has been sitting in a comment.

The only non-expanding residency for a gather table is `kKeepF16`, which requires the
file to store ggml type **1 (F16) exactly**. That makes the table 102.4 GB on disk, and
it is **CPU-only**, because `EmbeddingKernelCuda` refuses anything but f32/bf16.

**Second blocker, cheap to avoid because we author the converter.**
`moe_intermediate_size = 640` makes `ffn_down_exps` Q4_K-illegal on its reduction dim
(`640 % 256 = 128`), and `hc_lowrank = 320` is the same class. llama.cpp's substitution
for a ragged-K Q4_K tensor is believed to be Q5_0 -- **flagged as UNVERIFIED, and owed
a check against the pinned llama.cpp oracle before it becomes an assertion.** The
dependent fact IS verified in-tree and is the one that bites: this repository's GGUF
reader knows ggml type ids `0,1,2,8,10..14,16,18,19,22..28,30,39,40,41,66` and **has no
entry for 3 (Q4_1), 6 (Q5_0), 7 (Q5_1) or 20 (IQ4_NL)**, so such a file fails at header
parse with "unknown ggml type id". A stock `llama-quantize -Q4_K_M` output for this
model would not open at all. The fix is ours: emit **Q4_0** on every K=640 and K=320
reduction dim -- block 32, the same 4.5 bpw as Q4_K, and keep-quant capable.

**`ENG-WEIGHT-OFFLOAD` will not help, now or later.** It moves zero bytes today
(`ConsiderWeight` has no production callers, `supports_weight_offload` is false
everywhere and a test pins that), and it is separately documented inert on GB10 because
it moves bytes inside one physical pool. **Do not budget for it.**

**The tier that does work already ships**, and the 2.4T model is the proof: mmap the
GGUF `MAP_PRIVATE`, borrow tensors in place, and alias the host pointer into the kernel
on a host-addressable device. That serves 369.97 GiB from a 119.631 GiB box at ~62 GiB
resident. Set `vllm_cpp.mmap.prefault: false`, or `PrefaultBorrowedSpan` touches every
page of the table at load and OOM-reboots the box.

**Two routes, and the recommendation is to do both in order.**

- **Route A, runs on today's code, CPU only.** F16 n-gram table, Q4_0 on the ragged
  reduction dims, Q4_K elsewhere, mmap borrow with prefault off. Delivers a correct
  first run and the token baseline Route B needs. No shared-kernel changes.
- **Route B, the arm actually chosen.** Add a dequantizing gather to `vt::Embedding`
  across CPU and CUDA, then make `kEmbeddingTable` keep-quant eligible gated on that
  op's availability. Order matters: the assertion above is CORRECT today and only
  becomes wrong once the op lands. Then the table is Q4_K at **28.8 GB** on disk,
  borrowed, device-aliased, gathered on device -- smaller than the 51 GB the arm was
  scoped at. Roughly 400 lines.

**Corrected sizing.** Backbone ~67.7 GiB resident in the expected arm; whole process
~73.5 GiB of 119.631 at 32K context single stream, leaving ~46 GiB of headroom that is
exactly what pays for the table's page cache. The original ~76 GB estimate was right to
within 10%. The design works because the per-token demand is tiny: 16 lookups x 160
dims x 2 B = 5120 B/token over at most 16 distinct pages, so **<= 64 KiB of reads per
token**, against the 2.4T expert lane's 6.95 GB/token. That contrast is the whole
argument for this arm, and it is why the table is offloadable where MoE experts are not.

Two further hazards, both with escapes: the `--device cuda` load-time device-fit
refusal counts every tensor in the file including the table, and a misaligned mmap
borrow is silently STAGED into device memory with a full `Alloc` -- pad the n-gram
tensor's data offset to 256 in our writer, since `kDeviceAliasAlignment` is 256 while
GGUF guarantees only 32.

Finally, **there is no GGUF writer in this repository.** Authoring the conversion means
authoring it outside this tree; what this repo controls is only what it will accept.
And a latent trap for exactly that writer: the parse-time and dequant-time divisibility
checks test `numel % block_elems`, not `K % block_elems`, so a hand-rolled ragged-K
K-quant tensor decodes across row boundaries into structurally wrong values with **no
error**. Assert K-divisibility in the converter.

## Risks

- **Porting the eager reference as written.** The stated risk of the oracle split.
  The QSA indexer and the n-gram ID construction are both written as scalar Python
  in transformers. A reviewer should mutate for this: an implementation whose QSA
  path has no block-level kernel is a correctness result, not a port.
- **Reaching for DSA.** This tree has a working DSA indexer, and it is the wrong
  base. See above.
- **Silent n-gram mis-indexing.** See above.
- **Sizing estimates hardening into measurements.** The `## Hardware` table is
  arithmetic on published counts. `.agents/` already records this failure mode
  (a quoted number becoming a measured one); do not let the 76 GB row be cited as
  an observation.
- **A GGUF arm with no oracle.** llama.cpp does not implement `qwen4_exp`, so the
  usual quant-arm cross-check against a quant-matched llama.cpp does not exist. A
  k-quant arm here can only be gated against our own higher-precision path, which
  is a weaker gate, and the spec must say so rather than imply parity.
- **`transformers_version: 5.8.0.dev0`** in the published config is older than both
  our pin and the branch that merged `Qwen4Exp`. It records the branch the config
  was authored on and is not a usable pin. Do not resolve the oracle from it.

## Tests to port

AGENTS.md requires the upstream tests in the same change, preserving parameters,
modes, fixtures, tolerances, failure cases and the revision anchor. The upstream
suite is `tests/models/qwen4_exp/test_modeling_qwen4_exp.py` at transformers #48337,
707 lines, two classes. Inventory, read live 2026-08-26:

**`Qwen4ExpTextModelTest`** (`Qwen4ExpTextModelTester`, a `CausalLMModelTester`):

| Upstream case | Ports to | Note |
|---|---|---|
| `test_ple_layers_must_use_linear_attention` | W1 | a config invariant; cheap and load-bearing |
| `test_ple_padding_and_static_cache_match_unpadded_sequence` | W2 | the padding/EOS-segment semantics of the n-gram history |
| `test_all_layer_types_cached_forward_match_full_forward` | W4/W5 | cached vs full forward across BOTH layer types; this is the incremental-decode gate |
| `test_ple_beam_generation` | W5 | PLE under beam search, where the conv and n-gram states must follow the beam |
| `test_ple_sharded_checkpoint_loads_and_forwards` | W5 | 131 shards here, so sharded load is not optional |
| `test_generate_with_ple_and_inputs_embeds` | W5 | drives `reverse_embedding`, the inputs-embeds path |
| `test_reverse_loading_mapping` | W1 | weight-name mapping both directions |
| `test_attention_outputs`, `test_hidden_states_output` | W3/W4 | both are OVERRIDDEN upstream because the hyper-connection stream changes the shapes; port the override, not the base |
| `test_tp_plan_matches_params` | not ported | tensor-parallel plan; no TP surface in this row |
| `test_generate_compile_model_forward_fullgraph`, `test_generate_compilation_all_outputs`, `test_multi_gpu_data_parallel_forward`, `test_generate_with_quant_cache` | not ported | torch.compile / multi-GPU / torch quant-cache harness, no counterpart here |

**`Qwen4ExpCompositeModelTest`** (`Qwen4ExpVisionText2TextModelTester`, a
`VLMModelTester`): `test_mismatching_num_image_tokens`, `test_video_forward`,
`test_composite_checkpoint_loads_as_causal_lm`,
`test_base_model_checkpoint_loads_as_conditional_generation`,
`test_generate_with_ple_and_inputs_embeds`, plus its own `test_attention_outputs` /
`test_hidden_states_output` overrides. All port to W5. The remaining cases in that
class are the same harness-only skips as above.

Adaptations must be documented per AGENTS.md, and only where genuinely unavoidable.
"Our harness differs" is not one; "upstream asserts against a `torch.compile`
fullgraph we do not have" is.

### Local red-first tests

Red-first, smallest failing test per slice, each entering through a production entry
point per AGENTS.md "Nothing lands dead". A unit test that constructs the type by
hand does not discharge this.

1. N-gram ID construction against transformers goldens: prime head vocab sizes, the
   splitmix64 multipliers, the shift-and-XOR mix, EOS segment handling. Integer
   equality, no tolerance.
2. Grouped RMSNorm against vLLM's `group_size` form.
3. Gated Residual forward against transformers, both `use_combine` arms.
4. QSA block selection: selected token index sets equal to transformers on the same
   inputs, including the ragged tail beyond the last complete block.
5. PLE layer end to end, including the dilated depthwise conv and its state.
6. Config resolution: the `full_attention` -> `qwen_sparse_attention` rewrite that
   upstream `__post_init__` performs, and every rejection in `validate_architecture`.
7. Loader coverage against the published index, with the refusal path naming any
   unimplemented arm.
8. Inertness: existing Qwen3.5/3.6/3.8 goldens byte-identical.

## Gates

No token gate is claimable until an arm runs. In order:

1. **G0, component goldens.** Tests 1-6 above against transformers at the lane pin.
   This is the only gate reachable today, and it is reachable without the weights.
2. **G1, load plan.** Every published tensor accounted against a committed manifest,
   per arm, with refusals naming what is missing.
3. **G2, token-exact greedy** with **at least one prompt past `indexer_budget` = 2048
   tokens of context**, because below that QSA selects every candidate and the gate
   cannot distinguish a correct implementation from one attending pooled keys. Vs
   transformers at the lane pin, on whichever arm
   `## Hardware` makes runnable first. Strict token equality; the near-tie
   distributional doctrine applies only if the oracle's greedy decode is shown
   non-deterministic, which is not assumed here.
4. **G3, quantized arms.** Per arm, with the lower-bound requirement this repository
   places on quantized gates, and with the missing-llama.cpp-oracle limitation stated
   in the result rather than omitted.
5. **Speed: nothing.** No throughput, latency or memory number is admissible from
   this row until G2 passes. There is no vLLM denominator for this model, so when a
   speed axis does open, the spec must first say what the denominator is.

## Evidence required

Per gate: the exact build and run recipe, the lane transformers revision, the
checkpoint repo **and revision** plus sha256 for any quantized artifact, the device,
and the contention state. `docs/USAGE.md` gains the checkpoint pins in the same
change that makes any arm reachable, not later.

## Stop conditions

- vLLM registers `qwen4_exp`: **stop and reconcile onto vLLM** before continuing.
  This is the designed end of the transformers exception.
- SGLang #36497 merges: re-survey the op mapping; it does not displace vLLM.
- The transformers lane pin is rejected in review: the row holds at `READY` and the
  gate stays `PENDING`. Do not proceed on an unpinned oracle.
- No arm is made to fit any fleet device: the row holds with G0 passed and G1-G3
  `PENDING` on hardware, recorded as visible debt, and no token claim is made.

## Owed

- [#1978](https://github.com/mudler/vllm.cpp/issues/1978): this port. No product
  code lands under the spec pull request.
- GGUF k-quant arms, including authoring the `qwen4_exp` architecture on our side,
  and the statement that no llama.cpp oracle exists for them.
- MTP depth > 1.
- The 1M-token RoPE extension above the native 262144.
- The non-resident n-gram table on CUDA: the dequantizing gather op and the
  `kEmbeddingTable` keep-quant policy change (Route B), and a measurement of the
  page-cache cost that the <= 64 KiB/token arithmetic only bounds.
- **UNVERIFIED and owed a check against the pinned llama.cpp oracle:** llama.cpp's exact
  substitution for a ragged-K Q4_K tensor, asserted here as Q5_0.
- A K-divisibility assertion in whatever writes our GGUF files.
- A speed denominator, once one exists.

## Now

`READY`. Spec committed, no implementation.

Both decisions this spec was blocked on are **settled** (developer, 2026-08-26) and
recorded in place rather than left as proposals: the transformers lane pin is
ACCEPTED at 5.16.0 (`## Oracles`), and the first runnable arm is the Q4_K_M backbone
with a non-resident n-gram table (`## Hardware`).

Next actions, in order: W0 lands this spec; W1 through W3 are reachable today
against the lane pin with tiny random configs and need neither a checkpoint nor a
GPU lease; W6b's mechanism is the unknown that decides whether the chosen arm is
schedulable, and it should be spiked before W6 is planned.
