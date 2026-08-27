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

### Merge sequencing for the `ACTIVE` transition and its claim (operator note)

W1 and W6a BOTH moved this row `READY -> ACTIVE` on their own branches, independently
and correctly — AGENTS.md "Records" requires the matrix row to move with the lifecycle
state, and each wave was the first product code from its own point of view. The result
is a collision that a clean three-way merge will NOT catch, and it is recorded here
because the second merge is where it bites:

- **The counts happen to be safe.** Both branches make the IDENTICAL edit, `ACTIVE`
  10 -> 11 and `READY` 4 -> 3, so a three-way merge with a base of 10/4 and both sides
  at 11/3 resolves to 11/3. That is luck, not design: two branches making DIFFERENT
  one-line edits to the same counter merge cleanly and apply BOTH, which is the failure
  AGENTS.md names under "Never store a measurement of one file inside another file".
  **Verify these two numbers by COUNTING ROWS at every merge, never by trusting the
  merge.**
- **The claim owner is NOT safe.** W1 wrote owner `CLAIM-MODEL-MM-QWEN4-EXP-W1` with
  `.agents/claims/CLAIM-MODEL-MM-QWEN4-EXP-W1.md`; W6a wrote `CLAIM-MODEL-MM-QWEN4-EXP`
  with its own file. Two different owners for one cell, and two claim files for one row.

**Resolution: the row-level claim `CLAIM-MODEL-MM-QWEN4-EXP` wins**, because the claim
covers the whole campaign rather than one wave, and `check-agent-record.py` binds an
owner to a ROW. Whichever of W1/W6a merges second drops its own transition and its own
claim file, keeping only the survivor. This is a merge-time reconciliation, not a
defect in either branch.

The same shape will recur for W2, W3 and W4: each is the first product code from its
own vantage, none of them should re-make the transition, and each should drop the edit
if it finds the row already `ACTIVE` on `main`.

## Why this needs a spec before code

Three of this row's decisions are expensive to reverse and cheap to get wrong, and
all three have already been made incorrectly once by an agent reading a related
record. They are settled here so a fresh implementer does not re-derive them.

1. **This is not a Qwen3.8 row.** `.agents/specs/qwen38-27b-bf16-gate.md` records
   `Qwen/Qwen3.8-27B` as the Qwen3.6-27B shape retrained, differing in exactly one
   config key. That precedent does not extend here. `qwen4_exp` shares an ancestor
   with `qwen3_5` and diverges in four load-bearing places.
2. **QSA's twin in vLLM is DeepSeek-V4's C4 indexer lane, not MiniMax-M3.** See
   `## Design`. This REVERSES the row's first reading, which rested on treating
   `MLAAttentionSpec` as an MLA claim; it is a per-state BUDGET shape, and M3 —
   itself plain GQA — uses it too. Nine independent structural matches tie QSA to
   DeepSeek-V4, `compress_ratio == 4` literally the same number. Building QSA on M3
   is the wrong port and it fails hard rather than subtly: M3 welds
   `SPARSE_BLOCK_SIZE = 128` to the KV page size, so ratio 4 forces a page size of 4
   and breaks `tl.dot`, whose tile needs >= 16. What M3 does contribute is a wiring
   precedent and not an algorithm: a plain-GQA model owning a key-only side cache
   through `MLAAttentionSpec`. The DSA/MLA reflex remains the trap, because this tree
   already has that path — the correction is which side of it QSA sits on
   ([#2049](https://github.com/mudler/vllm.cpp/issues/2049)).
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

**The GGUF converter already folds it, and it folds far more than `hc_norm`.** Read at
source rather than relayed, because W5 writes the loader and the narrow version of this
sentence causes the defect it warns about. Every anchor below is read at our recorded
llama.cpp pin, stock upstream tag `b10451` (`10bf611e533d81f739128304991c5e133c6aebd8`,
[`../oracles/llama-cpp.md`](../oracles/llama-cpp.md)). Stock upstream has no `qwen4exp`
at all there (`git grep -il qwen4exp`: nothing tree-wide, so a released llama.cpp can
neither convert nor load this architecture). The converter is ggml-org/llama.cpp
[#27742](https://github.com/ggml-org/llama.cpp/pull/27742), head
`035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`, **still OPEN**. Its `conversion/qwen4exp.py`
declares `class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase)`;
`_LinearAttentionVReorderBase` is `conversion/qwen.py:438`, a subclass of
`Qwen3NextModel` (`:365`, whose own signature is
`class Qwen3NextModel(_QwenMtpMixin, Qwen2MoeModel)`); and the PR's `modify_tensors` has
**no `hc_norm` branch**, so `hc_norm.weight` falls through to `super()`. The `+1` is the
inherited Qwen3-Next rule at `conversion/qwen.py:387-388`:

```python
elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
    data_torch = data_torch + 1
```

So the rule a loader implements is **every `*norm.weight` carries the fold, with
`linear_attn.norm.weight` (the GDN `ssm_norm`) the one exception** -- `hc_norm`,
`attn_q_norm` and `attn_k_norm` all match it, and the PLE and indexer gammas are folded
by the PR's own early-returning branch. A loader that skips the fold for `hc_norm` alone
double-folds everything else, which is the same silent ~2x defect one tensor to the left.
Two consequences for W5. The property belongs to one in-flight converter, not to "GGUF":
#27742 can change before it merges and another publisher's tool need not match it, so the
loader treats the fold as a provenance question and checks it -- cheaply, since an
unfolded `hc_norm` is a zero-init gamma and a folded one is centred on 1.0. And it was
corroborated on published artifacts during fresh review of #1988
(`unsloth/Qwen3.8-Flash-Next-GGUF` `UD-IQ1_S` and `UD-Q4_K_XL`, `vumpt/...-Q4_K_M`, read
by HTTP range request against the bf16 HF tensors): every `*hc_norm.weight` is HF + 1.0
exactly, elementwise, while `ssm_norm` is unfolded and sits in [0.875, 1.023].

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
for a ragged-K Q4_K tensor is **Q5_0, now VERIFIED** and no longer owed: the
`tensor_type_fallback` table in `src/llama-quant.cpp` maps `Q4_K -> Q5_0`,
`Q5_K -> Q5_1`, `Q6_K -> Q8_0`, `Q2_K/Q3_K/TQ* -> Q4_0` and every `IQ*` including
`IQ4_XS -> IQ4_NL`, then falls to `F16` if the result still does not divide. So the
answer depends on the RECIPE, which is why the shipped `unsloth` UD-IQ1_S file shows
`IQ4_NL` on `ffn_down_exps` rather than Q5_0 -- it asked for an IQ type, not Q4_K. A
`-Q4_K_M` build of this model would land on Q5_0, and on Q8_0 wherever `use_more_bits`
promotes `ffn_down` to Q6_K. The
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

## Mutation record — W6a (#1989)

Committed because the first fresh review could not re-run W6a's claimed
mutations: no table for the wave existed anywhere in the tree, so the reviewer
designed and ran fourteen of their own. This section is the reproducible list.
Every row is one textual change applied to a pristine tree, rebuilt, run,
restored, and rebuilt again with the source `touch`ed after restore — without
that touch ninja skips the rebuild and the mutations ACCUMULATE, which fails
toward RED and makes a weak gate read strong.

Reviewer battery (14, at `beedfdf31`; R8b and R11-R13 are what the review's
findings F2 and F7 are made of):

| # | mutation | target(s) | result |
|---|---|---|---|
| R1 | `kValuesIq4nl[8]` `1` -> `0` | dequant, embedding | RED, RED |
| R2 | `DequantQ5_0` upper-half `qh` shift `j+12` -> `j+16` | dequant, embedding | RED, RED |
| R3 | `DequantIQ4_NL` swap the two nibble halves | dequant, embedding | RED, RED |
| R4 | reader `GgmlTypeTraits` IQ4_NL `block_bytes` 18 -> 17 | load_plan, traits | RED, RED |
| R5 | `vt` `BlockGeometry` Q5_0 `block_bytes` 22 -> 21 | traits | RED |
| R6 | delete the block arm of `EmbeddingKernel` | embedding, qwen36_loader | RED, RED |
| R7 | `KeepQuantKDim(kEmbeddingTable)` back to `-1` | keep_quant, qwen36_loader, load_plan | RED x3 |
| R8a | `DeviceQuantGatherSupported` INVERTED | keep_quant | RED |
| R8b | `DeviceQuantGatherSupported` widened to every device but ROCm | keep_quant | SURVIVED — only the CPU branch is reachable on a CPU host, the same limitation `DeviceKeepQuantSupported` already has |
| R9 | remove the NVFP4 `role != kEmbeddingTable` exclusion | keep_quant | RED |
| R10 | delete the `kGgufArchArms` `qwen4exp` row | model_loader_gguf | RED |
| R11 | neuter `vt::Embedding`'s whole-block precondition | embedding | SURVIVED at `beedfdf31` -> **RED after the F7 repair** |
| R12 | `VecDotIQ4_NLQ8_0`: swap the two nibble halves | all 8 suites | SURVIVED x8 at `beedfdf31` -> **RED after the F2 repair** |
| R13 | `VecDotQ5_0Q8_0`: upper-half `qh` shift `j+12` -> `j+16` | all 8 suites | SURVIVED x8 at `beedfdf31` -> **RED after the F2 repair** |

Repair battery (this change; each restored byte-identically and re-verified
green afterwards):

| # | mutation | target(s) | result |
|---|---|---|---|
| R11 | neuter `vt::Embedding`'s whole-block precondition (`% BlockElems` -> `% 1`) | `test_ops_embedding_quant` | RED |
| R12 | `VecDotIQ4_NLQ8_0`: swap the two nibble halves | `test_ops_quant_dot` | RED |
| R13 | `VecDotQ5_0Q8_0`: `>> (j + 12)` -> `>> (j + 16)` | `test_ops_quant_dot` | RED |
| R14 | `NoKeepQuant` made a no-op (the F1 defect, restored) | `test_deepseek_v4_gguf_load`, `test_laguna_gguf_load` | RED, RED |
| R15 | delete the block arm's per-id bounds check (`id % v`) | `test_ops_embedding_quant` | RED |
| R16 | `ResidentWeight`'s CPU alias offset by one byte | `test_gguf_qwen36_loader` | RED |

Anchor repairs in W6a: **three**, not nine. Measured with the repository's own
checker on both trees — parent `ok=876, stale=31, broken=6 -> rot 37`; head
`ok=879, stale=28, broken=6 -> rot 34`. The three are
`KERNEL-ATTN-DFLASH-BLOCK -> cpu_ops.cpp`, `SPEC-DFLASH-GGUF -> :773 -> :1015`
and `SPEC-MTP-GGUF -> :971 -> :1425`. All three were stale BEFORE W6a. The
DFlash one landed with a label that disagreed with its own href and is corrected
here.

## Mutation record — W5a (#2031)

The W5a pull request claimed a "14/14 red" battery. **That claim was
inaccurate and is not repeated here.** The fresh review of `a68312c79` re-ran
it, found two survivors, and this section is the honest list — every survivor
kept in the table rather than dropped, because a mutation table whose only rows
are reds is a table nobody re-ran.

Method, unchanged from the W6a section above: one textual change applied to a
pristine tree, `touch`ed, rebuilt, run, restored, `sha256sum`-verified
byte-identical against the pre-mutation copy, `touch`ed again and rebuilt. Every
row below was measured on this branch by the W5a REPAIR, not relayed, and every
row of the repair battery was re-measured on the FINAL head rather than at the
point in the repair where its fix landed — the case COUNT moves as cases are
added, and a count carried forward from an earlier build is a number nobody
measured.

**Reviewer battery at `a68312c79` (the immutable head), reproduced by the
repair before changing anything:**

| # | mutation | target | result at `a68312c79` |
|---|---|---|---|
| M1 | delete the `LoadQwen4ExpFromGguf` call site in `qwen4_exp_registry.cpp` (return `Qwen4ExpWeights{}`) | `test_qwen4_exp_gguf_weights` | **PARTIAL** — 1 of 10 cases red (6 assertions). Only "a malformed file refuses BY NAME" reddened; the headline case "the production load_weights hook LOADS the file" stayed GREEN, because `REQUIRE_NOTHROW` + `model != nullptr` is satisfied by a stub |
| M5 | swap `g` and `t` in `qwen4_exp_weights.cpp`'s `ReorderVRows`/`ReorderVCols` | `test_qwen4_exp_gguf_weights` | RED, 2 of 10 cases, 41 assertions |
| M6 | the SAME swap in `qwen3_5_gguf_weights.cpp`'s copy | `test_gguf_qwen36_loader` 7/7 555, `test_model_loader_gguf` 7/7 23, `test_gguf_nvfp4` 14/14 2352, `test_gguf_keep_quant` 42/42 6340 | **SURVIVED x4** — the cause is the FIXTURES, not the loader, and the cause first recorded here was itself wrong. Measured in `tests/vllm/test_gguf_qwen36_loader.cpp`: the fixture DEFAULT is `num_k = 2, num_v = 2` (`:120`), written out as `ssm.group_count = 2, ssm.time_step_rank = 2` (`:149-150`), so `K = 2` and `R = 1` and the permutation is the IDENTITY — the reorder is inactive. The ONE case that reaches it at all, `TEST_CASE("LoadQwen3_5MoeFromGguf: V-head reorder when num_v != num_k")` (`:361`), sets `num_v = 4`, giving `K == R == 2`, where the permutation is its own INVERSE and the mutated loader emits byte-identical weights. So exactly one case exercises the reorder, at the one shape where forwards and backwards are indistinguishable. The other three suites declare no `ssm.group_count` at all, so no buffer passes through the reorder there. Owned by #2081 |
| MUT-C | delete `text["ple_embed_dim"] = ple_row * ngram_heads_gguf;` from `Qwen4ExpHfConfigFromGguf` | `test_qwen4_exp_gguf_weights` | **SURVIVED** — 10/10, 2938/2938. The fixture defined `kPleRow` as `kH / kNgramHeads`, so `ple_row * ngram_heads == hidden_size` BY CONSTRUCTION and the fallback was indistinguishable from the value |

**Repair battery (this change). Each restored byte-identically, sha256-verified,
and re-run green afterwards:**

| # | mutation | target | result AFTER the repair |
|---|---|---|---|
| M1 | delete the `LoadQwen4ExpFromGguf` call site | `test_qwen4_exp_gguf_weights` | **RED, 2 of 11 cases, 8 assertions** — the headline case now opens the handle with `ModelAs<Qwen4ExpLoadedModel>` and reads loaded tensor bytes, which a stub cannot produce |
| MUT-C | delete the `ple_embed_dim` line | `test_qwen4_exp_gguf_weights` | **RED, 9 of 11 cases, 7 assertions** — the fixture's `kPleRow` is independent of `kH`, so the total is not the `hidden_size` fallback and the PLE projections refuse by shape. Re-measured at `kPleRow = 96` by the W5a-3 repair: still RED, 9 of 11, 7 assertions of 2553 run |
| MUT-G1 | disable the new `DeviceQuantGatherSupported` guard (`if (false && ...)`) | `test_qwen4_exp_gguf_weights` | **RED, 1 of 11 cases, 8 assertions** — "a device with no block gather refuses BEFORE the load" |
| MUT-G2 | pin the production device ARGUMENT to `vt::DeviceType::kCPU` in `qwen4_exp_registry.cpp` | `test_qwen4_exp_gguf_weights` | **SURVIVED, and it cannot do otherwise on this host.** A CPU-only build registers no other platform, so `CurrentPlatform().device_type()` and the literal `kCPU` are the same value and no test can tell them apart. The GUARD is gated (MUT-G1); the ARGUMENT is not, and this row exists so nobody records that as coverage. Closing it needs a CUDA host |
| M5 | swap `g`/`t` in our reorder copy | `test_qwen4_exp_gguf_weights` | RED, 2 of 11 cases, 41 assertions |
| M6 | the same swap in the `qwen3_5` copy | the same four suites, re-run at THIS head | **SURVIVED x4, deliberately unfixed.** Filed as [#2081](https://github.com/mudler/vllm.cpp/issues/2081): re-shaping a shipped model's fixtures changes `qwen35`, `qwen35moe` and `qwen3next` coverage and is not this row's scope. The source comment and the `## Owed` entry that both claimed "gated on both sides" are corrected to say so |

**Two survivors stand at the end of this repair, and both are named rather than
closed:** M6 (owned by #2081) and MUT-G2 (owned by the CUDA gather arm under
[#2083](https://github.com/mudler/vllm.cpp/issues/2083), which needs a device
this repair did not have).

**W5a-3 battery (this change).** Three repairs: the missing `issue-index.md` row
for #2081, the residual `kPleRow`/`kH` coincidence the re-review found, and the
`Closes #2064` note below. Each mutation was applied to a pristine tree,
`touch`ed, rebuilt, run, restored from a byte-identical copy,
`sha256sum -c`-verified, rebuilt and re-run green.

| # | mutation | target | result |
|---|---|---|---|
| M6 | swap `g` and `t` in `qwen3_5_gguf_weights.cpp`'s `ReorderVRows`/`ReorderVCols` | the four suites, re-run at THIS head | **SURVIVED x4, re-measured not relayed** — `test_gguf_qwen36_loader` 7/7 555, `test_model_loader_gguf` 7/7 23, `test_gguf_nvfp4` 14/14 2352, `test_gguf_keep_quant` 42/42 6340, every count identical to the un-mutated baseline. This is the measurement the appended #2081 index row states |
| M5 | swap `g` and `t` in OUR `qwen4_exp_weights.cpp` copy | `test_qwen4_exp_gguf_weights` | **RED, 2 of 11 cases, 41 of 2970 assertions**, re-measured at this head. The pair M5/M6 is what makes "only one copy is gated" a measurement rather than a reading |
| MUT-D | `text["ple_embed_dim"] = ReqInt(gguf, p + "embedding_length") * ngram_heads_gguf` in `Qwen4ExpHfConfigFromGguf` — `hidden_size` where the per-head row width belongs | `test_qwen4_exp_gguf_weights` | **SURVIVED at `kPleRow = 64`** — 11/11, 2969/2969. `kH` is also 64, so `hidden_size * ngram_heads` and `ple_row * ngram_heads` were the same 128 and the wrong source was unobservable. This is the residual coincidence the re-review named, and the reason MUT-C alone did not close #2064 |
| MUT-D | the same mutation AFTER `kPleRow` moved to 96 | `test_qwen4_exp_gguf_weights` | **RED, 9 of 11 cases, 7 of 2553 assertions.** At 96 the correct total is 192, the `hidden_size` product is 128 and the bare `hidden_size` fallback is 64: all three distinct, so each wrong source refuses the file by shape |

96 is the smallest legal replacement. `head_dim_per_ngram() == kPleEmbedDim /
kNgramHeads == kPleRow` must stay a whole number of Q8_0 blocks, because the
n-gram table is the one gather this model keeps quantized, so `kPleRow` is a
multiple of 32; 32 is `kH / kNgramHeads` and 64 is `kH`, and 96 is the next one.
A third `static_assert` now pins `kPleEmbedDim != kH * kNgramHeads` beside the
two that were already there, and the suite carries a third `CHECK` on
`ple.embed_dim` so both wrong answers are visible to a reader, not only to a
mutation.

**#2064 closes when W5a lands, and the closing keyword lives in the pull request
body.** The wave fixed it in flow — MUT-C and both MUT-D legs are its
instruments — but the issue is still open, and this branch has no pull request
yet. Whoever opens it puts `Closes #2064` in the BODY, which is the landed commit
message here (`squash_merge_commit_message = PR_BODY`), so the merge closes the
issue. Do not close it by hand: a hand-closed issue leaves the commit that fixed
it unlinked, and that link is the only thing tying the fix to its record.

## Stop conditions

- vLLM registers `qwen4_exp`: **stop and reconcile onto vLLM** before continuing.
  This is the designed end of the transformers exception.
- SGLang #36497 merges: re-survey the op mapping; it does not displace vLLM.
- The transformers lane pin is rejected in review: the row holds at `READY` and the
  gate stays `PENDING`. Do not proceed on an unpinned oracle.
- No arm is made to fit any fleet device: the row holds with G0 passed and G1-G3
  `PENDING` on hardware, recorded as visible debt, and no token claim is made.

## The refusal boundary

W1's whole product is a boundary: which configs this port accepts and which it
refuses. This row has **no reachable token gate** (`## Gates`, `gateable = no`), so
nothing downstream will ever catch a wrong default by running the model — a
`partial_rotary_factor` read from the wrong place, an n-gram field defaulted to
zero, or a missing `eos_token_id` all produce a config that parses, resolves, and
is silently wrong for W2 and W4. The config layer is the last place any of it is
checkable, so the boundary is **measured** here rather than described.

**The oracle runs.** `transformers` 5.16.0 installs and imports without torch —
it says so itself ("only tokenizers, configuration and file/data utilities can be
used") — and `Qwen4ExpConfig.from_dict()` runs `Qwen4ExpTextConfig.__post_init__`
and `validate_architecture` in full. That makes the CONFIG layer of this row
gateable even though the MODEL layer is not, and it is the only layer of this row
that is. `gateable = no` in `oracles/transformers.md` still stands: it is a
statement about running the model, and nothing here runs one.

### Two-direction sweep

39 configs, each derived from the committed fixture, put through
`Qwen4ExpConfig.from_dict` on one side and `LoadHfConfig -> ModelRegistry::Resolve
-> factory->parse_config -> ParseQwen4ExpParams` on the other. **35 agree; 4
differ, and over these 39 all 4 are ours refusing what upstream accepts** — the
safe direction, since the reverse is what lets a bad checkpoint through.

**That is a claim about the measured set, and it is bounded on purpose.** An earlier
draft said "never the reverse" as an absolute, and a fresh re-review falsified it with
a fortieth case outside the sweep: `rope_parameters` carrying **`rope_dim = 64`
alongside `partial_rotary_factor = 1.0`**. Upstream ignores `rope_dim` entirely —
`validate_architecture` computes `int(self.head_dim * partial_rotary_factor)` = 256
unconditionally at `configuration_qwen4_exp.py:225-226` — and refuses, because
256 > `indexer_head_dim` 128. We take `rope_dim` in preference, following vLLM's
`get_rope` semantics in the shared reader (`hf_config.cpp:545-547`), and **ACCEPT at
`rotary_dim = 64`**, handing W4 a 64-of-256 slice. That is the same failure mode and
the same direction as the finding that failed this wave's first review, reached
through a different key.

It is narrow and it is not a defect in this model's code: `rope_dim` has **zero
occurrences** in `modeling_rope_utils.py` at v5.16.0, so no transformers path writes
or reads it and no published checkpoint carries it — the oracle tolerates the key and
ignores it. The divergence lives in the shared reader, which is deliberately mirroring
vLLM rather than transformers on that point.

It is recorded rather than repaired because the fix belongs to whoever reconciles the
shared reader's rope resolution, not to this row, and because the honest form of a
boundary claim in the row whose whole product is that boundary is either **true or
bounded**. Owed: either a `rope_dim` case in the sweep with the divergence stated, or
a shared-reader change that makes it moot.

Reproduce (transformers 5.16.0 in a venv; the probe links `build/libvllm.a` with
`-Wl,--whole-archive` so the model's self-registration survives):

| upstream verdict | ours | cases |
|---|---|---|
| ACCEPT | ACCEPT | baseline; `prf` top 1.0 / rope .25; `prf` top .25 / rope absent; `eos_token_id` null with PLE OFF; every n-gram default omitted; no `output_gate_type` with `hidden_act` silu; all five indexer keys erased; `layer_types` erased (interval synthesis) |
| REFUSE | REFUSE | `prf` absent everywhere; `prf` only in rope 1.0; `prf` top .25 / rope 1.0; `eos_token_id` null with PLE ON; `eos_token_id` `[]`; no `output_gate_type` with `hidden_act` gelu; `output_gate_type` swish; `output_gate_type` gelu; `ple_embed_dim` -2560; `ple_embed_dim` 2561; `hc_count` 1; `num_experts` 0; `num_experts_per_tok` 513; `moe_intermediate_size` 0; partial QSA group; `indexer_n_heads` 0; `indexer_kv_heads` 2; `indexer_budget` 2049; `sliding_attention`; `ple_layer_ids` [0]/[4]/[49]; `ngram_size` 1; `heads_per_ngram` 0; interval 0; short `layer_types`; `num_hidden_layers` 0 |
| ACCEPT | **REFUSE** | `hc_lowrank` 0; `ple_conv_kernel_size` 0; `mtp_num_hidden_layers` -1; `partial_rotary_factor` -0.25 |

### Each upstream rejection, and the line that implements it

`configuration_qwen4_exp.py` at `v5.16.0`; local lines in
`src/vllm/model_executor/models/qwen4_exp.cpp` unless stated.

| # | upstream | our implementation | exercised by |
|---|---|---|---|
| 1 | `:190-192` unsupported `layer_types` | `KindFromString` | "an unsupported layer type" |
| 2 | `:193-195` `output_gate_type or hidden_act` not in {sigmoid, silu} | the raw-text gate resolution, NOT `config.output_gate_type` | "[UP] an absent output_gate_type falls back to hidden_act", "[UP] an explicit output_gate_type outside {sigmoid, silu}" |
| 3 | `:196-197` `hc_count <= 1` | the `hc_count` refusal | "hc_count must exceed 1" |
| 4 | `:198-199` `num_experts <= 0` | the `num_experts` refusal | "[UP] num_experts must be positive" |
| 5 | `:200-204` `num_experts_per_tok` outside [1, num_experts] | the `num_experts_per_tok` refusal | "num_experts_per_tok above num_experts" |
| 6 | `:205-206` MoE intermediate sizes | the MoE-size refusal | "[UP] the MoE intermediate sizes must be positive" |
| 7 | `:216-218` partial QSA group | the `present != 5` refusal, naming the missing fields | "a partial QSA group names what is missing" |
| 8 | `:219-220` QSA values not positive | the QSA positivity refusal | "[UP] QSA values must be positive" |
| 9 | `:221-222` `indexer_kv_heads != 1` | the `kv_heads` refusal | "QSA requires exactly one indexer kv head" |
| 10 | `:223-224` `indexer_budget % indexer_compress_ratio` | the divisibility refusal | "the indexer budget must divide by the compress ratio" |
| 11 | `:225-231` `rotary_dim > indexer_head_dim` | the refusal, over `config.rotary_dim` from the SHARED reader | "absent everywhere: 1.0, rotary_dim 256, and upstream REFUSES", "top-level 0.25 does NOT rescue a rope dict that says 1.0" |
| 12 | `:235-239` `ngram_heads <= 0 or ple_embed_dim <= 0 or ple_embed_dim % ngram_heads` | split three ways so the message names the field: `ngram_size < 2`, `heads_per_ngram <= 0`, then `heads <= 0 \|\| embed_dim <= 0 \|\| embed_dim % heads` | "[LOCAL] ngram_size below 2", "[LOCAL] heads_per_ngram must be positive", "[UP] a NEGATIVE ple_embed_dim", "[UP] a ple_embed_dim that does not divide by the head count" |
| 13 | `:240-247` `ple_layer_ids` outside [1, num_hidden_layers] | the one-indexed range refusal | "a PLE id outside the one-indexed range" |
| 14 | `:248-255` PLE on a non-`linear_attention` layer | the layer-kind refusal | "a PLE id on a sparse-attention layer" |
| 15 | `:256-257` `eos_token_id` unset with PLE enabled | the `eos_token_id` refusal | "[UP] eos_token_id must be set when PLE is enabled", "[UP] an EMPTY eos_token_id list is refused too" |

`__post_init__` behaviors, which are not rejections but decide what the rejections
see: `full_attention -> qwen_sparse_attention` (`:180-184`), the interval synthesis
(`:174-179`), `ple_embed_dim` defaulting to `hidden_size` (`:168`),
`sorted(set(ple_layer_ids))` (`:167`), and `number_of_conv_states` (`:172`). Each
has its own case.

**Upstream's ORDER inside the PLE block is mirrored**, and deliberately: head count
and embedding width first, then the id range, then the layer kind, then EOS. A
config violating two at once has to report the one upstream reports, or a reader
comparing the two runtimes is sent to a different field.

### Every refusal is mutated ONE AT A TIME

A sweep is an accept/reject comparison; it does not say whether OUR TESTS would
notice a refusal going missing. So each of the 23 refusals in
`ParseQwen4ExpParams` was deleted individually — `if (<guard>) {` rewritten to
`if (false) {`, proved applied by a non-empty `git diff --stat`, rebuilt, run,
and restored by byte comparison. **All 23 red.** Before this change a single
mutation deleting 13 of them at once left the suite green.

Deleting them as a UNION is not equivalent and would have hidden two defects: the
first union mutation SIGFPE'd on `(i + 1) % 0` at the second subcase and never
reached the other eleven. Run one at a time, two of the new subcases turned out
to be weak — `num_hidden_layers = 0` asserted the bare field name, which the next
refusal down ("`layer_types` has 48 entries but `num_hidden_layers` is 0") also
prints, and `num_experts = 0` the same against the `num_experts_per_tok` range
message. Both now assert the distinguishing text. That is the general shape:
**a substring assertion is a weak gate wherever two refusals share a word**, and
only a per-guard mutation finds it.

The three production entry points were mutated too. Gutting the registered
`parse_config` hook to `(void)config;` reds 3 cases / 42 assertions; removing the
forward's `VT_CHECK` reds 5 assertions; removing the GGUF arm's throw reds 4.
Before this change all three were green.

### Refusals we impose that upstream does not

Each is deliberate, each is exercised, and each is a row in the sweep above. None
of them lets a config through that upstream refuses.

| ours | upstream | why we keep it |
|---|---|---|
| `num_hidden_layers <= 0` | none | a zero-layer stack is unrepresentable downstream; upstream refuses the same fixture for a different reason (the PLE id range collapses to [1, 0]) |
| `layer_types` length vs `num_hidden_layers` | none | upstream indexes `layer_types[layer_id - 1]` and would `IndexError`; in C++ that is an out-of-bounds read |
| `full_attention_interval <= 0` | none | `(i + 1) % 0` is UB in C++ where Python raises `ZeroDivisionError` |
| `hc_lowrank <= 0` | none | a non-positive rank cannot size the hyper-connection mixer W3 builds |
| `ple_conv_kernel_size <= 0` | none | `short_conv_state_len()` goes negative and W2 sizes a conv state from it |
| `mtp_num_hidden_layers < 0` | none (not even a declared field of `Qwen4ExpTextConfig`) | a negative depth cannot be built |
| `ngram_size < 2` / `heads_per_ngram <= 0` | folded into `ngram_heads <= 0` | same accept/reject boundary, a message that names the field |
| non-integer / non-array JSON where a number or list belongs | Python coerces or raises later | a typed reader has to refuse at the boundary |
| `partial_rotary_factor` outside (0, 1] | none | **belongs to the SHARED reader**, `hf_config.cpp`, not to this model. It fires before this parse runs, which is why there is no local guard: one would be unreachable. Recorded here because the sweep sees it as ours |

### What the config layer still cannot see

`Qwen4ExpParams` resolves the fields W1 through W5 consume. It does NOT yet carry
`linear_num_key_heads` (16), `linear_num_value_heads` (**48**, against upstream's
declared default of 32), `linear_key_head_dim`, `linear_value_head_dim`,
`linear_conv_kernel_dim`, `norm_topk_prob`, `max_position_embeddings` or the
resolved `output_gate_type` value. The shared reader types most of them, so
nothing is lost — but a wave titled "config resolution" owes the statement, and it
is listed under `## Owed`.

## Owed

- [#1978](https://github.com/mudler/vllm.cpp/issues/1978): this port, the campaign
  row. W0 landed the spec with no product code.
- [#1981](https://github.com/mudler/vllm.cpp/issues/1981): **W1**, the config
  surface — resolution, validation, registration, refuse-by-name on everything
  else. LANDED. Recorded here because every `Refuse()` message this code emits
  ends "See `.agents/specs/qwen4-exp-flash-next.md` and issue #1981", and a reader
  who follows that pointer has to find the issue at the other end of it.
- **`Qwen4ExpParams` resolves 60% of the config.** `linear_num_key_heads`,
  `linear_num_value_heads` (48 in the checkpoint, against upstream's declared
  default of 32 — a difference W2 must not inherit from the docstring),
  `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`,
  `norm_topk_prob`, `max_position_embeddings` and the resolved `output_gate_type`
  are read by the shared `HfConfig` and dropped by this struct. Nothing is lost
  yet; W2/W3 owe carrying the ones they consume.
- **A model-layer oracle.** The config layer is gateable and now gated
  (`## The refusal boundary`); nothing above it is. `gateable = no` stands.
- GGUF k-quant arms, including authoring the `qwen4_exp` architecture on our side,
  and the statement that no llama.cpp oracle exists for them.
- MTP depth > 1.
- **W2 (#1987) lands UNREACHED, by AGENTS.md "Nothing lands dead".**
  `src/vllm/model_executor/models/qwen4_exp_ple.{h,cpp}` is a host reference
  for the n-gram hashed embedding and the PLE dilated depthwise conv. No
  production entry point calls it: `qwen4_exp` has no registry entry, no
  loader and no `ModelRegistry::Forward` arm until W5 assembles the model.
  The wiring is owned by row `MODEL-MM-QWEN4-EXP` (W5) and tracked by
  campaign issue [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
  Also owed from that wave: the batched device arm (the host signatures are
  per-sequence precisely so it drops in), the 128-shard NUMERIC table
  reassembly, and the prefix-caching decision for a conv state written by a
  chunked prefill shorter than 9 columns, which `## Design` records as
  AMBIGUOUS and not resolvable from upstream.
- **W2's float path has never been compared at MODEL WIDTH, and that is the one
  gap its own gate cannot close.** `tests/vllm/models/test_qwen4_exp_ple.cpp`
  runs at `hidden_size = 8`, `hc_count = 2`, `heads_per_ngram = 2`,
  `ngram_vocab_size_base = 20`. Only the multipliers, the prime head sizes and
  the offsets are pinned at the released config, and those are INTEGERS, where
  width cannot change an answer. Everything float — the grouped RMSNorm, the
  gate reduction that is 2560 wide in the real model, the 10240-channel dilated
  conv — is gated at width 16 with 8-wide groups. Every structural mutation in
  the W2 table dies there by orders of magnitude, so the instrument is sound for
  structure; a REDUCTION-ORDER difference at width 2560 is what it cannot see,
  and it is exactly the class of difference that a device arm introduces.
  Owed: a first real-width numeric comparison against the lane pin. It must
  derive a **relative** bound, not reuse W2's absolute `1e-5`. W3's repair on
  the sibling branch measured the reason: an exact-double evaluation of the
  oracle's own algorithm for the gated residual already exceeds a 1e-5 absolute
  bound at model width, because torch runs the reduction in fp32, so an absolute
  bound at that width tests the accumulator and not the port.
- The `conv_mask` contract beyond the host arm. W2 gates the masking itself
  (both tensors, and through the 9-column state), but the PAIRED obligation it
  documents — a masked position must already carry EOS in `input_ids`, because
  the hash reads ids and not activations — is a CALLER obligation with no caller
  yet. W5 owns asserting it where the mask is built.
- The 1M-token RoPE extension above the native 262144.
- The non-resident n-gram table on CUDA. **W6a (#1989) discharged the CPU half**:
  the dequantizing gather (`vt::Embedding` over a block table) and the
  `kEmbeddingTable` keep-quant policy change both landed, gated bit-exactly
  against llama.cpp `b10451` decoding real bytes of the shipped tensor. What is
  still owed is the **CUDA arm**: `EmbeddingKernelCuda` (`src/vt/cuda/cuda_ops.cu`)
  refuses anything but f32/bf16, so `DeviceQuantGatherSupported` returns false on
  CUDA and the table keeps its expand-bf16 residency there. That is the honest
  state and it is also the expensive one — a device-resident quantized table
  gathered on device is precisely the shape llama.cpp's #27742 does NOT have (it
  pins the table to the CPU by tensor class), so the CUDA arm is where this
  model's high-concurrency advantage lives, not a tidying task. Still owed with
  it: a measurement of the page-cache cost that the <= 64 KiB/token arithmetic
  only bounds.
- **VERIFIED 2026-08-26, no longer owed:** llama.cpp's substitution for a
  ragged-K tensor is read at the pin, `src/llama-quant.cpp:374-405 @ b10451`
  (`tensor_type_fallback`). `Q4_K -> Q5_0` is confirmed exactly as this spec
  asserted, and `IQ4_XS -> IQ4_NL` beside it, which is why the shipped UD-IQ1_S
  carries 49 IQ4_NL tensors. Both encodings landed in W6a.
- **NEW, from reading that table:** the same function maps `Q5_K -> Q5_1` (ggml
  type 7) and `Q2_K`/`Q3_K` -> `Q4_0`. Q5_1 and Q4_1 (3) are still absent from
  our reader, so a `-Q5_K_M` recipe of THIS model — whose `ffn_down_shexp` row is
  640 and therefore ragged for any K-quant — would refuse at header parse. Not
  in W6a's scope, which the shipped file does not need; recorded rather than
  quietly added.
- **A keep-quant gather for `deepseek4` and `laguna`.** W6a made
  `GgufTensorRole::kEmbeddingTable` keep-quant eligible, which is a change to a
  SHARED policy with three consumers. Only `qwen3_5_gguf_weights.cpp` was given
  the residency; `deepseek_v4_weights.cpp` and `laguna_weights.cpp` consume
  `token_embd` as a flat host f32 array (and, on a tied file, hand the same f32
  image to the final projection), so both now narrow the policy for that tensor
  through `NoKeepQuant` and keep expanding it. That is correct and it is not
  free: on a real deepseek4 or laguna checkpoint the vocab matrix is still
  materialized in f32. Decoding it per gathered row instead needs those two
  forwards to take a `vt::Tensor` rather than a `std::vector<float>`, which is
  model work and not policy work. Owed to
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978).
- The non-resident n-gram table on CUDA: the dequantizing gather op and the
  `kEmbeddingTable` keep-quant policy change (Route B), and a measurement of the
  page-cache cost that the <= 64 KiB/token arithmetic only bounds.
- ~~llama.cpp's ragged-K substitution~~ **RESOLVED, AND NOW READ AT THE PIN**:
  `Q4_K -> Q5_0`, `IQ4_XS -> IQ4_NL`, from `tensor_type_fallback` in
  `src/llama-quant.cpp:374-406` of the `llama-cpp` oracle at its recorded revision
  `10bf611e533d81f739128304991c5e133c6aebd8` (`b10451`,
  [`../oracles/llama-cpp.md`](../oracles/llama-cpp.md)) — not at `master`, which is
  where the claim was first read and which is not an oracle. The complete table at
  that revision: `IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`/`IQ4_XS
  -> IQ4_NL`; `Q2_0`/`Q2_K`/`Q3_K`/`TQ1_0`/`TQ2_0 -> Q4_0`; `Q4_K -> Q5_0`;
  `Q5_K -> Q5_1`; `Q6_K -> Q8_0`; anything else throws. Both are reachable for this
  model depending on the recipe, and our reader supports NEITHER (no `case 6`, no
  `case 20`), so W6 owes both.
- **A published GGUF now EXISTS**, which supersedes this spec's "no GGUF exists and no
  tool can produce one": `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, 67.56 GiB of
  weights in 3 shards, `general.architecture = qwen4exp`, 1224 tensors. **PINNED**, and
  it needed to be — the repo's `lastModified` moved to `2026-08-26T15:54:43Z`, after
  W1's pull request was opened, which is exactly the re-quantize-in-place case AGENTS.md
  "Say which weights, and from where" names. Revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`; at that revision, shard sizes
  10,946,624 + 49,990,818,368 + 22,544,696,352 = **72,546,461,344 bytes = 67.564 GiB**,
  with sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`,
  `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6` and
  `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a`. Those digests are
  the Hub API's `lfs.oid` values and are NOT locally computed; W6 owes a local sha256
  when it stages the file. The "1224 tensors" count remains UNVERIFIED: shard 1 is the
  metadata shard and reports `n_tensors = 0`. It FITS GB10
  with ~52 GiB of headroom, and two things in OUR tree stop us loading it: the missing
  IQ4_NL reader arm, and the gather-table expansion. Its metadata independently
  confirms this spec's n-gram derivation to the digit --
  `ple.layer_multipliers = [23703573157769, 20109073645365, 8052911324071]` and
  `ple.head_vocab_sizes = [20000003, 20000023, ...]`.
- **Mirror the `qwen4exp` GGUF key and tensor names rather than inventing ours.** Two
  competing llama.cpp PRs (#27742 open, #27739 closed-by-courtesy) already disagree on
  `ple.*` key spellings and on whether the n-gram table is model-level
  (`per_layer_token_embd`) or per-layer (`blk.N.ple_ngram_embd`), and a maintainer has
  asked for a rename, so the names are NOT settled. Re-check before W6a commits to a
  layout; a wrong guess makes every published GGUF unreadable by us.
- A K-divisibility assertion in whatever writes our GGUF files.
- A speed denominator, once one exists.
- **W4's QSA slice lands UNREACHED**, and this entry is what AGENTS.md "Nothing
  lands dead" requires in exchange.
  `src/vllm/model_executor/models/qwen4_exp_qsa.{h,cpp}`
  ([#1991](https://github.com/mudler/vllm.cpp/issues/1991)) ship the indexer, the
  side-cache sizing and the GATHER consumer as host reference math with no
  production call site: `Qwen4ExpTextModel` does not exist yet, its PLE
  ([#1987](https://github.com/mudler/vllm.cpp/issues/1987)), hyper-connection
  stream ([#1988](https://github.com/mudler/vllm.cpp/issues/1988)) and GGUF
  reader ([#1989](https://github.com/mudler/vllm.cpp/issues/1989)) are sibling
  waves, and the registry entry plus runner wiring belong to W5. Row
  `MODEL-MM-QWEN4-EXP` owns that wiring and
  [#1978](https://github.com/mudler/vllm.cpp/issues/1978) tracks it.
- **The QSA device arm.** `qwen4_exp_qsa.cpp` is the portable oracle a CUDA
  kernel is written against, the way `deepseek_v4_dsa.h` is for
  `src/vt/cuda/cuda_deepseek_v4.cu`. Nothing in W4 runs on a GPU, so the gather's
  cost advantage over the mask is stated by a `keys_visited` count and NOT by a
  measurement. The speed axis opens at G4.
- **`QsaCompressNormRope` assumes a contiguous visible range.** Upstream forms
  blocks over `local_visible_indices` of a padded batch; a serving engine's
  ragged batch has no interior masking, so the two coincide and the function
  asserts `num_keys % compress_ratio == 0` instead of accepting an arbitrary
  visibility set. A padded-batch caller would need the general form.
- **The row's lifecycle record is owed the W4 transition, and W5 lands it.** W4
  ([#1991](https://github.com/mudler/vllm.cpp/issues/1991)) is this row's first
  product code: `src/vllm/model_executor/models/qwen4_exp_qsa.cpp` joins
  `add_library(vllm ...)` at its merge commit. `.agents/model-matrix.md` still
  carries the row at `READY` with the note "SPEC ONLY, NO PRODUCT CODE, NO TOKEN,
  NO SPEED", which was true at the merge base and is false from W4 onwards. That
  cell is NOT edited here: W1 through W3 are live on the same file and the
  operator is sequencing those writes, and a per-wave edit to one shared row is
  exactly the lock AGENTS.md "Records" forbids. W5, which lands the registry entry
  and the runner wiring, moves the row to `ACTIVE`, rewrites that note and updates
  `## Now` in the one change. Until then this entry is where the discrepancy is
  visible.
- **Nothing gates the interleaved-mRoPE section layout, in W4 or anywhere yet.**
  `gen_qwen4_exp_qsa_goldens.py` passes a 2-D `position_ids`, which
  `Qwen4ExpTextRotaryEmbedding.forward` expands into three IDENTICAL streams, so
  `apply_interleaved_mrope` runs value-blind and the captured `cos`/`sin` are
  indistinguishable from plain RoPE. `qwen4_exp_qsa.h` scopes the tables out of W4
  ("this function does not build them") and W4 is honest about that, but no wave
  currently owns building them, and a multimodal caller with genuinely different
  t/h/w streams would be running an untested section layout. The wave that builds
  the cos/sin tables owes a case with three DISTINCT position streams.
- **W3's host reference lands UNREACHED, and this is the record of it** per
  AGENTS.md "Nothing lands dead".
  `src/vllm/model_executor/models/qwen4_exp_hc.{h,cpp}`
  ([#1988](https://github.com/mudler/vllm.cpp/issues/1988)) is reached only by
  `tests/vllm/models/test_qwen4_exp_hc.cpp`. No production entry point calls it
  at its merge commit: W1 config registration
  ([#1986](https://github.com/mudler/vllm.cpp/issues/1986)) was still in review,
  so no `qwen4_exp` resolves through the loader and there is nothing for the
  gated-residual stream to hang off. The wiring is owed by **W5, assembly**,
  under [#1978](https://github.com/mudler/vllm.cpp/issues/1978), which is the
  wave that widens the residual buffers to `hc_count * hidden_size` and calls
  the module twice per layer.
- The **model-matrix lifecycle cell** for
  `MODEL-MM-qwen4-exp-qwen4-exp-for-conditional-generation`, which still reads
  `SPEC ONLY`. Left to W1 deliberately rather than by omission: W1 is the wave
  whose scope IS registration, its pull request is already open, and
  `.agents/model-matrix.md` is a single shared file, so three parallel waves
  editing one cell is the write-lock AGENTS.md "Records" names. Whichever of
  W1/W2/W3 lands last owes the correction.
- The **device arm of the gated residual**, and with it one check this host wave
  cannot make: that `RMSNormGated.forward_cuda`'s flash-linear-attention Triton
  kernel is numerically correct in its GROUPED mode (unverified upstream, see
  `## Design`).
- **W3's `kTol = 1e-5` is an absolute bound that does not survive a rescale, and
  the host reference is the first thing it fails.** Recorded because an earlier
  draft of the bullet above framed the tolerance question as the DEVICE arm's
  problem, and it is not. Measured against the pinned oracle itself, at the
  model's own shape (hidden_size 2560, hc_count 4, hc_lowrank 320, eps 1e-6, two
  tokens), max|diff| on `mixed_input`. This is ONE draw of random inputs, and the
  ratios below move from draw to draw; the ordering and the conclusion do not.

  | | t=0 | t=1 |
  |---|---|---|
  | ours (fp32) vs oracle | 2.325e-05 | 2.137e-05 |
  | exact double vs oracle | 1.360e-05 | 5.431e-06 |
  | ours (fp32) vs exact double | 3.684e-05 | 1.606e-05 |

  At the suite's own widths (flat = 24 and 15) the implementation is bit-identical
  to the oracle -- max|diff| over every golden array of cases A, B and C is
  2.384e-07 -- so kTol carries a 42x margin there and constrains nothing. At model
  width our fp32 interior is 2.1x to 2.3x over it, driven by `LinearNoBias`'s
  sequential fp32 accumulation over 10240 terms. **The second row is the one that
  settles it: the ORACLE is itself of the same ORDER as kTol against an exact
  evaluation of its own algorithm -- 1.36x on the draw above, 0.91x and 0.82x on
  an independent draw taken during fresh review -- because torch runs this in
  fp32 too.** No fp32
  implementation of this function meets a 1e-5 ABSOLUTE bound at hidden_size
  2560, and widening our accumulator cannot rescue one. W5 therefore does not
  reuse kTol at model width; the file carries a real-width case with a relative
  bound (`kRealWidthMixedRel`, 4e-5, derived as 6.6x the sqrt(K)*u random-walk
  bound for K = 10240) that all three measurements sit inside. **What is still
  owed** is agreement with the ORACLE at model width, which needs a real
  checkpoint and cannot be closed in-suite: the in-suite case compares against
  the double reference, because dumping one token of oracle IO at this width is
  26 MB of `.inc`.
- **The double accumulator is now gated, and the device arm inherits the
  consequence.** `GroupedRmsNorm` accumulates the per-group sum of squares in
  `double`, and at the suite's group sizes of 5 and 6 that convention had zero
  discriminating power -- replacing it with `float` left the suite 280/280 green.
  It is gated at the model's real group size of 2560, on magnitude-separated
  data, where the two accumulators differ by 742x (3.168e-06 against 2.352e-03,
  bound 1e-4). The convention is kept rather than dropped because it makes the
  host reference more accurate than the oracle rather than less, which is what a
  reference is for. What follows for the device arm, stated here so it is not
  discovered: **a straight fp32-accumulate device reduction will not meet
  `kRealWidthNormTol` on that data.** That is the correct signal, not a defect in
  the gate -- it says the device kernel must accumulate wider than fp32 or be
  gated against the oracle directly rather than against this reference. Deciding
  which is the device wave's, and it is owed.
- The **fused rank-1 write-back**. `GatedResidualWriteBackInPlace` is the seam
  and is already the primitive, but no device kernel replaces it yet. Both
  llama.cpp implementations of this architecture materialise the update as a
  `repeat_4d` + `mul`, i.e. 96 dense `[2560, 4, T]` broadcasts built and thrown
  away per forward pass at 48 layers x 2 sites, which is where a
  beat-llama.cpp-at-concurrency claim would come from. Not claimed here: no arm
  runs.

- **Nothing detects two claim files owning one matrix row**
  ([#2056](https://github.com/mudler/vllm.cpp/issues/2056)), and this row proved it
  rather than supposed it. W1 and W6a each wrote their own `CLAIM-*` for this row,
  both correct in isolation because each wave was the first product code from its
  own vantage. Copying one beside the other and running the checker gives
  `agent record OK ... rc=0`: git cannot conflict on it because the two sides touch
  different PATHS, and no gate reads for duplicate ownership. The collision was
  resolved by MERGE ORDER — W6a landed the row-level claim, W1 dropped its
  `-W1` file and deferred — which is an operator remembering, not a gate. Filed
  rather than fixed in flow because it changes checker semantics and so owes its
  own row, spec and red-before test per AGENTS.md §"Changing the rules or a
  checker". Owned by `MODEL-MM-QWEN4-EXP` until re-homed.

- **W5a (#2031) lands the GGUF WEIGHT LOADER, and it lands REACHED.**
  `src/vllm/model_executor/models/qwen4_exp_weights.{h,cpp}` materialize the
  text tower from a `qwen4exp` file, and `LoadQwen4ExpForConditionalGeneration`
  — the registry's `load_weights` hook, which a `qwen4exp` file already reaches
  through the `kGgufArchArms` dispatch row W6a added — calls it instead of
  refusing. This is the first slice of this row with a production call site.
  What it does NOT do is make the architecture SERVE: the forward and the
  KV-cache spec still refuse by name, so nothing decodes a token. Those two are
  W5b and W5c below.
- **W5b, the forward, is OWED and it is the row's remaining barrier.** The
  scope is `Qwen4ExpTextModel::Forward` over 48 layers in `vt::` ops — the
  10240-wide hyper-connection stream, 36 Gated DeltaNet layers, 12 QSA layers,
  the 512-expert MoE with its shared expert, the PLE layer on 0-based layer 1,
  and the `use_combine=false` mixer that collapses the stream at the end. Two
  structural facts about this tree shape it, both MEASURED during W5a rather
  than assumed, and neither was in this spec before:
    * **`qwen3_5.cpp` lines 1209-7890 are one anonymous namespace.**
      `GdnBlockPaged`, `FullAttnBlockPaged`, `MoeBlock`, `SharedExpert`,
      `RunLayerPaged`, `StepDevInputs` and `BuildStepDevInputs` all have
      INTERNAL LINKAGE, so no new translation unit can call any of them. Reuse
      needs them hoisted into a header the way `dense_attn_block.h` was hoisted
      out of `qwen3.cpp` — a documented in-tree precedent, and an edit to a
      1745-line-plus file several other rows are working in. That extraction is
      its own unit of work and should be its own row.
    * **What IS free is the `vt::` op layer**, and it is enough to build the
      forward from: every `vt::Gdn*` entry point has a registered CPU kernel as
      well as a CUDA one, `vt::Moe*`, `vt::FusedChain`, `MRotaryEmbedding`,
      `dense_attn::AttnBlock` and `layers::MlpGateUpMethodBase` are all
      header-inline or externally linked. `muse_glimmer.cpp` builds a complete
      forward that way in 510 lines and is the shape to follow.
  W2/W3/W4 remain host-float references with `std::vector<float>` signatures;
  the forward needs their arithmetic in `vt::` ops, which is the "device arm"
  each of those waves already records as owed. Writing a host-float forward
  instead would be the hand-written parallel path AGENTS.md §"Shared seams"
  forbids, and it is recorded here so the shortcut is refused deliberately
  rather than rediscovered.
- **W5c, the KV-cache spec, is OWED and blocked behind W5b.** It needs three
  conv states on a PLE layer (GDN conv, PLE conv, and an int64 n-gram token
  history) plus the QSA indexer side cache. One naming correction found in
  W5a: this tree's `MLAAttentionSpec` has NO `tokens_per_state` field. The
  compression knob is spelled **`compress_ratio`**
  (`include/vllm/v1/kv_cache_interface.h`), and `storage_block_size()` returns
  `block_size / compress_ratio`, which is the same semantics under a different
  name. The only `tokens_per_state` identifier in the repository is W4's own
  `QsaSideCacheSpec`. A W5c implementer reaching for the upstream spelling
  finds nothing.
- **The VISION path is owed and has no GGUF artifact to load.** The tower is an
  unchanged `Qwen3_5MoeVisionModel`, but the shipped `unsloth` UD-IQ1_S file is
  TEXT-ONLY: its 1224 tensors are 768 hyper-connection/MoE, 324 Gated DeltaNet,
  120 QSA, 6 PLE and 6 model-level, and there is not one `v.blk.*` or `mm.*`
  among them. So the multimodal arm needs either a companion mmproj that does
  not exist yet or the safetensors arm that no device we own can hold. Recorded
  as a BLOCKER rather than as scheduling.
- **The safetensors arm is refused for a reason, and the refusal now says it.**
  Every published safetensors artifact — bf16 ~360 GB, FP8 ~180 GB, NVFP4
  ~128 GB — exceeds every device this project owns, so an arm that read them
  would be code nothing could run. W5a rewrote the message from "not ported
  yet", which reads as scheduling, to the size argument plus the name of the
  arm that IS supported.
- **The converter and the algorithm oracle DISAGREE on which EOS the n-gram
  hash uses, and a GGUF-only load has to take the converter's.** llama.cpp
  #27742 resolves `qwen4exp.ple.eos_token_id` as `int(eos[-1])`, the LAST
  element of the HF list; `Qwen4ExpTextModel.forward` takes element `[0]`. On a
  single-entry list they coincide and nothing shows. On a longer one they
  disagree, and the disagreement is invisible because both runtimes emit fluent
  text from different n-gram segment boundaries. `ParseQwen4ExpParams` follows
  the algorithm oracle wherever a `config.json` is present; from a GGUF the
  container is the only source. **LATENT on this checkpoint, not active:** the
  released `Qwen/Qwen3.8-Flash-Next` `config.json` carries
  `eos_token_id: 248044` as a bare integer, so `[0]` and `[-1]` are the same
  value and the two runtimes agree today. Owed: the same check on any future
  checkpoint of this family whose `eos_token_id` is a list.
- **The V-head reorder costs the Gated DeltaNet tower its keep-quant
  residency. MEASURED: net +2.446 GiB, and NOT fit-threatening.**
  `_LinearAttentionVReorderBase` fires whenever `num_k_heads != num_v_heads`,
  which is 16 vs 48 here, so every GDN projection of all 36 linear layers is
  layout-rewritten at load and therefore `kTransformedWeight` — it expands to
  bf16 instead of staying Q5_K/Q6_K. The ROW reorders could in principle be done
  inside the block stream (a k-quant row is a whole number of superblocks, so
  moving whole rows never cuts one), but `out_proj`'s is a COLUMN permutation
  and can never be. `qwen3_5_gguf_weights.cpp` already has this property for the
  27B.

  The earlier version of this entry said "resident-bytes unmeasured, because no
  real file has been loaded". **No file is needed.** The committed 1224-tensor
  manifest (`tests/vllm/models/qwen4_exp_gguf_manifest.inc`,
  `unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ1_S` @
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`) carries every name, `ne` and ggml
  type id, and the block geometries are `src/vt/dtype.cpp`'s own table (Q5_K
  256/176, Q6_K 256/210, F32 1/4). The whole file sums to 72,535,436,800 B =
  **67.554 GiB**, which agrees with the 67.56 GiB this spec states from the
  repository listing — so the arithmetic below is cross-checked against a number
  measured a different way.

  The five tensors the reorder moves off the keep-quant arm, over 36 GDN layers:

  | tensor | ggml type | on disk | resident bf16 |
  |---|---|---|---|
  | `attn_qkv.weight` [10240, 2560] | Q5_K | 652,288,000 B (0.6075 GiB) | 1,887,436,800 B (1.7578 GiB) |
  | `attn_gate.weight` [6144, 2560] | Q5_K | 391,372,800 B (0.3645 GiB) | 1,132,462,080 B (1.0547 GiB) |
  | `ssm_out.weight` [2560, 6144] | Q6_K | 464,486,400 B (0.4326 GiB) | 1,132,462,080 B (1.0547 GiB) |
  | `ssm_alpha.weight` [48, 2560] | F32 | 17,694,720 B (0.0165 GiB) | 8,847,360 B (0.0082 GiB) |
  | `ssm_beta.weight` [48, 2560] | F32 | 17,694,720 B (0.0165 GiB) | 8,847,360 B (0.0082 GiB) |
  | **total** | | **1,543,536,640 B (1.4375 GiB)** | **4,170,055,680 B (3.8837 GiB)** |

  **Net +2,626,519,040 B = +2.446 GiB.** The two `ssm_*` rows NARROW, because
  the file stores them F32 and we hold them bf16; only the three quantized
  projections grow.

  `ssm_conv1d.weight` is deliberately NOT in that table even though it also
  lands bf16 (5,898,240 B -> 2,949,120 B over 36 layers). It is a depthwise
  filter that `LoadGdn` dequantizes whether or not the reorder fires, so its
  residency is not a cost of the reorder. Including it would move the "on disk"
  total to 1.4430 GiB and the net to +2.443 GiB, which is the difference between
  this figure and a first reading of it.

  **Not fit-threatening.** 67.554 GiB on disk against ~119.6 GiB usable on GB10
  leaves ~52 GiB, and +2.446 GiB is 4.7% of that headroom. The residency
  question that DOES threaten the fit is the n-gram gather table on a non-CPU
  device, which is +68.5 GiB and is the next entry.
- **The CUDA arm cannot gather from the n-gram table, so it is REFUSED at load
  ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)).**
  `DeviceQuantGatherSupported` is true for `kCPU` alone, so on any other device
  `RouteGgufTensor` sends `per_layer_token_embd.weight` to `kExpandBf16`. From
  the same manifest that tensor is [320001536, 160] IQ4_NL: **26.822 GiB on
  disk, 95.368 GiB expanded** (320001536 x 160 x 2 = 102,400,491,520 B), on a
  box with ~119.6 GiB for everything. The #1123 device-fit guard sums the file's
  ON-DISK bytes — 67.554 GiB, comfortably inside the budget — so it admits the
  load and the expansion happens after it, which is `model_loader.cpp`'s own
  stated worst case: "Loading for 26 minutes and dying mid-stream is the worst
  of the available behaviours."

  W5a's repair takes the device as an argument to `LoadQwen4ExpFromGguf` — no
  default, so a caller cannot disable the guard by saying nothing — and refuses
  BY NAME ahead of any tensor I/O. **Owed: the CUDA block-decoding gather
  kernel.** Until it exists this is a CPU-only arm, which is now a named refusal
  instead of a discovery.

  **Also owed, and only NARROWED by that guard: the load still runs to
  completion on `--device cpu` and then dies in `MakeQwen4ExpKVCache`.** Before
  W5a the loader refused at once; after it, a CPU user pays the full load first.
  W5c closes this by making that function return a config rather than throw.
- **`ReorderVRows`/`ReorderVCols` exist twice in this tree, and only ONE copy is
  gated.** `qwen3_5_gguf_weights.cpp` has them in an anonymous namespace with no
  header, and `qwen4_exp_weights.cpp` has its own copy. Four lines of index
  arithmetic, deliberately duplicated rather than hoisted: the hoist edits a
  1745-line translation unit other rows are working in, which is the same
  shared-file lock the `qwen3_5.cpp` extraction above runs into. Owed to
  whichever row does that extraction.

  **The earlier version of this entry said "gated on both sides", and that was
  FALSE.** Measured at `a68312c79` by the W5a repair: swapping `g` and `t` in
  the `qwen4_exp_weights.cpp` copy reddens `test_qwen4_exp_gguf_weights` (2
  cases, 41 assertions, mutation M5), while the same swap in the
  `qwen3_5_gguf_weights.cpp` copy leaves `test_gguf_qwen36_loader` (7/7, 555
  assertions),
  `test_model_loader_gguf` (7/7), `test_gguf_nvfp4` (14/14) and
  `test_gguf_keep_quant` (42/42) ALL green — every synthetic `qwen35`/
  `qwen35moe` fixture in the tree is `ssm.group_count = 2,
  ssm.time_step_rank = 4`, i.e. K == R, where the permutation is its own
  inverse. The qwen3_5 side is tracked by
  [#2081](https://github.com/mudler/vllm.cpp/issues/2081) and is deliberately
  NOT fixed under this row: re-shaping a shipped model's fixtures changes
  `qwen35`, `qwen35moe` and `qwen3next` coverage and is not this row's scope.

  **#2081 now has an `issue-index.md` row, and it did not before.** The issue was
  open on GitHub and named here, but the index carried nothing for it, so two of
  the three places AGENTS.md requires to agree did not. The appended row names
  `MODEL-MM-qwen3-5-qwen3-5-for-conditional-generation` as the owner — the
  shipped model whose fixtures have to change — and points back at this `## Owed`
  entry. Two corrections ride with it, both from re-measuring MUT-M6 on this
  branch rather than relaying the earlier numbers. The survival holds exactly
  (7/7 555, 7/7 23, 14/14 2352, 42/42 6340, every count identical to the
  un-mutated baseline). The stated CAUSE was too narrow: the fixtures are not all
  `ssm.time_step_rank = 4`. `test_gguf_qwen36_loader`'s default shape is
  `group_count = 2, time_step_rank = 2`, i.e. R = 1, where the map is the
  IDENTITY and no inversion is even expressible; only the one case named "V-head
  reorder when num_v != num_k" reaches R = 2, and that is the self-inverse
  K == R. Both roads end at the same place, but a reader chasing "K == R" through
  the default fixture would not find it.

## Now

`ACTIVE`. Six reviewed waves have landed. Five of them are unreached by design
and the sixth, W5a, is the first with a production call site:

| Wave | Lands | Issue |
|---|---|---|
| W1 | the config layer: `qwen4_exp` resolves, parses and VALIDATES | [#1981](https://github.com/mudler/vllm.cpp/issues/1981) |
| W2 | the hashed n-gram index and the PLE dilated depthwise conv | [#1987](https://github.com/mudler/vllm.cpp/issues/1987) |
| W3 | the 4-branch gated-residual hyper-connection stream | [#1988](https://github.com/mudler/vllm.cpp/issues/1988) |
| W4 | Qwen Sparse Attention with a GATHER consumer | [#1991](https://github.com/mudler/vllm.cpp/issues/1991) |
| W6a | IQ4_NL, Q5_0 and a dequantizing gather, so the artifact OPENS | [#1989](https://github.com/mudler/vllm.cpp/issues/1989) |
| W5a | the GGUF weight loader, REACHED through the `load_weights` hook | [#2031](https://github.com/mudler/vllm.cpp/issues/2031) |

**Reached, and LOADING — on a CPU device:** a `qwen4exp` file lands on
`Qwen4ExpHfConfigFromGguf` through the `kGgufArchArms` dispatch row, the registry
resolves `Qwen4ExpForConditionalGeneration`, and W5a's `load_weights` hook now
materializes the whole text tower instead of refusing. That is the first
production call site this row has had. On any device that cannot gather from a
block table the load REFUSES BY NAME ahead of any tensor I/O, because the
n-gram table would otherwise expand from 26.822 GiB to 95.368 GiB of host
memory ([#2083](https://github.com/mudler/vllm.cpp/issues/2083)); the CUDA
gather arm is owed.

**Reached, and still refusing:** the forward and the KV-cache spec. Nothing
decodes a token, so there is still no token number, no speed number, no
`examples/server` e2e and no `docs/USAGE.md` weights row — that row is owed in
the same change that makes an arm SERVE, which is W5b, not W5a. W2, W3 and W4
remain host reference math with no production call site.

**What is owed, in order.** W5b, the forward in `vt::` ops
([#2031](https://github.com/mudler/vllm.cpp/issues/2031)); W5c, the KV-cache
spec with three conv states and the QSA side cache; then the first served
request, G2 with a prompt past 2048 tokens, and only then the G4 speed axis,
which additionally waits on `dgx:gpu0`. MTP/speculators are W7
([#1993](https://github.com/mudler/vllm.cpp/issues/1993)). Each is scoped under
`## Owed` above with the structural facts W5a measured.

Both decisions this spec was blocked on are **settled** (developer, 2026-08-26) and
recorded in place rather than left as proposals: the transformers lane pin is
ACCEPTED at 5.16.0 (`## Oracles`), and the first runnable arm is the Q4_K_M backbone
with a non-resident n-gram table (`## Hardware`).

Next actions, in order: W2 (hashed n-gram embedding + PLE dilated depthwise conv) and
W3 (hyper-connection residual stream) are both reachable today against the lane pin
with tiny random configs and need neither a checkpoint nor a GPU lease — and both
inherit a config layer whose boundary is measured, so a golden that disagrees is a
port defect and not a config question. W6b's mechanism is the unknown that decides
whether the chosen arm is schedulable, and it should be spiked before W6 is planned.
